"""Contract tests for the official RoboTwin-MeM admission integration."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lewm.envs.robotwin_mem import (
    DEFAULT_MEMORY_BUDGET,
    MATCHED_MEMORY_CONDITIONS,
    OFFICIAL_COMMIT,
    OFFICIAL_DATASET_REVISION,
    EpisodeRecord,
    assert_matched_budget,
    decide_admission_gate,
    deterministic_episode_split,
    load_episode_records,
    policy_view,
    raw_memory_bytes,
    recent_suffix_audit,
    select_memory_indices,
    source_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = (
    ROOT / "outputs/robotwin_mem_admission_v1/external/RoboTwin-MeM"
)


def _records(task_id: str, count: int = 50) -> list[EpisodeRecord]:
    return [
        EpisodeRecord(
            task_id=task_id,
            episode_index=index,
            episode_seed=index,
            length=700,
            instruction="official instruction",
            keyframe_steps=(95, 303, 510),
            query_steps=(616,),
        )
        for index in range(count)
    ]


def test_official_source_versions_and_licenses_are_pinned() -> None:
    receipt = source_receipt()
    assert receipt["repository_commit"] == OFFICIAL_COMMIT
    assert receipt["dataset_revision"] == OFFICIAL_DATASET_REVISION
    assert receipt["code_license"] == "MIT"
    assert receipt["dataset_license"] == "Apache-2.0"
    assert set(receipt["tasks"]) == {
        "pick_the_unhidden_block",
        "pick_objects_in_order",
        "cover_blocks_hard",
    }


def test_deterministic_split_is_disjoint() -> None:
    records = _records("pick_the_unhidden_block")
    first = deterministic_episode_split(
        "pick_the_unhidden_block", records
    )
    second = deterministic_episode_split(
        "pick_the_unhidden_block", records
    )
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 30,
        "validation": 10,
        "test": 10,
    }
    assert not (
        set(first["train"]) & set(first["validation"])
        or set(first["train"]) & set(first["test"])
        or set(first["validation"]) & set(first["test"])
    )


def test_matched_conditions_use_identical_frame_and_byte_budget() -> None:
    record = _records("pick_the_unhidden_block", 1)[0]
    selections = {
        condition: select_memory_indices(
            condition,
            record,
            random_seed=7,
        )
        for condition in MATCHED_MEMORY_CONDITIONS
    }
    assert_matched_budget(selections)
    assert all(
        len(indices) == DEFAULT_MEMORY_BUDGET
        for indices in selections.values()
    )
    assert raw_memory_bytes() == (
        DEFAULT_MEMORY_BUDGET * 3 * 480 * 640 * 3
    )
    assert np.all(selections["no_memory"] == -1)
    assert selections["recent_only"].tolist() == [612, 613, 614, 615]


def test_random_selector_does_not_consult_oracle_events() -> None:
    left = _records("pick_the_unhidden_block", 1)[0]
    right = EpisodeRecord(
        **{
            **left.__dict__,
            "keyframe_steps": (1, 2, 3),
        }
    )
    left_indices = select_memory_indices(
        "random_event", left, random_seed=19
    )
    right_indices = select_memory_indices(
        "random_event", right, random_seed=19
    )
    assert np.array_equal(left_indices, right_indices)


def test_recent_suffix_leakage_audit_is_fail_closed() -> None:
    clean = _records("pick_the_unhidden_block", 1)[0]
    audit = recent_suffix_audit(clean)
    assert audit["passed"]
    assert audit["event_overlap_frames"] == 0
    assert audit["gap_frames"] == 101

    leaked = EpisodeRecord(
        **{
            **clean.__dict__,
            "keyframe_steps": (95, 303, 610),
        }
    )
    assert not recent_suffix_audit(leaked)["passed"]


def test_policy_surface_rejects_evaluator_metadata() -> None:
    rgb = np.zeros((3, 480, 640, 3), dtype=np.uint8)
    proprio = np.zeros(14, dtype=np.float32)
    view = policy_view(
        rgb=rgb,
        proprio=proprio,
        instruction="official query",
    )
    assert set(view) == {"rgb", "proprio", "language_instruction"}
    with pytest.raises(AssertionError, match="evaluator-only"):
        policy_view(
            rgb=rgb,
            proprio=proprio,
            instruction="official query",
            metadata={"keyframe_steps": [10]},
        )


def test_training_path_has_no_oracle_condition_or_scene_labels() -> None:
    path = ROOT / "scripts/run_robotwin_mem_admission.py"
    tree = ast.parse(path.read_text())
    training_modes = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "TRAINING_MODES"
            for target in node.targets
        )
    )
    literals = {
        node.value
        for node in ast.walk(training_modes)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "oracle_event_set" not in literals
    assert "oracle_best_event" not in literals
    assert literals == {"full_history", "auto_surprise", "random_event"}

    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_automatic_surprise_indices"
    )
    surprise_literals = {
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not (
        surprise_literals
        & {"keyframe_steps", "scene_info", "target_visible_block_id"}
    )


def test_paired_gate_requires_positive_oracle_confidence_interval() -> None:
    recent = np.zeros((3, 10), dtype=np.float64)
    no_memory = np.zeros_like(recent)
    oracle = np.ones_like(recent)
    probe = np.full_like(recent, 0.25)
    decision = decide_admission_gate(
        recent_success=recent,
        oracle_success=oracle,
        no_memory_success=no_memory,
        recent_probe_accuracy=probe,
        candidate_count=4,
    )
    assert decision.passed
    assert decision.metrics["oracle_minus_recent_ci_low"] > 0

    tied = decide_admission_gate(
        recent_success=oracle,
        oracle_success=oracle,
        no_memory_success=no_memory,
        recent_probe_accuracy=probe,
        candidate_count=4,
    )
    assert not tied.passed
    assert "oracle_gain" in tied.reasons


@pytest.mark.skipif(
    not DATASET_ROOT.exists(), reason="official RoboTwin-MeM data not downloaded"
)
def test_official_dataset_contract_and_deterministic_loading() -> None:
    records_a = load_episode_records(
        DATASET_ROOT, "pick_the_unhidden_block"
    )
    records_b = load_episode_records(
        DATASET_ROOT, "pick_the_unhidden_block"
    )
    assert records_a == records_b
    assert len(records_a) == 50
    assert records_a[0].length == 697
    assert records_a[0].keyframe_steps == (94, 302, 509)
    assert records_a[0].query_steps == (616,)

    smoke = json.loads(
        (ROOT / "outputs/robotwin_mem_admission_v1/smoke_receipt.json").read_text()
    )
    assert smoke["passed"]
    assert smoke["deterministic_decode"]
    assert smoke["state_shape"] == [697, 14]
    assert smoke["action_shape"] == [697, 14]
    assert set(smoke["cameras"]) == {
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    }


@pytest.mark.skipif(
    not (
        ROOT
        / "outputs/robotwin_mem_admission_v1/"
        "vlm_control_protocol_registration.json"
    ).exists(),
    reason="confirmatory VLM control not registered",
)
def test_confirmatory_control_source_and_episodes_are_frozen() -> None:
    registration = json.loads(
        (
            ROOT
            / "outputs/robotwin_mem_admission_v1/"
            "vlm_control_protocol_registration.json"
        ).read_text()
    )
    source = ROOT / "scripts/run_robotwin_mem_vlm_control.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    expected_exclusions = {
        "pick_the_unhidden_block": [0],
        "pick_objects_in_order": [1],
        "cover_blocks_hard": [3],
    }
    for task_id, task in registration["tasks"].items():
        assert task["script_sha256"] == source_hash
        assert task["excluded_development_episodes"] == expected_exclusions[task_id]
        assert len(task["confirmatory_test_episodes"]) == 9
        assert not (
            set(task["confirmatory_test_episodes"])
            & set(task["excluded_development_episodes"])
        )
        assert task["control_seeds"] == [17, 29, 43]


@pytest.mark.skipif(
    not (
        ROOT
        / "outputs/robotwin_mem_admission_v1/"
        "predictions_vlm_cover_blocks_hard.json"
    ).exists(),
    reason="confirmatory VLM control not evaluated",
)
def test_confirmatory_control_keeps_matched_memory_budgets_and_gpu_contract() -> None:
    for task_id in (
        "pick_the_unhidden_block",
        "pick_objects_in_order",
        "cover_blocks_hard",
    ):
        prediction = json.loads(
            (
                ROOT
                / "outputs/robotwin_mem_admission_v1"
                / f"predictions_vlm_{task_id}.json"
            ).read_text()
        )
        assert prediction["controller"]["cuda_visible_devices"] in {"0", "1", "2"}
        assert prediction["controller"]["cuda_visible_devices"] != "3"
        assert prediction["controller"]["event_labels_or_times_used_for_calibration"] is False
        assert prediction["controller"]["task_state_labels_used"] is False
        assert len(prediction["rows"]) == 27
        for row in prediction["rows"]:
            for condition in MATCHED_MEMORY_CONDITIONS:
                assert len(row["conditions"][condition]["memory_indices"]) == 4
