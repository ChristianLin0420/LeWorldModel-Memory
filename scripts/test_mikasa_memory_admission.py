"""Contract tests for the canonical MIKASA admission harness."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from lewm.envs.mikasa_memory import (
    ButtonPressController,
    DEFAULT_MEMORY_BUDGET,
    GateThresholds,
    MATCHED_MEMORY_CONDITIONS,
    assert_matched_budget,
    decide_admission_gate,
    deterministic_episode_split,
    materialize_memory,
    recent_suffix_audit,
    select_memory_indices,
)


def test_episode_splits_are_seed_disjoint_and_deterministic() -> None:
    seeds = list(range(100, 350))
    first = deterministic_episode_split(
        "GatherAndRecall3-VLA-v0",
        seeds,
        validation_episodes=35,
        test_episodes=35,
    )
    second = deterministic_episode_split(
        "GatherAndRecall3-VLA-v0",
        seeds,
        validation_episodes=35,
        test_episodes=35,
    )
    assert first == second
    assert len(first["train"]) == 180
    assert len(first["validation"]) == 35
    assert len(first["test"]) == 35
    assert not (
        set(first["train"])
        & set(first["validation"])
        | set(first["train"])
        & set(first["test"])
        | set(first["validation"])
        & set(first["test"])
    )


def test_matched_memory_conditions_use_exact_budget() -> None:
    flash = np.zeros(120, dtype=bool)
    flash[31:42] = True
    selections = {
        condition: select_memory_indices(
            condition,
            len(flash),
            budget=DEFAULT_MEMORY_BUDGET,
            flash_mask=flash,
            random_seed=7,
        )
        for condition in MATCHED_MEMORY_CONDITIONS
    }
    assert_matched_budget(selections)
    assert all(len(indices) == DEFAULT_MEMORY_BUDGET for indices in selections.values())
    assert np.all(selections["no_memory"] == -1)
    assert selections["recent_only"].tolist() == list(range(112, 120))
    assert flash[selections["oracle_event"]].sum() == DEFAULT_MEMORY_BUDGET
    assert flash[selections["oracle_full_event"]].all()


def test_random_event_does_not_consult_oracle_mask() -> None:
    left = np.zeros(90, dtype=bool)
    right = np.ones(90, dtype=bool)
    selected_left = select_memory_indices(
        "random_event",
        90,
        flash_mask=left,
        random_seed=19,
    )
    selected_right = select_memory_indices(
        "random_event",
        90,
        flash_mask=right,
        random_seed=19,
    )
    assert np.array_equal(selected_left, selected_right)


def test_recent_suffix_audit_detects_any_flash_leakage() -> None:
    flash = np.zeros(50, dtype=bool)
    flash[10:18] = True
    clean = recent_suffix_audit(flash, np.arange(42, 50))
    leaked = recent_suffix_audit(flash, np.arange(14, 22))
    assert clean["passed"]
    assert clean["gap_frames"] == 24
    assert not leaked["passed"]
    assert leaked["flash_overlap_frames"] == 4


def test_null_memory_is_explicitly_masked() -> None:
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    selected, valid = materialize_memory(values, [-1, 1, -1, 4])
    assert valid.tolist() == [False, True, False, True]
    assert np.all(selected[~valid] == 0)
    assert np.array_equal(selected[valid], values[[1, 4]])


def test_gate_passes_only_with_paired_oracle_gain_and_clean_suffix() -> None:
    recent = np.zeros((3, 60), dtype=np.float64)
    recent[:, ::3] = 1.0
    no_memory = np.roll(recent, 1, axis=1)
    oracle = np.ones_like(recent)
    probe = recent.copy()
    decision = decide_admission_gate(
        recent_success=recent,
        oracle_success=oracle,
        no_memory_success=no_memory,
        recent_probe_accuracy=probe,
        thresholds=GateThresholds(),
    )
    assert decision.passed
    assert decision.metrics["oracle_minus_recent_ci_low"] > 0

    leaked_probe = np.ones_like(probe)
    failed = decide_admission_gate(
        recent_success=recent,
        oracle_success=oracle,
        no_memory_success=no_memory,
        recent_probe_accuracy=leaked_probe,
        thresholds=GateThresholds(),
    )
    assert not failed.passed
    assert failed.reasons == ("recent_suffix_probe",)


def test_controller_source_is_identical_for_every_condition() -> None:
    controller = ButtonPressController()
    digests = {
        condition: controller.digest() for condition in MATCHED_MEMORY_CONDITIONS
    }
    assert len(set(digests.values())) == 1


def test_model_contract_has_no_manual_crop_or_cue_input() -> None:
    module_path = (
        Path(__file__).resolve().parents[1] / "lewm/envs/mikasa_memory.py"
    )
    tree = ast.parse(module_path.read_text())
    forbidden_policy_names = {
        "lamp_crop",
        "cue_label",
        "cue_time",
        "known_cue_time",
        "saliency_mask",
        "future_frame",
    }
    policy_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "policy_view"
    )
    loaded = {
        node.id
        for node in ast.walk(policy_function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert not (loaded & forbidden_policy_names)
    literals = {
        node.value
        for node in ast.walk(policy_function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert not (literals & {"flash_color", "flash_active", "oracle_info"})
