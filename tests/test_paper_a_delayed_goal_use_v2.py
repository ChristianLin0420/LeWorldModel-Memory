from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.launch_paper_a_delayed_goal_use_v2 import build_jobs
from scripts.paper_a_delayed_goal_v2_spec import (
    CANDIDATE_IDS,
    DEFAULT_SPEC,
    DelayedGoalV2SpecError,
    controller_protocol,
    development_indices,
    load_controller_lock,
    load_locked_spec,
    load_v1_provenance,
    resolve_path,
    seal_json,
    select_candidate_id,
    validate_controller_lock_payload,
    validate_device,
)
from scripts.select_paper_a_delayed_goal_v2_controller import summarize_oracle


def _passing_results(spec: dict) -> list[dict]:
    results = []
    for candidate_id in CANDIDATE_IDS:
        tasks = {
            task: {
                "episodes": 240,
                "oracle_success": 0.95,
                "oracle_return": 0.94,
                "mean_distance": 0.1,
                "per_class_oracle_success": [0.9, 0.95, 0.95, 1.0],
            }
            for task in ("t1", "t3")
        }
        results.append({
            "id": candidate_id,
            "protocol": controller_protocol(spec, candidate_id),
            "tasks": tasks,
            "equal_task_oracle_success": 0.95,
            "equal_task_oracle_return": 0.94,
            "development_gate_pass": True,
        })
    return results


def _lock_payload(spec: dict) -> dict:
    results = _passing_results(spec)
    selected = CANDIDATE_IDS[0]
    return {
        "schema_version": 1,
        "study": spec["study"],
        "spec": spec["_spec_record"],
        "status": "controller_locked",
        "development_source": "parent training bank only",
        "validation_data_accessed": False,
        "validation_artifacts_absent_at_lock": True,
        "development_subsets": {
            task: {
                "episodes": 240,
                "class_counts": [60, 60, 60, 60],
                "index_sha256": spec["development"]["index_sha256"][task],
                "source_cache": spec["parent"]["train_caches"][task],
            }
            for task in ("t1", "t3")
        },
        "candidate_results": results,
        "selected_candidate_id": selected,
        "selected_protocol": controller_protocol(spec, selected),
        "v1_failure_provenance": spec["v1"]["provenance_manifest"],
        "v1_repairs_reused": True,
        "repair_retraining_performed": False,
    }


def test_v2_spec_authenticates_v1_failure_and_every_repair() -> None:
    spec = load_locked_spec()
    provenance = load_v1_provenance(spec)
    failure = provenance["v1_failure"]
    assert failure["observed_oracle_success"] == 0.8875
    assert failure["registered_oracle_success_min"] == 0.90
    assert failure["shortcut_gates_passed"] is True
    assert failure["valid_for_use_claim"] is False
    assert len(failure["completed_evaluations"]) == 10
    repairs = provenance["repair_checkpoints"]
    assert len(repairs) == 40
    assert len({record["path"] for record in repairs}) == 40
    assert spec["v1"]["repair_retraining_permitted"] is False


def test_v2_locked_spec_rejects_byte_change(tmp_path: Path) -> None:
    config = tmp_path / DEFAULT_SPEC.name
    lock = config.with_suffix(".sha256")
    shutil.copyfile(DEFAULT_SPEC, config)
    shutil.copyfile(DEFAULT_SPEC.with_suffix(".sha256"), lock)
    config.write_text(config.read_text() + "\n# changed\n")
    with pytest.raises(DelayedGoalV2SpecError, match="hash mismatch"):
        load_locked_spec(config, verify_artifacts=False, root=tmp_path)


def test_development_subsets_are_fixed_balanced_training_only() -> None:
    spec = load_locked_spec()
    for task in ("t1", "t3"):
        path = resolve_path(spec["parent"]["train_caches"][task]["path"])
        with np.load(path) as source:
            labels = np.asarray(source["xi"], dtype=np.int64)
        first = development_indices(labels, task, spec)
        second = development_indices(labels, task, spec)
        np.testing.assert_array_equal(first, second)
        assert first.shape == (240,)
        assert [int(np.sum(labels[first] == category))
                for category in range(4)] == [60, 60, 60, 60]
    assert spec["development"]["validation_data_permitted"] is False


def test_candidate_deck_changes_only_horizon_or_pd_gains() -> None:
    spec = load_locked_spec()
    fixed = spec["executed_choice"]["fixed"]
    protocols = [controller_protocol(spec, candidate)
                 for candidate in CANDIDATE_IDS]
    for protocol in protocols:
        assert protocol["joint_goals"] == fixed["joint_goals"]
        assert protocol["success_tolerance_radians"] == 0.35
        assert protocol["return_scale_radians"] == 0.50
    assert [protocol["executed_horizon"] for protocol in protocols] == [
        120, 160, 200, 160]
    assert protocols[0]["proportional_gain"] == 1.5
    assert protocols[0]["derivative_gain"] == 0.25


def test_selection_is_first_passing_candidate_and_fails_closed() -> None:
    spec = load_locked_spec()
    results = _passing_results(spec)
    assert select_candidate_id(spec, results) == CANDIDATE_IDS[0]
    results[0]["development_gate_pass"] = False
    assert select_candidate_id(spec, results) == CANDIDATE_IDS[1]
    for result in results:
        result["development_gate_pass"] = False
    assert select_candidate_id(spec, results) is None


def test_controller_lock_payload_cannot_select_unhealthy_or_later_candidate() -> None:
    spec = load_locked_spec()
    payload = _lock_payload(spec)
    validate_controller_lock_payload(payload, spec)
    payload["selected_candidate_id"] = CANDIDATE_IDS[1]
    payload["selected_protocol"] = controller_protocol(
        spec, CANDIDATE_IDS[1])
    with pytest.raises(DelayedGoalV2SpecError, match="not locked/healthy"):
        validate_controller_lock_payload(payload, spec)


def test_controller_lock_round_trip_is_hash_sealed(tmp_path: Path) -> None:
    spec = load_locked_spec()
    spec = dict(spec)
    spec["output"] = dict(spec["output"])
    spec["executed_choice"] = dict(spec["executed_choice"])
    spec["executed_choice"]["lock"] = {
        "path": "outputs/v2/controller.lock.json",
        "sha256_path": "outputs/v2/controller.lock.sha256",
        "must_precede_validation": True,
        "refuse_overwrite": True,
    }
    payload = _lock_payload(spec)
    path = tmp_path / "outputs/v2/controller.lock.json"
    sidecar = tmp_path / "outputs/v2/controller.lock.sha256"
    seal_json(path, sidecar, payload)
    loaded, record = load_controller_lock(spec, root=tmp_path)
    assert loaded == payload
    assert len(record["sha256"]) == 64
    with pytest.raises(FileExistsError, match="overwrite"):
        seal_json(path, sidecar, payload)


@pytest.mark.parametrize("device", ["cuda:0", "cuda:3"])
def test_v2_forbidden_devices_fail_before_job_creation(device: str) -> None:
    spec = load_locked_spec()
    with pytest.raises(DelayedGoalV2SpecError, match="forbidden"):
        validate_device(spec, device)
    with pytest.raises(DelayedGoalV2SpecError, match="forbidden"):
        build_jobs(spec, "evaluate", device, DEFAULT_SPEC)


def test_v2_jobs_have_no_repair_training_and_use_new_root() -> None:
    spec = load_locked_spec()
    expected = {"controller-select": 1, "evaluate": 10, "aggregate": 1}
    v1_root = resolve_path(spec["v1"]["output_root"])
    v2_root = resolve_path(spec["output"]["root"])
    for wave, count in expected.items():
        jobs = build_jobs(spec, wave, "cuda:2", DEFAULT_SPEC)
        assert len(jobs) == count
        assert all(v2_root in job.done_file.parents for job in jobs)
        assert all(v1_root not in job.done_file.parents for job in jobs)
        command = "\n".join(" ".join(job.command) for job in jobs)
        assert "train_paper_a_delayed_goal_repair.py" not in command
        assert "--execute" in command
    eval_names = {job.name for job in build_jobs(
        spec, "evaluate", "cuda:1", DEFAULT_SPEC)}
    assert "transient-marker-recall_checkpoint-seed-0" in eval_names
    assert "drifting-color-recall_checkpoint-seed-4" in eval_names


def test_validation_is_forbidden_before_controller_lock(tmp_path: Path) -> None:
    spec = load_locked_spec()
    spec = dict(spec)
    spec["executed_choice"] = dict(spec["executed_choice"])
    spec["executed_choice"]["lock"] = {
        "path": "missing/controller.lock.json",
        "sha256_path": "missing/controller.lock.sha256",
        "must_precede_validation": True,
        "refuse_overwrite": True,
    }
    with pytest.raises(DelayedGoalV2SpecError, match="forbidden before"):
        load_controller_lock(spec, root=tmp_path)
    assert spec["executed_choice"]["validation_oracle_success_min"] == 0.90
    assert spec["controls"]["shortcut_accuracy_max"] == 0.35


def test_oracle_summary_reports_per_class_health() -> None:
    labels = np.repeat(np.arange(4), 2)
    execution = {
        "success": np.asarray([1, 1, 1, 0, 1, 1, 0, 0], dtype=bool),
        "return": np.asarray([.9, .8, .7, .6, .9, .9, .2, .3]),
        "distance": np.asarray([.1, .2, .2, .4, .1, .1, .8, .7]),
    }
    result = summarize_oracle(execution, labels)
    assert result["oracle_success"] == 0.625
    assert result["per_class_oracle_success"] == [1.0, 0.5, 1.0, 0.0]
