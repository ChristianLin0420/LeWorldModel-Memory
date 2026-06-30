#!/usr/bin/env python3
"""Closed-world and gate tests for the independent V14 screen auditor."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.audit_cf_ebo_v14_screen as audit
import scripts.run_cf_ebo_v14_screen as runner
from scripts.run_cf_ebo_v14_screen import SOURCE_PATHS
from scripts.test_analyze_cf_ebo_v14_screen import _rows


def _audit_rows():
    return [{
        "task": row["task"],
        "design": row["design"],
        "metrics": copy.deepcopy(row["metrics"]),
        "wandb_run_id": row["wandb"]["run_id"],
        "wandb_url": row["wandb"]["url"],
        "artifact_sha256": row["artifact_sha256"],
    } for row in _rows()]


def test_complete_negative_is_a_successful_audit_outcome() -> None:
    status, passed = audit.audit_status(
        artifact_integrity=True, analyzer_consistent=True, scientific_gate=False)
    assert status == "PASS_COMPLETE_NEGATIVE"
    assert passed
    assert audit.audit_exit_code({"passed": passed}) == 0


def test_complete_positive_is_distinct() -> None:
    status, passed = audit.audit_status(
        artifact_integrity=True, analyzer_consistent=True, scientific_gate=True)
    assert status == "PASS_COMPLETE"
    assert passed


def test_missing_protocol_and_artifacts_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        report = audit.audit(Path(directory))
    assert report["status"] == "FAIL_CLOSED"
    assert not report["passed"]
    assert not report["artifact_integrity_passed"]
    assert report["validated_cells"] == 0
    assert report["errors"]
    assert audit.audit_exit_code(report) == 2


def test_independent_positive_recomputation_covers_all_gates() -> None:
    gates = audit.recompute_gates(_audit_rows())
    assert gates["representation_passed"]
    assert gates["numerical_passed"]
    assert gates["external_passed"]
    assert gates["internal_passed"]
    assert gates["mechanism_passed"]
    assert gates["robustness_passed"]
    assert gates["complement_passed"]
    assert gates["convergence_passed"]
    assert gates["scientific_gate_passed"]


def test_padded_active_support_is_valid_but_leakage_fails() -> None:
    rows = _audit_rows()
    pendulum = next(row for row in rows if row["task"] == "pendulum.swingup"
                    and row["design"] == "cfebov14")
    assert pendulum["metrics"]["cf_ebo_core_energy_inactive_padding"] == 2
    assert audit.recompute_gates(rows)["numerical_passed"]
    pendulum["metrics"]["cf_ebo_core_energy_support_correction_max_abs"] = 1e-2
    gates = audit.recompute_gates(rows)
    assert not gates["numerical_passed"]
    assert any("support_correction" in value for value in gates["numerical_failures"])


def test_weaker_direction_and_robust_energy_are_fail_closed() -> None:
    rows = _audit_rows()
    candidate = next(row for row in rows if row["design"] == "cfebov14")
    candidate["metrics"]["cf_ebo_fit_computed_correction_reliability"] = .85
    gates = audit.recompute_gates(rows)
    assert not gates["numerical_passed"]

    rows = _audit_rows()
    candidate = next(row for row in rows if row["design"] == "cfebov14")
    candidate["metrics"]["cf_ebo_gaussian_noise_correction_energy_max"] = 1e6
    gates = audit.recompute_gates(rows)
    assert not gates["robustness_passed"]
    assert not gates["scientific_gate_passed"]


def test_auditor_source_manifest_matches_runner_freeze() -> None:
    assert set(audit.SOURCE_MANIFEST) == {str(path) for path in SOURCE_PATHS}
    assert "scripts/analyze_cf_ebo_v14_screen.py" in audit.SOURCE_MANIFEST
    assert "scripts/audit_cf_ebo_v14_screen.py" in audit.SOURCE_MANIFEST


def test_independent_command_reconstruction_matches_every_runner_token() -> None:
    root = Path("/tmp/frozen-v14-exact").resolve()
    for task in audit.TASKS:
        for design in audit.DESIGNS:
            expected = runner.train_command(
                str(runner.FROZEN_PYTHON), root, runner.DEFAULT_STUDY,
                audit.EPOCHS, task, design)
            assert audit.expected_train_command(
                root, audit.STUDY, audit.EPOCHS, task, design) == expected
    tampered = audit.expected_train_command(
        root, audit.STUDY, audit.EPOCHS, audit.TASKS[0], audit.DESIGNS[0])
    tampered[tampered.index("--encoder-layers") + 1] = "7"
    assert tampered != runner.train_command(
        str(runner.FROZEN_PYTHON), root, runner.DEFAULT_STUDY,
        audit.EPOCHS, audit.TASKS[0], audit.DESIGNS[0])


def test_runner_receipt_hash_is_bound_to_protocol_command() -> None:
    rows = _audit_rows()
    commands = {
        task: [["python", task, design] for design in audit.DESIGNS]
        for task in audit.TASKS
    }
    protocol = {
        "task_pinned_gpu": dict(zip(
            audit.TASKS, ("0", "1", "2", "3"), strict=True)),
        "commands": commands,
    }
    records = []
    for row in rows:
        task, design = row["task"], row["design"]
        records.append({
            "task": task, "design": design,
            "gpu": protocol["task_pinned_gpu"][task],
            "seed": audit.SEED, "seconds": 1.0,
            "command_sha256": audit.json_sha256(
                commands[task][audit.DESIGNS.index(design)]),
            "artifact_sha256": row["artifact_sha256"],
        })
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "screen_runs.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        audit.validate_runner(root, rows, protocol)
        records[-1]["command_sha256"] = "f" * 64
        path.write_text(json.dumps(records), encoding="utf-8")
        try:
            audit.validate_runner(root, rows, protocol)
        except audit.AuditFailure as exc:
            assert "runner command hash" in str(exc)
        else:
            raise AssertionError("tampered runner command hash was accepted")


def test_deep_receipt_equality_rejects_nested_or_tensor_drift() -> None:
    import torch

    first = {"fold": {"alpha": .5}, "tensor": torch.tensor([1.0, 2.0])}
    assert audit._deep_equal(first, copy.deepcopy(first))
    second = copy.deepcopy(first)
    second["fold"]["alpha"] = .6
    assert not audit._deep_equal(first, second)
    second = copy.deepcopy(first)
    second["tensor"][0] = 9.0
    assert not audit._deep_equal(first, second)


def main() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} V14 independent-audit tests passed.")


if __name__ == "__main__":
    main()
