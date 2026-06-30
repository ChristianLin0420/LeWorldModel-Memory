#!/usr/bin/env python3
"""Closed-world and gate tests for the independent V15 auditor."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.audit_cvpf_v15_screen as audit
import scripts.run_cvpf_v15_screen as runner
from scripts.test_analyze_cvpf_v15_screen import rows


def audit_rows() -> list[dict[str, object]]:
    return [{
        "task": row["task"],
        "design": row["design"],
        "metrics": copy.deepcopy(row["metrics"]),
        "wandb_run_id": row["wandb"]["run_id"],
        "wandb_url": row["wandb"]["url"],
        "artifact_sha256": row["artifact_sha256"],
    } for row in rows()]


def test_complete_positive_and_negative_statuses() -> None:
    assert audit.audit_status(
        artifact_integrity=True, analyzer_consistent=True,
        scientific_gate=True) == ("PASS_COMPLETE", True)
    assert audit.audit_status(
        artifact_integrity=True, analyzer_consistent=True,
        scientific_gate=False) == ("PASS_COMPLETE_NEGATIVE", True)
    assert audit.audit_status(
        artifact_integrity=False, analyzer_consistent=True,
        scientific_gate=False) == ("FAIL_CLOSED", False)


def test_missing_protocol_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        report = audit.audit(Path(directory))
    assert report["status"] == "FAIL_CLOSED"
    assert not report["passed"]
    assert report["validated_cells"] == 0
    assert report["errors"]
    assert audit.audit_exit_code(report) == 2


def test_independent_positive_recomputation_covers_every_gate() -> None:
    gates = audit.recompute_gates(audit_rows())
    for key in (
            "representation_passed", "structural_passed",
            "mode_gain_exact_ablation_passed", "baseline_passed",
            "direct_controls_passed", "active_identification_envelope_passed",
            "mechanism_passed", "convergence_passed", "scientific_gate_passed"):
        assert gates[key], key


def test_shift_bound_and_exact_modes_fail_closed() -> None:
    synthetic = audit_rows()
    candidate = next(row for row in synthetic if row["design"] == audit.CANDIDATE)
    candidate["metrics"]["cvpf_shift_closure_relative"] = 1.01
    gates = audit.recompute_gates(synthetic)
    assert not gates["structural_passed"]
    synthetic = audit_rows()
    candidate = next(row for row in synthetic if row["design"] == audit.CANDIDATE)
    candidate["metrics"][
        "cvpf_core_observation_deployed_to_fit_innovation_rms_ratio"] = .499
    gates = audit.recompute_gates(synthetic)
    assert not gates["structural_passed"]
    synthetic = audit_rows()
    norho = next(row for row in synthetic if row["design"] == "cvpfv15_norho")
    norho["metrics"]["cvpf_core_rho"] = 0.0
    gates = audit.recompute_gates(synthetic)
    assert not gates["mode_gain_exact_ablation_passed"]
    synthetic = audit_rows()
    anchor = next(row for row in synthetic if row["design"] == "cvpfv15_anchoronly")
    anchor["metrics"]["cvpf_exact_nocorrect"] = False
    assert not audit.recompute_gates(synthetic)["mode_gain_exact_ablation_passed"]


def test_each_baseline_and_control_is_conjunctive() -> None:
    synthetic = audit_rows()
    for row in synthetic:
        if row["design"] == "cfebov14_norisk":
            row["metrics"][audit.PRIMARY] = 1.01
    assert not audit.recompute_gates(synthetic)["baseline_passed"]
    synthetic = audit_rows()
    for row in synthetic:
        if row["design"] == "cvpfv15_noenvelope":
            row["metrics"][audit.PRIMARY] = 1.01
    gates = audit.recompute_gates(synthetic)
    assert not gates["direct_controls_passed"]
    assert not gates["active_identification_envelope_passed"]


def test_source_manifest_and_command_reconstruction_match_runner() -> None:
    assert set(audit.SOURCE_MANIFEST) == {str(path) for path in runner.SOURCE_PATHS}
    root = Path("/tmp/frozen-v15-exact").resolve()
    for task in audit.TASKS:
        for design in audit.DESIGNS:
            assert audit.expected_train_command(
                root, audit.STUDY, audit.EPOCHS, task, design) == runner.train_command(
                    str(runner.FROZEN_PYTHON), root, runner.DEFAULT_STUDY,
                    audit.EPOCHS, task, design)


def test_runner_receipt_is_bound_to_protocol_command() -> None:
    synthetic = audit_rows()
    commands = {
        task: [["python", task, design] for design in audit.DESIGNS]
        for task in audit.TASKS}
    protocol = {
        "task_pinned_gpu": dict(zip(
            audit.TASKS, ("0", "1", "2", "3"), strict=True)),
        "commands": commands,
    }
    records = []
    for row in synthetic:
        task, design = row["task"], row["design"]
        records.append({
            "task": task, "design": design,
            "gpu": protocol["task_pinned_gpu"][task], "seed": audit.SEED,
            "seconds": 1.0,
            "command_sha256": audit.json_sha256(
                commands[task][audit.DESIGNS.index(design)]),
            "artifact_sha256": row["artifact_sha256"],
        })
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "screen_runs.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        audit.validate_runner(root, synthetic, protocol)
        records[-1]["command_sha256"] = "f" * 64
        path.write_text(json.dumps(records), encoding="utf-8")
        try:
            audit.validate_runner(root, synthetic, protocol)
        except audit.AuditFailure as exc:
            assert "command hash" in str(exc)
        else:
            raise AssertionError("tampered command hash accepted")


def test_deep_receipt_equality_rejects_nested_and_tensor_drift() -> None:
    import torch

    first = {"fold": {"gain": .5}, "tensor": torch.tensor([1.0, 2.0])}
    assert audit._deep_equal(first, copy.deepcopy(first))
    second = copy.deepcopy(first)
    second["fold"]["gain"] = .6
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
    print(f"All {len(tests)} V15 independent-audit tests passed.")


if __name__ == "__main__":
    main()
