#!/usr/bin/env python3
"""Synthetic decision and integrity tests for the V15 screen analyzer."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.analyze_cvpf_v15_screen as analysis


ACTION_DIMS = {
    "cartpole.swingup": 1, "fish.swim": 5,
    "pendulum.swingup": 1, "walker.walk": 6,
}


def metrics(task: str, design: str, *, positive: bool = True) -> dict[str, object]:
    if design == analysis.CANDIDATE:
        primary = 1.0 if positive else 3.0
    elif design in analysis.BASELINES:
        primary = 2.0
    else:
        primary = 1.2
    result: dict[str, object] = {
        analysis.PRIMARY: primary,
        analysis.CLEAN: primary,
        "initial_encoder_integrator_probe_nmse": 2.0,
        "predictive_loss_convergence_relative_change": .01 if positive else -.01,
        "encoder_mean_channel_variance": .1,
        "encoder_covariance_effective_rank": 32.0,
        "encoder_singleton_max_abs": 0.0,
        "encoder_prefix_max_abs": 0.0,
        "action_dim": ACTION_DIMS[task],
    }
    if design in analysis.CVPF_DESIGNS:
        action = 0.0 if design in ("cvpfv15_noaction", "cvpfv15_anchoronly") else .8
        correction = 0.0 if design in (
            "cvpfv15_nocorrect", "cvpfv15_anchoronly") else .8
        risk = 1.0 if design == "cvpfv15_norisk" else .7
        rho = 1.0 if design == "cvpfv15_norho" else .6
        envelope = design != "cvpfv15_noenvelope"
        result.update({
            "fit_updates": 31,
            "cvpf_fit_fit_index": 30,
            "cvpf_fit_fit_episode_count": 1200,
            "cvpf_fit_fit_uses_validation": False,
            "cvpf_fit_fit_gradient_active": design != "cvpfv15_detachid",
            "cvpf_streaming_max_abs": 0.0,
            "cvpf_prefix_closure_max_abs": 0.0,
            "cvpf_shift_closure_relative": .8,
            "cvpf_core_observation_deployed_to_fit_innovation_rms_ratio": 1.0,
            "cvpf_core_action_gain": action,
            "cvpf_core_correction_gain": correction,
            "cvpf_core_risk_gain": risk,
            "cvpf_core_rho": rho,
            "cvpf_exact_nocorrect": design in (
                "cvpfv15_nocorrect", "cvpfv15_anchoronly"),
            "cvpf_exact_noaction": design in (
                "cvpfv15_noaction", "cvpfv15_anchoronly"),
            "cvpf_exact_norisk": design == "cvpfv15_norisk",
            "cvpf_exact_norho": design == "cvpfv15_norho",
            "cvpf_exact_anchoronly": design == "cvpfv15_anchoronly",
            "cvpf_identification_detached": design == "cvpfv15_detachid",
            "cvpf_envelope_active": envelope,
            "cvpf_envelope_weight": 1.0 if envelope else 0.0,
            "cvpf_action_crossfit_mean_gain": .1,
            "cvpf_correction_crossfit_mean_gain": .1,
            "cvpf_true_action_prior_advantage": .1,
            "cvpf_action_pair_accuracy": .6,
        })
    return result


def rows(*, positive: bool = True) -> list[dict[str, object]]:
    return [{
        "task": task,
        "design": design,
        "metrics": metrics(task, design, positive=positive),
        "wandb": {"run_id": f"{task}-{design}", "url": "https://example.test/run"},
        "wandb_epoch_indices": list(range(1, 31)),
        "artifact_sha256": {"model.pt": "a"},
    } for task in analysis.TASKS for design in analysis.DESIGNS]


def test_complete_positive_is_screen_go_without_launch() -> None:
    result = analysis.analyze(rows(), [])
    assert result["artifact_integrity_passed"]
    assert result["status"] == "SCREEN_GO"
    assert result["scientific_gate_passed"]
    assert result["baseline_gate_passed"]
    assert result["direct_control_gate_passed"]
    assert result["active_identification_envelope_gate"]["passed"]
    assert result["action_correction_suffix_gate"]["passed"]
    assert result["structural_integrity_gate"]["passed"]
    assert result["mode_gain_exact_ablation_gate"]["passed"]
    assert result["conditional_authorization_status"] == "AUTHORIZED_NOT_LAUNCHED"
    assert result["automatic_continuation_launch_performed"] is False
    assert analysis.analysis_exit_code(result) == 0


def test_norho_and_anchoronly_exact_semantics() -> None:
    result = analysis.analyze(rows(), [])
    assert result["mode_gain_exact_ablation_gate"]["passed"]
    broken = rows()
    norho = next(row for row in broken if row["design"] == "cvpfv15_norho")
    norho["metrics"]["cvpf_core_rho"] = 0.0
    result = analysis.analyze(broken, [])
    assert not result["mode_gain_exact_ablation_gate"]["passed"]
    broken = rows()
    anchor = next(row for row in broken if row["design"] == "cvpfv15_anchoronly")
    anchor["metrics"]["cvpf_exact_noaction"] = False
    result = analysis.analyze(broken, [])
    assert not result["mode_gain_exact_ablation_gate"]["passed"]


def test_projected_shift_uses_relative_zero_predictor_bound() -> None:
    valid = rows()
    for row in valid:
        if row["design"] in analysis.CVPF_DESIGNS:
            row["metrics"]["cvpf_shift_closure_relative"] = 1.0
    assert analysis.analyze(valid, [])["structural_integrity_gate"]["passed"]
    invalid = copy.deepcopy(valid)
    candidate = next(row for row in invalid if row["design"] == analysis.CANDIDATE)
    candidate["metrics"]["cvpf_shift_closure_relative"] = 1.001
    result = analysis.analyze(invalid, [])
    assert not result["structural_integrity_gate"]["passed"]
    assert not result["scientific_gate_passed"]
    invalid = copy.deepcopy(valid)
    candidate = next(row for row in invalid if row["design"] == analysis.CANDIDATE)
    candidate["metrics"][
        "cvpf_core_observation_deployed_to_fit_innovation_rms_ratio"] = 2.001
    result = analysis.analyze(invalid, [])
    assert not result["structural_integrity_gate"]["passed"]


def test_every_baseline_integrator_and_control_is_conjunctive() -> None:
    broken = rows()
    for row in broken:
        if row["design"] == "ssm":
            row["metrics"][analysis.PRIMARY] = 1.01
    result = analysis.analyze(broken, [])
    assert not result["baseline_gate_passed"]
    broken = rows()
    for row in broken:
        if row["design"] == "cvpfv15_detachid":
            row["metrics"][analysis.PRIMARY] = 1.01
    result = analysis.analyze(broken, [])
    assert not result["direct_control_gate_passed"]
    assert not result["active_identification_envelope_gate"]["passed"]
    broken = rows()
    for row in broken:
        if row["design"] == analysis.CANDIDATE:
            row["metrics"]["initial_encoder_integrator_probe_nmse"] = 1.01
    assert not analysis.analyze(broken, [])["baseline_gate_passed"]


def test_mechanisms_require_three_of_four_for_each_receipt() -> None:
    broken = rows()
    for task in analysis.TASKS[:2]:
        row = next(row for row in broken if row["task"] == task
                   and row["design"] == analysis.CANDIDATE)
        row["metrics"]["cvpf_true_action_prior_advantage"] = -1.0
    result = analysis.analyze(broken, [])
    assert result["action_correction_suffix_gate"]["passed_task_counts"][
        "suffix_advantage"] == 2
    assert not result["action_correction_suffix_gate"]["passed"]


def test_convergence_requires_signed_nonnegative_full_tasks() -> None:
    broken = rows()
    candidate = next(row for row in broken if row["design"] == analysis.CANDIDATE)
    candidate["metrics"]["predictive_loss_convergence_relative_change"] = -1e-6
    result = analysis.analyze(broken, [])
    assert not result["convergence_gate"]["passed"]
    assert result["status"] == "SCREEN_NO_GO"
    assert result["conditional_authorization_status"] == "CONDITIONAL_NOT_AUTHORIZED"


def test_complete_negative_is_successful_evidence() -> None:
    result = analysis.analyze(rows(positive=False), [])
    assert result["artifact_integrity_passed"]
    assert result["status"] == "SCREEN_NO_GO"
    assert not result["scientific_gate_passed"]
    assert analysis.analysis_exit_code(result) == 0


def test_missing_cell_fails_closed() -> None:
    result = analysis.analyze(rows()[:-1], ["missing model.pt"])
    assert result["status"] == "INCOMPLETE_OR_INVALID"
    assert not result["artifact_integrity_passed"]
    assert analysis.analysis_exit_code(result) == 2


def test_runner_receipt_binds_exact_command_hash() -> None:
    synthetic = rows()
    commands = {
        task: [["python", task, design] for design in analysis.DESIGNS]
        for task in analysis.TASKS}
    protocol = {
        "task_pinned_gpu": dict(zip(
            analysis.TASKS, ("0", "1", "2", "3"), strict=True)),
        "commands": commands,
    }
    records = []
    for row in synthetic:
        task, design = row["task"], row["design"]
        records.append({
            "task": task, "design": design,
            "gpu": protocol["task_pinned_gpu"][task], "seed": analysis.SEED,
            "seconds": 1.0,
            "command_sha256": analysis.json_sha256(
                commands[task][analysis.DESIGNS.index(design)]),
            "artifact_sha256": row["artifact_sha256"],
        })
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "screen_runs.json").write_text(json.dumps(records), encoding="utf-8")
        assert analysis.validate_runner_receipt(root, synthetic, protocol) == []
        records[0]["command_sha256"] = "0" * 64
        (root / "screen_runs.json").write_text(json.dumps(records), encoding="utf-8")
        assert any("command hash mismatch" in error for error in
                   analysis.validate_runner_receipt(root, synthetic, protocol))


def main() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} V15 screen-analysis tests passed.")


if __name__ == "__main__":
    main()
