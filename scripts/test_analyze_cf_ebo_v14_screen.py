#!/usr/bin/env python3
"""Decision, mechanism, and integrity tests for the V14 screen analyzer."""

from __future__ import annotations

import copy
import contextlib
import dataclasses
import io
import json
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.analyze_cf_ebo_v14_screen as analysis
import scripts.train_cf_ebo_v14 as train
from scripts.test_cf_ebo_v14_integration import _synthetic, _world
from scripts.train_cf_ebo_v14 import CF_EBO_DESIGNS, DESIGNS


ACTION_DIMS = {
    "cartpole.swingup": 1,
    "fish.swim": 5,
    "pendulum.swingup": 1,
    "walker.walk": 6,
}


def _metrics(task: str, design: str, *, positive: bool = True) -> dict[str, object]:
    action_dim = ACTION_DIMS[task]
    candidate = design == analysis.CANDIDATE
    if candidate:
        primary = 1.0 if positive else 3.0
    elif design in {"cfhirov13_nocorrect", "ssm", "hacssmv8", "kdiov11",
                    "cfebov14_noaction"}:
        primary = 2.0
    else:
        primary = 1.2
    values: dict[str, object] = {
        "heldout_prior_state_nmse": primary,
        "clean_prior_state_nmse": primary,
        "initial_encoder_integrator_probe_nmse": 2.0,
        "predictive_loss_convergence_relative_change": .01 if positive else -.01,
        "encoder_mean_channel_variance": .1,
        "encoder_covariance_effective_rank": 32.0,
        "encoder_singleton_max_abs": 0.0,
        "encoder_prefix_max_abs": 0.0,
        "action_dim": action_dim,
    }
    if design in CF_EBO_DESIGNS:
        mode = design.removeprefix("cfebov14_") if design != "cfebov14" else "full"
        alpha_b = 0.0 if mode == "noaction" else 1.0 if mode == "norisk" else .8
        alpha_k = 0.0 if mode == "nocorrect" else 1.0 if mode == "norisk" else .8
        codimension = 0 if task == "walker.walk" else 1
        state_dim = min(23 * 128, 24 * action_dim)
        active_rank = state_dim - (2 if task == "pendulum.swingup" else 0)
        values.update({
            "fit_updates": 31,
            "memory_state_dim": float(min(23 * 128, 24 * action_dim)),
            "cf_ebo_fit_fit_index": 30,
            "cf_ebo_fit_fit_episode_count": 1200,
            "cf_ebo_fit_fit_length": 48,
            "cf_ebo_fit_markov_lag_count": 47,
            "cf_ebo_fit_even_episodes": 600,
            "cf_ebo_fit_odd_episodes": 600,
            "cf_ebo_fit_action_even_action_refit_lags": 47,
            "cf_ebo_fit_action_odd_action_refit_lags": 47,
            "cf_ebo_fit_action_pooled_action_refit_lags": 47,
            "cf_ebo_fit_fit_uses_validation": False,
            "cf_ebo_fit_fit_gradient_active": False,
            "cf_ebo_fit_action_combination": "minimum_directional_positive_part_EB",
            "cf_ebo_fit_correction_combination": "minimum_directional_positive_part_EB",
            "cf_ebo_fit_action_first_direction_reliability": .8,
            "cf_ebo_fit_action_second_direction_reliability": .9,
            "cf_ebo_fit_action_combined_risk_reliability": .8,
            "cf_ebo_fit_computed_action_reliability": .8,
            "cf_ebo_fit_correction_first_direction_reliability": .8,
            "cf_ebo_fit_correction_second_direction_reliability": .85,
            "cf_ebo_fit_correction_combined_risk_reliability": .8,
            "cf_ebo_fit_computed_correction_reliability": .8,
            "cf_ebo_fit_energy_dissipativity_max_abs": 1e-10,
            "cf_ebo_fit_energy_lyapunov_relative_residual": 1e-12,
            "cf_ebo_fit_output_projector_idempotence_max_abs": 1e-10,
            "cf_ebo_fit_complement_projector_idempotence_max_abs": 1e-10,
            "cf_ebo_fit_direct_sum_projector_sum_max_abs": 1e-10,
            "cf_ebo_fit_complement_read_orthogonality_max_abs": 1e-10,
            "cf_ebo_fit_energy_support_projector_symmetry_max_abs": 0.0,
            "cf_ebo_fit_energy_support_projector_idempotence_max_abs": 0.0,
            "cf_ebo_fit_energy_state_rank": active_rank,
            "cf_ebo_fit_energy_inactive_padding": state_dim - active_rank,
            "cf_ebo_fit_energy_support_projector_rank": active_rank,
            "cf_ebo_streaming_max_abs": 0.0,
            "cf_ebo_initial_reconstruction_max_abs": 1e-8,
            "cf_ebo_core_energy_identity_max_abs": 1e-8,
            "cf_ebo_core_energy_support_rank": active_rank,
            "cf_ebo_core_energy_inactive_padding": state_dim - active_rank,
            "cf_ebo_core_energy_support_projector_symmetry_max_abs": 0.0,
            "cf_ebo_core_energy_support_projector_idempotence_max_abs": 0.0,
            "cf_ebo_core_energy_support_state_left_max_abs": 0.0,
            "cf_ebo_core_energy_support_state_right_max_abs": 0.0,
            "cf_ebo_core_energy_support_read_max_abs": 0.0,
            "cf_ebo_core_energy_support_action_max_abs": 0.0,
            "cf_ebo_core_energy_support_raw_action_max_abs": 0.0,
            "cf_ebo_core_energy_support_correction_max_abs": 0.0,
            "cf_ebo_core_energy_support_raw_correction_max_abs": 0.0,
            "cf_ebo_core_energy_support_initial_map_max_abs": 0.0,
            "cf_ebo_core_state_spectral_radius": .9,
            "cf_ebo_core_state_operator_norm": .95,
            "cf_ebo_core_deployed_correction_operator_norm": (
                2.0 if mode == "noenergycap" else 1.0),
            "cf_ebo_core_action_reliability": alpha_b,
            "cf_ebo_core_correction_reliability": alpha_k,
            "cf_ebo_core_innovation_rank": 4,
            "cf_ebo_core_gradient_parameter_count": 0,
            "cf_ebo_core_streaming_covariance_floats": 0,
            "cf_ebo_core_energy_cap_active": mode != "noenergycap",
            "cf_ebo_core_radial_gate_active": mode != "noradial",
            "cf_ebo_core_complement_codimension": codimension,
            "cf_ebo_core_complement_present": codimension > 0,
            "cf_ebo_exact_noaction": True,
            "cf_ebo_exact_nocorrect": True,
            "cf_ebo_fit_action_even_to_odd_mean_improvement": .1,
            "cf_ebo_fit_action_odd_to_even_mean_improvement": .1,
            "cf_ebo_fit_correction_even_to_odd_mean_improvement": .1,
            "cf_ebo_fit_correction_odd_to_even_mean_improvement": .1,
            "cf_ebo_true_action_suffix_advantage": .1,
            "cf_ebo_action_pair_accuracy": .6,
        })
        for fold in ("even", "odd", "pooled"):
            values.update({
                f"cf_ebo_fit_correction_{fold}_fit_innovation_score_mean": 4.0,
                f"cf_ebo_fit_correction_{fold}_fit_innovation_score_max": 12.0,
                f"cf_ebo_fit_correction_{fold}_fit_radial_gate_mean": .8,
                f"cf_ebo_fit_correction_{fold}_fit_radial_gate_min": .2,
                f"cf_ebo_fit_correction_{fold}_fit_radial_gate_max": 1.0,
            })
        bound = alpha_k * alpha_k * 4
        for condition in analysis.CONDITIONS:
            values.update({
                f"cf_ebo_{condition}_innovation_score_mean": (
                    2.0 if condition == "gaussian_noise" else 1.0),
                f"cf_ebo_{condition}_radial_gate_mean": (
                    .2 if condition == "gaussian_noise" else .8),
                f"cf_ebo_{condition}_correction_energy_max": (
                    0.0 if mode == "nocorrect" else min(2.0, bound)),
                f"cf_ebo_{condition}_evidence_samples": 100,
            })
    return values


def _rows(*, positive: bool = True):
    return [{
        "task": task,
        "design": design,
        "metrics": _metrics(task, design, positive=positive),
        "wandb": {"run_id": f"{task}-{design}", "url": "https://example.test/run"},
        "wandb_epoch_indices": list(range(1, 31)),
        "artifact_sha256": {"model.pt": "a"},
    } for task in analysis.TASKS for design in DESIGNS]


def test_complete_positive_authorizes_exact_96_cell_manifest() -> None:
    result = analysis.analyze(
        _rows(), [], epochs=30, study=analysis.DEFAULT_STUDY)
    assert result["artifact_integrity_passed"]
    assert result["representation_gate_passed"]
    assert result["numerical_gate"]["passed"]
    assert result["action_correction_mechanism_gate"]["passed"]
    assert result["robustness_gate"]["passed"]
    assert result["rank_aware_complement_gate"]["passed"]
    assert result["scientific_gate_passed"]
    assert result["status"] == "SCREEN_PASS_100E_MANIFEST"
    assert result["contingent_100e_manifest"] == "contingent_100e_launch_manifest.json"
    assert analysis.analysis_exit_code(result) == 0
    assert len(analysis.CONTINUATION_DESIGNS) == 8
    assert len(analysis.TASKS) * 8 * 3 == 96


def test_zero_codimension_is_valid_without_complement_energy() -> None:
    rows = _rows()
    walker = next(row for row in rows if row["task"] == "walker.walk"
                  and row["design"] == analysis.CANDIDATE)
    assert walker["metrics"]["cf_ebo_core_complement_codimension"] == 0
    result = analysis.analyze(rows, [], epochs=30, study=analysis.DEFAULT_STUDY)
    assert result["numerical_gate"]["passed"]
    assert result["rank_aware_complement_gate"]["passed"]


def test_risk_must_equal_weaker_direction() -> None:
    rows = _rows()
    candidate = next(row for row in rows if row["design"] == analysis.CANDIDATE)
    candidate["metrics"]["cf_ebo_fit_computed_action_reliability"] = .9
    result = analysis.analyze(rows, [], epochs=30, study=analysis.DEFAULT_STUDY)
    assert not result["numerical_gate"]["passed"]
    assert any("directional minimum" in value
               for value in result["numerical_gate"]["failures"])


def test_gaussian_shift_and_energy_bound_are_conjunctive() -> None:
    rows = _rows()
    for task in analysis.TASKS[:2]:
        candidate = next(row for row in rows if row["task"] == task
                         and row["design"] == analysis.CANDIDATE)
        candidate["metrics"]["cf_ebo_gaussian_noise_innovation_score_mean"] = .5
    result = analysis.analyze(rows, [], epochs=30, study=analysis.DEFAULT_STUDY)
    assert not result["robustness_gate"]["passed"]
    assert result["robustness_gate"]["distribution_shift_passed_tasks"] == 2

    rows = _rows()
    candidate = next(row for row in rows if row["design"] == analysis.CANDIDATE)
    candidate["metrics"]["cf_ebo_freeze_correction_energy_max"] = 99.0
    result = analysis.analyze(rows, [], epochs=30, study=analysis.DEFAULT_STUDY)
    assert not result["robustness_gate"]["passed"]
    assert not result["scientific_gate_passed"]


def test_complete_scientific_negative_exits_successfully() -> None:
    result = analysis.analyze(
        _rows(positive=False), [], epochs=30, study=analysis.DEFAULT_STUDY)
    assert result["artifact_integrity_passed"]
    assert not result["scientific_gate_passed"]
    assert result["status"] == "SCREEN_NO_GO"
    assert result["contingent_100e_manifest"] is None
    assert analysis.analysis_exit_code(result) == 0


def test_missing_artifact_is_separate_and_fails_closed() -> None:
    rows = _rows()[:-1]
    result = analysis.analyze(
        rows, ["walker.walk/kdiov11: missing model.pt"],
        epochs=30, study=analysis.DEFAULT_STUDY)
    assert not result["artifact_integrity_passed"]
    assert result["representation_gate_passed"] is None
    assert result["status"] == "INCOMPLETE_OR_INVALID"
    assert analysis.analysis_exit_code(result) == 2


def test_main_missing_or_malformed_protocol_writes_structured_exit2() -> None:
    for contents in (None, "{broken", "{}"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if contents is not None:
                (root / "screen_protocol.json").write_text(contents, encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                try:
                    analysis.main(["--root", str(root), "--write"])
                except SystemExit as exc:
                    assert exc.code == 2
                else:
                    raise AssertionError("invalid protocol did not exit 2")
            rendered = json.loads(output.getvalue())
            assert rendered["status"] == "INCOMPLETE_OR_INVALID"
            assert rendered["artifact_integrity_passed"] is False
            assert json.loads((root / "screen_analysis.json").read_text())["status"] \
                == "INCOMPLETE_OR_INVALID"
            assert json.loads((root / "screen_decision.json").read_text())[\
                "continue_to_100_epochs"] is False


def test_runner_receipt_authenticates_each_exact_command_hash() -> None:
    rows = _rows()
    commands = {
        task: [["python", task, design] for design in DESIGNS]
        for task in analysis.TASKS
    }
    protocol = {
        "task_pinned_gpu": dict(zip(
            analysis.TASKS, ("0", "1", "2", "3"), strict=True)),
        "commands": commands,
    }
    records = []
    for row in rows:
        task, design = row["task"], row["design"]
        records.append({
            "task": task,
            "design": design,
            "gpu": protocol["task_pinned_gpu"][task],
            "seed": analysis.SEED,
            "seconds": 1.0,
            "command_sha256": analysis._json_sha256(
                commands[task][DESIGNS.index(design)]),
            "artifact_sha256": row["artifact_sha256"],
        })
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "screen_runs.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        assert analysis.validate_runner_receipt(root, rows, protocol) == []
        records[0]["command_sha256"] = "0" * 64
        path.write_text(json.dumps(records), encoding="utf-8")
        errors = analysis.validate_runner_receipt(root, rows, protocol)
        assert any("command hash mismatch" in error for error in errors)


def test_checkpoint_validator_deeply_authenticates_extra_state_receipts() -> None:
    clean, observed, actions = _synthetic()
    fit = train._fit_candidate(clean, observed, actions, "cfebov14")
    fit = dataclasses.replace(
        fit, receipts={**fit.receipts, "fit_index": 1})
    model = train.CFEBOExperimentModel(
        _world("cfebov14", train._fit_state_dim(fit)))
    model.world.mem_cfebov14.install_fit(fit)
    model.world.mem_cfebov14.install_fit(fit)
    metrics = {
        **train.scalar_fit_receipts(fit.receipts),
        "fit_updates": 2,
    }
    payload = {
        "fit_history": [{"fit_index": 0}, {"fit_index": 1}],
        "final_operator_fit": train.operator_fit_payload(fit),
        "model_state_dict": model.state_dict(),
    }
    analysis._validate_candidate_checkpoint(payload, metrics, 1, "synthetic")
    corrupted = copy.deepcopy(payload)
    extra = corrupted["model_state_dict"][
        "world.mem_cfebov14._extra_state"]
    extra["fit_receipts"]["fit_index"] = 0
    try:
        analysis._validate_candidate_checkpoint(corrupted, metrics, 1, "synthetic")
    except analysis.IntegrityError as exc:
        assert "serialized fit receipts" in str(exc)
    else:
        raise AssertionError("corrupted serialized fit receipts were accepted")


def main() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} V14 screen-analysis tests passed.")


if __name__ == "__main__":
    main()
