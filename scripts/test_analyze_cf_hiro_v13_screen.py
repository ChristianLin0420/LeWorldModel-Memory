#!/usr/bin/env python3
"""Decision and integrity-separation tests for the V13 screen analyzer."""

from __future__ import annotations

import copy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.analyze_cf_hiro_v13_screen as analysis
from scripts.train_cf_hiro_v13 import CF_HIRO_DESIGNS, CORE_MODES, DESIGNS


ACTION_DIMS = {
    "cartpole.swingup": 1, "fish.swim": 5,
    "pendulum.swingup": 1, "walker.walk": 6,
}


def _metrics(task: str, design: str, *, positive: bool = True):
    action_dim = ACTION_DIMS[task]
    candidate = design == "cfhirov13"
    if candidate:
        primary = 1.0 if positive else 3.0
    elif design in {"ssm", "hacssmv8", "kdiov11", "cfhirov13_noaction"}:
        primary = 2.0
    else:
        primary = 1.2
    values = {
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
    if design in CF_HIRO_DESIGNS:
        values.update({
            "fit_updates": 31,
            "memory_state_dim": float(min(23 * 128, 24 * action_dim)),
            "cf_hiro_fit_fit_index": 30,
            "cf_hiro_fit_fit_episode_count": 1200,
            "cf_hiro_fit_fit_length": 48,
            "cf_hiro_fit_markov_lag_count": 47,
            "cf_hiro_fit_even_episodes": 600,
            "cf_hiro_fit_odd_episodes": 600,
            "cf_hiro_fit_action_refit_lags": 47,
            "cf_hiro_fit_fit_uses_validation": False,
            "cf_hiro_fit_fit_gradient_active": False,
            "cf_hiro_core_gradient_parameter_count": 0,
            "cf_hiro_core_streaming_covariance_floats": 0,
            "cf_hiro_core_online_covariance_update": "none_fixed_steady_gain_mean_only",
            "cf_hiro_fit_fold_agreement_mode": (
                "unit" if design == "cfhirov13_noshrink" else "empirical_bayes"),
            "cf_hiro_fit_transition_deployment": (
                "triangular" if design == "cfhirov13_triangular" else "normal"),
            "cf_hiro_streaming_max_abs": 0.0,
            "cf_hiro_projector_algebra_max_abs": 1e-8,
            "cf_hiro_initial_reconstruction_max_abs": 1e-8,
            "cf_hiro_complement_dynamic_orthogonality_max_abs": 1e-8,
            "cf_hiro_core_steady_riccati_relative_residual": 1e-10,
            "cf_hiro_core_state_spectral_radius": .9,
            "cf_hiro_core_state_operator_norm": .9,
            "cf_hiro_core_state_is_real_normal_contraction":
                design != "cfhirov13_triangular",
            "cf_hiro_exact_noaction": True,
            "cf_hiro_exact_nocorrect": True,
        })
    if candidate:
        values.update({
            "cf_hiro_fit_held_fold_action_r2_even_to_odd": .1,
            "cf_hiro_fit_held_fold_action_r2_odd_to_even": .1,
            "cf_hiro_true_action_suffix_advantage": .01,
            "cf_hiro_action_pair_accuracy": .6,
            "cf_hiro_complement_anchor_rms": .1,
            "cf_hiro_dynamic_initial_rms": .1,
        })
    return values


def _rows(*, positive: bool):
    return [{
        "task": task,
        "design": design,
        "metrics": _metrics(task, design, positive=positive),
        "wandb": {"run_id": f"{task}-{design}", "url": "https://example.test/run"},
        "wandb_epoch_indices": list(range(1, 31)),
        "artifact_sha256": {"model.pt": "a"},
    } for task in analysis.TASKS for design in DESIGNS]


def test_complete_positive_authorizes_exact_72_cell_manifest() -> None:
    result = analysis.analyze(
        _rows(positive=True), [], epochs=30, study=analysis.DEFAULT_STUDY)
    assert result["artifact_integrity_passed"]
    assert result["scientific_gate_passed"]
    assert result["status"] == "SCREEN_PASS_100E_MANIFEST"
    assert analysis.analysis_exit_code(result) == 0
    assert len(analysis.CONTINUATION_DESIGNS) == 6
    assert analysis.CONTINUATION_SEEDS == (13002, 13003, 13004)
    assert 4 * 6 * 3 == 72


def test_complete_scientific_negative_exits_successfully() -> None:
    result = analysis.analyze(
        _rows(positive=False), [], epochs=30, study=analysis.DEFAULT_STUDY)
    assert result["artifact_integrity_passed"]
    assert not result["scientific_gate_passed"]
    assert result["status"] == "SCREEN_NO_GO"
    assert result["contingent_100e_manifest"] is None
    assert analysis.analysis_exit_code(result) == 0


def test_missing_artifact_is_separate_and_fails_closed() -> None:
    rows = _rows(positive=False)[:-1]
    result = analysis.analyze(
        rows, ["walker.walk/kdiov11: missing model.pt"],
        epochs=30, study=analysis.DEFAULT_STUDY)
    assert not result["artifact_integrity_passed"]
    assert result["representation_gate_passed"] is None
    assert result["status"] == "INCOMPLETE_OR_INVALID"
    assert analysis.analysis_exit_code(result) == 2


def main() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} V13 screen-analysis tests passed.")


if __name__ == "__main__":
    main()
