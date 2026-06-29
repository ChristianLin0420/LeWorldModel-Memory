#!/usr/bin/env python3
"""Focused trainer, protocol, and analyzer contracts for KDIO-v11."""

from __future__ import annotations

import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.analyze_hacssm_v11 as analyzer
import scripts.run_hacssm_v11 as runner
from scripts.train_hacssm_v11 import (
    CALIBRATED_DESIGNS, DESIGNS, HISTORY_KEYS, SUFFIX_DESIGNS,
    _clean_calibration_priors,
    build_model, compute_v11_losses, memory_representations,
)


def _args(design: str) -> Namespace:
    return Namespace(
        memory_mode=design, img_size=16, patch_size=8, embed_dim=8,
        encoder_layers=1, encoder_heads=2, predictor_layers=1, predictor_heads=2,
        history_len=2, dropout=0.0, sigreg_lambda=0.1, sigreg_projections=8,
        seed=17,
    )


def _batch():
    generator = torch.Generator().manual_seed(111)
    observed = torch.rand(3, 5, 3, 16, 16, generator=generator)
    clean = torch.rand(3, 5, 3, 16, 16, generator=generator)
    actions = torch.rand(3, 4, 2, generator=generator) * 2 - 1
    return observed, clean, actions


def test_objective_algebra_suffix_and_action_swap_contracts() -> None:
    observed, clean, actions = _batch()
    for design in DESIGNS:
        model = build_model(
            _args(design), 2, np.zeros(2, np.float32), np.ones(2, np.float32))
        losses = compute_v11_losses(model, observed, clean, actions, design)
        assert set(losses) == set(HISTORY_KEYS)
        assert torch.allclose(
            losses["predictive_loss"],
            .5 * (losses["context_loss"] + losses["suffix_loss"]))
        expected = sum(losses[key] for key in (
            "predictive_loss", "action_swap_loss", "variance_loss", "covariance_loss"))
        assert torch.allclose(losses["loss"], expected)
        if design in CALIBRATED_DESIGNS:
            assert losses["calibration_applicable"].item() == 1
            assert torch.isfinite(losses["calibration_nll"])
        else:
            assert losses["calibration_applicable"].item() == 0
            assert losses["calibration_nll"].item() == 0
        if design in SUFFIX_DESIGNS:
            assert losses["suffix_applicable"].item() == 1
            wanted = 1 if design == "kdiov11_h1" else actions.shape[1]
            assert losses["suffix_horizons"].item() == wanted
            assert losses["action_swap_applicable"].item() == 1
            assert losses["action_swap_horizons"].item() == wanted
            assert torch.allclose(
                losses["action_swap_advantage"],
                losses["action_swap_negative_energy"]
                - losses["action_swap_positive_energy"])
            assert 0 <= losses["action_swap_pair_accuracy"].item() <= 1
            if design == "kdiov11_noactionswap":
                assert losses["action_swap_loss"].item() == 0
                assert losses["action_swap_diagnostic_loss"].item() > 0
            else:
                assert torch.equal(
                    losses["action_swap_loss"], losses["action_swap_diagnostic_loss"])
        else:
            assert losses["suffix_loss"] is losses["context_loss"]
            assert losses["suffix_applicable"].item() == 0
            assert losses["suffix_horizons"].item() == 0
            assert losses["action_swap_applicable"].item() == 0
            assert losses["action_swap_horizons"].item() == 0
            assert losses["action_swap_loss"].item() == 0
            assert losses["action_swap_diagnostic_loss"].item() == 0
        if design == "kdiov11_nosuffix":
            assert model.world.memory_impl == "kdiov11"
            assert model.world.mem_kdiov11.mode == "full"
        assert not hasattr(model, "inverse_head")
        assert not any(key.startswith("inverse_head.") for key in model.state_dict())
        losses["loss"].backward()


def test_no_train_inverse_head_and_suffix_uses_observed_anchor() -> None:
    observed, clean, actions = _batch()
    torch.manual_seed(711)
    model = build_model(
        _args("kdiov11"), 2, np.zeros(2, np.float32), np.ones(2, np.float32))
    assert tuple(model._modules) == ("world",)
    first = compute_v11_losses(model, observed, clean, actions, "kdiov11")
    changed = observed.clone()
    changed[:, 2] = 1.0 - changed[:, 2]
    second = compute_v11_losses(model, changed, clean, actions, "kdiov11")
    # Clean targets/actions are identical; only the deployed posterior anchor changed.
    assert not torch.allclose(first["suffix_loss"], second["suffix_loss"])


def test_action_frame_initialization_is_exactly_paired_across_kdio_designs() -> None:
    reference_action = None
    for index, design in enumerate(DESIGNS):
        # Deliberately perturb the ambient RNG before each architecture construction.
        torch.manual_seed(9_000 + index)
        model = build_model(
            _args(design), 2, np.zeros(2, np.float32), np.ones(2, np.float32))
        assert not any(key.startswith("inverse_head.") for key in model.state_dict())
        if design.startswith("kdiov11"):
            action = model.world.mem_kdiov11.W_a.weight.detach().clone()
            assert model.world.mem_kdiov11.log_action_scale.item() == 0.0
            if reference_action is None:
                reference_action = action
            else:
                assert torch.equal(action, reference_action)


def test_primary_prior_is_strictly_preobservation() -> None:
    _observed, clean, actions = _batch()
    for design in DESIGNS:
        model = build_model(
            _args(design), 2, np.zeros(2, np.float32), np.ones(2, np.float32))
        model.eval()
        z = model.world.encode(clean)
        changed = z.clone()
        changed[:, 3] = changed[:, 3] + torch.linspace(1, 2, z.shape[-1])
        first = memory_representations(model, z, actions)
        second = memory_representations(model, changed, actions)
        # The t=3 prior has consumed only t<=2 observations and action_2.
        assert torch.allclose(first["prior"][:, 3], second["prior"][:, 3], atol=1e-6)
        # The posterior is allowed (and expected) to consume z_3.
        assert not torch.allclose(first["posterior"][:, 3], second["posterior"][:, 3])


def test_clean_calibration_path_is_precision_independent_and_reliability_open() -> None:
    _observed, clean, actions = _batch()
    model = build_model(
        _args("kdiov11"), 2, np.zeros(2, np.float32), np.ones(2, np.float32))
    clean_z = model.world.encode(clean)
    first = _clean_calibration_priors(model, clean_z, actions)
    memory = model.world.mem_kdiov11
    with torch.no_grad():
        memory.innovation_precision_packed.normal_()
        memory.clean_innovation_mean.normal_()
    second = _clean_calibration_priors(model, clean_z, actions)
    assert torch.equal(first, second)
    _, details = memory(
        clean_z, actions, reliability_override=1.0, return_details=True)
    assert torch.equal(first, details["q_priors"][:, 1:])
    assert torch.equal(details["reliability"], torch.ones_like(details["reliability"]))


def test_ssm_prior_retains_native_affine_bias_and_alignment() -> None:
    _observed, clean, actions = _batch()
    model = build_model(
        _args("ssm"), 2, np.zeros(2, np.float32), np.ones(2, np.float32))
    with torch.no_grad():
        model.world.mem_ssm.in_proj.bias.copy_(torch.linspace(-.2, .3, 8))
    z = model.world.encode(clean)
    representations = memory_representations(model, z, actions)
    posterior = representations["details"]["states"]
    decay = torch.sigmoid(model.world.mem_ssm.raw_decay)
    expected_t3 = ((1 - decay) * posterior[:, 2]
                   + decay * model.world.mem_ssm.in_proj.bias)
    assert torch.allclose(
        representations["details"]["priors"][:, 3], expected_t3, atol=1e-7)
    assert torch.allclose(
        representations["prior"][:, 3],
        expected_t3 * torch.rsqrt(expected_t3.square().mean(-1, keepdim=True) + 1e-6),
        atol=1e-6)


def test_runner_grid_args_and_four_gpu_contract() -> None:
    assert tuple(runner.DESIGNS) == tuple(DESIGNS)
    assert len(DESIGNS) == 16
    assert "kdiov11_unconstrained" in DESIGNS
    assert "kdiov11_fixedscale" in DESIGNS
    assert "kdiov11_noactionswap" in DESIGNS
    assert "kdiov11_noinverse" not in DESIGNS
    assert len(runner.PILOT_JOBS) == 240
    assert len(runner.COMPLETION_JOBS) == 160
    assert len(runner.ALL_JOBS) == 400
    assert len({job.run_name for job in runner.ALL_JOBS}) == 400
    args = runner.expected_args(runner.ALL_JOBS[0])
    assert args["eval_target_key"] == "task_observation"
    assert args["corruption_seed"] == 11_012
    assert args["batch_size"] == 64
    assert runner.COMMON["innovation_calibration_gradient_active"] is False
    assert runner.COMMON["calibration_nll_optimized"] is False
    assert "identity_and_diagonal_controls" in runner.COMMON[
        "innovation_calibration_family"]
    assert len(runner.observer_metric_names()) == 725
    assert len(set(runner.observer_metric_names())) == 725
    contract = runner.memory_contract()
    gauge = 6 * 7 // 2
    assert contract["stiefel_gauge_dimension"] == gauge
    assert contract["kdio_mode_by_design"]["kdiov11_unconstrained"] == "unconstrained"
    assert contract["kdio_mode_by_design"]["kdiov11_fixedscale"] == "fixedscale"
    assert contract["kdio_nominal_optimizer_scalars"] == 17_796
    assert contract["kdio_nominal_fitted_oas_scalars"] == 8_255
    assert contract["kdio_nominal_total_memory_scalars"] == 26_051
    effective = contract["kdio_functionally_effective_optimizer_dof_by_design"]
    assert effective["kdiov11"] == 17_796 - gauge
    assert effective["kdiov11_unconstrained"] == 17_795
    assert effective["kdiov11_fixedscale"] == 17_796 - gauge - 1
    assert effective["kdiov11_noaction"] == 17_796 - 6 * 128 - 1
    active_fitted = contract["kdio_functionally_active_fitted_scalars_by_design"]
    assert active_fitted["kdiov11"] == 8_255
    assert active_fitted["kdiov11_diagonal"] == 254
    assert active_fitted["kdiov11_nocalibration"] == 0
    assert active_fitted["kdiov11_noreliability"] == 0
    assert contract["trainable_inverse_head_parameters"] == 0
    assert runner.build_parser().parse_args([]).workers == 4
    assert runner.build_parser().parse_args([]).gpus == ("0", "1", "2", "3")


def test_runner_accepts_only_the_exact_nosuffix_history_receipt() -> None:
    values = {
        "loss": .7, "predictive_loss": .5, "context_loss": .5,
        "suffix_loss": .5, "action_swap_loss": 0.0,
        "action_swap_diagnostic_loss": 0.0,
        "action_swap_positive_energy": 0.0, "action_swap_negative_energy": 0.0,
        "action_swap_advantage": 0.0, "action_swap_pair_accuracy": 0.0,
        "action_swap_applicable": 0.0, "action_swap_horizons": 0.0,
        "calibration_nll": .05, "calibration_applicable": 1.0,
        "variance_loss": .1, "covariance_loss": .1,
        "suffix_applicable": 0.0, "suffix_horizons": 0.0,
    }
    history = [
        {"epoch": epoch, "epoch_seconds": 1.0,
         "train": dict(values), "val": dict(values)}
        for epoch in range(1, runner.COMMON["epochs"] + 1)
    ]
    job = Namespace(run_name="nosuffix-contract", design="kdiov11_nosuffix")
    runner.validate_history(history, job)
    history[0]["train"]["suffix_applicable"] = 1.0
    try:
        runner.validate_history(history, job)
    except runner.RunnerError:
        pass
    else:
        raise AssertionError("runner accepted an applicable suffix for kdiov11_nosuffix")

    # The calibration NLL is a diagnostic: adding it to the optimized total must fail.
    history[0]["train"]["suffix_applicable"] = 0.0
    history[0]["train"]["loss"] += history[0]["train"]["calibration_nll"]
    try:
        runner.validate_history(history, job)
    except runner.RunnerError:
        pass
    else:
        raise AssertionError("runner accepted calibration NLL in the optimized total")


def test_runner_accepts_negative_gaussian_calibration_nll() -> None:
    values = {
        "loss": .1, "predictive_loss": .1, "context_loss": .1,
        "suffix_loss": .1, "action_swap_loss": 0.0,
        "action_swap_diagnostic_loss": 0.0,
        "action_swap_positive_energy": 0.0, "action_swap_negative_energy": 0.0,
        "action_swap_advantage": 0.0, "action_swap_pair_accuracy": 0.0,
        "action_swap_applicable": 0.0, "action_swap_horizons": 0.0,
        "calibration_nll": -.3, "calibration_applicable": 1.0,
        "variance_loss": 0.0, "covariance_loss": 0.0,
        "suffix_applicable": 0.0, "suffix_horizons": 0.0,
    }
    history = [
        {"epoch": epoch, "epoch_seconds": 1.0,
         "train": dict(values), "val": dict(values)}
        for epoch in range(1, runner.COMMON["epochs"] + 1)
    ]
    runner.validate_history(
        history, Namespace(run_name="negative-nll", design="kdiov11_nosuffix"))


def test_runner_noactionswap_keeps_diagnostic_but_removes_optimized_asr() -> None:
    values = {
        "loss": .7, "predictive_loss": .5, "context_loss": .5,
        "suffix_loss": .5, "action_swap_loss": 0.0,
        "action_swap_diagnostic_loss": .4,
        "action_swap_positive_energy": .5, "action_swap_negative_energy": .6,
        "action_swap_advantage": .1, "action_swap_pair_accuracy": .6,
        "action_swap_applicable": 1.0, "action_swap_horizons": 47.0,
        "calibration_nll": .05, "calibration_applicable": 1.0,
        "variance_loss": .1, "covariance_loss": .1,
        "suffix_applicable": 1.0, "suffix_horizons": 47.0,
    }
    history = [
        {"epoch": epoch, "epoch_seconds": 1.0,
         "train": dict(values), "val": dict(values)}
        for epoch in range(1, runner.COMMON["epochs"] + 1)
    ]
    job = Namespace(run_name="noactionswap", design="kdiov11_noactionswap")
    runner.validate_history(history, job)
    history[0]["train"]["action_swap_loss"] = .4
    history[0]["train"]["loss"] = 1.1
    try:
        runner.validate_history(history, job)
    except runner.RunnerError:
        pass
    else:
        raise AssertionError("runner accepted active ASR in noactionswap")


def test_runner_rejects_nonzero_inapplicable_calibration_nll() -> None:
    values = {
        "loss": .3, "predictive_loss": .1, "context_loss": .1,
        "suffix_loss": .1, "action_swap_loss": .2,
        "action_swap_diagnostic_loss": .2,
        "action_swap_positive_energy": .5, "action_swap_negative_energy": .6,
        "action_swap_advantage": .1, "action_swap_pair_accuracy": .6,
        "action_swap_applicable": 1.0, "action_swap_horizons": 47.0,
        "calibration_nll": .2, "calibration_applicable": 0.0,
        "variance_loss": 0.0, "covariance_loss": 0.0,
        "suffix_applicable": 1.0, "suffix_horizons": 47.0,
    }
    history = [
        {"epoch": epoch, "epoch_seconds": 1.0,
         "train": dict(values), "val": dict(values)}
        for epoch in range(1, runner.COMMON["epochs"] + 1)
    ]
    try:
        runner.validate_history(
            history, Namespace(run_name="nocalibration", design="kdiov11_nocalibration"))
    except runner.RunnerError:
        pass
    else:
        raise AssertionError("runner accepted a nonzero inapplicable calibration diagnostic")


def _synthetic_rows(seeds, candidate=.75):
    rows = []
    for env_index, environment in enumerate(analyzer.ENVIRONMENTS):
        for design in analyzer.DESIGNS:
            for seed in seeds:
                primary = candidate if design == analyzer.CANDIDATE else 1.0
                run = f"lewm-{environment}-{design}-s{seed}"
                row = {"run": run, "env": environment, "design": design, "seed": seed,
                       analyzer.PRIMARY: primary, analyzer.CLEAN_METRIC: primary}
                row["action_only_integrator_probe_nmse"] = 1.0
                row["initial_encoder_integrator_probe_nmse"] = 1.0
                row["inverse_action_probe_output_dim"] = 2.0
                for split in ("train", "val"):
                    row[f"final_{split}_action_swap_loss"] = .5
                    row[f"final_{split}_action_swap_diagnostic_loss"] = .5
                    row[f"final_{split}_action_swap_advantage"] = .1
                    row[f"final_{split}_action_swap_pair_accuracy"] = .6
                for key in analyzer.ROW_METRICS:
                    row.setdefault(
                        key, 1e-8 if "error" in key or "max_abs" in key else .5)
                for key in analyzer.KDIO_DIAGNOSTIC_METRICS:
                    row[key] = (
                        1e-8 if "error" in key or "violation" in key else .5
                    ) if design in analyzer.KDIO_DESIGNS else ""
                for key in analyzer.KDIO_CHECKPOINT_DIAGNOSTICS:
                    row[key] = .5 if design in analyzer.KDIO_DESIGNS else ""
                if design in analyzer.KDIO_DESIGNS:
                    row["action_parameter_checkpoint_numerical_rank"] = 2.0
                    row["memory_calibration_updates"] = 100.0
                    row["memory_calibration_samples"] = 56_400.0
                    row["memory_innovation_precision_singular_min"] = .5
                    row["memory_innovation_precision_singular_max"] = 1.0
                    row["memory_innovation_precision_condition"] = 2.0
                    row["memory_calibration_covariance_condition"] = 4.0
                    row["memory_action_transport"] = (
                        0.0 if design == "kdiov11_noaction" else 1.0)
                    row["memory_action_scale"] = 1.0
                    row["memory_action_log_scale"] = 0.0
                    row["memory_action_scale_parameter_retained"] = 1.0
                    row["memory_action_scale_gradient_active"] = float(
                        design not in {"kdiov11_fixedscale", "kdiov11_noaction"})
                    row["memory_action_kick_norm"] = math.sqrt(2.0)
                    row["memory_action_raw_norm"] = math.sqrt(2.0)
                    row["memory_action_direction_norm"] = math.sqrt(2.0)
                    row["memory_action_effective_norm"] = math.sqrt(2.0)
                    row["memory_action_parameter_norm"] = math.sqrt(2.0)
                    row["memory_action_parameter_singular_min"] = 1.0
                    row["memory_action_parameter_singular_max"] = 1.0
                    row["memory_action_parameter_condition"] = 1.0
                    row["memory_action_frame_gram_error"] = 0.0
                    row["memory_action_frame_singular_min"] = 1.0
                    row["memory_action_frame_singular_max"] = 1.0
                    row["memory_action_frame_condition"] = 1.0
                    row["memory_action_frame_constrained"] = float(
                        design != "kdiov11_unconstrained")
                    row["memory_action_direction_gram_error"] = 0.0
                    row["memory_action_direction_singular_min"] = 1.0
                    row["memory_action_direction_singular_max"] = 1.0
                    row["memory_action_direction_condition"] = 1.0
                    row["memory_nominal_optimizer_scalars"] = 17_796.0
                    row["memory_fitted_memory_scalars"] = 8_255.0
                    row["memory_total_memory_scalars"] = 26_051.0
                    row["memory_functionally_effective_optimizer_dof"] = 17_775.0
                    row["memory_functionally_active_fitted_scalars"] = 8_255.0
                    row["memory_functionally_effective_plus_fitted_scalars"] = 26_030.0
                    row["action_scale_checkpoint"] = 1.0
                    row["action_direction_checkpoint_norm"] = math.sqrt(2.0)
                    row["action_direction_checkpoint_gram_error"] = 0.0
                    row["action_direction_checkpoint_singular_min"] = 1.0
                    row["action_direction_checkpoint_singular_max"] = 1.0
                    row["action_direction_checkpoint_condition"] = 1.0
                    row["action_frame_checkpoint_norm"] = math.sqrt(2.0)
                    row["action_frame_checkpoint_gram_error"] = 0.0
                    row["action_frame_checkpoint_singular_min"] = 1.0
                    row["action_frame_checkpoint_singular_max"] = 1.0
                    row["action_frame_checkpoint_condition"] = 1.0
                    row["kdio_action_swap_pair_accuracy"] = .6
                    for horizon in (1, 4, 8, 16, 47):
                        row[f"kdio_true_action_advantage_h{horizon}"] = .1
                    if design == "kdiov11_noaction":
                        for key in analyzer.KDIO_ACTION_DIAGNOSTIC_METRICS:
                            row[key] = 0.0
                        row["kdio_true_action_one_step_mse"] = .5
                        row["kdio_shuffled_action_one_step_mse"] = .5
                        row["kdio_true_action_suffix_mse"] = .5
                        row["kdio_shuffled_action_suffix_mse"] = .5
                        row["kdio_action_swap_pair_accuracy"] = .5
                        for split in ("train", "val"):
                            row[f"final_{split}_action_swap_loss"] = math.log(2.0)
                            row[f"final_{split}_action_swap_diagnostic_loss"] = math.log(2.0)
                            row[f"final_{split}_action_swap_advantage"] = 0.0
                            row[f"final_{split}_action_swap_pair_accuracy"] = .5
                    for dataset in analyzer.OBSERVER_DATASETS:
                        row[f"{dataset}_observer_ordered_gain_violation_max"] = 0.0
                row["encoder_mean_channel_variance"] = .5
                row["encoder_covariance_effective_rank"] = 32.0
                row["prior_probe_ceiling_state_nmse"] = .8
                row["inverse_action_r2"] = .5
                rows.append(row)
    return rows


def _valid_kdio_receipts(action_dim: int = 2, design: str = "kdiov11"):
    metrics = {key: 0.5 for key in runner.CALIBRATION_MEMORY_METRICS}
    metrics.update({key: 0.5 for key in runner.KDIO_MECHANISM_METRICS})
    metrics.update({key: 0.5 for key in runner.KDIO_ACTION_DIAGNOSTIC_METRICS})
    dimension = runner.COMMON["embed_dim"]
    gradient = dimension**2 + action_dim * dimension + 5 * dimension + 4
    fitted = dimension * (dimension - 1) // 2 + dimension - 1
    action_parameter = torch.eye(dimension, action_dim)
    action_parameter[:, 0].mul_(2.0)
    log_action_scale = torch.tensor(math.log(2.0) if design == "kdiov11_fixedscale"
                                    else math.log(1.5))
    action_scale = torch.tensor(1.0) if design == "kdiov11_fixedscale" \
        else log_action_scale.exp()
    if design == "kdiov11_unconstrained":
        action_direction = (
            math.sqrt(action_dim) * action_parameter / action_parameter.norm())
    else:
        action_direction, triangular = torch.linalg.qr(action_parameter, mode="reduced")
        sign = torch.where(
            torch.diagonal(triangular) < 0.0,
            -torch.ones(action_dim), torch.ones(action_dim))
        action_direction = action_direction * sign.unsqueeze(0)
    action_frame = action_scale * action_direction
    raw_singular = torch.linalg.svdvals(action_parameter)
    direction_singular = torch.linalg.svdvals(action_direction)
    frame_singular = torch.linalg.svdvals(action_frame)
    direction_gram_error = float((
        action_direction.T @ action_direction - torch.eye(action_dim)).abs().max())
    gram_error = float((
        action_frame.T @ action_frame
        - action_scale.square() * torch.eye(action_dim)).abs().max())
    noaction = design == "kdiov11_noaction"
    metrics.update({
        "memory_innovation_precision_singular_min": .5,
        "memory_innovation_precision_singular_max": 1.0,
        "memory_innovation_precision_condition": 2.0,
        "memory_calibration_updates": 100.0,
        "memory_calibration_samples": 56_400.0,
        "memory_calibration_oas_shrinkage": .5,
        "memory_calibration_covariance_condition": 4.0,
        "memory_calibration_diagonal_only": 0.0,
        "memory_gradient_trained_parameters": float(gradient),
        "memory_nominal_optimizer_scalars": float(gradient),
        "memory_fitted_memory_scalars": float(fitted),
        "memory_total_memory_scalars": float(gradient + fitted),
        "memory_kick_drift": 1.0,
        "memory_velocity_carry": 1.0,
        "memory_position_drift": 1.0,
        "memory_action_transport": float(not noaction),
        "memory_autonomous_transport": 1.0,
        "memory_prior_conditioned_correction": 1.0,
        "memory_innovation_reliability": 1.0,
        "memory_ordered_correction": 1.0,
        "memory_invertible_transition": 1.0,
        "memory_recurrent_floats": 256.0,
        "memory_action_scale": float(action_scale),
        "memory_action_log_scale": float(log_action_scale),
        "memory_action_scale_parameter_retained": 1.0,
        "memory_action_scale_gradient_active": float(
            design not in {"kdiov11_fixedscale", "kdiov11_noaction"}),
        "memory_action_kick_norm": float(action_frame.norm()),
        "memory_action_raw_norm": float(action_parameter.norm()),
        "memory_action_direction_norm": float(action_direction.norm()),
        "memory_action_effective_norm": float(action_frame.norm()),
        "memory_action_parameter_norm": float(action_parameter.norm()),
        "memory_action_parameter_singular_min": float(raw_singular.min()),
        "memory_action_parameter_singular_max": float(raw_singular.max()),
        "memory_action_parameter_condition": float(
            raw_singular.max() / raw_singular.min()),
        "memory_action_frame_gram_error": gram_error,
        "memory_action_frame_singular_min": float(frame_singular.min()),
        "memory_action_frame_singular_max": float(frame_singular.max()),
        "memory_action_frame_condition": float(
            frame_singular.max() / frame_singular.min()),
        "memory_action_frame_constrained": float(
            design != "kdiov11_unconstrained"),
        "memory_action_direction_gram_error": direction_gram_error,
        "memory_action_direction_singular_min": float(direction_singular.min()),
        "memory_action_direction_singular_max": float(direction_singular.max()),
        "memory_action_direction_condition": float(
            direction_singular.max() / direction_singular.min()),
        "memory_state_kick_norm": 0.0,
        "memory_autonomous_kick_norm": 0.0,
    })
    if noaction:
        for key in runner.KDIO_ACTION_DIAGNOSTIC_METRICS:
            metrics[key] = 0.0
        metrics.update({
            "kdio_true_action_one_step_mse": .5,
            "kdio_shuffled_action_one_step_mse": .5,
            "kdio_true_action_suffix_mse": .5,
            "kdio_shuffled_action_suffix_mse": .5,
            "kdio_action_swap_pair_accuracy": .5,
        })
    else:
        metrics.update({
            "kdio_action_effect_rms": .25,
            "kdio_true_action_one_step_mse": .5,
            "kdio_shuffled_action_one_step_mse": .6,
            "kdio_true_action_one_step_advantage": 1.0 / 6.0,
            "kdio_true_action_suffix_mse": .7,
            "kdio_shuffled_action_suffix_mse": .8,
            "kdio_true_action_suffix_advantage": .125,
            "kdio_action_swap_pair_accuracy": .6,
            "kdio_true_action_advantage_h1": 1.0 / 6.0,
            "kdio_true_action_advantage_h4": .14,
            "kdio_true_action_advantage_h8": .12,
            "kdio_true_action_advantage_h16": .10,
            "kdio_true_action_advantage_h47": .08,
            "kdio_action_rollout_divergence_h1": .1,
            "kdio_action_rollout_divergence_h4": .15,
            "kdio_action_rollout_divergence_h8": .2,
            "kdio_action_rollout_divergence_h16": .3,
            "kdio_action_rollout_divergence_h47": .4,
        })
    bounded = {
        "innovation_ratio", "reliability", "position_base_gain",
        "velocity_base_gain", "velocity_base_ratio", "q_gates", "v_gates",
        "action_tanh_derivative_mean", "action_tanh_saturation_proxy",
    }
    for dataset in runner.OBSERVER_DATASETS:
        metrics[f"{dataset}_observer_ordered_gain_violation_max"] = 0.0
        for key in runner.OBSERVER_KEYS:
            for phase in runner.OBSERVER_PHASES:
                observed_mean = .5 if key in bounded else 1.0
                observed_std = .1
                if noaction and key == "action_effect_norm":
                    observed_mean, observed_std = 0.0, 0.0
                metrics[f"{dataset}_observer_{key}_{phase}_mean"] = observed_mean
                metrics[f"{dataset}_observer_{key}_{phase}_std"] = observed_std
    state = {
        f"world.mem_kdiov11.{name}": torch.zeros(1)
        for name in (
            "clean_innovation_mean", "innovation_precision_packed",
            "calibration_updates", "calibration_samples", "calibration_oas_shrinkage",
            "calibration_covariance_condition", "calibration_diagonal_only",
        )
    }
    state["world.mem_kdiov11.W_a.weight"] = action_parameter
    state["world.mem_kdiov11.log_action_scale"] = log_action_scale
    state["world.mem_kdiov11.w_q"] = torch.zeros(dimension)
    state["world.mem_kdiov11.b_f"] = torch.zeros(dimension)
    return metrics, state


def test_runner_requires_complete_calibration_and_observer_receipts() -> None:
    metrics, state = _valid_kdio_receipts()
    job = Namespace(run_name="calibration-receipts", design="kdiov11")
    runner._validate_kdio_calibration_receipts(job, metrics, state, action_dim=2)
    missing = dict(metrics)
    del missing["clean_observer_reliability_primary_mean"]
    try:
        runner._validate_kdio_calibration_receipts(job, missing, state, action_dim=2)
    except runner.RunnerError:
        pass
    else:
        raise AssertionError("runner accepted a missing observer receipt")


def test_runner_distinguishes_stiefel_raw_and_unconstrained_action_frames() -> None:
    constrained, constrained_state = _valid_kdio_receipts(design="kdiov11")
    runner._validate_kdio_calibration_receipts(
        Namespace(run_name="stiefel", design="kdiov11"),
        constrained, constrained_state, action_dim=2)
    assert math.isclose(
        constrained["memory_action_kick_norm"], 1.5 * math.sqrt(2.0), abs_tol=1e-6)
    assert not math.isclose(
        constrained["memory_action_kick_norm"],
        constrained["memory_action_parameter_norm"])
    assert constrained["memory_action_frame_constrained"] == 1.0

    unconstrained, unconstrained_state = _valid_kdio_receipts(
        design="kdiov11_unconstrained")
    runner._validate_kdio_calibration_receipts(
        Namespace(run_name="raw-control", design="kdiov11_unconstrained"),
        unconstrained, unconstrained_state, action_dim=2)
    assert unconstrained["memory_action_frame_constrained"] == 0.0
    assert unconstrained["memory_action_frame_gram_error"] > 0.0
    assert math.isclose(
        unconstrained["memory_action_direction_norm"], math.sqrt(2.0), abs_tol=1e-6)
    assert math.isclose(
        unconstrained["memory_action_effective_norm"] / math.sqrt(2.0),
        unconstrained["memory_action_scale"], abs_tol=1e-6)

    fixed, fixed_state = _valid_kdio_receipts(design="kdiov11_fixedscale")
    runner._validate_kdio_calibration_receipts(
        Namespace(run_name="fixed-scale", design="kdiov11_fixedscale"),
        fixed, fixed_state, action_dim=2)
    assert fixed["memory_action_scale"] == 1.0
    assert fixed["memory_action_log_scale"] != 0.0
    assert fixed["memory_action_scale_gradient_active"] == 0.0


def test_runner_noaction_keeps_frame_but_zeroes_semantic_action_effects() -> None:
    metrics, state = _valid_kdio_receipts(design="kdiov11_noaction")
    runner._validate_kdio_calibration_receipts(
        Namespace(run_name="noaction", design="kdiov11_noaction"),
        metrics, state, action_dim=2)
    assert metrics["memory_action_parameter_norm"] > 0.0
    assert metrics["memory_action_kick_norm"] > 0.0
    assert metrics["memory_action_transport"] == 0.0
    assert metrics["kdio_action_effect_rms"] == 0.0
    assert metrics["kdio_action_swap_pair_accuracy"] == .5
    assert metrics["kdio_true_action_advantage_h4"] == 0.0
    assert metrics["kdio_action_rollout_divergence_h47"] == 0.0


def test_analyzer_pairing_and_action_only_control() -> None:
    rows = _synthetic_rows(analyzer.PILOT_SEEDS)
    summary = analyzer.pairwise_summary(rows, analyzer.CANDIDATE, "ssm")
    assert np.isclose(summary["mean_paired_relative_reduction"], .25)
    assert summary["paired_wins"] == 15
    convergence = [{"run": row["run"], "relative_improvement": .001} for row in rows]
    observed = analyzer.observed_summary(rows, convergence)
    assert np.isclose(
        observed["action_only_control"]["mean_paired_relative_reduction"], .25)
    assert observed["action_only_control"]["environment_mean_wins"] == 5
    calibration = observed["candidate_calibration_receipts"]
    assert calibration["updates_min"] == 100
    assert calibration["samples_max"] == 56_400
    assert calibration["precision_singular_min"] > 0
    assert set(calibration["heldout_primary"]) == set(analyzer.HELDOUT_CONDITIONS)
    assert np.isclose(
        observed["candidate_action_path_receipts"]["action_map_norm_mean"],
        math.sqrt(2.0))
    assert observed["candidate_action_path_receipts"][
        "action_map_numerical_rank_min"] == 2.0
    assert set(observed["action_path_receipts_by_design"]) == set(
        analyzer.KDIO_DESIGNS)
    assert observed["action_path_receipts_by_design"]["kdiov11"][
        "effective_frame"]["constrained"]["min"] == 1.0
    assert observed["action_path_receipts_by_design"]["kdiov11_unconstrained"][
        "effective_frame"]["constrained"]["max"] == 0.0


def test_analyzer_comparison_validity_covers_candidate_and_all_headline_anchors() -> None:
    rows = _synthetic_rows(analyzer.PILOT_SEEDS)
    convergence = [{"run": row["run"], "relative_improvement": .001} for row in rows]
    observed = analyzer.observed_summary(rows, convergence)
    assert tuple(observed["comparison_validity_receipts"]) == (
        analyzer.CANDIDATE, *analyzer.HEADLINE_REFERENCES)
    criteria, _ = analyzer.decision_criteria(observed, rows, final=False)
    assert all(criteria.values())

    # Every quality/causality dimension is fail-closed for every headline encoder,
    # even when the candidate's own receipt remains healthy.
    failures = (
        ("encoder_mean_channel_variance", 0.0, "ssm_encoder_variance_ge_1e_5"),
        ("encoder_covariance_effective_rank", 15.0, "ssm_encoder_rank_ge_16"),
        ("encoder_singleton_max_abs", 2e-5, "ssm_encoder_singleton_le_1e_5"),
        ("encoder_prefix_max_abs", 2e-5, "ssm_encoder_prefix_le_1e_5"),
    )
    for metric, invalid, criterion in failures:
        invalid_rows = [dict(row) for row in rows]
        for row in invalid_rows:
            if row["design"] == "ssm":
                row[metric] = invalid
        observed = analyzer.observed_summary(invalid_rows, convergence)
        criteria, _ = analyzer.decision_criteria(observed, invalid_rows, final=False)
        assert criteria["kdiov11_encoder_variance_ge_1e_5"]
        assert not criteria[criterion]


def test_analyzer_gamma_asr_semantic_and_integrator_gates_fail_closed() -> None:
    rows = _synthetic_rows(analyzer.PILOT_SEEDS)
    convergence = [{"run": row["run"], "relative_improvement": .001} for row in rows]

    def criteria_after(design, metric, value):
        changed = [dict(row) for row in rows]
        for row in changed:
            if row["design"] == design:
                row[metric] = value
        observed = analyzer.observed_summary(changed, convergence)
        return analyzer.decision_criteria(observed, changed, final=False)[0]

    criteria = criteria_after("kdiov11", "kdio_action_swap_pair_accuracy", .5)
    assert not criteria["asr_pair_accuracy_equal_task_mean_gt_chance"]
    criteria = criteria_after("kdiov11", "kdio_true_action_advantage_h4", 0.0)
    assert not criteria["true_action_advantage_h4_equal_task_mean_gt_0"]
    criteria = criteria_after("kdiov11_fixedscale", "memory_action_scale", 1.1)
    assert not criteria["fixedscale_gamma_eq_1"]
    criteria = criteria_after("kdiov11_noaction", "kdio_action_swap_pair_accuracy", .6)
    assert not criteria["noaction_pair_accuracy_eq_chance"]
    criteria = criteria_after(
        "kdiov11", "initial_encoder_integrator_probe_nmse", .7)
    assert not criteria["vs_initial_encoder_integrator_equal_task_mean_below"]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"ok: {len(tests)} V11 experiment tests")
