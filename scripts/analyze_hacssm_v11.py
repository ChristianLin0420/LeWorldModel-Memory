#!/usr/bin/env python3
"""Fail-closed paired analysis for the frozen 400-cell KDIO-v11 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.analyze_hacssm_v5 as shared


ENVIRONMENTS = (
    "dmc:walker.walk", "dmc:hopper.hop", "dmc:cartpole.swingup",
    "dmc:pendulum.swingup", "dmc:fish.swim",
)
DESIGNS = (
    "ssm", "hacssmv8", "orbitv10", "kdiov11", "kdiov11_unconstrained",
    "kdiov11_fixedscale", "kdiov11_nocalibration",
    "kdiov11_diagonal", "kdiov11_h1",
    "kdiov11_firstorder", "kdiov11_nodrift", "kdiov11_noautonomy",
    "kdiov11_noaction", "kdiov11_noactionswap", "kdiov11_nosuffix",
    "kdiov11_noreliability",
)
KDIO_DESIGNS = frozenset(
    design for design in DESIGNS if design.startswith("kdiov11"))
PILOT_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 100
WINDOW = 10
CALIBRATION_SAMPLES = 1_200 * (48 - 1)
PRIMARY = "heldout_prior_state_nmse"
CLEAN_METRIC = "clean_prior_state_nmse"
CANDIDATE = "kdiov11"
HEADLINE_REFERENCES = ("ssm", "hacssmv8", "orbitv10")
COMPARISON_VALIDITY_DESIGNS = (CANDIDATE, *HEADLINE_REFERENCES)
MECHANISM_CONTROLS = (
    "kdiov11_unconstrained", "kdiov11_fixedscale", "kdiov11_noactionswap",
    "kdiov11_nocalibration", "kdiov11_diagonal", "kdiov11_h1",
    "kdiov11_firstorder", "kdiov11_nodrift",
    "kdiov11_noautonomy", "kdiov11_noaction",
    "kdiov11_nosuffix", "kdiov11_noreliability",
)
HELDOUT_CONDITIONS = ("freeze", "gaussian_noise", "checkerboard", "long_freeze")
SCOPE = "adaptive_development_after_v10_host_audit"
DEFAULT_ROOT = Path("outputs/hacssm_v11_shared")
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 11_011
BOOTSTRAP_CONTRACT = {
    "schema_version": 1,
    "algorithm": "crossed_environment_seed_percentile_bootstrap",
    "draws": BOOTSTRAP_DRAWS,
    "seed": BOOTSTRAP_SEED,
    "rng": "numpy.random.Generator(numpy.random.PCG64)",
    "resampling": "independent environment and optimizer-seed indices; Cartesian product mean",
    "estimand": "mean paired relative prior-state NMSE reduction",
    "quantiles": {"method": "linear", "reported": [0.05, 0.95]},
}
BOOTSTRAP_CONTRACT_SHA256 = hashlib.sha256(json.dumps(
    BOOTSTRAP_CONTRACT, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode()).hexdigest()

ROW_METRICS = (
    PRIMARY, CLEAN_METRIC, "val_predictive_loss", "final_val_loss",
    "mean_epoch_seconds", "peak_vram_bytes",
    "prior_probe_ceiling_state_nmse", "prior_probe_ceiling_r2",
    "posterior_probe_ceiling_state_nmse", "posterior_probe_ceiling_r2",
    "encoder_probe_ceiling_state_nmse", "predictor_probe_ceiling_state_nmse",
    "inverse_action_nmse", "inverse_action_r2", "inverse_action_probe_samples",
    "inverse_action_probe_input_dim", "inverse_action_probe_output_dim",
    "action_history_probe_nmse", "action_history_probe_r2",
    "action_only_integrator_probe_nmse", "action_only_integrator_probe_r2",
    "initial_encoder_integrator_probe_nmse", "initial_encoder_integrator_probe_r2",
    "predictive_loss_convergence_relative_change",
    "action_swap_loss_convergence_relative_change",
    "calibration_nll_convergence_relative_change", "loss_convergence_relative_change",
    "encoder_mean_channel_variance", "encoder_covariance_effective_rank",
    "encoder_singleton_max_abs", "encoder_prefix_max_abs",
    "orbit_orthogonality_error_max", "orbit_streaming_max_abs",
    "kdio_inverse_error_max", "kdio_volume_error_max", "kdio_streaming_max_abs",
    *(f"{condition}_prior_state_nmse" for condition in HELDOUT_CONDITIONS),
    *(f"{condition}_{coordinate}_state_nmse_{phase}"
      for condition in HELDOUT_CONDITIONS
      for coordinate in ("prior", "encoder", "predictor")
      for phase in ("deep", "first_post", "post")),
)

OBSERVER_KEYS = (
    "innovation_ratio", "innovation_energy", "process_tolerance", "reliability",
    "position_base_gain", "velocity_base_gain", "velocity_base_ratio",
    "q_gates", "v_gates", "action_effect_norm",
    "action_tanh_derivative_mean", "action_tanh_saturation_proxy",
)
OBSERVER_PHASES = ("all", "gap", "deep", "first_post", "post", "primary")
OBSERVER_DATASETS = ("clean", *HELDOUT_CONDITIONS)
CALIBRATION_MEMORY_METRICS = (
    "memory_innovation_precision_diagonal_mean",
    "memory_innovation_precision_logdet_per_dim",
    "memory_innovation_precision_offdiagonal_norm",
    "memory_innovation_precision_singular_min",
    "memory_innovation_precision_singular_max",
    "memory_innovation_precision_condition",
    "memory_calibration_updates", "memory_calibration_samples",
    "memory_calibration_oas_shrinkage",
    "memory_calibration_covariance_condition",
    "memory_calibration_diagonal_only", "memory_gradient_trained_parameters",
    "memory_nominal_optimizer_scalars", "memory_fitted_memory_scalars",
    "memory_total_memory_scalars", "memory_clean_innovation_mean_norm",
)
KDIO_MECHANISM_METRICS = (
    "memory_action_scale", "memory_action_log_scale",
    "memory_action_scale_parameter_retained", "memory_action_scale_gradient_active",
    "memory_action_kick_norm", "memory_action_raw_norm",
    "memory_action_direction_norm", "memory_action_effective_norm",
    "memory_action_parameter_norm",
    "memory_action_parameter_singular_min", "memory_action_parameter_singular_max",
    "memory_action_parameter_condition", "memory_action_frame_gram_error",
    "memory_action_frame_singular_min", "memory_action_frame_singular_max",
    "memory_action_frame_condition", "memory_action_frame_constrained",
    "memory_action_direction_gram_error", "memory_action_direction_singular_min",
    "memory_action_direction_singular_max", "memory_action_direction_condition",
    "memory_state_kick_norm",
    "memory_autonomous_kick_norm", "memory_position_gain_vector_norm",
    "memory_velocity_ratio_vector_norm", "memory_process_tolerance_vector_norm",
    "memory_kick_drift", "memory_velocity_carry", "memory_position_drift",
    "memory_action_transport", "memory_autonomous_transport",
    "memory_prior_conditioned_correction", "memory_innovation_reliability",
    "memory_ordered_correction", "memory_invertible_transition",
    "memory_recurrent_floats",
)
KDIO_ACTION_DIAGNOSTIC_METRICS = (
    "kdio_action_effect_rms",
    "kdio_true_action_one_step_mse", "kdio_shuffled_action_one_step_mse",
    "kdio_true_action_one_step_advantage",
    "kdio_true_action_suffix_mse", "kdio_shuffled_action_suffix_mse",
    "kdio_true_action_suffix_advantage",
    "kdio_action_swap_pair_accuracy",
    "kdio_true_action_advantage_h1", "kdio_true_action_advantage_h4",
    "kdio_true_action_advantage_h8", "kdio_true_action_advantage_h16",
    "kdio_true_action_advantage_h47",
    "kdio_action_rollout_divergence_h1", "kdio_action_rollout_divergence_h4",
    "kdio_action_rollout_divergence_h8",
    "kdio_action_rollout_divergence_h16", "kdio_action_rollout_divergence_h47",
)
OBSERVER_METRICS = tuple(
    metric
    for dataset in OBSERVER_DATASETS
    for metric in (
        *(f"{dataset}_observer_{key}_{phase}_{stat}"
          for key in OBSERVER_KEYS
          for phase in OBSERVER_PHASES
          for stat in ("mean", "std")),
        f"{dataset}_observer_ordered_gain_violation_max",
    )
)
KDIO_DIAGNOSTIC_METRICS = (
    CALIBRATION_MEMORY_METRICS + KDIO_MECHANISM_METRICS
    + KDIO_ACTION_DIAGNOSTIC_METRICS + OBSERVER_METRICS)
KDIO_CHECKPOINT_DIAGNOSTICS = (
    "action_parameter_checkpoint_singular_min",
    "action_parameter_checkpoint_singular_max",
    "action_parameter_checkpoint_numerical_rank",
    "action_scale_checkpoint", "action_direction_checkpoint_norm",
    "action_direction_checkpoint_gram_error",
    "action_direction_checkpoint_singular_min",
    "action_direction_checkpoint_singular_max",
    "action_direction_checkpoint_condition",
    "action_frame_checkpoint_norm", "action_frame_checkpoint_gram_error",
    "action_frame_checkpoint_singular_min", "action_frame_checkpoint_singular_max",
    "action_frame_checkpoint_condition",
    "memory_functionally_effective_optimizer_dof",
    "memory_functionally_active_fitted_scalars",
    "memory_functionally_effective_plus_fitted_scalars",
)


def _finite(value: Any, context: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{context} is not finite: {value!r}")
    return float(value)


def load_cells(root: Path, seeds: Sequence[int]):
    rows, convergence = [], []
    for environment in ENVIRONMENTS:
        for design in DESIGNS:
            for seed in seeds:
                run = f"lewm-{environment}-{design}-s{seed}"
                run_dir = root / run
                try:
                    metrics = shared.read_json(run_dir / "metrics.json")
                    checkpoint = torch.load(
                        run_dir / "model.pt", map_location="cpu", weights_only=False)
                except Exception as exc:
                    raise ValueError(f"cannot load {run_dir}: {exc}") from exc
                if metrics != checkpoint.get("final_metrics"):
                    raise ValueError(f"{run}: metrics/checkpoint mismatch")
                history = checkpoint.get("history")
                if not isinstance(history, list) or len(history) != EPOCHS:
                    raise ValueError(f"{run}: expected {EPOCHS} history records")
                row = {
                    "run": run, "env": environment, "design": design, "seed": int(seed),
                    "trainable_parameters": int(metrics["trainable_parameters"]),
                }
                for key in ROW_METRICS:
                    row[key] = _finite(metrics[key], f"{run}.{key}")
                for key in KDIO_DIAGNOSTIC_METRICS:
                    row[key] = (
                        _finite(metrics[key], f"{run}.{key}")
                        if design in KDIO_DESIGNS else "")
                if design in KDIO_DESIGNS:
                    model_state = checkpoint["model_state_dict"]
                    action_parameter = model_state.get(
                        "world.mem_kdiov11.W_a.weight")
                    log_action_scale = model_state.get(
                        "world.mem_kdiov11.log_action_scale")
                    if (not isinstance(action_parameter, torch.Tensor)
                            or action_parameter.dim() != 2
                            or not isinstance(log_action_scale, torch.Tensor)
                            or log_action_scale.numel() != 1):
                        raise ValueError(f"{run}: missing KDIO action-frame checkpoint tensors")
                    action_parameter = action_parameter.float()
                    log_action_scale = log_action_scale.float().reshape(())
                    dimension = action_parameter.shape[0]
                    action_dim = action_parameter.shape[1]
                    parameter_singular = torch.linalg.svdvals(action_parameter)
                    if design == "kdiov11_unconstrained":
                        action_direction = (
                            math.sqrt(action_dim) * action_parameter
                            / action_parameter.norm().clamp_min(
                                torch.finfo(torch.float32).tiny))
                    else:
                        action_direction, triangular = torch.linalg.qr(
                            action_parameter, mode="reduced")
                        sign = torch.where(
                            torch.diagonal(triangular) < 0.0,
                            -torch.ones_like(torch.diagonal(triangular)),
                            torch.ones_like(torch.diagonal(triangular)),
                        )
                        action_direction = action_direction * sign.unsqueeze(0)
                    action_scale = (
                        torch.ones((), dtype=torch.float32)
                        if design == "kdiov11_fixedscale"
                        else torch.exp(log_action_scale))
                    action_frame = action_scale * action_direction
                    direction_singular = torch.linalg.svdvals(action_direction)
                    frame_singular = torch.linalg.svdvals(action_frame)
                    direction_gram = action_direction.T @ action_direction
                    frame_gram = action_frame.T @ action_frame
                    frame_identity = torch.eye(
                        action_frame.shape[1], dtype=action_frame.dtype)
                    gauge_dimension = (
                        action_frame.shape[1] * (action_frame.shape[1] + 1) // 2)
                    nominal = row["memory_nominal_optimizer_scalars"]
                    if design == "kdiov11_unconstrained":
                        effective_optimizer = nominal - 1
                    elif design == "kdiov11_noaction":
                        effective_optimizer = nominal - action_parameter.numel() - 1
                    elif design == "kdiov11_fixedscale":
                        effective_optimizer = nominal - gauge_dimension - 1
                    elif design == "kdiov11_noautonomy":
                        effective_optimizer = nominal - gauge_dimension - 2 * dimension
                    elif design == "kdiov11_noreliability":
                        effective_optimizer = nominal - gauge_dimension - (dimension + 1)
                    else:
                        effective_optimizer = nominal - gauge_dimension
                    active_fitted = (
                        0.0 if design in {
                            "kdiov11_nocalibration", "kdiov11_noreliability"} else
                        2.0 * (dimension - 1) if design == "kdiov11_diagonal" else
                        row["memory_fitted_memory_scalars"])
                    effective_plus_fitted = (
                        effective_optimizer + active_fitted)
                    checkpoint_diagnostics = {
                        "action_parameter_checkpoint_singular_min": float(
                            parameter_singular.min()),
                        "action_parameter_checkpoint_singular_max": float(
                            parameter_singular.max()),
                        "action_parameter_checkpoint_numerical_rank": float(
                            torch.linalg.matrix_rank(action_parameter)),
                        "action_scale_checkpoint": float(action_scale),
                        "action_direction_checkpoint_norm": float(
                            action_direction.norm()),
                        "action_direction_checkpoint_gram_error": float(
                            (direction_gram - frame_identity).abs().max()),
                        "action_direction_checkpoint_singular_min": float(
                            direction_singular.min()),
                        "action_direction_checkpoint_singular_max": float(
                            direction_singular.max()),
                        "action_direction_checkpoint_condition": float(
                            direction_singular.max()
                            / direction_singular.min().clamp_min(
                                torch.finfo(torch.float32).tiny)),
                        "action_frame_checkpoint_norm": float(action_frame.norm()),
                        "action_frame_checkpoint_gram_error": float(
                            (frame_gram - action_scale.square() * frame_identity).abs().max()),
                        "action_frame_checkpoint_singular_min": float(
                            frame_singular.min()),
                        "action_frame_checkpoint_singular_max": float(
                            frame_singular.max()),
                        "action_frame_checkpoint_condition": float(
                            frame_singular.max() / frame_singular.min().clamp_min(
                                torch.finfo(torch.float32).tiny)),
                        "memory_functionally_effective_optimizer_dof": float(
                            effective_optimizer),
                        "memory_functionally_active_fitted_scalars": float(active_fitted),
                        "memory_functionally_effective_plus_fitted_scalars": float(
                            effective_plus_fitted),
                    }
                    checkpoint_pairs = {
                        "memory_action_scale": checkpoint_diagnostics[
                            "action_scale_checkpoint"],
                        "memory_action_log_scale": float(log_action_scale),
                        "memory_action_raw_norm": float(action_parameter.norm()),
                        "memory_action_parameter_norm": float(action_parameter.norm()),
                        "memory_action_kick_norm": checkpoint_diagnostics[
                            "action_frame_checkpoint_norm"],
                        "memory_action_effective_norm": checkpoint_diagnostics[
                            "action_frame_checkpoint_norm"],
                        "memory_action_parameter_singular_min":
                            checkpoint_diagnostics[
                                "action_parameter_checkpoint_singular_min"],
                        "memory_action_parameter_singular_max":
                            checkpoint_diagnostics[
                                "action_parameter_checkpoint_singular_max"],
                        "memory_action_frame_gram_error": checkpoint_diagnostics[
                            "action_frame_checkpoint_gram_error"],
                        "memory_action_frame_singular_min": checkpoint_diagnostics[
                            "action_frame_checkpoint_singular_min"],
                        "memory_action_frame_singular_max": checkpoint_diagnostics[
                            "action_frame_checkpoint_singular_max"],
                        "memory_action_frame_condition": checkpoint_diagnostics[
                            "action_frame_checkpoint_condition"],
                        "memory_action_direction_norm": checkpoint_diagnostics[
                            "action_direction_checkpoint_norm"],
                        "memory_action_direction_gram_error": checkpoint_diagnostics[
                            "action_direction_checkpoint_gram_error"],
                        "memory_action_direction_singular_min": checkpoint_diagnostics[
                            "action_direction_checkpoint_singular_min"],
                        "memory_action_direction_singular_max": checkpoint_diagnostics[
                            "action_direction_checkpoint_singular_max"],
                        "memory_action_direction_condition": checkpoint_diagnostics[
                            "action_direction_checkpoint_condition"],
                    }
                    for metric, measured in checkpoint_pairs.items():
                        if not math.isclose(row[metric], measured,
                                            rel_tol=1e-3, abs_tol=1e-6):
                            raise ValueError(
                                f"{run}: metric/checkpoint action receipt mismatch {metric}")
                else:
                    checkpoint_diagnostics = {
                        key: "" for key in KDIO_CHECKPOINT_DIAGNOSTICS}
                row.update(checkpoint_diagnostics)
                row["final_train_calibration_nll"] = _finite(
                    history[-1]["train"]["calibration_nll"],
                    f"{run}.final_train_calibration_nll")
                row["final_val_calibration_nll"] = _finite(
                    history[-1]["val"]["calibration_nll"],
                    f"{run}.final_val_calibration_nll")
                for split in ("train", "val"):
                    for key in (
                            "action_swap_loss", "action_swap_diagnostic_loss",
                            "action_swap_advantage", "action_swap_pair_accuracy"):
                        label = f"final_{split}_{key}"
                        row[label] = _finite(history[-1][split][key], f"{run}.{label}")
                condition_values = [row[f"{condition}_prior_state_nmse"]
                                    for condition in HELDOUT_CONDITIONS]
                if not math.isclose(row[PRIMARY], mean(condition_values),
                                    rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError(f"{run}: headline is not the equal condition mean")
                previous = mean(_finite(item["val"]["predictive_loss"], run)
                                for item in history[-2 * WINDOW:-WINDOW])
                recent = mean(_finite(item["val"]["predictive_loss"], run)
                              for item in history[-WINDOW:])
                emitted = row["predictive_loss_convergence_relative_change"]
                relative = (previous - recent) / max(previous, 1e-12)
                if not math.isclose(emitted, relative, rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError(f"{run}: convergence receipt mismatch")
                convergence.append({
                    "run": run, "env": environment, "design": design, "seed": seed,
                    "previous_window_mean": previous, "recent_window_mean": recent,
                    "relative_improvement": relative,
                })
                rows.append(row)
    expected = len(ENVIRONMENTS) * len(DESIGNS) * len(seeds)
    if len(rows) != expected:
        raise ValueError(f"grid has {len(rows)} rows, expected {expected}")
    return rows, convergence


def _grid(rows, metric):
    seeds = tuple(sorted({int(row["seed"]) for row in rows}))
    lookup = {(row["env"], row["design"], int(row["seed"])):
              _finite(row[metric], f"row.{metric}") for row in rows}
    wanted = {(env, design, seed) for env in ENVIRONMENTS
              for design in DESIGNS for seed in seeds}
    if set(lookup) != wanted or any(value <= 0 for value in lookup.values()):
        raise ValueError(f"invalid locked grid for {metric}")
    return lookup, seeds


def pairwise_matrix(rows, candidate, reference, metric=PRIMARY):
    lookup, seeds = _grid(rows, metric)
    return np.asarray([[
        (lookup[(environment, reference, seed)] - lookup[(environment, candidate, seed)])
        / lookup[(environment, reference, seed)]
        for seed in seeds] for environment in ENVIRONMENTS], dtype=np.float64)


def pairwise_summary(rows, candidate, reference, metric=PRIMARY):
    matrix = pairwise_matrix(rows, candidate, reference, metric)
    environment_effects = {
        environment: float(matrix[index].mean())
        for index, environment in enumerate(ENVIRONMENTS)}
    return {
        "candidate": candidate, "reference": reference, "metric": metric,
        "n_pairs": int(matrix.size), "mean_paired_relative_reduction": float(matrix.mean()),
        "paired_wins": int((matrix > 0).sum()), "paired_ties": int((matrix == 0).sum()),
        "environment_mean_wins": int(sum(value > 0 for value in environment_effects.values())),
        "environment_mean_reductions": environment_effects,
    }


def crossed_bootstrap(matrix: np.ndarray, label: str):
    if matrix.shape != (len(ENVIRONMENTS), len(FINAL_SEEDS)):
        raise ValueError(f"{label}: wrong final matrix shape {matrix.shape}")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    env_index = rng.integers(0, matrix.shape[0], (BOOTSTRAP_DRAWS, matrix.shape[0]))
    seed_index = rng.integers(0, matrix.shape[1], (BOOTSTRAP_DRAWS, matrix.shape[1]))
    draws = matrix[env_index[:, :, None], seed_index[:, None, :]].mean(axis=(1, 2))
    quantiles = np.quantile(draws, (0.05, 0.95), method="linear")
    return {"label": label, "point": float(matrix.mean()),
            "ci90": [float(quantiles[0]), float(quantiles[1])],
            "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
            "contract_sha256": BOOTSTRAP_CONTRACT_SHA256}


def _metric_summary(rows, metric):
    values = [float(row[metric]) for row in rows]
    return {"min": min(values), "mean": float(np.mean(values)), "max": max(values)}


def _action_path_summary(rows):
    if not rows:
        raise ValueError("cannot summarize an empty KDIO action path")
    return {
        "scale": {
            "gamma": _metric_summary(rows, "memory_action_scale"),
            "log_gamma": _metric_summary(rows, "memory_action_log_scale"),
            "gradient_active": _metric_summary(
                rows, "memory_action_scale_gradient_active"),
            "parameter_retained": _metric_summary(
                rows, "memory_action_scale_parameter_retained"),
        },
        "raw_parameter": {
            "norm": _metric_summary(rows, "memory_action_parameter_norm"),
            "singular_min": _metric_summary(
                rows, "memory_action_parameter_singular_min"),
            "singular_max": _metric_summary(
                rows, "memory_action_parameter_singular_max"),
            "condition": _metric_summary(rows, "memory_action_parameter_condition"),
            "checkpoint_numerical_rank": _metric_summary(
                rows, "action_parameter_checkpoint_numerical_rank"),
        },
        "normalized_direction": {
            "norm": _metric_summary(rows, "memory_action_direction_norm"),
            "gram_error": _metric_summary(
                rows, "memory_action_direction_gram_error"),
            "singular_min": _metric_summary(
                rows, "memory_action_direction_singular_min"),
            "singular_max": _metric_summary(
                rows, "memory_action_direction_singular_max"),
            "condition": _metric_summary(rows, "memory_action_direction_condition"),
        },
        "effective_frame": {
            "norm": _metric_summary(rows, "memory_action_kick_norm"),
            "gram_error": _metric_summary(rows, "memory_action_frame_gram_error"),
            "singular_min": _metric_summary(
                rows, "memory_action_frame_singular_min"),
            "singular_max": _metric_summary(
                rows, "memory_action_frame_singular_max"),
            "condition": _metric_summary(rows, "memory_action_frame_condition"),
            "constrained": _metric_summary(rows, "memory_action_frame_constrained"),
        },
        "parameter_accounting": {
            "caveat": (
                "nominal optimizer tensors are retained across controls; effective DOF "
                "subtract QR/radial gauges and exactly disconnected control tensors"),
            "nominal_optimizer": _metric_summary(
                rows, "memory_nominal_optimizer_scalars"),
            "functionally_effective_optimizer_dof": _metric_summary(
                rows, "memory_functionally_effective_optimizer_dof"),
            "nominal_fitted_oas": _metric_summary(
                rows, "memory_fitted_memory_scalars"),
            "functionally_active_fitted": _metric_summary(
                rows, "memory_functionally_active_fitted_scalars"),
            "nominal_total": _metric_summary(rows, "memory_total_memory_scalars"),
            "effective_plus_fitted": _metric_summary(
                rows, "memory_functionally_effective_plus_fitted_scalars"),
        },
        "final_asr_objective": {
            f"{split}_{metric}": _metric_summary(
                rows, f"final_{split}_{metric}")
            for split in ("train", "val")
            for metric in (
                "action_swap_loss", "action_swap_diagnostic_loss",
                "action_swap_advantage", "action_swap_pair_accuracy")
        },
        "semantic_action_diagnostics": {
            metric: _metric_summary(rows, metric)
            for metric in KDIO_ACTION_DIAGNOSTIC_METRICS
        },
        "clean_observer_all": {
            key: _metric_summary(rows, f"clean_observer_{key}_all_mean")
            for key in (
                "action_effect_norm", "action_tanh_derivative_mean",
                "action_tanh_saturation_proxy")
        },
        "heldout_observer_primary": {
            condition: {
                key: _metric_summary(
                    rows, f"{condition}_observer_{key}_primary_mean")
                for key in (
                    "action_effect_norm", "action_tanh_derivative_mean",
                    "action_tanh_saturation_proxy")
            }
            for condition in HELDOUT_CONDITIONS
        },
        "action_transport": _metric_summary(rows, "memory_action_transport"),
    }


def observed_summary(rows, convergence):
    candidate = [row for row in rows if row["design"] == CANDIDATE]
    noaction_rows = [row for row in rows if row["design"] == "kdiov11_noaction"]
    fixedscale_rows = [row for row in rows if row["design"] == "kdiov11_fixedscale"]
    unconstrained_rows = [
        row for row in rows if row["design"] == "kdiov11_unconstrained"]
    absolute = np.abs([row["relative_improvement"] for row in convergence])
    receipts = {
        "kdio_inverse_max": max(row["kdio_inverse_error_max"] for row in candidate),
        "kdio_volume_max": max(row["kdio_volume_error_max"] for row in candidate),
        "kdio_streaming_max": max(row["kdio_streaming_max_abs"] for row in candidate),
        "prior_ceiling_max": max(row["prior_probe_ceiling_state_nmse"] for row in candidate),
        "inverse_r2_min": min(row["inverse_action_r2"] for row in candidate),
        "observer_ordered_gain_violation_max": max(
            row[f"{dataset}_observer_ordered_gain_violation_max"]
            for row in candidate for dataset in OBSERVER_DATASETS),
        "action_frame_gram_error_max": max(
            row["memory_action_frame_gram_error"] for row in candidate),
        "action_scale_min": min(row["memory_action_scale"] for row in candidate),
        "action_scale_max": max(row["memory_action_scale"] for row in candidate),
        "action_scale_gradient_active_min": min(
            row["memory_action_scale_gradient_active"] for row in candidate),
        "action_scale_parameter_retained_min": min(
            row["memory_action_scale_parameter_retained"] for row in candidate),
        "action_frame_singular_over_gamma_min": min(
            row["memory_action_frame_singular_min"] / row["memory_action_scale"]
            for row in candidate),
        "action_frame_singular_over_gamma_max": max(
            row["memory_action_frame_singular_max"] / row["memory_action_scale"]
            for row in candidate),
        "action_frame_condition_max": max(
            row["memory_action_frame_condition"] for row in candidate),
        "action_direction_gram_error_max": max(
            row["memory_action_direction_gram_error"] for row in candidate),
        "action_frame_constrained_min": min(
            row["memory_action_frame_constrained"] for row in candidate),
        "asr_pair_accuracy_equal_task_mean": float(np.mean([
            row["kdio_action_swap_pair_accuracy"] for row in candidate])),
        "true_action_advantage_equal_task_means": {
            horizon: float(np.mean([
                row[f"kdio_true_action_advantage_h{horizon}"]
                for row in candidate]))
            for horizon in (1, 4, 8, 16, 47)
        },
    }
    semantic_controls = {
        "fixedscale_gamma_max_abs_error_from_one": max(
            abs(row["memory_action_scale"] - 1.0) for row in fixedscale_rows),
        "unconstrained_rms_singular_gamma_max_abs_error": max(
            abs(row["memory_action_effective_norm"]
                / math.sqrt(row["inverse_action_probe_output_dim"])
                - row["memory_action_scale"])
            for row in unconstrained_rows),
        "noaction_pair_accuracy_max_abs_error_from_chance": max(
            abs(row["kdio_action_swap_pair_accuracy"] - 0.5)
            for row in noaction_rows),
        "noaction_semantic_zero_max_abs": max(
            abs(row[key])
            for row in noaction_rows
            for key in (
                "kdio_action_effect_rms", "kdio_true_action_one_step_advantage",
                "kdio_true_action_suffix_advantage",
                "kdio_true_action_advantage_h1", "kdio_true_action_advantage_h4",
                "kdio_true_action_advantage_h8", "kdio_true_action_advantage_h16",
                "kdio_true_action_advantage_h47",
                "kdio_action_rollout_divergence_h1",
                "kdio_action_rollout_divergence_h4",
                "kdio_action_rollout_divergence_h8",
                "kdio_action_rollout_divergence_h16",
                "kdio_action_rollout_divergence_h47")),
    }
    calibration = {
        "updates_min": min(row["memory_calibration_updates"] for row in candidate),
        "updates_max": max(row["memory_calibration_updates"] for row in candidate),
        "samples_min": min(row["memory_calibration_samples"] for row in candidate),
        "samples_max": max(row["memory_calibration_samples"] for row in candidate),
        "oas_shrinkage_min": min(
            row["memory_calibration_oas_shrinkage"] for row in candidate),
        "oas_shrinkage_mean": float(np.mean([
            row["memory_calibration_oas_shrinkage"] for row in candidate])),
        "oas_shrinkage_max": max(
            row["memory_calibration_oas_shrinkage"] for row in candidate),
        "covariance_condition_max": max(
            row["memory_calibration_covariance_condition"] for row in candidate),
        "precision_condition_max": max(
            row["memory_innovation_precision_condition"] for row in candidate),
        "precision_singular_min": min(
            row["memory_innovation_precision_singular_min"] for row in candidate),
        "clean_innovation_energy_all_mean": float(np.mean([
            row["clean_observer_innovation_energy_all_mean"] for row in candidate])),
        "clean_reliability_all_mean": float(np.mean([
            row["clean_observer_reliability_all_mean"] for row in candidate])),
        "heldout_primary": {
            condition: {
                key: float(np.mean([
                    row[f"{condition}_observer_{key}_primary_mean"]
                    for row in candidate]))
                for key in ("innovation_energy", "process_tolerance", "reliability",
                            "q_gates", "v_gates")
            }
            for condition in HELDOUT_CONDITIONS
        },
    }
    action_path = {
        "architecture": (
            "gamma*qf(M), U^T U=I_A; fixed gamma=1 and normalized-free controls"),
        "detailed": _action_path_summary(candidate),
        # Compatibility aliases for the original flat decision-ledger fields.  The detailed
        # record above explicitly separates the raw parameter from its effective frame.
        "action_map_norm_min": min(row["memory_action_kick_norm"] for row in candidate),
        "action_map_norm_mean": float(np.mean([
            row["memory_action_kick_norm"] for row in candidate])),
        "action_map_norm_max": max(row["memory_action_kick_norm"] for row in candidate),
        "action_map_singular_min": min(
            row["action_parameter_checkpoint_singular_min"] for row in candidate),
        "action_map_singular_max": max(
            row["action_parameter_checkpoint_singular_max"] for row in candidate),
        "action_map_numerical_rank_min": min(
            row["action_parameter_checkpoint_numerical_rank"] for row in candidate),
        "action_map_numerical_rank_mean": float(np.mean([
            row["action_parameter_checkpoint_numerical_rank"] for row in candidate])),
        "state_kick_norm_mean": float(np.mean([
            row["memory_state_kick_norm"] for row in candidate])),
        "autonomous_kick_norm_mean": float(np.mean([
            row["memory_autonomous_kick_norm"] for row in candidate])),
        "action_transport_flag_min": min(
            row["memory_action_transport"] for row in candidate),
    }
    action_path_by_design = {
        design: _action_path_summary([
            row for row in rows if row["design"] == design])
        for design in DESIGNS if design in KDIO_DESIGNS
    }
    comparison_validity = {}
    for design in COMPARISON_VALIDITY_DESIGNS:
        selected = [row for row in rows if row["design"] == design]
        if not selected:
            raise ValueError(f"missing comparison-validity rows for {design}")
        comparison_validity[design] = {
            "encoder_variance_min": min(
                row["encoder_mean_channel_variance"] for row in selected),
            "encoder_rank_min": min(
                row["encoder_covariance_effective_rank"] for row in selected),
            "encoder_singleton_max": max(
                row["encoder_singleton_max_abs"] for row in selected),
            "encoder_prefix_max": max(
                row["encoder_prefix_max_abs"] for row in selected),
        }
    def external_control(metric):
        lookup = {(row["env"], row["seed"]): row for row in candidate}
        matrix = np.asarray([[
            (lookup[(environment, seed)][metric] - lookup[(environment, seed)][PRIMARY])
            / lookup[(environment, seed)][metric]
            for seed in sorted({row["seed"] for row in candidate})]
            for environment in ENVIRONMENTS], dtype=np.float64)
        return {
            "reference": metric, "metric": PRIMARY,
            "mean_paired_relative_reduction": float(matrix.mean()),
            "paired_wins": int((matrix > 0).sum()),
            "environment_mean_wins": int((matrix.mean(axis=1) > 0).sum()),
            "environment_mean_reductions": {
                environment: float(matrix[index].mean())
                for index, environment in enumerate(ENVIRONMENTS)},
        }
    action_control = external_control("action_only_integrator_probe_nmse")
    initial_control = external_control("initial_encoder_integrator_probe_nmse")
    return {
        "pairwise": {reference: pairwise_summary(rows, CANDIDATE, reference)
                     for reference in DESIGNS if reference != CANDIDATE},
        "clean_pairwise": {reference: pairwise_summary(
            rows, CANDIDATE, reference, CLEAN_METRIC) for reference in HEADLINE_REFERENCES},
        "all_grid_convergence_absolute": {
            "median": float(np.quantile(absolute, .5, method="linear")),
            "p95": float(np.quantile(absolute, .95, method="linear")),
            "max": float(np.max(absolute)),
        },
        "candidate_receipts": receipts,
        "candidate_calibration_receipts": calibration,
        "candidate_action_path_receipts": action_path,
        "action_path_receipts_by_design": action_path_by_design,
        "action_semantic_control_receipts": semantic_controls,
        "comparison_validity_receipts": comparison_validity,
        "action_only_control": action_control,
        "initial_encoder_integrator_control": initial_control,
    }


def _pair_gate(summary, reduction, wins, env_wins):
    return (summary["mean_paired_relative_reduction"] >= reduction,
            summary["paired_wins"] >= wins,
            summary["environment_mean_wins"] >= env_wins)


def decision_criteria(observed, rows, *, final):
    criteria = {}
    pairwise = observed["pairwise"]
    for reference in HEADLINE_REFERENCES:
        threshold = 15 if final else 9
        for suffix, passed in zip(
                ("reduction_ge_5pct", f"wins_ge_{threshold}", "env_wins_ge_4_of_5"),
                _pair_gate(pairwise[reference], .05, threshold, 4)):
            criteria[f"vs_{reference}_{suffix}"] = passed
    for reference in (
            "kdiov11_unconstrained", "kdiov11_fixedscale",
            "kdiov11_noactionswap"):
        criteria[f"vs_{reference}_equal_task_paired_reduction_ge_2pct"] = (
            pairwise[reference]["mean_paired_relative_reduction"] >= .02)
    mechanism_reduction = {
        "kdiov11_nocalibration": .02, "kdiov11_diagonal": .02,
        "kdiov11_h1": .02, "kdiov11_firstorder": .02, "kdiov11_nodrift": .05,
        "kdiov11_noautonomy": .02, "kdiov11_noaction": .05,
        "kdiov11_nosuffix": .02,
        "kdiov11_noreliability": .02,
    }
    for reference, reduction in mechanism_reduction.items():
        threshold = 14 if final else 9
        for suffix, passed in zip(
                (f"reduction_ge_{int(reduction*100)}pct", f"wins_ge_{threshold}",
                 "env_wins_ge_3_of_5"),
                _pair_gate(pairwise[reference], reduction, threshold, 3)):
            criteria[f"vs_{reference}_{suffix}"] = passed
    for reference, summary in observed["clean_pairwise"].items():
        criteria[f"clean_harm_vs_{reference}_le_2pct"] = (
            summary["mean_paired_relative_reduction"] >= -.02)
    convergence = observed["all_grid_convergence_absolute"]
    receipt = observed["candidate_receipts"]
    calibration = observed["candidate_calibration_receipts"]
    initial = observed["initial_encoder_integrator_control"]
    semantic = observed["action_semantic_control_receipts"]
    criteria.update({
        "vs_initial_encoder_integrator_equal_task_mean_below": (
            initial["mean_paired_relative_reduction"] > 0.0),
        "vs_initial_encoder_integrator_env_wins_ge_3_of_5": (
            initial["environment_mean_wins"] >= 3),
        "convergence_median_lt_1pct": convergence["median"] < .01,
        "convergence_p95_lt_3pct": convergence["p95"] < .03,
        "convergence_max_lt_5pct": convergence["max"] < .05,
        "kdio_inverse_le_1e_5": receipt["kdio_inverse_max"] <= 1e-5,
        "kdio_volume_le_1e_7": receipt["kdio_volume_max"] <= 1e-7,
        "kdio_streaming_le_1e_5": receipt["kdio_streaming_max"] <= 1e-5,
        "prior_probe_ceiling_lt_1": receipt["prior_ceiling_max"] < 1,
        "inverse_action_r2_gt_0": receipt["inverse_r2_min"] > 0,
        "calibration_updates_eq_100": (
            calibration["updates_min"] == EPOCHS
            and calibration["updates_max"] == EPOCHS),
        "calibration_samples_eq_56400": (
            calibration["samples_min"] == CALIBRATION_SAMPLES
            and calibration["samples_max"] == CALIBRATION_SAMPLES),
        "calibration_oas_shrinkage_in_unit_interval": (
            0.0 <= calibration["oas_shrinkage_min"]
            and calibration["oas_shrinkage_max"] <= 1.0),
        "calibration_precision_positive": calibration["precision_singular_min"] > 0.0,
        "observer_ordered_gain_violation_le_1e_6": (
            receipt["observer_ordered_gain_violation_max"] <= 1e-6),
        "action_scale_finite_positive": receipt["action_scale_min"] > 0.0,
        "action_scale_gradient_active_eq_1": (
            receipt["action_scale_gradient_active_min"] == 1.0),
        "action_scale_parameter_retained_eq_1": (
            receipt["action_scale_parameter_retained_min"] == 1.0),
        "fixedscale_gamma_eq_1": (
            semantic["fixedscale_gamma_max_abs_error_from_one"] <= 1e-7),
        "unconstrained_rms_singular_eq_gamma": (
            semantic["unconstrained_rms_singular_gamma_max_abs_error"] <= 1e-5),
        "action_frame_gram_error_le_1e_5": (
            receipt["action_frame_gram_error_max"] <= 1e-5),
        "action_direction_gram_error_le_1e_5": (
            receipt["action_direction_gram_error_max"] <= 1e-5),
        "action_frame_singular_over_gamma_min_ge_0_99999": (
            receipt["action_frame_singular_over_gamma_min"] >= 1.0 - 1e-5),
        "action_frame_singular_over_gamma_max_le_1_00001": (
            receipt["action_frame_singular_over_gamma_max"] <= 1.0 + 1e-5),
        "action_frame_condition_le_1_00001": (
            receipt["action_frame_condition_max"] <= 1.0 + 1e-5),
        "action_frame_constrained_eq_1": (
            receipt["action_frame_constrained_min"] == 1.0),
        "asr_pair_accuracy_equal_task_mean_gt_chance": (
            receipt["asr_pair_accuracy_equal_task_mean"] > 0.5),
        "noaction_pair_accuracy_eq_chance": (
            semantic["noaction_pair_accuracy_max_abs_error_from_chance"] <= 1e-7),
        "noaction_semantic_receipts_eq_zero": (
            semantic["noaction_semantic_zero_max_abs"] <= 1e-7),
    })
    for horizon, advantage in receipt[
            "true_action_advantage_equal_task_means"].items():
        criteria[f"true_action_advantage_h{horizon}_equal_task_mean_gt_0"] = (
            advantage > 0.0)
    for design, validity in observed["comparison_validity_receipts"].items():
        criteria.update({
            f"{design}_encoder_variance_ge_1e_5": (
                validity["encoder_variance_min"] >= 1e-5),
            f"{design}_encoder_rank_ge_16": validity["encoder_rank_min"] >= 16,
            f"{design}_encoder_singleton_le_1e_5": (
                validity["encoder_singleton_max"] <= 1e-5),
            f"{design}_encoder_prefix_le_1e_5": (
                validity["encoder_prefix_max"] <= 1e-5),
        })
    bootstrap = {}
    if final:
        for reference in HEADLINE_REFERENCES:
            receipt = crossed_bootstrap(pairwise_matrix(rows, CANDIDATE, reference), reference)
            bootstrap[reference] = receipt
            criteria[f"bootstrap90_lower_vs_{reference}_gt_0"] = receipt["ci90"][0] > 0
    return criteria, bootstrap


def pilot_decision(rows, convergence):
    observed = observed_summary(rows, convergence)
    criteria, _ = decision_criteria(observed, rows, final=False)
    passed = all(criteria.values())
    return {
        "schema_version": 1, "phase": "pilot",
        "decision": "PILOT_CONFIRMATION_PASS" if passed else "NO_GO",
        "pilot_screen_passed": passed, "criteria": criteria,
        "observed": {**observed, "bootstrap_contract": BOOTSTRAP_CONTRACT,
                     "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256},
        "scope": SCOPE, "note": "Immutable seeds 0--2; completion is mandatory.",
    }


def final_decision(rows, convergence, *, pilot_screen_passed):
    observed = observed_summary(rows, convergence)
    criteria, bootstrap = decision_criteria(observed, rows, final=True)
    gates = all(criteria.values())
    confirmed = bool(pilot_screen_passed and gates)
    label = ("END_TO_END_CONFIRMATION_PASS" if confirmed else
             "PILOT_NO_GO_FINAL_DESCRIPTIVE" if not pilot_screen_passed else "NO_GO")
    return {
        "schema_version": 1, "phase": "final", "decision": label,
        "pilot_screen_passed": bool(pilot_screen_passed), "final_gates_passed": gates,
        "end_to_end_confirmation_passed": confirmed,
        "scoped_component_confirmation_passed": confirmed,
        "iclr_submission_ready": False, "criteria": criteria,
        "completed_runs": len(rows), "scope": SCOPE,
        "observed": {**observed, "bootstrap": bootstrap,
                     "bootstrap_contract": BOOTSTRAP_CONTRACT,
                     "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256},
        "limitations": [
            "V11 is adaptive development on the five already opened V10 tasks.",
            "Native task observations are used only by post-training train-split probes; raw physics state is unused.",
            "No executed policy-return claim is made by this study.",
        ],
    }


def contrast_rows(rows):
    output = []
    for metric in (PRIMARY, CLEAN_METRIC):
        for reference in DESIGNS:
            if reference == CANDIDATE:
                continue
            summary = pairwise_summary(rows, CANDIDATE, reference, metric)
            output.append({key: summary[key] for key in (
                "candidate", "reference", "metric", "n_pairs",
                "mean_paired_relative_reduction", "paired_wins", "paired_ties",
                "environment_mean_wins") } | {"env": "__overall__"})
            for environment, reduction in summary["environment_mean_reductions"].items():
                output.append({
                    "candidate": CANDIDATE, "reference": reference, "metric": metric,
                    "env": environment, "n_pairs": len({row["seed"] for row in rows}),
                    "mean_paired_relative_reduction": reduction,
                    "paired_wins": "", "paired_ties": "", "environment_mean_wins": "",
                })
    return output


def strict_validate_cells(root: Path, seeds: Sequence[int]):
    import scripts.run_hacssm_v11 as runner
    original = runner.OUTPUT_ROOT
    try:
        runner.OUTPUT_ROOT = root.resolve()
        runner.configure_shared()
        jobs = tuple(runner.shared.Job(
            "pilot" if seed in PILOT_SEEDS else "completion", seed,
            environment, environment, design)
            for seed in seeds for environment in ENVIRONMENTS for design in DESIGNS)
        for job in jobs:
            runner.validate_job(job, allow_missing=False)
    finally:
        runner.OUTPUT_ROOT = original
        runner.configure_shared()


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    args = parser.parse_args(argv)
    seeds = PILOT_SEEDS if args.phase == "pilot" else FINAL_SEEDS
    strict_validate_cells(args.root, seeds)
    rows, convergence = load_cells(args.root, seeds)
    grouped, contrasts = shared.grouped_rows(rows), contrast_rows(rows)
    prefix = "pilot_" if args.phase == "pilot" else ""
    if args.phase == "pilot":
        decision = pilot_decision(rows, convergence)
    else:
        pilot = shared.read_json(args.root / "pilot_decision.json")
        pilot_rows = [row for row in rows if row["seed"] in PILOT_SEEDS]
        pilot_convergence = [row for row in convergence if row["seed"] in PILOT_SEEDS]
        recomputed = pilot_decision(pilot_rows, pilot_convergence)
        if pilot != recomputed:
            raise ValueError("immutable pilot decision mismatch")
        decision = final_decision(
            rows, convergence, pilot_screen_passed=recomputed["pilot_screen_passed"])
    shared.atomic_csv(args.root / f"{prefix}per_run.csv", rows)
    shared.atomic_csv(args.root / f"{prefix}grouped.csv", grouped)
    shared.atomic_csv(args.root / f"{prefix}paired_contrasts.csv", contrasts)
    shared.atomic_csv(args.root / f"{prefix}convergence.csv", convergence)
    shared.atomic_json(
        args.root / ("pilot_decision.json" if args.phase == "pilot" else "decision.json"),
        decision)
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
