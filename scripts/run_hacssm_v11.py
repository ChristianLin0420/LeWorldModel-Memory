#!/usr/bin/env python3
"""Run the frozen 400-cell KDIO-v11 predictive-state study on four GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_hacssm_v5 as shared
from scripts.hacssm_v11_data import (
    ACTION_PROCESS, DEFAULT_CORRUPTION_SEED, DEFAULT_TRAIN_SEED,
    DEFAULT_VAL_SEED, SCHEMA_VERSION as DATA_SCHEMA_VERSION, TASKS,
    cache_name, load_cache, sidecar_path,
)
from scripts.train_hacssm_v11 import (
    ACTION_SWAP_DESIGNS, CALIBRATED_DESIGNS, DESIGNS, HELDOUT_CONDITIONS, HISTORY_KEYS,
    KDIO_DESIGNS, OBJECTIVE,
    ROLLOUT_SCHEMA_VERSION, SUFFIX_DESIGNS, _design_metadata,
)


REPO_ROOT = ROOT
TRAIN_SCRIPT = ROOT / "scripts" / "train_hacssm_v11.py"
DATA_SCRIPT = ROOT / "scripts" / "hacssm_v11_data.py"
ANALYZE_SCRIPT = ROOT / "scripts" / "analyze_hacssm_v11.py"
DATA_ROOT = ROOT / "outputs" / "hacssm_v11_data"
OUTPUT_ROOT = ROOT / "outputs" / "hacssm_v11_shared"
LOG_ROOT = ROOT / "logs" / "hacssm_v11_shared"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PILOT_DECISION_PATH = OUTPUT_ROOT / "pilot_decision.json"
FINAL_DECISION_PATH = OUTPUT_ROOT / "decision.json"
MANIFEST_PATH = OUTPUT_ROOT / "hacssm_v11_manifest.json"
MANIFEST_SHA_PATH = OUTPUT_ROOT / "hacssm_v11_manifest.sha256"
LOCK_PATH = OUTPUT_ROOT / ".run_hacssm_v11.lock"

WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
WANDB_MODE = "online"
WANDB_STUDY = "hacssm-v11"
EVAL_ROLLOUT_EPISODE = 0
SCOPE = "adaptive_development_after_v10_host_audit"
ENVIRONMENTS = tuple((f"dmc:{task}", f"dmc:{task}") for task in TASKS)
PILOT_SEEDS = (0, 1, 2)
COMPLETION_SEEDS = (3, 4)
ALL_SEEDS = PILOT_SEEDS + COMPLETION_SEEDS
HELDOUT_CORRUPTIONS = HELDOUT_CONDITIONS
COMMON = {
    "data_schema_version": DATA_SCHEMA_VERSION,
    "train_episodes": 1200, "val_episodes": 240, "length": 48,
    "img_size": 64, "embed_dim": 128, "batch_size": 64,
    "learning_rate": 3e-4, "weight_decay": 1e-5, "history_len": 3,
    "encoder_layers": 6, "encoder_heads": 4,
    "predictor_layers": 4, "predictor_heads": 8,
    "encoder_norm": "causal", "predictor_norm": "none", "epochs": 100,
    "train_dataloader_workers": 2, "train_rollout_seed": DEFAULT_TRAIN_SEED,
    "val_rollout_seed": DEFAULT_VAL_SEED,
    "corruption_seed": DEFAULT_CORRUPTION_SEED,
    "action_process": ACTION_PROCESS, "smooth_rho": 0.0,
    "sigreg_lambda": 0.1, "sigreg_projections": 512,
    "training_objective": OBJECTIVE,
    "prediction_loss_weight": 1.0, "action_swap_loss_weight": 1.0,
    "inverse_loss_weight": 0.0,
    "innovation_calibration_family": (
        "epoch_end_reliability_open_clean_oas_with_identity_and_diagonal_controls"),
    "innovation_calibration_gradient_active": False,
    "calibration_nll_optimized": False,
    "variance_loss_weight": 1.0, "covariance_loss_weight": 1.0,
    "state_probe_ridge": 1e-3, "eval_target_key": "task_observation",
    "wandb": True, "wandb_entity": WANDB_ENTITY,
    "wandb_project": WANDB_PROJECT, "wandb_mode": WANDB_MODE,
    "wandb_study": WANDB_STUDY, "eval_rollout_episode": EVAL_ROLLOUT_EPISODE,
}

SOURCE_FILES = (
    Path("scripts/run_hacssm_v11.py"), Path("scripts/analyze_hacssm_v11.py"),
    Path("scripts/train_hacssm_v11.py"), Path("scripts/hacssm_v11_data.py"),
    Path("scripts/run_hacssm_v5.py"), Path("scripts/analyze_hacssm_v5.py"),
    Path("lewm/models/encoder.py"), Path("lewm/models/leworldmodel.py"),
    Path("lewm/models/memory.py"), Path("lewm/models/memory_model.py"),
    Path("lewm/models/sigreg.py"),
)
PILOT_ANALYSIS_FILES = frozenset({
    "pilot_per_run.csv", "pilot_grouped.csv", "pilot_paired_contrasts.csv",
    "pilot_convergence.csv", "pilot_decision.json",
})
FINAL_ANALYSIS_FILES = frozenset({
    "per_run.csv", "grouped.csv", "paired_contrasts.csv", "convergence.csv",
    "decision.json",
})
TOP_LEVEL_OUTPUT_FILES = frozenset({
    PROTOCOL_PATH.name, LOCK_PATH.name, MANIFEST_PATH.name, MANIFEST_SHA_PATH.name,
    *PILOT_ANALYSIS_FILES, *FINAL_ANALYSIS_FILES,
})

OBSERVER_KEYS = (
    "innovation_ratio", "innovation_energy", "process_tolerance", "reliability",
    "position_base_gain", "velocity_base_gain", "velocity_base_ratio",
    "q_gates", "v_gates", "action_effect_norm",
    "action_tanh_derivative_mean", "action_tanh_saturation_proxy",
)
OBSERVER_PHASES = ("all", "gap", "deep", "first_post", "post", "primary")
OBSERVER_DATASETS = ("clean", *HELDOUT_CORRUPTIONS)
CALIBRATION_MEMORY_METRICS = (
    "memory_innovation_precision_diagonal_mean",
    "memory_innovation_precision_logdet_per_dim",
    "memory_innovation_precision_offdiagonal_norm",
    "memory_innovation_precision_singular_min",
    "memory_innovation_precision_singular_max",
    "memory_innovation_precision_condition",
    "memory_calibration_updates",
    "memory_calibration_samples",
    "memory_calibration_oas_shrinkage",
    "memory_calibration_covariance_condition",
    "memory_calibration_diagonal_only",
    "memory_gradient_trained_parameters",
    "memory_nominal_optimizer_scalars",
    "memory_fitted_memory_scalars",
    "memory_total_memory_scalars",
    "memory_clean_innovation_mean_norm",
)
KDIO_MECHANISM_METRICS = (
    "memory_action_scale", "memory_action_log_scale",
    "memory_action_scale_parameter_retained",
    "memory_action_scale_gradient_active",
    "memory_action_kick_norm", "memory_action_raw_norm",
    "memory_action_direction_norm", "memory_action_effective_norm",
    "memory_action_parameter_norm",
    "memory_action_parameter_singular_min", "memory_action_parameter_singular_max",
    "memory_action_parameter_condition", "memory_action_frame_gram_error",
    "memory_action_frame_singular_min", "memory_action_frame_singular_max",
    "memory_action_frame_condition", "memory_action_frame_constrained",
    "memory_action_direction_gram_error",
    "memory_action_direction_singular_min",
    "memory_action_direction_singular_max",
    "memory_action_direction_condition",
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


def observer_metric_names():
    """Every finite observer phase receipt emitted for a KDIO evaluation dataset."""
    names = []
    for dataset in OBSERVER_DATASETS:
        for key in OBSERVER_KEYS:
            for phase in OBSERVER_PHASES:
                names.extend((
                    f"{dataset}_observer_{key}_{phase}_mean",
                    f"{dataset}_observer_{key}_{phase}_std",
                ))
        names.append(f"{dataset}_observer_ordered_gain_violation_max")
    return tuple(names)


def make_jobs(stage: str, seeds: Sequence[int]):
    return tuple(shared.Job(stage, seed, environment, clean, design)
                 for seed in seeds for environment, clean in ENVIRONMENTS
                 for design in DESIGNS)


PILOT_JOBS = make_jobs("pilot", PILOT_SEEDS)
COMPLETION_JOBS = make_jobs("completion", COMPLETION_SEEDS)
ALL_JOBS = PILOT_JOBS + COMPLETION_JOBS
assert len(PILOT_JOBS) == 240 and len(COMPLETION_JOBS) == 160
assert len(ALL_JOBS) == 400 and len({job.run_name for job in ALL_JOBS}) == 400
RunnerError = shared.RunnerError


def data_paths(environment: str):
    task = environment.removeprefix("dmc:")
    return (
        DATA_ROOT / cache_name(
            task, "train", COMMON["train_episodes"], COMMON["length"],
            COMMON["img_size"], COMMON["train_rollout_seed"]),
        DATA_ROOT / cache_name(
            task, "val", COMMON["val_episodes"], COMMON["length"],
            COMMON["img_size"], COMMON["val_rollout_seed"]),
        DATA_ROOT / "manifest.json",
    )


def source_snapshot():
    records = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise RunnerError(f"missing V11 source {relative}")
        records[relative.as_posix()] = shared.file_record(path)
    return dict(sorted(records.items()))


def data_snapshot():
    records, expected_paths = {}, set()
    task_schemas = {}
    for environment, _ in ENVIRONMENTS:
        train_path, val_path, manifest_path = data_paths(environment)
        for split, path, episodes, seed in (
            ("train", train_path, COMMON["train_episodes"], COMMON["train_rollout_seed"]),
            ("val", val_path, COMMON["val_episodes"], COMMON["val_rollout_seed"]),
        ):
            metadata = load_cache(path, verify=True)
            expected = (
                environment.removeprefix("dmc:"), split, seed, episodes,
                COMMON["length"], COMMON["img_size"], 0.0,
            )
            actual = (
                metadata.env_id, metadata.split, metadata.seed, metadata.episodes,
                metadata.length, metadata.img_size, metadata.smooth_rho,
            )
            if actual != expected:
                raise RunnerError(f"{path}: V11 cache metadata {actual} != {expected}")
            if metadata.task_observation_dim < 1 or not metadata.task_observation_keys:
                raise RunnerError(f"{path}: missing task-observation schema")
            schema = (metadata.task_observation_dim, metadata.task_observation_keys,
                      metadata.task_observation_shapes)
            previous = task_schemas.setdefault(environment, schema)
            if previous != schema:
                raise RunnerError(f"{environment}: train/val task-observation schemas differ")
            for artifact in (path, sidecar_path(path)):
                expected_paths.add(artifact.resolve())
                records[shared.rel(artifact)] = shared.file_record(artifact)
        expected_paths.add(manifest_path.resolve())
        expected_paths.add((DATA_ROOT / "manifest.sha256").resolve())
    manifest = DATA_ROOT / "manifest.json"
    sidecar = DATA_ROOT / "manifest.sha256"
    records[shared.rel(manifest)] = shared.file_record(manifest)
    records[shared.rel(sidecar)] = shared.file_record(sidecar)
    actual_paths = {path.resolve() for path in DATA_ROOT.iterdir()} if DATA_ROOT.is_dir() else set()
    if actual_paths != expected_paths:
        raise RunnerError(
            f"V11 data namespace differs: missing={sorted(map(str, expected_paths-actual_paths))[:4]} "
            f"extra={sorted(map(str, actual_paths-expected_paths))[:4]}")
    payload = shared.read_json(manifest)
    protocol = payload.get("protocol") if isinstance(payload, dict) else None
    wanted = {
        "tasks": list(TASKS), "splits": ["train", "val"],
        "train_episodes": COMMON["train_episodes"],
        "val_episodes": COMMON["val_episodes"], "length": COMMON["length"],
        "img_size": COMMON["img_size"], "train_seed": COMMON["train_rollout_seed"],
        "val_seed": COMMON["val_rollout_seed"], "smooth_rho": 0.0,
        "action_process": ACTION_PROCESS,
        "primary_evaluation_target": "flattened_native_task_observation",
        "secondary_evaluation_target": "raw_physics_state",
        "evaluation_targets_used_for_training": False,
        "cache_role": "clean_only_corruptions_are_deterministic_dataset_views",
        "corruption_seed": DEFAULT_CORRUPTION_SEED,
    }
    if not shared.stable_equal(protocol, wanted):
        raise RunnerError("V11 data manifest protocol mismatch")
    return dict(sorted(records.items()))


def precollect_data(python: str):
    command = [
        python, shared.rel(DATA_SCRIPT), "--root", shared.rel(DATA_ROOT), "--all",
        "--train-episodes", str(COMMON["train_episodes"]),
        "--val-episodes", str(COMMON["val_episodes"]),
        "--length", str(COMMON["length"]), "--img-size", str(COMMON["img_size"]),
        "--train-seed", str(COMMON["train_rollout_seed"]),
        "--val-seed", str(COMMON["val_rollout_seed"]), "--smooth-rho", "0.0",
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise RunnerError(f"V11 data preparation failed with status {result.returncode}")
    data_snapshot()


def memory_contract():
    from lewm.models.memory import (
        KDIOv11Memory, ORBITv10Memory, SSMMemory, SharedActionShrinkageMemory)
    dimension, action_dim = COMMON["embed_dim"], 6
    kdio_modes = {
        "kdiov11": "full", "kdiov11_unconstrained": "unconstrained",
        "kdiov11_fixedscale": "fixedscale", "kdiov11_h1": "full",
        "kdiov11_noactionswap": "full",
        "kdiov11_nosuffix": "full", "kdiov11_nocalibration": "full",
        "kdiov11_diagonal": "full", "kdiov11_noreliability": "noreliability",
        "kdiov11_firstorder": "firstorder", "kdiov11_nodrift": "nodrift",
        "kdiov11_noaction": "noaction", "kdiov11_noautonomy": "noautonomy",
    }
    models = {name: KDIOv11Memory(dimension, action_dim, mode=mode)
              for name, mode in kdio_modes.items()}
    signatures = {name: [(key, list(value.shape)) for key, value in model.named_parameters()]
                  for name, model in models.items()}
    counts = {name: model.parameter_count() for name, model in models.items()}
    nominal_fitted_counts = {
        name: model.expected_fitted_scalar_count(dimension, action_dim)
        for name, model in models.items()
    }
    total_counts = {
        name: model.expected_total_scalar_count(dimension, action_dim)
        for name, model in models.items()
    }
    stiefel_gauge = action_dim * (action_dim + 1) // 2
    action_matrix_scalars = dimension * action_dim
    effective_optimizer_dof = {
        name: (
            counts[name] - 1 if mode == "unconstrained" else
            counts[name] - action_matrix_scalars - 1 if mode == "noaction" else
            counts[name] - stiefel_gauge - 1 if mode == "fixedscale" else
            counts[name] - stiefel_gauge - 2 * dimension
            if mode == "noautonomy" else
            counts[name] - stiefel_gauge - (dimension + 1)
            if mode == "noreliability" else
            counts[name] - stiefel_gauge)
        for name, mode in kdio_modes.items()
    }
    full_calibration_scalars = dimension * (dimension - 1) // 2 + dimension - 1
    diagonal_calibration_scalars = 2 * (dimension - 1)
    active_fitted_counts = {
        name: (0 if name in {"kdiov11_nocalibration", "kdiov11_noreliability"}
               else diagonal_calibration_scalars if name == "kdiov11_diagonal"
               else full_calibration_scalars)
        for name in kdio_modes
    }
    effective_plus_fitted = {
        name: effective_optimizer_dof[name] + active_fitted_counts[name]
        for name in kdio_modes}
    if set(kdio_modes) != set(KDIO_DESIGNS):
        raise RunnerError("KDIO mode map does not cover the frozen design set")
    if len({json.dumps(value) for value in signatures.values()}) != 1 or len(set(counts.values())) != 1:
        raise RunnerError("KDIO controls are not parameter/signature matched")
    return {
        "embed_dim": dimension, "reference_action_dim": action_dim,
        "memory_parameters": {
            **counts,
            "ssm": sum(p.numel() for p in SSMMemory(dimension).parameters()),
            "hacssmv8": SharedActionShrinkageMemory(
                dimension, action_dim, mode="learned").parameter_count(),
            "orbitv10": ORBITv10Memory(
                dimension, action_dim, mode="orthogonal").parameter_count(),
        },
        "streaming_recurrent_floats": {
            **{design: 2 * dimension for design in KDIO_DESIGNS},
            "ssm": dimension, "hacssmv8": 2 * dimension, "orbitv10": dimension,
        },
        "kdio_parameter_signature": signatures["kdiov11"],
        "kdio_mode_by_design": kdio_modes,
        "kdio_nominal_optimizer_scalars_by_design": counts,
        "kdio_nominal_fitted_oas_scalars_by_design": nominal_fitted_counts,
        "kdio_nominal_total_memory_scalars_by_design": total_counts,
        "kdio_functionally_effective_optimizer_dof_by_design": effective_optimizer_dof,
        "kdio_functionally_active_fitted_scalars_by_design": active_fitted_counts,
        "kdio_nominal_optimizer_scalars": counts["kdiov11"],
        "kdio_nominal_fitted_oas_scalars": nominal_fitted_counts["kdiov11"],
        "kdio_nominal_total_memory_scalars": total_counts["kdiov11"],
        "kdio_effective_plus_fitted_scalars_by_design": effective_plus_fitted,
        "active_count_caveat": (
            "effective counts are functional dimensions, not optimizer tensor counts; "
            "QR fibers, Frobenius radial normalization, and exact controls retain tensors"),
        "stiefel_gauge_dimension": stiefel_gauge,
        "action_frame_map": (
            "full gamma*qf(M); fixedscale qf(M); unconstrained "
            "gamma*sqrt(A)*M/||M||_F"),
        "action_scale_map": "gamma=exp(log_action_scale) in FP32 without clipping",
        "kdio_calibration_parameters_require_grad": False,
        "trainable_inverse_head_parameters": 0,
        "serialized_evaluation_inverse_ridge_shapes": {
            "x_mean": [3 * dimension], "x_std": [3 * dimension],
            "y_mean": [action_dim], "y_std": [action_dim],
            "weights": [3 * dimension + 1, action_dim],
        },
    }


def build_protocol(commit: str, clean: bool, wandb_preflight):
    import scripts.analyze_hacssm_v11 as analysis
    return {
        "schema_version": 1,
        "study": "KDIO-v11 end-to-end predictive-state adaptive study",
        "study_id": WANDB_STUDY, "scope": SCOPE,
        "producer_git_commit": commit, "producer_git_clean": clean,
        "common_protocol": COMMON,
        "data_contract": {
            "tasks": [environment for environment, _ in ENVIRONMENTS],
            "iid_actions": True, "smooth_rho": 0.0,
            "training_corruptions": ["cutout", "meanframe"],
            "heldout_corruptions": list(HELDOUT_CORRUPTIONS),
            "primary_evaluation_target": "flattened native DMC task observation",
            "physics_state_used_for_training": False,
            "task_observation_used_for_training": False,
            "primary_metric": "heldout_prior_state_nmse",
            "primary_representation": "strictly pre-observation transition prior",
        },
        "architecture_contract": {
            "candidate": "kdiov11",
            "state": "position q and velocity v, each D",
            "transition": "state-dependent action kick then velocity-to-position drift",
            "action_frame": {
                "candidate_and_structural_modes": (
                    "gamma*U, U=qf(M), U^T U=I_A, gamma=exp(log_gamma)"),
                "fixedscale_control": "U=qf(M), gamma exactly one",
                "unconstrained_control": "gamma*sqrt(A)*M/||M||_F",
                "qr_arithmetic": "canonical-sign thin QR in FP32",
                "scale_arithmetic": "unclipped positive exponential in FP32",
            },
            "read": "RMSNorm(q+v)", "fixed_decay_or_horizon": False,
            "one_token_predictor": True,
            "predictive_average": "equal mean of context and suffix losses",
            "suffix_weighting": "equal over every available horizon k",
            "suffix_anchor": "observed/corrupted deployed posterior",
            "action_swap_ranking": (
                "softplus(E_true-E_batch-deranged) averaged equally over horizons"),
            "inverse_action_training": False,
            "inverse_action_evaluation": (
                "serialized clean-train ridge [z[t-1],z[t],z[t+1]] -> a[t]"),
            "objective": OBJECTIVE,
            "objective_weights": {"predictive": 1.0, "action_swap": 1.0,
                                  "inverse": 0.0,
                                  "variance": 1.0, "covariance": 1.0,
                                  "calibration_nll": 0.0},
            "innovation_calibration": {
                "coordinate": "fixed orthonormal Helmert D-1 contrast",
                "fit_input": "clean deployed recurrence with reliability r forced to one",
                "estimator": "epoch-end closed-form OAS; no tuned ridge or loss weight",
                "metric": "FP32 Gaussian diagnostic NLL excluded from optimized total",
                "precision_convention": "lower whitening C with Lambda=C^T C",
                "gradient_semantics": (
                    "C and mu are non-gradient OAS statistics; tau remains predictive"),
                "corruption_label_or_visibility_mask": False,
            },
            "action_path_diagnostics": {
                "available": (
                    "gamma, raw/direction/effective action-frame Gram and singular receipts, "
                    "per-phase effect/saturation diagnostics, ASR pair accuracy, per-horizon "
                    "true-action advantages and rollout divergence, exact mode flags"),
                "causal_control": "paired kdiov11_noaction performance contrast",
            },
            "variants": {design: _design_metadata(design) for design in DESIGNS},
        },
        "memory_contract": memory_contract(),
        "data_artifacts": data_snapshot(), "source_artifacts": source_snapshot(),
        "output_root": shared.rel(OUTPUT_ROOT), "log_root": shared.rel(LOG_ROOT),
        "wandb": wandb_preflight,
        "wandb_requirements": {"all_cells_online": True,
                               "complete_epoch_history_per_cell": COMMON["epochs"],
                               "total_epoch_rows": 40_000,
                               "rollout_npz_table_video_per_cell": True,
                               "oas_epoch_rows_for_calibrated_kdio": COMMON["epochs"],
                               "calibration_observer_action_asr_summary_receipts": True},
        "stages": {
            "pilot": {"seeds": list(PILOT_SEEDS), "runs": len(PILOT_JOBS)},
            "completion": {"seeds": list(COMPLETION_SEEDS),
                           "runs": len(COMPLETION_JOBS), "mandatory": True},
        },
        "pilot_success_criteria": {
            "vs_each_ssm_v8_orbit": ">=5%, >=9/15 paired wins, >=4/5 task means",
            "vs_unconstrained_fixedscale_noactionswap": (
                ">=2% equal-task paired held-out reduction"),
            "vs_h1_nosuffix_nocalibration_diagonal_firstorder_noautonomy_noreliability": (
                ">=2%, >=9/15, >=3/5"),
            "vs_nodrift_noaction": ">=5%, >=9/15, >=3/5",
            "vs_action_only_integrator": "descriptive",
            "vs_initial_frame_integrator": "equal-task mean below, >=3/5 task means",
            "clean_harm_vs_each_anchor": "<=2%",
            "all_grid_convergence": "absolute median <1%, p95 <3%, max <5%",
            "kdio_inverse_and_streaming": "<=1e-5; volume error <=1e-7",
            "comparison_encoder_causality": (
                "candidate, SSM, V8, and ORBIT singleton and prefix max <=1e-5"),
            "comparison_encoder_quality": (
                "candidate, SSM, V8, and ORBIT variance >=1e-5 and effective rank >=16"),
            "belief_quality": (
                "prior probe ceiling <1 and evaluation-only inverse-ridge action R2 >0"),
            "gamma_and_action_semantics": (
                "gamma finite positive; fixedscale=1; ASR pair accuracy >.5 and true-action "
                "advantage >0 at h1/4/8/16/47; noaction exactly chance/zero"),
            "calibration_receipt": (
                "100 OAS updates, 56,400 clean innovations, finite positive precision, "
                "ordered observer gains"),
        },
        "final_success_criteria": {
            "requires_pilot_pass": True,
            "vs_each_ssm_v8_orbit": ">=5%, >=15/25 paired wins, >=4/5 task means",
            "vs_unconstrained_fixedscale_noactionswap": (
                ">=2% equal-task paired held-out reduction"),
            "vs_h1_nosuffix_nocalibration_diagonal_firstorder_noautonomy_noreliability": (
                ">=2%, >=14/25, >=3/5"),
            "vs_nodrift_noaction": ">=5%, >=14/25, >=3/5",
            "vs_action_only_integrator": "descriptive",
            "vs_initial_frame_integrator": "equal-task mean below, >=3/5 task means",
            "bootstrap_vs_each_anchor": "crossed task x seed 90% lower bound >0",
            "clean_harm_vs_each_anchor": "<=2%",
            "all_grid_convergence": "absolute median <1%, p95 <3%, max <5%",
            "mechanism_and_representation_receipts": "same as pilot",
        },
        "analysis_contract": {
            "primary": analysis.PRIMARY,
            "clean_metric": analysis.CLEAN_METRIC,
            "candidate": analysis.CANDIDATE,
            "headline_references": list(analysis.HEADLINE_REFERENCES),
            "mechanism_controls": list(analysis.MECHANISM_CONTROLS),
            "bootstrap": analysis.BOOTSTRAP_CONTRACT,
            "bootstrap_contract_sha256": analysis.BOOTSTRAP_CONTRACT_SHA256,
            "failed_pilot_cannot_be_reopened": True,
            "completion_is_mandatory": True,
        },
        "job_names": {"pilot": [job.run_name for job in PILOT_JOBS],
                      "completion": [job.run_name for job in COMPLETION_JOBS]},
    }


def expected_args(job):
    train_path, val_path, _ = data_paths(job.clean_env)
    return {
        "train_data": shared.rel(train_path), "val_data": shared.rel(val_path),
        "memory_mode": job.design, "seed": job.seed,
        "output_dir": shared.rel(OUTPUT_ROOT), "epochs": COMMON["epochs"],
        "batch_size": COMMON["batch_size"], "lr": COMMON["learning_rate"],
        "weight_decay": COMMON["weight_decay"],
        "num_workers": COMMON["train_dataloader_workers"],
        "img_size": COMMON["img_size"], "patch_size": 8,
        "embed_dim": COMMON["embed_dim"], "encoder_layers": COMMON["encoder_layers"],
        "encoder_heads": COMMON["encoder_heads"],
        "predictor_layers": COMMON["predictor_layers"],
        "predictor_heads": COMMON["predictor_heads"],
        "history_len": COMMON["history_len"], "dropout": 0.1,
        "sigreg_lambda": COMMON["sigreg_lambda"],
        "sigreg_projections": COMMON["sigreg_projections"],
        "probe_ridge": COMMON["state_probe_ridge"],
        "eval_target_key": COMMON["eval_target_key"],
        "corruption_seed": COMMON["corruption_seed"],
        "eval_rollout_episode": EVAL_ROLLOUT_EPISODE, "no_amp": False,
        "device": "cuda", "wandb": True, "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT, "wandb_mode": WANDB_MODE,
        "wandb_study": WANDB_STUDY, "extra_tag": "",
    }


def train_command(python: str, job):
    command = [python, shared.rel(TRAIN_SCRIPT)]
    for key, value in expected_args(job).items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
        else:
            command.extend((option, str(value)))
    return command


def validate_history(history, job):
    if not isinstance(history, list) or len(history) != COMMON["epochs"]:
        raise RunnerError(f"{job.run_name}: expected 100 history rows")
    for epoch, record in enumerate(history, 1):
        if not isinstance(record, dict) or set(record) != {
                "epoch", "epoch_seconds", "train", "val"} \
                or record["epoch"] != epoch:
            raise RunnerError(f"{job.run_name}: malformed history epoch {epoch}")
        if type(record["epoch_seconds"]) not in (int, float) \
                or not math.isfinite(float(record["epoch_seconds"])) \
                or float(record["epoch_seconds"]) <= 0:
            raise RunnerError(f"{job.run_name}: invalid epoch timing receipt")
        for split in ("train", "val"):
            values = record[split]
            if not isinstance(values, dict) or set(values) != set(HISTORY_KEYS):
                raise RunnerError(f"{job.run_name}: wrong {split} history keys")
            shared.assert_finite_tree(values, f"{job.run_name}.{epoch}.{split}")
            if any(float(values[key]) < 0 for key in HISTORY_KEYS
                   if key not in {"calibration_nll", "action_swap_advantage"}):
                raise RunnerError(f"{job.run_name}: negative history metric")
            predictive = .5 * (float(values["context_loss"]) + float(values["suffix_loss"]))
            if not math.isclose(float(values["predictive_loss"]), predictive,
                                rel_tol=1e-5, abs_tol=1e-7):
                raise RunnerError(f"{job.run_name}: predictive average mismatch")
            total = sum(float(values[key]) for key in (
                "predictive_loss", "action_swap_loss",
                "variance_loss", "covariance_loss"))
            if not math.isclose(float(values["loss"]), total, rel_tol=1e-5, abs_tol=1e-7):
                raise RunnerError(f"{job.run_name}: objective sum mismatch")
            if job.design in SUFFIX_DESIGNS:
                wanted_horizons = 1 if job.design == "kdiov11_h1" else COMMON["length"] - 1
                if (values["suffix_applicable"] != 1.0
                        or values["suffix_horizons"] != wanted_horizons
                        or values["action_swap_applicable"] != 1.0
                        or values["action_swap_horizons"] != wanted_horizons):
                    raise RunnerError(f"{job.run_name}: suffix contract mismatch")
                if not 0.0 <= float(values["action_swap_pair_accuracy"]) <= 1.0:
                    raise RunnerError(f"{job.run_name}: invalid ASR pair accuracy")
                expected_advantage = (
                    float(values["action_swap_negative_energy"])
                    - float(values["action_swap_positive_energy"]))
                if not math.isclose(float(values["action_swap_advantage"]),
                                    expected_advantage, rel_tol=1e-5, abs_tol=1e-7):
                    raise RunnerError(f"{job.run_name}: ASR energy advantage mismatch")
                if job.design == "kdiov11_noactionswap":
                    if values["action_swap_loss"] != 0.0:
                        raise RunnerError(
                            f"{job.run_name}: noactionswap optimized loss is nonzero")
                elif not math.isclose(
                        float(values["action_swap_loss"]),
                        float(values["action_swap_diagnostic_loss"]),
                        rel_tol=1e-6, abs_tol=1e-8):
                    raise RunnerError(f"{job.run_name}: active ASR diagnostic mismatch")
                if job.design == "kdiov11_noaction":
                    noaction_asr = {
                        "action_swap_advantage": 0.0,
                        "action_swap_pair_accuracy": 0.5,
                        "action_swap_loss": math.log(2.0),
                        "action_swap_diagnostic_loss": math.log(2.0),
                    }
                    for key, wanted in noaction_asr.items():
                        if not math.isclose(float(values[key]), wanted,
                                            rel_tol=0.0, abs_tol=1e-5):
                            raise RunnerError(
                                f"{job.run_name}: noaction ASR receipt {key} mismatch")
            elif (values["suffix_applicable"] != 0.0 or values["suffix_horizons"] != 0.0
                  or not math.isclose(values["suffix_loss"], values["context_loss"],
                                      rel_tol=0.0, abs_tol=0.0)):
                raise RunnerError(
                    f"{job.run_name}: no-suffix objective must equal context exactly")
            if job.design not in SUFFIX_DESIGNS:
                for key in (
                        "action_swap_loss", "action_swap_diagnostic_loss",
                        "action_swap_positive_energy", "action_swap_negative_energy",
                        "action_swap_advantage", "action_swap_pair_accuracy",
                        "action_swap_applicable", "action_swap_horizons"):
                    if values[key] != 0.0:
                        raise RunnerError(
                            f"{job.run_name}: inapplicable ASR receipt {key} is nonzero")
            wanted_calibration = 1.0 if job.design in CALIBRATED_DESIGNS else 0.0
            if values["calibration_applicable"] != wanted_calibration:
                raise RunnerError(f"{job.run_name}: calibration diagnostic contract mismatch")
            if not wanted_calibration and values["calibration_nll"] != 0.0:
                raise RunnerError(
                    f"{job.run_name}: inapplicable calibration NLL must be exactly zero")


def _validate_probe(probe, job, label, feature_dim, target_dim):
    wanted = {"x_mean": (feature_dim,), "x_std": (feature_dim,),
              "y_mean": (target_dim,), "y_std": (target_dim,),
              "weights": (feature_dim + 1, target_dim)}
    if not isinstance(probe, dict) or set(probe) != set(wanted):
        raise RunnerError(f"{job.run_name}: malformed {label} probe")
    for key, shape in wanted.items():
        value = np.asarray(probe[key])
        if value.shape != shape or value.dtype.hasobject or not np.isfinite(value).all():
            raise RunnerError(f"{job.run_name}: invalid {label}.{key}")
        if key.endswith("_std") and np.any(value <= 0):
            raise RunnerError(f"{job.run_name}: nonpositive {label}.{key}")


def validate_rollout(job, metrics, target_dim):
    if shared.sha256_file(job.eval_rollout_path) != metrics.get("eval_rollout_sha256"):
        raise RunnerError(f"{job.run_name}: rollout hash mismatch")
    with np.load(job.eval_rollout_path, allow_pickle=False) as rollout:
        base = {"schema_version", "episode_index", "conditions", "condition",
                "target_times", "phase", "state_target"}
        base |= {f"{coordinate}_{field}" for coordinate in (
            "prior", "posterior", "encoder", "predictor")
                 for field in ("state_prediction", "state_nmse")}
        per_condition = {f"{condition}_{field}" for condition in HELDOUT_CORRUPTIONS
                         for field in (
                             "target_times", "phase", "gap_start", "gap_end", "observed_rgb",
                             "clean_rgb", "actions", "evaluation_target",
                             *(f"{coordinate}_{kind}" for coordinate in
                               ("prior", "posterior", "encoder", "predictor")
                               for kind in ("state_prediction", "state_nmse_by_target_t")),
                         )}
        if set(rollout.files) != base | per_condition:
            raise RunnerError(f"{job.run_name}: rollout field mismatch")
        if int(rollout["schema_version"]) != ROLLOUT_SCHEMA_VERSION \
                or int(rollout["episode_index"]) != EVAL_ROLLOUT_EPISODE:
            raise RunnerError(f"{job.run_name}: rollout schema/episode mismatch")
        if tuple(np.asarray(rollout["conditions"]).astype(str).tolist()) != HELDOUT_CORRUPTIONS:
            raise RunnerError(f"{job.run_name}: rollout condition order mismatch")
        for name in rollout.files:
            value = np.asarray(rollout[name])
            if value.dtype.hasobject or value.size == 0 or (
                    np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all()):
                raise RunnerError(f"{job.run_name}: invalid rollout array {name}")
        for condition in HELDOUT_CORRUPTIONS:
            if rollout[f"{condition}_evaluation_target"].shape != (
                    COMMON["length"] - COMMON["history_len"], target_dim):
                raise RunnerError(f"{job.run_name}: rollout target shape mismatch")
            for coordinate in ("prior", "posterior", "encoder", "predictor"):
                if rollout[f"{condition}_{coordinate}_state_prediction"].shape != (
                        COMMON["length"] - COMMON["history_len"], target_dim):
                    raise RunnerError(f"{job.run_name}: rollout prediction shape mismatch")


def validate_tracking_receipt(job, metrics):
    receipt = shared.read_json(job.wandb_run_path)
    run_id = receipt.get("run_id") if isinstance(receipt, dict) else None
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise RunnerError(f"{job.run_name}: invalid W&B receipt")
    wanted = {
        "schema_version": 2, "run_name": f"{WANDB_STUDY}-{job.run_name}",
        "url": f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}",
        "entity": WANDB_ENTITY, "project": WANDB_PROJECT, "mode": WANDB_MODE,
        "study": WANDB_STUDY, "state": "finished",
        "eval_rollout_sha256": metrics["eval_rollout_sha256"],
        "eval_rollout_episode": EVAL_ROLLOUT_EPISODE,
    }
    for key, value in wanted.items():
        if not shared.stable_equal(receipt.get(key), value):
            raise RunnerError(f"{job.run_name}: W&B receipt {key} mismatch")
    if not receipt.get("eval_rollout_artifact_name") or not list(job.run_dir.rglob("run-*.wandb")):
        raise RunnerError(f"{job.run_name}: W&B transaction/artifact receipt absent")


def _finite_metric(metrics, key, job):
    value = metrics.get(key)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise RunnerError(f"{job.run_name}: missing/nonfinite metric {key}")
    return float(value)


def _validate_kdio_calibration_receipts(job, metrics, state, action_dim):
    """Fail closed on the closed-form calibration and observer diagnostic contract."""
    import torch

    for key in CALIBRATION_MEMORY_METRICS:
        _finite_metric(metrics, key, job)
    for key in KDIO_MECHANISM_METRICS:
        _finite_metric(metrics, key, job)
    for key in KDIO_ACTION_DIAGNOSTIC_METRICS:
        _finite_metric(metrics, key, job)
    for key in observer_metric_names():
        _finite_metric(metrics, key, job)

    required_state = {
        "world.mem_kdiov11.clean_innovation_mean",
        "world.mem_kdiov11.innovation_precision_packed",
        "world.mem_kdiov11.calibration_updates",
        "world.mem_kdiov11.calibration_samples",
        "world.mem_kdiov11.calibration_oas_shrinkage",
        "world.mem_kdiov11.calibration_covariance_condition",
        "world.mem_kdiov11.calibration_diagonal_only",
        "world.mem_kdiov11.W_a.weight", "world.mem_kdiov11.log_action_scale",
        "world.mem_kdiov11.w_q",
        "world.mem_kdiov11.b_f",
    }
    if not required_state.issubset(state):
        raise RunnerError(
            f"{job.run_name}: missing KDIO calibration state "
            f"{sorted(required_state - set(state))}")

    calibrated = job.design in CALIBRATED_DESIGNS
    expected_updates = COMMON["epochs"] if calibrated else 0
    expected_samples = (
        COMMON["train_episodes"] * (COMMON["length"] - 1) if calibrated else 0)
    expected_diagonal = 1.0 if job.design == "kdiov11_diagonal" else 0.0
    exact = {
        "memory_calibration_updates": float(expected_updates),
        "memory_calibration_samples": float(expected_samples),
        "memory_calibration_diagonal_only": expected_diagonal,
    }
    for key, wanted in exact.items():
        if not math.isclose(float(metrics[key]), wanted, rel_tol=0.0, abs_tol=0.0):
            raise RunnerError(f"{job.run_name}: exact calibration receipt {key} mismatch")

    firstorder = job.design == "kdiov11_firstorder"
    nodrift = job.design == "kdiov11_nodrift"
    noaction = job.design == "kdiov11_noaction"
    noautonomy = job.design == "kdiov11_noautonomy"
    noreliability = job.design == "kdiov11_noreliability"
    mechanism_exact = {
        "memory_kick_drift": float(not (firstorder or nodrift)),
        "memory_velocity_carry": float(not firstorder),
        "memory_position_drift": float(not nodrift),
        "memory_action_transport": float(not noaction),
        "memory_autonomous_transport": float(not noautonomy),
        "memory_prior_conditioned_correction": 1.0,
        "memory_innovation_reliability": float(not noreliability),
        "memory_ordered_correction": 1.0,
        "memory_invertible_transition": float(not firstorder),
        "memory_recurrent_floats": float(2 * COMMON["embed_dim"]),
    }
    for key, wanted in mechanism_exact.items():
        if float(metrics[key]) != wanted:
            raise RunnerError(f"{job.run_name}: exact KDIO mechanism receipt {key} mismatch")
    for key in (
            "memory_action_scale", "memory_action_kick_norm",
            "memory_action_raw_norm", "memory_action_direction_norm",
            "memory_action_effective_norm", "memory_action_parameter_norm",
            "memory_action_parameter_singular_min",
            "memory_action_parameter_singular_max",
            "memory_action_parameter_condition", "memory_action_frame_gram_error",
            "memory_action_frame_singular_min", "memory_action_frame_singular_max",
            "memory_action_frame_condition", "memory_action_direction_gram_error",
            "memory_action_direction_singular_min",
            "memory_action_direction_singular_max",
            "memory_action_direction_condition", "memory_state_kick_norm",
            "memory_autonomous_kick_norm", "memory_position_gain_vector_norm",
            "memory_velocity_ratio_vector_norm", "memory_process_tolerance_vector_norm"):
        if float(metrics[key]) < 0.0:
            raise RunnerError(f"{job.run_name}: negative KDIO mechanism norm {key}")

    raw_action_frame = state["world.mem_kdiov11.W_a.weight"].detach().float()
    if tuple(raw_action_frame.shape) != (COMMON["embed_dim"], action_dim):
        raise RunnerError(f"{job.run_name}: malformed raw KDIO action parameter")
    raw_singular = torch.linalg.svdvals(raw_action_frame)
    raw_min = raw_singular.min()
    raw_condition = raw_singular.max() / raw_min.clamp_min(
        torch.finfo(torch.float32).tiny)
    log_action_scale = state[
        "world.mem_kdiov11.log_action_scale"].detach().float().reshape(())
    fixedscale = job.design == "kdiov11_fixedscale"
    unconstrained = job.design == "kdiov11_unconstrained"
    action_scale = torch.ones((), dtype=torch.float32) if fixedscale else torch.exp(
        log_action_scale)
    if unconstrained:
        action_direction = (
            math.sqrt(action_dim) * raw_action_frame
            / raw_action_frame.norm().clamp_min(torch.finfo(torch.float32).tiny))
    else:
        action_direction, triangular = torch.linalg.qr(
            raw_action_frame, mode="reduced")
        sign = torch.where(
            torch.diagonal(triangular) < 0.0,
            -torch.ones(action_dim, dtype=action_direction.dtype),
            torch.ones(action_dim, dtype=action_direction.dtype),
        )
        action_direction = action_direction * sign.unsqueeze(0)
    effective_action_frame = action_scale * action_direction
    direction_singular = torch.linalg.svdvals(action_direction)
    direction_min = direction_singular.min()
    direction_condition = direction_singular.max() / direction_min.clamp_min(
        torch.finfo(torch.float32).tiny)
    effective_singular = torch.linalg.svdvals(effective_action_frame)
    effective_min = effective_singular.min()
    effective_condition = effective_singular.max() / effective_min.clamp_min(
        torch.finfo(torch.float32).tiny)
    identity = torch.eye(action_dim, dtype=effective_action_frame.dtype)
    direction_gram_error = (
        action_direction.T @ action_direction - identity).abs().max()
    gram_error = (
        effective_action_frame.T @ effective_action_frame
        - action_scale.square() * identity).abs().max()

    norm_receipts = {
        "memory_action_scale": float(action_scale),
        "memory_action_log_scale": float(log_action_scale),
        "memory_action_scale_parameter_retained": 1.0,
        "memory_action_scale_gradient_active": float(not (fixedscale or noaction)),
        "memory_action_kick_norm": float(effective_action_frame.norm()),
        "memory_action_raw_norm": float(raw_action_frame.norm()),
        "memory_action_direction_norm": float(action_direction.norm()),
        "memory_action_effective_norm": float(effective_action_frame.norm()),
        "memory_action_parameter_norm": float(raw_action_frame.norm()),
        "memory_action_parameter_singular_min": float(raw_min),
        "memory_action_parameter_singular_max": float(raw_singular.max()),
        "memory_action_parameter_condition": float(raw_condition),
        "memory_action_frame_gram_error": float(gram_error),
        "memory_action_frame_singular_min": float(effective_min),
        "memory_action_frame_singular_max": float(effective_singular.max()),
        "memory_action_frame_condition": float(effective_condition),
        "memory_action_direction_gram_error": float(direction_gram_error),
        "memory_action_direction_singular_min": float(direction_min),
        "memory_action_direction_singular_max": float(direction_singular.max()),
        "memory_action_direction_condition": float(direction_condition),
        "memory_action_frame_constrained": float(not unconstrained),
        "memory_state_kick_norm": float(
            state["world.mem_kdiov11.w_q"].float().norm()),
        "memory_autonomous_kick_norm": float(
            state["world.mem_kdiov11.b_f"].float().tanh().norm()),
    }
    for key, measured in norm_receipts.items():
        # Final horizons are computed on CUDA while checkpoint reconstruction is on CPU;
        # tolerate only the expected FP32 backend difference in small singular values.
        if not math.isclose(float(metrics[key]), measured, rel_tol=1e-3, abs_tol=1e-6):
            raise RunnerError(f"{job.run_name}: checkpoint/mechanism norm mismatch {key}")
    if float(metrics["memory_action_scale"]) <= 0.0:
        raise RunnerError(f"{job.run_name}: action scale is not positive")
    if fixedscale and float(metrics["memory_action_scale"]) != 1.0:
        raise RunnerError(f"{job.run_name}: fixedscale gamma is not exactly one")
    if not unconstrained:
        stiefel_receipts = {
            "memory_action_frame_gram_error": 0.0,
            "memory_action_frame_singular_min": float(action_scale),
            "memory_action_frame_singular_max": float(action_scale),
            "memory_action_frame_condition": 1.0,
            "memory_action_direction_gram_error": 0.0,
            "memory_action_direction_singular_min": 1.0,
            "memory_action_direction_singular_max": 1.0,
            "memory_action_direction_condition": 1.0,
        }
        for key, wanted in stiefel_receipts.items():
            if not math.isclose(float(metrics[key]), wanted,
                                rel_tol=1e-5, abs_tol=1e-5):
                raise RunnerError(f"{job.run_name}: constrained action-frame receipt {key} failed")
    if not math.isclose(float(metrics["memory_action_direction_norm"]),
                        math.sqrt(action_dim), rel_tol=1e-5, abs_tol=1e-6):
        raise RunnerError(f"{job.run_name}: action direction is not Frobenius normalized")
    rms_singular = float(metrics["memory_action_effective_norm"]) / math.sqrt(action_dim)
    if not math.isclose(rms_singular, float(metrics["memory_action_scale"]),
                        rel_tol=1e-5, abs_tol=1e-6):
        raise RunnerError(f"{job.run_name}: normalized action RMS singular is not gamma")
    if float(metrics["memory_action_parameter_norm"]) <= 0.0 \
            or float(metrics["memory_action_kick_norm"]) <= 0.0:
        raise RunnerError(f"{job.run_name}: KDIO lost its nonzero raw/effective action frame")
    if noautonomy and (float(metrics["memory_state_kick_norm"]) != 0.0
                       or float(metrics["memory_autonomous_kick_norm"]) != 0.0):
        raise RunnerError(f"{job.run_name}: noautonomy control learned autonomous force")

    dimension = COMMON["embed_dim"]
    gradient_parameters = dimension * dimension + action_dim * dimension + 5 * dimension + 4
    fitted_scalars = dimension * (dimension - 1) // 2 + dimension - 1
    total_scalars = gradient_parameters + fitted_scalars
    count_receipts = {
        "memory_gradient_trained_parameters": float(gradient_parameters),
        "memory_nominal_optimizer_scalars": float(gradient_parameters),
        "memory_fitted_memory_scalars": float(fitted_scalars),
        "memory_total_memory_scalars": float(total_scalars),
    }
    if any(float(metrics[key]) != wanted for key, wanted in count_receipts.items()):
        raise RunnerError(f"{job.run_name}: KDIO parameter-count receipt mismatch")

    shrinkage = float(metrics["memory_calibration_oas_shrinkage"])
    covariance_condition = float(metrics["memory_calibration_covariance_condition"])
    precision_condition = float(metrics["memory_innovation_precision_condition"])
    singular_min = float(metrics["memory_innovation_precision_singular_min"])
    singular_max = float(metrics["memory_innovation_precision_singular_max"])
    if not 0.0 <= shrinkage <= 1.0:
        raise RunnerError(f"{job.run_name}: OAS shrinkage is outside [0,1]")
    if covariance_condition < 1.0 - 1e-5 or precision_condition < 1.0 - 1e-5 \
            or singular_min <= 0.0 or singular_max < singular_min:
        raise RunnerError(f"{job.run_name}: invalid calibration conditioning receipt")

    if job.design in {"kdiov11_nocalibration", "kdiov11_diagonal"}:
        if abs(float(metrics["memory_innovation_precision_offdiagonal_norm"])) > 1e-7:
            raise RunnerError(f"{job.run_name}: diagonal calibration has off-diagonal precision")
    if job.design == "kdiov11_nocalibration":
        identity_receipts = {
            "memory_innovation_precision_diagonal_mean": 1.0,
            "memory_innovation_precision_logdet_per_dim": 0.0,
            "memory_innovation_precision_singular_min": 1.0,
            "memory_innovation_precision_singular_max": 1.0,
            "memory_innovation_precision_condition": 1.0,
            "memory_clean_innovation_mean_norm": 0.0,
            "memory_calibration_oas_shrinkage": 0.0,
            "memory_calibration_covariance_condition": 1.0,
        }
        for key, wanted in identity_receipts.items():
            if not math.isclose(float(metrics[key]), wanted, rel_tol=0.0, abs_tol=1e-6):
                raise RunnerError(f"{job.run_name}: identity calibration receipt {key} mismatch")

    bounded_means = {
        "innovation_ratio", "reliability", "position_base_gain",
        "velocity_base_gain", "velocity_base_ratio", "q_gates", "v_gates",
        "action_tanh_derivative_mean", "action_tanh_saturation_proxy",
    }
    nonnegative_means = {
        "innovation_energy", "process_tolerance", "action_effect_norm"}
    for dataset in OBSERVER_DATASETS:
        violation = _finite_metric(
            metrics, f"{dataset}_observer_ordered_gain_violation_max", job)
        if not 0.0 <= violation <= 1e-6:
            raise RunnerError(f"{job.run_name}: ordered observer gain receipt failed")
        for key in OBSERVER_KEYS:
            for phase in OBSERVER_PHASES:
                mean_key = f"{dataset}_observer_{key}_{phase}_mean"
                std_key = f"{dataset}_observer_{key}_{phase}_std"
                observed_mean = float(metrics[mean_key])
                observed_std = float(metrics[std_key])
                if observed_std < 0.0:
                    raise RunnerError(f"{job.run_name}: negative observer std {std_key}")
                if key in bounded_means and not 0.0 <= observed_mean <= 1.0:
                    raise RunnerError(f"{job.run_name}: observer mean outside [0,1] {mean_key}")
                if key in nonnegative_means and observed_mean < 0.0:
                    raise RunnerError(f"{job.run_name}: negative observer mean {mean_key}")
                if noaction and key == "action_effect_norm" and (
                        abs(observed_mean) > 1e-7 or observed_std > 1e-7):
                    raise RunnerError(
                        f"{job.run_name}: noaction control has nonzero action effect")
                if job.design == "kdiov11_noreliability" and key == "reliability":
                    if not math.isclose(observed_mean, 1.0, rel_tol=0.0, abs_tol=1e-6) \
                            or observed_std > 1e-6:
                        raise RunnerError(
                            f"{job.run_name}: noreliability control did not keep r=1")
        for phase in OBSERVER_PHASES:
            derivative = float(metrics[
                f"{dataset}_observer_action_tanh_derivative_mean_{phase}_mean"])
            saturation = float(metrics[
                f"{dataset}_observer_action_tanh_saturation_proxy_{phase}_mean"])
            if not math.isclose(derivative + saturation, 1.0,
                                rel_tol=0.0, abs_tol=1e-2):
                raise RunnerError(
                    f"{job.run_name}: inconsistent tanh observer diagnostics")

    for key in (
            "kdio_action_effect_rms", "kdio_true_action_one_step_mse",
            "kdio_shuffled_action_one_step_mse", "kdio_true_action_suffix_mse",
            "kdio_shuffled_action_suffix_mse", "kdio_action_rollout_divergence_h1",
            "kdio_action_rollout_divergence_h4",
            "kdio_action_rollout_divergence_h8",
            "kdio_action_rollout_divergence_h16",
            "kdio_action_rollout_divergence_h47"):
        if float(metrics[key]) < 0.0:
            raise RunnerError(f"{job.run_name}: negative action diagnostic {key}")
    if not 0.0 <= float(metrics["kdio_action_swap_pair_accuracy"]) <= 1.0:
        raise RunnerError(f"{job.run_name}: action-swap pair accuracy is outside [0,1]")
    for true_key, shuffled_key, advantage_key in (
            ("kdio_true_action_one_step_mse", "kdio_shuffled_action_one_step_mse",
             "kdio_true_action_one_step_advantage"),
            ("kdio_true_action_suffix_mse", "kdio_shuffled_action_suffix_mse",
             "kdio_true_action_suffix_advantage")):
        shuffled = float(metrics[shuffled_key])
        expected_advantage = (
            shuffled - float(metrics[true_key])) / max(
                abs(shuffled), float(np.finfo(np.float32).eps))
        if not math.isclose(float(metrics[advantage_key]), expected_advantage,
                            rel_tol=1e-5, abs_tol=1e-6):
            raise RunnerError(
                f"{job.run_name}: inconsistent action advantage {advantage_key}")
    if not math.isclose(
            float(metrics["kdio_true_action_advantage_h1"]),
            float(metrics["kdio_true_action_one_step_advantage"]),
            rel_tol=1e-6, abs_tol=1e-7):
        raise RunnerError(f"{job.run_name}: h1 action advantage mismatch")
    if noaction:
        noaction_zero = (
            "kdio_action_effect_rms", "kdio_true_action_one_step_advantage",
            "kdio_true_action_suffix_advantage",
            "kdio_true_action_advantage_h1", "kdio_true_action_advantage_h4",
            "kdio_true_action_advantage_h8", "kdio_true_action_advantage_h16",
            "kdio_true_action_advantage_h47",
            "kdio_action_rollout_divergence_h1",
            "kdio_action_rollout_divergence_h4",
            "kdio_action_rollout_divergence_h8",
            "kdio_action_rollout_divergence_h16",
            "kdio_action_rollout_divergence_h47",
        )
        if any(abs(float(metrics[key])) > 1e-7 for key in noaction_zero):
            raise RunnerError(f"{job.run_name}: noaction counterfactual receipt is nonzero")
        if not math.isclose(float(metrics["kdio_action_swap_pair_accuracy"]), 0.5,
                            rel_tol=0.0, abs_tol=1e-7):
            raise RunnerError(f"{job.run_name}: noaction ASR pair accuracy is not chance")
        for true_key, shuffled_key in (
                ("kdio_true_action_one_step_mse", "kdio_shuffled_action_one_step_mse"),
                ("kdio_true_action_suffix_mse", "kdio_shuffled_action_suffix_mse")):
            if not math.isclose(float(metrics[true_key]), float(metrics[shuffled_key]),
                                rel_tol=0.0, abs_tol=1e-7):
                raise RunnerError(
                    f"{job.run_name}: noaction true/shuffled diagnostic mismatch")


def validate_job(job, *, allow_missing):
    required = (job.model_path, job.metrics_path, job.eval_rollout_path, job.wandb_run_path)
    present = [path.is_file() and path.stat().st_size > 0 for path in required]
    if any(present) and not all(present):
        raise RunnerError(f"partial V11 run: {job.run_dir}")
    if not all(present):
        if job.run_dir.exists():
            raise RunnerError(f"incomplete V11 run directory: {job.run_dir}")
        if allow_missing:
            return False
        raise RunnerError(f"missing V11 run: {job.run_dir}")
    metrics = shared.read_json(job.metrics_path)
    shared.assert_finite_tree(metrics, f"{job.run_name}.metrics")
    import torch
    checkpoint = torch.load(job.model_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
            "model_state_dict", "args", "final_metrics", "history", "state_probes",
            "inverse_action_probe", "action_history_probe"}:
        raise RunnerError(f"{job.run_name}: checkpoint structure mismatch")
    if not shared.stable_equal(metrics, checkpoint["final_metrics"]) \
            or not shared.stable_equal(checkpoint["args"], expected_args(job)):
        raise RunnerError(f"{job.run_name}: checkpoint metadata mismatch")
    train_path, val_path, _ = data_paths(job.clean_env)
    train_meta, val_meta = load_cache(train_path), load_cache(val_path)
    target_dim = train_meta.task_observation_dim
    if (target_dim != val_meta.task_observation_dim
            or train_meta.task_observation_keys != val_meta.task_observation_keys
            or train_meta.task_observation_shapes != val_meta.task_observation_shapes):
        raise RunnerError(f"{job.run_name}: task-observation schema mismatch")
    probes = checkpoint["state_probes"]
    if not isinstance(probes, dict) or set(probes) != {
            "prior", "posterior", "encoder", "predictor"}:
        raise RunnerError(f"{job.run_name}: state probe set mismatch")
    for label, probe in probes.items():
        _validate_probe(probe, job, label, COMMON["embed_dim"], target_dim)
    _validate_probe(
        checkpoint["inverse_action_probe"], job, "inverse_action",
        3 * COMMON["embed_dim"], train_meta.action_dim)
    state = checkpoint["model_state_dict"]
    if not isinstance(state, dict) or not state or not any(
            name.startswith("world.encoder.") for name in state):
        raise RunnerError(f"{job.run_name}: wrapped encoder state absent")
    if any(name.startswith("inverse_head.") for name in state):
        raise RunnerError(f"{job.run_name}: training-time inverse head is present")
    memory_prefix = {"ssm": "world.mem_ssm.", "hacssmv8": "world.mem_hacssmv8.",
                     "orbitv10": "world.mem_orbitv10."}.get(
                         job.design, "world.mem_kdiov11.")
    if not any(name.startswith(memory_prefix) for name in state):
        raise RunnerError(f"{job.run_name}: expected memory namespace absent")
    for name, tensor in state.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise RunnerError(f"{job.run_name}: nonfinite model tensor {name}")
    validate_history(checkpoint["history"], job)
    required_metrics = (
        "heldout_prior_state_nmse", "clean_prior_state_nmse",
        "heldout_posterior_state_nmse", "posterior_probe_ceiling_state_nmse",
        "encoder_probe_ceiling_state_nmse", "predictor_probe_ceiling_state_nmse",
        "inverse_action_nmse", "inverse_action_r2", "inverse_action_probe_samples",
        "inverse_action_probe_input_dim", "inverse_action_probe_output_dim",
        "action_history_probe_nmse",
        "action_only_integrator_probe_nmse", "initial_encoder_integrator_probe_nmse",
        "initial_encoder_integrator_probe_r2", "final_val_loss", "val_predictive_loss",
        "predictive_loss_convergence_relative_change",
        "action_swap_loss_convergence_relative_change",
        "calibration_nll_convergence_relative_change", "loss_convergence_relative_change",
        "encoder_mean_channel_variance", "encoder_covariance_effective_rank",
        "encoder_singleton_max_abs", "encoder_prefix_max_abs",
        "orbit_orthogonality_error_max", "orbit_streaming_max_abs",
        "kdio_inverse_error_max", "kdio_volume_error_max", "kdio_streaming_max_abs",
        "mean_epoch_seconds", "peak_vram_bytes",
        *(f"{condition}_prior_state_nmse" for condition in HELDOUT_CORRUPTIONS),
        *(f"action_only_integrator_{condition}_nmse" for condition in HELDOUT_CORRUPTIONS),
        *(f"initial_encoder_integrator_{condition}_nmse"
          for condition in HELDOUT_CORRUPTIONS),
        *KDIO_ACTION_DIAGNOSTIC_METRICS,
    )
    for key in required_metrics:
        _finite_metric(metrics, key, job)
    if job.design in KDIO_DESIGNS:
        _validate_kdio_calibration_receipts(
            job, metrics, state, train_meta.action_dim)
    observed_mean_seconds = mean(float(row["epoch_seconds"])
                                 for row in checkpoint["history"])
    if not math.isclose(metrics["mean_epoch_seconds"], observed_mean_seconds,
                        rel_tol=1e-6, abs_tol=1e-8):
        raise RunnerError(f"{job.run_name}: mean epoch timing mismatch")
    if metrics["peak_vram_bytes"] <= 0:
        raise RunnerError(f"{job.run_name}: missing CUDA peak-memory receipt")
    exact = {
        "schema_version": 2, "env": job.clean_env, "design": job.design,
        "seed": job.seed, "epochs": COMMON["epochs"],
        "training_objective": OBJECTIVE, "headline_metric": "heldout_prior_state_nmse",
        "eval_target_key": "task_observation", "eval_target_dim": target_dim,
        "probe_ridge": COMMON["state_probe_ridge"],
        "train_data_sha256": train_meta.file_sha256, "val_data_sha256": val_meta.file_sha256,
        "train_data_content_sha256": train_meta.content_sha256,
        "val_data_content_sha256": val_meta.content_sha256,
        "train_episodes": COMMON["train_episodes"], "val_episodes": COMMON["val_episodes"],
        "length": COMMON["length"], "action_dim": train_meta.action_dim,
        "state_dim": train_meta.state_dim, "encoder_norm": "causal",
        "predictor_norm": "none", "one_token_predictor": True,
        "prediction_loss_weight": 1.0, "inverse_loss_weight": 0.0,
        "action_swap_loss_weight": float(
            job.design in ACTION_SWAP_DESIGNS
            and job.design != "kdiov11_noactionswap"),
        "variance_loss_weight": 1.0, "covariance_loss_weight": 1.0,
        "inverse_gradient_active": False,
        "innovation_calibration_method": (
            _design_metadata(job.design)["innovation_calibration"]),
        "innovation_calibration_gradient_active": False,
        **_design_metadata(job.design),
    }
    for key, wanted in exact.items():
        if not shared.stable_equal(metrics.get(key), wanted):
            raise RunnerError(f"{job.run_name}: exact metric {key} mismatch")
    inverse_probe_exact = {
        "inverse_action_probe_samples": float(
            COMMON["val_episodes"] * (COMMON["length"] - 2)),
        "inverse_action_probe_input_dim": float(3 * COMMON["embed_dim"]),
        "inverse_action_probe_output_dim": float(train_meta.action_dim),
    }
    if any(float(metrics[key]) != wanted
           for key, wanted in inverse_probe_exact.items()):
        raise RunnerError(f"{job.run_name}: evaluation-only inverse ridge receipt mismatch")
    condition_mean = mean(float(metrics[f"{condition}_prior_state_nmse"])
                          for condition in HELDOUT_CORRUPTIONS)
    if not math.isclose(metrics["heldout_prior_state_nmse"], condition_mean,
                        rel_tol=1e-6, abs_tol=1e-8):
        raise RunnerError(f"{job.run_name}: headline condition mean mismatch")
    if not 0 <= metrics["encoder_singleton_max_abs"] <= 1e-5 \
            or not 0 <= metrics["encoder_prefix_max_abs"] <= 1e-5:
        raise RunnerError(f"{job.run_name}: encoder causality receipt failed")
    validate_rollout(job, metrics, target_dim)
    validate_tracking_receipt(job, metrics)
    return True


def expected_wandb_artifact_metadata(job, rollout_sha256):
    return {
        "schema_version": ROLLOUT_SCHEMA_VERSION, "study": WANDB_STUDY,
        "env": job.clean_env, "design": job.design, "seed": job.seed,
        "episode": EVAL_ROLLOUT_EPISODE, "sha256": rollout_sha256,
        "semantics": "heldout pre-observation-prior normalized task-observation trace",
        **_design_metadata(job.design),
    }


def verify_wandb_cloud(jobs):
    import wandb
    expected = {}
    for job in jobs:
        receipt = shared.read_json(job.wandb_run_path)
        run_id = receipt["run_id"]
        if run_id in expected:
            raise RunnerError(f"duplicate W&B run id {run_id!r}")
        expected[run_id] = (job, receipt)

    def inspect(run_id):
        job, receipt = expected[run_id]
        try:
            run = wandb.Api(timeout=45).run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
            if run.state != "finished" or run.name != f"{WANDB_STUDY}-{job.run_name}":
                return f"{run_id}: state/name mismatch"
            history_keys = [
                "epoch", "train/action_swap_loss", "val/action_swap_loss",
                "train/action_swap_pair_accuracy", "val/action_swap_pair_accuracy",
            ]
            if job.design in KDIO_DESIGNS:
                history_keys.extend((
                    "mem/action_scale", "mem/action_scale_gradient_active",
                    "mem/action_kick_norm", "mem/calibration_updates",
                    "mem/calibration_samples",
                ))
                if job.design in CALIBRATED_DESIGNS:
                    history_keys.extend((
                        "cal/samples", "cal/oas_shrinkage",
                        "cal/covariance_condition", "cal/diagonal_only",
                    ))
            history = list(run.scan_history(keys=history_keys))
            epochs = {int(row["epoch"]) for row in history
                      if row.get("epoch") is not None}
            if epochs != set(range(1, COMMON["epochs"] + 1)):
                return f"{run_id}: epoch history mismatch"
            asr_epochs = {
                int(row["epoch"]) for row in history
                if row.get("train/action_swap_loss") is not None
                and row.get("val/action_swap_loss") is not None
                and row.get("train/action_swap_pair_accuracy") is not None
                and row.get("val/action_swap_pair_accuracy") is not None
            }
            if asr_epochs != epochs:
                return f"{run_id}: ASR epoch histories absent"
            if job.design in KDIO_DESIGNS:
                mem_epochs = {
                    int(row["epoch"]) for row in history
                    if row.get("mem/action_scale") is not None
                    and row.get("mem/action_scale_gradient_active") is not None
                    and row.get("mem/action_kick_norm") is not None
                    and row.get("mem/calibration_updates") is not None
                    and row.get("mem/calibration_samples") is not None
                }
                if mem_epochs != epochs:
                    return f"{run_id}: KDIO memory epoch receipts absent"
                calibration_epochs = {
                    int(row["epoch"]) for row in history
                    if row.get("cal/samples") is not None
                    and row.get("cal/oas_shrinkage") is not None
                    and row.get("cal/covariance_condition") is not None
                    and row.get("cal/diagonal_only") is not None
                }
                wanted_calibration_epochs = epochs if job.design in CALIBRATED_DESIGNS else set()
                if calibration_epochs != wanted_calibration_epochs:
                    return f"{run_id}: OAS calibration epoch receipts mismatch"
            summary = dict(run.summary)
            if job.design in KDIO_DESIGNS:
                for key in (*CALIBRATION_MEMORY_METRICS, *KDIO_MECHANISM_METRICS,
                            *KDIO_ACTION_DIAGNOSTIC_METRICS,
                            *observer_metric_names()):
                    value = summary.get(key)
                    if type(value) not in (int, float) or not math.isfinite(float(value)):
                        return f"{run_id}: W&B summary missing KDIO receipt {key}"
            table = dict(summary.get("eval/rollout_trace", {}))
            video = dict(summary.get("eval/paired_rollout", {}))
            if not table or not video:
                return f"{run_id}: rollout table/video absent"
            if (table.get("nrows"), table.get("ncols")) != (
                    len(HELDOUT_CORRUPTIONS) * (COMMON["length"] - COMMON["history_len"]), 7):
                return f"{run_id}: rollout table shape mismatch"
            if (video.get("height"), video.get("width")) != (
                    len(HELDOUT_CORRUPTIONS) * COMMON["img_size"],
                    2 * COMMON["img_size"] + 4):
                return f"{run_id}: rollout video shape mismatch"
            if not re.fullmatch(r"[0-9a-f]{64}", str(video.get("sha256", ""))):
                return f"{run_id}: rollout video hash absent"
            artifacts = [artifact for artifact in run.logged_artifacts()
                         if artifact.type == "evaluation-rollout"]
            if len(artifacts) != 1:
                return f"{run_id}: expected one rollout artifact"
            wanted = expected_wandb_artifact_metadata(job, receipt["eval_rollout_sha256"])
            if not shared.stable_equal(dict(artifacts[0].metadata), wanted):
                return f"{run_id}: artifact metadata mismatch"
            if set(artifacts[0].manifest.entries) != {"eval_rollout.npz"}:
                return f"{run_id}: rollout artifact manifest mismatch"
            return None
        except Exception as exc:
            return f"{run_id}: {exc}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        errors = [error for error in executor.map(inspect, expected) if error]
    if errors:
        raise RunnerError("W&B cloud verification failed: " + "; ".join(errors[:8]))
    return {"verified_runs": len(expected), "verified_epoch_histories": len(expected),
            "verified_asr_epoch_histories": len(expected),
            "verified_gamma_epoch_histories": sum(
                job.design in KDIO_DESIGNS for job, _receipt in expected.values()),
            "verified_rollout_artifacts": len(expected),
            "verified_rollout_tables": len(expected), "verified_rollout_videos": len(expected)}


def read_pilot_decision():
    decision = shared.read_json(PILOT_DECISION_PATH)
    passed = decision.get("pilot_screen_passed") if isinstance(decision, dict) else None
    if type(passed) is not bool or decision.get("scope") != SCOPE:
        raise RunnerError("invalid V11 pilot decision")
    return passed, decision


def write_final_manifest(protocol, pilot, pilot_passed, gpu_ids, workers, cloud, final):
    for job in ALL_JOBS:
        validate_job(job, allow_missing=False)
    manifest = {
        "schema_version": 1, "study": protocol["study"], "study_id": WANDB_STUDY,
        "scope": SCOPE, "producer_git_commit": protocol["producer_git_commit"],
        "producer_git_clean": True, "completed_runs": len(ALL_JOBS),
        "expected_runs": len(ALL_JOBS), "all_requested_runs_completed": True,
        "pilot_screen_passed": pilot_passed, "pilot_decision": pilot,
        "final_decision": final, "execution": {"gpu_ids": list(gpu_ids), "workers": workers},
        "protocol": {shared.rel(PROTOCOL_PATH): shared.file_record(PROTOCOL_PATH)},
        "data_artifacts": protocol["data_artifacts"],
        "source_artifacts": protocol["source_artifacts"], "wandb": protocol["wandb"],
        "wandb_cloud_verification": cloud,
        "wandb_runs": {job.run_name: shared.read_json(job.wandb_run_path) for job in ALL_JOBS},
        "output_artifacts": shared.output_file_snapshot(),
        "log_artifacts": shared.log_file_snapshot(),
    }
    shared.atomic_write_json(MANIFEST_PATH, manifest)
    digest = shared.sha256_file(MANIFEST_PATH)
    shared.atomic_write_bytes(MANIFEST_SHA_PATH, f"{digest}  {MANIFEST_PATH.name}\n".encode())


def configure_shared():
    assignments = {
        "TRAIN_SCRIPT": TRAIN_SCRIPT, "ANALYZE_SCRIPT": ANALYZE_SCRIPT,
        "FEATURE_ROOT": DATA_ROOT, "OUTPUT_ROOT": OUTPUT_ROOT, "LOG_ROOT": LOG_ROOT,
        "DATA_ROOT": DATA_ROOT, "PROTOCOL_PATH": PROTOCOL_PATH,
        "DECISION_PATH": PILOT_DECISION_PATH, "FINAL_DECISION_PATH": FINAL_DECISION_PATH,
        "MANIFEST_PATH": MANIFEST_PATH, "MANIFEST_SHA_PATH": MANIFEST_SHA_PATH,
        "LOCK_PATH": LOCK_PATH, "WANDB_ENTITY": WANDB_ENTITY,
        "WANDB_PROJECT": WANDB_PROJECT, "WANDB_MODE": WANDB_MODE,
        "WANDB_STUDY": WANDB_STUDY, "EVAL_ROLLOUT_EPISODE": EVAL_ROLLOUT_EPISODE,
        "ENVIRONMENTS": ENVIRONMENTS, "DESIGNS": DESIGNS,
        "PILOT_SEEDS": PILOT_SEEDS, "COMPLETION_SEEDS": COMPLETION_SEEDS,
        "ALL_SEEDS": ALL_SEEDS, "V5_DESIGNS": KDIO_DESIGNS,
        "HIER_DESIGNS": frozenset(), "NO_AUX_DESIGNS": frozenset(DESIGNS),
        "COMMON": COMMON, "SOURCE_FILES": SOURCE_FILES,
        "PILOT_ANALYSIS_FILES": PILOT_ANALYSIS_FILES,
        "FINAL_ANALYSIS_FILES": FINAL_ANALYSIS_FILES,
        "TOP_LEVEL_OUTPUT_FILES": TOP_LEVEL_OUTPUT_FILES,
        "PILOT_JOBS": PILOT_JOBS, "COMPLETION_JOBS": COMPLETION_JOBS,
        "ALL_JOBS": ALL_JOBS, "CLOUD_VERIFY_EPOCH_HISTORY": True,
    }
    for name, value in assignments.items():
        setattr(shared, name, value)
    shared.feature_snapshot = data_snapshot
    shared.eval_rollout_snapshot = data_snapshot
    shared.source_snapshot = source_snapshot
    shared.memory_contract = memory_contract
    shared.build_protocol = build_protocol
    shared.expected_args = expected_args
    shared.validate_history = validate_history
    shared.validate_job = validate_job
    shared.train_command = train_command
    shared.expected_wandb_artifact_metadata = expected_wandb_artifact_metadata
    shared.read_pilot_decision = read_pilot_decision


def verify_provenance_unchanged(protocol):
    if not shared.stable_equal(shared.read_json(PROTOCOL_PATH), protocol) \
            or not shared.stable_equal(source_snapshot(), protocol["source_artifacts"]) \
            or not shared.stable_equal(data_snapshot(), protocol["data_artifacts"]):
        raise RunnerError("V11 source/data/protocol changed during launch")
    commit, porcelain = shared.git_provenance()
    if commit != protocol["producer_git_commit"] or porcelain:
        raise RunnerError("V11 Git provenance changed during launch")


def check_command_interfaces(python):
    for script, required in (
        (TRAIN_SCRIPT, ("--train-data", "--eval-target-key", *DESIGNS)),
        (DATA_SCRIPT, ("--root", "--all", "--smooth-rho")),
        (ANALYZE_SCRIPT, ("--phase", "pilot", "final")),
    ):
        result = subprocess.run([python, str(script), "--help"], cwd=ROOT,
                                capture_output=True, text=True, check=False)
        output = result.stdout + result.stderr
        if result.returncode or any(token not in output for token in required):
            raise RunnerError(f"V11 command interface failed for {script}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--gpus", type=shared.parse_gpu_ids,
                        default=shared.parse_gpu_ids("0,1,2,3"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None):
    configure_shared()
    args = build_parser().parse_args(argv)
    if args.workers != 4 or len(args.gpus) != 4 or len(set(args.gpus)) != 4:
        raise RunnerError("V11 is frozen to four workers on four distinct GPU ids")
    commit, porcelain = shared.git_provenance()
    clean = not porcelain
    if not args.dry_run and not clean:
        raise RunnerError("launch requires a clean committed worktree")
    shared.check_python(args.python)
    check_command_interfaces(args.python)
    lock_stream = None
    if not args.dry_run:
        lock_stream = shared.acquire_lock()
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        precollect_data(args.python)
    try:
        data_snapshot()
        wandb_preflight = shared.check_wandb_online(args.python)
        protocol = build_protocol(commit, clean, wandb_preflight)
        shared.establish_protocol(protocol, args.dry_run)
        shared.reject_temporary_artifacts()
        completed = shared.validate_artifact_space(ALL_JOBS)
        shared.status(f"preflight validated {len(completed)}/400")
        if args.dry_run:
            digest = hashlib.sha256(json.dumps(
                protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
            shared.status(f"DRY RUN: no writes/launches; protocol digest={digest}")
            return 0
        shared.check_gpus(args.python, args.gpus)
        verify_provenance_unchanged(protocol)
        shared.run_stage(args.python, PILOT_JOBS, args.gpus, args.workers)
        for job in PILOT_JOBS:
            validate_job(job, allow_missing=False)
        shared.run_analyzer(args.python, "pilot")
        pilot_passed, pilot = read_pilot_decision()
        shared.run_stage(args.python, COMPLETION_JOBS, args.gpus, args.workers)
        for job in ALL_JOBS:
            validate_job(job, allow_missing=False)
        verify_provenance_unchanged(protocol)
        cloud = verify_wandb_cloud(ALL_JOBS)
        shared.run_analyzer(args.python, "final")
        final = shared.read_json(FINAL_DECISION_PATH)
        if final.get("completed_runs") != len(ALL_JOBS):
            raise RunnerError("invalid V11 final decision")
        write_final_manifest(
            protocol, pilot, pilot_passed, args.gpus, args.workers, cloud, final)
        shared.status(
            f"KDIO-v11 complete: 400/400; confirmation="
            f"{final.get('end_to_end_confirmation_passed')}")
        return 0
    finally:
        if lock_stream is not None:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        shared.terminate_active_processes()
        raise SystemExit(130)
    except RunnerError as exc:
        shared.terminate_active_processes()
        print(f"V11 RUNNER ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
