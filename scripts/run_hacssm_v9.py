#!/usr/bin/env python3
"""Run the locked 325-cell LOIF-v9 adaptive-development study.

All thirteen designs are trained from scratch under one visible-target next-latent
objective.  Seeds 0--2 form the immutable pilot; seeds 3--4 are a mandatory
completion even when that pilot is ``NO_GO``.  This runner reuses only the audited
V5 execution machinery and the frozen feature/rollout inputs--never an older model,
optimizer state, history, or checkpoint.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_hacssm_v5 as shared


REPO_ROOT = ROOT
TRAIN_SCRIPT = ROOT / "scripts" / "train_popgym.py"
ANALYZE_SCRIPT = ROOT / "scripts" / "analyze_hacssm_v9.py"
FEATURE_ROOT = ROOT / "outputs" / "smt_v3_shared" / "dino_features_d128"
OUTPUT_ROOT = ROOT / "outputs" / "hacssm_v9_shared"
LOG_ROOT = ROOT / "logs" / "hacssm_v9_shared"
DATA_ROOT = ROOT / "outputs" / "popgym_data"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PILOT_DECISION_PATH = OUTPUT_ROOT / "pilot_decision.json"
FINAL_DECISION_PATH = OUTPUT_ROOT / "decision.json"
MANIFEST_PATH = OUTPUT_ROOT / "hacssm_v9_manifest.json"
MANIFEST_SHA_PATH = OUTPUT_ROOT / "hacssm_v9_manifest.sha256"
LOCK_PATH = OUTPUT_ROOT / ".run_hacssm_v9.lock"

WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
WANDB_MODE = "online"
WANDB_STUDY = "hacssm-v9"
EVAL_ROLLOUT_EPISODE = 0

ENVIRONMENTS = shared.ENVIRONMENTS
V9_DESIGNS = (
    "loifv9",
    "loifv9_fixedalpha",
    "loifv9_globalR",
    "loifv9_innovationonly",
    "loifv9_latentonly",
    "loifv9_uniformfusion",
    "loifv9_noaction",
    "loifv9_singlebank",
)
REFERENCE_DESIGNS = (
    "ssm",
    "hacssmv7_sharedaction",
    "hacssmv8",
    "hacssmv8_dynamic",
    "hacssmv8_static",
)
DESIGNS = V9_DESIGNS + REFERENCE_DESIGNS
PILOT_SEEDS = (0, 1, 2)
COMPLETION_SEEDS = (3, 4)
ALL_SEEDS = PILOT_SEEDS + COMPLETION_SEEDS
V7_DESIGNS = frozenset({"hacssmv7_sharedaction"})
V8_DESIGNS = frozenset({"hacssmv8", "hacssmv8_dynamic", "hacssmv8_static"})
HIER_DESIGNS = V7_DESIGNS
NO_AUX_DESIGNS = frozenset(DESIGNS)

COMMON = {
    "train_episodes": 600,
    "val_episodes": 150,
    "length": 32,
    "feature_dim": 128,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "history_len": 3,
    "predictor_norm": "none",
    "first_post_loss_weight": 0.0,
    "epochs": 200,
    "train_dataloader_workers": 2,
    "prototype_seed": 0,
    "train_rollout_seed": 0,
    "val_rollout_seed": 7777,
    "smt_router": "sigmoid",
    "fixed_alpha": False,
    "wandb": True,
    "wandb_entity": WANDB_ENTITY,
    "wandb_project": WANDB_PROJECT,
    "wandb_mode": WANDB_MODE,
    "wandb_study": WANDB_STUDY,
    "eval_rollout_episode": EVAL_ROLLOUT_EPISODE,
}

SOURCE_FILES = (
    Path("scripts/run_hacssm_v9.py"),
    Path("scripts/analyze_hacssm_v9.py"),
    Path("scripts/hacssm_v9_diagnostics.py"),
    Path("scripts/run_hacssm_v5.py"),
    Path("scripts/analyze_hacssm_v5.py"),
    Path("scripts/train_popgym.py"),
    Path("lewm/__init__.py"),
    Path("lewm/data.py"),
    Path("lewm/envs/__init__.py"),
    Path("lewm/envs/memory_envs.py"),
    Path("lewm/envs/two_room.py"),
    Path("lewm/eval/__init__.py"),
    Path("lewm/eval/memory_probe.py"),
    Path("lewm/eval/probing.py"),
    Path("lewm/models/__init__.py"),
    Path("lewm/models/encoder.py"),
    Path("lewm/models/leworldmodel.py"),
    Path("lewm/models/memory.py"),
    Path("lewm/models/memory_model.py"),
    Path("lewm/models/sigreg.py"),
)

PILOT_ANALYSIS_FILES = frozenset({
    "pilot_per_run.csv",
    "pilot_grouped.csv",
    "pilot_paired_contrasts.csv",
    "pilot_phase_contrasts.csv",
    "pilot_intervention_contrasts.csv",
    "pilot_convergence.csv",
    "pilot_decision.json",
})
FINAL_ANALYSIS_FILES = frozenset({
    "per_run.csv",
    "grouped.csv",
    "paired_contrasts.csv",
    "phase_contrasts.csv",
    "intervention_contrasts.csv",
    "convergence.csv",
    "decision.json",
})
TOP_LEVEL_OUTPUT_FILES = frozenset({
    PROTOCOL_PATH.name,
    LOCK_PATH.name,
    MANIFEST_PATH.name,
    MANIFEST_SHA_PATH.name,
    *PILOT_ANALYSIS_FILES,
    *FINAL_ANALYSIS_FILES,
})

BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 8_008
BOOTSTRAP_CONTRACT = {
    "schema_version": 1,
    "algorithm": "crossed_environment_seed_percentile_bootstrap",
    "draws": BOOTSTRAP_DRAWS,
    "seed": BOOTSTRAP_SEED,
    "rng": "numpy.random.Generator(numpy.random.PCG64)",
    "resampling": (
        "independently sample E environment indices and S optimizer-seed indices "
        "with replacement; evaluate the E-by-S Cartesian product; equal-weight mean"
    ),
    "estimand": "mean paired relative reduction (reference-candidate)/reference",
    "quantiles": {"method": "linear", "reported": [0.05, 0.95]},
}
BOOTSTRAP_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        BOOTSTRAP_CONTRACT, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
).hexdigest()

V9_DIAGNOSTIC_PHASES = (
    "visible", "blackout_transition", "deep_blackout", "recovery"
)
V9_DIAGNOSTIC_STATS = (
    "log_R", "K_fast", "K_slow", "log_P_fast", "log_P_slow",
    "omega_fast", "omega_slow", "pi_fast", "pi_slow",
    "direct_fast", "direct_slow", "innovation_norm",
    "action_state_influence", "action_output_influence",
)
V9_INTERVENTION_PHASES = ("first_post", "deep_blackout", "all")


def make_jobs(stage: str, seeds: Sequence[int]) -> tuple[shared.Job, ...]:
    return tuple(
        shared.Job(stage, seed, occ, clean, design)
        for seed in seeds
        for occ, clean in ENVIRONMENTS
        for design in DESIGNS
    )


PILOT_JOBS = make_jobs("pilot", PILOT_SEEDS)
COMPLETION_JOBS = make_jobs("completion", COMPLETION_SEEDS)
ALL_JOBS = PILOT_JOBS + COMPLETION_JOBS
assert len(V9_DESIGNS) == 8 and len(DESIGNS) == 13
assert len(PILOT_JOBS) == 195 and len(COMPLETION_JOBS) == 130
assert len(ALL_JOBS) == 325 and len({job.run_name for job in ALL_JOBS}) == 325

RunnerError = shared.RunnerError
_BASE_VALIDATE_MODEL_STATE = shared.validate_model_state
_BASE_VALIDATE_JOB = shared.validate_job


def source_snapshot() -> dict[str, dict[str, Any]]:
    """Hash the loaded repository source closure, including empty package files."""
    records: dict[str, dict[str, Any]] = {}
    for source in SOURCE_FILES:
        path = REPO_ROOT / source
        if not path.is_file():
            raise RunnerError(f"required source file is missing: {path}")
        records[source.as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": shared.sha256_file(path),
        }
    return records


def design_aux_contract(design: str) -> tuple[float, str, bool]:
    if design not in DESIGNS:
        raise RunnerError(f"unknown V9-study design {design!r}")
    return 0.0, "fixed", False


def scheduled_weight(base: float, schedule: str, epoch: int) -> float:
    if epoch < 1 or schedule != "fixed" or float(base) != 0.0:
        raise RunnerError(f"invalid V9 auxiliary contract {base}/{schedule}/{epoch}")
    return 0.0


def objective_metadata(design: str) -> dict[str, Any]:
    from scripts.train_popgym import (
        counterfactual_recovery_metadata,
        learned_ordered_innovation_metadata,
        shared_action_shrinkage_metadata,
    )

    if design in V9_DESIGNS:
        return learned_ordered_innovation_metadata(design)
    if design in V8_DESIGNS:
        return shared_action_shrinkage_metadata(design)
    if design in V7_DESIGNS:
        return counterfactual_recovery_metadata(design)
    return {}


def memory_contract() -> dict[str, Any]:
    from lewm.models.memory import (
        HierarchicalCounterfactualRecoveryMemory,
        LearnedOrderedInnovationFilterMemory,
        SharedActionShrinkageMemory,
        SSMMemory,
    )

    dimension, action_dim = 128, 6
    mode_map = {
        "loifv9": "learned",
        "loifv9_fixedalpha": "fixedalpha",
        "loifv9_globalR": "globalR",
        "loifv9_innovationonly": "innovationonly",
        "loifv9_latentonly": "latentonly",
        "loifv9_uniformfusion": "uniformfusion",
        "loifv9_noaction": "noaction",
        "loifv9_singlebank": "singlebank",
    }
    models = {
        design: LearnedOrderedInnovationFilterMemory(
            dimension, action_dim, mode=mode
        )
        for design, mode in mode_map.items()
    }
    counts = {design: model.parameter_count() for design, model in models.items()}
    if set(counts.values()) != {34_563}:
        raise RunnerError(f"V9 controls are not parameter matched at 34,563: {counts}")
    signatures = {
        design: [
            [name, list(parameter.shape)]
            for name, parameter in model.named_parameters()
        ]
        for design, model in models.items()
    }
    if len({json.dumps(value) for value in signatures.values()}) != 1:
        raise RunnerError("V9 controls do not share one parameter signature")

    ssm = sum(parameter.numel() for parameter in SSMMemory(dimension).parameters())
    v7 = HierarchicalCounterfactualRecoveryMemory(
        dimension, action_dim, mode="sharedaction"
    ).parameter_count()
    v8 = SharedActionShrinkageMemory(
        dimension, action_dim, mode="learned"
    ).parameter_count()
    if (ssm, v7, v8) != (33_024, 36_102, 34_566):
        raise RunnerError(f"reference memory contract changed: {(ssm, v7, v8)}")
    return {
        "embed_dim": dimension,
        "action_dim": action_dim,
        "memory_parameters": {
            "loifv9_all_modes": 34_563,
            "ssm": ssm,
            "hacssmv7_sharedaction_student": v7,
            "hacssmv7_sharedaction_inert_teacher": v7,
            "hacssmv8_all_references": v8,
        },
        "streaming_recurrent_floats": {
            "loifv9": 2 * dimension + 2,
            "ssm": dimension,
            "hacssmv7_sharedaction": 2 * dimension,
            "hacssmv8": 2 * dimension,
        },
        "v9_counts_by_design": counts,
        "v9_parameter_signature": next(iter(signatures.values())),
        "parameter_matching_scope": "the eight LOIF modes are exactly parameter matched",
    }


def diagnostic_contract() -> dict[str, Any]:
    from scripts.hacssm_v9_diagnostics import (
        DIAGNOSTICS_SCHEMA_VERSION,
        DONOR_CONTRACT,
        DONOR_CONTRACT_SHA256,
        DONOR_SEED,
        LOG_SCALE_EXTREME_THRESHOLD,
        SATURATION_TOLERANCE,
        STREAMING_TOLERANCE,
    )

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "donor_contract": DONOR_CONTRACT,
        "donor_contract_sha256": DONOR_CONTRACT_SHA256,
        "donor_seed": DONOR_SEED,
        "saturation_tolerance": SATURATION_TOLERANCE,
        "log_scale_extreme_threshold": LOG_SCALE_EXTREME_THRESHOLD,
        "streaming_tolerance": STREAMING_TOLERANCE,
        "required_candidate_global_fields": [
            "alpha_fast", "alpha_slow", "q_fast", "q_slow"
        ],
        "required_candidate_phase_fields": [
            f"loif_{stat}_{phase}"
            for phase in V9_DIAGNOSTIC_PHASES
            for stat in V9_DIAGNOSTIC_STATS
        ],
        "required_candidate_intervention_fields": [
            f"clean_mse_{phase}_resistance_{kind}"
            for phase in V9_INTERVENTION_PHASES
            for kind in ("permuted", "mean")
        ],
        "required_candidate_receipt_fields": [
            "loif_pole_separation",
            "loif_fast_boundary_margin",
            "loif_slow_boundary_margin",
            "loif_pole_boundary_margin",
            "loif_pole_collapsed",
            "loif_boundary_saturated",
            "loif_gain_saturated_fraction",
            "loif_log_R_extreme_fraction",
            "loif_log_P_extreme_fraction",
            "loif_nonfinite_diagnostic_count",
            "loif_streaming_equivalent",
        ],
        "training_only_donors": True,
        "future_or_validation_donors": False,
    }


def build_protocol(
    commit: str, clean: bool, wandb_preflight: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": "HACSSM-v9 learned ordered innovation filter adaptive-development study",
        "adaptive_development_only": True,
        "producer_git_commit": commit,
        "producer_git_clean": clean,
        "common_protocol": COMMON,
        "memory_contract": memory_contract(),
        "v9_architecture_contract": {
            "candidate": "loifv9",
            "training_signal": "ordinary visible-target next-latent MSE only",
            "internal_auxiliary": "none",
            "teacher": "none",
            "hidden_clean_blackout_targets_used": False,
            "first_post_loss_weight": 0.0,
            "hier_loss_weight": 0.0,
            "teacher_or_hidden_clean_training_target": False,
            "learned_ordered_poles": True,
            "fixed_memory_timescale": False,
            "retention_initialization": (
                "tau-mapped alpha=(exp(-1/2),exp(-1/8)); matched to fixedalpha "
                "at step zero, then unconstrained within the ordered stable domain"
            ),
            "fixed_timescale_initialization_prior_present": True,
            "parameter_initialization": {
                "W_x": "identity",
                "W_a": "zero",
                "W_o": "zero",
                "w_z": "zero",
                "w_e": "zero",
                "b_R": "zero",
            },
            "memory_specific_objective_weight": False,
            "reference_exception": (
                "retrained hacssmv7_sharedaction retains its inert EMA teacher and "
                "hierarchy diagnostics, but its effective auxiliary coefficient is exactly zero"
            ),
            "variants": {
                "loifv9": "learned poles, dynamic evidence, inverse-scale fusion",
                "loifv9_fixedalpha": "tau-mapped fixed-alpha control",
                "loifv9_globalR": "learned global resistance only",
                "loifv9_innovationonly": "innovation evidence branch only",
                "loifv9_latentonly": "latent evidence branch only",
                "loifv9_uniformfusion": "uniform prior and posterior fusion",
                "loifv9_noaction": "shared action innovation zeroed",
                "loifv9_singlebank": "slow-bank inference path only",
            },
        },
        "design_protocol": {
            design: {
                "first_post_loss_weight": 0.0,
                "hier_loss_weight": 0.0,
                "hier_loss_schedule": "fixed",
                "auxiliary_gradients_active": False,
                "trained_from_scratch": True,
                **objective_metadata(design),
            }
            for design in DESIGNS
        },
        "fresh_training_contract": {
            "all_cells_trained_from_scratch": True,
            "checkpoint_reuse_allowed": False,
            "optimizer_state_reuse_allowed": False,
            "history_reuse_allowed": False,
            "sealed_v7_v8_models_are_not_inputs": True,
            "objective_mismatched_checkpoint_reuse": False,
            "reused_inputs_only": ["fixed feature caches", "fixed rollout pixel caches"],
            "command_has_checkpoint_or_resume_input": False,
        },
        "diagnostics_contract": diagnostic_contract(),
        "output_root": shared.rel(OUTPUT_ROOT),
        "log_root": shared.rel(LOG_ROOT),
        "feature_root": shared.rel(FEATURE_ROOT),
        "feature_artifacts": shared.feature_snapshot(),
        "eval_rollout_artifacts": shared.eval_rollout_snapshot(),
        "source_artifacts": source_snapshot(),
        "wandb": wandb_preflight,
        "wandb_requirements": {
            "all_cells_online": True,
            "complete_epoch_history_per_cell": 200,
            "evaluation_rollout_npz_table_video_per_cell": True,
        },
        "environments": [
            {"occluded": occ, "clean_target": clean_env}
            for occ, clean_env in ENVIRONMENTS
        ],
        "stages": {
            "pilot": {
                "designs": list(DESIGNS), "seeds": list(PILOT_SEEDS), "runs": 195,
            },
            "completion": {
                "designs": list(DESIGNS),
                "seeds": list(COMPLETION_SEEDS),
                "runs": 130,
                "completed_total_runs": 325,
                "runs_regardless_of_pilot_screen": True,
            },
        },
        "analysis_gate": {
            "command": "scripts/analyze_hacssm_v9.py --phase pilot",
            "decision_file": shared.rel(PILOT_DECISION_PATH),
            "fail_closed_result": "NO_GO",
            "pilot_pass_result": "PILOT_OVERALL_BEST_PASS",
            "scope": "adaptive_development_only",
        },
        "endpoint_envelope": {
            "candidate": "loifv9",
            "references": ["hacssmv8_dynamic", "hacssmv8_static"],
            "cell_selection": "minimum reference MSE independently in each environment-seed cell",
            "environment_selection": "minimum of the two separate environment means",
        },
        "pilot_success_criteria": {
            "vs_ssm": ">=6% reduction, >=10/15 cells, >=4/5 environments",
            "vs_each_headline_reference": ">=0.5% reduction, >=9/15, >=3/5",
            "vs_each_adaptive_evidence_control": ">=0.25% reduction, >=9/15, >=3/5",
            "vs_better_v8_endpoint_envelope": ">0% reduction, >=9/15, >=3/5",
            "vs_uniform_fusion": ">0% reduction, >=8/15, >=3/5",
            "vs_each_structural_control": ">=3% reduction, >=11/15, >=3/5",
            "convergence": "absolute median <1%, p95 <3%, maximum <5%",
        },
        "final_success_criteria": {
            "requires_pilot_pass": True,
            "vs_ssm": ">=7% reduction, >=20/25 cells, >=4/5 environments",
            "vs_each_headline_reference": ">=1% reduction, >=15/25, >=3/5",
            "vs_each_adaptive_evidence_control": ">=0.5% reduction, >=14/25, >=3/5",
            "vs_better_v8_endpoint_envelope": ">=0.5% reduction, >=14/25, >=3/5",
            "vs_uniform_fusion": ">=0.5% reduction, >=14/25, >=3/5",
            "vs_each_structural_control": ">=3% reduction, >=17/25, >=3/5",
            "bootstrap": (
                "crossed environment x seed 90% lower bound >0 for both headline "
                "references, four adaptive-evidence controls, and endpoint envelope"
            ),
            "full_grid_environment_envelope": ">=3/5",
            "last_visible_hold": ">=4/5",
            "convergence": "absolute median <1%, p95 <3%, maximum <5%",
        },
        "phase_success_criteria": {
            "reference": "hacssmv7_sharedaction",
            "metrics": ["clean_mse_deep_blackout", "clean_mse_all"],
            "each_metric": ">-1% paired reduction and >=3/5 environment effects >-1%",
        },
        "intervention_success_criteria": {
            "candidate": "loifv9",
            "metric": "clean_mse_first_post",
            "pilot_each_intervention": ">=0.25% reduction, >=9/15, >=3/5",
            "final_each_intervention": ">=0.5% reduction, >=14/25, >=3/5",
            "interventions": ["resistance_permuted", "resistance_mean"],
        },
        "bootstrap_contract": BOOTSTRAP_CONTRACT,
        "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
        "expected_runs": {
            "pilot": [job.run_name for job in PILOT_JOBS],
            "completion": [job.run_name for job in COMPLETION_JOBS],
        },
    }


def expected_args(job: shared.Job) -> dict[str, Any]:
    train_path, val_path, manifest_path = shared.feature_paths(job.clean_env)
    return {
        "env_id": job.occ_env,
        "memory_mode": job.design,
        "smt_router": COMMON["smt_router"],
        "seed": job.seed,
        "output_dir": shared.rel(OUTPUT_ROOT),
        "num_episodes": COMMON["train_episodes"],
        "val_episodes": COMMON["val_episodes"],
        "data_dir": shared.rel(DATA_ROOT),
        "prototype_seed": COMMON["prototype_seed"],
        "target_env_id": job.clean_env,
        "mask_occluded_target_loss": True,
        "first_post_loss_weight": 0.0,
        "encoder_checkpoint": None,
        "encoder_stats": None,
        "freeze_encoder": False,
        "encoder_type": "precomputed",
        "train_feature_cache": shared.rel(train_path),
        "val_feature_cache": shared.rel(val_path),
        "feature_manifest": shared.rel(manifest_path),
        "length": COMMON["length"],
        "img_size": 64,
        "epochs": COMMON["epochs"],
        "batch_size": COMMON["batch_size"],
        "lr": COMMON["learning_rate"],
        "weight_decay": COMMON["weight_decay"],
        "num_workers": COMMON["train_dataloader_workers"],
        "no_amp": False,
        "patch_size": 8,
        "embed_dim": COMMON["feature_dim"],
        "encoder_layers": 6,
        "encoder_heads": 4,
        "predictor_layers": 4,
        "predictor_heads": 8,
        "predictor_norm": COMMON["predictor_norm"],
        "history_len": COMMON["history_len"],
        "dropout": 0.1,
        "sigreg_lambda": 0.1,
        "sigreg_projections": 512,
        "hier_loss_weight": 0.0,
        "hier_loss_schedule": "fixed",
        "tau_fast": 3.0,
        "tau_slow": 25.0,
        "fixed_alpha": False,
        "wandb": True,
        "wandb_project": WANDB_PROJECT,
        "wandb_entity": WANDB_ENTITY,
        "wandb_mode": WANDB_MODE,
        "wandb_study": WANDB_STUDY,
        "eval_rollout_cache": shared.rel(shared.eval_rollout_cache(job.clean_env)),
        "eval_rollout_episode": EVAL_ROLLOUT_EPISODE,
        "extra_tag": "",
        "device": "cuda",
        "feature_manifest_sha256": shared.sha256_file(manifest_path),
        "eval_rollout_cache_sha256": shared.sha256_file(
            shared.eval_rollout_cache(job.clean_env)
        ),
    }


def expected_metric_metadata(job: shared.Job) -> dict[str, Any]:
    _, _, manifest_path = shared.feature_paths(job.clean_env)
    return {
        "env": job.occ_env,
        "design": job.design,
        "n_actions": 6,
        "influence_schema_version": 2,
        "influence_kind": (
            "single_or_undifferentiated_total"
            if job.design == "ssm" else "per_level_and_total"
        ),
        "prototype_seed": COMMON["prototype_seed"],
        "dataset_schema_version": 3,
        "feature_schema_version": 1,
        "feature_manifest": shared.rel(manifest_path),
        "feature_manifest_sha256": shared.sha256_file(manifest_path),
        "target_env": job.clean_env,
        "masked_clean_blackout_loss": True,
        "first_post_loss_weight": 0.0,
        "hier_loss_weight": 0.0,
        "hier_loss_schedule": "fixed",
        "hier_loss_weight_final": 0.0,
        "hier_loss_weight_base_effective": 0.0,
        "hier_loss_weight_effective": 0.0,
        "val_pred_loss_target_kind": "observed_pre_post_only",
        "deep_blackout_target_kind": "evaluation_only_hidden_clean",
        "primary_common_target_metric": "clean_mse_first_post",
        "encoder_frozen": False,
        "encoder_type": "precomputed",
        "predictor_norm": COMMON["predictor_norm"],
        "external_features_fixed": True,
        "encoder_checkpoint": None,
        "encoder_stats": None,
        "encoder_stats_sha256": None,
        "wandb_enabled": True,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_mode": WANDB_MODE,
        "wandb_study": WANDB_STUDY,
        "eval_rollout_cache": shared.rel(shared.eval_rollout_cache(job.clean_env)),
        "eval_rollout_cache_sha256": shared.sha256_file(
            shared.eval_rollout_cache(job.clean_env)
        ),
        "eval_rollout_episode": EVAL_ROLLOUT_EPISODE,
        **objective_metadata(job.design),
    }


def expected_wandb_artifact_metadata(
    job: shared.Job, rollout_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": WANDB_STUDY,
        "env": job.occ_env,
        "design": job.design,
        "seed": job.seed,
        "episode": EVAL_ROLLOUT_EPISODE,
        "sha256": rollout_sha256,
        "semantics": "closed-loop-on-observations clean-next-latent evaluation trace",
        **objective_metadata(job.design),
    }


def validate_history(history: Any, job: shared.Job) -> None:
    if not isinstance(history, list) or len(history) != COMMON["epochs"]:
        raise RunnerError(f"{job.run_name}: expected 200 history records")
    for epoch, record in enumerate(history, 1):
        if not isinstance(record, dict) or record.get("epoch") != epoch:
            raise RunnerError(f"{job.run_name}: malformed epoch {epoch}")
        if set(record) != {"epoch", "train", "val"}:
            raise RunnerError(f"{job.run_name}: unexpected history fields at epoch {epoch}")
        for split in ("train", "val"):
            values = record.get(split)
            if not isinstance(values, dict):
                raise RunnerError(f"{job.run_name}: missing {split} epoch {epoch}")
            for key in ("loss", "pred_loss", "sigreg_loss"):
                value = values.get(key)
                if type(value) not in (int, float) or not math.isfinite(float(value)):
                    raise RunnerError(f"{job.run_name}: invalid {split}.{key} at {epoch}")
            shared.assert_finite_tree(values, f"{job.run_name}.history[{epoch}].{split}")
            expected_total = float(values["pred_loss"]) + 0.1 * float(values["sigreg_loss"])
            if not math.isclose(
                float(values["loss"]), expected_total, rel_tol=2e-5, abs_tol=2e-6
            ):
                raise RunnerError(
                    f"{job.run_name}: loss contains a non-primary term at {epoch}/{split}"
                )
            hierarchical = sorted(key for key in values if key.startswith("hier_"))
            if job.design not in V7_DESIGNS and hierarchical:
                raise RunnerError(
                    f"{job.run_name}: no-aux design logged hierarchy fields {hierarchical}"
                )
            if job.design in V7_DESIGNS:
                required = {
                    "hier_loss", "hier_loss_fast", "hier_loss_medium",
                    "hier_loss_weight", "hier_loss_bridge", "hier_loss_recovery",
                    "hier_overlap",
                }
                if not required.issubset(values):
                    raise RunnerError(
                        f"{job.run_name}: V7 diagnostic fields missing at {epoch}/{split}"
                    )
                if float(values["hier_loss_weight"]) != 0.0:
                    raise RunnerError(f"{job.run_name}: V7 auxiliary is not disconnected")
                if float(values["hier_overlap"]) != 0.0:
                    raise RunnerError(f"{job.run_name}: V7 diagnostic overlaps hidden targets")


def validate_model_state(state: Any, job: shared.Job) -> None:
    _BASE_VALIDATE_MODEL_STATE(state, job)
    names = set(state) if isinstance(state, dict) else set()
    teacher_names = {name for name in names if "teacher" in name}
    if job.design in V9_DESIGNS:
        if not any(name.startswith("mem_loifv9.") for name in names):
            raise RunnerError(f"{job.run_name}: missing LOIF-v9 namespace")
        if teacher_names or any(
            name.startswith(("mem_hacssmv7.", "mem_hacssmv8.")) for name in names
        ):
            raise RunnerError(f"{job.run_name}: V9 checkpoint contains foreign memory tensors")
        required = {
            "mem_loifv9.u_fast", "mem_loifv9.u_delta", "mem_loifv9.b_R",
            "mem_loifv9.W_x.weight", "mem_loifv9.W_a.weight",
            "mem_loifv9.W_o.weight", "mem_loifv9.w_z", "mem_loifv9.w_e",
        }
        if not required.issubset(names):
            raise RunnerError(
                f"{job.run_name}: missing V9 tensors {sorted(required - names)}"
            )
    elif job.design in V8_DESIGNS:
        if not any(name.startswith("mem_hacssmv8.") for name in names) or teacher_names:
            raise RunnerError(f"{job.run_name}: invalid V8 reference checkpoint")
    elif job.design in V7_DESIGNS:
        if not any(name.startswith("mem_hacssmv7.") for name in names):
            raise RunnerError(f"{job.run_name}: missing V7 student namespace")
        if not any(name.startswith("mem_hacssmv7_teacher.") for name in names):
            raise RunnerError(f"{job.run_name}: missing inert V7 diagnostic teacher")
    elif teacher_names:
        raise RunnerError(f"{job.run_name}: unexpected teacher tensors")


def _finite_metric(metrics: Mapping[str, Any], key: str, job: shared.Job) -> float:
    value = metrics.get(key)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise RunnerError(f"{job.run_name}: required metric {key} is not finite")
    return float(value)


def validate_candidate_diagnostics(metrics: Mapping[str, Any], job: shared.Job) -> None:
    if job.design != "loifv9":
        return
    from scripts.hacssm_v9_diagnostics import (
        DIAGNOSTICS_SCHEMA_VERSION,
        DONOR_CONTRACT_SHA256,
        DONOR_SEED,
        LOG_SCALE_EXTREME_THRESHOLD,
        SATURATION_TOLERANCE,
        STREAMING_TOLERANCE,
    )

    exact = {
        "loif_diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "loif_donor_contract_sha256": DONOR_CONTRACT_SHA256,
        "loif_donor_seed": DONOR_SEED,
        "loif_donor_train_episodes": COMMON["train_episodes"],
        "loif_donor_val_episodes": COMMON["val_episodes"],
    }
    for key, wanted in exact.items():
        if metrics.get(key) != wanted:
            raise RunnerError(
                f"{job.run_name}: diagnostic {key}={metrics.get(key)!r}, expected {wanted!r}"
            )
    alpha_fast = _finite_metric(metrics, "alpha_fast", job)
    alpha_slow = _finite_metric(metrics, "alpha_slow", job)
    q_fast = _finite_metric(metrics, "q_fast", job)
    q_slow = _finite_metric(metrics, "q_slow", job)
    if not 0.0 <= alpha_fast <= alpha_slow < 1.0:
        raise RunnerError(f"{job.run_name}: learned poles are outside the ordered domain")
    for alpha, q in ((alpha_fast, q_fast), (alpha_slow, q_slow)):
        if not math.isclose(q, (1.0 - alpha) * (1.0 + alpha), rel_tol=1e-6, abs_tol=1e-7):
            raise RunnerError(f"{job.run_name}: q is inconsistent with its pole")
    for phase in V9_DIAGNOSTIC_PHASES:
        for stat in V9_DIAGNOSTIC_STATS:
            _finite_metric(metrics, f"loif_{stat}_{phase}", job)
        correlation = _finite_metric(
            metrics, f"loif_innovation_log_R_corr_{phase}", job
        )
        if not -1.0 <= correlation <= 1.0:
            raise RunnerError(f"{job.run_name}: invalid innovation/log-R correlation")
        constant = metrics.get(f"loif_innovation_or_log_R_constant_{phase}")
        if type(constant) is not bool:
            raise RunnerError(f"{job.run_name}: invalid constant-signal receipt")

    separation = _finite_metric(metrics, "loif_pole_separation", job)
    fast_margin = _finite_metric(metrics, "loif_fast_boundary_margin", job)
    slow_margin = _finite_metric(metrics, "loif_slow_boundary_margin", job)
    boundary_margin = _finite_metric(metrics, "loif_pole_boundary_margin", job)
    if not math.isclose(separation, alpha_slow - alpha_fast, abs_tol=1e-8):
        raise RunnerError(f"{job.run_name}: pole-separation receipt is inconsistent")
    if not math.isclose(fast_margin, alpha_fast, abs_tol=1e-8):
        raise RunnerError(f"{job.run_name}: fast boundary margin is inconsistent")
    if not math.isclose(slow_margin, 1.0 - alpha_slow, abs_tol=1e-8):
        raise RunnerError(f"{job.run_name}: slow boundary margin is inconsistent")
    if not math.isclose(boundary_margin, min(fast_margin, slow_margin), abs_tol=1e-8):
        raise RunnerError(f"{job.run_name}: aggregate boundary margin is inconsistent")
    exact_receipts = {
        "loif_saturation_tolerance": SATURATION_TOLERANCE,
        "loif_log_scale_extreme_threshold": LOG_SCALE_EXTREME_THRESHOLD,
        "loif_pole_collapsed": separation <= SATURATION_TOLERANCE,
        "loif_boundary_saturated": boundary_margin <= SATURATION_TOLERANCE,
        "loif_nonfinite_diagnostic_count": 0,
        "loif_streaming_batch_size": 1,
        "loif_streaming_tolerance": STREAMING_TOLERANCE,
    }
    for key, wanted in exact_receipts.items():
        if metrics.get(key) != wanted:
            raise RunnerError(
                f"{job.run_name}: receipt {key}={metrics.get(key)!r}, expected {wanted!r}"
            )
    for key in (
        "loif_gain_saturated_fraction", "loif_log_R_extreme_fraction",
        "loif_log_P_extreme_fraction",
    ):
        value = _finite_metric(metrics, key, job)
        if not 0.0 <= value <= 1.0:
            raise RunnerError(f"{job.run_name}: invalid saturation fraction {key}")
    for key in (
        "loif_streaming_mixed_max_abs", "loif_streaming_state_max_abs",
        "loif_streaming_log_P_max_abs",
    ):
        if _finite_metric(metrics, key, job) < 0.0:
            raise RunnerError(f"{job.run_name}: negative streaming error {key}")
    if type(metrics.get("loif_streaming_equivalent")) is not bool:
        raise RunnerError(f"{job.run_name}: invalid streaming-equivalence receipt")
    for phase in V9_INTERVENTION_PHASES:
        for kind in ("permuted", "mean"):
            value = _finite_metric(
                metrics, f"clean_mse_{phase}_resistance_{kind}", job
            )
            if value <= 0.0:
                raise RunnerError(f"{job.run_name}: intervention MSE must be positive")
    for key in ("loif_resistance_permuted_sha256", "loif_resistance_mean_sha256"):
        value = metrics.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise RunnerError(f"{job.run_name}: invalid intervention receipt {key}")


def validate_job(job: shared.Job, *, allow_missing: bool) -> bool:
    complete = _BASE_VALIDATE_JOB(job, allow_missing=allow_missing)
    if not complete:
        return False
    metrics = shared.read_json(job.metrics_path)
    if not isinstance(metrics, dict):
        raise RunnerError(f"{job.run_name}: metrics are not an object")
    validate_candidate_diagnostics(metrics, job)
    return True


def train_command(python: str, job: shared.Job) -> list[str]:
    train_path, val_path, manifest_path = shared.feature_paths(job.clean_env)
    return [
        python, shared.rel(TRAIN_SCRIPT),
        "--env-id", job.occ_env,
        "--target-env-id", job.clean_env,
        "--mask-occluded-target-loss",
        "--memory-mode", job.design,
        "--smt-router", COMMON["smt_router"],
        "--seed", str(job.seed),
        "--encoder-type", "precomputed",
        "--train-feature-cache", shared.rel(train_path),
        "--val-feature-cache", shared.rel(val_path),
        "--feature-manifest", shared.rel(manifest_path),
        "--prototype-seed", str(COMMON["prototype_seed"]),
        "--data-dir", shared.rel(DATA_ROOT),
        "--output-dir", shared.rel(OUTPUT_ROOT),
        "--num-episodes", str(COMMON["train_episodes"]),
        "--val-episodes", str(COMMON["val_episodes"]),
        "--length", str(COMMON["length"]),
        "--img-size", "64",
        "--epochs", str(COMMON["epochs"]),
        "--batch-size", str(COMMON["batch_size"]),
        "--lr", str(COMMON["learning_rate"]),
        "--weight-decay", str(COMMON["weight_decay"]),
        "--num-workers", str(COMMON["train_dataloader_workers"]),
        "--patch-size", "8",
        "--embed-dim", str(COMMON["feature_dim"]),
        "--encoder-layers", "6",
        "--encoder-heads", "4",
        "--predictor-layers", "4",
        "--predictor-heads", "8",
        "--predictor-norm", COMMON["predictor_norm"],
        "--history-len", str(COMMON["history_len"]),
        "--dropout", "0.1",
        "--sigreg-lambda", "0.1",
        "--sigreg-projections", "512",
        "--hier-loss-weight", "0.0",
        "--hier-loss-schedule", "fixed",
        "--tau-fast", "3.0",
        "--tau-slow", "25.0",
        "--first-post-loss-weight", "0.0",
        "--wandb",
        "--wandb-project", WANDB_PROJECT,
        "--wandb-entity", WANDB_ENTITY,
        "--wandb-mode", WANDB_MODE,
        "--wandb-study", WANDB_STUDY,
        "--eval-rollout-cache", shared.rel(shared.eval_rollout_cache(job.clean_env)),
        "--eval-rollout-episode", str(EVAL_ROLLOUT_EPISODE),
        "--device", "cuda",
    ]


def read_pilot_decision() -> tuple[bool, dict[str, Any]]:
    decision = shared.read_json(PILOT_DECISION_PATH)
    if not isinstance(decision, dict) or type(decision.get("pilot_screen_passed")) is not bool:
        raise RunnerError("pilot decision requires boolean pilot_screen_passed")
    expected = "PILOT_OVERALL_BEST_PASS" if decision["pilot_screen_passed"] else "NO_GO"
    if decision.get("decision") != expected:
        raise RunnerError(f"pilot decision label conflicts with its boolean: {decision}")
    if decision.get("scope") != "adaptive_development_only":
        raise RunnerError("pilot decision has invalid scope")
    shared.assert_finite_tree(decision, "pilot_decision")
    return decision["pilot_screen_passed"], decision


def validate_final_decision(final: Any, pilot_passed: bool) -> bool:
    if not isinstance(final, dict):
        raise RunnerError("final decision is not an object")
    best = final.get("best_in_locked_grid")
    if (
        final.get("decision") not in {
            "OVERALL_BEST_ADAPTIVE_DEV", "NO_GO", "PILOT_NO_GO_FINAL_DESCRIPTIVE"
        }
        or final.get("completed_runs") != 325
        or final.get("pilot_screen_passed") is not pilot_passed
        or final.get("scope") != "adaptive_development_only"
        or type(best) is not bool
    ):
        raise RunnerError("invalid final analyzer decision contract")
    if final["decision"] == "OVERALL_BEST_ADAPTIVE_DEV" and not (pilot_passed and best):
        raise RunnerError("overall-best label conflicts with pilot/final gates")
    if final["decision"] == "PILOT_NO_GO_FINAL_DESCRIPTIVE" and (pilot_passed or best):
        raise RunnerError("failed-pilot label conflicts with pilot/final gates")
    if final["decision"] == "NO_GO" and (not pilot_passed or best):
        raise RunnerError("NO_GO label conflicts with pilot/final gates")
    return best


def write_final_manifest(
    protocol: dict[str, Any],
    pilot_decision: dict[str, Any],
    pilot_screen_passed: bool,
    gpu_ids: Sequence[str],
    workers: int,
    cloud: dict[str, Any],
    final: dict[str, Any],
) -> None:
    shared.write_final_manifest(
        protocol, pilot_decision, pilot_screen_passed, gpu_ids, workers, cloud
    )
    manifest = shared.read_json(MANIFEST_PATH)
    if not isinstance(manifest, dict):
        raise RunnerError("fresh V9 manifest is not an object")
    manifest["final_decision"] = final
    manifest["fresh_training_contract"] = protocol["fresh_training_contract"]
    manifest["diagnostics_contract"] = protocol["diagnostics_contract"]
    shared.atomic_write_json(MANIFEST_PATH, manifest)
    digest = shared.sha256_file(MANIFEST_PATH)
    shared.atomic_write_bytes(
        MANIFEST_SHA_PATH, f"{digest}  {MANIFEST_PATH.name}\n".encode()
    )
    if shared.sha256_file(MANIFEST_PATH) != digest:
        raise RunnerError("V9 manifest changed after final-decision publication")


def configure_shared() -> None:
    assignments = {
        "TRAIN_SCRIPT": TRAIN_SCRIPT,
        "ANALYZE_SCRIPT": ANALYZE_SCRIPT,
        "FEATURE_ROOT": FEATURE_ROOT,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "LOG_ROOT": LOG_ROOT,
        "DATA_ROOT": DATA_ROOT,
        "PROTOCOL_PATH": PROTOCOL_PATH,
        "DECISION_PATH": PILOT_DECISION_PATH,
        "FINAL_DECISION_PATH": FINAL_DECISION_PATH,
        "MANIFEST_PATH": MANIFEST_PATH,
        "MANIFEST_SHA_PATH": MANIFEST_SHA_PATH,
        "LOCK_PATH": LOCK_PATH,
        "WANDB_ENTITY": WANDB_ENTITY,
        "WANDB_PROJECT": WANDB_PROJECT,
        "WANDB_MODE": WANDB_MODE,
        "WANDB_STUDY": WANDB_STUDY,
        "EVAL_ROLLOUT_EPISODE": EVAL_ROLLOUT_EPISODE,
        "ENVIRONMENTS": ENVIRONMENTS,
        "DESIGNS": DESIGNS,
        "PILOT_SEEDS": PILOT_SEEDS,
        "COMPLETION_SEEDS": COMPLETION_SEEDS,
        "ALL_SEEDS": ALL_SEEDS,
        "V5_DESIGNS": frozenset(V9_DESIGNS),
        "HIER_DESIGNS": HIER_DESIGNS,
        "NO_AUX_DESIGNS": NO_AUX_DESIGNS,
        "COMMON": COMMON,
        "SOURCE_FILES": SOURCE_FILES,
        "PILOT_ANALYSIS_FILES": PILOT_ANALYSIS_FILES,
        "FINAL_ANALYSIS_FILES": FINAL_ANALYSIS_FILES,
        "TOP_LEVEL_OUTPUT_FILES": TOP_LEVEL_OUTPUT_FILES,
        "PILOT_JOBS": PILOT_JOBS,
        "COMPLETION_JOBS": COMPLETION_JOBS,
        "ALL_JOBS": ALL_JOBS,
        "CLOUD_VERIFY_EPOCH_HISTORY": True,
    }
    for name, value in assignments.items():
        setattr(shared, name, value)
    shared.design_aux_contract = design_aux_contract
    shared.source_snapshot = source_snapshot
    shared.scheduled_weight = scheduled_weight
    shared.memory_contract = memory_contract
    shared.build_protocol = build_protocol
    shared.expected_args = expected_args
    shared.expected_metric_metadata = expected_metric_metadata
    shared.validate_history = validate_history
    shared.validate_model_state = validate_model_state
    shared.validate_job = validate_job
    shared.train_command = train_command
    shared.expected_wandb_artifact_metadata = expected_wandb_artifact_metadata
    shared.read_pilot_decision = read_pilot_decision


def check_command_interfaces(python: str) -> None:
    for script, required in (
        (TRAIN_SCRIPT, ("--hier-loss-schedule", *DESIGNS)),
        (ANALYZE_SCRIPT, ("--phase", "pilot", "final")),
    ):
        result = shared.subprocess.run(
            [python, str(script), "--help"], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RunnerError(f"cannot inspect {script}: {result.stderr.strip()}")
        text = result.stdout + result.stderr
        missing = [token for token in required if token not in text]
        if missing:
            raise RunnerError(f"{script}: command interface lacks {missing}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the locked 325-cell LOIF-v9 study.")
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument(
        "--gpus", type=shared.parse_gpu_ids,
        default=shared.parse_gpu_ids("0,1,2,3"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_shared()
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise RunnerError("--workers must be positive")
    commit, porcelain = shared.git_provenance()
    clean = not porcelain
    if not args.dry_run and not clean:
        raise RunnerError(
            "launch requires a committed clean worktree: "
            + " | ".join(porcelain.splitlines()[:8])
        )
    if args.dry_run and not clean:
        shared.status("DRY RUN NOTE: launch remains disabled until changes are committed")

    shared.check_python(args.python)
    check_command_interfaces(args.python)
    wandb_preflight = shared.check_wandb_online(args.python)
    protocol = build_protocol(commit, clean, wandb_preflight)

    lock_stream = None
    if not args.dry_run:
        lock_stream = shared.acquire_lock()
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        shared.establish_protocol(protocol, args.dry_run)
        shared.reject_temporary_artifacts()
        completed = shared.validate_artifact_space(ALL_JOBS)
        pilot_done = sum(job.run_name in completed for job in PILOT_JOBS)
        shared.status(
            f"preflight validated {len(completed)}/325 runs; pilot={pilot_done}/195"
        )
        if args.dry_run:
            digest = hashlib.sha256(
                json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            shared.status(f"DRY RUN: no writes or launches; protocol digest={digest}")
            return 0

        shared.check_gpus(args.python, args.gpus)
        shared.verify_provenance_unchanged(protocol)
        shared.run_stage(args.python, PILOT_JOBS, args.gpus, args.workers)
        for job in PILOT_JOBS:
            shared.validate_job(job, allow_missing=False)
        shared.verify_provenance_unchanged(protocol)
        shared.status("running immutable V9 pilot analyzer")
        shared.run_analyzer(args.python, "pilot")
        pilot_passed, pilot_decision = shared.read_pilot_decision()
        shared.status(
            f"V9 pilot decision={pilot_decision['decision']} passed={pilot_passed}; "
            "seeds 3-4 run regardless"
        )

        shared.run_stage(args.python, COMPLETION_JOBS, args.gpus, args.workers)
        for job in ALL_JOBS:
            shared.validate_job(job, allow_missing=False)
        shared.verify_provenance_unchanged(protocol)
        cloud = shared.verify_wandb_cloud(ALL_JOBS)
        shared.status("running final five-seed V9 analyzer")
        shared.run_analyzer(args.python, "final")
        final = shared.read_json(FINAL_DECISION_PATH)
        best = validate_final_decision(final, pilot_passed)
        shared.verify_provenance_unchanged(protocol)
        shared.validate_artifact_space(ALL_JOBS)
        write_final_manifest(
            protocol, pilot_decision, pilot_passed,
            args.gpus, args.workers, cloud, final,
        )
        shared.status(
            f"LOIF-v9 complete: 325/325 validated; best_in_locked_grid={best}"
        )
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
        print(f"LOIF-v9 runner error: {exc}", file=sys.stderr)
        raise SystemExit(2)
