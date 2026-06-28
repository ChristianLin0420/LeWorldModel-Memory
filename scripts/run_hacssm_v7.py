#!/usr/bin/env python3
"""Run the locked HACSSM/HCRD-v7 counterfactual-recovery study.

The audited V5 harness supplies execution, W&B rollout, hashing, resume, and
cloud-verification primitives.  This module replaces every study-specific
contract before using them.  The immutable pilot is 5 environments x 13
designs x seeds 0--2 (195 cells); seeds 3--4 always run, for 325 cells total,
even when the prospective pilot screen fails.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_hacssm_v5 as shared


REPO_ROOT = ROOT
TRAIN_SCRIPT = ROOT / "scripts" / "train_popgym.py"
ANALYZE_SCRIPT = ROOT / "scripts" / "analyze_hacssm_v7.py"
FEATURE_ROOT = ROOT / "outputs" / "smt_v3_shared" / "dino_features_d128"
OUTPUT_ROOT = ROOT / "outputs" / "hacssm_v7_shared"
LOG_ROOT = ROOT / "logs" / "hacssm_v7_shared"
DATA_ROOT = ROOT / "outputs" / "popgym_data"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PILOT_DECISION_PATH = OUTPUT_ROOT / "pilot_decision.json"
FINAL_DECISION_PATH = OUTPUT_ROOT / "decision.json"
MANIFEST_PATH = OUTPUT_ROOT / "hacssm_v7_manifest.json"
MANIFEST_SHA_PATH = OUTPUT_ROOT / "hacssm_v7_manifest.sha256"
LOCK_PATH = OUTPUT_ROOT / ".run_hacssm_v7.lock"

WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
WANDB_MODE = "online"
WANDB_STUDY = "hacssm-v7"
EVAL_ROLLOUT_EPISODE = 0

ENVIRONMENTS = shared.ENVIRONMENTS
DESIGNS = (
    "ssm",
    "hacsmv4_two_noaux",
    "hacssmv6",
    "hacssmv6_static",
    "hacssmv7_noaux",
    "hacssmv7_sharedaction",
    "hacssmv7_noshrink",
    "hacssmv7_actiononly",
    "hacssmv7_uniform",
    "hacssmv7_norecovery",
    "hacssmv7_noaction",
    "hacssmv7_single",
    "hacssmv7",
)
PILOT_SEEDS = (0, 1, 2)
COMPLETION_SEEDS = (3, 4)
ALL_SEEDS = PILOT_SEEDS + COMPLETION_SEEDS
V7_DESIGNS = frozenset(d for d in DESIGNS if d.startswith("hacssmv7"))
V6_DESIGNS = frozenset({"hacssmv6", "hacssmv6_static"})
HIER_DESIGNS = frozenset(
    d for d in DESIGNS if d.startswith(("hacsmv4", "hacssmv6", "hacssmv7"))
)
NO_AUX_DESIGNS = frozenset({"hacsmv4_two_noaux", "hacssmv7_noaux"})

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
    "first_post_loss_weight": 0.5,
    "epochs": 200,
    "train_dataloader_workers": 2,
    "prototype_seed": 0,
    "train_rollout_seed": 0,
    "val_rollout_seed": 7777,
    "smt_router": "sigmoid",
    "fixed_alpha": True,
    "wandb": True,
    "wandb_entity": WANDB_ENTITY,
    "wandb_project": WANDB_PROJECT,
    "wandb_mode": WANDB_MODE,
    "wandb_study": WANDB_STUDY,
    "eval_rollout_episode": EVAL_ROLLOUT_EPISODE,
}

SOURCE_FILES = (
    Path("scripts/run_hacssm_v7.py"),
    Path("scripts/run_hacssm_v5.py"),
    Path("scripts/analyze_hacssm_v7.py"),
    Path("scripts/analyze_hacssm_v5.py"),
    Path("scripts/train_popgym.py"),
    Path("lewm/data.py"),
    Path("lewm/models/encoder.py"),
    Path("lewm/models/leworldmodel.py"),
    Path("lewm/models/memory.py"),
    Path("lewm/models/memory_model.py"),
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
assert len(PILOT_JOBS) == 195
assert len(COMPLETION_JOBS) == 130
assert len(ALL_JOBS) == 325
assert len({job.run_name for job in ALL_JOBS}) == 325


RunnerError = shared.RunnerError


def design_aux_contract(design: str) -> tuple[float, str, bool]:
    if design in V7_DESIGNS:
        return 0.02, "v6_bootstrap", design != "hacssmv7_noaux"
    if design in V6_DESIGNS:
        return 0.02, "v6_bootstrap", True
    if design == "hacsmv4_two_noaux":
        return 0.1, "fixed", False
    if design == "ssm":
        return 0.0, "fixed", False
    raise RunnerError(f"no auxiliary contract for design {design!r}")


def objective_metadata(design: str) -> dict[str, Any]:
    from scripts.train_popgym import (
        counterfactual_recovery_metadata,
        hierarchical_objective_metadata,
    )

    if design in V7_DESIGNS:
        return counterfactual_recovery_metadata(design)
    if design in V6_DESIGNS:
        return hierarchical_objective_metadata(design)
    return {}


def scheduled_weight(base: float, schedule: str, epoch: int) -> float:
    if epoch < 1:
        raise RunnerError(f"epoch must be positive, got {epoch}")
    if schedule == "fixed":
        return float(base)
    if schedule == "v6_bootstrap":
        if epoch <= 40:
            return float(base)
        if epoch <= 100:
            return float(base) * 0.5 * (1.0 + math.cos(math.pi * (epoch - 40) / 60.0))
        return 0.0
    raise RunnerError(f"unknown hierarchy schedule {schedule!r}")


def memory_contract() -> dict[str, Any]:
    from lewm.models.memory import (
        HierarchicalActionConditionedMemory,
        HierarchicalCounterfactualRecoveryMemory,
        SSMMemory,
    )

    dimension, action_dim = 128, 6
    modes = (
        "dynamic", "noaux", "sharedaction", "noshrink", "actiononly",
        "uniform", "norecovery", "noaction", "single",
    )
    instances = [
        HierarchicalCounterfactualRecoveryMemory(dimension, action_dim, mode=mode)
        for mode in modes
    ]
    signatures = [
        [[name, list(parameter.shape)] for name, parameter in model.named_parameters()]
        for model in instances
    ]
    counts = [model.parameter_count() for model in instances]
    expected = HierarchicalCounterfactualRecoveryMemory.expected_parameter_count(
        dimension, action_dim)
    if expected != 36_102 or len(set(counts)) != 1 or counts[0] != expected:
        raise RunnerError(f"V7 modes are not parameter matched: expected={expected}, got={counts}")
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RunnerError("V7 mode parameter names/shapes differ")

    ssm = SSMMemory(dimension)
    ssm_count = sum(parameter.numel() for parameter in ssm.parameters())
    v4 = HierarchicalActionConditionedMemory(
        dimension, action_dim, mode="dynamic", taus=(2.0, 8.0))
    if ssm_count != 33_024 or v4.parameter_count() != 34_564:
        raise RunnerError("historical SSM/V4-two parameter contract changed")
    return {
        "embed_dim": dimension,
        "action_dim": action_dim,
        "memory_parameters": {
            "ssm": ssm_count,
            "hacsmv4_two_noaux": v4.parameter_count(),
            "hacssmv6_all_modes": v4.parameter_count(),
            "hacssmv7_all_modes": counts[0],
            "hacssmv7_ema_teacher_frozen_copy": counts[0],
        },
        "trainable_memory_parameters_include_teacher": False,
        "checkpoint_contains_frozen_ema_teacher": True,
        "streaming_recurrent_floats": {
            "ssm": dimension,
            "hacsmv4_two_noaux": 2 * dimension,
            "hacssmv6_all_modes": 2 * dimension,
            "hacssmv7_all_modes": 2 * dimension,
        },
        "v7_parameter_signature": signatures[0],
        "v7_parameter_matching": (
            "all V7 inference and objective ablations instantiate identical student and "
            "frozen-teacher schemas; only functional paths differ"
        ),
    }


def build_protocol(commit: str, clean: bool, wandb_preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": "HACSSM/HCRD-v7 visible-only counterfactual-recovery study",
        "producer_git_commit": commit,
        "producer_git_clean": clean,
        "common_protocol": COMMON,
        "memory_contract": memory_contract(),
        "self_supervision_contract": {
            "inference_anchor": (
                "fixed taus 2 and 8; level-specific action predict/correct; learned per-level "
                "convex shrinkage between exact static and dynamic gates; joint read"
            ),
            "teacher": (
                "frozen EMA memory copy (momentum .99) consuming only the original occluded "
                "trajectory; targets are detached same-level teacher posteriors"
            ),
            "counterfactual_spans": (
                "all deterministic windows [source, h synthetic black frames, restored frame] "
                "lying entirely inside originally visible runs"
            ),
            "canonical_black_token": (
                "mean observed latent over the original blackout input; no clean blackout latent"
            ),
            "source": "detached online student state at the visible source",
            "distance": "SmoothL1(LayerNorm(student), LayerNorm(EMA-teacher target))",
            "losses": "equal bridge/recovery mean within horizon, then equal fast/medium mean",
            "hierarchy": {"fast": [1, 2], "medium": [4, 8]},
            "schedule": ".02 epochs 1-40; cosine to zero at 100; zero epochs 101-200",
            "auxiliary_gradient_scope": (
                "student level-specific action heads, gate parameters, and shrinkage only; "
                "W_x outputs/source/teacher are detached and W_o/route/predictor are primary-only"
            ),
            "hidden_clean_blackout_targets_used": False,
            "required_original_hidden_overlap": 0,
            "variants": {
                "hacssmv7_noaux": "identical V7 inference and EMA plumbing, zero auxiliary weight",
                "hacssmv7_sharedaction": "average the level action heads before recurrence",
                "hacssmv7_noshrink": "fix rho=1 for the dynamic correction expert",
                "hacssmv7_actiononly": "V6-style detached action rollout objective only",
                "hacssmv7_uniform": "both levels receive horizons 1,2,4,8",
                "hacssmv7_norecovery": "counterfactual bridge objective without restored-frame loss",
                "hacssmv7_noaction": "zero action features in inference and auxiliary",
                "hacssmv7_single": "medium-only inference read",
                "hacssmv7": "level-specific action, learned shrinkage, bridge plus recovery",
            },
        },
        "design_protocol": {
            design: {
                "hier_loss_weight": design_aux_contract(design)[0],
                "hier_loss_schedule": design_aux_contract(design)[1],
                "auxiliary_gradients_active": design_aux_contract(design)[2],
                **objective_metadata(design),
            }
            for design in DESIGNS
        },
        "output_root": shared.rel(OUTPUT_ROOT),
        "log_root": shared.rel(LOG_ROOT),
        "feature_root": shared.rel(FEATURE_ROOT),
        "feature_artifacts": shared.feature_snapshot(),
        "eval_rollout_artifacts": shared.eval_rollout_snapshot(),
        "source_artifacts": shared.source_snapshot(),
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
            "pilot": {"designs": list(DESIGNS), "seeds": list(PILOT_SEEDS), "runs": 195},
            "completion": {
                "designs": list(DESIGNS), "seeds": list(COMPLETION_SEEDS), "runs": 130,
                "completed_total_runs": 325, "runs_regardless_of_pilot_screen": True,
            },
        },
        "analysis_gate": {
            "command": "scripts/analyze_hacssm_v7.py --phase pilot",
            "decision_file": shared.rel(PILOT_DECISION_PATH),
            "fail_closed_result": "NO_GO",
            "final_claim_field": "good_enough_for_overall_best_claim",
        },
        "final_success_criteria": {
            "vs_ssm": ">=6% reduction, >=20/25 paired wins, >=4/5 environment wins",
            "vs_v4_two": ">=1.5% reduction, >=17/25 paired wins, >=4/5 environment wins",
            "vs_v6": ">=1% reduction, >=15/25 paired wins, >=3/5 environment wins",
            "vs_v6_static": "positive reduction, >=13/25 paired wins, >=3/5 environment wins",
            "vs_v7_noaux": ">=1% reduction, >=15/25 paired wins, >=3/5 environment wins",
            "mechanism_controls": (
                "positive reduction, >=13/25 paired wins, >=3/5 environment wins against "
                "sharedaction/noshrink/actiononly/uniform/norecovery"
            ),
            "structural_controls": (
                ">=3% reduction, >=17/25 paired wins, >=3/5 environment wins against "
                "noaction/single"
            ),
            "envelope_and_hold": ">=3/5 locked-grid envelope wins and >=4/5 hold wins",
            "convergence": "absolute median <1%, p95 <3%, max <5%",
        },
        "expected_runs": {
            "pilot": [job.run_name for job in PILOT_JOBS],
            "completion": [job.run_name for job in COMPLETION_JOBS],
        },
    }


def expected_args(job: shared.Job) -> dict[str, Any]:
    train_path, val_path, manifest_path = shared.feature_paths(job.clean_env)
    base, schedule, _ = design_aux_contract(job.design)
    return {
        "env_id": job.occ_env,
        "memory_mode": job.design,
        "smt_router": COMMON["smt_router"],
        "seed": job.seed,
        "output_dir": shared.rel(OUTPUT_ROOT),
        "num_episodes": 600,
        "val_episodes": 150,
        "data_dir": shared.rel(DATA_ROOT),
        "prototype_seed": 0,
        "target_env_id": job.clean_env,
        "mask_occluded_target_loss": True,
        "first_post_loss_weight": 0.5,
        "encoder_checkpoint": None,
        "encoder_stats": None,
        "freeze_encoder": False,
        "encoder_type": "precomputed",
        "train_feature_cache": shared.rel(train_path),
        "val_feature_cache": shared.rel(val_path),
        "feature_manifest": shared.rel(manifest_path),
        "length": 32,
        "img_size": 64,
        "epochs": 200,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "num_workers": 2,
        "no_amp": False,
        "patch_size": 8,
        "embed_dim": 128,
        "encoder_layers": 6,
        "encoder_heads": 4,
        "predictor_layers": 4,
        "predictor_heads": 8,
        "predictor_norm": "none",
        "history_len": 3,
        "dropout": 0.1,
        "sigreg_lambda": 0.1,
        "sigreg_projections": 512,
        "hier_loss_weight": base,
        "hier_loss_schedule": schedule,
        "tau_fast": 3.0,
        "tau_slow": 25.0,
        "fixed_alpha": True,
        "wandb": True,
        "wandb_project": WANDB_PROJECT,
        "wandb_entity": WANDB_ENTITY,
        "wandb_mode": WANDB_MODE,
        "wandb_study": WANDB_STUDY,
        "eval_rollout_cache": shared.rel(shared.eval_rollout_cache(job.clean_env)),
        "eval_rollout_episode": 0,
        "extra_tag": "",
        "device": "cuda",
        "feature_manifest_sha256": shared.sha256_file(manifest_path),
        "eval_rollout_cache_sha256": shared.sha256_file(
            shared.eval_rollout_cache(job.clean_env)),
    }


def expected_metric_metadata(job: shared.Job) -> dict[str, Any]:
    _, _, manifest_path = shared.feature_paths(job.clean_env)
    base, schedule, active = design_aux_contract(job.design)
    final_weight = scheduled_weight(base, schedule, 200)
    return {
        "env": job.occ_env,
        "design": job.design,
        "n_actions": 6,
        "influence_schema_version": 2,
        "influence_kind": (
            "single_or_undifferentiated_total" if job.design == "ssm"
            else "per_level_and_total"
        ),
        "prototype_seed": 0,
        "dataset_schema_version": 3,
        "feature_schema_version": 1,
        "feature_manifest": shared.rel(manifest_path),
        "feature_manifest_sha256": shared.sha256_file(manifest_path),
        "target_env": job.clean_env,
        "masked_clean_blackout_loss": True,
        "first_post_loss_weight": 0.5,
        "hier_loss_weight": base,
        "hier_loss_schedule": schedule,
        "hier_loss_weight_final": final_weight,
        "hier_loss_weight_base_effective": base if active else 0.0,
        "hier_loss_weight_effective": final_weight if active else 0.0,
        "val_pred_loss_target_kind": "observed_pre_post_only",
        "deep_blackout_target_kind": "evaluation_only_hidden_clean",
        "primary_common_target_metric": "clean_mse_first_post",
        "encoder_frozen": False,
        "encoder_type": "precomputed",
        "predictor_norm": "none",
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
            shared.eval_rollout_cache(job.clean_env)),
        "eval_rollout_episode": 0,
        **objective_metadata(job.design),
    }


def expected_wandb_artifact_metadata(
    job: shared.Job, rollout_sha256: str,
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
    if not isinstance(history, list) or len(history) != 200:
        raise RunnerError(f"{job.run_name}: expected 200 history records")
    base, schedule, active = design_aux_contract(job.design)
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
            if job.design in HIER_DESIGNS:
                for key in (
                    "hier_loss", "hier_loss_fast", "hier_loss_medium", "hier_loss_weight",
                ):
                    value = values.get(key)
                    if type(value) not in (int, float) or not math.isfinite(float(value)):
                        raise RunnerError(f"{job.run_name}: invalid {split}.{key} at {epoch}")
                wanted = scheduled_weight(base, schedule, epoch) if active else 0.0
                if not math.isclose(float(values["hier_loss_weight"]), wanted,
                                    rel_tol=1e-6, abs_tol=1e-8):
                    raise RunnerError(
                        f"{job.run_name}: {split} hierarchy weight at {epoch} "
                        f"is {values['hier_loss_weight']}, expected {wanted}")
            if job.design in V7_DESIGNS:
                for key in ("hier_loss_bridge", "hier_loss_recovery", "hier_overlap"):
                    value = values.get(key)
                    if type(value) not in (int, float) or not math.isfinite(float(value)):
                        raise RunnerError(f"{job.run_name}: invalid {split}.{key} at {epoch}")
                if float(values["hier_overlap"]) != 0.0:
                    raise RunnerError(
                        f"{job.run_name}: {split} V7 auxiliary overlaps original hidden "
                        f"targets at epoch {epoch}: {values['hier_overlap']}")


def train_command(python: str, job: shared.Job) -> list[str]:
    train_path, val_path, manifest_path = shared.feature_paths(job.clean_env)
    base, schedule, _ = design_aux_contract(job.design)
    return [
        python, shared.rel(TRAIN_SCRIPT),
        "--env-id", job.occ_env,
        "--target-env-id", job.clean_env,
        "--mask-occluded-target-loss",
        "--memory-mode", job.design,
        "--smt-router", "sigmoid",
        "--seed", str(job.seed),
        "--fixed-alpha",
        "--encoder-type", "precomputed",
        "--train-feature-cache", shared.rel(train_path),
        "--val-feature-cache", shared.rel(val_path),
        "--feature-manifest", shared.rel(manifest_path),
        "--prototype-seed", "0",
        "--data-dir", shared.rel(DATA_ROOT),
        "--output-dir", shared.rel(OUTPUT_ROOT),
        "--num-episodes", "600",
        "--val-episodes", "150",
        "--length", "32",
        "--img-size", "64",
        "--epochs", "200",
        "--batch-size", "64",
        "--lr", "3e-4",
        "--weight-decay", "1e-5",
        "--num-workers", "2",
        "--patch-size", "8",
        "--embed-dim", "128",
        "--encoder-layers", "6",
        "--encoder-heads", "4",
        "--predictor-layers", "4",
        "--predictor-heads", "8",
        "--predictor-norm", "none",
        "--history-len", "3",
        "--dropout", "0.1",
        "--sigreg-lambda", "0.1",
        "--sigreg-projections", "512",
        "--hier-loss-weight", str(base),
        "--hier-loss-schedule", schedule,
        "--tau-fast", "3.0",
        "--tau-slow", "25.0",
        "--first-post-loss-weight", "0.5",
        "--wandb",
        "--wandb-project", WANDB_PROJECT,
        "--wandb-entity", WANDB_ENTITY,
        "--wandb-mode", WANDB_MODE,
        "--wandb-study", WANDB_STUDY,
        "--eval-rollout-cache", shared.rel(shared.eval_rollout_cache(job.clean_env)),
        "--eval-rollout-episode", "0",
        "--device", "cuda",
    ]


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
        "V5_DESIGNS": V7_DESIGNS,
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
    shared.scheduled_weight = scheduled_weight
    shared.memory_contract = memory_contract
    shared.build_protocol = build_protocol
    shared.expected_args = expected_args
    shared.expected_metric_metadata = expected_metric_metadata
    shared.validate_history = validate_history
    shared.train_command = train_command
    shared.expected_wandb_artifact_metadata = expected_wandb_artifact_metadata


def check_command_interfaces(python: str) -> None:
    for script, required in (
        (TRAIN_SCRIPT, ("--hier-loss-schedule", "v6_bootstrap", *DESIGNS)),
        (ANALYZE_SCRIPT, ("--phase", "pilot", "final")),
    ):
        result = shared.subprocess.run(
            [python, str(script), "--help"], cwd=ROOT,
            capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RunnerError(f"cannot inspect {script}: {result.stderr.strip()}")
        help_text = result.stdout + result.stderr
        missing = [token for token in required if token not in help_text]
        if missing:
            raise RunnerError(f"{script}: command interface lacks {missing}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the locked 325-cell HACSSM-v7 study.")
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument(
        "--gpus", type=shared.parse_gpu_ids, default=shared.parse_gpu_ids("0,1,2,3"))
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
            + " | ".join(porcelain.splitlines()[:8]))
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
        shared.status(f"preflight validated {len(completed)}/325 runs; pilot={pilot_done}/195")
        if args.dry_run:
            digest = hashlib.sha256(
                json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
            shared.status(f"DRY RUN: no writes or launches; protocol digest={digest}")
            return 0

        shared.check_gpus(args.python, args.gpus)
        shared.verify_provenance_unchanged(protocol)
        shared.run_stage(args.python, PILOT_JOBS, args.gpus, args.workers)
        for job in PILOT_JOBS:
            shared.validate_job(job, allow_missing=False)
        shared.verify_provenance_unchanged(protocol)
        shared.status("running immutable V7 pilot analyzer")
        shared.run_analyzer(args.python, "pilot")
        pilot_passed, pilot_decision = shared.read_pilot_decision()
        shared.status(
            f"V7 pilot decision={pilot_decision['decision']} passed={pilot_passed}; "
            "seeds 3-4 run regardless")

        shared.run_stage(args.python, COMPLETION_JOBS, args.gpus, args.workers)
        for job in ALL_JOBS:
            shared.validate_job(job, allow_missing=False)
        shared.verify_provenance_unchanged(protocol)
        cloud = shared.verify_wandb_cloud(ALL_JOBS)
        shared.status("running final five-seed V7 analyzer")
        shared.run_analyzer(args.python, "final")
        final = shared.read_json(FINAL_DECISION_PATH)
        allowed = {
            "OVERALL_BEST_IN_LOCKED_GRID", "PROMISING_NOT_OVERALL_BEST", "NO_GO",
            "PILOT_NO_GO_FINAL_DESCRIPTIVE",
        }
        if (not isinstance(final, dict) or final.get("decision") not in allowed
                or final.get("completed_runs") != 325
                or final.get("pilot_screen_passed") is not pilot_passed
                or type(final.get("good_enough_for_overall_best_claim")) is not bool):
            raise RunnerError(f"invalid final analyzer decision: {FINAL_DECISION_PATH}")
        shared.verify_provenance_unchanged(protocol)
        shared.validate_artifact_space(ALL_JOBS)
        shared.write_final_manifest(
            protocol, pilot_decision, pilot_passed, args.gpus, args.workers, cloud)
        shared.status(
            "HACSSM-v7 complete: 325/325 validated; "
            f"overall_best={final['good_enough_for_overall_best_claim']}")
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
        print(f"HACSSM-v7 runner error: {exc}", file=sys.stderr)
        raise SystemExit(2)
