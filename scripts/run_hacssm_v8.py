#!/usr/bin/env python3
"""Run the locked 325-cell HACSSM-v8 adaptive-development study.

The audited V5 harness supplies execution, W&B rollout, hashing, resume, and cloud
verification.  This module replaces every study-specific contract before using it.
V8 is deliberately inference-only: every V8 cell has hierarchical loss weight zero,
no teacher, and no ``hier_*`` history fields.  Seeds 0--2 form an immutable 195-cell
pilot; seeds 3--4 always run for a 325-cell final descriptive cohort.
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
ANALYZE_SCRIPT = ROOT / "scripts" / "analyze_hacssm_v8.py"
FEATURE_ROOT = ROOT / "outputs" / "smt_v3_shared" / "dino_features_d128"
OUTPUT_ROOT = ROOT / "outputs" / "hacssm_v8_shared"
LOG_ROOT = ROOT / "logs" / "hacssm_v8_shared"
DATA_ROOT = ROOT / "outputs" / "popgym_data"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PILOT_DECISION_PATH = OUTPUT_ROOT / "pilot_decision.json"
FINAL_DECISION_PATH = OUTPUT_ROOT / "decision.json"
RECEIPT_PATH = OUTPUT_ROOT / "equivalence_receipts.json"
MANIFEST_PATH = OUTPUT_ROOT / "hacssm_v8_manifest.json"
MANIFEST_SHA_PATH = OUTPUT_ROOT / "hacssm_v8_manifest.sha256"
LOCK_PATH = OUTPUT_ROOT / ".run_hacssm_v8.lock"

V7_REFERENCE_ROOT = ROOT / "outputs" / "hacssm_v7_shared"
V7_REFERENCE_MANIFEST = V7_REFERENCE_ROOT / "hacssm_v7_manifest.json"
V7_REFERENCE_MANIFEST_SHA = V7_REFERENCE_ROOT / "hacssm_v7_manifest.sha256"
V7_REFERENCE_SHA256 = "98eda8abec229753381bed5f22c70317428242470cc6f40b6a3f9c16d0f55c11"

BOOTSTRAP_CONTRACT = {
    "schema_version": 1,
    "algorithm": "crossed_environment_seed_percentile_bootstrap",
    "draws": 100_000,
    "seed": 8_008,
    "rng": "numpy.random.Generator(numpy.random.PCG64)",
    "resampling": (
        "independently sample E environment indices and S optimizer-seed indices "
        "with replacement; evaluate the E-by-S Cartesian product; equal-weight mean"
    ),
    "estimand": "mean paired relative reduction (reference-candidate)/reference",
    "quantiles": {"method": "linear", "reported": [0.05, 0.025, 0.975, 0.95]},
}
BOOTSTRAP_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        BOOTSTRAP_CONTRACT, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
).hexdigest()
assert BOOTSTRAP_CONTRACT_SHA256 == (
    "b387010d207f96e9e6777c272ec51629764bfc190cbfd3f323fe6196c38f969e"
)

WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
WANDB_MODE = "online"
WANDB_STUDY = "hacssm-v8"
EVAL_ROLLOUT_EPISODE = 0

ENVIRONMENTS = shared.ENVIRONMENTS
DESIGNS = (
    "ssm",
    "hacssmv6",
    "hacssmv6_static",
    "hacssmv7_noaux",
    "hacssmv7_sharedaction",
    "hacssmv7_norecovery",
    "hacssmv8",
    "hacssmv8_dynamic",
    "hacssmv8_static",
    "hacssmv8_levelaction",
    "hacssmv8_redundant",
    "hacssmv8_noaction",
    "hacssmv8_single",
)
PILOT_SEEDS = (0, 1, 2)
COMPLETION_SEEDS = (3, 4)
ALL_SEEDS = PILOT_SEEDS + COMPLETION_SEEDS

V6_DESIGNS = frozenset({"hacssmv6", "hacssmv6_static"})
V7_DESIGNS = frozenset({
    "hacssmv7_noaux", "hacssmv7_sharedaction", "hacssmv7_norecovery",
})
V8_DESIGNS = frozenset(design for design in DESIGNS if design.startswith("hacssmv8"))
COMPACT_V8_DESIGNS = frozenset(
    V8_DESIGNS - {"hacssmv8_levelaction", "hacssmv8_redundant"})
EXPANDED_V8_DESIGNS = frozenset({"hacssmv8_levelaction", "hacssmv8_redundant"})
PERFORMANCE_ENVELOPE_DESIGNS = tuple(
    design for design in DESIGNS
    if design not in {"hacssmv8", "hacssmv8_redundant"}
)
ANCHOR_DESIGNS = frozenset({
    "ssm", "hacssmv6", "hacssmv6_static", *V7_DESIGNS,
})
HIER_DESIGNS = frozenset(V6_DESIGNS | V7_DESIGNS)
NO_AUX_DESIGNS = frozenset({"hacssmv7_noaux", *V8_DESIGNS})

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
    Path("scripts/run_hacssm_v8.py"),
    Path("scripts/run_hacssm_v5.py"),
    Path("scripts/analyze_hacssm_v8.py"),
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
    PROTOCOL_PATH.name, LOCK_PATH.name, RECEIPT_PATH.name,
    MANIFEST_PATH.name, MANIFEST_SHA_PATH.name,
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
_BASE_VALIDATE_MODEL_STATE = shared.validate_model_state


def design_aux_contract(design: str) -> tuple[float, str, bool]:
    if design in V8_DESIGNS or design == "ssm":
        return 0.0, "fixed", False
    if design in V6_DESIGNS:
        return 0.02, "v6_bootstrap", True
    if design in V7_DESIGNS:
        return 0.02, "v6_bootstrap", design != "hacssmv7_noaux"
    raise RunnerError(f"no auxiliary contract for design {design!r}")


def objective_metadata(design: str) -> dict[str, Any]:
    from scripts.train_popgym import (
        counterfactual_recovery_metadata,
        hierarchical_objective_metadata,
        shared_action_shrinkage_metadata,
    )

    if design in V8_DESIGNS:
        return shared_action_shrinkage_metadata(design)
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
        SharedActionShrinkageMemory,
        SSMMemory,
    )

    dimension, action_dim = 128, 6
    mode_map = {
        "hacssmv8": "learned",
        "hacssmv8_dynamic": "rho1",
        "hacssmv8_static": "rho0",
        "hacssmv8_levelaction": "levelaction",
        "hacssmv8_redundant": "redundant",
        "hacssmv8_noaction": "noaction",
        "hacssmv8_single": "single",
    }
    instances = {
        design: SharedActionShrinkageMemory(dimension, action_dim, mode=mode)
        for design, mode in mode_map.items()
    }
    counts = {design: model.parameter_count() for design, model in instances.items()}
    compact_expected = SharedActionShrinkageMemory.expected_parameter_count(
        dimension, action_dim)
    expanded_expected = SharedActionShrinkageMemory.expected_parameter_count(
        dimension, action_dim, wide_action=True)
    if compact_expected != 34_566 or expanded_expected != 36_102:
        raise RunnerError(
            f"invalid V8 expected counts compact={compact_expected}, expanded={expanded_expected}")
    if any(counts[design] != compact_expected for design in COMPACT_V8_DESIGNS):
        raise RunnerError(f"compact V8 parameter mismatch: {counts}")
    if any(counts[design] != expanded_expected for design in EXPANDED_V8_DESIGNS):
        raise RunnerError(f"expanded V8 parameter mismatch: {counts}")

    compact_signatures = {
        design: [[name, list(parameter.shape)] for name, parameter in instances[design].named_parameters()]
        for design in COMPACT_V8_DESIGNS
    }
    if len({json.dumps(value, sort_keys=True) for value in compact_signatures.values()}) != 1:
        raise RunnerError("compact V8 parameter signatures differ")
    expanded_signatures = {
        design: [[name, list(parameter.shape)] for name, parameter in instances[design].named_parameters()]
        for design in EXPANDED_V8_DESIGNS
    }
    if len({json.dumps(value, sort_keys=True) for value in expanded_signatures.values()}) != 1:
        raise RunnerError("expanded V8 parameter signatures differ")

    ssm_count = sum(parameter.numel() for parameter in SSMMemory(dimension).parameters())
    v6_count = HierarchicalActionConditionedMemory(
        dimension, action_dim, mode="dynamic", taus=(2.0, 8.0)).parameter_count()
    v7_count = HierarchicalCounterfactualRecoveryMemory(
        dimension, action_dim, mode="noaux").parameter_count()
    if (ssm_count, v6_count, v7_count) != (33_024, 34_564, 36_102):
        raise RunnerError(
            f"historical parameter contract changed: {(ssm_count, v6_count, v7_count)}")
    return {
        "embed_dim": dimension,
        "action_dim": action_dim,
        "memory_parameters": {
            "ssm": ssm_count,
            "hacssmv6_all_modes": v6_count,
            "hacssmv7_student": v7_count,
            "hacssmv7_frozen_teacher": v7_count,
            "hacssmv8_compact_modes": compact_expected,
            "hacssmv8_expanded_controls": expanded_expected,
        },
        "v8_trainable_memory_parameters_include_teacher": False,
        "v8_checkpoint_contains_teacher": False,
        "streaming_recurrent_floats": {
            "ssm": dimension,
            "hacssmv6": 2 * dimension,
            "hacssmv7": 2 * dimension,
            "hacssmv8": 2 * dimension,
        },
        "v8_counts_by_design": counts,
        "compact_signature": next(iter(compact_signatures.values())),
        "expanded_signature": next(iter(expanded_signatures.values())),
        "parameter_matching_scope": (
            "compact modes match each other; levelaction/redundant match each other; "
            "the complete V8 ladder is intentionally not parameter matched"
        ),
    }


def _load_v7_reference_manifest() -> dict[str, Any]:
    if not V7_REFERENCE_MANIFEST.is_file() or not V7_REFERENCE_MANIFEST_SHA.is_file():
        raise RunnerError("sealed V7 manifest/sidecar is missing")
    actual = shared.sha256_file(V7_REFERENCE_MANIFEST)
    if actual != V7_REFERENCE_SHA256:
        raise RunnerError(f"sealed V7 manifest SHA changed: {actual}")
    wanted_sidecar = f"{actual}  {V7_REFERENCE_MANIFEST.name}\n"
    if V7_REFERENCE_MANIFEST_SHA.read_text() != wanted_sidecar:
        raise RunnerError("sealed V7 manifest sidecar mismatch")
    manifest = shared.read_json(V7_REFERENCE_MANIFEST)
    if (
        not isinstance(manifest, dict)
        or manifest.get("completed_runs") != 325
        or manifest.get("all_requested_runs_completed") is not True
        or manifest.get("producer_git_clean") is not True
    ):
        raise RunnerError("sealed V7 manifest has an invalid completion contract")
    return manifest


def v7_reference_contract() -> dict[str, Any]:
    manifest = _load_v7_reference_manifest()
    return {
        "manifest": shared.rel(V7_REFERENCE_MANIFEST),
        "manifest_sha256": V7_REFERENCE_SHA256,
        "sidecar": shared.rel(V7_REFERENCE_MANIFEST_SHA),
        "producer_git_commit": manifest.get("producer_git_commit"),
        "completed_runs": manifest.get("completed_runs"),
        "identity_designs": sorted(ANCHOR_DESIGNS),
    }


def build_protocol(commit: str, clean: bool, wandb_preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": "HACSSM-v8 compact shared-action shrinkage adaptive-development study",
        "adaptive_development_only": True,
        "producer_git_commit": commit,
        "producer_git_clean": clean,
        "common_protocol": COMMON,
        "memory_contract": memory_contract(),
        "v8_architecture_contract": {
            "inference": (
                "fixed taus 2 and 8; shared causal action predict/correct; learned per-level "
                "convex shrinkage; joint global read"
            ),
            "internal_auxiliary": "none",
            "teacher": "none",
            "hier_loss_weight": 0.0,
            "hier_loss_schedule": "fixed",
            "hidden_clean_blackout_targets_used": False,
            "training_signal": (
                "ordinary next-latent prediction on visible targets only; no special "
                "internal-state self-supervision"
            ),
            "variants": {
                "hacssmv8": "compact physically shared action head and learned shrinkage",
                "hacssmv8_dynamic": "compact shared head; rho fixed to one",
                "hacssmv8_static": "compact shared head; rho fixed to zero",
                "hacssmv8_levelaction": "expanded level-specific action heads",
                "hacssmv8_redundant": "expanded heads averaged before both transitions",
                "hacssmv8_noaction": "compact tensors; action features forced to zero",
                "hacssmv8_single": "compact tensors; medium-only inference read",
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
        "identity_contract": {
            "sealed_v7_reference": v7_reference_contract(),
            "v7_anchor_reruns": "model tensors, histories, primary metrics, rollout arrays bit-exact",
            "v8_levelaction_vs_v7_noaux": (
                "student tensors after namespace mapping, primary histories/metrics, and rollouts bit-exact"
            ),
            "v8_redundant": "two action-head blocks remain bit-exact equal",
            "receipt_file": shared.rel(RECEIPT_PATH),
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
            "command": "scripts/analyze_hacssm_v8.py --phase pilot",
            "decision_file": shared.rel(PILOT_DECISION_PATH),
            "fail_closed_result": "NO_GO",
            "pilot_pass_result": "PILOT_OVERALL_BEST_PASS",
            "final_claim_field": "best_in_focused_locked_grid",
            "scope": "adaptive_development_only",
        },
        "performance_envelope": {
            "designs": list(PERFORMANCE_ENVELOPE_DESIGNS),
            "excluded": ["hacssmv8", "hacssmv8_redundant"],
            "rationale": (
                "the candidate cannot be its own reference; redundant is a statistical "
                "equivalence/optimization receipt, not a distinct deployable architecture"
            ),
        },
        "endpoint_envelope": {
            "pairwise_reduction_and_wins": (
                "select min(dynamic, static) independently within every environment-seed cell"
            ),
            "environment_wins": (
                "compare the candidate environment mean with the minimum of the dynamic and "
                "static endpoint environment means"
            ),
        },
        "pilot_success_criteria": {
            "vs_ssm": ">=6% reduction, >=10/15 wins, >=4/5 environment wins",
            "vs_v7_sharedaction": ">=0.5% reduction, >=9/15 wins, >=3/5 environment wins",
            "vs_v6_static": "positive reduction, >=8/15 wins, >=3/5 environment wins",
            "vs_full_v6": ">=1% reduction, >=9/15 wins, >=3/5 environment wins",
            "vs_v7_norecovery": "positive reduction, >=8/15 wins, >=3/5 environment wins",
            "redundant_vs_levelaction": ">=0.5% reduction, >=9/15 wins, >=3/5 environment wins",
            "vs_each_shrinkage_endpoint": (
                "positive reduction, >=8/15 wins, >=3/5 environment wins"
            ),
            "vs_better_endpoint_envelope": ">=0.5% reduction, >=9/15 wins, >=3/5 environment wins",
            "compact_redundant_equivalence": (
                "absolute mean <=0.25%, every environment within +/-1%, crossed-bootstrap "
                "90% interval inside +/-1%"
            ),
            "structural_controls": ">=3% reduction, >=11/15 wins, >=3/5 environment wins",
            "convergence": "absolute median <1%, p95 <3%, max <5%",
        },
        "final_success_criteria": {
            "requires_pilot_pass": True,
            "vs_ssm": ">=7% reduction, >=20/25 wins, >=4/5 environment wins",
            "vs_v7_sharedaction": ">=1% reduction, >=15/25 wins, >=3/5 environment wins",
            "vs_v6_static": ">=1% reduction, >=15/25 wins, >=3/5 environment wins",
            "vs_full_v6": ">=1% reduction, >=15/25 wins, >=3/5 environment wins",
            "vs_v7_norecovery": "positive reduction, >=13/25 wins, >=3/5 environment wins",
            "redundant_vs_levelaction": ">=0.5% reduction, >=15/25 wins, >=3/5 environment wins",
            "vs_better_endpoint_envelope": ">=1% reduction, >=15/25 wins, >=3/5 environment wins",
            "vs_each_shrinkage_endpoint": (
                "positive reduction, >=13/25 wins, >=3/5 environment wins"
            ),
            "compact_redundant_equivalence": (
                "absolute mean <=0.25%, every environment within +/-1%, crossed-bootstrap "
                "90% interval inside +/-1%"
            ),
            "structural_controls": ">=3% reduction, >=17/25 wins, >=3/5 environment wins",
            "envelope_and_hold": (
                ">=3/5 wins over the frozen performance-envelope design list (redundant "
                "excluded) and >=4/5 last-visible-hold wins"
            ),
            "convergence": "absolute median <1%, p95 <3%, max <5%",
        },
        "compact_noninferiority_criteria": {
            "shared_requirements": {
                "vs_v7_sharedaction": (
                    "point estimate >-0.5%, >=4/5 environment effects >-1%, deterministic "
                    "crossed-bootstrap 95% lower bound >-1%"
                ),
                "vs_ssm": ">=6% reduction",
                "compact_redundant_equivalence": (
                    "absolute mean <=0.25%, every environment within +/-1%, crossed-bootstrap "
                    "90% interval inside +/-1%"
                ),
                "convergence": "absolute median <1%, p95 <3%, max <5%",
            },
            "pilot": {
                "redundant_vs_levelaction": (
                    ">=0.5% reduction, >=9/15 wins, >=3/5 environment wins"
                ),
                "vs_better_endpoint_envelope": (
                    ">=0.5% reduction, >=9/15 wins, >=3/5 environment wins"
                ),
                "role": "diagnostic receipt only; cannot reopen a failed strict pilot",
            },
            "final": {
                "requires_strict_pilot_pass": True,
                "redundant_vs_levelaction": (
                    ">=0.5% reduction, >=15/25 wins, >=3/5 environment wins"
                ),
                "vs_better_endpoint_envelope": (
                    ">=1% reduction, >=15/25 wins, >=3/5 environment wins"
                ),
            },
            "bootstrap_contract": BOOTSTRAP_CONTRACT,
            "bootstrap_contract_sha256": BOOTSTRAP_CONTRACT_SHA256,
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
        "num_episodes": COMMON["train_episodes"],
        "val_episodes": COMMON["val_episodes"],
        "data_dir": shared.rel(DATA_ROOT),
        "prototype_seed": COMMON["prototype_seed"],
        "target_env_id": job.clean_env,
        "mask_occluded_target_loss": True,
        "first_post_loss_weight": COMMON["first_post_loss_weight"],
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
        "eval_rollout_episode": EVAL_ROLLOUT_EPISODE,
        "extra_tag": "",
        "device": "cuda",
        "feature_manifest_sha256": shared.sha256_file(manifest_path),
        "eval_rollout_cache_sha256": shared.sha256_file(
            shared.eval_rollout_cache(job.clean_env)),
    }


def expected_metric_metadata(job: shared.Job) -> dict[str, Any]:
    _, _, manifest_path = shared.feature_paths(job.clean_env)
    base, schedule, active = design_aux_contract(job.design)
    final_weight = scheduled_weight(base, schedule, COMMON["epochs"])
    return {
        "env": job.occ_env,
        "design": job.design,
        "n_actions": 6,
        "influence_schema_version": 2,
        "influence_kind": (
            "single_or_undifferentiated_total" if job.design == "ssm"
            else "per_level_and_total"
        ),
        "prototype_seed": COMMON["prototype_seed"],
        "dataset_schema_version": 3,
        "feature_schema_version": 1,
        "feature_manifest": shared.rel(manifest_path),
        "feature_manifest_sha256": shared.sha256_file(manifest_path),
        "target_env": job.clean_env,
        "masked_clean_blackout_loss": True,
        "first_post_loss_weight": COMMON["first_post_loss_weight"],
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
            shared.eval_rollout_cache(job.clean_env)),
        "eval_rollout_episode": EVAL_ROLLOUT_EPISODE,
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
    if not isinstance(history, list) or len(history) != COMMON["epochs"]:
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
            if job.design in V8_DESIGNS:
                forbidden = sorted(key for key in values if key.startswith("hier_"))
                if forbidden:
                    raise RunnerError(
                        f"{job.run_name}: V8 history contains forbidden hierarchy fields "
                        f"at epoch {epoch}: {forbidden}")
            if job.design in HIER_DESIGNS:
                for key in ("hier_loss", "hier_loss_fast", "hier_loss_medium", "hier_loss_weight"):
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
                        f"{job.run_name}: {split} V7 auxiliary overlaps hidden targets at {epoch}")


def validate_model_state(state: Any, job: shared.Job) -> None:
    _BASE_VALIDATE_MODEL_STATE(state, job)
    names = set(state) if isinstance(state, dict) else set()
    if job.design not in V8_DESIGNS:
        if any(name.startswith("mem_hacssmv8.") for name in names):
            raise RunnerError(f"{job.run_name}: non-V8 checkpoint contains V8 tensors")
        return
    if not any(name.startswith("mem_hacssmv8.") for name in names):
        raise RunnerError(f"{job.run_name}: V8 checkpoint has no V8 memory namespace")
    if any("teacher" in name or name.startswith("mem_hacssmv7.") for name in names):
        raise RunnerError(f"{job.run_name}: V8 checkpoint contains teacher/V7 tensors")
    action = state.get("mem_hacssmv8.W_a.weight")
    shrink = state.get("mem_hacssmv8.shrink_logits")
    wanted_action = (512, 6) if job.design in EXPANDED_V8_DESIGNS else (256, 6)
    if action is None or tuple(action.shape) != wanted_action:
        shape = None if action is None else tuple(action.shape)
        raise RunnerError(f"{job.run_name}: V8 action shape {shape}, expected {wanted_action}")
    if shrink is None or tuple(shrink.shape) != (2,):
        raise RunnerError(f"{job.run_name}: invalid V8 shrinkage tensor")
    if job.design == "hacssmv8_redundant" and not action[:256].equal(action[256:]):
        raise RunnerError(f"{job.run_name}: redundant action heads diverged")


def train_command(python: str, job: shared.Job) -> list[str]:
    train_path, val_path, manifest_path = shared.feature_paths(job.clean_env)
    base, schedule, _ = design_aux_contract(job.design)
    return [
        python, shared.rel(TRAIN_SCRIPT),
        "--env-id", job.occ_env,
        "--target-env-id", job.clean_env,
        "--mask-occluded-target-loss",
        "--memory-mode", job.design,
        "--smt-router", COMMON["smt_router"],
        "--seed", str(job.seed),
        "--fixed-alpha",
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
        "--hier-loss-weight", str(base),
        "--hier-loss-schedule", schedule,
        "--tau-fast", "3.0",
        "--tau-slow", "25.0",
        "--first-post-loss-weight", str(COMMON["first_post_loss_weight"]),
        "--wandb",
        "--wandb-project", WANDB_PROJECT,
        "--wandb-entity", WANDB_ENTITY,
        "--wandb-mode", WANDB_MODE,
        "--wandb-study", WANDB_STUDY,
        "--eval-rollout-cache", shared.rel(shared.eval_rollout_cache(job.clean_env)),
        "--eval-rollout-episode", str(EVAL_ROLLOUT_EPISODE),
        "--device", "cuda",
    ]


def _assert_v7_reference_file(path: Path, manifest: Mapping[str, Any]) -> None:
    key = shared.rel(path)
    expected = manifest.get("output_artifacts", {}).get(key)
    observed = {"kind": "file", **shared.file_record(path)}
    if not isinstance(expected, dict) or not shared.stable_equal(expected, observed):
        raise RunnerError(f"sealed V7 reference file changed or is unmanifested: {key}")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RunnerError(f"cannot load receipt checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise RunnerError(f"receipt checkpoint is not an object: {path}")
    return checkpoint


def _assert_tensor_state_equal(left: Mapping[str, Any], right: Mapping[str, Any], label: str) -> None:
    if left.keys() != right.keys():
        raise RunnerError(f"{label}: state keys differ")
    for name in left:
        if not left[name].equal(right[name]):
            raise RunnerError(f"{label}: tensor differs at {name}")


def _assert_rollout_equal(left: Path, right: Path, label: str) -> None:
    import numpy as np

    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if a.files != b.files:
            raise RunnerError(f"{label}: rollout fields differ")
        for name in a.files:
            if not np.array_equal(a[name], b[name]):
                raise RunnerError(f"{label}: rollout differs at {name}")


def _primary_metric_subset(metrics: Mapping[str, Any]) -> dict[str, Any]:
    prefixes = (
        "clean_mse_", "clean_input_mse_", "constant_mse_", "persistence_mse_",
        "last_visible_mse_",
    )
    exact = {"val_pred_loss", "infl_all", "infl_fast", "infl_slow"}
    return {
        key: value for key, value in metrics.items()
        if key in exact or key.startswith(prefixes)
    }


def _base_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "loss", "pred_loss", "sigreg_loss", "pred_loss_all_valid", "pred_loss_first_post",
    )
    return [
        {
            "epoch": record["epoch"],
            "train": {key: record["train"][key] for key in keys if key in record["train"]},
            "val": {key: record["val"][key] for key in keys if key in record["val"]},
        }
        for record in history
    ]


def _canonical_v7_student_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for name, tensor in state.items():
        if name.startswith("mem_hacssmv7_teacher."):
            continue
        if name.startswith("mem_hacssmv7."):
            name = "mem_hacssmv8." + name.removeprefix("mem_hacssmv7.")
        result[name] = tensor
    return result


def validate_equivalence_receipts(jobs: Sequence[shared.Job]) -> dict[str, Any]:
    manifest = _load_v7_reference_manifest()
    completed = {job.run_name: job for job in jobs}
    anchor_records = []
    levelaction_records = []
    redundant_records = []

    for job in jobs:
        if job.design in ANCHOR_DESIGNS:
            reference_dir = V7_REFERENCE_ROOT / job.run_name
            reference_model = reference_dir / "model.pt"
            reference_metrics = reference_dir / "metrics.json"
            reference_rollout = reference_dir / "eval_rollout.npz"
            for path in (reference_model, reference_metrics, reference_rollout):
                _assert_v7_reference_file(path, manifest)
            current = _load_checkpoint(job.model_path)
            reference = _load_checkpoint(reference_model)
            _assert_tensor_state_equal(
                current["model_state_dict"], reference["model_state_dict"],
                f"{job.run_name} sealed-anchor")
            if not shared.stable_equal(current["history"], reference["history"]):
                raise RunnerError(f"{job.run_name}: sealed-anchor history differs")
            current_metrics = shared.read_json(job.metrics_path)
            reference_metric_values = shared.read_json(reference_metrics)
            if not shared.stable_equal(
                _primary_metric_subset(current_metrics),
                _primary_metric_subset(reference_metric_values),
            ):
                raise RunnerError(f"{job.run_name}: sealed-anchor primary metrics differ")
            _assert_rollout_equal(job.eval_rollout_path, reference_rollout, job.run_name)
            anchor_records.append({
                "run": job.run_name,
                "reference_model_sha256": shared.sha256_file(reference_model),
                "current_model_sha256": shared.sha256_file(job.model_path),
                "model_tensors_exact": True,
                "history_exact": True,
                "primary_metrics_exact": True,
                "rollout_exact": True,
            })

        if job.design == "hacssmv8_redundant":
            checkpoint = _load_checkpoint(job.model_path)
            action = checkpoint["model_state_dict"]["mem_hacssmv8.W_a.weight"]
            if not action[:256].equal(action[256:]):
                raise RunnerError(f"{job.run_name}: redundant heads differ in receipt audit")
            redundant_records.append({"run": job.run_name, "head_blocks_exact": True})

        if job.design == "hacssmv8_levelaction":
            peer_name = shared.Job(
                job.stage, job.seed, job.occ_env, job.clean_env, "hacssmv7_noaux").run_name
            peer = completed.get(peer_name)
            if peer is None:
                raise RunnerError(f"{job.run_name}: missing V7-noaux identity peer")
            v8 = _load_checkpoint(job.model_path)
            v7 = _load_checkpoint(peer.model_path)
            _assert_tensor_state_equal(
                v8["model_state_dict"],
                _canonical_v7_student_state(v7["model_state_dict"]),
                f"{job.run_name} V8-levelaction/V7-noaux")
            if not shared.stable_equal(_base_history(v8["history"]), _base_history(v7["history"])):
                raise RunnerError(f"{job.run_name}: V8/V7 base histories differ")
            if not shared.stable_equal(
                _primary_metric_subset(shared.read_json(job.metrics_path)),
                _primary_metric_subset(shared.read_json(peer.metrics_path)),
            ):
                raise RunnerError(f"{job.run_name}: V8/V7 primary metrics differ")
            _assert_rollout_equal(job.eval_rollout_path, peer.eval_rollout_path, job.run_name)
            levelaction_records.append({
                "run": job.run_name,
                "peer": peer.run_name,
                "student_tensors_exact": True,
                "base_history_exact": True,
                "primary_metrics_exact": True,
                "rollout_exact": True,
            })

    return {
        "schema_version": 1,
        "sealed_v7_manifest_sha256": V7_REFERENCE_SHA256,
        "validated_jobs": len(jobs),
        "sealed_anchor_identities": anchor_records,
        "v8_levelaction_v7_noaux_identities": levelaction_records,
        "v8_redundant_head_receipts": redundant_records,
        "counts": {
            "sealed_anchor_identities": len(anchor_records),
            "v8_levelaction_v7_noaux_identities": len(levelaction_records),
            "v8_redundant_head_receipts": len(redundant_records),
        },
    }


def read_pilot_decision() -> tuple[bool, dict[str, Any]]:
    decision = shared.read_json(PILOT_DECISION_PATH)
    if not isinstance(decision, dict) or type(decision.get("pilot_screen_passed")) is not bool:
        raise RunnerError(
            f"{PILOT_DECISION_PATH} requires boolean pilot_screen_passed")
    expected = "PILOT_OVERALL_BEST_PASS" if decision["pilot_screen_passed"] else "NO_GO"
    if decision.get("decision") != expected:
        raise RunnerError(
            f"{PILOT_DECISION_PATH}: decision={decision.get('decision')!r}, expected {expected!r}")
    if decision.get("adaptive_development_only") is not True:
        raise RunnerError(f"{PILOT_DECISION_PATH}: invalid adaptive-development scope")
    if decision.get("scope") not in (None, "adaptive_development_only"):
        raise RunnerError(f"{PILOT_DECISION_PATH}: conflicting scope")
    shared.assert_finite_tree(decision, "pilot_decision")
    return decision["pilot_screen_passed"], decision


def write_final_manifest(
    protocol: dict[str, Any], pilot_decision: dict[str, Any], pilot_screen_passed: bool,
    gpu_ids: Sequence[str], workers: int, cloud: dict[str, Any], final: dict[str, Any],
) -> None:
    """Publish through the audited writer, then bind the validated final decision itself."""
    shared.write_final_manifest(
        protocol, pilot_decision, pilot_screen_passed, gpu_ids, workers, cloud)
    manifest = shared.read_json(MANIFEST_PATH)
    if not isinstance(manifest, dict):
        raise RunnerError("fresh V8 manifest is not an object")
    manifest["final_decision"] = final
    manifest["equivalence_receipts"] = {
        "path": shared.rel(RECEIPT_PATH),
        **shared.file_record(RECEIPT_PATH),
    }
    shared.atomic_write_json(MANIFEST_PATH, manifest)
    manifest_sha = shared.sha256_file(MANIFEST_PATH)
    shared.atomic_write_bytes(
        MANIFEST_SHA_PATH, f"{manifest_sha}  {MANIFEST_PATH.name}\n".encode())
    if shared.sha256_file(MANIFEST_PATH) != manifest_sha:
        raise RunnerError("V8 manifest changed immediately after final-decision publication")


def validate_final_decision(final: Any, pilot_passed: bool) -> tuple[bool, bool]:
    allowed = {
        "OVERALL_BEST_ADAPTIVE_DEV", "COMPACT_NONINFERIOR_ADAPTIVE_DEV",
        "NO_GO", "PILOT_NO_GO_FINAL_DESCRIPTIVE",
    }
    if not isinstance(final, dict):
        raise RunnerError(f"invalid final analyzer decision: {FINAL_DECISION_PATH}")
    best = final.get("best_in_focused_locked_grid")
    noninferior = final.get("compact_noninferior_to_v7_leader")
    if (
        final.get("decision") not in allowed
        or final.get("completed_runs") != 325
        or final.get("pilot_screen_passed") is not pilot_passed
        or final.get("scope") != "adaptive_development_only"
        or type(best) is not bool
        or type(noninferior) is not bool
    ):
        raise RunnerError(f"invalid final analyzer decision: {FINAL_DECISION_PATH}")
    if final["decision"] == "OVERALL_BEST_ADAPTIVE_DEV" and not (pilot_passed and best):
        raise RunnerError("overall-best label conflicts with pilot/best fields")
    if final["decision"] == "OVERALL_BEST_ADAPTIVE_DEV" and noninferior:
        raise RunnerError("overall-best decision must not also carry the fallback label")
    if final["decision"] == "COMPACT_NONINFERIOR_ADAPTIVE_DEV" and not (
        pilot_passed and not best and noninferior
    ):
        raise RunnerError("compact-noninferior label conflicts with decision fields")
    if final["decision"] == "NO_GO" and not (
        pilot_passed and not best and not noninferior
    ):
        raise RunnerError("NO_GO label conflicts with decision fields")
    if final["decision"] == "PILOT_NO_GO_FINAL_DESCRIPTIVE" and not (
        not pilot_passed and not best and not noninferior
    ):
        raise RunnerError("failed-pilot label conflicts with decision fields")
    return best, noninferior


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
        "V5_DESIGNS": V8_DESIGNS,
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
    shared.validate_model_state = validate_model_state
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
            capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RunnerError(f"cannot inspect {script}: {result.stderr.strip()}")
        help_text = result.stdout + result.stderr
        missing = [token for token in required if token not in help_text]
        if missing:
            raise RunnerError(f"{script}: command interface lacks {missing}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the locked 325-cell HACSSM-v8 study.")
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
        validate_equivalence_receipts(PILOT_JOBS)
        shared.verify_provenance_unchanged(protocol)
        shared.status("running immutable V8 pilot analyzer")
        shared.run_analyzer(args.python, "pilot")
        pilot_passed, pilot_decision = shared.read_pilot_decision()
        shared.status(
            f"V8 pilot decision={pilot_decision['decision']} passed={pilot_passed}; "
            "seeds 3-4 run regardless")

        shared.run_stage(args.python, COMPLETION_JOBS, args.gpus, args.workers)
        for job in ALL_JOBS:
            shared.validate_job(job, allow_missing=False)
        receipts = validate_equivalence_receipts(ALL_JOBS)
        shared.atomic_write_json(RECEIPT_PATH, receipts)
        shared.verify_provenance_unchanged(protocol)
        cloud = shared.verify_wandb_cloud(ALL_JOBS)
        shared.status("running final five-seed V8 analyzer")
        shared.run_analyzer(args.python, "final")
        final = shared.read_json(FINAL_DECISION_PATH)
        best, noninferior = validate_final_decision(final, pilot_passed)
        shared.verify_provenance_unchanged(protocol)
        shared.validate_artifact_space(ALL_JOBS)
        write_final_manifest(
            protocol, pilot_decision, pilot_passed, args.gpus, args.workers, cloud, final)
        shared.status(
            "HACSSM-v8 complete: 325/325 validated; "
            f"focused_best={best} compact_noninferior={noninferior}")
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
        print(f"HACSSM-v8 runner error: {exc}", file=sys.stderr)
        raise SystemExit(2)
