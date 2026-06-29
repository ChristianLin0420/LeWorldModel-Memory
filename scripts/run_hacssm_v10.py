#!/usr/bin/env python3
"""Run the frozen 225-cell ORBIT-v10-J/R1 adaptive confirmation study.

Five DMC tasks are trained from native RGB with a joint-gradient clean target
and equal-weight prediction, variance, and covariance losses.  Each environment
owns one immutable train and validation bundle; the
bundles contain clean simulator state plus deterministic train and held-out
corruption views.  Seeds 0--2 are an immutable pilot and seeds 3--4 always run.

R1 is an explicitly adaptive revision after the invalid predecessor's
normalization audit.  It retains the exact 225-cell grid and data while moving
every new local and W&B artifact into an isolated namespace and sealing the
predecessor provenance in the protocol.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_hacssm_v5 as shared


REPO_ROOT = ROOT
TRAIN_SCRIPT = ROOT / "scripts" / "train_hacssm_v10.py"
DATA_SCRIPT = ROOT / "scripts" / "hacssm_v10_data.py"
ANALYZE_SCRIPT = ROOT / "scripts" / "analyze_hacssm_v10.py"
DATA_ROOT = ROOT / "outputs" / "hacssm_v10_data"
OUTPUT_ROOT = ROOT / "outputs" / "hacssm_v10_r1_shared"
LOG_ROOT = ROOT / "logs" / "hacssm_v10_r1_shared"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
PILOT_DECISION_PATH = OUTPUT_ROOT / "pilot_decision.json"
FINAL_DECISION_PATH = OUTPUT_ROOT / "decision.json"
MANIFEST_PATH = OUTPUT_ROOT / "hacssm_v10_r1_manifest.json"
MANIFEST_SHA_PATH = OUTPUT_ROOT / "hacssm_v10_r1_manifest.sha256"
LOCK_PATH = OUTPUT_ROOT / ".run_hacssm_v10_r1.lock"

WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
WANDB_MODE = "online"
WANDB_STUDY = "hacssm-v10-r1"
EVAL_ROLLOUT_EPISODE = 0
SCOPE = "adaptive_after_normalization_audit"
V10J_OBJECTIVE = "v10j_joint_pred_variance_covariance_equal_weight"
PREDECESSOR_PROVENANCE = {
    "study": "hacssm-v10",
    "producer_git_commit": "5d561cc2a5e312f0e9c06d2492859e85fc1debe9",
    "protocol_sha256": "d446b70abb0ece3560ea7939117bc4c8b9b909dbab6c9517790971d3b1c20934",
    "protocol_archive": (
        "outputs/hacssm_v10_invalid_none_norm_20260629T1707/protocol.json"
    ),
    "output_archive": "outputs/hacssm_v10_invalid_none_norm_20260629T1707",
    "log_archive": "logs/hacssm_v10_invalid_none_norm_20260629T1707",
    "wandb_run_ids": ["jqf47nm9", "zlk8974u", "kbn9rxpt", "69sb8eod"],
    "completed_cells": 0,
    "partially_trained_cells": 4,
    "invalidation_reason": (
        "encoder_norm=none failed the representation-quality audit; all predecessor "
        "partial cells are excluded from R1"
    ),
}

ENVIRONMENTS = (
    ("dmc:walker.walk", "dmc:walker.walk"),
    ("dmc:hopper.hop", "dmc:hopper.hop"),
    ("dmc:cartpole.swingup", "dmc:cartpole.swingup"),
    ("dmc:pendulum.swingup", "dmc:pendulum.swingup"),
    ("dmc:fish.swim", "dmc:fish.swim"),
)
DESIGNS = (
    "none",
    "gru",
    "ssm",
    "hacssmv8",
    "orbitv10",
    "orbitv10_noaction",
    "orbitv10_additive",
    "orbitv10_scaled",
    "orbitv10_static",
)
ORBIT_DESIGNS = frozenset(design for design in DESIGNS if design.startswith("orbitv10"))
PILOT_SEEDS = (0, 1, 2)
COMPLETION_SEEDS = (3, 4)
ALL_SEEDS = PILOT_SEEDS + COMPLETION_SEEDS

TRAIN_CORRUPTIONS = ("cutout", "meanframe")
HELDOUT_CORRUPTIONS = ("freeze", "gaussian_noise", "checkerboard", "long_freeze")
ENCODER_RECEIPT_BASES = (
    "encoder_mean_channel_variance",
    "encoder_covariance_effective_rank",
    "encoder_singleton_max_abs",
    "encoder_prefix_max_abs",
)
COMMON = {
    "train_episodes": 1200,
    "val_episodes": 240,
    "length": 48,
    "img_size": 64,
    "embed_dim": 128,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "history_len": 3,
    "encoder_layers": 6,
    "encoder_heads": 4,
    "predictor_layers": 4,
    "predictor_heads": 8,
    "encoder_norm": "causal",
    "predictor_norm": "none",
    "epochs": 100,
    "train_dataloader_workers": 2,
    "prototype_seed": 0,
    "train_rollout_seed": 27_100,
    "val_rollout_seed": 92_710,
    "corruption_seed": 10_012,
    "sigreg_lambda": 0.1,
    "sigreg_projections": 512,
    "training_objective": V10J_OBJECTIVE,
    "prediction_loss_weight": 1.0,
    "variance_loss_weight": 1.0,
    "covariance_loss_weight": 1.0,
    "state_probe_ridge": 1e-3,
    "first_post_loss_weight": 0.0,
    "hier_loss_weight": 0.0,
    "wandb": True,
    "wandb_entity": WANDB_ENTITY,
    "wandb_project": WANDB_PROJECT,
    "wandb_mode": WANDB_MODE,
    "wandb_study": WANDB_STUDY,
    "eval_rollout_episode": EVAL_ROLLOUT_EPISODE,
}

SOURCE_FILES = (
    Path("scripts/run_hacssm_v10.py"),
    Path("scripts/analyze_hacssm_v10.py"),
    Path("scripts/run_hacssm_v5.py"),
    Path("scripts/analyze_hacssm_v5.py"),
    Path("scripts/train_hacssm_v10.py"),
    Path("scripts/hacssm_v10_data.py"),
    Path("lewm/data.py"),
    Path("lewm/envs/dmc_collect.py"),
    Path("lewm/models/__init__.py"),
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
        shared.Job(stage, seed, environment, environment, design)
        for seed in seeds
        for environment, _clean in ENVIRONMENTS
        for design in DESIGNS
    )


PILOT_JOBS = make_jobs("pilot", PILOT_SEEDS)
COMPLETION_JOBS = make_jobs("completion", COMPLETION_SEEDS)
ALL_JOBS = PILOT_JOBS + COMPLETION_JOBS
assert len(PILOT_JOBS) == 135
assert len(COMPLETION_JOBS) == 90
assert len(ALL_JOBS) == 225 and len({job.run_name for job in ALL_JOBS}) == 225

RunnerError = shared.RunnerError


def safe_environment(environment: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", environment).strip("_")


def data_paths(environment: str) -> tuple[Path, Path, Path]:
    task = environment.removeprefix("dmc:").replace(".", "_")
    return (
        DATA_ROOT / (
            f"dmc_{task}_train_n{COMMON['train_episodes']}_L{COMMON['length']}"
            f"_s{COMMON['img_size']}_seed{COMMON['train_rollout_seed']}.npz"
        ),
        DATA_ROOT / (
            f"dmc_{task}_val_n{COMMON['val_episodes']}_L{COMMON['length']}"
            f"_s{COMMON['img_size']}_seed{COMMON['val_rollout_seed']}.npz"
        ),
        DATA_ROOT / "manifest.json",
    )


def source_snapshot() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise RunnerError(f"required V10 source is missing: {path}")
        records[relative.as_posix()] = shared.file_record(path)
    return records


def _scalar(data: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in data:
        raise RunnerError(f"{path}: missing scalar {key}")
    value = np.asarray(data[key])
    if value.shape != ():
        raise RunnerError(f"{path}: {key} is not scalar")
    return value.item()


def validate_data_bundle(
    path: Path, *, environment: str, split: str, episodes: int,
) -> dict[str, Any]:
    """Validate the immutable multi-view native-RGB/state bundle."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise RunnerError(f"missing V10 data bundle: {path}")
    try:
        from scripts.hacssm_v10_data import load_cache
        metadata = load_cache(path, verify=True)
    except Exception as exc:
        raise RunnerError(f"{path}: canonical cache verification failed: {exc}") from exc
    if (
        metadata.split != split
        or metadata.seed != (
            COMMON["train_rollout_seed"] if split == "train" else COMMON["val_rollout_seed"]
        )
        or metadata.length != COMMON["length"]
        or metadata.img_size != COMMON["img_size"]
        or metadata.episodes != episodes
    ):
        raise RunnerError(f"{path}: canonical metadata differs from V10 protocol")
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "schema_version", "env_id", "split", "seed", "length", "img_size",
                "smooth_rho", "obs", "actions", "physics_state", "rewards",
                "action_min", "action_max", "content_sha256",
            }
            missing = required - set(data.files)
            if missing:
                raise RunnerError(f"{path}: missing fields {sorted(missing)}")
            if int(_scalar(data, "schema_version", path)) != 1:
                raise RunnerError(f"{path}: unsupported schema version")
            if str(_scalar(data, "env_id", path)) not in {
                environment, environment.removeprefix("dmc:")
            }:
                raise RunnerError(f"{path}: environment mismatch")
            if str(_scalar(data, "split", path)) != split:
                raise RunnerError(f"{path}: split mismatch")
            clean = np.asarray(data["obs"])
            actions = np.asarray(data["actions"])
            state = np.asarray(data["physics_state"])
            expected_pixels = (episodes, COMMON["length"], 64, 64, 3)
            if clean.shape != expected_pixels or clean.dtype != np.uint8:
                raise RunnerError(f"{path}: invalid clean pixels {clean.shape}/{clean.dtype}")
            if actions.shape[:2] != (episodes, COMMON["length"] - 1):
                raise RunnerError(f"{path}: invalid action shape {actions.shape}")
            if state.shape[:2] != (episodes, COMMON["length"]):
                raise RunnerError(f"{path}: invalid physics-state shape {state.shape}")
            if not np.issubdtype(state.dtype, np.floating) or not np.isfinite(state).all():
                raise RunnerError(f"{path}: non-finite/non-floating physics state")
            if int(_scalar(data, "seed", path)) != (
                COMMON["train_rollout_seed"] if split == "train" else COMMON["val_rollout_seed"]
            ):
                raise RunnerError(f"{path}: rollout seed mismatch")
            if int(_scalar(data, "length", path)) != COMMON["length"]:
                raise RunnerError(f"{path}: sequence length metadata mismatch")
            if int(_scalar(data, "img_size", path)) != COMMON["img_size"]:
                raise RunnerError(f"{path}: image-size metadata mismatch")
            content_sha = str(_scalar(data, "content_sha256", path))
            if not re.fullmatch(r"[0-9a-f]{64}", content_sha):
                raise RunnerError(f"{path}: invalid canonical content hash")
            for name in data.files:
                value = np.asarray(data[name])
                if value.dtype.hasobject:
                    raise RunnerError(f"{path}: object array forbidden at {name}")
                if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                    raise RunnerError(f"{path}: non-finite values at {name}")
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(f"cannot validate V10 data {path}: {exc}") from exc
    return shared.file_record(path)


def data_snapshot() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    expected: set[Path] = set()
    manifest_path = DATA_ROOT / "manifest.json"
    manifest_sidecar = DATA_ROOT / "manifest.sha256"
    manifest = shared.read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RunnerError(f"{manifest_path}: manifest is not an object")
    for environment, _ in ENVIRONMENTS:
        train_path, val_path, _shared_manifest_path = data_paths(environment)
        if _shared_manifest_path != manifest_path:
            raise RunnerError("V10 environments do not share one root manifest")
        expected.update((train_path, val_path, train_path.with_suffix(".npz.sha256"),
                         val_path.with_suffix(".npz.sha256")))
        records[shared.rel(train_path)] = validate_data_bundle(
            train_path, environment=environment, split="train",
            episodes=COMMON["train_episodes"],
        )
        records[shared.rel(val_path)] = validate_data_bundle(
            val_path, environment=environment, split="val",
            episodes=COMMON["val_episodes"],
        )
        for path in (train_path, val_path):
            sidecar = path.with_suffix(".npz.sha256")
            if not sidecar.is_file():
                raise RunnerError(f"missing data sidecar: {sidecar}")
            tokens = sidecar.read_text().strip().split()
            if len(tokens) < 1 or tokens[0] != shared.sha256_file(path):
                raise RunnerError(f"{sidecar}: NPZ hash mismatch")
            records[shared.rel(sidecar)] = shared.file_record(sidecar)
    expected.update((manifest_path, manifest_sidecar))
    manifest_tokens = manifest_sidecar.read_text().strip().split()
    if len(manifest_tokens) < 1 or manifest_tokens[0] != shared.sha256_file(manifest_path):
        raise RunnerError("V10 data manifest sidecar mismatch")
    records[shared.rel(manifest_path)] = shared.file_record(manifest_path)
    records[shared.rel(manifest_sidecar)] = shared.file_record(manifest_sidecar)
    inventory = manifest.get("files") or manifest.get("artifacts")
    if not isinstance(inventory, (dict, list)):
        raise RunnerError("V10 data manifest has no inventory")
    protocol = manifest.get("protocol")
    expected_protocol = {
        "tasks": [environment.removeprefix("dmc:") for environment, _ in ENVIRONMENTS],
        "splits": ["train", "val"],
        "train_episodes": COMMON["train_episodes"],
        "val_episodes": COMMON["val_episodes"],
        "length": COMMON["length"],
        "img_size": COMMON["img_size"],
        "train_seed": COMMON["train_rollout_seed"],
        "val_seed": COMMON["val_rollout_seed"],
        "smooth_rho": 0.85,
        "action_process": "bounded_tanh_ar1",
        "cache_role": "clean_only_corruptions_are_deterministic_dataset_views",
    }
    if not shared.stable_equal(protocol, expected_protocol):
        raise RunnerError("V10 data manifest protocol differs from frozen runner contract")
    actual = {path.resolve() for path in DATA_ROOT.iterdir()} if DATA_ROOT.is_dir() else set()
    wanted_paths = {path.resolve() for path in expected}
    if actual != wanted_paths:
        raise RunnerError(
            f"V10 data namespace is not exact; missing={sorted(map(str, wanted_paths-actual))[:8]}, "
            f"extra={sorted(map(str, actual-wanted_paths))[:8]}"
        )
    return dict(sorted(records.items()))


def precollect_data(python: str) -> None:
    command = [
        python, shared.rel(DATA_SCRIPT), "--root", shared.rel(DATA_ROOT), "--all",
        "--train-episodes", str(COMMON["train_episodes"]),
        "--val-episodes", str(COMMON["val_episodes"]),
        "--length", str(COMMON["length"]),
        "--img-size", str(COMMON["img_size"]),
        "--train-seed", str(COMMON["train_rollout_seed"]),
        "--val-seed", str(COMMON["val_rollout_seed"]),
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RunnerError(f"V10 data preparation failed with status {result.returncode}")
    data_snapshot()


def design_metadata(design: str) -> dict[str, Any]:
    if design not in DESIGNS:
        raise RunnerError(f"unknown V10 design {design!r}")
    variant = design.removeprefix("orbitv10_") if design != "orbitv10" else "orthogonal"
    return {
        "memory_arch_schema_version": 10 if design in ORBIT_DESIGNS else None,
        "memory_architecture": "orthogonal_recurrent_belief" if design in ORBIT_DESIGNS else design,
        "memory_v10_variant": variant if design in ORBIT_DESIGNS else None,
        "memory_internal_auxiliary": "none",
        "memory_teacher_present": False,
        "memory_fixed_horizon": False if design in ORBIT_DESIGNS else None,
        "encoder_trained_end_to_end": True,
        "encoder_ema_teacher_present": False,
        "target_stop_gradient": False,
        "training_objective": V10J_OBJECTIVE,
    }


def memory_contract() -> dict[str, Any]:
    from lewm.models.memory import (
        GRUMemory, ORBITv10Memory, SSMMemory, SharedActionShrinkageMemory,
    )

    from scripts.train_hacssm_v10 import _matched_gru_hidden

    # A=6 is a transparent reference count (the largest action width in the
    # development lineage), not a claim that all five new tasks have six controls.
    # Official models instantiate the native continuous action width sealed in each
    # environment's data bundle.
    dimension, action_dim = COMMON["embed_dim"], 6
    modes = {
        "orbitv10": "orthogonal",
        "orbitv10_noaction": "noaction",
        "orbitv10_additive": "additive",
        "orbitv10_scaled": "scaled",
        "orbitv10_static": "static",
    }
    orbit = {
        design: ORBITv10Memory(dimension, action_dim, mode=mode)
        for design, mode in modes.items()
    }
    signatures = {
        design: [[name, list(parameter.shape)] for name, parameter in model.named_parameters()]
        for design, model in orbit.items()
    }
    counts = {design: model.parameter_count() for design, model in orbit.items()}
    if len({json.dumps(value) for value in signatures.values()}) != 1:
        raise RunnerError("ORBIT controls do not share one parameter signature")
    if len(set(counts.values())) != 1:
        raise RunnerError(f"ORBIT controls are not parameter matched: {counts}")
    v8 = SharedActionShrinkageMemory(dimension, action_dim, mode="learned").parameter_count()
    ssm = sum(parameter.numel() for parameter in SSMMemory(dimension).parameters())
    gru_hidden = _matched_gru_hidden(dimension)
    gru = sum(
        parameter.numel()
        for parameter in GRUMemory(dimension, hidden=gru_hidden).parameters()
    )
    return {
        "embed_dim": dimension,
        "reference_action_dim": action_dim,
        "reference_count_scope": "D=128,A=6; official action_dim is native per-task cache metadata",
        "official_action_dimension": "per-task continuous control width",
        "memory_parameters": {**counts, "hacssmv8": v8, "ssm": ssm, "gru": gru},
        "gru_hidden": gru_hidden,
        "streaming_recurrent_floats": {
            **{design: dimension for design in ORBIT_DESIGNS},
            "hacssmv8": 2 * dimension, "ssm": dimension, "gru": gru_hidden,
        },
        "orbit_parameter_signature": signatures["orbitv10"],
        "parameter_matching_scope": "all five ORBIT modes",
    }


def build_protocol(
    commit: str, clean: bool, wandb_preflight: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": "ORBIT-v10-J/R1 adaptive end-to-end held-out confirmation",
        "study_id": WANDB_STUDY,
        "scope": SCOPE,
        "adaptive_revision": "R1",
        "predecessor_provenance": PREDECESSOR_PROVENANCE,
        "producer_git_commit": commit,
        "producer_git_clean": clean,
        "common_protocol": COMMON,
        "data_contract": {
            "tasks": [environment for environment, _ in ENVIRONMENTS],
            "unopened_task_claim": False,
            "adaptive_reuse_of_predecessor_data": True,
            "training_corruptions": list(TRAIN_CORRUPTIONS),
            "heldout_corruptions": list(HELDOUT_CORRUPTIONS),
            "deterministic_by_split_episode": True,
            "synchronized_clean_view_targets_used_for_training": True,
            "clean_view_target_kind": (
                "same-trajectory clean RGB encoded by the joint-gradient online encoder"
            ),
            "clean_view_target_gradient_active": True,
            "simulator_physics_state_used_for_training": False,
            "simulator_physics_state_role": "evaluation-only normalized ridge-probe target",
            "primary_metric": "heldout_state_nmse",
            "primary_definition": (
                "equal mean of per-checkpoint normalized physics-state probe MSE over "
                "freeze, gaussian_noise, checkerboard, and long_freeze at deep+first-post targets"
            ),
            "private_latent_mse_cross_model_comparison": False,
        },
        "architecture_contract": {
            "candidate": "orbitv10",
            "single_state": True,
            "transport": "action-conditioned exact block-orthogonal rotations",
            "correction": "causal learned innovation gate",
            "fixed_decay_or_horizon": False,
            "memory_auxiliary_or_teacher": False,
            "encoder_normalization": "causal affine-free per-frame LayerNorm",
            "encoder_ema_teacher": False,
            "target_stop_gradient": False,
            "clean_target_gradient_active": True,
            "training_objective": V10J_OBJECTIVE,
            "objective_weights": {
                "prediction": 1.0,
                "variance": 1.0,
                "covariance": 1.0,
            },
            "sigreg_role": "zero-weight diagnostic only",
            "variants": {
                "orbitv10_noaction": "identity memory transport; predictor actions retained",
                "orbitv10_additive": "V8-style additive action transport control",
                "orbitv10_scaled": "action rotation plus learned contraction/expansion control",
                "orbitv10_static": "static observation-correction gate control",
            },
        },
        "design_protocol": {design: design_metadata(design) for design in DESIGNS},
        "memory_contract": memory_contract(),
        "data_artifacts": data_snapshot(),
        "source_artifacts": source_snapshot(),
        "output_root": shared.rel(OUTPUT_ROOT),
        "log_root": shared.rel(LOG_ROOT),
        "wandb": wandb_preflight,
        "wandb_requirements": {
            "all_cells_online": True,
            "complete_epoch_history_per_cell": COMMON["epochs"],
            "evaluation_rollout_npz_table_video_per_cell": True,
        },
        "stages": {
            "pilot": {"seeds": list(PILOT_SEEDS), "designs": list(DESIGNS), "runs": 135},
            "completion": {
                "seeds": list(COMPLETION_SEEDS), "designs": list(DESIGNS), "runs": 90,
                "completed_total_runs": 225, "runs_regardless_of_pilot_screen": True,
            },
        },
        "pilot_success_criteria": {
            "vs_each_ssm_and_v8": ">=5%, >=9/15 cells, >=4/5 environments",
            "vs_each_additive_and_scaled": ">=2%, >=9/15, >=3/5",
            "vs_noaction": ">=5%, >=11/15, >=3/5",
            "vs_static": ">=1%, >=9/15, >=3/5",
            "clean_harm_vs_each_ssm_and_v8": "<=2%",
            "convergence": "absolute median <1%, p95 <3%, maximum <5%",
            "orbit_orthogonality_and_streaming": "each <=1e-5",
            "online_encoder_quality": (
                "finite variance >=1e-5, covariance effective rank >=16, singleton "
                "and prefix parity <=1e-5, clean state-probe ceiling <1"
            ),
        },
        "final_success_criteria": {
            "requires_pilot_pass": True,
            "vs_each_ssm_and_v8": ">=5%, >=15/25 cells, >=4/5 environments",
            "bootstrap_vs_each_ssm_and_v8": "crossed environment x seed 90% lower bound >0",
            "vs_each_additive_and_scaled": ">=2%, >=14/25, >=3/5",
            "vs_noaction": ">=5%, >=17/25, >=3/5",
            "vs_static": ">=1%, >=14/25, >=3/5",
            "clean_harm_vs_each_ssm_and_v8": "<=2%",
            "convergence": "absolute median <1%, p95 <3%, maximum <5%",
            "orbit_orthogonality_and_streaming": "each <=1e-5",
            "encoder_quality": (
                "online encoder finite; mean channel variance >=1e-5; covariance effective "
                "rank >=16; singleton and prefix parity <=1e-5; clean state-probe ceiling <1"
            ),
        },
        "analysis_gate": {
            "command": "scripts/analyze_hacssm_v10.py --phase pilot",
            "decision_file": shared.rel(PILOT_DECISION_PATH),
            "fail_closed_result": "NO_GO",
            "pilot_pass_result": "PILOT_CONFIRMATION_PASS",
        },
        "expected_runs": {
            "pilot": [job.run_name for job in PILOT_JOBS],
            "completion": [job.run_name for job in COMPLETION_JOBS],
        },
    }


def expected_args(job: shared.Job) -> dict[str, Any]:
    train_path, val_path, _manifest = data_paths(job.clean_env)
    return {
        "train_data": shared.rel(train_path),
        "val_data": shared.rel(val_path),
        "memory_mode": job.design,
        "seed": job.seed,
        "output_dir": shared.rel(OUTPUT_ROOT),
        "epochs": COMMON["epochs"],
        "batch_size": COMMON["batch_size"],
        "lr": COMMON["learning_rate"],
        "weight_decay": COMMON["weight_decay"],
        "num_workers": COMMON["train_dataloader_workers"],
        "img_size": COMMON["img_size"],
        "patch_size": 8,
        "embed_dim": COMMON["embed_dim"],
        "encoder_layers": COMMON["encoder_layers"],
        "encoder_heads": COMMON["encoder_heads"],
        "predictor_layers": COMMON["predictor_layers"],
        "predictor_heads": COMMON["predictor_heads"],
        "history_len": COMMON["history_len"],
        "dropout": 0.1,
        "sigreg_lambda": COMMON["sigreg_lambda"],
        "sigreg_projections": COMMON["sigreg_projections"],
        "probe_ridge": COMMON["state_probe_ridge"],
        "corruption_seed": COMMON["corruption_seed"],
        "no_amp": False,
        "wandb": True,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_mode": WANDB_MODE,
        "wandb_study": WANDB_STUDY,
        "extra_tag": "",
        "eval_rollout_episode": EVAL_ROLLOUT_EPISODE,
        "device": "cuda",
    }


def train_command(python: str, job: shared.Job) -> list[str]:
    args = expected_args(job)
    command = [python, shared.rel(TRAIN_SCRIPT)]
    for key, value in args.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
        else:
            command.extend((option, str(value)))
    return command


def validate_history(history: Any, job: shared.Job) -> None:
    if not isinstance(history, list) or len(history) != COMMON["epochs"]:
        raise RunnerError(f"{job.run_name}: expected 100 history rows")
    for epoch, record in enumerate(history, 1):
        if not isinstance(record, dict) or record.get("epoch") != epoch:
            raise RunnerError(f"{job.run_name}: malformed history epoch {epoch}")
        if set(record) != {"epoch", "train", "val"}:
            raise RunnerError(f"{job.run_name}: unexpected history fields")
        expected_metrics = {
            "loss", "pred_loss", "variance_loss", "covariance_loss", "sigreg_loss",
        }
        for split in ("train", "val"):
            values = record.get(split)
            if not isinstance(values, dict) or set(values) != expected_metrics:
                raise RunnerError(f"{job.run_name}: absent {split} history")
            for key in expected_metrics:
                value = values.get(key)
                if type(value) not in (int, float) or not math.isfinite(float(value)):
                    raise RunnerError(f"{job.run_name}: invalid {split}.{key} at {epoch}")
            for key in (
                "loss", "pred_loss", "variance_loss", "covariance_loss", "sigreg_loss",
            ):
                if float(values[key]) < 0.0:
                    raise RunnerError(f"{job.run_name}: negative {split}.{key} at {epoch}")
            optimized_sum = sum(
                float(values[key])
                for key in ("pred_loss", "variance_loss", "covariance_loss")
            )
            if not math.isclose(
                float(values["loss"]), optimized_sum, rel_tol=1e-5, abs_tol=1e-7
            ):
                raise RunnerError(
                    f"{job.run_name}: {split} loss is not the equal-weight V10-J sum at {epoch}"
                )
            if any(key.startswith("hier_") for key in values):
                raise RunnerError(f"{job.run_name}: forbidden hierarchy metric in history")
            shared.assert_finite_tree(values, f"{job.run_name}.{epoch}.{split}")


def validate_model_state(state: Any, job: shared.Job) -> None:
    import torch

    if not isinstance(state, dict) or not state:
        raise RunnerError(f"{job.run_name}: empty model state")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise RunnerError(f"{job.run_name}: malformed model tensor {name!r}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise RunnerError(f"{job.run_name}: non-finite tensor {name}")
    names = set(state)
    if not any(name.startswith("encoder.") for name in names):
        raise RunnerError(f"{job.run_name}: native encoder state is missing")
    expected_prefix = {
        "gru": "mem_gru.", "ssm": "mem_ssm.", "hacssmv8": "mem_hacssmv8.",
    }.get(job.design, "mem_orbitv10." if job.design in ORBIT_DESIGNS else None)
    if expected_prefix and not any(name.startswith(expected_prefix) for name in names):
        raise RunnerError(f"{job.run_name}: missing memory namespace {expected_prefix}")
    if job.design in ORBIT_DESIGNS and any(
        name.startswith(("mem_hacssmv8.", "mem_loifv9.")) for name in names
    ):
        raise RunnerError(f"{job.run_name}: ORBIT checkpoint contains foreign memory tensors")


def validate_probe(probe: Any, job: shared.Job, label: str) -> None:
    expected = {"x_mean", "x_std", "y_mean", "y_std", "weights"}
    if not isinstance(probe, dict) or set(probe) != expected:
        raise RunnerError(f"{job.run_name}: invalid {label} state-probe payload")
    for key, value in probe.items():
        array = np.asarray(value)
        if array.dtype.hasobject or array.size == 0 or not np.isfinite(array).all():
            raise RunnerError(f"{job.run_name}: invalid {label} state-probe array {key}")
        if key.endswith("_std") and np.any(array <= 0.0):
            raise RunnerError(f"{job.run_name}: non-positive {label} state-probe scale {key}")

def _finite_metric(metrics: Mapping[str, Any], key: str, job: shared.Job) -> float:
    value = metrics.get(key)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise RunnerError(f"{job.run_name}: missing/non-finite metric {key}")
    return float(value)


def validate_rollout(job: shared.Job, metrics: Mapping[str, Any]) -> None:
    if shared.sha256_file(job.eval_rollout_path) != metrics.get("eval_rollout_sha256"):
        raise RunnerError(f"{job.run_name}: rollout hash mismatch")
    try:
        with np.load(job.eval_rollout_path, allow_pickle=False) as rollout:
            required = {
                "schema_version", "episode_index", "conditions", "condition",
                "target_times", "phase", "state_target", "state_prediction", "state_nmse",
            }
            per_condition = {
                f"{condition}_{field}"
                for condition in HELDOUT_CORRUPTIONS
                for field in (
                    "target_times", "phase", "gap_start", "gap_end", "observed_rgb",
                    "clean_rgb", "actions", "physics_state_target",
                    "physics_state_prediction", "state_nmse_by_target_t",
                )
            }
            required |= per_condition
            if set(rollout.files) != required:
                raise RunnerError(
                    f"{job.eval_rollout_path}: rollout field mismatch; "
                    f"missing={sorted(required-set(rollout.files))}, "
                    f"extra={sorted(set(rollout.files)-required)}"
                )
            if int(rollout["schema_version"]) != 1:
                raise RunnerError(f"{job.eval_rollout_path}: schema mismatch")
            if int(rollout["episode_index"]) != EVAL_ROLLOUT_EPISODE:
                raise RunnerError(f"{job.eval_rollout_path}: episode mismatch")
            for name in rollout.files:
                value = np.asarray(rollout[name])
                if value.dtype.hasobject or value.size == 0:
                    raise RunnerError(f"{job.eval_rollout_path}: invalid {name}")
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    raise RunnerError(f"{job.eval_rollout_path}: non-finite {name}")
            conditions = tuple(np.asarray(rollout["conditions"]).astype(str).tolist())
            if conditions != HELDOUT_CORRUPTIONS:
                raise RunnerError(f"{job.eval_rollout_path}: held-out conditions mismatch")
            for condition in HELDOUT_CORRUPTIONS:
                target_times = np.asarray(rollout[f"{condition}_target_times"])
                phase = np.asarray(rollout[f"{condition}_phase"])
                errors = np.asarray(rollout[f"{condition}_state_nmse_by_target_t"])
                if target_times.shape != (COMMON["length"] - COMMON["history_len"],):
                    raise RunnerError(f"{job.eval_rollout_path}: {condition} target shape")
                if phase.shape != target_times.shape or errors.shape != target_times.shape:
                    raise RunnerError(f"{job.eval_rollout_path}: {condition} trace shape")
    except RunnerError:
        raise
    except Exception as exc:
        raise RunnerError(f"cannot validate {job.eval_rollout_path}: {exc}") from exc


def validate_tracking_receipt(job: shared.Job, metrics: Mapping[str, Any]) -> None:
    receipt = shared.read_json(job.wandb_run_path)
    if not isinstance(receipt, dict):
        raise RunnerError(f"{job.wandb_run_path}: receipt is not an object")
    run_id = receipt.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise RunnerError(f"{job.wandb_run_path}: invalid run id")
    wanted = {
        "schema_version": 1,
        "run_name": f"{WANDB_STUDY}-{job.run_name}",
        "url": f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}",
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "mode": WANDB_MODE,
        "study": WANDB_STUDY,
        "state": "finished",
        "eval_rollout_sha256": metrics.get("eval_rollout_sha256"),
        "eval_rollout_episode": EVAL_ROLLOUT_EPISODE,
    }
    for key, value in wanted.items():
        if not shared.stable_equal(receipt.get(key), value):
            raise RunnerError(f"{job.wandb_run_path}: {key} mismatch")
    artifact = receipt.get("eval_rollout_artifact_name")
    if not isinstance(artifact, str) or not artifact:
        raise RunnerError(f"{job.wandb_run_path}: missing rollout artifact name")
    if not list(job.run_dir.rglob("run-*.wandb")):
        raise RunnerError(f"{job.run_name}: W&B transaction file is absent")


def validate_job(job: shared.Job, *, allow_missing: bool) -> bool:
    required = (job.model_path, job.metrics_path, job.eval_rollout_path, job.wandb_run_path)
    present = [path.is_file() and path.stat().st_size > 0 for path in required]
    if any(present) and not all(present):
        raise RunnerError(f"partial V10 run: {job.run_dir}")
    if not all(present):
        if job.run_dir.exists():
            raise RunnerError(f"incomplete V10 run directory: {job.run_dir}")
        if allow_missing:
            return False
        raise RunnerError(f"missing V10 run: {job.run_dir}")
    metrics = shared.read_json(job.metrics_path)
    if not isinstance(metrics, dict):
        raise RunnerError(f"{job.metrics_path}: not an object")
    shared.assert_finite_tree(metrics, f"{job.run_name}.metrics")
    import torch
    try:
        checkpoint = torch.load(job.model_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RunnerError(f"cannot load {job.model_path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != {
        "model_state_dict", "args", "final_metrics", "history", "state_probe",
    }:
        raise RunnerError(f"{job.model_path}: unexpected checkpoint structure")
    if not shared.stable_equal(metrics, checkpoint["final_metrics"]):
        raise RunnerError(f"{job.run_name}: metrics/checkpoint mismatch")
    validate_probe(checkpoint["state_probe"], job, "online")
    if not shared.stable_equal(checkpoint["args"], expected_args(job)):
        actual = checkpoint.get("args")
        differing = sorted(
            key for key in set(actual or {}) | set(expected_args(job))
            if not isinstance(actual, dict)
            or key not in actual or key not in expected_args(job)
            or not shared.stable_equal(actual[key], expected_args(job)[key])
        )
        raise RunnerError(f"{job.run_name}: checkpoint args differ at {differing[:12]}")
    validate_model_state(checkpoint["model_state_dict"], job)
    validate_history(checkpoint["history"], job)
    required_metrics = (
        "heldout_state_nmse", "clean_state_nmse", "final_val_loss", "val_pred_loss",
        "probe_ceiling_state_nmse", "probe_ceiling_r2",
        *ENCODER_RECEIPT_BASES,
        "convergence_relative_change", "trainable_parameters",
        *(f"{condition}_state_nmse" for condition in HELDOUT_CORRUPTIONS),
        *(f"{condition}_predicted_state_r2" for condition in HELDOUT_CORRUPTIONS),
        *(
            f"{condition}_state_nmse_{phase}"
            for condition in HELDOUT_CORRUPTIONS
            for phase in ("deep", "first_post", "post")
        ),
    )
    for key in required_metrics:
        _finite_metric(metrics, key, job)
    from scripts.hacssm_v10_data import load_cache
    train_path, val_path, _manifest = data_paths(job.clean_env)
    train_metadata = load_cache(train_path, verify=True)
    val_metadata = load_cache(val_path, verify=True)
    exact_metadata = {
        "schema_version": 1,
        "env": job.clean_env,
        "design": job.design,
        "seed": job.seed,
        "epochs": COMMON["epochs"],
        "headline_metric": "heldout_state_nmse",
        "train_data": str(train_path.resolve()),
        "val_data": str(val_path.resolve()),
        "train_data_sha256": train_metadata.file_sha256,
        "val_data_sha256": val_metadata.file_sha256,
        "train_data_content_sha256": train_metadata.content_sha256,
        "val_data_content_sha256": val_metadata.content_sha256,
        "train_episodes": COMMON["train_episodes"],
        "val_episodes": COMMON["val_episodes"],
        "length": COMMON["length"],
        "action_dim": train_metadata.action_dim,
        "state_dim": train_metadata.state_dim,
        "probe_ridge": COMMON["state_probe_ridge"],
        "probe_fit_split": "clean_train_online_joint_only",
        "training_objective": V10J_OBJECTIVE,
    }
    for key, wanted in exact_metadata.items():
        if not shared.stable_equal(metrics.get(key), wanted):
            raise RunnerError(
                f"{job.run_name}: metric {key}={metrics.get(key)!r}, expected {wanted!r}"
            )
    condition_mean = sum(
        float(metrics[f"{condition}_state_nmse"]) for condition in HELDOUT_CORRUPTIONS
    ) / len(HELDOUT_CORRUPTIONS)
    if not math.isclose(
        float(metrics["heldout_state_nmse"]), condition_mean, rel_tol=1e-6, abs_tol=1e-8
    ):
        raise RunnerError(f"{job.run_name}: heldout_state_nmse is not an equal mean")
    if metrics.get("encoder_norm") != "causal" or metrics.get("predictor_norm") != "none":
        raise RunnerError(f"{job.run_name}: noncausal encoder/predictor normalization")
    if (
        metrics.get("encoder_type") != "vit"
        or metrics.get("encoder_frozen") is not False
        or metrics.get("end_to_end_rgb") is not True
        or metrics.get("clean_target_gradient_active") is not True
        or metrics.get("ema_target_active") is not False
        or metrics.get("target_stop_gradient") is not False
        or metrics.get("vicreg_gradient_active") is not True
        or metrics.get("sigreg_gradient_active") is not False
    ):
        raise RunnerError(f"{job.run_name}: not native joint-gradient V10-J LeWM")
    forbidden_metrics = {
        "ema_schedule", "ema_optimizer_steps", "ema_final_momentum",
        "teacher_student_parameter_mse", "student_probe_fit_split",
        "student_probe_ceiling_state_nmse", "student_probe_ceiling_r2",
    }
    forbidden_metrics.update(
        key for key in metrics if key.startswith("student_encoder_")
    )
    present_forbidden = sorted(forbidden_metrics & set(metrics))
    if present_forbidden:
        raise RunnerError(
            f"{job.run_name}: forbidden EMA/student metrics {present_forbidden}"
        )
    for key in ("prediction_loss_weight", "variance_loss_weight", "covariance_loss_weight"):
        if metrics.get(key) != 1.0:
            raise RunnerError(f"{job.run_name}: unequal V10-J objective weight {key}")
    if metrics.get("sigreg_lambda") != 0.1 or metrics.get("sigreg_optimization_weight") != 0.0:
        raise RunnerError(f"{job.run_name}: SIGReg diagnostic-only contract mismatch")
    variance = _finite_metric(metrics, "encoder_mean_channel_variance", job)
    effective_rank = _finite_metric(metrics, "encoder_covariance_effective_rank", job)
    singleton = _finite_metric(metrics, "encoder_singleton_max_abs", job)
    prefix_error = _finite_metric(metrics, "encoder_prefix_max_abs", job)
    probe_ceiling = _finite_metric(metrics, "probe_ceiling_state_nmse", job)
    if variance < 1e-5:
        raise RunnerError(f"{job.run_name}: collapsed online encoder variance")
    if effective_rank < 16.0:
        raise RunnerError(f"{job.run_name}: low-rank online encoder")
    if not 0.0 <= singleton <= 1e-5 or not 0.0 <= prefix_error <= 1e-5:
        raise RunnerError(f"{job.run_name}: noncausal online encoder receipt")
    if not 0.0 <= probe_ceiling < 1.0:
        raise RunnerError(f"{job.run_name}: failed online probe ceiling")
    if job.design in ORBIT_DESIGNS:
        for key in ("orbit_orthogonality_error_max", "orbit_streaming_max_abs"):
            if _finite_metric(metrics, key, job) < 0.0:
                raise RunnerError(f"{job.run_name}: negative {key}")
    if int(metrics["trainable_parameters"]) <= 0:
        raise RunnerError(f"{job.run_name}: invalid parameter count")
    last_val = checkpoint["history"][-1]["val"]
    if not shared.stable_equal(metrics["final_val_loss"], last_val.get("loss")):
        raise RunnerError(f"{job.run_name}: final val loss mismatch")
    if not shared.stable_equal(metrics["val_pred_loss"], last_val.get("pred_loss")):
        raise RunnerError(f"{job.run_name}: final prediction loss mismatch")
    previous = sum(
        float(record["val"]["pred_loss"])
        for record in checkpoint["history"][-20:-10]
    ) / 10.0
    recent = sum(
        float(record["val"]["pred_loss"])
        for record in checkpoint["history"][-10:]
    ) / 10.0
    if previous <= 0.0:
        raise RunnerError(f"{job.run_name}: non-positive convergence denominator")
    convergence = (previous - recent) / previous
    if not math.isclose(
        float(metrics["convergence_relative_change"]), convergence,
        rel_tol=1e-6, abs_tol=1e-8,
    ):
        raise RunnerError(f"{job.run_name}: convergence receipt mismatch")
    validate_rollout(job, metrics)
    validate_tracking_receipt(job, metrics)
    return True


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
        "semantics": "heldout-corruption normalized physics-state evaluation trace",
        **design_metadata(job.design),
    }


def verify_wandb_cloud(jobs: Sequence[shared.Job]) -> dict[str, Any]:
    """Verify V10's 100-row history, 180x4 table, 256x132 video, and artifact."""
    import wandb

    expected: dict[str, dict[str, Any]] = {}
    for job in jobs:
        receipt = shared.read_json(job.wandb_run_path)
        run_id = receipt.get("run_id") if isinstance(receipt, dict) else None
        if not isinstance(run_id, str) or not run_id:
            raise RunnerError(f"{job.wandb_run_path}: missing W&B run id")
        if run_id in expected:
            raise RunnerError(f"duplicate W&B run id {run_id}")
        rollout_sha = str(receipt["eval_rollout_sha256"])
        expected[run_id] = {
            "url": str(receipt["url"]),
            "artifact_name": str(receipt["eval_rollout_artifact_name"]),
            "artifact_metadata": expected_wandb_artifact_metadata(job, rollout_sha),
        }

    def inspect(run_id: str) -> str | None:
        try:
            api = wandb.Api(timeout=30)
            run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
            wanted = expected[run_id]
            if run.state != "finished" or run.url != wanted["url"]:
                return f"state/url mismatch: {run.state}/{run.url}"
            epochs = [
                row.get("epoch") for row in run.scan_history(keys=["epoch"])
                if row.get("epoch") is not None
            ]
            if epochs != list(range(1, COMMON["epochs"] + 1)):
                return f"epoch history mismatch count={len(epochs)}"
            try:
                table = dict(run.summary.get("eval/rollout_trace"))
                video = dict(run.summary.get("eval/paired_rollout"))
            except (TypeError, ValueError):
                return "missing rollout table/video summary"
            if not (
                table.get("_type") == "table-file"
                and table.get("nrows") == len(HELDOUT_CORRUPTIONS) * (
                    COMMON["length"] - COMMON["history_len"]
                )
                and table.get("ncols") == 4
                and isinstance(table.get("size"), int) and table["size"] > 0
                and isinstance(table.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", table["sha256"])
            ):
                return f"malformed 180x4 rollout table: {table}"
            if not (
                video.get("_type") == "video-file"
                and video.get("height") == len(HELDOUT_CORRUPTIONS) * COMMON["img_size"]
                and video.get("width") == 2 * COMMON["img_size"] + 4
                and isinstance(video.get("size"), int) and video["size"] > 0
                and isinstance(video.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", video["sha256"])
            ):
                return f"malformed 256x132 rollout video: {video}"
            artifacts = [
                artifact for artifact in run.logged_artifacts()
                if artifact.type == "evaluation-rollout"
                and artifact.name.split(":", 1)[0] == wanted["artifact_name"]
            ]
            if len(artifacts) != 1:
                return f"expected one rollout artifact, found {len(artifacts)}"
            artifact = artifacts[0]
            if not shared.stable_equal(artifact.metadata, wanted["artifact_metadata"]):
                return f"artifact metadata mismatch: {artifact.metadata}"
            if set(artifact.manifest.entries) != {"eval_rollout.npz"}:
                return f"artifact entries mismatch: {sorted(artifact.manifest.entries)}"
            entry = artifact.manifest.entries["eval_rollout.npz"]
            if not isinstance(entry.size, int) or entry.size <= 0:
                return "empty rollout artifact entry"
            return None
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    last_problem = "no verification attempt"
    for attempt in range(1, 7):
        try:
            api = wandb.Api(timeout=30)
            runs = api.runs(
                f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                filters={"config.wandb_study": WANDB_STUDY},
            )
            observed = {run.id for run in runs if run.id in expected}
            missing = sorted(set(expected) - observed)
            problems: dict[str, str] = {}
            if not missing:
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    futures = {executor.submit(inspect, run_id): run_id for run_id in expected}
                    for future in concurrent.futures.as_completed(futures):
                        problem = future.result()
                        if problem is not None:
                            problems[futures[future]] = problem
            if not missing and not problems:
                record = {
                    "entity": WANDB_ENTITY,
                    "project": WANDB_PROJECT,
                    "study": WANDB_STUDY,
                    "verified_finished_runs": len(expected),
                    "verified_complete_epoch_histories": len(expected),
                    "verified_rollout_artifacts": len(expected),
                    "verified_rollout_tables": len(expected),
                    "verified_rollout_videos": len(expected),
                    "rollout_table_shape": [180, 4],
                    "rollout_video_shape": [256, 132],
                    "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
                shared.status(f"W&B cloud verified all {len(expected)} V10 runs")
                return record
            last_problem = f"missing={missing[:8]}, problems={list(problems.items())[:4]}"
        except Exception as exc:
            last_problem = f"{type(exc).__name__}: {exc}"
        if attempt < 6:
            time.sleep(5)
    raise RunnerError(f"V10 W&B cloud verification failed: {last_problem}")


def read_pilot_decision() -> tuple[bool, dict[str, Any]]:
    decision = shared.read_json(PILOT_DECISION_PATH)
    if not isinstance(decision, dict) or type(decision.get("pilot_screen_passed")) is not bool:
        raise RunnerError("pilot decision lacks boolean pilot_screen_passed")
    expected = "PILOT_CONFIRMATION_PASS" if decision["pilot_screen_passed"] else "NO_GO"
    if decision.get("decision") != expected:
        raise RunnerError("pilot decision label/boolean conflict")
    if decision.get("scope") != SCOPE:
        raise RunnerError("pilot decision scope mismatch")
    shared.assert_finite_tree(decision, "pilot_decision")
    return bool(decision["pilot_screen_passed"]), decision


def validate_final_decision(final: Any, pilot_passed: bool) -> bool:
    if not isinstance(final, dict):
        raise RunnerError("final decision is not an object")
    confirmed = final.get("end_to_end_confirmation_passed")
    if (
        final.get("decision") not in {
            "END_TO_END_CONFIRMATION_PASS", "NO_GO", "PILOT_NO_GO_FINAL_DESCRIPTIVE"
        }
        or final.get("completed_runs") != 225
        or final.get("pilot_screen_passed") is not pilot_passed
        or type(confirmed) is not bool
        or final.get("scope") != SCOPE
    ):
        raise RunnerError("invalid final decision contract")
    if confirmed and (not pilot_passed or final["decision"] != "END_TO_END_CONFIRMATION_PASS"):
        raise RunnerError("confirmation label conflicts with gates")
    if not pilot_passed and final["decision"] != "PILOT_NO_GO_FINAL_DESCRIPTIVE":
        raise RunnerError("failed pilot was reopened")
    return confirmed


def write_final_manifest(
    protocol: dict[str, Any], pilot: dict[str, Any], pilot_passed: bool,
    gpu_ids: Sequence[str], workers: int, cloud: dict[str, Any], final: dict[str, Any],
) -> None:
    shared.reject_temporary_artifacts()
    for job in ALL_JOBS:
        validate_job(job, allow_missing=False)
    manifest = {
        "schema_version": 1,
        "study": protocol["study"],
        "study_id": WANDB_STUDY,
        "scope": SCOPE,
        "adaptive_revision": "R1",
        "predecessor_provenance": PREDECESSOR_PROVENANCE,
        "producer_git_commit": protocol["producer_git_commit"],
        "producer_git_clean": True,
        "completed_runs": 225,
        "expected_runs": 225,
        "all_requested_runs_completed": True,
        "pilot_screen_passed": pilot_passed,
        "pilot_decision": pilot,
        "final_decision": final,
        "execution": {"gpu_ids": list(gpu_ids), "workers": workers},
        "protocol": {shared.rel(PROTOCOL_PATH): shared.file_record(PROTOCOL_PATH)},
        "data_artifacts": protocol["data_artifacts"],
        "source_artifacts": protocol["source_artifacts"],
        "wandb": protocol["wandb"],
        "wandb_cloud_verification": cloud,
        "wandb_runs": {job.run_name: shared.read_json(job.wandb_run_path) for job in ALL_JOBS},
        "output_artifacts": shared.output_file_snapshot(),
        "log_artifacts": shared.log_file_snapshot(),
    }
    shared.atomic_write_json(MANIFEST_PATH, manifest)
    digest = shared.sha256_file(MANIFEST_PATH)
    shared.atomic_write_bytes(
        MANIFEST_SHA_PATH, f"{digest}  {MANIFEST_PATH.name}\n".encode()
    )


def configure_shared() -> None:
    assignments = {
        "TRAIN_SCRIPT": TRAIN_SCRIPT,
        "ANALYZE_SCRIPT": ANALYZE_SCRIPT,
        "FEATURE_ROOT": DATA_ROOT,
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
        "V5_DESIGNS": ORBIT_DESIGNS,
        "HIER_DESIGNS": frozenset(),
        "NO_AUX_DESIGNS": frozenset(DESIGNS),
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
    shared.feature_snapshot = data_snapshot
    shared.eval_rollout_snapshot = data_snapshot
    shared.source_snapshot = source_snapshot
    shared.memory_contract = memory_contract
    shared.build_protocol = build_protocol
    shared.expected_args = expected_args
    shared.validate_history = validate_history
    shared.validate_model_state = validate_model_state
    shared.validate_job = validate_job
    shared.train_command = train_command
    shared.expected_wandb_artifact_metadata = expected_wandb_artifact_metadata
    shared.read_pilot_decision = read_pilot_decision


def verify_provenance_unchanged(protocol: Mapping[str, Any]) -> None:
    if not shared.stable_equal(shared.read_json(PROTOCOL_PATH), protocol):
        raise RunnerError("protocol changed after publication")
    if not shared.stable_equal(source_snapshot(), protocol["source_artifacts"]):
        raise RunnerError("source changed during V10")
    if not shared.stable_equal(data_snapshot(), protocol["data_artifacts"]):
        raise RunnerError("data changed during V10")
    commit, porcelain = shared.git_provenance()
    if commit != protocol["producer_git_commit"] or porcelain:
        raise RunnerError("Git provenance changed during V10")


def check_command_interfaces(python: str) -> None:
    checks = (
        (TRAIN_SCRIPT, (
            "--train-data", "--val-data", "--memory-mode", "--wandb-study",
            "--eval-rollout-episode", *DESIGNS,
        )),
        (DATA_SCRIPT, ("--root", "--all", "--train-episodes", "--val-episodes")),
        (ANALYZE_SCRIPT, ("--phase", "pilot", "final")),
    )
    for script, required in checks:
        if not script.is_file():
            raise RunnerError(f"required V10 script is missing: {script}")
        result = subprocess.run(
            [python, str(script), "--help"], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise RunnerError(f"cannot inspect {script}: {result.stderr[-1200:]}")
        output = result.stdout + result.stderr
        missing = [token for token in required if token not in output]
        if missing:
            raise RunnerError(f"{script}: interface lacks {missing}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen 225-cell ORBIT-v10-R1 adaptive study."
    )
    parser.add_argument("--python", default=str(ROOT / ".venv" / "bin" / "python"))
    parser.add_argument(
        "--gpus", type=shared.parse_gpu_ids, default=shared.parse_gpu_ids("0,1,2,3")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_shared()
    args = build_parser().parse_args(argv)
    if args.workers != 4 or len(args.gpus) != 4:
        raise RunnerError("V10-R1 is frozen to exactly four workers on four GPU ids")
    commit, porcelain = shared.git_provenance()
    clean = not porcelain
    if not args.dry_run and not clean:
        raise RunnerError(
            "launch requires a clean committed worktree: "
            + " | ".join(porcelain.splitlines()[:8])
        )
    if args.dry_run and not clean:
        shared.status("DRY RUN NOTE: launch remains disabled until Git is clean")
    shared.check_python(args.python)
    check_command_interfaces(args.python)

    lock_stream = None
    if not args.dry_run:
        lock_stream = shared.acquire_lock()
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        precollect_data(args.python)
    try:
        # A dry run is strictly read-only and therefore requires pre-existing data.
        data_snapshot()
        wandb_preflight = shared.check_wandb_online(args.python)
        protocol = build_protocol(commit, clean, wandb_preflight)
        shared.establish_protocol(protocol, args.dry_run)
        shared.reject_temporary_artifacts()
        completed = shared.validate_artifact_space(ALL_JOBS)
        shared.status(
            f"preflight validated {len(completed)}/225; "
            f"pilot={sum(job.run_name in completed for job in PILOT_JOBS)}/135"
        )
        if args.dry_run:
            digest = hashlib.sha256(
                json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            shared.status(f"DRY RUN: no writes/launches; protocol digest={digest}")
            return 0

        shared.check_gpus(args.python, args.gpus)
        verify_provenance_unchanged(protocol)
        shared.run_stage(args.python, PILOT_JOBS, args.gpus, args.workers)
        for job in PILOT_JOBS:
            validate_job(job, allow_missing=False)
        verify_provenance_unchanged(protocol)
        shared.status("running immutable ORBIT-v10-R1 pilot analyzer")
        shared.run_analyzer(args.python, "pilot")
        pilot_passed, pilot = read_pilot_decision()
        shared.status(
            f"V10-R1 pilot={pilot['decision']} passed={pilot_passed}; completion is mandatory"
        )

        shared.run_stage(args.python, COMPLETION_JOBS, args.gpus, args.workers)
        for job in ALL_JOBS:
            validate_job(job, allow_missing=False)
        verify_provenance_unchanged(protocol)
        cloud = verify_wandb_cloud(ALL_JOBS)
        shared.status("running final ORBIT-v10-R1 analyzer")
        shared.run_analyzer(args.python, "final")
        final = shared.read_json(FINAL_DECISION_PATH)
        confirmed = validate_final_decision(final, pilot_passed)
        verify_provenance_unchanged(protocol)
        shared.validate_artifact_space(ALL_JOBS)
        write_final_manifest(
            protocol, pilot, pilot_passed, args.gpus, args.workers, cloud, final
        )
        shared.status(
            f"ORBIT-v10-R1 complete: 225/225; end_to_end_confirmation_passed={confirmed}"
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
        print(f"ORBIT-v10-R1 runner error: {exc}", file=sys.stderr)
        raise SystemExit(2)
