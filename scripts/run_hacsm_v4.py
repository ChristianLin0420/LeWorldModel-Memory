#!/usr/bin/env python3
"""Run the decision-gated HACSM-v4 shared-feature experiment.

This runner intentionally owns orchestration and provenance only.  It reuses the
immutable DINO-PCA feature bundles from the SMT-v3 study and invokes
``train_popgym.py`` once per independently trained factorial cell.

Stage A (pilot)
    5 environments x 9 designs x seeds 0,1,2 = 135 runs.

Stage B (only when ``pilot_decision.json["expand"]`` is exactly ``true``)
    * the same 9 designs x seeds 3,4 = 90 runs, and
    * GRU/SMT-v1/SMT-v2 x all 5 seeds = 75 runs.

The expanded grid therefore contains 300 runs.  Existing runs are never trusted
from filenames alone: both artifacts, every scientific argument, the complete
200-epoch history, final-metric equality, finiteness, and feature hashes are
validated before a checkpoint may be skipped.
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
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_popgym.py"
ANALYZE_SCRIPT = REPO_ROOT / "scripts" / "analyze_hacsm_v4.py"
FEATURE_ROOT = REPO_ROOT / "outputs" / "smt_v3_shared" / "dino_features_d128"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "hacsm_v4_shared"
LOG_ROOT = REPO_ROOT / "logs" / "hacsm_v4_shared"
DATA_ROOT = REPO_ROOT / "outputs" / "popgym_data"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
DECISION_PATH = OUTPUT_ROOT / "pilot_decision.json"
MANIFEST_PATH = OUTPUT_ROOT / "hacsm_v4_manifest.json"
MANIFEST_SHA_PATH = OUTPUT_ROOT / "hacsm_v4_manifest.sha256"
LOCK_PATH = OUTPUT_ROOT / ".run_hacsm_v4.lock"


ENVIRONMENTS = (
    ("dmc:reacher.hard.occ", "dmc:reacher.hard"),
    ("dmc:ball_in_cup.catch.occ", "dmc:ball_in_cup.catch"),
    ("dmc:finger.spin.occ", "dmc:finger.spin"),
    ("dmc:cheetah.run.occ", "dmc:cheetah.run"),
    ("ogbench:cube-single.occ", "ogbench:cube-single"),
)
PILOT_DESIGNS = (
    "none",
    "multi",
    "ssm",
    "smtv3",
    "hacsmv4_static",
    "hacsmv4_noaction",
    "hacsmv4_noaux",
    "hacsmv4_single",
    "hacsmv4",
)
EXPANSION_BASELINES = ("gru", "smtv1", "smtv2")
PILOT_SEEDS = (0, 1, 2)
EXPANSION_SEEDS = (3, 4)
ALL_SEEDS = (0, 1, 2, 3, 4)

# These values are scientific protocol, not convenience defaults.  Changing one
# requires a new output namespace and a new runner/protocol schema.
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
    "hier_loss_weight": 0.1,
    "epochs": 200,
    "train_dataloader_workers": 2,
    "prototype_seed": 0,
    "train_rollout_seed": 0,
    "val_rollout_seed": 7777,
    "smt_router": "sigmoid",
    "fixed_alpha": True,
    "wandb": False,
}

SOURCE_FILES = (
    Path("scripts/run_hacsm_v4.py"),
    Path("scripts/train_popgym.py"),
    Path("scripts/analyze_hacsm_v4.py"),
    Path("lewm/data.py"),
    Path("lewm/models/encoder.py"),
    Path("lewm/models/leworldmodel.py"),
    Path("lewm/models/memory.py"),
    Path("lewm/models/memory_model.py"),
    Path("lewm/models/sigreg.py"),
)

_ACTIVE_PROCESSES: set[subprocess.Popen[Any]] = set()
_PROCESS_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


@dataclass(frozen=True, order=True)
class Job:
    stage: str
    seed: int
    occ_env: str
    clean_env: str
    design: str

    @property
    def run_name(self) -> str:
        return f"lewm-{self.occ_env}-{self.design}-s{self.seed}"

    @property
    def run_dir(self) -> Path:
        return OUTPUT_ROOT / self.run_name

    @property
    def model_path(self) -> Path:
        return self.run_dir / "model.pt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def log_path(self) -> Path:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", self.run_name).strip("_")
        return LOG_ROOT / f"{safe}.log"


def make_jobs(
    stage: str, designs: Sequence[str], seeds: Sequence[int]
) -> tuple[Job, ...]:
    # Seed-major ordering is part of the fixed worker sharding.
    return tuple(
        Job(stage, seed, occ, clean, design)
        for seed in seeds
        for occ, clean in ENVIRONMENTS
        for design in designs
    )


PILOT_JOBS = make_jobs("pilot", PILOT_DESIGNS, PILOT_SEEDS)
EXPANSION_MAIN_JOBS = make_jobs("expansion", PILOT_DESIGNS, EXPANSION_SEEDS)
EXPANSION_BASELINE_JOBS = make_jobs("expansion", EXPANSION_BASELINES, ALL_SEEDS)
EXPANSION_JOBS = EXPANSION_MAIN_JOBS + EXPANSION_BASELINE_JOBS
ALL_JOBS = PILOT_JOBS + EXPANSION_JOBS

assert len(PILOT_JOBS) == 135
assert len(EXPANSION_MAIN_JOBS) == 90
assert len(EXPANSION_BASELINE_JOBS) == 75
assert len(ALL_JOBS) == 300
assert len({job.run_name for job in ALL_JOBS}) == len(ALL_JOBS)


class RunnerError(RuntimeError):
    """A protocol or artifact invariant was violated."""


def rel(path: Path) -> str:
    """Return a stable repository-relative path."""
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RunnerError(f"required nonempty file is missing: {path}")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def reject_non_rfc_json(token: str) -> None:
    raise ValueError(f"non-RFC JSON constant {token}")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(), parse_constant=reject_non_rfc_json)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RunnerError(f"invalid JSON at {path}: {exc}") from exc


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    if temporary.exists():
        raise RunnerError(f"refusing to reuse temporary file: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_write_bytes(path, payload)


def stable_equal(left: Any, right: Any) -> bool:
    """Strict recursive equality (in particular, bool is not equal to int)."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            stable_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            stable_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def assert_finite_tree(value: Any, context: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_tree(child, f"{context}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RunnerError(f"non-finite value at {context}: {value!r}")


def safe_env(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def feature_paths(clean_env: str) -> tuple[Path, Path, Path]:
    prefix = FEATURE_ROOT / safe_env(clean_env)
    return (
        Path(f"{prefix}_train.npz"),
        Path(f"{prefix}_val.npz"),
        Path(f"{prefix}_manifest.json"),
    )


def feature_snapshot() -> dict[str, dict[str, Any]]:
    if not FEATURE_ROOT.is_dir():
        raise RunnerError(f"fixed feature root is missing: {FEATURE_ROOT}")
    expected: set[Path] = set()
    records: dict[str, dict[str, Any]] = {}
    for occ_env, clean_env in ENVIRONMENTS:
        train_path, val_path, manifest_path = feature_paths(clean_env)
        expected.update((train_path, val_path, manifest_path))
        manifest = read_json(manifest_path)
        config = manifest.get("config") if isinstance(manifest, dict) else None
        exact_config = {
            "occ_env": occ_env,
            "clean_env": clean_env,
            "train_episodes": COMMON["train_episodes"],
            "val_episodes": COMMON["val_episodes"],
            "length": COMMON["length"],
            "feature_dim": COMMON["feature_dim"],
            "feature_schema_version": 1,
            "prototype_seed": COMMON["prototype_seed"],
            "train_rollout_seed": COMMON["train_rollout_seed"],
            "val_rollout_seed": COMMON["val_rollout_seed"],
        }
        if not isinstance(config, dict):
            raise RunnerError(f"feature manifest has no config object: {manifest_path}")
        for key, wanted in exact_config.items():
            if config.get(key) != wanted:
                raise RunnerError(
                    f"{manifest_path}: config.{key}={config.get(key)!r}, expected {wanted!r}"
                )
        artifacts = manifest.get("artifact_files")
        if artifacts != {"train": train_path.name, "val": val_path.name}:
            raise RunnerError(f"{manifest_path}: artifact_files does not match fixed bundle")
        for path in (train_path, val_path, manifest_path):
            records[rel(path)] = file_record(path)

    actual = {path.resolve() for path in FEATURE_ROOT.iterdir() if path.is_file()}
    expected_resolved = {path.resolve() for path in expected}
    if actual != expected_resolved:
        missing = sorted(str(path) for path in expected_resolved - actual)
        extra = sorted(str(path) for path in actual - expected_resolved)
        raise RunnerError(f"feature bundle is not exact; missing={missing}, extra={extra}")
    return dict(sorted(records.items()))


def source_snapshot() -> dict[str, dict[str, Any]]:
    return {
        source.as_posix(): file_record(REPO_ROOT / source)
        for source in SOURCE_FILES
    }


def build_protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": "HACSM-v4 shared-target causal-normalization study",
        "common_protocol": COMMON,
        "output_root": rel(OUTPUT_ROOT),
        "log_root": rel(LOG_ROOT),
        "feature_root": rel(FEATURE_ROOT),
        "feature_artifacts": feature_snapshot(),
        "source_artifacts": source_snapshot(),
        "environments": [
            {"occluded": occ, "clean_target": clean} for occ, clean in ENVIRONMENTS
        ],
        "stages": {
            "pilot": {
                "designs": list(PILOT_DESIGNS),
                "seeds": list(PILOT_SEEDS),
                "runs": len(PILOT_JOBS),
            },
            "expansion_if_pilot_expand": {
                "main_designs": list(PILOT_DESIGNS),
                "main_seeds": list(EXPANSION_SEEDS),
                "baseline_designs": list(EXPANSION_BASELINES),
                "baseline_seeds": list(ALL_SEEDS),
                "runs": len(EXPANSION_JOBS),
                "expanded_total_runs": len(ALL_JOBS),
            },
        },
        "analysis_gate": {
            "command": "scripts/analyze_hacsm_v4.py --phase pilot",
            "decision_file": rel(DECISION_PATH),
            "required_field": {"expand": "boolean"},
            "if_false": "terminal pilot; publish 135-run manifest",
            "if_true": (
                "run 165 expansion cells, then scripts/analyze_hacsm_v4.py --phase final"
            ),
        },
        "expected_runs": {
            "pilot": [job.run_name for job in PILOT_JOBS],
            "expansion": [job.run_name for job in EXPANSION_JOBS],
        },
    }


def establish_protocol(protocol: dict[str, Any], dry_run: bool) -> None:
    if PROTOCOL_PATH.exists():
        existing = read_json(PROTOCOL_PATH)
        if not stable_equal(existing, protocol):
            raise RunnerError(
                f"{PROTOCOL_PATH} differs from current feature/source/protocol snapshot; "
                "refusing to mix cohorts"
            )
        return
    if dry_run:
        return
    # A grid without its immutable protocol cannot be safely resumed.
    prior_entries = [
        path for path in OUTPUT_ROOT.rglob("*")
        if path.resolve() != LOCK_PATH.resolve()
    ]
    if LOG_ROOT.exists():
        prior_entries.extend(LOG_ROOT.rglob("*"))
    if prior_entries:
        raise RunnerError(
            f"output/log entries exist without {PROTOCOL_PATH}: {prior_entries[:5]}"
        )
    atomic_write_json(PROTOCOL_PATH, protocol)


def expected_args(job: Job) -> dict[str, Any]:
    train_path, val_path, manifest_path = feature_paths(job.clean_env)
    router = "softmax" if job.design == "smtv1" else "sigmoid"
    return {
        "env_id": job.occ_env,
        "memory_mode": job.design,
        "smt_router": router,
        "seed": job.seed,
        "output_dir": rel(OUTPUT_ROOT),
        "num_episodes": COMMON["train_episodes"],
        "val_episodes": COMMON["val_episodes"],
        "data_dir": rel(DATA_ROOT),
        "prototype_seed": COMMON["prototype_seed"],
        "target_env_id": job.clean_env,
        "mask_occluded_target_loss": True,
        "first_post_loss_weight": COMMON["first_post_loss_weight"],
        "encoder_checkpoint": None,
        "encoder_stats": None,
        "freeze_encoder": False,
        "encoder_type": "precomputed",
        "train_feature_cache": rel(train_path),
        "val_feature_cache": rel(val_path),
        "feature_manifest": rel(manifest_path),
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
        "hier_loss_weight": COMMON["hier_loss_weight"],
        "tau_fast": 3.0,
        "tau_slow": 25.0,
        "fixed_alpha": True,
        "wandb": False,
        "wandb_project": "lewm-memory-popgym",
        "extra_tag": "",
        "device": "cuda",
        "feature_manifest_sha256": sha256_file(manifest_path),
    }


def expected_metric_metadata(job: Job) -> dict[str, Any]:
    _, _, manifest_path = feature_paths(job.clean_env)
    effective_hier = (
        0.0
        if job.design == "hacsmv4_noaux" or not job.design.startswith("hacsmv4")
        else COMMON["hier_loss_weight"]
    )
    return {
        "env": job.occ_env,
        "design": job.design,
        "n_actions": 6,
        "prototype_seed": 0,
        "dataset_schema_version": 3,
        "feature_schema_version": 1,
        "feature_manifest": rel(manifest_path),
        "feature_manifest_sha256": sha256_file(manifest_path),
        "target_env": job.clean_env,
        "masked_clean_blackout_loss": True,
        "first_post_loss_weight": COMMON["first_post_loss_weight"],
        "hier_loss_weight": COMMON["hier_loss_weight"],
        "hier_loss_weight_effective": effective_hier,
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
    }


def validate_history(history: Any, job: Job) -> None:
    if not isinstance(history, list) or len(history) != COMMON["epochs"]:
        length = len(history) if isinstance(history, list) else None
        raise RunnerError(
            f"{job.run_name}: history length {length}, expected {COMMON['epochs']}"
        )
    for epoch, record in enumerate(history, 1):
        if not isinstance(record, dict) or record.get("epoch") != epoch:
            raise RunnerError(f"{job.run_name}: malformed history epoch {epoch}")
        if set(record) != {"epoch", "train", "val"}:
            raise RunnerError(f"{job.run_name}: unexpected history fields at epoch {epoch}")
        for split in ("train", "val"):
            values = record.get(split)
            if not isinstance(values, dict):
                raise RunnerError(f"{job.run_name}: missing {split} history at epoch {epoch}")
            for key in ("loss", "pred_loss", "sigreg_loss"):
                value = values.get(key)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise RunnerError(
                        f"{job.run_name}: invalid {split}.{key} at epoch {epoch}: {value!r}"
                    )
            assert_finite_tree(values, f"{job.run_name}.history[{epoch}].{split}")


def validate_model_state(state: Any, job: Job) -> None:
    # Importing torch lazily keeps --help and protocol-only inspection lightweight.
    import torch

    if not isinstance(state, dict) or not state:
        raise RunnerError(f"{job.run_name}: empty/non-dictionary model_state_dict")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise RunnerError(f"{job.run_name}: malformed model state entry {name!r}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise RunnerError(f"{job.run_name}: non-finite model tensor {name}")


def validate_job(job: Job, *, allow_missing: bool) -> bool:
    model_exists = job.model_path.is_file() and job.model_path.stat().st_size > 0
    metrics_exists = job.metrics_path.is_file() and job.metrics_path.stat().st_size > 0
    if model_exists != metrics_exists:
        raise RunnerError(f"partial run artifacts: {job.run_dir}")
    if not model_exists:
        if job.model_path.exists() or job.metrics_path.exists() or job.run_dir.exists():
            raise RunnerError(f"empty or incomplete run directory: {job.run_dir}")
        if allow_missing:
            return False
        raise RunnerError(f"missing required run: {job.run_dir}")

    metrics = read_json(job.metrics_path)
    if not isinstance(metrics, dict):
        raise RunnerError(f"{job.metrics_path}: expected a JSON object")
    assert_finite_tree(metrics, f"{job.run_name}.metrics")

    import torch

    try:
        checkpoint = torch.load(job.model_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # torch has several serialization exception classes
        raise RunnerError(f"cannot load checkpoint {job.model_path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise RunnerError(f"{job.model_path}: checkpoint is not a dictionary")
    validate_model_state(checkpoint.get("model_state_dict"), job)
    history = checkpoint.get("history")
    validate_history(history, job)
    final_metrics = checkpoint.get("final_metrics")
    if not stable_equal(metrics, final_metrics):
        raise RunnerError(f"{job.run_name}: metrics.json != checkpoint final_metrics")

    wanted_args = expected_args(job)
    actual_args = checkpoint.get("args")
    if not stable_equal(actual_args, wanted_args):
        if isinstance(actual_args, dict):
            differing = sorted(
                key for key in set(actual_args) | set(wanted_args)
                if key not in actual_args
                or key not in wanted_args
                or not stable_equal(actual_args[key], wanted_args[key])
            )
        else:
            differing = ["<args is not a dictionary>"]
        raise RunnerError(f"{job.run_name}: checkpoint args differ at {differing[:12]}")

    for key, wanted in expected_metric_metadata(job).items():
        if key not in metrics or not stable_equal(metrics[key], wanted):
            raise RunnerError(
                f"{job.run_name}: metric {key}={metrics.get(key)!r}, expected {wanted!r}"
            )
    required_finite = (
        "val_pred_loss",
        "infl_fast",
        "infl_slow",
        "clean_mse_deep_blackout",
        "clean_mse_deep_blackout_ablated",
        "clean_mse_first_post",
        "clean_mse_first_post_ablated",
        "constant_mse_first_post",
        "last_visible_mse_first_post",
        "clean_input_mse_first_post",
    )
    for key in required_finite:
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise RunnerError(f"{job.run_name}: invalid metric {key}={value!r}")
    last_val = history[-1]["val"]
    if not stable_equal(metrics["val_pred_loss"], last_val.get("pred_loss")):
        raise RunnerError(f"{job.run_name}: val_pred_loss differs from final history")
    if job.design.startswith("hacsmv4"):
        for key in ("val_hier_loss", "val_hier_loss_fast", "val_hier_loss_medium", "val_hier_loss_slow"):
            value = metrics.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise RunnerError(f"{job.run_name}: missing/invalid HACSM metric {key}")
        if not stable_equal(metrics["val_hier_loss"], last_val.get("hier_loss")):
            raise RunnerError(f"{job.run_name}: val_hier_loss differs from final history")
    return True


def validate_artifact_space(jobs: Sequence[Job]) -> set[str]:
    expected_names = {job.run_name for job in ALL_JOBS}
    actual_names = {
        path.name for path in OUTPUT_ROOT.glob("lewm-*") if path.is_dir()
    } if OUTPUT_ROOT.is_dir() else set()
    unexpected_names = actual_names - expected_names
    if unexpected_names:
        raise RunnerError(f"unexpected run directories: {sorted(unexpected_names)[:8]}")

    expected_models = {job.model_path.resolve() for job in ALL_JOBS}
    expected_metrics = {job.metrics_path.resolve() for job in ALL_JOBS}
    actual_models = {path.resolve() for path in OUTPUT_ROOT.rglob("model.pt")} if OUTPUT_ROOT.exists() else set()
    actual_metrics = {path.resolve() for path in OUTPUT_ROOT.rglob("metrics.json")} if OUTPUT_ROOT.exists() else set()
    if actual_models - expected_models or actual_metrics - expected_metrics:
        raise RunnerError(
            "unexpected checkpoint artifacts: "
            f"models={sorted(map(str, actual_models - expected_models))[:4]}, "
            f"metrics={sorted(map(str, actual_metrics - expected_metrics))[:4]}"
        )

    expected_training_logs = {job.log_path.resolve() for job in ALL_JOBS}
    actual_training_logs = {
        path.resolve() for path in LOG_ROOT.glob("lewm_*.log")
    } if LOG_ROOT.exists() else set()
    if actual_training_logs - expected_training_logs:
        raise RunnerError(
            f"unexpected training logs: {sorted(map(str, actual_training_logs - expected_training_logs))[:8]}"
        )

    completed: set[str] = set()
    for job in jobs:
        is_complete = validate_job(job, allow_missing=True)
        if job.log_path.exists() and (
            not job.log_path.is_file() or job.log_path.stat().st_size <= 0
        ):
            raise RunnerError(f"empty/non-file training log: {job.log_path}")
        if not is_complete and job.log_path.exists():
            raise RunnerError(
                f"training log exists without a complete run: {job.log_path}"
            )
        if is_complete:
            completed.add(job.run_name)
    return completed


def train_command(python: str, job: Job) -> list[str]:
    train_path, val_path, manifest_path = feature_paths(job.clean_env)
    return [
        python,
        rel(TRAIN_SCRIPT),
        "--env-id", job.occ_env,
        "--target-env-id", job.clean_env,
        "--mask-occluded-target-loss",
        "--memory-mode", job.design,
        "--smt-router", "sigmoid",
        "--seed", str(job.seed),
        "--fixed-alpha",
        "--encoder-type", "precomputed",
        "--train-feature-cache", rel(train_path),
        "--val-feature-cache", rel(val_path),
        "--feature-manifest", rel(manifest_path),
        "--prototype-seed", "0",
        "--data-dir", rel(DATA_ROOT),
        "--output-dir", rel(OUTPUT_ROOT),
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
        "--predictor-norm", COMMON["predictor_norm"],
        "--history-len", "3",
        "--dropout", "0.1",
        "--sigreg-lambda", "0.1",
        "--sigreg-projections", "512",
        "--hier-loss-weight", "0.1",
        "--tau-fast", "3.0",
        "--tau-slow", "25.0",
        "--first-post-loss-weight", "0.5",
        "--device", "cuda",
        "--no-wandb",
    ]


def timestamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def status(message: str) -> None:
    with _PRINT_LOCK:
        print(f"{timestamp()} {message}", flush=True)


def run_logged_process(command: Sequence[str], log_path: Path, env: dict[str, str]) -> int:
    if log_path.exists():
        raise RunnerError(f"refusing to overwrite existing log for an incomplete run: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        with _PROCESS_LOCK:
            _ACTIVE_PROCESSES.add(process)
        try:
            return process.wait()
        finally:
            with _PROCESS_LOCK:
                _ACTIVE_PROCESSES.discard(process)


def terminate_active_processes() -> None:
    with _PROCESS_LOCK:
        active = list(_ACTIVE_PROCESSES)
    for process in active:
        if process.poll() is None:
            process.terminate()
    for process in active:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def run_stage(
    python: str,
    jobs: Sequence[Job],
    gpu_ids: Sequence[str],
    workers: int,
) -> None:
    # Refuse stale evidence before any worker starts.  Without this all other shards
    # could begin while one shard discovers a partial prior attempt.
    for job in jobs:
        complete = validate_job(job, allow_missing=True)
        if not complete and job.log_path.exists():
            raise RunnerError(
                f"stale log exists for missing run {job.run_name}: {job.log_path}"
            )
    # Shard the full fixed list before skip decisions, so resume never changes which
    # worker/GPU owns a cell.
    shards = [tuple(jobs[slot::workers]) for slot in range(workers)]
    stop = threading.Event()

    def worker(slot: int) -> None:
        gpu = gpu_ids[slot % len(gpu_ids)]
        for job in shards[slot]:
            if stop.is_set():
                return
            if validate_job(job, allow_missing=True):
                status(f"[worker {slot} gpu {gpu}] skip validated {job.run_name}")
                continue
            if job.log_path.exists():
                raise RunnerError(
                    f"stale log exists for missing run {job.run_name}: {job.log_path}"
                )
            status(f"[worker {slot} gpu {gpu}] >>> {job.run_name}")
            child_env = os.environ.copy()
            child_env.update({"CUDA_VISIBLE_DEVICES": gpu, "MUJOCO_GL": "egl"})
            return_code = run_logged_process(
                train_command(python, job), job.log_path, child_env
            )
            if return_code != 0:
                stop.set()
                raise RunnerError(
                    f"training failed with status {return_code}: {job.run_name}; "
                    f"see {job.log_path}"
                )
            if not validate_job(job, allow_missing=False):  # pragma: no cover - always true
                raise AssertionError("required validation unexpectedly returned false")
            status(f"[worker {slot} gpu {gpu}] <<< {job.run_name}")

    errors: list[BaseException] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, slot) for slot in range(workers)]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except BaseException as exc:
                stop.set()
                errors.append(exc)
    if errors:
        details = "; ".join(str(error) for error in errors[:4])
        raise RunnerError(f"stage failed in {len(errors)} worker(s): {details}") from errors[0]


def replaceable_analysis_log(phase: str) -> tuple[Path, Path]:
    final = LOG_ROOT / f"analyze_{phase}.log"
    temporary = LOG_ROOT / f".analyze_{phase}.{os.getpid()}.tmp"
    if temporary.exists():
        raise RunnerError(f"stale analysis temporary log: {temporary}")
    return temporary, final


def run_analyzer(python: str, phase: str) -> None:
    temporary, final = replaceable_analysis_log(phase)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        rel(ANALYZE_SCRIPT),
        "--root", rel(OUTPUT_ROOT),
        "--phase", phase,
    ]
    with temporary.open("xb") as log:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    os.replace(temporary, final)
    if result.returncode != 0:
        raise RunnerError(
            f"{phase} analyzer failed with status {result.returncode}; see {final}"
        )


def read_pilot_decision() -> tuple[bool, dict[str, Any]]:
    decision = read_json(DECISION_PATH)
    if not isinstance(decision, dict) or type(decision.get("expand")) is not bool:
        raise RunnerError(
            f"{DECISION_PATH} must be an object with top-level boolean 'expand'"
        )
    assert_finite_tree(decision, "pilot_decision")
    return decision["expand"], decision


def check_command_interfaces(python: str) -> None:
    if not TRAIN_SCRIPT.is_file():
        raise RunnerError(f"training script is missing: {TRAIN_SCRIPT}")
    if not ANALYZE_SCRIPT.is_file():
        raise RunnerError(f"analysis script is missing: {ANALYZE_SCRIPT}")
    for script, required in (
        (
            TRAIN_SCRIPT,
            (
                "--predictor-norm",
                "--hier-loss-weight",
                "--first-post-loss-weight",
                *PILOT_DESIGNS,
                *EXPANSION_BASELINES,
            ),
        ),
        (ANALYZE_SCRIPT, ("--phase", "pilot", "final")),
    ):
        result = subprocess.run(
            [python, str(script), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RunnerError(
                f"{script} --help failed ({result.returncode}): {result.stderr[-1000:]}"
            )
        help_text = result.stdout + result.stderr
        absent = [token for token in required if token not in help_text]
        if absent:
            raise RunnerError(f"{script} --help is missing required tokens: {absent}")


def check_python(python: str) -> None:
    result = subprocess.run(
        [python, "-c", "import torch; print(torch.__version__)"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RunnerError(f"Python/torch preflight failed: {result.stderr.strip()}")


def check_gpus(python: str, gpu_ids: Sequence[str]) -> None:
    for gpu in dict.fromkeys(gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        result = subprocess.run(
            [
                python,
                "-c",
                "import torch; assert torch.cuda.is_available(); "
                "assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RunnerError(
                f"GPU preflight failed for CUDA_VISIBLE_DEVICES={gpu!r}: "
                f"{result.stderr.strip()}"
            )
        status(f"GPU {gpu}: {result.stdout.strip()}")


def verify_provenance_unchanged(protocol: dict[str, Any]) -> None:
    if not stable_equal(read_json(PROTOCOL_PATH), protocol):
        raise RunnerError("protocol.json changed after protocol creation")
    if not stable_equal(source_snapshot(), protocol["source_artifacts"]):
        raise RunnerError("producer/analyzer source files changed after protocol creation")
    if not stable_equal(feature_snapshot(), protocol["feature_artifacts"]):
        raise RunnerError("fixed feature artifacts changed after protocol creation")


def reject_temporary_artifacts() -> None:
    offenders = []
    for root in (OUTPUT_ROOT, LOG_ROOT):
        if root.exists():
            offenders.extend(
                path for path in root.rglob("*")
                if path.is_file() and (path.name.endswith(".tmp") or path.name.startswith(".tmp"))
            )
    if offenders:
        raise RunnerError(f"temporary/partial files remain: {offenders[:8]}")


def output_file_snapshot() -> dict[str, dict[str, Any]]:
    excluded = {
        LOCK_PATH.resolve(),
        MANIFEST_PATH.resolve(),
        MANIFEST_SHA_PATH.resolve(),
    }
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(OUTPUT_ROOT.rglob("*")):
        if path.is_file() and path.resolve() not in excluded:
            records[rel(path)] = file_record(path)
    return records


def log_file_snapshot() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if LOG_ROOT.exists():
        for path in sorted(LOG_ROOT.rglob("*")):
            if path.is_file():
                records[rel(path)] = file_record(path)
    return records


def write_final_manifest(
    protocol: dict[str, Any], decision: dict[str, Any], expanded: bool,
    gpu_ids: Sequence[str], workers: int,
) -> None:
    reject_temporary_artifacts()
    required_jobs = ALL_JOBS if expanded else PILOT_JOBS
    for job in required_jobs:
        validate_job(job, allow_missing=False)
    manifest = {
        "schema_version": 1,
        "study": protocol["study"],
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "expanded": expanded,
        "completed_runs": len(required_jobs),
        "expected_runs": 300 if expanded else 135,
        "pilot_decision": decision,
        "execution": {"gpu_ids": list(gpu_ids), "workers": workers},
        "protocol": {rel(PROTOCOL_PATH): file_record(PROTOCOL_PATH)},
        "feature_artifacts": protocol["feature_artifacts"],
        "source_artifacts": protocol["source_artifacts"],
        "output_artifacts": output_file_snapshot(),
        "log_artifacts": log_file_snapshot(),
    }
    atomic_write_json(MANIFEST_PATH, manifest)
    manifest_sha = sha256_file(MANIFEST_PATH)
    atomic_write_bytes(MANIFEST_SHA_PATH, f"{manifest_sha}  {MANIFEST_PATH.name}\n".encode())
    if sha256_file(MANIFEST_PATH) != manifest_sha:
        raise RunnerError("manifest changed immediately after atomic publication")


def parse_gpu_ids(raw: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one GPU id is required")
    for value in values:
        if any(char.isspace() for char in value) or "," in value or "=" in value:
            raise argparse.ArgumentTypeError(f"invalid CUDA device token: {value!r}")
    return values


def acquire_lock() -> Any:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stream = LOCK_PATH.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RunnerError(f"another HACSM-v4 runner holds {LOCK_PATH}") from exc
    return stream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed, decision-gated 300-cell HACSM-v4 experiment."
    )
    parser.add_argument(
        "--python", default=str(REPO_ROOT / ".venv" / "bin" / "python"),
        help="Python executable used for training and analysis",
    )
    parser.add_argument(
        "--gpus", type=parse_gpu_ids, default=parse_gpu_ids("0,1,2,3"),
        help="comma-separated physical GPU ids (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="fixed orchestration shards, assigned round-robin to --gpus (default: 8)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="read-only interface/hash/artifact audit; launch no training or analysis",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise RunnerError("--workers must be positive")
    check_python(args.python)
    check_command_interfaces(args.python)
    protocol = build_protocol()

    lock_stream = None
    if not args.dry_run:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        lock_stream = acquire_lock()
    try:
        establish_protocol(protocol, args.dry_run)
        reject_temporary_artifacts()
        completed = validate_artifact_space(ALL_JOBS)
        status(
            f"preflight validated {len(completed)}/300 possible runs; "
            f"pilot={sum(job.run_name in completed for job in PILOT_JOBS)}/135"
        )
        if args.dry_run:
            protocol_digest = hashlib.sha256(
                json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            status(
                "DRY RUN: no files written and no experiments launched; "
                f"protocol content digest={protocol_digest}"
            )
            return 0

        check_gpus(args.python, args.gpus)
        verify_provenance_unchanged(protocol)
        run_stage(args.python, PILOT_JOBS, args.gpus, args.workers)
        for job in PILOT_JOBS:
            validate_job(job, allow_missing=False)

        verify_provenance_unchanged(protocol)
        status("running pilot analyzer")
        run_analyzer(args.python, "pilot")
        expanded, decision = read_pilot_decision()
        status(f"pilot decision: expand={expanded}")

        if expanded:
            run_stage(args.python, EXPANSION_JOBS, args.gpus, args.workers)
            for job in ALL_JOBS:
                validate_job(job, allow_missing=False)
            verify_provenance_unchanged(protocol)
            status("running final analyzer")
            run_analyzer(args.python, "final")
        else:
            # A negative decision may never silently coexist with previously launched
            # expansion cells.  The prospective pilot decision is terminal in this
            # branch; the final analyzer is defined over the 300-cell expanded grid.
            expansion_complete = [
                job.run_name for job in EXPANSION_JOBS
                if validate_job(job, allow_missing=True)
            ]
            if expansion_complete:
                raise RunnerError(
                    "pilot declined expansion but expansion artifacts exist: "
                    f"{expansion_complete[:8]}"
                )

        verify_provenance_unchanged(protocol)
        write_final_manifest(protocol, decision, expanded, args.gpus, args.workers)
        count = 300 if expanded else 135
        status(f"HACSM-v4 study complete: {count}/{count} validated")
        return 0
    finally:
        if lock_stream is not None:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
            lock_stream.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        terminate_active_processes()
        print("interrupted; active child processes terminated", file=sys.stderr)
        raise SystemExit(130)
    except RunnerError as exc:
        terminate_active_processes()
        print(f"HACSM-v4 runner error: {exc}", file=sys.stderr)
        raise SystemExit(2)
