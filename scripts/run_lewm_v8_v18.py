#!/usr/bin/env python3
"""Run the frozen 200-cell LeWM+V8 unopened-task confirmation grid.

Five fresh DMC tasks, eight matched memory designs, and five optimizer seeds
are trained for 100 epochs.  Four GPUs are used through a frozen assignment;
GPU 0 runs the acrobot and stacker queues serially.  The robust V17 cell
runner supplies atomic ledgers, resume, artifact checks, and online W&B
receipts, while this module replaces its four-task orchestration contract.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import hacssm_v18_data as data
import scripts.run_autovisreg_v17 as base
import scripts.train_lewm_v8_v18 as trainer


TASKS = data.TASKS
MEMORY_VARIANTS = trainer.MEMORIES
DESIGNS = trainer.DESIGNS
SEEDS = (18_001, 18_002, 18_003, 18_004, 18_005)
EPOCHS = 100
DEFAULT_STUDY = "lewm-v8-v18-confirmation"
DEFAULT_OUTPUT_ROOT = Path("outputs/lewm_v8_v18_confirmation")
DEFAULT_LOG_ROOT = Path("logs/lewm_v8_v18_confirmation")
DATA_ROOT = Path(data.DEFAULT_ROOT)

# One task is deliberately queued after another on GPU 0.  No task migrates
# across devices, which keeps the execution receipt deterministic.
TASK_QUEUES = (
    ("acrobot.swingup", "stacker.stack_4"),
    ("manipulator.bring_ball",),
    ("quadruped.run",),
    ("swimmer.swimmer15",),
)

SOURCE_PATHS = (
    Path("docs/V18_LEWM_V8_CONFIRMATION.md"),
    Path("lewm/__init__.py"),
    Path("lewm/models/__init__.py"),
    Path("lewm/models/leworldmodel.py"),
    Path("lewm/models/memory.py"),
    Path("lewm/models/memory_model.py"),
    Path("lewm/models/encoder.py"),
    Path("lewm/models/sigreg.py"),
    Path("lewm/models/siro.py"),
    Path("lewm/models/cf_hiro.py"),
    Path("lewm/models/cf_ebo.py"),
    Path("lewm/models/cvpf.py"),
    Path("scripts/hacssm_v10_data.py"),
    Path("scripts/hacssm_v11_data.py"),
    Path("scripts/hacssm_v18_data.py"),
    Path("scripts/train_hacssm_v10.py"),
    Path("scripts/train_hacssm_v11.py"),
    Path("scripts/train_subjepa_v16.py"),
    Path("scripts/train_lewm_v8_v18.py"),
    Path("scripts/run_autovisreg_v17.py"),
    Path("scripts/run_lewm_v8_v18.py"),
    Path("scripts/analyze_lewm_v8_v18.py"),
)

REQUIRED_FINITE_METRICS = (
    "heldout_prior_state_nmse",
    "clean_prior_state_nmse",
    "initial_encoder_integrator_probe_nmse",
    "encoder_mean_channel_variance",
    "encoder_covariance_effective_rank",
    "predictive_loss_convergence_relative_change",
    "val_predictive_loss",
)

_BASE_CONTRACT_FIELDS = (
    "TASKS", "OBJECTIVE_FAMILIES", "MEMORY_VARIANTS", "DESIGNS", "SEEDS",
    "EPOCHS", "DEFAULT_STUDY", "DEFAULT_OUTPUT_ROOT", "DEFAULT_LOG_ROOT",
    "DATA_ROOT", "DEFAULT_TRAIN_EPISODES", "DEFAULT_VAL_EPISODES",
    "DEFAULT_LENGTH", "DEFAULT_IMG_SIZE", "DEFAULT_TRAIN_SEED",
    "DEFAULT_VAL_SEED", "PROTOCOL_NAME", "RUNS_NAME", "ATTEMPTS_NAME",
    "SUMMARY_NAME", "LOCK_NAME", "SOURCE_PATHS", "REQUIRED_FINITE_METRICS",
    "design_parts", "train_command", "validate_inputs", "protocol_payload",
    "validate_core_artifacts",
)
_BASE_CONTRACT_ORIGINALS = {
    name: getattr(base, name) for name in _BASE_CONTRACT_FIELDS}


class ProtocolIntegrityGuard:
    """Revalidate the frozen source and cache identity throughout execution."""

    def __init__(self, source: Mapping[str, str], cohort: Mapping[str, Any]):
        self.source = dict(source)
        self.manifest = dict(cohort.get("__manifest__", {}))
        self.cohort = {
            str(task): dict(record) for task, record in cohort.items()
            if task != "__manifest__"}
        self._data_stats = {
            (task, role): self._stat(self._path(record[role]))
            for task, record in self.cohort.items()
            for role in ("train", "val")
        }
        self._manifest_stats = {
            role: self._stat(self._path(self.manifest[role]))
            for role in ("path", "sidecar")}

    @staticmethod
    def _path(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    @staticmethod
    def _stat(path: Path) -> tuple[int, int, int, int]:
        value = path.stat()
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)

    def assert_sources(self) -> None:
        for label, expected in self.source.items():
            path = self._path(label)
            actual = base.file_sha256(path)
            if actual != expected:
                raise base.ArtifactError(
                    f"frozen V18 source changed: {label}: {actual} != {expected}")

    def assert_task_data(self, task: str, *, full_hash: bool) -> None:
        if task not in self.cohort:
            raise base.ArtifactError(f"task {task!r} is absent from V18 protocol data")
        record = self.cohort[task]
        for role in ("train", "val"):
            path = self._path(record[role])
            if self._stat(path) != self._data_stats[(task, role)]:
                raise base.ArtifactError(
                    f"frozen V18 {task} {role} cache metadata changed")
            if full_hash:
                actual = base.file_sha256(path)
                expected = record[f"{role}_sha256"]
                if actual != expected:
                    raise base.ArtifactError(
                        f"frozen V18 {task} {role} cache changed: "
                        f"{actual} != {expected}")

    def assert_manifest(self, *, full_hash: bool) -> None:
        for role in ("path", "sidecar"):
            path = self._path(self.manifest[role])
            if self._stat(path) != self._manifest_stats[role]:
                raise base.ArtifactError(f"frozen V18 cohort manifest {role} changed")
            if full_hash:
                actual = base.file_sha256(path)
                expected = self.manifest[f"{role}_sha256"]
                if actual != expected:
                    raise base.ArtifactError(
                        f"frozen V18 cohort manifest {role} hash changed")

    def assert_before_cell(self, task: str) -> None:
        self.assert_sources()
        self.assert_manifest(full_hash=False)
        self.assert_task_data(task, full_hash=False)

    def assert_all(self) -> None:
        self.assert_sources()
        self.assert_manifest(full_hash=True)
        for task in TASKS:
            self.assert_task_data(task, full_hash=True)


_ACTIVE_INTEGRITY_GUARD: ProtocolIntegrityGuard | None = None


def design_parts(design: str) -> tuple[str, str]:
    _, memory, _ = trainer.parse_design(design)
    return "vicreg", memory


def train_command(
        python: str, output_root: Path, study: str, epochs: int,
        task: str, design: str, seed: int, *, wandb: bool) -> list[str]:
    if _ACTIVE_INTEGRITY_GUARD is not None:
        _ACTIVE_INTEGRITY_GUARD.assert_before_cell(task)
    train_data, val_data = base.data_paths(task)
    return [
        python,
        str(ROOT / "scripts" / "train_lewm_v8_v18.py"),
        "--train-data", str(train_data),
        "--val-data", str(val_data),
        "--design", design,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--output-dir", str(output_root),
        "--batch-size", "64",
        "--lr", "0.0003",
        "--weight-decay", "0.00001",
        "--num-workers", "2",
        "--img-size", "64",
        "--patch-size", "8",
        "--embed-dim", "128",
        "--encoder-layers", "6",
        "--encoder-heads", "4",
        "--predictor-layers", "4",
        "--predictor-heads", "8",
        "--history-len", "3",
        "--dropout", "0.1",
        "--sigreg-lambda", "0.0",
        "--sigreg-projections", "512",
        "--probe-ridge", "0.001",
        "--eval-target-key", "task_observation",
        "--corruption-seed", str(data.DEFAULT_CORRUPTION_SEED),
        "--eval-rollout-episode", "0",
        "--device", "cuda",
        "--wandb" if wandb else "--no-wandb",
        "--wandb-entity", base.WANDB_ENTITY,
        "--wandb-project", base.WANDB_PROJECT,
        "--wandb-mode", "online",
        "--wandb-study", study,
        "--extra-tag", "confirmation-grid,lewm-v8-v18,unopened-tasks",
    ]


def validate_inputs() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    missing = [path for path in SOURCE_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing V18 confirmation sources: {missing}")
    if trainer.DESIGNS != DESIGNS:
        raise RuntimeError("V18 runner and trainer design registries differ")
    if tuple(task for queue in TASK_QUEUES for task in queue) != (
            TASKS[0], TASKS[3], TASKS[1], TASKS[2], TASKS[4]):
        raise RuntimeError("V18 task queues differ from the frozen task registry")

    cohort: dict[str, dict[str, str]] = {}
    for task in TASKS:
        train_path, val_path = base.data_paths(task)
        for path in (train_path, val_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing V18 confirmation cache {path}")
        train = data.load_cache(train_path)
        val = data.load_cache(val_path)
        if (train.env_id, val.env_id) != (task, task):
            raise RuntimeError(f"V18 cache task mismatch for {task}")
        if (train.seed, val.seed) != (
                data.DEFAULT_TRAIN_SEED, data.DEFAULT_VAL_SEED):
            raise RuntimeError(f"V18 cache seed mismatch for {task}")
        if (train.episodes, val.episodes) != (
                data.DEFAULT_TRAIN_EPISODES, data.DEFAULT_VAL_EPISODES):
            raise RuntimeError(f"V18 cache episode count mismatch for {task}")
        if train.smooth_rho != 0.0 or val.smooth_rho != 0.0:
            raise RuntimeError(f"V18 cache action process mismatch for {task}")
        cohort[task] = {
            "train": str(train_path),
            "train_sha256": data.sha256_file(train_path),
            "train_content_sha256": train.content_sha256,
            "val": str(val_path),
            "val_sha256": data.sha256_file(val_path),
            "val_content_sha256": val.content_sha256,
        }
    data_root = DATA_ROOT if DATA_ROOT.is_absolute() else ROOT / DATA_ROOT
    manifest_path = data_root / "manifest.json"
    sidecar_path = data_root / "manifest.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(
            "V18 requires the complete write-once cohort manifest and sidecar")
    manifest_hash = base.file_sha256(manifest_path)
    if sidecar_path.read_text(encoding="utf-8") != (
            f"{manifest_hash}  {manifest_path.name}\n"):
        raise base.ArtifactError("V18 cohort manifest sidecar differs")
    manifest = base.load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise base.ArtifactError("V18 cohort manifest must be an object")
    if manifest.get("schema_version") != data.v11.SCHEMA_VERSION:
        raise base.ArtifactError("V18 cohort manifest schema version differs")
    expected_collection = {
        "study": DEFAULT_STUDY,
        "scope": "prospectively_frozen_unopened_task_confirmation",
        "tasks": list(TASKS),
        "splits": ["train", "val"],
        "train_episodes": data.DEFAULT_TRAIN_EPISODES,
        "val_episodes": data.DEFAULT_VAL_EPISODES,
        "length": data.DEFAULT_LENGTH,
        "img_size": data.DEFAULT_IMG_SIZE,
        "train_seed": data.DEFAULT_TRAIN_SEED,
        "val_seed": data.DEFAULT_VAL_SEED,
        "smooth_rho": 0.0,
        "action_process": data.ACTION_PROCESS,
        "primary_evaluation_target": "flattened_native_task_observation",
        "secondary_evaluation_target": "raw_physics_state",
        "evaluation_targets_used_for_training": False,
        "cache_role": "clean_only_corruptions_are_deterministic_dataset_views",
        "corruption_seed": data.DEFAULT_CORRUPTION_SEED,
    }
    if manifest.get("protocol") != expected_collection:
        raise base.ArtifactError("V18 cohort manifest collection protocol differs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2 * len(TASKS):
        raise base.ArtifactError("V18 cohort manifest must contain exactly ten caches")
    indexed_artifacts = {
        str(Path(str(record.get("path", ""))).resolve()): record
        for record in artifacts if isinstance(record, Mapping)}
    if len(indexed_artifacts) != 2 * len(TASKS):
        raise base.ArtifactError("V18 cohort manifest contains malformed/duplicate caches")
    for task, record in cohort.items():
        for role in ("train", "val"):
            path = Path(record[role])
            path = path if path.is_absolute() else ROOT / path
            artifact = indexed_artifacts.get(str(path.resolve()))
            expected_artifact = {
                "path": str(path.resolve()),
                "sha256": record[f"{role}_sha256"],
                "content_sha256": record[f"{role}_content_sha256"],
                "bytes": path.stat().st_size,
            }
            if artifact != expected_artifact:
                raise base.ArtifactError(
                    f"V18 cohort manifest artifact differs for {task}/{role}")
    cohort["__manifest__"] = {
        "path": str(manifest_path.resolve()),
        "path_sha256": manifest_hash,
        "sidecar": str(sidecar_path.resolve()),
        "sidecar_sha256": base.file_sha256(sidecar_path),
    }
    source = {
        str(path): base.file_sha256(ROOT / path) for path in SOURCE_PATHS}
    return cohort, source


def _task_gpu_map(gpu_ids: Sequence[str]) -> dict[str, str]:
    if len(gpu_ids) != len(TASK_QUEUES):
        raise ValueError("V18 requires exactly four GPU identifiers")
    return {
        task: gpu
        for gpu, queue in zip(gpu_ids, TASK_QUEUES, strict=True)
        for task in queue
    }


def protocol_payload(
        *, python: str, output_root: Path, log_root: Path, study: str,
        epochs: int, gpu_ids: Sequence[str], wandb: bool,
        data: Mapping[str, Any], source: Mapping[str, str]) -> dict[str, Any]:
    commands = base.command_records(
        python, output_root, study, epochs, wandb=wandb)
    task_gpu = _task_gpu_map(gpu_ids)
    git = base.git_receipt()
    if not git.get("git_worktree_clean") or not git.get("git_commit"):
        raise RuntimeError(
            "V18 protocol requires the exact source to be committed in a clean worktree")
    git["git_clean_or_pushed_required"] = True
    git["git_clean_required"] = True
    payload = {
        "schema_version": 1,
        "scope": "lewm_v8_v18_unopened_task_confirmation",
        "created_at": base.utc_now(),
        "tasks": list(TASKS),
        "designs": list(DESIGNS),
        "objective_families": ["vicreg_stable_lewm_host"],
        "memory_variants": list(MEMORY_VARIANTS),
        "seeds": list(SEEDS),
        "epochs": epochs,
        "runs": len(commands),
        "gpus": list(gpu_ids),
        "gpu_task_queues": {
            gpu: list(queue)
            for gpu, queue in zip(gpu_ids, TASK_QUEUES, strict=True)},
        "task_pinned_gpu": task_gpu,
        "study": study,
        "output_root": str(output_root),
        "log_root": str(log_root),
        "python": python,
        "wandb_enabled": wandb,
        "wandb_entity": base.WANDB_ENTITY,
        "wandb_project": base.WANDB_PROJECT,
        "wandb_mode": "online",
        "blas_threads_per_process": base.BLAS_THREADS,
        "data": dict(data),
        "source_sha256": dict(source),
        "commands": commands,
        "commands_sha256": base.json_sha256(commands),
        "resume_supported": True,
        "resume_granularity": "complete_cell_only",
        "core_artifacts": list(base.CORE_ARTIFACTS),
        "candidate_ssl_selectable_hyperparameters": [],
        "candidate_gradient_policy": "ordinary_joint_end_to_end_backpropagation",
        "claim_under_test": (
            "an explicit persistent action-conditioned recurrent state extends "
            "the usable context of a stabilized finite-context LeWM host under "
            "partial observability"),
        "primary_metric": "heldout_prior_state_nmse",
        "secondary_metric": "val_predictive_loss",
        "mechanism_controls": [
            "vicreg_hacssmv8_noaction", "vicreg_hacssmv8_single"],
        "endpoint_controls": [
            "vicreg_hacssmv8_static", "vicreg_hacssmv8_dynamic"],
        "recurrent_baselines": ["vicreg_gru", "vicreg_ssm"],
        "confirmation_requires_executed_return": False,
        "executed_return_claim_permitted": False,
        "data_opened_only_after_architecture_and_grid_freeze": True,
        **git,
    }
    return payload


def validate_frozen_protocol(
        output_root: Path) -> tuple[dict[str, Any], ProtocolIntegrityGuard]:
    """Load a stored protocol and reproduce every source/data/command identity."""
    protocol_path = output_root / base.PROTOCOL_NAME
    value = base.load_json(protocol_path)
    if not isinstance(value, Mapping):
        raise base.ArtifactError(f"{protocol_path} must contain an object")
    protocol = dict(value)
    cohort, source = validate_inputs()
    try:
        gpu_ids = tuple(str(item) for item in protocol["gpus"])
        expected = protocol_payload(
            python=str(protocol["python"]),
            output_root=Path(str(protocol["output_root"])),
            log_root=Path(str(protocol["log_root"])),
            study=str(protocol["study"]),
            epochs=int(protocol["epochs"]),
            gpu_ids=gpu_ids,
            wandb=bool(protocol["wandb_enabled"]),
            data=cohort,
            source=source,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise base.ArtifactError(
            f"malformed frozen V18 protocol: {type(exc).__name__}: {exc}") from exc
    errors = base._protocol_errors(protocol, expected)
    for key in (
            "gpu_task_queues", "primary_metric", "secondary_metric",
            "mechanism_controls", "endpoint_controls", "recurrent_baselines",
            "git_clean_required"):
        if protocol.get(key) != expected.get(key):
            errors.append(f"confirmation protocol {key} differs")
    if Path(str(protocol.get("output_root", ""))).resolve() != output_root.resolve():
        errors.append("confirmation protocol output_root differs from analysis root")
    if errors:
        raise base.ArtifactError("invalid frozen V18 protocol: " + "; ".join(errors))
    guard = ProtocolIntegrityGuard(protocol["source_sha256"], protocol["data"])
    guard.assert_all()
    return protocol, guard


def validate_analysis_bundle(output_root: Path) -> bool:
    """Return false for no bundle; reject partial, stale, or invalid bundles."""
    analysis_path = output_root / "confirmation_analysis.json"
    cells_path = output_root / "confirmation_cells.csv"
    contrasts_path = output_root / "confirmation_contrasts.csv"
    paths = (analysis_path, cells_path, contrasts_path)
    present = [path.is_file() for path in paths]
    if not any(present):
        return False
    if not all(present):
        raise base.ArtifactError(
            "partial V18 analysis bundle exists; refusing to skip/rewrite it")
    report = base.load_json(analysis_path)
    if not isinstance(report, Mapping) \
            or report.get("status") != "COMPLETE" \
            or report.get("expected_cells") != len(base.cell_specs()) \
            or report.get("completed_valid_cells") != len(base.cell_specs()) \
            or report.get("artifact_integrity_passed") is not True \
            or report.get("scientific_label") not in {
                "STABILIZED_LEWM_V8_CONFIRMATION_PASS", "CONFIRMATION_FAILED"}:
        raise base.ArtifactError("stale or invalid V18 analysis decision exists")
    protocol_path = output_root / base.PROTOCOL_NAME
    if report.get("input_protocol_sha256") != base.file_sha256(protocol_path):
        raise base.ArtifactError("V18 analysis is not bound to the frozen protocol")
    ledger_rows = base.load_json(output_root / base.RUNS_NAME)
    if not isinstance(ledger_rows, list) or len(ledger_rows) != len(base.cell_specs()):
        raise base.ArtifactError("V18 analysis run ledger is incomplete")
    if report.get("input_artifact_manifest_sha256") != artifact_manifest_sha256(
            ledger_rows):
        raise base.ArtifactError("V18 analysis is not bound to current artifacts")
    with cells_path.open(newline="", encoding="utf-8") as stream:
        cell_rows = list(csv.DictReader(stream))
    expected_cells = {
        (task, str(seed), design)
        for task, design, seed in base.cell_specs()}
    actual_cells = {
        (row.get("task"), row.get("seed"), row.get("design"))
        for row in cell_rows}
    if len(cell_rows) != len(expected_cells) or actual_cells != expected_cells:
        raise base.ArtifactError("V18 analysis cell CSV does not cover the frozen grid")
    with contrasts_path.open(newline="", encoding="utf-8") as stream:
        contrast_rows = list(csv.DictReader(stream))
    names = [row.get("contrast") for row in contrast_rows]
    if len(contrast_rows) != 33 or len(set(names)) != 33 or None in names:
        raise base.ArtifactError("V18 analysis contrast CSV is incomplete or duplicated")
    if report.get("cells_csv_sha256") != base.file_sha256(cells_path) \
            or report.get("contrasts_csv_sha256") != base.file_sha256(contrasts_path):
        raise base.ArtifactError("V18 analysis JSON/CSV hashes differ")
    return True


def artifact_manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    entries = [{
        "task": str(row.get("task")),
        "seed": int(row.get("seed")),
        "design": str(row.get("design")),
        "artifact_sha256": row.get("artifact_sha256"),
    } for row in rows]
    entries.sort(key=lambda row: (row["task"], row["seed"], row["design"]))
    if any(not isinstance(row["artifact_sha256"], Mapping) for row in entries):
        raise base.ArtifactError("V18 artifact manifest contains a malformed hash record")
    return base.json_sha256(entries)


def validate_core_artifacts(
        output_root: Path, task: str, design: str, seed: int, epochs: int,
        *, wandb_expected: bool) -> dict[str, Any]:
    """Validate a V18 cell without V17-only gradient-composition fields."""
    import numpy as np
    import torch

    directory = base.run_directory(output_root, task, design, seed)
    paths = {name: directory / name for name in base.CORE_ARTIFACTS}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise base.ArtifactError(
                f"{task}/{design}/s{seed}: missing nonempty {name}")

    metrics = base.load_json(paths["metrics.json"])
    if not isinstance(metrics, dict):
        raise base.ArtifactError(
            f"{task}/{design}/s{seed}: metrics is not an object")
    _, memory, _ = trainer.parse_design(design)
    expected_metrics = {
        "env": f"dmc:{task}",
        "design": design,
        "seed": seed,
        "epochs": epochs,
        "end_to_end_rgb": True,
        "encoder_norm": "causal",
        "predictor_norm": "none",
        "regularizer": "vicreg",
        "one_token_predictor": False,
        "predictor_history": trainer.PREDICTOR_HISTORY,
        "memory_architecture": memory,
        "training_objective": trainer.OBJECTIVE,
        "memory_specific_loss_weight": 0.0,
        "unopened_task_cohort": True,
        "train_episodes": data.DEFAULT_TRAIN_EPISODES,
        "val_episodes": data.DEFAULT_VAL_EPISODES,
        "length": data.DEFAULT_LENGTH,
    }
    for key, value in expected_metrics.items():
        if metrics.get(key) != value:
            raise base.ArtifactError(
                f"{task}/{design}/s{seed}: metrics {key} differs")
    if _ACTIVE_INTEGRITY_GUARD is not None:
        record = _ACTIVE_INTEGRITY_GUARD.cohort[task]
        for role in ("train", "val"):
            for suffix in ("sha256", "content_sha256"):
                metric_key = f"{role}_data_{suffix}"
                record_key = f"{role}_{suffix}"
                if metrics.get(metric_key) != record.get(record_key):
                    raise base.ArtifactError(
                        f"{task}/{design}/s{seed}: {metric_key} differs from protocol")
    for key in REQUIRED_FINITE_METRICS:
        base._finite_metric(metrics, key, f"{task}/{design}/s{seed}")
    base._finite_tree(metrics, f"{task}/{design}/s{seed}.metrics")

    try:
        with np.load(paths["eval_rollout.npz"], allow_pickle=False) as rollout:
            if not rollout.files:
                raise base.ArtifactError(
                    f"{task}/{design}/s{seed}: rollout contains no arrays")
            for key in rollout.files:
                value = rollout[key]
                if value.dtype.hasobject:
                    raise base.ArtifactError(
                        f"{task}/{design}/s{seed}: rollout {key} is object dtype")
                if np.issubdtype(value.dtype, np.number) \
                        and not bool(np.isfinite(value).all()):
                    raise base.ArtifactError(
                        f"{task}/{design}/s{seed}: rollout {key} is nonfinite")
    except (OSError, ValueError) as exc:
        raise base.ArtifactError(
            f"{task}/{design}/s{seed}: invalid rollout: {exc}") from exc
    rollout_hash = base.file_sha256(paths["eval_rollout.npz"])
    if metrics.get("eval_rollout_sha256") != rollout_hash:
        raise base.ArtifactError(
            f"{task}/{design}/s{seed}: rollout SHA-256 differs")

    try:
        checkpoint = torch.load(
            paths["model.pt"], map_location="cpu", weights_only=False)
    except Exception as exc:
        raise base.ArtifactError(
            f"{task}/{design}/s{seed}: cannot load checkpoint: {exc}") from exc
    if not isinstance(checkpoint, Mapping):
        raise base.ArtifactError(
            f"{task}/{design}/s{seed}: checkpoint is not a mapping")
    saved_args = base._as_args_dict(checkpoint.get("args"))
    expected_args = {
        "design": design,
        "seed": seed,
        "epochs": epochs,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "embed_dim": 128,
        "history_len": 3,
        "img_size": 64,
        "patch_size": 8,
        "encoder_layers": 6,
        "encoder_heads": 4,
        "predictor_layers": 4,
        "predictor_heads": 8,
        "dropout": 0.1,
        "sigreg_lambda": 0.0,
        "corruption_seed": data.DEFAULT_CORRUPTION_SEED,
    }
    for key, value in expected_args.items():
        if saved_args.get(key) != value:
            raise base.ArtifactError(
                f"{task}/{design}/s{seed}: checkpoint arg {key} differs")
    if checkpoint.get("final_metrics") != metrics:
        raise base.ArtifactError(
            f"{task}/{design}/s{seed}: checkpoint metrics differ")
    history = checkpoint.get("history")
    if not isinstance(history, list) or len(history) != epochs:
        raise base.ArtifactError(
            f"{task}/{design}/s{seed}: checkpoint history length differs")
    indices = [row.get("epoch") for row in history if isinstance(row, Mapping)]
    if indices != list(range(1, epochs + 1)):
        raise base.ArtifactError(
            f"{task}/{design}/s{seed}: checkpoint epoch indices differ")
    base._finite_tree(history, f"{task}/{design}/s{seed}.history")
    base._finite_tree(
        checkpoint.get("model_state_dict"),
        f"{task}/{design}/s{seed}.model_state_dict")

    receipt = base.load_json(paths["wandb_run.json"])
    if not isinstance(receipt, Mapping):
        raise base.ArtifactError(
            f"{task}/{design}/s{seed}: W&B receipt is not an object")
    base._validate_wandb_receipt(
        receipt, expected=wandb_expected, task=task, design=design,
        seed=seed, rollout_hash=rollout_hash)
    if wandb_expected:
        for key in ("run_id", "run_name", "url", "eval_rollout_artifact_name"):
            if str(receipt.get(key)).strip().lower() in {"", "none", "null", "unknown"}:
                raise base.ArtifactError(
                    f"{task}/{design}/s{seed}: invalid W&B receipt {key}")
        tags = receipt.get("tags")
        required_tags = {
            "lewm-memory", "end-to-end-rgb", "lewm-v8-v18",
            "unopened-task-confirmation", "confirmation-grid",
            "unopened-tasks"}
        if not isinstance(tags, list) or not required_tags.issubset(set(tags)):
            raise base.ArtifactError(
                f"{task}/{design}/s{seed}: V18 W&B tags differ")
    return {
        "directory": str(directory),
        "metrics": metrics,
        "headline_metric": float(metrics["heldout_prior_state_nmse"]),
        "wandb_state": receipt["state"],
        "artifact_sha256": {
            name: base.file_sha256(path) for name, path in paths.items()},
    }


def _install_contract() -> None:
    base.TASKS = TASKS
    base.OBJECTIVE_FAMILIES = ("vicreg_stable_lewm_host",)
    base.MEMORY_VARIANTS = MEMORY_VARIANTS
    base.DESIGNS = DESIGNS
    base.SEEDS = SEEDS
    base.EPOCHS = EPOCHS
    base.DEFAULT_STUDY = DEFAULT_STUDY
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.DEFAULT_LOG_ROOT = DEFAULT_LOG_ROOT
    base.DATA_ROOT = DATA_ROOT
    base.DEFAULT_TRAIN_EPISODES = data.DEFAULT_TRAIN_EPISODES
    base.DEFAULT_VAL_EPISODES = data.DEFAULT_VAL_EPISODES
    base.DEFAULT_LENGTH = data.DEFAULT_LENGTH
    base.DEFAULT_IMG_SIZE = data.DEFAULT_IMG_SIZE
    base.DEFAULT_TRAIN_SEED = data.DEFAULT_TRAIN_SEED
    base.DEFAULT_VAL_SEED = data.DEFAULT_VAL_SEED
    base.PROTOCOL_NAME = "confirmation_protocol.json"
    base.RUNS_NAME = "confirmation_runs.json"
    base.ATTEMPTS_NAME = "confirmation_attempts.json"
    base.SUMMARY_NAME = "confirmation_summary.json"
    base.LOCK_NAME = ".lewm_v8_v18_confirmation.lock"
    base.SOURCE_PATHS = SOURCE_PATHS
    base.REQUIRED_FINITE_METRICS = REQUIRED_FINITE_METRICS
    base.design_parts = design_parts
    base.train_command = train_command
    base.validate_inputs = validate_inputs
    base.protocol_payload = protocol_payload
    base.validate_core_artifacts = validate_core_artifacts


def _restore_contract() -> None:
    for name, value in _BASE_CONTRACT_ORIGINALS.items():
        setattr(base, name, value)


def _run_gpu_queue(
        gpu: str, tasks: Sequence[str], *, python: str,
        output_root: Path, log_root: Path, study: str, epochs: int,
        wandb: bool, resume: bool, ledger: base.RunLedger) -> None:
    for task in tasks:
        if _ACTIVE_INTEGRITY_GUARD is None:
            raise RuntimeError("V18 integrity guard is not active")
        _ACTIVE_INTEGRITY_GUARD.assert_task_data(task, full_hash=True)
        if resume:
            _assert_resume_attempt_history(
                task, ledger, output_root=output_root, epochs=epochs,
                wandb=wandb)
        base._run_task_queue(
            gpu, task, python=python, output_root=output_root,
            log_root=log_root, study=study, epochs=epochs, wandb=wandb,
            resume=resume, ledger=ledger)
        _assert_task_complete(task, ledger)


def _assert_resume_attempt_history(
        task: str, ledger: base.RunLedger, *, output_root: Path,
        epochs: int, wandb: bool) -> None:
    """Never re-execute a cell that already has a terminal attempt receipt."""
    with ledger.lock:
        attempts = [
            dict(row) for row in ledger.attempts
            if str(row.get("task")) == task]
    seen: set[str] = set()
    for row in attempts:
        design = str(row.get("design"))
        seed = int(row.get("seed"))
        key = base.cell_key(task, design, seed)
        if key in seen:
            raise base.ArtifactError(
                f"V18 resume found multiple terminal attempts for {key}")
        seen.add(key)
        if row.get("status") != "complete":
            directory = base.run_directory(output_root, task, design, seed)
            science = [
                name for name in ("model.pt", "metrics.json", "eval_rollout.npz")
                if (directory / name).exists()]
            if science:
                raise base.ArtifactError(
                    f"V18 refuses to retrain failed attempt {key} after science "
                    f"artifacts were produced: {science}")
            # A single transparent retry is allowed only when the process died
            # before producing any scientific artifact. The failed attempt
            # remains permanently recorded in the append-only attempts ledger.
            continue
        validation = validate_core_artifacts(
            output_root, task, design, seed, epochs,
            wandb_expected=wandb)
        if row.get("artifact_sha256") != validation["artifact_sha256"]:
            raise base.ArtifactError(
                f"V18 completed-attempt artifacts changed for {key}")


def _assert_task_complete(task: str, ledger: base.RunLedger) -> None:
    expected = {
        base.cell_key(task, design, seed)
        for seed in SEEDS for design in DESIGNS}
    with ledger.lock:
        rows = {
            key: dict(value) for key, value in ledger.records.items()
            if str(value.get("task")) == task}
    complete = {
        key for key, value in rows.items() if value.get("status") == "complete"}
    missing = sorted(expected - complete)
    unexpected = sorted(set(rows) - expected)
    if missing or unexpected:
        raise base.ArtifactError(
            f"V18 task barrier failed for {task}: "
            f"missing_or_failed={missing[:8]}, unexpected={unexpected[:8]}")


def _main_installed(argv: Sequence[str] | None = None) -> None:
    global _ACTIVE_INTEGRITY_GUARD
    args = base.build_parser().parse_args(
        list(argv) if argv is not None else sys.argv[1:])
    output_root = (
        args.output_root if args.output_root.is_absolute()
        else ROOT / args.output_root).resolve()
    log_root = (
        args.log_root if args.log_root.is_absolute()
        else ROOT / args.log_root).resolve()
    python_path = Path(args.python)
    python_path = python_path if python_path.is_absolute() else ROOT / python_path
    python_path = Path(os.path.abspath(python_path))
    gpu_ids = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if gpu_ids != ("0", "1", "2", "3"):
        raise ValueError("LeWM+V8 V18 requires task-pinned GPUs 0,1,2,3")
    if args.epochs != EPOCHS:
        raise ValueError(f"LeWM+V8 V18 requires exactly {EPOCHS} epochs")
    if args.study != DEFAULT_STUDY:
        raise ValueError(f"V18 study must be {DEFAULT_STUDY!r}")
    if args.no_wandb and not args.dry_run:
        raise ValueError("V18 confirmation requires finished online W&B receipts")
    if args.skip_analysis and not args.dry_run:
        raise ValueError("V18 confirmation requires the write-once final analysis")
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable not found: {python_path}")

    cohort, source = validate_inputs()
    expected_protocol = protocol_payload(
        python=str(python_path), output_root=output_root, log_root=log_root,
        study=args.study, epochs=args.epochs, gpu_ids=gpu_ids,
        wandb=not args.no_wandb, data=cohort, source=source)
    protocol_path = output_root / base.PROTOCOL_NAME
    if args.resume:
        if not protocol_path.is_file():
            raise FileNotFoundError(
                f"--resume requires an existing {protocol_path}")
        value = base.load_json(protocol_path)
        if not isinstance(value, Mapping):
            raise base.ArtifactError(f"{protocol_path} must contain an object")
        errors = base._protocol_errors(value, expected_protocol)
        if value.get("gpu_task_queues") != expected_protocol["gpu_task_queues"]:
            errors.append("confirmation protocol gpu_task_queues differs")
        if errors:
            raise RuntimeError("cannot resume mixed protocol: " + "; ".join(errors))
        protocol = dict(value)
    else:
        protocol = expected_protocol
        if protocol_path.exists():
            raise FileExistsError(
                f"confirmation namespace exists; use --resume: {output_root}")

    _ACTIVE_INTEGRITY_GUARD = ProtocolIntegrityGuard(
        protocol["source_sha256"], protocol["data"])
    _ACTIVE_INTEGRITY_GUARD.assert_all()

    if args.dry_run:
        complete = invalid = absent = 0
        for task, design, seed in base.cell_specs():
            directory = base.run_directory(output_root, task, design, seed)
            present = base._core_artifacts_present(directory)
            if len(present) == len(base.CORE_ARTIFACTS):
                try:
                    validate_core_artifacts(
                        output_root, task, design, seed, args.epochs,
                        wandb_expected=not args.no_wandb)
                    complete += 1
                except (base.ArtifactError, OSError, ValueError):
                    invalid += 1
            elif present:
                invalid += 1
            else:
                absent += 1
        print(json.dumps({
            "scope": protocol["scope"],
            "gpus": list(gpu_ids),
            "gpu_task_queues": protocol["gpu_task_queues"],
            "task_pinned_gpu": protocol["task_pinned_gpu"],
            "tasks": list(TASKS),
            "designs": list(DESIGNS),
            "seeds": list(SEEDS),
            "runs": len(base.cell_specs()),
            "epochs": args.epochs,
            "study": args.study,
            "wandb_enabled": not args.no_wandb,
            "output_root": str(output_root),
            "log_root": str(log_root),
            "commands_sha256": protocol["commands_sha256"],
            "resume": args.resume,
            "local_cells": {
                "complete": complete,
                "invalid_or_partial": invalid,
                "absent": absent},
            "commands": protocol["commands"],
        }, indent=2, sort_keys=True))
        return

    if not args.resume:
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(
                f"fresh output namespace is not empty: {output_root}")
        if log_root.exists() and any(log_root.iterdir()):
            raise FileExistsError(
                f"fresh log namespace is not empty: {log_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        base.atomic_write_json(protocol_path, protocol)
    lock_path = output_root / base.LOCK_NAME
    base.create_lock(lock_path, resume=args.resume)
    summary: dict[str, Any]
    try:
        ledger = base.RunLedger(output_root, resume=args.resume)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    _run_gpu_queue, gpu, queue, python=str(python_path),
                    output_root=output_root, log_root=log_root,
                    study=args.study, epochs=args.epochs,
                    wandb=not args.no_wandb, resume=args.resume,
                    ledger=ledger)
                for gpu, queue in zip(gpu_ids, TASK_QUEUES, strict=True)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        _ACTIVE_INTEGRITY_GUARD.assert_all()
        rows = list(ledger.records.values())
        complete = sum(row.get("status") == "complete" for row in rows)
        failed = [row for row in rows if row.get("status") != "complete"]
        summary = {
            "schema_version": 1,
            "scope": protocol["scope"],
            "status": (
                "COMPLETE"
                if complete == len(base.cell_specs()) and not failed
                else "INCOMPLETE_OR_INVALID"),
            "expected_cells": len(base.cell_specs()),
            "completed_cells": complete,
            "failed_or_invalid_cells": len(failed),
            "finished_at": base.utc_now(),
            "resume": args.resume,
            "wandb_enabled": not args.no_wandb,
            "failures": [{
                "task": row.get("task"),
                "design": row.get("design"),
                "seed": row.get("seed"),
                "status": row.get("status"),
                "error": row.get("error")}
                for row in failed],
        }
        base.atomic_write_json(output_root / base.SUMMARY_NAME, summary)
    finally:
        lock_path.unlink(missing_ok=True)

    analysis_returncode = 0
    if summary["status"] == "COMPLETE" and not args.skip_analysis:
        if not validate_analysis_bundle(output_root):
            analysis = subprocess.run([
                str(python_path),
                str(ROOT / "scripts" / "analyze_lewm_v8_v18.py"),
                "--root", str(output_root), "--write",
            ], cwd=ROOT, check=False)
            analysis_returncode = analysis.returncode
            if analysis_returncode == 0:
                validate_analysis_bundle(output_root)
    if summary["status"] != "COMPLETE" or analysis_returncode:
        raise SystemExit(2)


def main(argv: Sequence[str] | None = None) -> None:
    global _ACTIVE_INTEGRITY_GUARD
    _install_contract()
    try:
        _main_installed(argv)
    finally:
        _ACTIVE_INTEGRITY_GUARD = None
        _restore_contract()


if __name__ == "__main__":
    main()
