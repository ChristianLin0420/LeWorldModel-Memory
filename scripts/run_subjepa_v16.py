#!/usr/bin/env python3
"""Run the 144-cell Sub-JEPA-v16 development grid on four task-pinned GPUs.

This is a resumable *development* runner, not a frozen confirmation protocol.
It records source/data/command hashes without requiring a clean or pushed git
checkout.  A resumed run adopts a cell only after validating its local core
artifacts (checkpoint, metrics, and rollout); it never silently overwrites a
partial or malformed result.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.hacssm_v11_data import (
    DEFAULT_IMG_SIZE,
    DEFAULT_LENGTH,
    DEFAULT_TRAIN_EPISODES,
    DEFAULT_TRAIN_SEED,
    DEFAULT_VAL_EPISODES,
    DEFAULT_VAL_SEED,
    cache_name,
    sha256_file,
)


TASKS = (
    "cartpole.swingup",
    "fish.swim",
    "pendulum.swingup",
    "walker.walk",
)
MEMORY_VARIANTS = ("none", "ssm", "hacssmv8")
OBJECTIVE_FAMILIES = ("fullsig", "subjepa16", "subjepa32", "vicreg")
FULLSIG_DESIGNS = tuple(
    f"{family}_{memory}"
    for family in OBJECTIVE_FAMILIES
    for memory in MEMORY_VARIANTS
)
DESIGNS = FULLSIG_DESIGNS
SEEDS = (16_001, 16_002, 16_003)
EPOCHS = 30

DEFAULT_STUDY = "subjepa-v16-development"
DEFAULT_OUTPUT_ROOT = Path("outputs/subjepa_v16_development")
DEFAULT_LOG_ROOT = Path("logs/subjepa_v16_development")
DATA_ROOT = Path("outputs/hacssm_v11_data")
WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
BLAS_THREADS = 4
DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"

PROTOCOL_NAME = "development_protocol.json"
RUNS_NAME = "development_runs.json"
ATTEMPTS_NAME = "development_attempts.json"
SUMMARY_NAME = "development_summary.json"
LOCK_NAME = ".subjepa_v16_development.lock"
CORE_ARTIFACTS = ("model.pt", "metrics.json", "eval_rollout.npz")

# Closed source set used by the new trainer and its inherited LeWM host.  The
# development protocol records these hashes but intentionally does not impose
# a clean/pushed-git requirement.
SOURCE_PATHS = (
    Path("lewm/models/__init__.py"),
    Path("lewm/models/memory.py"),
    Path("lewm/models/memory_model.py"),
    Path("lewm/models/leworldmodel.py"),
    Path("lewm/models/encoder.py"),
    Path("lewm/models/sigreg.py"),
    Path("scripts/hacssm_v10_data.py"),
    Path("scripts/hacssm_v11_data.py"),
    Path("scripts/train_hacssm_v10.py"),
    Path("scripts/train_hacssm_v11.py"),
    Path("scripts/train_subjepa_v16.py"),
    Path("scripts/run_subjepa_v16.py"),
    Path("scripts/analyze_subjepa_v16.py"),
)


class ArtifactError(RuntimeError):
    """A local cell exists but does not satisfy the core artifact contract."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read JSON {path}: {exc}") from exc


def _finite_tree(value: Any, label: str) -> None:
    """Reject non-finite numeric values recursively while allowing metadata."""
    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - environment preflight
        raise ArtifactError(f"artifact validation dependency unavailable: {exc}") from exc

    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                raise ArtifactError(f"{label} contains a non-finite tensor")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all()):
            raise ArtifactError(f"{label} contains a non-finite array")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ArtifactError(f"{label} is non-finite")


def design_parts(design: str) -> tuple[str, str, int | None]:
    if design not in DESIGNS:
        raise ValueError(f"unknown Sub-JEPA-v16 design {design!r}")
    family, memory = design.rsplit("_", 1)
    tokens = 16 if family == "subjepa16" else 32 if family == "subjepa32" else None
    return family, memory, tokens


def _slug(task: str) -> str:
    return "dmc_" + task.replace(".", "_")


def cell_key(task: str, design: str, seed: int) -> str:
    return f"{task}|{design}|{seed}"


def data_paths(task: str) -> tuple[Path, Path]:
    return (
        DATA_ROOT / cache_name(
            task, "train", DEFAULT_TRAIN_EPISODES, DEFAULT_LENGTH,
            DEFAULT_IMG_SIZE, DEFAULT_TRAIN_SEED),
        DATA_ROOT / cache_name(
            task, "val", DEFAULT_VAL_EPISODES, DEFAULT_LENGTH,
            DEFAULT_IMG_SIZE, DEFAULT_VAL_SEED),
    )


def run_name(task: str, design: str, seed: int) -> str:
    return f"lewm-dmc:{task}-{design}-s{seed}"


def run_directory(output_root: Path, task: str, design: str, seed: int) -> Path:
    return output_root / run_name(task, design, seed)


def cell_specs() -> list[tuple[str, str, int]]:
    return [
        (task, design, seed)
        for task in TASKS
        for seed in SEEDS
        for design in DESIGNS
    ]


def train_command(
        python: str, output_root: Path, study: str, epochs: int,
        task: str, design: str, seed: int, *, wandb: bool) -> list[str]:
    train_data, val_data = data_paths(task)
    command = [
        python,
        str(ROOT / "scripts" / "train_subjepa_v16.py"),
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
        "--sigreg-lambda", "0.1",
        "--sigreg-projections", "512",
        "--probe-ridge", "0.001",
        "--eval-target-key", "task_observation",
        "--corruption-seed", "11012",
        "--eval-rollout-episode", "0",
        "--device", "cuda",
        "--wandb" if wandb else "--no-wandb",
        "--wandb-entity", WANDB_ENTITY,
        "--wandb-project", WANDB_PROJECT,
        "--wandb-mode", "online",
        "--wandb-study", study,
        "--extra-tag", "development-grid,subjepa-v16",
    ]
    return command


def command_records(
        python: str, output_root: Path, study: str, epochs: int,
        *, wandb: bool) -> list[dict[str, Any]]:
    return [{
        "task": task,
        "design": design,
        "seed": seed,
        "argv": train_command(
            python, output_root, study, epochs, task, design, seed,
            wandb=wandb),
    } for task, design, seed in cell_specs()]


def _git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ("git", *arguments), cwd=ROOT, text=True, capture_output=True,
        check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_receipt() -> dict[str, Any]:
    """Record best-effort git state; never gate this development run on it."""
    status = _git_value("status", "--porcelain", "--untracked-files=all")
    head = _git_value("rev-parse", "HEAD")
    upstream = _git_value("rev-parse", "@{upstream}")
    return {
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": head,
        "git_upstream_commit": upstream,
        "git_worktree_clean": status == "" if status is not None else None,
        "git_head_pushed": head == upstream if head and upstream else None,
        "git_status_sha256": (
            hashlib.sha256(status.encode()).hexdigest()
            if status is not None else None),
        "git_clean_or_pushed_required": False,
    }


def validate_inputs() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    missing_sources = [path for path in SOURCE_PATHS if not (ROOT / path).is_file()]
    if missing_sources:
        raise FileNotFoundError(f"missing Sub-JEPA-v16 source files: {missing_sources}")

    # The trainer owns the canonical public registry.  Keep the runner's closed
    # command grid synchronized without importing trainer code at module import.
    import scripts.train_subjepa_v16 as trainer
    trainer_designs = getattr(
        trainer, "FULLSIG_DESIGNS", getattr(trainer, "DESIGNS", None))
    if trainer_designs is None:
        raise RuntimeError(
            "trainer must export FULLSIG_DESIGNS (or the compatible DESIGNS alias)")
    if tuple(trainer_designs) != DESIGNS:
        raise RuntimeError(
            f"runner designs {DESIGNS} differ from trainer registry "
            f"{tuple(trainer_designs)}")

    data: dict[str, dict[str, str]] = {}
    for task in TASKS:
        train_path, val_path = data_paths(task)
        for path in (train_path, val_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing V16 development cache {path}")
        data[task] = {
            "train": str(train_path),
            "train_sha256": sha256_file(train_path),
            "val": str(val_path),
            "val_sha256": sha256_file(val_path),
        }
    source = {str(path): file_sha256(ROOT / path) for path in SOURCE_PATHS}
    return data, source


def protocol_payload(
        *, python: str, output_root: Path, log_root: Path, study: str,
        epochs: int, gpu_ids: Sequence[str], wandb: bool,
        data: Mapping[str, Any], source: Mapping[str, str]) -> dict[str, Any]:
    commands = command_records(
        python, output_root, study, epochs, wandb=wandb)
    return {
        "schema_version": 1,
        "scope": "subjepa_v16_excluded_adaptive_development",
        "created_at": utc_now(),
        "tasks": list(TASKS),
        "designs": list(DESIGNS),
        "objective_families": list(OBJECTIVE_FAMILIES),
        "memory_variants": list(MEMORY_VARIANTS),
        "seeds": list(SEEDS),
        "epochs": epochs,
        "runs": len(commands),
        "gpus": list(gpu_ids),
        "task_pinned_gpu": dict(zip(TASKS, gpu_ids, strict=True)),
        "study": study,
        "output_root": str(output_root),
        "log_root": str(log_root),
        "python": python,
        "wandb_enabled": wandb,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_mode": "online",
        "blas_threads_per_process": BLAS_THREADS,
        "blas_environment_variables": [
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS"],
        "data": dict(data),
        "source_sha256": dict(source),
        "commands": commands,
        "commands_sha256": json_sha256(commands),
        "resume_supported": True,
        "core_artifacts": list(CORE_ARTIFACTS),
        **git_receipt(),
    }


def _as_args_dict(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__") and isinstance(vars(value), dict):
        return vars(value)
    raise ArtifactError("checkpoint args must be a mapping or Namespace")


def validate_core_artifacts(
        output_root: Path, task: str, design: str, seed: int, epochs: int
        ) -> dict[str, Any]:
    """Validate one finished cell without relying on W&B or remote state."""
    import numpy as np
    import torch

    directory = run_directory(output_root, task, design, seed)
    paths = {name: directory / name for name in CORE_ARTIFACTS}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise ArtifactError(f"{task}/{design}/s{seed}: missing nonempty {name}")

    metrics = load_json(paths["metrics.json"])
    if not isinstance(metrics, dict):
        raise ArtifactError(f"{task}/{design}/s{seed}: metrics must be a JSON object")
    expected = {
        "env": f"dmc:{task}",
        "design": design,
        "seed": seed,
        "epochs": epochs,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ArtifactError(
                f"{task}/{design}/s{seed}: metrics {key}={metrics.get(key)!r}, "
                f"expected {value!r}")
    for key in (
            "heldout_prior_state_nmse", "initial_encoder_integrator_probe_nmse",
            "encoder_covariance_effective_rank",
            "predictive_loss_convergence_relative_change"):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            raise ArtifactError(
                f"{task}/{design}/s{seed}: missing finite metric {key}")
    _finite_tree(metrics, f"{task}/{design}/s{seed}.metrics")

    try:
        with np.load(paths["eval_rollout.npz"], allow_pickle=False) as rollout:
            if not rollout.files:
                raise ArtifactError(
                    f"{task}/{design}/s{seed}: rollout contains no arrays")
            for key in rollout.files:
                array = rollout[key]
                if array.dtype.hasobject:
                    raise ArtifactError(
                        f"{task}/{design}/s{seed}: rollout {key} uses object dtype")
                if np.issubdtype(array.dtype, np.number) \
                        and not bool(np.isfinite(array).all()):
                    raise ArtifactError(
                        f"{task}/{design}/s{seed}: rollout {key} is non-finite")
    except (OSError, ValueError) as exc:
        raise ArtifactError(
            f"{task}/{design}/s{seed}: invalid rollout: {exc}") from exc

    rollout_hash = file_sha256(paths["eval_rollout.npz"])
    declared_hash = metrics.get("eval_rollout_sha256")
    if declared_hash is not None and declared_hash != rollout_hash:
        raise ArtifactError(f"{task}/{design}/s{seed}: rollout SHA-256 mismatch")

    try:
        checkpoint = torch.load(
            paths["model.pt"], map_location="cpu", weights_only=False)
    except Exception as exc:  # torch emits several serialization exception types
        raise ArtifactError(
            f"{task}/{design}/s{seed}: cannot load checkpoint: {exc}") from exc
    if not isinstance(checkpoint, Mapping):
        raise ArtifactError(f"{task}/{design}/s{seed}: checkpoint is not a mapping")
    saved_args = _as_args_dict(checkpoint.get("args"))
    for key, value in {
            "design": design, "seed": seed, "epochs": epochs}.items():
        if saved_args.get(key) != value:
            raise ArtifactError(
                f"{task}/{design}/s{seed}: checkpoint arg {key} differs")
    if checkpoint.get("final_metrics") != metrics:
        raise ArtifactError(
            f"{task}/{design}/s{seed}: checkpoint final_metrics differ from metrics.json")
    history = checkpoint.get("history")
    if not isinstance(history, list) or len(history) != epochs:
        raise ArtifactError(
            f"{task}/{design}/s{seed}: checkpoint history length differs")
    epoch_indices = [row.get("epoch") for row in history if isinstance(row, Mapping)]
    if epoch_indices != list(range(1, epochs + 1)):
        raise ArtifactError(
            f"{task}/{design}/s{seed}: checkpoint epoch indices differ")
    _finite_tree(checkpoint.get("model_state_dict"),
                 f"{task}/{design}/s{seed}.model_state_dict")

    artifact_paths = dict(paths)
    wandb_path = directory / "wandb_run.json"
    if wandb_path.is_file():
        receipt = load_json(wandb_path)
        if not isinstance(receipt, Mapping):
            raise ArtifactError(
                f"{task}/{design}/s{seed}: W&B receipt is not an object")
        artifact_paths[wandb_path.name] = wandb_path

    return {
        "directory": str(directory),
        "metrics": metrics,
        "headline_metric": float(metrics["heldout_prior_state_nmse"]),
        "wandb_receipt_present": wandb_path.is_file(),
        "artifact_sha256": {
            name: file_sha256(path) for name, path in artifact_paths.items()},
    }


def _core_artifacts_present(directory: Path) -> list[str]:
    return [name for name in CORE_ARTIFACTS if (directory / name).exists()]


def _next_log_path(log_root: Path, task: str, design: str, seed: int) -> Path:
    base = log_root / f"{_slug(task)}-{design}-s{seed}.log"
    if not base.exists():
        return base
    attempt = 2
    while True:
        candidate = base.with_name(f"{base.stem}.attempt{attempt}{base.suffix}")
        if not candidate.exists():
            return candidate
        attempt += 1


class RunLedger:
    """Thread-safe, crash-resilient current-cell and attempt receipts."""

    def __init__(self, output_root: Path, *, resume: bool):
        self.runs_path = output_root / RUNS_NAME
        self.attempts_path = output_root / ATTEMPTS_NAME
        self.lock = threading.Lock()
        self.records: dict[str, dict[str, Any]] = {}
        self.attempts: list[dict[str, Any]] = []
        if resume and self.runs_path.is_file():
            value = load_json(self.runs_path)
            if not isinstance(value, list):
                raise ArtifactError(f"{self.runs_path} must contain a list")
            for row in value:
                if not isinstance(row, dict):
                    raise ArtifactError(f"{self.runs_path} contains a non-object")
                key = cell_key(str(row.get("task")), str(row.get("design")),
                               int(row.get("seed")))
                if key in self.records:
                    raise ArtifactError(f"duplicate ledger cell {key}")
                self.records[key] = row
        if resume and self.attempts_path.is_file():
            value = load_json(self.attempts_path)
            if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
                raise ArtifactError(f"{self.attempts_path} must contain an object list")
            self.attempts = list(value)

    def _persist(self) -> None:
        rows = sorted(self.records.values(), key=lambda row: (
            TASKS.index(str(row["task"])), int(row["seed"]),
            DESIGNS.index(str(row["design"]))))
        atomic_write_json(self.runs_path, rows)
        atomic_write_json(self.attempts_path, self.attempts)

    def record(self, row: Mapping[str, Any], *, attempt: bool = False) -> None:
        copied = dict(row)
        key = cell_key(str(copied["task"]), str(copied["design"]),
                       int(copied["seed"]))
        with self.lock:
            self.records[key] = copied
            if attempt:
                self.attempts.append(copied)
            self._persist()


def _record_for_valid_cell(
        task: str, design: str, seed: int, gpu: str, command: Sequence[str],
        validation: Mapping[str, Any], *, seconds: float,
        log_path: Path | None, resumed_existing: bool) -> dict[str, Any]:
    return {
        "task": task,
        "design": design,
        "seed": seed,
        "gpu": gpu,
        "status": "complete",
        "resumed_existing": resumed_existing,
        "seconds": seconds,
        "completed_at": utc_now(),
        "command_sha256": json_sha256(list(command)),
        "log": str(log_path) if log_path else None,
        "directory": validation["directory"],
        "headline_metric": validation["headline_metric"],
        "wandb_receipt_present": validation["wandb_receipt_present"],
        "artifact_sha256": validation["artifact_sha256"],
    }


def _run_task_queue(
        gpu: str, task: str, *, python: str, output_root: Path,
        log_root: Path, study: str, epochs: int, wandb: bool,
        resume: bool, ledger: RunLedger) -> None:
    for seed in SEEDS:
        for design in DESIGNS:
            command = train_command(
                python, output_root, study, epochs, task, design, seed,
                wandb=wandb)
            directory = run_directory(output_root, task, design, seed)
            existing = _core_artifacts_present(directory)

            if resume and len(existing) == len(CORE_ARTIFACTS):
                try:
                    validation = validate_core_artifacts(
                        output_root, task, design, seed, epochs)
                    row = _record_for_valid_cell(
                        task, design, seed, gpu, command, validation,
                        seconds=0.0, log_path=None, resumed_existing=True)
                    ledger.record(row)
                    print(f"[gpu {gpu}] resume-valid {task}/{design}/s{seed}",
                          flush=True)
                    continue
                except (ArtifactError, OSError, ValueError) as exc:
                    row = {
                        "task": task, "design": design, "seed": seed,
                        "gpu": gpu, "status": "invalid_existing",
                        "resumed_existing": False, "seconds": 0.0,
                        "completed_at": utc_now(),
                        "command_sha256": json_sha256(command),
                        "log": None, "directory": str(directory),
                        "error": str(exc),
                    }
                    ledger.record(row)
                    print(f"[gpu {gpu}] INVALID {task}/{design}/s{seed}: {exc}",
                          flush=True)
                    continue

            if existing:
                row = {
                    "task": task, "design": design, "seed": seed,
                    "gpu": gpu, "status": "partial_existing",
                    "resumed_existing": False, "seconds": 0.0,
                    "completed_at": utc_now(),
                    "command_sha256": json_sha256(command),
                    "log": None, "directory": str(directory),
                    "present_core_artifacts": existing,
                    "error": "refusing to overwrite a partial cell",
                }
                ledger.record(row)
                print(f"[gpu {gpu}] PARTIAL {task}/{design}/s{seed}: {existing}",
                      flush=True)
                continue

            log_path = _next_log_path(log_root, task, design, seed)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["MUJOCO_GL"] = "egl"
            for variable in (
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
                environment[variable] = str(BLAS_THREADS)
            started = time.time()
            print(f"[gpu {gpu}] starting {task}/{design}/s{seed}", flush=True)
            returncode: int | None = None
            error: str | None = None
            validation: Mapping[str, Any] | None = None
            try:
                with log_path.open("x", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command, cwd=ROOT, env=environment, stdout=log,
                        stderr=subprocess.STDOUT, text=True, check=False)
                returncode = completed.returncode
                if returncode:
                    error = f"trainer exited with status {returncode}"
                else:
                    validation = validate_core_artifacts(
                        output_root, task, design, seed, epochs)
            except Exception as exc:  # preserve progress across isolated cell failures
                error = f"{type(exc).__name__}: {exc}"
            seconds = time.time() - started
            if validation is not None and error is None:
                row = _record_for_valid_cell(
                    task, design, seed, gpu, command, validation,
                    seconds=seconds, log_path=log_path, resumed_existing=False)
                ledger.record(row, attempt=True)
                print(
                    f"[gpu {gpu}] finished {task}/{design}/s{seed} "
                    f"in {seconds:.1f}s", flush=True)
            else:
                row = {
                    "task": task, "design": design, "seed": seed,
                    "gpu": gpu, "status": "failed",
                    "resumed_existing": False, "seconds": seconds,
                    "completed_at": utc_now(),
                    "returncode": returncode,
                    "command_sha256": json_sha256(command),
                    "log": str(log_path), "directory": str(directory),
                    "error": error or "unknown cell failure",
                }
                ledger.record(row, attempt=True)
                print(
                    f"[gpu {gpu}] FAILED {task}/{design}/s{seed}: "
                    f"{row['error']} (see {log_path})", flush=True)


def _lock_is_live(path: Path) -> bool:
    try:
        value = load_json(path)
        pid = int(value["pid"])
        host = str(value["hostname"])
    except (ArtifactError, KeyError, TypeError, ValueError):
        return True
    if host != socket.gethostname():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def create_lock(path: Path, *, resume: bool) -> None:
    if path.exists():
        if not resume or _lock_is_live(path):
            raise RuntimeError(f"development runner lock is active: {path}")
        path.unlink()
    with path.open("x", encoding="utf-8") as stream:
        json.dump({
            "pid": os.getpid(), "hostname": socket.gethostname(),
            "created_at": utc_now()}, stream, sort_keys=True)
        stream.write("\n")


def _protocol_errors(
        existing: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    keys = (
        "schema_version", "scope", "tasks", "designs", "objective_families",
        "memory_variants", "seeds", "epochs", "runs", "gpus",
        "task_pinned_gpu", "study", "output_root", "log_root", "python",
        "wandb_enabled", "wandb_entity", "wandb_project", "wandb_mode",
        "data", "source_sha256", "commands", "commands_sha256",
    )
    for key in keys:
        if existing.get(key) != expected.get(key):
            errors.append(f"development protocol {key} differs")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--study", default=DEFAULT_STUDY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_root = (
        args.output_root if args.output_root.is_absolute()
        else ROOT / args.output_root).resolve()
    log_root = (
        args.log_root if args.log_root.is_absolute()
        else ROOT / args.log_root).resolve()
    python_path = Path(args.python)
    python_path = python_path if python_path.is_absolute() else ROOT / python_path
    # Preserve a virtualenv symlink in the executable string.  Resolving it to
    # /usr/bin/python would bypass pyvenv.cfg and launch outside the environment.
    python_path = Path(os.path.abspath(python_path))
    gpu_ids = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if gpu_ids != ("0", "1", "2", "3"):
        raise ValueError("Sub-JEPA-v16 requires task-pinned GPUs 0,1,2,3")
    if args.epochs != EPOCHS:
        raise ValueError(f"Sub-JEPA-v16 development requires exactly {EPOCHS} epochs")
    if args.study != DEFAULT_STUDY:
        raise ValueError(f"Sub-JEPA-v16 development study must be {DEFAULT_STUDY!r}")
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable not found: {python_path}")

    data, source = validate_inputs()
    expected_protocol = protocol_payload(
        python=str(python_path), output_root=output_root, log_root=log_root,
        study=args.study, epochs=args.epochs, gpu_ids=gpu_ids,
        wandb=not args.no_wandb, data=data, source=source)
    protocol_path = output_root / PROTOCOL_NAME

    if args.resume:
        if not protocol_path.is_file():
            raise FileNotFoundError(
                f"--resume requires an existing {protocol_path}")
        existing_protocol = load_json(protocol_path)
        if not isinstance(existing_protocol, Mapping):
            raise ArtifactError(f"{protocol_path} must contain an object")
        errors = _protocol_errors(existing_protocol, expected_protocol)
        if errors:
            raise RuntimeError("cannot resume mixed protocol: " + "; ".join(errors))
        protocol = dict(existing_protocol)
    else:
        protocol = expected_protocol
        if protocol_path.exists():
            raise FileExistsError(
                f"development namespace exists; use --resume: {output_root}")

    if args.dry_run:
        complete, invalid, absent = 0, 0, 0
        for task, design, seed in cell_specs():
            directory = run_directory(output_root, task, design, seed)
            if len(_core_artifacts_present(directory)) == len(CORE_ARTIFACTS):
                try:
                    validate_core_artifacts(
                        output_root, task, design, seed, args.epochs)
                    complete += 1
                except (ArtifactError, OSError, ValueError):
                    invalid += 1
            elif _core_artifacts_present(directory):
                invalid += 1
            else:
                absent += 1
        print(json.dumps({
            "scope": protocol["scope"],
            "gpus": list(gpu_ids),
            "task_pinned_gpu": protocol["task_pinned_gpu"],
            "tasks": list(TASKS),
            "designs": list(DESIGNS),
            "seeds": list(SEEDS),
            "runs": len(cell_specs()),
            "epochs": args.epochs,
            "study": args.study,
            "wandb_enabled": not args.no_wandb,
            "output_root": str(output_root),
            "log_root": str(log_root),
            "commands_sha256": protocol["commands_sha256"],
            "resume": args.resume,
            "local_cells": {
                "complete": complete, "invalid_or_partial": invalid,
                "absent": absent},
            "commands": protocol["commands"],
        }, indent=2, sort_keys=True))
        return

    if not args.resume:
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(
                f"fresh development output namespace is not empty: {output_root}")
        if log_root.exists() and any(log_root.iterdir()):
            raise FileExistsError(
                f"fresh development log namespace is not empty: {log_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        atomic_write_json(protocol_path, protocol)
    lock_path = output_root / LOCK_NAME
    create_lock(lock_path, resume=args.resume)
    try:
        ledger = RunLedger(output_root, resume=args.resume)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    _run_task_queue, gpu, task, python=str(python_path),
                    output_root=output_root, log_root=log_root,
                    study=args.study, epochs=args.epochs,
                    wandb=not args.no_wandb, resume=args.resume, ledger=ledger)
                for task, gpu in zip(TASKS, gpu_ids, strict=True)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        rows = list(ledger.records.values())
        complete = sum(row.get("status") == "complete" for row in rows)
        failed = [row for row in rows if row.get("status") != "complete"]
        summary = {
            "schema_version": 1,
            "scope": protocol["scope"],
            "status": "COMPLETE" if complete == len(cell_specs()) and not failed
                      else "INCOMPLETE_OR_INVALID",
            "expected_cells": len(cell_specs()),
            "completed_cells": complete,
            "failed_or_invalid_cells": len(failed),
            "finished_at": utc_now(),
            "resume": args.resume,
            "wandb_enabled": not args.no_wandb,
            "failures": [{
                "task": row.get("task"), "design": row.get("design"),
                "seed": row.get("seed"), "status": row.get("status"),
                "error": row.get("error")}
                for row in failed],
        }
        atomic_write_json(output_root / SUMMARY_NAME, summary)
    finally:
        lock_path.unlink(missing_ok=True)

    analysis_returncode = 0
    if not args.skip_analysis:
        completed_analysis = subprocess.run([
            str(python_path), str(ROOT / "scripts" / "analyze_subjepa_v16.py"),
            "--root", str(output_root), "--write",
        ], cwd=ROOT, check=False)
        analysis_returncode = completed_analysis.returncode

    if summary["status"] != "COMPLETE" or analysis_returncode:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
