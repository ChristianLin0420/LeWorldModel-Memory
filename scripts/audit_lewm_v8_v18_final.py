#!/usr/bin/env python3
"""Independent read-only final auditor for the frozen LeWM+V8 V18 study.

This file deliberately does not import the repository's runner or analyzer.
It refuses to inspect checkpoints or W&B until the local 200/200 completion
and write-once analysis markers are present.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np


TASKS = (
    "acrobot.swingup",
    "manipulator.bring_ball",
    "quadruped.run",
    "stacker.stack_4",
    "swimmer.swimmer15",
)
DESIGNS = (
    "vicreg_none",
    "vicreg_gru",
    "vicreg_ssm",
    "vicreg_hacssmv8",
    "vicreg_hacssmv8_static",
    "vicreg_hacssmv8_dynamic",
    "vicreg_hacssmv8_noaction",
    "vicreg_hacssmv8_single",
)
SEEDS = (18_001, 18_002, 18_003, 18_004, 18_005)
EPOCHS = 100
EXPECTED_CELLS = len(TASKS) * len(DESIGNS) * len(SEEDS)
STUDY = "lewm-v8-v18-confirmation"
SCOPE = "lewm_v8_v18_unopened_task_confirmation"
WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
CORE_ARTIFACTS = (
    "model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json")
HISTORY_FIELDS = (
    "loss", "predictive_loss", "regularizer_loss", "sigreg_loss",
    "variance_loss", "covariance_loss")

PRIMARY = "heldout_prior_state_nmse"
CLEAN = "clean_prior_state_nmse"
SECONDARY = "val_predictive_loss"
DEEP = "deep_prior_state_nmse"
VARIANCE = "encoder_mean_channel_variance"
RANK = "encoder_covariance_effective_rank"
CONVERGENCE = "predictive_loss_convergence_relative_change"
INTEGRATOR = "initial_encoder_integrator_probe_nmse"
CONDITIONS = ("freeze", "gaussian_noise", "checkerboard", "long_freeze")

NONE = "vicreg_none"
GRU = "vicreg_gru"
SSM = "vicreg_ssm"
CANDIDATE = "vicreg_hacssmv8"
STATIC = "vicreg_hacssmv8_static"
DYNAMIC = "vicreg_hacssmv8_dynamic"
NO_ACTION = "vicreg_hacssmv8_noaction"
SINGLE = "vicreg_hacssmv8_single"
DIRECT_REFERENCES = tuple(item for item in DESIGNS if item != CANDIDATE)
RECURRENT_REFERENCES = (GRU, SSM)
ENDPOINT_REFERENCES = (DYNAMIC, STATIC)

BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 18_018

PROTOCOL_NAME = "confirmation_protocol.json"
RUNS_NAME = "confirmation_runs.json"
ATTEMPTS_NAME = "confirmation_attempts.json"
SUMMARY_NAME = "confirmation_summary.json"
ANALYSIS_NAME = "confirmation_analysis.json"
CELLS_NAME = "confirmation_cells.csv"
CONTRASTS_NAME = "confirmation_contrasts.csv"
LOCK_NAME = ".lewm_v8_v18_confirmation.lock"

SOURCE_PATHS = (
    "docs/V18_LEWM_V8_CONFIRMATION.md",
    "lewm/__init__.py",
    "lewm/models/__init__.py",
    "lewm/models/leworldmodel.py",
    "lewm/models/memory.py",
    "lewm/models/memory_model.py",
    "lewm/models/encoder.py",
    "lewm/models/sigreg.py",
    "lewm/models/siro.py",
    "lewm/models/cf_hiro.py",
    "lewm/models/cf_ebo.py",
    "lewm/models/cvpf.py",
    "scripts/hacssm_v10_data.py",
    "scripts/hacssm_v11_data.py",
    "scripts/hacssm_v18_data.py",
    "scripts/train_hacssm_v10.py",
    "scripts/train_hacssm_v11.py",
    "scripts/train_subjepa_v16.py",
    "scripts/train_lewm_v8_v18.py",
    "scripts/run_autovisreg_v17.py",
    "scripts/run_lewm_v8_v18.py",
    "scripts/analyze_lewm_v8_v18.py",
)

CONTENT_FIELDS = (
    "schema_version", "action_process", "env_id", "split", "seed",
    "length", "img_size", "smooth_rho", "obs", "actions",
    "task_observation", "task_observation_keys",
    "task_observation_shape_offsets", "task_observation_shape_values",
    "task_observation_slices", "physics_state", "rewards", "action_min",
    "action_max",
)
REQUIRED_CACHE_FIELDS = frozenset((*CONTENT_FIELDS, "content_sha256"))
ACTION_PROCESS = "bounded_tanh_iid_gaussian"
TRAINING_OBJECTIVE = (
    "v18_paired_next_clean_sliding_h3_plus_vicreg_causal_memory_confirmation")

GPU_BY_TASK = {
    "acrobot.swingup": "0",
    "stacker.stack_4": "0",
    "manipulator.bring_ball": "1",
    "quadruped.run": "2",
    "swimmer.swimmer15": "3",
}


class AuditRefused(RuntimeError):
    """The terminal 200/200 markers do not yet permit an audit."""


class AuditFailure(RuntimeError):
    """One independently checked invariant is false."""


class RemoteAuditFailure(AuditFailure):
    """Local audit passed but requested cloud evidence could not be verified."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditFailure(f"cannot read JSON {path}: {exc}") from exc


def resolve_from(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def expected_keys() -> set[tuple[str, int, str]]:
    return {
        (task, seed, design)
        for task in TASKS for seed in SEEDS for design in DESIGNS
    }


def run_name(task: str, design: str, seed: int) -> str:
    return f"lewm-dmc:{task}-{design}-s{seed}"


def run_directory(root: Path, task: str, design: str, seed: int) -> Path:
    return root / run_name(task, design, seed)


def cell_label(key: tuple[str, int, str]) -> str:
    task, seed, design = key
    return f"{task}/{design}/s{seed}"


def finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise AuditFailure(f"{label} is not a finite scalar")
    return float(value)


def finite_tree(value: Any, label: str) -> None:
    # Torch is intentionally duck-typed here so self-test and preflight do not
    # import it before the 200/200 refusal boundary.
    if value.__class__.__module__.startswith("torch") \
            and hasattr(value, "is_floating_point"):
        import torch
        if (value.is_floating_point() or value.is_complex()) \
                and not bool(torch.isfinite(value).all()):
            raise AuditFailure(f"{label} contains a nonfinite tensor")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) \
                and not bool(np.isfinite(value).all()):
            raise AuditFailure(f"{label} contains a nonfinite array")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise AuditFailure(f"{label} is nonfinite")


def close(a: Any, b: Any, *, atol: float = 5e-15,
          rtol: float = 0.0) -> bool:
    try:
        return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)
    except (TypeError, ValueError, OverflowError):
        return False


def require_equal_numeric(a: Any, b: Any, label: str, *,
                          atol: float = 5e-15, rtol: float = 0.0) -> None:
    if not close(a, b, atol=atol, rtol=rtol):
        raise AuditFailure(f"{label} differs: {a!r} != {b!r}")


def preflight(repo: Path, root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Refuse before checkpoint/W&B access unless final markers are complete."""
    del repo
    summary_path = root / SUMMARY_NAME
    if not summary_path.is_file():
        raise AuditRefused(f"missing terminal summary {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditRefused(f"unreadable terminal summary: {exc}") from exc
    if not isinstance(summary, Mapping):
        raise AuditRefused("terminal summary is not a JSON object")
    wanted_summary = {
        "status": "COMPLETE",
        "expected_cells": EXPECTED_CELLS,
        "completed_cells": EXPECTED_CELLS,
        "failed_or_invalid_cells": 0,
        "wandb_enabled": True,
    }
    differences = [
        f"{key}={summary.get(key)!r} (expected {value!r})"
        for key, value in wanted_summary.items() if summary.get(key) != value
    ]
    if differences:
        raise AuditRefused("terminal summary is not 200/200 COMPLETE: " + "; ".join(differences))
    if (root / LOCK_NAME).exists():
        raise AuditRefused(f"runner lock still exists: {root / LOCK_NAME}")
    mandatory = (
        PROTOCOL_NAME, RUNS_NAME, ATTEMPTS_NAME, ANALYSIS_NAME,
        CELLS_NAME, CONTRASTS_NAME,
    )
    missing = [name for name in mandatory if not (root / name).is_file()]
    if missing:
        raise AuditRefused(
            "write-once runner/analysis bundle is not complete: " + ", ".join(missing))
    # The runner removes its lock before starting analysis, and the analyzer
    # writes its JSON completion marker non-atomically. Parse and validate that
    # marker here so an empty/partial file remains a refusal, not permission to
    # load checkpoints.
    try:
        analysis = json.loads((root / ANALYSIS_NAME).read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditRefused(f"analysis completion marker is unreadable/partial: {exc}") from exc
    if not isinstance(analysis, Mapping):
        raise AuditRefused("analysis completion marker is not a JSON object")
    analysis_exact = {
        "status": "COMPLETE",
        "expected_cells": EXPECTED_CELLS,
        "completed_valid_cells": EXPECTED_CELLS,
        "artifact_integrity_passed": True,
    }
    differences = [
        f"{key}={analysis.get(key)!r} (expected {value!r})"
        for key, value in analysis_exact.items() if analysis.get(key) != value]
    if analysis.get("scientific_label") not in {
            "STABILIZED_LEWM_V8_CONFIRMATION_PASS", "CONFIRMATION_FAILED"}:
        differences.append(f"scientific_label={analysis.get('scientific_label')!r}")
    try:
        cells_csv_sha = file_sha256(root / CELLS_NAME)
        contrasts_csv_sha = file_sha256(root / CONTRASTS_NAME)
    except OSError as exc:
        raise AuditRefused(f"analysis CSV disappeared/changed during preflight: {exc}") from exc
    if analysis.get("cells_csv_sha256") != cells_csv_sha:
        differences.append("cells_csv_sha256 does not match current CSV bytes")
    if analysis.get("contrasts_csv_sha256") != contrasts_csv_sha:
        differences.append("contrasts_csv_sha256 does not match current CSV bytes")
    if differences:
        raise AuditRefused("analysis bundle is not terminal/consistent: " + "; ".join(differences))
    try:
        runs = json.loads((root / RUNS_NAME).read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditRefused(f"unreadable final runs ledger: {exc}") from exc
    if not isinstance(runs, list) or len(runs) != EXPECTED_CELLS:
        raise AuditRefused(
            f"final runs ledger is not exactly {EXPECTED_CELLS} rows")
    keys: list[tuple[str, int, str]] = []
    try:
        for row in runs:
            if not isinstance(row, Mapping) or row.get("status") != "complete":
                raise ValueError("non-object or noncomplete row")
            keys.append((str(row["task"]), int(row["seed"]), str(row["design"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditRefused(f"malformed/noncomplete final runs ledger: {exc}") from exc
    if len(set(keys)) != EXPECTED_CELLS or set(keys) != expected_keys():
        raise AuditRefused("final runs ledger does not cover the exact frozen grid")
    missing_artifacts: list[str] = []
    for key in sorted(expected_keys()):
        directory = run_directory(root, key[0], key[2], key[1])
        for name in CORE_ARTIFACTS:
            path = directory / name
            try:
                present = path.is_file() and path.stat().st_size > 0
            except OSError:
                present = False
            if not present:
                missing_artifacts.append(f"{cell_label(key)}:{name}")
    if missing_artifacts:
        raise AuditRefused(
            f"{len(missing_artifacts)} terminal core artifacts are absent/empty; "
            f"first={missing_artifacts[:5]!r}")
    return dict(summary), [dict(row) for row in runs]


def expected_command(repo: Path, root: Path, protocol: Mapping[str, Any],
                     task: str, design: str, seed: int) -> list[str]:
    data = protocol["data"][task]
    return [
        str(protocol["python"]),
        str(repo / "scripts" / "train_lewm_v8_v18.py"),
        "--train-data", str(data["train"]),
        "--val-data", str(data["val"]),
        "--design", design,
        "--seed", str(seed),
        "--epochs", "100",
        "--output-dir", str(root),
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
        "--corruption-seed", "270711",
        "--eval-rollout-episode", "0",
        "--device", "cuda",
        "--wandb",
        "--wandb-entity", WANDB_ENTITY,
        "--wandb-project", WANDB_PROJECT,
        "--wandb-mode", "online",
        "--wandb-study", STUDY,
        "--extra-tag", "confirmation-grid,lewm-v8-v18,unopened-tasks",
    ]


def audit_protocol(repo: Path, root: Path) -> tuple[
        dict[str, Any], dict[tuple[str, int, str], list[str]], dict[str, Any]]:
    protocol_path = root / PROTOCOL_NAME
    protocol = load_json(protocol_path)
    if not isinstance(protocol, Mapping):
        raise AuditFailure("confirmation protocol is not an object")
    protocol = dict(protocol)
    exact = {
        "schema_version": 1,
        "scope": SCOPE,
        "tasks": list(TASKS),
        "designs": list(DESIGNS),
        "objective_families": ["vicreg_stable_lewm_host"],
        "memory_variants": [item.removeprefix("vicreg_") for item in DESIGNS],
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "runs": EXPECTED_CELLS,
        "gpus": ["0", "1", "2", "3"],
        "gpu_task_queues": {
            "0": ["acrobot.swingup", "stacker.stack_4"],
            "1": ["manipulator.bring_ball"],
            "2": ["quadruped.run"],
            "3": ["swimmer.swimmer15"],
        },
        "task_pinned_gpu": GPU_BY_TASK,
        "study": STUDY,
        "wandb_enabled": True,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_mode": "online",
        "blas_threads_per_process": 4,
        "core_artifacts": list(CORE_ARTIFACTS),
        "candidate_ssl_selectable_hyperparameters": [],
        "candidate_gradient_policy": "ordinary_joint_end_to_end_backpropagation",
        "primary_metric": PRIMARY,
        "secondary_metric": SECONDARY,
        "mechanism_controls": [NO_ACTION, SINGLE],
        "endpoint_controls": [STATIC, DYNAMIC],
        "recurrent_baselines": [GRU, SSM],
        "confirmation_requires_executed_return": False,
        "executed_return_claim_permitted": False,
        "data_opened_only_after_architecture_and_grid_freeze": True,
        "resume_supported": True,
        "resume_granularity": "complete_cell_only",
        "git_clean_required": True,
        "git_clean_or_pushed_required": True,
        "git_worktree_clean": True,
    }
    errors = [
        f"protocol {key} differs: {protocol.get(key)!r} != {value!r}"
        for key, value in exact.items() if protocol.get(key) != value
    ]
    if resolve_from(repo, str(protocol.get("output_root", ""))) != root:
        errors.append("protocol output_root does not resolve to audit root")
    python_path = resolve_from(repo, str(protocol.get("python", "")))
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        errors.append("protocol Python executable is missing/nonexecutable")
    log_root_raw = protocol.get("log_root")
    if not isinstance(log_root_raw, str) or not log_root_raw:
        errors.append("protocol log_root is absent")
    source = protocol.get("source_sha256")
    if not isinstance(source, Mapping) or set(source) != set(SOURCE_PATHS):
        errors.append("protocol source_sha256 does not have the exact 22-file key set")
    else:
        commit = protocol.get("git_commit")
        if not isinstance(commit, str) or not commit:
            errors.append("protocol git_commit is absent")
        for relative in SOURCE_PATHS:
            path = repo / relative
            if not path.is_file():
                errors.append(f"frozen source missing: {relative}")
                continue
            actual = file_sha256(path)
            if actual != source[relative]:
                errors.append(
                    f"frozen source hash differs: {relative}: {actual} != {source[relative]}")
            if isinstance(commit, str) and commit:
                result = subprocess.run(
                    ["git", "show", f"{commit}:{relative}"], cwd=repo,
                    capture_output=True, check=False)
                if result.returncode != 0:
                    errors.append(f"source is absent from protocol Git commit: {relative}")
                else:
                    committed = hashlib.sha256(result.stdout).hexdigest()
                    if committed != source[relative]:
                        errors.append(
                            f"protocol source is not bound to Git commit: {relative}")

    commands = protocol.get("commands")
    expected_records: list[dict[str, Any]] = []
    command_index: dict[tuple[str, int, str], list[str]] = {}
    if not isinstance(protocol.get("data"), Mapping):
        errors.append("protocol data is not an object")
    else:
        for task in TASKS:
            if not isinstance(protocol["data"].get(task), Mapping):
                errors.append(f"protocol data record is missing for {task}")
        if not errors or all(isinstance(protocol["data"].get(t), Mapping) for t in TASKS):
            for task in TASKS:
                for seed in SEEDS:
                    for design in DESIGNS:
                        argv = expected_command(repo, root, protocol, task, design, seed)
                        key = (task, seed, design)
                        command_index[key] = argv
                        expected_records.append({
                            "task": task, "design": design,
                            "seed": seed, "argv": argv,
                        })
    if commands != expected_records:
        errors.append("protocol commands differ from the independently rebuilt 200 commands")
    if protocol.get("commands_sha256") != json_sha256(expected_records):
        errors.append("protocol commands_sha256 differs")
    if errors:
        raise AuditFailure("; ".join(errors))
    return protocol, command_index, {
        "sha256": file_sha256(protocol_path),
        "sources_matched": len(SOURCE_PATHS),
        "commands_matched": len(expected_records),
        "git_commit": protocol["git_commit"],
    }


def _array_digest(name: str, value: Any) -> bytes:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(b"\0")
    # hashlib accepts a contiguous buffer directly; avoid an additional copy
    # of the ~700 MiB training RGB array solely for hashing.
    digest.update(memoryview(array).cast("B"))
    return digest.digest()


def audit_cache(path: Path, *, task: str, split: str,
                expected_file_sha: str, expected_content_sha: str) -> dict[str, Any]:
    if not path.is_file():
        raise AuditFailure(f"missing cache {path}")
    actual_file_sha = file_sha256(path)
    if actual_file_sha != expected_file_sha:
        raise AuditFailure(f"cache file SHA differs for {task}/{split}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != (
            f"{actual_file_sha}  {path.name}\n"):
        raise AuditFailure(f"cache sidecar differs for {task}/{split}")
    digest = hashlib.sha256()
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, np.dtype[Any]] = {}
    retained: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_CACHE_FIELDS:
            raise AuditFailure(f"cache field set differs for {task}/{split}")
        scalar = {
            name: np.asarray(archive[name])
            for name in (
                "schema_version", "action_process", "env_id", "split", "seed",
                "length", "img_size", "smooth_rho", "content_sha256")
        }
        for name in CONTENT_FIELDS:
            value = np.asarray(archive[name])
            if name in {
                    "obs", "actions", "task_observation", "physics_state",
                    "rewards", "action_min", "action_max"}:
                shapes[name] = tuple(value.shape)
                dtypes[name] = value.dtype
            if name in {
                    "actions", "task_observation", "physics_state", "rewards",
                    "action_min", "action_max", "task_observation_keys",
                    "task_observation_shape_offsets",
                    "task_observation_shape_values", "task_observation_slices"}:
                retained[name] = value
            if np.issubdtype(value.dtype, np.number) and name != "obs" \
                    and not bool(np.isfinite(value).all()):
                raise AuditFailure(f"cache contains nonfinite {name}: {task}/{split}")
            digest.update(_array_digest(name, value))
    content = digest.hexdigest()
    stored_content = str(scalar["content_sha256"])
    expected_seed = 270_701 if split == "train" else 270_702
    expected_episodes = 1_200 if split == "train" else 240
    exact = {
        "schema_version": int(scalar["schema_version"]),
        "action_process": str(scalar["action_process"]),
        "env_id": str(scalar["env_id"]),
        "split": str(scalar["split"]),
        "seed": int(scalar["seed"]),
        "length": int(scalar["length"]),
        "img_size": int(scalar["img_size"]),
        "smooth_rho": float(scalar["smooth_rho"]),
    }
    wanted = {
        "schema_version": 2,
        "action_process": ACTION_PROCESS,
        "env_id": task,
        "split": split,
        "seed": expected_seed,
        "length": 48,
        "img_size": 64,
        "smooth_rho": 0.0,
    }
    if exact != wanted:
        raise AuditFailure(f"cache metadata differs for {task}/{split}: {exact!r}")
    # Shapes were captured while hashing, without a second decompression pass.
    obs_shape = shapes["obs"]
    actions_shape = shapes["actions"]
    task_shape = shapes["task_observation"]
    physics_shape = shapes["physics_state"]
    rewards_shape = shapes["rewards"]
    if dtypes["obs"] != np.dtype(np.uint8) or len(obs_shape) != 5 \
            or obs_shape[:2] != (expected_episodes, 48) \
            or obs_shape[2:] != (64, 64, 3):
        raise AuditFailure(f"cache RGB shape differs for {task}/{split}: {obs_shape}")
    if len(actions_shape) != 3 or len(task_shape) != 3 or len(physics_shape) != 3 \
            or actions_shape[:2] != (expected_episodes, 47) \
            or task_shape[:2] != (expected_episodes, 48) \
            or physics_shape[:2] != (expected_episodes, 48) \
            or rewards_shape != (expected_episodes, 47) \
            or actions_shape[-1] < 1 or task_shape[-1] < 1 or physics_shape[-1] < 1 \
            or shapes["action_min"] != (actions_shape[-1],) \
            or shapes["action_max"] != (actions_shape[-1],):
        raise AuditFailure(f"cache trajectory shapes differ for {task}/{split}")
    for name in ("actions", "task_observation", "physics_state", "rewards",
                 "action_min", "action_max"):
        if not np.issubdtype(dtypes[name], np.number) \
                or not bool(np.isfinite(retained[name]).all()):
            raise AuditFailure(f"cache numeric dtype/finiteness differs: {task}/{split}/{name}")
    action_min = retained["action_min"]
    action_max = retained["action_max"]
    if not bool(np.all(action_max > action_min)):
        raise AuditFailure(f"cache action bounds differ for {task}/{split}")
    actions = retained["actions"]
    if bool(np.any(actions < action_min.reshape(1, 1, -1) - 1e-6)) \
            or bool(np.any(actions > action_max.reshape(1, 1, -1) + 1e-6)):
        raise AuditFailure(f"cache action lies outside bounds for {task}/{split}")
    keys_array = retained["task_observation_keys"]
    offsets = retained["task_observation_shape_offsets"]
    shape_values = retained["task_observation_shape_values"]
    slices = retained["task_observation_slices"]
    if keys_array.ndim != 1 or keys_array.dtype.kind not in {"U", "S"} \
            or len(keys_array) < 1:
        raise AuditFailure(f"cache task-observation keys differ for {task}/{split}")
    keys = tuple(str(value) for value in keys_array)
    if any(not value for value in keys) or len(set(keys)) != len(keys):
        raise AuditFailure(f"cache task-observation keys are invalid for {task}/{split}")
    if not np.issubdtype(offsets.dtype, np.integer) \
            or offsets.shape != (len(keys) + 1,) \
            or not np.issubdtype(shape_values.dtype, np.integer) \
            or shape_values.ndim != 1 \
            or not np.issubdtype(slices.dtype, np.integer) \
            or slices.shape != (len(keys), 2) \
            or offsets[0] != 0 or offsets[-1] != len(shape_values) \
            or bool(np.any(np.diff(offsets) < 0)):
        raise AuditFailure(f"cache task-observation schema differs for {task}/{split}")
    cursor = 0
    for index in range(len(keys)):
        shape = tuple(int(value) for value in shape_values[offsets[index]:offsets[index + 1]])
        if any(value <= 0 for value in shape):
            raise AuditFailure(f"cache has nonpositive task shape for {task}/{split}")
        width = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if tuple(int(value) for value in slices[index]) != (cursor, cursor + width):
            raise AuditFailure(f"cache task slices are noncontiguous for {task}/{split}")
        cursor += width
    if cursor != task_shape[-1]:
        raise AuditFailure(f"cache task schema width differs for {task}/{split}")
    if content != stored_content or content != expected_content_sha:
        raise AuditFailure(f"cache content SHA differs for {task}/{split}")
    return {
        "path": str(path), "sha256": actual_file_sha,
        "content_sha256": content, "bytes": path.stat().st_size,
        "_dimensions": {
            "action_dim": actions_shape[-1],
            "state_dim": physics_shape[-1],
            "eval_target_dim": task_shape[-1],
        },
    }


def audit_caches(repo: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    cohort = protocol["data"]
    if set(cohort) != {*TASKS, "__manifest__"}:
        raise AuditFailure("protocol cohort does not contain exactly five tasks and manifest")
    records: list[dict[str, Any]] = []
    by_path: dict[str, dict[str, Any]] = {}
    dimensions: dict[str, dict[str, int]] = {}
    for task in TASKS:
        record = cohort[task]
        if not isinstance(record, Mapping):
            raise AuditFailure(f"malformed protocol cache record for {task}")
        expected_record_keys = {
            "train", "train_sha256", "train_content_sha256",
            "val", "val_sha256", "val_content_sha256"}
        if set(record) != expected_record_keys:
            raise AuditFailure(f"protocol cache record key set differs for {task}")
        task_dimensions: list[dict[str, int]] = []
        for split in ("train", "val"):
            path = resolve_from(repo, str(record[split]))
            audited = audit_cache(
                path, task=task, split=split,
                expected_file_sha=str(record[f"{split}_sha256"]),
                expected_content_sha=str(record[f"{split}_content_sha256"]),
            )
            task_dimensions.append(audited.pop("_dimensions"))
            records.append(audited)
            by_path[str(path)] = audited
            print(
                f"[v18-audit] cache hash/content verified {len(records)}/10 "
                f"({task}/{split})", file=sys.stderr)
        if task_dimensions[0] != task_dimensions[1]:
            raise AuditFailure(f"train/val cache dimensions differ for {task}")
        dimensions[task] = task_dimensions[0]

    manifest_record = cohort["__manifest__"]
    if not isinstance(manifest_record, Mapping):
        raise AuditFailure("malformed protocol manifest record")
    if set(manifest_record) != {
            "path", "path_sha256", "sidecar", "sidecar_sha256"}:
        raise AuditFailure("protocol manifest record key set differs")
    manifest_path = resolve_from(repo, str(manifest_record["path"]))
    manifest_sidecar = resolve_from(repo, str(manifest_record["sidecar"]))
    manifest_sha = file_sha256(manifest_path)
    sidecar_sha = file_sha256(manifest_sidecar)
    if manifest_sha != manifest_record.get("path_sha256") \
            or sidecar_sha != manifest_record.get("sidecar_sha256"):
        raise AuditFailure("cohort manifest/sidecar hash differs from protocol")
    if manifest_sidecar.read_text(encoding="utf-8") != (
            f"{manifest_sha}  {manifest_path.name}\n"):
        raise AuditFailure("cohort manifest sidecar does not bind manifest bytes")
    manifest = load_json(manifest_path)
    collection = {
        "study": STUDY,
        "scope": "prospectively_frozen_unopened_task_confirmation",
        "tasks": list(TASKS),
        "splits": ["train", "val"],
        "train_episodes": 1_200,
        "val_episodes": 240,
        "length": 48,
        "img_size": 64,
        "train_seed": 270_701,
        "val_seed": 270_702,
        "smooth_rho": 0.0,
        "action_process": ACTION_PROCESS,
        "primary_evaluation_target": "flattened_native_task_observation",
        "secondary_evaluation_target": "raw_physics_state",
        "evaluation_targets_used_for_training": False,
        "cache_role": "clean_only_corruptions_are_deterministic_dataset_views",
        "corruption_seed": 270_711,
    }
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 2 \
            or manifest.get("protocol") != collection:
        raise AuditFailure("cohort manifest collection protocol differs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 10:
        raise AuditFailure("cohort manifest does not contain exactly ten artifacts")
    manifest_index = {
        str(Path(str(row.get("path", ""))).resolve()): row
        for row in artifacts if isinstance(row, Mapping)
    }
    if len(manifest_index) != 10:
        raise AuditFailure("cohort manifest contains malformed/duplicate artifacts")
    for path, audited in by_path.items():
        if manifest_index.get(path) != audited:
            raise AuditFailure(f"cohort manifest artifact record differs: {path}")
    return {
        "matched": len(records), "expected": 10,
        "manifest_sha256": manifest_sha,
        "manifest_sidecar_sha256": sidecar_sha,
        "dimensions": dimensions,
    }


def as_args_dict(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__") and isinstance(vars(value), dict):
        return vars(value)
    raise AuditFailure("checkpoint args are not a mapping/Namespace")


def recompute_convergence(history: Sequence[Mapping[str, Any]],
                          loss_name: str) -> float:
    if len(history) != EPOCHS:
        raise AuditFailure(f"history length is {len(history)}, expected {EPOCHS}")
    epochs = [row.get("epoch") if isinstance(row, Mapping) else None for row in history]
    if epochs != list(range(1, EPOCHS + 1)):
        raise AuditFailure("history epoch indices are not exactly 1..100")
    early = np.asarray(
        [finite_scalar(row["val"][loss_name], f"epoch {row['epoch']} val {loss_name}")
         for row in history[80:90]], dtype=np.float64)
    late = np.asarray(
        [finite_scalar(row["val"][loss_name], f"epoch {row['epoch']} val {loss_name}")
         for row in history[90:100]], dtype=np.float64)
    previous = float(np.mean(early))
    recent = float(np.mean(late))
    return float((previous - recent) / max(abs(previous), 1e-12))


def expected_design_metadata(design: str) -> dict[str, Any]:
    memory = design.removeprefix("vicreg_")
    is_v8 = memory.startswith("hacssmv8")
    correction = (
        "static" if memory == "hacssmv8_static" else
        "dynamic" if memory == "hacssmv8_dynamic" else
        "learned_shrinkage" if is_v8 else None)
    return {
        "method": "LeWM-SAS-PC-v18-confirmation",
        "evidence_scope": "unopened_task_confirmation",
        "wandb_method_tag": "lewm-v8-v18",
        "wandb_scope_tag": "unopened-task-confirmation",
        "confirmation_evidence": True,
        "executed_return_evaluation": False,
        "regularizer": "vicreg",
        "regularizer_family": "clean_target_variance_covariance",
        "regularizer_source": "active_clean_target",
        "memory_architecture": memory,
        "memory_specific_loss_weight": 0.0,
        "new_memory_architecture": False,
        "one_token_predictor": False,
        "predictor_history": 3,
        "predictor_window_policy": "all_aligned_length_h_windows",
        "evaluation_predictor_window_policy": "zero_padded_aligned_length_h_windows",
        "paired_clean_target": True,
        "hidden_clean_targets_included": True,
        "clean_target_gradient_active": True,
        "target_stop_gradient": False,
        "reward_used_for_training": False,
        "state_labels_used_for_training": False,
        "training_objective": TRAINING_OBJECTIVE,
        "unopened_task_cohort": True,
        "causal_action_timing": "a_t_maps_z_t_to_z_t_plus_1",
        "v8_action_transport_enabled": (
            memory != "hacssmv8_noaction" if is_v8 else None),
        "v8_joint_read_enabled": (
            memory != "hacssmv8_single" if is_v8 else None),
        "v8_correction_mode": correction,
        "memory_timescales": [2.0, 8.0] if is_v8 else [],
        "gru_probe_prior_contract": (
            "D_dimensional_read_of_h_t_minus_1_before_z_t"
            if memory == "gru" else None),
        "embedding_dimension": 128,
    }


def expected_metric_identity(task: str, design: str, seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "env": f"dmc:{task}",
        "design": design,
        "seed": seed,
        "epochs": EPOCHS,
        "encoder_type": "vit",
        "encoder_frozen": False,
        "encoder_norm": "causal",
        "predictor_norm": "none",
        "end_to_end_rgb": True,
        "train_episodes": 1_200,
        "val_episodes": 240,
        "length": 48,
        "prediction_loss_weight": 1.0,
        "sigreg_lambda": 0.0,
        "sigreg_projections_per_subspace": 512,
        "sigreg_quad_nodes": 17,
        "probe_ridge": 0.001,
        "headline_metric": PRIMARY,
        "eval_target_key": "task_observation",
        "eval_rollout_episode": 0,
        **expected_design_metadata(design),
    }


def audit_cell(
    repo: Path,
    root: Path,
    protocol: Mapping[str, Any],
    cache_dimensions: Mapping[str, Mapping[str, int]],
    command_index: Mapping[tuple[str, int, str], list[str]],
    ledger: Mapping[str, Any],
    key: tuple[str, int, str],
) -> dict[str, Any]:
    import torch

    task, seed, design = key
    label = cell_label(key)
    directory = run_directory(root, task, design, seed)
    paths = {name: directory / name for name in CORE_ARTIFACTS}
    # Bind bytes to the final ledger before deserializing the trusted-but-pickled
    # checkpoint payload.
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    if ledger.get("artifact_sha256") != hashes:
        raise AuditFailure(f"{label}: ledger artifact hashes differ before checkpoint load")
    metrics = load_json(paths["metrics.json"])
    if not isinstance(metrics, dict):
        raise AuditFailure(f"{label}: metrics.json is not an object")
    finite_tree(metrics, f"{label}.metrics")
    expected_identity = expected_metric_identity(task, design, seed)
    expected_identity.update({
        "train_data": str(resolve_from(repo, protocol["data"][task]["train"])),
        "val_data": str(resolve_from(repo, protocol["data"][task]["val"])),
        **cache_dimensions[task],
    })
    for name, expected in expected_identity.items():
        if metrics.get(name) != expected:
            raise AuditFailure(
                f"{label}: metrics identity {name}={metrics.get(name)!r}, expected {expected!r}")
    cache_record = protocol["data"][task]
    for split in ("train", "val"):
        for suffix in ("sha256", "content_sha256"):
            metric_key = f"{split}_data_{suffix}"
            protocol_key = f"{split}_{suffix}"
            if metrics.get(metric_key) != cache_record.get(protocol_key):
                raise AuditFailure(f"{label}: {metric_key} differs from protocol cache")
    required_scalars = (
        PRIMARY, CLEAN, SECONDARY, VARIANCE, RANK, CONVERGENCE, INTEGRATOR,
        "final_train_loss", "final_val_loss", "val_regularizer_loss",
        "mean_epoch_seconds",
        *(f"{condition}_prior_state_nmse_deep" for condition in CONDITIONS),
    )
    for name in required_scalars:
        finite_scalar(metrics.get(name), f"{label}.metrics.{name}")

    try:
        checkpoint = torch.load(
            paths["model.pt"], map_location="cpu", weights_only=False)
    except Exception as exc:
        raise AuditFailure(f"{label}: cannot load model.pt: {exc}") from exc
    if not isinstance(checkpoint, Mapping):
        raise AuditFailure(f"{label}: model.pt payload is not a mapping")
    checkpoint_keys = {
        "model_state_dict", "args", "final_metrics", "history",
        "state_probes", "inverse_action_probe",
    }
    if set(checkpoint) != checkpoint_keys:
        raise AuditFailure(
            f"{label}: checkpoint key set differs: "
            f"extra={sorted(set(checkpoint) - checkpoint_keys)}, "
            f"missing={sorted(checkpoint_keys - set(checkpoint))}")
    for name in ("model_state_dict", "state_probes", "inverse_action_probe"):
        if not isinstance(checkpoint[name], Mapping) or not checkpoint[name]:
            raise AuditFailure(f"{label}: checkpoint {name} is not a nonempty mapping")
    if not all(torch.is_tensor(value) for value in checkpoint["model_state_dict"].values()):
        raise AuditFailure(f"{label}: model_state_dict contains a non-tensor value")
    for probe_name, probe in checkpoint["state_probes"].items():
        if not isinstance(probe, Mapping) or not probe \
                or not all(isinstance(value, np.ndarray) for value in probe.values()):
            raise AuditFailure(f"{label}: state probe {probe_name} is malformed")
    if not all(isinstance(value, np.ndarray)
               for value in checkpoint["inverse_action_probe"].values()):
        raise AuditFailure(f"{label}: inverse_action_probe contains a non-array value")
    if checkpoint.get("final_metrics") != metrics:
        raise AuditFailure(f"{label}: checkpoint final_metrics != metrics.json")
    args = as_args_dict(checkpoint.get("args"))
    expected_args = {
        "train_data": str(protocol["data"][task]["train"]),
        "val_data": str(protocol["data"][task]["val"]),
        "design": design,
        "seed": seed,
        "epochs": EPOCHS,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "num_workers": 2,
        "img_size": 64,
        "patch_size": 8,
        "embed_dim": 128,
        "encoder_layers": 6,
        "encoder_heads": 4,
        "predictor_layers": 4,
        "predictor_heads": 8,
        "history_len": 3,
        "dropout": 0.1,
        "sigreg_lambda": 0.0,
        "sigreg_projections": 512,
        "sigreg_quad_nodes": 17,
        "probe_ridge": 0.001,
        "eval_target_key": "task_observation",
        "corruption_seed": 270_711,
        "eval_rollout_episode": 0,
        "no_amp": False,
        "device": "cuda",
        "wandb": True,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_mode": "online",
        "wandb_study": STUDY,
        "output_dir": str(root),
        "extra_tag": "confirmation-grid,lewm-v8-v18,unopened-tasks",
    }
    if set(args) != set(expected_args):
        raise AuditFailure(
            f"{label}: checkpoint arg key set differs: "
            f"extra={sorted(set(args) - set(expected_args))}, "
            f"missing={sorted(set(expected_args) - set(args))}")
    for name, expected in expected_args.items():
        if args.get(name) != expected:
            raise AuditFailure(
                f"{label}: checkpoint arg {name}={args.get(name)!r}, expected {expected!r}")

    history = checkpoint.get("history")
    if not isinstance(history, list):
        raise AuditFailure(f"{label}: checkpoint history is not a list")
    finite_tree(history, f"{label}.history")
    if [row.get("epoch") if isinstance(row, Mapping) else None for row in history] != (
            list(range(1, EPOCHS + 1))):
        raise AuditFailure(f"{label}: history epoch indices differ")
    for epoch_row in history:
        if not isinstance(epoch_row.get("train"), Mapping) \
                or not isinstance(epoch_row.get("val"), Mapping):
            raise AuditFailure(f"{label}: malformed epoch metrics")
        for namespace in ("train", "val"):
            if set(epoch_row[namespace]) != set(HISTORY_FIELDS):
                raise AuditFailure(
                    f"{label}: e{epoch_row['epoch']} {namespace} history key set differs")
            for name in HISTORY_FIELDS:
                finite_scalar(
                    epoch_row[namespace].get(name),
                    f"{label}.history.e{epoch_row['epoch']}.{namespace}.{name}")
        finite_scalar(epoch_row.get("epoch_seconds"), f"{label}.epoch_seconds")

    recomputed_convergence: dict[str, float] = {}
    convergence_scalar_errors: dict[str, float] = {}
    for loss_name in ("predictive_loss", "regularizer_loss", "loss"):
        value = recompute_convergence(history, loss_name)
        stored_name = f"{loss_name}_convergence_relative_change"
        require_equal_numeric(
            metrics.get(stored_name), value,
            f"{label}: stored versus checkpoint-derived {stored_name}")
        recomputed_convergence[stored_name] = value
        convergence_scalar_errors[stored_name] = abs(float(metrics[stored_name]) - value)
    require_equal_numeric(
        metrics["final_train_loss"], history[-1]["train"]["loss"],
        f"{label}: final_train_loss")
    require_equal_numeric(
        metrics["final_val_loss"], history[-1]["val"]["loss"],
        f"{label}: final_val_loss")
    require_equal_numeric(
        metrics[SECONDARY], history[-1]["val"]["predictive_loss"],
        f"{label}: val_predictive_loss")
    require_equal_numeric(
        metrics["val_regularizer_loss"], history[-1]["val"]["regularizer_loss"],
        f"{label}: val_regularizer_loss")
    require_equal_numeric(
        metrics["mean_epoch_seconds"],
        float(np.mean([row["epoch_seconds"] for row in history])),
        f"{label}: mean_epoch_seconds")
    for coordinate in ("prior", "posterior", "encoder", "predictor"):
        values = [metrics[f"{condition}_{coordinate}_state_nmse"]
                  for condition in CONDITIONS]
        require_equal_numeric(
            metrics[f"heldout_{coordinate}_state_nmse"], float(np.mean(values)),
            f"{label}: heldout_{coordinate}_state_nmse")
    deep = float(np.mean([
        metrics[f"{condition}_prior_state_nmse_deep"] for condition in CONDITIONS]))

    finite_tree(checkpoint.get("model_state_dict"), f"{label}.model_state_dict")
    finite_tree(checkpoint.get("state_probes"), f"{label}.state_probes")
    finite_tree(checkpoint.get("inverse_action_probe"), f"{label}.inverse_action_probe")

    try:
        with np.load(paths["eval_rollout.npz"], allow_pickle=False) as rollout:
            if not rollout.files:
                raise AuditFailure(f"{label}: rollout has no arrays")
            for name in rollout.files:
                value = rollout[name]
                if value.dtype.hasobject:
                    raise AuditFailure(f"{label}: rollout {name} has object dtype")
                if np.issubdtype(value.dtype, np.number) \
                        and not bool(np.isfinite(value).all()):
                    raise AuditFailure(f"{label}: rollout {name} is nonfinite")
    except (OSError, ValueError) as exc:
        raise AuditFailure(f"{label}: invalid rollout NPZ: {exc}") from exc
    rollout_sha = file_sha256(paths["eval_rollout.npz"])
    if metrics.get("eval_rollout_sha256") != rollout_sha:
        raise AuditFailure(f"{label}: metrics rollout SHA differs")

    receipt = load_json(paths["wandb_run.json"])
    if not isinstance(receipt, Mapping):
        raise AuditFailure(f"{label}: W&B receipt is not an object")
    exact_receipt = {
        "schema_version": 1,
        "state": "finished",
        "mode": "online",
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "study": STUDY,
        "run_name": f"{STUDY}-{run_name(task, design, seed)}",
        "eval_rollout_sha256": rollout_sha,
    }
    for name, expected in exact_receipt.items():
        if receipt.get(name) != expected:
            raise AuditFailure(f"{label}: W&B receipt {name} differs")
    run_id = receipt.get("run_id")
    url = receipt.get("url")
    artifact_name = receipt.get("eval_rollout_artifact_name")
    if not isinstance(run_id, str) or not run_id.strip() \
            or run_id.lower() in {"none", "null", "unknown"}:
        raise AuditFailure(f"{label}: invalid W&B run ID")
    if not isinstance(url, str) or run_id not in url:
        raise AuditFailure(f"{label}: invalid/unbound W&B URL")
    if artifact_name != f"eval-rollout-{run_id}":
        raise AuditFailure(f"{label}: W&B artifact name is not bound to run ID")
    expected_tags = Counter([
        "lewm-memory", "end-to-end-rgb", "lewm-v8-v18",
        "unopened-task-confirmation", f"env:dmc:{task}", f"design:{design}",
        f"study:{STUDY}", "confirmation-grid", "lewm-v8-v18",
        "unopened-tasks",
    ])
    if not isinstance(receipt.get("tags"), list) \
            or Counter(str(item) for item in receipt["tags"]) != expected_tags:
        raise AuditFailure(f"{label}: W&B receipt tag multiset differs")

    expected_command_hash = json_sha256(command_index[key])
    exact_ledger = {
        "task": task,
        "design": design,
        "seed": seed,
        "gpu": GPU_BY_TASK[task],
        "status": "complete",
        "command_sha256": expected_command_hash,
        "directory": str(directory),
        "wandb_state": "finished",
        "artifact_sha256": hashes,
    }
    for name, expected in exact_ledger.items():
        if ledger.get(name) != expected:
            raise AuditFailure(
                f"{label}: ledger {name}={ledger.get(name)!r}, expected {expected!r}")
    require_equal_numeric(
        ledger.get("headline_metric"), metrics[PRIMARY],
        f"{label}: ledger headline metric", atol=0.0)
    return {
        "task": task,
        "seed": seed,
        "design": design,
        "directory": str(directory),
        "metrics": metrics,
        DEEP: deep,
        "history": history,
        "args": dict(args),
        "recomputed_convergence": recomputed_convergence[CONVERGENCE],
        "recomputed_convergence_all": recomputed_convergence,
        "convergence_scalar_errors": convergence_scalar_errors,
        "artifact_sha256": hashes,
        "receipt": dict(receipt),
    }


def row_index(rows: Sequence[Mapping[str, Any]]) -> dict[
        tuple[str, int, str], Mapping[str, Any]]:
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["task"]), int(row["seed"]), str(row["design"]))
        if key in result:
            raise AuditFailure(f"duplicate analyzed cell {key}")
        result[key] = row
    if set(result) != expected_keys():
        raise AuditFailure("analyzed rows do not cover exact frozen grid")
    return result


def metric(row: Mapping[str, Any], name: str) -> float:
    if name == DEEP:
        return finite_scalar(row[DEEP], f"{row['task']}/{row['design']}/{DEEP}")
    return finite_scalar(
        row["metrics"].get(name), f"{row['task']}/{row['design']}/{name}")


def crossed_bootstrap(values: np.ndarray, *,
                      draws: int = BOOTSTRAP_DRAWS) -> dict[str, float | int]:
    if values.shape != (len(TASKS), len(SEEDS)) or not np.isfinite(values).all():
        raise AuditFailure(f"invalid crossed-bootstrap matrix {values.shape}")
    if draws <= 0:
        raise AuditFailure("bootstrap draw count must be positive")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    chunks: list[np.ndarray] = []
    remaining = draws
    while remaining:
        count = min(10_000, remaining)
        task_ids = rng.integers(0, values.shape[0], size=(count, values.shape[0]))
        seed_ids = rng.integers(0, values.shape[1], size=(count, values.shape[1]))
        sampled = values[task_ids[:, :, None], seed_ids[:, None, :]]
        chunks.append(sampled.mean(axis=(1, 2)))
        remaining -= count
    estimates = np.concatenate(chunks)
    return {
        "ci90_low": float(np.quantile(estimates, 0.05, method="linear")),
        "ci90_high": float(np.quantile(estimates, 0.95, method="linear")),
        "ci95_low": float(np.quantile(estimates, 0.025, method="linear")),
        "ci95_high": float(np.quantile(estimates, 0.975, method="linear")),
        "draws": draws,
        "seed": BOOTSTRAP_SEED,
    }


ReferenceSelector = Callable[[str, int], tuple[float, str]]


def selected_contrast(
    index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    *,
    candidate: str,
    reference_label: str,
    metric_name: str,
    select_reference: ReferenceSelector,
    draws: int,
) -> dict[str, Any]:
    matrix = np.empty((len(TASKS), len(SEEDS)), dtype=np.float64)
    candidate_values: dict[str, list[float]] = {task: [] for task in TASKS}
    reference_values: dict[str, list[float]] = {task: [] for task in TASKS}
    selected_counts: dict[str, int] = {}
    for task_index, task in enumerate(TASKS):
        for seed_index, seed in enumerate(SEEDS):
            cand = metric(index[(task, seed, candidate)], metric_name)
            ref, selected = select_reference(task, seed)
            ref = finite_scalar(ref, f"{task}/{reference_label}/{metric_name}")
            matrix[task_index, seed_index] = (ref - cand) / max(abs(ref), 1e-12)
            candidate_values[task].append(cand)
            reference_values[task].append(ref)
            selected_counts[selected] = selected_counts.get(selected, 0) + 1
    task_effects = {
        task: float(
            (np.mean(reference_values[task]) - np.mean(candidate_values[task]))
            / max(abs(np.mean(reference_values[task])), 1e-12))
        for task in TASKS
    }
    flat_candidate = [v for values in candidate_values.values() for v in values]
    flat_reference = [v for values in reference_values.values() for v in values]
    return {
        "candidate": candidate,
        "reference": reference_label,
        "metric": metric_name,
        "direction": "lower_is_better",
        "mean_paired_relative_reduction": float(matrix.mean()),
        "paired_wins": int((matrix > 0).sum()),
        "paired_ties": int((matrix == 0).sum()),
        "pairs": int(matrix.size),
        "task_mean_wins": int(sum(value > 0 for value in task_effects.values())),
        "task_effects": task_effects,
        "candidate_mean": float(np.mean(flat_candidate)),
        "reference_mean": float(np.mean(flat_reference)),
        "selected_reference_counts": selected_counts,
        "bootstrap": crossed_bootstrap(matrix, draws=draws),
        "cell_effects": matrix.tolist(),
    }


def paired_contrast(
        index: Mapping[tuple[str, int, str], Mapping[str, Any]],
        candidate: str, reference: str, metric_name: str,
        draws: int) -> dict[str, Any]:
    def select(task: str, seed: int) -> tuple[float, str]:
        return metric(index[(task, seed, reference)], metric_name), reference
    return selected_contrast(
        index, candidate=candidate, reference_label=reference,
        metric_name=metric_name, select_reference=select, draws=draws)


def envelope_contrast(
        index: Mapping[tuple[str, int, str], Mapping[str, Any]],
        candidate: str, references: Sequence[str], metric_name: str,
        *, label: str, draws: int) -> dict[str, Any]:
    references = tuple(references)
    def select(task: str, seed: int) -> tuple[float, str]:
        values = [
            (metric(index[(task, seed, reference)], metric_name), reference)
            for reference in references]
        return min(values, key=lambda item: (item[0], item[1]))
    result = selected_contrast(
        index, candidate=candidate, reference_label=label,
        metric_name=metric_name, select_reference=select, draws=draws)
    result["envelope_members"] = list(references)
    result["envelope_policy"] = "per_task_seed_lower_error"
    return result


def selected_identity_contrast(
        index: Mapping[tuple[str, int, str], Mapping[str, Any]],
        candidate: str, references: Sequence[str], metric_name: str,
        *, selection_metric: str, label: str, draws: int) -> dict[str, Any]:
    references = tuple(references)
    def select(task: str, seed: int) -> tuple[float, str]:
        chosen = min(
            references,
            key=lambda reference: (
                metric(index[(task, seed, reference)], selection_metric), reference))
        return metric(index[(task, seed, chosen)], metric_name), chosen
    result = selected_contrast(
        index, candidate=candidate, reference_label=label,
        metric_name=metric_name, select_reference=select, draws=draws)
    result["envelope_members"] = list(references)
    result["envelope_policy"] = "per_task_seed_identity_selected_once"
    result["selection_metric"] = selection_metric
    return result


def integrator_contrast(
        index: Mapping[tuple[str, int, str], Mapping[str, Any]],
        *, draws: int) -> dict[str, Any]:
    def select(task: str, seed: int) -> tuple[float, str]:
        value = index[(task, seed, CANDIDATE)]["metrics"][INTEGRATOR]
        return finite_scalar(value, f"{task}/{CANDIDATE}/{INTEGRATOR}"), (
            "checkpoint_matched_initial_frame_action_integrator")
    return selected_contrast(
        index, candidate=CANDIDATE,
        reference_label="checkpoint_matched_initial_frame_action_integrator",
        metric_name=PRIMARY, select_reference=select, draws=draws)


def superiority_receipt(contrast: Mapping[str, Any], *, magnitude: float,
                        paired_wins: int, task_wins: int,
                        require_positive_ci95: bool) -> dict[str, Any]:
    observed = {
        "mean_paired_relative_reduction": contrast["mean_paired_relative_reduction"],
        "paired_wins": contrast["paired_wins"],
        "task_mean_wins": contrast["task_mean_wins"],
        "ci95_low": contrast["bootstrap"]["ci95_low"],
    }
    thresholds = {
        "minimum_mean_paired_relative_reduction": magnitude,
        "minimum_paired_wins": paired_wins,
        "minimum_task_mean_wins": task_wins,
        "require_ci95_low_strictly_positive": require_positive_ci95,
    }
    passed = (
        observed["mean_paired_relative_reduction"] >= magnitude
        and observed["paired_wins"] >= paired_wins
        and observed["task_mean_wins"] >= task_wins
        and (not require_positive_ci95 or observed["ci95_low"] > 0.0)
    )
    return {"passed": bool(passed), "thresholds": thresholds, "observed": observed}


def compute_analysis(rows: Sequence[Mapping[str, Any]], *,
                     draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    index = row_index(rows)
    metrics_to_report = (PRIMARY, CLEAN, SECONDARY, DEEP)
    contrasts = {
        f"{CANDIDATE}_vs_{reference}:{metric_name}": paired_contrast(
            index, CANDIDATE, reference, metric_name, draws)
        for reference in DIRECT_REFERENCES
        for metric_name in metrics_to_report
    }
    recurrent_primary = selected_identity_contrast(
        index, CANDIDATE, RECURRENT_REFERENCES, PRIMARY,
        selection_metric=PRIMARY, label="per_cell_better_of_gru_ssm", draws=draws)
    recurrent_deep = selected_identity_contrast(
        index, CANDIDATE, RECURRENT_REFERENCES, DEEP,
        selection_metric=PRIMARY, label="per_cell_better_of_gru_ssm", draws=draws)
    recurrent_clean = selected_identity_contrast(
        index, CANDIDATE, RECURRENT_REFERENCES, CLEAN,
        selection_metric=PRIMARY, label="per_cell_better_of_gru_ssm", draws=draws)
    endpoint_primary = envelope_contrast(
        index, CANDIDATE, ENDPOINT_REFERENCES, PRIMARY,
        label="per_cell_better_of_static_dynamic", draws=draws)
    integrator = integrator_contrast(index, draws=draws)
    contrasts.update({
        f"{CANDIDATE}_vs_recurrent_envelope:{PRIMARY}": recurrent_primary,
        f"{CANDIDATE}_vs_recurrent_envelope:{DEEP}": recurrent_deep,
        f"{CANDIDATE}_vs_recurrent_envelope:{CLEAN}": recurrent_clean,
        f"{CANDIDATE}_vs_endpoint_envelope:{PRIMARY}": endpoint_primary,
        f"{CANDIDATE}_vs_checkpoint_integrator:{PRIMARY}": integrator,
    })
    if len(contrasts) != 33:
        raise AuditFailure(f"internal contrast registry produced {len(contrasts)}, expected 33")

    variance_values = [
        finite_scalar(row["metrics"][VARIANCE], f"{row['task']}/{row['design']}/{VARIANCE}")
        for row in rows]
    rank_values = [
        finite_scalar(row["metrics"][RANK], f"{row['task']}/{row['design']}/{RANK}")
        for row in rows]
    convergence_values = [
        abs(finite_scalar(
            row["recomputed_convergence"],
            f"{row['task']}/{row['design']}/recomputed_convergence"))
        for row in rows]

    recurrent_receipt = superiority_receipt(
        recurrent_primary, magnitude=0.03, paired_wins=18, task_wins=4,
        require_positive_ci95=True)
    none_receipt = superiority_receipt(
        contrasts[f"{CANDIDATE}_vs_{NONE}:{PRIMARY}"], magnitude=0.05,
        paired_wins=20, task_wins=4, require_positive_ci95=False)
    integrator_receipt = superiority_receipt(
        integrator, magnitude=0.03, paired_wins=18, task_wins=4,
        require_positive_ci95=False)
    action_receipt = superiority_receipt(
        contrasts[f"{CANDIDATE}_vs_{NO_ACTION}:{PRIMARY}"], magnitude=0.05,
        paired_wins=18, task_wins=4, require_positive_ci95=True)
    single_receipt = superiority_receipt(
        contrasts[f"{CANDIDATE}_vs_{SINGLE}:{PRIMARY}"], magnitude=0.03,
        paired_wins=18, task_wins=4, require_positive_ci95=True)
    deep_receipt = {
        "passed": bool(recurrent_deep["bootstrap"]["ci95_low"] > 0.0
                       and recurrent_deep["task_mean_wins"] >= 3),
        "thresholds": {
            "require_ci95_low_strictly_positive": True,
            "minimum_task_mean_wins": 3,
        },
        "observed": {
            "mean_paired_relative_reduction": recurrent_deep[
                "mean_paired_relative_reduction"],
            "task_mean_wins": recurrent_deep["task_mean_wins"],
            "ci95_low": recurrent_deep["bootstrap"]["ci95_low"],
        },
    }
    endpoint_receipt = {
        "passed": bool(
            endpoint_primary["mean_paired_relative_reduction"] >= -0.01
            and endpoint_primary["bootstrap"]["ci95_low"] >= -0.01),
        "thresholds": {
            "minimum_mean_paired_relative_reduction": -0.01,
            "minimum_ci95_low": -0.01,
            "interpretation": "learned V8 noninferior within one percent",
        },
        "observed": {
            "mean_paired_relative_reduction": endpoint_primary[
                "mean_paired_relative_reduction"],
            "ci95_low": endpoint_primary["bootstrap"]["ci95_low"],
        },
    }
    clean_degradation = -float(recurrent_clean["mean_paired_relative_reduction"])
    clean_receipt = {
        "passed": bool(clean_degradation <= 0.03),
        "thresholds": {"maximum_mean_paired_relative_degradation": 0.03},
        "observed": {
            "mean_paired_relative_degradation": clean_degradation,
            "mean_paired_relative_reduction": recurrent_clean[
                "mean_paired_relative_reduction"],
            "ci95_low_reduction": recurrent_clean["bootstrap"]["ci95_low"],
            "ci95_high_reduction": recurrent_clean["bootstrap"]["ci95_high"],
        },
    }
    representation_receipt = {
        "passed": bool(min(variance_values) >= 1e-4 and min(rank_values) >= 16.0),
        "thresholds": {
            "minimum_every_cell_channel_variance": 1e-4,
            "minimum_every_cell_effective_rank": 16.0,
        },
        "observed": {
            "cells": len(rows),
            "minimum_channel_variance": min(variance_values),
            "minimum_effective_rank": min(rank_values),
            "variance_passing_cells": sum(value >= 1e-4 for value in variance_values),
            "rank_passing_cells": sum(value >= 16.0 for value in rank_values),
        },
    }
    convergence_receipt = {
        "passed": bool(max(convergence_values) <= 0.05),
        "thresholds": {"maximum_every_cell_absolute_relative_change": 0.05},
        "observed": {
            "cells": len(rows),
            "maximum_absolute_relative_change": max(convergence_values),
            "passing_cells": sum(value <= 0.05 for value in convergence_values),
        },
    }
    gate_receipts = {
        "v8_vs_per_cell_better_gru_ssm": recurrent_receipt,
        "v8_vs_none": none_receipt,
        "v8_vs_checkpoint_integrator": integrator_receipt,
        "action_causality": action_receipt,
        "joint_state_use": single_receipt,
        "deep_vs_per_cell_better_gru_ssm": deep_receipt,
        "learned_v8_vs_static_dynamic_envelope_noninferiority": endpoint_receipt,
        "clean_prior_guard_vs_per_cell_better_gru_ssm": clean_receipt,
        "healthy_representation": representation_receipt,
        "convergence": convergence_receipt,
    }
    gates = {name: bool(receipt["passed"]) for name, receipt in gate_receipts.items()}
    gates["integrity"] = True
    passed = all(gates.values())
    return {
        "contrasts": contrasts,
        "gate_receipts": gate_receipts,
        "gates": gates,
        "representation": representation_receipt,
        "convergence": convergence_receipt,
        "integrator_guard": integrator,
        "scientific_label": (
            "STABILIZED_LEWM_V8_CONFIRMATION_PASS" if passed
            else "CONFIRMATION_FAILED"),
        "official_confirmation_result": passed,
    }


def compare_tree(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AuditFailure(
                f"{label}: mapping keys differ: actual={sorted(actual) if isinstance(actual, Mapping) else type(actual)} "
                f"expected={sorted(expected)}")
        for key in expected:
            compare_tree(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AuditFailure(f"{label}: list shape differs")
        for index, value in enumerate(expected):
            compare_tree(actual[index], value, f"{label}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise AuditFailure(f"{label}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (int, float, np.integer, np.floating)):
        if not close(actual, expected, atol=1e-15, rtol=1e-12):
            raise AuditFailure(f"{label}: numeric value differs: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise AuditFailure(f"{label}: value differs: {actual!r} != {expected!r}")


def artifact_manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    entries = [{
        "task": str(row["task"]),
        "seed": int(row["seed"]),
        "design": str(row["design"]),
        "artifact_sha256": row["artifact_sha256"],
    } for row in rows]
    entries.sort(key=lambda row: (row["task"], row["seed"], row["design"]))
    return json_sha256(entries)


def audit_attempts(
        root: Path,
        command_index: Mapping[tuple[str, int, str], list[str]],
        rows_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]]) -> dict[str, Any]:
    attempts = load_json(root / ATTEMPTS_NAME)
    if not isinstance(attempts, list) or not all(isinstance(row, Mapping) for row in attempts):
        raise AuditFailure("confirmation_attempts.json is not an object list")
    status_counts: Counter[str] = Counter()
    complete_counts: Counter[tuple[str, int, str]] = Counter()
    for ordinal, attempt in enumerate(attempts):
        try:
            key = (
                str(attempt["task"]), int(attempt["seed"]),
                str(attempt["design"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditFailure(f"attempt row {ordinal} has malformed identity") from exc
        if key not in command_index:
            raise AuditFailure(f"attempt row {ordinal} is outside frozen grid: {key}")
        status = str(attempt.get("status"))
        if status not in {"complete", "failed"}:
            raise AuditFailure(f"attempt row {ordinal} has invalid status {status!r}")
        status_counts[status] += 1
        if attempt.get("gpu") != GPU_BY_TASK[key[0]]:
            raise AuditFailure(f"attempt GPU differs for {cell_label(key)}")
        expected_directory = str(Path(rows_by_key[key]["directory"]))
        if attempt.get("directory") != expected_directory:
            raise AuditFailure(f"attempt directory differs for {cell_label(key)}")
        if attempt.get("command_sha256") != json_sha256(command_index[key]):
            raise AuditFailure(f"attempt command SHA differs for {cell_label(key)}")
        if status == "complete":
            complete_counts[key] += 1
            if attempt.get("artifact_sha256") != rows_by_key[key]["artifact_sha256"]:
                raise AuditFailure(
                    f"complete attempt artifact hashes differ for {cell_label(key)}")
            if attempt.get("wandb_state") != "finished":
                raise AuditFailure(f"complete attempt W&B state differs for {cell_label(key)}")
        elif "artifact_sha256" in attempt:
            raise AuditFailure(f"failed attempt unexpectedly claims artifact hashes: {cell_label(key)}")
    duplicated = [cell_label(key) for key, count in complete_counts.items() if count > 1]
    if duplicated:
        raise AuditFailure(f"multiple complete attempts recorded: {duplicated[:5]!r}")
    return {
        "rows": len(attempts),
        "status_counts": dict(sorted(status_counts.items())),
        "cells_with_complete_attempt": len(complete_counts),
        "note": "current complete runs ledger is authoritative; resume adoption may lack an attempt row",
    }


def audit_analysis_bundle(
        root: Path, protocol_sha: str, rows: Sequence[Mapping[str, Any]],
        computed: Mapping[str, Any]) -> dict[str, Any]:
    report = load_json(root / ANALYSIS_NAME)
    if not isinstance(report, Mapping):
        raise AuditFailure("confirmation_analysis.json is not an object")
    exact = {
        "schema_version": 2,
        "scope": "lewm_v8_v18_unopened_task_confirmation",
        "expected_cells": EXPECTED_CELLS,
        "completed_valid_cells": EXPECTED_CELLS,
        "artifact_integrity_passed": True,
        "artifact_integrity_errors": [],
        "protocol_contract_errors": [],
        "status": "COMPLETE",
        "primary_metric": PRIMARY,
        "input_protocol_sha256": protocol_sha,
        "input_artifact_manifest_sha256": artifact_manifest_sha256(rows),
        "cells_csv_sha256": file_sha256(root / CELLS_NAME),
        "contrasts_csv_sha256": file_sha256(root / CONTRASTS_NAME),
        "scientific_label": computed["scientific_label"],
        "official_confirmation_result": computed["official_confirmation_result"],
        "frozen_grid": {
            "tasks": 5, "designs": 8, "seeds": 5, "epochs": 100,
            "cells": 200, "task_ids": list(TASKS), "seed_ids": list(SEEDS),
            "design_ids": list(DESIGNS),
        },
        "claim_boundary": (
            "V18 licenses only a persistent causal-memory claim for the stabilized "
            "VICReg LeWM host on the frozen partial-observation cohort. It licenses no "
            "executed-return, planning, original-SIGReg, learned-timescale, semantic-"
            "hierarchy, or calibrated-uncertainty claim."),
    }
    for name, expected in exact.items():
        if report.get(name) != expected:
            raise AuditFailure(
                f"official analysis {name}={report.get(name)!r}, expected {expected!r}")
    for name in (
            "contrasts", "gate_receipts", "gates", "representation",
            "convergence", "integrator_guard"):
        compare_tree(report.get(name), computed[name], f"official_analysis.{name}")

    with (root / CELLS_NAME).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected_cell_fields = [
            "task", "seed", "design", PRIMARY, CLEAN, SECONDARY, DEEP,
            VARIANCE, RANK, CONVERGENCE, INTEGRATOR]
        if reader.fieldnames != expected_cell_fields:
            raise AuditFailure("confirmation_cells.csv header/order differs")
        csv_rows = list(reader)
    if len(csv_rows) != EXPECTED_CELLS:
        raise AuditFailure("confirmation_cells.csv does not have exactly 200 rows")
    csv_index: dict[tuple[str, int, str], Mapping[str, str]] = {}
    for csv_row in csv_rows:
        try:
            key = (str(csv_row["task"]), int(csv_row["seed"]), str(csv_row["design"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditFailure("malformed cell CSV identity") from exc
        if key in csv_index:
            raise AuditFailure(f"duplicate cell CSV key {key}")
        csv_index[key] = csv_row
    rows_by_key = {
        (str(row["task"]), int(row["seed"]), str(row["design"])): row
        for row in rows}
    if set(csv_index) != expected_keys():
        raise AuditFailure("cell CSV key set differs from frozen grid")
    expected_order = [
        (str(row["task"]), int(row["seed"]), str(row["design"])) for row in rows]
    actual_order = [
        (str(row["task"]), int(row["seed"]), str(row["design"])) for row in csv_rows]
    if actual_order != expected_order:
        raise AuditFailure("cell CSV row order differs from frozen order")
    scalar_fields = (PRIMARY, CLEAN, SECONDARY, DEEP, VARIANCE, RANK, CONVERGENCE, INTEGRATOR)
    for key, csv_row in csv_index.items():
        row = rows_by_key[key]
        for field in scalar_fields:
            expected = (
                row[DEEP] if field == DEEP
                else row["recomputed_convergence"] if field == CONVERGENCE
                else row["metrics"][field])
            require_equal_numeric(
                csv_row.get(field), expected,
                f"cell CSV {cell_label(key)} {field}", atol=5e-15)

    with (root / CONTRASTS_NAME).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected_contrast_fields = [
            "contrast", "metric", "mean_paired_relative_reduction",
            "paired_wins", "pairs", "task_mean_wins", "ci95_low", "ci95_high"]
        if reader.fieldnames != expected_contrast_fields:
            raise AuditFailure("confirmation_contrasts.csv header/order differs")
        contrast_rows = list(reader)
    if len(contrast_rows) != 33:
        raise AuditFailure("confirmation_contrasts.csv does not have exactly 33 rows")
    contrast_index = {row.get("contrast"): row for row in contrast_rows}
    if len(contrast_index) != 33 or set(contrast_index) != set(computed["contrasts"]):
        raise AuditFailure("contrast CSV key set differs from 33 registered contrasts")
    if [row["contrast"] for row in contrast_rows] != list(computed["contrasts"]):
        raise AuditFailure("contrast CSV row order differs from registered order")
    for name, expected in computed["contrasts"].items():
        row = contrast_index[name]
        if row.get("metric") != expected["metric"]:
            raise AuditFailure(f"contrast CSV metric differs for {name}")
        for field, value in {
            "mean_paired_relative_reduction": expected["mean_paired_relative_reduction"],
            "paired_wins": expected["paired_wins"],
            "pairs": expected["pairs"],
            "task_mean_wins": expected["task_mean_wins"],
            "ci95_low": expected["bootstrap"]["ci95_low"],
            "ci95_high": expected["bootstrap"]["ci95_high"],
        }.items():
            require_equal_numeric(
                row.get(field), value, f"contrast CSV {name} {field}",
                atol=1e-15, rtol=1e-12)
    return {
        "agrees": True,
        "analysis_sha256": file_sha256(root / ANALYSIS_NAME),
        "cells_csv_sha256": exact["cells_csv_sha256"],
        "contrasts_csv_sha256": exact["contrasts_csv_sha256"],
        "cell_rows": len(csv_rows),
        "contrast_rows": len(contrast_rows),
    }


def contrast_summaries(contrasts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "metric": value["metric"],
            "mean_paired_relative_reduction": value[
                "mean_paired_relative_reduction"],
            "paired_wins": value["paired_wins"],
            "paired_ties": value["paired_ties"],
            "pairs": value["pairs"],
            "task_mean_wins": value["task_mean_wins"],
            "ci95_low": value["bootstrap"]["ci95_low"],
            "ci95_high": value["bootstrap"]["ci95_high"],
            "selected_reference_counts": value["selected_reference_counts"],
            "task_effects": value["task_effects"],
        }
        for name, value in contrasts.items()
    }


def audit_local(repo: Path, root: Path, summary: Mapping[str, Any],
                runs: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol, command_index, protocol_result = audit_protocol(repo, root)
    cache_result = audit_caches(repo, protocol)
    ledger_index: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for ledger in runs:
        key = (str(ledger["task"]), int(ledger["seed"]), str(ledger["design"]))
        if key in ledger_index:
            raise AuditFailure(f"duplicate final ledger key {key}")
        ledger_index[key] = ledger
    rows: list[dict[str, Any]] = []
    ordered_keys = [
        (task, seed, design)
        for task in TASKS for seed in SEEDS for design in DESIGNS]
    for ordinal, key in enumerate(ordered_keys, 1):
        rows.append(audit_cell(
            repo, root, protocol, cache_result["dimensions"],
            command_index, ledger_index[key], key))
        if ordinal % 10 == 0 or ordinal == EXPECTED_CELLS:
            print(f"[v18-audit] locally validated {ordinal}/{EXPECTED_CELLS} cells", file=sys.stderr)

    run_ids = [str(row["receipt"]["run_id"]) for row in rows]
    run_names = [str(row["receipt"]["run_name"]) for row in rows]
    urls = [str(row["receipt"]["url"]) for row in rows]
    artifact_names = [str(row["receipt"]["eval_rollout_artifact_name"]) for row in rows]
    for label, values in (
            ("W&B run IDs", run_ids), ("W&B run names", run_names),
            ("W&B URLs", urls), ("W&B artifact names", artifact_names)):
        if len(set(values)) != EXPECTED_CELLS:
            raise AuditFailure(f"{label} are not unique across 200 cells")
    expected_models = {
        run_directory(root, task, design, seed) / "model.pt"
        for task, seed, design in expected_keys()}
    actual_models = set(root.rglob("model.pt"))
    if actual_models != expected_models:
        raise AuditFailure(
            f"unexpected/missing model.pt paths: extra={sorted(actual_models - expected_models)[:5]}, "
            f"missing={sorted(expected_models - actual_models)[:5]}")
    expected_run_dirs = {path.parent for path in expected_models}
    actual_top_level_dirs = {path for path in root.iterdir() if path.is_dir()}
    if actual_top_level_dirs != expected_run_dirs:
        raise AuditFailure(
            f"top-level run-directory set differs: "
            f"extra={sorted(actual_top_level_dirs - expected_run_dirs)[:5]}, "
            f"missing={sorted(expected_run_dirs - actual_top_level_dirs)[:5]}")
    rows_by_key = {
        (row["task"], row["seed"], row["design"]): row for row in rows}
    attempts_result = audit_attempts(root, command_index, rows_by_key)

    print("[v18-audit] recomputing 33 registered contrasts", file=sys.stderr)
    computed = compute_analysis(rows)
    analysis_result = audit_analysis_bundle(
        root, protocol_result["sha256"], rows, computed)

    representation_failures = [{
        "task": row["task"], "seed": row["seed"], "design": row["design"],
        "variance": row["metrics"][VARIANCE], "rank": row["metrics"][RANK],
    } for row in rows if (
        row["metrics"][VARIANCE] < 1e-4 or row["metrics"][RANK] < 16.0)]
    convergence_failures = [{
        "task": row["task"], "seed": row["seed"], "design": row["design"],
        "relative_change": row["recomputed_convergence"],
        "absolute_relative_change": abs(row["recomputed_convergence"]),
    } for row in rows if abs(row["recomputed_convergence"]) > 0.05]
    worst_convergence = sorted(
        ({
            "task": row["task"], "seed": row["seed"], "design": row["design"],
            "relative_change": row["recomputed_convergence"],
            "absolute_relative_change": abs(row["recomputed_convergence"]),
        } for row in rows),
        key=lambda item: item["absolute_relative_change"], reverse=True)[:10]

    local = {
        "preflight": {
            "summary_complete": True,
            "summary_sha256": file_sha256(root / SUMMARY_NAME),
            "cells": EXPECTED_CELLS,
            "analysis_bundle_complete": True,
            "finished_at": summary.get("finished_at"),
        },
        "integrity": {
            "passed": True,
            "protocol": protocol_result,
            "caches": cache_result,
            "ledger": {"complete": len(runs), "expected": EXPECTED_CELLS},
            "attempts": attempts_result,
            "artifacts": {
                "cells_matched": len(rows),
                "files_matched": len(rows) * len(CORE_ARTIFACTS),
                "expected_files": EXPECTED_CELLS * len(CORE_ARTIFACTS),
                "manifest_sha256": artifact_manifest_sha256(rows),
            },
            "wandb_local": {
                "finished": len(rows), "unique_run_ids": len(set(run_ids)),
                "unique_artifact_names": len(set(artifact_names)),
            },
            "analysis_hash_binding": analysis_result,
        },
        "metric_agreement": {
            "checkpoint_vs_json": len(rows),
            "history_derived_scalars": len(rows),
            "convergence_stored_agreement": len(rows),
            "cells_csv": analysis_result["cell_rows"],
            "contrasts_csv": analysis_result["contrast_rows"],
            "maximum_convergence_scalar_error": max(
                error for row in rows for error in row["convergence_scalar_errors"].values()),
        },
        "representation": {
            **computed["representation"]["observed"],
            "finite": len(rows),
            "both_passing_cells": sum(
                row["metrics"][VARIANCE] >= 1e-4 and row["metrics"][RANK] >= 16.0
                for row in rows),
            "failures": representation_failures,
        },
        "convergence": {
            **computed["convergence"]["observed"],
            "source": "model.pt/history",
            "early_epochs": [81, 90],
            "late_epochs": [91, 100],
            "finite": len(rows),
            "stored_agreement": len(rows),
            "failures": convergence_failures,
            "worst_cells": worst_convergence,
            "maximum_absolute_regularizer_loss_relative_change": max(
                abs(row["recomputed_convergence_all"][
                    "regularizer_loss_convergence_relative_change"])
                for row in rows),
            "maximum_absolute_total_loss_relative_change": max(
                abs(row["recomputed_convergence_all"][
                    "loss_convergence_relative_change"])
                for row in rows),
        },
        "gates": {
            "integrity": {"passed": True},
            **computed["gate_receipts"],
        },
        "scientific_label": computed["scientific_label"],
        "official_confirmation_result": computed["official_confirmation_result"],
        "official_analysis_agrees": True,
        "contrasts": {
            "registered": 33,
            "recomputed": 33,
            "items": contrast_summaries(computed["contrasts"]),
        },
    }
    return local, rows


def remote_scalar_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected or actual == expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return close(actual, expected, atol=1e-12, rtol=1e-10)
    return actual == expected


def remote_query_indicates_missing_evidence(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in (
        "could not find run", "run not found", "404", "does not exist"))


def audit_wandb(rows: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    if mode == "none":
        return {
            "mode": "none", "status": "not_requested",
            "note": "local hashed W&B receipts were verified; cloud was not queried",
        }
    for name, value in {
        "WANDB_CACHE_DIR": "/tmp/v18-wandb-cache",
        "WANDB_CONFIG_DIR": "/tmp/v18-wandb-config",
        "WANDB_DATA_DIR": "/tmp/v18-wandb-data",
        "WANDB_ARTIFACT_DIR": "/tmp/v18-wandb-artifacts",
        "WANDB_DIR": "/tmp/v18-wandb-runs",
        "WANDB_SILENT": "true",
    }.items():
        os.environ[name] = value
    try:
        import wandb
    except Exception as exc:
        raise RemoteAuditFailure(f"cannot import wandb for requested cloud audit: {exc}") from exc
    try:
        api = wandb.Api(timeout=90)
    except Exception as exc:
        raise RemoteAuditFailure(f"cannot initialize W&B API: {exc}") from exc

    state_matched = 0
    histories_matched = 0
    summaries_matched = 0
    artifacts_matched = 0
    mismatches: list[str] = []
    unavailable: list[str] = []
    for ordinal, row in enumerate(rows, 1):
        task = str(row["task"])
        design = str(row["design"])
        seed = int(row["seed"])
        label = f"{task}/{design}/s{seed}"
        receipt = row["receipt"]
        run_id = str(receipt["run_id"])
        try:
            remote = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        except Exception as exc:
            detail = f"{label}/{run_id}: {type(exc).__name__}: {exc}"
            if remote_query_indicates_missing_evidence(exc):
                mismatches.append(detail)
            else:
                unavailable.append(detail)
            continue
        try:
            exact = {
                "state": "finished",
                "name": receipt["run_name"],
                "group": f"{STUDY}:dmc:{task}",
                "job_type": design,
                "entity": WANDB_ENTITY,
                "project": WANDB_PROJECT,
            }
            for name, expected in exact.items():
                if getattr(remote, name, None) != expected:
                    raise AuditFailure(
                        f"cloud run {name}={getattr(remote, name, None)!r}, expected {expected!r}")
            required_tags = {
                "lewm-memory", "end-to-end-rgb", "lewm-v8-v18",
                "unopened-task-confirmation", f"env:dmc:{task}",
                f"design:{design}", f"study:{STUDY}", "confirmation-grid",
                "unopened-tasks",
            }
            if not required_tags.issubset(set(str(item) for item in remote.tags)):
                raise AuditFailure("cloud run tags differ")
            config = dict(remote.config)
            expected_config = {
                **row["args"],
                "env": f"dmc:{task}",
                "action_dim": row["metrics"]["action_dim"],
                "state_dim": row["metrics"]["state_dim"],
                "eval_target_dim": row["metrics"]["eval_target_dim"],
                "prediction_loss_weight": 1.0,
                "sigreg_loss_weight": 0.0,
                "variance_loss_weight": 1.0,
                "covariance_loss_weight": 1.0,
                **expected_design_metadata(design),
            }
            if set(config) != set(expected_config):
                raise AuditFailure(
                    f"cloud config key set differs: "
                    f"extra={sorted(set(config) - set(expected_config))}, "
                    f"missing={sorted(set(expected_config) - set(config))}")
            for name, expected in expected_config.items():
                if not remote_scalar_equal(config.get(name), expected):
                    raise AuditFailure(
                        f"cloud config {name}={config.get(name)!r}, expected {expected!r}")
            state_matched += 1

            if mode == "full":
                history = row["history"]
                train_fields = sorted(HISTORY_FIELDS)
                val_fields = sorted(HISTORY_FIELDS)
                keys = ["epoch"] + [f"train/{name}" for name in train_fields] + [
                    f"val/{name}" for name in val_fields]
                cloud_rows = list(remote.scan_history(
                    keys=keys, page_size=1000, use_cache=False))
                cloud_by_epoch: dict[int, Mapping[str, Any]] = {}
                for cloud_row in cloud_rows:
                    epoch_raw = cloud_row.get("epoch")
                    if epoch_raw is None:
                        continue
                    epoch = int(epoch_raw)
                    if epoch in cloud_by_epoch:
                        raise AuditFailure(f"duplicate cloud history epoch {epoch}")
                    cloud_by_epoch[epoch] = cloud_row
                if set(cloud_by_epoch) != set(range(1, EPOCHS + 1)):
                    raise AuditFailure(
                        f"cloud epoch set differs: got {len(cloud_by_epoch)} rows")
                for checkpoint_row in history:
                    cloud_row = cloud_by_epoch[int(checkpoint_row["epoch"])]
                    for namespace, fields in (("train", train_fields), ("val", val_fields)):
                        for name in fields:
                            expected = checkpoint_row[namespace][name]
                            actual = cloud_row.get(f"{namespace}/{name}")
                            if not remote_scalar_equal(actual, expected):
                                raise AuditFailure(
                                    f"cloud history e{checkpoint_row['epoch']} "
                                    f"{namespace}/{name} differs")
                histories_matched += 1

                summary = dict(remote.summary)
                for name, expected in row["metrics"].items():
                    if isinstance(expected, bool) or (
                            isinstance(expected, (int, float))
                            and math.isfinite(float(expected))):
                        if name not in summary or not remote_scalar_equal(summary[name], expected):
                            raise AuditFailure(f"cloud summary scalar {name} differs")
                summaries_matched += 1

                artifact_base = str(receipt["eval_rollout_artifact_name"])
                logged = list(remote.logged_artifacts())
                matches = [
                    artifact for artifact in logged
                    if str(artifact.name).rsplit("/", 1)[-1].split(":", 1)[0]
                    == artifact_base]
                if len(matches) != 1:
                    raise AuditFailure(
                        f"expected one logged artifact {artifact_base}, found {len(matches)}")
                artifact = matches[0]
                if getattr(artifact, "type", None) != "evaluation-rollout":
                    raise AuditFailure("cloud artifact type differs")
                metadata = dict(artifact.metadata or {})
                expected_metadata = {
                    "schema_version": 2,
                    "study": STUDY,
                    "env": f"dmc:{task}",
                    "design": design,
                    "seed": seed,
                    "episode": 0,
                    "sha256": row["metrics"]["eval_rollout_sha256"],
                    **expected_design_metadata(design),
                }
                if set(metadata) != set(expected_metadata):
                    raise AuditFailure(
                        f"cloud artifact metadata key set differs: "
                        f"extra={sorted(set(metadata) - set(expected_metadata))}, "
                        f"missing={sorted(set(expected_metadata) - set(metadata))}")
                for name, expected in expected_metadata.items():
                    if not remote_scalar_equal(metadata.get(name), expected):
                        raise AuditFailure(f"cloud artifact metadata {name} differs")
                with tempfile.TemporaryDirectory(
                        prefix=f"v18-wandb-{run_id}-", dir="/tmp") as temporary:
                    manifest_paths = set(str(path) for path in artifact.manifest.entries)
                    if manifest_paths != {"eval_rollout.npz"}:
                        raise AuditFailure(
                            f"cloud artifact manifest differs: {sorted(manifest_paths)!r}")
                    candidate = Path(artifact.get_entry("eval_rollout.npz").download(
                        root=temporary, skip_cache=True))
                    if not candidate.is_file():
                        raise AuditFailure("downloaded artifact entry is not a file")
                    if file_sha256(candidate) != row["metrics"]["eval_rollout_sha256"]:
                        raise AuditFailure("downloaded cloud rollout SHA-256 differs")
                artifacts_matched += 1
        except AuditFailure as exc:
            mismatches.append(f"{label}/{run_id}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            unavailable.append(f"{label}/{run_id}: {type(exc).__name__}: {exc}")
        if ordinal % 10 == 0 or ordinal == EXPECTED_CELLS:
            print(
                f"[v18-audit] W&B {mode} checked {ordinal}/{EXPECTED_CELLS} runs",
                file=sys.stderr)
    if mismatches:
        raise AuditFailure(
            f"W&B {mode} evidence mismatch for {len(mismatches)}/{EXPECTED_CELLS} runs; "
            f"first={mismatches[:5]!r}")
    if unavailable:
        raise RemoteAuditFailure(
            f"W&B {mode} unavailable for {len(unavailable)}/{EXPECTED_CELLS} runs; "
            f"first={unavailable[:5]!r}")
    return {
        "mode": mode,
        "status": "verified",
        "queried": len(rows),
        "finished_state_matched": state_matched,
        "histories_matched": histories_matched if mode == "full" else None,
        "summaries_matched": summaries_matched if mode == "full" else None,
        "downloaded_artifact_sha256_matched": (
            artifacts_matched if mode == "full" else None),
    }


def synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(TASKS):
        for seed_index, seed in enumerate(SEEDS):
            alternating = (task_index + seed_index) % 2 == 0
            for design in DESIGNS:
                if design == NONE:
                    primary, clean, deep = 1.20, 1.10, 1.20
                elif design == GRU:
                    value = 1.00 if alternating else 1.10
                    primary, clean, deep = value, value, value
                elif design == SSM:
                    value = 1.10 if alternating else 1.00
                    primary, clean, deep = value, value, value
                elif design == CANDIDATE:
                    primary, clean, deep = 0.90, 1.02, 0.90
                elif design == DYNAMIC:
                    value = 0.895 if alternating else 0.905
                    primary, clean, deep = value, 1.02, value
                elif design == STATIC:
                    value = 0.905 if alternating else 0.895
                    primary, clean, deep = value, 1.02, value
                elif design == NO_ACTION:
                    primary, clean, deep = 1.05, 1.02, 1.05
                elif design == SINGLE:
                    primary, clean, deep = 1.00, 1.02, 1.00
                else:
                    raise AssertionError(design)
                rows.append({
                    "task": task, "seed": seed, "design": design,
                    DEEP: deep, "recomputed_convergence": 0.01,
                    "metrics": {
                        PRIMARY: primary, CLEAN: clean, SECONDARY: 0.05,
                        VARIANCE: 0.01, RANK: 24.0, CONVERGENCE: 0.01,
                        INTEGRATOR: 1.0,
                    },
                })
    return rows


def self_test() -> dict[str, Any]:
    tests: list[str] = []
    history = [{
        "epoch": epoch,
        "val": {"predictive_loss": 1.0 if epoch <= 90 else 0.96},
    } for epoch in range(1, 101)]
    value = recompute_convergence(history, "predictive_loss")
    if not close(value, 0.04, atol=1e-15):
        raise AssertionError(f"convergence helper returned {value}")
    tests.append("checkpoint_window_convergence")

    computed = compute_analysis(synthetic_rows(), draws=2_000)
    if len(computed["contrasts"]) != 33 \
            or computed["scientific_label"] != "STABILIZED_LEWM_V8_CONFIRMATION_PASS" \
            or not all(computed["gates"].values()):
        raise AssertionError("synthetic passing grid did not pass all 33-contrast gates")
    recurrent = computed["contrasts"][
        f"{CANDIDATE}_vs_recurrent_envelope:{PRIMARY}"]
    if recurrent["selected_reference_counts"] != {GRU: 13, SSM: 12} \
            or not close(recurrent["mean_paired_relative_reduction"], 0.10):
        raise AssertionError("recurrent envelope/tie policy differs")
    tests.append("registered_contrasts_and_gates")

    tie_rows = synthetic_rows()
    tie_index = row_index(tie_rows)
    first = (TASKS[0], SEEDS[0])
    tie_index[(*first, GRU)]["metrics"][PRIMARY] = 1.0
    tie_index[(*first, SSM)]["metrics"][PRIMARY] = 1.0
    tie_index[(*first, GRU)][DEEP] = 4.0
    tie_index[(*first, SSM)][DEEP] = 0.25
    tie_result = selected_identity_contrast(
        tie_index, CANDIDATE, RECURRENT_REFERENCES, DEEP,
        selection_metric=PRIMARY, label="tie_test", draws=2_000)
    expected_first = (4.0 - tie_index[(*first, CANDIDATE)][DEEP]) / 4.0
    if not close(tie_result["cell_effects"][0][0], expected_first):
        raise AssertionError("exact recurrent tie did not select GRU")
    endpoint_before = envelope_contrast(
        tie_index, CANDIDATE, ENDPOINT_REFERENCES, PRIMARY,
        label="endpoint_tie_test", draws=2_000)
    second = (TASKS[0], SEEDS[1])
    tie_index[(*second, DYNAMIC)]["metrics"][PRIMARY] = 0.895
    tie_index[(*second, STATIC)]["metrics"][PRIMARY] = 0.895
    endpoint_after = envelope_contrast(
        tie_index, CANDIDATE, ENDPOINT_REFERENCES, PRIMARY,
        label="endpoint_tie_test", draws=2_000)
    if endpoint_after["selected_reference_counts"].get(DYNAMIC, 0) != (
            endpoint_before["selected_reference_counts"].get(DYNAMIC, 0) + 1):
        raise AssertionError("exact endpoint tie did not select lexicographic dynamic ID")
    tests.append("exact_reference_tie_policies")

    failing = synthetic_rows()
    failing[0]["recomputed_convergence"] = -0.051
    failed_analysis = compute_analysis(failing, draws=2_000)
    if failed_analysis["gates"]["convergence"] is not False \
            or failed_analysis["scientific_label"] != "CONFIRMATION_FAILED":
        raise AssertionError("absolute convergence gate did not reject synthetic failure")
    tests.append("absolute_convergence_failure")

    with tempfile.TemporaryDirectory(prefix="v18-audit-preflight-", dir="/tmp") as temporary:
        temp = Path(temporary)
        (temp / SUMMARY_NAME).write_text(json.dumps({
            "status": "INCOMPLETE_OR_INVALID", "expected_cells": 200,
            "completed_cells": 199, "failed_or_invalid_cells": 0,
            "wandb_enabled": True,
        }), encoding="utf-8")
        try:
            preflight(temp, temp)
        except AuditRefused:
            pass
        else:
            raise AssertionError("preflight did not refuse an incomplete synthetic grid")
    tests.append("incomplete_preflight_refusal")

    with tempfile.TemporaryDirectory(prefix="v18-audit-race-", dir="/tmp") as temporary:
        temp = Path(temporary)
        (temp / SUMMARY_NAME).write_text(json.dumps({
            "status": "COMPLETE", "expected_cells": 200,
            "completed_cells": 200, "failed_or_invalid_cells": 0,
            "wandb_enabled": True,
        }), encoding="utf-8")
        for name in (PROTOCOL_NAME, RUNS_NAME, ATTEMPTS_NAME, CELLS_NAME, CONTRASTS_NAME):
            (temp / name).write_text("[]\n", encoding="utf-8")
        (temp / ANALYSIS_NAME).write_text("{\"status\":", encoding="utf-8")
        try:
            preflight(temp, temp)
        except AuditRefused:
            pass
        else:
            raise AssertionError("preflight admitted a partial analysis JSON race")
    tests.append("partial_analysis_preflight_refusal")
    if not remote_query_indicates_missing_evidence(
            ValueError("Could not find run entity/project/id")) \
            or remote_query_indicates_missing_evidence(TimeoutError("timed out")):
        raise AssertionError("remote missing-evidence/unavailable classifier differs")
    if not issubclass(RemoteAuditFailure, AuditFailure):
        raise AssertionError("remote audit exception hierarchy differs")
    tests.append("remote_mismatch_vs_unavailable_classification")
    return {
        "schema_version": 1,
        "audit": "v18_independent_read_only_final_audit",
        "self_test": "PASS",
        "tests": tests,
        "live_grid_accessed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path,
        default=Path("/home/chrislin/projects/LeWorldModel-Memory"))
    parser.add_argument(
        "--root", type=Path,
        default=Path("outputs/lewm_v8_v18_confirmation"))
    parser.add_argument(
        "--wandb-check", choices=("none", "state", "full"), default="full",
        help="none=local receipts only; state=query all cloud run states; "
             "full=also compare cloud history/summary and download rollout artifacts")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    repo = args.repo.resolve()
    root = args.root if args.root.is_absolute() else repo / args.root
    root = root.resolve()
    base_report: dict[str, Any] = {
        "schema_version": 1,
        "audit": "v18_independent_read_only_final_audit",
        "auditor_path": str(Path(__file__).resolve()),
        "auditor_sha256": file_sha256(Path(__file__).resolve()),
        "repo": str(repo),
        "root": str(root),
        "wandb_check_requested": args.wandb_check,
    }
    try:
        summary, runs = preflight(repo, root)
    except AuditRefused as exc:
        report = {
            **base_report,
            "audit_status": "REFUSED_INCOMPLETE",
            "scientific_label": "NOT_EVALUATED",
            "errors": [str(exc)],
        }
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 3
    try:
        local, rows = audit_local(repo, root, summary, runs)
    except Exception as exc:
        report = {
            **base_report,
            "audit_status": "INVALID",
            "scientific_label": "INCOMPLETE_OR_INVALID",
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 2
    try:
        remote = audit_wandb(rows, args.wandb_check)
    except RemoteAuditFailure as exc:
        report = {
            **base_report,
            **local,
            "audit_status": "REMOTE_UNVERIFIED",
            "wandb_remote": {
                "mode": args.wandb_check, "status": "failed",
                "error": str(exc),
            },
            "errors": [str(exc)],
        }
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 4
    except AuditFailure as exc:
        report = {
            **base_report,
            **local,
            "audit_status": "INVALID",
            "scientific_label": "INCOMPLETE_OR_INVALID",
            "wandb_remote": {
                "mode": args.wandb_check, "status": "evidence_mismatch",
                "error": str(exc),
            },
            "errors": [str(exc)],
        }
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 2
    status = {
        "none": "VERIFIED_LOCAL",
        "state": "VERIFIED_STATE",
        "full": "VERIFIED",
    }[args.wandb_check]
    report = {
        **base_report,
        **local,
        "audit_status": status,
        "final_audit": args.wandb_check == "full",
        "wandb_remote": remote,
        "errors": [],
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
