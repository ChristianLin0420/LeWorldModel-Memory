#!/usr/bin/env python3
"""Create a deterministic, create-only closeout for the Sub-JEPA-v16 grid.

This auditor is deliberately separate from the frozen development runner and
analyzer.  It verifies their protocol, ledgers, local artifacts, method
invariants, and scientific summaries without changing any input.  Remote W&B
verification is opt-in; local receipts are never presented as cloud proof.

The resulting evidence remains excluded adaptive-development evidence, not an
official confirmation result.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "outputs" / "subjepa_v16_development"

TASKS = (
    "cartpole.swingup",
    "fish.swim",
    "pendulum.swingup",
    "walker.walk",
)
FAMILIES = ("fullsig", "subjepa16", "subjepa32", "vicreg")
MEMORIES = ("none", "ssm", "hacssmv8")
DESIGNS = tuple(f"{family}_{memory}" for family in FAMILIES for memory in MEMORIES)
SEEDS = (16_001, 16_002, 16_003)
EPOCHS = 30
EXPECTED_CELLS = len(TASKS) * len(DESIGNS) * len(SEEDS)

SCOPE = "subjepa_v16_excluded_adaptive_development"
STUDY = "subjepa-v16-development"
WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
CORE_ARTIFACTS = ("model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json")
SOURCE_PATHS = (
    "lewm/models/__init__.py",
    "lewm/models/encoder.py",
    "lewm/models/leworldmodel.py",
    "lewm/models/memory.py",
    "lewm/models/memory_model.py",
    "lewm/models/sigreg.py",
    "scripts/analyze_subjepa_v16.py",
    "scripts/hacssm_v10_data.py",
    "scripts/hacssm_v11_data.py",
    "scripts/run_subjepa_v16.py",
    "scripts/train_hacssm_v10.py",
    "scripts/train_hacssm_v11.py",
    "scripts/train_subjepa_v16.py",
)

HISTORY_KEYS = (
    "loss",
    "predictive_loss",
    "regularizer_loss",
    "sigreg_loss",
    "variance_loss",
    "covariance_loss",
)
PRIMARY_METRIC = "heldout_prior_state_nmse"
CLEAN_METRIC = "clean_prior_state_nmse"
VAL_PREDICTIVE_METRIC = "val_predictive_loss"
INTEGRATOR_METRIC = "initial_encoder_integrator_probe_nmse"
CONTRAST_METRICS = (PRIMARY_METRIC, CLEAN_METRIC, VAL_PREDICTIVE_METRIC)

RANK_THRESHOLD = 16.0
CONVERGENCE_ABS_THRESHOLD = 0.05
CAUSALITY_ABS_THRESHOLD = 1e-5
SEED_BLOCK_T95_DF2 = 4.302652729911275

PROTOCOL_NAME = "development_protocol.json"
RUNS_NAME = "development_runs.json"
ATTEMPTS_NAME = "development_attempts.json"
SUMMARY_NAME = "development_summary.json"
ANALYSIS_NAME = "development_analysis.json"


class AuditError(RuntimeError):
    """A deterministic closeout invariant failed."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


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


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError("invalid_json", f"cannot read {path}: {exc}") from exc


def canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AuditError("noncanonical_json", str(exc)) from exc


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise AuditError("invalid_numeric", f"{label} is boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AuditError("invalid_numeric", f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise AuditError("nonfinite", f"{label} is non-finite")
    return result


def finite_tree(value: Any, label: str) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                raise AuditError("nonfinite", f"{label} contains a non-finite tensor")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not bool(np.isfinite(value).all()):
            raise AuditError("nonfinite", f"{label} contains a non-finite array")
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
        raise AuditError("nonfinite", f"{label} is non-finite")


def close_enough(left: Any, right: Any, *, atol: float = 1e-10, rtol: float = 1e-9) -> bool:
    try:
        return math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol)
    except (TypeError, ValueError, OverflowError):
        return False


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def snapshot_files(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for path in sorted({item.resolve() for item in paths}, key=str):
        key = relative_path(path)
        if path.is_file():
            stat = path.stat()
            snapshot[key] = {
                "exists": True,
                "size_bytes": int(stat.st_size),
                "sha256": file_sha256(path),
            }
        else:
            snapshot[key] = {"exists": False, "size_bytes": None, "sha256": None}
    return snapshot


def add_event(
    events: list[dict[str, Any]], level: str, code: str, message: str,
    *, cell: str | None = None,
) -> None:
    events.append({"level": level, "code": code, "cell": cell or "", "message": message})


def local_error_events(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return integrity errors caused by local evidence, not remote availability."""
    return [
        event
        for event in events
        if event.get("level") == "error" and event.get("code") != "remote_wandb"
    ]


def run_name(task: str, design: str, seed: int) -> str:
    return f"lewm-dmc:{task}-{design}-s{seed}"


def cell_tuple(record: Mapping[str, Any]) -> tuple[str, str, int] | None:
    try:
        task = str(record["task"])
        design = str(record["design"])
        seed = int(record["seed"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return task, design, seed


def cell_label(key: tuple[str, str, int]) -> str:
    return f"{key[0]}|{key[1]}|{key[2]}"


def design_parts(design: str) -> tuple[str, str]:
    try:
        family, memory = design.rsplit("_", 1)
    except ValueError as exc:
        raise AuditError("invalid_design", f"invalid design {design!r}") from exc
    if family not in FAMILIES or memory not in MEMORIES:
        raise AuditError("invalid_design", f"invalid design {design!r}")
    return family, memory


def output_path_for(
    protocol_path: Path,
    output_parent: Path | None = None,
    revision: str | None = None,
) -> Path:
    protocol_hash = file_sha256(protocol_path)
    parent = output_parent or protocol_path.parent.parent
    suffix = f"_{revision}" if revision else ""
    return parent.resolve() / f"subjepa_v16_closeout_{protocol_hash[:12]}{suffix}"


def _flag_values(argv: Sequence[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    index = 2
    boolean_flags = {"--wandb", "--no-wandb", "--no-amp"}
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            raise AuditError("invalid_command", f"unexpected command token {token!r}")
        if token in boolean_flags:
            if token == "--wandb":
                values["wandb"] = True
            elif token == "--no-wandb":
                values["wandb"] = False
            else:
                values["no_amp"] = True
            index += 1
            continue
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise AuditError("invalid_command", f"flag {token!r} has no value")
        values[token[2:].replace("-", "_")] = argv[index + 1]
        index += 2
    return values


ARG_TYPES: dict[str, type] = {
    "train_data": str,
    "val_data": str,
    "design": str,
    "seed": int,
    "output_dir": str,
    "epochs": int,
    "batch_size": int,
    "lr": float,
    "weight_decay": float,
    "num_workers": int,
    "img_size": int,
    "patch_size": int,
    "embed_dim": int,
    "encoder_layers": int,
    "encoder_heads": int,
    "predictor_layers": int,
    "predictor_heads": int,
    "history_len": int,
    "dropout": float,
    "sigreg_lambda": float,
    "sigreg_projections": int,
    "probe_ridge": float,
    "eval_target_key": str,
    "corruption_seed": int,
    "eval_rollout_episode": int,
    "device": str,
    "wandb_entity": str,
    "wandb_project": str,
    "wandb_mode": str,
    "wandb_study": str,
    "extra_tag": str,
}


def expected_checkpoint_args(argv: Sequence[str]) -> dict[str, Any]:
    raw = _flag_values(argv)
    expected: dict[str, Any] = {}
    for key, converter in ARG_TYPES.items():
        if key not in raw:
            raise AuditError("invalid_command", f"command omitted --{key.replace('_', '-')}")
        try:
            expected[key] = converter(raw[key])
        except (TypeError, ValueError, OverflowError) as exc:
            raise AuditError("invalid_command", f"invalid value for {key}") from exc
    expected["wandb"] = bool(raw.get("wandb", False))
    expected["no_amp"] = bool(raw.get("no_amp", False))
    expected["sigreg_quad_nodes"] = int(raw.get("sigreg_quad_nodes", 17))
    return expected


def audit_protocol(
    input_root: Path,
    protocol: Mapping[str, Any],
    protocol_hash: str,
) -> tuple[dict[tuple[str, str, int], Mapping[str, Any]], dict[str, Any], list[str]]:
    errors: list[str] = []
    expected_exact = {
        "schema_version": 1,
        "scope": SCOPE,
        "tasks": list(TASKS),
        "designs": list(DESIGNS),
        "objective_families": list(FAMILIES),
        "memory_variants": list(MEMORIES),
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "runs": EXPECTED_CELLS,
        "study": STUDY,
        "wandb_enabled": True,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_mode": "online",
        "resume_supported": True,
        "core_artifacts": ["model.pt", "metrics.json", "eval_rollout.npz"],
        "git_clean_or_pushed_required": False,
    }
    for key, expected in expected_exact.items():
        if protocol.get(key) != expected:
            errors.append(f"protocol field {key!r} differs")
    if protocol.get("output_root") != str(input_root.resolve()):
        errors.append("protocol output_root differs from audited input root")

    source = protocol.get("source_sha256")
    if not isinstance(source, Mapping) or set(source) != set(SOURCE_PATHS):
        errors.append("protocol source manifest has the wrong closed set")
    else:
        for relative in SOURCE_PATHS:
            path = ROOT / relative
            if not path.is_file() or source.get(relative) != file_sha256(path):
                errors.append(f"source hash mismatch: {relative}")

    data = protocol.get("data")
    if not isinstance(data, Mapping) or set(data) != set(TASKS):
        errors.append("protocol data manifest has the wrong task set")
    else:
        for task in TASKS:
            record = data.get(task)
            if not isinstance(record, Mapping):
                errors.append(f"invalid data record: {task}")
                continue
            for split in ("train", "val"):
                value = record.get(split)
                if not isinstance(value, str):
                    errors.append(f"missing data path: {task}/{split}")
                    continue
                path = (ROOT / value).resolve()
                if not path.is_file() or record.get(f"{split}_sha256") != file_sha256(path):
                    errors.append(f"data hash mismatch: {task}/{split}")

    commands = protocol.get("commands")
    command_map: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    if not isinstance(commands, list):
        errors.append("protocol commands is not a list")
        commands = []
    if len(commands) != EXPECTED_CELLS:
        errors.append(f"protocol has {len(commands)} commands, expected {EXPECTED_CELLS}")
    if protocol.get("commands_sha256") != json_sha256(commands):
        errors.append("protocol command-list SHA-256 differs")
    expected_keys = {
        (task, design, seed) for task in TASKS for seed in SEEDS for design in DESIGNS
    }
    for index, command in enumerate(commands):
        if not isinstance(command, Mapping):
            errors.append(f"protocol command {index} is not an object")
            continue
        key = cell_tuple(command)
        argv = command.get("argv")
        if key is None or key not in expected_keys:
            errors.append(f"protocol command {index} has an invalid cell key")
            continue
        if key in command_map:
            errors.append(f"duplicate protocol command: {cell_label(key)}")
            continue
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            errors.append(f"protocol command argv is invalid: {cell_label(key)}")
            continue
        if len(argv) < 3 or Path(argv[1]).resolve() != (ROOT / "scripts/train_subjepa_v16.py").resolve():
            errors.append(f"protocol trainer path differs: {cell_label(key)}")
        try:
            parsed = expected_checkpoint_args(argv)
            if (parsed["design"], parsed["seed"], parsed["epochs"]) != (
                key[1], key[2], EPOCHS
            ):
                errors.append(f"protocol command identity differs: {cell_label(key)}")
            if Path(parsed["output_dir"]).resolve() != input_root.resolve():
                errors.append(f"protocol command output root differs: {cell_label(key)}")
            if parsed["wandb_entity"] != WANDB_ENTITY or parsed["wandb_project"] != WANDB_PROJECT:
                errors.append(f"protocol command W&B target differs: {cell_label(key)}")
            if parsed["wandb_mode"] != "online" or parsed["wandb_study"] != STUDY:
                errors.append(f"protocol command W&B semantics differ: {cell_label(key)}")
        except AuditError as exc:
            errors.append(f"{cell_label(key)}: {exc}")
        command_map[key] = command
    missing = expected_keys - set(command_map)
    if missing:
        errors.append(f"protocol omitted {len(missing)} expected cell commands")

    report = {
        "protocol_sha256": protocol_hash,
        "commands_sha256": protocol.get("commands_sha256"),
        "expected_cells": EXPECTED_CELLS,
        "command_cells": len(command_map),
        "source_files": len(SOURCE_PATHS),
        "data_files": 2 * len(TASKS),
        "passed": not errors,
        "errors": errors,
    }
    return command_map, report, errors


def _ledger_index(
    name: str,
    rows: Any,
    expected_keys: set[tuple[str, str, int]],
) -> tuple[dict[tuple[str, str, int], Mapping[str, Any]], list[str]]:
    errors: list[str] = []
    index: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    counts: Counter[tuple[str, str, int]] = Counter()
    if not isinstance(rows, list):
        return {}, [f"{name} is not a list"]
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"{name}[{position}] is not an object")
            continue
        key = cell_tuple(row)
        if key is None:
            errors.append(f"{name}[{position}] has no valid cell key")
            continue
        counts[key] += 1
        index.setdefault(key, row)
    duplicates = [key for key, count in counts.items() if count != 1]
    if duplicates:
        errors.append(f"{name} has {len(duplicates)} duplicate cell keys")
    missing = expected_keys - set(index)
    extra = set(index) - expected_keys
    if missing:
        errors.append(f"{name} omitted {len(missing)} expected cells")
    if extra:
        errors.append(f"{name} contains {len(extra)} unexpected cells")
    return index, errors


def audit_attempt_ledgers(
    expected_keys: set[tuple[str, str, int]],
    command_map: Mapping[tuple[str, str, int], Mapping[str, Any]],
    runs: Any,
    attempts: Any,
) -> tuple[dict[str, Any], dict[tuple[str, str, int], Mapping[str, Any]], list[dict[str, Any]]]:
    run_index, errors = _ledger_index("development_runs", runs, expected_keys)
    attempt_index, attempt_errors = _ledger_index(
        "development_attempts", attempts, expected_keys)
    errors.extend(attempt_errors)
    audit_rows: list[dict[str, Any]] = []
    # Inspect every append-only attempt row, including duplicate/prior rows that
    # are not selected by the unique-key index.  This is what makes a failed
    # first try followed by a successful relaunch visible rather than letting
    # the final current-row ledger conceal it.
    prior_or_relaunch = 0
    if isinstance(attempts, list):
        for row in attempts:
            if not isinstance(row, Mapping):
                prior_or_relaunch += 1
                continue
            if row.get("status") != "complete" or row.get("resumed_existing") is not False:
                prior_or_relaunch += 1
    for key in sorted(expected_keys):
        run = run_index.get(key)
        attempt = attempt_index.get(key)
        if run is None or attempt is None:
            continue
        if canonical_json(run) != canonical_json(attempt):
            errors.append(f"current run and sole attempt differ: {cell_label(key)}")
        command = command_map.get(key)
        argv = command.get("argv") if isinstance(command, Mapping) else None
        expected_command_hash = json_sha256(argv) if isinstance(argv, list) else None
        for ledger_name, row in (("run", run), ("attempt", attempt)):
            if row.get("status") != "complete":
                errors.append(f"{ledger_name} is not complete: {cell_label(key)}")
            if row.get("resumed_existing") is not False:
                errors.append(f"{ledger_name} was resumed/relaunched: {cell_label(key)}")
            if row.get("command_sha256") != expected_command_hash:
                errors.append(f"{ledger_name} command hash differs: {cell_label(key)}")
            if row.get("wandb_receipt_present") is not True:
                errors.append(f"{ledger_name} lacks a W&B receipt flag: {cell_label(key)}")
            if not isinstance(row.get("artifact_sha256"), Mapping):
                errors.append(f"{ledger_name} lacks artifact hashes: {cell_label(key)}")
        audit_rows.append({
            "task": key[0],
            "design": key[1],
            "seed": key[2],
            "status": attempt.get("status"),
            "resumed_existing": attempt.get("resumed_existing"),
            "command_sha256": attempt.get("command_sha256"),
            "seconds": attempt.get("seconds"),
            "gpu": attempt.get("gpu"),
            "completed_at": attempt.get("completed_at"),
            "directory": attempt.get("directory"),
            "log": attempt.get("log"),
            "artifact_sha256": json.dumps(
                attempt.get("artifact_sha256"), sort_keys=True, separators=(",", ":")),
        })
    report = {
        "expected_cells": len(expected_keys),
        "current_run_rows": len(runs) if isinstance(runs, list) else 0,
        "attempt_rows": len(attempts) if isinstance(attempts, list) else 0,
        "unique_current_cells": len(run_index),
        "unique_attempt_cells": len(attempt_index),
        "exactly_one_attempt_per_cell": (
            isinstance(attempts, list)
            and len(attempts) == len(expected_keys)
            and len(attempt_index) == len(expected_keys)
        ),
        "prior_failed_invalid_partial_or_relaunch_count": prior_or_relaunch,
        "result_dependent_relaunch_detected": prior_or_relaunch != 0,
        "passed": not errors,
        "errors": errors,
    }
    return report, run_index, audit_rows


def expected_metadata(task: str, design: str, seed: int) -> dict[str, Any]:
    family, memory = design_parts(design)
    gaussian = family != "vicreg"
    subspaces = {"fullsig": 1, "subjepa16": 16, "subjepa32": 32}.get(family)
    return {
        "schema_version": 1,
        "env": f"dmc:{task}",
        "design": design,
        "seed": seed,
        "epochs": EPOCHS,
        "method": "Sub-JEPA-v16-development",
        "regularizer": family,
        "regularizer_family": (
            "epps_pulley_frozen_orthogonal_subspaces"
            if gaussian else "vicreg_variance_covariance"
        ),
        "num_subspaces": subspaces,
        "subspace_dim": 128 // subspaces if gaussian else None,
        "projection_policy": (
            "independent_frozen_row_orthonormal_qr" if gaussian else "not_applicable"
        ),
        "projection_requires_grad": False,
        "sketch_direction_policy": (
            "fresh_unit_directions_per_forward" if gaussian else "not_applicable"
        ),
        "regularizer_source": "active_clean_target",
        "memory_architecture": memory,
        "memory_specific_loss_weight": 0.0,
        "new_memory_architecture": False,
        "observation_correction_branch": False,
        "one_token_predictor": True,
        "paired_clean_target": True,
        "clean_target_gradient_active": True,
        "target_stop_gradient": False,
        "reward_used_for_training": False,
        "state_labels_used_for_training": False,
        "training_objective": "v16_paired_next_clean_plus_collapse_regularizer",
        "confirmation_evidence": False,
        "sigreg_projections_per_subspace": 512,
        "sigreg_quad_nodes": 17,
        "sigreg_lambda": 0.1,
        "eval_rollout_episode": 0,
        "eval_target_key": "task_observation",
    }


def _validate_history(
    history: Any,
    task: str,
    design: str,
    seed: int,
) -> list[dict[str, Any]]:
    label = cell_label((task, design, seed))
    family, memory = design_parts(design)
    if not isinstance(history, list) or len(history) != EPOCHS:
        raise AuditError("history_length", f"{label}: history is not exactly {EPOCHS} rows")
    csv_rows: list[dict[str, Any]] = []
    for expected_epoch, row in enumerate(history, start=1):
        if not isinstance(row, Mapping) or row.get("epoch") != expected_epoch:
            raise AuditError("history_epoch", f"{label}: epoch sequence differs")
        epoch_seconds = finite_number(row.get("epoch_seconds"), f"{label}.epoch_seconds")
        if epoch_seconds <= 0:
            raise AuditError("history_epoch_seconds", f"{label}: non-positive epoch duration")
        epoch_csv: dict[str, Any] = {
            "task": task,
            "design": design,
            "family": family,
            "memory": memory,
            "seed": seed,
            "epoch": expected_epoch,
            "epoch_seconds": epoch_seconds,
        }
        for split in ("train", "val"):
            metrics = row.get(split)
            if not isinstance(metrics, Mapping) or set(metrics) != set(HISTORY_KEYS):
                raise AuditError("history_schema", f"{label}: {split} history keys differ")
            values = {
                key: finite_number(metrics.get(key), f"{label}.e{expected_epoch}.{split}.{key}")
                for key in HISTORY_KEYS
            }
            if not close_enough(
                values["loss"], values["predictive_loss"] + values["regularizer_loss"],
                atol=2e-6, rtol=2e-6,
            ):
                raise AuditError("loss_decomposition", f"{label}: total loss decomposition differs")
            if family == "vicreg":
                if values["sigreg_loss"] != 0.0 or not close_enough(
                    values["regularizer_loss"],
                    values["variance_loss"] + values["covariance_loss"],
                    atol=2e-6, rtol=2e-6,
                ):
                    raise AuditError("vicreg_decomposition", f"{label}: VICReg terms differ")
            else:
                if values["variance_loss"] != 0.0 or values["covariance_loss"] != 0.0:
                    raise AuditError("sigreg_decomposition", f"{label}: inactive VICReg term is nonzero")
                if not close_enough(
                    values["regularizer_loss"], 0.1 * values["sigreg_loss"],
                    atol=2e-6, rtol=2e-6,
                ):
                    raise AuditError("sigreg_decomposition", f"{label}: SIGReg weight differs")
            epoch_csv.update({f"{split}_{key}": value for key, value in values.items()})
        csv_rows.append(epoch_csv)
    return csv_rows


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - old torch fallback
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise AuditError("checkpoint_load", f"cannot load {path}: {exc}") from exc
    if not isinstance(checkpoint, Mapping):
        raise AuditError("checkpoint_schema", f"{path} is not a checkpoint object")
    return checkpoint


def _validate_projection(
    state: Mapping[str, Any],
    metrics: Mapping[str, Any],
    family: str,
    label: str,
) -> dict[str, Any]:
    if family == "vicreg":
        expected = {
            "subspace_projection_frozen": False,
            "subspace_projection_orthogonality_max_abs": 0.0,
            "subspace_projection_count": None,
            "subspace_projection_dimension": None,
            "subspace_projection_sha256": None,
        }
        for key, value in expected.items():
            if metrics.get(key) != value:
                raise AuditError("vicreg_projection_metadata", f"{label}: {key} differs")
        return {"projection_sha256": None, "orthogonality_max_abs": None}

    subspaces = {"fullsig": 1, "subjepa16": 16, "subjepa32": 32}[family]
    width = 128 // subspaces
    prefix = "world.sigreg."
    required = {
        "projection_matrices": (subspaces, width, 128),
        "t": (17,),
        "phi": (17,),
        "weights": (17,),
    }
    tensors: dict[str, torch.Tensor] = {}
    for name, shape in required.items():
        tensor = state.get(prefix + name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise AuditError("projection_shape", f"{label}: {prefix + name} shape differs")
        if tensor.dtype != torch.float32 or tensor.requires_grad:
            raise AuditError("projection_dtype", f"{label}: {prefix + name} is not frozen FP32")
        if not bool(torch.isfinite(tensor).all()):
            raise AuditError("nonfinite", f"{label}: {prefix + name} is non-finite")
        tensors[name] = tensor.detach().cpu().contiguous()

    matrices = tensors["projection_matrices"]
    identity = torch.eye(width, dtype=torch.float32).expand(subspaces, width, width)
    gram_error = float((matrices @ matrices.transpose(-1, -2) - identity).abs().max())
    if gram_error > 1e-5:
        raise AuditError("projection_orthogonality", f"{label}: row Gram error {gram_error}")
    projection_bytes = matrices.numpy().astype("<f4", copy=False).tobytes(order="C")
    projection_sha = hashlib.sha256(projection_bytes).hexdigest()
    expected_t = torch.linspace(0.0, 3.0, 17, dtype=torch.float32)
    expected_phi = torch.exp(-expected_t.square() / 2.0)
    dt_value = 3.0 / 16.0
    quadrature = torch.full((17,), 2.0 * dt_value, dtype=torch.float32)
    quadrature[[0, -1]] = dt_value
    expected_weights = quadrature * expected_phi
    if not torch.equal(tensors["t"], expected_t):
        raise AuditError("projection_knots", f"{label}: 17-knot grid differs")
    if not torch.equal(tensors["phi"], expected_phi):
        raise AuditError("projection_phi", f"{label}: target characteristic function differs")
    if not torch.equal(tensors["weights"], expected_weights):
        raise AuditError("projection_weights", f"{label}: quadrature weights differ")
    expected_metrics = {
        "subspace_projection_frozen": True,
        "subspace_projection_count": subspaces,
        "subspace_projection_dimension": width,
        "subspace_projection_sha256": projection_sha,
    }
    for key, value in expected_metrics.items():
        if metrics.get(key) != value:
            raise AuditError("projection_receipt", f"{label}: projection receipt {key} differs")
    if not close_enough(
        metrics.get("subspace_projection_orthogonality_max_abs"), gram_error,
        # The trainer computed this diagnostic with CUDA GEMM while closeout
        # recomputes it on CPU.  The tensor SHA is exact; the derived maximum is
        # backend-sensitive at roughly one FP32 ulp, so compare it accordingly.
        atol=1e-6, rtol=1e-5,
    ):
        raise AuditError("projection_receipt", f"{label}: recorded Gram error differs")
    return {"projection_sha256": projection_sha, "orthogonality_max_abs": gram_error}


def _validate_rollout(path: Path, metrics: Mapping[str, Any], label: str) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) == set():
                raise AuditError("rollout_schema", f"{label}: rollout is empty")
            for key in archive.files:
                array = archive[key]
                if array.dtype.hasobject:
                    raise AuditError("rollout_object", f"{label}: rollout {key} has object dtype")
                if np.issubdtype(array.dtype, np.number) and not bool(np.isfinite(array).all()):
                    raise AuditError("nonfinite", f"{label}: rollout {key} is non-finite")
            if int(archive["schema_version"]) != 2 or int(archive["episode_index"]) != 0:
                raise AuditError("rollout_schema", f"{label}: rollout header differs")
            if archive["conditions"].tolist() != [
                "freeze", "gaussian_noise", "checkerboard", "long_freeze"
            ]:
                raise AuditError("rollout_conditions", f"{label}: held-out conditions differ")
            for condition in ("freeze", "gaussian_noise", "checkerboard", "long_freeze"):
                for suffix in (
                    "target_times", "phase", "observed_rgb", "clean_rgb", "actions",
                    "evaluation_target", "encoder_state_prediction",
                    "prior_state_prediction", "posterior_state_prediction",
                    "predictor_state_prediction",
                ):
                    if f"{condition}_{suffix}" not in archive.files:
                        raise AuditError(
                            "rollout_schema", f"{label}: missing {condition}_{suffix}")
    except AuditError:
        raise
    except Exception as exc:
        raise AuditError("rollout_load", f"{label}: cannot load rollout: {exc}") from exc
    if metrics.get("eval_rollout_sha256") != file_sha256(path):
        raise AuditError("rollout_hash", f"{label}: metrics rollout hash differs")


def _validate_receipt(
    receipt: Mapping[str, Any],
    metrics: Mapping[str, Any],
    task: str,
    design: str,
    seed: int,
) -> str:
    label = cell_label((task, design, seed))
    expected_name = f"{STUDY}-{run_name(task, design, seed)}"
    exact = {
        "schema_version": 1,
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "mode": "online",
        "state": "finished",
        "study": STUDY,
        "run_name": expected_name,
        "eval_rollout_sha256": metrics.get("eval_rollout_sha256"),
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            raise AuditError("wandb_receipt", f"{label}: receipt {key} differs")
    run_id = receipt.get("run_id")
    if not isinstance(run_id, str) or not run_id or run_id == "unknown":
        raise AuditError("wandb_receipt", f"{label}: invalid W&B run ID")
    if receipt.get("eval_rollout_artifact_name") != f"eval-rollout-{run_id}":
        raise AuditError("wandb_receipt", f"{label}: artifact name differs")
    expected_url = f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}"
    if receipt.get("url") != expected_url:
        raise AuditError("wandb_receipt", f"{label}: run URL differs")
    return run_id


def _recompute_convergence(history: Sequence[Mapping[str, Any]], key: str) -> float:
    previous = np.mean([row["val"][key] for row in history[-20:-10]], dtype=np.float64)
    recent = np.mean([row["val"][key] for row in history[-10:]], dtype=np.float64)
    return float((previous - recent) / max(abs(float(previous)), 1e-12))


def audit_cell(
    input_root: Path,
    key: tuple[str, str, int],
    command: Mapping[str, Any],
    run_record: Mapping[str, Any],
    protocol: Mapping[str, Any],
    before: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], Mapping[str, Any], Mapping[str, Any], str]:
    task, design, seed = key
    family, memory = design_parts(design)
    label = cell_label(key)
    directory = input_root / run_name(task, design, seed)
    if Path(str(run_record.get("directory", ""))).resolve() != directory.resolve():
        raise AuditError("ledger_directory", f"{label}: ledger directory differs")
    artifact_hashes = run_record.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping) or set(artifact_hashes) != set(CORE_ARTIFACTS):
        raise AuditError("ledger_artifacts", f"{label}: ledger artifact set differs")
    for filename in CORE_ARTIFACTS:
        path = directory / filename
        snap = before.get(relative_path(path), {})
        if not snap.get("exists") or artifact_hashes.get(filename) != snap.get("sha256"):
            raise AuditError("artifact_hash", f"{label}: {filename} hash differs from ledger")

    metrics = load_json(directory / "metrics.json")
    receipt = load_json(directory / "wandb_run.json")
    if not isinstance(metrics, Mapping) or not isinstance(receipt, Mapping):
        raise AuditError("artifact_schema", f"{label}: metrics/receipt is not an object")
    finite_tree(metrics, f"{label}.metrics")
    for field, expected in expected_metadata(task, design, seed).items():
        if metrics.get(field) != expected:
            raise AuditError("method_invariant", f"{label}: metric metadata {field} differs")

    data_record = protocol["data"][task]
    for split in ("train", "val"):
        expected_path = (ROOT / data_record[split]).resolve()
        if Path(str(metrics.get(f"{split}_data", ""))).resolve() != expected_path:
            raise AuditError("data_receipt", f"{label}: {split} data path differs")
        if metrics.get(f"{split}_data_sha256") != data_record[f"{split}_sha256"]:
            raise AuditError("data_receipt", f"{label}: {split} data hash differs")

    required_numeric = (
        PRIMARY_METRIC,
        CLEAN_METRIC,
        VAL_PREDICTIVE_METRIC,
        INTEGRATOR_METRIC,
        "encoder_mean_channel_variance",
        "encoder_covariance_effective_rank",
        "encoder_singleton_max_abs",
        "encoder_prefix_max_abs",
        "predictive_loss_convergence_relative_change",
        "regularizer_loss_convergence_relative_change",
        "final_train_loss",
        "final_val_loss",
        "val_regularizer_loss",
        "mean_epoch_seconds",
        "peak_vram_bytes",
        "trainable_parameters",
    )
    numeric = {field: finite_number(metrics.get(field), f"{label}.{field}") for field in required_numeric}
    if any(numeric[field] < 0 for field in (
        PRIMARY_METRIC, CLEAN_METRIC, VAL_PREDICTIVE_METRIC, INTEGRATOR_METRIC,
        "encoder_mean_channel_variance", "encoder_covariance_effective_rank",
        "encoder_singleton_max_abs", "encoder_prefix_max_abs", "mean_epoch_seconds",
        "peak_vram_bytes", "trainable_parameters",
    )):
        raise AuditError("invalid_metric", f"{label}: negative nonnegative metric")

    _validate_rollout(directory / "eval_rollout.npz", metrics, label)
    run_id = _validate_receipt(receipt, metrics, task, design, seed)

    checkpoint = _load_checkpoint(directory / "model.pt")
    if set(checkpoint) != {
        "model_state_dict", "args", "final_metrics", "history", "state_probes",
        "inverse_action_probe",
    }:
        raise AuditError("checkpoint_schema", f"{label}: checkpoint top-level keys differ")
    finite_tree(checkpoint, f"{label}.checkpoint")
    if canonical_json(checkpoint["final_metrics"]) != canonical_json(metrics):
        raise AuditError("checkpoint_metrics", f"{label}: checkpoint final metrics differ")
    argv = command.get("argv")
    if not isinstance(argv, list):
        raise AuditError("invalid_command", f"{label}: command argv missing")
    expected_args = expected_checkpoint_args(argv)
    if canonical_json(checkpoint["args"]) != canonical_json(expected_args):
        raise AuditError("checkpoint_args", f"{label}: checkpoint arguments differ token-for-token")
    probes = checkpoint["state_probes"]
    inverse = checkpoint["inverse_action_probe"]
    if not isinstance(probes, Mapping) or set(probes) != {"prior", "posterior", "encoder", "predictor"}:
        raise AuditError("probe_schema", f"{label}: state probes differ")
    if not isinstance(inverse, Mapping) or set(inverse) != {"x_mean", "x_std", "y_mean", "y_std", "weights"}:
        raise AuditError("probe_schema", f"{label}: inverse-action probe differs")
    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping):
        raise AuditError("checkpoint_schema", f"{label}: model state is not a mapping")
    projection = _validate_projection(state, metrics, family, label)
    history_rows = _validate_history(checkpoint["history"], task, design, seed)
    history = checkpoint["history"]
    last = history[-1]
    final_pairs = {
        "final_train_loss": last["train"]["loss"],
        "final_val_loss": last["val"]["loss"],
        "val_predictive_loss": last["val"]["predictive_loss"],
        "val_regularizer_loss": last["val"]["regularizer_loss"],
        "mean_epoch_seconds": statistics.fmean(row["epoch_seconds"] for row in history),
    }
    for field, expected in final_pairs.items():
        if not close_enough(metrics.get(field), expected, atol=1e-12, rtol=1e-12):
            raise AuditError("history_receipt", f"{label}: {field} differs from history")
    for loss_name in ("predictive_loss", "regularizer_loss", "loss"):
        field = f"{loss_name}_convergence_relative_change"
        if not close_enough(
            metrics.get(field), _recompute_convergence(history, loss_name),
            atol=1e-12, rtol=1e-12,
        ):
            raise AuditError("history_convergence", f"{label}: {field} differs")

    rank_pass = numeric["encoder_covariance_effective_rank"] >= RANK_THRESHOLD
    convergence_pass = (
        abs(numeric["predictive_loss_convergence_relative_change"])
        <= CONVERGENCE_ABS_THRESHOLD
    )
    causality_max = max(
        numeric["encoder_singleton_max_abs"], numeric["encoder_prefix_max_abs"])
    causality_pass = causality_max <= CAUSALITY_ABS_THRESHOLD
    cell_row = {
        "task": task,
        "design": design,
        "family": family,
        "memory": memory,
        "seed": seed,
        PRIMARY_METRIC: numeric[PRIMARY_METRIC],
        CLEAN_METRIC: numeric[CLEAN_METRIC],
        VAL_PREDICTIVE_METRIC: numeric[VAL_PREDICTIVE_METRIC],
        INTEGRATOR_METRIC: numeric[INTEGRATOR_METRIC],
        "encoder_mean_channel_variance": numeric["encoder_mean_channel_variance"],
        "encoder_covariance_effective_rank": numeric["encoder_covariance_effective_rank"],
        "encoder_singleton_max_abs": numeric["encoder_singleton_max_abs"],
        "encoder_prefix_max_abs": numeric["encoder_prefix_max_abs"],
        "predictive_loss_convergence_relative_change": numeric[
            "predictive_loss_convergence_relative_change"],
        "regularizer_loss_convergence_relative_change": numeric[
            "regularizer_loss_convergence_relative_change"],
        "trainable_parameters": int(numeric["trainable_parameters"]),
        "mean_epoch_seconds": numeric["mean_epoch_seconds"],
        "peak_vram_bytes": int(numeric["peak_vram_bytes"]),
        "rank_gate_pass": rank_pass,
        "predictive_convergence_gate_pass": convergence_pass,
        "causality_gate_pass": causality_pass,
        "projection_sha256": projection["projection_sha256"],
        "projection_orthogonality_max_abs": projection["orthogonality_max_abs"],
        "wandb_run_id": run_id,
        "local_bundle_passed": True,
    }
    return cell_row, history_rows, metrics, receipt, run_id


def summarize(values: Iterable[float]) -> dict[str, Any]:
    vector = [finite_number(value, "summary value") for value in values]
    if not vector:
        raise AuditError("empty_summary", "cannot summarize an empty vector")
    return {
        "n": len(vector),
        "mean": statistics.fmean(vector),
        "std": statistics.stdev(vector) if len(vector) > 1 else 0.0,
        "min": min(vector),
        "max": max(vector),
    }


def seed_block_summary(
    effects: Mapping[tuple[str, int], float],
    *,
    tasks: Sequence[str] = TASKS,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    expected = {(task, seed) for task in tasks for seed in seeds}
    if set(effects) != expected:
        missing = expected - set(effects)
        extra = set(effects) - expected
        raise AuditError(
            "unbalanced_contrast",
            f"paired effect grid differs (missing={len(missing)}, extra={len(extra)})",
        )
    for key, value in effects.items():
        finite_number(value, f"paired effect {key}")
    task_means = {
        task: statistics.fmean(effects[(task, seed)] for seed in seeds) for task in tasks
    }
    seed_blocks = {
        str(seed): statistics.fmean(effects[(task, seed)] for task in tasks) for seed in seeds
    }
    blocks = list(seed_blocks.values())
    point = statistics.fmean(blocks)
    sd = statistics.stdev(blocks)
    half_width = SEED_BLOCK_T95_DF2 * sd / math.sqrt(len(blocks))
    return {
        "effect_definition": "(reference-candidate)/reference; positive favors candidate",
        "aggregation": "mean of within-task/seed paired relative effects",
        "raw_cross_task_ratio_of_means_used": False,
        "n_task_seed_pairs": len(effects),
        "n_tasks": len(tasks),
        "n_seed_blocks": len(seeds),
        "point_estimate": point,
        "task_means": task_means,
        "seed_blocks": seed_blocks,
        "seed_block_std": sd,
        "ci95_low": point - half_width,
        "ci95_high": point + half_width,
        "ci_method": "fixed-task seed-blocked Student-t interval, df=2",
        "t_critical": SEED_BLOCK_T95_DF2,
        "cell_wins": sum(value > 0 for value in effects.values()),
        "cell_ties": sum(value == 0 for value in effects.values()),
        "task_wins": sum(value > 0 for value in task_means.values()),
        "adaptive_descriptive_only": True,
    }


def paired_contrast(
    metrics: Mapping[tuple[str, str, int], Mapping[str, Any]],
    *,
    comparison: str,
    candidate_design: str,
    reference_design: str | None,
    metric: str,
    reference_metric: str | None = None,
) -> dict[str, Any]:
    effects: dict[tuple[str, int], float] = {}
    values: list[dict[str, Any]] = []
    for task in TASKS:
        for seed in SEEDS:
            candidate = finite_number(
                metrics[(task, candidate_design, seed)].get(metric),
                f"{candidate_design}.{metric}",
            )
            if reference_design is None:
                reference = finite_number(
                    metrics[(task, candidate_design, seed)].get(reference_metric),
                    f"{candidate_design}.{reference_metric}",
                )
            else:
                reference = finite_number(
                    metrics[(task, reference_design, seed)].get(metric),
                    f"{reference_design}.{metric}",
                )
            if reference == 0:
                raise AuditError("zero_reference", f"zero reference in {comparison}")
            effect = (reference - candidate) / reference
            effects[(task, seed)] = effect
            values.append({
                "task": task,
                "seed": seed,
                "candidate": candidate,
                "reference": reference,
                "paired_relative_effect": effect,
            })
    result = seed_block_summary(effects)
    result.update({
        "comparison": comparison,
        "candidate_design": candidate_design,
        "reference_design": reference_design or f"same-checkpoint:{reference_metric}",
        "metric": metric,
        "lower_is_better": True,
        "paired_effects": values,
        "harm_cell_count": sum(item["paired_relative_effect"] < 0 for item in values),
        "clean_harm_summary": metric == CLEAN_METRIC,
    })
    return result


def build_contrasts(
    metrics: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    contrasts: list[dict[str, Any]] = []
    host_pairs = (
        ("subjepa16_vs_fullsig", "subjepa16", "fullsig"),
        ("subjepa16_vs_vicreg", "subjepa16", "vicreg"),
        ("subjepa32_vs_subjepa16", "subjepa32", "subjepa16"),
    )
    for name, candidate_family, reference_family in host_pairs:
        for memory in MEMORIES:
            for metric in CONTRAST_METRICS:
                contrasts.append(paired_contrast(
                    metrics,
                    comparison=f"{name}:{memory}:{metric}",
                    candidate_design=f"{candidate_family}_{memory}",
                    reference_design=f"{reference_family}_{memory}",
                    metric=metric,
                ))
    for family in FAMILIES:
        for memory in ("ssm", "hacssmv8"):
            contrasts.append(paired_contrast(
                metrics,
                comparison=f"memory_vs_none:{family}:{memory}:{PRIMARY_METRIC}",
                candidate_design=f"{family}_{memory}",
                reference_design=f"{family}_none",
                metric=PRIMARY_METRIC,
            ))
    for design in DESIGNS:
        contrasts.append(paired_contrast(
            metrics,
            comparison=f"model_vs_checkpoint_integrator:{design}:{PRIMARY_METRIC}",
            candidate_design=design,
            reference_design=None,
            metric=PRIMARY_METRIC,
            reference_metric=INTEGRATOR_METRIC,
        ))
    return contrasts


def build_interactions(
    metrics: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    interactions: list[dict[str, Any]] = []
    host_pairs = (
        ("subjepa16_vs_fullsig", "subjepa16", "fullsig"),
        ("subjepa16_vs_vicreg", "subjepa16", "vicreg"),
        ("subjepa32_vs_subjepa16", "subjepa32", "subjepa16"),
    )
    for name, candidate_family, reference_family in host_pairs:
        for memory in ("ssm", "hacssmv8"):
            for metric in CONTRAST_METRICS:
                effects: dict[tuple[str, int], float] = {}
                per_cell: list[dict[str, Any]] = []
                for task in TASKS:
                    for seed in SEEDS:
                        values: dict[str, float] = {}
                        for mem in ("none", memory):
                            candidate = finite_number(
                                metrics[(task, f"{candidate_family}_{mem}", seed)].get(metric),
                                f"interaction candidate {metric}",
                            )
                            reference = finite_number(
                                metrics[(task, f"{reference_family}_{mem}", seed)].get(metric),
                                f"interaction reference {metric}",
                            )
                            if candidate <= 0 or reference <= 0:
                                raise AuditError("nonpositive_log", f"non-positive interaction value: {name}")
                            values[mem] = math.log(reference / candidate)
                        interaction = values[memory] - values["none"]
                        effects[(task, seed)] = interaction
                        per_cell.append({
                            "task": task,
                            "seed": seed,
                            "candidate_benefit_log_ratio_none": values["none"],
                            "candidate_benefit_log_ratio_memory": values[memory],
                            "interaction": interaction,
                        })
                result = seed_block_summary(effects)
                result.update({
                    "interaction": f"{name}:{memory}_minus_none:{metric}",
                    "candidate_family": candidate_family,
                    "reference_family": reference_family,
                    "memory": memory,
                    "baseline_memory": "none",
                    "metric": metric,
                    "interaction_definition": (
                        "log(reference/candidate)_memory - "
                        "log(reference/candidate)_none"
                    ),
                    "positive_interpretation": "memory amplifies the candidate-family benefit",
                    "paired_interactions": per_cell,
                })
                interactions.append(result)
    return interactions


def diagnostics_by_design(cell_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        grouped[str(row["design"])].append(row)
    result: list[dict[str, Any]] = []
    numeric_fields = (
        "encoder_mean_channel_variance",
        "encoder_covariance_effective_rank",
        "encoder_singleton_max_abs",
        "encoder_prefix_max_abs",
        "predictive_loss_convergence_relative_change",
        "regularizer_loss_convergence_relative_change",
        CLEAN_METRIC,
        "trainable_parameters",
        "mean_epoch_seconds",
        "peak_vram_bytes",
    )
    for design in DESIGNS:
        rows = grouped.get(design, [])
        item: dict[str, Any] = {"design": design, "n": len(rows)}
        for field in numeric_fields:
            item[field] = summarize(float(row[field]) for row in rows)
        item.update({
            "rank_gate_pass_count": sum(bool(row["rank_gate_pass"]) for row in rows),
            "predictive_convergence_gate_pass_count": sum(
                bool(row["predictive_convergence_gate_pass"]) for row in rows),
            "causality_gate_pass_count": sum(bool(row["causality_gate_pass"]) for row in rows),
            "rank_gate": f"encoder_covariance_effective_rank >= {RANK_THRESHOLD}",
            "predictive_convergence_gate": (
                "abs(predictive_loss_convergence_relative_change) <= "
                f"{CONVERGENCE_ABS_THRESHOLD}"
            ),
            "causality_gate": (
                "max(encoder_singleton_max_abs,encoder_prefix_max_abs) <= "
                f"{CAUSALITY_ABS_THRESHOLD}"
            ),
            "variance_gate_invented": False,
        })
        result.append(item)
    return result


def validate_parameter_parity(cell_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    grouped: dict[tuple[str, int, str], dict[str, int]] = defaultdict(dict)
    for row in cell_rows:
        grouped[(str(row["task"]), int(row["seed"]), str(row["memory"]))][
            str(row["family"])
        ] = int(row["trainable_parameters"])
    errors: list[str] = []
    for key, values in grouped.items():
        if set(values) != set(FAMILIES) or len(set(values.values())) != 1:
            errors.append(f"trainable-parameter host parity differs: {key}: {values}")
    return errors


def _history_remote_rows(run: Any) -> list[Mapping[str, Any]]:
    keys = ["epoch"]
    keys.extend(f"{split}/{field}" for split in ("train", "val") for field in HISTORY_KEYS)
    keys.append("perf/epoch_seconds")
    # W&B's scan_history endpoint can intermittently reject valid runs with
    # ``Step column '_step' not found in schema``.  These runs expose the exact
    # 30-row history through the public history API, which is also what the
    # dashboard uses.  Request every expected row explicitly, without sampling.
    return [
        dict(row)
        for row in run.history(keys=keys, samples=10_000, pandas=False)
    ]


def remote_config_matches(
    config: Mapping[str, Any], expected: Mapping[str, Any],
) -> bool:
    """Match remote argparse names to the immutable receipt vocabulary."""
    config_keys = {
        "design": "design",
        "seed": "seed",
        "env": "env",
        "regularizer": "regularizer",
        "regularizer_family": "regularizer_family",
        "num_subspaces": "num_subspaces",
        "subspace_dim": "subspace_dim",
        "memory_architecture": "memory_architecture",
        "regularizer_source": "regularizer_source",
        "clean_target_gradient_active": "clean_target_gradient_active",
        "target_stop_gradient": "target_stop_gradient",
        "sigreg_projections_per_subspace": "sigreg_projections",
        "sigreg_quad_nodes": "sigreg_quad_nodes",
    }
    return (
        all(
            config.get(remote_name) == expected[receipt_name]
            for receipt_name, remote_name in config_keys.items()
        )
        and config.get("wandb_entity") == WANDB_ENTITY
        and config.get("wandb_project") == WANDB_PROJECT
        and config.get("wandb_study") == STUDY
    )


def remote_media_matches(
    summary: Mapping[str, Any], logged_artifacts: Sequence[Any],
) -> tuple[bool, bool]:
    """Validate the persisted evaluation table and paired-video receipts."""
    table = summary.get("eval/rollout_trace")
    video = summary.get("eval/paired_rollout")
    # Public API summary children are SummarySubDict objects: dict-like, but
    # intentionally not registered as collections.abc.Mapping.
    if hasattr(table, "items"):
        table = dict(table.items())
    if hasattr(video, "items"):
        video = dict(video.items())
    run_tables = [
        artifact for artifact in logged_artifacts
        if getattr(artifact, "type", None) == "run_table"
    ]
    table_verified = (
        isinstance(table, Mapping)
        and table.get("_type") == "table-file"
        and int(table.get("nrows", 0)) > 0
        and bool(table.get("sha256"))
        and len(run_tables) == 1
    )
    video_verified = (
        isinstance(video, Mapping)
        and video.get("_type") == "video-file"
        and int(video.get("size", 0)) > 0
        and bool(video.get("sha256"))
    )
    return table_verified, video_verified


def audit_remote_wandb(
    receipts: Mapping[tuple[str, str, int], Mapping[str, Any]],
    metrics: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    try:
        import wandb
    except ImportError as exc:
        return {
            "status": "REMOTE_VERIFICATION_FAILED",
            "requested": True,
            "verified": False,
            "errors": [f"wandb import failed: {exc}"],
            "runs": [],
        }
    try:
        api = wandb.Api(timeout=60)
    except Exception as exc:
        return {
            "status": "REMOTE_VERIFICATION_FAILED",
            "requested": True,
            "verified": False,
            "errors": [f"wandb API initialization failed: {type(exc).__name__}: {exc}"],
            "runs": [],
        }
    for key in sorted(receipts):
        task, design, seed = key
        label = cell_label(key)
        receipt = receipts[key]
        run_id = str(receipt["run_id"])
        record = {
            "task": task,
            "design": design,
            "seed": seed,
            "run_id": run_id,
            "state_verified": False,
            "config_verified": False,
            "history_verified": False,
            "table_verified": False,
            "video_verified": False,
            "artifact_metadata_verified": False,
            "artifact_download_sha256_verified": False,
        }
        try:
            run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
            record["state_verified"] = getattr(run, "state", None) == "finished"
            config = dict(getattr(run, "config", {}) or {})
            expected = expected_metadata(task, design, seed)
            record["config_verified"] = remote_config_matches(config, expected)
            history = _history_remote_rows(run)
            epoch_rows = [row for row in history if row.get("epoch") in range(1, EPOCHS + 1)]
            by_epoch = {int(row["epoch"]): row for row in epoch_rows}
            record["history_verified"] = (
                len(epoch_rows) == EPOCHS
                and set(by_epoch) == set(range(1, EPOCHS + 1))
                and all(
                    key_name in by_epoch[epoch]
                    and math.isfinite(float(by_epoch[epoch][key_name]))
                    for epoch in range(1, EPOCHS + 1)
                    for key_name in [
                        *(f"{split}/{field}" for split in ("train", "val") for field in HISTORY_KEYS),
                        "perf/epoch_seconds",
                    ]
                )
            )
            summary = dict(getattr(run, "summary", {}) or {})
            logged_artifacts = list(run.logged_artifacts())
            record["table_verified"], record["video_verified"] = (
                remote_media_matches(summary, logged_artifacts)
            )
            matching = [
                artifact for artifact in logged_artifacts
                if getattr(artifact, "type", None) == "evaluation-rollout"
                and str(getattr(artifact, "name", "")).split(":", 1)[0]
                == receipt["eval_rollout_artifact_name"]
            ]
            if len(matching) == 1:
                artifact = matching[0]
                metadata = dict(getattr(artifact, "metadata", {}) or {})
                record["artifact_metadata_verified"] = (
                    metadata.get("sha256") == metrics[key]["eval_rollout_sha256"]
                    and metadata.get("study") == STUDY
                    and metadata.get("env") == f"dmc:{task}"
                    and metadata.get("design") == design
                    and int(metadata.get("seed", -1)) == seed
                )
                with tempfile.TemporaryDirectory(prefix="subjepa-v16-wandb-") as temporary:
                    artifact.download(root=temporary)
                    downloaded = Path(temporary) / "eval_rollout.npz"
                    record["artifact_download_sha256_verified"] = (
                        downloaded.is_file()
                        and file_sha256(downloaded) == metrics[key]["eval_rollout_sha256"]
                    )
            failed_fields = [
                name for name, value in record.items() if name.endswith("_verified") and not value
            ]
            if failed_fields:
                errors.append(f"{label}: remote checks failed: {','.join(failed_fields)}")
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
        rows.append(record)
    return {
        "status": "REMOTE_VERIFIED" if not errors and len(rows) == EXPECTED_CELLS else "REMOTE_VERIFICATION_FAILED",
        "requested": True,
        "verified": not errors and len(rows) == EXPECTED_CELLS,
        "runs_checked": len(rows),
        "errors": errors,
        "runs": rows,
    }


def csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_exclusive(path: Path, data: str | bytes) -> None:
    mode = "xb" if isinstance(data, bytes) else "x"
    kwargs = {} if isinstance(data, bytes) else {"encoding": "utf-8", "newline": ""}
    with path.open(mode, **kwargs) as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _contrast_csv_rows(contrasts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "comparison", "candidate_design", "reference_design", "metric",
        "n_task_seed_pairs", "n_tasks", "n_seed_blocks", "point_estimate",
        "seed_block_std", "ci95_low", "ci95_high", "cell_wins", "cell_ties",
        "task_wins", "harm_cell_count", "ci_method", "adaptive_descriptive_only",
    )
    return [{field: row.get(field) for field in fields} for row in contrasts]


def _interaction_csv_rows(interactions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "interaction", "candidate_family", "reference_family", "memory",
        "baseline_memory", "metric", "n_task_seed_pairs", "n_seed_blocks",
        "point_estimate", "seed_block_std", "ci95_low", "ci95_high",
        "cell_wins", "task_wins", "ci_method", "adaptive_descriptive_only",
    )
    return [{field: row.get(field) for field in fields} for row in interactions]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--output-parent", type=Path, default=None,
        help="Parent for protocol-hash-keyed output (default: repository outputs directory)",
    )
    parser.add_argument(
        "--remote-wandb", action="store_true",
        help="Query W&B and download every logged rollout artifact for hash verification",
    )
    parser.add_argument(
        "--revision", default=None,
        help=(
            "Optional create-only revision suffix. Use only when preserving an earlier "
            "immutable closeout; the protocol hash remains in the directory key."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = args.input_root.resolve()
    protocol_path = input_root / PROTOCOL_NAME
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing protocol: {protocol_path}")
    if args.revision is not None and (
        not args.revision
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in args.revision)
    ):
        raise AuditError("invalid_revision", "revision must contain only letters, digits, '-' or '_'")
    output_dir = output_path_for(protocol_path, args.output_parent, args.revision)
    if output_dir.exists():
        raise FileExistsError(f"create-only closeout already exists: {output_dir}")

    # Preliminary reads discover the closed input set.  Everything is reread
    # after the before-snapshot, and the complete set is hashed again at end.
    preliminary_protocol = load_json(protocol_path)
    preliminary_runs = load_json(input_root / RUNS_NAME)
    preliminary_attempts = load_json(input_root / ATTEMPTS_NAME)
    consumed_paths: set[Path] = {
        protocol_path,
        input_root / RUNS_NAME,
        input_root / ATTEMPTS_NAME,
        input_root / SUMMARY_NAME,
        input_root / ANALYSIS_NAME,
        Path(__file__).resolve(),
    }
    if isinstance(preliminary_protocol, Mapping):
        source = preliminary_protocol.get("source_sha256", {})
        if isinstance(source, Mapping):
            consumed_paths.update(ROOT / str(relative) for relative in source)
        data = preliminary_protocol.get("data", {})
        if isinstance(data, Mapping):
            for record in data.values():
                if isinstance(record, Mapping):
                    for split in ("train", "val"):
                        if isinstance(record.get(split), str):
                            consumed_paths.add(ROOT / record[split])
        commands = preliminary_protocol.get("commands", [])
        if isinstance(commands, list):
            for command in commands:
                if not isinstance(command, Mapping):
                    continue
                key = cell_tuple(command)
                if key is None:
                    continue
                directory = input_root / run_name(*key)
                consumed_paths.update(directory / name for name in CORE_ARTIFACTS)
    for ledger in (preliminary_runs, preliminary_attempts):
        if isinstance(ledger, list):
            for row in ledger:
                if isinstance(row, Mapping) and isinstance(row.get("log"), str):
                    consumed_paths.add(Path(row["log"]))

    before = snapshot_files(consumed_paths)
    protocol = load_json(protocol_path)
    runs = load_json(input_root / RUNS_NAME)
    attempts = load_json(input_root / ATTEMPTS_NAME)
    summary = load_json(input_root / SUMMARY_NAME)
    frozen_analysis = load_json(input_root / ANALYSIS_NAME)
    if not isinstance(protocol, Mapping):
        raise AuditError("protocol_schema", "development protocol is not an object")
    protocol_hash = before[relative_path(protocol_path)]["sha256"]
    events: list[dict[str, Any]] = []

    command_map, protocol_report, protocol_errors = audit_protocol(
        input_root, protocol, str(protocol_hash))
    for message in protocol_errors:
        add_event(events, "error", "protocol", message)
    expected_keys = {
        (task, design, seed) for task in TASKS for seed in SEEDS for design in DESIGNS
    }
    attempt_report, run_index, attempt_rows = audit_attempt_ledgers(
        expected_keys, command_map, runs, attempts)
    for message in attempt_report["errors"]:
        add_event(events, "error", "attempt_ledger", message)

    summary_errors: list[str] = []
    expected_summary = {
        "schema_version": 1,
        "scope": SCOPE,
        "status": "COMPLETE",
        "expected_cells": EXPECTED_CELLS,
        "completed_cells": EXPECTED_CELLS,
        "failed_or_invalid_cells": 0,
        "failures": [],
        "resume": False,
        "wandb_enabled": True,
    }
    if not isinstance(summary, Mapping):
        summary_errors.append("development summary is not an object")
    else:
        for field, expected in expected_summary.items():
            if summary.get(field) != expected:
                summary_errors.append(f"development summary {field} differs")
    if not isinstance(frozen_analysis, Mapping):
        summary_errors.append("frozen analysis is not an object")
    else:
        expected_analysis = {
            "scope": SCOPE,
            "status": "COMPLETE",
            "expected_cells": EXPECTED_CELLS,
            "completed_valid_cells": EXPECTED_CELLS,
            "artifact_integrity_passed": True,
            "artifact_integrity_errors": [],
            "development_protocol_sha256": protocol_hash,
            "commands_sha256": protocol.get("commands_sha256"),
            "official_confirmation_result": False,
        }
        for field, expected in expected_analysis.items():
            if frozen_analysis.get(field) != expected:
                summary_errors.append(f"frozen analysis {field} differs")
    for message in summary_errors:
        add_event(events, "error", "frozen_closeout_receipt", message)

    cell_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    metric_map: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    receipt_map: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    run_ids: dict[str, tuple[str, str, int]] = {}
    for index, key in enumerate(sorted(expected_keys), start=1):
        command = command_map.get(key)
        run_record = run_index.get(key)
        if command is None or run_record is None:
            continue
        try:
            cell, cell_history, metrics, receipt, run_id = audit_cell(
                input_root, key, command, run_record, protocol, before)
            if run_id in run_ids:
                raise AuditError(
                    "wandb_run_id_duplicate",
                    f"{cell_label(key)} shares W&B run ID with {cell_label(run_ids[run_id])}",
                )
            run_ids[run_id] = key
            cell_rows.append(cell)
            history_rows.extend(cell_history)
            metric_map[key] = metrics
            receipt_map[key] = receipt
            if not cell["rank_gate_pass"]:
                add_event(
                    events, "scientific_diagnostic", "rank_gate_failed",
                    "encoder covariance effective rank is below the reference gate",
                    cell=cell_label(key),
                )
            if not cell["predictive_convergence_gate_pass"]:
                add_event(
                    events, "scientific_diagnostic", "predictive_convergence_gate_failed",
                    "absolute late predictive-loss change exceeds the reference gate",
                    cell=cell_label(key),
                )
            if not cell["causality_gate_pass"]:
                add_event(
                    events, "scientific_diagnostic", "causality_gate_failed",
                    "singleton/prefix discrepancy exceeds the reference gate",
                    cell=cell_label(key),
                )
        except AuditError as exc:
            add_event(events, "error", exc.code, str(exc), cell=cell_label(key))
        except Exception as exc:  # keep an auditable closeout on unexpected input damage
            add_event(
                events, "error", "unexpected_cell_error",
                f"{type(exc).__name__}: {exc}", cell=cell_label(key),
            )
        if index % 24 == 0:
            print(f"audited {index}/{EXPECTED_CELLS} local cells", flush=True)

    parameter_errors = validate_parameter_parity(cell_rows) if cell_rows else [
        "no cells available for trainable-parameter parity"
    ]
    for message in parameter_errors:
        add_event(events, "error", "parameter_parity", message)

    contrasts: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    if len(metric_map) == EXPECTED_CELLS:
        try:
            contrasts = build_contrasts(metric_map)
            interactions = build_interactions(metric_map)
            diagnostics = diagnostics_by_design(cell_rows)
        except AuditError as exc:
            add_event(events, "error", exc.code, str(exc))
    else:
        add_event(
            events, "error", "scientific_grid_incomplete",
            f"scientific summaries require 144 valid cells; found {len(metric_map)}",
        )

    if args.remote_wandb and len(receipt_map) == EXPECTED_CELLS:
        remote = audit_remote_wandb(receipt_map, metric_map)
        for message in remote["errors"]:
            add_event(events, "error", "remote_wandb", message)
    elif args.remote_wandb:
        remote = {
            "status": "REMOTE_VERIFICATION_FAILED",
            "requested": True,
            "verified": False,
            "runs_checked": 0,
            "errors": ["local grid invalid; remote verification not attempted"],
            "runs": [],
        }
        add_event(events, "error", "remote_wandb", remote["errors"][0])
    else:
        remote = {
            "status": "UNVERIFIED_NOT_REQUESTED",
            "requested": False,
            "verified": None,
            "runs_checked": 0,
            "errors": [],
            "runs": [],
            "meaning": "local finished receipts are not proof of remote W&B state",
        }

    after = snapshot_files(consumed_paths)
    drift = [key for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
    for path in drift:
        add_event(events, "error", "consumed_file_drift", f"input changed during audit: {path}")

    integrity_errors = [event for event in events if event["level"] == "error"]
    local_integrity_errors = local_error_events(events)
    local_integrity = (
        not local_integrity_errors
        and len(cell_rows) == EXPECTED_CELLS
        and len(history_rows) == EXPECTED_CELLS * EPOCHS
        and len(run_ids) == EXPECTED_CELLS
        and len(contrasts) == 47
        and len(interactions) == 18
    )
    remote_verified = remote.get("verified") is True
    if not local_integrity:
        status = "INCOMPLETE_OR_INVALID"
    elif remote_verified:
        status = "COMPLETE_ADAPTIVE_DEVELOPMENT"
    elif args.remote_wandb:
        status = "LOCAL_COMPLETE_REMOTE_FAILED"
    else:
        status = "LOCAL_COMPLETE_REMOTE_UNVERIFIED"

    receipt_report = {
        "expected": EXPECTED_CELLS,
        "locally_valid_finished_receipts": len(receipt_map),
        "unique_run_ids": len(run_ids),
        "local_semantics_passed": len(receipt_map) == EXPECTED_CELLS and len(run_ids) == EXPECTED_CELLS,
        "remote_status": remote["status"],
        "remote_proof_inferred_from_local_receipts": False,
    }
    closeout = {
        "schema_version": 1,
        "created_at": utc_now(),
        "scope": SCOPE,
        "status": status,
        "input_root": str(input_root),
        "output_directory": str(output_dir),
        "closeout_revision": args.revision,
        "development_protocol_sha256": protocol_hash,
        "closeout_source_sha256": before[relative_path(Path(__file__))]["sha256"],
        "commands_sha256": protocol.get("commands_sha256"),
        "local_integrity_passed": local_integrity,
        "remote_integrity_passed": remote.get("verified"),
        "frozen_full_artifact_contract_passed": local_integrity and remote_verified,
        "official_confirmation_result": False,
        "adaptive_descriptive_only": True,
        "protocol_audit": protocol_report,
        "attempt_ledger_audit": attempt_report,
        "frozen_runner_analyzer_receipt_audit": {
            "passed": not summary_errors,
            "errors": summary_errors,
        },
        "local_wandb_receipt_audit": receipt_report,
        "counts": {
            "expected_cells": EXPECTED_CELLS,
            "valid_local_cells": len(cell_rows),
            "exact_epoch_rows": len(history_rows),
            "history_csv_rows": len(history_rows),
            "paired_contrasts": len(contrasts),
            "host_memory_interactions": len(interactions),
            "consumed_files": len(before),
            "consumed_file_drift": len(drift),
            "integrity_errors": len(integrity_errors),
            "scientific_diagnostic_events": sum(
                event["level"] == "scientific_diagnostic" for event in events),
            "nonfinite_events": sum(event["code"] == "nonfinite" for event in events),
        },
        "method_invariants": {
            "per_cell_checked": len(cell_rows),
            "regularizer_clean_target_only": True if len(cell_rows) == EXPECTED_CELLS else None,
            "clean_target_gradient_active": True if len(cell_rows) == EXPECTED_CELLS else None,
            "target_stop_gradient": False if len(cell_rows) == EXPECTED_CELLS else None,
            "subspace_counts_checked": [1, 16, 32],
            "ambient_dimension": 128,
            "projections_per_subspace": 512,
            "quadrature_knots": 17,
            "fresh_direction_policy_checked_from_pinned_source_and_receipts": True,
            "frozen_row_orthogonal_projection_tensors_checked": 108 if len(cell_rows) == EXPECTED_CELLS else None,
            "memory_specific_loss_weight": 0.0,
            "observation_correction_branch": False,
            "trainable_parameter_host_parity_passed": not parameter_errors,
        },
        "scientific_gates": {
            "integrity_status_affected_by_scientific_gate_failures": False,
            "rank_threshold": RANK_THRESHOLD,
            "predictive_convergence_abs_threshold": CONVERGENCE_ABS_THRESHOLD,
            "causality_abs_threshold": CAUSALITY_ABS_THRESHOLD,
            "variance_gate_invented": False,
        },
        "diagnostics_by_design": diagnostics,
        "paired_contrasts": contrasts,
        "host_memory_interactions": interactions,
        "consumed_file_hashes": {
            "before_manifest": "manifest.json:consumed_files_before",
            "after_manifest": "manifest.json:consumed_files_after",
            "drift_paths": drift,
        },
        "remote_wandb": {
            "status": remote["status"],
            "requested": remote["requested"],
            "verified": remote.get("verified"),
            "runs_checked": remote.get("runs_checked", 0),
            "errors": remote["errors"],
        },
        "events": events,
        "output_files": [
            "closeout.json", "cells.csv", "contrasts.csv", "interactions.csv",
            "history.csv", "attempts.csv", "events.csv", "remote_wandb.json",
            "manifest.json", "manifest.sha256",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    cell_fields = (
        "task", "design", "family", "memory", "seed", PRIMARY_METRIC,
        CLEAN_METRIC, VAL_PREDICTIVE_METRIC, INTEGRATOR_METRIC,
        "encoder_mean_channel_variance", "encoder_covariance_effective_rank",
        "encoder_singleton_max_abs", "encoder_prefix_max_abs",
        "predictive_loss_convergence_relative_change",
        "regularizer_loss_convergence_relative_change", "trainable_parameters",
        "mean_epoch_seconds", "peak_vram_bytes", "rank_gate_pass",
        "predictive_convergence_gate_pass", "causality_gate_pass",
        "projection_sha256", "projection_orthogonality_max_abs", "wandb_run_id",
        "local_bundle_passed",
    )
    history_fields = (
        "task", "design", "family", "memory", "seed", "epoch", "epoch_seconds",
        *(f"train_{key}" for key in HISTORY_KEYS),
        *(f"val_{key}" for key in HISTORY_KEYS),
    )
    attempt_fields = (
        "task", "design", "seed", "status", "resumed_existing", "command_sha256",
        "seconds", "gpu", "completed_at", "directory", "log", "artifact_sha256",
    )
    contrast_fields = tuple(_contrast_csv_rows(contrasts)[0]) if contrasts else (
        "comparison", "candidate_design", "reference_design", "metric")
    interaction_fields = tuple(_interaction_csv_rows(interactions)[0]) if interactions else (
        "interaction", "candidate_family", "reference_family", "memory", "metric")
    event_fields = ("level", "code", "cell", "message")
    payloads = {
        "closeout.json": json_text(closeout),
        "cells.csv": csv_text(cell_rows, cell_fields),
        "contrasts.csv": csv_text(_contrast_csv_rows(contrasts), contrast_fields),
        "interactions.csv": csv_text(_interaction_csv_rows(interactions), interaction_fields),
        "history.csv": csv_text(history_rows, history_fields),
        "attempts.csv": csv_text(attempt_rows, attempt_fields),
        "events.csv": csv_text(events, event_fields),
        "remote_wandb.json": json_text(remote),
    }
    for filename, payload in payloads.items():
        write_exclusive(output_dir / filename, payload)
    produced = {
        filename: {
            "size_bytes": (output_dir / filename).stat().st_size,
            "sha256": file_sha256(output_dir / filename),
        }
        for filename in sorted(payloads)
    }
    manifest = {
        "schema_version": 1,
        "development_protocol_sha256": protocol_hash,
        "closeout_revision": args.revision,
        "create_only_output": True,
        "consumed_files_before": before,
        "consumed_files_after": after,
        "consumed_file_drift": drift,
        "produced_files": produced,
        "manifest_self_hash_stored_in": "manifest.sha256",
    }
    write_exclusive(output_dir / "manifest.json", json_text(manifest))
    manifest_hash = file_sha256(output_dir / "manifest.json")
    write_exclusive(output_dir / "manifest.sha256", f"{manifest_hash}  manifest.json\n")
    print(json.dumps({
        "status": status,
        "local_integrity_passed": local_integrity,
        "remote_status": remote["status"],
        "valid_cells": len(cell_rows),
        "history_rows": len(history_rows),
        "output_directory": str(output_dir),
        "manifest_sha256": manifest_hash,
    }, indent=2, sort_keys=True))
    return 0 if local_integrity and (not args.remote_wandb or remote_verified) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, FileExistsError, FileNotFoundError) as exc:
        print(f"closeout failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
