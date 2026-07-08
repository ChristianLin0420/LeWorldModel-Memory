#!/usr/bin/env python3
"""Causally reevaluate frozen-carrier decision probes without retraining.

The registered categorical endpoint is

    [flatten(z[:, q-3:q]), prior_read[:, q]],  q = L - 1,

where ``prior_read[:, q]`` is produced before ``z[:, q]`` is observed.  The
final decision observation is therefore never a probe input.  This program
loads the already-trained carrier state, recomputes train/validation priors,
and atomically synchronizes ``metrics.json`` with the metrics embedded in
``carrier.pt``.  It never instantiates or optimizes the LeWM host.

Use ``--task/--arm/--seed`` for one cell or ``--all`` for the complete grid.
The complete-grid mode preflights every expected checkpoint before changing
any cell, so an incomplete or identity-mismatched grid fails closed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import (
    FROZEN_CARRIER_NAMES,
    make_frozen_carrier,
    parameter_report,
)
from lewm.models.official_lewm import OFFICIAL_ACTION_DIM, OFFICIAL_EMBED_DIM
from scripts.train_frozen_official_swap import (
    TASKS,
    carrier_outputs,
    load_cache,
    probe_categorical,
    probe_categorical_trajectory,
    state_digest,
)


DEFAULT_CONFIG = ROOT / "configs/paper_a_expansion.yaml"
DEFAULT_CACHE_ROOT = ROOT / "outputs/paper_a_expansion/cache"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/paper_a_expansion/frozen_swap"
MUTABLE_METRIC_KEYS = {"probe", "trajectory_probe"}
LEARNED_CARRIER_DESCRIPTION_KEYS = {"decay_init"}


@dataclass(frozen=True)
class Cell:
    task: str
    arm: str
    seed: int

    @property
    def label(self) -> str:
        return f"{self.task}/{self.arm}/s{self.seed}"


@dataclass
class PreparedCell:
    cell: Cell
    directory: Path
    metrics: dict[str, Any]
    checkpoint: dict[str, Any]
    state_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"carrier state {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"carrier state {name!r} contains a non-finite value")
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"cannot safely load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a checkpoint mapping")
    if not isinstance(value.get("carrier_state_dict"), Mapping):
        raise ValueError(f"{path} misses carrier_state_dict")
    if not isinstance(value.get("metrics"), dict):
        raise ValueError(f"{path} misses embedded metrics")
    return value


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read config {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    frozen = value.get("frozen_carrier_swap")
    host = value.get("official_host")
    if not isinstance(frozen, dict) or not isinstance(host, dict):
        raise ValueError(f"{path} misses frozen_carrier_swap/official_host")
    return value


def expected_cells(config: Mapping[str, Any]) -> list[Cell]:
    wave = config["frozen_carrier_swap"]
    return [
        Cell(str(task), str(arm), int(seed))
        for task in wave["tasks"]
        for arm in wave["arms"]
        for seed in wave["seeds"]
    ]


def _same_number(actual: Any, expected: Any) -> bool:
    return (isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and math.isclose(float(actual), float(expected),
                             rel_tol=1e-10, abs_tol=1e-12))


def _static_carrier_description(value: Any) -> Any:
    """Remove diagnostic values recomputed from trainable state.

    The diagonal SSM historically called its fitted decay vector
    ``decay_init``.  That vector is already authenticated by the complete
    carrier-state digest and can differ by one float32 rounding step when
    reconstructed through ``sigmoid``.  Every immutable architecture and
    causal-convention field remains part of this identity comparison.
    """

    if not isinstance(value, dict):
        return value
    return {
        key: item for key, item in value.items()
        if key not in LEARNED_CARRIER_DESCRIPTION_KEYS
    }


def _validate_cache(cache: Mapping[str, Any], cell: Cell, split: str,
                    config: Mapping[str, Any]) -> None:
    z = np.asarray(cache["z"])
    actions = np.asarray(cache["actions"])
    labels = np.asarray(cache["xi"])
    meta = cache.get("meta")
    if z.shape[1:] != (64, OFFICIAL_EMBED_DIM):
        raise ValueError(
            f"{cell.label} {split} cache has unexpected z shape {z.shape}")
    if actions.shape != (len(z), 63, OFFICIAL_ACTION_DIM):
        raise ValueError(
            f"{cell.label} {split} cache has unexpected action shape {actions.shape}")
    if labels.shape != (len(z),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{cell.label} requires one categorical label per episode")
    if not isinstance(meta, dict):
        raise ValueError(f"{cell.label} {split} cache has no metadata object")
    host = config["official_host"]
    official = meta.get("official_checkpoint", {})
    checks = {
        "task": meta.get("task") == cell.task,
        "split": meta.get("split") == split,
        "schema": meta.get("schema") == "official_lewm_reacher_latents_v1",
        "source_stream": meta.get("source_stream") == "clean",
        "label_training": meta.get("representation_label_training") is False,
        "checkpoint_source": official.get("source") == host["source"],
        "checkpoint_hash": official.get("sha256") == host["weights_sha256"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"{cell.label} {split} cache identity failed: {', '.join(failed)}")


def load_cell_caches(cell: Cell, cache_root: Path,
                     config: Mapping[str, Any]) -> tuple[dict, dict]:
    train_path = cache_root / cell.task / "train.npz"
    val_path = cache_root / cell.task / "val.npz"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(f"{cell.label}: missing train/validation cache")
    train = load_cache(train_path)
    validation = load_cache(val_path)
    _validate_cache(train, cell, "train", config)
    _validate_cache(validation, cell, "val", config)
    return train, validation


def preflight_cell(cell: Cell, output_root: Path,
                   config: Mapping[str, Any]) -> PreparedCell:
    directory = output_root / cell.task / cell.arm / f"s{cell.seed}"
    required = [directory / name for name in (
        "history.csv", "metrics.json", "carrier.pt", "eval_export.npz")]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{cell.label}: incomplete trained cell; missing {missing}")

    metrics = _load_json(directory / "metrics.json")
    checkpoint = _load_checkpoint(directory / "carrier.pt")
    if checkpoint["metrics"] != metrics:
        raise ValueError(
            f"{cell.label}: metrics.json and carrier.pt metrics disagree")

    wave = config["frozen_carrier_swap"]
    host = config["official_host"]
    expected_epochs = 0 if cell.arm == "none" else int(wave["epochs"])
    identity_checks = {
        "schema_version": metrics.get("schema_version") == 1,
        "study": metrics.get("study") == "official-lewm-frozen-carrier-swap",
        "task": metrics.get("task") == cell.task,
        "arm": metrics.get("arm") == cell.arm,
        "seed": metrics.get("seed") == cell.seed,
        "official_host": metrics.get("official_host") == host["source"],
        "frozen_host": metrics.get("frozen_host_unchanged") is True,
        "host_trainable_parameters": metrics.get("host_trainable_parameters") == 0,
        "epochs": metrics.get("epochs") == expected_epochs,
        "batch_size": _same_number(metrics.get("batch_size"), wave["batch_size"]),
        "learning_rate": _same_number(
            metrics.get("learning_rate"), wave["learning_rate"]),
    }
    failed = [name for name, passed in identity_checks.items() if not passed]
    if failed:
        raise ValueError(
            f"{cell.label}: metric identity failed: {', '.join(failed)}")
    before = metrics.get("official_host_state_sha256_before")
    after = metrics.get("official_host_state_sha256_after")
    if not isinstance(before, str) or len(before) != 64 or before != after:
        raise ValueError(f"{cell.label}: frozen host state hashes are invalid")
    mse = metrics.get("val_next_latent_mse")
    if not _same_number(mse, mse) or float(mse) < 0:
        raise ValueError(f"{cell.label}: validation next-latent MSE is invalid")

    carrier = make_frozen_carrier(
        cell.arm, OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM)
    if metrics.get("carrier_parameters") != carrier.parameter_count():
        raise ValueError(f"{cell.label}: carrier parameter count mismatch")
    if metrics.get("parameter_matching") != parameter_report(
            OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM):
        raise ValueError(f"{cell.label}: parameter-matching ledger mismatch")
    try:
        carrier.load_state_dict(checkpoint["carrier_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"{cell.label}: carrier checkpoint state is incompatible: {error}") from error
    # Architecture/convention metadata is checked directly.  Trainable state
    # is checked independently below by a byte-level digest.
    if _static_carrier_description(metrics.get("carrier_config")) != \
            _static_carrier_description(carrier.describe()):
        raise ValueError(f"{cell.label}: carrier configuration mismatch")
    state_sha256 = state_dict_digest(checkpoint["carrier_state_dict"])
    if state_digest(carrier) != _legacy_state_digest(
            checkpoint["carrier_state_dict"]):
        raise ValueError(f"{cell.label}: loaded carrier state digest mismatch")
    prior_reevaluation = metrics.get("probe", {}).get("reevaluation", {})
    recorded_state = (prior_reevaluation.get("carrier_state_sha256")
                      if isinstance(prior_reevaluation, dict) else None)
    if recorded_state is not None and recorded_state != state_sha256:
        raise ValueError(
            f"{cell.label}: prior reevaluation records a different carrier state")
    return PreparedCell(cell, directory, metrics, checkpoint, state_sha256)


def _legacy_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    """Match train_frozen_official_swap.state_digest exactly."""

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _protected_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in metrics.items()
            if key not in MUTABLE_METRIC_KEYS}


def _temporary_path(target: Path, tag: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.{tag}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    return Path(name)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stage_json(target: Path, value: Mapping[str, Any]) -> Path:
    temporary = _temporary_path(target, "new")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
    _fsync_file(temporary)
    return temporary


def _stage_checkpoint(target: Path, value: Mapping[str, Any]) -> Path:
    temporary = _temporary_path(target, "new")
    torch.save(value, temporary)
    os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
    _fsync_file(temporary)
    return temporary


def _backup_file(target: Path) -> Path:
    backup = _temporary_path(target, "rollback")
    shutil.copyfile(target, backup)
    os.chmod(backup, stat.S_IMODE(target.stat().st_mode))
    _fsync_file(backup)
    return backup


def atomic_publish_metrics_pair(directory: Path, metrics: Mapping[str, Any],
                                checkpoint: Mapping[str, Any]) -> None:
    """Atomically replace both files, rolling back a caught partial failure."""

    metrics_path = directory / "metrics.json"
    checkpoint_path = directory / "carrier.pt"
    staged_metrics = _stage_json(metrics_path, metrics)
    staged_checkpoint = _stage_checkpoint(checkpoint_path, checkpoint)
    backup_metrics = _backup_file(metrics_path)
    backup_checkpoint = _backup_file(checkpoint_path)
    replaced_metrics = replaced_checkpoint = False
    try:
        os.replace(staged_checkpoint, checkpoint_path)
        replaced_checkpoint = True
        os.replace(staged_metrics, metrics_path)
        replaced_metrics = True
        _fsync_directory(directory)
    except BaseException:
        if replaced_checkpoint:
            os.replace(backup_checkpoint, checkpoint_path)
        if replaced_metrics:
            os.replace(backup_metrics, metrics_path)
        _fsync_directory(directory)
        raise
    finally:
        for path in (staged_metrics, staged_checkpoint,
                     backup_metrics, backup_checkpoint):
            path.unlink(missing_ok=True)


def _acquire_cell_lock(directory: Path) -> tuple[int, Path]:
    path = directory / ".probe_reevaluation.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"refusing concurrent reevaluation: {path}") from error
    os.write(descriptor, f"pid={os.getpid()}\n".encode())
    os.fsync(descriptor)
    return descriptor, path


def reevaluate_cell(prepared: PreparedCell, train: dict, validation: dict,
                    device: torch.device, batch_size: int,
                    cache_hashes: Mapping[str, str]) -> dict[str, Any]:
    cell = prepared.cell
    descriptor, lock_path = _acquire_cell_lock(prepared.directory)
    try:
        carrier = make_frozen_carrier(
            cell.arm, OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM)
        carrier.load_state_dict(
            prepared.checkpoint["carrier_state_dict"], strict=True)
        carrier = carrier.to(device)
        before = state_dict_digest(carrier.state_dict())
        if before != prepared.state_sha256:
            raise RuntimeError(f"{cell.label}: device-loaded carrier state changed")

        train_z = np.asarray(train["z"], dtype=np.float32)
        train_actions = np.asarray(train["actions"], dtype=np.float32)
        val_z = np.asarray(validation["z"], dtype=np.float32)
        val_actions = np.asarray(validation["actions"], dtype=np.float32)
        _, train_prior = carrier_outputs(
            carrier, train_z, train_actions, device, batch_size)
        _, val_prior = carrier_outputs(
            carrier, val_z, val_actions, device, batch_size)
        after = state_dict_digest(carrier.state_dict())
        if before != after:
            raise RuntimeError(f"{cell.label}: evaluation mutated carrier state")

        probe = probe_categorical(train, train_prior, validation, val_prior)
        trajectory = probe_categorical_trajectory(
            train, train_prior, validation, val_prior)
        reevaluation = {
            "mode": "checkpoint_only_no_retraining",
            "training_performed": False,
            "host_instantiated": False,
            "carrier_state_sha256": before,
            "carrier_state_unchanged": True,
            "train_cache_sha256": cache_hashes["train"],
            "validation_cache_sha256": cache_hashes["validation"],
            "checkpoint_metrics_synchronized": True,
        }
        probe["reevaluation"] = reevaluation
        trajectory["reevaluation"] = copy.deepcopy(reevaluation)

        updated = copy.deepcopy(prepared.metrics)
        protected_before = _protected_metrics(updated)
        updated["probe"] = probe
        updated["trajectory_probe"] = trajectory
        if _protected_metrics(updated) != protected_before:
            raise RuntimeError(
                f"{cell.label}: reevaluation altered protected training/MSE fields")
        new_checkpoint = copy.deepcopy(prepared.checkpoint)
        new_checkpoint["metrics"] = copy.deepcopy(updated)
        if state_dict_digest(new_checkpoint["carrier_state_dict"]) != before:
            raise RuntimeError(f"{cell.label}: checkpoint state changed before publish")

        atomic_publish_metrics_pair(
            prepared.directory, updated, new_checkpoint)

        published_metrics = _load_json(prepared.directory / "metrics.json")
        published_checkpoint = _load_checkpoint(
            prepared.directory / "carrier.pt")
        if published_metrics != updated or published_checkpoint["metrics"] != updated:
            raise RuntimeError(f"{cell.label}: published metrics are not synchronized")
        if state_dict_digest(
                published_checkpoint["carrier_state_dict"]) != before:
            raise RuntimeError(f"{cell.label}: published carrier state changed")
        if _protected_metrics(published_metrics) != protected_before:
            raise RuntimeError(f"{cell.label}: published training/MSE fields changed")
        return updated
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="reevaluate the complete preregistered 50-cell grid")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--arm", choices=FROZEN_CARRIER_NAMES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def _selected_cells(args: argparse.Namespace,
                    config: Mapping[str, Any]) -> list[Cell]:
    expected = expected_cells(config)
    if args.all:
        if any(value is not None for value in (args.task, args.arm, args.seed)):
            raise ValueError("--all cannot be combined with --task/--arm/--seed")
        return expected
    if any(value is None for value in (args.task, args.arm, args.seed)):
        raise ValueError("one-cell mode requires --task, --arm, and --seed")
    cell = Cell(args.task, args.arm, args.seed)
    if cell not in expected:
        raise ValueError(f"cell {cell.label} is outside the preregistered grid")
    return [cell]


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    config = load_config(args.config)
    cells = _selected_cells(args, config)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
    device = torch.device(args.device)

    # Complete preflight before the first mutation.  This is what makes an
    # accidentally early --all invocation fail closed while training runs.
    prepared = [preflight_cell(cell, args.output_root, config) for cell in cells]
    cache_by_task: dict[str, tuple[dict, dict]] = {}
    hashes_by_task: dict[str, dict[str, str]] = {}
    for task in sorted({cell.task for cell in cells}):
        representative = next(cell for cell in cells if cell.task == task)
        cache_by_task[task] = load_cell_caches(
            representative, args.cache_root, config)
        hashes_by_task[task] = {
            "train": sha256_file(args.cache_root / task / "train.npz"),
            "validation": sha256_file(args.cache_root / task / "val.npz"),
        }
    print(f"[reevaluate] preflight passed for {len(prepared)} cell(s)", flush=True)

    for index, item in enumerate(prepared, start=1):
        train, validation = cache_by_task[item.cell.task]
        old_score = item.metrics.get("probe", {}).get("mean")
        updated = reevaluate_cell(
            item, train, validation, device, args.batch_size,
            hashes_by_task[item.cell.task])
        print(
            f"[reevaluate] {index}/{len(prepared)} {item.cell.label}: "
            f"accuracy {old_score} -> {updated['probe']['mean']:.6f}; "
            "state/training/MSE preserved",
            flush=True,
        )


if __name__ == "__main__":
    main()
