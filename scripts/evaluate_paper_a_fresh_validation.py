#!/usr/bin/env python3
"""Evaluate an existing frozen carrier on one fresh validation bank.

The source checkpoint and original training cache are read-only.  A fresh
readout is fit on the locked parent training bank and evaluated on the named
fresh bank; no carrier or LeWM parameter is trained or changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import (  # noqa: E402
    FROZEN_CARRIER_NAMES,
    make_frozen_carrier,
)
from lewm.models.official_lewm import (  # noqa: E402
    OFFICIAL_ACTION_DIM,
    OFFICIAL_EMBED_DIM,
)
from scripts.cache_official_lewm import _atomic_text, _stable_json  # noqa: E402
from scripts.cache_paper_a_fresh_validation import cache_directory  # noqa: E402
from scripts.paper_a_robustness_spec import (  # noqa: E402
    DEFAULT_SPEC,
    PARENT_SEEDS,
    TASKS,
    load_locked_spec,
    resolve_spec_path,
    sha256_file,
    validate_device,
)
from scripts.reevaluate_frozen_official_probes import (  # noqa: E402
    Cell,
    load_config,
    preflight_cell,
    state_dict_digest,
)
from scripts.train_frozen_official_swap import (  # noqa: E402
    carrier_outputs,
    load_cache,
    probe_categorical,
    probe_categorical_trajectory,
)


def evaluation_directory(spec: Mapping, task: str, bank_id: str,
                         arm: str, seed: int) -> Path:
    root = resolve_spec_path(
        spec, spec["output"]["fresh_validation_evaluation"])
    return root / task / bank_id / arm / f"s{seed}"


def _load_fresh_manifest(spec: Mapping, task: str,
                         bank_id: str) -> tuple[Mapping, Path]:
    directory = cache_directory(spec, task, bank_id)
    manifest_path = directory / "manifest.json"
    digest_path = directory / "manifest.sha256"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read fresh-cache manifest {manifest_path}: {error}") from error
    tokens = digest_path.read_text().strip().split() if digest_path.is_file() else []
    actual = sha256_file(manifest_path)
    if tokens != [actual, "manifest.json"]:
        raise ValueError(f"fresh-cache manifest hash mismatch: {manifest_path}")
    if manifest.get("task") != task or manifest.get("bank_id") != bank_id:
        raise ValueError("fresh-cache manifest identity mismatch")
    if manifest.get("strengthening_spec") != spec["_spec_record"]:
        raise ValueError("fresh-cache manifest uses a different robustness spec")
    availability = manifest.get("availability", {})
    if availability.get("passed") is not True:
        raise ValueError(
            f"fresh bank {task}/{bank_id} failed representation availability")
    cache_path = directory / "val.npz"
    if manifest.get("artifact", {}).get("sha256") != sha256_file(cache_path):
        raise ValueError(f"fresh validation cache hash mismatch: {cache_path}")
    return manifest, cache_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--arm", required=True, choices=FROZEN_CARRIER_NAMES)
    parser.add_argument("--seed", required=True, type=int,
                        choices=PARENT_SEEDS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec)
    if args.arm not in spec["fresh_validation"]["checkpoint_arms"]:
        raise ValueError(f"arm {args.arm!r} is outside the fresh-bank grid")
    if args.seed not in spec["fresh_validation"]["checkpoint_seeds"]:
        raise ValueError(f"seed {args.seed} is outside the fresh-bank grid")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device_name = args.device or spec["execution"]["default_device"]
    validate_device(spec, device_name, allow_cpu=True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")

    parent_config_record = spec["parent"]["config"]
    parent_config_path = resolve_spec_path(spec, parent_config_record["path"])
    parent_config = load_config(parent_config_path)
    parent_checkpoint_root = resolve_spec_path(
        spec, spec["parent"]["frozen_checkpoint_root"])
    cell = Cell(args.task, args.arm, args.seed)
    prepared = preflight_cell(cell, parent_checkpoint_root, parent_config)
    source_directory = prepared.directory

    train_record = spec["parent"]["train_caches"][args.task]
    train_path = resolve_spec_path(spec, train_record["path"])
    if sha256_file(train_path) != train_record["sha256"]:
        raise ValueError("parent training cache changed after robustness lock")
    fresh_manifest, validation_path = _load_fresh_manifest(
        spec, args.task, args.bank)
    train = load_cache(train_path)
    validation = load_cache(validation_path)
    if train.get("meta", {}).get("task") != args.task \
            or validation.get("meta", {}).get("task") != args.task:
        raise ValueError("cache task identity mismatch")

    output = evaluation_directory(
        spec, args.task, args.bank, args.arm, args.seed)
    result_path = output / "metrics.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite fresh evaluation {output}")

    carrier = make_frozen_carrier(
        args.arm, OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM)
    carrier.load_state_dict(
        prepared.checkpoint["carrier_state_dict"], strict=True)
    carrier = carrier.to(device)
    before = state_dict_digest(carrier.state_dict())
    if before != prepared.state_sha256:
        raise RuntimeError("device-loaded carrier state differs from checkpoint")
    train_z = np.asarray(train["z"], dtype=np.float32)
    train_actions = np.asarray(train["actions"], dtype=np.float32)
    val_z = np.asarray(validation["z"], dtype=np.float32)
    val_actions = np.asarray(validation["actions"], dtype=np.float32)
    _, train_prior = carrier_outputs(
        carrier, train_z, train_actions, device, args.batch_size)
    _, validation_prior = carrier_outputs(
        carrier, val_z, val_actions, device, args.batch_size)
    after = state_dict_digest(carrier.state_dict())
    if before != after:
        raise RuntimeError("fresh validation evaluation mutated carrier state")

    probe = probe_categorical(
        train, train_prior, validation, validation_prior)
    trajectory = probe_categorical_trajectory(
        train, train_prior, validation, validation_prior)
    result = {
        "schema_version": 1,
        "study": "paper-a-reacher-fresh-validation-v1",
        "task": args.task,
        "bank_id": args.bank,
        "arm": args.arm,
        "seed": args.seed,
        "training_performed": False,
        "host_instantiated": False,
        "parent_artifacts_modified": False,
        "strengthening_spec": spec["_spec_record"],
        "source_checkpoint": {
            "metrics_path": str((source_directory / "metrics.json").relative_to(ROOT)),
            "metrics_sha256": sha256_file(source_directory / "metrics.json"),
            "checkpoint_path": str((source_directory / "carrier.pt").relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(source_directory / "carrier.pt"),
            "carrier_state_sha256": before,
        },
        "parent_training_cache": train_record,
        "fresh_validation_cache": {
            "path": str(validation_path.relative_to(ROOT)),
            "sha256": sha256_file(validation_path),
            "manifest_sha256": sha256_file(
                cache_directory(spec, args.task, args.bank) / "manifest.json"),
            "availability": fresh_manifest["availability"],
        },
        "probe": probe,
        "trajectory_probe": trajectory,
        "carrier_state_unchanged": True,
    }
    output.mkdir(parents=True, exist_ok=False)
    _atomic_text(result_path, _stable_json(result))
    print(
        f"[robust-eval] {args.task}/{args.bank}/{args.arm}/s{args.seed}: "
        f"accuracy={probe['mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
