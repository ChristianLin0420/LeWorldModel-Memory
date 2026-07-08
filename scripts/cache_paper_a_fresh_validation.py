#!/usr/bin/env python3
"""Encode one locked fresh validation bank with the official LeWM encoder."""

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

from lewm.models.official_lewm import load_official_reacher_checkpoint  # noqa: E402
from scripts.cache_official_lewm import (  # noqa: E402
    FRAME_BATCH_SIZE,
    OFFICIAL_WEIGHTS_SHA256,
    _atomic_text,
    _spaced_indices,
    _stable_json,
    cache_split,
    categorical_availability,
    configure_determinism,
    sha256_file,
)
from scripts.make_paper_a_robustness_data import bank_path  # noqa: E402
from scripts.paper_a_robustness_spec import (  # noqa: E402
    DEFAULT_SPEC,
    TASKS,
    load_locked_spec,
    resolve_spec_path,
    validate_device,
)
from scripts.train_frozen_official_swap import load_cache  # noqa: E402


def cache_directory(spec: Mapping, task: str, bank_id: str) -> Path:
    root = resolve_spec_path(spec, spec["output"]["fresh_validation_cache"])
    return root / task / bank_id


def cached_cue_features(data: Mapping[str, np.ndarray]) -> np.ndarray:
    z = np.asarray(data["z"], dtype=np.float32)
    cue_on = np.asarray(data["event_cue_on"], dtype=np.int64)
    cue_off = np.asarray(data["event_cue_off"], dtype=np.int64)
    indices = _spaced_indices(cue_on, cue_off - 1)
    selected = z[np.arange(len(z))[:, None], indices]
    return selected.reshape(len(z), -1)


def _read_receipt(source: Path, spec: Mapping, task: str,
                  bank_id: str) -> Mapping:
    path = source.parent / "collection_receipt.json"
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read fresh-bank receipt {path}: {error}") from error
    if receipt.get("task") != task or receipt.get("bank_id") != bank_id:
        raise ValueError(f"fresh-bank receipt identity mismatch: {path}")
    if receipt.get("strengthening_spec") != spec["_spec_record"]:
        raise ValueError(f"fresh-bank receipt uses a different spec: {path}")
    artifact = receipt.get("artifact", {})
    if artifact.get("sha256") != sha256_file(source):
        raise ValueError(f"fresh-bank receipt hash mismatch: {source}")
    return receipt


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--frame-batch-size", type=int,
                        default=FRAME_BATCH_SIZE)
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec)
    device_name = args.device or spec["execution"]["default_device"]
    validate_device(spec, device_name, allow_cpu=True)
    if args.frame_batch_size < 1:
        raise ValueError("--frame-batch-size must be positive")
    if not 0 <= args.compression_level <= 9:
        raise ValueError("--compression-level must be in [0,9]")

    source = bank_path(spec, args.task, args.bank)
    if not source.is_file():
        raise FileNotFoundError(f"fresh validation bank is missing: {source}")
    receipt = _read_receipt(source, spec, args.task, args.bank)
    destination_dir = cache_directory(spec, args.task, args.bank)
    destination = destination_dir / "val.npz"
    manifest_path = destination_dir / "manifest.json"
    candidates = [destination, destination.with_suffix(".npz.json"),
                  manifest_path, destination_dir / "manifest.sha256"]
    existing = [path for path in candidates if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite fresh cache: "
            + ", ".join(str(path) for path in existing))

    train_record = spec["parent"]["train_caches"][args.task]
    train_path = resolve_spec_path(spec, train_record["path"])
    if sha256_file(train_path) != train_record["sha256"]:
        raise ValueError("parent training cache changed after robustness lock")
    train = load_cache(train_path)
    metadata = train.get("meta", {})
    if metadata.get("task") != args.task:
        raise ValueError("parent training cache task mismatch")
    transform = metadata.get("action_transform", {})
    action_mean = np.asarray(transform.get("training_mean"), dtype=np.float64)
    action_std = np.asarray(transform.get("training_std_ddof0"), dtype=np.float64)
    if action_mean.shape != (10,) or action_std.shape != (10,):
        raise ValueError("parent training cache lacks the locked 10-D action transform")

    weights_record = spec["parent"]["official_weights"]
    weights = resolve_spec_path(spec, weights_record["path"])
    weights_sha256 = sha256_file(weights)
    if weights_sha256 != OFFICIAL_WEIGHTS_SHA256 \
            or weights_sha256 != weights_record["sha256"]:
        raise ValueError("official weights changed after robustness lock")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    configure_determinism(0)
    model = load_official_reacher_checkpoint(weights, device).eval()
    for parameter in model.parameters():
        if parameter.requires_grad:
            raise AssertionError("official host must be frozen")

    destination_dir.mkdir(parents=True, exist_ok=False)
    record, validation_features, validation_targets = cache_split(
        model=model,
        device=device,
        source_path=source,
        destination=destination,
        task=args.task,
        split="val",
        source_stream="clean",
        action_mean=action_mean,
        action_std=action_std,
        weights_sha256=weights_sha256,
        frame_batch_size=args.frame_batch_size,
        compression_level=args.compression_level,
        overwrite=False,
        quiet=args.quiet,
    )
    availability = categorical_availability(
        cached_cue_features(train), np.asarray(train["xi"]),
        validation_features, validation_targets)
    threshold = float(spec["fresh_validation"]["categorical_availability_min"])
    availability.update({
        "task": args.task,
        "bank_id": args.bank,
        "train_episodes": int(len(train["xi"])),
        "validation_episodes": int(len(validation_targets)),
        "threshold": threshold,
        "passed": bool(availability["value"] >= threshold),
        "representation_frozen": True,
        "representation_label_training": False,
    })
    manifest = {
        "schema_version": 1,
        "study": spec["study"],
        "task": args.task,
        "bank_id": args.bank,
        "strengthening_spec": spec["_spec_record"],
        "source_collection_receipt_sha256": sha256_file(
            source.parent / "collection_receipt.json"),
        "source_collection": receipt,
        "parent_training_cache": train_record,
        "official_weights": weights_record,
        "artifact": record,
        "availability": availability,
        "parent_artifacts_modified": False,
    }
    manifest_sha256 = _atomic_text(manifest_path, _stable_json(manifest))
    _atomic_text(destination_dir / "manifest.sha256",
                 f"{manifest_sha256}  manifest.json\n")
    print(
        f"[robust-cache] {args.task}/{args.bank}: "
        f"availability={availability['value']:.4f} "
        f"passed={availability['passed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
