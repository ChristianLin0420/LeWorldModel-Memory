#!/usr/bin/env python3
"""Evaluate one existing-checkpoint evidence-age read-time cell."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import make_frozen_carrier
from lewm.official_tasks.artifacts import atomic_text, stable_json
from lewm.official_tasks.pusht_pipeline import pusht_carrier_directory
from lewm.official_tasks.pusht_spec import (
    load_locked_pusht_spec,
    pusht_lock_receipt,
)
from scripts.paper_a_evidence_age import (
    age_name,
    configure_determinism,
    endpoint_features,
    fit_readout,
    read_indices,
)
from scripts.paper_a_evidence_age_spec import (
    ARMS,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    SEEDS,
    host_tasks,
    load_locked_spec,
    output_root,
    resolve_path,
    sha256_file,
    validate_device,
)
from scripts.reevaluate_frozen_official_probes import Cell, load_config, preflight_cell
from scripts.train_frozen_official_swap import carrier_outputs, load_cache, state_digest
from scripts.train_official_pusht_carrier import _carrier_prior, _load_admitted


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--host", required=True, choices=HOSTS)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _safe_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"cannot safely load {path}: {error}") from error
    if not isinstance(value, dict) \
            or not isinstance(value.get("carrier_state_dict"), Mapping) \
            or not isinstance(value.get("metrics"), dict):
        raise ValueError(f"malformed carrier checkpoint {path}")
    return value


def _load_reacher(spec: dict[str, Any], task: str, arm: str, seed: int,
                  device: torch.device
                  ) -> tuple[dict, dict, np.ndarray, np.ndarray, dict]:
    parent = spec["parents"]["reacher"]
    config = load_config(resolve_path(parent["config"]["path"]))
    cell = preflight_cell(
        Cell(task, arm, seed), resolve_path(parent["checkpoint_root"]), config)
    train = load_cache(resolve_path(f"{parent['cache_root']}/{task}/train.npz"))
    validation = load_cache(resolve_path(f"{parent['cache_root']}/{task}/val.npz"))
    carrier = make_frozen_carrier(arm, 192, 10).to(device)
    carrier.load_state_dict(cell.checkpoint["carrier_state_dict"], strict=True)
    before = state_digest(carrier)
    _, train_prior = carrier_outputs(
        carrier, train["z"].astype(np.float32),
        train["actions"].astype(np.float32), device)
    _, validation_prior = carrier_outputs(
        carrier, validation["z"].astype(np.float32),
        validation["actions"].astype(np.float32), device)
    if state_digest(carrier) != before:
        raise RuntimeError("read-time inference mutated Reacher carrier")
    checkpoint = cell.directory / "carrier.pt"
    return train, validation, train_prior, validation_prior, {
        "path": str(checkpoint.relative_to(ROOT)),
        "sha256": sha256_file(checkpoint),
        "carrier_state_sha256": before,
    }


def _load_pusht(spec: dict[str, Any], task: str, arm: str, seed: int,
                device: torch.device
                ) -> tuple[dict, dict, np.ndarray, np.ndarray, dict]:
    parent_record = spec["parents"]["pusht"]
    parent = load_locked_pusht_spec(
        resolve_path(parent_record["config"]["path"]),
        resolve_path(parent_record["lock"]["path"]))
    train = _load_admitted(parent, task, "train")
    validation = _load_admitted(parent, task, "validation")
    directory = pusht_carrier_directory(parent, task, arm, seed)
    paths = {name: directory / name for name in
             ("carrier.pt", "metrics.json", "manifest.json")}
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"incomplete PushT carrier cell {directory}")
    manifest = json.loads(paths["manifest.json"].read_text())
    metrics = json.loads(paths["metrics.json"].read_text())
    if manifest.get("formal_lock") != pusht_lock_receipt(parent) \
            or metrics.get("formal_lock") != pusht_lock_receipt(parent) \
            or metrics.get("task_key") != task \
            or metrics.get("arm") != arm \
            or metrics.get("seed") != seed:
        raise ValueError(f"PushT carrier provenance mismatch: {directory}")
    artifacts = manifest.get("artifacts", {})
    if artifacts.get("checkpoint", {}).get("sha256") != sha256_file(
            paths["carrier.pt"]) \
            or artifacts.get("metrics", {}).get("sha256") != sha256_file(
                paths["metrics.json"]):
        raise ValueError(f"PushT carrier artifact hash mismatch: {directory}")
    checkpoint = _safe_checkpoint(paths["carrier.pt"])
    if checkpoint["metrics"] != metrics:
        raise ValueError(f"PushT checkpoint metrics mismatch: {directory}")
    carrier = make_frozen_carrier(arm, 192, 10).to(device)
    carrier.load_state_dict(checkpoint["carrier_state_dict"], strict=True)
    before = state_digest(carrier)
    train_prior = _carrier_prior(
        carrier, train["z"].astype(np.float32),
        train["actions"].astype(np.float32), device)
    validation_prior = _carrier_prior(
        carrier, validation["z"].astype(np.float32),
        validation["actions"].astype(np.float32), device)
    if state_digest(carrier) != before:
        raise RuntimeError("read-time inference mutated PushT carrier")
    return train, validation, train_prior, validation_prior, {
        "path": str(paths["carrier.pt"].relative_to(ROOT)),
        "sha256": sha256_file(paths["carrier.pt"]),
        "carrier_state_sha256": before,
    }


def _ages(spec: Mapping[str, Any], host: str) -> list[int | str]:
    return list(spec["read_time"][f"{host}_ages"])


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing evidence-age writes without --execute")
    spec = load_locked_spec(args.spec, args.sha)
    validate_device(spec, args.device)
    if args.task not in host_tasks(spec, args.host):
        raise ValueError(f"task {args.task!r} is not registered for {args.host}")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError("evidence-age evaluation requires physical cuda:0")
    device = torch.device(args.device)
    configure_determinism(args.seed)
    destination = (output_root(spec, "read_time") / args.host / args.task
                   / f"seed-{args.seed}" / "metrics.json")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")

    load = _load_reacher if args.host == "reacher" else _load_pusht
    arms: dict[str, Any] = {}
    labels = None
    for arm in ARMS:
        train, validation, train_prior, validation_prior, artifact = load(
            spec, args.task, arm, args.seed, device)
        train_y = np.asarray(
            train["xi"] if args.host == "reacher" else train["labels"],
            dtype=np.int64)
        validation_y = np.asarray(
            validation["xi"] if args.host == "reacher" else validation["labels"],
            dtype=np.int64)
        if labels is None:
            labels = validation_y
        elif not np.array_equal(labels, validation_y):
            raise ValueError("validation labels changed across carrier arms")
        if args.host == "reacher":
            train_off = np.asarray(train["event_cue_off"], dtype=np.int64)
            validation_off = np.asarray(
                validation["event_cue_off"], dtype=np.int64)
        else:
            cue_off = (int(spec["parents"]["pusht"].get("cue_off", 4))
                       if "cue_off" in spec["parents"]["pusht"] else 4)
            train_off = np.full(len(train_y), cue_off, dtype=np.int64)
            validation_off = np.full(len(validation_y), cue_off, dtype=np.int64)
        age_results = {}
        for age in _ages(spec, args.host):
            train_q = read_indices(train_off, age, length=train["z"].shape[1])
            validation_q = read_indices(
                validation_off, age, length=validation["z"].shape[1])
            result = fit_readout(
                endpoint_features(train["z"], train_prior, train_q), train_y,
                endpoint_features(
                    validation["z"], validation_prior, validation_q),
                validation_y, spec["read_time"]["readout"],
                balanced=args.host == "pusht")
            result["endpoint"] = {
                "evidence_age": age,
                "read_index_min": int(validation_q.min()),
                "read_index_max": int(validation_q.max()),
                "context_history": 3,
                "current_observation_excluded": True,
            }
            age_results[age_name(age)] = result
        arms[arm] = {"artifact": artifact, "ages": age_results}
        del train_prior, validation_prior
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "study": spec["study"],
        "branch": "existing-checkpoint-read-time",
        "lock": spec["_lock"],
        "host": args.host,
        "task": args.task,
        "seed": args.seed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "labels": labels.tolist(),
        "arms": arms,
        "host_checkpoint_training_performed": False,
        "carrier_training_performed": False,
        "validation_labels_used_for_fitting": False,
        "current_observation_excluded": True,
    }
    atomic_text(destination, stable_json(payload))
    print(f"[evidence-age/read-time] wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
