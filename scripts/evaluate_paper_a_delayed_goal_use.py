#!/usr/bin/env python3
"""Evaluate one locked delayed-goal executed-choice cell."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import make_frozen_carrier
from lewm.models.official_lewm import OFFICIAL_ACTION_DIM, OFFICIAL_EMBED_DIM
from scripts.paper_a_delayed_goal_spec import (
    CARRIER_ARMS,
    DEFAULT_SPEC,
    REPAIR_ARMS,
    REPAIR_CONDITIONS,
    SEEDS,
    SOURCE_IDS,
    evaluation_directory,
    load_locked_spec,
    repair_directory,
    resolve_path,
    sha256_file,
    source_slug,
    validate_device,
)
from scripts.paper_a_delayed_goal_use import (
    action_time_interface,
    carrier_interface,
    cue_window_interface,
    decision_metrics,
    execute_reacher_choices,
    fit_shared_consumer,
    fit_shortcut_consumer,
    long_context_interface,
)
from scripts.reevaluate_frozen_official_probes import (
    Cell,
    load_config,
    preflight_cell,
)
from scripts.train_frozen_official_swap import (
    carrier_outputs,
    load_cache,
    state_digest,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--task", required=True, choices=("t1", "t3"))
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--device", default=None)
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
        raise ValueError(f"malformed repair checkpoint {path}")
    return value


def _load_repair_carrier(spec: Mapping[str, Any], task: str, arm: str,
                         seed: int, condition: str,
                         parent_state_sha256: str,
                         device: torch.device) -> tuple[torch.nn.Module, dict]:
    directory = repair_directory(spec, task, arm, seed, condition)
    checkpoint_path = directory / "repair.pt"
    metrics_path = directory / "metrics.json"
    history_path = directory / "history.csv"
    if not all(path.is_file() for path in (
            checkpoint_path, metrics_path, history_path)):
        raise FileNotFoundError(f"incomplete repair cell {directory}")
    checkpoint = _safe_checkpoint(checkpoint_path)
    metrics = json.loads(metrics_path.read_text())
    if checkpoint["metrics"] != metrics:
        raise ValueError(f"repair checkpoint/metrics mismatch: {directory}")
    expected = {
        "study": spec["study"], "task": task, "arm": arm,
        "condition": condition, "seed": seed,
        "spec": spec["_spec_record"],
        "parent_carrier_state_sha256": parent_state_sha256,
        "label_arrays_loaded": False, "label_values_consumed": False,
        "validation_used_for_optimization": False,
        "final_frame_excluded_from_target": True,
        "repair_read": spec["repair"]["read"],
        "target_gradient": spec["repair"]["target_gradient"],
        "epochs": spec["repair"]["epochs"],
        "batch_size": spec["repair"]["batch_size"],
        "learning_rate": spec["repair"]["learning_rate"],
        "weight_decay": spec["repair"]["weight_decay"],
        "next_latent_weight": spec["repair"]["next_latent_weight"],
        "optimizer": "AdamW", "scheduler": "CosineAnnealingLR",
        "source_caches": {
            "train": spec["parent"]["train_caches"][task],
            "validation": spec["parent"]["validation_caches"][task],
        },
    }
    failed = [key for key, value in expected.items()
              if metrics.get(key) != value]
    if failed or metrics.get("target_frame_index_min", -1) < 0 \
            or metrics.get("target_frame_index_max", 63) >= 63 \
            or metrics.get("validation_target_frame_index_max", 63) >= 63:
        raise ValueError(
            f"repair provenance/leakage check failed for {directory}: {failed}")
    if metrics.get("repair_weight") != spec["repair"][
            "cue_repair_weight"][condition]:
        raise ValueError(f"repair weight changed for {directory}")
    host_before = metrics.get("official_host_state_sha256_before")
    if not isinstance(host_before, str) or len(host_before) != 64 \
            or metrics.get("official_host_state_sha256_after") != host_before:
        raise ValueError(f"frozen host digest mismatch for {directory}")
    carrier = make_frozen_carrier(
        arm, OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM).to(device)
    carrier.load_state_dict(checkpoint["carrier_state_dict"], strict=True)
    if state_digest(carrier) != metrics.get("carrier_state_sha256"):
        raise ValueError(f"repair carrier state digest mismatch: {directory}")
    decoder = torch.nn.Linear(OFFICIAL_EMBED_DIM, 4 * OFFICIAL_EMBED_DIM)
    try:
        decoder.load_state_dict(checkpoint["repair_head_state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError(f"repair head state mismatch: {directory}") from error
    if state_digest(decoder) != metrics.get("repair_head_state_sha256"):
        raise ValueError(f"repair head digest mismatch: {directory}")
    target_mean = np.asarray(checkpoint.get("target_mean"), dtype=np.float32)
    target_scale = np.asarray(checkpoint.get("target_scale"), dtype=np.float32)
    if target_mean.shape != (4 * OFFICIAL_EMBED_DIM,) \
            or target_scale.shape != target_mean.shape \
            or not np.isfinite(target_mean).all() \
            or not np.isfinite(target_scale).all() \
            or np.any(target_scale <= 0):
        raise ValueError(f"repair target normalization mismatch: {directory}")
    return carrier, {
        "path": str(checkpoint_path.relative_to(ROOT)),
        "sha256": sha256_file(checkpoint_path),
        "carrier_state_sha256": metrics["carrier_state_sha256"],
        "parent_carrier_state_sha256": parent_state_sha256,
        "repair_head_initial_state_sha256": metrics[
            "repair_head_initial_state_sha256"],
        "device": metrics["device"],
        "cuda_device_name": metrics["cuda_device_name"],
    }


def _consumer_kwargs(spec: Mapping[str, Any]) -> dict[str, Any]:
    consumer = spec["consumer"]
    return {
        "c": float(consumer["logistic_c"]),
        "solver": str(consumer["solver"]),
        "max_iter": int(consumer["max_iter"]),
        "random_state": int(consumer["random_state"]),
    }


def _source_record(spec: Mapping[str, Any], source: str,
                   metrics: Mapping[str, Any]) -> dict[str, Any]:
    registered = next(item for item in spec["representation_sources"]
                      if item["id"] == source)
    return {
        "id": source,
        "name": registered["name"],
        "slug": registered["slug"],
        "kind": registered["kind"],
        "metrics": dict(metrics),
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing evaluation without explicit --execute")
    spec = load_locked_spec(args.spec)
    device_name = args.device or spec["execution"]["default_device"]
    validate_device(spec, device_name)
    if not torch.cuda.is_available():
        raise RuntimeError("locked use evaluation requires CUDA; no CPU fallback")
    device = torch.device(device_name)
    output_dir = evaluation_directory(spec, args.task, args.seed)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    parent_config = load_config(resolve_path(spec["parent"]["config"]["path"]))
    train = load_cache(resolve_path(
        spec["parent"]["train_caches"][args.task]["path"]))
    validation = load_cache(resolve_path(
        spec["parent"]["validation_caches"][args.task]["path"]))
    train_labels = np.asarray(train["xi"], dtype=np.int64)
    validation_labels = np.asarray(validation["xi"], dtype=np.int64)
    train_features: dict[str, np.ndarray] = {}
    validation_features: dict[str, np.ndarray] = {}
    artifacts: dict[str, Any] = {}
    prepared: dict[str, Any] = {}

    for arm in CARRIER_ARMS:
        cell = preflight_cell(
            Cell(args.task, arm, args.seed),
            resolve_path(spec["parent"]["checkpoint_root"]), parent_config)
        prepared[arm] = cell
        carrier = make_frozen_carrier(
            arm, OFFICIAL_EMBED_DIM, OFFICIAL_ACTION_DIM).to(device)
        carrier.load_state_dict(cell.checkpoint["carrier_state_dict"], strict=True)
        before = state_digest(carrier)
        _, train_prior = carrier_outputs(
            carrier, np.asarray(train["z"], dtype=np.float32),
            np.asarray(train["actions"], dtype=np.float32), device)
        _, validation_prior = carrier_outputs(
            carrier, np.asarray(validation["z"], dtype=np.float32),
            np.asarray(validation["actions"], dtype=np.float32), device)
        if state_digest(carrier) != before:
            raise RuntimeError(f"evaluation mutated frozen {arm} carrier")
        train_features[arm] = carrier_interface(train, train_prior)
        validation_features[arm] = carrier_interface(
            validation, validation_prior)
        checkpoint_path = cell.directory / "carrier.pt"
        artifacts[arm] = {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": sha256_file(checkpoint_path),
            "carrier_state_sha256": cell.state_sha256,
        }

    repair_initializations: dict[str, set[str]] = {
        arm: set() for arm in REPAIR_ARMS}
    repair_devices: dict[str, set[tuple[str, str]]] = {
        arm: set() for arm in REPAIR_ARMS}
    for arm in REPAIR_ARMS:
        for condition in REPAIR_CONDITIONS:
            source = f"{arm}_{condition}"
            carrier, record = _load_repair_carrier(
                spec, args.task, arm, args.seed, condition,
                prepared[arm].state_sha256, device)
            repair_initializations[arm].add(
                record["repair_head_initial_state_sha256"])
            repair_devices[arm].add(
                (record["device"], record["cuda_device_name"]))
            before = state_digest(carrier)
            _, train_prior = carrier_outputs(
                carrier, np.asarray(train["z"], dtype=np.float32),
                np.asarray(train["actions"], dtype=np.float32), device)
            _, validation_prior = carrier_outputs(
                carrier, np.asarray(validation["z"], dtype=np.float32),
                np.asarray(validation["actions"], dtype=np.float32), device)
            if state_digest(carrier) != before:
                raise RuntimeError(f"evaluation mutated {source}")
            train_features[source] = carrier_interface(train, train_prior)
            validation_features[source] = carrier_interface(
                validation, validation_prior)
            artifacts[source] = record
    if any(len(values) != 1 for values in repair_initializations.values()):
        raise ValueError("repair condition pair did not share head initialization")
    if any(len(values) != 1 for values in repair_devices.values()):
        raise ValueError("repair condition pair did not share one CUDA device")

    train_features["long_context_56"] = long_context_interface(train)
    validation_features["long_context_56"] = long_context_interface(validation)
    train_features["cue_window"] = cue_window_interface(train)
    validation_features["cue_window"] = cue_window_interface(validation)
    consumer = fit_shared_consumer(
        train_features, train_labels, list(SOURCE_IDS),
        **_consumer_kwargs(spec))
    permutation = np.random.default_rng(
        int(spec["controls"]["label_shuffle"]["seed_base"]) + args.seed
    ).permutation(len(train_labels))
    shuffled_consumer = fit_shared_consumer(
        train_features, train_labels, list(SOURCE_IDS),
        label_permutation=permutation, **_consumer_kwargs(spec))
    shortcut_consumer = fit_shortcut_consumer(
        action_time_interface(train), train_labels, spec["consumer"])

    execution = execute_reacher_choices(
        np.asarray(validation["endo_state"], dtype=np.float64)[:, 63],
        validation_labels, spec["executed_choice"])
    sources: dict[str, Any] = {}
    shuffled: dict[str, Any] = {}
    for source in SOURCE_IDS:
        sources[source_slug(spec, source)] = _source_record(
            spec, source, decision_metrics(
                consumer.predict(validation_features[source]),
                validation_labels, execution))
        shuffled[source_slug(spec, source)] = _source_record(
            spec, source, decision_metrics(
                shuffled_consumer.predict(validation_features[source]),
                validation_labels, execution))
    shortcut_metrics = decision_metrics(
        shortcut_consumer.predict(action_time_interface(validation)),
        validation_labels, execution)
    oracle_metrics = decision_metrics(
        validation_labels, validation_labels, execution)
    max_shuffled = max(record["metrics"]["goal_decision_accuracy"]
                       for record in shuffled.values())
    shortcut_limit = float(spec["controls"]["shortcut_accuracy_max"])
    validity = {
        "label_oracle_success_pass": (
            oracle_metrics["executed_success_rate"] >=
            float(spec["executed_choice"]["oracle_success_min"])),
        "label_shuffle_accuracy_pass": max_shuffled <= shortcut_limit,
        "action_time_accuracy_pass": (
            shortcut_metrics["goal_decision_accuracy"] <= shortcut_limit),
        "max_label_shuffle_accuracy": max_shuffled,
        "shortcut_accuracy_limit": shortcut_limit,
    }
    validity["valid_for_use_claim"] = all(
        validity[key] for key in (
            "label_oracle_success_pass", "label_shuffle_accuracy_pass",
            "action_time_accuracy_pass"))
    result = {
        "schema_version": 1,
        "study": spec["study"],
        "spec": spec["_spec_record"],
        "task": args.task,
        "task_name": spec["tasks"][args.task]["name"],
        "task_slug": spec["tasks"][args.task]["slug"],
        "checkpoint_seed": args.seed,
        "consumer": {
            "digest": consumer.digest(),
            "shared_across_all_sources": True,
            "fit_split": "authenticated parent training bank only",
            "source_order": list(SOURCE_IDS),
            "validation_labels_available_during_fit": False,
        },
        "labels": validation_labels.tolist(),
        "sources": sources,
        "label_oracle": oracle_metrics,
        "controls": {
            "label_shuffle": shuffled,
            "label_shuffle_consumer_digest": shuffled_consumer.digest(),
            "training_episode_permutation": permutation.tolist(),
            "action_time": shortcut_metrics,
            "action_time_consumer_digest": shortcut_consumer.digest(),
        },
        "repair_pairing": {
            arm: {"shared_head_initialization": True,
                  "repair_head_initial_state_sha256": next(iter(values)),
                  "shared_cuda_device": True,
                  "cuda_device": list(next(iter(repair_devices[arm])))}
            for arm, values in repair_initializations.items()
        },
        "validity": validity,
        "artifacts": artifacts,
        "claim_boundary": spec["claim_boundary"],
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        (stage / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"[delayed-goal-use] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
