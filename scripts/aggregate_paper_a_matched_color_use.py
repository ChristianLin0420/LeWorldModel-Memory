#!/usr/bin/env python3
"""Audit and aggregate Wave-1b TwoRoom color-to-waypoint use."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import atomic_text, stable_json  # noqa: E402
from scripts.aggregate_paper_a_tworoom_use import (  # noqa: E402
    _summarize, hierarchical_paired_bootstrap,
)
from scripts.evaluate_paper_a_matched_color_use import (  # noqa: E402
    _load_deck, use_cell_directory,
)
from scripts.paper_a_matched_color_spec import (  # noqa: E402
    ARMS, DEFAULT_SHA, DEFAULT_SPEC, SEEDS, load_locked_spec, output_path,
    sha256_file,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def _load_cells(spec: dict, deck: Mapping[str, np.ndarray]
                ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    values = {
        key: np.empty((len(ARMS), len(SEEDS), 480), dtype=np.float64)
        for key in ("goal_correct", "executed_success", "selected_distance",
                    "distance_regret")}
    labels = np.asarray(deck["color_label"], dtype=np.int64)
    joint = np.asarray(deck["combination_label"], dtype=np.int64)
    random_choice = np.asarray(deck["random_choice"], dtype=np.int64)
    success = np.asarray(deck["success_matrix"], dtype=np.int64)
    distance = np.asarray(deck["distance_matrix"], dtype=np.float64)
    rows = np.arange(480)
    records: dict[str, Any] = {}
    consumer_digest: dict[int, str] = {}
    expected_dirs: set[Path] = set()
    for arm_index, arm in enumerate(ARMS):
        for seed_index, seed in enumerate(SEEDS):
            directory = use_cell_directory(spec, arm, seed)
            expected_dirs.add(directory.resolve())
            metrics_path = directory / "metrics.json"
            manifest_path = directory / "manifest.json"
            if not metrics_path.is_file() or not manifest_path.is_file() \
                    or {path.name for path in directory.iterdir()} \
                    != {"metrics.json", "manifest.json"}:
                raise FileNotFoundError(f"incomplete Wave-1b use cell {directory}")
            metrics = json.loads(metrics_path.read_text())
            manifest = json.loads(manifest_path.read_text())
            expected = {
                "schema_version": 1, "study": spec["study"],
                "branch": "matched-color-tworoom-external-waypoint-use",
                "lock": spec["_lock"], "host": "tworoom",
                "target": "color", "nuisance": "location", "cue_age": 15,
                "arm": arm, "seed": seed, "device": "cuda:0",
                "physical_gpu": 0, "arm_blind_consumer": True,
                "consumer_training_arms": list(ARMS), "consumer_rows": 6000,
                "consumer_seed": seed, "validation_labels_used_for_fitting": False,
                "carrier_state_unchanged": True,
            }
            failed = [key for key, value in expected.items()
                      if metrics.get(key) != value]
            if failed or manifest.get("lock") != spec["_lock"] \
                    or manifest.get("arm") != arm \
                    or manifest.get("seed") != seed \
                    or manifest.get("artifacts", {}).get(
                        "metrics", {}).get("sha256") != sha256_file(metrics_path):
                raise ValueError(f"invalid Wave-1b use cell {directory}: {failed}")
            digest = metrics.get("consumer_state_sha256")
            if not isinstance(digest, str) or len(digest) != 64 \
                    or metrics.get("consumer", {}).get(
                        "parameter_sha256") != digest:
                raise ValueError("Wave-1b shared consumer digest differs")
            prior = consumer_digest.setdefault(seed, digest)
            if prior != digest:
                raise ValueError(f"consumer differs across arms for seed {seed}")
            prediction = np.asarray(metrics["predictions"], dtype=np.int64)
            if prediction.shape != (480,) \
                    or not np.array_equal(metrics["labels"], labels) \
                    or not np.array_equal(metrics["joint_labels"], joint):
                raise ValueError("Wave-1b use episode pairing differs")
            expected_arrays = {
                "goal_correct": (prediction == labels).astype(np.int8),
                "executed_success": success[rows, prediction, labels],
                "selected_distance": distance[rows, prediction, labels],
                "distance_regret": (
                    distance[rows, prediction, labels]
                    - distance[rows, labels, labels]),
            }
            for key, expected_values in expected_arrays.items():
                current = np.asarray(metrics[key], dtype=np.float64)
                if current.shape != (480,) or not np.allclose(
                        current, expected_values, rtol=0, atol=1e-6):
                    raise ValueError(f"Wave-1b use {key} is not deck-derived")
                values[key][arm_index, seed_index] = current
            records[f"{arm}/seed-{seed}"] = {
                "metrics": _record(metrics_path),
                "manifest": _record(manifest_path),
            }
    cells_root = output_path(spec, "use") / "cells"
    actual = {path.parent.resolve() for path in cells_root.glob(
        "*/seed-*/metrics.json")}
    if actual != expected_dirs:
        raise ValueError("Wave-1b use directory grid differs")
    baselines = {
        "realized_random_goal_correct": (random_choice == labels).astype(float),
        "realized_random_success": success[rows, random_choice, labels].astype(float),
        "realized_random_distance": distance[rows, random_choice, labels],
        "realized_random_regret": (
            distance[rows, random_choice, labels]
            - distance[rows, labels, labels]),
        "oracle_goal_correct": np.ones(480, dtype=float),
        "oracle_success": success[rows, labels, labels].astype(float),
        "oracle_distance": distance[rows, labels, labels],
        "oracle_regret": np.zeros(480, dtype=float),
    }
    return values, {
        "baselines": baselines, "joint": joint, "cells": records,
        "shared_consumer_sha256_by_seed": {
            str(seed): consumer_digest[seed] for seed in SEEDS},
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing Wave-1b use aggregation without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    root = output_path(spec, "use")
    destination, audit_path = root / "summary.json", root / "final_audit.json"
    if destination.exists() or audit_path.exists():
        raise FileExistsError("Wave-1b use aggregation already exists")
    deck, deck_provenance = _load_deck(spec)
    values, loaded = _load_cells(spec, deck)
    use = spec["tworoom_use"]
    points, samples, baseline_points, baseline_samples = \
        hierarchical_paired_bootstrap(
            values, loaded["joint"], loaded["baselines"],
            draws=int(use["bootstrap_draws"]), seed=int(use["bootstrap_seed"]))
    analysis = _summarize(
        values, points, samples, baseline_points, baseline_samples)
    summary = {
        "schema_version": 1, "study": spec["study"],
        "branch": "matched-color-tworoom-external-waypoint-use",
        "lock": spec["_lock"], "status": "complete",
        "host": "tworoom", "target": "color", "nuisance": "location",
        "cue_age": 15, "episodes": 480,
        "arms": list(ARMS), "seeds": list(SEEDS),
        "bootstrap": {
            "draws": use["bootstrap_draws"], "seed": use["bootstrap_seed"],
            "joint_seed_resampling": True,
            "shared_16_way_stratified_episode_resampling": True,
            "paired": True,
        },
        "no_pooled_memory_score": True, **analysis,
        "claim_boundary": use["claim_boundary"],
    }
    summary_hash = atomic_text(destination, stable_json(summary))
    audit = {
        "schema_version": 1, "study": spec["study"],
        "branch": summary["branch"], "lock": spec["_lock"],
        "status": "complete", "complete_cells": 25,
        "expected_cells": 25, "physical_gpu_counts": {"0": 25},
        "execution_deck": deck_provenance,
        "cell_artifacts": loaded["cells"],
        "shared_consumer_sha256_by_seed": loaded[
            "shared_consumer_sha256_by_seed"],
        "summary": {"path": str(destination.relative_to(ROOT)),
                    "sha256": summary_hash},
        "bootstrap_draws": use["bootstrap_draws"], "cuda3_used": False,
    }
    atomic_text(audit_path, stable_json(audit))
    print(f"[matched-color/use] wrote {destination} and {audit_path}", flush=True)


if __name__ == "__main__":
    main()
