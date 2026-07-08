#!/usr/bin/env python3
"""Aggregate matched-token TwoRoom external-use cells."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import atomic_text, stable_json  # noqa: E402
from scripts.aggregate_paper_a_tworoom_use import _summarize, hierarchical_paired_bootstrap  # noqa: E402
from scripts.evaluate_paper_a_matched_token_use import _deck, use_cell_directory  # noqa: E402
from scripts.paper_a_matched_token_spec import (  # noqa: E402
    ARMS, DEFAULT_SHA, DEFAULT_SPEC, SEEDS, load_locked_spec, output_path,
    sha256_file,
)


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _load(spec, deck):
    values = {key: np.empty((5, 5, 480), dtype=np.float64)
              for key in ("goal_correct", "executed_success",
                          "selected_distance", "distance_regret")}
    labels, joint = deck["token_label"], deck["combination_label"]
    success, distance = deck["success_matrix"], deck["distance_matrix"]
    random_choice, rows = deck["random_choice"], np.arange(480)
    records, digests, expected_dirs = {}, {}, set()
    for ai, arm in enumerate(ARMS):
        for si, seed in enumerate(SEEDS):
            directory = use_cell_directory(spec, arm, seed)
            expected_dirs.add(directory.resolve())
            mp, manifest_path = directory / "metrics.json", directory / "manifest.json"
            metrics, manifest = json.loads(mp.read_text()), json.loads(manifest_path.read_text())
            if metrics.get("lock") != spec["_lock"] \
                    or metrics.get("arm") != arm or metrics.get("seed") != seed \
                    or metrics.get("consumer_rows") != 6000 \
                    or metrics.get("consumer_training_arms") != list(ARMS) \
                    or manifest["artifacts"]["metrics"]["sha256"] != sha256_file(mp):
                raise ValueError("matched-token use cell identity differs")
            digest = metrics["consumer_state_sha256"]
            consumer = metrics.get("consumer", {})
            training_sources = consumer.get("training_sources", {})
            state_before = metrics.get("carrier_state_sha256_before")
            if not isinstance(digest, str) or len(digest) != 64 \
                    or not isinstance(state_before, str) or len(state_before) != 64 \
                    or consumer.get("parameter_sha256") != digest \
                    or set(training_sources) != set(ARMS) \
                    or any(source.get("state_unchanged") is not True
                           for source in training_sources.values()) \
                    or state_before != metrics.get("carrier_state_sha256_after"):
                raise ValueError("shared consumer provenance differs")
            if digests.setdefault(seed, digest) != digest:
                raise ValueError("shared consumer differs across arms")
            prediction = np.asarray(metrics["predictions"], dtype=np.int64)
            if prediction.shape != (480,) \
                    or not np.array_equal(metrics.get("labels"), labels) \
                    or not np.array_equal(metrics.get("joint_labels"), joint) \
                    or not np.isin(prediction, np.arange(4)).all():
                raise ValueError("use episode pairing differs")
            expected = {"goal_correct": prediction == labels,
                        "executed_success": success[rows, prediction, labels],
                        "selected_distance": distance[rows, prediction, labels],
                        "distance_regret": distance[rows, prediction, labels]
                        - distance[rows, labels, labels]}
            for key, value in expected.items():
                current = np.asarray(metrics[key], dtype=np.float64)
                if not np.allclose(current, value, rtol=0, atol=1e-6):
                    raise ValueError(f"use {key} not deck-derived")
                values[key][ai, si] = current
            records[f"{arm}/seed-{seed}"] = {
                "metrics": {"path": str(mp.relative_to(ROOT)),
                            "sha256": sha256_file(mp)},
                "manifest": {"path": str(manifest_path.relative_to(ROOT)),
                             "sha256": sha256_file(manifest_path)}}
    actual = {path.parent.resolve() for path in output_path(
        spec, "use").glob("cells/*/seed-*/metrics.json")}
    if actual != expected_dirs:
        raise ValueError("matched-token use cell grid differs")
    baselines = {"realized_random_goal_correct": (random_choice == labels).astype(float),
                 "realized_random_success": success[rows, random_choice, labels],
                 "realized_random_distance": distance[rows, random_choice, labels],
                 "realized_random_regret": distance[rows, random_choice, labels]
                 - distance[rows, labels, labels],
                 "oracle_goal_correct": np.ones(480),
                 "oracle_success": success[rows, labels, labels],
                 "oracle_distance": distance[rows, labels, labels],
                 "oracle_regret": np.zeros(480)}
    return values, baselines, records, digests


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing matched-token use aggregation without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    root = output_path(spec, "use")
    if (root / "summary.json").exists() or (root / "final_audit.json").exists():
        raise FileExistsError("matched-token use aggregation exists")
    deck, provenance = _deck(spec)
    values, baselines, records, digests = _load(spec, deck)
    use = spec["tworoom_use"]
    points, samples, bp, bs = hierarchical_paired_bootstrap(
        values, deck["combination_label"], baselines,
        draws=use["bootstrap_draws"], seed=use["bootstrap_seed"])
    analysis = _summarize(values, points, samples, bp, bs)
    summary = {"schema_version": 1, "study": spec["study"],
               "branch": "matched-token-tworoom-external-waypoint-use",
               "lock": spec["_lock"], "status": "complete",
               "host": "tworoom", "target": "token", "nuisance": "location",
               "cue_age": 15, "episodes": 480, "arms": list(ARMS),
               "seeds": list(SEEDS), "bootstrap": {"draws": 20000,
               "seed": 20260975, "paired_seed_and_16_way_episode_resampling": True},
               "no_pooled_memory_score": True, **analysis,
               "claim_boundary": use["claim_boundary"]}
    summary_path = root / "summary.json"
    digest = atomic_text(summary_path, stable_json(summary))
    atomic_text(root / "final_audit.json", stable_json({
        "schema_version": 1, "study": spec["study"], "lock": spec["_lock"],
        "status": "complete", "complete_cells": 25, "expected_cells": 25,
        "physical_gpu_counts": {"0": 25}, "execution_deck": provenance,
        "cell_artifacts": records,
        "shared_consumer_sha256_by_seed": {str(k): v for k, v in digests.items()},
        "summary": {"path": str(summary_path.relative_to(ROOT)),
                    "sha256": digest}, "bootstrap_draws": 20000,
        "cuda3_used": False}))


if __name__ == "__main__":
    main()
