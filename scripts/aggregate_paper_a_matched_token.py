#!/usr/bin/env python3
"""Fail-closed hierarchical aggregation for matched composite-token recall."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import atomic_text, stable_json  # noqa: E402
from scripts.paper_a_matched_token_spec import (  # noqa: E402
    AGES, ARMS, DEFAULT_SHA, DEFAULT_SPEC, HOSTS, SEEDS, load_locked_spec,
    output_path, sha256_file,
)
from scripts.prepare_paper_a_matched_token import host_manifest_path  # noqa: E402
from scripts.train_paper_a_matched_token import carrier_directory  # noqa: E402


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _stat(point: float, samples: np.ndarray, seeds: np.ndarray) -> dict[str, Any]:
    return {"mean": float(point),
            "ci95": np.quantile(samples, (.025, .975)).astype(float).tolist(),
            "ci90": np.quantile(samples, (.05, .95)).astype(float).tolist(),
            "seed_values": np.asarray(seeds, dtype=float).tolist()}


def _weights(joint: np.ndarray, draws: int,
             rng: np.random.Generator) -> np.ndarray:
    if not np.array_equal(
            np.bincount(joint, minlength=16), np.full(16, 30)):
        raise ValueError("matched-token labels are not 16-way balanced")
    weights = np.zeros((draws, 480), dtype=np.float32)
    rows = np.arange(draws)[:, None]
    for label in range(16):
        indices = np.flatnonzero(joint == label)
        selected = indices[rng.integers(0, 30, size=(draws, 30))]
        np.add.at(weights, (np.broadcast_to(rows, selected.shape), selected), 1)
    return weights / 480.0


def _load(spec: dict):
    correct = np.empty((3, 5, 5, 3, 480), dtype=np.float32)
    observed = np.empty((3, 5, 5, 3), dtype=np.float64)
    joint = np.empty((3, 480), dtype=np.int64)
    location = np.empty((3, 480), dtype=np.int64)
    records, expected_dirs = {}, set()
    admissions = {}
    for hi, host in enumerate(HOSTS):
        admission_path = host_manifest_path(spec, host)
        admission = json.loads(admission_path.read_text())
        if admission.get("lock") != spec["_lock"] \
                or admission.get("status") != "admitted" \
                or admission.get("all_token_ages_admitted") is not True \
                or admission.get("frozen_host_unchanged") is not True:
            raise ValueError(f"host not admitted: {host}")
        admissions[host] = {"path": str(admission_path.relative_to(ROOT)),
                            "sha256": sha256_file(admission_path)}
        reference = None
        for ai, arm in enumerate(ARMS):
            for si, seed in enumerate(SEEDS):
                directory = carrier_directory(spec, host, arm, seed)
                expected_dirs.add(directory.resolve())
                paths = {"metrics": directory / "metrics.json",
                         "checkpoint": directory / "carrier.pt",
                         "history": directory / "history.csv",
                         "manifest": directory / "manifest.json"}
                if not all(path.is_file() for path in paths.values()):
                    raise FileNotFoundError(f"incomplete cell {directory}")
                metrics = json.loads(paths["metrics"].read_text())
                manifest = json.loads(paths["manifest"].read_text())
                expected = {"schema_version": 1, "study": spec["study"],
                            "branch": "matched-composite-token-fixed-endpoint",
                            "lock": spec["_lock"],
                            "host": host, "target": "token",
                            "nuisance": "location", "arm": arm, "seed": seed,
                            "device": "cuda:0", "physical_gpu": 0,
                            "frozen_host_unchanged": True,
                            "validation_labels_used_for_fitting": False,
                            "v1_metrics_used_for_inference": False}
                if any(metrics.get(k) != v for k, v in expected.items()) \
                        or metrics.get("ages") != list(AGES) \
                        or manifest.get("schema_version") != 1 \
                        or manifest.get("study") != spec["study"] \
                        or manifest.get("lock") != spec["_lock"] \
                        or manifest.get("host") != host \
                        or manifest.get("arm") != arm \
                        or manifest.get("seed") != seed \
                        or set(manifest.get("artifacts", {})) \
                        != {"metrics", "checkpoint", "history"}:
                    raise ValueError(f"cell identity differs {directory}")
                artifacts = manifest["artifacts"]
                for name in ("metrics", "checkpoint", "history"):
                    if artifacts[name]["sha256"] != sha256_file(paths[name]):
                        raise ValueError(f"cell hash differs {directory}")
                current_ref = (metrics["labels"],
                               metrics["nuisance_location_labels"],
                               metrics["joint_labels"])
                if reference is None:
                    reference = current_ref
                elif current_ref != reference:
                    raise ValueError("labels differ across cells")
                for gi, age in enumerate(AGES):
                    result = metrics["readouts"][f"age-{age}"]
                    current = np.asarray(result["correct"], dtype=np.float64)
                    if current.shape != (480,) \
                            or abs(current.mean()
                                   - result["balanced_accuracy"]) > 1e-12:
                        raise ValueError("readout receipt differs")
                    correct[hi, ai, si, gi] = current
                    observed[hi, ai, si, gi] = result["balanced_accuracy"]
                records[f"{host}/{arm}/seed-{seed}"] = {
                    name: {"path": str(path.relative_to(ROOT)),
                           "sha256": sha256_file(path)}
                    for name, path in paths.items()}
        assert reference is not None
        labels, nuisance, combined = map(lambda x: np.asarray(x, dtype=np.int64),
                                         reference)
        if not np.array_equal(combined, labels * 4 + nuisance):
            raise ValueError("joint labels differ")
        joint[hi], location[hi] = combined, nuisance
    actual = {path.parent.resolve() for path in output_path(
        spec, "carriers").glob("*/*/seed-*/metrics.json")}
    if actual != expected_dirs:
        raise ValueError("carrier directory grid differs")
    return correct, observed, joint, location, {
        "admissions": admissions, "cell_artifacts": records,
        "complete_cells": 75, "expected_cells": 75,
        "physical_gpu_counts": {"0": 75, "1": 0, "2": 0, "3": 0}}


def _bootstrap(correct, joint, location, draws, seed):
    rng = np.random.default_rng(seed)
    picked = rng.integers(0, 5, size=(draws, 5))
    sw = np.zeros((draws, 5), dtype=np.float32)
    rows = np.arange(draws)[:, None]
    np.add.at(sw, (np.broadcast_to(rows, picked.shape), picked), 1)
    sw /= 5
    point = correct.mean(axis=(2, 4))
    samples = np.empty((draws, 3, 5, 3), dtype=np.float32)
    loc_point = np.empty((3, 5, 3, 4), dtype=np.float64)
    loc_samples = np.empty((draws, 3, 5, 3, 4), dtype=np.float32)
    for hi in range(3):
        ew = _weights(joint[hi], draws, rng)
        episode = np.einsum("asge,be->basg", correct[hi], ew, optimize=True)
        samples[:, hi] = np.einsum("basg,bs->bag", episode, sw, optimize=True)
        for loc in range(4):
            mask = (location[hi] == loc).astype(np.float32)
            part = np.einsum(
                "asge,be,e->basg", correct[hi], ew, mask, optimize=True) * 4
            loc_samples[:, hi, :, :, loc] = np.einsum(
                "basg,bs->bag", part, sw, optimize=True)
            loc_point[hi, :, :, loc] = correct[hi][..., mask.astype(bool)].mean(
                axis=(1, 3))
    return point, samples, loc_point, loc_samples


def _analysis(spec, correct, observed, location, point, samples,
              loc_point, loc_samples):
    none, fixed, ssm = 0, ARMS.index("fixed_trust"), ARMS.index("ssm")
    result = {}
    for hi, host in enumerate(HOSTS):
        host_value = {"arms": {}, "paired_vs_none": {}, "fixed_minus_ssm": {}}
        for ai, arm in enumerate(ARMS):
            host_value["arms"][arm] = {}
            for gi, age in enumerate(AGES):
                seed_loc = np.asarray([[correct[hi, ai, si, gi,
                    location[hi] == loc].mean() for loc in range(4)]
                    for si in range(5)])
                record = _stat(point[hi, ai, gi], samples[:, hi, ai, gi],
                               observed[hi, ai, :, gi])
                record["nuisance_location"] = {
                    "per_location": [_stat(
                        loc_point[hi, ai, gi, loc],
                        loc_samples[:, hi, ai, gi, loc], seed_loc[:, loc])
                        for loc in range(4)],
                    "worst_location": _stat(
                        loc_point[hi, ai, gi].min(),
                        loc_samples[:, hi, ai, gi].min(1), seed_loc.min(1))}
                host_value["arms"][arm][f"age-{age}"] = record
                if ai:
                    host_value["paired_vs_none"].setdefault(arm, {})[
                        f"age-{age}"] = _stat(
                            point[hi, ai, gi] - point[hi, none, gi],
                            samples[:, hi, ai, gi] - samples[:, hi, none, gi],
                            observed[hi, ai, :, gi] - observed[hi, none, :, gi])
            host_value["arms"][arm]["shortest_minus_longest"] = _stat(
                point[hi, ai, 0] - point[hi, ai, 2],
                samples[:, hi, ai, 0] - samples[:, hi, ai, 2],
                observed[hi, ai, :, 0] - observed[hi, ai, :, 2])
        for gi, age in enumerate(AGES):
            host_value["fixed_minus_ssm"][f"age-{age}"] = _stat(
                point[hi, fixed, gi] - point[hi, ssm, gi],
                samples[:, hi, fixed, gi] - samples[:, hi, ssm, gi],
                observed[hi, fixed, :, gi] - observed[hi, ssm, :, gi])
        result[host] = host_value
    r, p, g = HOSTS.index("reacher"), HOSTS.index("pusht"), 2
    primary_point = (point[p, fixed, g] - point[p, ssm, g]
                     - point[r, fixed, g] + point[r, ssm, g])
    primary_samples = (samples[:, p, fixed, g] - samples[:, p, ssm, g]
                       - samples[:, r, fixed, g] + samples[:, r, ssm, g])
    seed_values = (observed[p, fixed, :, g] - observed[p, ssm, :, g]
                   - observed[r, fixed, :, g] + observed[r, ssm, :, g])
    primary = _stat(primary_point, primary_samples, seed_values)
    margin = spec["statistics"]["equivalence_margin"]
    primary.update({"resolved_nonzero": primary["ci95"][0] > 0
                    or primary["ci95"][1] < 0,
                    "equivalent_within_margin": primary["ci90"][0] >= -margin
                    and primary["ci90"][1] <= margin,
                    "equivalence_margin": margin})
    return {"hosts": result, "primary_ranking_interaction": primary}


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing matched-token aggregation without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    root = output_path(spec, "root")
    if (root / "summary.json").exists() or (root / "final_audit.json").exists():
        raise FileExistsError("matched-token aggregation exists")
    correct, observed, joint, location, audit = _load(spec)
    point, samples, loc_point, loc_samples = _bootstrap(
        correct, joint, location, spec["statistics"]["bootstrap_draws"],
        spec["statistics"]["bootstrap_seed"])
    analysis = _analysis(
        spec, correct, observed, location, point, samples, loc_point, loc_samples)
    summary = {"schema_version": 1, "study": spec["study"],
               "branch": "matched-composite-token-fixed-endpoint",
               "lock": spec["_lock"], "status": "complete",
               "target": "token", "nuisance": "location",
               "hosts": list(HOSTS), "ages": list(AGES),
               "arms": list(ARMS), "seeds": list(SEEDS),
               "bootstrap": {"draws": 20000, "seed": 20260921,
                             "joint_seed_resampling": True,
                             "independent_host_16_way_episode_resampling": True},
               "no_pooled_host_memory_score": True, **analysis}
    summary_path = root / "summary.json"
    digest = atomic_text(summary_path, stable_json(summary))
    atomic_text(root / "final_audit.json", stable_json({
        "schema_version": 1, "study": spec["study"], "lock": spec["_lock"],
        "status": "complete", **audit,
        "summary": {"path": str(summary_path.relative_to(ROOT)),
                    "sha256": digest}, "bootstrap_draws": 20000,
        "cuda3_used": False}))


if __name__ == "__main__":
    main()
