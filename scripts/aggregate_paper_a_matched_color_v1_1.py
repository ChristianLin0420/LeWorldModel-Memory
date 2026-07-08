#!/usr/bin/env python3
"""Fail-closed aggregation for adaptive Wave 1.1 matched color recall."""

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
from scripts.paper_a_matched_color_v1_1_spec import (  # noqa: E402
    AGES,
    ARMS,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    SEEDS,
    load_locked_spec,
    output_path,
    sha256_file,
)
from scripts.prepare_paper_a_matched_color_v1_1 import host_manifest_path  # noqa: E402
from scripts.train_paper_a_matched_color_v1_1 import carrier_directory  # noqa: E402


EPISODES = 480


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def _stat(point: float, samples: np.ndarray,
          seed_values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean": float(point),
        "ci95": [float(value) for value in np.quantile(values, (.025, .975))],
        "ci90": [float(value) for value in np.quantile(values, (.05, .95))],
        "seed_values": [float(value) for value in np.asarray(seed_values)],
    }


def _stratified_weights(joint: np.ndarray, draws: int,
                        rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(joint, dtype=np.int64)
    if labels.shape != (EPISODES,) or not np.array_equal(
            np.bincount(labels, minlength=16), np.full(16, 30)):
        raise ValueError("Wave 1.1 labels are not exactly 16-way balanced")
    weights = np.zeros((draws, EPISODES), dtype=np.float32)
    rows = np.arange(draws)[:, None]
    for label in range(16):
        indices = np.flatnonzero(labels == label)
        sampled = indices[
            rng.integers(0, len(indices), size=(draws, len(indices)))]
        np.add.at(weights, (
            np.broadcast_to(rows, sampled.shape), sampled), 1.0)
    return weights / float(EPISODES)


def _load_grid(spec: Mapping[str, Any]) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    # H,A,S,G,E where G indexes evidence age.
    correct = np.empty((len(HOSTS), len(ARMS), len(SEEDS), len(AGES),
                        EPISODES), dtype=np.float32)
    observed = np.empty(correct.shape[:-1], dtype=np.float64)
    joint = np.empty((len(HOSTS), EPISODES), dtype=np.int64)
    location = np.empty_like(joint)
    admissions: dict[str, Any] = {}
    cell_records: dict[str, Any] = {}
    expected_directories: set[Path] = set()
    for host_index, host in enumerate(HOSTS):
        admission_path = host_manifest_path(dict(spec), host)
        admission = json.loads(admission_path.read_text())
        if admission.get("lock") != spec["_lock"] \
                or admission.get("status") != "admitted" \
                or admission.get("all_color_ages_admitted") is not True \
                or admission.get("frozen_host_unchanged") is not True:
            raise ValueError(f"invalid Wave 1.1 admission: {host}")
        provenance = admission.get("provenance", {})
        fresh = (provenance.get("freshness", {}).get(
            "prior_trajectories_reused") is False if host == "reacher"
            else provenance.get("prior_screen_fresh_data_exclusion", {}).get(
                "zero_overlap_proven") is True)
        if not fresh:
            raise ValueError(f"Wave 1.1 freshness proof differs: {host}")
        admissions[host] = _record(admission_path)
        reference: tuple[list[int], list[int], list[int]] | None = None
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(SEEDS):
                directory = carrier_directory(spec, host, arm, seed)
                expected_directories.add(directory.resolve())
                paths = {
                    "metrics": directory / "metrics.json",
                    "checkpoint": directory / "carrier.pt",
                    "history": directory / "history.csv",
                    "manifest": directory / "manifest.json",
                }
                if not all(path.is_file() for path in paths.values()):
                    raise FileNotFoundError(f"incomplete Wave 1.1 cell {directory}")
                metrics = json.loads(paths["metrics"].read_text())
                manifest = json.loads(paths["manifest"].read_text())
                expected = {
                    "schema_version": 1, "study": spec["study"],
                    "branch": "admission-informed-matched-color-v1-1",
                    "lock": spec["_lock"], "host": host,
                    "target": "color", "nuisance": "location",
                    "arm": arm, "seed": seed, "device": "cuda:0",
                    "physical_gpu": 0, "age_balanced_mixture": True,
                    "frozen_host_unchanged": True,
                    "validation_labels_used_for_fitting": False,
                    "prior_admission_metrics_used_for_wave1_1_inference": False,
                    "global_all_host_admission_verified": True,
                }
                failed = [key for key, value in expected.items()
                          if metrics.get(key) != value]
                global_admissions = metrics.get("global_admissions", {})
                expected_global = {
                    candidate: _record(host_manifest_path(
                        dict(spec), candidate))
                    for candidate in HOSTS
                }
                if global_admissions != expected_global:
                    failed.append("global_admissions")
                artifacts = manifest.get("artifacts", {})
                hashes = {
                    name: sha256_file(paths[name])
                    for name in ("metrics", "checkpoint", "history")}
                if failed or metrics.get("ages") != list(AGES) \
                        or set(artifacts) != set(hashes) \
                        or any(artifacts[name].get("sha256") != digest
                               for name, digest in hashes.items()):
                    raise ValueError(f"invalid Wave 1.1 cell {directory}: {failed}")
                labels = metrics.get("labels")
                nuisance = metrics.get("nuisance_location_labels")
                joint_labels = metrics.get("joint_labels")
                current_reference = (labels, nuisance, joint_labels)
                if reference is None:
                    reference = current_reference
                elif current_reference != reference:
                    raise ValueError(f"Wave 1.1 labels differ across {host} cells")
                color_array = np.asarray(labels, dtype=np.int64)
                nuisance_array = np.asarray(nuisance, dtype=np.int64)
                joint_array = np.asarray(joint_labels, dtype=np.int64)
                if any(value.shape != (EPISODES,) for value in (
                        color_array, nuisance_array, joint_array)) \
                        or not np.array_equal(
                            joint_array, color_array * 4 + nuisance_array) \
                        or not np.array_equal(
                            np.bincount(joint_array, minlength=16),
                            np.full(16, 30)):
                    raise ValueError(f"Wave 1.1 nuisance deck differs: {host}")
                for age_index, age in enumerate(AGES):
                    result = metrics["readouts"][f"age-{age}"]
                    current = np.asarray(result["correct"], dtype=np.float64)
                    if current.shape != (EPISODES,) \
                            or result.get("metric") != "balanced_accuracy" \
                            or abs(float(current.mean())
                                   - float(result["balanced_accuracy"])) > 1e-12:
                        raise ValueError(
                            f"invalid Wave 1.1 readout {host}/{arm}/{seed}/{age}")
                    by_location = [float(current[nuisance_array == value].mean())
                                   for value in range(4)]
                    nuisance_record = result.get("nuisance_location", {})
                    if not np.allclose(
                            nuisance_record.get("per_location_accuracy"),
                            by_location, rtol=0, atol=1e-12) \
                            or abs(float(nuisance_record.get(
                                "worst_location_accuracy", -1))
                                - min(by_location)) > 1e-12:
                        raise ValueError("nuisance-location receipt differs")
                    correct[host_index, arm_index, seed_index, age_index] = current
                    observed[host_index, arm_index, seed_index, age_index] = float(
                        result["balanced_accuracy"])
                key = f"{host}/{arm}/seed-{seed}"
                cell_records[key] = {
                    name: _record(path) for name, path in paths.items()}
        assert reference is not None
        joint[host_index] = np.asarray(reference[2], dtype=np.int64)
        location[host_index] = np.asarray(reference[1], dtype=np.int64)
    carrier_root = output_path(spec, "carriers")
    actual = {path.parent.resolve() for path in carrier_root.glob(
        "*/*/seed-*/metrics.json")}
    if actual != expected_directories:
        raise ValueError("Wave 1.1 carrier directory set differs")
    audit = {
        "admissions": admissions, "cell_artifacts": cell_records,
        "complete_cells": len(cell_records), "expected_cells": 50,
        "hashed_cell_artifacts": 4 * len(cell_records),
        "physical_gpu_counts": {"0": 50, "1": 0, "2": 0, "3": 0},
        "unexpected_carrier_directories": 0,
    }
    return correct, observed, joint, location, audit


def _bootstrap(correct: np.ndarray, joint: np.ndarray,
               location: np.ndarray, *, draws: int, seed: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(SEEDS), size=(draws, len(SEEDS)))
    seed_counts = np.zeros((draws, len(SEEDS)), dtype=np.float32)
    rows = np.arange(draws)[:, None]
    np.add.at(seed_counts, (
        np.broadcast_to(rows, selected.shape), selected), 1.0)
    seed_weights = seed_counts / float(len(SEEDS))
    point = correct.mean(axis=(2, 4))
    location_point = np.empty((*point.shape, 4), dtype=np.float64)
    samples = np.empty((draws, *point.shape), dtype=np.float32)
    location_samples = np.empty((draws, *point.shape, 4), dtype=np.float32)
    for host_index in range(len(HOSTS)):
        episode_weights = _stratified_weights(joint[host_index], draws, rng)
        episode_mean = np.einsum(
            "asge,be->basg", correct[host_index], episode_weights,
            optimize=True)
        samples[:, host_index] = np.einsum(
            "basg,bs->bag", episode_mean, seed_weights, optimize=True)
        for nuisance in range(4):
            mask = (location[host_index] == nuisance).astype(np.float32)
            # Exact 16-way stratification gives each nuisance group mass .25.
            nuisance_episode = np.einsum(
                "asge,be,e->basg", correct[host_index], episode_weights,
                mask, optimize=True) * 4.0
            location_samples[:, host_index, :, :, nuisance] = np.einsum(
                "basg,bs->bag", nuisance_episode, seed_weights, optimize=True)
            subset = correct[host_index][
                ..., location[host_index] == nuisance]
            location_point[host_index, :, :, nuisance] = subset.mean(
                axis=(1, 3))
    return point, samples, location_point, location_samples


def _summarize(spec: Mapping[str, Any], correct: np.ndarray,
               observed: np.ndarray, point: np.ndarray,
               samples: np.ndarray, location_point: np.ndarray,
               location_samples: np.ndarray,
               nuisance_labels: np.ndarray) -> dict[str, Any]:
    none, fixed, ssm = (ARMS.index(name)
                        for name in ("none", "fixed_trust", "ssm"))
    age4, age15 = AGES.index(4), AGES.index(15)
    hosts: dict[str, Any] = {}
    for host_index, host in enumerate(HOSTS):
        value: dict[str, Any] = {
            "metric": "balanced_accuracy", "episodes": EPISODES,
            "arms": {}, "paired_vs_none": {}, "fixed_minus_ssm": {},
        }
        for arm_index, arm in enumerate(ARMS):
            value["arms"][arm] = {}
            for age_index, age in enumerate(AGES):
                seed_location = np.asarray([
                    correct[host_index, arm_index, seed_index, age_index,
                            nuisance_labels[host_index] == nuisance].mean()
                    for seed_index in range(len(SEEDS))
                    for nuisance in range(4)
                ]).reshape(len(SEEDS), 4)
                worst_point = float(location_point[
                    host_index, arm_index, age_index].min())
                worst_samples = location_samples[
                    :, host_index, arm_index, age_index].min(axis=1)
                record = _stat(
                    point[host_index, arm_index, age_index],
                    samples[:, host_index, arm_index, age_index],
                    observed[host_index, arm_index, :, age_index])
                record["nuisance_location"] = {
                    "per_location_accuracy": [
                        _stat(
                            location_point[host_index, arm_index, age_index, loc],
                            location_samples[:, host_index, arm_index,
                                             age_index, loc],
                            seed_location[:, loc])
                        for loc in range(4)],
                    "worst_location_accuracy": _stat(
                        worst_point, worst_samples,
                        seed_location.min(axis=1)),
                }
                value["arms"][arm][f"age-{age}"] = record
                if arm != "none":
                    value["paired_vs_none"].setdefault(arm, {})[
                        f"age-{age}"] = _stat(
                            point[host_index, arm_index, age_index]
                            - point[host_index, none, age_index],
                            samples[:, host_index, arm_index, age_index]
                            - samples[:, host_index, none, age_index],
                            observed[host_index, arm_index, :, age_index]
                            - observed[host_index, none, :, age_index])
                if arm in (fixed, ssm):
                    pass
            value["arms"][arm]["shortest_minus_longest"] = _stat(
                point[host_index, arm_index, age4]
                - point[host_index, arm_index, age15],
                samples[:, host_index, arm_index, age4]
                - samples[:, host_index, arm_index, age15],
                observed[host_index, arm_index, :, age4]
                - observed[host_index, arm_index, :, age15])
        for age_index, age in enumerate(AGES):
            value["fixed_minus_ssm"][f"age-{age}"] = _stat(
                point[host_index, fixed, age_index]
                - point[host_index, ssm, age_index],
                samples[:, host_index, fixed, age_index]
                - samples[:, host_index, ssm, age_index],
                observed[host_index, fixed, :, age_index]
                - observed[host_index, ssm, :, age_index])
        hosts[host] = value

    reacher, pusht = HOSTS.index("reacher"), HOSTS.index("pusht")
    interaction_point = (
        point[pusht, fixed, age15] - point[pusht, ssm, age15]
        - point[reacher, fixed, age15] + point[reacher, ssm, age15])
    interaction_samples = (
        samples[:, pusht, fixed, age15] - samples[:, pusht, ssm, age15]
        - samples[:, reacher, fixed, age15]
        + samples[:, reacher, ssm, age15])
    seed_values = (
        observed[pusht, fixed, :, age15] - observed[pusht, ssm, :, age15]
        - observed[reacher, fixed, :, age15]
        + observed[reacher, ssm, :, age15])
    primary = _stat(interaction_point, interaction_samples, seed_values)
    margin = float(spec["statistics"]["equivalence_margin"])
    primary.update({
        "resolved_nonzero": bool(
            primary["ci95"][0] > 0 or primary["ci95"][1] < 0),
        "equivalent_within_margin": bool(
            primary["ci90"][0] >= -margin
            and primary["ci90"][1] <= margin),
        "equivalence_margin": margin,
    })
    return {"hosts": hosts, "primary_ranking_interaction": primary}


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing Wave 1.1 aggregation writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    destination = output_path(spec, "root") / "summary.json"
    audit_path = output_path(spec, "root") / "final_audit.json"
    if destination.exists() or audit_path.exists():
        raise FileExistsError("Wave 1.1 aggregation is already closed")
    correct, observed, joint, nuisance, audit = _load_grid(spec)
    draws = int(spec["statistics"]["bootstrap_draws"])
    point, samples, location_point, location_samples = _bootstrap(
        correct, joint, nuisance, draws=draws,
        seed=int(spec["statistics"]["bootstrap_seed"]))
    analysis = _summarize(
        spec, correct, observed, point, samples,
        location_point, location_samples, nuisance)
    summary = {
        "schema_version": 1, "study": spec["study"],
        "branch": "admission-informed-matched-color-v1-1",
        "lock": spec["_lock"], "status": "complete",
        "admission_informed_after_two_preserved_global_stops": True,
        "deterministic_selection_rule": spec["adaptive_origin"][
            "deterministic_selection_rule"],
        "prior_admission_metrics_used_for_wave1_1_inference": False,
        "hosts": list(HOSTS), "target": "color",
        "nuisance": "location", "ages": list(AGES),
        "arms": list(ARMS), "seeds": list(SEEDS),
        "bootstrap": {
            "draws": draws, "seed": spec["statistics"]["bootstrap_seed"],
            "joint_seed_resampling_across_hosts": True,
            "independent_host_episode_resampling": True,
            "episode_stratification": "16-way color-location joint label",
            "paired_within_host": ["age", "arm"],
        },
        "no_pooled_host_memory_score": True,
        **analysis,
        "claim_boundary": (
            "Admission-informed color-only Reacher/PushT comparison; location "
            "is an exact-balanced nuisance. It reduces cue-semantic and age "
            "confounding but does not isolate checkpoint identity from host "
            "environment, background, dynamics, or training distribution, and "
            "it is not a TwoRoom carrier result."),
    }
    summary_hash = atomic_text(destination, stable_json(summary))
    final_audit = {
        "schema_version": 1, "study": spec["study"],
        "branch": summary["branch"], "lock": spec["_lock"],
        "status": "complete", **audit,
        "summary": {"path": str(destination.relative_to(ROOT)),
                    "sha256": summary_hash},
        "bootstrap_draws": draws, "cuda3_used": False,
    }
    atomic_text(audit_path, stable_json(final_audit))
    print(f"[matched-color-v1.1] wrote {destination} and {audit_path}", flush=True)


if __name__ == "__main__":
    main()
