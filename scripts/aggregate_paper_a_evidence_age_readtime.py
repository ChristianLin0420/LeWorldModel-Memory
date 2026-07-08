#!/usr/bin/env python3
"""Aggregate the existing-checkpoint evidence-age sweep with paired CIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import atomic_text, stable_json
from scripts.paper_a_evidence_age import age_name
from scripts.paper_a_evidence_age_spec import (
    ARMS,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    SEEDS,
    host_tasks,
    load_locked_spec,
    output_root,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _episode_weights(rng: np.random.Generator, draws: int, labels: np.ndarray,
                     stratified: bool) -> np.ndarray:
    episodes = len(labels)
    weights = np.zeros((draws, episodes), dtype=np.float32)
    if not stratified:
        sampled = rng.integers(0, episodes, size=(draws, episodes))
        rows = np.repeat(np.arange(draws), episodes)
        np.add.at(weights, (rows, sampled.reshape(-1)), 1.0 / episodes)
        return weights
    classes = np.unique(labels)
    for category in classes:
        members = np.flatnonzero(labels == category)
        sampled_local = rng.integers(
            0, len(members), size=(draws, len(members)))
        sampled = members[sampled_local]
        rows = np.repeat(np.arange(draws), len(members))
        np.add.at(
            weights, (rows, sampled.reshape(-1)),
            1.0 / (len(classes) * len(members)))
    return weights


def _bootstrap(values: np.ndarray, labels: np.ndarray, *, draws: int,
               seed: int, stratified: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return point values and paired seed/episode bootstrap samples.

    ``values`` is ``(P,S,E)``.  Every bootstrap draw uses one shared seed and
    episode resample across all P endpoints, preserving all contrasts.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(SEEDS), len(labels)) \
            or not np.isfinite(values).all():
        raise ValueError(f"invalid bootstrap tensor {values.shape}")
    rng = np.random.default_rng(seed)
    point_episode = _episode_weights(
        rng, 1, labels, stratified=stratified)[0].astype(np.float64)
    point_episode[:] = 0.0
    if stratified:
        classes = np.unique(labels)
        for category in classes:
            members = np.flatnonzero(labels == category)
            point_episode[members] = 1.0 / (len(classes) * len(members))
    else:
        point_episode[:] = 1.0 / len(labels)
    point = np.einsum(
        "pse,s,e->p", values,
        np.full(len(SEEDS), 1.0 / len(SEEDS)), point_episode)
    samples = np.empty((draws, values.shape[0]), dtype=np.float64)
    batch = 400
    offset = 0
    while offset < draws:
        count = min(batch, draws - offset)
        seed_samples = rng.integers(
            0, len(SEEDS), size=(count, len(SEEDS)))
        seed_weights = np.zeros((count, len(SEEDS)), dtype=np.float32)
        rows = np.repeat(np.arange(count), len(SEEDS))
        np.add.at(seed_weights, (rows, seed_samples.reshape(-1)),
                  1.0 / len(SEEDS))
        episode_weights = _episode_weights(
            rng, count, labels, stratified=stratified)
        samples[offset:offset + count] = np.einsum(
            "bs,pse,be->bp", seed_weights, values, episode_weights,
            optimize=True)
        offset += count
    return point, samples


def _stat(point: float, samples: np.ndarray, seed_values: np.ndarray,
          draws: int, seed: int) -> dict[str, Any]:
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "mean": float(point),
        "ci95": [float(low), float(high)],
        "seed_values": [float(value) for value in seed_values],
        "bootstrap": {
            "draws": draws, "seed": seed,
            "unit": "paired checkpoint seed and validation episode",
            "interval": "percentile", "confidence": 0.95,
        },
    }


def _load_task(spec: dict[str, Any], host: str, task: str
               ) -> tuple[np.ndarray, dict[str, Any]]:
    records = []
    root = output_root(spec, "read_time")
    for seed in SEEDS:
        path = root / host / task / f"seed-{seed}" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing read-time cell {path}")
        value = json.loads(path.read_text())
        expected = {
            "schema_version": 1, "study": spec["study"],
            "branch": "existing-checkpoint-read-time", "lock": spec["_lock"],
            "host": host, "task": task, "seed": seed,
            "device": "cuda:0", "host_checkpoint_training_performed": False,
            "carrier_training_performed": False,
            "validation_labels_used_for_fitting": False,
            "current_observation_excluded": True,
        }
        failed = [key for key, expected_value in expected.items()
                  if value.get(key) != expected_value]
        if failed or set(value.get("arms", {})) != set(ARMS) \
                or not isinstance(value.get("cuda_device_name"), str):
            raise ValueError(f"invalid read-time cell {path}: {failed}")
        records.append(value)
    labels = np.asarray(records[0]["labels"], dtype=np.int64)
    if any(not np.array_equal(labels, np.asarray(item["labels"]))
           for item in records[1:]):
        raise ValueError(f"validation labels differ across {host}/{task}")
    return labels, {str(item["seed"]): item for item in records}


def _task_summary(spec: dict[str, Any], host: str, task: str,
                  seed_offset: int) -> dict[str, Any]:
    labels, records = _load_task(spec, host, task)
    ages = list(spec["read_time"][f"{host}_ages"])
    shape = (len(ARMS), len(ages), len(SEEDS), len(labels))
    correct = np.empty(shape, dtype=np.float64)
    observed_values = np.empty(shape[:-1], dtype=np.float64)
    for arm_index, arm in enumerate(ARMS):
        for age_index, age in enumerate(ages):
            key = age_name(age)
            for seed_index, seed in enumerate(SEEDS):
                result = records[str(seed)]["arms"][arm]["ages"][key]
                correct[arm_index, age_index, seed_index] = np.asarray(
                    result["correct"], dtype=np.float64)
                observed_values[arm_index, age_index, seed_index] = float(
                    result["value"])
    draws = int(spec["statistics"]["bootstrap_draws"])
    bootstrap_seed = int(spec["statistics"]["bootstrap_seed"]) + seed_offset
    flattened = correct.reshape(-1, len(SEEDS), len(labels))
    point, samples = _bootstrap(
        flattened, labels, draws=draws, seed=bootstrap_seed,
        stratified=host == "pusht")
    point = point.reshape(len(ARMS), len(ages))
    samples = samples.reshape(draws, len(ARMS), len(ages))

    arm_summary: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    none_index = ARMS.index("none")
    for arm_index, arm in enumerate(ARMS):
        arm_summary[arm] = {}
        for age_index, age in enumerate(ages):
            arm_summary[arm][age_name(age)] = _stat(
                point[arm_index, age_index],
                samples[:, arm_index, age_index],
                observed_values[arm_index, age_index], draws, bootstrap_seed)
        if arm != "none":
            contrasts[arm] = {}
            for age_index, age in enumerate(ages):
                delta = samples[:, arm_index, age_index] \
                    - samples[:, none_index, age_index]
                contrasts[arm][age_name(age)] = _stat(
                    point[arm_index, age_index] - point[none_index, age_index],
                    delta,
                    observed_values[arm_index, age_index]
                    - observed_values[none_index, age_index],
                    draws, bootstrap_seed)

    age_change = None
    if host == "reacher":
        age15 = ages.index(15)
        final = ages.index("final")
        age_change = {
            arm: _stat(
                point[index, age15] - point[index, final],
                samples[:, index, age15] - samples[:, index, final],
                observed_values[index, age15] - observed_values[index, final],
                draws, bootstrap_seed)
            for index, arm in enumerate(ARMS)
        }
    return {
        "host": host, "task": task,
        "metric": "balanced_accuracy" if host == "pusht" else "accuracy",
        "validation_episodes": int(len(labels)),
        "arms": arm_summary,
        "paired_vs_no_carrier": contrasts,
        "age15_minus_final": age_change,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing aggregation writes without --execute")
    spec = load_locked_spec(args.spec, args.sha)
    tasks = {}
    offset = 0
    for host in HOSTS:
        tasks[host] = {}
        for task in host_tasks(spec, host):
            tasks[host][task] = _task_summary(spec, host, task, offset)
            offset += 1
    summary = {
        "schema_version": 1, "study": spec["study"],
        "branch": "existing-checkpoint-read-time",
        "lock": spec["_lock"], "complete": True,
        "tasks": tasks,
        "interpretation_boundary": (
            "Read time varies within the frozen checkpoint and paired bank; "
            "local physical state therefore changes with evidence age. This is "
            "not the strict fixed-endpoint cue-offset intervention."),
    }
    destination = output_root(spec, "read_time") / "summary.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    atomic_text(destination, stable_json(summary))
    lines = ["# Existing-checkpoint evidence-age sweep", ""]
    for host, host_tasks_value in tasks.items():
        lines.extend([f"## {host}", ""])
        for task, task_value in host_tasks_value.items():
            lines.append(f"### {task}")
            for arm, ages in task_value["paired_vs_no_carrier"].items():
                rendered = ", ".join(
                    f"{age}: {stat['mean']:+.3f} "
                    f"[{stat['ci95'][0]:+.3f},{stat['ci95'][1]:+.3f}]"
                    for age, stat in ages.items())
                lines.append(f"- {arm} minus none: {rendered}")
            lines.append("")
    atomic_text(destination.with_suffix(".md"), "\n".join(lines) + "\n")
    print(f"[evidence-age/read-time] wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
