#!/usr/bin/env python3
"""Aggregate or formally close the strict fixed-endpoint age branch."""

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
from scripts.aggregate_paper_a_evidence_age_readtime import _bootstrap, _stat
from scripts.paper_a_evidence_age_spec import (
    ARMS,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    SEEDS,
    host_tasks,
    load_locked_spec,
    output_root,
    sha256_file,
)
from scripts.train_paper_a_evidence_age_strict import carrier_directory


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _resolution(spec: dict[str, Any], host: str, task: str) -> dict[str, Any]:
    root = output_root(spec, "strict") / "cache" / host / task
    manifest = root / "manifest.json"
    stopped = root / "prerequisite-stopped.json"
    if stopped.is_file():
        return {"status": "stopped-prerequisite-failure",
                "receipt": json.loads(stopped.read_text())}
    if not manifest.is_file():
        return {"status": "missing-prerequisite-resolution"}
    value = json.loads(manifest.read_text())
    if value.get("lock") != spec["_lock"]:
        raise ValueError(f"strict cache uses a different lock: {manifest}")
    if value.get("status") != "admitted":
        return {"status": value.get("status", "stopped"), "receipt": value}
    return {"status": "admitted", "manifest": value,
            "manifest_sha256": sha256_file(manifest)}


def _task_summary(spec: dict[str, Any], host: str, task: str,
                  offset: int) -> dict[str, Any]:
    ages = [int(value) for value in spec["strict_fixed_endpoint"][host]["ages"]]
    records: dict[tuple[str, int], dict] = {}
    labels = None
    for arm in ARMS:
        for seed in SEEDS:
            directory = carrier_directory(spec, host, task, arm, seed)
            metrics_path = directory / "metrics.json"
            manifest_path = directory / "manifest.json"
            if not metrics_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"incomplete strict carrier {directory}")
            value = json.loads(metrics_path.read_text())
            manifest = json.loads(manifest_path.read_text())
            expected = {
                "schema_version": 1, "study": spec["study"],
                "branch": "strict-fixed-endpoint-cue-offset",
                "lock": spec["_lock"], "host": host, "task": task,
                "arm": arm, "seed": seed, "device": "cuda:0",
                "age_balanced_mixture": True,
                "frozen_host_unchanged": True,
                "validation_labels_used_for_fitting": False,
            }
            failed = [key for key, expected_value in expected.items()
                      if value.get(key) != expected_value]
            if failed or value.get("ages") != ages \
                    or manifest.get("artifacts", {}).get("metrics", {}).get(
                        "sha256") != sha256_file(metrics_path):
                raise ValueError(f"invalid strict carrier {directory}: {failed}")
            current_labels = np.asarray(value["labels"], dtype=np.int64)
            if labels is None:
                labels = current_labels
            elif not np.array_equal(labels, current_labels):
                raise ValueError("strict labels differ across carrier cells")
            records[(arm, seed)] = value

    correct = np.empty(
        (len(ARMS), len(ages), len(SEEDS), len(labels)), dtype=np.float64)
    observed = np.empty((len(ARMS), len(ages), len(SEEDS)), dtype=np.float64)
    for arm_index, arm in enumerate(ARMS):
        for age_index, age in enumerate(ages):
            for seed_index, seed in enumerate(SEEDS):
                result = records[(arm, seed)]["readouts"][f"age-{age}"]
                correct[arm_index, age_index, seed_index] = result["correct"]
                observed[arm_index, age_index, seed_index] = result["value"]
    draws = int(spec["statistics"]["bootstrap_draws"])
    bootstrap_seed = int(spec["statistics"]["bootstrap_seed"]) + 100 + offset
    point, samples = _bootstrap(
        correct.reshape(-1, len(SEEDS), len(labels)), labels,
        draws=draws, seed=bootstrap_seed, stratified=host == "pusht")
    point = point.reshape(len(ARMS), len(ages))
    samples = samples.reshape(draws, len(ARMS), len(ages))
    none = ARMS.index("none")
    arms, contrasts, decay = {}, {}, {}
    for arm_index, arm in enumerate(ARMS):
        arms[arm] = {}
        for age_index, age in enumerate(ages):
            arms[arm][f"age-{age}"] = _stat(
                point[arm_index, age_index], samples[:, arm_index, age_index],
                observed[arm_index, age_index], draws, bootstrap_seed)
        if arm != "none":
            contrasts[arm] = {}
            for age_index, age in enumerate(ages):
                contrasts[arm][f"age-{age}"] = _stat(
                    point[arm_index, age_index] - point[none, age_index],
                    samples[:, arm_index, age_index]
                    - samples[:, none, age_index],
                    observed[arm_index, age_index] - observed[none, age_index],
                    draws, bootstrap_seed)
        decay[arm] = _stat(
            point[arm_index, 0] - point[arm_index, -1],
            samples[:, arm_index, 0] - samples[:, arm_index, -1],
            observed[arm_index, 0] - observed[arm_index, -1],
            draws, bootstrap_seed)
    return {
        "status": "complete", "host": host, "task": task, "ages": ages,
        "metric": "balanced_accuracy" if host == "pusht" else "accuracy",
        "validation_episodes": int(len(labels)), "arms": arms,
        "paired_vs_no_carrier": contrasts,
        "shortest_age_minus_longest_age": decay,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing strict aggregation writes without --execute")
    spec = load_locked_spec(args.spec, args.sha)
    resolutions, tasks = {}, {}
    complete = True
    offset = 0
    for host in HOSTS:
        resolutions[host], tasks[host] = {}, {}
        for task in host_tasks(spec, host):
            resolution = _resolution(spec, host, task)
            resolutions[host][task] = resolution
            if resolution["status"] == "admitted":
                tasks[host][task] = _task_summary(spec, host, task, offset)
            else:
                complete = False
                tasks[host][task] = {"status": resolution["status"]}
            offset += 1
    summary = {
        "schema_version": 1, "study": spec["study"],
        "branch": "strict-fixed-endpoint-cue-offset",
        "lock": spec["_lock"], "complete": complete,
        "resolutions": resolutions, "tasks": tasks,
        "training_contract": (
            "One carrier per arm/seed is trained on the equal-age replicated "
            "training mixture and evaluated on paired base episodes at every age."),
    }
    destination = output_root(spec, "strict") / "summary.json"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    atomic_text(destination, stable_json(summary))
    lines = ["# Strict fixed-endpoint cue-offset study", "",
             f"Complete: {complete}", ""]
    for host, host_tasks_value in tasks.items():
        lines.extend([f"## {host}", ""])
        for task, value in host_tasks_value.items():
            lines.append(f"### {task}: {value['status']}")
            for arm, ages in value.get("paired_vs_no_carrier", {}).items():
                text = ", ".join(
                    f"{age}: {stat['mean']:+.3f} "
                    f"[{stat['ci95'][0]:+.3f},{stat['ci95'][1]:+.3f}]"
                    for age, stat in ages.items())
                lines.append(f"- {arm} minus none: {text}")
            lines.append("")
    atomic_text(destination.with_suffix(".md"), "\n".join(lines) + "\n")
    print(f"[evidence-age/strict] wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
