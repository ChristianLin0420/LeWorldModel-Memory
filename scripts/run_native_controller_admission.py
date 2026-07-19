#!/usr/bin/env python3
"""Controller-only admission audit for OGBench native-use rows.

This script does not use ME-JEPA readouts.  It loads the validation labels from
an existing feature root, executes the fixed controller with selected=true
label, and reports whether the controller itself is valid enough to admit a
native-use score.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_masked_evidence_jepa_native_use_all import (  # noqa: E402
    AGES,
    SEEDS,
    is_supported,
    load_cell,
    selected_success,
)


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def metric(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
    }


def audit_cell(args: argparse.Namespace, env_name: str, age: int, seed: int) -> dict[str, Any]:
    cell = load_cell(args.feature_root, env_name, int(age), int(seed))
    labels = cell["val_labels"].astype(np.int64)
    rows = min(int(args.max_rows), int(len(labels)))
    labels = labels[:rows]
    successes: list[float] = []
    rewards: list[float] = []
    steps: list[float] = []
    by_label: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"success": [], "reward_sum": [], "steps": []}
    )
    failures: list[dict[str, Any]] = []

    for row_index, true_label in enumerate(labels):
        result = selected_success(env_name, int(true_label), int(true_label), row_index, args)
        success = float(bool(result["success"]))
        reward = float(result.get("reward_sum", 0.0))
        step_count = float(result.get("steps", 0))
        successes.append(success)
        rewards.append(reward)
        steps.append(step_count)
        label_bucket = by_label[int(true_label)]
        label_bucket["success"].append(success)
        label_bucket["reward_sum"].append(reward)
        label_bucket["steps"].append(step_count)
        if not success and len(failures) < int(args.failure_examples):
            failures.append({
                "row_index": int(row_index),
                "label": int(true_label),
                "reward_sum": reward,
                "steps": int(step_count),
            })
        if (row_index + 1) % max(1, int(args.progress_every)) == 0:
            print(
                f"[controller-admission] {env_name} age={age} seed={seed} "
                f"{row_index + 1}/{rows}",
                flush=True,
            )

    success_rate = float(sum(successes) / len(successes)) if successes else float("nan")
    return {
        "env_name": env_name,
        "age": int(age),
        "seed": int(seed),
        "rows": int(rows),
        "success_rate": success_rate,
        "admitted": bool(success_rate >= float(args.oracle_success_min)),
        "oracle_success_min": float(args.oracle_success_min),
        "reward_sum": metric(rewards),
        "steps": metric(steps),
        "label_distribution": {
            str(label): int(np.sum(labels == label))
            for label in sorted(set(int(value) for value in labels))
        },
        "by_label": {
            str(label): {
                "success_rate": metric(values["success"]),
                "reward_sum": metric(values["reward_sum"]),
                "steps": metric(values["steps"]),
            }
            for label, values in sorted(by_label.items())
        },
        "failure_examples": failures,
    }


def summarize(cells: list[dict[str, Any]], unavailable: list[dict[str, Any]],
              oracle_success_min: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[(str(cell["env_name"]), int(cell["age"]))].append(cell)

    for (env_name, age), values in sorted(grouped.items()):
        success_values = [float(value["success_rate"]) for value in values]
        admitted_count = sum(1 for value in values if bool(value["admitted"]))
        rows.append({
            "env_name": env_name,
            "age": int(age),
            "status": "admitted" if admitted_count == len(values) else "failed",
            "seed_count": int(len(values)),
            "admitted_count": int(admitted_count),
            "success_rate": metric(success_values),
            "failed_seeds": [
                int(value["seed"])
                for value in values
                if not bool(value["admitted"])
            ],
        })

    rows.extend(unavailable)
    admitted_rows = [row for row in rows if row.get("status") == "admitted"]
    failed_rows = [row for row in rows if row.get("status") == "failed"]
    return {
        "schema": "native_controller_admission_summary_v1",
        "status": "completed",
        "oracle_success_min": float(oracle_success_min),
        "cell_count": int(len(cells)),
        "unavailable_count": int(len(unavailable)),
        "env_age_count": int(len(rows)),
        "admitted_env_age_count": int(len(admitted_rows)),
        "failed_env_age_count": int(len(failed_rows)),
        "rows": rows,
        "claim_boundary": (
            "Controller-only audit.  It admits native-use scoring only when the "
            "fixed controller solves the true-label condition; it does not score "
            "ME-JEPA memory."
        ),
    }


def write_progress(args: argparse.Namespace, cells: list[dict[str, Any]],
                   unavailable: list[dict[str, Any]]) -> None:
    progress = {
        "schema": "native_controller_admission_progress_v1",
        "feature_root": str(args.feature_root.relative_to(ROOT)),
        "cell_count": int(len(cells)),
        "unavailable_count": int(len(unavailable)),
        "cells": cells,
        "unavailable": unavailable,
    }
    (args.output / "progress.json").write_text(stable_json(progress))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", nargs="*", required=True)
    parser.add_argument("--ages", type=int, nargs="*", default=list(AGES))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--max-rows", type=int, default=120)
    parser.add_argument("--pointmaze-horizon", type=int, default=1200)
    parser.add_argument("--cube-horizon", type=int, default=1800)
    parser.add_argument("--scene-horizon", type=int, default=2400)
    parser.add_argument("--puzzle-horizon", type=int, default=900)
    parser.add_argument("--oracle-success-min", type=float, default=0.85)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--failure-examples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.feature_root = args.feature_root if args.feature_root.is_absolute() else ROOT / args.feature_root
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")

    cells: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for env_name in args.envs:
        if not is_supported(env_name):
            for age in args.ages:
                unavailable.append({
                    "env_name": env_name,
                    "age": int(age),
                    "status": "controller_unavailable",
                    "reason": "No audited fixed controller is available in this repository.",
                })
            write_progress(args, cells, unavailable)
            continue
        for age in args.ages:
            for seed in args.seeds:
                cells.append(audit_cell(args, env_name, int(age), int(seed)))
                write_progress(args, cells, unavailable)

    summary = summarize(cells, unavailable, float(args.oracle_success_min))
    summary["feature_root"] = str(args.feature_root.relative_to(ROOT))
    (args.output / "cells.json").write_text(stable_json(cells))
    (args.output / "summary.json").write_text(stable_json(summary))
    print(stable_json(summary), flush=True)


if __name__ == "__main__":
    main()
