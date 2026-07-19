#!/usr/bin/env python3
"""Native-use evaluation for autonomous Masked-Evidence JEPA memory states.

This script converts a trained ME-JEPA memory state into a selected target via
a post-hoc linear readout, then evaluates fixed-controller environment success.
It supports only environments with an audited controller available locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_FEATURE_ROOT = ROOT / "outputs" / "masked_evidence_jepa_ogbench_v1"
DEFAULT_OUTPUT = ROOT / "outputs" / "masked_evidence_jepa_native_use_v1"
AGES = (4, 8, 15)
SEEDS = (0, 1, 2)
CLASSES = 4


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def env_key(env_name: str) -> str:
    return env_name.replace("/", "_")


def feature_path(feature_root: Path, env_name: str, age: int, seed: int) -> Path:
    return feature_root / env_key(env_name) / f"age_{age}" / f"s{seed}" / "features.npz"


def load_cell(feature_root: Path, env_name: str, age: int, seed: int) -> dict[str, np.ndarray]:
    path = feature_path(feature_root, env_name, age, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def metric(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "count": int(arr.size),
        "values": arr.astype(float).tolist(),
    }


def predict_arms(cell: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    readout = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    readout.fit(cell["train_memory"], cell["train_labels"].astype(np.int64))
    return {
        "full": readout.predict(cell["val_full_memory"]).astype(np.int64),
        "reset": readout.predict(cell["val_reset_memory"]).astype(np.int64),
        "recent": readout.predict(cell["val_no_state_memory"]).astype(np.int64),
    }


def build_success_cube(env_name: str, labels: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    import os
    from scripts.run_ogbench_native_use_stage import (  # noqa: WPS433
        cube_selected_success,
        pointmaze_selected_success,
    )

    os.environ.setdefault("MUJOCO_GL", "egl")
    labels = np.asarray(labels, dtype=np.int64)
    rows = int(len(labels))
    success = np.zeros((rows, CLASSES), dtype=np.uint8)
    reward = np.zeros((rows, CLASSES), dtype=np.float32)
    if env_name.startswith("pointmaze"):
        runner = pointmaze_selected_success
        horizon = int(args.pointmaze_horizon)
    elif env_name == "cube-single-play-v0":
        runner = cube_selected_success
        horizon = int(args.cube_horizon)
    else:
        raise NotImplementedError(f"no native controller audited for {env_name}")
    for row_index, true_label in enumerate(labels):
        for selected_label in range(CLASSES):
            result = runner(env_name, int(true_label), int(selected_label), row_index, horizon)
            success[row_index, selected_label] = int(result["success"])
            reward[row_index, selected_label] = float(result["reward_sum"])
        if (row_index + 1) % 20 == 0:
            print(f"[me-jepa-use] {env_name} cube {row_index + 1}/{rows}", flush=True)
    rng = np.random.default_rng(int(args.random_seed))
    random_choice = rng.integers(0, CLASSES, size=rows, dtype=np.int64)
    return {
        "labels": labels,
        "success": success,
        "reward": reward,
        "random_choice": random_choice,
    }


def evaluate_cell(feature_root: Path, env_name: str, age: int, seed: int,
                  args: argparse.Namespace) -> dict[str, Any]:
    cell = load_cell(feature_root, env_name, age, seed)
    labels = cell["val_labels"].astype(np.int64)
    pred = predict_arms(cell)
    cube = build_success_cube(env_name, labels, args)
    rows = np.arange(len(labels), dtype=np.int64)

    def arm(name: str, choice: np.ndarray) -> dict[str, float]:
        choice = np.asarray(choice, dtype=np.int64)
        executed = cube["success"][rows, choice].astype(np.float64)
        rewards = cube["reward"][rows, choice].astype(np.float64)
        return {
            "goal_accuracy": float((choice == labels).mean()),
            "executed_success": float(executed.mean()),
            "reward_sum_mean": float(rewards.mean()),
        }

    return {
        "env_name": env_name,
        "age": int(age),
        "seed": int(seed),
        "rows": int(len(labels)),
        "arms": {
            "full": arm("full", pred["full"]),
            "reset": arm("reset", pred["reset"]),
            "recent": arm("recent", pred["recent"]),
            "random": arm("random", cube["random_choice"]),
            "oracle": arm("oracle", labels),
        },
        "claim_boundary": "Fixed-controller native use from ME-JEPA memory readout; not native world-model planning.",
    }


def summarize(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault((cell["env_name"], int(cell["age"])), []).append(cell)
    rows = []
    for (env_name, age), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "env_name": env_name,
            "age": int(age),
            "seed_count": int(len(values)),
            "seeds": [int(v["seed"]) for v in values],
        }
        for arm in ("full", "reset", "recent", "random", "oracle"):
            row[arm] = {
                "goal_accuracy": metric([v["arms"][arm]["goal_accuracy"] for v in values]),
                "executed_success": metric([v["arms"][arm]["executed_success"] for v in values]),
                "reward_sum_mean": metric([v["arms"][arm]["reward_sum_mean"] for v in values]),
            }
        row["full_vs_recent_success"] = float(
            row["full"]["executed_success"]["mean"]
            - row["recent"]["executed_success"]["mean"])
        row["full_vs_random_success"] = float(
            row["full"]["executed_success"]["mean"]
            - row["random"]["executed_success"]["mean"])
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--envs", nargs="*", default=["pointmaze-large-navigate-v0", "cube-single-play-v0"])
    parser.add_argument("--ages", type=int, nargs="*", default=list(AGES))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--pointmaze-horizon", type=int, default=600)
    parser.add_argument("--cube-horizon", type=int, default=220)
    parser.add_argument("--random-seed", type=int, default=1187)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.feature_root = args.feature_root if args.feature_root.is_absolute() else ROOT / args.feature_root
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.output.mkdir(parents=True, exist_ok=True)
    cells = []
    for env_name in args.envs:
        for age in args.ages:
            for seed in args.seeds:
                cells.append(evaluate_cell(args.feature_root, env_name, int(age), int(seed), args))
    summary = {
        "schema": "masked_evidence_jepa_native_use_summary_v1",
        "status": "completed",
        "feature_root": str(args.feature_root.relative_to(ROOT)),
        "cell_count": int(len(cells)),
        "rows": summarize(cells),
        "claim_boundary": "Fixed-controller native use from ME-JEPA memory readout; labels are post-hoc readout only.",
    }
    (args.output / "cells.json").write_text(stable_json(cells))
    (args.output / "summary.json").write_text(stable_json(summary))
    print(stable_json(summary), flush=True)


if __name__ == "__main__":
    main()
