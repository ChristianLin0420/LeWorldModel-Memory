#!/usr/bin/env python3
"""Plot compact OGBench native-use baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINTMAZE = ROOT / "outputs" / "ogbench_native_use_stage_v2_all_envs" / "summary.json"
DEFAULT_CUBE = ROOT / "outputs" / "ogbench_native_use_stage_v1" / "summary.json"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "ogbench_native_success_comparison.svg"
ARMS = ("full", "reset", "no_state", "random", "oracle")
LABELS = {
    "pointmaze-large-navigate-v0": "PM-large",
    "pointmaze-medium-navigate-v0": "PM-medium",
    "pointmaze-giant-navigate-v0": "PM-giant",
    "pointmaze-teleport-navigate-v0": "PM-teleport",
    "cube-single-play-v0": "Cube-single",
}
COLORS = {
    "full": "#111827",
    "reset": "#9ca3af",
    "no_state": "#d4d3cb",
    "random": "#656760",
    "oracle": "#fbd45b",
}


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return json.load(handle)["rows"]


def collect(pointmaze_summary: Path, cube_summary: Path) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in [*load_rows(pointmaze_summary), *load_rows(cube_summary)]:
        env_name = row["env_name"]
        if env_name not in LABELS:
            continue
        grouped.setdefault(env_name, {arm: [] for arm in ARMS})
        for arm in ARMS:
            grouped[env_name][arm].append(
                float(row[arm]["executed_success"]["mean"]))
    return {
        env_name: {arm: float(np.mean(values)) for arm, values in arms.items()}
        for env_name, arms in grouped.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointmaze-summary", type=Path, default=DEFAULT_POINTMAZE)
    parser.add_argument("--cube-summary", type=Path, default=DEFAULT_CUBE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect(args.pointmaze_summary, args.cube_summary)
    envs = [
        "pointmaze-large-navigate-v0",
        "pointmaze-medium-navigate-v0",
        "pointmaze-giant-navigate-v0",
        "pointmaze-teleport-navigate-v0",
        "cube-single-play-v0",
    ]
    x = np.arange(len(envs), dtype=np.float64)
    width = 0.15
    fig, ax = plt.subplots(figsize=(8.8, 2.6), dpi=180)
    fig.patch.set_facecolor("#fbfbf9")
    ax.set_facecolor("#fbfbf9")
    for index, arm in enumerate(ARMS):
        offset = (index - (len(ARMS) - 1) / 2.0) * width
        values = [rows[env][arm] for env in envs]
        edge = "#111827" if arm == "oracle" else "none"
        ax.bar(x + offset, values, width=width * 0.92, label=arm.replace("_", "-"),
               color=COLORS[arm], edgecolor=edge, linewidth=0.7)
    ax.set_ylim(0.0, 1.06)
    ax.set_ylabel("native success", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[env] for env in envs], fontsize=8)
    ax.tick_params(axis="y", labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color="#d4d3cb", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333b49")
        spine.set_linewidth(0.7)
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.20),
              frameon=False, fontsize=8, handlelength=1.4, columnspacing=1.0)
    ax.set_title("Fixed-controller native use: memory selection vs baselines",
                 loc="left", fontsize=10, fontweight="bold", pad=18)
    fig.text(
        0.01, -0.02,
        "PointMaze-teleport has a lower controller-oracle ceiling; full memory matches that ceiling rather than reaching 1.0.",
        fontsize=7, color="#656760",
    )
    fig.tight_layout(pad=0.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="svg", bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
