#!/usr/bin/env python3
"""Plot autonomous Masked-Evidence JEPA OGBench summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "outputs" / "masked_evidence_jepa_ogbench_v1" / "summary.json"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "masked_evidence_jepa_ogbench_summary.svg"


def short_env(env_name: str) -> str:
    return {
        "pointmaze-large-navigate-v0": "PointMaze",
        "cube-single-play-v0": "Cube-single",
    }.get(env_name, env_name.replace("-navigate-v0", "").replace("-play-v0", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    rows = summary["rows"]
    labels = [f"{short_env(r['env_name'])}\nage {r['age']}" for r in rows]
    full = np.asarray([r["full_bacc_mean"] for r in rows], dtype=np.float64)
    reset = np.asarray([r["reset_bacc_mean"] for r in rows], dtype=np.float64)
    none = np.asarray([r["no_state_bacc_mean"] for r in rows], dtype=np.float64)
    top1 = np.asarray([r["retrieval_top1_mean"] for r in rows], dtype=np.float64)
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 2.7), dpi=180, gridspec_kw={"width_ratios": [1.8, 1.0]})
    fig.patch.set_facecolor("#fbfbf9")
    for ax in axes:
        ax.set_facecolor("#fbfbf9")
        ax.grid(axis="y", color="#d4d3cb", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#333b49")
            spine.set_linewidth(0.7)
        ax.tick_params(length=0, labelsize=7)
    axes[0].bar(x - width, full, width, color="#111827", label="full")
    axes[0].bar(x, reset, width, color="#9ca3af", label="reset")
    axes[0].bar(x + width, none, width, color="#d4d3cb", edgecolor="#333b49", linewidth=0.4, label="recent-only")
    axes[0].axhline(0.25, color="#7f1d1d", linestyle="--", linewidth=0.8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("post-hoc BAcc", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_title("Memory readout after self-supervised training", fontsize=9, loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    axes[1].bar(x, top1, width=0.52, color="#fbd45b", edgecolor="#111827", linewidth=0.5)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("latent retrieval top-1", fontsize=8)
    axes[1].set_title("JEPA target retrieval", fontsize=9, loc="left", fontweight="bold")
    fig.text(
        0.01,
        -0.02,
        "Training loss uses no cue labels or cue-feature handoff; labels are used only for post-hoc readout.",
        fontsize=7,
        color="#656760",
    )
    fig.tight_layout(pad=0.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower().lstrip(".") or "svg"
    fig.savefig(args.output, format=suffix, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
