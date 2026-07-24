#!/usr/bin/env python3
"""White-background figures for decision-conditioned memory."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/cem_decision_memory_v1"
ASSETS = ROOT / "docs/assets"
CACHE = (
    ROOT / "outputs/multiview_patchset_color_jepa_native_v1/cache"
    / "pointmaze-large-navigate-v0/render_cache.npz"
)


def save(fig: plt.Figure, name: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = ASSETS / f"cem_decision_{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def main() -> None:
    report = json.loads((OUTPUT / "report.json").read_text())
    ASSETS.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    methods = [
        "no_memory", "recent_only", "random_event", "surprise",
        "oracle_frame", "oracle_discovered_event", "all_history_upper",
    ]
    labels = ["None", "Recent", "Random", "Surprise", "Frame oracle", "Event oracle", "All history"]
    fig, axis = plt.subplots(figsize=(9.2, 3.8))
    values = [report["aggregate_action_accuracy"][name]["mean"] * 100 for name in methods]
    lows = [report["aggregate_action_accuracy"][name]["ci95"][0] * 100 for name in methods]
    highs = [report["aggregate_action_accuracy"][name]["ci95"][1] * 100 for name in methods]
    x = np.arange(len(methods))
    axis.bar(x, values, color=["#777777", "#444444", "#999999", "#7c4d9e", "#d2772f", "#28659c", "#3a7d44"])
    axis.errorbar(x, values, yerr=[np.asarray(values)-lows, np.asarray(highs)-values], fmt="none", color="black")
    axis.set(xticks=x, xticklabels=labels, ylabel="Candidate-action top-1 (%)", title="Decision-conditioned oracle ladder")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    artifacts["oracle_ladder"] = save(fig, "oracle_ladder")

    fig, axis = plt.subplots(figsize=(7.0, 3.7))
    gaps = [32, 64, 128]
    for method, color, label in (
        ("recent_only", "#444444", "Recent"),
        ("oracle_discovered_event", "#28659c", "Event oracle"),
        ("all_history_upper", "#3a7d44", "All history"),
    ):
        axis.plot(gaps, [next(row for row in report["gaps"] if row["gap"] == gap)["methods"][method]["accuracy"]["mean"] * 100 for gap in gaps], "o-", color=color, label=label)
    axis.set(xlabel="Gap (frames)", ylabel="Action top-1 (%)", title="Anti-recency action ranking", xticks=gaps)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    artifacts["action_ranking"] = save(fig, "action_ranking")

    gate2 = report["gate2"]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    axes[0].bar(["Spearman", "Pairwise"], [gate2["du_spearman"]["mean"], gate2["du_pairwise_accuracy"]["mean"]], color=["#d2772f", "#28659c"])
    axes[0].axhline(0.2, color="#a33a2b", linestyle="--")
    axes[0].axhline(0.65, color="#a33a2b", linestyle=":")
    axes[0].set_title("Conditional utility calibration")
    deletion = gate2["high_minus_random_deletion"]
    axes[1].bar(["High − random"], [deletion["mean"]], color="#7c4d9e")
    axes[1].errorbar([0], [deletion["mean"]], yerr=[[deletion["mean"]-deletion["ci95"][0]], [deletion["ci95"][1]-deletion["mean"]]], fmt="none", color="black")
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].set_title("DU deletion audit")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    artifacts["du"] = save(fig, "du_calibration_deletion")

    fig, axis = plt.subplots(figsize=(6.4, 3.6))
    coverage = gate2["activation_coverage"]["values"]
    precision = gate2["activation_precision"]["values"]
    axis.scatter(np.asarray(coverage) * 100, np.asarray(precision) * 100, color="#28659c", s=60)
    axis.set(xlabel="Activation coverage (%)", ylabel="Positive-utility precision (%)", title="Conservative router coverage–precision", xlim=(-2, 102), ylim=(-2, 102))
    axis.axhline(50, color="#777777", linestyle="--")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    artifacts["activation"] = save(fig, "activation_coverage_risk")

    with np.load(OUTPUT / "build/pointmaze-large-navigate-v0/pairs.npz", allow_pickle=False) as pairs:
        sources = pairs["test_sources"][0]
        donor = int(pairs["test_donors"][0])
    with np.load(CACHE, allow_pickle=False) as data:
        frames = data["frames"]
        indices = [
            (int(sources[0]), 4, "branch A old event"),
            (int(sources[1]), 4, "branch B old event"),
            (donor, 12, "shared suffix A"),
            (donor, 12, "shared suffix B"),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.5))
        for axis, (episode, time_index, title) in zip(axes, indices):
            axis.imshow(Image.fromarray(frames[episode, time_index]).resize((180, 180)))
            axis.set_title(title, fontsize=8)
            axis.axis("off")
    fig.suptitle("Unmodified raw frames · branch-specific history, identical recent suffix")
    fig.tight_layout()
    artifacts["storyboard"] = save(fig, "paired_storyboard")

    receipt = {"schema": "cem_decision_figure_receipt_v1", "white_background": True, "artifacts": artifacts}
    (OUTPUT / "figure_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
