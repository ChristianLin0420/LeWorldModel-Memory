#!/usr/bin/env python3
"""White-background figures for patch alignment and masking."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/cem_patch_alignment_v1"
ASSETS = ROOT / "docs/assets"


def save(fig: plt.Figure, name: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = ASSETS / f"cem_patch_alignment_{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def main() -> None:
    report = json.loads((OUTPUT / "report.json").read_text())
    ASSETS.mkdir(parents=True, exist_ok=True)
    factorial = report["factorial"]
    artifacts = {}

    names = [
        "A_no_alignment", "B_random_25", "B_random_50", "B_random_75",
        "C_semantic_change_50", "D_causal_alignment", "E_random25_causal",
    ]
    labels = ["None", "Rnd 25", "Rnd 50", "Rnd 75", "Semantic", "Causal", "Hybrid"]
    recovery = [
        factorial[name]["mean"] if name == "A_no_alignment"
        else factorial[name]["recovery"]["mean"]
        for name in names
    ]
    low = [
        factorial[name]["ci95"][0] if name == "A_no_alignment"
        else factorial[name]["recovery"]["ci95"][0]
        for name in names
    ]
    high = [
        factorial[name]["ci95"][1] if name == "A_no_alignment"
        else factorial[name]["recovery"]["ci95"][1]
        for name in names
    ]
    fig, axis = plt.subplots(figsize=(9.2, 3.8))
    x = np.arange(len(names))
    axis.bar(x, np.asarray(recovery) * 100, color=["#666666"] + ["#28659c"] * 3 + ["#7c4d9e", "#d2772f", "#3a7d44"])
    axis.errorbar(x, np.asarray(recovery) * 100, yerr=[(np.asarray(recovery)-low)*100, (np.asarray(high)-recovery)*100], fmt="none", color="black")
    axis.axhline(50, color="#a33a2b", linestyle="--", label="50% Gate B")
    axis.set(xticks=x, xticklabels=labels, ylabel="Oracle opportunity recovered (%)", title="Patch alignment factorial · six full cells")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    artifacts["factorial_recovery"] = save(fig, "factorial_recovery")

    ratios = [25, 50, 75]
    random_names = ["B_random_25", "B_random_50", "B_random_75"]
    host = [factorial[name]["recovery"]["mean"] * 100 for name in random_names]
    recon = [factorial[name]["reconstruction_mse"]["mean"] for name in random_names]
    fig, left = plt.subplots(figsize=(6.2, 3.7))
    right = left.twinx()
    line1, = left.plot(ratios, host, "o-", color="#28659c", label="Host recovery")
    line2, = right.plot(ratios, recon, "s-", color="#d2772f", label="DINO reconstruction MSE")
    left.axhline(factorial["A_no_alignment"]["mean"] * 100, color="#666666", linestyle="--", label="No alignment")
    left.set(xlabel="Random mask ratio (%)", ylabel="Host recovery (%)", title="Random masking: reconstruction vs host utility", xticks=ratios)
    right.set_ylabel("Masked-patch reconstruction MSE")
    left.legend([line1, line2], ["Host recovery", "Reconstruction MSE"], frameon=False)
    left.spines["top"].set_visible(False); right.spines["top"].set_visible(False)
    fig.tight_layout()
    artifacts["mask_ratio"] = save(fig, "mask_ratio_curve")

    gap_data = report["environment_gap_results"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    for axis, (env, title) in zip(axes, (("cube-single-play-v0", "Cube"), ("pointmaze-large-navigate-v0", "PointMaze"))):
        gaps = [32, 64, 128]
        for method, color, label in (
            ("A_no_alignment", "#666666", "No alignment"),
            ("B_random_25", "#28659c", "Random 25"),
            ("D_causal_alignment", "#d2772f", "Causal"),
            ("E_random25_causal", "#3a7d44", "Hybrid"),
        ):
            axis.plot(gaps, [gap_data[env][str(gap)][method]["recovery"]["mean"] * 100 for gap in gaps], "o-", color=color, label=label)
        axis.axhline(50, color="#a33a2b", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel="Gap (frames)", xticks=gaps)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Recovery (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    artifacts["environment_gap"] = save(fig, "environment_gap")

    fig, axis = plt.subplots(figsize=(6.2, 3.7))
    for name, label, color in zip(names[1:], labels[1:], ["#28659c", "#3b78ae", "#5590c2", "#7c4d9e", "#d2772f", "#3a7d44"]):
        axis.scatter(factorial[name]["attention_entropy"]["mean"], factorial[name]["recovery"]["mean"] * 100, s=55, color=color, label=label)
    axis.axvline(np.log(16), color="#666666", linestyle="--")
    axis.set(xlabel="Attention entropy", ylabel="Recovery (%)", title="Uniform attention persists")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    artifacts["entropy_pareto"] = save(fig, "entropy_recovery")

    causal_names = ["B_random_25", "C_semantic_change_50", "D_causal_alignment", "E_random25_causal"]
    fig, axis = plt.subplots(figsize=(7.0, 3.6))
    values = [factorial[name]["high_minus_random_deletion"]["mean"] for name in causal_names]
    lows = [factorial[name]["high_minus_random_deletion"]["ci95"][0] for name in causal_names]
    highs = [factorial[name]["high_minus_random_deletion"]["ci95"][1] for name in causal_names]
    x = np.arange(len(values))
    axis.bar(x, values, color=["#28659c", "#7c4d9e", "#d2772f", "#3a7d44"])
    axis.errorbar(x, values, yerr=[np.asarray(values)-lows, np.asarray(highs)-values], fmt="none", color="black")
    axis.axhline(0, color="#444444", linewidth=0.8)
    axis.set(xticks=x, xticklabels=["Rnd25", "Semantic", "Causal", "Hybrid"], ylabel="High-effect minus random deletion", title="Attention does not identify causal patches")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    artifacts["deletion"] = save(fig, "deletion_gap")

    receipt = {"schema": "cem_patch_alignment_figure_receipt_v1", "white_background": True, "artifacts": artifacts}
    (OUTPUT / "figure_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
