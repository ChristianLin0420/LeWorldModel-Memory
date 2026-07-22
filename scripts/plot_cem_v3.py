#!/usr/bin/env python3
"""Render real CEM-v3 LeWM factorial and grouped-CE diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/cem_lewm_v3"
ASSETS = ROOT / "docs/assets"
CONFIGS = ("A", "B", "C", "D")
LEVELS = ("cue_latent", "memory_only", "injected_context", "host_output")


def load(config: str, seed: int) -> dict:
    path = (RUN / config / "multi-item-visual-binding-recall"
            / f"s{seed}" / "summary.json")
    return json.loads(path.read_text())


def save(fig, stem: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 220}), ("pdf", {})):
        fig.savefig(ASSETS / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def exposure() -> None:
    values = np.asarray([
        [[load(config, seed)["diagnostic_ladder"][level]
          for level in LEVELS] for seed in range(3)]
        for config in CONFIGS])
    means, stds = values.mean(1), values.std(1)
    x = np.arange(len(LEVELS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = ("#8c8c8c", "#42a5f5", "#ef6c00", "#5e35b1")
    for index, config in enumerate(CONFIGS):
        ax.bar(x + (index - 1.5) * width, means[index], width,
               yerr=stds[index], capsize=3, label=config,
               color=colors[index], edgecolor="black", linewidth=0.5)
    ax.axhline(1 / 6, color="black", linestyle=":", label="chance")
    ax.axhline(0.75, color="#c62828", linestyle="--", label="pass gate")
    ax.set_xticks(x, ("Cue latent", "Memory only", "Injected context",
                     "Frozen host output"))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("CEM-v3 LeWM exposure ladder · age 15 · mean ± SD, 3 seeds")
    ax.legend(ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "cem_v3_lewm_exposure_factorial")


def calibration() -> None:
    single_true, single_hat, group_true, group_hat = [], [], [], []
    for seed in range(3):
        summary = load("D", seed)
        group = summary["causal_deletion"]["group_ce"]
        group_true.extend(group["ce_true_norm"])
        group_hat.extend(group["ce_hat"])
        log_path = (RUN / "D/multi-item-visual-binding-recall"
                    / f"s{seed}/decision_log.json")
        log = json.loads(log_path.read_text())
        single_true.extend(e["ce_true"] for e in log["events"])
        single_hat.extend(e["ce_hat"] for e in log["events"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    axes[0].scatter(single_true, single_hat, s=28, alpha=0.7,
                    color="#757575", edgecolor="black", linewidth=0.3)
    axes[0].set_title("Single-frame CE")
    axes[1].scatter(group_true, group_hat, s=40, alpha=0.8,
                    color="#5e35b1", edgecolor="black", linewidth=0.3)
    axes[1].set_title("Matched 3-frame group CE")
    for ax in axes:
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel("True deletion effect")
        ax.set_ylabel("Predicted CE")
        ax.grid(alpha=0.25)
    fig.suptitle("CEM-v3 calibration: redundant cues require group deletion")
    fig.tight_layout()
    save(fig, "cem_v3_group_vs_single_ce")


def dinowm_semantic() -> None:
    policies = ("pooled", "random", "surprise_semantic")
    rows = []
    for policy in policies:
        seeds = (0, 1, 2) if policy == "surprise_semantic" else (0,)
        summaries = [
            json.loads((ROOT / "outputs/cem_dinowm_v3" / policy
                        / "wall" / f"s{seed}/summary.json").read_text())
            for seed in seeds]
        rows.append({
            "loss_gain": np.asarray([
                x["endpoint_improvement"] for x in summaries]),
            "audit": np.asarray([x["audit"]["full"] for x in summaries]),
            "spearman": np.asarray([
                x["surrogate_spearman"] for x in summaries]),
        })
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    colors = ("#757575", "#42a5f5", "#5e35b1")
    for ax, key, title in zip(
            axes, ("loss_gain", "audit", "spearman"),
            ("Host-loss reduction", "Full audit BAcc", "CE Spearman")):
        means = [row[key].mean() for row in rows]
        errs = [row[key].std() for row in rows]
        ax.bar(range(3), means, yerr=errs, capsize=3,
               color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(3), ("Pooled", "Random", "Surprise\nsemantic"))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[1].axhline(0.75, color="#c62828", linestyle="--")
    axes[2].axhline(0.5, color="#c62828", linestyle="--")
    fig.suptitle("DINO-WM Wall · frozen DINOv2 semantic targets · age 15")
    fig.tight_layout()
    save(fig, "cem_v3_dinowm_semantic_comparison")


if __name__ == "__main__":
    exposure()
    calibration()
    dinowm_semantic()
    print("wrote CEM-v3 factorial figures")
