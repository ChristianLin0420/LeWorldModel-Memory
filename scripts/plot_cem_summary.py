#!/usr/bin/env python3
"""Aggregate CEM (Causal-Effect Memory) results into a single summary figure.

Reads every ``outputs/cem_*_v1/**/summary.json`` plus the shortcut-probe
summary and renders a 4-panel results chart used by the docs and the NVIDIA
dashboard:

  1. Host future-latent loss WITH vs WITHOUT memory (the CEM endpoint).
  2. Fail-closed audit balanced accuracy (full / reset / no-state).
  3. Causal-deletion: delta host-loss when deleting high-CEhat vs random slots.
  4. Surprise-WRITE vs saliency-miner label BAcc on the colour shortcut.

Writes docs/assets/cem_results_summary.{png,pdf}.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "docs" / "assets"
PI = {
    "ink": "#111827", "cream": "#f5f4ef", "paper": "#fbfbf9", "yellow": "#fbd45b",
    "yellow_deep": "#d8a900", "muted": "#656760", "line": "#d4d3cb",
    "good": "#315b2c", "bad": "#7f1d1d", "gray": "#9ca3af", "blue": "#2563eb",
}


def _style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "axes.edgecolor": PI["ink"],
        "axes.labelcolor": PI["ink"], "axes.titlecolor": PI["ink"],
        "xtick.color": PI["ink"], "ytick.color": PI["ink"], "text.color": PI["ink"],
        "figure.facecolor": "#ffffff", "axes.facecolor": PI["paper"],
        "savefig.facecolor": "#ffffff", "grid.color": PI["line"],
        "legend.frameon": False,
    })


def load_summaries():
    groups = defaultdict(list)
    for p in sorted(glob.glob(str(REPO / "outputs/cem_*_v1/**/summary.json"),
                              recursive=True)):
        d = json.loads(Path(p).read_text())
        mode = d.get("endpoint_mode") or "cue_conditioned"
        key = (d["host"], d["env"], mode)
        groups[key].append(d)
    return groups


def label_for(key):
    host, env, mode = key
    env_short = {"transient-visual-token-recall": "transient",
                 "multi-item-visual-binding-recall": "multi-item",
                 "wall": "wall"}.get(env, env)
    tag = "own-roll" if mode == "own_rollout" else "recall"
    if host == "dinowm":
        return f"DINO\n{env_short}"
    return f"LeWM\n{env_short}\n({tag})"


def mean(vs):
    return float(np.mean(vs)) if vs else float("nan")


def main():
    _style()
    groups = load_summaries()
    keys = sorted(groups.keys(), key=lambda k: (k[0] != "dinowm", k[1], k[2]))
    labels = [label_for(k) for k in keys]
    x = np.arange(len(keys))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    fig.suptitle("Causal-Effect Memory (CEM) on frozen world-model hosts \u2014 real results (v1)",
                 fontsize=14, x=0.02, ha="left")

    # Panel 1: endpoint loss with vs without memory (normalized per config since
    # LeWM/DINO losses live on different scales -> show as fraction of no-memory).
    ax = axes[0, 0]
    lm = [mean([d["host_loss_with_memory"] for d in groups[k]]) for k in keys]
    ln = [mean([d["host_loss_without_memory"] for d in groups[k]]) for k in keys]
    frac = [m / n if n else np.nan for m, n in zip(lm, ln)]
    ax.bar(x - 0.2, [1.0] * len(keys), 0.4, color=PI["gray"], label="without memory")
    ax.bar(x + 0.2, frac, 0.4, color=PI["yellow_deep"], label="with memory")
    for i, (m, n) in enumerate(zip(lm, ln)):
        ax.text(i + 0.2, (m / n if n else 0) + 0.02, f"{m:.3g}\n/{n:.3g}",
                ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("host future-latent loss\n(fraction of no-memory)")
    ax.set_title("Endpoint: memory lowers host's own loss", fontsize=10.5, loc="left")
    ax.axhline(1.0, color=PI["ink"], lw=0.8, ls=":")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.4)

    # Panel 2: audit
    ax = axes[0, 1]
    full = [mean([d["audit"]["full"] for d in groups[k]]) for k in keys]
    reset = [mean([d["audit"]["reset"] for d in groups[k]]) for k in keys]
    nost = [mean([d["audit"]["no_state"] for d in groups[k]]) for k in keys]
    ax.bar(x - 0.25, full, 0.25, color=PI["good"], label="full (mem)")
    ax.bar(x, reset, 0.25, color=PI["gray"], label="reset")
    ax.bar(x + 0.25, nost, 0.25, color=PI["line"], label="no-state")
    ax.axhline(0.75, color=PI["bad"], lw=1.0, ls="--", label="pass \u2265 0.75")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("label balanced accuracy")
    ax.set_title("Fail-closed audit (labels post-hoc only)", fontsize=10.5, loc="left")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax.grid(True, axis="y", alpha=0.4)

    # Panel 3: causal-deletion high-CEhat vs random
    ax = axes[1, 0]
    hi = [mean([d["causal_deletion"]["delta_loss_delete_high_ce_hat"] for d in groups[k]]) for k in keys]
    rnd = [mean([d["causal_deletion"]["delta_loss_delete_random"] for d in groups[k]]) for k in keys]
    ax.bar(x - 0.2, hi, 0.4, color=PI["yellow_deep"], label="delete high-$\\hat{CE}$")
    ax.bar(x + 0.2, rnd, 0.4, color=PI["gray"], label="delete random")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("$\\Delta$ host loss when deleted")
    ax.set_title("Causal-deletion: does high-$\\hat{CE}$ matter more?", fontsize=10.5, loc="left")
    ax.axhline(0.0, color=PI["ink"], lw=0.8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.4)
    # annotate spearman
    rho = [mean([d["surrogate_spearman"] for d in groups[k]]) for k in keys]
    for i, r in enumerate(rho):
        ax.text(i, ax.get_ylim()[1] * 0.9, f"$\\rho$={r:.2f}", ha="center",
                fontsize=7, color=PI["blue"])

    # Panel 4: shortcut probe
    ax = axes[1, 1]
    sc_path = REPO / "outputs/cem_shortcut_v1/summary.json"
    if sc_path.is_file():
        sc = json.loads(sc_path.read_text())
        modes = list(sc["modes"].keys())
        xs = np.arange(len(modes))
        sur = [sc["modes"][m]["surprise_label_bacc"] for m in modes]
        sal = [sc["modes"][m]["saliency_label_bacc"] for m in modes]
        tru = [sc["modes"][m]["true_cue_label_bacc"] for m in modes]
        ax.bar(xs - 0.25, sur, 0.25, color=PI["yellow_deep"], label="surprise WRITE")
        ax.bar(xs, sal, 0.25, color=PI["bad"], label="saliency miner")
        ax.bar(xs + 0.25, tru, 0.25, color=PI["good"], label="true-cue ceiling")
        ax.axhline(sc["chance_bacc"], color=PI["ink"], lw=0.9, ls=":", label="chance")
        ax.set_xticks(xs); ax.set_xticklabels(modes, fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("label balanced accuracy")
        ax.set_title("Colour shortcut: surprise vs saliency miner", fontsize=10.5, loc="left")
        ax.legend(fontsize=7.5, loc="upper right", ncol=2)
        ax.grid(True, axis="y", alpha=0.4)
    else:
        ax.text(0.5, 0.5, "shortcut probe not run", ha="center", va="center")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        p = ASSETS / f"cem_results_summary.{ext}"
        fig.savefig(p, bbox_inches="tight", **kw)
        out.append(p)
    plt.close(fig)
    for p in out:
        print("wrote", p)


if __name__ == "__main__":
    main()
