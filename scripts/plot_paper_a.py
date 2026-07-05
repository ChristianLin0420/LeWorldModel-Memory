#!/usr/bin/env python3
"""Paper A figures, artifact-sourced (no hand-entered numbers).

fig_a_delay  — registered-probe accuracy vs episode length for the three
               carriers, chance line, and the F3 rescaled-spectrum overlay.
fig_a_sstar  — the two-host x two-scene salience-ladder grid: sighted-probe
               scores per rung against the 0.75 gate; host encoded by hue,
               scene by marker/linestyle (grayscale-safe composite encoding).

Writes docs/figures/fig_a_{delay,sstar}.{pdf,png}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"

# dataviz reference palette (validated: CVD dE 47.2; aqua/yellow relief via
# direct labels + marker shapes)
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def fig_delay() -> None:
    delay = load("outputs/v21_x3/delay_scaling.json")["curves"]
    tau = load("outputs/v21_x3/tau_rescale.json")["curves"]
    lengths = (64, 96, 128)
    series = [
        ("lkc_rfix", "filter (rfix)", BLUE, "o", "-", (6, 0)),
        ("gdelta_l10", "delta cell", AQUA, "s", "-", (6, 6)),
        ("acgru", "GRU", YELLOW, "^", "-", (6, -8)),
    ]
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for arm, label, color, marker, style, offset in series:
        means = [delay[f"{arm}@L{l}"]["mean"] for l in lengths]
        sds = [delay[f"{arm}@L{l}"]["sd"] for l in lengths]
        ax.errorbar(lengths, means, yerr=sds, color=color, marker=marker,
                    linestyle=style, linewidth=2, markersize=6.5,
                    capsize=2.5, elinewidth=1, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=3)
        ax.annotate(label, (lengths[-1], means[-1]),
                    xytext=offset, textcoords="offset points",
                    va="center", fontsize=8.2, color=INK)
    rescaled = [tau[f"L{l}"]["rescaled_mean"] for l in lengths]
    ax.plot(lengths, rescaled, color=BLUE, marker="o", linestyle=(0, (4, 3)),
            linewidth=1.6, markersize=6, markerfacecolor="white",
            markeredgecolor=BLUE, markeredgewidth=1.4, zorder=2)
    ax.annotate("filter, spectrum\nre-derived per L", (96, tau["L96"]["rescaled_mean"]),
                xytext=(-4, 14), textcoords="offset points", fontsize=7.6,
                color=INK2, ha="right")
    ax.axhline(0.25, color=INK2, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
    ax.annotate("chance", (65, 0.25), xytext=(0, 4),
                textcoords="offset points", fontsize=7.6, color=INK2)
    ax.axvline(64, color=GRID, linewidth=1, zorder=0)
    ax.annotate("training length", (64, 0.55), fontsize=7.6, color=INK2,
                rotation=90, xytext=(-11, 0), textcoords="offset points")
    ax.set_xticks(lengths)
    ax.set_xlim(58, 152)
    ax.set_ylim(0.2, 0.58)
    ax.set_xlabel("episode length L (cue-to-decision delay ≈ L − 14)")
    ax.set_ylabel("registered probe accuracy")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_a_delay.{ext}", dpi=300)
    plt.close(fig)


def fig_sstar() -> None:
    rungs = ["t1s0c", "t1s0b", "t1s0a", "t1s1", "t1s2", "t1s3", "t1"]
    x = {r: i for i, r in enumerate(rungs)}

    def dino_levels(rel: str) -> dict[str, list[float]]:
        return {lvl: row["scores"]
                for lvl, row in load(rel)["levels"].items()}

    dino_reacher = dino_levels("outputs/v21_x3/dino_sstar.json")
    dino_reacher.update(dino_levels("outputs/v21_x3/dino_sstar_ext.json"))
    dino_pm = dino_levels("outputs/v21_x3/dino_sstar_pointmass.json")
    w0 = load("outputs/v20_w0/w0_summary.json")["ladder_readout"]["vicreg"]
    vicreg_reacher = {lvl: row["sighted_scores"]
                      for lvl, row in w0["levels"].items()}
    vicreg_pm = {"t1s1": [
        load("outputs/v21_f2b/certificates/t1s1/vicreg/s0.json")["sighted"]["score"],
        load("outputs/v21_f2b/certificates/t1s1/vicreg/s1.json")["sighted"]["score"]]}

    series = [
        (vicreg_reacher, "task-trained VICReg · reacher", BLUE, "o", "-", True),
        (vicreg_pm, "task-trained VICReg · point-mass", BLUE, "s", (0, (4, 3)), False),
        (dino_reacher, "frozen DINOv2 · reacher", AQUA, "o", "-", True),
        (dino_pm, "frozen DINOv2 · point-mass", AQUA, "s", (0, (4, 3)), False),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    dodges = (-0.09, -0.03, 0.03, 0.09)
    for (levels, label, color, marker, style, filled), dodge in zip(series,
                                                                    dodges):
        xs = sorted(x[lvl] for lvl in levels)
        means = [float(np.mean(levels[rungs[i]])) for i in xs]
        los = [means[j] - min(levels[rungs[i]]) for j, i in enumerate(xs)]
        his = [max(levels[rungs[i]]) - means[j] for j, i in enumerate(xs)]
        ax.errorbar([i + dodge for i in xs], means, yerr=[los, his],
                    color=color, marker=marker,
                    linestyle=style if len(xs) > 1 else "none",
                    linewidth=1.8, markersize=6.5, capsize=2.5, elinewidth=1,
                    markerfacecolor=color if filled else "white",
                    markeredgecolor=color, markeredgewidth=1.3,
                    label=label, zorder=3)
    ax.axhline(0.75, color=INK2, linewidth=1, linestyle=(0, (2, 3)), zorder=1)
    ax.annotate("sighted gate 0.75", (0.05, 0.75), xytext=(0, -10),
                textcoords="offset points", fontsize=7.6, color=INK2)
    ax.annotate("s*(vicreg·reacher) = t1s2", (x["t1s2"] + 0.15, 0.30),
                fontsize=7.6, color=INK, ha="center")
    ax.annotate("", (x["t1s2"], 0.50), (x["t1s2"], 0.36),
                arrowprops={"arrowstyle": "->", "color": INK2, "lw": 1})
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels(["s0c", "s0b", "s0a", "s1", "s2", "s3", "t1"])
    ax.set_xlabel("salience ladder (increasing cue salience →)")
    ax.set_ylabel("sighted-probe score")
    ax.set_ylim(0.15, 1.06)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              fontsize=7.4, frameon=False, handlelength=2.6,
              columnspacing=1.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_a_sstar.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_delay()
    fig_sstar()
    print(f"[plot-a] wrote fig_a_delay + fig_a_sstar under {FIG}")


if __name__ == "__main__":
    main()
