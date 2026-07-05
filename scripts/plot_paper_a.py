#!/usr/bin/env python3
"""Paper A figures, artifact-sourced (no hand-entered numbers).

NVIDIA-themed palette (validated: worst adjacent CVD dE 31.5; the
below-3:1 green carries relief via direct labels + marker shapes):
green #76B900 (hero/filter), deep blue #1C5CAB, orange #EB6834.

fig_a_protocol — compact motivation strip: the two-sided demand
                 certificate on an episode timeline, annotated with the
                 external-audit numbers.
fig_a_arch     — pipeline/architecture diagram: pixels -> host encoder ->
                 belief carrier -> readouts, with the certificates bound
                 to the stages they gate.
fig_a_results  — one row, two panels: (a) two-host x two-scene s* grid,
                 (b) delay curves with the falsified spectrum repair.

Writes docs/figures/fig_a_{protocol,arch,results}.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"

NV_GREEN, NV_BLUE, NV_ORANGE = "#76B900", "#1C5CAB", "#EB6834"
NV_GREEN_DK = "#5A8C00"
INK, INK2, GRID = "#1A1A1A", "#666666", "#E5E5E5"
TINT_GREEN, TINT_GRAY, TINT_BLUE = "#EFF6DC", "#F2F2F2", "#E4EDF7"

plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def fig_protocol() -> None:
    f1 = load("outputs/v21_f1/certification.json")
    banks = [f1["banks"][k] for k in sorted(f1["banks"])]
    sighted = float(np.mean([b["sighted"] for b in banks]))
    floor = float(np.mean([b["floor"] for b in banks]))
    leak = float(np.mean([b["leakage"] for b in banks]))
    chance = f1["chance"]

    fig, ax = plt.subplots(figsize=(4.8, 1.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    spans = [(2, 20, TINT_GREEN, "cue window\n$\\xi$ visible"),
             (20, 36, TINT_GRAY, "delay"),
             (36, 98, TINT_BLUE, "decision window ($\\xi$ unmarked)")]
    for x0, x1, face, label in spans:
        ax.fill_between([x0, x1], 6.0, 9.4, color=face, zorder=1)
        ax.annotate(label, ((x0 + x1) / 2, 7.7), ha="center", va="center",
                    fontsize=7.0, color=INK)
    ax.plot([2, 98], [6.0, 6.0], color=INK2, lw=0.9)
    ax.plot([2, 98], [9.4, 9.4], color=INK2, lw=0.9)
    ax.annotate("$o_0$", (2, 9.8), fontsize=7.5, color=INK, ha="left")
    probes = [
        (11, "sighted $\\geq$ 0.75", f"{sighted:.2f}  PASS", NV_GREEN_DK),
        (44, "integrator floor $[\\mathrm{enc}(o_0);a]$ $\\approx$ chance",
         f"{floor:.2f} $\\gg$ {chance:.2f}  FAIL", NV_ORANGE),
        (80, "leakage $\\approx$ chance", f"{leak:.2f}  PASS", NV_GREEN_DK),
    ]
    for x0, rule, verdict, color in probes:
        ax.annotate("", (x0, 5.9), (x0, 4.6),
                    arrowprops={"arrowstyle": "->", "color": color,
                                "lw": 1.5})
        ax.annotate(rule, (x0, 3.9), ha="center", va="top", fontsize=6.6,
                    color=INK)
        ax.annotate(verdict, (x0, 1.7), ha="center", va="top", fontsize=6.8,
                    color=color, fontweight="bold")
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_a_protocol.{ext}", dpi=300)
    plt.close(fig)


def _box(ax, x, y, w, h, text, face, edge, tcolor=INK, fs=7.0, lw=1.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=3))
    ax.annotate(text, (x + w / 2, y + h / 2), ha="center", va="center",
                fontsize=fs, color=tcolor, zorder=4)


def _arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.3):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
        color=color, linewidth=lw, zorder=2))


def fig_arch() -> None:
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")
    row = 26
    # main pipeline
    _box(ax, 1, row, 12, 11, "pixels\n$o_{0:L-1}$, $a$", TINT_GRAY, INK2,
         fs=6.4)
    _box(ax, 17, row, 19, 11,
         "host encoder\n$z_t = E(o_t)$\n(trained or frozen)",
         TINT_GRAY, INK2, fs=6.4)
    _box(ax, 40, row, 24, 11, "belief carrier\npredict–correct state $m_t$\n"
         "(filter / GRU / delta cell)", TINT_GRAY, INK2, fs=6.4)
    _box(ax, 68, row + 6.5, 31, 8, "probe readout\n(registered family)",
         TINT_BLUE, NV_BLUE, fs=6.4)
    _box(ax, 68, row - 3.5, 31, 8,
         "selector $\\to$ planner $\\to$ return", TINT_BLUE, NV_BLUE,
         fs=6.4)
    _arrow(ax, 13, row + 5.5, 17, row + 5.5)
    _arrow(ax, 36, row + 5.5, 40, row + 5.5)
    _arrow(ax, 64, row + 5.5, 68, row + 10.5)
    _arrow(ax, 64, row + 5.5, 68, row + 0.5)
    # certificate gates (the contribution) below, clamped to stages
    gates = [
        (17.5, "memory-demand\ncertificate\n(sighted / floor / leakage)", 9.5),
        (40.0, "$s^*$ salience instrument\nper (encoder, scene)", 29.0),
        (62.5, "rollout-competence\ncertificate", 53.0),
        (85.0, "return-floor certificate\n(oracle vs integrator gap)", 83.0),
    ]
    for gx, label, ax_x in gates:
        _box(ax, gx - 11, 2, 22, 10, label, NV_GREEN, NV_GREEN_DK,
             tcolor="white", fs=6.6, lw=1.0)
        _arrow(ax, gx, 12, ax_x, row - 0.6, color=NV_GREEN_DK, lw=1.2)
    ax.annotate("certificates gate every stage before any memory claim",
                (50, 0.1), ha="center", va="bottom", fontsize=6.8,
                color=NV_GREEN_DK, style="italic")
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_a_arch.{ext}", dpi=300)
    plt.close(fig)


def _panel_sstar(ax) -> None:
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
        (vicreg_reacher, "VICReg·reacher", NV_ORANGE, "o", "-", True),
        (vicreg_pm, "VICReg·point-mass", NV_ORANGE, "s", (0, (4, 3)), False),
        (dino_reacher, "DINOv2·reacher", NV_GREEN, "o", "-", True),
        (dino_pm, "DINOv2·point-mass", NV_GREEN, "s", (0, (4, 3)), False),
    ]
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
                    linewidth=1.6, markersize=5, capsize=2, elinewidth=0.9,
                    markerfacecolor=color if filled else "white",
                    markeredgecolor=color, markeredgewidth=1.1,
                    label=label, zorder=3)
    ax.axhline(0.75, color=INK2, linewidth=0.9, linestyle=(0, (2, 3)),
               zorder=1)
    ax.annotate("gate 0.75", (0.02, 0.765), fontsize=6.4, color=INK2)
    ax.annotate("$s^*$(vicreg·reacher) = t1s2", (0.05, 0.56),
                fontsize=6.6, color=INK, ha="left")
    ax.annotate("",
                (x["t1s2"] - 0.35, 0.82), (x["t1s1"] + 0.55, 0.585),
                arrowprops={"arrowstyle": "->", "color": INK2, "lw": 0.9})
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels(["s0c", "s0b", "s0a", "s1", "s2", "s3", "t1"],
                       fontsize=7)
    ax.set_xlabel("salience ladder (salience →)", fontsize=7.5)
    ax.set_ylabel("sighted-probe score", fontsize=7.5)
    ax.set_ylim(0.2, 1.06)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), fontsize=5.8,
              frameon=False, handlelength=2.0, borderaxespad=0.1,
              labelspacing=0.3)
    ax.set_title("(a) $s^*$: two hosts × two scenes", fontsize=7.8,
                 color=INK)


def _panel_delay(ax) -> None:
    delay = load("outputs/v21_x3/delay_scaling.json")["curves"]
    tau = load("outputs/v21_x3/tau_rescale.json")["curves"]
    lengths = (64, 96, 128)
    series = [
        ("lkc_rfix", "filter", NV_GREEN, "o", (5, 1)),
        ("gdelta_l10", "delta cell", NV_BLUE, "s", (5, 5)),
        ("acgru", "GRU", NV_ORANGE, "^", (5, -8)),
    ]
    for arm, label, color, marker, offset in series:
        means = [delay[f"{arm}@L{l}"]["mean"] for l in lengths]
        sds = [delay[f"{arm}@L{l}"]["sd"] for l in lengths]
        ax.errorbar(lengths, means, yerr=sds, color=color, marker=marker,
                    linewidth=1.6, markersize=5, capsize=2, elinewidth=0.9,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.annotate(label, (lengths[-1], means[-1]), xytext=offset,
                    textcoords="offset points", va="center", fontsize=6.6,
                    color=INK)
    rescaled = [tau[f"L{l}"]["rescaled_mean"] for l in lengths]
    ax.plot(lengths, rescaled, color=NV_GREEN, marker="o",
            linestyle=(0, (4, 3)), linewidth=1.3, markersize=4.6,
            markerfacecolor="white", markeredgecolor=NV_GREEN,
            markeredgewidth=1.1, zorder=2)
    ax.annotate("spectrum re-derived\nper L (no gain)",
                (96, tau["L96"]["rescaled_mean"]), xytext=(14, 14),
                textcoords="offset points", fontsize=6.2, color=INK2,
                ha="left")
    ax.axhline(0.25, color=INK2, linewidth=0.9, linestyle=(0, (2, 3)),
               zorder=1)
    ax.annotate("chance", (65, 0.253), fontsize=6.4, color=INK2)
    ax.axvline(64, color=GRID, linewidth=0.9, zorder=0)
    ax.set_xticks(lengths)
    ax.set_xlim(58, 148)
    ax.set_ylim(0.2, 0.55)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("episode length L (train = 64)", fontsize=7.5)
    ax.set_ylabel("probe accuracy", fontsize=7.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("(b) delay scaling + falsified repair", fontsize=7.8,
                 color=INK)


def fig_results() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45),
                             constrained_layout=True)
    _panel_sstar(axes[0])
    _panel_delay(axes[1])
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_a_results.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_protocol()
    fig_arch()
    fig_results()
    print(f"[plot-a] wrote fig_a_protocol + fig_a_arch + fig_a_results "
          f"under {FIG}")


if __name__ == "__main__":
    main()
