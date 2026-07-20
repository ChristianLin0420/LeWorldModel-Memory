#!/usr/bin/env python3
"""Bar chart for the frozen-LeWM host-writer control battery.

Reads ``outputs/lewm_host_controls_v1/controls.json`` and renders
``paper_c/figures/fig_c_lewm_controls.pdf`` (and ``.png``) in the Physical
Intelligence black/cream/yellow theme.  Does not touch scripts/plot_paper_c.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/lewm_host_controls_v1/controls.json"
DEFAULT_OUTPUT = ROOT / "paper_c/figures/fig_c_lewm_controls"

PI = {
    "ink": "#111827",
    "black": "#000000",
    "cream": "#f5f4ef",
    "yellow": "#fbd45b",
    "yellow_deep": "#d8a900",
    "muted": "#656760",
    "line": "#d4d3cb",
    "gray": "#c4b5a5",
    "good": "#315b2c",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    order = data["condition_order"]
    conditions = data["conditions"]
    chance = float(data["chance_level"])
    gate = float(data["gate"]["full_minimum"])
    n_seeds = len(data["seeds"])

    labels = [conditions[name]["label"] for name in order]
    means = [conditions[name]["mean"] for name in order]
    stds = [conditions[name]["std"] for name in order]

    # The recovery bar (correct) and its slot-memory probe are the signal;
    # everything else is a control expected near chance.
    colors = []
    for name in order:
        if name in {"correct", "memory_only"}:
            colors.append(PI["yellow"])
        elif name in {"host_only", "no_state"}:
            colors.append(PI["gray"])
        else:
            colors.append("#e7e4d8")

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": PI["ink"],
        "axes.linewidth": 0.9,
    })

    fig, ax = plt.subplots(figsize=(6.6, 3.5), dpi=200)
    fig.patch.set_facecolor(PI["cream"])
    ax.set_facecolor(PI["cream"])

    x = range(len(order))
    bars = ax.bar(x, means, width=0.68, color=colors,
                  edgecolor=PI["ink"], linewidth=0.9, zorder=3)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=PI["ink"],
                elinewidth=1.0, capsize=3.0, capthick=1.0, zorder=4)

    for xi, mean, std in zip(x, means, stds):
        ax.text(xi, mean + std + 0.025, f"{mean:.2f}", ha="center",
                va="bottom", fontsize=8.0, color=PI["ink"], zorder=5)

    ax.axhline(gate, color=PI["good"], lw=1.3, ls="--", zorder=2)
    ax.text(len(order) - 0.5, gate + 0.012, f"pass gate = {gate:.2f}",
            ha="right", va="bottom", fontsize=7.6, color=PI["good"])
    ax.axhline(chance, color=PI["yellow_deep"], lw=1.2, ls=":", zorder=2)
    ax.text(len(order) - 0.5, chance + 0.012,
            f"chance = 1/6 = {chance:.2f}", ha="right", va="bottom",
            fontsize=7.6, color=PI["yellow_deep"])

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.4)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("host-output balanced accuracy")
    ax.set_title(
        "Frozen PushT LeWM host-writer: control battery "
        f"(age 15, {n_seeds} seeds, mean $\\pm$ s.d.)",
        fontsize=9.2, color=PI["ink"], pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=PI["ink"])
    ax.grid(axis="y", color=PI["line"], lw=0.6, zorder=0)

    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    fig.savefig(out.with_suffix(".png"), facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {out.with_suffix('.pdf')}")
    print(f"wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
