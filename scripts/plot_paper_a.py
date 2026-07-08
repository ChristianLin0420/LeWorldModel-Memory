#!/usr/bin/env python3
"""Paper A figures, artifact-sourced (no hand-entered numbers).

NVIDIA-themed palette (validated: worst adjacent CVD dE 31.5; the
below-3:1 green carries relief via direct labels + marker shapes):
green #76B900 (hero/filter), teal #2A6F77, brown-orange #B5622D.

fig_a_arch     — modular host, semantic carrier roles, and legal frozen taps.
fig_a_protocol — core task frames and official-LeWM availability gates.
fig_a_evidence — frozen-host carrier swap and long-context control.
fig_a_results  — learned-model rollout error and action sensitivity.

Writes docs/figures/fig_a_{protocol,arch,evidence,results}.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"

PDF_METADATA = {
    "Creator": "plot_paper_a.py",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}
PNG_METADATA = {"Software": "plot_paper_a.py"}

NV_GREEN, NV_BLUE, NV_ORANGE = "#76B900", "#2A6F77", "#B5622D"
NV_GREEN_DK = "#5A8C00"
INK, INK2, GRID = "#1A1A1A", "#666666", "#E5E5E5"
TINT_GREEN, TINT_GRAY, TINT_BLUE = "#EFF6DC", "#F2F2F2", "#E4EDF7"

plt.rcParams.update({
    "font.size": 9.0, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save_figure_pair(fig: plt.Figure, stem: str) -> None:
    """Write stable vector and raster copies without wall-clock metadata."""
    fig.savefig(FIG / f"{stem}.pdf", dpi=300, metadata=PDF_METADATA)
    fig.savefig(FIG / f"{stem}.png", dpi=300, metadata=PNG_METADATA)


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def _box(ax, x, y, w, h, text, face, edge, tcolor=INK, fs=7.0,
         lw=1.2, ls="-", zorder=3):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.55",
        facecolor=face, edgecolor=edge, linewidth=lw, linestyle=ls,
        zorder=zorder)
    ax.add_patch(patch)
    ax.annotate(text, (x + w / 2, y + h / 2), ha="center", va="center",
                fontsize=fs, color=tcolor, zorder=zorder + 1)
    return patch


def _arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.2, ls="-",
           mutation_scale=8, zorder=2):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=mutation_scale,
        color=color, linewidth=lw, linestyle=ls, zorder=zorder))


def _architecture_frames() -> list[np.ndarray]:
    """Four consecutive frames from the official-LeWM memory-task bank."""
    path = (ROOT / "outputs/paper_a_expansion/data/t1/"
            "val_clean_e240_s270702.npz")
    with np.load(path, allow_pickle=False) as bank:
        return [bank["frames"][0, t].copy() for t in (20, 21, 22, 23)]


def _token(ax, x: float, y: float, label: str, face: str, edge: str,
           w: float = 3.0, h: float = 6.0, fs: float = 6.8,
           lw: float = 1.0) -> None:
    _box(ax, x, y, w, h, label, face, edge, fs=fs, lw=lw)


def fig_arch() -> None:
    """Frozen official-LeWM carrier swap and its causal memory interface."""
    frames = _architecture_frames()
    fig, ax = plt.subplots(figsize=(5.45, 2.90))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    frozen_ls = (0, (3.0, 2.0))

    ax.annotate("(a) Frozen SIGReg LeWM carrier swap", (1.0, 98.0),
                ha="left", va="top", fontsize=8.5, fontweight="bold")

    # Compact, explicit visual key: line style carries trainability in print.
    legend_y = 93.0
    ax.add_patch(Rectangle((59.0, legend_y - 1.2), 2.3, 2.4,
                           facecolor=TINT_BLUE, edgecolor=NV_BLUE,
                           linewidth=1.0, linestyle=frozen_ls))
    ax.annotate("frozen released LeWM", (62.2, legend_y), ha="left",
                va="center", fontsize=6.5, color=INK2)
    ax.add_patch(Rectangle((81.0, legend_y - 1.2), 2.3, 2.4,
                           facecolor=TINT_GREEN, edgecolor=NV_GREEN_DK,
                           linewidth=1.2))
    ax.annotate("trainable carrier", (84.2, legend_y), ha="left",
                va="center", fontsize=6.5, color=INK2)

    # Three real context frames from the exact official-LeWM evaluation bank.
    for i, frame in enumerate(frames[:3]):
        x = 1.5 + 1.65 * i
        ax.imshow(frame, extent=(x, x + 6.2, 69.2, 82.0),
                  interpolation="nearest", aspect="auto", zorder=2 + i)
        ax.add_patch(Rectangle((x, 69.2), 6.2, 12.8, facecolor="none",
                               edgecolor=INK2, linewidth=0.8, zorder=3 + i))
    ax.annotate("real context", (6.2, 66.9), ha="center", fontsize=6.7,
                color=INK2)

    _box(ax, 13.2, 68.0, 12.0, 16.0, "Image encoder\n+ projector",
         TINT_BLUE, NV_BLUE, fs=7.2, lw=1.1, ls=frozen_ls)
    _arrow(ax, 9.7, 75.8, 12.9, 75.8, color=NV_BLUE, lw=1.1)

    for j, lab in enumerate(("$z_{t-2}$", "$z_{t-1}$", "$z_t$")):
        _token(ax, 28.0 + 3.05 * j, 72.5, lab, "white", NV_BLUE,
               w=2.45, h=6.4, fs=6.0)
    _arrow(ax, 25.2, 76.0, 27.7, 76.0, color=NV_BLUE, lw=1.0)

    _box(ax, 40.0, 67.5, 13.5, 17.0, "Persistent\ncarrier",
         TINT_GREEN, NV_GREEN_DK, fs=8.0, lw=1.5)
    _arrow(ax, 36.7, 75.7, 39.7, 75.7, color=NV_GREEN_DK, lw=1.25)
    ax.plot([52.0, 52.0, 41.5], [84.6, 88.8, 88.8],
            color=NV_GREEN_DK, lw=0.95, zorder=4)
    _arrow(ax, 41.5, 88.8, 41.5, 84.7, color=NV_GREEN_DK, lw=0.95,
           zorder=4)
    ax.annotate("episode state", (46.7, 89.7), ha="center", fontsize=6.3,
                color=NV_GREEN_DK)

    for j, lab in enumerate(("$\\tilde z_{t-2}$", "$\\tilde z_{t-1}$",
                             "$\\tilde z_t$")):
        _token(ax, 56.5 + 3.05 * j, 72.5, lab, TINT_GREEN, NV_GREEN_DK,
               w=2.45, h=6.4, fs=5.7)
    _arrow(ax, 53.5, 75.7, 56.2, 75.7, color=NV_GREEN_DK, lw=1.25)
    ax.plot([56.3, 65.1], [81.0, 81.0], color=INK2, lw=0.75)
    ax.plot([56.3, 56.3], [80.2, 81.0], color=INK2, lw=0.75)
    ax.plot([65.1, 65.1], [80.2, 81.0], color=INK2, lw=0.75)
    ax.annotate("$H=3$", (60.7, 82.4), ha="center", fontsize=6.6,
                color=INK2)

    _box(ax, 68.0, 67.5, 11.5, 17.0,
         "Predictor\n+ output\nprojection", TINT_BLUE,
         NV_BLUE, fs=6.3, lw=1.1, ls=frozen_ls)
    _arrow(ax, 65.2, 75.7, 67.7, 75.7, color=NV_BLUE, lw=1.1)
    _token(ax, 82.2, 72.5, "$\\hat z_{t+1}$", "white", NV_BLUE,
           w=4.2, h=6.4, fs=6.5, lw=1.0)
    _arrow(ax, 79.5, 75.7, 81.9, 75.7, color=NV_BLUE, lw=1.0)
    _box(ax, 91.0, 69.5, 8.0, 12.0, "Next-latent\nloss", "#F8EFE8",
         NV_ORANGE, fs=6.7, lw=1.15)
    _arrow(ax, 86.4, 75.7, 90.7, 75.7, color=NV_ORANGE, lw=1.0)

    # The standardized raw 10-D action blocks fork before frozen embedding:
    # raw a enters the carrier, while encoded u conditions only the predictor.
    for j, lab in enumerate(("$a_{t-2}$", "$a_{t-1}$", "$a_t$")):
        _token(ax, 2.0 + 3.1 * j, 52.4, lab, "#F8EFE8", NV_ORANGE,
               w=2.5, h=5.0, fs=5.7, lw=0.85)
    ax.annotate("10-D action blocks", (6.1, 49.9), ha="center",
                va="top", fontsize=5.8, color=INK2)
    _box(ax, 13.2, 50.5, 12.0, 9.5, "Action encoder", TINT_BLUE,
         NV_BLUE, fs=7.1, lw=1.05, ls=frozen_ls)
    _arrow(ax, 10.6, 55.2, 12.9, 55.2, color=NV_ORANGE, lw=0.95)
    ax.scatter([11.2], [55.2], s=10, color=NV_ORANGE, zorder=5)
    ax.plot([11.2, 11.2, 37.5], [55.2, 63.0, 63.0],
            color=NV_ORANGE, lw=0.95, zorder=2)
    _arrow(ax, 37.5, 63.0, 40.0, 68.8, color=NV_ORANGE, lw=0.95)
    ax.annotate("raw $a$", (27.0, 64.0), ha="center", va="bottom",
                fontsize=6.1, color=NV_ORANGE, fontweight="bold")
    for j, lab in enumerate(("$u_{t-2}$", "$u_{t-1}$", "$u_t$")):
        _token(ax, 28.0 + 3.05 * j, 52.0, lab, "white", NV_BLUE,
               w=2.45, h=6.0, fs=5.8, lw=0.9)
    _arrow(ax, 25.2, 55.2, 27.7, 55.2, color=NV_BLUE, lw=0.95)
    ax.plot([36.7, 73.8], [55.2, 55.2], color=NV_BLUE, lw=0.9)
    _arrow(ax, 73.8, 55.2, 73.8, 67.2, color=NV_BLUE, lw=0.9)

    # The raw target is a cached output of the frozen official checkpoint.
    ax.imshow(frames[3], extent=(76.0, 81.7, 47.0, 59.0),
              interpolation="nearest", aspect="auto", zorder=2)
    ax.add_patch(Rectangle((76.0, 47.0), 5.7, 12.0, facecolor="none",
                           edgecolor=NV_ORANGE, linewidth=0.9, zorder=3))
    ax.annotate("next frame", (78.85, 45.7), ha="center", va="top",
                fontsize=5.9, color=INK2)
    _token(ax, 84.0, 50.1, "$z_{t+1}$", "white", NV_ORANGE,
           w=4.1, h=6.2, fs=6.2, lw=1.0)
    _arrow(ax, 81.7, 53.2, 83.7, 53.2, color=NV_ORANGE, lw=0.9, ls=frozen_ls)
    ax.annotate("frozen E+proj cache", (86.05, 57.4),
                ha="center", va="bottom", fontsize=5.45, color=NV_ORANGE)
    ax.plot([88.1, 95.0], [53.2, 53.2], color=NV_ORANGE, lw=0.9)
    _arrow(ax, 95.0, 53.2, 95.0, 69.2, color=NV_ORANGE, lw=0.9)

    ax.plot([1.0, 99.0], [41.5, 41.5], color=GRID, lw=0.8)

    ax.annotate("(b) Read before the current observation", (1.0, 39.7), ha="left",
                va="top", fontsize=8.5, fontweight="bold")
    _token(ax, 2.0, 21.2, "$m_{t-1}$", "white", NV_GREEN_DK,
           w=5.4, h=6.5, fs=6.8)
    _token(ax, 2.0, 11.7, "$a_{t-1}$", "#F8EFE8", NV_ORANGE,
           w=5.4, h=6.5, fs=6.5)
    _box(ax, 16.0, 16.2, 9.5, 8.5, "Predict", TINT_GREEN,
         NV_GREEN_DK, fs=7.7, lw=1.15)
    _arrow(ax, 7.4, 24.4, 15.7, 21.4, color=NV_GREEN_DK, lw=1.0)
    _arrow(ax, 7.4, 14.9, 15.7, 19.4, color=NV_ORANGE, lw=1.0)
    _token(ax, 29.0, 17.2, "$m_t^-$", "white", NV_GREEN_DK,
           w=5.5, h=6.5, fs=6.8)
    _arrow(ax, 25.5, 20.4, 28.7, 20.4, color=NV_GREEN_DK, lw=1.0)

    # The causal read branches from the prior before the observation update.
    ax.plot([34.5, 37.5], [20.4, 30.0], color=NV_GREEN_DK, lw=1.0)
    _box(ax, 37.5, 26.5, 10.5, 7.0, "Causal read", TINT_GREEN,
         NV_GREEN_DK, fs=7.3, lw=1.05)
    _token(ax, 50.5, 27.0, "$b_t$", "white", NV_GREEN_DK,
           w=4.2, h=6.0, fs=6.8)
    _arrow(ax, 48.0, 30.0, 50.2, 30.0, color=NV_GREEN_DK, lw=1.0)
    ax.annotate("legal retention tap", (56.2, 30.0), ha="left", va="center",
                fontsize=6.2, color=NV_GREEN_DK)

    _box(ax, 44.5, 16.2, 9.5, 8.5, "Correct", TINT_GREEN,
         NV_GREEN_DK, fs=7.7, lw=1.15)
    _arrow(ax, 34.5, 20.4, 44.2, 20.4, color=NV_GREEN_DK, lw=1.0)
    _token(ax, 57.5, 17.2, "$m_t$", "white", NV_GREEN_DK,
           w=5.5, h=6.5, fs=6.8)
    _arrow(ax, 54.0, 20.4, 57.2, 20.4, color=NV_GREEN_DK, lw=1.0)
    _box(ax, 72.0, 16.2, 9.0, 8.5, "Fuse", TINT_GREEN,
         NV_GREEN_DK, fs=7.7, lw=1.15)
    _arrow(ax, 63.0, 20.4, 71.7, 20.4, color=NV_GREEN_DK, lw=1.0)
    _token(ax, 85.0, 17.2, "$\\tilde z_t$", "white", NV_GREEN_DK,
           w=5.5, h=6.5, fs=6.8)
    _arrow(ax, 81.0, 20.4, 84.7, 20.4, color=NV_GREEN_DK, lw=1.0)

    _token(ax, 46.3, 5.9, "$z_t$", "white", NV_BLUE,
           w=5.5, h=6.0, fs=6.8)
    _arrow(ax, 49.05, 11.9, 49.05, 15.9, color=NV_BLUE, lw=1.0)
    ax.plot([51.8, 76.5], [8.9, 8.9], color=NV_BLUE, lw=0.9)
    _arrow(ax, 76.5, 8.9, 76.5, 15.9, color=NV_BLUE, lw=0.9)

    # One recurrence arc closes the state update for the next step.
    ax.plot([60.25, 60.25, 1.0, 1.0], [17.2, 3.0, 3.0, 24.4],
            color=NV_GREEN_DK, lw=0.95)
    _arrow(ax, 1.0, 24.4, 1.7, 24.4, color=NV_GREEN_DK, lw=0.95)
    ax.annotate("persistent across windows", (31.5, 3.8), ha="center",
                va="bottom",
                fontsize=6.3, color=NV_GREEN_DK)

    fig.subplots_adjust(left=0.008, right=0.995, bottom=0.008, top=0.995)
    save_figure_pair(fig, "fig_a_arch")
    plt.close(fig)


def _core_task_evidence() -> list[dict]:
    """Artifact-sourced task frames, frozen-encoder scores, and audit gates."""
    # Use the Paper-A expansion configuration as the sole gate authority.
    import yaml
    registration = yaml.safe_load(
        (ROOT / "configs/paper_a_expansion.yaml").read_text())
    categorical_gate = float(
        registration["availability_gate"]["categorical_accuracy_min"])
    continuous_gate = float(
        registration["availability_gate"]["continuous_r2_min"])

    specs = {
        "t1": {
            "name": "Transient-marker\nrecall",
            "phase_labels": ("marker cue", "marker absent", "pre-decision"),
        },
        "t3": {
            "name": "Drifting-color\nrecall",
            "phase_labels": ("color cue", "post-cue drift", "pre-decision"),
        },
        "t4": {
            "name": "Occluded-target\nprediction",
            "phase_labels": ("last pre-gap", "observation frozen",
                             "held-out outcome"),
        },
    }
    rows: list[dict] = []
    for task, spec in specs.items():
        availability = load(
            f"outputs/paper_a_expansion/cache/{task}/availability.json")
        path = (ROOT / "outputs/paper_a_expansion/data" / task /
                "val_clean_e240_s270702.npz")
        with np.load(path, allow_pickle=False) as bank:
            if task in ("t1", "t3"):
                cue_on = int(bank["event_cue_on"][0])
                cue_off = int(bank["event_cue_off"][0])
                times = (cue_on + 1, cue_off + 8, 62)
                gate = categorical_gate
                metric = "Acc."
            else:
                gap_on = int(bank["event_gap_on"][0])
                gap_off = int(bank["event_gap_off"][0])
                times = (gap_on - 1, (gap_on + gap_off) // 2, gap_off + 2)
                gate = continuous_gate
                metric = "$R^2$"
            frames = [bank["frames"][0, t].copy() for t in times]

        value = float(availability["value"])
        rows.append({**spec, "task": task, "frames": frames,
                     "times": times, "value": value, "gate": gate,
                     "metric": metric, "passed": value >= gate})
    return rows


def fig_protocol() -> None:
    """Core task frames and official-LeWM encoder-availability gates."""
    rows = _core_task_evidence()

    fig, ax = plt.subplots(figsize=(5.45, 3.15))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.annotate("Core memory tasks and representation availability", (1.0, 99.0),
                ha="left", va="top", fontsize=8.8, fontweight="bold")
    ax.annotate("representative validation episode", (23.5, 90.5),
                ha="left", va="center", fontsize=6.7, color=INK2)
    ax.annotate("frozen encoder + projector", (73.5, 90.5),
                ha="left", va="center", fontsize=6.7, color=INK2)

    row_y = (74.0, 46.0, 18.0)
    frame_x = (23.5, 40.0, 56.5)
    frame_w, frame_h = 9.8, 17.0
    border_colors = (NV_GREEN_DK, INK2, NV_BLUE)
    for row_index, (row, y) in enumerate(zip(rows, row_y)):
        if row_index:
            ax.plot([1.0, 99.0], [y + 14.0, y + 14.0], color=GRID,
                    linewidth=0.8, zorder=0)
        ax.annotate(row["name"], (1.5, y + 1.3), ha="left", va="center",
                    fontsize=7.5, fontweight="bold", linespacing=1.02)
        target_kind = "categorical cue" if row["metric"] == "Acc." else \
            "continuous target"
        ax.annotate(target_kind, (1.5, y - 7.0), ha="left", va="center",
                    fontsize=6.2, color=INK2)

        for j, (frame, t, phase, x0, edge) in enumerate(zip(
                row["frames"], row["times"], row["phase_labels"],
                frame_x, border_colors)):
            y0 = y - frame_h / 2
            ax.imshow(frame, extent=(x0, x0 + frame_w, y0, y0 + frame_h),
                      interpolation="nearest", aspect="auto", zorder=2)
            ax.add_patch(Rectangle((x0, y0), frame_w, frame_h,
                                   facecolor="none", edgecolor=edge,
                                   linewidth=1.05, zorder=3))
            phase_fs = 5.35 if len(phase) > 14 else 6.2
            ax.annotate(phase, (x0 + frame_w / 2, y + 10.4),
                        ha="center", va="center", fontsize=phase_fs,
                        color=edge, fontweight="bold")
            ax.annotate(f"$t={t}$", (x0 + frame_w / 2, y - 10.2),
                        ha="center", va="center", fontsize=5.9, color=INK2)
            if j < 2:
                _arrow(ax, x0 + frame_w + 0.5, y,
                       frame_x[j + 1] - 0.5, y, color=INK2, lw=0.8,
                       mutation_scale=6.5)

        value = float(row["value"])
        gate = float(row["gate"])
        passed = bool(row["passed"])
        status_color = NV_GREEN_DK if passed else NV_ORANGE
        status = "meets gate" if passed else "availability\nnot established"
        score = (f"{row['metric']} {value:.3f}" if row["metric"] == "Acc."
                 else f"{row['metric']}={value:.3f}")
        ax.annotate(score, (73.5, y + 5.1), ha="left", va="center",
                    fontsize=8.0, fontweight="bold", color=INK)
        ax.annotate(status, (98.5, y + 5.1), ha="right", va="center",
                    fontsize=6.6, fontweight="bold", color=status_color)

        gauge_x0, gauge_x1, gauge_y = 73.7, 98.2, y - 1.3
        gate_x = gauge_x0 + np.clip(gate, 0.0, 1.0) * (gauge_x1 - gauge_x0)
        value_x = gauge_x0 + np.clip(value, 0.0, 1.0) * (gauge_x1 - gauge_x0)
        ax.plot([gauge_x0, gauge_x1], [gauge_y, gauge_y], color=GRID,
                linewidth=3.0, solid_capstyle="round", zorder=1)
        ax.plot([gate_x, gauge_x1], [gauge_y, gauge_y], color=TINT_GREEN,
                linewidth=4.2, solid_capstyle="butt", zorder=1)
        ax.plot([gate_x, gate_x], [gauge_y - 2.2, gauge_y + 2.2],
                color=INK2, linewidth=0.85, linestyle=(0, (3, 2)), zorder=2)
        ax.scatter([value_x], [gauge_y], s=28, color=status_color,
                   edgecolor="white", linewidth=0.55, zorder=4)
        ax.annotate(f"gate $\\geq$ {gate:.3f}", (73.5, y - 7.0),
                    ha="left", va="center", fontsize=6.1, color=INK2)

    fig.subplots_adjust(left=0.008, right=0.995, bottom=0.008, top=0.995)
    save_figure_pair(fig, "fig_a_protocol")
    plt.close(fig)


def _health_counts(arm: str) -> tuple[int, int, int]:
    rows = []
    for task in ("t1", "t3", "t4"):
        for path in sorted((ROOT / f"outputs/v21_x1/{task}/{arm}").glob(
                "s*/gates.json")):
            rows.append(json.loads(path.read_text()))
    if not rows:
        raise RuntimeError(f"no X1 health receipts for {arm}")
    return (sum(bool(row["rank_pass"]) for row in rows),
            sum(bool(row["overall_pass"]) for row in rows), len(rows))


def _legacy_fig_evidence() -> None:
    """Primary effects, planning, and the training-health stopping rule."""
    gate = load("outputs/v21_x1/x1_gates.json")
    control = load("outputs/v21_x2/x2_results_v3.json")
    delta = load("outputs/v21_x2/x2_results_envelope.json")
    _, filter_joint, filter_total = _health_counts("lkc_rfix")
    _, delta_joint, delta_total = _health_counts("gdelta_l10")

    fig = plt.figure(figsize=(5.45, 3.60), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[4.1, 0.62, 1.65],
                          width_ratios=[1.02, 1.18], hspace=0.08, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    labels = ["T1", "T3", "T4$^\\dagger$", "Pooled"]
    tasks = ("t1", "t3", "t4")
    vals = [gate["per_task_d"][task] for task in tasks] + [gate["pooled_d"]]
    wins = [gate["per_task_wins"][task] for task in tasks] + ["3/3 tasks"]
    y = np.arange(4)[::-1]
    ax.axvline(0, color=INK2, lw=0.9, linestyle=(0, (2, 2)))
    ax.scatter(vals[:3], y[:3], s=34, color=NV_GREEN,
               edgecolor=NV_GREEN_DK, linewidth=0.8, zorder=3)
    ax.scatter(vals[2], y[2], s=56, facecolor="none", edgecolor=NV_ORANGE,
               linewidth=1.0, zorder=4)
    lo, hi = gate["ci95"]
    ax.errorbar(vals[3], y[3], xerr=[[vals[3] - lo], [hi - vals[3]]],
                fmt="D", color=NV_GREEN_DK, markerfacecolor=NV_GREEN,
                markersize=5.3, capsize=2.5, lw=1.4, zorder=4)
    for xv, yv, win in zip(vals, y, wins):
        offset = (5, 6) if yv == 0 else (5, 0)
        valign = "bottom" if yv == 0 else "center"
        ax.annotate(f"{xv:+.2f}  ({win})", (xv, yv), xytext=offset,
                    textcoords="offset points", ha="left", va=valign,
                    fontsize=6.0, color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "alpha": 1.0, "pad": 0.10})
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.18, 2.55)
    ax.set_ylim(-0.55, 3.55)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("paired effect $d$  (fixed trust $-$ delta)",
                  fontsize=6.8)
    ax.tick_params(labelsize=6.8)
    ax.grid(axis="x", color=GRID, lw=0.55)
    ax.set_axisbelow(True)
    ax.set_title("(a) Registered full-system effect", fontsize=8.2,
                 fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    rows = [
        ("Oracle $\\xi$", "oracle", INK2, 5.0),
        ("Initial/action", "floor_integrator", INK2, 4.0),
        ("Ours (fixed trust)", "rfix_argmax", NV_GREEN, 3.0),
        ("GRU", "acgru_argmax", "#555555", 2.0),
        ("Delta", None, NV_BLUE, 1.0),
        ("No carrier", "none_selector", INK2, 0.0),
    ]
    means, sds = [], []
    for _, key, _, _ in rows:
        if key is None:
            values = [v["gdelta_l10_argmax"]["success_rate"]
                      for v in delta["per_seed"].values()]
        else:
            values = [v[key]["success_rate"]
                      for v in control["per_seed"].values()]
        means.append(float(np.mean(values)))
        sds.append(float(np.std(values, ddof=1)))
    ax.axhspan(3.55, 5.45, color=TINT_GRAY, zorder=0)
    ax.axhspan(1.55, 3.45, color=TINT_GREEN, zorder=0)
    ax.axhspan(-0.45, 1.45, color=TINT_BLUE, alpha=0.65, zorder=0)
    baseline = means[1]
    ax.axvline(baseline, color=INK2, lw=0.9, linestyle=(0, (2, 2)))
    for (label, _, color, yv), mean, sd in zip(rows, means, sds):
        ax.errorbar(mean, yv, xerr=sd, fmt="o", color=color,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.5, markersize=5.4, capsize=2.2,
                    lw=1.15, zorder=3)
        ax.annotate(f"{mean:.3f}", (mean, yv), xytext=(5, 1),
                    textcoords="offset points", va="center", fontsize=6.2,
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "alpha": 0.82, "pad": 0.12})
    ax.annotate("references", (0.13, 5.35), fontsize=5.8, color=INK2,
                fontweight="bold")
    ax.annotate("registered pair", (0.13, 3.35), fontsize=5.8,
                color=NV_GREEN_DK, fontweight="bold")
    ax.annotate("exploratory", (0.13, 1.35), fontsize=5.8,
                color=NV_BLUE, fontweight="bold")
    ax.annotate("ours > GRU in 3/3 checkpoint pairs", (0.98, 0.46),
                xycoords="axes fraction", ha="right", fontsize=5.7,
                color=NV_GREEN_DK, fontweight="bold")
    ax.set_yticks([row[3] for row in rows], [row[0] for row in rows])
    ax.set_xlim(0.12, 1.02)
    ax.set_ylim(-0.45, 5.55)
    ax.set_xlabel("oracle-physics planning success", fontsize=6.8)
    ax.tick_params(labelsize=6.7)
    ax.grid(axis="x", color=GRID, lw=0.55)
    ax.set_axisbelow(True)
    ax.set_title("(b) Oracle-physics planning", fontsize=8.2,
                 fontweight="bold")

    # Dedicated, non-data legend rows.
    key_a = fig.add_subplot(gs[1, 0])
    key_a.set_xlim(0, 1)
    key_a.set_ylim(0, 1)
    key_a.axis("off")
    key_a.scatter([0.03], [0.72], s=28, color=NV_GREEN,
                  edgecolor=NV_GREEN_DK, linewidth=0.7)
    key_a.annotate("task effect", (0.08, 0.72), ha="left", va="center",
                   fontsize=6.5)
    key_a.plot([0.35, 0.47], [0.72, 0.72], color=NV_GREEN_DK, lw=1.2)
    key_a.scatter([0.41], [0.72], s=30, marker="D", color=NV_GREEN,
                  edgecolor=NV_GREEN_DK, linewidth=0.7)
    key_a.annotate("pooled $d$ + 95% CI", (0.50, 0.72), ha="left",
                   va="center", fontsize=6.5)
    key_a.scatter([0.03], [0.18], s=38, facecolor="none",
                  edgecolor=NV_ORANGE, linewidth=1.0)
    key_a.annotate("T4 audit exception", (0.08, 0.18), ha="left",
                   va="center", fontsize=6.5)

    key_b = fig.add_subplot(gs[1, 1])
    key_b.set_xlim(0, 1)
    key_b.set_ylim(0, 1)
    key_b.axis("off")
    key_b.scatter([0.04], [0.72], s=27, color=INK2,
                  edgecolor="white", linewidth=0.4)
    key_b.annotate("mean", (0.09, 0.72), va="center", fontsize=6.5)
    key_b.errorbar([0.43], [0.72], xerr=[0.07], fmt="none", color=INK2,
                   capsize=2.5, lw=1.0)
    key_b.annotate("$\\pm1$ SD", (0.52, 0.72), va="center", fontsize=6.5)
    key_b.plot([0.04, 0.04], [0.05, 0.40], color=INK2, lw=0.9,
               linestyle=(0, (3, 2)))
    key_b.annotate("initial/action reference", (0.09, 0.22), va="center",
                   fontsize=6.5)

    strip = fig.add_subplot(gs[2, :])
    strip.set_title("(c) Health gate and claim boundary", loc="left",
                    fontsize=8.2, fontweight="bold", pad=2)
    labels_health = ["Ours (fixed trust)", "Delta comparator"]
    pass_counts = [filter_joint, delta_joint]
    totals = [filter_total, delta_total]
    y_health = [1.0, 0.15]
    strip.barh(y_health, pass_counts, color=NV_GREEN, edgecolor=NV_GREEN_DK,
               height=0.46, linewidth=0.7, label="pass")
    strip.barh(y_health, [t - p for t, p in zip(totals, pass_counts)],
               left=pass_counts, color="#F1D9CB", edgecolor=NV_ORANGE,
               height=0.46, linewidth=0.7, label="fail")
    for yv, passed, total in zip(y_health, pass_counts, totals):
        label_x = passed / 2 if passed else 1.1
        strip.annotate(f"{passed}/{total} pass", (label_x, yv),
                       ha="center" if passed else "left", va="center",
                       fontsize=6.2, color=INK, fontweight="bold")
    strip.annotate(f"{filter_total - filter_joint} fail", (28.5, 1.0),
                   ha="center", va="center", fontsize=5.6,
                   color=NV_ORANGE, fontweight="bold")
    strip.annotate(f"{delta_total - delta_joint} fail", (15.0, 0.15),
                   ha="center", va="center", fontsize=5.6,
                   color=NV_ORANGE, fontweight="bold")
    strip.set_yticks(y_health, labels_health)
    strip.set_xlim(0, 50)
    strip.set_ylim(-0.28, 1.72)
    strip.set_xticks([0, 10, 20, 30])
    strip.set_xlabel("task × seed cells", fontsize=6.7)
    strip.tick_params(labelsize=6.5)
    strip.grid(axis="x", color=GRID, lw=0.5)
    strip.set_axisbelow(True)
    strip.spines[["left", "bottom"]].set_visible(False)
    strip.plot([33.0, 33.0], [-0.2, 1.55], color=GRID, lw=0.9)
    strip.add_patch(Rectangle((20.0, 1.48), 1.2, 0.18,
                              facecolor=NV_GREEN, edgecolor=NV_GREEN_DK,
                              linewidth=0.6))
    strip.annotate("pass", (21.7, 1.57), va="center", fontsize=6.5)
    strip.add_patch(Rectangle((25.5, 1.48), 1.2, 0.18,
                              facecolor="#F1D9CB", edgecolor=NV_ORANGE,
                              linewidth=0.6))
    strip.annotate("fail", (27.2, 1.57), va="center", fontsize=6.5)
    strip.annotate("ENDPOINT PASS\npositive full-system effect",
                   (48.5, 1.15), ha="right", va="center", fontsize=5.9,
                   color=NV_GREEN_DK, fontweight="bold",
                   bbox={"boxstyle": "round,pad=0.18", "facecolor": TINT_GREEN,
                         "edgecolor": NV_GREEN_DK, "linewidth": 0.7})
    strip.annotate("STOP\nintrinsic carrier superiority",
                   (48.5, 0.05), ha="right", va="center", fontsize=5.9,
                   color=NV_ORANGE, fontweight="bold",
                   bbox={"boxstyle": "round,pad=0.18", "facecolor": "#F8EFE8",
                         "edgecolor": NV_ORANGE, "linewidth": 0.7})

    save_figure_pair(fig, "fig_a_evidence")
    plt.close(fig)


def _salience_data() -> tuple[list[str], list[tuple[str, dict[str, list[float]], str]]]:
    rungs = ["t1s0c", "t1s0b", "t1s0a", "t1s1", "t1s2", "t1s3", "t1"]

    def dino_levels(rel: str) -> dict[str, list[float]]:
        return {lvl: row["scores"] for lvl, row in load(rel)["levels"].items()}

    dino_reacher = dino_levels("outputs/v21_x3/dino_sstar.json")
    dino_reacher.update(dino_levels("outputs/v21_x3/dino_sstar_ext.json"))
    dino_pm = dino_levels("outputs/v21_x3/dino_sstar_pointmass.json")
    w0 = load("outputs/v20_w0/w0_summary.json")["ladder_readout"]["vicreg"]
    vicreg_reacher = {lvl: row["sighted_scores"]
                      for lvl, row in w0["levels"].items()}
    vicreg_pm = {"t1s1": [
        load("outputs/v21_f2b/certificates/t1s1/vicreg/s0.json")["sighted"]["score"],
        load("outputs/v21_f2b/certificates/t1s1/vicreg/s1.json")["sighted"]["score"]]}
    rows = [
        ("VICReg / reacher", vicreg_reacher, "$s^*=S2$"),
        ("VICReg / point-mass", vicreg_pm, "S1 pass only"),
        ("DINOv2 / reacher", dino_reacher, "$s^*\\leq S0c$"),
        ("DINOv2 / point-mass", dino_pm, "$s^*\\leq S0c$"),
    ]
    return rungs, rows


def _panel_sstar_matrix(ax) -> None:
    rungs, rows = _salience_data()
    col_x = np.arange(len(rungs), dtype=float) * 1.13
    boundary_x = 8.88
    ax.set_xlim(-3.75, 9.65)
    ax.set_ylim(-0.28, 4.30)
    ax.axis("off")
    ax.annotate("(a) Representation salience audit", (-3.65, 4.26),
                ha="left", va="top", fontsize=8.2, fontweight="bold")
    for j, rung in enumerate(rungs):
        ax.annotate(rung.replace("t1s", "S").replace("t1", "T1").upper(),
                    (col_x[j], 3.78), ha="center", va="center", fontsize=6.15,
                    color=INK2, fontweight="bold")
    ax.annotate("result", (boundary_x, 3.78), ha="center", va="center",
                fontsize=6.0, color=INK2, fontweight="bold")
    label_family = ["Ours—VICReg", "Ours—VICReg", "DINOv2", "DINOv2"]
    label_scene = ["Reacher", "Point-mass", "Reacher", "Point-mass"]
    boundary = ["S2", "S1+", "≤S0c", "≤S0c"]
    for row_index, (label, levels, summary) in enumerate(rows):
        y = 3.05 - row_index
        is_ours = row_index < 2
        if is_ours:
            ax.add_patch(Rectangle((-3.60, y - 0.39), 11.40, 0.78,
                                   facecolor=TINT_GREEN, edgecolor="none",
                                   zorder=-2))
            ax.add_patch(Rectangle((-3.60, y - 0.39), 0.12, 0.78,
                                   facecolor=NV_GREEN, edgecolor="none",
                                   zorder=-1))
            label_color = NV_GREEN_DK
        else:
            label_color = NV_BLUE
        ax.annotate(label_family[row_index], (-0.62, y + 0.10), ha="right",
                    va="center", fontsize=6.15, color=label_color,
                    fontweight="bold")
        ax.annotate(label_scene[row_index], (-0.62, y - 0.14), ha="right",
                    va="center", fontsize=5.75, color=label_color)
        for col, rung in enumerate(rungs):
            x = col_x[col]
            if rung not in levels:
                rect = Rectangle((x - 0.43, y - 0.30), 0.86, 0.60,
                                 facecolor="white", edgecolor="#C8CDD2",
                                 linewidth=0.7, hatch="////")
                ax.add_patch(rect)
                continue
            values = [float(value) for value in levels[rung]]
            mean = float(np.mean(values))
            passes = sum(value >= 0.75 for value in values) > len(values) / 2
            edge = NV_GREEN_DK if passes else NV_ORANGE
            face = TINT_GREEN if passes else "#F8EFE8"
            ax.add_patch(Rectangle((x - 0.43, y - 0.30), 0.86, 0.60,
                                   facecolor=face, edgecolor=edge,
                                   linewidth=1.0))
            ax.annotate(f"{mean:.2f}", (x, y), ha="center", va="center",
                        fontsize=5.15, color=INK, fontweight="bold")
        ax.annotate(boundary[row_index], (boundary_x, y), ha="center",
                    va="center",
                    fontsize=6.4,
                    color=NV_ORANGE if row_index == 1 else NV_GREEN_DK,
                    fontweight="bold")
    ax.plot([7.82, 7.82], [-0.15, 3.55], color=GRID, lw=0.7)


def _panel_matrix_key(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        (0.02, TINT_GREEN, NV_GREEN_DK, None, "pass $\\geq .75$"),
        (0.35, "#F8EFE8", NV_ORANGE, None, "fail"),
        (0.52, "white", "#C8CDD2", "////", "not tested"),
    ]
    for x, face, edge, hatch, label in items:
        ax.add_patch(Rectangle((x, 0.38), 0.035, 0.28, facecolor=face,
                               edgecolor=edge, linewidth=0.8, hatch=hatch))
        ax.annotate(label, (x + 0.047, 0.52), ha="left", va="center",
                    fontsize=6.5, color=INK2)
    ax.annotate("cell number = bank mean", (0.76, 0.52), ha="left",
                va="center", fontsize=6.5, color=INK2)


def _panel_delay(ax) -> None:
    delay = load("outputs/v21_x3/delay_scaling.json")["curves"]
    tau_receipt = load("outputs/v21_x3/tau_rescale.json")
    tau = tau_receipt["curves"]
    lengths = (64, 96, 128)
    ax.axvspan(65.0, 142, color=TINT_GRAY, alpha=0.75, zorder=0)
    ax.axvline(64, color=INK2, linewidth=0.9, zorder=1)
    ax.annotate("trained to $L=64$", (64, 0.545), ha="left", va="top",
                fontsize=5.6, color=INK2)
    series = [
        ("lkc_rfix", "Ours (fixed trust)", NV_GREEN, "o", (5, 4)),
        ("gdelta_l10", "delta", NV_BLUE, "s", (5, 5)),
        ("acgru", "GRU", "#555555", "^", (5, -1)),
    ]
    for arm, label, color, marker, offset in series:
        means = [float(delay[f"{arm}@L{length}"]["mean"]) for length in lengths]
        sds = [float(delay[f"{arm}@L{length}"]["sd"]) for length in lengths]
        ax.errorbar(lengths, means, yerr=sds, color=color, marker=marker,
                    linewidth=1.5, markersize=5.0, capsize=2.0, elinewidth=0.9,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    rescaled = [float(tau[f"L{length}"]["rescaled_mean"]) for length in lengths]
    ax.plot(lengths, rescaled, color=NV_ORANGE, marker="o",
            linestyle=(0, (4, 3)), linewidth=1.1, markersize=4.6,
            markerfacecolor="white", markeredgecolor=NV_ORANGE,
            markeredgewidth=1.0, zorder=2)
    ax.axhline(0.25, color=INK2, linewidth=0.9, linestyle=(0, (2, 3)),
               zorder=1)
    ax.annotate("chance", (66, 0.253), fontsize=5.8, color=INK2)
    ax.set_xticks(lengths)
    ax.set_xlim(58, 143)
    ax.set_ylim(0.2, 0.55)
    ax.tick_params(labelsize=6.7)
    ax.set_xlabel("episode length $L$", fontsize=6.8)
    ax.set_ylabel("probe accuracy", fontsize=6.8)
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_title("(b) Delay extrapolation", fontsize=8.2,
                 fontweight="bold")


def _panel_delay_key(ax) -> None:
    ax.axis("off")
    handles = [
        Line2D([0], [0], color=NV_GREEN, marker="o", markersize=4.5,
               label="Ours (fixed trust)"),
        Line2D([0], [0], color=NV_BLUE, marker="s", markersize=4.5,
               label="Delta"),
        Line2D([0], [0], color="#555555", marker="^", markersize=4.5,
               label="GRU reference"),
        Line2D([0], [0], color=NV_ORANGE, marker="o",
               markerfacecolor="white", linestyle=(0, (4, 3)),
               markersize=4.3, label="Failed repair"),
        Line2D([0], [0], color=INK2, marker="|", markersize=8,
               label="Whiskers: $\\pm1$ SD"),
    ]
    ax.legend(handles=handles, loc="center", ncol=2, fontsize=6.5,
              frameon=False, handlelength=2.0, handletextpad=0.45,
              columnspacing=0.9, labelspacing=0.35)


def _legacy_fig_results() -> None:
    fig = plt.figure(figsize=(5.45, 3.10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1.0],
                          height_ratios=[4.0, 1.10, 0.55],
                          hspace=0.05, wspace=0.23)
    matrix = fig.add_subplot(gs[0, 0])
    delay = fig.add_subplot(gs[0, 1])
    matrix_key = fig.add_subplot(gs[1, 0])
    delay_key = fig.add_subplot(gs[1, 1])
    takeaway = fig.add_subplot(gs[2, 1])
    blank = fig.add_subplot(gs[2, 0])
    blank.axis("off")
    takeaway.axis("off")
    _panel_sstar_matrix(matrix)
    _panel_delay(delay)
    _panel_matrix_key(matrix_key)
    _panel_delay_key(delay_key)
    takeaway.annotate("All systems → chance; repair misses the +.05 bar.",
                      (0.5, 0.55), ha="center", va="center", fontsize=6.5,
                      color=NV_ORANGE, fontweight="bold")
    save_figure_pair(fig, "fig_a_results")
    plt.close(fig)


EXPANSION_SUMMARY = "outputs/paper_a_expansion/summary.json"
EXPANSION_TASKS = ("t1", "t3")
EXPANSION_HISTORIES = (3, 16, 32, 56)
EXPANSION_HORIZONS = (1, 2, 4, 8, 16)
EXPANSION_REFERENCES = ("gru", "lstm", "ssm")


def _require_complete_expansion(summary: dict) -> dict:
    """Reject any publication figure assembled from an incomplete grid."""
    completion = summary.get("completion", {})
    validation = summary.get("validation", {})
    checks = {
        "completion.complete": completion.get("complete") is True,
        "validation.grid_complete": validation.get("grid_complete") is True,
        "validation.all_discovered_cells_schema_and_provenance_valid":
            validation.get(
                "all_discovered_cells_schema_and_provenance_valid") is True,
        "validation.official_host_file_hash_matches_preregistration":
            validation.get(
                "official_host_file_hash_matches_preregistration") is True,
        "validation.frozen_host_unchanged_within_every_completed_cell":
            validation.get(
                "frozen_host_unchanged_within_every_completed_cell") is True,
        "validation.frozen_host_state_consistent_across_completed_cells":
            validation.get(
                "frozen_host_state_consistent_across_completed_cells") is True,
        "validation.parameter_matching_ledger_consistent_across_completed_cells":
            validation.get(
                "parameter_matching_ledger_consistent_across_completed_cells")
            is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        missing = completion.get("missing_count", "unknown")
        raise RuntimeError(
            "Paper-A expansion figures are publication-locked: "
            f"{missing} cells remain or validation failed "
            f"({', '.join(failed)}). Re-aggregate the complete grid first."
        )
    return summary


def _load_complete_expansion() -> dict:
    path = ROOT / EXPANSION_SUMMARY
    if not path.is_file():
        raise RuntimeError(f"missing expansion summary: {path}")
    return _require_complete_expansion(json.loads(path.read_text()))


def _task_style(summary: dict, task_id: str) -> dict:
    styles = {
        "t1": {"color": NV_BLUE, "marker": "o"},
        "t3": {"color": NV_ORANGE, "marker": "s"},
    }
    if task_id not in styles:
        raise RuntimeError(f"no publication style registered for {task_id}")
    task_names = summary.get("semantic_task_names", {})
    label = task_names.get(task_id)
    if not label:
        raise RuntimeError(f"missing semantic task name for {task_id}")
    return {**styles[task_id], "label": label}


def _finite_stat(stat: dict, *, field: str) -> tuple[np.ndarray, float, tuple[float, float]]:
    values = np.asarray(stat.get("values", []), dtype=float)
    mean = stat.get("mean")
    ci = stat.get("ci95", [])
    seeds = stat.get("seeds", [])
    if (values.size == 0 or len(seeds) != values.size or mean is None or
            len(ci) != 2 or ci[0] is None or ci[1] is None):
        raise RuntimeError(f"incomplete aggregate statistic: {field}")
    packed = np.r_[values, float(mean), float(ci[0]), float(ci[1])]
    if not np.all(np.isfinite(packed)):
        raise RuntimeError(f"non-finite aggregate statistic: {field}")
    return values, float(mean), (float(ci[0]), float(ci[1]))


def _seed_offsets(n: int, half_width: float = 0.045) -> np.ndarray:
    if n <= 1:
        return np.zeros(n, dtype=float)
    return np.linspace(-half_width, half_width, n)


def _mean_ci(ax, x: float, mean: float, ci: tuple[float, float], *,
             color: str, marker: str, zorder: int = 4) -> None:
    ax.errorbar(
        [x], [mean], yerr=[[mean - ci[0]], [ci[1] - mean]],
        fmt=marker, color=color, markerfacecolor="white",
        markeredgecolor=color, markeredgewidth=1.15, markersize=6.2,
        capsize=3.0, capthick=1.0, elinewidth=1.2, zorder=zorder,
    )


def _mean_ci_x(ax, y: float, mean: float, ci: tuple[float, float], *,
               color: str, marker: str, zorder: int = 5) -> None:
    ax.errorbar(
        [mean], [y], xerr=[[mean - ci[0]], [ci[1] - mean]],
        fmt=marker, color=color, markerfacecolor="white",
        markeredgecolor=color, markeredgewidth=1.15, markersize=6.2,
        capsize=3.0, capthick=1.0, elinewidth=1.2, zorder=zorder,
    )


def _common_figure_axes(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5, length=3.0, width=0.7)
    ax.xaxis.label.set_size(8.0)
    ax.yaxis.label.set_size(8.0)


def _cue_reachable(context: dict, *, task_id: str, history: int) -> bool:
    coverage = context["raw_legal_context_readout"]["validation_coverage"]
    key = "cue_any_frame_reachable"
    if key not in coverage:
        key = "cue_any_frame_reachable_from_context"
    if key not in coverage:
        raise RuntimeError(
            f"missing cue-coverage receipt for {task_id}, H={history}")
    return bool(coverage[key])


def _figure3_legend(summary: dict) -> list[Line2D]:
    handles: list[Line2D] = []
    for task_id in EXPANSION_TASKS:
        style = _task_style(summary, task_id)
        handles.append(Line2D(
            [0], [0], color=style["color"], marker=style["marker"],
            linewidth=1.8, markersize=5.7, label=style["label"],
        ))
    handles.extend([
        Line2D([0], [0], color=INK, marker="D", linestyle="none",
               markersize=5.2, markerfacecolor="white",
               label="equal-task contrast"),
        Line2D([0], [0], color=INK, linewidth=1.7, marker="o",
               label="trained predictor output"),
        Line2D([0], [0], color=INK, linewidth=1.5,
               linestyle=(0, (4, 2.2)), marker="o",
               markerfacecolor="white", label="raw legal context"),
        Line2D([0], [0], linestyle="none", marker="o", color=INK2,
               alpha=0.42, markersize=4.0, label="individual model seeds"),
        Line2D([0], [0], color=INK2, linewidth=1.0,
               linestyle=(0, (2, 2)), label="chance = .25"),
    ])
    return handles


def fig_evidence(summary: dict | None = None) -> None:
    """Frozen-host carrier isolation and the long-context control."""
    summary = _require_complete_expansion(
        summary if summary is not None else _load_complete_expansion())
    frozen = summary["frozen_carrier_swap"]["tasks"]
    context_tasks = summary["long_context"]["tasks"]

    fig, (carrier_ax, context_ax) = plt.subplots(
        2, 1, figsize=(7.0, 4.55), gridspec_kw={"height_ratios": [0.92, 1.08]},
    )

    # (a) Directly visualize the paper's paired estimands.  Task rows expose
    # heterogeneity; the diamond is the equal-task aggregate reported in text.
    pooled = summary["frozen_carrier_swap"]["pooled_equal_task_contrasts"]
    y_base = np.arange(len(EXPANSION_REFERENCES), dtype=float)[::-1]
    y_offsets = {"t1": 0.20, "pooled": 0.0, "t3": -0.20}
    reference_labels = {
        "gru": "GRU", "lstm": "LSTM", "ssm": "Diagonal SSM",
    }
    for row, reference in zip(y_base, EXPANSION_REFERENCES):
        task_values: dict[str, np.ndarray] = {}
        for task_id in EXPANSION_TASKS:
            style = _task_style(summary, task_id)
            stat = frozen[task_id]["paired_contrasts"][reference]
            values, mean, ci = _finite_stat(
                stat, field=f"frozen/{task_id}/fixed_trust-minus-{reference}")
            task_values[task_id] = values
            y = row + y_offsets[task_id]
            carrier_ax.scatter(
                values, y + _seed_offsets(values.size, half_width=0.035),
                s=16, marker=style["marker"], color=style["color"],
                alpha=0.28, linewidths=0, zorder=3,
            )
            _mean_ci_x(carrier_ax, y, mean, ci, color=style["color"],
                       marker=style["marker"])

        pooled_stat = pooled[reference]
        pooled_mean = float(pooled_stat["mean"])
        pooled_ci = tuple(float(value) for value in pooled_stat["ci95"])
        if not np.all(np.isfinite([pooled_mean, *pooled_ci])):
            raise RuntimeError(f"invalid pooled frozen contrast: {reference}")
        if task_values["t1"].shape != task_values["t3"].shape:
            raise RuntimeError(f"unpaired task contrasts: {reference}")
        pooled_seed_values = (task_values["t1"] + task_values["t3"]) / 2.0
        y = row + y_offsets["pooled"]
        carrier_ax.scatter(
            pooled_seed_values,
            y + _seed_offsets(pooled_seed_values.size, half_width=0.035),
            s=15, marker="D", color=INK, alpha=0.25, linewidths=0, zorder=3,
        )
        _mean_ci_x(carrier_ax, y, pooled_mean, pooled_ci,
                   color=INK, marker="D")

    carrier_ax.axvline(
        0.0, color=INK2, linewidth=1.0, linestyle=(0, (2, 2)), zorder=1)
    carrier_ax.set_yticks(
        y_base, [reference_labels[item] for item in EXPANSION_REFERENCES])
    carrier_ax.set_xlim(-0.058, 0.045)
    carrier_ax.set_ylim(-0.48, len(EXPANSION_REFERENCES) - 0.52)
    carrier_ax.set_xlabel(
        "Paired accuracy difference  (positive favors fixed-trust)")
    carrier_ax.set_ylabel("Comparator")
    carrier_ax.annotate(
        "favors comparator", (0.02, 0.94), xycoords="axes fraction",
        ha="left", va="top", fontsize=7.2, color=INK2,
    )
    carrier_ax.annotate(
        "favors fixed-trust", (0.98, 0.94), xycoords="axes fraction",
        ha="right", va="top", fontsize=7.2, color=INK2,
    )
    carrier_ax.set_title(
        "(a) Paired carrier contrasts reveal task heterogeneity",
        loc="left", fontsize=8.7, fontweight="bold", pad=5,
    )
    _common_figure_axes(carrier_ax)

    # (b) The raw readout asks whether legal context contains the cue; the
    # trained-predictor readout is a separate, seed-varying estimand.
    reachable_by_h: dict[int, set[bool]] = {
        history: set() for history in EXPANSION_HISTORIES}
    for task_id in EXPANSION_TASKS:
        style = _task_style(summary, task_id)
        task = context_tasks[task_id]
        raw_means: list[float] = []
        trained_means: list[float] = []
        trained_lows: list[float] = []
        trained_highs: list[float] = []
        for index, history in enumerate(EXPANSION_HISTORIES):
            row = task["contexts"][str(history)]
            reachable_by_h[history].add(
                _cue_reachable(row, task_id=task_id, history=history))
            raw = float(row["raw_legal_context_readout"]["value"])
            if not np.isfinite(raw):
                raise RuntimeError(
                    f"non-finite raw legal-context readout: {task_id}, H={history}")
            raw_means.append(raw)
            stat = row["trained_predictor_semantic_accuracy"]
            values, mean, ci = _finite_stat(
                stat, field=f"context/{task_id}/H={history}/accuracy")
            trained_means.append(mean)
            trained_lows.append(ci[0])
            trained_highs.append(ci[1])
            context_ax.scatter(
                history + _seed_offsets(values.size, half_width=0.55), values,
                s=17, marker=style["marker"], color=style["color"],
                alpha=0.28, linewidths=0, zorder=4,
            )
        histories = np.asarray(EXPANSION_HISTORIES, dtype=float)
        context_ax.plot(
            histories, raw_means, color=style["color"],
            marker=style["marker"], markerfacecolor="white",
            markeredgewidth=1.0, markersize=5.1, linewidth=1.5,
            linestyle=(0, (4, 2.2)), zorder=3,
        )
        context_ax.plot(
            histories, trained_means, color=style["color"],
            marker=style["marker"], markersize=5.3, linewidth=1.8,
            zorder=4,
        )
        context_ax.fill_between(
            histories, trained_lows, trained_highs,
            color=style["color"], alpha=0.10, linewidth=0, zorder=2,
        )

    expected_reachability = {3: False, 16: False, 32: False, 56: True}
    observed_reachability = {
        history: next(iter(states)) if len(states) == 1 else None
        for history, states in reachable_by_h.items()
    }
    if observed_reachability != expected_reachability:
        raise RuntimeError(
            "unexpected cue coverage in long-context summary: "
            f"{observed_reachability}")
    context_ax.axvspan(0, 44, color=TINT_GRAY, alpha=0.72, zorder=0)
    context_ax.axvspan(44, 60, color=TINT_GREEN, alpha=0.62, zorder=0)
    context_ax.axvline(44, color=INK2, linewidth=0.8,
                       linestyle=(0, (2, 2)), zorder=1)
    context_ax.annotate(
        "cue outside legal context (H ≤ 32)", (0.02, 0.96),
        xycoords="axes fraction", ha="left", va="top", fontsize=7.5,
        color=INK2,
    )
    context_ax.annotate(
        "cue reachable at H = 56", (0.97, 0.96), xycoords="axes fraction",
        ha="right", va="top", fontsize=7.5, color=NV_GREEN_DK,
    )
    context_ax.axhline(
        0.25, color=INK2, linewidth=1.0, linestyle=(0, (2, 2)), zorder=1)
    context_ax.set_xticks(EXPANSION_HISTORIES)
    context_ax.set_xlim(0, 60)
    context_ax.set_ylim(0.12, 1.02)
    context_ax.set_xlabel("Legal context length H")
    context_ax.set_ylabel("Decision accuracy")
    context_ax.set_title(
        "(b) Context length separates raw access from predictor exposure",
        loc="left", fontsize=8.7, fontweight="bold", pad=5,
    )
    _common_figure_axes(context_ax)

    fig.legend(
        handles=_figure3_legend(summary), loc="lower center", ncol=4,
        bbox_to_anchor=(0.50, 0.006), fontsize=7.2, frameon=False,
        handlelength=2.3, handletextpad=0.55, columnspacing=1.25,
        labelspacing=0.45,
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.965, bottom=0.225,
                        hspace=0.58)
    save_figure_pair(fig, "fig_a_evidence")
    plt.close(fig)


def _paired_ratio_values(numerator: dict, denominator: dict, *,
                         field: str) -> np.ndarray:
    numerator_values = dict(zip(numerator.get("seeds", []),
                                numerator.get("values", [])))
    denominator_values = dict(zip(denominator.get("seeds", []),
                                  denominator.get("values", [])))
    if set(numerator_values) != set(denominator_values) or not numerator_values:
        raise RuntimeError(f"unpaired rollout seeds: {field}")
    ratios = np.asarray([
        float(numerator_values[seed]) / float(denominator_values[seed])
        for seed in sorted(numerator_values)
    ], dtype=float)
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0):
        raise RuntimeError(f"invalid relative rollout error: {field}")
    return ratios


def _bootstrap_mean_ci(values: np.ndarray, summary: dict, *,
                       salt: str) -> tuple[float, tuple[float, float]]:
    analysis = summary["analysis"]
    draws = int(analysis["bootstrap_draws"])
    level = float(analysis["confidence_level"])
    base_seed = int(analysis["bootstrap_seed"])
    stable_salt = sum((index + 1) * ord(char)
                      for index, char in enumerate(salt))
    rng = np.random.default_rng(base_seed + stable_salt)
    samples = values[rng.integers(0, values.size,
                                  size=(draws, values.size))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return float(values.mean()), (float(low), float(high))


def _rollout_series(summary: dict, task_id: str, objective_id: str,
                    metric: str) -> list[dict]:
    objective = summary["learned_rollout"]["tasks"][task_id]["objectives"][
        objective_id]
    series = []
    for horizon in EXPANSION_HORIZONS:
        row = objective["horizons"][str(horizon)]
        if metric == "relative_latent_mse":
            values = _paired_ratio_values(
                row["normalized_latent_mse"],
                row["copy_last_normalized_mse"],
                field=f"rollout/{task_id}/{objective_id}/K={horizon}",
            )
            mean, ci = _bootstrap_mean_ci(
                values, summary,
                salt=f"{task_id}/{objective_id}/{metric}/{horizon}",
            )
        else:
            values, mean, ci = _finite_stat(
                row[metric],
                field=f"rollout/{task_id}/{objective_id}/K={horizon}/{metric}",
            )
        series.append({"horizon": horizon, "values": values,
                       "mean": mean, "ci": ci})
    return series


def _plot_rollout_series(ax, summary: dict, *, metric: str) -> list[float]:
    positions = np.arange(len(EXPANSION_HORIZONS), dtype=float)
    objective_styles = {
        "one_step": {"linestyle": "-", "offset": -0.055,
                     "markerfacecolor": None},
        "overshoot_8": {"linestyle": (0, (4, 2.2)), "offset": 0.055,
                        "markerfacecolor": "white"},
    }
    observed: list[float] = []
    for task_index, task_id in enumerate(EXPANSION_TASKS):
        style = _task_style(summary, task_id)
        for objective_id, objective_style in objective_styles.items():
            rows = _rollout_series(
                summary, task_id, objective_id, metric)
            means = np.asarray([row["mean"] for row in rows])
            lows = np.asarray([row["ci"][0] for row in rows])
            highs = np.asarray([row["ci"][1] for row in rows])
            observed.extend(
                value for row in rows for value in row["values"].tolist())
            observed.extend(lows.tolist() + highs.tolist())
            line_x = positions + objective_style["offset"]
            markerfacecolor = objective_style["markerfacecolor"]
            if markerfacecolor is None:
                markerfacecolor = style["color"]
            ax.fill_between(
                line_x, lows, highs, color=style["color"], alpha=0.075,
                linewidth=0, zorder=2,
            )
            ax.plot(
                line_x, means, color=style["color"],
                linestyle=objective_style["linestyle"], linewidth=1.75,
                marker=style["marker"], markersize=5.2,
                markerfacecolor=markerfacecolor,
                markeredgecolor=style["color"], markeredgewidth=0.9,
                zorder=4,
            )
            for index, row in enumerate(rows):
                seed_x = (line_x[index] +
                          _seed_offsets(len(row["values"]), half_width=0.025))
                ax.scatter(
                    seed_x, row["values"], s=13, color=style["color"],
                    marker=style["marker"], alpha=0.22, linewidths=0,
                    zorder=3,
                )
    return observed


def _rollout_legend(summary: dict) -> list:
    handles: list = []
    for task_id in EXPANSION_TASKS:
        style = _task_style(summary, task_id)
        handles.append(Line2D(
            [0], [0], color=style["color"], marker=style["marker"],
            linewidth=1.8, markersize=5.7, label=style["label"],
        ))
    handles.extend([
        Line2D([0], [0], color=INK, linewidth=1.8, linestyle="-",
               label="one-step objective"),
        Line2D([0], [0], color=INK, linewidth=1.6,
               linestyle=(0, (4, 2.2)), label="eight-step objective"),
        Line2D([0], [0], color=INK2, marker="o", linestyle="none",
               markersize=3.8, alpha=0.35, label="individual model seeds"),
        Patch(facecolor=INK2, edgecolor="none", alpha=0.12,
              label="95% bootstrap CI"),
        Line2D([0], [0], color=INK2, linewidth=1.0,
               linestyle=(0, (2, 2)),
               label="references: error ratio 1; advantage 0"),
        Patch(facecolor=TINT_GREEN, edgecolor="none",
              label="pre-specified gate: K ≤ 8"),
        Patch(facecolor=TINT_GRAY, edgecolor="none",
              label="diagnostic: K = 16"),
    ])
    return handles


def _rollout_regions(ax) -> None:
    ax.axvspan(-0.42, 3.48, color=TINT_GREEN, alpha=0.47, zorder=0)
    ax.axvspan(3.48, 4.42, color=TINT_GRAY, alpha=0.85, zorder=0)
    ax.axvline(3.48, color=INK2, linewidth=0.8,
               linestyle=(0, (2, 2)), zorder=1)
    ax.annotate(
        "pre-specified gate", (0.02, 0.93), xycoords="axes fraction",
        ha="left", va="top", fontsize=7.5, color=NV_GREEN_DK,
    )
    ax.annotate(
        "diagnostic", (0.98, 0.93), xycoords="axes fraction",
        ha="right", va="top", fontsize=7.5, color=INK2,
    )


def fig_results(summary: dict | None = None) -> None:
    """Learned-model rollout quality and action dependence."""
    summary = _require_complete_expansion(
        summary if summary is not None else _load_complete_expansion())
    fig, (error_ax, action_ax) = plt.subplots(
        2, 1, figsize=(7.0, 4.55), gridspec_kw={"height_ratios": [1, 1]},
    )

    error_values = _plot_rollout_series(
        error_ax, summary, metric="relative_latent_mse")
    _rollout_regions(error_ax)
    error_ax.axhline(
        1.0, color=INK2, linewidth=1.0, linestyle=(0, (2, 2)), zorder=1)
    error_ax.set_yscale("log")
    positive = np.asarray([value for value in error_values if value > 0])
    error_ax.set_ylim(max(0.008, positive.min() * 0.72),
                      max(1.28, positive.max() * 1.28))
    candidate_ticks = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    y_low, y_high = error_ax.get_ylim()
    ticks = [tick for tick in candidate_ticks if y_low <= tick <= y_high]
    error_ax.set_yticks(ticks, [f"{tick:g}" for tick in ticks])
    error_ax.set_xticks(np.arange(len(EXPANSION_HORIZONS)),
                        EXPANSION_HORIZONS)
    error_ax.set_xlim(-0.42, 4.42)
    error_ax.set_ylabel("Latent MSE / copy-last MSE")
    error_ax.set_title(
        "(a) Learned-rollout error relative to copy-last dynamics",
        loc="left", fontsize=8.7, fontweight="bold", pad=5,
    )
    _common_figure_axes(error_ax)

    action_values = _plot_rollout_series(
        action_ax, summary, metric="true_action_advantage")
    _rollout_regions(action_ax)
    action_ax.axhline(
        0.0, color=INK2, linewidth=1.0, linestyle=(0, (2, 2)), zorder=1)
    finite_action = np.asarray(action_values, dtype=float)
    span = max(0.1, finite_action.max() - min(0.0, finite_action.min()))
    action_ax.set_ylim(min(-0.04, finite_action.min() - 0.08 * span),
                       finite_action.max() + 0.18 * span)
    action_ax.set_xticks(np.arange(len(EXPANSION_HORIZONS)),
                         EXPANSION_HORIZONS)
    action_ax.set_xlim(-0.42, 4.42)
    action_ax.set_xlabel("Rollout horizon K")
    action_ax.set_ylabel("Normalized true-action advantage")
    action_ax.set_title(
        "(b) Action sensitivity across rollout horizons",
        loc="left", fontsize=8.7, fontweight="bold", pad=5,
    )
    _common_figure_axes(action_ax)

    fig.legend(
        handles=_rollout_legend(summary), loc="lower center", ncol=3,
        bbox_to_anchor=(0.50, 0.006), fontsize=7.5, frameon=False,
        handlelength=2.4, handletextpad=0.55, columnspacing=1.15,
        labelspacing=0.47,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.965, bottom=0.225,
                        hspace=0.43)
    save_figure_pair(fig, "fig_a_results")
    plt.close(fig)


def _paired_stat_arrays(left: dict, right: dict, *, field: str
                        ) -> tuple[np.ndarray, list[int]]:
    left_map = dict(zip(left.get("seeds", []), left.get("values", [])))
    right_map = dict(zip(right.get("seeds", []), right.get("values", [])))
    if not left_map or set(left_map) != set(right_map):
        raise RuntimeError(f"unpaired statistic values: {field}")
    seeds = sorted(left_map)
    values = np.asarray(
        [float(left_map[seed]) - float(right_map[seed]) for seed in seeds],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"non-finite paired statistic values: {field}")
    return values, seeds


def fig_appendix_probe(summary: dict | None = None) -> None:
    """Exploratory temporal aggregation versus the final causal endpoint."""
    summary = _require_complete_expansion(
        summary if summary is not None else _load_complete_expansion())
    tasks = summary["frozen_carrier_swap"]["tasks"]
    arms = ("none", "gru", "lstm", "ssm", "fixed_trust")
    arm_labels = {
        "none": "No carrier", "gru": "GRU", "lstm": "LSTM",
        "ssm": "Diagonal SSM", "fixed_trust": "Fixed-trust",
    }
    markers = {
        "none": "x", "gru": "o", "lstm": "s", "ssm": "^",
        "fixed_trust": "D",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), sharex=True,
                             sharey=True)
    all_values: list[float] = []
    for ax, task_id in zip(axes, EXPANSION_TASKS):
        style = _task_style(summary, task_id)
        y_positions = np.arange(len(arms), dtype=float)[::-1]
        for y, arm in zip(y_positions, arms):
            item = tasks[task_id]["arms"][arm]
            differences, _ = _paired_stat_arrays(
                item["trajectory_accuracy"], item["accuracy"],
                field=f"appendix-probe/{task_id}/{arm}",
            )
            mean, ci = _bootstrap_mean_ci(
                differences, summary,
                salt=f"appendix-probe/{task_id}/{arm}",
            )
            all_values.extend(differences.tolist() + [ci[0], ci[1]])
            if arm == "none":
                ax.scatter([mean], [y], marker=markers[arm], s=36,
                           color=INK2, linewidths=1.2, zorder=5)
            else:
                ax.scatter(
                    differences,
                    y + _seed_offsets(len(differences), half_width=0.055),
                    marker=markers[arm], s=17, color=style["color"],
                    alpha=0.28, linewidths=0, zorder=3,
                )
                _mean_ci_x(ax, y, mean, ci, color=style["color"],
                           marker=markers[arm])
        ax.axvline(0.0, color=INK2, linewidth=1.0,
                   linestyle=(0, (2, 2)), zorder=1)
        ax.set_yticks(y_positions, [arm_labels[arm] for arm in arms])
        ax.set_title(style["label"], loc="left", fontsize=8.7,
                     fontweight="bold", pad=5)
        ax.set_xlabel("Trajectory diagnostic − final accuracy")
        _common_figure_axes(ax)
    axes[0].set_ylabel("Carrier")
    low = min(-0.025, min(all_values) - 0.02)
    high = max(0.25, max(all_values) + 0.02)
    for ax in axes:
        ax.set_xlim(low, high)
    handles = [
        Line2D([0], [0], color=INK2, marker="o", linestyle="none",
               markersize=4.0, alpha=0.35, label="individual paired seeds"),
        Line2D([0], [0], color=INK, marker="o", markerfacecolor="white",
               linewidth=1.2, markersize=5.5, label="mean and 95% CI"),
        Line2D([0], [0], color=INK2, marker="x", linestyle="none",
               markersize=5.0, label="deterministic no-carrier"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.005), fontsize=7.3, frameon=False,
               columnspacing=1.25, handletextpad=0.5)
    fig.suptitle(
        "Temporal aggregation yields additional linearly decodable signal",
        x=0.085, y=0.99, ha="left", fontsize=9.2, fontweight="bold",
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.83, bottom=0.23,
                        wspace=0.10)
    save_figure_pair(fig, "fig_a_appendix_probe")
    plt.close(fig)


def fig_appendix_context(summary: dict | None = None) -> None:
    """Context prediction loss versus delayed semantic exposure."""
    summary = _require_complete_expansion(
        summary if summary is not None else _load_complete_expansion())
    tasks = summary["long_context"]["tasks"]
    markers = {3: "o", 16: "s", 32: "^", 56: "D"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), sharey=True)
    for ax, task_id in zip(axes, EXPANSION_TASKS):
        style = _task_style(summary, task_id)
        task = tasks[task_id]
        by_seed: dict[int, tuple[list[float], list[float]]] = {
            seed: ([], []) for seed in (0, 1, 2)
        }
        means_x: list[float] = []
        means_y: list[float] = []
        for history in EXPANSION_HISTORIES:
            row = task["contexts"][str(history)]
            mse = row["validation_next_latent_mse"]
            accuracy = row["trained_predictor_semantic_accuracy"]
            if mse["seeds"] != accuracy["seeds"]:
                raise RuntimeError(
                    f"context appendix seeds are unpaired: {task_id}/H={history}")
            for seed, x_value, y_value in zip(
                    mse["seeds"], mse["values"], accuracy["values"],
                    strict=True):
                by_seed[int(seed)][0].append(float(x_value))
                by_seed[int(seed)][1].append(float(y_value))
            x_mean = float(mse["mean"])
            y_mean = float(accuracy["mean"])
            means_x.append(x_mean)
            means_y.append(y_mean)
            x_ci = [float(value) for value in mse["ci95"]]
            y_ci = [float(value) for value in accuracy["ci95"]]
            ax.errorbar(
                [x_mean], [y_mean],
                xerr=[[x_mean - x_ci[0]], [x_ci[1] - x_mean]],
                yerr=[[y_mean - y_ci[0]], [y_ci[1] - y_mean]],
                fmt=markers[history], color=(NV_GREEN_DK if history == 56
                                             else style["color"]),
                markerfacecolor="white", markeredgewidth=1.0,
                markersize=5.7, capsize=2.5, elinewidth=1.0, zorder=5,
            )
            if task_id == "t1":
                offsets = {3: (5, 5), 16: (5, -12),
                           32: (4, -13), 56: (8, 7)}
            else:
                offsets = {3: (5, 5), 16: (5, -12),
                           32: (5, 7), 56: (8, 7)}
            ax.annotate(
                f"H={history}", (x_mean, y_mean), xytext=offsets[history],
                textcoords="offset points", fontsize=7.1,
                color=(NV_GREEN_DK if history == 56 else INK2),
            )
        for seed, (x_values, y_values) in by_seed.items():
            ax.plot(x_values, y_values, color=style["color"], alpha=0.16,
                    linewidth=0.8, zorder=2)
        ax.plot(means_x, means_y, color=style["color"], linewidth=1.3,
                alpha=0.75, zorder=3)
        ax.axhline(0.25, color=INK2, linewidth=1.0,
                   linestyle=(0, (2, 2)), zorder=1)
        raw_h56 = float(task["contexts"]["56"][
            "raw_legal_context_readout"]["value"])
        ax.annotate(
            f"H=56 raw-context readout = {raw_h56:.3f}",
            (0.97, 0.94), xycoords="axes fraction", ha="right", va="top",
            fontsize=7.1, color=NV_GREEN_DK,
        )
        if task_id == "t1":
            ticks = (0.00, 0.02, 0.04, 0.06)
            labels = ("0.00", "0.02", "0.04", "0.06")
            ax.set_xlim(0.0, 0.06)
        else:
            ticks = (0.05, 0.06, 0.07, 0.08, 0.09)
            labels = ("0.05", "0.06", "0.07", "0.08", "0.09")
            ax.set_xlim(0.05, 0.09)
        ax.set_xticks(ticks, labels)
        ax.minorticks_off()
        ax.set_xlabel("Validation next-latent MSE")
        ax.set_title(style["label"], loc="left", fontsize=8.7,
                     fontweight="bold", pad=5)
        _common_figure_axes(ax)
    axes[0].set_ylabel("Predictor semantic accuracy")
    axes[0].set_ylim(0.16, 0.29)
    handles = [
        Line2D([0], [0], color=INK2, linewidth=0.9, alpha=0.35,
               label="individual model-seed path"),
        Line2D([0], [0], color=INK, marker="o", markerfacecolor="white",
               linewidth=1.2, markersize=5.4, label="mean and 95% intervals"),
        Line2D([0], [0], color=NV_GREEN_DK, marker="D", linestyle="none",
               markerfacecolor="white", label="H=56: cue reachable"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.005), fontsize=7.7, frameon=False,
               columnspacing=1.25, handletextpad=0.5)
    fig.suptitle(
        "Lower local prediction error does not imply delayed semantic exposure",
        x=0.085, y=0.99, ha="left", fontsize=9.2, fontweight="bold",
    )
    fig.subplots_adjust(left=0.11, right=0.985, top=0.83, bottom=0.23,
                        wspace=0.16)
    save_figure_pair(fig, "fig_a_appendix_context")
    plt.close(fig)


def _rank_ratio_series(summary: dict, task_id: str, objective_id: str
                       ) -> list[dict]:
    objective = summary["learned_rollout"]["tasks"][task_id]["objectives"][
        objective_id]
    rows: list[dict] = []
    for horizon in EXPANSION_HORIZONS:
        metrics = objective["horizons"][str(horizon)]
        values = _paired_ratio_values(
            metrics["predicted_effective_rank"],
            metrics["target_effective_rank"],
            field=f"rank-ratio/{task_id}/{objective_id}/K={horizon}",
        )
        mean, ci = _bootstrap_mean_ci(
            values, summary,
            salt=f"rank-ratio/{task_id}/{objective_id}/K={horizon}",
        )
        rows.append({"values": values, "mean": mean, "ci": ci})
    return rows


def fig_appendix_rollout(summary: dict | None = None) -> None:
    """Secondary physical-readout and rank-collapse rollout diagnostics."""
    summary = _require_complete_expansion(
        summary if summary is not None else _load_complete_expansion())
    fig, (pose_ax, rank_ax) = plt.subplots(
        2, 1, figsize=(7.0, 4.55), gridspec_kw={"height_ratios": [1, 1]},
    )
    objective_styles = {
        "one_step": {"linestyle": "-", "offset": -0.055,
                     "markerfacecolor": None},
        "overshoot_8": {"linestyle": (0, (4, 2.2)), "offset": 0.055,
                        "markerfacecolor": "white"},
    }
    positions = np.arange(len(EXPANSION_HORIZONS), dtype=float)
    for task_id in EXPANSION_TASKS:
        style = _task_style(summary, task_id)
        for objective_id, objective_style in objective_styles.items():
            pose_rows = _rollout_series(
                summary, task_id, objective_id, "pose_angular_mae")
            rank_rows = _rank_ratio_series(summary, task_id, objective_id)
            for ax, rows in ((pose_ax, pose_rows), (rank_ax, rank_rows)):
                means = np.asarray([row["mean"] for row in rows])
                lows = np.asarray([row["ci"][0] for row in rows])
                highs = np.asarray([row["ci"][1] for row in rows])
                line_x = positions + objective_style["offset"]
                face = objective_style["markerfacecolor"] or style["color"]
                ax.fill_between(line_x, lows, highs, color=style["color"],
                                alpha=0.075, linewidth=0, zorder=2)
                ax.plot(line_x, means, color=style["color"],
                        linestyle=objective_style["linestyle"], linewidth=1.75,
                        marker=style["marker"], markersize=5.2,
                        markerfacecolor=face, markeredgecolor=style["color"],
                        markeredgewidth=0.9, zorder=4)
                for index, row in enumerate(rows):
                    seed_x = (line_x[index] + _seed_offsets(
                        len(row["values"]), half_width=0.025))
                    ax.scatter(seed_x, row["values"], s=13,
                               color=style["color"], marker=style["marker"],
                               alpha=0.22, linewidths=0, zorder=3)
    for ax in (pose_ax, rank_ax):
        _rollout_regions(ax)
        ax.set_xticks(positions, EXPANSION_HORIZONS)
        ax.set_xlim(-0.42, 4.42)
        _common_figure_axes(ax)
    pose_ax.set_ylabel("Pose angular MAE (rad)")
    pose_ax.set_title(
        "(a) Linear pose readout across rollout horizons", loc="left",
        fontsize=8.7, fontweight="bold", pad=5,
    )
    rank_ax.axhline(1.0, color=INK2, linewidth=1.0,
                    linestyle=(0, (2, 2)), zorder=1)
    rank_ax.set_ylim(0.94, 1.06)
    rank_ax.set_ylabel("Predicted / target effective rank\n(zoomed scale)")
    rank_ax.set_xlabel("Rollout horizon K")
    rank_ax.set_title(
        "(b) Effective-rank ratio shows no obvious global rank collapse",
        loc="left", fontsize=8.7, fontweight="bold", pad=5,
    )
    handles = _rollout_legend(summary)[:6]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.006), fontsize=7.3, frameon=False,
               handlelength=2.3, columnspacing=1.2, handletextpad=0.5)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.965, bottom=0.20,
                        hspace=0.43)
    save_figure_pair(fig, "fig_a_appendix_rollout")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = _load_complete_expansion()
    fig_protocol()
    fig_arch()
    fig_evidence(summary)
    fig_results(summary)
    fig_appendix_probe(summary)
    fig_appendix_context(summary)
    fig_appendix_rollout(summary)
    print(f"[plot-a] wrote fig_a_protocol + fig_a_arch + fig_a_evidence + "
          f"fig_a_results + appendix diagnostics "
          f"under {FIG}")


if __name__ == "__main__":
    main()
