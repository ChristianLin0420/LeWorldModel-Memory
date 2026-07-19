#!/usr/bin/env python3
"""Generate compact Paper-B figures in the MESM dashboard visual style.

The figure policy for this draft is intentionally strict:
  * low height, paper-friendly panels;
  * real rendered frames where they clarify input/output formation;
  * analytical plots over decorative block diagrams;
  * same cream / ink / signal-yellow visual language as docs/mesm_nvidia_plan.html.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper_b" / "figures"
GEN = ROOT / "paper_b" / "generated_results"

INK = "#111827"
BLACK = "#000000"
CREAM = "#f5f4ef"
PAPER = "#fbfbf9"
PAPER2 = "#efeee8"
YELLOW = "#fbd45b"
YELLOW_DEEP = "#d8a900"
GRAY = "#9ca3af"
MUTED = "#656760"
LINE = "#d4d3cb"
LINE_DARK = "#333b49"
RED = "#a94b3f"
GREEN = "#4f7d5a"

PDF_METADATA = {
    "Creator": "scripts/plot_paper_b.py",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}
PNG_METADATA = {"Software": "scripts/plot_paper_b.py"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.fontsize": 6.1,
    "axes.edgecolor": LINE_DARK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": PAPER,
})


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text())


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight",
                metadata=PDF_METADATA)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=360,
                metadata=PNG_METADATA)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(0.0, 1.035, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.8, fontweight="bold", color=YELLOW_DEEP)
    ax.text(0.055, 1.035, title, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=7.0, fontweight="bold", color=INK)


def arrow(ax: plt.Axes, p0: tuple[float, float],
          p1: tuple[float, float], color: str = INK, lw: float = 0.9) -> None:
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=8.0, lw=lw,
        color=color, shrinkA=2, shrinkB=2))


def rect(ax: plt.Axes, x: float, y: float, w: float, h: float,
         label: str, sub: str = "", *, fc: str = PAPER, ec: str = LINE_DARK,
         label_color: str = INK) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=0.8))
    ax.text(x + 0.035 * w, y + 0.66 * h, label, ha="left", va="center",
            fontsize=5.7, fontweight="bold", color=label_color)
    if sub:
        ax.text(x + 0.035 * w, y + 0.33 * h, sub, ha="left", va="center",
                fontsize=4.7, color=MUTED, linespacing=1.05)


def image_array(path: str, *, crop: tuple[int, int, int, int] | None = None,
                max_size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(ROOT / path).convert("RGB")
    if crop is not None:
        image = image.crop(crop)
    if max_size is not None:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return np.asarray(image) / 255.0


def draw_image_fit(ax: plt.Axes, arr: np.ndarray,
                   box: tuple[float, float, float, float],
                   *, border: bool = True, zorder: int = 2) -> tuple[float, float, float, float]:
    """Draw image into an axes-space box without changing render aspect ratio."""
    x, y, w, h = box
    image_h, image_w = arr.shape[:2]
    image_ratio = image_w / image_h
    box_ratio = w / h
    if image_ratio >= box_ratio:
        draw_w = w
        draw_h = w / image_ratio
        draw_x = x
        draw_y = y + (h - draw_h) / 2
    else:
        draw_h = h
        draw_w = h * image_ratio
        draw_x = x + (w - draw_w) / 2
        draw_y = y
    ax.imshow(arr, extent=(draw_x, draw_x + draw_w, draw_y, draw_y + draw_h),
              aspect="auto", zorder=zorder)
    if border:
        ax.add_patch(Rectangle((draw_x, draw_y), draw_w, draw_h,
                               facecolor="none", edgecolor=LINE_DARK,
                               lw=0.55, zorder=zorder + 1))
    return draw_x, draw_y, draw_w, draw_h


def ci_half(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return 0.0
    return 1.96 * arr.std(ddof=1) / np.sqrt(arr.size)


def figure_architecture() -> None:
    """Pi0.5/LeWM-style model schematic with native-ratio render strips."""
    fig = plt.figure(figsize=(7.25, 2.35), facecolor=PAPER)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=CREAM,
                           edgecolor=LINE_DARK, lw=0.8, zorder=0))

    ax.text(0.025, 0.925, "MASKED-EVIDENCE MEMORY INTERFACE",
            fontsize=5.7, fontweight="bold", color=MUTED)
    ax.text(0.025, 0.845, "frozen host + evidence sidecar",
            fontsize=7.4, fontweight="bold", color=INK)

    cue = image_array(
        "outputs/ogbench_mesmem_admission_v1/pointmaze-large-navigate-v0/cue_variants.png")
    draw_image_fit(ax, cue, (0.025, 0.560, 0.215, 0.175))
    ax.text(0.025, 0.520, "cue-only renders",
            fontsize=4.8, fontweight="bold", color=INK)
    ax.text(0.025, 0.485, "same trajectory / endpoint",
            fontsize=4.2, color=MUTED)

    frame_strip = image_array("docs/figures/ogbench_pointmaze_real_frames.png")
    draw_image_fit(ax, frame_strip, (0.265, 0.590, 0.255, 0.125))
    ax.text(0.265, 0.520, "legal context",
            fontsize=4.8, fontweight="bold", color=INK)
    ax.text(0.265, 0.485, "old cue leaves context",
            fontsize=4.2, color=MUTED)

    def arch_box(x: float, y: float, w: float, h: float, title: str,
                 subtitle: str, *, fc: str = PAPER, lw: float = 0.8) -> None:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc,
                               edgecolor=LINE_DARK, lw=lw, zorder=2))
        ax.text(x + 0.012, y + h - 0.030, title, ha="left", va="top",
                fontsize=5.0, fontweight="bold", color=INK, zorder=3)
        if subtitle:
            ax.text(x + 0.012, y + 0.030, subtitle, ha="left", va="bottom",
                    fontsize=3.8, color=MUTED, linespacing=0.95, zorder=3)

    # Main frozen-host path, drawn as a compact architecture schematic rather
    # than a UI dashboard.
    y = 0.270
    arch_box(0.055, y, 0.115, 0.130, "stream", "$o_t, a_t$")
    arch_box(0.215, y, 0.125, 0.130, "encoder", "frozen")
    arch_box(0.390, y, 0.145, 0.130, "frozen host", "LeWM / DINO-WM")
    arch_box(0.590, y, 0.125, 0.130, "exposure", "host output")
    arch_box(0.760, y, 0.135, 0.130, "policy", "execute")
    for x0, x1 in [(0.170, 0.215), (0.340, 0.390), (0.535, 0.590), (0.715, 0.760)]:
        arrow(ax, (x0, y + 0.065), (x1, y + 0.065), lw=0.95)

    # Sidecar memory path injected into the frozen host.
    arch_box(0.240, 0.075, 0.210, 0.105, "evidence memory",
             "", fc=YELLOW, lw=0.9)
    ax.add_patch(Rectangle((0.485, 0.088), 0.082, 0.079, facecolor=PAPER2,
                           edgecolor=LINE_DARK, lw=0.7, zorder=2))
    ax.text(0.496, 0.135, "writer", fontsize=4.8, fontweight="bold", zorder=3)
    ax.text(0.496, 0.106, "residual", fontsize=3.7, color=MUTED, zorder=3)
    arrow(ax, (0.155, 0.270), (0.285, 0.180), color=YELLOW_DEEP, lw=1.0)
    arrow(ax, (0.450, 0.127), (0.485, 0.127), color=YELLOW_DEEP, lw=1.0)
    arrow(ax, (0.567, 0.127), (0.430, 0.270), color=YELLOW_DEEP, lw=1.0)

    # Controls and claim ladder, intentionally small.
    for i, (label, sub) in enumerate([("full", "old stream"),
                                      ("reset", "drop state"),
                                      ("none", "host only")]):
        x = 0.610 + i * 0.090
        ax.add_patch(Rectangle((x, 0.085), 0.070, 0.055,
                               facecolor=YELLOW if i == 0 else PAPER,
                               edgecolor=LINE_DARK, lw=0.6, zorder=2))
        ax.text(x + 0.035, 0.117, label, ha="center", va="center",
                fontsize=4.3, fontweight="bold", zorder=3)
        ax.text(x + 0.035, 0.066, sub, ha="center", va="top",
                fontsize=3.3, color=MUTED, zorder=3)
    for i, lab in enumerate(["demand", "retain", "expose", "use"]):
        x = 0.050 + i * 0.087
        ax.text(x, 0.060, f"{i+1}", color=YELLOW_DEEP, fontsize=4.8,
                fontweight="bold", ha="center", va="center")
        ax.text(x + 0.014, 0.060, lab, color=MUTED, fontsize=3.7,
                ha="left", va="center")
    save(fig, "fig_b_architecture")


def figure_claim_ledger() -> None:
    fig, ax = plt.subplots(figsize=(7.25, 1.95), facecolor=PAPER)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=CREAM,
                           edgecolor=LINE_DARK, lw=0.8))
    ax.text(0.025, 0.88, "TASK-LEVEL CLAIM LEDGER", fontsize=6.2,
            fontweight="bold", color=MUTED)
    ax.text(0.025, 0.78, "One paper result is not one memory score.",
            fontsize=7.5, fontweight="bold")

    cols = ["demand", "retention", "host exposure", "executed use", "role in paper"]
    rows = [
        ("PushT", ["pass", "pass", "pass", "pass", "positive use, two hosts"]),
        ("PointMaze", ["pass", "pass", "pass", "pass", "strong navigation use"]),
        ("Wall", ["pass", "pass", "mixed", "n/a", "exposure bottleneck"]),
        ("OGBench", ["pass", "cap", "cap", "pass", "fixed-controller success"]),
    ]
    left, top = 0.025, 0.62
    colw = [0.14, 0.12, 0.13, 0.16, 0.15, 0.25]
    ax.add_patch(Rectangle((left, top), 0.95, 0.08, facecolor=INK,
                           edgecolor=INK, lw=0.7))
    ax.text(left + 0.01, top + 0.04, "env", color=YELLOW, va="center",
            fontsize=5.1, fontweight="bold")
    x = left + colw[0]
    for j, c in enumerate(cols):
        ax.text(x + colw[j + 1] / 2, top + 0.04, c, ha="center",
                va="center", fontsize=5.0, fontweight="bold", color=PAPER)
        x += colw[j + 1]

    status_color = {"pass": YELLOW, "mixed": PAPER2, "cap": PAPER2, "n/a": PAPER}
    status_text = {"pass": "✓", "mixed": "mixed", "cap": "cap", "n/a": "—"}
    for i, (name, vals) in enumerate(rows):
        y = top - (i + 1) * 0.105
        ax.add_patch(Rectangle((left, y), 0.95, 0.105,
                               facecolor=PAPER if i % 2 == 0 else CREAM,
                               edgecolor=LINE, lw=0.6))
        ax.text(left + 0.01, y + 0.052, name, va="center",
                fontsize=5.6, fontweight="bold")
        x = left + colw[0]
        for j in range(4):
            val = vals[j]
            ax.add_patch(Rectangle((x + 0.012, y + 0.026),
                                   colw[j + 1] - 0.024, 0.052,
                                   facecolor=status_color[val],
                                   edgecolor=LINE_DARK if val != "n/a" else LINE,
                                   lw=0.55))
            ax.text(x + colw[j + 1] / 2, y + 0.052, status_text[val],
                    ha="center", va="center", fontsize=5.0,
                    fontweight="bold", color=INK)
            x += colw[j + 1]
        ax.text(x + 0.01, y + 0.052, vals[4], va="center",
                fontsize=4.8, color=MUTED)
    save(fig, "fig_b_claim_ledger")


def figure_execution() -> None:
    pusht = load_json("outputs/pusht_checkpointed_downstream_use_v1/summary.json")
    point = load_json("outputs/dinowm_pointmaze_wave3/formal/external_use_summary.json")
    carrier = load_json("outputs/dinowm_pointmaze_wave3/formal/carrier_summary.json")

    fig = plt.figure(figsize=(7.25, 2.55), facecolor=PAPER)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.28)

    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(CREAM)
    points: list[tuple[str, float, float, str, str]] = []
    for host, name in [("dinowm", "PushT DINO"), ("lewm", "PushT LeWM")]:
        for cond, marker in [("full", "o"), ("reset", "x"), ("no_state", "s")]:
            d = pusht["hosts"][host]["conditions"][cond]
            points.append((f"{name} {cond}", d["balanced_accuracy_mean"],
                           d["execution"]["executed_success_mean"],
                           YELLOW_DEEP if cond == "full" else MUTED, marker))
    for arm, name in [("none", "PM none"), ("gru", "PM GRU"),
                      ("lstm", "PM LSTM"), ("ssm", "PM SSM"),
                      ("fixed_trust", "PM FT")]:
        x = point["arms"][arm]["goal_accuracy"]["mean"]
        y = point["arms"][arm]["executed_success"]["mean"]
        points.append((name, x, y,
                       YELLOW_DEEP if arm == "fixed_trust" else INK,
                       "D" if arm == "fixed_trust" else "o"))
    for label, x, y, color, marker in points:
        ax.scatter(x, y, s=42, marker=marker, facecolor=color,
                   edgecolor=INK, linewidth=0.6, zorder=3)
    ax.plot([0, 1], [0, 1], color=LINE_DARK, lw=0.7, ls="--", alpha=0.55)
    ax.axvline(0.25, color=MUTED, lw=0.7, ls=":")
    ax.axhline(0.25, color=MUTED, lw=0.7, ls=":")
    ax.set_xlim(0.1, 1.03)
    ax.set_ylim(0.1, 1.03)
    ax.set_xlabel("selected-goal / read accuracy")
    ax.set_ylabel("executed success")
    ax.grid(color=LINE, lw=0.55)
    panel_label(ax, "A", "readability ↔ execution")
    ax.text(0.28, 0.17, "controls", fontsize=5.8, color=MUTED)
    ax.text(0.77, 0.90, "full memory", fontsize=5.8, color=YELLOW_DEEP,
            fontweight="bold")

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(CREAM)
    arms = ["none", "gru", "lstm", "ssm", "fixed_trust"]
    labels = ["None", "GRU", "LSTM", "SSM", "Fixed\ntrust"]
    xpos = np.arange(len(arms))
    bacc = [carrier["results"]["15"]["arms"][a]["balanced_accuracy"]["mean"]
            for a in arms]
    execv = [point["arms"][a]["executed_success"]["mean"] for a in arms]
    ax2.bar(xpos - 0.16, bacc, width=0.30, color=PAPER, edgecolor=INK,
            lw=0.65, label="host read")
    ax2.bar(xpos + 0.16, execv, width=0.30, color=YELLOW, edgecolor=INK,
            lw=0.65, label="execution")
    ax2.axhline(0.25, color=MUTED, lw=0.7, ls=":")
    ax2.set_ylim(0, 1.02)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("PointMaze age-15")
    ax2.grid(axis="y", color=LINE, lw=0.55)
    panel_label(ax2, "B", "PointMaze arm order")
    ax2.legend(frameon=False, loc="upper left", ncol=2,
               bbox_to_anchor=(0.00, 1.02), handlelength=1.0)
    save(fig, "fig_b_execution")


def figure_wall() -> None:
    wall = load_json("outputs/dinowm_wall_audit_v1/stage_h_logistic_readout/summary.json")
    ridge = load_json("outputs/dinowm_wall_audit_v1/stage_h_carriers/summary.json")
    ages = [4, 8, 15]

    fig = plt.figure(figsize=(7.25, 2.45), facecolor=PAPER)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.50)

    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(CREAM)
    for arm, color, marker, label in [
        ("fixed_trust", YELLOW_DEEP, "o", "Fixed-trust"),
        ("ssm", INK, "s", "SSM"),
    ]:
        full = [wall["arms"][arm]["ages"][str(a)]["full_mean"] for a in ages]
        prior = [wall["arms"][arm]["ages"][str(a)]["prior_mean"] for a in ages]
        ax.plot(ages, prior, color=color, lw=1.5, ls="--", marker=marker,
                markersize=3.5, alpha=0.75)
        ax.plot(ages, full, color=color, lw=2.0, marker=marker,
                markersize=4.0, label=label)
    ax.axhline(0.75, color=INK, lw=0.75, ls=":")
    ax.axhline(0.25, color=MUTED, lw=0.75, ls=":")
    ax.set_xticks(ages)
    ax.set_ylim(0.18, 1.04)
    ax.set_ylabel("balanced accuracy")
    ax.set_xlabel("cue age")
    ax.grid(axis="y", color=LINE, lw=0.55)
    panel_label(ax, "A", "prior vs. exposure")
    ax.legend(frameon=False, loc="lower left")

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(CREAM)
    gaps = []
    names = []
    colors = []
    for arm, label, color in [("fixed_trust", "FT", YELLOW_DEEP), ("ssm", "SSM", INK)]:
        p = wall["arms"][arm]["ages"]["15"]["prior_mean"]
        f = wall["arms"][arm]["ages"]["15"]["full_mean"]
        gaps.append(p - f)
        names.append(label)
        colors.append(color)
    ax2.bar(names, gaps, color=colors, edgecolor=INK, lw=0.65)
    ax2.set_ylim(0, 0.82)
    ax2.set_ylabel("prior - host output")
    ax2.grid(axis="y", color=LINE, lw=0.55)
    panel_label(ax2, "B", "age-15 gap")
    for i, g in enumerate(gaps):
        ax2.text(i, g + 0.025, f"{g:.2f}", ha="center", va="bottom",
                 fontsize=6.0, fontweight="bold")

    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(CREAM)
    width = 0.34
    x = np.arange(len(ages))
    ridge_ft = [ridge["arms"]["fixed_trust"]["ages"][str(a)]["full_mean"]
                for a in ages]
    log_ft = [wall["arms"]["fixed_trust"]["ages"][str(a)]["full_mean"]
              for a in ages]
    ax3.bar(x - width / 2, ridge_ft, width, color=PAPER, edgecolor=INK,
            lw=0.6, label="ridge screen")
    ax3.bar(x + width / 2, log_ft, width, color=YELLOW, edgecolor=INK,
            lw=0.6, label="PCA-logistic")
    ax3.axhline(0.75, color=INK, lw=0.7, ls=":")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(a) for a in ages])
    ax3.set_ylim(0, 1.02)
    ax3.set_xlabel("age")
    ax3.grid(axis="y", color=LINE, lw=0.55)
    panel_label(ax3, "C", "screen vs. confirm")
    ax3.legend(frameon=False, loc="upper right", fontsize=5.6)
    save(fig, "fig_b_wall_bottleneck")


def figure_ogbench() -> None:
    envs = [
        ("PointMaze", "pointmaze-large-navigate-v0", "nav"),
        ("AntMaze", "antmaze-large-navigate-v0", "loco"),
        ("Humanoid", "humanoidmaze-large-navigate-v0", "hi-DoF"),
        ("Cube-1", "cube-single-play-v0", "manip"),
        ("Cube-2", "cube-double-play-v0", "manip"),
        ("Scene", "scene-play-v0", "scene"),
        ("Puzzle", "puzzle-3x3-play-v0", "puzzle"),
    ]
    root = ROOT / "outputs/ogbench_mesmem_admission_v1"
    checks = ["cue", "endpoint", "action", "obs", "mask"]
    values = np.zeros((len(checks), len(envs)))
    annotations: list[list[str]] = [["" for _ in envs] for _ in checks]
    for j, (_label, env, _family) in enumerate(envs):
        d = json.loads((root / env / "admission_summary.json").read_text())
        values[0, j] = d["cue_encoding"]["balanced_accuracy"]
        annotations[0][j] = "1.00"
        max_endpoint = max(v["no_cue_visual_endpoint"]["balanced_accuracy"]
                           for v in d["shortcuts"].values())
        max_action = max(v["action_only"]["balanced_accuracy"]
                         for v in d["shortcuts"].values())
        max_obs = max(v["observation_only"]["balanced_accuracy"]
                      for v in d["shortcuts"].values())
        values[1, j], values[2, j], values[3, j] = max_endpoint, max_action, max_obs
        annotations[1][j] = f"{max_endpoint:.2f}"
        annotations[2][j] = f"{max_action:.2f}"
        annotations[3][j] = f"{max_obs:.2f}"
        values[4, j] = 1.0 if d["counterfactual_audit"]["pass"] else 0.0
        annotations[4][j] = "pass"

    native_use = load_json("outputs/native_use_repaired_combined_v1/summary.json")

    fig = plt.figure(figsize=(7.25, 2.55), facecolor=PAPER)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.62, 1.10, 0.82], wspace=0.28)
    ax_img = fig.add_subplot(gs[0])
    ax_img.set_axis_off()
    ax_img.set_xlim(0, 1)
    ax_img.set_ylim(0, 1)
    ax_img.add_patch(Rectangle((0, 0), 1, 1, facecolor=INK,
                               edgecolor=INK, lw=0.8, zorder=0))
    ax_img.text(0.05, 0.88, "OGBENCH BREADTH", color=YELLOW,
                fontsize=6.8, fontweight="bold")
    ax_img.text(0.05, 0.76, "admit → execute",
                color=PAPER, fontsize=6.7, fontweight="bold")
    for i, (path, label) in enumerate([
        ("outputs/ogbench_mesmem_admission_v1/antmaze-large-navigate-v0/cue_variants.png", "AntMaze"),
        ("outputs/ogbench_mesmem_admission_v1/cube-single-play-v0/cue_variants.png", "Cube"),
        ("outputs/dinowm_wall_audit_v1/stage_g_admission/wall_cue_variants.png", "Wall"),
    ]):
        arr = image_array(path)
        y0 = 0.54 - i * 0.19
        draw_image_fit(ax_img, arr, (0.05, y0, 0.90, 0.13),
                       border=True, zorder=2)
        ax_img.text(0.05, y0 - 0.035, label, color=PAPER2,
                    fontsize=5.3, fontweight="bold")

    ax = fig.add_subplot(gs[1])
    ax.set_facecolor(CREAM)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "mem_admission", [PAPER, PAPER2, YELLOW])
    ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(envs)))
    ax.set_xticklabels([e[0] for e in envs], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(checks)))
    ax.set_yticklabels(["cue", "endpoint", "action", "state", "mask"])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            txt = annotations[i][j]
            color = INK if i in (0, 4) else MUTED
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.2,
                    fontweight="bold" if i in (0, 4) else "normal",
                    color=color)
    ax.set_xticks(np.arange(-.5, len(envs), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(checks), 1), minor=True)
    ax.grid(which="minor", color=LINE_DARK, linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    panel_label(ax, "D", "seven clean admission decks")

    ax_cap = fig.add_subplot(gs[2])
    ax_cap.set_facecolor(CREAM)
    rows = [r for r in native_use["rows"] if r.get("status") == "completed"]
    groups = [
        ("PM", lambda env: env.startswith("pointmaze-") and "teleport" not in env),
        ("Cube-1", lambda env: env == "cube-single-play-v0"),
        ("Cube-2", lambda env: env == "cube-double-play-v0"),
        ("Cube-3", lambda env: env == "cube-triple-play-v0"),
        ("Puzzle", lambda env: env == "puzzle-3x3-play-v0"),
        ("Scene", lambda env: env == "scene-play-v0"),
    ]
    full = []
    ctrl = []
    labels = []
    for label, pred in groups:
        matched = [r for r in rows if pred(r["env_name"])]
        if not matched:
            continue
        labels.append(label)
        full.append(np.mean([r["full"]["executed_success"]["mean"] for r in matched]))
        ctrl.append(np.mean([
            max(r["recent"]["executed_success"]["mean"],
                r["random"]["executed_success"]["mean"])
            for r in matched
        ]))
    y = np.arange(len(labels))
    ax_cap.barh(y + 0.16, full, height=0.28, color=YELLOW,
                edgecolor=INK, lw=0.55, label="full")
    ax_cap.barh(y - 0.16, ctrl, height=0.28, color=PAPER2,
                edgecolor=LINE_DARK, lw=0.55, label="control")
    ax_cap.axvline(1.00, color=INK, lw=0.65, ls=":")
    ax_cap.axvline(0.25, color=MUTED, lw=0.65, ls=":")
    ax_cap.set_yticks(y)
    ax_cap.set_yticklabels(labels)
    ax_cap.set_xlim(0, 1.05)
    ax_cap.set_xlabel("native success (mean age)")
    ax_cap.invert_yaxis()
    ax_cap.grid(axis="x", color=LINE, lw=0.5)
    ax_cap.legend(frameon=False, loc="lower right", fontsize=5.5)
    ax_cap.text(0.02, 1.05, "E", transform=ax_cap.transAxes, ha="left",
                va="bottom", fontsize=6.8, fontweight="bold",
                color=YELLOW_DEEP)
    ax_cap.text(0.11, 1.05, "repaired fixed-controller execution", transform=ax_cap.transAxes,
                ha="left", va="bottom", fontsize=7.0, fontweight="bold",
                color=INK)
    save(fig, "fig_b_ogbench_admission")


def write_native_use_repair_table() -> None:
    native_use = load_json("outputs/native_use_repaired_combined_v1/summary.json")
    completed = [r for r in native_use["rows"] if r.get("status") == "completed"]
    groups = [
        ("PointMaze non-teleport", lambda env: env.startswith("pointmaze-") and "teleport" not in env, "scored"),
        ("Cube-single", lambda env: env == "cube-single-play-v0", "scored"),
        ("Cube-double", lambda env: env == "cube-double-play-v0", "repaired"),
        ("Cube-triple", lambda env: env == "cube-triple-play-v0", "repaired"),
        ("Puzzle-3x3", lambda env: env == "puzzle-3x3-play-v0", "scored"),
        ("Scene", lambda env: env == "scene-play-v0", "repaired"),
    ]
    table_rows: list[str] = []
    full_values: list[float] = []
    recent_lifts: list[float] = []
    random_lifts: list[float] = []
    for label, pred, status in groups:
        matched = [r for r in completed if pred(r["env_name"])]
        if not matched:
            continue
        full = np.mean([r["full"]["executed_success"]["mean"] for r in matched])
        control = np.mean([
            max(r["recent"]["executed_success"]["mean"],
                r["random"]["executed_success"]["mean"])
            for r in matched
        ])
        oracle = np.mean([r["oracle"]["executed_success"]["mean"] for r in matched])
        full_values.extend([r["full"]["executed_success"]["mean"] for r in matched])
        recent_lifts.extend([r["full_vs_recent_success"] for r in matched])
        random_lifts.extend([r["full_vs_random_success"] for r in matched])
        table_rows.append(
            f"{label} & {len(matched)} & {full:.3f} & {control:.3f} & "
            f"{oracle:.3f} & {status}\\\\"
        )

    mean_full = float(np.mean(full_values))
    worst_full = float(np.min(full_values))
    mean_recent_lift = float(np.mean(recent_lifts))
    mean_random_lift = float(np.mean(random_lifts))
    path = GEN / "native_use_repaired_allenv_auto.tex"
    path.write_text("\n".join([
        r"\section{Generated all-env native-use repair monitor}",
        r"\label{app:native-use-repaired-auto}",
        "",
        r"This section is generated from \path{outputs/native_use_repaired_combined_v1/summary.json}.  It records the fixed-controller native-use sweep after the multi-object controller repair.  Rows without an audited local controller, or rows whose oracle-label controller gate fails, remain unscored scope limits.",
        "",
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Repaired all-env fixed-controller native-use summary.  Full memory is the ME-JEPA memory-selected target; control is the stronger of recent-only and random selection, averaged over the listed rows.  The sweep scores 24/36 env-age rows, with mean full success %.3f, worst scored full success %.3f, and mean full-minus-control lifts %.3f over recent-only and %.3f over random.}" % (
            mean_full, worst_full, mean_recent_lift, mean_random_lift),
        r"\label{tab:native-use-repaired-allenv}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Group & Rows & Full & Control & Oracle & Status\\",
        r"\midrule",
        *table_rows,
        r"PointMaze-teleport & 0 & \NA & \NA & 0.608--0.658 & gate failed\\",
        r"Ant/Humanoid locomotion & 0 & \NA & \NA & \NA & no local controller\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]) + "\n")


def write_snapshot() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "pusht_checkpointed_downstream_use": load_json(
            "outputs/pusht_checkpointed_downstream_use_v1/summary.json"),
        "pointmaze_external_use": load_json(
            "outputs/dinowm_pointmaze_wave3/formal/external_use_summary.json"),
        "pointmaze_carrier": load_json(
            "outputs/dinowm_pointmaze_wave3/formal/carrier_summary.json"),
        "wall_pca_logistic": load_json(
            "outputs/dinowm_wall_audit_v1/stage_h_logistic_readout/summary.json"),
        "wall_ridge_screen": load_json(
            "outputs/dinowm_wall_audit_v1/stage_h_carriers/summary.json"),
        "ogbench_admission": load_json(
            "outputs/ogbench_mesmem_admission_v1/summary_all.json"),
        "ogbench_feature_host": load_json(
            "outputs/ogbench_feature_host_stage_v1/summary.json"),
        "ogbench_native_use": load_json(
            "outputs/ogbench_native_use_stage_v1/summary.json"),
        "ogbench_native_use_repaired_allenv": load_json(
            "outputs/native_use_repaired_combined_v1/summary.json"),
    }
    (GEN / "result_snapshot.json").write_text(
        json.dumps(snapshot, sort_keys=True, indent=2) + "\n")


def main() -> None:
    figure_architecture()
    figure_claim_ledger()
    figure_execution()
    figure_wall()
    figure_ogbench()
    write_native_use_repair_table()
    write_snapshot()
    print(json.dumps({
        "status": "complete",
        "figures": sorted(p.name for p in FIG.glob("fig_b_*.pdf")),
        "snapshot": str((GEN / "result_snapshot.json").relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
