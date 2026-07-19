#!/usr/bin/env python3
"""Generate Paper C figures and result tables from local JSON artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_c"
FIGURES = PAPER / "figures"
GENERATED = PAPER / "generated_results"

SLOT_ROOT = ROOT / "outputs" / "multiview_patchset_color_jepa_native_v1"
BASELINE_ROOT = ROOT / "outputs" / "memory_arch_baselines_v1"
NATIVE_STATS = ROOT / "outputs" / "native_use_repaired_combined_v2" / "stats.json"

METHOD_LABELS = {
    "slot_current": "Slot memory (ours)",
    "gru": "GRU",
    "lstm": "LSTM",
    "mamba_lite": "Mamba-lite",
}
METHOD_ORDER = ["slot_current", "gru", "lstm", "mamba_lite"]
AGES = [4, 8, 15]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_cell_rows(root: Path, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/result.json")):
        obj = load_json(path)
        readout = obj.get("readout", {})
        if not {"full", "reset", "no_state"}.issubset(readout):
            continue
        rows.append(
            {
                "method": method,
                "env": str(obj["env_name"]),
                "age": int(obj["age"]),
                "seed": int(obj["seed"]),
                "full": float(readout["full"]["balanced_accuracy"]),
                "reset": float(readout["reset"]["balanced_accuracy"]),
                "no_state": float(readout["no_state"]["balanced_accuracy"]),
                "pass": bool(obj.get("gate", {}).get("pass")),
            }
        )
    return rows


def aggregate_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    env_age: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        env_age[(row["env"], row["age"])].append(row)
    return {
        "jobs": len(rows),
        "mean_full": mean(row["full"] for row in rows),
        "mean_reset": mean(row["reset"] for row in rows),
        "mean_no_state": mean(row["no_state"] for row in rows),
        "seed_pass": sum(row["pass"] for row in rows),
        "allpass_rows": sum(1 for values in env_age.values() if len(values) == 3 and all(v["pass"] for v in values)),
        "anypass_rows": sum(1 for values in env_age.values() if any(v["pass"] for v in values)),
        "age": {
            str(age): {
                "mean_full": mean(row["full"] for row in rows if row["age"] == age),
                "seed_pass": sum(row["pass"] for row in rows if row["age"] == age),
                "seed_count": sum(1 for row in rows if row["age"] == age),
            }
            for age in AGES
        },
    }


def all_rows() -> list[dict[str, Any]]:
    rows = load_cell_rows(SLOT_ROOT, "slot_current")
    for method in ["gru", "lstm", "mamba_lite"]:
        rows.extend(load_cell_rows(BASELINE_ROOT / method, method))
    return rows


def method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = aggregate_method(method_rows)

    by_env_age: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_env_age[(row["env"], row["age"])][row["method"]].append(float(row["full"]))
    wins = defaultdict(int)
    slot_better = 0
    slot_ge = 0
    for methods in by_env_age.values():
        means = {method: mean(values) for method, values in methods.items() if len(values) == 3}
        if not set(METHOD_ORDER).issubset(means):
            continue
        best = max(means, key=means.get)
        wins[best] += 1
        best_baseline = max(value for method, value in means.items() if method != "slot_current")
        slot_better += int(means["slot_current"] > best_baseline)
        slot_ge += int(means["slot_current"] >= best_baseline)
    return {
        "methods": summary,
        "wins_by_env_age": dict(wins),
        "slot_better_than_best_baseline": slot_better,
        "slot_ge_best_baseline": slot_ge,
    }


def env_label(name: str) -> str:
    replacements = {
        "pointmaze-medium-navigate-v0": "PointMaze-M",
        "pointmaze-large-navigate-v0": "PointMaze-L",
        "pointmaze-giant-navigate-v0": "PointMaze-G",
        "pointmaze-teleport-navigate-v0": "PointMaze-T",
        "antmaze-large-navigate-v0": "AntMaze-L",
        "antmaze-giant-navigate-v0": "AntMaze-G",
        "humanoidmaze-large-navigate-v0": "Humanoid",
        "cube-single-play-v0": "Cube-1",
        "cube-double-play-v0": "Cube-2",
        "cube-triple-play-v0": "Cube-3",
        "puzzle-3x3-play-v0": "Puzzle",
        "scene-play-v0": "Scene",
    }
    return replacements.get(name, name.replace("-navigate-v0", "").replace("-play-v0", ""))


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_main_table(snapshot: dict[str, Any]) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Controlled-age retention and exposure on 12 OGBench decks, three ages, and three seeds. Full, reset, and no-state are balanced accuracies from the post-hoc audit readout.}",
        "\\label{tab:main-comparison}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Full & Reset & No-state & Seed pass & Env-age pass\\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        item = snapshot["methods"][method]
        lines.append(
            f"{METHOD_LABELS[method]} & {fmt(item['mean_full'])} & {fmt(item['mean_reset'])} & "
            f"{fmt(item['mean_no_state'])} & {item['seed_pass']}/{item['jobs']} & "
            f"{item['allpass_rows']}/36\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (GENERATED / "main_comparison.tex").write_text("\n".join(lines))


def write_age_table(snapshot: dict[str, Any]) -> None:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Age-wise full-memory balanced accuracy. The age-15 row is the main long-memory stress test.}",
        "\\label{tab:age-comparison}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Age 4 & Age 8 & Age 15\\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        item = snapshot["methods"][method]["age"]
        values = " & ".join(fmt(item[str(age)]["mean_full"]) for age in AGES)
        lines.append(f"{METHOD_LABELS[method]} & {values}\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (GENERATED / "age_comparison.tex").write_text("\n".join(lines))


def write_native_table() -> None:
    stats = load_json(NATIVE_STATS)
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Fixed-controller native-use summary for supported OGBench rows. AntMaze and HumanoidMaze native-use rows are excluded because no audited local low-level controller is available.}",
        "\\label{tab:native-use}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Metric & Value\\\\",
        "\\midrule",
        f"Scored env-age rows & {stats['completed_rows']}/{stats['total_env_age_rows']}\\\\",
        f"Mean full-memory executed success & {fmt(stats['mean_full'])}\\\\",
        f"Worst scored full-memory row & {fmt(stats['worst_full'])}\\\\",
        f"Mean recent-only success & {fmt(stats['mean_recent'])}\\\\",
        f"Mean random-target success & {fmt(stats['mean_random'])}\\\\",
        f"Mean oracle success & {fmt(stats['mean_oracle'])}\\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    (GENERATED / "native_use.tex").write_text("\n".join(lines))


def write_hard_rows_table(rows: list[dict[str, Any]]) -> None:
    by: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        by[(row["env"], row["age"], row["method"])].append(float(row["full"]))
    selected = [
        ("puzzle-3x3-play-v0", 15),
        ("scene-play-v0", 15),
        ("pointmaze-teleport-navigate-v0", 15),
        ("antmaze-giant-navigate-v0", 15),
    ]
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Representative hard and long-age rows.  Values are mean full-memory balanced accuracy over three seeds.}",
        "\\label{tab:hard-rows}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Env-age row & Ours & GRU & LSTM & Mamba-lite\\\\",
        "\\midrule",
    ]
    for env, age in selected:
        values = []
        for method in METHOD_ORDER:
            vals = by[(env, age, method)]
            values.append(fmt(mean(vals)))
        lines.append(f"{env_label(env)} age {age} & " + " & ".join(values) + "\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (GENERATED / "hard_rows.tex").write_text("\n".join(lines))


def draw_box(ax, xy, wh, text, *, fc, ec="#18212f", size=8.3, weight="normal") -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.15, joinstyle="round")
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight)


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ink = "#18212f"
    ground = "#f4efe2"
    sensor = "#d9ead3"
    token = "#dbeafe"
    slot = "#f7d65b"
    motor = "#f6c7a5"
    audit = "#e7ddff"
    swatches = ["#dc2626", "#2563eb", "#16a34a", "#d8a900"]

    def panel(x: float, y: float, w: float, h: float, title: str, fc: str) -> None:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ink, lw=1.2))
        ax.text(x + 0.018, y + h - 0.04, title, fontsize=9, weight="bold", color=ink, va="top")

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": ink})

    ax.add_patch(plt.Rectangle((0.005, 0.02), 0.99, 0.94, facecolor=ground, edgecolor="#c9bfa8", lw=1.0))
    panel(0.035, 0.16, 0.22, 0.68, "1  sequence", sensor)
    panel(0.295, 0.16, 0.18, 0.68, "2  mask", "#fff7d6")
    panel(0.515, 0.16, 0.20, 0.68, "3  slots", "#fff8db")
    panel(0.755, 0.16, 0.20, 0.68, "4  audit", audit)

    # Sequence panel: rendered-frame cards and legal context strip.
    for i, x in enumerate([0.06, 0.105, 0.15, 0.195]):
        ax.add_patch(plt.Rectangle((x, 0.55), 0.035, 0.16, facecolor="#f8fafc", edgecolor=ink, lw=0.8))
        ax.add_patch(plt.Circle((x + 0.017, 0.61), 0.009, facecolor=swatches[i % 4], edgecolor=ink, lw=0.4))
        if i == 1:
            ax.add_patch(plt.Rectangle((x + 0.005, 0.665), 0.025, 0.022, facecolor="#fde68a", edgecolor=ink, lw=0.35))
    for i in range(18):
        color = "#d8a900" if i in [2, 3, 4] else "#93c5fd" if i in [14, 15, 16] else "#d1d5db"
        ax.add_patch(plt.Rectangle((0.055 + 0.0095 * i, 0.40), 0.0065, 0.055, facecolor=color, edgecolor="none"))
    ax.text(0.055, 0.34, "cue", fontsize=7.2, color="#6b5b00")
    ax.text(0.175, 0.34, "legal context", fontsize=7.2, color="#1d4ed8")

    # Mask panel: mined patch set and target bank.
    for r in range(3):
        for c in range(4):
            color = swatches[(r + c) % 4] if (r + c) % 3 != 0 else "#111827"
            ax.add_patch(plt.Rectangle((0.325 + c * 0.029, 0.58 - r * 0.06), 0.022, 0.038, facecolor=color, edgecolor=ink, lw=0.35))
    ax.text(0.326, 0.36, "salient patch set", fontsize=7.3, weight="bold")
    ax.text(0.326, 0.31, "labels hidden", fontsize=7.1, color="#5f5b52")

    # Slot transformer panel: attention stack and slot bank.
    draw_box(ax, (0.545, 0.63), (0.14, 0.055), "attention", fc=slot, size=7.2, weight="bold")
    draw_box(ax, (0.56, 0.535), (0.11, 0.045), "norm", fc="#fdf2c2", size=7.0, weight="bold")
    draw_box(ax, (0.545, 0.445), (0.14, 0.055), "MLP", fc="#ffe8a3", size=7.2, weight="bold")
    for i in range(8):
        ax.add_patch(plt.Circle((0.548 + i * 0.019, 0.335), 0.008, facecolor="#d8a900", edgecolor=ink, lw=0.45))
    ax.text(0.55, 0.285, "$M_t$", fontsize=8, weight="bold", color=ink)
    ax.text(0.59, 0.285, "8 slots", fontsize=7.2, color="#5f5b52")

    # Audit panel: readout bars and controller handoff.
    for j, (name, value, color) in enumerate([("full", 0.94, "#18212f"), ("reset", 0.25, "#9ca3af"), ("none", 0.25, "#c4b5a5")]):
        y = 0.64 - j * 0.10
        ax.text(0.78, y, name, fontsize=7.2, va="center")
        ax.add_patch(plt.Rectangle((0.825, y - 0.017), 0.095, 0.034, facecolor="#f8fafc", edgecolor=ink, lw=0.45))
        ax.add_patch(plt.Rectangle((0.825, y - 0.017), 0.095 * value, 0.034, facecolor=color, edgecolor="none"))
    ax.add_patch(plt.Rectangle((0.80, 0.28), 0.09, 0.055, facecolor=motor, edgecolor=ink, lw=0.8))
    ax.add_patch(plt.Circle((0.905, 0.307), 0.018, facecolor="#f8fafc", edgecolor=ink, lw=0.7))
    ax.text(0.792, 0.22, "target selected", fontsize=7.2, color="#5f5b52")

    arrow(0.255, 0.50, 0.295, 0.50)
    arrow(0.475, 0.50, 0.515, 0.50)
    arrow(0.715, 0.50, 0.755, 0.50)
    arrow(0.845, 0.38, 0.845, 0.335)
    ax.text(0.035, 0.90, "Masked-evidence slot-memory interface", fontsize=12, weight="bold", color=ink)
    ax.text(0.62, 0.90, "frames -> patches -> slots -> target", fontsize=8.3, color="#5f5b52")
    fig.savefig(FIGURES / "fig_c_architecture.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / "fig_c_architecture.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def plot_cue_inference() -> None:
    feature_path = SLOT_ROOT / "puzzle-3x3-play-v0" / "age_15" / "s0" / "features.npz"
    cell = np.load(feature_path)
    readout = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    readout.fit(cell["train_memory"], cell["train_labels"].astype(np.int64))

    labels = cell["val_labels"].astype(np.int64)
    selected: list[int] = []
    for cue in range(4):
        selected.extend(np.where(labels == cue)[0][:2].tolist())
    selected = selected[:8]
    condition_arrays = {
        "full memory": cell["val_full_memory"][selected],
        "reset": cell["val_reset_memory"][selected],
        "no state": cell["val_no_state_memory"][selected],
    }
    evidence = {
        name: _softmax(readout.decision_function(values))
        for name, values in condition_arrays.items()
    }
    truth = labels[selected]
    cue_colors = ["#dc2626", "#2563eb", "#16a34a", "#d8a900"]

    fig = plt.figure(figsize=(10.4, 3.15), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[0.58, 1.0, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_xlim(0, 1)
    ax0.set_ylim(len(selected), 0)
    ax0.axis("off")
    ax0.set_title("true cue", fontsize=9, pad=4)
    for i, cue in enumerate(truth):
        ax0.add_patch(plt.Rectangle((0.15, i + 0.12), 0.28, 0.76, facecolor=cue_colors[int(cue)], edgecolor="#18212f", lw=0.6))
        ax0.text(0.58, i + 0.5, f"cue {int(cue)}", va="center", fontsize=7.6)

    for col, (name, probs) in enumerate(evidence.items(), start=1):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(probs, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_title(name, fontsize=9, pad=4)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels([str(i) for i in range(4)], fontsize=7)
        ax.set_yticks([])
        for i, cue in enumerate(truth):
            pred = int(np.argmax(probs[i]))
            ax.add_patch(plt.Rectangle((pred - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111827", lw=1.1))
            ax.add_patch(plt.Rectangle((int(cue) - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor="#a94b3f", lw=0.7, linestyle="--"))
    cbar = fig.colorbar(im, ax=fig.axes[1:], fraction=0.025, pad=0.01)
    cbar.set_label("readout evidence", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(FIGURES / "fig_c_cue_inference.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / "fig_c_cue_inference.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_results(snapshot: dict[str, Any]) -> None:
    labels = ["Ours", "GRU", "LSTM", "Mamba-lite"]
    x = np.arange(len(labels))
    mean_full = [snapshot["methods"][m]["mean_full"] for m in METHOD_ORDER]
    age15 = [snapshot["methods"][m]["age"]["15"]["mean_full"] for m in METHOD_ORDER]
    pass_rows = [snapshot["methods"][m]["allpass_rows"] for m in METHOD_ORDER]
    colors = ["#111827", "#7a7f87", "#9ca3af", "#d8a900"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)
    axes[0].bar(x, mean_full, color=colors)
    axes[0].set_title("Mean full BAcc")
    axes[0].set_ylim(0.0, 1.05)
    axes[1].bar(x, age15, color=colors)
    axes[1].set_title("Age-15 full BAcc")
    axes[1].set_ylim(0.0, 1.05)
    axes[2].bar(x, pass_rows, color=colors)
    axes[2].set_title("Env-age all-pass rows")
    axes[2].set_ylim(0, 36)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(FIGURES / "fig_c_baseline_comparison.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / "fig_c_baseline_comparison.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_age_curves(snapshot: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    colors = {
        "slot_current": "#18212f",
        "gru": "#6b7280",
        "lstm": "#b45309",
        "mamba_lite": "#d8a900",
    }
    markers = {"slot_current": "o", "gru": "s", "lstm": "^", "mamba_lite": "D"}
    for method in METHOD_ORDER:
        vals = [snapshot["methods"][method]["age"][str(age)]["mean_full"] for age in AGES]
        ax.plot(AGES, vals, marker=markers[method], lw=2.2, color=colors[method], label=METHOD_LABELS[method])
    ax.axhline(0.75, color="#a94b3f", lw=1.0, ls="--", label="gate")
    ax.set_xlabel("Evidence age")
    ax.set_ylabel("Full-memory BAcc")
    ax.set_xticks(AGES)
    ax.set_ylim(0.2, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(pad=0.35)
    fig.savefig(FIGURES / "fig_c_age_decay.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / "fig_c_age_decay.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_env_heatmap(rows: list[dict[str, Any]]) -> None:
    envs = sorted({row["env"] for row in rows})
    envs = [
        "pointmaze-medium-navigate-v0",
        "pointmaze-large-navigate-v0",
        "pointmaze-giant-navigate-v0",
        "pointmaze-teleport-navigate-v0",
        "antmaze-large-navigate-v0",
        "antmaze-giant-navigate-v0",
        "humanoidmaze-large-navigate-v0",
        "cube-single-play-v0",
        "cube-double-play-v0",
        "cube-triple-play-v0",
        "puzzle-3x3-play-v0",
        "scene-play-v0",
    ]
    by: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        by[(row["env"], row["age"], row["method"])].append(float(row["full"]))
    gaps = np.zeros((len(AGES), len(envs)))
    for i, env in enumerate(envs):
        for j, age in enumerate(AGES):
            ours = mean(by[(env, age, "slot_current")])
            base_best = max(mean(by[(env, age, method)]) for method in ["gru", "lstm", "mamba_lite"])
            gaps[j, i] = ours - base_best
    fig, ax = plt.subplots(figsize=(10.4, 2.45))
    im = ax.imshow(gaps, cmap="YlGnBu", vmin=0.0, vmax=max(0.35, float(gaps.max())))
    ax.set_xticks(np.arange(len(envs)))
    ax.set_xticklabels([env_label(env) for env in envs], fontsize=7, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(AGES)))
    ax.set_yticklabels([str(age) for age in AGES], fontsize=8)
    ax.set_ylabel("Age")
    ax.set_title("Ours minus best recurrent baseline")
    for j in range(len(AGES)):
        for i in range(len(envs)):
            ax.text(i, j, f"{gaps[j, i]:.2f}", ha="center", va="center", fontsize=6.5, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.012)
    cbar.set_label("BAcc gap")
    fig.tight_layout(pad=0.25)
    fig.savefig(FIGURES / "fig_c_env_gap_heatmap.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / "fig_c_env_gap_heatmap.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_native_use() -> None:
    stats = load_json(NATIVE_STATS)
    labels = ["Full", "Recent", "Random", "Oracle"]
    values = [stats["mean_full"], stats["mean_recent"], stats["mean_random"], stats["mean_oracle"]]
    colors = ["#18212f", "#9ca3af", "#c4b5a5", "#d8a900"]
    fig, ax = plt.subplots(figsize=(5.3, 3.1))
    ax.bar(np.arange(len(labels)), values, color=colors)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Executed success")
    ax.set_title("Fixed-controller native-use rows")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(i, value + 0.025, f"{value:.3f}", ha="center", fontsize=8)
    ax.text(
        0.5,
        -0.24,
        f"{stats['completed_rows']}/{stats['total_env_age_rows']} env-age rows scored; Ant/Humanoid controller-unavailable.",
        ha="center",
        transform=ax.transAxes,
        fontsize=8,
        color="#5f5b52",
    )
    fig.tight_layout(pad=0.35)
    fig.savefig(FIGURES / "fig_c_native_use.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / "fig_c_native_use.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    rows = all_rows()
    snapshot = method_summary(rows)
    snapshot["native_use"] = load_json(NATIVE_STATS)
    snapshot["sources"] = {
        "slot_current": str(SLOT_ROOT / "summary.json"),
        "gru": str(BASELINE_ROOT / "gru" / "summary.json"),
        "lstm": str(BASELINE_ROOT / "lstm" / "summary.json"),
        "mamba_lite": str(BASELINE_ROOT / "mamba_lite" / "summary.json"),
        "native_use": str(NATIVE_STATS),
    }
    (GENERATED / "result_snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    write_main_table(snapshot)
    write_age_table(snapshot)
    write_native_table()
    write_hard_rows_table(rows)
    plot_architecture()
    plot_cue_inference()
    plot_results(snapshot)
    plot_age_curves(snapshot)
    plot_env_heatmap(rows)
    plot_native_use()
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
