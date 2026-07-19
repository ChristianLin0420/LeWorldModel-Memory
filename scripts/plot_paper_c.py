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
        "\\begin{table}[t]",
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
        "\\begin{table}[t]",
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
        "\\begin{table}[t]",
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
        "\\begin{table}[t]",
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
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
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
    ax.add_patch(plt.Rectangle((0.01, 0.02), 0.98, 0.94, facecolor=ground, edgecolor="#c9bfa8", lw=1.0))
    ax.text(
        0.03,
        0.93,
        "Masked-evidence slot transformer interface",
        fontsize=13,
        weight="bold",
        color=ink,
        va="top",
    )
    ax.text(
        0.03,
        0.88,
        "Physical-intelligence view: perception tokens, action/time tokens, structured memory, and bounded controller use.",
        fontsize=9,
        color="#5f5b52",
        va="top",
    )

    # Token lanes.
    draw_box(ax, (0.04, 0.66), (0.14, 0.12), "frame patch\ntokens", fc=sensor, weight="bold")
    draw_box(ax, (0.04, 0.49), (0.14, 0.12), "action\ntokens", fc=motor, weight="bold")
    draw_box(ax, (0.04, 0.32), (0.14, 0.12), "time\ntokens", fc="#f8e7c8", weight="bold")

    # Frozen tokenization block, with bars separated from text.
    ax.add_patch(plt.Rectangle((0.23, 0.48), 0.17, 0.27, facecolor=token, edgecolor=ink, lw=1.15))
    ax.text(0.315, 0.705, "frozen visual\nencoder", ha="center", va="center", fontsize=8.3, weight="bold")
    ax.text(0.315, 0.635, "+ token mixer", ha="center", va="center", fontsize=8.0, color="#334155")
    for i in range(5):
        ax.add_patch(
            plt.Rectangle(
                (0.252 + 0.025 * i, 0.535),
                0.017,
                0.055,
                facecolor="#93c5fd",
                edgecolor=ink,
                lw=0.6,
            )
        )

    # Transformer-style writer block.
    ax.add_patch(plt.Rectangle((0.44, 0.27), 0.24, 0.50, facecolor="#fff8db", edgecolor=ink, lw=1.25))
    ax.text(0.56, 0.735, "slot transformer writer", ha="center", va="center", fontsize=9.2, weight="bold")
    draw_box(ax, (0.47, 0.63), (0.18, 0.065), "multi-head slot attention", fc=slot, size=7.3, weight="bold")
    draw_box(ax, (0.49, 0.535), (0.14, 0.055), "Add & Norm", fc="#fdf2c2", size=7.2, weight="bold")
    draw_box(ax, (0.47, 0.435), (0.18, 0.065), "MLP slot update", fc="#ffe8a3", size=7.4, weight="bold")
    ax.text(0.56, 0.365, "8 persistent memory slots", ha="center", va="center", fontsize=7.8, weight="bold")
    for i in range(8):
        ax.add_patch(plt.Circle((0.482 + i * 0.022, 0.322), 0.008, facecolor="#d8a900", edgecolor=ink, lw=0.5))
    ax.text(0.56, 0.292, "$M_t \\in \\mathbb{R}^{8\\times d}$", ha="center", va="center", fontsize=7.4, color="#334155")

    # Target and audit side.
    draw_box(ax, (0.74, 0.66), (0.20, 0.11), "masked salient\npatch-set target", fc="#d9ead3", weight="bold")
    draw_box(ax, (0.74, 0.50), (0.20, 0.11), "set InfoNCE\n+ cosine + std", fc="#fde68a", weight="bold")
    draw_box(ax, (0.74, 0.34), (0.20, 0.11), "post-hoc readout\nfull / reset / no-state", fc=audit, weight="bold")
    draw_box(ax, (0.74, 0.18), (0.20, 0.11), "fixed controller\nnative-use check", fc=motor, weight="bold")

    arrows = [
        ((0.18, 0.72), (0.23, 0.68)),
        ((0.18, 0.55), (0.23, 0.615)),
        ((0.18, 0.38), (0.23, 0.55)),
        ((0.40, 0.615), (0.44, 0.665)),
        ((0.56, 0.63), (0.56, 0.59)),
        ((0.56, 0.535), (0.56, 0.50)),
        ((0.68, 0.665), (0.74, 0.715)),
        ((0.68, 0.545), (0.74, 0.555)),
        ((0.68, 0.355), (0.74, 0.395)),
        ((0.84, 0.34), (0.84, 0.29)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": ink})
    ax.text(0.44, 0.83, "causal stream", fontsize=8, color="#5f5b52")
    ax.text(0.735, 0.83, "labels withheld during training", fontsize=8, color="#5f5b52")
    ax.text(0.71, 0.09, "Execution is scoped: memory-selected target + audited fixed controller.", fontsize=8, color="#5f5b52")
    fig.tight_layout(pad=0.2)
    fig.savefig(FIGURES / "fig_c_architecture.pdf")
    fig.savefig(FIGURES / "fig_c_architecture.png", dpi=180)
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
    fig.savefig(FIGURES / "fig_c_baseline_comparison.pdf")
    fig.savefig(FIGURES / "fig_c_baseline_comparison.png", dpi=180)
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
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_c_age_decay.pdf")
    fig.savefig(FIGURES / "fig_c_age_decay.png", dpi=180)
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
    gaps = np.zeros((len(envs), len(AGES)))
    best_base = np.zeros_like(gaps)
    for i, env in enumerate(envs):
        for j, age in enumerate(AGES):
            ours = mean(by[(env, age, "slot_current")])
            base_best = max(mean(by[(env, age, method)]) for method in ["gru", "lstm", "mamba_lite"])
            gaps[i, j] = ours - base_best
            best_base[i, j] = base_best
    fig, ax = plt.subplots(figsize=(5.7, 5.1))
    im = ax.imshow(gaps, cmap="YlGnBu", vmin=0.0, vmax=max(0.35, float(gaps.max())))
    ax.set_xticks(np.arange(len(AGES)))
    ax.set_xticklabels([str(age) for age in AGES])
    ax.set_yticks(np.arange(len(envs)))
    ax.set_yticklabels([env_label(env) for env in envs], fontsize=8)
    ax.set_xlabel("Evidence age")
    ax.set_title("Ours minus best recurrent baseline")
    for i in range(len(envs)):
        for j in range(len(AGES)):
            ax.text(j, i, f"{gaps[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("BAcc gap")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_c_env_gap_heatmap.pdf")
    fig.savefig(FIGURES / "fig_c_env_gap_heatmap.png", dpi=180)
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
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_c_native_use.pdf")
    fig.savefig(FIGURES / "fig_c_native_use.png", dpi=180)
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
    plot_results(snapshot)
    plot_age_curves(snapshot)
    plot_env_heatmap(rows)
    plot_native_use()
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
