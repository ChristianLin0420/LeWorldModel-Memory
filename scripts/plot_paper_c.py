#!/usr/bin/env python3
"""Generate Paper C figures and result tables from local JSON artifacts.

Visual theme follows the Physical Intelligence dashboard: black / cream /
yellow, high-contrast panels, no purple glow.  Figures are kept compact so the
text carries the paper; large multi-panel collages are avoided.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
NATIVE_SUMMARY = ROOT / "outputs" / "native_use_repaired_combined_v2" / "summary.json"
AUTOAGE_ROOT = ROOT / "outputs" / "multiview_patchset_auto_age_v1"
TRAIN_AGES = {4, 8, 15}
EVAL_AGES = [4, 6, 8, 10, 12, 15, 18]
AUTO_AGE_ROOT = ROOT / "outputs" / "multiview_patchset_auto_age_v1"

METHOD_LABELS = {
    "slot_current": "Slot memory (ours)",
    "gru": "GRU",
    "lstm": "LSTM",
    "mamba_lite": "Mamba-lite",
}
METHOD_SHORT = {"slot_current": "Ours", "gru": "GRU", "lstm": "LSTM", "mamba_lite": "Mamba"}
METHOD_ORDER = ["slot_current", "gru", "lstm", "mamba_lite"]
AGES = [4, 8, 15]
AUTO_EVAL_AGES = [4, 6, 8, 10, 12, 15, 18]
ENV_ORDER = [
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
NATIVE_ENVS = [
    "pointmaze-medium-navigate-v0",
    "pointmaze-large-navigate-v0",
    "pointmaze-giant-navigate-v0",
    "pointmaze-teleport-navigate-v0",
    "cube-single-play-v0",
    "cube-double-play-v0",
    "cube-triple-play-v0",
    "puzzle-3x3-play-v0",
    "scene-play-v0",
]

PI = {
    "ink": "#111827",
    "black": "#000000",
    "cream": "#f5f4ef",
    "paper": "#fbfbf9",
    "paper2": "#efeee8",
    "yellow": "#fbd45b",
    "yellow_deep": "#d8a900",
    "muted": "#656760",
    "line": "#d4d3cb",
    "line_dark": "#333b49",
    "good": "#315b2c",
    "bad": "#7f1d1d",
    "gray": "#9ca3af",
    "gray2": "#6b7280",
    "blue": "#2563eb",
    "white": "#ffffff",
}
METHOD_COLORS = {
    "slot_current": PI["black"],
    "gru": PI["gray2"],
    "lstm": "#8a5d00",
    "mamba_lite": PI["yellow_deep"],
}
PI_HEAT = LinearSegmentedColormap.from_list(
    "pi_heat", [PI["cream"], PI["yellow"], PI["yellow_deep"], PI["ink"]]
)
PI_GAP = LinearSegmentedColormap.from_list(
    "pi_gap", ["#f8f7f2", PI["yellow"], PI["yellow_deep"], "#3d3200", PI["black"]]
)
MONO = "DejaVu Sans Mono"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": PI["ink"],
            "axes.labelcolor": PI["ink"],
            "axes.titlecolor": PI["ink"],
            "xtick.color": PI["ink"],
            "ytick.color": PI["ink"],
            "text.color": PI["ink"],
            "figure.facecolor": PI["white"],
            "axes.facecolor": PI["paper"],
            "savefig.facecolor": PI["white"],
            "grid.color": PI["line"],
            "grid.linewidth": 0.6,
            "axes.linewidth": 1.0,
            "legend.frameon": False,
        }
    )


def savefig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- #
# Data aggregation
# --------------------------------------------------------------------------- #
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


def load_summary_rows(path: Path, method: str) -> list[dict[str, Any]]:
    data = load_json(path)
    out = []
    for row in data["rows"]:
        out.append(
            {
                "method": method,
                "env": str(row["env_name"]),
                "age": int(row["age"]),
                "full": float(row["full_bacc_mean"]),
                "reset": float(row["reset_bacc_mean"]),
                "no_state": float(row["no_state_bacc_mean"]),
                "retrieval": float(row.get("retrieval_top1_mean", float("nan"))),
                "pass_count": int(row.get("pass_count", 0)),
                "all_pass": bool(row.get("all_pass", False)),
            }
        )
    return out


def summary_rows() -> list[dict[str, Any]]:
    rows = load_summary_rows(SLOT_ROOT / "summary.json", "slot_current")
    for method in ["gru", "lstm", "mamba_lite"]:
        rows.extend(load_summary_rows(BASELINE_ROOT / method / "summary.json", method))
    return rows


def index_summary(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {(r["method"], r["env"], r["age"]): r for r in rows}


def aggregate_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    env_age: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        env_age[(row["env"], row["age"])].append(row)
    return {
        "jobs": len(rows),
        "mean_full": mean(r["full"] for r in rows),
        "mean_reset": mean(r["reset"] for r in rows),
        "mean_no_state": mean(r["no_state"] for r in rows),
        "seed_pass": sum(r["pass"] for r in rows),
        "allpass_rows": sum(1 for v in env_age.values() if len(v) == 3 and all(x["pass"] for x in v)),
        "age": {
            str(age): {
                "mean_full": mean(r["full"] for r in rows if r["age"] == age),
                "mean_reset": mean(r["reset"] for r in rows if r["age"] == age),
                "mean_no_state": mean(r["no_state"] for r in rows if r["age"] == age),
                "seed_pass": sum(r["pass"] for r in rows if r["age"] == age),
                "seed_count": sum(1 for r in rows if r["age"] == age),
            }
            for age in AGES
        },
    }


def cell_rows() -> list[dict[str, Any]]:
    rows = load_cell_rows(SLOT_ROOT, "slot_current")
    for method in ["gru", "lstm", "mamba_lite"]:
        rows.extend(load_cell_rows(BASELINE_ROOT / method, method))
    return rows


def method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {m: aggregate_method([r for r in rows if r["method"] == m]) for m in METHOD_ORDER}
    by_env_age: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_env_age[(row["env"], row["age"])][row["method"]].append(float(row["full"]))
    wins: dict[str, int] = defaultdict(int)
    slot_better = slot_ge = 0
    for methods in by_env_age.values():
        means = {m: mean(v) for m, v in methods.items() if len(v) == 3}
        if not set(METHOD_ORDER).issubset(means):
            continue
        wins[max(means, key=means.get)] += 1
        best_baseline = max(v for m, v in means.items() if m != "slot_current")
        slot_better += int(means["slot_current"] > best_baseline)
        slot_ge += int(means["slot_current"] >= best_baseline)
    return {
        "methods": summary,
        "wins_by_env_age": dict(wins),
        "slot_better_than_best_baseline": slot_better,
        "slot_ge_best_baseline": slot_ge,
    }


def load_auto_age_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(AUTO_AGE_ROOT.glob("**/result.json")):
        obj = load_json(path)
        env = str(obj["env_name"])
        seed = int(obj["seed"])
        for item in obj.get("eval", []):
            readout = item["readout"]
            rows.append(
                {
                    "env": env,
                    "seed": seed,
                    "age": int(item["age"]),
                    "trained_age": bool(item.get("trained_age", False)),
                    "full": float(readout["full"]["balanced_accuracy"]),
                    "reset": float(readout["reset"]["balanced_accuracy"]),
                    "no_state": float(readout["no_state"]["balanced_accuracy"]),
                    "pass": bool(item.get("gate", {}).get("pass", False)),
                }
            )
    return rows


def auto_age_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_age: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_env_age: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_age[int(row["age"])].append(row)
        by_env_age[(str(row["env"]), int(row["age"]))].append(row)

    age_rows = []
    for age in AUTO_EVAL_AGES:
        values = by_age[age]
        age_rows.append(
            {
                "age": age,
                "trained_age": age in AGES,
                "seed_cells": len(values),
                "pass_count": sum(1 for row in values if row["pass"]),
                "mean_full": mean(row["full"] for row in values),
                "mean_reset": mean(row["reset"] for row in values),
                "mean_no_state": mean(row["no_state"] for row in values),
            }
        )

    env_age_rows = []
    for env in ENV_ORDER:
        for age in AUTO_EVAL_AGES:
            values = by_env_age[(env, age)]
            if not values:
                continue
            full = mean(row["full"] for row in values)
            reset = mean(row["reset"] for row in values)
            no_state = mean(row["no_state"] for row in values)
            env_age_rows.append(
                {
                    "env": env,
                    "age": age,
                    "seed_count": len(values),
                    "pass_count": sum(1 for row in values if row["pass"]),
                    "mean_full": full,
                    "mean_reset": reset,
                    "mean_no_state": no_state,
                    "margin": full - max(reset, no_state),
                }
            )

    worst = min(env_age_rows, key=lambda row: float(row["mean_full"]))
    worst_margin = min(env_age_rows, key=lambda row: float(row["margin"]))
    return {
        "train_ages": AGES,
        "eval_ages": AUTO_EVAL_AGES,
        "jobs": len({(row["env"], row["seed"]) for row in rows}),
        "eval_cells": len(rows),
        "pass_count": sum(1 for row in rows if row["pass"]),
        "mean_full": mean(row["full"] for row in rows),
        "mean_reset": mean(row["reset"] for row in rows),
        "mean_no_state": mean(row["no_state"] for row in rows),
        "age_rows": age_rows,
        "env_age_rows": env_age_rows,
        "worst_full": worst,
        "worst_margin": worst_margin,
        "claim_boundary": (
            "Mixed-age training over ages 4/8/15 with evaluation on 4/6/8/10/12/15/18. "
            "This tests delay generalization, but not a learned explicit age router or automatic maximum-age search."
        ),
    }


def env_label(name: str) -> str:
    return {
        "pointmaze-medium-navigate-v0": "PM-M",
        "pointmaze-large-navigate-v0": "PM-L",
        "pointmaze-giant-navigate-v0": "PM-G",
        "pointmaze-teleport-navigate-v0": "PM-T",
        "antmaze-large-navigate-v0": "Ant-L",
        "antmaze-giant-navigate-v0": "Ant-G",
        "humanoidmaze-large-navigate-v0": "Humanoid",
        "cube-single-play-v0": "Cube-1",
        "cube-double-play-v0": "Cube-2",
        "cube-triple-play-v0": "Cube-3",
        "puzzle-3x3-play-v0": "Puzzle",
        "scene-play-v0": "Scene",
    }.get(name, name)


def env_full(name: str) -> str:
    return {
        "pointmaze-medium-navigate-v0": "PointMaze-medium",
        "pointmaze-large-navigate-v0": "PointMaze-large",
        "pointmaze-giant-navigate-v0": "PointMaze-giant",
        "pointmaze-teleport-navigate-v0": "PointMaze-teleport",
        "antmaze-large-navigate-v0": "AntMaze-large",
        "antmaze-giant-navigate-v0": "AntMaze-giant",
        "humanoidmaze-large-navigate-v0": "HumanoidMaze-large",
        "cube-single-play-v0": "Cube-single",
        "cube-double-play-v0": "Cube-double",
        "cube-triple-play-v0": "Cube-triple",
        "puzzle-3x3-play-v0": "Puzzle-3x3",
        "scene-play-v0": "Scene",
    }.get(name, name)


def fmt(value: float) -> str:
    return f"{value:.3f}"


# --------------------------------------------------------------------------- #
# Main tables
# --------------------------------------------------------------------------- #
def write_main_table(snapshot: dict[str, Any]) -> None:
    tabular = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Full & Reset & No-state & Seed pass & Env-age pass\\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        item = snapshot["methods"][method]
        row = (
            f"{METHOD_LABELS[method]} & {fmt(item['mean_full'])} & {fmt(item['mean_reset'])} & "
            f"{fmt(item['mean_no_state'])} & {item['seed_pass']}/{item['jobs']} & {item['allpass_rows']}/36\\\\"
        )
        if method == "slot_current":
            row = "\\rowcolor{MemCream}" + row
        tabular.append(row)
    tabular.extend(["\\bottomrule", "\\end{tabular}", ""])
    (GENERATED / "main_comparison_tabular.tex").write_text("\n".join(tabular))


def write_age_table(snapshot: dict[str, Any]) -> None:
    tabular = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Age 4 & Age 8 & Age 15\\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        item = snapshot["methods"][method]["age"]
        values = " & ".join(fmt(item[str(age)]["mean_full"]) for age in AGES)
        row = f"{METHOD_LABELS[method]} & {values}\\\\"
        if method == "slot_current":
            row = "\\rowcolor{MemCream}" + row
        tabular.append(row)
    tabular.extend(["\\bottomrule", "\\end{tabular}", ""])
    (GENERATED / "age_comparison_tabular.tex").write_text("\n".join(tabular))


def write_native_table() -> None:
    stats = load_json(NATIVE_STATS)
    tabular = [
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
        "",
    ]
    (GENERATED / "native_use_tabular.tex").write_text("\n".join(tabular))


def write_auto_age_table(auto: dict[str, Any]) -> None:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Eval age & Trained? & Pass & Full & Reset & No-state\\\\",
        "\\midrule",
    ]
    for row in auto["age_rows"]:
        trained = "yes" if row["trained_age"] else "held-out"
        lines.append(
            f"{row['age']} & {trained} & {row['pass_count']}/{row['seed_cells']} & "
            f"{fmt(row['mean_full'])} & {fmt(row['mean_reset'])} & {fmt(row['mean_no_state'])}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (GENERATED / "auto_age_summary_tabular.tex").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Appendix tables
# --------------------------------------------------------------------------- #
def write_retention_by_age(index: dict[tuple[str, str, int], dict[str, Any]]) -> None:
    for age in AGES:
        lines = [
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            "\\multirow{2}{*}{Environment} & \\multicolumn{3}{c}{Slot memory (ours)} & \\multicolumn{3}{c}{Full-memory BAcc}\\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
            " & Full & Reset & No-st. & GRU & LSTM & Mamba\\\\",
            "\\midrule",
        ]
        for env in ENV_ORDER:
            ours = index[("slot_current", env, age)]
            gru = index[("gru", env, age)]
            lstm = index[("lstm", env, age)]
            mamba = index[("mamba_lite", env, age)]
            lines.append(
                f"{env_full(env)} & {fmt(ours['full'])} & {fmt(ours['reset'])} & {fmt(ours['no_state'])} & "
                f"{fmt(gru['full'])} & {fmt(lstm['full'])} & {fmt(mamba['full'])}\\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}", ""])
        (GENERATED / f"appendix_retention_age{age}.tex").write_text("\n".join(lines))


def write_retrieval_table(index: dict[tuple[str, str, int], dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Environment & Age 4 & Age 8 & Age 15\\\\",
        "\\midrule",
    ]
    for env in ENV_ORDER:
        vals = " & ".join(fmt(index[("slot_current", env, age)]["retrieval"]) for age in AGES)
        lines.append(f"{env_full(env)} & {vals}\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (GENERATED / "appendix_retrieval.tex").write_text("\n".join(lines))


def write_seed_pass_table(index: dict[tuple[str, str, int], dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Environment (age 15) & Ours & GRU & LSTM & Mamba\\\\",
        "\\midrule",
    ]
    for env in ENV_ORDER:
        vals = " & ".join(f"{index[(m, env, 15)]['pass_count']}/3" for m in METHOD_ORDER)
        lines.append(f"{env_full(env)} & {vals}\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (GENERATED / "appendix_seed_pass.tex").write_text("\n".join(lines))


def write_native_full_table() -> None:
    summary = load_json(NATIVE_SUMMARY)
    by = {(str(r["env_name"]), int(r["age"])): r for r in summary["rows"] if r["status"] == "completed"}
    lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Environment & Age & Full & Recent & Random & Oracle & Gain\\\\",
        "\\midrule",
    ]
    for env in NATIVE_ENVS:
        for age in AGES:
            row = by.get((env, age))
            if row is None:
                continue
            full = row["full"]["executed_success"]["mean"]
            std = row["full"]["executed_success"]["std"]
            recent = row["recent"]["executed_success"]["mean"]
            rnd = row["random"]["executed_success"]["mean"]
            oracle = row["oracle"]["executed_success"]["mean"]
            gain = full - max(recent, rnd)
            lines.append(
                f"{env_full(env)} & {age} & {full:.3f}\\,$\\pm$\\,{std:.3f} & {recent:.3f} & "
                f"{rnd:.3f} & {oracle:.3f} & +{gain:.3f}\\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (GENERATED / "appendix_native_full.tex").write_text("\n".join(lines))


def write_hyperparam_table() -> None:
    rows = [
        ("Epochs", "36"),
        ("Batch size", "96"),
        ("Model dimension $D$", "160"),
        ("Output memory slots $S$", "8"),
        ("Attention heads (slot writer)", "4"),
        ("Optimizer", "AdamW"),
        ("Learning rate", "$3\\times10^{-4}$"),
        ("Weight decay", "$10^{-4}$"),
        ("Target patch grid", "$4\\times4$ pooled"),
        ("Target views per episode", "3 temporal bins"),
        ("Loss", "set InfoNCE $+$ cosine $+$ std reg."),
        ("Readout", "ridge classifier (post-hoc)"),
        ("Evidence ages", "4, 8, 15"),
        ("Seeds", "0, 1, 2"),
        ("Validation split", "held-out episodes per cell"),
    ]
    lines = ["\\begin{tabular}{ll}", "\\toprule", "Hyperparameter & Value\\\\", "\\midrule"]
    lines += [f"{k} & {v}\\\\" for k, v in rows]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (GENERATED / "appendix_hyperparams.tex").write_text("\n".join(lines))


def write_auto_age_full_table(auto: dict[str, Any]) -> None:
    by = {(row["env"], row["age"]): row for row in auto["env_age_rows"]}
    lines = [
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Environment & 4 & 6 & 8 & 10 & 12 & 15 & 18\\\\",
        "\\midrule",
    ]
    for env in ENV_ORDER:
        vals = []
        for age in AUTO_EVAL_AGES:
            row = by[(env, age)]
            vals.append(f"{row['mean_full']:.3f}")
        lines.append(f"{env_full(env)} & " + " & ".join(vals) + "\\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (GENERATED / "appendix_auto_age_full.tex").write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Architecture figure (redesigned: explicit data + train/eval flow)
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, *, fc, ec=None, lw=1.2, round_pad=0.012, z=3):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={round_pad},rounding_size={round_pad*1.4}",
        facecolor=fc,
        edgecolor=ec or PI["ink"],
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)


def _label(ax, x, y, text, *, size=8.0, color=None, weight="normal", mono=False, ha="center", va="center", z=6):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        color=color or PI["ink"],
        weight=weight,
        zorder=z,
        fontfamily=MONO if mono else "DejaVu Sans",
    )


def _arrow(ax, p0, p1, *, color, style="-|>", lw=1.6, ls="-", z=5, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=13,
            lw=lw,
            color=color,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            zorder=z,
        )
    )


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(PI["cream"])
    ax.set_facecolor(PI["cream"])
    ax.add_patch(plt.Rectangle((0.008, 0.02), 0.984, 0.965, facecolor=PI["paper"], edgecolor=PI["line_dark"], lw=1.3, zorder=1))

    _label(ax, 0.022, 0.945, "MASKED-EVIDENCE SLOT MEMORY", size=11.5, weight="bold", mono=True, ha="left")
    # legend
    _arrow(ax, (0.63, 0.948), (0.675, 0.948), color=PI["yellow_deep"], lw=2.2)
    _label(ax, 0.68, 0.948, "training gradient path", size=7.6, color=PI["muted"], ha="left")
    _arrow(ax, (0.82, 0.948), (0.865, 0.948), color=PI["ink"], ls=(0, (3, 2)), lw=1.7)
    _label(ax, 0.87, 0.948, "audit (eval only)", size=7.6, color=PI["muted"], ha="left")

    yc = 0.60  # main row center

    # 1. Causal stream ------------------------------------------------------ #
    _box(ax, 0.03, 0.44, 0.155, 0.36, fc="#e8ebe2")
    _label(ax, 0.107, 0.76, "1  CAUSAL STREAM", size=7.8, weight="bold", mono=True)
    swatch = ["#dc2626", PI["blue"], "#16a34a", PI["yellow_deep"], "#7c3aed", "#0891b2"]
    for i in range(6):
        x = 0.045 + i * 0.0225
        hot = i in (1,)
        ax.add_patch(plt.Rectangle((x, 0.63), 0.019, 0.09, facecolor=PI["white"], edgecolor=PI["ink"], lw=0.7, zorder=5))
        ax.add_patch(plt.Circle((x + 0.0095, 0.675), 0.006, facecolor=swatch[i], edgecolor=PI["ink"], lw=0.3, zorder=6))
        if hot:
            ax.add_patch(plt.Rectangle((x, 0.723), 0.019, 0.012, facecolor=PI["yellow"], edgecolor=PI["ink"], lw=0.3, zorder=6))
    _label(ax, 0.05, 0.60, "cue frame (early)", size=6.4, color="#b45309", ha="left")
    for i in range(14):
        c = PI["yellow"] if i < 2 else "#93c5fd" if i >= 11 else PI["line"]
        ax.add_patch(plt.Rectangle((0.045 + i * 0.0092, 0.535), 0.006, 0.045, facecolor=c, edgecolor="none", zorder=5))
    _label(ax, 0.045, 0.505, "cue", size=6.0, color="#b45309", ha="left")
    _label(ax, 0.15, 0.505, "legal $K$", size=6.0, color="#1d4ed8", ha="right")
    _label(ax, 0.107, 0.475, r"$x_{0:T},\ a_{0:T-1},\ \tau$", size=7.0, mono=True)

    # 2. Tokenizer ---------------------------------------------------------- #
    _box(ax, 0.205, 0.50, 0.10, 0.24, fc=PI["paper2"])
    _label(ax, 0.255, 0.71, "2  TOKENIZE", size=7.6, weight="bold", mono=True)
    _label(ax, 0.255, 0.655, "frame CNN", size=7.0)
    _label(ax, 0.255, 0.615, "+ action emb", size=7.0)
    _label(ax, 0.255, 0.575, "+ time emb", size=7.0)
    _label(ax, 0.255, 0.535, r"tokens $[T{\times}D]$", size=6.6, mono=True, color=PI["muted"])

    # 3. Slot writer -------------------------------------------------------- #
    _box(ax, 0.325, 0.46, 0.155, 0.32, fc=PI["black"], ec=PI["ink"])
    _label(ax, 0.4025, 0.745, "3  SLOT WRITER", size=7.8, weight="bold", mono=True, color=PI["yellow"])
    for lab, yy, hot in [("cross-attn (queries=slots)", 0.685, True), ("LayerNorm", 0.62, False), ("MLP update", 0.555, True)]:
        _box(ax, 0.34, yy - 0.028, 0.125, 0.05, fc=PI["yellow"] if hot else PI["paper2"], ec=PI["yellow"] if hot else PI["line"], lw=0.9, z=5)
        _label(ax, 0.4025, yy - 0.003, lab, size=6.6, weight="bold")
    for i in range(8):
        ax.add_patch(plt.Circle((0.352 + i * 0.0145, 0.50), 0.007, facecolor=PI["yellow"], edgecolor=PI["ink"], lw=0.4, zorder=6))
    _label(ax, 0.4025, 0.478, r"memory $M_t=[S{\times}D],\ S{=}8$", size=6.6, mono=True, color=PI["yellow"])

    # 4. Target miner (offline, below) ------------------------------------- #
    _box(ax, 0.205, 0.10, 0.275, 0.26, fc=PI["yellow"])
    _label(ax, 0.3425, 0.335, "4  TARGET MINER  (label-free, offline)", size=7.6, weight="bold", mono=True)
    stages = ["temporal\nbins", "saliency\nselect", "mask\nfrom view", r"fixed $\phi$" + "\nstop-grad"]
    for i, s in enumerate(stages):
        x = 0.220 + i * 0.065
        _box(ax, x, 0.16, 0.056, 0.10, fc=PI["paper"], ec=PI["ink"], lw=0.9, z=5)
        _label(ax, x + 0.028, 0.21, s, size=5.9)
        if i < 3:
            _arrow(ax, (x + 0.056, 0.21), (x + 0.065, 0.21), color=PI["ink"], lw=1.0)
    _label(ax, 0.3425, 0.125, r"target set $Y=\{y_j\}=[N{\times}D]$", size=6.6, mono=True, color=PI["ink"])

    # 5. Set predictor + loss (training) ----------------------------------- #
    _box(ax, 0.52, 0.46, 0.14, 0.32, fc=PI["paper2"])
    _label(ax, 0.59, 0.745, "5  SET PREDICTOR", size=7.6, weight="bold", mono=True)
    _label(ax, 0.59, 0.68, r"$p(M_t)\rightarrow \hat{Y}$", size=7.4, mono=True)
    _box(ax, 0.53, 0.55, 0.12, 0.085, fc=PI["yellow"], ec=PI["yellow_deep"], lw=1.1, z=5)
    _label(ax, 0.59, 0.61, "set InfoNCE", size=6.6, weight="bold")
    _label(ax, 0.59, 0.585, "+ cosine + std", size=6.4)
    _label(ax, 0.59, 0.492, "matches $\\hat{Y}$ to $Y$", size=6.4, color=PI["muted"])

    # 6. Audit readout (eval-only) ----------------------------------------- #
    _box(ax, 0.70, 0.40, 0.265, 0.40, fc="#eceae2")
    _label(ax, 0.8325, 0.765, "6  FAIL-CLOSED AUDIT  (eval only)", size=7.6, weight="bold", mono=True)
    _label(ax, 0.72, 0.715, "ridge readout fit on train $M$; applied to:", size=6.6, color=PI["muted"], ha="left")
    for name, val, col, yy in [("full  $M_t$", 0.96, PI["black"], 0.66), ("reset (clear at $K$)", 0.25, PI["gray"], 0.585), ("no-state", 0.25, "#c4b5a5", 0.51)]:
        _label(ax, 0.715, yy, name, size=6.6, ha="left")
        ax.add_patch(plt.Rectangle((0.845, yy - 0.017), 0.11, 0.03, facecolor=PI["white"], edgecolor=PI["ink"], lw=0.6, zorder=5))
        ax.add_patch(plt.Rectangle((0.845, yy - 0.017), 0.11 * val, 0.03, facecolor=col, edgecolor="none", zorder=6))
        _label(ax, 0.96, yy, f"{val:.2f}" if val > 0.3 else "~.25", size=6.0, ha="left", color=PI["muted"])
    _label(ax, 0.715, 0.455, "pass: full high $\\wedge$ controls $\\approx$ chance", size=6.6, weight="bold", color=PI["ink"], ha="left")
    _box(ax, 0.715, 0.415, 0.10, 0.03, fc=PI["yellow"], ec=PI["ink"], lw=0.7, z=5)
    _label(ax, 0.765, 0.43, "target $\\rightarrow$ controller", size=6.0)

    # Flows ---------------------------------------------------------------- #
    _arrow(ax, (0.185, yc), (0.205, 0.62), color=PI["ink"], lw=1.5)  # stream->tokenize
    _arrow(ax, (0.305, 0.62), (0.325, 0.62), color=PI["ink"], lw=1.5)  # tokenize->slots
    _arrow(ax, (0.185, 0.55), (0.205, 0.30), color=PI["ink"], lw=1.3, rad=-0.2)  # stream->miner
    _arrow(ax, (0.48, 0.62), (0.52, 0.62), color=PI["yellow_deep"], lw=2.2)  # slots->predictor
    _arrow(ax, (0.3425, 0.36), (0.55, 0.55), color=PI["yellow_deep"], lw=2.2, rad=0.18)  # target->loss
    _arrow(ax, (0.59, 0.55), (0.4025, 0.50), color=PI["yellow_deep"], lw=2.0, ls="-", rad=0.28)  # loss grad back to slots
    _label(ax, 0.50, 0.435, "gradient", size=6.2, color=PI["yellow_deep"])
    _arrow(ax, (0.462, 0.74), (0.72, 0.80), color=PI["ink"], ls=(0, (3, 2)), lw=1.7, rad=-0.22)  # M_t -> audit (eval)
    _label(ax, 0.60, 0.90, r"frozen $M_t$ (no labels used in training)", size=6.6, color=PI["muted"])

    savefig(fig, "fig_c_architecture")


# --------------------------------------------------------------------------- #
# Result figures (compact)
# --------------------------------------------------------------------------- #
def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def plot_shortcut_diagnostics() -> None:
    """Cue inference (left) and confusion matrices (right) in one row."""
    cell = np.load(SLOT_ROOT / "puzzle-3x3-play-v0" / "age_15" / "s0" / "features.npz")
    readout = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    readout.fit(cell["train_memory"], cell["train_labels"].astype(np.int64))
    labels = cell["val_labels"].astype(np.int64)
    selected: list[int] = []
    for cue in range(4):
        selected.extend(np.where(labels == cue)[0][:2].tolist())
    selected = selected[:8]
    arrays = {
        "full": cell["val_full_memory"][selected],
        "reset": cell["val_reset_memory"][selected],
        "no state": cell["val_no_state_memory"][selected],
    }
    evidence = {n: _softmax(readout.decision_function(v)) for n, v in arrays.items()}
    truth = labels[selected]
    cue_colors = ["#dc2626", PI["blue"], "#16a34a", PI["yellow_deep"]]

    result = load_json(SLOT_ROOT / "puzzle-3x3-play-v0" / "age_15" / "s0" / "result.json")
    conf = {
        "full": np.asarray(result["readout"]["full"]["confusion_matrix"], float),
        "reset": np.asarray(result["readout"]["reset"]["confusion_matrix"], float),
        "no state": np.asarray(result["readout"]["no_state"]["confusion_matrix"], float),
    }

    fig = plt.figure(figsize=(9.4, 2.35), constrained_layout=True)
    fig.patch.set_facecolor(PI["white"])
    gs = fig.add_gridspec(1, 8, width_ratios=[0.32, 1, 1, 1, 0.30, 1, 1, 1])

    axc = fig.add_subplot(gs[0, 0])
    axc.set_xlim(0, 1)
    axc.set_ylim(len(selected), 0)
    axc.axis("off")
    axc.set_title("cue", fontsize=7.5, pad=2, weight="bold")
    for i, cue in enumerate(truth):
        axc.add_patch(plt.Rectangle((0.2, i + 0.15), 0.55, 0.7, facecolor=cue_colors[int(cue)], edgecolor=PI["ink"], lw=0.6))
    im = None
    for col, (name, probs) in zip([1, 2, 3], evidence.items()):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(probs, vmin=0, vmax=1, cmap=PI_HEAT, aspect="auto")
        ax.set_title(name, fontsize=7.5, pad=2, weight="bold")
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(range(4), fontsize=6)
        ax.set_yticks([])
        for i, cue in enumerate(truth):
            pred = int(np.argmax(probs[i]))
            ax.add_patch(plt.Rectangle((pred - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=PI["ink"], lw=0.9))
            ax.add_patch(plt.Rectangle((int(cue) - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor=PI["bad"], lw=0.7, linestyle="--"))
        ax.set_facecolor(PI["paper"])
    fig.text(0.255, 1.02, "(a) cue inference", ha="center", fontsize=8, weight="bold")

    imc = None
    for col, (name, matrix) in zip([5, 6, 7], conf.items()):
        ax = fig.add_subplot(gs[0, col])
        rs = matrix.sum(axis=1, keepdims=True)
        norm = np.divide(matrix, rs, out=np.zeros_like(matrix), where=rs > 0)
        imc = ax.imshow(norm, cmap=PI_HEAT, vmin=0, vmax=1)
        ax.set_title(name, fontsize=7.5, pad=2, weight="bold")
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(4))
        ax.set_xticklabels(range(4), fontsize=6)
        ax.set_yticklabels(list(range(4)) if col == 5 else [], fontsize=6)
        for j in range(4):
            for i in range(4):
                v = norm[j, i]
                ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=4.8, color=PI["white"] if v > 0.55 else PI["ink"])
    fig.text(0.74, 1.02, "(b) row-normalized confusion", ha="center", fontsize=8, weight="bold")
    savefig(fig, "fig_c_cue_inference")


def plot_results(snapshot: dict[str, Any]) -> None:
    labels = [METHOD_SHORT[m] for m in METHOD_ORDER]
    x = np.arange(len(labels))
    mean_full = [snapshot["methods"][m]["mean_full"] for m in METHOD_ORDER]
    age15 = [snapshot["methods"][m]["age"]["15"]["mean_full"] for m in METHOD_ORDER]
    pass_rows = [snapshot["methods"][m]["allpass_rows"] for m in METHOD_ORDER]
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.35), constrained_layout=True)
    fig.patch.set_facecolor(PI["white"])
    panels = [
        (mean_full, "Mean full BAcc", (0, 1.08), False),
        (age15, "Age-15 full BAcc", (0, 1.08), False),
        (pass_rows, "All-pass / 36", (0, 40), True),
    ]
    for ax, (vals, title, ylim, is_count) in zip(axes, panels):
        bars = ax.bar(x, vals, color=colors, edgecolor=PI["ink"], linewidth=0.7, width=0.72)
        bars[0].set_edgecolor(PI["yellow_deep"])
        bars[0].set_linewidth(1.5)
        ax.set_title(title, fontsize=8.5, weight="bold", pad=4)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + (ylim[1] * 0.015), f"{v:.2f}" if not is_count else f"{int(v)}", ha="center", va="bottom", fontsize=6.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
    savefig(fig, "fig_c_baseline_comparison")


def plot_age_and_autoage(snapshot: dict[str, Any], auto: dict[str, Any]) -> None:
    """Left: architecture age decay (fixed-age). Right: auto-age unseen generalization."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.7), constrained_layout=True)
    fig.patch.set_facecolor(PI["white"])

    ax = axes[0]
    markers = {"slot_current": "o", "gru": "s", "lstm": "^", "mamba_lite": "D"}
    for method in METHOD_ORDER:
        vals = [snapshot["methods"][method]["age"][str(age)]["mean_full"] for age in AGES]
        ax.plot(
            AGES, vals, marker=markers[method], markersize=6, lw=2.0, color=METHOD_COLORS[method],
            markerfacecolor=PI["yellow"] if method == "slot_current" else METHOD_COLORS[method],
            markeredgecolor=PI["ink"], markeredgewidth=0.8, label=METHOD_LABELS[method],
            zorder=4 if method == "slot_current" else 3,
        )
    ax.axhline(0.75, color=PI["bad"], lw=1.0, ls="--", label="gate")
    ax.set_xlabel("Evidence age", fontsize=8.5)
    ax.set_ylabel("Full-memory BAcc", fontsize=8.5)
    ax.set_xticks(AGES)
    ax.set_ylim(0.25, 1.03)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.2, loc="lower left")
    ax.set_title("(a) architecture vs. age (fixed-age training)", fontsize=8.5, weight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    # Prefer the long-horizon age-scaling curve (to 128) if available; else the
    # standard mixed-age summary. This extends the delay axis well beyond 15.
    long_curve = None
    agescale_root = ROOT / "outputs" / "paper_c_agescale_v1"
    pts = sorted(agescale_root.glob("*/auto_age/s*/result.json"))
    if pts:
        try:
            d0 = load_json(pts[0])
            eval_ages = [int(e["age"]) for e in d0["eval"]]
            train_ages = set(int(a) for a in d0.get("train_ages", []))
            # average full/reset/no-state over available long-horizon envs
            accum: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for p in pts:
                dd = load_json(p)
                for e in dd["eval"]:
                    r = e["readout"]
                    accum[int(e["age"])]["full"].append(r["full"]["balanced_accuracy"])
                    accum[int(e["age"])]["reset"].append(r["reset"]["balanced_accuracy"])
                    accum[int(e["age"])]["no_state"].append(r["no_state"]["balanced_accuracy"])
            eval_ages = sorted(accum)
            full = [mean(accum[a]["full"]) for a in eval_ages]
            reset = [mean(accum[a]["reset"]) for a in eval_ages]
            ns = [mean(accum[a]["no_state"]) for a in eval_ages]
            long_curve = True
        except Exception:
            long_curve = None
    if not long_curve:
        age_rows = {int(r["age"]): r for r in auto["age_rows"]}
        eval_ages = [int(a) for a in auto["eval_ages"]]
        train_ages = set(int(a) for a in auto["train_ages"])
        full = [age_rows[a]["mean_full"] for a in eval_ages]
        reset = [age_rows[a]["mean_reset"] for a in eval_ages]
        ns = [age_rows[a]["mean_no_state"] for a in eval_ages]
    seen_mask = [a in train_ages for a in eval_ages]
    if max(eval_ages) > 40:
        ax.set_xscale("log", base=2)
    ax.plot(eval_ages, full, "-", lw=2.0, color=PI["ink"], zorder=3)
    for a, f, seen in zip(eval_ages, full, seen_mask):
        ax.scatter([a], [f], s=64 if seen else 58, zorder=4,
                   marker="o" if seen else "D",
                   facecolor=PI["ink"] if seen else PI["yellow"],
                   edgecolor=PI["ink"], linewidth=0.9)
    ax.plot(eval_ages, reset, "-", lw=1.4, color=PI["gray"], label="reset")
    ax.plot(eval_ages, ns, "--", lw=1.4, color="#c4b5a5", label="no-state")
    ax.axhline(0.75, color=PI["bad"], lw=1.0, ls="--")
    # annotate seen vs unseen
    ax.scatter([], [], marker="o", facecolor=PI["ink"], edgecolor=PI["ink"], label="trained age")
    ax.scatter([], [], marker="D", facecolor=PI["yellow"], edgecolor=PI["ink"], label="unseen age")
    ax.set_xlabel("Evaluation age (delay)", fontsize=8.5)
    ax.set_ylabel("Balanced accuracy", fontsize=8.5)
    ax.set_xticks(eval_ages)
    ax.set_xticklabels([str(a) for a in eval_ages])
    ax.set_ylim(0.15, 1.03)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.2, loc="lower left")
    ax.set_title("(b) mixed-age training generalizes across delays", fontsize=8.5, weight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7)
    savefig(fig, "fig_c_age_generalization")


def _autoage_env_curves() -> dict[str, dict[int, dict[str, float]]]:
    """Per-environment full/reset/no-state balanced accuracy vs evaluation age,
    averaged over seeds, from the mixed-age run (continuous 7-age grid)."""
    root = ROOT / "outputs" / "multiview_patchset_auto_age_v1"
    acc: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for p in sorted(root.glob("*/auto_age/s*/result.json")):
        try:
            d = load_json(p)
        except Exception:
            continue
        env = d["env_name"]
        for e in d.get("eval", []):
            r = e["readout"]
            acc[env][int(e["age"])]["full"].append(r["full"]["balanced_accuracy"])
            acc[env][int(e["age"])]["reset"].append(r["reset"]["balanced_accuracy"])
            acc[env][int(e["age"])]["no_state"].append(r["no_state"]["balanced_accuracy"])
    out: dict[str, dict[int, dict[str, float]]] = {}
    for env, ages in acc.items():
        out[env] = {age: {k: mean(v) for k, v in cond.items()} for age, cond in ages.items()}
    return out


def plot_env_heatmaps(rows: list[dict[str, Any]]) -> None:
    """Per-environment memory-vs-delay LINE plots (replaces the fixed-age heatmap
    value maps): full-memory accuracy and audit margin over a continuous age grid."""
    curves = _autoage_env_curves()
    if not curves:
        return
    envs = [e for e in ENV_ORDER if e in curves]
    ages = sorted({a for e in envs for a in curves[e]})
    cmap = plt.get_cmap("cividis")
    colors = {e: cmap(i / max(1, len(envs) - 1)) for i, e in enumerate(envs)}

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), constrained_layout=True)
    fig.patch.set_facecolor(PI["white"])
    ax = axes[0]
    for e in envs:
        xs = [a for a in ages if a in curves[e]]
        ys = [curves[e][a]["full"] for a in xs]
        ax.plot(xs, ys, "-o", ms=3, lw=1.4, color=colors[e], label=env_label(e))
    ax.axhline(0.75, color=PI["bad"], ls="--", lw=1.0)
    ax.set_xlabel("Evidence age (delay)", fontsize=9)
    ax.set_ylabel("Full-memory BAcc", fontsize=9)
    ax.set_ylim(0.4, 1.02)
    ax.set_title("(a) retention vs delay, per environment", fontsize=9, weight="bold")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    ax = axes[1]
    for e in envs:
        xs = [a for a in ages if a in curves[e]]
        ys = [curves[e][a]["full"] - max(curves[e][a]["reset"], curves[e][a]["no_state"]) for a in xs]
        ax.plot(xs, ys, "-o", ms=3, lw=1.4, color=colors[e])
    ax.set_xlabel("Evidence age (delay)", fontsize=9)
    ax.set_ylabel("Audit margin (full $-$ control)", fontsize=9)
    ax.set_ylim(0.0, 0.9)
    ax.set_title("(b) audit margin vs delay", fontsize=9, weight="bold")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7.5)
    axes[0].legend(fontsize=5.6, ncol=2, loc="lower left", handlelength=1.2)
    savefig(fig, "fig_c_env_heatmaps")


def plot_native_heatmap() -> None:
    summary = load_json(NATIVE_SUMMARY)
    values = np.full((len(AGES), len(NATIVE_ENVS)), np.nan)
    deltas = np.full((len(AGES), len(NATIVE_ENVS)), np.nan)
    for row in summary["rows"]:
        if row["status"] != "completed" or str(row["env_name"]) not in NATIVE_ENVS:
            continue
        j = AGES.index(int(row["age"]))
        i = NATIVE_ENVS.index(str(row["env_name"]))
        full = float(row["full"]["executed_success"]["mean"])
        recent = float(row["recent"]["executed_success"]["mean"])
        rnd = float(row["random"]["executed_success"]["mean"])
        values[j, i] = full
        deltas[j, i] = full - max(recent, rnd)
    fig, ax = plt.subplots(figsize=(6.4, 1.95))
    fig.patch.set_facecolor(PI["white"])
    im = ax.imshow(values, cmap=PI_HEAT, vmin=0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(NATIVE_ENVS)))
    ax.set_xticklabels([env_label(e) for e in NATIVE_ENVS], fontsize=7, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(AGES)))
    ax.set_yticklabels([str(a) for a in AGES], fontsize=7.5)
    ax.set_ylabel("Age", fontsize=8)
    for j in range(len(AGES)):
        for i in range(len(NATIVE_ENVS)):
            v = values[j, i]
            if not np.isnan(v):
                ax.text(i, j, f"{v:.2f}\n+{deltas[j, i]:.2f}", ha="center", va="center", fontsize=5.2, color=PI["white"] if v > 0.75 else PI["ink"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.01)
    cbar.set_label("exec. success", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.2)
    savefig(fig, "fig_c_native_heatmap")


def plot_auto_age_curve(auto: dict[str, Any]) -> None:
    ages = [int(row["age"]) for row in auto["age_rows"]]
    full = [float(row["mean_full"]) for row in auto["age_rows"]]
    reset = [float(row["mean_reset"]) for row in auto["age_rows"]]
    no_state = [float(row["mean_no_state"]) for row in auto["age_rows"]]
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    ax.plot(ages, full, marker="o", lw=2.2, color=PI["black"], label="full memory")
    ax.plot(ages, reset, marker="s", lw=1.6, color=PI["gray2"], label="reset")
    ax.plot(ages, no_state, marker="^", lw=1.6, color="#8a5d00", label="no-state")
    for age in AGES:
        ax.axvline(age, color=PI["yellow_deep"], lw=0.8, alpha=0.45)
    ax.axhline(0.75, color=PI["bad"], lw=0.9, ls="--", label="gate")
    ax.set_xticks(ages)
    ax.set_xlabel("Evaluation evidence age", fontsize=8.5)
    ax.set_ylabel("Balanced accuracy", fontsize=8.5)
    ax.set_ylim(0.15, 1.04)
    ax.set_title("Mixed-age training generalizes to held-out ages", fontsize=10, weight="bold")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(loc="center right", fontsize=7.2)
    fig.tight_layout(pad=0.25)
    savefig(fig, "fig_c_auto_age_curve")


def plot_auto_age_heatmap(auto: dict[str, Any]) -> None:
    by = {(row["env"], row["age"]): row for row in auto["env_age_rows"]}
    margins = np.zeros((len(ENV_ORDER), len(AUTO_EVAL_AGES)), dtype=np.float64)
    for i, env in enumerate(ENV_ORDER):
        for j, age in enumerate(AUTO_EVAL_AGES):
            margins[i, j] = float(by[(env, age)]["margin"])

    fig, ax = plt.subplots(figsize=(6.5, 3.05))
    im = ax.imshow(margins, cmap=PI_GAP, vmin=0.55, vmax=0.80, aspect="auto")
    ax.set_xticks(np.arange(len(AUTO_EVAL_AGES)))
    ax.set_xticklabels([str(age) for age in AUTO_EVAL_AGES], fontsize=7.2)
    ax.set_yticks(np.arange(len(ENV_ORDER)))
    ax.set_yticklabels([env_label(env) for env in ENV_ORDER], fontsize=7.0)
    ax.set_xlabel("Evaluation age", fontsize=8)
    ax.set_title("Auto-age audit margin: full - max(reset, no-state)", fontsize=10, weight="bold")
    for i in range(len(ENV_ORDER)):
        for j in range(len(AUTO_EVAL_AGES)):
            value = margins[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.2, color=PI["white"] if value > 0.70 else PI["ink"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.012)
    cbar.set_label("BAcc margin", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=0.25)
    savefig(fig, "fig_c_auto_age_heatmap")


# --------------------------------------------------------------------------- #
def convert_architecture_svg() -> None:
    svg = FIGURES / "fig_c_architecture.svg"
    if not svg.exists():
        return
    try:
        import cairosvg  # type: ignore
    except Exception:
        print("cairosvg unavailable; keeping existing fig_c_architecture.pdf")
        return
    cairosvg.svg2pdf(url=str(svg), write_to=str(FIGURES / "fig_c_architecture.pdf"))
    cairosvg.svg2png(url=str(svg), write_to=str(FIGURES / "fig_c_architecture.png"), output_width=1400)


# --------------------------------------------------------------------------- #
# Revision figures/tables (reviewer response): streaming cost, capacity,
# structured baselines, shortcut robustness, LeWM host, age scaling.
# Each reads local artifacts with graceful fallback if a sweep is incomplete.
# --------------------------------------------------------------------------- #
REV = ROOT / "outputs" / "paper_c_revision_v1"
STREAM_COST = ROOT / "outputs" / "streaming_cost_v1" / "streaming_cost.json"
LEWM_HOST = ROOT / "outputs" / "lewm_pusht_host_writer_counterfactual_checkpointed_v1" / "summary.json"
AGESCALE = ROOT / "outputs" / "paper_c_agescale_v1" / "pointmaze-large-navigate-v0" / "auto_age" / "s0" / "result.json"

REV_METHOD_LABELS = {
    "slot": "Slot (ours)", "gru": "GRU", "lstm": "LSTM", "mamba_lite": "Mamba-lite",
    "slotssm": "SlotSSM", "gsa": "GSA", "retrieval": "Top-k retr.",
    "txl": "Transf.-XL", "parallel_gru": "8x GRU", "rec_queries": "GRU+queries",
}


def _rev_cells(root: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(root.glob("*/*/age_*/s*/result.json")):
        try:
            d = load_json(p)
        except Exception:
            continue
        r = d.get("readout", {})
        if not {"full", "reset", "no_state"}.issubset(r):
            continue
        rows.append({
            "baseline": d.get("baseline", "slot"),
            "env": d["env_name"], "age": int(d["age"]),
            "full": float(r["full"]["balanced_accuracy"]),
            "reset": float(r["reset"]["balanced_accuracy"]),
            "no_state": float(r["no_state"]["balanced_accuracy"]),
            "state_scalars": int(d.get("state_scalars", 0) or 0),
            "param_count": int(d.get("param_count", 0) or 0),
            "streaming": bool(d.get("streaming", False)),
            "cue_mode": d.get("cue_mode", "color"),
        })
    return rows


def plot_streaming_cost() -> None:
    if not STREAM_COST.is_file():
        return
    data = load_json(STREAM_COST)
    rows = data["rows"]
    L = [r["length"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5), constrained_layout=True)
    fig.patch.set_facecolor(PI["white"])
    ax = axes[0]
    ax.plot(L, [r["one_shot_peak_mb"] for r in rows], "o-", color=PI["gray2"], lw=2, label="one-shot (full prefix)")
    ax.plot(L, [r["streaming_peak_mb"] for r in rows], "o-", color=PI["black"],
            markerfacecolor=PI["yellow"], markeredgecolor=PI["ink"], lw=2, label="streaming (evict)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length", fontsize=8.5)
    ax.set_ylabel("Peak activation (MB)", fontsize=8.5)
    ax.set_title("(a) peak memory", fontsize=9, weight="bold")
    ax.legend(fontsize=6.6, loc="upper left")
    ax = axes[1]
    ax.plot(L, [r["one_shot_time_ms"] for r in rows], "o-", color=PI["gray2"], lw=2, label="one-shot / call")
    ax.plot(L, [r["streaming_per_step_ms"] for r in rows], "o-", color=PI["black"],
            markerfacecolor=PI["yellow"], markeredgecolor=PI["ink"], lw=2, label="streaming / step")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length", fontsize=8.5)
    ax.set_ylabel("Compute (ms)", fontsize=8.5)
    ax.set_title("(b) per-step compute is constant", fontsize=9, weight="bold")
    ax.legend(fontsize=6.6, loc="center left")
    for ax in axes:
        ax.grid(alpha=0.3); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7)
    savefig(fig, "fig_c_streaming_cost")


def plot_lewm_host() -> None:
    if not LEWM_HOST.is_file():
        return
    d = load_json(LEWM_HOST)
    task = next(iter(d["tasks"].values()))
    cell = task.get("15") or next(iter(task.values()))
    vals = [cell["full_mean"], cell["reset_mean"], cell["no_state_mean"]]
    labels = ["full", "reset", "no-state"]
    colors = [PI["black"], PI["gray"], "#c4b5a5"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.patch.set_facecolor(PI["white"])
    bars = ax.bar(range(3), vals, color=colors, edgecolor=PI["ink"], linewidth=0.85, width=0.66)
    bars[0].set_edgecolor(PI["yellow_deep"]); bars[0].set_linewidth(1.6)
    ax.axhline(cell["gate"]["full_minimum"], color=PI["bad"], ls="--", lw=1.0)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Host-output BAcc", fontsize=8.5)
    ax.set_title("Frozen LeWM host: exposed\nold evidence (age 15, 5 seeds)", fontsize=8.5, weight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout(pad=0.3)
    savefig(fig, "fig_c_lewm_host")


def _agg_rev(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["baseline"]].append(r)
    out = {}
    for bl, vs in by.items():
        out[bl] = {
            "full": mean(v["full"] for v in vs),
            "reset": mean(v["reset"] for v in vs),
            "no_state": mean(v["no_state"] for v in vs),
            "state_scalars": vs[0]["state_scalars"],
            "n": len(vs),
        }
    return out


def plot_capacity_and_structured() -> None:
    rows = _rev_cells(REV / "stream_age15_s0")
    if len(rows) < 4:
        return
    # Compare over environments present for every method (fair aggregation).
    envs_by_method: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        envs_by_method[r["baseline"]].add(r["env"])
    if len(envs_by_method) >= 2:
        common = set.intersection(*envs_by_method.values())
        if common:
            rows = [r for r in rows if r["env"] in common]
    agg = _agg_rev(rows)
    order = [b for b in ["slot", "slotssm", "gsa", "retrieval", "txl", "parallel_gru",
                          "rec_queries", "gru", "lstm", "mamba_lite"] if b in agg]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.7), constrained_layout=True)
    fig.patch.set_facecolor(PI["white"])
    # (a) structured/arch comparison bar (age 15 full BAcc)
    ax = axes[0]
    vals = [agg[b]["full"] for b in order]
    colors = [PI["black"] if b == "slot" else PI["yellow_deep"] if b in {"slotssm", "gsa", "retrieval", "txl"} else PI["gray2"] for b in order]
    bars = ax.bar(range(len(order)), vals, color=colors, edgecolor=PI["ink"], linewidth=0.7, width=0.74)
    ax.axhline(0.75, color=PI["bad"], ls="--", lw=1.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([REV_METHOD_LABELS.get(b, b) for b in order], fontsize=6.6, rotation=32, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Full BAcc (age 15)", fontsize=8.5)
    ax.set_title("(a) structured + capacity baselines", fontsize=9, weight="bold")
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    # (b) accuracy vs exposed state scalars
    ax = axes[1]
    for b in order:
        x = max(1, agg[b]["state_scalars"])
        ax.scatter([x], [agg[b]["full"]], s=70,
                   color=PI["black"] if b == "slot" else PI["yellow_deep"],
                   edgecolor=PI["ink"], zorder=3, marker="o" if b == "slot" else "D")
        ax.annotate(REV_METHOD_LABELS.get(b, b), (x, agg[b]["full"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Exposed state scalars", fontsize=8.5)
    ax.set_ylabel("Full BAcc (age 15)", fontsize=8.5)
    ax.set_title("(b) accuracy vs memory capacity", fontsize=9, weight="bold")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7)
    savefig(fig, "fig_c_capacity_structured")

    # structured baseline table
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Memory & Full & Reset & No-state & State scalars\\\\", "\\midrule"]
    for b in order:
        a = agg[b]
        row = f"{REV_METHOD_LABELS.get(b, b)} & {a['full']:.3f} & {a['reset']:.3f} & {a['no_state']:.3f} & {a['state_scalars']}\\\\"
        if b == "slot":
            row = "\\rowcolor{MemCream}" + row
        lines.append(row)
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (GENERATED / "structured_baselines_tabular.tex").write_text("\n".join(lines))


def plot_shortcut() -> None:
    rows = _rev_cells(REV / "shortcut_shape_age15_s0")
    if len(rows) < 3:
        return
    agg = _agg_rev(rows)
    order = [b for b in ["slot", "gru", "retrieval"] if b in agg]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    fig.patch.set_facecolor(PI["white"])
    x = np.arange(len(order)); w = 0.26
    ax.bar(x - w, [agg[b]["full"] for b in order], w, label="full", color=PI["black"], edgecolor=PI["ink"], linewidth=0.7)
    ax.bar(x, [agg[b]["reset"] for b in order], w, label="reset", color=PI["gray"], edgecolor=PI["ink"], linewidth=0.7)
    ax.bar(x + w, [agg[b]["no_state"] for b in order], w, label="no-state", color="#c4b5a5", edgecolor=PI["ink"], linewidth=0.7)
    ax.axhline(0.75, color=PI["bad"], ls="--", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([REV_METHOD_LABELS.get(b, b) for b in order], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("BAcc", fontsize=8.5)
    ax.set_title("Randomized-colour identity cue\n(colour shortcut removed)", fontsize=8.5, weight="bold")
    ax.legend(fontsize=6.6, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout(pad=0.3)
    savefig(fig, "fig_c_shortcut")


def plot_age_scaling() -> None:
    root = ROOT / "outputs" / "paper_c_agescale_v1"
    pts = sorted(root.glob("*/auto_age/s*/result.json"))
    if not pts:
        return
    # one curve per environment, plus a mean
    per_env: dict[str, list[tuple[int, float]]] = {}
    train_ages: set[int] = set()
    reset_by_age: dict[int, list[float]] = defaultdict(list)
    for p in pts:
        d = load_json(p)
        env = d["env_name"]
        train_ages |= set(int(a) for a in d.get("train_ages", []))
        pts_env = []
        for e in d["eval"]:
            a = int(e["age"])
            pts_env.append((a, float(e["readout"]["full"]["balanced_accuracy"])))
            reset_by_age[a].append(float(e["readout"]["reset"]["balanced_accuracy"]))
        per_env[env] = sorted(pts_env)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    fig.patch.set_facecolor(PI["white"])
    cmap = plt.get_cmap("cividis")
    envs = list(per_env)
    for i, env in enumerate(envs):
        xs = [a for a, _ in per_env[env]]
        ys = [f for _, f in per_env[env]]
        ax.plot(xs, ys, "-o", ms=4, lw=1.6, color=cmap(i / max(1, len(envs) - 1)), label=env_label(env))
    ages_sorted = sorted(reset_by_age)
    ax.plot(ages_sorted, [mean(reset_by_age[a]) for a in ages_sorted], "--", color=PI["gray"], lw=1.2, label="reset (mean)")
    ax.axhline(0.75, color=PI["bad"], ls="--", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({a for e in per_env.values() for a, _ in e}))
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Evidence age (delay, log scale)", fontsize=8.5)
    ax.set_ylabel("Full-memory BAcc", fontsize=8.5)
    ax.set_ylim(0.15, 1.03)
    ax.set_title("Long-delay scaling to age 128 (streaming, per env)", fontsize=8.5, weight="bold")
    ax.legend(fontsize=6.2, loc="lower left", ncol=2, handlelength=1.3)
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout(pad=0.3)
    savefig(fig, "fig_c_age_scaling")


def write_ablation_tables() -> None:
    """Slot-count (carrier) and loss-component ablations from launch_paper_c_ablations."""
    abl = REV / "ablations"
    if not abl.is_dir():
        return
    def _read(tag: str) -> dict[str, float] | None:
        cells = list((abl / tag).glob("*/age_*/s*/result.json"))
        if not cells:
            return None
        full, reset, nostate = [], [], []
        for p in cells:
            try:
                d = load_json(p)
            except Exception:
                continue
            r = d["readout"]
            full.append(r["full"]["balanced_accuracy"])
            reset.append(r["reset"]["balanced_accuracy"])
            nostate.append(r["no_state"]["balanced_accuracy"])
        if not full:
            return None
        return {"full": mean(full), "reset": mean(reset), "no_state": mean(nostate), "n": len(full)}

    # Slot-count / carrier ablation (S=1 is the single-vector carrier cell).
    slot_rows = []
    for s in (1, 2, 4, 8, 16):
        r = _read(f"slots{s}")
        if r:
            slot_rows.append((s, r))
    if slot_rows:
        lines = ["\\begin{tabular}{lccc}", "\\toprule",
                 "Slots $S$ & Full & Reset & No-state\\\\", "\\midrule"]
        for s, r in slot_rows:
            tag = f"{s}" + (" (single vector)" if s == 1 else "")
            row = f"{tag} & {r['full']:.3f} & {r['reset']:.3f} & {r['no_state']:.3f}\\\\"
            if s == 8:
                row = "\\rowcolor{MemCream}" + row
            lines.append(row)
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        (GENERATED / "ablation_slots_tabular.tex").write_text("\n".join(lines))

    # Loss-component ablation.
    loss_map = [("slots8", "full loss (NCE+cos+std)"), ("loss_nce_only", "InfoNCE only"),
                ("loss_no_cos", "no cosine"), ("loss_no_std", "no std reg."),
                ("stream_k4", "streaming ($K{=}4$)")]
    loss_rows = [(lbl, _read(tag)) for tag, lbl in loss_map]
    loss_rows = [(lbl, r) for lbl, r in loss_rows if r]
    if loss_rows:
        lines = ["\\begin{tabular}{lccc}", "\\toprule",
                 "Configuration & Full & Reset & No-state\\\\", "\\midrule"]
        for lbl, r in loss_rows:
            row = f"{lbl} & {r['full']:.3f} & {r['reset']:.3f} & {r['no_state']:.3f}\\\\"
            if lbl.startswith("full loss"):
                row = "\\rowcolor{MemCream}" + row
            lines.append(row)
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        (GENERATED / "ablation_loss_tabular.tex").write_text("\n".join(lines))


def run_revision_figures() -> None:
    for fn in (plot_streaming_cost, plot_lewm_host, plot_capacity_and_structured, plot_shortcut, plot_age_scaling, write_ablation_tables):
        try:
            fn()
        except Exception as exc:  # keep the main build resilient to partial sweeps
            print(f"[revision-fig] {fn.__name__} skipped: {exc}")


def main() -> None:
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    cells = cell_rows()
    snapshot = method_summary(cells)
    auto = auto_age_summary(load_auto_age_rows())
    snapshot["native_use"] = load_json(NATIVE_STATS)
    snapshot["auto_age"] = auto
    (GENERATED / "result_snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")

    index = index_summary(summary_rows())

    write_main_table(snapshot)
    write_age_table(snapshot)
    write_native_table()
    write_auto_age_table(auto)
    write_auto_age_full_table(auto)
    write_retention_by_age(index)
    write_retrieval_table(index)
    write_seed_pass_table(index)
    write_native_full_table()
    write_hyperparam_table()

    convert_architecture_svg()
    plot_shortcut_diagnostics()
    plot_results(snapshot)
    plot_age_and_autoage(snapshot, auto)
    plot_env_heatmaps(cells)
    plot_native_heatmap()
    run_revision_figures()

    unseen = [r for r in auto["age_rows"] if not r["trained_age"]]
    print(json.dumps({
        "slot_better_than_best_baseline": snapshot["slot_better_than_best_baseline"],
        "autoage_unseen_pass": f"{sum(r['pass_count'] for r in unseen)}/{sum(r['seed_cells'] for r in unseen)}",
        "autoage_unseen_mean_full": round(mean(r["mean_full"] for r in unseen), 3),
    }, indent=2))


if __name__ == "__main__":
    main()
