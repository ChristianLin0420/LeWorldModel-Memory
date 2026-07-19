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


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 2.6))
    ax.axis("off")
    boxes = [
        (0.04, 0.55, "Rendered stream\\nframes/actions/time"),
        (0.27, 0.55, "Frozen encoder\\npatch tokens"),
        (0.50, 0.55, "Slot memory writer\\n8 compact slots"),
        (0.73, 0.55, "Masked target JEPA\\npatch-set prediction"),
        (0.50, 0.12, "Audit readout\\nfull vs reset/no-state"),
    ]
    for x, y, text in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), 0.19, 0.28, facecolor="#f5f4ef", edgecolor="#111827", lw=1.2)
        )
        ax.text(x + 0.095, y + 0.14, text, ha="center", va="center", fontsize=9)
    arrows = [
        ((0.23, 0.69), (0.27, 0.69)),
        ((0.46, 0.69), (0.50, 0.69)),
        ((0.69, 0.69), (0.73, 0.69)),
        ((0.595, 0.55), (0.595, 0.40)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#111827"})
    ax.text(
        0.5,
        0.96,
        "Labels are withheld during training and used only after training for the audit.",
        ha="center",
        va="top",
        fontsize=9,
        color="#656760",
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "fig_c_architecture.pdf")
    fig.savefig(FIGURES / "fig_c_architecture.png", dpi=180)
    plt.close(fig)


def plot_results(snapshot: dict[str, Any]) -> None:
    labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    x = np.arange(len(labels))
    mean_full = [snapshot["methods"][m]["mean_full"] for m in METHOD_ORDER]
    age15 = [snapshot["methods"][m]["age"]["15"]["mean_full"] for m in METHOD_ORDER]
    pass_rows = [snapshot["methods"][m]["allpass_rows"] for m in METHOD_ORDER]
    colors = ["#111827", "#7a7f87", "#9ca3af", "#d8a900"]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
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
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_c_baseline_comparison.pdf")
    fig.savefig(FIGURES / "fig_c_baseline_comparison.png", dpi=180)
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
    plot_architecture()
    plot_results(snapshot)
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
