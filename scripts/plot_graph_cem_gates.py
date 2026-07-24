#!/usr/bin/env python3
"""Generate exact Graph-CEM gate figures and the campaign machine report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONDITIONAL = ROOT / "outputs/graph_cem_conditional_v1/report.json"
LONG_GAP = ROOT / "outputs/graph_cem_long_gap_v1/report.json"
OUTPUT = ROOT / "outputs/graph_cem_report.json"
ASSETS = ROOT / "docs/assets"

INK = "#171717"
BLUE = "#2563eb"
ORANGE = "#d97706"
GREEN = "#15803d"
RED = "#b91c1c"
PURPLE = "#7e22ce"
GRAY = "#6b7280"
LIGHT = "#d1d5db"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, name: str) -> list[str]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, options in (("png", {"dpi": 220}), ("pdf", {})):
        path = ASSETS / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **options)
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def short_env(name: str) -> str:
    if name.startswith("pointmaze"):
        return "PointMaze-large"
    if name.startswith("cube"):
        return "Cube-single"
    if name.startswith("puzzle"):
        return "Puzzle-3x3"
    return name


def conditional_figure(report: dict[str, Any]) -> list[str]:
    environments = report["environments"]
    labels = [short_env(row["environment"]) for row in environments]
    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.55))
    specs = (
        ("spearman", "Within-query Spearman", 0.2),
        ("pairwise", "Pairwise ranking accuracy", 0.65),
        ("deletion_gap", "High − random conditional deletion", 0.0),
    )
    for axis, (metric, title, threshold) in zip(axes, specs):
        for offset, method, color, label in (
            (-width / 2, "singleton", GRAY, "singleton CE"),
            (width / 2, "conditional", BLUE, "conditional CE"),
        ):
            rows = [row["methods"][method][metric] for row in environments]
            values = np.asarray([row["mean"] for row in rows])
            low = values - np.asarray([row["ci95"][0] for row in rows])
            high = np.asarray([row["ci95"][1] for row in rows]) - values
            axis.bar(
                x + offset,
                values,
                width,
                color=color,
                alpha=0.9,
                label=label,
            )
            axis.errorbar(
                x + offset,
                values,
                yerr=np.stack([low, high]),
                fmt="none",
                ecolor=INK,
                capsize=2,
                lw=0.8,
            )
        axis.axhline(
            threshold,
            color=RED if threshold else INK,
            ls=":",
            lw=1.1,
            label="gate threshold" if metric != "deletion_gap" else "zero",
        )
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color=LIGHT, alpha=0.45)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Flat conditional-target diagnostic · 3 seeds per environment",
        x=0.04,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, "graph_cem_conditional_calibration")


def oracle_ladder_figure(report: dict[str, Any]) -> list[str]:
    gaps = np.asarray([row["gap"] for row in report["aggregate_gaps"]])
    methods = (
        ("oracle_frame", "oracle frame", BLUE),
        ("oracle_automatic_node", "oracle discovered node", GREEN),
        ("oracle_event_set", "oracle event set", PURPLE),
        ("conditional_ce", "conditional CE", ORANGE),
        ("singleton_ce", "singleton CE", GRAY),
        ("surprise", "surprise", RED),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    for method, label, color in methods:
        rows = [row["methods"][method] for row in report["aggregate_gaps"]]
        values = np.asarray([row["paired_gain_vs_recent"] for row in rows])
        low = values - np.asarray([row["ci95"][0] for row in rows])
        high = np.asarray([row["ci95"][1] for row in rows]) - values
        axes[0].errorbar(
            gaps,
            values,
            yerr=np.stack([low, high]),
            marker="o",
            ms=4,
            capsize=2,
            lw=1.4,
            color=color,
            label=label,
        )
    axes[0].axhline(0, color=INK, ls=":", lw=1)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(gaps, [str(value) for value in gaps])
    axes[0].set_xlabel("Event-to-query gap (frames)")
    axes[0].set_ylabel("Paired host-loss gain vs recent-only")
    axes[0].set_title("Equal-byte oracle and selector ladder", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color=LIGHT, alpha=0.45)
    axes[0].legend(fontsize=7, ncol=2)

    recovery = np.asarray(
        [row["automatic_node_recovery"] for row in report["aggregate_gaps"]]
    )
    closure = np.asarray(
        [row["conditional_selection_closure"] for row in report["aggregate_gaps"]]
    )
    axes[1].plot(
        gaps,
        recovery,
        marker="o",
        color=GREEN,
        label="automatic-node recovery",
    )
    axes[1].plot(
        gaps,
        closure,
        marker="s",
        color=ORANGE,
        label="conditional selection closure",
    )
    axes[1].axhline(0.70, color=RED, ls=":", lw=1, label="Gate 2 recovery")
    axes[1].axhline(0, color=INK, ls="--", lw=0.8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(gaps, [str(value) for value in gaps])
    axes[1].set_xlabel("Event-to-query gap (frames)")
    axes[1].set_ylabel("Fraction of oracle gain")
    axes[1].set_title("Discovery and learned-selection closure", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color=LIGHT, alpha=0.45)
    axes[1].legend(fontsize=7)
    fig.suptitle(
        "Controlled suffix collision · exact recent suffixes · 3 envs × 3 seeds",
        x=0.04,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, "graph_cem_oracle_ladder_vs_gap")


def gate_dashboard(
    conditional: dict[str, Any],
    long_gap: dict[str, Any],
) -> list[str]:
    gate1 = conditional["gate"]
    gate2 = long_gap["gate"]
    values = [
        conditional["aggregate"]["conditional"]["spearman"]["ci95"][0],
        conditional["aggregate"]["conditional"]["pairwise"]["mean"],
        gate1["deletion_criterion"]["count"],
        gate2["automatic_discovery_criterion"]["high_gap_recovery"],
    ]
    thresholds = [0.2, 0.65, 2.0, 0.70]
    labels = [
        "Spearman\nlower CI",
        "Pairwise\naccuracy",
        "Positive deletion\nenvironments",
        "High-gap automatic\nnode recovery",
    ]
    normalized = np.asarray(values) / np.asarray(thresholds)
    colors = [GREEN if value >= 1 else RED for value in normalized]
    fig, axes = plt.subplots(
        1, 2, figsize=(10.7, 3.75), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    x = np.arange(len(labels))
    axes[0].bar(x, normalized, color=colors)
    axes[0].axhline(1, color=INK, ls=":", lw=1.1, label="pass threshold")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Observed / required")
    axes[0].set_title("Hard-gate metrics", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color=LIGHT, alpha=0.45)
    for index, (raw, threshold) in enumerate(zip(values, thresholds)):
        axes[0].text(
            index,
            normalized[index] + 0.04,
            f"{raw:.3f} / {threshold:g}",
            ha="center",
            fontsize=7,
        )
    axes[0].legend(fontsize=7)

    phases = ["1\nConditional", "2\nOracle", "3\nGraph", "4\nEdges", "5–7\nScale/use"]
    status = ["FAIL", "FAIL", "STOP", "N/R", "N/R"]
    status_colors = [RED, RED, RED, GRAY, GRAY]
    axes[1].scatter(
        np.arange(len(phases)),
        np.ones(len(phases)),
        s=750,
        c=status_colors,
        marker="s",
    )
    for index, text in enumerate(status):
        axes[1].text(index, 1, text, ha="center", va="center", color="white", weight="bold")
    axes[1].plot(np.arange(len(phases)), np.ones(len(phases)), color=LIGHT, zorder=0)
    axes[1].set_xticks(np.arange(len(phases)), phases)
    axes[1].set_yticks([])
    axes[1].set_ylim(0.75, 1.25)
    axes[1].set_title("Campaign stop/go outcome", loc="left", fontweight="bold")
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    fig.suptitle(
        "Graph-CEM gate dashboard · graph implementation hard-stopped",
        x=0.04,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, "graph_cem_gate_dashboard")


def machine_report(
    conditional: dict[str, Any],
    long_gap: dict[str, Any],
    figures: list[str],
) -> dict[str, Any]:
    phase1_pass = bool(conditional["gate"]["passed"])
    phase2_pass = bool(long_gap["gate"]["passed"])
    stop_reason = (
        "Gate 1 failed causal ranking and deletion discrimination; Gate 2 "
        "showed oracle opportunity but missed automatic-discovery recovery."
    )
    return {
        "schema": "graph_cem_campaign_report_v1",
        "status": "completed",
        "source_commit": "660cfe763c9e49f6ea30cf841003929b0e4a9c33",
        "gpus_allowed": [0, 1, 2],
        "gpu3_used": False,
        "phases": {
            "1_flat_conditional_target": {
                "status": "completed",
                "gate": conditional["gate"],
                "source": str(CONDITIONAL.relative_to(ROOT)),
            },
            "2_suffix_collision_oracle": {
                "status": "completed",
                "gate": long_gap["gate"],
                "source": str(LONG_GAP.relative_to(ROOT)),
            },
            "3_simple_graph": {
                "status": "not_run_hard_gate",
                "reason": stop_reason,
            },
            "4_edge_structure_ablation": {
                "status": "not_run_hard_gate",
                "reason": "No graph was built.",
            },
            "5_scale_hierarchical": {
                "status": "not_run_hard_gate",
                "reason": "Graph gate was not reached.",
            },
            "6_raw_official_dinowm": {
                "status": "not_run_hard_gate",
                "reason": "Graph and prediction gates did not pass.",
            },
            "7_executed_control": {
                "status": "not_run_hard_gate",
                "reason": "Prediction and graph gates did not pass.",
            },
        },
        "gate_summary": {
            "gate_1_passed": phase1_pass,
            "gate_2_passed": phase2_pass,
            "both_mandatory_gates_passed": phase1_pass and phase2_pass,
            "graph_built": False,
            "graph_vs_flat_same_nodes": None,
        },
        "recommendation": {
            "decision": "stop_due_selection_and_discovery_headroom",
            "text": (
                "Do not build Graph-CEM or prefer the current flat conditional "
                "head. Preserve the demonstrated oracle opportunity, add a "
                "sparse raw-frame fallback to automatic candidates, and solve "
                "query-conditioned value ranking before reconsidering edges."
            ),
            "top_next_step": (
                "Re-run a flat frame-plus-event selector with cross-fitted "
                "long-horizon conditional targets; require Gate 1 and >=70% "
                "automatic recovery before any graph implementation."
            ),
        },
        "jobs_still_running": [],
        "artifacts": {
            "conditional_report": str(CONDITIONAL.relative_to(ROOT)),
            "long_gap_report": str(LONG_GAP.relative_to(ROOT)),
            "figures": figures,
        },
    }


def main() -> None:
    style()
    conditional = json.loads(CONDITIONAL.read_text())
    long_gap = json.loads(LONG_GAP.read_text())
    figures = []
    figures += conditional_figure(conditional)
    figures += oracle_ladder_figure(long_gap)
    figures += gate_dashboard(conditional, long_gap)
    report = machine_report(conditional, long_gap, figures)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "graph_cem_figure_receipt_v1",
        "conditional_report_sha256": sha256_file(CONDITIONAL),
        "long_gap_report_sha256": sha256_file(LONG_GAP),
        "figures": figures,
        "white_background": True,
    }
    receipt_path = LONG_GAP.parent / "figure_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
