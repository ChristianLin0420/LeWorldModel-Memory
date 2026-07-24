#!/usr/bin/env python3
"""Plot fallback-selector results and finalize its machine decision report."""
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
SOURCE = ROOT / "outputs/cem_fallback_selector_v1/report.json"
GRAPH_SOURCE = ROOT / "outputs/graph_cem_report.json"
OUTPUT = ROOT / "outputs/cem_fallback_report.json"
ASSETS = ROOT / "docs/assets"

INK = "#171717"
BLUE = "#2563eb"
GREEN = "#15803d"
ORANGE = "#d97706"
PURPLE = "#7e22ce"
RED = "#b91c1c"
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
    if name.startswith("cube"):
        return "Cube-single"
    if name.startswith("pointmaze"):
        return "PointMaze-large"
    if name.startswith("puzzle"):
        return "Puzzle-3x3"
    return name


def ladder(report: dict[str, Any]) -> list[str]:
    gaps = np.asarray([row["gap"] for row in report["aggregate_gaps"]])
    specs = (
        ("oracle_frame_vs_recent", "oracle frame", BLUE),
        ("oracle_union_vs_recent", "oracle frame+event union", PURPLE),
        ("fallback_vs_recent", "learned fallback", GREEN),
        ("event_vs_recent", "event-only CE", ORANGE),
        ("surprise_vs_recent", "surprise", RED),
        ("random_vs_recent", "random", GRAY),
    )
    fig, axis = plt.subplots(figsize=(7.8, 4.1))
    for key, label, color in specs:
        rows = [row["comparisons"][key] for row in report["aggregate_gaps"]]
        value = np.asarray([row["mean"] for row in rows])
        low = value - np.asarray([row["ci95"][0] for row in rows])
        high = np.asarray([row["ci95"][1] for row in rows]) - value
        axis.errorbar(
            gaps,
            value,
            yerr=np.stack([low, high]),
            marker="o",
            capsize=2,
            lw=1.4,
            ms=4,
            color=color,
            label=label,
        )
    axis.axhline(0, color=INK, ls=":", lw=1)
    axis.set_xscale("log", base=2)
    axis.set_xticks(gaps, [str(value) for value in gaps])
    axis.set_xlabel("Event-to-query gap (frames)")
    axis.set_ylabel("Paired host-loss gain vs recent-only")
    axis.set_title(
        "Equal-byte frame+event fallback ladder",
        loc="left",
        fontweight="bold",
    )
    axis.grid(axis="y", color=LIGHT, alpha=0.45)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return save(fig, "cem_fallback_candidate_oracle_ladder")


def calibration(report: dict[str, Any]) -> list[str]:
    environments = report["environments"]
    labels = [short_env(row["environment"]) for row in environments]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.6))
    for axis, metric, title, threshold in (
        (axes[0], "spearman", "Within-store Spearman", 0.2),
        (axes[1], "pairwise", "Pairwise accuracy", 0.65),
        (axes[2], "deletion_gap", "High − random deletion", 0.0),
    ):
        rows = [row["ranking"][metric] for row in environments]
        value = np.asarray([row["mean"] for row in rows])
        low = value - np.asarray([row["ci95"][0] for row in rows])
        high = np.asarray([row["ci95"][1] for row in rows]) - value
        axis.bar(x, value, color=BLUE)
        axis.errorbar(
            x,
            value,
            yerr=np.stack([low, high]),
            fmt="none",
            ecolor=INK,
            capsize=2,
        )
        axis.axhline(
            threshold,
            color=RED if threshold else INK,
            ls=":",
        )
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color=LIGHT, alpha=0.45)
    nominal = np.asarray([0.5, 0.8, 0.9])
    for environment, color in zip(
        environments, (BLUE, ORANGE, GREEN)
    ):
        cells = [
            json.loads(path.read_text())
            for path in sorted(
                (
                    ROOT
                    / "outputs/cem_fallback_selector_v1/cells"
                    / environment["environment"]
                ).glob("s*/result.json")
            )
        ]
        empirical = np.asarray(
            [
                np.mean(
                    [
                        cell["test_uncertainty"]["interval_coverage"][
                            str(value)
                        ]
                        for cell in cells
                    ]
                )
                for value in nominal
            ]
        )
        axes[3].plot(
            nominal,
            empirical,
            marker="o",
            color=color,
            label=short_env(environment["environment"]),
        )
    axes[3].plot(nominal, nominal, color=INK, ls=":", label="ideal")
    axes[3].set_xlabel("Nominal interval coverage")
    axes[3].set_ylabel("Empirical coverage")
    axes[3].set_title(
        "Ensemble uncertainty calibration",
        loc="left",
        fontweight="bold",
    )
    axes[3].legend(fontsize=6.5)
    axes[3].grid(color=LIGHT, alpha=0.45)
    fig.suptitle(
        "Cross-fitted conditional ranking and uncertainty",
        x=0.03,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, "cem_fallback_calibration_ranking")


def selections(report: dict[str, Any]) -> list[str]:
    rows = report["aggregate_gaps"]
    gaps = np.asarray([row["gap"] for row in rows])
    frame = np.asarray(
        [
            row["selected_candidate_type_frequency"]["frame"]
            for row in rows
        ]
    )
    event = np.asarray(
        [
            row["selected_candidate_type_frequency"]["event"]
            for row in rows
        ]
    )
    recent = np.asarray(
        [
            row["selected_candidate_type_frequency"]["recent"]
            for row in rows
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    axes[0].bar(gaps, event, color=ORANGE, label="event")
    axes[0].bar(gaps, frame, bottom=event, color=BLUE, label="frame")
    axes[0].bar(
        gaps,
        recent,
        bottom=event + frame,
        color=GRAY,
        label="recent",
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(gaps, [str(value) for value in gaps])
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Gap (frames)")
    axes[0].set_ylabel("Selected-token fraction")
    axes[0].set_title(
        "Frame/event fallback selections",
        loc="left",
        fontweight="bold",
    )
    axes[0].legend(fontsize=7)
    proposal_hit = np.asarray(
        [row["proposal_oracle_frame_hit_rate"] for row in rows]
    )
    union_recovery = np.asarray([row["union_recovery"] for row in rows])
    learned_recovery = np.asarray([row["learned_recovery"] for row in rows])
    high = gaps >= 32
    axes[1].plot(
        gaps[high],
        proposal_hit[high],
        marker="o",
        color=BLUE,
        label="oracle-frame proposal hit",
    )
    axes[1].plot(
        gaps[high],
        union_recovery[high],
        marker="s",
        color=PURPLE,
        label="oracle-union recovery",
    )
    axes[1].plot(
        gaps[high],
        learned_recovery[high],
        marker="^",
        color=GREEN,
        label="learned recovery",
    )
    axes[1].axhline(0.70, color=RED, ls=":", label="recovery gate")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(
        gaps[high], [str(value) for value in gaps[high]]
    )
    axes[1].set_xlabel("Gap (frames)")
    axes[1].set_ylabel("Fraction")
    axes[1].set_title(
        "Proposal and selection closure",
        loc="left",
        fontweight="bold",
    )
    axes[1].grid(axis="y", color=LIGHT, alpha=0.45)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    return save(fig, "cem_fallback_frame_event_selections")


def uncertainty_pareto(report: dict[str, Any]) -> list[str]:
    environments = report["environments"]
    labels = [short_env(row["environment"]) for row in environments]
    x = np.arange(len(labels))
    ece = [row["uncertainty"]["coverage_ece_mean"] for row in environments]
    coverage = [
        row["uncertainty"]["coverage_90_mean"] for row in environments
    ]
    ratio = [row["efficiency_ratio_mean"] for row in environments]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].bar(x - 0.18, coverage, 0.36, color=GREEN, label="90% coverage")
    axes[0].bar(x + 0.18, ece, 0.36, color=ORANGE, label="coverage ECE")
    axes[0].axhline(0.9, color=INK, ls=":")
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_title("Uncertainty quality", loc="left", fontweight="bold")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", color=LIGHT, alpha=0.45)
    axes[1].bar(x, ratio, color=BLUE)
    axes[1].axhline(1.5, color=RED, ls=":", label="efficiency gate")
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylabel("Fallback / event-only recall latency")
    axes[1].set_title(
        "Matched-byte non-host recall",
        loc="left",
        fontweight="bold",
    )
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", color=LIGHT, alpha=0.45)
    fig.tight_layout()
    return save(fig, "cem_fallback_uncertainty_pareto")


def machine_report(
    report: dict[str, Any],
    figures: list[str],
) -> dict[str, Any]:
    passed = bool(report["gates"]["all_passed"])
    return {
        "schema": "cem_fallback_campaign_report_v1",
        "status": "completed",
        "source_commit": "660cfe763c9e49f6ea30cf841003929b0e4a9c33",
        "fallback_report": str(SOURCE.relative_to(ROOT)),
        "prior_graph_report": str(GRAPH_SOURCE.relative_to(ROOT)),
        "gates": report["gates"],
        "fallback_beats_recovery_gates": passed,
        "graph_reconsideration": {
            "authorized": passed,
            "status": "pending" if passed else "not_run_hard_gate",
            "reason": (
                "Fallback gates passed; run minimal graph reconsideration."
                if passed
                else "One or more fallback recovery gates failed."
            ),
        },
        "scale_official_control": {
            "status": "not_run_prerequisite"
            if not passed
            else "pending_graph_decision",
        },
        "recommendation": (
            "reconsider_minimal_graph"
            if passed
            else "stop_graph_and_retain_best_flat_baseline"
        ),
        "jobs_still_running": [],
        "artifacts": {
            "figures": figures,
            "cells": "outputs/cem_fallback_selector_v1/cells/<env>/s<seed>",
        },
    }


def main() -> None:
    style()
    report = json.loads(SOURCE.read_text())
    figures = []
    figures += ladder(report)
    figures += calibration(report)
    figures += selections(report)
    figures += uncertainty_pareto(report)
    machine = machine_report(report, figures)
    OUTPUT.write_text(json.dumps(machine, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "cem_fallback_figure_receipt_v1",
        "source_sha256": sha256_file(SOURCE),
        "white_background": True,
        "figures": figures,
    }
    (
        ROOT
        / "outputs/cem_fallback_selector_v1/figure_receipt.json"
    ).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(machine, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
