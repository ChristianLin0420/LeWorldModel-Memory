#!/usr/bin/env python3
"""White-background figures for the native long-trajectory recovery campaign."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_cem_native_long import HORIZON  # noqa: E402

OUTPUT = ROOT / "outputs/cem_native_long_v1"
REPORT = OUTPUT / "report.json"
ASSETS = ROOT / "docs/assets"
CACHE = ROOT / "outputs/paper_c_agescale_v1/cache"
COLORS = ("#1f5a94", "#d2772f", "#3a7d44", "#7c4d9e")


def save(fig: plt.Figure, name: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = ASSETS / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def short_env(name: str) -> str:
    return {
        "pointmaze-large-navigate-v0": "PointMaze",
        "cube-single-play-v0": "Cube",
        "puzzle-3x3-play-v0": "Puzzle",
        "scene-play-v0": "Scene",
    }.get(name, name)


def opportunity_figure(report: dict[str, Any]) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for color, environment in zip(COLORS, report["environments"]):
        gaps = [row["gap"] for row in environment["gaps"]]
        gain = [
            row["oracle_gain_vs_recent"]["mean"]
            for row in environment["gaps"]
        ]
        coverage = [
            100.0 * row["retained_fraction"]["mean"]
            for row in environment["gaps"]
        ]
        label = short_env(environment["environment"])
        axes[0].plot(gaps, gain, "o-", color=color, label=label)
        axes[1].plot(gaps, coverage, "o-", color=color, label=label)
    axes[0].axhline(0.0, color="#444444", linewidth=0.8)
    axes[0].set(
        xlabel="Historical separation (frames)",
        ylabel="Oracle gain vs recent (host MSE)",
        title="Native oracle opportunity",
        xticks=[16, 32, 64, 128],
    )
    axes[1].axhline(20.0, color="#a33a2b", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Historical separation (frames)",
        ylabel="Retained opportunities (%)",
        title="Fixed train-only opportunity filter",
        xticks=[16, 32, 64, 128],
    )
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Unmodified 140-frame OGBench trajectories · 3 seeds per environment",
        fontsize=11,
        weight="bold",
    )
    fig.tight_layout()
    return save(fig, "cem_native_long_opportunity")


def conditioner_figure(report: dict[str, Any]) -> list[str]:
    environments = [
        row
        for row in report["environments"]
        if row["conditioner_recovery"]["mean"] is not None
    ]
    labels = [short_env(row["environment"]) for row in environments]
    recovery = [
        100.0 * row["conditioner_recovery"]["mean"]
        for row in environments
    ]
    degradation = [
        100.0 * row["ordinary_degradation"]["mean"]
        for row in environments
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.5))
    axes[0].bar(x, recovery, color=COLORS[: len(labels)])
    axes[0].axhline(50.0, color="#a33a2b", linestyle="--", label="50% gate")
    axes[0].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Oracle opportunity recovered (%)",
        title="Historical-token exposure",
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x, degradation, color=COLORS[: len(labels)])
    axes[1].axhline(5.0, color="#a33a2b", linestyle="--", label="+5% limit")
    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].set(
        xticks=x,
        xticklabels=labels,
        ylabel="Ordinary recent loss change (%)",
        title="Recent-only fidelity",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Separated recent/history residual paths · base host frozen",
        fontsize=11,
        weight="bold",
    )
    fig.tight_layout()
    return save(fig, "cem_native_long_conditioner")


def load_conditioned_arrays() -> list[dict[str, np.ndarray]]:
    rows = []
    for result_path in sorted((OUTPUT / "cells").glob("*/*/result.json")):
        result = json.loads(result_path.read_text())
        if result["gates"]["B_conditioner"] is None:
            continue
        with np.load(
            ROOT / result["artifacts"]["evaluation"],
            allow_pickle=False,
        ) as data:
            rows.append({key: np.asarray(data[key]) for key in data.files})
    return rows


def selective_figures(
    arrays: list[dict[str, np.ndarray]],
) -> tuple[list[str], list[str]]:
    utility = np.concatenate(
        [
            (row["loss_conditioned_recent"] - row["loss_conditioned_robust"])
            / np.maximum(row["loss_conditioned_recent"], 1e-8)
            for row in arrays
        ]
    )
    absolute = np.concatenate(
        [
            row["loss_conditioned_recent"] - row["loss_conditioned_robust"]
            for row in arrays
        ]
    )
    order = np.argsort(utility)[::-1]
    coverage_grid = np.linspace(0.02, 1.0, 50)
    risk, precision = [], []
    for coverage in coverage_grid:
        count = max(1, int(round(coverage * len(order))))
        chosen = utility[order[:count]]
        risk.append(float(-chosen.mean()))
        precision.append(float(np.mean(chosen > 0.0)))
    fig, left = plt.subplots(figsize=(6.7, 3.8))
    right = left.twinx()
    risk_line, = left.plot(
        coverage_grid * 100.0,
        risk,
        color=COLORS[0],
        label="Selective risk",
    )
    precision_line, = right.plot(
        coverage_grid * 100.0,
        np.asarray(precision) * 100.0,
        color=COLORS[1],
        label="Positive precision",
    )
    safe_point = left.scatter(
        [0],
        [0],
        color="#222222",
        zorder=5,
        label="Safe default",
    )
    left.set(
        xlabel="Activation coverage (%)",
        ylabel="Post-hoc oracle selective risk",
        title="Target-aware upper bound; calibrated gate not reached",
    )
    right.set_ylabel("Post-hoc positive precision (%)")
    left.axhline(0.0, color="#555555", linewidth=0.8)
    left.spines["top"].set_visible(False)
    right.spines["top"].set_visible(False)
    left.grid(axis="y", alpha=0.2)
    left.legend(
        [risk_line, precision_line, safe_point],
        ["Selective risk", "Positive precision", "Safe default"],
        frameon=False,
    )
    fig.tight_layout()
    risk_paths = save(fig, "cem_native_long_selective_risk")

    cell_always = []
    cell_oracle = []
    for row in arrays:
        gain = row["loss_conditioned_recent"] - row["loss_conditioned_robust"]
        cell_always.append(float(gain.mean()))
        cell_oracle.append(float(np.maximum(gain, 0.0).mean()))
    values = [
        0.0,
        float(np.mean(cell_always)),
        0.0,
        float(np.mean(cell_oracle)),
    ]
    fig, axis = plt.subplots(figsize=(6.5, 3.7))
    axis.bar(
        np.arange(4),
        values,
        color=("#777777", COLORS[1], "#222222", COLORS[2]),
    )
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.set(
        xticks=np.arange(4),
        xticklabels=(
            "Recent-only",
            "Always memory",
            "Gated (abstain)",
            "Oracle activation",
        ),
        ylabel="Mean host-MSE gain vs recent",
        title="Policy comparison after Gate B hard stop",
    )
    axis.text(
        2,
        0.0,
        "coverage 0%",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    policy_paths = save(fig, "cem_native_long_policy_comparison")
    return risk_paths, policy_paths


def timeline_figure(
    arrays: list[dict[str, np.ndarray]],
) -> list[str]:
    best = None
    for result_path in sorted((OUTPUT / "cells").glob("*/*/result.json")):
        result = json.loads(result_path.read_text())
        with np.load(
            ROOT / result["artifacts"]["evaluation"],
            allow_pickle=False,
        ) as data:
            opportunity = np.asarray(data["opportunity"], dtype=bool)
            if not opportunity.any():
                continue
            gain = np.asarray(data["loss_raw_recent"]) - np.asarray(
                data["loss_raw_oracle"]
            )
            eligible = np.flatnonzero(opportunity)
            row = int(eligible[np.argmax(gain[eligible])])
            candidate = {
                "environment": result["environment"],
                "episode": int(np.asarray(data["episode_id"])[row]),
                "query_t": int(np.asarray(data["query_t"])[row]),
                "source": int(
                    np.asarray(data["raw_oracle_frame_index"])[row]
                ),
                "gap": int(np.asarray(data["gap"])[row]),
                "gain": float(gain[row]),
            }
            if best is None or candidate["gain"] > best["gain"]:
                best = candidate
    if best is None:
        return []
    with np.load(
        CACHE / best["environment"] / "render_cache.npz",
        allow_pickle=False,
    ) as data:
        frames = np.asarray(data["frames"][best["episode"]])
    query_t = best["query_t"]
    indices = [
        best["source"],
        max(0, query_t - 12),
        max(0, query_t - 7),
        query_t - 4,
        query_t,
        min(len(frames) - 1, query_t + HORIZON),
    ]
    labels = (
        "oracle old frame",
        "long-gap history",
        "recent memory",
        "recent memory",
        "query",
        "realized future",
    )
    fig, axes = plt.subplots(1, len(indices), figsize=(11.5, 2.5))
    for axis, index, label in zip(axes, indices, labels):
        image = Image.fromarray(frames[index]).resize(
            (180, 180),
            Image.Resampling.NEAREST,
        )
        axis.imshow(image)
        axis.set_title(f"t={index}\n{label}", fontsize=8)
        axis.axis("off")
    fig.suptitle(
        f"{short_env(best['environment'])} · gap {best['gap']} · "
        "post-hoc opportunity, activation ABSTAINED (Gate B failed)",
        fontsize=10,
        weight="bold",
    )
    fig.tight_layout()
    return save(fig, "cem_native_long_activation_timeline")


def main() -> None:
    if not REPORT.is_file():
        raise FileNotFoundError(REPORT)
    ASSETS.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text())
    arrays = load_conditioned_arrays()
    artifacts = {
        "opportunity": opportunity_figure(report),
        "conditioner": conditioner_figure(report),
    }
    if arrays:
        risk, policy = selective_figures(arrays)
        artifacts["selective_risk_upper_bound"] = risk
        artifacts["policy_comparison"] = policy
        artifacts["activation_timeline"] = timeline_figure(arrays)
    receipt = {
        "schema": "cem_native_long_figure_receipt_v1",
        "white_background": True,
        "calibrated_gate_curve_available": False,
        "selective_curve_is_posthoc_oracle_upper_bound": True,
        "artifacts": artifacts,
    }
    (OUTPUT / "figure_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
