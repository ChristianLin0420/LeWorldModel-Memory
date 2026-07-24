#!/usr/bin/env python3
"""White-background figures for the spatial conditioner confirmation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/cem_spatial_conditioner_v1"
ASSETS = ROOT / "docs/assets"
CACHE = ROOT / "outputs/paper_c_agescale_v1/cache"


def save(fig: plt.Figure, name: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = ASSETS / f"{name}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def main() -> None:
    report = json.loads((OUTPUT / "report.json").read_text())
    ASSETS.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    global_value = report["factorial"]["A_global_bottleneck_recovery_mean"]
    patch = report["factorial"]["B_patch_grid_recovery"]
    fig, axis = plt.subplots(figsize=(5.8, 3.5))
    values = [100 * global_value, 100 * patch["mean"]]
    error = [
        [0.0, 100 * (patch["mean"] - patch["ci95"][0])],
        [0.0, 100 * (patch["ci95"][1] - patch["mean"])],
    ]
    axis.bar([0, 1], values, color=("#777777", "#28659c"))
    axis.errorbar([0, 1], values, yerr=np.asarray(error), fmt="none", color="black")
    axis.axhline(50, color="#a33a2b", linestyle="--", label="50% target")
    axis.set(
        xticks=[0, 1],
        xticklabels=["Global bottleneck", "Patch-grid + 2D"],
        ylabel="Oracle opportunity recovered (%)",
        title="Gate-B conditioner comparison",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    artifacts["factorial"] = save(fig, "cem_spatial_factorial_recovery")

    fig, axis = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(3)
    width = 0.34
    for offset, (env, color, label) in enumerate(
        (
            ("cube-single-play-v0", "#2f7d45", "Cube"),
            ("pointmaze-large-navigate-v0", "#d2772f", "PointMaze"),
        )
    ):
        rows = report["family_high_gap_gains"][env]
        values = [rows[str(gap)]["memory_gain"] for gap in (32, 64, 128)]
        low = [values[i] - rows[str(gap)]["ci95"][0] for i, gap in enumerate((32, 64, 128))]
        high = [rows[str(gap)]["ci95"][1] - values[i] for i, gap in enumerate((32, 64, 128))]
        positions = x + (offset - 0.5) * width
        axis.bar(positions, values, width=width, color=color, label=label)
        axis.errorbar(positions, values, yerr=[low, high], fmt="none", color="black")
    axis.axhline(0, color="#444444", linewidth=0.8)
    axis.set(
        xticks=x,
        xticklabels=["gap 32", "gap 64", "gap 128"],
        ylabel="Memory gain vs recent (host MSE)",
        title="High-gap spatial memory gain by family",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    artifacts["family_gap"] = save(fig, "cem_spatial_family_gap")

    cells = []
    for path in sorted((OUTPUT / "cells").glob("*/*/result.json")):
        cells.append(json.loads(path.read_text()))
    fig, axis = plt.subplots(figsize=(5.8, 3.7))
    for environment, color, label in (
        ("cube-single-play-v0", "#2f7d45", "Cube"),
        ("pointmaze-large-navigate-v0", "#d2772f", "PointMaze"),
    ):
        rows = [row for row in cells if row["environment"] == environment]
        axis.scatter(
            [100 * row["variants"]["B_patch_grid_position"]["ordinary_recent_degradation"] for row in rows],
            [100 * row["variants"]["B_patch_grid_position"]["recovery"] for row in rows],
            color=color,
            s=45,
            label=label,
        )
    axis.axhline(50, color="#a33a2b", linestyle="--")
    axis.axvline(5, color="#a33a2b", linestyle="--")
    axis.set(
        xlabel="Ordinary recent loss change (%)",
        ylabel="Oracle opportunity recovered (%)",
        title="Exposure–fidelity Pareto",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    artifacts["pareto"] = save(fig, "cem_spatial_exposure_fidelity")

    decision = json.loads(
        (
            OUTPUT / "cells/cube-single-play-v0/s0/decision_log.json"
        ).read_text()
    )
    row = max(
        decision["queries"],
        key=lambda value: value["realized_oracle_delta_audit"],
    )
    with np.load(
        CACHE / "cube-single-play-v0/render_cache.npz",
        allow_pickle=False,
    ) as data:
        frames = data["frames"][row["episode_id"]]
    source = row["oracle_frame_index"]
    query = row["query_t"]
    coordinate = row["memory_patch_coordinates"]
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.8))
    for axis, index, title in (
        (axes[0], source, f"memory t={source}"),
        (axes[1], query, f"query t={query}"),
    ):
        image = Image.fromarray(frames[index]).resize((256, 256))
        if axis is axes[0]:
            draw = ImageDraw.Draw(image)
            x = int((coordinate[0] + 1) * 128)
            y = int((coordinate[1] + 1) * 128)
            draw.rectangle([x - 32, y - 32, x + 32, y + 32], outline="red", width=4)
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle("Real unmodified Cube frames · highest-attended memory patch")
    fig.tight_layout()
    artifacts["attention"] = save(fig, "cem_spatial_attention_example")

    receipt = {
        "schema": "cem_spatial_figure_receipt_v1",
        "white_background": True,
        "artifacts": artifacts,
    }
    (OUTPUT / "figure_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
