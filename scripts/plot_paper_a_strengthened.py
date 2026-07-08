#!/usr/bin/env python3
"""Generate the strengthened Paper-A figures from authenticated artifacts.

The main figures deliberately use conventional model blocks, environment
renderings, absolute outcomes, and direct labels.  Equations and complete
seed grids remain in the appendix text rather than inside diagrams.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping

import h5py
import hdf5plugin  # noqa: F401  # registers the official PushT Blosc filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "paper_a" / "figures"

from lewm.official_tasks.pusht_memory import render_single_overlay  # noqa: E402
from lewm.official_tasks.matched_memory import render_joint_cue  # noqa: E402
from lewm.official_tasks.dinowm_pointmaze import (  # noqa: E402
    goal_card,
    render_transient_goal_cue,
)


GREEN = "#6B9F12"
GREEN_DARK = "#4E760B"
GREEN_PALE = "#EEF5E3"
TEAL = "#28747D"
ORANGE = "#C96F2D"
BLUE = "#4D6F91"
BLUE_PALE = "#EAF1F7"
INK = "#24282C"
MID = "#666D73"
LIGHT = "#E4E7E9"
PALE = "#F5F6F6"
RED = "#A94B3F"

ARM_COLORS = {
    "none": "#8A8F93",
    "gru": BLUE,
    "lstm": "#8772A4",
    "ssm": TEAL,
    "fixed_trust": GREEN,
}
ARM_SHORT = {
    "none": "No state",
    "gru": "GRU",
    "lstm": "LSTM",
    "ssm": "State-space",
    "fixed_trust": "Fixed-trust",
}

PDF_METADATA = {
    "Creator": "plot_paper_a_strengthened.py",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}
PNG_METADATA = {"Software": "plot_paper_a_strengthened.py"}

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.titlesize": 8.4,
    "axes.labelsize": 7.7,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.edgecolor": MID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MID,
    "ytick.color": MID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_current_lock(config_path: Path, lock: Mapping[str, Any],
                         expected_grid: Mapping[str, int]) -> str:
    protocol = str(lock.get("protocol_sha256", ""))
    if len(protocol) != 64 or _sha256(config_path) != protocol:
        raise RuntimeError(f"protocol lock changed: {config_path}")
    if lock.get("grid") != dict(expected_grid):
        raise RuntimeError(f"registered grid changed: {config_path}")
    for relative, expected in lock.get("source_sha256", {}).items():
        source = Path(relative)
        if not source.is_absolute():
            source = ROOT / source
        if not source.is_file() or _sha256(source) != expected:
            raise RuntimeError(f"locked source changed: {relative}")
    return protocol


def _require_final_audit_receipts() -> None:
    """Require both independent receipts and bind them to current summaries."""
    cross_path = ROOT / "outputs/paper_a_cross_wave_completion/receipt.json"
    stats_path = ROOT / "outputs/paper_a_statistics_independent/receipt.json"
    if not cross_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError("final independent audit receipts are unavailable")
    cross = json.loads(cross_path.read_text())
    stats = json.loads(stats_path.read_text())
    if cross.get("schema") != "paper_a_cross_wave_completion_receipt_v1" \
            or cross.get("status") != "complete" \
            or cross.get("read_only_verification") is not True \
            or cross.get("sealed_locks_modified") is not False \
            or cross.get("paper_files_modified") is not False \
            or cross.get("totals", {}).get("formal_cells") != 125 \
            or cross.get("totals", {}).get("physical_gpu_cell_counts") \
            != {"0": 50, "1": 50, "2": 25, "3": 0} \
            or cross.get("totals", {}).get("cuda3_used") is not False:
        raise RuntimeError("cross-wave completion receipt differs")
    if stats.get("schema") != "paper_a_statistics_independent_receipt_v1" \
            or stats.get("status") != "verified" \
            or stats.get("read_only") is not True \
            or stats.get("statistics_computed") is not True \
            or stats.get("imports_producer_statistics") is not False \
            or stats.get("experiment_roots_modified") is not False \
            or stats.get("scientific_cross_family_pooling") is not False \
            or stats.get("totals", {}).get("formal_cells") != 75:
        raise RuntimeError("independent statistics receipt differs")
    cross_waves = cross.get("waves", {})
    stats_waves = stats.get("waves", {})
    if any(cross_waves.get(name, {}).get("status") != "verified"
           for name in ("wave1_1", "wave2_v1_1", "wave3")) \
            or any(stats_waves.get(name, {}).get("status") != "verified"
                   for name in ("wave2", "wave3")):
        raise RuntimeError("an independent receipt has an unverified wave")

    summary_paths = {
        "matched": ROOT / "outputs/paper_a_matched_color_v1_1/summary.json",
        "wave2": ROOT / (
            "outputs/dinowm_wave2_spatial_carrier_v1_1/formal/summary.json"),
        "wave3_top": ROOT / "outputs/dinowm_pointmaze_wave3/formal/summary.json",
        "wave3_carrier": ROOT / (
            "outputs/dinowm_pointmaze_wave3/formal/carrier_summary.json"),
        "wave3_use": ROOT / (
            "outputs/dinowm_pointmaze_wave3/formal/external_use_summary.json"),
    }
    actual = {name: _sha256(path) for name, path in summary_paths.items()}
    if cross_waves["wave1_1"].get("summary_sha256") != actual["matched"] \
            or cross_waves["wave2_v1_1"].get("summary_sha256") != actual["wave2"] \
            or cross_waves["wave3"].get("summary_sha256") \
            != actual["wave3_top"] \
            or cross_waves["wave3"].get("carrier_summary_sha256") \
            != actual["wave3_carrier"] \
            or cross_waves["wave3"].get("external_use_summary_sha256") \
            != actual["wave3_use"]:
        raise RuntimeError("cross-wave receipt is stale relative to summaries")
    if stats_waves["wave2"].get("summary_sha256") != actual["wave2"] \
            or stats_waves["wave3"].get("combined_summary_sha256") \
            != actual["wave3_top"] \
            or stats_waves["wave3"].get("carrier_summary_sha256") \
            != actual["wave3_carrier"] \
            or stats_waves["wave3"].get("external_use_summary_sha256") \
            != actual["wave3_use"]:
        raise RuntimeError("statistics receipt is stale relative to summaries")
    auditors = (
        (cross, ROOT / "scripts/audit_paper_a_cross_wave_completion.py"),
        (stats, ROOT / "scripts/audit_paper_a_statistics_independent.py"),
    )
    for receipt, script in auditors:
        if receipt.get("auditor", {}).get("sha256") != _sha256(script):
            raise RuntimeError(f"audit script changed after receipt: {script.name}")


def load_verified_dinowm_pusht() -> dict[str, Any]:
    """Load Wave-2 results only after its fail-closed verifier succeeds."""
    config = ROOT / "configs/dinowm_wave2_spatial_carrier_v1_1.yaml"
    lock = json.loads(config.with_suffix(".lock.json").read_text())
    protocol = _assert_current_lock(
        config, lock, {"tasks": 2, "arms": 5, "seeds": 5, "cells": 50})
    formal = ROOT / "outputs/dinowm_wave2_spatial_carrier_v1_1/formal"
    verification_path = formal / "verification.json"
    summary_path = formal / "summary.json"
    if not verification_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("verified DINO-WM PushT result is unavailable")
    verification = json.loads(verification_path.read_text())
    required = {
        "schema": "dinowm_wave2_spatial_verification_v1_1",
        "verified": True,
        "protocol_sha256": protocol,
        "physical_gpu": 1,
        "cells": 50,
        "paired_bootstrap_draws": 20_000,
        "host_unchanged": True,
        "preoutcome_numerical_amendment_verified": True,
        "paper_modified_by_wave2": False,
    }
    for key, expected in required.items():
        if verification.get(key) != expected:
            raise RuntimeError(f"DINO-WM PushT verification differs: {key}")
    if _sha256(summary_path) != verification.get("summary_sha256"):
        raise RuntimeError("DINO-WM PushT verified summary hash differs")
    if (formal / "stop_receipt.json").exists():
        raise RuntimeError("DINO-WM PushT formal stop receipt exists")
    summary = json.loads(summary_path.read_text())
    if summary.get("schema") != "dinowm_wave2_spatial_carrier_summary_v1" \
            or summary.get("status") != "complete" \
            or summary.get("protocol_sha256") != protocol \
            or summary.get("grid") != lock["grid"]:
        raise RuntimeError("DINO-WM PushT summary is not the locked full grid")
    if set(summary.get("results", {})) != {
            "transient-visual-token-recall",
            "multi-item-visual-binding-recall"}:
        raise RuntimeError("DINO-WM PushT task set differs")
    _require_final_audit_receipts()
    return summary


def load_verified_pointmaze() -> tuple[dict[str, Any], dict[str, Any],
                                       dict[str, Any]]:
    """Load PointMaze summaries only after the official verifier succeeds."""
    config = ROOT / "configs/dinowm_pointmaze_wave3.yaml"
    lock = json.loads(config.with_suffix(".lock.json").read_text())
    protocol = _assert_current_lock(
        config, lock, {"tasks": 1, "arms": 5, "seeds": 5, "cells": 25})
    root = ROOT / "outputs/dinowm_pointmaze_wave3"
    formal = root / "formal"
    verification_path = formal / "verification.json"
    paths = {
        "summary": formal / "summary.json",
        "carrier_summary": formal / "carrier_summary.json",
        "external_use_summary": formal / "external_use_summary.json",
    }
    if not verification_path.is_file() or not all(
            path.is_file() for path in paths.values()):
        raise FileNotFoundError("verified DINO-WM PointMaze result is unavailable")
    verification = json.loads(verification_path.read_text())
    required = {
        "schema": "dinowm_pointmaze_wave3_verification_v1",
        "verified": True,
        "protocol_sha256": protocol,
        "physical_gpu": 2,
        "cells": 25,
        "paired_bootstrap_draws": 20_000,
        "native_validation_episode_clusters": 120,
        "host_unchanged": True,
        "current_mujoco_execution": True,
        "paper_modified_by_wave3": False,
    }
    for key, expected in required.items():
        if verification.get(key) != expected:
            raise RuntimeError(f"PointMaze verification differs: {key}")
    hash_fields = {
        "summary": "summary_sha256",
        "carrier_summary": "carrier_summary_sha256",
        "external_use_summary": "external_use_summary_sha256",
    }
    for name, field in hash_fields.items():
        if _sha256(paths[name]) != verification.get(field):
            raise RuntimeError(f"PointMaze verified hash differs: {name}")
    for stop in (root / "cache/stop_receipt.json", formal / "stop_receipt.json",
                 formal / "formal_stop_receipt.json"):
        if stop.exists():
            raise RuntimeError(f"PointMaze stop receipt exists: {stop}")
    top, carrier, use = (json.loads(paths[name].read_text()) for name in (
        "summary", "carrier_summary", "external_use_summary"))
    if any(value.get("status") != "complete"
           or value.get("protocol_sha256") != protocol
           for value in (top, carrier, use)):
        raise RuntimeError("PointMaze summaries are not locked and complete")
    if top.get("schema") != "dinowm_pointmaze_wave3_summary_v1" \
            or carrier.get("schema") \
            != "dinowm_pointmaze_wave3_carrier_summary_v1" \
            or use.get("schema") != "dinowm_pointmaze_wave3_external_use_v1":
        raise RuntimeError("PointMaze summary schema differs")
    if carrier.get("grid") != lock["grid"] \
            or top.get("carrier_summary_path") != "carrier_summary.json" \
            or top.get("external_use_summary_path") \
            != "external_use_summary.json":
        raise RuntimeError("PointMaze grid or summary pointer differs")
    receipts = use.get("consumer_receipts", [])
    if use.get("scope", {}).get("native_planner") is not False \
            or len(receipts) != 5 \
            or not all(value.get("arm_blind") is True
                       and value.get("arm_identifier_feature") is False
                       for value in receipts):
        raise RuntimeError("PointMaze external-consumer boundary differs")
    resolved = sorted(
        arm for arm, value in use.get("arms", {}).items()
        if value.get("resolved_execution_gain") is True)
    if resolved != sorted(top.get("resolved_external_use_arms", [])):
        raise RuntimeError("PointMaze resolved execution arm set differs")
    _require_final_audit_receipts()
    return top, carrier, use


def save(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf", dpi=300, metadata=PDF_METADATA,
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{stem}.png", dpi=240, metadata=PNG_METADATA,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _figure_assets(directory: Path) -> list[Path]:
    return [
        path for path in sorted(directory.glob("fig_mem_*"))
        if path.is_file() and path.suffix in {".pdf", ".png"}
    ]


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_figure_manifest(
        *, final_directory: Path | None = None) -> dict[str, Any]:
    """Bind staged figures to their final published paths."""
    _require_final_audit_receipts()
    published = final_directory or FIG
    completion = ROOT / "outputs/paper_a_cross_wave_completion/receipt.json"
    statistics = ROOT / "outputs/paper_a_statistics_independent/receipt.json"
    summary_paths = {
        "matched": ROOT / "outputs/paper_a_matched_color_v1_1/summary.json",
        "pusht": ROOT / (
            "outputs/dinowm_wave2_spatial_carrier_v1_1/formal/summary.json"),
        "pointmaze": ROOT / "outputs/dinowm_pointmaze_wave3/formal/summary.json",
        "pointmaze_carrier": ROOT / (
            "outputs/dinowm_pointmaze_wave3/formal/carrier_summary.json"),
        "pointmaze_use": ROOT / (
            "outputs/dinowm_pointmaze_wave3/formal/external_use_summary.json"),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for path in _figure_assets(FIG):
        final_path = published / path.name
        artifacts[path.name] = {
            "path": str(final_path.relative_to(ROOT)),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
    required = {
        "fig_mem_architecture.pdf", "fig_mem_tasks.pdf",
        "fig_mem_matched.pdf", "fig_mem_dinowm_carrier.pdf",
        "fig_mem_pointmaze.pdf",
    }
    if not required.issubset(artifacts):
        raise RuntimeError("final main-figure set is incomplete")
    manifest = {
        "schema": "paper_a_figure_manifest_v1",
        "status": "complete",
        "generator": {
            "path": "scripts/plot_paper_a_strengthened.py",
            "sha256": _sha256(
                ROOT / "scripts/plot_paper_a_strengthened.py"),
        },
        "audit_receipts": {
            "completion": {
                "path": str(completion.relative_to(ROOT)),
                "sha256": _sha256(completion),
            },
            "statistics": {
                "path": str(statistics.relative_to(ROOT)),
                "sha256": _sha256(statistics),
            },
        },
        "summaries": {
            name: {"path": str(path.relative_to(ROOT)),
                   "sha256": _sha256(path)}
            for name, path in summary_paths.items()
        },
        "artifacts": artifacts,
    }
    destination = FIG / "manifest.json"
    descriptor, temporary = tempfile.mkstemp(
        prefix=".manifest.", suffix=".tmp", dir=FIG)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(FIG)
    finally:
        temporary_path.unlink(missing_ok=True)
    return manifest


def _validate_staged_figure_manifest(
        staging: Path, final_directory: Path) -> dict[str, Any]:
    manifest_path = staging / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("staged figure manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "paper_a_figure_manifest_v1" \
            or manifest.get("status") != "complete":
        raise RuntimeError("staged figure manifest is incomplete")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("staged figure artifact ledger is missing")
    staged = {path.name: path for path in _figure_assets(staging)}
    if set(artifacts) != set(staged):
        raise RuntimeError("staged figure artifact ledger differs")
    required = {
        "fig_mem_architecture.pdf", "fig_mem_tasks.pdf",
        "fig_mem_matched.pdf", "fig_mem_dinowm_carrier.pdf",
        "fig_mem_pointmaze.pdf",
    }
    if not required.issubset(staged):
        raise RuntimeError("staged main-figure set is incomplete")
    for name, path in staged.items():
        record = artifacts[name]
        expected_path = str((final_directory / name).relative_to(ROOT))
        if not isinstance(record, Mapping) \
                or record.get("path") != expected_path \
                or record.get("size") != path.stat().st_size \
                or record.get("sha256") != _sha256(path):
            raise RuntimeError(f"staged figure binding differs: {name}")
    return manifest


def _publish_staged_figures(staging: Path, final_directory: Path) -> None:
    """Publish a validated staged set, with its manifest committed last."""
    _validate_staged_figure_manifest(staging, final_directory)
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    final_directory.mkdir(parents=True, exist_ok=True)
    quarantine = Path(tempfile.mkdtemp(
        prefix=".figures-quarantine.", dir=final_directory.parent))
    published: list[Path] = []
    old_manifest = final_directory / "manifest.json"
    try:
        # Invalidate the old trust root durably before changing any figure.
        if old_manifest.is_file():
            os.replace(old_manifest, quarantine / "manifest.json")
        _fsync_directory(final_directory)
        _fsync_directory(quarantine)

        # Quarantine the old generated set so successful publication cannot
        # retain an unbound stale figure. Non-generated files are untouched.
        for path in _figure_assets(final_directory):
            os.replace(path, quarantine / path.name)
        _fsync_directory(final_directory)
        _fsync_directory(quarantine)

        for path in _figure_assets(staging):
            _fsync_file(path)
            destination = final_directory / path.name
            os.replace(path, destination)
            published.append(destination)
        _fsync_directory(final_directory)

        # The complete manifest is the final commit record.
        staged_manifest = staging / "manifest.json"
        _fsync_file(staged_manifest)
        os.replace(staged_manifest, final_directory / "manifest.json")
        _fsync_directory(final_directory)
    except Exception:
        # A failed commit must never leave either the old or a new manifest.
        (final_directory / "manifest.json").unlink(missing_ok=True)
        for path in published:
            path.unlink(missing_ok=True)
        for path in _figure_assets(quarantine):
            os.replace(path, final_directory / path.name)
        _fsync_directory(final_directory)
        raise
    finally:
        shutil.rmtree(quarantine, ignore_errors=True)
        _fsync_directory(final_directory.parent)


def _box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float,
         label: str, *, face: str, edge: str, dashed: bool = False,
         fontsize: float = 8.0, weight: str = "normal") -> None:
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.009,rounding_size=0.006",
        facecolor=face, edgecolor=edge, linewidth=1.05,
        linestyle=(0, (3, 2)) if dashed else "-", zorder=3)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, linespacing=1.05, zorder=4)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
           *, color: str = MID, dashed: bool = False, width: float = 1.15,
           connectionstyle: str = "arc3") -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8.5,
        linewidth=width, color=color,
        linestyle=(0, (3, 2)) if dashed else "-",
        connectionstyle=connectionstyle, zorder=2))


def _reacher_context_thumbnails() -> list[np.ndarray]:
    with np.load(
            ROOT / "outputs/paper_a_expansion/data/t1/val_clean_e240_s270702.npz",
            allow_pickle=False) as bank:
        return [bank["frames"][0, index].copy() for index in (12, 13, 14)]


def figure_architecture() -> None:
    """Frozen host, one swapped sidecar, and the legal pre-decision read."""
    fig, ax = plt.subplots(figsize=(5.45, 2.56))
    muted = "#50585E"
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def node(x: float, y: float, width: float, height: float,
             title: str, subtitle: str, *, face: str, edge: str,
             dashed: bool = False, title_size: float = 7.0,
             subtitle_size: float = 6.0) -> None:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.006,rounding_size=0.003",
            facecolor=face, edgecolor=edge, linewidth=1.0,
            linestyle=(0, (3, 2)) if dashed else "-", zorder=3)
        ax.add_patch(patch)
        center = x + width / 2
        ax.text(center, y + 0.61 * height, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", zorder=4)
        if subtitle:
            ax.text(center, y + 0.29 * height, subtitle, ha="center",
                    va="center", fontsize=subtitle_size, color=muted,
                    linespacing=0.96, zorder=4)

    # Panel A: one semantic host group plus one trainable sidecar.
    ax.text(0.01, 0.982, "(a) Freeze the released host; swap one persistent carrier",
            fontsize=8.2, fontweight="bold", va="top")
    key_y = 0.927
    for x, face, edge, dashed, label in (
            (0.632, BLUE_PALE, BLUE, True, "frozen host"),
            (0.760, GREEN_PALE, GREEN_DARK, False, "carrier path"),
            (0.902, PALE, MID, True, "evaluation only")):
        ax.add_patch(Rectangle(
            (x, key_y - 0.014), 0.018, 0.028, facecolor=face,
            edgecolor=edge, linewidth=0.8,
            linestyle=(0, (3, 2)) if dashed else "-"))
        ax.text(x + 0.024, key_y, label, fontsize=5.9, color=MID,
                va="center", ha="left")

    frame = _matched_reacher_visuals()[0][0]
    inset = ax.inset_axes([0.015, 0.635, 0.090, 0.174])
    inset.imshow(frame, interpolation="nearest", aspect="equal")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color(BLUE)
        spine.set_linewidth(0.85)
    ax.text(0.060, 0.603, "real observation", fontsize=6.1,
            ha="center", color=MID)

    # The frozen checkpoint is deliberately a single grouped object.
    host = FancyBboxPatch(
        (0.145, 0.605), 0.445, 0.225,
        boxstyle="round,pad=0.008,rounding_size=0.003",
        facecolor=BLUE_PALE, edgecolor=BLUE, linewidth=1.05,
        linestyle=(0, (3, 2)), zorder=2)
    ax.add_patch(host)
    ax.text(0.162, 0.797, "FROZEN RELEASED HOST", fontsize=6.1,
            fontweight="bold", color=BLUE, va="center")
    ax.text(0.162, 0.764, "same weights and short-context interface in every arm",
            fontsize=5.9, color=muted, va="center")
    ax.text(0.232, 0.689, "Image encoder", fontsize=7.0,
            fontweight="bold", ha="center")
    ax.text(0.232, 0.650, "vector or 196-patch grid", fontsize=5.9,
            color=muted, ha="center")
    _arrow(ax, (0.307, 0.690), (0.350, 0.690), color=BLUE, width=1.0)
    ax.text(0.440, 0.689, "Short-context predictor", fontsize=7.0,
            fontweight="bold", ha="center")
    ax.text(0.440, 0.650, "actions; DINO-WM also uses proprio", fontsize=5.8,
            color=muted, ha="center")
    _arrow(ax, (0.105, 0.714), (0.145, 0.714), color=BLUE, width=1.0)

    node(0.655, 0.625, 0.175, 0.175, "Label-free feature loss",
         "frozen target; labels absent", face="#EDF7F5", edge=TEAL,
         title_size=6.7, subtitle_size=5.9)
    _arrow(ax, (0.590, 0.714), (0.655, 0.714), color=TEAL, width=1.0)
    ax.text(0.852, 0.741, "next latent target", fontsize=6.0,
            color=MID, ha="left")
    _arrow(ax, (0.970, 0.714), (0.830, 0.714), color=TEAL, width=0.95)

    node(0.245, 0.405, 0.310, 0.125, "Carrier arm (swapped)",
         "None · GRU · LSTM · State-space · Fixed-trust",
         face=GREEN_PALE, edge=GREEN_DARK, title_size=7.0,
         subtitle_size=5.6)
    node(0.015, 0.420, 0.135, 0.095, "Action block", "host-native",
         face="#FBF2E8", edge=ORANGE, title_size=6.7,
         subtitle_size=5.7)
    _arrow(ax, (0.150, 0.468), (0.245, 0.468), color=ORANGE, width=1.0)
    _arrow(ax, (0.270, 0.605), (0.315, 0.530), color=GREEN_DARK,
           width=1.0)
    _arrow(ax, (0.500, 0.530), (0.470, 0.605), color=GREEN_DARK,
           width=1.0)
    ax.text(0.270, 0.558, "update state", fontsize=5.4,
            color=GREEN_DARK, ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.15))
    ax.text(0.515, 0.558, "fuse/read", fontsize=5.4,
            color=GREEN_DARK, ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.15))
    _arrow(ax, (0.110, 0.515), (0.175, 0.605), color=ORANGE,
           connectionstyle="arc3,rad=-0.10", width=0.95)
    ax.add_patch(FancyArrowPatch(
        (0.505, 0.410), (0.300, 0.410), arrowstyle="-|>",
        mutation_scale=8.5, linewidth=0.95, color=GREEN_DARK,
        connectionstyle="arc3,rad=-0.35", zorder=2))
    ax.text(0.405, 0.372, "episode state persists", fontsize=6.0,
            color=GREEN_DARK, ha="center")
    ax.plot([0.625, 0.625], [0.405, 0.525], color=GREEN_DARK,
            linewidth=1.3)
    ax.text(0.640, 0.495, "LeWM: one vector stream", fontsize=5.9,
            color=INK, ha="left")
    ax.text(0.640, 0.448, "DINO-WM: one tied carrier × 196 patches",
            fontsize=5.9, color=INK, ha="left")

    ax.plot([0.01, 0.99], [0.345, 0.345], color=LIGHT, linewidth=0.9)

    # Panel B: actual evidence timing and the evaluation boundary.
    ax.text(0.01, 0.317, "(b) Read state before the decision observation; then evaluate",
            fontsize=8.1, fontweight="bold", va="top")
    cue_frames = _matched_reacher_visuals()[0]
    cue_times = (2, 9, 18)
    frame_x = (0.018, 0.132, 0.246)
    frame_edges = (GREEN, MID, TEAL)
    frame_labels = ("cue visible", "cue-free delay", "pre-decision")
    for x, image_value, time, edge, label in zip(
            frame_x, cue_frames, cue_times, frame_edges, frame_labels):
        frame_ax = ax.inset_axes([x, 0.085, 0.082, 0.158])
        frame_ax.imshow(image_value, interpolation="nearest", aspect="equal")
        frame_ax.set_xticks([])
        frame_ax.set_yticks([])
        for spine in frame_ax.spines.values():
            spine.set_visible(True)
            spine.set_color(edge)
            spine.set_linewidth(1.0 if edge != MID else 0.75)
        ax.text(x + 0.041, 0.058, label, fontsize=5.8, color=edge,
                ha="center", fontweight="bold" if edge != MID else "normal")
        ax.text(x + 0.041, 0.031, f"$t={time}$", fontsize=5.6,
                color=MID, ha="center")
    _arrow(ax, (0.103, 0.165), (0.126, 0.165), color=GREEN_DARK,
           width=0.85)
    _arrow(ax, (0.217, 0.165), (0.240, 0.165), color=GREEN_DARK,
           width=0.85)
    ax.plot([0.338, 0.338], [0.075, 0.275], color=ORANGE,
            linewidth=0.9, linestyle=(0, (2, 2)))
    ax.text(0.345, 0.246, "state read", fontsize=6.0,
            color=ORANGE, fontweight="bold", ha="left")
    ax.text(0.345, 0.214, "decision observation\nexcluded", fontsize=5.4,
            color=muted, ha="left", va="top", linespacing=0.94)

    evaluation = FancyBboxPatch(
        (0.475, 0.048), 0.515, 0.222,
        boxstyle="round,pad=0.007,rounding_size=0.003",
        facecolor=PALE, edgecolor=MID, linewidth=0.95,
        linestyle=(0, (3, 2)), zorder=1)
    ax.add_patch(evaluation)
    ax.text(0.490, 0.251, "EVALUATION ONLY · NO TRAINING SIGNAL",
            fontsize=6.0, color=MID, fontweight="bold", va="center")
    node(0.485, 0.078, 0.190, 0.145, "Legal pre-decision read",
         "LeWM: $H$ latents + prior\nDINO-WM: full-state output",
         face="white", edge=MID, title_size=6.5, subtitle_size=5.8)
    _arrow(ax, (0.405, 0.158), (0.492, 0.158), color=GREEN_DARK,
           width=0.95)
    node(0.715, 0.165, 0.145, 0.062, "Retention readout", "accuracy",
         face="white", edge=MID, dashed=True, title_size=6.3,
         subtitle_size=5.7)
    node(0.715, 0.070, 0.145, 0.072, "Fixed consumer", "controller execution",
         face="white", edge=MID, dashed=True, title_size=6.3,
         subtitle_size=5.7)
    _arrow(ax, (0.675, 0.151), (0.715, 0.196), color=MID, width=0.85)
    _arrow(ax, (0.675, 0.151), (0.715, 0.106), color=MID, width=0.85)
    _arrow(ax, (0.860, 0.196), (0.895, 0.196), color=MID, width=0.85)
    _arrow(ax, (0.860, 0.106), (0.895, 0.106), color=MID, width=0.85)
    ax.text(0.904, 0.196, "retained?", fontsize=6.2, color=INK,
            va="center", fontweight="bold")
    ax.text(0.904, 0.106, "used?", fontsize=6.2, color=INK,
            va="center", fontweight="bold")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    save(fig, "fig_mem_architecture")


def _reacher_task_frames(
        task: str) -> tuple[list[np.ndarray], str, tuple[int, int, int]]:
    path = ROOT / f"outputs/paper_a_expansion/data/{task}/val_clean_e240_s270702.npz"
    with np.load(path, allow_pickle=False) as bank:
        cue_on = int(bank["event_cue_on"][0])
        cue_off = int(bank["event_cue_off"][0])
        times = (cue_on + 1, min(cue_off + 12, 45), 62)
        frames = [bank["frames"][0, time].copy() for time in times]
        evidence_ages = 63 - np.asarray(bank["event_cue_off"], dtype=np.int64)
    age_range = f"{int(evidence_ages.min())}\N{EN DASH}{int(evidence_ages.max())}"
    return frames, age_range, times


def _pusht_task_frames(
        task_key: str) -> tuple[list[np.ndarray], int, tuple[int, int, int]]:
    base_path = ROOT / "outputs/official_pusht_memory/cache/base/validation.npz"
    task_path = (ROOT / "outputs/official_pusht_memory/cache/tasks" /
                 task_key / "validation.npz")
    with np.load(base_path, allow_pickle=False) as base, \
            np.load(task_path, allow_pickle=False) as task:
        episode = int(base["episode_index"][0])
        local_start = int(base["local_start"][0])
        label = int(task["labels"][0])
    hdf5_path = ROOT / "outputs/paper_a_strengthening/data/pusht_expert_train.h5"
    with h5py.File(hdf5_path, "r", swmr=True) as handle:
        offset = int(handle["ep_offset"][episode])
        global_indices = offset + local_start + np.arange(20) * 5
        frames = np.asarray(handle["pixels"][global_indices]).copy()
    semantic = ({
        "transient-visual-token-recall": "PushT transient visual-token recall",
        "multi-item-visual-binding-recall": "PushT multi-item visual-binding recall",
    })[task_key]
    overlaid = render_single_overlay(frames, semantic, label, 1, 3)
    # Prior q=19 has processed observations 4..18 after the last cue frame
    # at index 3: 15 post-cue observations under the paper's convention.
    times = (2, 9, 18)
    return [overlaid[index].copy() for index in times], 15, times


def _task_row(ax: plt.Axes, frames: list[np.ndarray],
              times: tuple[int, int, int], title: str, classes: str,
              cue_read: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.008, 0.63, title, ha="left", va="center", fontsize=7.0,
            fontweight="bold", linespacing=0.98)
    ax.text(0.008, 0.27, classes, ha="left", va="center", fontsize=6.0,
            color=MID)
    xs = (0.198, 0.423, 0.648)
    borders = (GREEN, MID, TEAL)
    for frame, time, x, border in zip(frames, times, xs, borders):
        inset = ax.inset_axes([x, 0.17, 0.125, 0.76])
        interpolation = "nearest" if frame.shape[0] <= 64 else "antialiased"
        inset.imshow(frame, interpolation=interpolation, aspect="equal")
        inset.set_box_aspect(1)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color(border)
            spine.set_linewidth(1.15 if border != MID else 0.75)
        ax.text(x + 0.0625, 0.075, f"$t={time}$", ha="center", va="center",
                fontsize=5.7, color=MID)
    _arrow(ax, (0.329, 0.54), (0.414, 0.54), color=MID, width=0.85)
    _arrow(ax, (0.554, 0.54), (0.639, 0.54), color=MID, width=0.85)
    _arrow(ax, (0.779, 0.54), (0.845, 0.54), color=GREEN_DARK, width=0.9)
    receipt = FancyBboxPatch(
        (0.855, 0.23), 0.132, 0.60,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=GREEN_PALE, edgecolor="#B7C99A", linewidth=0.7)
    ax.add_patch(receipt)
    ax.text(0.909, 0.62, cue_read, ha="center", va="center",
            fontsize=6.7, fontweight="bold", color=GREEN_DARK)
    ax.plot([0.951, 0.958, 0.973], [0.615, 0.565, 0.675],
            color=GREEN_DARK, linewidth=1.35, solid_capstyle="round")
    ax.text(0.921, 0.39, "decoded", ha="center", va="center",
            fontsize=5.7, color=MID)


def _host_strip(ax: plt.Axes, host: str, endpoint: str,
                evidence_age: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.002, 0.09), 0.985, 0.82,
        boxstyle="round,pad=0.006,rounding_size=0.009",
        facecolor=BLUE_PALE, edgecolor="#CEDCE8", linewidth=0.65))
    ax.text(0.012, 0.50, host, fontsize=6.35, fontweight="bold",
            color=BLUE, va="center")
    ax.text(0.135, 0.50, endpoint, fontsize=6.0, color=INK, va="center")
    ax.text(0.982, 0.50, f"observations since cue  {evidence_age}", fontsize=5.8,
            color=GREEN_DARK, fontweight="bold", ha="right", va="center")


def _matched_reacher_visuals() \
        -> tuple[list[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Illustrative matched-cue renders on a clean Reacher trajectory slice."""
    path = ROOT / "outputs/paper_a_expansion/data/t1/val_clean_e240_s270702.npz"
    with np.load(path, allow_pickle=False) as bank:
        # The original marker cue occurs before this slice; the matched cue is
        # therefore the only controlled overlay in the displayed window.
        base = np.asarray(bank["frames"][0, 30:50]).copy()
    shown = render_joint_cue(base, 0, 3, 1, 3)
    paired = render_joint_cue(base, 2, 3, 1, 3)
    return [shown[index].copy() for index in (2, 9, 18)], \
        (shown[2].copy(), paired[2].copy())


def _matched_pusht_visuals() \
        -> tuple[list[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Illustrative formal-selection PushT renders for the matched color cue."""
    cache = ROOT / (
        "outputs/paper_a_matched_color_v1_1/cache/pusht/base/validation.npz")
    with np.load(cache, allow_pickle=False) as values:
        indices = np.asarray(values["global_frame_indices"][0], dtype=np.int64)
    hdf5_path = ROOT / "outputs/paper_a_strengthening/data/pusht_expert_train.h5"
    with h5py.File(hdf5_path, "r", swmr=True) as handle:
        base = np.asarray(handle["pixels"][indices]).copy()
    shown = render_joint_cue(base, 0, 1, 1, 3)
    paired = render_joint_cue(base, 2, 1, 1, 3)
    return [shown[index].copy() for index in (2, 9, 18)], \
        (shown[2].copy(), paired[2].copy())


def _pointmaze_visuals() \
        -> tuple[list[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load one registered PointMaze window from the pinned durable archive."""
    import io
    import zipfile
    import torch

    selection = load_json(
        "outputs/dinowm_pointmaze_wave3/cache/selection.json")["values"]
    item = next(value for value in selection if value["split"] == "validation")
    episode = int(item["episode_index"])
    local_start = int(item["local_start"])
    filename = f"episode_{episode:03d}.pth"
    archive = ROOT / "outputs/dinowm_pointmaze_wave3/downloads/point_maze.zip"
    with zipfile.ZipFile(archive) as handle:
        payload = handle.read(f"point_maze/obses/{filename}")
    tensor = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=False)
    episode_frames = np.asarray(tensor.cpu().numpy(), dtype=np.uint8)
    indices = local_start + np.arange(20, dtype=np.int64) * 5
    base = episode_frames[indices].copy()
    shown = render_transient_goal_cue(base, 0, cue_start=1, cue_length=3)
    paired = render_transient_goal_cue(base, 3, cue_start=1, cue_length=3)
    return [shown[index].copy() for index in (2, 9, 18)], \
        (shown[2].copy(), paired[2].copy())


def _paired_cue_crops(pair: tuple[np.ndarray, np.ndarray]
                      ) -> tuple[np.ndarray, np.ndarray]:
    left, right = (np.asarray(value) for value in pair)
    values = np.any(left != right, axis=-1)
    yy, xx = np.nonzero(values)
    if not len(yy):
        raise RuntimeError("paired cue renders do not differ")
    span = max(int(np.ptp(yy)) + 1, int(np.ptp(xx)) + 1)
    pad = max(6, int(round(0.90 * span)))
    y0, y1 = max(0, int(yy.min()) - pad), min(len(values), int(yy.max()) + pad + 1)
    x0, x1 = max(0, int(xx.min()) - pad), min(values.shape[1], int(xx.max()) + pad + 1)
    return left[y0:y1, x0:x1].copy(), right[y0:y1, x0:x1].copy()


def _compact_task_row(ax: plt.Axes, *, host: str, task: str,
                      frames: list[np.ndarray],
                      difference: tuple[np.ndarray, np.ndarray],
                      cue_score: float, shortcut_score: float,
                      shaded: bool) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if shaded:
        ax.add_patch(Rectangle((0.0, 0.02), 1.0, 0.96,
                               facecolor="#F7F8F8", edgecolor="none"))
    ax.text(0.008, 0.64, host, fontsize=6.8, fontweight="bold",
            color=BLUE if "DINO" not in host else TEAL, va="center")
    ax.text(0.008, 0.32, task, fontsize=6.1, color=MID, va="center")

    positions = (0.195, 0.390, 0.585)
    edges = (GREEN, MID, TEAL)
    labels = ("$t=2$", "$t=9$", "$t=18$")
    for image_value, x, edge, label in zip(frames, positions, edges, labels):
        inset = ax.inset_axes([x, 0.13, 0.125, 0.76])
        interpolation = "nearest" if image_value.shape[0] <= 64 else "antialiased"
        inset.imshow(image_value, interpolation=interpolation, aspect="equal")
        inset.set_box_aspect(1)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color(edge)
            spine.set_linewidth(1.15 if edge != MID else 0.75)
        ax.text(x + 0.0625, 0.055, label, fontsize=5.7, color=MID,
                ha="center", va="center")
    _arrow(ax, (0.326, 0.52), (0.382, 0.52), color=MID, width=0.85)
    _arrow(ax, (0.521, 0.52), (0.577, 0.52), color=MID, width=0.85)

    cue_crops = _paired_cue_crops(difference)
    crop_width = 0.050
    for index, (x, crop) in enumerate(zip((0.738, 0.796), cue_crops)):
        diff_ax = ax.inset_axes([x, 0.24, crop_width, 0.54])
        diff_ax.imshow(crop, interpolation="nearest", aspect="equal")
        diff_ax.set_box_aspect(1)
        diff_ax.set_xticks([])
        diff_ax.set_yticks([])
        for spine in diff_ax.spines.values():
            spine.set_visible(True)
            spine.set_color(ORANGE)
            spine.set_linewidth(0.85)
        ax.text(x + crop_width / 2, 0.190, "A" if index == 0 else "B",
                fontsize=5.8, color=ORANGE, ha="center", va="center",
                fontweight="bold")

    receipt = FancyBboxPatch(
        (0.855, 0.16), 0.133, 0.68,
        boxstyle="round,pad=0.006,rounding_size=0.006",
        facecolor=GREEN_PALE, edgecolor="#B7C99A", linewidth=0.7)
    ax.add_patch(receipt)
    cue_text = f"{cue_score:.3f}"
    late_text = f"{shortcut_score:.3f}"
    if cue_text.startswith("0."):
        cue_text = cue_text[1:]
    if late_text.startswith("0."):
        late_text = late_text[1:]
    ax.plot([0.870, 0.876, 0.888], [0.615, 0.595, 0.642],
            color=GREEN_DARK, linewidth=1.35, solid_capstyle="round")
    ax.text(0.897, 0.62, f"cue {cue_text}", fontsize=5.9,
            color=GREEN_DARK,
            ha="left", va="center", fontweight="bold")
    ax.plot([0.870, 0.876, 0.888], [0.425, 0.405, 0.452],
            color=GREEN_DARK, linewidth=1.35, solid_capstyle="round")
    ax.text(0.897, 0.43, f"late {late_text} ≤ .300",
            fontsize=5.6, color=GREEN_DARK,
            ha="left", va="center", fontweight="bold")
    ax.text(0.921, 0.25, "ages 4 · 8 · 15", fontsize=5.6, color=MID,
            ha="center", va="center")


def figure_tasks() -> None:
    maze_top, _, _ = load_verified_pointmaze()
    reacher, reacher_difference = _matched_reacher_visuals()
    pusht, pusht_difference = _matched_pusht_visuals()
    maze, maze_difference = _pointmaze_visuals()
    matched_gates = {}
    for host in ("reacher", "pusht"):
        manifest = load_json(
            f"outputs/paper_a_matched_color_v1_1/cache/{host}/manifest.json")
        records = [value["color"] for value in manifest["admission"].values()]
        matched_gates[host] = (
            min(float(value["cue_probe"]["balanced_accuracy"])
                for value in records),
            max(float(value[name]["balanced_accuracy"])
                for value in records
                for name in ("final_context_latent_shortcut",
                             "final_action_shortcut", "final_state_shortcut")),
        )
    maze_admission = maze_top["admission"]
    maze_shortcut = max(
        float(source["balanced_accuracy"])
        for age in maze_admission["shortcuts"].values()
        for source in age.values())
    maze_gates = (float(maze_admission["cue_encoding"]["balanced_accuracy"]),
                  maze_shortcut)
    fig = plt.figure(figsize=(5.45, 2.70))
    grid = fig.add_gridspec(4, 1, height_ratios=(0.30, 1.0, 1.0, 1.0),
                            hspace=0.035)
    header = fig.add_subplot(grid[0])
    header.axis("off")
    header.text(0.008, 0.58, "host · controlled target", fontsize=6.1,
                color=MID, va="center")
    for x, label, color in (
            (0.2575, "cue only", GREEN_DARK),
            (0.4525, "cue-free delay", MID),
            (0.6475, "pre-decision", TEAL),
            (0.791, "paired cue crop", ORANGE),
            (0.921, "admission", GREEN_DARK)):
        header.text(x, 0.58, label, ha="center", va="center",
                    fontsize=6.15, color=color,
                    fontweight="bold" if color != MID else "normal")
    axes = [fig.add_subplot(grid[index]) for index in (1, 2, 3)]
    _compact_task_row(
        axes[0], host="LeWM · Reacher",
        task="matched color · 4-way", frames=reacher,
        difference=reacher_difference, cue_score=matched_gates["reacher"][0],
        shortcut_score=matched_gates["reacher"][1], shaded=False)
    _compact_task_row(
        axes[1], host="LeWM · PushT",
        task="matched color · 4-way", frames=pusht,
        difference=pusht_difference, cue_score=matched_gates["pusht"][0],
        shortcut_score=matched_gates["pusht"][1], shaded=True)
    _compact_task_row(
        axes[2], host="DINO-WM · PointMaze",
        task="transient goal · 4-way", frames=maze,
        difference=maze_difference, cue_score=maze_gates[0],
        shortcut_score=maze_gates[1], shaded=False)
    fig.subplots_adjust(left=0.010, right=0.995, top=0.99, bottom=0.012)
    save(fig, "fig_mem_tasks")


def _fresh_task_mean(robust: Mapping[str, Any], task: str,
                     arm: str) -> float:
    banks = robust["fresh_validation"]["task_banks"][task]["banks"]
    values = []
    for seed in range(5):
        values.append(float(np.mean([
            banks[bank]["arms"][arm]["accuracy"]["values"][seed]
            for bank in ("fresh-a", "fresh-b")
        ])))
    return float(np.mean(values))


def _reacher_original_stat(robust: Mapping[str, Any],
                            parent: Mapping[str, Any], task: str,
                            arm: str) -> dict[str, Any]:
    if arm in ("none", "lstm"):
        return parent["frozen_carrier_swap"]["tasks"][task]["arms"][arm][
            "accuracy"]
    return robust["seed_extension"]["tasks"][task]["arms"][arm]["accuracy"]


def _forest_point(ax: plt.Axes, y: float, stat: Mapping[str, Any], color: str,
                  marker: str, *, filled: bool = True, size: float = 4.8,
                  zorder: int = 4) -> None:
    mean = float(stat["mean"])
    low, high = map(float, stat["ci95"])
    ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt=marker,
                markersize=size, markerfacecolor=color if filled else "white",
                markeredgecolor=color, markeredgewidth=1.0,
                color=color, linewidth=1.25, capsize=2.2, zorder=zorder)


def _retention_panel(ax: plt.Axes, title: str, chance: float,
                     records: Mapping[str, Mapping[str, Any]],
                     xlim: tuple[float, float], panel: str) -> None:
    arms = ("gru", "lstm", "ssm", "fixed_trust")
    markers = {"gru": "o", "lstm": "s", "ssm": "^", "fixed_trust": "D"}
    y = np.arange(len(arms))[::-1]
    baseline = float(records["none"]["mean"])
    span = xlim[1] - xlim[0]
    for position, arm in zip(y, arms):
        stat = records[arm]
        mean = float(stat["mean"])
        low, high = map(float, stat["ci95"])
        arm_color = ARM_COLORS[arm]
        if low > baseline:
            status_color, status, filled = GREEN_DARK, "✓", True
        elif high < baseline:
            status_color, status, filled = RED, "×", False
        else:
            status_color, status, filled = MID, "~", False
        ax.plot([baseline, mean], [position, position], color=arm_color,
                linewidth=1.0, alpha=0.38, zorder=1)
        ax.plot(baseline, position, marker="o", markersize=3.7,
                markerfacecolor="white", markeredgecolor=MID,
                markeredgewidth=0.9, linestyle="none", zorder=3)
        ax.errorbar(mean, position, xerr=[[mean - low], [high - mean]],
                    fmt=markers[arm], markersize=4.7,
                    markerfacecolor=arm_color if filled else "white",
                    markeredgecolor=arm_color, markeredgewidth=1.0,
                    color=arm_color, linewidth=1.15, capsize=2.0, zorder=4)
        ax.text(xlim[1] - 0.012 * span, position,
                f"{mean - baseline:+.3f} {status}", ha="right", va="center",
                fontsize=5.9, color=status_color,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.35),
                zorder=6)
    ax.axvline(chance, color=MID, linestyle=(0, (2, 2)), linewidth=0.8)
    ax.axvline(baseline, color="#9DA5AA", linestyle=(0, (5, 2)),
               linewidth=0.75, zorder=0)
    ax.text(chance, len(arms) - 0.68, f"chance {chance:.3f}", ha="center",
            va="bottom", fontsize=5.7, color=MID,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.25), zorder=5)
    ax.text(baseline, -0.43, f"no state {baseline:.3f}", ha="center",
            va="bottom", fontsize=5.7, color=MID,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2), zorder=5)
    ax.set_xlim(*xlim)
    ax.set_yticks(y, [ARM_SHORT[arm] for arm in arms])
    ax.set_ylim(-0.55, len(arms) - 0.45)
    ax.grid(axis="x", color=LIGHT, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_title(f"({panel}) {title}", loc="left", fontweight="bold", pad=4)


def figure_retention() -> None:
    robust = load_json("outputs/paper_a_robustness_v1/summary.json")
    parent = load_json("outputs/paper_a_expansion/summary.json")
    pusht = load_json("outputs/official_pusht_memory/summary.json")
    arms = ("none", "gru", "lstm", "ssm", "fixed_trust")
    reacher_tasks = (
        ("t1", "Transient-marker recall"),
        ("t3", "Drifting-color recall"),
    )
    pusht_tasks = (
        ("transient-visual-token-recall", "Transient visual-token recall"),
        ("multi-item-visual-binding-recall", "Visual-binding recall (six-way)"),
    )
    fig, axes_array = plt.subplots(2, 2, figsize=(5.45, 3.65))
    axes = tuple(axes_array.ravel())
    for panel, (ax, (task, title)) in enumerate(
            zip(axes[:2], reacher_tasks), start=0):
        original = {arm: _reacher_original_stat(robust, parent, task, arm)
                    for arm in arms}
        short = "marker" if task == "t1" else "color"
        _retention_panel(ax, f"Reacher · {short}", 0.25, original,
                         (0.19, 0.31), chr(ord("a") + panel))
    for panel, (ax, (task, title)) in enumerate(
            zip(axes[2:], pusht_tasks), start=2):
        records = {arm: pusht["results"]["tasks"][task]["arms"][arm]
                   for arm in arms}
        xlim = (0.21, 0.79) if task.startswith("transient") else (0.145, 0.245)
        chance = pusht["results"]["tasks"][task]["chance"]
        short = "token" if task.startswith("transient") else "binding (six-way)"
        _retention_panel(ax, f"PushT · {short}", chance, records, xlim,
                         chr(ord("a") + panel))
    fig.supxlabel("pre-decision readout accuracy  ·  Reacher ordinary / PushT balanced  ·  facet-specific scales",
                  x=0.56, y=0.015, fontsize=6.6, color=MID)
    fig.subplots_adjust(left=0.16, right=0.995, top=0.95, bottom=0.12,
                        wspace=0.42, hspace=0.52)
    save(fig, "fig_mem_retention")


def figure_matched_lewm() -> None:
    """Matched color/age result and the registered age-15 interaction."""
    summary = load_json("outputs/paper_a_matched_color_v1_1/summary.json")
    ages = [int(value) for value in summary["ages"]]
    arms = ("none", "gru", "lstm", "ssm", "fixed_trust")
    markers = {"none": "o", "gru": "o", "lstm": "s",
               "ssm": "^", "fixed_trust": "D"}
    line_styles = {"none": (0, (3, 2)), "gru": "-", "lstm": "-",
                   "ssm": "-", "fixed_trust": "-"}
    line_width = {"none": 1.0, "gru": 0.95, "lstm": 0.95,
                  "ssm": 1.45, "fixed_trust": 1.45}

    fig = plt.figure(figsize=(5.45, 3.02))
    grid = fig.add_gridspec(
        2, 2, height_ratios=(1.52, 1.00), hspace=0.62, wspace=0.23)
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    forest = fig.add_subplot(grid[1, :])

    for panel, (ax, host, title) in enumerate((
            (axes[0], "reacher", "Reacher · matched color"),
            (axes[1], "pusht", "PushT · matched color"))):
        host_result = summary["hosts"][host]
        for arm in arms:
            stats = [host_result["arms"][arm][f"age-{age}"] for age in ages]
            mean = np.asarray([float(value["mean"]) for value in stats])
            low = np.asarray([float(value["ci95"][0]) for value in stats])
            high = np.asarray([float(value["ci95"][1]) for value in stats])
            color = ARM_COLORS[arm]
            alpha = 0.72 if arm in ("gru", "lstm") else 1.0
            ax.plot(ages, mean, marker=markers[arm], markersize=3.9,
                    markerfacecolor="white" if arm in ("none", "lstm") else color,
                    markeredgecolor=color, markeredgewidth=0.9,
                    color=color, linewidth=line_width[arm], alpha=alpha,
                    linestyle=line_styles[arm], zorder=4)
            ax.errorbar(ages, mean, yerr=[mean - low, high - mean],
                        fmt="none", color=color, linewidth=0.65,
                        capsize=1.45, alpha=0.76 * alpha, zorder=3)
        ax.axhline(0.25, color=MID, linewidth=0.8,
                   linestyle=(0, (2, 2)), zorder=0)
        ax.text(14.8, 0.266, "chance", fontsize=5.8, color=MID,
                ha="right", va="bottom")
        ax.set_xticks(ages)
        ax.set_xlim(3.2, 15.8)
        ax.set_ylim(0.18, 0.87)
        ax.grid(axis="both", color=LIGHT, linewidth=0.50)
        ax.set_axisbelow(True)
        ax.set_xlabel("real observations since cue")
        ax.set_title(f"({chr(ord('a') + panel)}) {title}", loc="left",
                     fontweight="bold", pad=3)
    axes[0].set_ylabel("balanced accuracy")
    axes[1].tick_params(labelleft=False)

    handles = tuple(Line2D(
        [0], [0], color=ARM_COLORS[arm], marker=markers[arm],
        markerfacecolor="white" if arm in ("none", "lstm")
        else ARM_COLORS[arm], markeredgecolor=ARM_COLORS[arm],
        linewidth=line_width[arm], linestyle=line_styles[arm],
        label=ARM_SHORT[arm]) for arm in arms)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=5, frameon=False, fontsize=6.4, handlelength=1.55,
               columnspacing=0.82, handletextpad=0.32)

    forest.axvspan(-5.0, 5.0, color=GREEN_PALE, alpha=0.86, zorder=-3)
    forest.axvline(0.0, color=INK, linewidth=0.85,
                   linestyle=(0, (3, 2)), zorder=0)
    forest.axvline(-5.0, color=GREEN_DARK, linewidth=0.6,
                   linestyle=(0, (2, 2)), zorder=0)
    forest.axvline(5.0, color=GREEN_DARK, linewidth=0.6,
                   linestyle=(0, (2, 2)), zorder=0)
    rows = (
        ("Reacher: Fixed-trust\n− state-space",
         summary["hosts"]["reacher"]["fixed_minus_ssm"]["age-15"]),
        ("PushT: Fixed-trust\n− state-space",
         summary["hosts"]["pusht"]["fixed_minus_ssm"]["age-15"]),
    )
    y_positions = (2.0, 1.0)
    for (label, stat), y in zip(rows, y_positions):
        mean = 100.0 * float(stat["mean"])
        low, high = (100.0 * float(value) for value in stat["ci95"])
        forest.errorbar(mean, y, xerr=[[mean - low], [high - mean]],
                        fmt="D", markersize=4.5, markerfacecolor=GREEN,
                        markeredgecolor=GREEN_DARK, markeredgewidth=0.9,
                        color=GREEN_DARK, linewidth=1.15, capsize=2.1,
                        zorder=4)
        forest.text(14.8, y, f"{mean:+.2f} pp", fontsize=6.2,
                    color=GREEN_DARK, ha="right", va="center",
                    fontweight="bold")

    interaction = summary["primary_ranking_interaction"]
    if interaction.get("equivalent_within_margin") is not True:
        raise RuntimeError(
            "refusing to label the registered host interaction equivalent")
    if float(interaction.get("equivalence_margin", -1.0)) != 0.05:
        raise RuntimeError("registered matched-host equivalence margin changed")
    y = 0.0
    mean = 100.0 * float(interaction["mean"])
    low95, high95 = (100.0 * float(value) for value in interaction["ci95"])
    low90, high90 = (100.0 * float(value) for value in interaction["ci90"])
    forest.errorbar(mean, y,
                    xerr=[[mean - low95], [high95 - mean]], fmt="none",
                    color=MID, linewidth=0.85, capsize=2.0, zorder=2)
    forest.errorbar(mean, y,
                    xerr=[[mean - low90], [high90 - mean]], fmt="D",
                    markersize=4.7, markerfacecolor="white",
                    markeredgecolor=GREEN_DARK, markeredgewidth=1.1,
                    color=GREEN_DARK, linewidth=2.15, capsize=2.8,
                    zorder=4)
    forest.text(14.8, y, f"{mean:+.2f} pp · TOST equivalent", fontsize=6.1,
                color=GREEN_DARK, ha="right", va="center",
                fontweight="bold")
    forest.text(0.0, 2.33, "±5 pp equivalence zone",
                fontsize=5.8, color=GREEN_DARK, ha="center", va="center")
    forest.set_yticks((2.0, 1.0, 0.0),
                      (rows[0][0], rows[1][0], "host interaction"))
    forest.set_xlim(-8.0, 15.5)
    forest.set_ylim(-0.42, 2.72)
    forest.set_xlabel("fixed-trust minus state-space (percentage points)")
    forest.grid(axis="x", color=LIGHT, linewidth=0.50)
    forest.set_axisbelow(True)
    forest.set_title("(c) Registered age-15 ranking contrast",
                     loc="left", fontweight="bold", pad=3)

    fig.subplots_adjust(left=0.145, right=0.995, top=0.88, bottom=0.13)
    save(fig, "fig_mem_matched")


def _paired_curve(ax: plt.Axes, ages: list[int], records: list[Mapping[str, Any]],
                  *, color: str, marker: str, label: str,
                  linestyle: str | tuple[int, tuple[int, ...]] = "-",
                  linewidth: float = 1.15) -> None:
    mean = np.asarray([float(value["mean"]) for value in records])
    low = np.asarray([float(value["ci95"][0]) for value in records])
    high = np.asarray([float(value["ci95"][1]) for value in records])
    ax.plot(ages, mean, color=color, marker=marker, markersize=3.6,
            markeredgewidth=0.75, linewidth=linewidth, linestyle=linestyle,
            label=label, zorder=4)
    ax.errorbar(ages, mean, yerr=[mean - low, high - mean], fmt="none",
                color=color, linewidth=0.65, capsize=1.4, alpha=0.82, zorder=3)


def figure_dinowm_carrier() -> None:
    """Verified DINO-WM PushT carrier and persistence contrasts."""
    summary = load_verified_dinowm_pusht()
    ages = [4, 8, 15]
    all_arms = ("none", "gru", "lstm", "ssm", "fixed_trust")
    arms = ("gru", "lstm", "ssm", "fixed_trust")
    markers = {"none": "o", "gru": "o", "lstm": "s", "ssm": "^",
               "fixed_trust": "D"}
    tasks = (
        ("transient-visual-token-recall", "Token recall"),
        ("multi-item-visual-binding-recall", "Visual binding"),
    )
    panels = (
        ("absolute", "full state"),
        ("paired_vs_none", r"$\Delta$ no state"),
        ("full_vs_context_reset", r"$\Delta$ reset"),
    )
    limits: dict[tuple[str, str], float] = {}
    for task, _ in tasks:
        for key, _ in panels[1:]:
            bound = 0.0
            for age in ages:
                records = summary["results"][task]["ages"][str(age)][key]
                for arm in arms:
                    bound = max(bound, *(abs(float(value))
                                         for value in records[arm]["ci95"]))
            limits[(task, key)] = max(0.04, 1.12 * bound)

    fig, axes = plt.subplots(2, 3, figsize=(5.45, 3.10), sharex=True)
    for row, (task, task_label) in enumerate(tasks):
        chance = float(summary["results"][task]["chance"])
        absolute_records = [
            summary["results"][task]["ages"][str(age)]["arms"][arm]
            ["balanced_accuracy"]
            for age in ages for arm in all_arms]
        absolute_low = min(float(record["ci95"][0])
                           for record in absolute_records)
        absolute_high = max(float(record["ci95"][1])
                            for record in absolute_records)
        absolute_span = max(0.10, absolute_high - min(chance, absolute_low))
        absolute_ylim = (
            max(0.0, min(chance, absolute_low) - 0.10 * absolute_span),
            min(1.0, absolute_high + 0.12 * absolute_span),
        )
        for column, (key, panel_label) in enumerate(panels):
            ax = axes[row, column]
            panel_arms = all_arms if key == "absolute" else arms
            for arm in panel_arms:
                if key == "absolute":
                    records = [summary["results"][task]["ages"][str(age)]
                               ["arms"][arm]["balanced_accuracy"]
                               for age in ages]
                else:
                    records = [summary["results"][task]["ages"][str(age)]
                               [key][arm] for age in ages]
                _paired_curve(ax, ages, records, color=ARM_COLORS[arm],
                              marker=markers[arm], label=ARM_SHORT[arm],
                              linestyle=(0, (3, 2)) if arm == "none" else "-",
                              linewidth=1.0 if arm == "none" else 1.15)
            if key == "absolute":
                ax.axhline(chance, color=MID, linewidth=0.85,
                           linestyle=(0, (3, 2)), zorder=0)
                ax.set_ylim(*absolute_ylim)
                ax.text(0.98, 0.05, f"chance {chance:.3f}",
                        transform=ax.transAxes, fontsize=5.3, color=MID,
                        ha="right", va="bottom",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  pad=0.15, alpha=0.88))
            else:
                ax.axhline(0.0, color=MID, linewidth=0.85,
                           linestyle=(0, (3, 2)), zorder=0)
                ax.set_ylim(-limits[(task, key)], limits[(task, key)])
            ax.set_xticks(ages)
            ax.grid(axis="both", color=LIGHT, linewidth=0.50)
            ax.set_axisbelow(True)
            panel = chr(ord("a") + 3 * row + column)
            short_task = "Token" if task_label == "Token recall" else "Binding"
            ax.set_title(f"({panel}) {short_task} · {panel_label}",
                         loc="left", fontweight="bold", pad=3)
            if column == 0:
                ax.set_ylabel("balanced accuracy")
            if row == 1:
                ax.set_xlabel("cue age")
    handles = tuple(Line2D(
        [0], [0], color=ARM_COLORS[arm], marker=markers[arm],
        linewidth=1.15, linestyle=(0, (3, 2)) if arm == "none" else "-",
        label=("No state (full-state only)" if arm == "none"
               else ARM_SHORT[arm])) for arm in all_arms)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=5, frameon=False, fontsize=6.15, handlelength=1.35,
               columnspacing=0.68, handletextpad=0.26)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.89, bottom=0.145,
                        wspace=0.33, hspace=0.43)
    save(fig, "fig_mem_dinowm_carrier")


def figure_pointmaze_retention_use() -> None:
    """Verified PointMaze retention controls and external executed use."""
    _, carrier, use = load_verified_pointmaze()
    ages = [4, 8, 15]
    arms = ("none", "gru", "lstm", "ssm", "fixed_trust")
    learned = arms[1:]
    markers = {"gru": "o", "lstm": "s", "ssm": "^", "fixed_trust": "D"}
    fig = plt.figure(figsize=(5.45, 3.30))
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1.0, 1.04), hspace=0.50, wspace=0.36)
    absolute = fig.add_subplot(grid[0, 0])
    absolute_markers = {
        "none": "o", "gru": "o", "lstm": "s", "ssm": "^",
        "fixed_trust": "D"}
    absolute_records: list[Mapping[str, Any]] = []
    for arm in arms:
        records = [carrier["results"][str(age)]["arms"][arm]
                   ["balanced_accuracy"] for age in ages]
        absolute_records.extend(records)
        _paired_curve(
            absolute, ages, records, color=ARM_COLORS[arm],
            marker=absolute_markers[arm], label=ARM_SHORT[arm],
            linestyle=(0, (3, 2)) if arm == "none" else "-",
            linewidth=1.0 if arm == "none" else 1.15)
    chance = 0.25
    absolute.axhline(chance, color=MID, linewidth=0.85,
                     linestyle=(0, (3, 2)), zorder=0)
    absolute_low = min(float(record["ci95"][0])
                       for record in absolute_records)
    absolute_high = max(float(record["ci95"][1])
                        for record in absolute_records)
    absolute_span = max(0.10, absolute_high - min(chance, absolute_low))
    absolute.set_ylim(
        max(0.0, min(chance, absolute_low) - 0.10 * absolute_span),
        min(1.0, absolute_high + 0.12 * absolute_span))
    absolute.set_xticks(ages)
    absolute.set_xlabel("cue age")
    absolute.set_ylabel("balanced accuracy")
    absolute.set_title("(a) Full-state retention", loc="left",
                       fontweight="bold", pad=3)
    absolute.grid(axis="both", color=LIGHT, linewidth=0.50)
    absolute.set_axisbelow(True)
    contrast_axes = (
        fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]))
    contrast_specs = (
        ("paired_vs_none", "(b) Full − no state"),
        ("full_vs_context_reset", "(c) Full − context reset"),
    )
    common_low = 0.0
    common_high = 0.0
    for key, _ in contrast_specs:
        for arm in learned:
            for age in ages:
                record = carrier["results"][str(age)][key][arm]
                low, high = (float(value) for value in record["ci95"])
                common_low = min(common_low, low)
                common_high = max(common_high, high)
    # Keep zero visible without devoting half of both panels to an empty
    # negative range.  The two contrasts retain a shared scale.
    common_ymin = min(-0.04, 1.12 * common_low)
    common_ymax = max(0.04, 1.08 * common_high)
    for ax, (key, title) in zip(contrast_axes, contrast_specs):
        for arm in learned:
            records = [carrier["results"][str(age)][key][arm] for age in ages]
            _paired_curve(ax, ages, records, color=ARM_COLORS[arm],
                          marker=markers[arm], label=ARM_SHORT[arm])
        ax.axhline(0.0, color=MID, linewidth=0.85,
                   linestyle=(0, (3, 2)), zorder=0)
        ax.set_ylim(common_ymin, common_ymax)
        ax.set_xticks(ages)
        ax.set_xlabel("cue age")
        ax.set_title(title, loc="left", fontweight="bold", pad=3)
        ax.grid(axis="both", color=LIGHT, linewidth=0.50)
        ax.set_axisbelow(True)
    contrast_axes[0].set_ylabel("balanced-accuracy change")
    contrast_axes[1].tick_params(labelleft=True)

    execution = fig.add_subplot(grid[1, 1])
    y = np.arange(len(arms))[::-1]
    none_mean = float(use["arms"]["none"]["executed_success"]["mean"])
    plotted_intervals: list[tuple[float, float]] = []
    for position, arm in zip(y, arms):
        record = use["arms"][arm]["executed_success"]
        mean = float(record["mean"])
        if arm == "none":
            low, high = (float(value) for value in record["ci95"])
        else:
            paired = use["arms"][arm]["contrast_vs_none"]
            paired_center = none_mean + float(paired["mean"])
            if not math.isclose(mean, paired_center, rel_tol=0.0,
                                abs_tol=1e-10):
                raise RuntimeError(
                    f"PointMaze absolute/paired execution mismatch: {arm}")
            low, high = (none_mean + float(value)
                         for value in paired["ci95"])
        plotted_intervals.append((low, high))
        resolved = bool(use["arms"][arm].get("resolved_execution_gain", False))
        execution.errorbar(mean, position,
                           xerr=[[mean - low], [high - mean]],
                           fmt="*" if resolved else "o",
                           markersize=6.0 if resolved else 4.0,
                           markerfacecolor=ARM_COLORS[arm],
                           markeredgecolor=ARM_COLORS[arm],
                           color=ARM_COLORS[arm], linewidth=0.85,
                           capsize=1.7, zorder=4)
    random_record = use["realized_random_goal"]
    random_mean = float(random_record["mean"])
    random_low, random_high = (float(value) for value in random_record["ci95"])
    oracle = float(use["oracle_executed_success"])
    execution.axvspan(random_low, random_high, color=ORANGE, alpha=0.12,
                      linewidth=0, zorder=-2)
    execution.axvline(random_mean, color=ORANGE, linewidth=1.0,
                      linestyle=(0, (3, 2)), zorder=0)
    execution.axvline(none_mean, color=ARM_COLORS["none"], linewidth=0.85,
                      linestyle=(0, (1, 2)), zorder=0)
    execution_labels = {
        "none": "No state", "gru": "GRU", "lstm": "LSTM",
        "ssm": "State-space", "fixed_trust": "Fixed-trust"}
    execution.set_yticks(y, [execution_labels[arm] for arm in arms])
    execution.tick_params(axis="y", labelsize=6.4, pad=2)
    data_low = min(random_low, *(value[0] for value in plotted_intervals))
    data_high = max(random_high, *(value[1] for value in plotted_intervals))
    padding = max(0.025, 0.13 * max(0.08, data_high - data_low))
    execution.set_xlim(max(0.0, data_low - padding),
                       min(1.0, data_high + padding))
    execution.set_ylim(-0.45, 4.45)
    execution.set_xlabel("executed success")
    execution.set_title("(d) External execution · age 15", loc="left",
                        fontweight="bold", pad=3)
    execution.grid(axis="x", color=LIGHT, linewidth=0.50)
    execution.set_axisbelow(True)
    execution.text(random_mean, 0.02, "random", color=ORANGE,
                   transform=execution.get_xaxis_transform(),
                   fontsize=5.8, ha="center", va="bottom",
                   bbox=dict(facecolor="white", edgecolor="none",
                             pad=0.10, alpha=0.86))
    execution.text(0.99, 0.89, f"oracle {oracle:.3f} · off axis",
                   transform=execution.transAxes, color=GREEN_DARK,
                   fontsize=5.6, ha="right", va="top")
    execution.text(0.99, 0.98, "★ resolved vs both controls",
                   transform=execution.transAxes, color=GREEN_DARK,
                   fontsize=5.4, ha="right", va="top")

    handles = tuple(Line2D(
        [0], [0], color=ARM_COLORS[arm], marker=absolute_markers[arm],
        linewidth=1.15,
        linestyle=(0, (3, 2)) if arm == "none" else "-",
        label=("No state (a,d)" if arm == "none" else ARM_SHORT[arm]))
        for arm in arms)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.995),
               ncol=5, frameon=False, fontsize=5.9, handlelength=1.25,
               columnspacing=0.62, handletextpad=0.24)
    fig.subplots_adjust(left=0.105, right=0.992, top=0.84, bottom=0.145)
    save(fig, "fig_mem_pointmaze")


def _stat_errorbar(ax: plt.Axes, stat: Mapping[str, Any], y: float,
                   color: str, marker: str = "o") -> None:
    _forest_point(ax, y, stat, color, marker, filled=True, size=4.8)


def figure_diagnosis() -> None:
    context_path = ROOT / "outputs/paper_a_context_rollout_seed_extension_v1/summary.json"
    if not context_path.is_file():
        raise FileNotFoundError(
            "aggregate the five-seed context/rollout extension first")
    context = json.loads(context_path.read_text())
    context_tasks = context["long_context"]["tasks"]
    task_specs = (
        ("transient-marker-recall", "o", "-", "marker"),
        ("drifting-color-recall", "s", (0, (4, 2)), "color"),
    )
    history_key = "histories"
    use = load_json("outputs/paper_a_delayed_goal_use_v1/summary.json")
    pusht_use = load_json("outputs/pusht_downstream_use_v1/summary.json")

    fig = plt.figure(figsize=(5.45, 2.92))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.42, 1.18), wspace=0.54)
    ax = fig.add_subplot(grid[0, 0])
    bx = fig.add_subplot(grid[0, 1])

    histories = [3, 16, 32, 56]
    ax.axvspan(1, 43.5, color="#F3F4F4", zorder=-3)
    ax.axvspan(43.5, 53.5, color="#FAF5E9", zorder=-3)
    ax.axvspan(53.5, 59, color=GREEN_PALE, alpha=0.82, zorder=-3)
    ax.axvline(43.5, color=MID, linewidth=0.65, linestyle=(0, (2, 2)))
    ax.axvline(53.5, color=MID, linewidth=0.65, linestyle=(0, (2, 2)))
    for task, marker, linestyle, short in task_specs:
        record = context_tasks[task][history_key]
        raw = [record[str(history)]["raw_legal_context_readout"]["value"]
               for history in histories]
        predictor_stats = [record[str(history)][
            "trained_predictor_semantic_accuracy"] for history in histories]
        predictor = [float(stat["mean"]) for stat in predictor_stats]
        pred_low = [float(stat["ci95"][0]) for stat in predictor_stats]
        pred_high = [float(stat["ci95"][1]) for stat in predictor_stats]
        ax.plot(histories, raw, color=GREEN_DARK, linewidth=1.35,
                marker=marker, markersize=4.1, markerfacecolor="white",
                markeredgewidth=1.0, linestyle=linestyle, zorder=4)
        ax.plot(histories, predictor, color=INK, linewidth=1.15,
                marker=marker, markersize=3.6, markerfacecolor=INK,
                linestyle=linestyle, zorder=3)
        ax.errorbar(histories, predictor,
                    yerr=[np.asarray(predictor) - np.asarray(pred_low),
                          np.asarray(pred_high) - np.asarray(predictor)],
                    fmt="none", color=INK, linewidth=0.8, capsize=1.7,
                    zorder=2)
    ax.axhline(0.25, color=MID, linewidth=0.9, linestyle=(0, (2, 2)))
    ax.text(35.0, 0.266, "chance .25", fontsize=5.8, color=MID,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.25))
    ax.text(21, 0.875, r"cue outside ($H\leq43$)", fontsize=6.0,
            color=MID, ha="center")
    ax.text(48.5, 0.892, "some", fontsize=5.7,
            color=ORANGE, ha="center", fontweight="bold")
    ax.text(56.2, 0.892, "all", fontsize=5.7,
            color=GREEN_DARK, ha="center", fontweight="bold")
    ax.text(55.4, 0.836, "color raw .825", fontsize=5.9,
            color=GREEN_DARK, ha="right", va="bottom",
            bbox=dict(facecolor=GREEN_PALE, edgecolor="none", pad=0.2),
            zorder=6)
    ax.text(55.4, 0.748, "marker raw .771", fontsize=5.9,
            color=GREEN_DARK, ha="right", va="top",
            bbox=dict(facecolor=GREEN_PALE, edgecolor="none", pad=0.2),
            zorder=6)
    ax.text(55.4, 0.271, "marker predictor .253", fontsize=5.8,
            color=INK, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.15), zorder=6)
    ax.text(55.4, 0.184, "color predictor .199", fontsize=5.8,
            color=INK, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.15), zorder=6)
    ax.text(2.5, 0.132, "○ marker   □ color", fontsize=5.9, color=MID,
            ha="left", va="bottom")
    ax.set_xticks(histories)
    ax.set_xlim(1, 59)
    ax.set_ylim(0.12, 0.91)
    ax.set_xlabel("pre-decision context length")
    ax.set_ylabel("semantic readout accuracy")
    ax.grid(axis="y", color=LIGHT, linewidth=0.55)
    ax.set_title("(a) Access appears; predictor output does not",
                 loc="left", fontweight="bold")

    reacher_task = use["tasks"]["t3"]
    task_token = pusht_use["tasks"]["transient-visual-token-recall"]
    task_binding = pusht_use["tasks"]["multi-item-visual-binding-recall"]
    reacher_sources = reacher_task["sources"]
    reacher_none = float(reacher_sources["no-persistent-carrier"]["endpoints"]
                         ["executed_success_rate"]["mean"])
    rows = (
        {
            "label": "Reacher color", "none": reacher_none,
            "ssm": float(reacher_sources["diagonal-state-space-carrier"]
                         ["endpoints"]["executed_success_rate"]["mean"]),
            "ssm_ci": tuple(float(reacher_none + value) for value in (
                reacher_task["paired_contrasts"]["ssm_vs_none"]
                ["executed_success_difference"]["ci_low"],
                reacher_task["paired_contrasts"]["ssm_vs_none"]
                ["executed_success_difference"]["ci_high"])),
            "fixed": float(reacher_sources["fixed-trust-predict-correct"]
                           ["endpoints"]["executed_success_rate"]["mean"]),
            "fixed_ci": tuple(float(reacher_none + value) for value in (
                reacher_task["paired_contrasts"]["fixed_trust_vs_none"]
                ["executed_success_difference"]["ci_low"],
                reacher_task["paired_contrasts"]["fixed_trust_vs_none"]
                ["executed_success_difference"]["ci_high"])),
            # The Reacher study did not execute a random-goal arm.  Four-way
            # label chance is not a physical-success reference, so do not put
            # a misleading .25 marker on the executed-success axis.
            "random": None, "chance": None, "oracle": 0.917,
        },
        {
            "label": "PushT token", "none": task_token["arms"]["none"]
                     ["executed_success"],
            "ssm": task_token["arms"]["ssm"]["executed_success"],
            "ssm_ci": tuple(task_token["arms"]["none"]["executed_success"]
                            + value for value in task_token["arms"]["ssm"]
                            ["contrast_vs_none"]["executed_success"]["ci95"]),
            "fixed": task_token["arms"]["fixed_trust"]["executed_success"],
            "fixed_ci": tuple(task_token["arms"]["none"]["executed_success"]
                              + value for value in task_token["arms"]
                              ["fixed_trust"]["contrast_vs_none"]
                              ["executed_success"]["ci95"]),
            "random": task_token["random_goal_executed_success"],
            "chance": 0.25, "oracle": task_token["oracle_executed_success"],
        },
        {
            "label": "PushT binding", "none": task_binding["arms"]["none"]
                     ["executed_success"],
            "ssm": task_binding["arms"]["ssm"]["executed_success"],
            "ssm_ci": tuple(task_binding["arms"]["none"]["executed_success"]
                            + value for value in task_binding["arms"]["ssm"]
                            ["contrast_vs_none"]["executed_success"]["ci95"]),
            "fixed": task_binding["arms"]["fixed_trust"]["executed_success"],
            "fixed_ci": tuple(task_binding["arms"]["none"]["executed_success"]
                              + value for value in task_binding["arms"]
                              ["fixed_trust"]["contrast_vs_none"]
                              ["executed_success"]["ci95"]),
            "random": task_binding["random_goal_executed_success"],
            "chance": 1 / 6, "oracle": task_binding["oracle_executed_success"],
        },
    )
    ys = np.arange(len(rows))[::-1]
    for y, row in zip(ys, rows):
        baseline = float(row["none"])
        bx.plot(baseline, y, marker="o", markersize=4.2,
                markerfacecolor="white", markeredgecolor=MID,
                markeredgewidth=1.0, linestyle="none", zorder=5)
        for offset, key, color, marker in (
                (0.11, "ssm", TEAL, "^"),
                (-0.11, "fixed", GREEN_DARK, "D")):
            mean = float(row[key])
            low, high = map(float, row[f"{key}_ci"])
            bx.plot([baseline, mean], [y + offset, y + offset], color=color,
                    linewidth=0.9, alpha=0.38, zorder=2)
            bx.errorbar(mean, y + offset,
                        xerr=[[mean - low], [high - mean]], fmt=marker,
                        color=color, markerfacecolor=color,
                        markeredgecolor=color, markersize=4.2,
                        linewidth=1.0, capsize=1.8, zorder=4)
            delta = 100 * (mean - baseline)
            bx.text(max(mean, high) + 0.004, y + offset,
                    f"{delta:+.2f}", fontsize=5.8, color=color,
                    ha="left", va="center", fontweight="bold")
        if row["random"] is not None:
            bx.plot(float(row["random"]), y, marker="*", markersize=5.8,
                    color=ORANGE, linestyle="none", zorder=5)
        if row["chance"] is not None:
            bx.plot(float(row["chance"]), y, marker="x", markersize=4.4,
                    color=MID, linestyle="none", zorder=4)
        bx.text(0.329, y + 0.25,
                f"oracle {float(row['oracle']):.3f} (off-axis)",
                fontsize=5.7, color=MID, ha="right", va="center")
    bx.set_xlim(0.14, 0.335)
    # Reserve a small in-panel header band for the key.  Keeping the key
    # inside the axes prevents it from colliding with the panel title or the
    # neighbouring context plot when the two-column figure is typeset.
    bx.set_ylim(-0.45, len(rows) + 0.05)
    bx.set_yticks(ys, [row["label"] for row in rows])
    bx.tick_params(axis="y", labelsize=6.1)
    bx.grid(axis="x", color=LIGHT, linewidth=0.55)
    bx.set_xlabel("absolute executed-success rate")
    bx.set_title("(b) External-consumer execution", loc="left",
                 fontweight="bold")
    handles = (
        Line2D([0], [0], marker="o", markerfacecolor="white",
               markeredgecolor=MID, linestyle="none", label="no state"),
        Line2D([0], [0], marker="^", color=TEAL, linestyle="none",
               label="state-space"),
        Line2D([0], [0], marker="D", color=GREEN_DARK,
               linestyle="none", label="fixed-trust"),
        Line2D([0], [0], marker="*", color=ORANGE,
               linestyle="none", label="random"),
        Line2D([0], [0], marker="x", color=MID,
               linestyle="none", label="nominal $1/K$"),
    )
    bx.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
              ncol=3, frameon=False, fontsize=6.0, handletextpad=0.2,
              columnspacing=0.50, labelspacing=0.25, borderaxespad=0.0)

    fig.subplots_adjust(left=0.10, right=0.995, top=0.90, bottom=0.19)
    save(fig, "fig_mem_diagnosis")


def figure_appendix_endpoint() -> None:
    """Show why trajectory-average decodability is not the legal final read."""
    robust = load_json("outputs/paper_a_robustness_v1/summary.json")
    arms = ("gru", "lstm", "ssm", "fixed_trust")
    data = robust["fresh_validation"]["equal_task_bank_arms"]
    fig, ax = plt.subplots(figsize=(5.45, 1.80))
    x = np.array([0.0, 1.0])
    styles = {
        "gru": ("o", "-", "#8E979C", 0.62),
        "lstm": ("s", (0, (4, 2)), "#737E84", 0.62),
        "ssm": ("^", "-", TEAL, 1.0),
        "fixed_trust": ("D", (0, (3, 1.5)), GREEN_DARK, 1.0),
    }
    ax.axvspan(-0.28, 0.5, color=BLUE_PALE, alpha=0.55, zorder=-3)
    ax.axvspan(0.5, 1.36, color="#F3F4F4", zorder=-3)
    label_offsets = {"gru": -0.009, "lstm": 0.009,
                     "ssm": 0.006, "fixed_trust": -0.006}
    for arm in arms:
        final = data[arm]["accuracy"]
        trajectory = data[arm]["trajectory_accuracy"]
        means = np.array([final["mean"], trajectory["mean"]])
        lows = np.array([final["ci95"][0], trajectory["ci95"][0]])
        highs = np.array([final["ci95"][1], trajectory["ci95"][1]])
        marker, linestyle, color, alpha = styles[arm]
        width = 1.35 if arm in ("ssm", "fixed_trust") else 0.95
        ax.plot(x, means, color=color, linewidth=width,
                marker=marker, linestyle=linestyle, markersize=4.4,
                alpha=alpha, zorder=3)
        ax.errorbar(x, means, yerr=[means - lows, highs - means],
                    fmt="none", color=color, linewidth=0.9,
                    capsize=1.8, alpha=alpha, zorder=2)
        ax.text(1.055, means[1] + label_offsets[arm], ARM_SHORT[arm],
                fontsize=6.7, color=color, va="center",
                fontweight="bold" if arm in ("ssm", "fixed_trust") else "normal")
    ax.axhline(0.25, color=MID, linewidth=0.9, linestyle=(0, (3, 2)))
    ax.text(0.5, 0.271, "four-way chance", fontsize=6.3, color=MID,
            ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88,
                      pad=0.35), zorder=6)
    ax.set_xticks(x, ["Pre-decision state read\n(primary)",
                      "Trajectory feature\n(exploratory)"])
    ax.set_xlim(-0.28, 1.34)
    ax.set_ylim(0.225, 0.475)
    ax.set_ylabel("fresh-bank accuracy\n(task/bank mean)", fontsize=7.8)
    ax.grid(axis="y", color=LIGHT, linewidth=0.55)
    ax.text(0.0, 0.468, "pre-decision state read", ha="center", va="top",
            fontsize=6.5, color=BLUE, fontweight="bold")
    ax.text(0.73, 0.468, "retrospective trajectory feature", ha="center",
            va="top", fontsize=6.25, color=MID, fontweight="bold")
    ax.tick_params(axis="both", labelsize=7.4)
    fig.subplots_adjust(left=0.12, right=0.91, top=0.95, bottom=0.29)
    save(fig, "fig_mem_appendix_endpoint")


def figure_cue_age() -> None:
    """Paired fixed-endpoint cue-placement curves from the registered sweep."""
    summary = load_json("outputs/paper_a_evidence_age_v1/strict/summary.json")
    panels = (
        ("reacher", (("t1", "marker"), ("t3", "color")),
         "Reacher · identical endpoint and trajectory"),
        ("pusht", (("transient-visual-token-recall", "token"),
                   ("multi-item-visual-binding-recall", "binding")),
         "PushT · identical endpoint and trajectory"),
    )
    carriers = (
        ("ssm", "State-space", TEAL),
        ("fixed_trust", "Fixed-trust", GREEN_DARK),
    )
    task_styles = (("o", "-"), ("s", (0, (4, 2))))
    fig = plt.figure(figsize=(5.45, 2.62))
    grid = fig.add_gridspec(
        2, 2, height_ratios=(0.42, 2.0), hspace=0.32, wspace=0.18)
    timeline = fig.add_subplot(grid[0, :])
    axes = (fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1], sharey=fig.axes[-1]))

    # A compact intervention key makes the line charts self-contained: only
    # cue timing changes; the endpoint and all non-cue episode content stay
    # fixed.  The three translucent cards denote registered cue positions,
    # not three cues in one episode.
    timeline.set_xlim(0, 1)
    timeline.set_ylim(0, 1)
    timeline.axis("off")
    timeline.add_patch(FancyBboxPatch(
        (0.002, 0.08), 0.992, 0.84,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        facecolor="#F7F9F5", edgecolor="#D8E1CC", linewidth=0.65))
    timeline.text(0.018, 0.65, "Fixed-endpoint intervention",
                  fontsize=6.6, fontweight="bold", color=INK, va="center")
    timeline.text(0.018, 0.30,
                  "same episode · actions · late inputs · one carrier per seed across ages",
                  fontsize=5.7, color=MID, va="center")
    timeline.plot([0.43, 0.955], [0.40, 0.40], color=MID,
                  linewidth=0.9, zorder=1)
    for index, x in enumerate((0.53, 0.69, 0.83)):
        timeline.add_patch(Rectangle(
            (x - 0.018, 0.28), 0.036, 0.24,
            facecolor=GREEN, edgecolor=GREEN_DARK, linewidth=0.65,
            alpha=(0.28, 0.52, 0.86)[index], zorder=3))
    timeline.add_patch(FancyArrowPatch(
        (0.84, 0.72), (0.52, 0.72), arrowstyle="-|>",
        mutation_scale=7.5, linewidth=0.8, color=GREEN_DARK))
    timeline.text(0.68, 0.77, "greater evidence age  ←  move cue earlier",
                  fontsize=5.8, color=GREEN_DARK, ha="center", va="bottom")
    timeline.plot([0.955, 0.955], [0.19, 0.73], color=TEAL,
                  linewidth=1.2, zorder=3)
    timeline.plot(0.955, 0.40, marker="<", color=TEAL, markersize=4.5)
    timeline.text(0.970, 0.47, "fixed pre-decision\nstate read",
                  fontsize=5.7, color=TEAL, ha="right", va="bottom",
                  linespacing=0.92)
    extrema = [0.0]
    for ax, (host, tasks, title), panel in zip(axes, panels, "ab"):
        for task_index, ((task, short), (marker, linestyle)) in enumerate(
                zip(tasks, task_styles)):
            record = summary["tasks"][host][task]
            ages = np.asarray(record["ages"], dtype=float)
            for carrier, carrier_label, color in carriers:
                stats = [record["paired_vs_no_carrier"][carrier][f"age-{age}"]
                         for age in record["ages"]]
                mean = np.asarray([float(stat["mean"]) for stat in stats])
                low = np.asarray([float(stat["ci95"][0]) for stat in stats])
                high = np.asarray([float(stat["ci95"][1]) for stat in stats])
                extrema.extend(low.tolist())
                extrema.extend(high.tolist())
                ax.plot(ages, mean, color=color, linestyle=linestyle,
                        marker=marker, markersize=3.9, linewidth=1.25,
                        markerfacecolor="white" if task_index else color,
                        markeredgecolor=color, markeredgewidth=0.9, zorder=4)
                ax.errorbar(ages, mean, yerr=[mean - low, high - mean],
                            fmt="none", color=color, linewidth=0.75,
                            capsize=1.5, alpha=0.82, zorder=3)
        ax.axhline(0.0, color=MID, linewidth=0.85, linestyle=(0, (2, 2)))
        ax.set_xticks(summary["tasks"][host][tasks[0][0]]["ages"])
        ax.grid(axis="both", color=LIGHT, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("observations since the cue")
        ax.set_title(f"({panel}) {title}", loc="left", fontweight="bold")
    low_limit = min(-0.04, min(extrema) - 0.035)
    high_limit = max(0.10, max(extrema) + 0.045)
    axes[0].set_ylim(low_limit, high_limit)
    axes[0].set_ylabel("accuracy gain over no state")
    legend = (
        Line2D([0], [0], color=TEAL, linewidth=1.4,
               label="State-space"),
        Line2D([0], [0], color=GREEN_DARK, linewidth=1.4,
               label="Fixed-trust"),
        Line2D([0], [0], color=MID, marker="o", markerfacecolor=MID,
               linewidth=1.2, label="marker / token"),
        Line2D([0], [0], color=MID, marker="s", markerfacecolor="white",
               linestyle=(0, (4, 2)), linewidth=1.2,
               label="color / binding"),
    )
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.52, -0.005), handlelength=1.7,
               columnspacing=1.25, handletextpad=0.45, fontsize=6.4)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.985, bottom=0.22,
                        wspace=0.18, hspace=0.32)
    save(fig, "fig_mem_cue_age")


def figure_appendix_dinowm() -> None:
    """Native DINO-WM imagination transport: semantics versus feature distance."""
    summary = load_json(
        "outputs/dinowm_native_pusht_audit_v2r2/formal/summary.json")
    tasks = (
        ("transient-visual-token-recall", "token", GREEN_DARK, "o", 0.25),
        ("multi-item-visual-binding-recall", "binding", TEAL, "s", 1 / 6),
    )
    ages = np.asarray([1, 4, 8, 15], dtype=float)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.45, 2.28))
    for key, label, color, marker, chance in tasks:
        record = summary["tasks"][key]
        probe = record["open_loop_decodability"]
        mean = np.asarray([probe[str(int(age))]["balanced_accuracy"]
                           for age in ages], dtype=float)
        low = np.asarray([probe[str(int(age))]
                          ["validation_accuracy_episode_bootstrap"]["lower"]
                          for age in ages], dtype=float)
        high = np.asarray([probe[str(int(age))]
                           ["validation_accuracy_episode_bootstrap"]["upper"]
                           for age in ages], dtype=float)
        ax.plot(ages, mean, color=color, marker=marker, linewidth=1.35,
                markersize=4.2, label=label, zorder=4)
        ax.errorbar(ages, mean, yerr=[mean - low, high - mean], fmt="none",
                    color=color, linewidth=0.8, capsize=1.7, zorder=3)
        ax.axhline(chance, color=color, linewidth=0.7,
                   linestyle=(0, (2, 2)), alpha=0.65)

        separation = record["paired_counterfactual_separation"]["ages"]
        ratio = np.asarray([separation[str(int(age))]["transport_ratio"]
                            for age in ages], dtype=float)
        ratio_low = np.asarray([
            separation[str(int(age))]["transport_ratio_lower"]
            for age in ages], dtype=float)
        ratio_high = np.asarray([
            separation[str(int(age))]["transport_ratio_upper"]
            for age in ages], dtype=float)
        bx.plot(ages, ratio, color=color, marker=marker, linewidth=1.35,
                markersize=4.2, label=label, zorder=4)
        bx.errorbar(ages, ratio,
                    yerr=[ratio - ratio_low, ratio_high - ratio], fmt="none",
                    color=color, linewidth=0.8, capsize=1.7, zorder=3)
    ax.set_title("(a) Semantic cue readout decays", loc="left",
                 fontweight="bold")
    ax.set_ylabel("open-loop balanced accuracy")
    ax.text(13.2, 0.255, "token chance", fontsize=5.7,
            color=GREEN_DARK, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2))
    ax.text(13.2, 0.172, "binding chance", fontsize=5.7,
            color=TEAL, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.2))
    ax.set_ylim(0.12, 1.03)
    bx.set_title("(b) Counterfactual separation persists", loc="left",
                 fontweight="bold")
    bx.axhline(1.0, color=MID, linewidth=0.75, linestyle=(0, (3, 2)))
    bx.text(14.7, 1.015, "cue-level RMS", fontsize=5.8, color=MID,
            ha="right", va="bottom")
    bx.set_ylabel("counterfactual RMS / cue RMS")
    bx.set_ylim(0.0, 1.23)
    for axis in (ax, bx):
        axis.set_xticks(ages.astype(int))
        axis.set_xlabel("imagined observations since cue")
        axis.grid(axis="both", color=LIGHT, linewidth=0.5)
        axis.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=2,
              handlelength=1.5, columnspacing=0.9)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.88, bottom=0.23,
                        wspace=0.34)
    save(fig, "fig_mem_appendix_dinowm")


def _rollout_task_record(summary: Mapping[str, Any], task: str,
                         objective: str, horizon: int) -> Mapping[str, Any]:
    return summary["learned_rollout"]["tasks"][task]["objectives"][objective][
        "horizons"][str(horizon)]


def figure_appendix_rollout() -> None:
    path = ROOT / "outputs/paper_a_context_rollout_seed_extension_v1/summary.json"
    if not path.is_file():
        raise FileNotFoundError(
            "aggregate the five-seed context/rollout extension first")
    summary = json.loads(path.read_text())
    task_keys = ("transient-marker-recall", "drifting-color-recall")
    horizons = [1, 2, 4, 8, 16]
    tasks = ((task_keys[0], TEAL, "marker"),
             (task_keys[1], ORANGE, "color"))
    objectives = (
        ("one-step", "o", "-", "one-step"),
        ("overshoot-8", "s", (0, (4, 2)), "overshoot"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.45, 2.90))
    for task, color, short in tasks:
        for objective, marker, linestyle, objective_short in objectives:
            rows = [_rollout_task_record(summary, task, objective, horizon)
                    for horizon in horizons]
            ratio = [row["model_to_copy_ratio"]["mean"] for row in rows]
            advantage = [row["true_action_advantage"]["mean"] for row in rows]
            ratio_low = [row["model_to_copy_ratio"]["ci95"][0]
                         for row in rows]
            ratio_high = [row["model_to_copy_ratio"]["ci95"][1]
                          for row in rows]
            advantage_low = [row["true_action_advantage"]["ci95"][0]
                             for row in rows]
            advantage_high = [row["true_action_advantage"]["ci95"][1]
                              for row in rows]
            label = f"{short} / {objective_short}"
            axes[0].errorbar(
                horizons, ratio,
                yerr=[np.asarray(ratio) - np.asarray(ratio_low),
                      np.asarray(ratio_high) - np.asarray(ratio)],
                color=color, marker=marker, linestyle=linestyle,
                linewidth=1.35, markersize=3.8, elinewidth=0.75,
                capsize=1.8, label=label)
            axes[1].errorbar(
                horizons, advantage,
                yerr=[np.asarray(advantage) - np.asarray(advantage_low),
                      np.asarray(advantage_high) - np.asarray(advantage)],
                color=color, marker=marker, linestyle=linestyle,
                linewidth=1.35, markersize=3.8, elinewidth=0.75,
                capsize=1.8, label=label)
    axes[0].axhline(1.0, color=MID, linewidth=0.9, linestyle=(0, (3, 2)))
    axes[1].axhline(0.0, color=MID, linewidth=0.9, linestyle=(0, (3, 2)))
    axes[0].set_yscale("log")
    axes[0].set_ylabel("model / copy-last error")
    axes[1].set_ylabel("true-action advantage")
    for panel, ax in zip(("a", "b"), axes):
        ax.set_xscale("log", base=2)
        ax.set_xticks(horizons, [str(horizon) for horizon in horizons])
        ax.set_xlabel("rollout horizon")
        ax.grid(axis="y", color=LIGHT, linewidth=0.7)
        ax.set_title(
            f"({panel}) " + ("Copy-last reference" if panel == "a" else
                              "Shuffled-action reference"),
            loc="left", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.99),
               ncol=4, frameon=False, fontsize=7.0,
               columnspacing=0.85, handlelength=1.7, handletextpad=0.35)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.79, bottom=0.19,
                        wspace=0.36)
    save(fig, "fig_mem_appendix_rollout")


def _gate_value(path: Path, gate: str, field: str) -> float:
    value = json.loads(path.read_text())["gates"][gate]["value"]
    return float(value[field] if isinstance(value, dict) else value)


def figure_appendix_shell_gates() -> None:
    """Visualize every shell-game stop without implying a carrier result."""
    entries: list[tuple[str, float, float]] = []
    for version, root in (
            ("Initial overlay · formal",
             ROOT / "outputs/official_shell_game_capacity/cache"),
            ("Salience overlay · dev",
             ROOT / "outputs/official_shell_game_capacity_v2/development-cache"),
            ("Cue card · dev",
             ROOT / "outputs/official_shell_game_capacity_v3/development-cache"),
            ("Cue card · formal",
             ROOT / "outputs/official_shell_game_capacity_v3/cache")):
        for stage, capacity in (("single-item", 1), ("two-item", 2),
                                ("four-item", 4)):
            if "dev" in version:
                path = root / stage / "salience_selection.json"
                if not path.is_file():
                    continue
                record = json.loads(path.read_text())
                admission = record["development_diagnostics"]
            else:
                path = root / stage / "admission.json"
                if not path.is_file():
                    continue
                admission = json.loads(path.read_text())
            cue = float(admission["gates"]["cue_initial_slot_availability"][
                "value"]["minimum_item_accuracy"])
            leakage = float(admission["gates"]["final_context_target_leakage"][
                "value"]["maximum_item_accuracy"])
            entries.append((f"{version} · {capacity} item", cue, leakage))
    y = np.arange(len(entries))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(5.45, 3.32), sharey=True)
    cue_threshold = 0.75
    leakage_threshold = 1 / 3 + 0.05
    for position, (label, cue, leakage) in zip(y, entries):
        cue_pass = cue >= cue_threshold
        leakage_pass = leakage <= leakage_threshold
        axes[0].plot(cue, position, marker="o" if cue_pass else "X",
                     color=GREEN if cue_pass else RED, markersize=4.8,
                     linestyle="none")
        axes[1].plot(leakage, position, marker="o" if leakage_pass else "X",
                     color=GREEN if leakage_pass else RED, markersize=4.8,
                     linestyle="none")
    axes[0].axvline(cue_threshold, color=MID, linestyle=(0, (3, 2)), linewidth=1.0)
    axes[1].axvline(leakage_threshold, color=MID, linestyle=(0, (3, 2)),
                    linewidth=1.0)
    axes[0].set_yticks(y, [entry[0] for entry in entries])
    axes[0].set_xlim(0.55, 1.0)
    axes[1].set_xlim(0.28, 0.42)
    axes[0].set_xlabel("minimum cue accuracy")
    axes[1].set_xlabel("maximum late-context accuracy")
    axes[0].set_title(r"(a) Cue read ($\geq .75$)", loc="left",
                      fontweight="bold")
    axes[1].set_title(r"(b) Late shortcut ($\leq .383$)", loc="left",
                      fontweight="bold")
    for ax in axes:
        ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    legend = [
        Line2D([0], [0], marker="o", color=GREEN, linestyle="none",
               label="this gate passes"),
        Line2D([0], [0], marker="X", color=RED, linestyle="none",
               label="this gate fails"),
    ]
    axes[0].legend(handles=legend, loc="upper left", bbox_to_anchor=(0.0, 1.23),
                   ncol=2, frameon=False)
    fig.subplots_adjust(left=0.37, right=0.99, top=0.83, bottom=0.16,
                        wspace=0.22)
    save(fig, "fig_mem_appendix_shell_gates")


def figure_appendix_repair() -> None:
    path = ROOT / "outputs/paper_a_delayed_repair_residual_v2/summary.json"
    if not path.is_file():
        raise FileNotFoundError("aggregate the residual-repair study first")
    summary = json.loads(path.read_text())
    rows = []
    for task, task_label in (("transient-marker-recall", "Marker"),
                             ("drifting-color-recall", "Color")):
        for arm, arm_label in (("gru", "GRU"), ("ssm", "State-space")):
            record = summary["tasks"][task][arm]
            ci = record["paired_objective_off_minus_repair_mse"]
            rows.append((f"{task_label} / {arm_label}", ci,
                         record["cue_residual_repair_normalized_mse_to_zero"],
                         record["diagnostic_support"]))
    y = np.arange(len(rows))[::-1]
    fig = plt.figure(figsize=(5.45, 2.68))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0), wspace=0.30)
    ax = fig.add_subplot(grid[0, 0])
    bx = fig.add_subplot(grid[0, 1], sharey=ax)
    for position, (label, ci, normalized, support) in zip(y, rows):
        stat = {"mean": ci["mean"], "ci95": [ci["ci_low"], ci["ci_high"]]}
        color = GREEN if support else MID
        _forest_point(ax, float(position), stat, color, "o", filled=True)
        bx.plot(normalized, position, marker="o" if support else "X",
                color=GREEN if support else RED, markersize=5.0,
                linestyle="none")
    ax.axvline(0, color=MID, linewidth=1.0, linestyle=(0, (3, 2)))
    bx.axvline(1.0, color=MID, linewidth=1.0, linestyle=(0, (3, 2)))
    ax.set_yticks(y, [row[0] for row in rows])
    high = max(row[1]["ci_high"] for row in rows)
    ax.set_xlim(-0.18, high * 1.08)
    bx.set_xlim(0.996, 1.042)
    ax.set_xlabel("MSE reduction vs twin")
    bx.set_xlabel("held-out NMSE / mean")
    ax.grid(axis="x", color=LIGHT, linewidth=0.7)
    bx.grid(axis="x", color=LIGHT, linewidth=0.7)
    ax.set_title("(a) Auxiliary target fits", loc="left", fontweight="bold")
    bx.set_title("(b) Mean baseline fails", loc="left", fontweight="bold")
    bx.tick_params(axis="y", labelleft=False, left=False)
    bx.text(1.0, len(rows) - 0.48, "required < 1", fontsize=6.8,
            ha="center", va="bottom", color=MID,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    fig.subplots_adjust(left=0.24, right=0.985, top=0.88, bottom=0.23)
    save(fig, "fig_mem_appendix_repair")


def _generate_all_figures() -> None:
    figure_architecture()
    figure_tasks()
    figure_matched_lewm()
    figure_dinowm_carrier()
    figure_pointmaze_retention_use()
    figure_retention()
    figure_diagnosis()
    figure_appendix_endpoint()
    if (ROOT / "outputs/paper_a_evidence_age_v1/strict/summary.json").is_file():
        figure_cue_age()
    if (ROOT / "outputs/dinowm_native_pusht_audit_v2r2/formal/summary.json").is_file():
        figure_appendix_dinowm()
    figure_appendix_rollout()
    figure_appendix_shell_gates()
    figure_appendix_repair()


def main() -> None:
    # Refuse every figure write until all three grids, official verifiers, and
    # both independent post-lock audits have bound the current summaries.
    _require_final_audit_receipts()
    global FIG
    final_directory = FIG
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=".figures-staging.", dir=final_directory.parent))
    try:
        FIG = staging
        _generate_all_figures()
        _write_figure_manifest(final_directory=final_directory)
        _validate_staged_figure_manifest(staging, final_directory)
        _publish_staged_figures(staging, final_directory)
    finally:
        FIG = final_directory
        shutil.rmtree(staging, ignore_errors=True)
    print(f"[paper-a-strengthened-plots] wrote figures to {final_directory}")


if __name__ == "__main__":
    main()
