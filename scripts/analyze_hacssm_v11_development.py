#!/usr/bin/env python3
"""Integrity-check and summarize an excluded KDIO-v11 development factorial.

This analyzer deliberately does not write an official V11 manifest or decision.  It reads the
one-seed adaptive screen receipts, verifies the local W&B/rollout contract, and emits paired
descriptive statistics suitable for the excluded-screen ledger in LEARNABLE_MEMORY.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TASKS = (
    "dmc:cartpole.swingup",
    "dmc:fish.swim",
    "dmc:pendulum.swingup",
    "dmc:walker.walk",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label(metrics: dict[str, Any]) -> str:
    design = str(metrics["design"])
    ranking = str(metrics.get(
        "kdio_action_rank_development_mode",
        metrics.get("development_action_ranking", "legacy"),
    ))
    if design == "kdiov11":
        return f"full::{ranking}"
    return f"{design}::{ranking}"


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def _metric(row: dict[str, Any], *keys: str) -> float:
    """Read the first present metric, permitting explicitly documented legacy aliases."""
    for key in keys:
        if key in row:
            return float(row[key])
    raise KeyError(f"none of the metric aliases are present: {keys}")


def _mean_metric(rows: list[dict[str, Any]], *keys: str) -> float:
    return sum(_metric(row, *keys) for row in rows) / len(rows)


def _paired_reduction(candidate: list[dict[str, Any]], reference: list[dict[str, Any]],
                      key: str) -> dict[str, Any]:
    left = {str(row["env"]): float(row[key]) for row in candidate}
    right = {str(row["env"]): float(row[key]) for row in reference}
    if set(left) != set(right):
        raise RuntimeError(f"paired task mismatch for {key}: {set(left)} != {set(right)}")
    reductions = {task: (right[task] - left[task]) / right[task] for task in sorted(left)}
    return {
        "aggregate_relative_reduction": (
            sum(right.values()) - sum(left.values())) / sum(right.values()),
        "equal_task_mean_relative_reduction": sum(reductions.values()) / len(reductions),
        "wins": sum(value > 0.0 for value in reductions.values()),
        "tasks": reductions,
    }


def _linear_quantile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile requires non-empty values and probability in [0,1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def analyze(root: Path, expected_cells: int | None) -> dict[str, Any]:
    metric_paths = sorted(root.glob("*/metrics.json"))
    if expected_cells is not None and len(metric_paths) != expected_cells:
        raise RuntimeError(
            f"expected {expected_cells} metrics in {root}, found {len(metric_paths)}")
    if not metric_paths:
        raise RuntimeError(f"no metrics found under {root}")

    rows: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    for metrics_path in metric_paths:
        run_dir = metrics_path.parent
        run_path = run_dir / "wandb_run.json"
        rollout_path = run_dir / "eval_rollout.npz"
        model_path = run_dir / "model.pt"
        for required in (run_path, rollout_path, model_path):
            if not required.is_file():
                raise RuntimeError(f"missing required receipt {required}")
        metrics = json.loads(metrics_path.read_text())
        wandb = json.loads(run_path.read_text())
        if wandb.get("mode") != "online" or wandb.get("state") != "finished":
            raise RuntimeError(f"invalid W&B state in {run_path}: {wandb}")
        actual_rollout_hash = _sha256(rollout_path)
        if actual_rollout_hash != wandb.get("eval_rollout_sha256"):
            raise RuntimeError(f"rollout hash mismatch in {run_dir}")
        row = dict(metrics)
        row["wandb_run_id"] = str(wandb["run_id"])
        row["wandb_url"] = str(wandb["url"])
        row["rollout_sha256"] = actual_rollout_hash
        row["label"] = _label(metrics)
        rows.append(row)
        artifact_hashes[run_dir.name] = actual_rollout_hash

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label"])].append(row)
    for label, group in grouped.items():
        tasks = tuple(sorted(str(row["env"]) for row in group))
        if tasks != tuple(sorted(TASKS)):
            raise RuntimeError(f"{label} task coverage {tasks} != {tuple(sorted(TASKS))}")

    summaries: dict[str, Any] = {}
    for label, group in sorted(grouped.items()):
        summaries[label] = {
            "heldout_prior_state_nmse": _mean(group, "heldout_prior_state_nmse"),
            "clean_prior_state_nmse": _mean(group, "clean_prior_state_nmse"),
            "initial_encoder_integrator_nmse": _mean(
                group, "initial_encoder_integrator_probe_nmse"),
            "memory_action_scale": _mean(group, "memory_action_scale"),
            "action_rank_pair_accuracy": _mean(group, "kdio_action_swap_pair_accuracy"),
            "action_rank_relative_advantage": _mean_metric(
                group, "kdio_action_rank_relative_advantage",
                "kdio_true_action_suffix_advantage"),
            "live_suffix_mse": _mean_metric(
                group, "kdio_live_suffix_mse", "kdio_true_action_suffix_mse"),
            "predictive_loss_late_relative_change": _mean(
                group, "predictive_loss_convergence_relative_change"),
            "tasks": {
                str(row["env"]): {
                    "heldout_prior_state_nmse": float(row["heldout_prior_state_nmse"]),
                    "clean_prior_state_nmse": float(row["clean_prior_state_nmse"]),
                    "initial_encoder_integrator_nmse": float(
                        row["initial_encoder_integrator_probe_nmse"]),
                    "memory_action_scale": float(row["memory_action_scale"]),
                    "action_rank_pair_accuracy": float(
                        row["kdio_action_swap_pair_accuracy"]),
                    "action_rank_relative_advantage": _metric(
                        row, "kdio_action_rank_relative_advantage",
                        "kdio_true_action_suffix_advantage"),
                    "predictive_loss_late_relative_change": float(
                        row["predictive_loss_convergence_relative_change"]),
                    "wandb_run_id": str(row["wandb_run_id"]),
                    "wandb_url": str(row["wandb_url"]),
                }
                for row in sorted(group, key=lambda item: str(item["env"]))
            },
        }

    comparisons: dict[str, Any] = {}
    default = (
        "full::relative_displacement_detached"
        if "full::relative_displacement_detached" in grouped else
        "full::legacy" if "full::legacy" in grouped else ""
    )
    if default in grouped:
        for label, group in sorted(grouped.items()):
            if label == default:
                continue
            comparisons[f"{default}_vs_{label}"] = _paired_reduction(
                grouped[default], group, "heldout_prior_state_nmse")
        integrator_rows = [
            dict(row, heldout_prior_state_nmse=row["initial_encoder_integrator_probe_nmse"])
            for row in grouped[default]
        ]
        comparisons[f"{default}_vs_initial_encoder_integrator"] = _paired_reduction(
            grouped[default], integrator_rows, "heldout_prior_state_nmse")

    late_absolute = [
        abs(float(row["predictive_loss_convergence_relative_change"]))
        for row in rows]
    best_label = min(
        summaries,
        key=lambda label: float(summaries[label]["heldout_prior_state_nmse"]))
    return {
        "schema_version": 1,
        "status": "EXCLUDED_ADAPTIVE_DEVELOPMENT_SCREEN",
        "root": str(root),
        "cells": len(rows),
        "labels": sorted(grouped),
        "summaries": summaries,
        "comparisons": comparisons,
        "best_screen_label": best_label,
        "best_screen_heldout_prior_state_nmse": float(
            summaries[best_label]["heldout_prior_state_nmse"]),
        "absolute_predictive_late_change": {
            "median": _linear_quantile(late_absolute, 0.5),
            "p95": _linear_quantile(late_absolute, 0.95),
            "maximum": max(late_absolute),
        },
        "rollout_sha256": artifact_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-cells", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root, args.expected_cells)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
