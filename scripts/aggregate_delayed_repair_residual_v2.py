#!/usr/bin/env python3
"""Aggregate paired label-free cue-residual diagnostics across formal cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.delayed_repair_residual_v2_spec import (  # noqa: E402
    ARMS,
    CONDITIONS,
    DEFAULT_LOCK,
    DEFAULT_SPEC,
    SEEDS,
    TASKS,
    load_locked_spec,
    lock_receipt,
    repair_directory,
    require_development_health,
    resolve_path,
    sha256_file,
    stable_json,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _seed(base: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def crossed_bootstrap_ci(matrix: np.ndarray, *, draws: int, seed: int,
                         confidence: float = 0.95) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 \
            or not np.isfinite(values).all():
        raise ValueError("crossed bootstrap requires a finite 2-D matrix")
    rng = np.random.default_rng(seed)
    rows, columns = values.shape
    row_weights = rng.multinomial(
        rows, np.full(rows, 1.0 / rows), size=draws)
    column_weights = rng.multinomial(
        columns, np.full(columns, 1.0 / columns), size=draws)
    samples = np.einsum(
        "dr,rc,dc->d", row_weights, values, column_weights,
        optimize=True) / float(rows * columns)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(samples, tail)),
        "ci_high": float(np.quantile(samples, 1.0 - tail)),
        "confidence": confidence,
        "draws": int(draws),
    }


def _load_cell(spec: dict[str, Any], task: str, arm: str,
               seed: int, condition: str) -> dict[str, Any]:
    directory = repair_directory(spec, task, arm, seed, condition)
    paths = {name: directory / filename for name, filename in (
        ("manifest", "manifest.json"),
        ("metrics", "metrics.json"),
        ("checkpoint", "repair.pt"),
        ("validation_export", "validation_export.npz"),
        ("history", "history.csv"),
    )}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"incomplete V2 repair cell {directory}")
    manifest = json.loads(paths["manifest"].read_text())
    metrics = json.loads(paths["metrics"].read_text())
    expected_identity = {
        "study": spec["study"], "task": task, "arm": arm,
        "condition": condition, "checkpoint_seed": seed,
        "formal_lock": lock_receipt(spec),
    }
    if any(manifest.get(key) != value
           for key, value in expected_identity.items()):
        raise ValueError(f"manifest identity mismatch for {directory}")
    for name in ("metrics", "checkpoint", "validation_export", "history"):
        record = manifest.get("artifacts", {}).get(name, {})
        if record.get("path") != paths[name].name \
                or record.get("sha256") != sha256_file(paths[name]):
            raise ValueError(f"artifact hash mismatch for {directory}/{name}")
    expected_metrics = {
        **expected_identity,
        "schema": "paper_a_delayed_repair_residual_metrics_v2",
        "scientific_role": "post_v1_diagnostic_repair",
        "preregistered_primary_result": False,
        "downstream_label_use_claim": False,
        "label_arrays_loaded": False,
        "label_values_consumed": False,
        "decision_frame_used_for_optimization": False,
        "validation_used_for_optimization": False,
        "development_statistics_used_for_formal_normalization": False,
        "repair_head": "Linear(192,192)",
        "repair_read": spec["cue_residual_target"]["read"],
    }
    if any(metrics.get(key) != value
           for key, value in expected_metrics.items()):
        raise ValueError(f"metrics identity/leakage mismatch for {directory}")
    if metrics.get("cue_residual_weight") != spec["formal_repair"][
            "cue_residual_weight"][condition]:
        raise ValueError(f"repair weight mismatch for {directory}")
    causal = metrics.get("causal_decision_intervention", {})
    if causal.get("exact_invariance") is not True \
            or causal.get("maximum_absolute_prior_difference") != 0.0:
        raise ValueError(f"causal read failed for {directory}")
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint.get("metrics") != metrics:
        raise ValueError(f"checkpoint/metrics mismatch for {directory}")
    with np.load(paths["validation_export"], allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    required = {
        "per_episode_mse", "per_episode_zero_mse",
        "prediction", "normalized_target",
    }
    if set(arrays) != required \
            or arrays["prediction"].shape != (240, 192) \
            or arrays["normalized_target"].shape != (240, 192) \
            or arrays["per_episode_mse"].shape != (240,) \
            or arrays["per_episode_zero_mse"].shape != (240,):
        raise ValueError(f"validation export shape mismatch for {directory}")
    expected_error = np.mean(np.square(
        arrays["prediction"] - arrays["normalized_target"]), axis=1)
    expected_zero = np.mean(np.square(arrays["normalized_target"]), axis=1)
    if not np.allclose(arrays["per_episode_mse"], expected_error,
                       rtol=2e-6, atol=2e-6) \
            or not np.allclose(arrays["per_episode_zero_mse"], expected_zero,
                               rtol=2e-6, atol=2e-6) \
            or not np.isclose(metrics["validation_cue_residual_mse"],
                              expected_error.mean(), rtol=2e-6, atol=2e-6) \
            or not np.isclose(metrics["validation_zero_predictor_mse"],
                              expected_zero.mean(), rtol=2e-6, atol=2e-6):
        raise ValueError(f"validation metrics mismatch for {directory}")
    return {"metrics": metrics, "arrays": arrays}


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing V2 aggregation without explicit --execute")
    spec = load_locked_spec(args.spec, args.lock)
    output = resolve_path(spec["output"]["summary"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    development = {
        task: require_development_health(spec, task) for task in TASKS}
    endpoints = spec["formal_endpoints"]
    draws = int(endpoints["bootstrap_draws"])
    base_seed = int(endpoints["bootstrap_seed"])
    results = {}
    for task in TASKS:
        task_result = {}
        for arm in ARMS:
            cells = {
                condition: [
                    _load_cell(spec, task, arm, seed, condition)
                    for seed in SEEDS]
                for condition in CONDITIONS
            }
            for seed_index, seed in enumerate(SEEDS):
                off = cells["objective-off"][seed_index]
                repair = cells["cue-residual-repair"][seed_index]
                for key in (
                        "parent_carrier_state_sha256",
                        "repair_head_initial_state_sha256",
                        "training_plan_sha256", "torch_seed",
                        "target_standardizer_sha256", "source_caches",
                        "device"):
                    if off["metrics"].get(key) != repair["metrics"].get(key):
                        raise ValueError(
                            f"twin pairing mismatch {task}/{arm}/seed-{seed}/{key}")
                if not np.array_equal(
                        off["arrays"]["per_episode_zero_mse"],
                        repair["arrays"]["per_episode_zero_mse"]):
                    raise ValueError(
                        f"twin validation targets differ {task}/{arm}/seed-{seed}")
            off_error = np.stack([
                cell["arrays"]["per_episode_mse"]
                for cell in cells["objective-off"]])
            repair_error = np.stack([
                cell["arrays"]["per_episode_mse"]
                for cell in cells["cue-residual-repair"]])
            zero_error = np.stack([
                cell["arrays"]["per_episode_zero_mse"]
                for cell in cells["cue-residual-repair"]])
            improvement = off_error - repair_error
            ci = crossed_bootstrap_ci(
                improvement, draws=draws,
                seed=_seed(base_seed, f"{task}/{arm}/residual-mse"))
            repaired_nmse = float(repair_error.mean() / zero_error.mean())
            next_mse = {
                condition: [float(cell["metrics"][
                    "validation_next_latent_mse"]) for cell in values]
                for condition, values in cells.items()
            }
            task_result[arm] = {
                "checkpoint_seeds": list(SEEDS),
                "objective_off_mse": float(off_error.mean()),
                "cue_residual_repair_mse": float(repair_error.mean()),
                "zero_predictor_mse": float(zero_error.mean()),
                "cue_residual_repair_normalized_mse_to_zero": repaired_nmse,
                "cue_residual_repair_r2_vs_training_mean": 1.0 - repaired_nmse,
                "paired_objective_off_minus_repair_mse": ci,
                "next_latent_mse_by_condition": next_mse,
                "mean_next_latent_mse_difference_repair_minus_off": float(
                    np.mean(next_mse["cue-residual-repair"])
                    - np.mean(next_mse["objective-off"])),
                "diagnostic_support": bool(
                    ci["ci_low"] > 0.0 and repaired_nmse < 1.0),
            }
        results[task] = task_result
    summary = {
        "schema": "paper_a_delayed_repair_residual_summary_v2",
        "study": spec["study"],
        "scientific_role": spec["scientific_role"]["classification"],
        "preregistered_primary_result": False,
        "downstream_label_use_claim": False,
        "formal_lock": lock_receipt(spec),
        "development_health": development,
        "tasks": results,
        "diagnostic_support_rule": endpoints["diagnostic_support_rule"],
        "all_task_arm_diagnostics_support_residual_repair": all(
            result["diagnostic_support"]
            for task_result in results.values()
            for result in task_result.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(stable_json(summary))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[delayed-residual-v2-aggregate] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
