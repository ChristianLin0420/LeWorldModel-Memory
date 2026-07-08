#!/usr/bin/env python3
"""Fail-closed aggregation for the locked Paper-A Reacher robustness stage.

The stage has two distinct inferential units and this script keeps them
separate:

* the seed extension combines optimizer seeds 0--4 from the immutable parent
  study with new seeds 5--9 on the original validation bank; and
* fresh validation evaluates the original seed-0--4 checkpoints on two new
  banks per task without retraining.

No confidence interval treats episodes, duplicated no-carrier checkpoints, or
the two validation banks as independent optimizer seeds.  Task and bank means
are formed inside a matched seed before paired seed bootstrapping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_paper_a_expansion import carrier_state_digest  # noqa: E402
from scripts.paper_a_robustness_spec import (  # noqa: E402
    DEFAULT_SPEC,
    load_locked_spec,
    resolve_spec_path,
    sha256_file,
)


BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_706
TASK_DISPLAY = {
    "t1": "Transient-marker recall",
    "t3": "Drifting-color recall",
}
ARM_DISPLAY = {
    "none": "No carrier",
    "gru": "Action-conditioned GRU",
    "lstm": "Action-conditioned LSTM",
    "ssm": "Diagonal SSM",
    "fixed_trust": "Fixed-trust memory",
}
METRICS = {
    "accuracy": ("probe", "mean"),
    "trajectory_accuracy": ("trajectory_probe", "mean"),
}


class RobustnessAggregationError(RuntimeError):
    """Raised when any expected artifact or provenance check fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RobustnessAggregationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RobustnessAggregationError(f"cannot read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _stable_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _seeded_rng(label: str) -> np.random.Generator:
    salt = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    return np.random.default_rng((BOOTSTRAP_SEED + salt) % (2**63 - 1))


def _summary(values: Iterable[float], label: str, *, effective_n: int | None = None
             ) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    _require(array.ndim == 1 and len(array) > 0 and np.isfinite(array).all(),
             f"{label}: summary values must be finite and nonempty")
    rng = _seeded_rng(label)
    draw = array[rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))]
    interval = np.quantile(draw.mean(axis=1), [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci95": [float(interval[0]), float(interval[1])],
        "n_seed_records": int(len(array)),
        "effective_independent_seeds": int(
            len(array) if effective_n is None else effective_n),
        "values": [float(value) for value in array],
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "unit": "matched optimizer seed",
            "interval": "percentile",
            "level": 0.95,
        },
    }


def _paired(left: Iterable[float], right: Iterable[float], label: str,
            left_name: str, right_name: str) -> dict[str, Any]:
    left_array = np.asarray(list(left), dtype=np.float64)
    right_array = np.asarray(list(right), dtype=np.float64)
    _require(left_array.shape == right_array.shape and left_array.ndim == 1,
             f"{label}: paired inputs are not aligned")
    result = _summary(left_array - right_array, label)
    result.update({
        "contrast": f"{left_name} minus {right_name}",
        "left": left_name,
        "right": right_name,
        "paired": True,
    })
    return result


def _validate_extension_cell(spec: Mapping[str, Any], task: str, arm: str,
                             seed: int) -> dict[str, Any]:
    root = resolve_spec_path(spec, spec["output"]["carrier_seed_extension"])
    directory = root / task / arm / f"s{seed}"
    metrics_path = directory / "metrics.json"
    checkpoint_path = directory / "carrier.pt"
    _require(metrics_path.is_file(), f"missing seed-extension metric {metrics_path}")
    _require(checkpoint_path.is_file(),
             f"missing seed-extension checkpoint {checkpoint_path}")
    metrics = _load_json(metrics_path)
    expected = {
        "schema_version": 1,
        "study": "official-lewm-frozen-carrier-seed-extension-v1",
        "task": task,
        "arm": arm,
        "seed": seed,
        "epochs": int(spec["carrier_seed_extension"]["epochs"]),
        "batch_size": int(spec["carrier_seed_extension"]["batch_size"]),
        "official_host": "quentinll/lewm-reacher",
        "host_trainable_parameters": 0,
        "frozen_host_unchanged": True,
        "strengthening_spec": spec["_spec_record"],
    }
    for key, value in expected.items():
        _require(metrics.get(key) == value,
                 f"{metrics_path}: {key}={metrics.get(key)!r}, expected {value!r}")
    _require(metrics.get("official_host_state_sha256_before") ==
             metrics.get("official_host_state_sha256_after"),
             f"{metrics_path}: official host state changed")
    _require(float(metrics.get("learning_rate", -1)) ==
             float(spec["carrier_seed_extension"]["learning_rate"]),
             f"{metrics_path}: learning rate changed")
    for split, record_name in (("train", "train_caches"),
                               ("validation", "validation_caches")):
        record = metrics.get("source_caches", {}).get(split, {})
        locked = spec["parent"][record_name][task]
        _require(record.get("sha256") == locked["sha256"],
                 f"{metrics_path}: {split} cache hash differs from lock")
        _require(Path(record.get("path", "")).resolve() ==
                 resolve_spec_path(spec, locked["path"]),
                 f"{metrics_path}: {split} cache path differs from lock")
    for key, _ in METRICS.values():
        probe = metrics.get(key, {})
        _require(probe.get("metric") == "accuracy" and
                 np.isfinite(float(probe.get("mean", np.nan))),
                 f"{metrics_path}: invalid {key}")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RobustnessAggregationError(
            f"cannot safely load {checkpoint_path}: {error}") from error
    _require(isinstance(checkpoint, dict),
             f"{checkpoint_path}: checkpoint root must be a mapping")
    _require(checkpoint.get("metrics") == metrics,
             f"{checkpoint_path}: embedded metrics differ from metrics.json")
    state = checkpoint.get("carrier_state_dict")
    _require(isinstance(state, Mapping),
             f"{checkpoint_path}: missing carrier_state_dict")
    _require(all(isinstance(value, torch.Tensor) for value in state.values()),
             f"{checkpoint_path}: carrier state contains non-tensors")
    return {
        "task": task,
        "arm": arm,
        "seed": seed,
        "accuracy": float(metrics["probe"]["mean"]),
        "trajectory_accuracy": float(metrics["trajectory_probe"]["mean"]),
        "validation_next_latent_mse": float(metrics["val_next_latent_mse"]),
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "metrics_sha256": sha256_file(metrics_path),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "carrier_state_sha256": carrier_state_digest(state),
    }


def _validate_fresh_cell(spec: Mapping[str, Any], task: str, bank: str,
                         arm: str, seed: int) -> dict[str, Any]:
    root = resolve_spec_path(
        spec, spec["output"]["fresh_validation_evaluation"])
    path = root / task / bank / arm / f"s{seed}" / "metrics.json"
    _require(path.is_file(), f"missing fresh-validation metric {path}")
    metrics = _load_json(path)
    expected = {
        "schema_version": 1,
        "study": "paper-a-reacher-fresh-validation-v1",
        "task": task,
        "bank_id": bank,
        "arm": arm,
        "seed": seed,
        "training_performed": False,
        "host_instantiated": False,
        "carrier_state_unchanged": True,
        "parent_artifacts_modified": False,
        "strengthening_spec": spec["_spec_record"],
    }
    for key, value in expected.items():
        _require(metrics.get(key) == value,
                 f"{path}: {key}={metrics.get(key)!r}, expected {value!r}")
    availability = metrics.get("fresh_validation_cache", {}).get(
        "availability", {})
    _require(availability.get("passed") is True and
             float(availability.get("value", -1)) >=
             float(spec["fresh_validation"]["categorical_availability_min"]),
             f"{path}: fresh bank failed availability")
    cache_record = metrics["fresh_validation_cache"]
    cache_path = ROOT / cache_record["path"]
    _require(cache_path.is_file() and
             sha256_file(cache_path) == cache_record["sha256"],
             f"{path}: fresh cache hash mismatch")
    source = metrics.get("source_checkpoint", {})
    source_checkpoint = ROOT / source.get("checkpoint_path", "")
    source_metrics = ROOT / source.get("metrics_path", "")
    _require(source_checkpoint.is_file() and
             sha256_file(source_checkpoint) == source.get("checkpoint_sha256"),
             f"{path}: parent checkpoint hash mismatch")
    _require(source_metrics.is_file() and
             sha256_file(source_metrics) == source.get("metrics_sha256"),
             f"{path}: parent metrics hash mismatch")
    try:
        checkpoint = torch.load(
            source_checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RobustnessAggregationError(
            f"cannot safely load {source_checkpoint}: {error}") from error
    state = checkpoint.get("carrier_state_dict", {})
    _require(carrier_state_digest(state) == source.get("carrier_state_sha256"),
             f"{path}: parent carrier state digest mismatch")
    for key, _ in METRICS.values():
        probe = metrics.get(key, {})
        _require(probe.get("metric") == "accuracy" and
                 np.isfinite(float(probe.get("mean", np.nan))),
                 f"{path}: invalid {key}")
    return {
        "task": task,
        "bank": bank,
        "arm": arm,
        "seed": seed,
        "accuracy": float(metrics["probe"]["mean"]),
        "trajectory_accuracy": float(metrics["trajectory_probe"]["mean"]),
        "availability": float(availability["value"]),
        "metrics_path": str(path.relative_to(ROOT)),
        "metrics_sha256": sha256_file(path),
    }


def _parent_values(parent: Mapping[str, Any], task: str, arm: str,
                   metric: str) -> list[float]:
    values = parent["frozen_carrier_swap"]["tasks"][task]["arms"][arm][metric][
        "values"]
    _require(len(values) == 5, f"parent {task}/{arm}/{metric} must have 5 seeds")
    return [float(value) for value in values]


def _aggregate_seed_extension(spec: Mapping[str, Any], parent: Mapping[str, Any],
                              rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = list(spec["carrier_seed_extension"]["arms"])
    tasks = list(spec["tasks"])
    output: dict[str, Any] = {"tasks": {}, "equal_task_arms": {},
                              "paired_contrasts": {}}
    aligned: dict[str, dict[str, list[float]]] = {
        metric: {} for metric in METRICS
    }
    for task in tasks:
        task_record = {"display_name": TASK_DISPLAY[task], "arms": {}}
        for arm in arms:
            arm_record: dict[str, Any] = {"display_name": ARM_DISPLAY[arm]}
            for metric in METRICS:
                parent_values = _parent_values(parent, task, arm, metric)
                extension_values = [
                    next(row[metric] for row in rows
                         if row["task"] == task and row["arm"] == arm
                         and row["seed"] == seed)
                    for seed in spec["carrier_seed_extension"]["seeds"]
                ]
                values = parent_values + extension_values
                arm_record[metric] = _summary(
                    values, f"seed-extension/{task}/{arm}/{metric}")
                arm_record[metric]["seeds"] = list(range(10))
            task_record["arms"][arm] = arm_record
        output["tasks"][task] = task_record
    for metric in METRICS:
        for arm in arms:
            per_seed = []
            for seed in range(10):
                per_seed.append(float(np.mean([
                    output["tasks"][task]["arms"][arm][metric]["values"][seed]
                    for task in tasks
                ])))
            aligned[metric][arm] = per_seed
            record = _summary(
                per_seed, f"seed-extension/equal-task/{arm}/{metric}")
            record.update({
                "display_name": ARM_DISPLAY[arm],
                "tasks": [TASK_DISPLAY[task] for task in tasks],
                "task_weighting": "equal within matched seed",
                "seeds": list(range(10)),
            })
            output["equal_task_arms"].setdefault(arm, {})[metric] = record
        for baseline in ("gru", "ssm"):
            output["paired_contrasts"].setdefault(baseline, {})[metric] = _paired(
                aligned[metric]["fixed_trust"], aligned[metric][baseline],
                f"seed-extension/fixed-minus-{baseline}/{metric}",
                ARM_DISPLAY["fixed_trust"], ARM_DISPLAY[baseline])
    output["design"] = {
        "new_training_seeds": list(spec["carrier_seed_extension"]["seeds"]),
        "combined_seeds": list(range(10)),
        "validation_bank": "original locked validation bank",
        "pooling": "equal task mean within matched seed",
    }
    return output


def _aggregate_fresh(spec: Mapping[str, Any], rows: list[dict[str, Any]]
                     ) -> dict[str, Any]:
    tasks = list(spec["tasks"])
    banks = [bank["id"] for bank in spec["fresh_validation"]["banks"]]
    arms = list(spec["fresh_validation"]["checkpoint_arms"])
    output: dict[str, Any] = {"task_banks": {}, "equal_task_bank_arms": {},
                              "paired_contrasts": {}}
    aligned: dict[str, dict[str, list[float]]] = {
        metric: {} for metric in METRICS
    }
    for task in tasks:
        output["task_banks"][task] = {
            "display_name": TASK_DISPLAY[task], "banks": {}}
        for bank in banks:
            bank_record: dict[str, Any] = {"arms": {}}
            bank_rows = [row for row in rows
                         if row["task"] == task and row["bank"] == bank]
            availability = sorted(set(row["availability"] for row in bank_rows))
            _require(len(availability) == 1,
                     f"fresh {task}/{bank}: availability is not invariant")
            bank_record["availability"] = availability[0]
            for arm in arms:
                arm_record: dict[str, Any] = {"display_name": ARM_DISPLAY[arm]}
                effective_n = 1 if arm == "none" else None
                for metric in METRICS:
                    values = [
                        next(row[metric] for row in bank_rows
                             if row["arm"] == arm and row["seed"] == seed)
                        for seed in spec["fresh_validation"]["checkpoint_seeds"]
                    ]
                    arm_record[metric] = _summary(
                        values, f"fresh/{task}/{bank}/{arm}/{metric}",
                        effective_n=effective_n)
                bank_record["arms"][arm] = arm_record
            output["task_banks"][task]["banks"][bank] = bank_record
    for metric in METRICS:
        for arm in arms:
            per_seed = []
            for seed in spec["fresh_validation"]["checkpoint_seeds"]:
                per_seed.append(float(np.mean([
                    next(row[metric] for row in rows
                         if row["task"] == task and row["bank"] == bank
                         and row["arm"] == arm and row["seed"] == seed)
                    for task in tasks for bank in banks
                ])))
            aligned[metric][arm] = per_seed
            record = _summary(
                per_seed, f"fresh/equal-task-bank/{arm}/{metric}",
                effective_n=1 if arm == "none" else None)
            record.update({
                "display_name": ARM_DISPLAY[arm],
                "tasks": [TASK_DISPLAY[task] for task in tasks],
                "banks": banks,
                "pooling": "equal task-bank mean within matched seed",
                "training_performed": False,
            })
            output["equal_task_bank_arms"].setdefault(arm, {})[metric] = record
        for baseline in ("gru", "lstm", "ssm"):
            output["paired_contrasts"].setdefault(baseline, {})[metric] = _paired(
                aligned[metric]["fixed_trust"], aligned[metric][baseline],
                f"fresh/fixed-minus-{baseline}/{metric}",
                ARM_DISPLAY["fixed_trust"], ARM_DISPLAY[baseline])
    output["design"] = {
        "banks_per_task": len(banks),
        "checkpoint_seeds": list(spec["fresh_validation"]["checkpoint_seeds"]),
        "training_performed": False,
        "readout_fit_source": spec["fresh_validation"]["readout_fit_source"],
        "pooling": "equal task-bank mean within matched seed",
    }
    return output


def _markdown(summary: Mapping[str, Any]) -> str:
    fresh = summary["fresh_validation"]["equal_task_bank_arms"]
    extension = summary["seed_extension"]["equal_task_arms"]
    lines = [
        "# Paper A robustness summary",
        "",
        "All labels below are semantic publication names. Confidence intervals "
        "resample matched optimizer seeds after equal task/bank pooling.",
        "",
        "## Fresh validation banks (no retraining)",
        "",
        "| Carrier | Final accuracy | Trajectory accuracy | Independent seeds |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARM_DISPLAY:
        final = fresh[arm]["accuracy"]
        trajectory = fresh[arm]["trajectory_accuracy"]
        lines.append(
            f"| {ARM_DISPLAY[arm]} | {final['mean']:.3f} "
            f"[{final['ci95'][0]:.3f}, {final['ci95'][1]:.3f}] | "
            f"{trajectory['mean']:.3f} "
            f"[{trajectory['ci95'][0]:.3f}, {trajectory['ci95'][1]:.3f}] | "
            f"{final['effective_independent_seeds']} |")
    lines.extend([
        "",
        "## Original-bank seed extension",
        "",
        "| Carrier | Final accuracy (10 seeds) | Trajectory accuracy (10 seeds) |",
        "|---|---:|---:|",
    ])
    for arm in ("gru", "ssm", "fixed_trust"):
        final = extension[arm]["accuracy"]
        trajectory = extension[arm]["trajectory_accuracy"]
        lines.append(
            f"| {ARM_DISPLAY[arm]} | {final['mean']:.3f} "
            f"[{final['ci95'][0]:.3f}, {final['ci95'][1]:.3f}] | "
            f"{trajectory['mean']:.3f} "
            f"[{trajectory['ci95'][0]:.3f}, {trajectory['ci95'][1]:.3f}] |")
    lines.extend([
        "",
        "Chance accuracy is 0.25. The trajectory diagnostic is exploratory; "
        "the final pre-observation endpoint remains the registered primary read.",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec)
    output_root = (args.output_root.resolve() if args.output_root else
                   resolve_spec_path(spec, spec["output"]["root"]))
    locked_root = resolve_spec_path(spec, spec["output"]["root"])
    _require(output_root == locked_root,
             "robustness summary must be written to the locked output root")
    parent_path = resolve_spec_path(spec, spec["parent"]["summary"]["path"])
    parent = _load_json(parent_path)
    _require(parent.get("completion", {}).get("complete") is True,
             "parent expansion is not complete")
    _require(parent.get("validation", {}).get("fail_closed") is True,
             "parent expansion did not pass fail-closed validation")

    extension_rows = [
        _validate_extension_cell(spec, task, arm, int(seed))
        for task in spec["tasks"]
        for arm in spec["carrier_seed_extension"]["arms"]
        for seed in spec["carrier_seed_extension"]["seeds"]
    ]
    fresh_rows = [
        _validate_fresh_cell(spec, task, bank["id"], arm, int(seed))
        for task in spec["tasks"]
        for bank in spec["fresh_validation"]["banks"]
        for arm in spec["fresh_validation"]["checkpoint_arms"]
        for seed in spec["fresh_validation"]["checkpoint_seeds"]
    ]
    _require(len(extension_rows) == 30, "seed-extension grid is not 30 cells")
    _require(len(fresh_rows) == 100, "fresh-validation grid is not 100 cells")

    summary = {
        "schema_version": 1,
        "study": spec["study"],
        "complete": True,
        "semantic_task_names": TASK_DISPLAY,
        "chance_accuracy": 0.25,
        "analysis": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "primary_endpoint": "final pre-observation decision read",
            "trajectory_role": "exploratory secondary diagnostic",
        },
        "provenance": {
            "strengthening_spec": spec["_spec_record"],
            "parent_summary": spec["parent"]["summary"],
            "extension_metric_sha256": {
                row["metrics_path"]: row["metrics_sha256"]
                for row in extension_rows
            },
            "fresh_metric_sha256": {
                row["metrics_path"]: row["metrics_sha256"]
                for row in fresh_rows
            },
        },
        "availability": {
            TASK_DISPLAY[task]: {
                bank["id"]: next(
                    row["availability"] for row in fresh_rows
                    if row["task"] == task and row["bank"] == bank["id"])
                for bank in spec["fresh_validation"]["banks"]
            }
            for task in spec["tasks"]
        },
        "seed_extension": _aggregate_seed_extension(
            spec, parent, extension_rows),
        "fresh_validation": _aggregate_fresh(spec, fresh_rows),
        "validation": {
            "expected_extension_cells": 30,
            "completed_extension_cells": len(extension_rows),
            "expected_fresh_cells": 100,
            "completed_fresh_cells": len(fresh_rows),
            "all_fresh_availability_gates_passed": all(
                row["availability"] >=
                float(spec["fresh_validation"]["categorical_availability_min"])
                for row in fresh_rows),
            "parent_artifacts_modified": False,
            "fail_closed": True,
        },
    }
    summary_path = output_root / "summary.json"
    markdown_path = output_root / "summary.md"
    _atomic_text(summary_path, _stable_json(summary))
    _atomic_text(markdown_path, _markdown(summary))
    print(f"[robust-aggregate] wrote {summary_path} and {markdown_path}")


if __name__ == "__main__":
    main()
