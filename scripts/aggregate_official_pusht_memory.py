#!/usr/bin/env python3
"""Fail-closed aggregation for the locked official-PushT memory audit.

The inferential unit is the matched carrier-training seed.  Task means and
contrasts are formed inside each seed before percentile bootstrapping; the
five duplicated no-carrier records are retained for alignment but count as
one effective independent model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import sha256_file  # noqa: E402
from lewm.official_tasks.pusht_pipeline import (  # noqa: E402
    pusht_admission_path,
    pusht_artifact_root,
    pusht_carrier_directory,
    pusht_task_manifest_path,
    pusht_task_spec,
)
from lewm.official_tasks.pusht_spec import (  # noqa: E402
    DEFAULT_PUSHT_LOCK,
    DEFAULT_PUSHT_SPEC,
    load_locked_pusht_spec,
    pusht_lock_receipt,
)


BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_710
ARM_DISPLAY = {
    "none": "No persistent carrier",
    "gru": "Action-conditioned GRU",
    "lstm": "Action-conditioned LSTM",
    "ssm": "Diagonal state-space carrier",
    "fixed_trust": "Fixed-trust predict--correct carrier",
}


class PushTAggregationError(RuntimeError):
    """Raised when a formal artifact or provenance check fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PushTAggregationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PushTAggregationError(f"cannot read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _stable_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _rng(label: str) -> np.random.Generator:
    salt = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    return np.random.default_rng((BOOTSTRAP_SEED + salt) % (2**63 - 1))


def _summary(values: Iterable[float], label: str, *, effective_n: int | None = None
             ) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    _require(array.ndim == 1 and len(array) > 0 and np.isfinite(array).all(),
             f"{label}: values must be finite and nonempty")
    draw = array[_rng(label).integers(
        0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))]
    interval = np.quantile(draw.mean(axis=1), (0.025, 0.975))
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci95": [float(interval[0]), float(interval[1])],
        "values": [float(value) for value in array],
        "seed_records": int(len(array)),
        "effective_independent_models": int(
            len(array) if effective_n is None else effective_n),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "matched carrier-training seed",
            "interval": "percentile",
            "level": 0.95,
        },
    }


def _paired(left: Iterable[float], right: Iterable[float], label: str,
            left_name: str, right_name: str) -> dict[str, Any]:
    left_array = np.asarray(list(left), dtype=np.float64)
    right_array = np.asarray(list(right), dtype=np.float64)
    _require(left_array.ndim == 1 and left_array.shape == right_array.shape,
             f"{label}: paired seed records are not aligned")
    result = _summary(left_array - right_array, label)
    result.update({
        "left": left_name,
        "right": right_name,
        "contrast": f"{left_name} minus {right_name}",
        "paired": True,
        "ci_excludes_zero": bool(
            result["ci95"][0] > 0 or result["ci95"][1] < 0),
    })
    return result


def _validate_admission(spec: Mapping[str, Any], task_key: str) -> dict[str, Any]:
    path = pusht_admission_path(spec, task_key)
    manifest_path = pusht_task_manifest_path(spec, task_key)
    admission = _load_json(path)
    manifest = _load_json(manifest_path)
    receipt = pusht_lock_receipt(spec)
    _require(admission.get("schema") == "official_pusht_frozen_admission_v1",
             f"{path}: unexpected admission schema")
    _require(admission.get("formal_lock") == receipt,
             f"{path}: formal lock differs")
    _require(admission.get("task_key") == task_key,
             f"{path}: task key differs")
    _require(admission.get("admitted") is True,
             f"{path}: task was not admitted")
    gates = admission.get("gates", {})
    _require(gates and all(gate.get("pass") is True for gate in gates.values()),
             f"{path}: not every admission gate passed")
    _require(manifest.get("formal_lock") == receipt,
             f"{manifest_path}: formal lock differs")
    manifest_admission = manifest.get("admission", {})
    _require(manifest_admission.get("admitted") is True and
             manifest_admission.get("sha256") == sha256_file(path),
             f"{manifest_path}: admission receipt differs")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "semantic_name": admission["semantic_name"],
        "classes": int(admission["classes"]),
        "chance": float(admission["chance"]),
        "gates": gates,
    }


def _validate_cell(spec: Mapping[str, Any], task_key: str,
                   arm: str, seed: int) -> dict[str, Any]:
    directory = pusht_carrier_directory(spec, task_key, arm, seed)
    manifest_path = directory / "manifest.json"
    metrics_path = directory / "metrics.json"
    checkpoint_path = directory / "carrier.pt"
    history_path = directory / "history.csv"
    for path in (manifest_path, metrics_path, checkpoint_path, history_path):
        _require(path.is_file(), f"missing formal PushT artifact {path}")
    manifest = _load_json(manifest_path)
    metrics = _load_json(metrics_path)
    receipt = pusht_lock_receipt(spec)
    expected = {
        "schema": "official_pusht_carrier_metrics_v1",
        "task_key": task_key,
        "semantic_name": pusht_task_spec(spec, task_key)["display_name"],
        "arm": arm,
        "seed": seed,
        "formal_lock": receipt,
        "frozen_host_unchanged": True,
    }
    for key, value in expected.items():
        _require(metrics.get(key) == value,
                 f"{metrics_path}: {key} differs from the lock")
    _require(metrics.get("frozen_host_sha256_before") ==
             metrics.get("frozen_host_sha256_after"),
             f"{metrics_path}: frozen host state changed")
    expected_epochs = 0 if arm == "none" else int(
        spec["carrier_training"]["epochs"])
    _require(metrics.get("epochs") == expected_epochs,
             f"{metrics_path}: epoch count differs")
    probe = metrics.get("primary_probe", {})
    accuracy = float(probe.get("balanced_accuracy", np.nan))
    _require(probe.get("metric") == "balanced_accuracy" and
             np.isfinite(accuracy) and 0 <= accuracy <= 1,
             f"{metrics_path}: invalid primary endpoint")
    endpoint = probe.get("endpoint", {})
    _require(endpoint.get("decision_index") == 19 and
             endpoint.get("decision_observation_excluded") is True and
             endpoint.get("raw_context_indices") == [16, 17, 18] and
             endpoint.get("carrier_prior_index") == 19,
             f"{metrics_path}: illegal endpoint")
    _require(manifest.get("schema") == "official_pusht_carrier_manifest_v1" and
             manifest.get("task_key") == task_key and
             manifest.get("arm") == arm and manifest.get("seed") == seed and
             manifest.get("formal_lock") == receipt,
             f"{manifest_path}: manifest identity differs")
    artifacts = manifest.get("artifacts", {})
    for key, path in (("metrics", metrics_path),
                      ("checkpoint", checkpoint_path),
                      ("history", history_path)):
        record = artifacts.get(key, {})
        _require(record.get("path") == path.name and
                 record.get("sha256") == sha256_file(path),
                 f"{manifest_path}: {key} hash differs")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PushTAggregationError(
            f"cannot safely load {checkpoint_path}: {error}") from error
    _require(isinstance(checkpoint, dict) and
             checkpoint.get("metrics") == metrics and
             isinstance(checkpoint.get("carrier_state_dict"), Mapping),
             f"{checkpoint_path}: embedded metrics/state differ")
    return {
        "task_key": task_key,
        "arm": arm,
        "seed": seed,
        "balanced_accuracy": accuracy,
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "metrics_sha256": sha256_file(metrics_path),
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def _aggregate(spec: Mapping[str, Any], rows: list[dict[str, Any]],
               admissions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    tasks = [task["key"] for task in spec["semantic_tasks"]]
    arms = list(spec["carrier_training"]["arms"])
    seeds = [int(seed) for seed in spec["carrier_training"]["seeds"]]
    task_results: dict[str, Any] = {}
    aligned: dict[str, dict[str, list[float]]] = {}
    for task_key in tasks:
        chance = float(admissions[task_key]["chance"])
        record: dict[str, Any] = {
            "semantic_name": admissions[task_key]["semantic_name"],
            "classes": admissions[task_key]["classes"],
            "chance": chance,
            "arms": {},
            "paired_vs_no_carrier": {},
        }
        aligned[task_key] = {}
        for arm in arms:
            values = [
                next(row["balanced_accuracy"] for row in rows
                     if row["task_key"] == task_key and
                     row["arm"] == arm and row["seed"] == seed)
                for seed in seeds
            ]
            aligned[task_key][arm] = values
            arm_summary = _summary(
                values, f"pusht/{task_key}/{arm}",
                effective_n=1 if arm == "none" else None)
            arm_summary.update({
                "display_name": ARM_DISPLAY[arm],
                "chance": chance,
                "chance_corrected_mean": float(
                    (arm_summary["mean"] - chance) / (1.0 - chance)),
                "ci_lower_above_chance": bool(
                    arm_summary["ci95"][0] > chance),
            })
            record["arms"][arm] = arm_summary
        for arm in arms:
            if arm == "none":
                continue
            record["paired_vs_no_carrier"][arm] = _paired(
                aligned[task_key][arm], aligned[task_key]["none"],
                f"pusht/{task_key}/{arm}-minus-none",
                ARM_DISPLAY[arm], ARM_DISPLAY["none"])
        task_results[task_key] = record

    equal_task: dict[str, Any] = {"arms": {}, "paired_vs_no_carrier": {}}
    equal_values: dict[str, list[float]] = {}
    equal_corrected: dict[str, list[float]] = {}
    for arm in arms:
        equal_values[arm] = [
            float(np.mean([aligned[task][arm][index] for task in tasks]))
            for index in range(len(seeds))
        ]
        equal_corrected[arm] = [
            float(np.mean([
                (aligned[task][arm][index] - admissions[task]["chance"]) /
                (1.0 - admissions[task]["chance"])
                for task in tasks
            ]))
            for index in range(len(seeds))
        ]
        equal_task["arms"][arm] = {
            "display_name": ARM_DISPLAY[arm],
            "raw_balanced_accuracy": _summary(
                equal_values[arm], f"pusht/equal-task/{arm}/raw",
                effective_n=1 if arm == "none" else None),
            "chance_corrected_accuracy": _summary(
                equal_corrected[arm],
                f"pusht/equal-task/{arm}/chance-corrected",
                effective_n=1 if arm == "none" else None),
        }
    for arm in arms:
        if arm == "none":
            continue
        equal_task["paired_vs_no_carrier"][arm] = {
            "raw_balanced_accuracy": _paired(
                equal_values[arm], equal_values["none"],
                f"pusht/equal-task/{arm}-minus-none/raw",
                ARM_DISPLAY[arm], ARM_DISPLAY["none"]),
            "chance_corrected_accuracy": _paired(
                equal_corrected[arm], equal_corrected["none"],
                f"pusht/equal-task/{arm}-minus-none/chance-corrected",
                ARM_DISPLAY[arm], ARM_DISPLAY["none"]),
        }
    equal_task["pooling"] = (
        "equal task mean formed within matched seed; chance-corrected score "
        "is (balanced_accuracy - chance) / (1 - chance)")
    return {"tasks": task_results, "equal_task": equal_task}


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Official PushT frozen-memory summary",
        "",
        "All confidence intervals resample matched carrier-training seeds. "
        "The repeated no-carrier row is deterministic and counts as one "
        "effective independent model.",
        "",
    ]
    for task in summary["results"]["tasks"].values():
        lines.extend([
            f"## {task['semantic_name']}",
            "",
            f"Chance balanced accuracy: {task['chance']:.3f}.",
            "",
            "| Carrier | Balanced accuracy [95% CI] | Delta vs no carrier [95% CI] |",
            "|---|---:|---:|",
        ])
        for arm, record in task["arms"].items():
            if arm == "none":
                delta = "--"
            else:
                contrast = task["paired_vs_no_carrier"][arm]
                delta = (f"{contrast['mean']:+.3f} "
                         f"[{contrast['ci95'][0]:+.3f}, "
                         f"{contrast['ci95'][1]:+.3f}]")
            lines.append(
                f"| {record['display_name']} | {record['mean']:.3f} "
                f"[{record['ci95'][0]:.3f}, {record['ci95'][1]:.3f}] | "
                f"{delta} |")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_PUSHT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_PUSHT_LOCK)
    parser.add_argument(
        "--execute", action="store_true",
        help="write summary.json and summary.md; otherwise validate and print")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_pusht_spec(args.spec, args.lock)
    tasks = [task["key"] for task in spec["semantic_tasks"]]
    arms = list(spec["carrier_training"]["arms"])
    seeds = [int(seed) for seed in spec["carrier_training"]["seeds"]]
    admissions = {
        task_key: _validate_admission(spec, task_key)
        for task_key in tasks
    }
    rows = [
        _validate_cell(spec, task_key, arm, seed)
        for task_key in tasks for arm in arms for seed in seeds
    ]
    expected = len(tasks) * len(arms) * len(seeds)
    _require(len(rows) == expected,
             f"formal PushT grid has {len(rows)} rows, expected {expected}")
    summary = {
        "schema_version": 1,
        "study": spec["study"],
        "complete": True,
        "formal_lock": pusht_lock_receipt(spec),
        "analysis": {
            "primary_endpoint": (
                "balanced accuracy from concat(raw latents 16:19, causal "
                "carrier prior at 19), excluding observation 19"),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "inference_unit": "matched carrier-training seed",
        },
        "admissions": admissions,
        "provenance": {
            "metric_sha256": {
                row["metrics_path"]: row["metrics_sha256"] for row in rows
            },
            "checkpoint_sha256": {
                row["checkpoint_path"]: row["checkpoint_sha256"] for row in rows
            },
        },
        "results": _aggregate(spec, rows, admissions),
        "validation": {
            "expected_cells": expected,
            "completed_cells": len(rows),
            "all_admission_gates_passed": True,
            "all_frozen_hosts_unchanged": True,
            "decision_observation_excluded": True,
            "fail_closed": True,
        },
    }
    root = pusht_artifact_root(spec)
    if not args.execute:
        print(_markdown(summary))
        return
    _atomic_text(root / "summary.json", _stable_json(summary))
    _atomic_text(root / "summary.md", _markdown(summary) + "\n")
    print(f"[pusht-aggregate] wrote {root / 'summary.json'} and "
          f"{root / 'summary.md'}")


if __name__ == "__main__":
    main()
