#!/usr/bin/env python3
"""Fail-closed aggregation for the admitted shell-game capacity V3 grid."""

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
from lewm.official_tasks.shell_game_pipeline_v3 import (  # noqa: E402
    admission_path_v3,
    cache_manifest_path_v3,
    carrier_directory_v3,
    lock_receipt_v3,
    require_all_selected_salience_v3,
    require_selected_salience_v3,
    stage_contract_v3,
)
from lewm.official_tasks.shell_game_spec_v3 import (  # noqa: E402
    DEFAULT_LOCK_V3,
    DEFAULT_SPEC_V3,
    load_locked_spec_v3,
    resolve_path_v3,
)


STAGES = ("single-item", "two-item", "four-item")
ARM_DISPLAY = {
    "none": "No persistent carrier",
    "gru": "Action-conditioned GRU",
    "lstm": "Action-conditioned LSTM",
    "ssm": "Diagonal state-space carrier",
    "fixed_trust": "Fixed-trust predict--correct carrier",
}
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_711


class ShellCapacityAggregationError(RuntimeError):
    """Raised when any formal V3 artifact fails validation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ShellCapacityAggregationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ShellCapacityAggregationError(f"cannot read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _rng(label: str) -> np.random.Generator:
    salt = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    return np.random.default_rng((BOOTSTRAP_SEED + salt) % (2**63 - 1))


def _summary(values: Iterable[float], label: str, *, effective_n: int | None = None
             ) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    _require(array.ndim == 1 and len(array) and np.isfinite(array).all(),
             f"{label}: summary values must be finite")
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
             f"{label}: paired records differ")
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


def _validate_admission(spec: Mapping[str, Any], stage: str) -> dict[str, Any]:
    selection = require_selected_salience_v3(spec, stage)
    path = admission_path_v3(spec, stage)
    manifest_path = cache_manifest_path_v3(spec, stage)
    admission = _load_json(path)
    manifest = _load_json(manifest_path)
    receipt = lock_receipt_v3(spec)
    _require(admission.get("admitted") is True and
             admission.get("formal_lock") == receipt and
             admission.get("development_selection") == selection,
             f"{path}: admission identity or gate differs")
    _require(admission.get("threshold_changed_from_v1_or_v2") is False,
             f"{path}: threshold changed")
    gates = admission.get("gates", {})
    _require(gates and all(gate.get("pass") is True for gate in gates.values()),
             f"{path}: not every formal gate passed")
    record = manifest.get("admission", {})
    _require(manifest.get("formal_lock") == receipt and
             manifest.get("development_selection") == selection and
             record.get("admitted") is True and
             record.get("sha256") == sha256_file(path),
             f"{manifest_path}: admission receipt differs")
    contract = stage_contract_v3(stage).stage
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "display_name": contract.display_name,
        "capacity": int(contract.capacity),
        "gates": gates,
        "development_selection": selection,
    }


def _validate_cell(spec: Mapping[str, Any], stage: str,
                   arm: str, seed: int) -> dict[str, Any]:
    directory = carrier_directory_v3(spec, stage, arm, seed)
    paths = {
        "manifest": directory / "manifest.json",
        "metrics": directory / "metrics.json",
        "checkpoint": directory / "carrier.pt",
        "validation_export": directory / "validation_export.npz",
        "history": directory / "history.csv",
    }
    for path in paths.values():
        _require(path.is_file(), f"missing formal V3 artifact {path}")
    manifest = _load_json(paths["manifest"])
    metrics = _load_json(paths["metrics"])
    receipt = lock_receipt_v3(spec)
    contract = stage_contract_v3(stage).stage
    expected = {
        "schema": "official_shell_game_carrier_metrics_v3",
        "study": spec["study"],
        "stage": stage,
        "display_name": contract.display_name,
        "capacity": int(contract.capacity),
        "arm": arm,
        "seed": seed,
        "formal_lock": receipt,
        "threshold_changed_from_v1_or_v2": False,
        "semantic_capacity_contract_changed_from_v1_or_v2": False,
        "carrier_definitions_changed_from_v1_or_v2": False,
        "frozen_host_unchanged": True,
    }
    for key, value in expected.items():
        _require(metrics.get(key) == value,
                 f"{paths['metrics']}: {key} differs")
    _require(metrics.get("official_host_state_sha256_before") ==
             metrics.get("official_host_state_sha256_after"),
             f"{paths['metrics']}: frozen host changed")
    expected_epochs = 0 if arm == "none" else int(
        spec["carrier_training"]["epochs"])
    _require(metrics.get("epochs") == expected_epochs,
             f"{paths['metrics']}: epoch count differs")
    probe = metrics.get("primary_probe", {})
    mean_item = float(probe.get("mean_per_item_balanced_accuracy", np.nan))
    exact_set = float(probe.get("exact_set_accuracy", np.nan))
    _require(probe.get("metric") == "mean_per_item_balanced_accuracy" and
             np.isfinite(mean_item) and np.isfinite(exact_set) and
             0 <= mean_item <= 1 and 0 <= exact_set <= 1,
             f"{paths['metrics']}: invalid primary probe")
    endpoint = probe.get("endpoint_contract", {})
    _require(endpoint.get("decision_observation_index") == 63 and
             endpoint.get("decision_observation_excluded") is True and
             endpoint.get("raw_context_indices") == [60, 61, 62] and
             endpoint.get("prior_index") == 63 and
             endpoint.get("temporal_aggregation") is False,
             f"{paths['metrics']}: endpoint differs")
    _require(len(probe.get("per_item", [])) == int(contract.capacity),
             f"{paths['metrics']}: per-item result count differs")
    _require(manifest.get("schema") ==
             "official_shell_game_carrier_manifest_v3" and
             manifest.get("stage") == stage and manifest.get("arm") == arm and
             manifest.get("seed") == seed and
             manifest.get("formal_lock") == receipt,
             f"{paths['manifest']}: manifest identity differs")
    artifacts = manifest.get("artifacts", {})
    for key in ("metrics", "checkpoint", "validation_export", "history"):
        record = artifacts.get(key, {})
        _require(record.get("path") == paths[key].name and
                 record.get("sha256") == sha256_file(paths[key]),
                 f"{paths['manifest']}: {key} receipt differs")
    try:
        checkpoint = torch.load(
            paths["checkpoint"], map_location="cpu", weights_only=True)
    except Exception as error:
        raise ShellCapacityAggregationError(
            f"cannot safely load {paths['checkpoint']}: {error}") from error
    _require(isinstance(checkpoint, dict) and
             checkpoint.get("metrics") == metrics and
             isinstance(checkpoint.get("carrier_state_dict"), Mapping),
             f"{paths['checkpoint']}: embedded metrics/state differ")
    return {
        "stage": stage,
        "capacity": int(contract.capacity),
        "display_name": contract.display_name,
        "arm": arm,
        "seed": seed,
        "mean_per_item_balanced_accuracy": mean_item,
        "minimum_per_item_balanced_accuracy": float(
            probe["minimum_per_item_balanced_accuracy"]),
        "exact_set_accuracy": exact_set,
        "per_item_chance": float(probe["per_item_chance"]),
        "exact_set_chance": float(probe["exact_set_chance"]),
        "metrics_path": str(paths["metrics"].relative_to(ROOT)),
        "metrics_sha256": sha256_file(paths["metrics"]),
        "checkpoint_path": str(paths["checkpoint"].relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
    }


def _aggregate(spec: Mapping[str, Any], rows: list[dict[str, Any]],
               admissions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    arms = list(spec["carrier_training"]["arms"])
    seeds = [int(seed) for seed in spec["carrier_training"]["seeds"]]
    metrics = ("mean_per_item_balanced_accuracy", "exact_set_accuracy")
    stage_results: dict[str, Any] = {}
    aligned: dict[str, dict[str, dict[str, list[float]]]] = {}
    for stage in STAGES:
        stage_record: dict[str, Any] = {
            "display_name": admissions[stage]["display_name"],
            "capacity": admissions[stage]["capacity"],
            "per_item_chance": 1.0 / 3.0,
            "exact_set_chance": float(3 ** -admissions[stage]["capacity"]),
            "arms": {},
            "paired_vs_no_carrier": {},
        }
        aligned[stage] = {}
        for arm in arms:
            aligned[stage][arm] = {}
            arm_record: dict[str, Any] = {"display_name": ARM_DISPLAY[arm]}
            for metric in metrics:
                values = [
                    next(row[metric] for row in rows
                         if row["stage"] == stage and row["arm"] == arm and
                         row["seed"] == seed)
                    for seed in seeds
                ]
                aligned[stage][arm][metric] = values
                arm_record[metric] = _summary(
                    values, f"shell-v3/{stage}/{arm}/{metric}",
                    effective_n=1 if arm == "none" else None)
            stage_record["arms"][arm] = arm_record
        for arm in arms:
            if arm == "none":
                continue
            stage_record["paired_vs_no_carrier"][arm] = {
                metric: _paired(
                    aligned[stage][arm][metric],
                    aligned[stage]["none"][metric],
                    f"shell-v3/{stage}/{arm}-minus-none/{metric}",
                    ARM_DISPLAY[arm], ARM_DISPLAY["none"])
                for metric in metrics
            }
        stage_results[stage] = stage_record

    capacity_effects: dict[str, Any] = {}
    for arm in arms:
        capacity_effects[arm] = {
            "display_name": ARM_DISPLAY[arm],
            "four_minus_one": {
                metric: _paired(
                    aligned["four-item"][arm][metric],
                    aligned["single-item"][arm][metric],
                    f"shell-v3/{arm}/four-minus-one/{metric}",
                    "Four-item shell-game recall",
                    "Single-item shell-game recall")
                for metric in metrics
            },
        }
    return {
        "stages": stage_results,
        "capacity_effects": capacity_effects,
        "design": {
            "capacities": [1, 2, 4],
            "matched_seeds": seeds,
            "primary_metric": "mean per-item balanced accuracy",
            "secondary_metric": "exact-set accuracy",
        },
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Official shell-game capacity V3 summary",
        "",
        "All three stages passed the unchanged formal admission gates. "
        "Intervals resample matched carrier-training seeds.",
        "",
    ]
    for stage in STAGES:
        result = summary["results"]["stages"][stage]
        lines.extend([
            f"## {result['display_name']}",
            "",
            "| Carrier | Per-item balanced accuracy [95% CI] | Exact-set accuracy [95% CI] | Per-item delta vs no carrier |",
            "|---|---:|---:|---:|",
        ])
        for arm, record in result["arms"].items():
            per_item = record["mean_per_item_balanced_accuracy"]
            exact = record["exact_set_accuracy"]
            if arm == "none":
                delta = "--"
            else:
                contrast = result["paired_vs_no_carrier"][arm][
                    "mean_per_item_balanced_accuracy"]
                delta = (f"{contrast['mean']:+.3f} "
                         f"[{contrast['ci95'][0]:+.3f}, "
                         f"{contrast['ci95'][1]:+.3f}]")
            lines.append(
                f"| {record['display_name']} | {per_item['mean']:.3f} "
                f"[{per_item['ci95'][0]:.3f}, {per_item['ci95'][1]:.3f}] | "
                f"{exact['mean']:.3f} [{exact['ci95'][0]:.3f}, "
                f"{exact['ci95'][1]:.3f}] | {delta} |")
        lines.append("")
    return "\n".join(lines)


def _stable_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_V3)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_V3)
    parser.add_argument(
        "--execute", action="store_true",
        help="write summary.json and summary.md; otherwise validate and print")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec_v3(args.spec, args.lock)
    require_all_selected_salience_v3(spec)
    admissions = {stage: _validate_admission(spec, stage) for stage in STAGES}
    arms = list(spec["carrier_training"]["arms"])
    seeds = [int(seed) for seed in spec["carrier_training"]["seeds"]]
    rows = [
        _validate_cell(spec, stage, arm, seed)
        for stage in STAGES for arm in arms for seed in seeds
    ]
    expected = len(STAGES) * len(arms) * len(seeds)
    _require(len(rows) == expected,
             f"formal V3 grid has {len(rows)} rows, expected {expected}")
    summary = {
        "schema_version": 1,
        "study": spec["study"],
        "complete": True,
        "formal_lock": lock_receipt_v3(spec),
        "analysis": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "inference_unit": "matched carrier-training seed",
            "decision_observation_excluded": True,
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
            "all_development_stages_selected": True,
            "all_formal_admission_gates_passed": True,
            "all_frozen_hosts_unchanged": True,
            "threshold_changed_from_v1_or_v2": False,
            "fail_closed": True,
        },
    }
    root = resolve_path_v3(spec["artifacts"]["root"])
    if not args.execute:
        print(_markdown(summary))
        return
    _atomic_text(root / "summary.json", _stable_json(summary))
    _atomic_text(root / "summary.md", _markdown(summary) + "\n")
    print(f"[shell-v3-aggregate] wrote {root / 'summary.json'} and "
          f"{root / 'summary.md'}")


if __name__ == "__main__":
    main()
