#!/usr/bin/env python3
"""Fail-closed five-seed aggregation for Reacher context and rollout controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_paper_a_expansion import (  # noqa: E402
    Cell as ParentCell,
    ROLLOUT_METRICS,
    raw_context_readout,
    validate_context as validate_parent_context,
    validate_rollout as validate_parent_rollout,
)
from scripts.paper_a_context_rollout_extension_spec import (  # noqa: E402
    COMBINED_SEEDS,
    CONTEXTS,
    DEFAULT_SPEC,
    EXTENSION_SEEDS,
    OBJECTIVES,
    OBJECTIVE_NAMES,
    PARENT_SEEDS,
    TASKS,
    TASK_NAMES,
    TASK_SLUGS,
    ExtensionCell,
    ExtensionSpecError,
    expected_cells,
    load_locked_spec,
    repo_path,
    sha256_file,
    task_record,
    validate_device,
)
from scripts.run_paper_a_context_rollout_extension import (  # noqa: E402
    PRODUCTS,
    stage_paths,
    underlying_command,
)


class ExtensionAggregationError(RuntimeError):
    """Raised when any grid, schema, or provenance check fails."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ExtensionAggregationError(f"{label} must be a mapping")
    return value


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _same_number(value: Any, expected: Any) -> bool:
    return (_finite(value) and _finite(expected)
            and math.isclose(float(value), float(expected),
                             rel_tol=1e-10, abs_tol=1e-12))


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ExtensionAggregationError(f"{label}: unreadable JSON ({error})") from error
    if not isinstance(payload, dict):
        raise ExtensionAggregationError(f"{label}: JSON root must be an object")
    return payload


def _validation_config(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = repo_path(spec["parent"]["config"]["path"], "parent.config.path")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ExtensionAggregationError("locked parent config is not a mapping")
    # Parent validation functions consume these exact decks.  Seed count does
    # not affect per-cell validation, but the combined deck is explicit here.
    payload = dict(payload)
    payload["long_context"] = dict(payload["long_context"])
    payload["learned_rollout"] = dict(payload["learned_rollout"])
    payload["long_context"]["seeds"] = list(COMBINED_SEEDS)
    payload["learned_rollout"]["seeds"] = list(COMBINED_SEEDS)
    return payload


def _parent_cell(cell: ExtensionCell) -> ParentCell:
    wave = "long_context" if cell.wave == "long_context" else "rollout"
    return ParentCell(wave, cell.task, cell.variant, cell.seed, cell.metrics_path)


def _validate_history(cell: ExtensionCell, directory: Path,
                      metrics: Mapping[str, Any], errors: list[str]) -> None:
    path = directory / "history.csv"
    if not path.is_file():
        errors.append(f"{cell.semantic_label}: missing history.csv")
        return
    try:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as error:
        errors.append(f"{cell.semantic_label}: unreadable history.csv ({error})")
        return
    expected_epochs = 60
    if len(rows) != expected_epochs:
        errors.append(
            f"{cell.semantic_label}: history has {len(rows)} rows, expected 60")
        return
    try:
        epochs = [int(row["epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        errors.append(f"{cell.semantic_label}: malformed epoch column")
        return
    if epochs != list(range(1, expected_epochs + 1)):
        errors.append(f"{cell.semantic_label}: history epoch sequence changed")
    numeric_fields = (("train_prediction_mse", "val_prediction_mse",
                       "val_target_windows", "learning_rate", "epoch_seconds")
                      if cell.wave == "long_context" else ("loss", "lr"))
    for field in numeric_fields:
        try:
            values = np.asarray([float(row[field]) for row in rows])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{cell.semantic_label}: malformed {field} history column")
            continue
        if not np.isfinite(values).all():
            errors.append(f"{cell.semantic_label}: non-finite {field} history value")
        if field not in {"lr", "learning_rate"} and (values < 0).any():
            errors.append(f"{cell.semantic_label}: negative {field} history value")
    if cell.wave == "learned_rollout":
        try:
            last_loss = float(rows[-1]["loss"])
        except (KeyError, TypeError, ValueError):
            return
        if not _same_number(metrics.get("final_train_loss"), last_loss):
            errors.append(
                f"{cell.semantic_label}: final loss disagrees with history")


def _validate_receipt(spec: Mapping[str, Any], cell: ExtensionCell,
                      directory: Path, receipt: Mapping[str, Any],
                      errors: list[str], artifact_ledger: dict[str, str]) -> None:
    label = cell.semantic_label
    expected_wave = cell.wave
    expected_task = task_record(spec, cell.task)
    expected_variant_name = (
        f"Context length {cell.variant.removeprefix('h')}"
        if cell.wave == "long_context" else OBJECTIVE_NAMES[cell.variant])
    checks = {
        "schema_version": 1,
        "study": spec["study"],
        "spec": spec["_spec_record"],
        "source": "new extension training cell",
        "wave": expected_wave,
        "semantic_task_name": expected_task["name"],
        "semantic_task_slug": expected_task["slug"],
        "internal_task_key": cell.task,
        "variant": cell.variant,
        "variant_name": expected_variant_name,
        "seed": cell.seed,
        "parent_config": spec["parent"]["config"],
        "official_weights": spec["parent"]["official_weights"],
        "parent_artifacts_modified": False,
    }
    trainer_key = ("long_context" if cell.wave == "long_context"
                   else "learned_rollout")
    checks["trainer"] = spec["parent"]["trainers"][trainer_key]
    for key, expected in checks.items():
        if receipt.get(key) != expected:
            errors.append(f"{label}: receipt {key} differs from locked contract")
    device = receipt.get("device")
    try:
        validate_device(spec, str(device))
    except ExtensionSpecError as error:
        errors.append(f"{label}: invalid receipt device ({error})")
        device = "cuda:1"
    _, staged_product, expected_final = stage_paths(
        spec, cell.wave, cell.task, cell.variant, cell.seed)
    if expected_final != directory:
        errors.append(f"{label}: extension directory is not canonical")
    expected_command = list(underlying_command(
        spec, cell.wave, cell.task, cell.variant, cell.seed,
        str(device), staged_product))
    if receipt.get("command") != expected_command:
        errors.append(f"{label}: receipt command differs from the locked trainer call")
    products = receipt.get("products")
    if not isinstance(products, dict) or set(products) != set(PRODUCTS):
        errors.append(f"{label}: receipt product ledger is incomplete")
        return
    for name in PRODUCTS:
        path = directory / name
        if not path.is_file():
            errors.append(f"{label}: missing receipted product {name}")
            continue
        item = products.get(name, {})
        observed_hash = sha256_file(path)
        observed_bytes = path.stat().st_size
        if item.get("sha256") != observed_hash:
            errors.append(f"{label}: {name} hash differs from receipt")
        if item.get("bytes") != observed_bytes:
            errors.append(f"{label}: {name} byte count differs from receipt")
        artifact_ledger[str(path.relative_to(ROOT))] = observed_hash


def _validate_context_deck(spec: Mapping[str, Any], cell: ExtensionCell,
                           metrics: Mapping[str, Any], errors: list[str]) -> None:
    deck = spec["long_context"]
    config = metrics.get("config", {})
    initialization = metrics.get("initialization", {})
    exact = {
        "history_len": int(cell.variant.removeprefix("h")),
        "position_init": deck["position_initialization"],
        "task_family": task_record(spec, cell.task)["family"],
        "epochs": deck["epochs"],
        "batch_size": deck["batch_size"],
        "lr": deck["learning_rate"],
        "weight_decay": deck["weight_decay"],
        "grad_clip": deck["grad_clip"],
        "seed": cell.seed,
        "amp": deck["amp"],
        "amp_dtype": deck["amp_dtype"],
        "objective": deck["objective"],
        "encoder_frozen": True,
        "encoder_instantiated_during_training": False,
    }
    for key, expected in exact.items():
        value = config.get(key)
        equal = (_same_number(value, expected)
                 if isinstance(expected, float) else value == expected)
        if not equal:
            errors.append(f"{cell.semantic_label}: config {key} changed")
    if initialization.get("position_initialization") != deck["position_initialization"]:
        errors.append(f"{cell.semantic_label}: position initialization changed")
    semantic = metrics.get("semantic_target_readout", {})
    for split in ("train_target_windows", "validation_target_windows"):
        coverage = semantic.get(split, {})
        if coverage.get("target_time_min") != deck["decision_index"] \
                or coverage.get("target_time_max") != deck["decision_index"]:
            errors.append(f"{cell.semantic_label}: semantic endpoint changed")
        if coverage.get("future_target_observation_consumed") is not False:
            errors.append(f"{cell.semantic_label}: decision observation was consumed")


def _validate_rollout_deck(spec: Mapping[str, Any], cell: ExtensionCell,
                           metrics: Mapping[str, Any], errors: list[str]) -> None:
    deck = spec["learned_rollout"]
    exact = {
        "task": cell.task,
        "objective": cell.variant,
        "overshoot_horizon": deck["objective_horizons"][cell.variant],
        "seed": cell.seed,
        "epochs": deck["epochs"],
        "batch_size": deck["batch_size"],
        "learning_rate": deck["learning_rate"],
        "frozen_encoder": True,
        "trainable_components": deck["trainable_components"],
        "anchor": deck["anchor"],
    }
    for key, expected in exact.items():
        value = metrics.get(key)
        equal = (_same_number(value, expected)
                 if isinstance(expected, float) else value == expected)
        if not equal:
            errors.append(f"{cell.semantic_label}: rollout field {key} changed")
    # Weight decay is not serialized by the parent trainer; the receipted,
    # reconstructed command above is therefore its fail-closed source of truth.
    if metrics.get("gate") != (
            "better than copy-last and positive true-action advantage at 1,2,4,8"):
        errors.append(f"{cell.semantic_label}: competence gate changed")


def validate_grid(spec: Mapping[str, Any]
                  ) -> tuple[dict[tuple[str, str, str, int], dict[str, Any]],
                             dict[str, str]]:
    validation_config = _validation_config(spec)
    parent_root = repo_path(spec["parent"]["root"], "parent.root")
    cells = expected_cells(spec)
    expected_metrics = {cell.metrics_path.resolve() for cell in cells
                        if cell.source == "extension"}
    expected_receipts = {cell.metrics_path.with_name("receipt.json").resolve()
                         for cell in cells if cell.source == "extension"}
    expected_artifacts = set(expected_metrics).union(expected_receipts)
    for metrics_path in expected_metrics:
        expected_artifacts.add(metrics_path.with_name("history.csv"))
        expected_artifacts.add(metrics_path.with_name("checkpoint.pt"))
    discovered_metrics = set()
    discovered_receipts = set()
    discovered_artifacts = set()
    for key in ("long_context", "learned_rollout"):
        directory = repo_path(spec["output"][key], f"output.{key}")
        if directory.exists():
            discovered_artifacts.update(
                path.resolve() for path in directory.glob("**/*") if path.is_file())
            discovered_metrics.update(
                path.resolve() for path in directory.glob("**/metrics.json"))
            discovered_receipts.update(
                path.resolve() for path in directory.glob("**/receipt.json"))
    errors = [
        f"unexpected extension metric: {path.relative_to(ROOT)}"
        for path in sorted(discovered_metrics.difference(expected_metrics))
    ]
    errors.extend(
        f"unexpected extension receipt: {path.relative_to(ROOT)}"
        for path in sorted(discovered_receipts.difference(expected_receipts)))
    errors.extend(
        f"unexpected extension artifact: {path.relative_to(ROOT)}"
        for path in sorted(discovered_artifacts.difference(expected_artifacts)))
    staging = repo_path(spec["output"]["staging"], "output.staging")
    if staging.exists() and any(staging.iterdir()):
        errors.append(f"incomplete staging artifacts remain under {staging}")

    records: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    artifact_ledger: dict[str, str] = {}
    for cell in cells:
        label = cell.semantic_label
        directory = cell.metrics_path.parent
        if not cell.metrics_path.is_file():
            errors.append(f"{label}: missing metrics.json")
            continue
        metrics = _load_json(cell.metrics_path, label)
        local_errors: list[str] = []
        parent_cell = _parent_cell(cell)
        hash_cache: dict[Path, str] = {}
        if cell.wave == "long_context":
            validate_parent_context(
                parent_cell, metrics, parent_root, validation_config,
                local_errors, hash_cache)
            _validate_context_deck(spec, cell, metrics, local_errors)
        else:
            validate_parent_rollout(
                parent_cell, metrics, validation_config, local_errors)
            _validate_rollout_deck(spec, cell, metrics, local_errors)
        _validate_history(cell, directory, metrics, local_errors)
        checkpoint = directory / "checkpoint.pt"
        if not checkpoint.is_file():
            local_errors.append(f"{label}: missing checkpoint.pt")
        if cell.source == "extension":
            receipt_path = directory / "receipt.json"
            if not receipt_path.is_file():
                local_errors.append(f"{label}: missing receipt.json")
            else:
                receipt = _load_json(receipt_path, label + " receipt")
                _validate_receipt(
                    spec, cell, directory, receipt, local_errors, artifact_ledger)
                artifact_ledger[str(receipt_path.relative_to(ROOT))] = \
                    sha256_file(receipt_path)
        else:
            # The locked parent summary already authenticates parent metrics.
            artifact_ledger[str(cell.metrics_path.relative_to(ROOT))] = \
                sha256_file(cell.metrics_path)
            if (directory / "history.csv").is_file():
                artifact_ledger[str((directory / "history.csv").relative_to(ROOT))] = \
                    sha256_file(directory / "history.csv")
            if checkpoint.is_file():
                artifact_ledger[str(checkpoint.relative_to(ROOT))] = \
                    sha256_file(checkpoint)
        errors.extend(local_errors)
        records[(cell.wave, cell.task, cell.variant, cell.seed)] = metrics

    if errors:
        raise ExtensionAggregationError(
            "five-seed context/rollout validation failed:\n- "
            + "\n- ".join(sorted(set(errors))))
    if len(records) != len(cells):
        raise ExtensionAggregationError(
            f"validated {len(records)} cells, expected {len(cells)}")
    return records, dict(sorted(artifact_ledger.items()))


def _rng(spec: Mapping[str, Any], key: str) -> np.random.Generator:
    material = f"{spec['analysis']['bootstrap_seed']}:{key}".encode()
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(seed)


def seed_summary(spec: Mapping[str, Any], values: Mapping[int, float],
                 key: str) -> dict[str, Any]:
    seeds = sorted(values)
    if seeds != list(COMBINED_SEEDS):
        raise ExtensionAggregationError(
            f"{key}: expected five seeds {COMBINED_SEEDS}, observed {seeds}")
    array = np.asarray([values[seed] for seed in seeds], dtype=np.float64)
    if not np.isfinite(array).all():
        raise ExtensionAggregationError(f"{key}: non-finite seed value")
    draws = int(spec["analysis"]["bootstrap_draws"])
    indices = _rng(spec, key).integers(0, len(array), size=(draws, len(array)))
    bootstrap = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "n": len(array),
        "seeds": seeds,
        "values": array.tolist(),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "ci95": [float(low), float(high)],
        "bootstrap": {
            "unit": "optimizer/model seed",
            "paired": False,
            "draws": draws,
            "interval": "percentile",
            "level": 0.95,
        },
    }


def paired_seed_summary(spec: Mapping[str, Any], candidate: Mapping[int, float],
                        reference: Mapping[int, float], key: str,
                        candidate_name: str, reference_name: str
                        ) -> dict[str, Any]:
    if sorted(candidate) != list(COMBINED_SEEDS) \
            or sorted(reference) != list(COMBINED_SEEDS):
        raise ExtensionAggregationError(f"{key}: incomplete matched-seed pair")
    differences = {
        seed: float(candidate[seed]) - float(reference[seed])
        for seed in COMBINED_SEEDS
    }
    output = seed_summary(spec, differences, key)
    output["bootstrap"]["paired"] = True
    output["bootstrap"]["pairing"] = "same task and optimizer/model seed"
    output.update({
        "estimand": f"{candidate_name} minus {reference_name}",
        "candidate": candidate_name,
        "reference": reference_name,
        "complete": True,
        "wins": int(sum(value > 0 for value in differences.values())),
        "ties": int(sum(value == 0 for value in differences.values())),
    })
    return output


def _equal_task_values(per_task: Mapping[str, Mapping[int, float]]) -> dict[int, float]:
    if set(per_task) != set(TASKS):
        raise ExtensionAggregationError("equal-task pool does not cover both tasks")
    return {
        seed: float(np.mean([per_task[task][seed] for task in TASKS]))
        for seed in COMBINED_SEEDS
    }


def aggregate_context(spec: Mapping[str, Any], records: Mapping[
        tuple[str, str, str, int], Mapping[str, Any]]) -> dict[str, Any]:
    parent_root = repo_path(spec["parent"]["root"], "parent.root")
    tasks: dict[str, Any] = {}
    accuracy_pool: dict[int, dict[str, dict[int, float]]] = {
        history: {} for history in CONTEXTS}
    mse_pool: dict[int, dict[str, dict[int, float]]] = {
        history: {} for history in CONTEXTS}
    for task in TASKS:
        histories: dict[str, Any] = {}
        accuracy_values: dict[int, dict[int, float]] = {}
        mse_values: dict[int, dict[int, float]] = {}
        for history in CONTEXTS:
            variant = f"h{history}"
            metrics = {
                seed: records[("long_context", task, variant, seed)]
                for seed in COMBINED_SEEDS
            }
            accuracy = {
                seed: float(item["semantic_target_readout"]["value"])
                for seed, item in metrics.items()
            }
            mse = {
                seed: float(item["best_checkpoint_prediction_mse"]["validation"])
                for seed, item in metrics.items()
            }
            accuracy_values[history] = accuracy
            mse_values[history] = mse
            accuracy_pool[history][task] = accuracy
            mse_pool[history][task] = mse
            histories[str(history)] = {
                "history": history,
                "raw_legal_context_readout": raw_context_readout(
                    parent_root, task, history),
                "trained_predictor_semantic_accuracy": seed_summary(
                    spec, accuracy,
                    f"context/{TASK_SLUGS[task]}/h{history}/accuracy"),
                "validation_next_latent_mse": seed_summary(
                    spec, mse, f"context/{TASK_SLUGS[task]}/h{history}/mse"),
            }
        comparisons = {
            str(history): {
                "trained_semantic_accuracy_delta": paired_seed_summary(
                    spec, accuracy_values[history], accuracy_values[3],
                    f"context/{TASK_SLUGS[task]}/h{history}-h3/accuracy",
                    f"context {history}", "context 3"),
                "validation_mse_delta": paired_seed_summary(
                    spec, mse_values[history], mse_values[3],
                    f"context/{TASK_SLUGS[task]}/h{history}-h3/mse",
                    f"context {history}", "context 3"),
            }
            for history in CONTEXTS if history != 3
        }
        tasks[TASK_SLUGS[task]] = {
            "display_name": TASK_NAMES[task],
            "histories": histories,
            "paired_vs_three-latent_context": comparisons,
        }
    pooled_accuracy = {
        history: _equal_task_values(accuracy_pool[history])
        for history in CONTEXTS}
    pooled_mse = {
        history: _equal_task_values(mse_pool[history])
        for history in CONTEXTS}
    pooled = {
        str(history): {
            "trained_predictor_semantic_accuracy": seed_summary(
                spec, pooled_accuracy[history],
                f"context/equal-task/h{history}/accuracy"),
            "validation_next_latent_mse": seed_summary(
                spec, pooled_mse[history], f"context/equal-task/h{history}/mse"),
            "paired_accuracy_delta_vs_context_3": (
                None if history == 3 else paired_seed_summary(
                    spec, pooled_accuracy[history], pooled_accuracy[3],
                    f"context/equal-task/h{history}-h3/accuracy",
                    f"context {history}", "context 3")),
            "paired_mse_delta_vs_context_3": (
                None if history == 3 else paired_seed_summary(
                    spec, pooled_mse[history], pooled_mse[3],
                    f"context/equal-task/h{history}-h3/mse",
                    f"context {history}", "context 3")),
        }
        for history in CONTEXTS
    }
    return {
        "tasks": tasks,
        "equal_task_matched_seed_pool": pooled,
        "pooling_contract": "average the two task values within seed, then bootstrap seeds",
    }


def aggregate_rollout(spec: Mapping[str, Any], records: Mapping[
        tuple[str, str, str, int], Mapping[str, Any]]) -> dict[str, Any]:
    horizons = tuple(spec["learned_rollout"]["evaluation_horizons"])
    tasks: dict[str, Any] = {}
    pool: dict[str, dict[int, dict[str, dict[str, dict[int, float]]]]] = {
        objective: {horizon: {metric: {} for metric in ROLLOUT_METRICS}
                    for horizon in horizons}
        for objective in OBJECTIVES
    }
    for task in TASKS:
        objective_output: dict[str, Any] = {}
        values: dict[str, dict[int, dict[str, dict[int, float]]]] = {}
        for objective in OBJECTIVES:
            cells = {
                seed: records[("learned_rollout", task, objective, seed)]
                for seed in COMBINED_SEEDS
            }
            values[objective] = {}
            horizon_output: dict[str, Any] = {}
            for horizon in horizons:
                values[objective][horizon] = {}
                metric_output: dict[str, Any] = {}
                for metric in ROLLOUT_METRICS:
                    by_seed = {
                        seed: float(item["horizons"][str(horizon)][metric])
                        for seed, item in cells.items()
                    }
                    values[objective][horizon][metric] = by_seed
                    pool[objective][horizon][metric][task] = by_seed
                    metric_output[metric] = seed_summary(
                        spec, by_seed,
                        f"rollout/{TASK_SLUGS[task]}/{objective}/h{horizon}/{metric}")
                copy = values[objective][horizon]["copy_last_normalized_mse"]
                if any(value <= 0 for value in copy.values()):
                    raise ExtensionAggregationError("copy-last normalized MSE is nonpositive")
                ratio = {
                    seed: values[objective][horizon]["normalized_latent_mse"][seed]
                    / copy[seed] for seed in COMBINED_SEEDS
                }
                metric_output["model_to_copy_ratio"] = seed_summary(
                    spec, ratio,
                    f"rollout/{TASK_SLUGS[task]}/{objective}/h{horizon}/model-copy")
                horizon_output[str(horizon)] = metric_output
            passed = sorted(seed for seed, item in cells.items()
                            if item["rollout_competent_through_8"])
            objective_output[objective.replace("_", "-")] = {
                "display_name": OBJECTIVE_NAMES[objective],
                "horizons": horizon_output,
                "competence_gate_through_horizon_8": {
                    "passed_seeds": passed,
                    "pass_count": len(passed),
                    "evaluated_seeds": list(COMBINED_SEEDS),
                    "all_five_seeds_pass": passed == list(COMBINED_SEEDS),
                },
            }
        contrasts: dict[str, Any] = {}
        for horizon in horizons:
            contrasts[str(horizon)] = {
                metric: paired_seed_summary(
                    spec,
                    values["overshoot_8"][horizon][metric],
                    values["one_step"][horizon][metric],
                    f"rollout/{TASK_SLUGS[task]}/overshoot-one/h{horizon}/{metric}",
                    "eight-step overshooting", "one-step objective")
                for metric in ROLLOUT_METRICS
            }
        tasks[TASK_SLUGS[task]] = {
            "display_name": TASK_NAMES[task],
            "objectives": objective_output,
            "paired_overshooting_minus_one_step": contrasts,
        }

    pooled: dict[str, Any] = {}
    pooled_values: dict[str, dict[int, dict[str, dict[int, float]]]] = {
        objective: {} for objective in OBJECTIVES}
    for objective in OBJECTIVES:
        pooled[objective.replace("_", "-")] = {"horizons": {}}
        for horizon in horizons:
            pooled_values[objective][horizon] = {}
            output_metrics: dict[str, Any] = {}
            for metric in ROLLOUT_METRICS:
                by_seed = _equal_task_values(pool[objective][horizon][metric])
                pooled_values[objective][horizon][metric] = by_seed
                output_metrics[metric] = seed_summary(
                    spec, by_seed,
                    f"rollout/equal-task/{objective}/h{horizon}/{metric}")
            pooled[objective.replace("_", "-")]["horizons"][str(horizon)] = \
                output_metrics
    pooled_contrasts = {
        str(horizon): {
            metric: paired_seed_summary(
                spec,
                pooled_values["overshoot_8"][horizon][metric],
                pooled_values["one_step"][horizon][metric],
                f"rollout/equal-task/overshoot-one/h{horizon}/{metric}",
                "eight-step overshooting", "one-step objective")
            for metric in ROLLOUT_METRICS
        }
        for horizon in horizons
    }
    return {
        "tasks": tasks,
        "equal_task_matched_seed_pool": pooled,
        "equal_task_paired_overshooting_minus_one_step": pooled_contrasts,
        "pooling_contract": "average the two task values within seed, then bootstrap seeds",
    }


def build_summary(spec: Mapping[str, Any]) -> dict[str, Any]:
    records, artifact_ledger = validate_grid(spec)
    return {
        "schema_version": 1,
        "study": spec["study"],
        "completion": {
            "complete": True,
            "expected_parent_cells": 36,
            "expected_extension_cells": 24,
            "validated_combined_cells": len(records),
            "combined_seed_count": 5,
            "seeds": list(COMBINED_SEEDS),
        },
        "semantic_task_names": {
            TASK_SLUGS[task]: TASK_NAMES[task] for task in TASKS},
        "long_context": aggregate_context(spec, records),
        "learned_rollout": aggregate_rollout(spec, records),
        "analysis": {
            "confidence_level": spec["analysis"]["confidence_level"],
            "interval": spec["analysis"]["interval"],
            "bootstrap_draws": spec["analysis"]["bootstrap_draws"],
            "bootstrap_seed": spec["analysis"]["bootstrap_seed"],
            "primary_resampling_unit": "matched optimizer/model seed",
            "paired_contrasts": (
                "same-seed context-length and objective differences are formed "
                "before seed bootstrap"),
            "task_pooling": spec["analysis"]["task_weighting"],
        },
        "validation": {
            "fail_closed": True,
            "missing_cells_permitted": False,
            "unexpected_cells_permitted": False,
            "parent_metric_hashes_match_locked_parent_summary": True,
            "extension_product_hashes_match_receipts": True,
            "trainer_source_hashes_match_locked_spec": True,
            "parent_output_modified": False,
            "allowed_training_devices": spec["execution"]["allowed_devices"],
        },
        "provenance": {
            "spec": dict(spec["_spec_record"]),
            "parent_config": dict(spec["parent"]["config"]),
            "parent_summary": dict(spec["parent"]["summary"]),
            "official_weights": dict(spec["parent"]["official_weights"]),
            "parent_caches": spec["parent"]["caches"],
            "trainers": spec["parent"]["trainers"],
            "parent_launcher": spec["parent"]["launcher"],
            "source_artifact_sha256": artifact_ledger,
        },
    }


def _stat(value: Mapping[str, Any], digits: int = 3) -> str:
    return (f"{value['mean']:.{digits}f} "
            f"[{value['ci95'][0]:.{digits}f}, {value['ci95'][1]:.{digits}f}]")


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Reacher long-context and learned-rollout five-seed extension",
        "",
        "All entries combine the three locked parent seeds with two new training "
        "seeds. Intervals are deterministic 95% percentile seed-bootstrap "
        "intervals; every comparison is differenced within seed before resampling.",
        "",
        "## Long-context control",
        "",
        "| Task | Context | Raw legal-context accuracy | Predictor accuracy | "
        "Paired accuracy change vs. context 3 | Validation next-latent MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in summary["long_context"]["tasks"].values():
        for history, row in task["histories"].items():
            delta = task["paired_vs_three-latent_context"].get(history)
            delta_text = ("reference" if delta is None else
                          _stat(delta["trained_semantic_accuracy_delta"]))
            lines.append(
                f"| {task['display_name']} | {history} | "
                f"{row['raw_legal_context_readout']['value']:.3f} | "
                f"{_stat(row['trained_predictor_semantic_accuracy'])} | "
                f"{delta_text} | {_stat(row['validation_next_latent_mse'], 4)} |")
    lines += [
        "",
        "## Learned-rollout control",
        "",
        "| Task | Objective | Horizon | Normalized latent MSE | "
        "Model/copy-last ratio | True-action advantage | Gate passes |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in summary["learned_rollout"]["tasks"].values():
        for objective in task["objectives"].values():
            passes = objective["competence_gate_through_horizon_8"]["pass_count"]
            for horizon, row in objective["horizons"].items():
                lines.append(
                    f"| {task['display_name']} | {objective['display_name']} | "
                    f"{horizon} | {_stat(row['normalized_latent_mse'], 4)} | "
                    f"{_stat(row['model_to_copy_ratio'], 3)} | "
                    f"{_stat(row['true_action_advantage'], 3)} | {passes}/5 |")
    lines += [
        "",
        "## Validation and provenance",
        "",
        "- Combined grid: 36 locked parent cells + 24 isolated extension cells "
        "= 60 validated cells.",
        "- Training devices: CUDA 1 and CUDA 2 only.",
        "- Parent metrics are authenticated by the locked parent summary; every "
        "new metric, history, and checkpoint is authenticated by its cell receipt.",
        "- Missing, extra, staged, schema-invalid, or hash-mismatched cells abort "
        "aggregation.",
        "",
    ]
    rendered = "\n".join(lines)
    # Paper-facing prose must use semantic task names.  Internal path keys are
    # intentionally confined to JSON provenance and never rendered here.
    for forbidden in (" T1", " T3", "`t1`", "`t3`", "/t1/", "/t3/"):
        if forbidden in rendered:
            raise ExtensionAggregationError(
                f"paper-facing summary leaked internal task key {forbidden!r}")
    return rendered


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_spec(args.spec)
    if not args.execute:
        print("[context-rollout-aggregate] preview only; --execute validates all "
              "60 cells and writes the deterministic summaries")
        return
    summary = build_summary(spec)
    json_path = repo_path(spec["output"]["summary_json"], "output.summary_json")
    markdown_path = repo_path(
        spec["output"]["summary_markdown"], "output.summary_markdown")
    digest_path = repo_path(
        spec["output"]["summary_sha256"], "output.summary_sha256")
    _atomic_write(json_path,
                  json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_write(markdown_path, render_markdown(summary) + "\n")
    digest = (
        f"{sha256_file(json_path)}  {json_path.name}\n"
        f"{sha256_file(markdown_path)}  {markdown_path.name}\n")
    _atomic_write(digest_path, digest)
    print(f"[context-rollout-aggregate] validated 60 cells; wrote {json_path}")


if __name__ == "__main__":
    main()
