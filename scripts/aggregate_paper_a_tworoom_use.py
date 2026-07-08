#!/usr/bin/env python3
"""Fail-closed aggregation for the matched-host TwoRoom use experiment.

The execution deck is shared by every arm and carrier seed.  Confidence
intervals therefore resample carrier seeds jointly across arms and resample
held-out episodes once per draw, stratified by the registered 16-way
color-location label.  All reported contrasts retain both kinds of pairing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    load_verified_npz,
    stable_json,
)
from lewm.official_tasks.matched_memory import balanced_joint_labels  # noqa: E402
from scripts.paper_a_matched_host_spec import (  # noqa: E402
    ARMS,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    SEEDS,
    load_locked_spec,
    output_path,
    resolve_input_path,
    sha256_file,
)
from scripts.prepare_paper_a_matched_host import host_manifest_path  # noqa: E402
from scripts.prepare_paper_a_tworoom_use import deck_path, gate_path  # noqa: E402
from scripts.train_paper_a_matched_host import carrier_directory  # noqa: E402


BRANCH = "tworoom-external-waypoint-use"
EPISODES = 480
ENDPOINTS = (
    "goal_selection_accuracy",
    "executed_success_rate",
    "mean_selected_distance",
    "mean_distance_regret",
)
ARRAY_FOR_ENDPOINT = {
    "goal_selection_accuracy": "goal_correct",
    "executed_success_rate": "executed_success",
    "mean_selected_distance": "selected_distance",
    "mean_distance_regret": "distance_regret",
}
CELL_ARRAYS = (
    "labels",
    "joint_labels",
    "predictions",
    "goal_correct",
    "executed_success",
    "selected_distance",
    "oracle_success",
    "random_success",
    "oracle_distance",
    "random_distance",
    "distance_regret",
)
DECK_KEYS = {
    "z",
    "actions",
    "color_label",
    "location_label",
    "combination_label",
    "decision_state",
    "goal_waypoints",
    "success_matrix",
    "distance_matrix",
    "controller_target_success",
    "controller_final_state",
    "reset_replay",
    "random_choice",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def use_cell_directory(spec: Mapping[str, Any], arm: str, seed: int) -> Path:
    """Return the one canonical output directory for a use-evaluation cell."""

    return output_path(spec, "use") / "cells" / arm / f"seed-{seed}"


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _record(path: Path) -> dict[str, str]:
    return {"path": _relative(path), "sha256": sha256_file(path)}


def _require_binary(values: np.ndarray, label: str) -> None:
    if values.shape != (EPISODES,) or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{label} must be a length-{EPISODES} binary array")


def _require_close(actual: Any, expected: float, label: str,
                   *, tolerance: float = 1e-7) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a scalar") from error
    if not np.isfinite(value) or not np.isclose(
            value, expected, rtol=tolerance, atol=tolerance):
        raise ValueError(f"{label} differs: {value} != {expected}")


def _validate_deck_and_gate(
        spec: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any],
                                         dict[str, Any]]:
    deck_file = deck_path(dict(spec))
    gate_file = gate_path(dict(spec))
    arrays, sidecar = load_verified_npz(deck_file)
    expected_sidecar = {
        "schema": "paper_a_matched_tworoom_use_deck_v1",
        "study": spec["study"],
        "lock": spec["_lock"],
        "episodes": EPISODES,
        "cue_age": 15,
        "prefix_label_blind": True,
        "physical_gpu": 0,
        "fixed_physics_seed": spec["tworoom_use"]["physics_seed"],
    }
    failed = [key for key, value in expected_sidecar.items()
              if sidecar.get(key) != value]
    if failed or set(arrays) != DECK_KEYS:
        raise ValueError(
            f"TwoRoom use deck schema differs: fields={failed}, "
            f"keys={sorted(set(arrays) ^ DECK_KEYS)}")

    labels = np.asarray(arrays["location_label"], dtype=np.int64)
    colors = np.asarray(arrays["color_label"], dtype=np.int64)
    joint = np.asarray(arrays["combination_label"], dtype=np.int64)
    random_choice = np.asarray(arrays["random_choice"], dtype=np.int64)
    expected_labels = balanced_joint_labels(
        EPISODES, int(spec["tworoom_use"]["label_seed"]))
    if any(value.shape != (EPISODES,) for value in (
            labels, colors, joint, random_choice)):
        raise ValueError("TwoRoom deck label vectors must have 480 rows")
    if not np.array_equal(colors, expected_labels.color) \
            or not np.array_equal(labels, expected_labels.location) \
            or not np.array_equal(joint, expected_labels.combination) \
            or not np.isin(labels, np.arange(4)).all() \
            or not np.isin(colors, np.arange(4)).all() \
            or not np.array_equal(joint, colors * 4 + labels) \
            or not np.array_equal(
                np.bincount(joint, minlength=16), np.full(16, 30)) \
            or not np.isin(random_choice, np.arange(4)).all():
        raise ValueError("TwoRoom deck is not the exact balanced 16-way deck")
    expected_random = np.random.default_rng(
        int(spec["tworoom_use"]["reset_seed"]) + 1).integers(
            0, 4, size=EPISODES)
    if not np.array_equal(random_choice, expected_random):
        raise ValueError("TwoRoom realized-random choices differ from their seed")
    expected_shapes = {
        "z": (EPISODES, 20, 192),
        "actions": (EPISODES, 19, 10),
        "decision_state": (EPISODES, 2),
        "goal_waypoints": (4, 2),
    }
    if any(np.asarray(arrays[key]).shape != shape
           for key, shape in expected_shapes.items()) \
            or np.asarray(arrays["z"]).dtype != np.float32 \
            or np.asarray(arrays["actions"]).dtype != np.float32 \
            or any(not np.isfinite(np.asarray(arrays[key])).all()
                   for key in expected_shapes):
        raise ValueError("TwoRoom deck model inputs have invalid shapes or values")
    if not np.allclose(
            np.asarray(arrays["goal_waypoints"], dtype=np.float64),
            np.asarray(spec["tworoom_use"]["goal_waypoints"],
                       dtype=np.float64), rtol=0, atol=1e-7):
        raise ValueError("TwoRoom deck waypoints differ from the protocol")
    success = np.asarray(arrays["success_matrix"])
    distance = np.asarray(arrays["distance_matrix"], dtype=np.float64)
    controller_target_success = np.asarray(arrays["controller_target_success"])
    controller_final_state = np.asarray(
        arrays["controller_final_state"], dtype=np.float64)
    reset_replay = np.asarray(arrays["reset_replay"])
    if success.shape != (EPISODES, 4, 4) \
            or not np.isin(success, (0, 1)).all() \
            or distance.shape != (EPISODES, 4, 4) \
            or not np.isfinite(distance).all() \
            or (distance < 0).any() \
            or controller_target_success.shape != (EPISODES, 4) \
            or not np.isin(controller_target_success, (0, 1)).all() \
            or controller_final_state.shape != (EPISODES, 4, 2) \
            or not np.isfinite(controller_final_state).all() \
            or reset_replay.shape != (EPISODES, 4) \
            or not np.isin(reset_replay, (0, 1)).all():
        raise ValueError("TwoRoom deck controller matrices are invalid")
    radius = float(spec["tworoom_use"]["success_radius"])
    goals = np.asarray(arrays["goal_waypoints"], dtype=np.float64)
    reproduced_distance = np.linalg.norm(
        controller_final_state[:, :, None, :] - goals[None, None, :, :],
        axis=-1)
    if not np.array_equal(success, distance < radius) \
            or not np.allclose(
                distance, reproduced_distance, rtol=1e-6, atol=1e-5) \
            or not np.array_equal(
                controller_target_success,
                success[:, np.arange(4), np.arange(4)]):
        raise ValueError("TwoRoom success matrices do not match final distances")

    try:
        gate = json.loads(gate_file.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read TwoRoom use gate: {error}") from error
    expected_gate = {
        "schema_version": 1,
        "study": spec["study"],
        "lock": spec["_lock"],
        "status": "admitted",
        "admitted": True,
        "episodes": EPISODES,
        "oracle_success_min": spec["tworoom_use"]["oracle_success_min"],
        "oracle_per_class_success_min": spec["tworoom_use"][
            "oracle_per_class_success_min"],
        "off_diagonal_false_success_max": spec["tworoom_use"][
            "off_diagonal_false_success_max"],
        "replay_fidelity_min": spec["tworoom_use"]["replay_fidelity_min"],
        "vendor_commit": spec["tworoom_use"]["upstream_environment"][
            "revision"],
        "vendor_clean": True,
    }
    failed = [key for key, value in expected_gate.items()
              if gate.get(key) != value]
    if failed:
        raise ValueError(f"TwoRoom execution gate is not admitted: {failed}")
    rows = np.arange(EPISODES)
    oracle_success = success[rows, labels, labels].astype(np.float64)
    random_success = success[
        rows, random_choice, labels].astype(np.float64)
    oracle_per_class = np.asarray([
        oracle_success[labels == label].mean() for label in range(4)
    ], dtype=np.float64)
    off_diagonal = success[
        :, ~np.eye(4, dtype=np.bool_)].reshape(-1).astype(np.float64)
    off_diagonal_rate = float(off_diagonal.mean())
    _require_close(
        gate.get("oracle_executed_success"), float(oracle_success.mean()),
        "gate.oracle_executed_success", tolerance=1e-12)
    _require_close(
        gate.get("realized_random_executed_success"),
        float(random_success.mean()),
        "gate.realized_random_executed_success", tolerance=1e-12)
    reported_per_class = np.asarray(
        gate.get("oracle_per_class_executed_success"), dtype=np.float64)
    if reported_per_class.shape != (4,) or not np.allclose(
            reported_per_class, oracle_per_class, rtol=0, atol=1e-12) \
            or oracle_per_class.min() < float(
                spec["tworoom_use"]["oracle_per_class_success_min"]):
        raise ValueError("TwoRoom per-class oracle controller gate differs")
    _require_close(
        gate.get("off_diagonal_false_success"), off_diagonal_rate,
        "gate.off_diagonal_false_success", tolerance=1e-12)
    if off_diagonal_rate > float(
            spec["tworoom_use"]["off_diagonal_false_success_max"]):
        raise ValueError("TwoRoom off-diagonal false success exceeds its gate")
    _require_close(
        gate.get("controller_selected_target_success"),
        float(controller_target_success.mean()),
        "gate.controller_selected_target_success", tolerance=1e-12)
    if float(gate.get("reset_replay_fidelity", -1)) \
            < float(spec["tworoom_use"]["replay_fidelity_min"]):
        raise ValueError("TwoRoom execution replay fidelity is below its gate")
    _require_close(
        gate.get("reset_replay_fidelity"), float(reset_replay.mean()),
        "gate.reset_replay_fidelity", tolerance=1e-12)

    deck_record = gate.get("deck")
    deck_sidecar = deck_file.with_suffix(deck_file.suffix + ".json")
    expected_record = {
        "path": str(deck_file),
        "sha256": sha256_file(deck_file),
        "sidecar": str(deck_sidecar),
        "sidecar_sha256": sha256_file(deck_sidecar),
    }
    if not isinstance(deck_record, dict) or any(
            deck_record.get(key) != value
            for key, value in expected_record.items()):
        raise ValueError("TwoRoom gate does not hash the admitted deck")
    provenance = {
        "deck": {
            **_record(deck_file),
            "sidecar": _relative(deck_sidecar),
            "sidecar_sha256": sha256_file(deck_sidecar),
        },
        "gate": _record(gate_file),
    }
    return arrays, gate, provenance


def _validate_admissions(spec: Mapping[str, Any]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for host in HOSTS:
        path = host_manifest_path(dict(spec), host)
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read matched admission {path}: {error}") \
                from error
        if value.get("lock") != spec["_lock"] \
                or value.get("status") != "admitted" \
                or value.get("all_targets_ages_admitted") is not True \
                or value.get("frozen_host_unchanged") is not True:
            raise ValueError(f"matched host is not admitted: {host}")
        records[host] = _record(path)
    return records


def _validate_manifest(
        path: Path, metrics_path: Path, spec: Mapping[str, Any], arm: str,
        seed: int) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read use manifest {path}: {error}") from error
    expected = {
        "schema_version": 1,
        "study": spec["study"],
        "lock": spec["_lock"],
        "branch": BRANCH,
        "host": "tworoom",
        "arm": arm,
        "seed": seed,
        "physical_gpu": 0,
    }
    failed = [key for key, expected_value in expected.items()
              if value.get(key) != expected_value]
    artifacts = value.get("artifacts")
    if failed or not isinstance(artifacts, dict) \
            or set(artifacts) != {"metrics"}:
        raise ValueError(f"invalid use manifest {path}: {failed}")
    metric_record = artifacts["metrics"]
    if not isinstance(metric_record, dict) \
            or metric_record.get("path") != "metrics.json" \
            or metric_record.get("sha256") != sha256_file(metrics_path):
        raise ValueError(f"use manifest does not authenticate metrics: {path}")
    return value


def _validate_cell(
        path: Path, spec: Mapping[str, Any], arm: str, seed: int,
        deck: Mapping[str, np.ndarray], provenance: Mapping[str, Any],
        gate: Mapping[str, Any]) -> tuple[
            dict[str, np.ndarray], dict[str, Any], str]:
    metrics_path = path / "metrics.json"
    manifest_path = path / "manifest.json"
    if not metrics_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"incomplete TwoRoom use cell {path}")
    if {entry.name for entry in path.iterdir()} != {"metrics.json", "manifest.json"}:
        raise ValueError(f"unexpected artifacts in TwoRoom use cell {path}")
    _validate_manifest(manifest_path, metrics_path, spec, arm, seed)
    try:
        value = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read use metrics {metrics_path}: {error}") \
            from error
    expected = {
        "schema_version": 1,
        "study": spec["study"],
        "lock": spec["_lock"],
        "branch": BRANCH,
        "host": "tworoom",
        "target": "location",
        "cue_age": 15,
        "episodes": EPISODES,
        "arm": arm,
        "seed": seed,
        "device": "cuda:0",
        "physical_gpu": 0,
    }
    failed = [key for key, expected_value in expected.items()
              if value.get(key) != expected_value]
    if failed:
        raise ValueError(f"use cell identity differs {metrics_path}: {failed}")
    if value.get("arm_blind_consumer") is not True \
            or value.get("validation_labels_used_for_fitting") is not False \
            or value.get("carrier_state_unchanged") is not True \
            or value.get("frozen_host_instantiated") is not False:
        raise ValueError(f"consumer boundary differs in {metrics_path}")
    state_before = value.get("carrier_state_sha256_before")
    state_after = value.get("carrier_state_sha256_after")
    if not isinstance(state_before, str) or len(state_before) != 64 \
            or state_after != state_before:
        raise ValueError(f"carrier state changed in {metrics_path}")
    consumer_digest = value.get("consumer_state_sha256")
    if value.get("consumer_training_arms") != list(ARMS) \
            or value.get("consumer_rows") != 6000 \
            or value.get("consumer_seed") != seed \
            or not isinstance(consumer_digest, str) \
            or len(consumer_digest) != 64:
        raise ValueError(f"shared arm-blind consumer differs in {metrics_path}")
    consumer = value.get("consumer")
    if not isinstance(consumer, dict) \
            or consumer.get("parameter_sha256") != consumer_digest:
        raise ValueError(f"consumer digest receipt differs in {metrics_path}")
    expected_consumer = {
        "model": "StandardScaler+multinomial LogisticRegression",
        "arm_id_feature_present": False,
        "target": "location",
        "fit_episodes": len(ARMS) * 1200,
        "feature_dimension": 768,
        "classes": [0, 1, 2, 3],
        "logistic_c": spec["readout"]["logistic_c"],
        "solver": spec["readout"]["solver"],
        "max_iter": spec["readout"]["max_iter"],
        "random_state": spec["readout"]["random_state"],
        "training_arms": list(ARMS),
        "rows_per_arm": 1200,
        "equal_arm_weighting": True,
        "shared_across_evaluation_arms": True,
    }
    failed = [key for key, expected_value in expected_consumer.items()
              if consumer.get(key) != expected_value]
    if failed:
        raise ValueError(
            f"shared consumer schema differs in {metrics_path}: {failed}")
    endpoint = value.get("endpoint")
    expected_endpoint = {
        "decision_index": 19,
        "context_indices": [16, 17, 18],
        "prior_index": 19,
        "current_observation_excluded": True,
        "feature": "concat(z[16],z[17],z[18],prior_read[19])",
    }
    if not isinstance(endpoint, dict) or endpoint != expected_endpoint:
        raise ValueError(f"use endpoint schema differs in {metrics_path}")

    training_sources = consumer.get("training_sources")
    if not isinstance(training_sources, dict) \
            or set(training_sources) != set(ARMS):
        raise ValueError(f"consumer source grid differs in {metrics_path}")
    for training_arm in ARMS:
        source_directory = carrier_directory(
            dict(spec), "tworoom", training_arm, seed)
        source_metrics = source_directory / "metrics.json"
        expected_source = {
            "carrier_manifest": _record(source_directory / "manifest.json"),
            "carrier_checkpoint": _record(source_directory / "carrier.pt"),
            "carrier_metrics": _record(source_metrics),
            "state_unchanged": True,
        }
        source_record = training_sources[training_arm]
        if not isinstance(source_record, dict) or any(
                source_record.get(key) != expected_value
                for key, expected_value in expected_source.items()):
            raise ValueError(
                f"consumer source {training_arm} differs in {metrics_path}")
        try:
            training_metrics = json.loads(source_metrics.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot read pooled carrier metrics: {source_metrics}") from error
        source_state = source_record.get("carrier_state_sha256")
        if not isinstance(source_state, str) or len(source_state) != 64 \
                or source_state != training_metrics.get("carrier_state_sha256"):
            raise ValueError(
                f"consumer carrier state differs for {training_arm} in "
                f"{metrics_path}")

    # Every cell must refer to the exact admitted deck and gate.  The evaluator
    # may include additional carrier/readout lineage, but these two records are
    # mandatory and immutable.
    cell_provenance = value.get("provenance")
    if not isinstance(cell_provenance, dict):
        raise ValueError(f"missing use provenance in {metrics_path}")
    expected_provenance_keys = {
        "gate", "deck", "carrier_manifest", "carrier_checkpoint",
        "carrier_metrics", "tworoom_host_weights",
    }
    if set(cell_provenance) != expected_provenance_keys:
        raise ValueError(f"use provenance grid differs in {metrics_path}")
    for key in ("deck", "gate"):
        if cell_provenance.get(key) != provenance[key]:
            raise ValueError(f"{key} provenance differs in {metrics_path}")
    source_directory = carrier_directory(dict(spec), "tworoom", arm, seed)
    expected_sources = {
        "carrier_manifest": source_directory / "manifest.json",
        "carrier_checkpoint": source_directory / "carrier.pt",
        "carrier_metrics": source_directory / "metrics.json",
        "tworoom_host_weights": resolve_input_path(
            spec["inputs"]["tworoom"]["weights"]),
    }
    for name, source in expected_sources.items():
        record = cell_provenance.get(name)
        if record != _record(source):
            raise ValueError(f"changed or mispaired {name} in {metrics_path}")
    try:
        source_manifest = json.loads(
            expected_sources["carrier_manifest"].read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read source carrier manifest: {error}") from error
    expected_carrier_identity = {
        "schema_version": 1,
        "study": spec["study"],
        "lock": spec["_lock"],
        "host": "tworoom",
        "arm": arm,
        "seed": seed,
    }
    failed = [key for key, expected_value in expected_carrier_identity.items()
              if source_manifest.get(key) != expected_value]
    source_artifacts = source_manifest.get("artifacts", {})
    expected_artifacts = {
        "metrics": source_directory / "metrics.json",
        "checkpoint": source_directory / "carrier.pt",
        "history": source_directory / "history.csv",
    }
    if failed or set(source_artifacts) != set(expected_artifacts) or any(
            source_artifacts[name].get("sha256") != sha256_file(source)
            for name, source in expected_artifacts.items()):
        raise ValueError(f"source carrier manifest is invalid for {metrics_path}")

    arrays: dict[str, np.ndarray] = {}
    integer = {"labels", "joint_labels", "predictions", "goal_correct",
               "executed_success", "oracle_success", "random_success"}
    for key in CELL_ARRAYS:
        dtype = np.int64 if key in integer else np.float64
        raw = np.asarray(value.get(key))
        current = np.asarray(raw, dtype=dtype)
        if current.shape != (EPISODES,) or not np.isfinite(current).all():
            raise ValueError(f"invalid {key} in {metrics_path}")
        if key in integer and not np.array_equal(raw, current):
            raise ValueError(f"non-integer values in {key} in {metrics_path}")
        arrays[key] = current

    labels = np.asarray(deck["location_label"], dtype=np.int64)
    joint = np.asarray(deck["combination_label"], dtype=np.int64)
    prediction = arrays["predictions"]
    if not np.array_equal(arrays["labels"], labels) \
            or not np.array_equal(arrays["joint_labels"], joint) \
            or not np.isin(prediction, np.arange(4)).all():
        raise ValueError(f"episode pairing or predictions differ in {metrics_path}")
    rows = np.arange(EPISODES)
    random_choice = np.asarray(deck["random_choice"], dtype=np.int64)
    success = np.asarray(deck["success_matrix"], dtype=np.int64)
    distance = np.asarray(deck["distance_matrix"], dtype=np.float64)
    expected_arrays = {
        "goal_correct": (prediction == labels).astype(np.int64),
        "executed_success": success[rows, prediction, labels],
        "selected_distance": distance[rows, prediction, labels],
        "oracle_success": success[rows, labels, labels],
        "random_success": success[rows, random_choice, labels],
        "oracle_distance": distance[rows, labels, labels],
        "random_distance": distance[rows, random_choice, labels],
        "distance_regret": (
            distance[rows, prediction, labels]
            - distance[rows, labels, labels]),
    }
    for key in ("goal_correct", "executed_success", "oracle_success",
                "random_success"):
        _require_binary(arrays[key], f"{metrics_path}/{key}")
    for key, expected_values in expected_arrays.items():
        if key in integer:
            valid = np.array_equal(arrays[key], expected_values)
        else:
            valid = np.allclose(
                arrays[key], expected_values, rtol=1e-6, atol=1e-5)
        if not valid:
            raise ValueError(f"{key} is not deck-derived in {metrics_path}")

    scalar_expectations = {
        "goal_selection_accuracy": arrays["goal_correct"].mean(),
        "executed_success_rate": arrays["executed_success"].mean(),
        "mean_selected_distance": arrays["selected_distance"].mean(),
        "mean_distance_regret": arrays["distance_regret"].mean(),
        "oracle_executed_success_rate": arrays["oracle_success"].mean(),
        "random_executed_success_rate": arrays["random_success"].mean(),
        "mean_oracle_distance": arrays["oracle_distance"].mean(),
        "mean_random_distance": arrays["random_distance"].mean(),
    }
    for key, expected_value in scalar_expectations.items():
        _require_close(value.get(key), float(expected_value),
                       f"{metrics_path}/{key}")
    _require_close(
        value.get("goal_selection_balanced_accuracy"),
        float(arrays["goal_correct"].mean()),
        f"{metrics_path}/goal_selection_balanced_accuracy")
    _require_close(
        value.get("oracle_executed_success_rate"),
        float(gate["oracle_executed_success"]),
        f"{metrics_path}/oracle gate pairing", tolerance=1e-12)
    _require_close(
        value.get("random_executed_success_rate"),
        float(gate["realized_random_executed_success"]),
        f"{metrics_path}/random gate pairing", tolerance=1e-12)
    return arrays, {
        "metrics": _record(metrics_path),
        "manifest": _record(manifest_path),
    }, consumer_digest


def _actual_cell_directories(root: Path) -> set[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"missing TwoRoom use cell root {root}")
    arm_paths = list(root.iterdir())
    if {path.name for path in arm_paths} != set(ARMS) \
            or any(not path.is_dir() for path in arm_paths):
        raise ValueError("TwoRoom use arm-directory grid differs")
    result: set[Path] = set()
    expected_seed_names = {f"seed-{seed}" for seed in SEEDS}
    for arm_path in arm_paths:
        seed_paths = list(arm_path.iterdir())
        if {path.name for path in seed_paths} != expected_seed_names \
                or any(not path.is_dir() for path in seed_paths):
            raise ValueError(f"use seed-directory grid differs: {arm_path}")
        result.update(path.resolve() for path in seed_paths)
    return result


def _load_grid(spec: Mapping[str, Any]) -> tuple[
        dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    deck, gate, provenance = _validate_deck_and_gate(spec)
    admissions = _validate_admissions(spec)
    values = {
        key: np.empty((len(ARMS), len(SEEDS), EPISODES), dtype=np.float64)
        for key in ("goal_correct", "executed_success", "selected_distance",
                    "distance_regret")
    }
    expected_directories: set[Path] = set()
    artifacts: dict[str, Any] = {}
    consumer_digests: dict[int, str] = {}
    for arm_index, arm in enumerate(ARMS):
        for seed_index, seed in enumerate(SEEDS):
            directory = use_cell_directory(spec, arm, seed)
            expected_directories.add(directory.resolve())
            arrays, records, consumer_digest = _validate_cell(
                directory, spec, arm, seed, deck, provenance, gate)
            previous_digest = consumer_digests.setdefault(seed, consumer_digest)
            if previous_digest != consumer_digest:
                raise ValueError(
                    f"consumer differs across arms for carrier seed {seed}")
            for key in values:
                values[key][arm_index, seed_index] = arrays[key]
            artifacts[f"{arm}/seed-{seed}"] = records
    actual = _actual_cell_directories(output_path(spec, "use") / "cells")
    if actual != expected_directories:
        missing = sorted(str(path) for path in expected_directories - actual)
        extra = sorted(str(path) for path in actual - expected_directories)
        raise ValueError(
            f"TwoRoom use grid differs: missing={missing}, extra={extra}")

    labels = np.asarray(deck["combination_label"], dtype=np.int64)
    rows = np.arange(EPISODES)
    target = np.asarray(deck["location_label"], dtype=np.int64)
    random_choice = np.asarray(deck["random_choice"], dtype=np.int64)
    success = np.asarray(deck["success_matrix"], dtype=np.float64)
    distance = np.asarray(deck["distance_matrix"], dtype=np.float64)
    baselines = {
        "realized_random_goal_correct": (random_choice == target).astype(float),
        "realized_random_success": success[rows, random_choice, target],
        "realized_random_distance": distance[rows, random_choice, target],
        "realized_random_regret": (
            distance[rows, random_choice, target]
            - distance[rows, target, target]),
        "oracle_goal_correct": np.ones(EPISODES, dtype=float),
        "oracle_success": success[rows, target, target],
        "oracle_distance": distance[rows, target, target],
        "oracle_regret": np.zeros(EPISODES, dtype=float),
    }
    audit = {
        "admissions": admissions,
        "execution_deck": provenance,
        "complete_cells": len(artifacts),
        "expected_cells": len(ARMS) * len(SEEDS),
        "hashed_cell_payloads": len(artifacts),
        "hashed_cell_manifests": len(artifacts),
        "physical_gpu_counts": {"0": 25, "1": 0, "2": 0, "3": 0},
        "unexpected_cell_directories": 0,
        "cell_artifacts": artifacts,
        "shared_consumer_sha256_by_seed": {
            str(seed): consumer_digests[seed] for seed in SEEDS},
    }
    return values, labels, baselines, audit


def _stratified_episode_weights(
        joint: np.ndarray, draws: int,
        rng: np.random.Generator) -> np.ndarray:
    labels = np.asarray(joint, dtype=np.int64)
    if labels.shape != (EPISODES,) or not np.array_equal(
            np.bincount(labels, minlength=16), np.full(16, 30)):
        raise ValueError("use labels are not exactly balanced over 16 strata")
    weights = np.zeros((draws, EPISODES), dtype=np.float32)
    draw_rows = np.arange(draws)[:, None]
    for label in range(16):
        indices = np.flatnonzero(labels == label)
        sampled = indices[
            rng.integers(0, len(indices), size=(draws, len(indices)))]
        np.add.at(
            weights, (np.broadcast_to(draw_rows, sampled.shape), sampled), 1.0)
    weights /= float(EPISODES)
    return weights


def hierarchical_paired_bootstrap(
        values: Mapping[str, np.ndarray], joint: np.ndarray,
        baselines: Mapping[str, np.ndarray], *, draws: int,
        seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                            dict[str, float], dict[str, np.ndarray]]:
    """Bootstrap all endpoints with shared seed and episode resamples."""

    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(SEEDS), size=(draws, len(SEEDS)))
    seed_counts = np.zeros((draws, len(SEEDS)), dtype=np.float32)
    rows = np.arange(draws)[:, None]
    np.add.at(seed_counts, (
        np.broadcast_to(rows, selected.shape), selected), 1.0)
    seed_weights = seed_counts / float(len(SEEDS))
    episode_weights = _stratified_episode_weights(joint, draws, rng)

    points: dict[str, np.ndarray] = {}
    samples: dict[str, np.ndarray] = {}
    for key, raw in values.items():
        array = np.asarray(raw, dtype=np.float64)
        if array.shape != (len(ARMS), len(SEEDS), EPISODES) \
                or not np.isfinite(array).all():
            raise ValueError(f"invalid use grid for {key}")
        points[key] = array.mean(axis=(1, 2))
        episode_means = np.einsum(
            "ase,be->bas", array, episode_weights, optimize=True)
        samples[key] = np.einsum(
            "bas,bs->ba", episode_means, seed_weights, optimize=True)

    baseline_points: dict[str, float] = {}
    baseline_samples: dict[str, np.ndarray] = {}
    for key, raw in baselines.items():
        array = np.asarray(raw, dtype=np.float64)
        if array.shape != (EPISODES,) or not np.isfinite(array).all():
            raise ValueError(f"invalid use baseline for {key}")
        baseline_points[key] = float(array.mean())
        baseline_samples[key] = episode_weights @ array
    return points, samples, baseline_points, baseline_samples


def _stat(point: float, samples: np.ndarray,
          seed_values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(samples, dtype=np.float64)
    per_seed = np.asarray(seed_values, dtype=np.float64)
    return {
        "mean": float(point),
        "ci95": [float(value) for value in np.quantile(values, (.025, .975))],
        "seed_values": [float(value) for value in per_seed],
    }


def _endpoint_arrays(
        values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {endpoint: values[array_key]
            for endpoint, array_key in ARRAY_FOR_ENDPOINT.items()}


def _baseline_key(endpoint: str, baseline: str) -> str:
    suffix = {
        "goal_selection_accuracy": "goal_correct",
        "executed_success_rate": "success",
        "mean_selected_distance": "distance",
        "mean_distance_regret": "regret",
    }[endpoint]
    return f"{baseline}_{suffix}"


def _summarize(
        values: Mapping[str, np.ndarray], points: Mapping[str, np.ndarray],
        samples: Mapping[str, np.ndarray],
        baseline_points: Mapping[str, float],
        baseline_samples: Mapping[str, np.ndarray]) -> dict[str, Any]:
    endpoint_values = _endpoint_arrays(values)
    endpoint_points = _endpoint_arrays(points)
    endpoint_samples = _endpoint_arrays(samples)
    none = ARMS.index("none")
    fixed = ARMS.index("fixed_trust")
    ssm = ARMS.index("ssm")

    arms: dict[str, Any] = {}
    versus_none: dict[str, Any] = {}
    versus_random: dict[str, Any] = {}
    versus_oracle: dict[str, Any] = {}
    for arm_index, arm in enumerate(ARMS):
        arms[arm] = {}
        versus_random[arm] = {}
        versus_oracle[arm] = {}
        for endpoint in ENDPOINTS:
            raw = endpoint_values[endpoint]
            point = endpoint_points[endpoint]
            sample = endpoint_samples[endpoint]
            arms[arm][endpoint] = _stat(
                point[arm_index], sample[:, arm_index],
                raw[arm_index].mean(axis=1))
            random_key = _baseline_key(endpoint, "realized_random")
            oracle_key = _baseline_key(endpoint, "oracle")
            versus_random[arm][endpoint] = _stat(
                point[arm_index] - baseline_points[random_key],
                sample[:, arm_index] - baseline_samples[random_key],
                raw[arm_index].mean(axis=1)
                - baseline_points[random_key])
            versus_oracle[arm][endpoint] = _stat(
                point[arm_index] - baseline_points[oracle_key],
                sample[:, arm_index] - baseline_samples[oracle_key],
                raw[arm_index].mean(axis=1)
                - baseline_points[oracle_key])
        if arm != "none":
            versus_none[arm] = {}
            for endpoint in ENDPOINTS:
                raw = endpoint_values[endpoint]
                point = endpoint_points[endpoint]
                sample = endpoint_samples[endpoint]
                versus_none[arm][endpoint] = _stat(
                    point[arm_index] - point[none],
                    sample[:, arm_index] - sample[:, none],
                    (raw[arm_index] - raw[none]).mean(axis=1))

    baselines: dict[str, Any] = {}
    for baseline in ("realized_random", "oracle"):
        baselines[baseline] = {}
        for endpoint in ENDPOINTS:
            key = _baseline_key(endpoint, baseline)
            baselines[baseline][endpoint] = _stat(
                baseline_points[key], baseline_samples[key],
                np.full(len(SEEDS), baseline_points[key]))

    fixed_minus_ssm: dict[str, Any] = {}
    for endpoint in ENDPOINTS:
        raw = endpoint_values[endpoint]
        point = endpoint_points[endpoint]
        sample = endpoint_samples[endpoint]
        fixed_minus_ssm[endpoint] = _stat(
            point[fixed] - point[ssm],
            sample[:, fixed] - sample[:, ssm],
            (raw[fixed] - raw[ssm]).mean(axis=1))

    use_claims: dict[str, Any] = {}
    for arm in ARMS[1:]:
        none_ci = versus_none[arm]["executed_success_rate"]["ci95"]
        random_ci = versus_random[arm]["executed_success_rate"]["ci95"]
        use_claims[arm] = {
            "carrier_minus_none_executed_success_ci_lower_above_zero": bool(
                none_ci[0] > 0.0),
            "carrier_minus_realized_random_executed_success_ci_lower_above_zero": bool(
                random_ci[0] > 0.0),
            "resolved_external_use": bool(
                none_ci[0] > 0.0 and random_ci[0] > 0.0),
            "decision_rule": (
                "resolved iff the paired carrier-minus-none executed-success "
                "95% CI lower bound and the carrier-minus-realized-random "
                "executed-success 95% CI lower bound are both above zero"),
        }
    return {
        "arms": arms,
        "baselines": baselines,
        "paired_arm_minus_none": versus_none,
        "arm_minus_realized_random": versus_random,
        "arm_minus_oracle": versus_oracle,
        "fixed_trust_minus_ssm": fixed_minus_ssm,
        "external_use_claims": use_claims,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing TwoRoom use aggregation writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=False)
    destination = output_path(spec, "use") / "summary.json"
    audit_path = output_path(spec, "use") / "final_audit.json"
    if destination.exists() or audit_path.exists():
        raise FileExistsError("TwoRoom use aggregation is already closed")

    values, joint, baselines, loaded_audit = _load_grid(spec)
    draws = int(spec["statistics"]["bootstrap_draws"])
    seed = int(spec["statistics"]["bootstrap_seed"])
    points, samples, baseline_points, baseline_samples = \
        hierarchical_paired_bootstrap(
            values, joint, baselines, draws=draws, seed=seed)
    analysis = _summarize(
        values, points, samples, baseline_points, baseline_samples)
    summary = {
        "schema_version": 1,
        "study": spec["study"],
        "branch": BRANCH,
        "lock": spec["_lock"],
        "status": "complete",
        "host": "tworoom",
        "target": "location",
        "cue_age": 15,
        "episodes": EPISODES,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "carrier_seed_resampling": "joint across all arms",
            "episode_resampling": (
                "shared across all arms/seeds/endpoints and stratified by "
                "16-way color-location combination"),
            "paired": True,
            "interval": "percentile 95%",
        },
        "no_pooled_memory_score": True,
        **analysis,
        "claim_boundary": spec["tworoom_use"]["claim_boundary"],
    }
    summary_hash = atomic_text(destination, stable_json(summary))
    audit = {
        "schema_version": 1,
        "study": spec["study"],
        "branch": BRANCH,
        "lock": spec["_lock"],
        "status": "complete",
        **loaded_audit,
        "summary": {"path": _relative(destination), "sha256": summary_hash},
        "bootstrap_draws": draws,
        "no_pooled_memory_score": True,
        "cuda3_used": False,
    }
    atomic_text(audit_path, stable_json(audit))
    print(f"[matched-host/use] wrote {destination} and {audit_path}", flush=True)


if __name__ == "__main__":
    main()
