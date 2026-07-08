#!/usr/bin/env python3
"""Independent post-lock statistics audit for Paper-A Waves 2 and 3.

This program is intentionally separate from the sealed producers.  It imports
none of their statistic or bootstrap helpers and consumes preserved
predictions/features, the independently receipted execution deck, immutable
configuration/lock metadata, and completed summaries.  It never writes inside
an experiment root.  A receipt is printed by default; ``--execute`` may create
exactly one receipt outside both experiment roots.

While either formal run is incomplete, the auditor reports that state without
opening a validation-prediction artifact or computing a statistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import warnings
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
WAVE2_CONFIG = Path("configs/dinowm_wave2_spatial_carrier_v1_1.yaml")
WAVE3_CONFIG = Path("configs/dinowm_pointmaze_wave3.yaml")
DEFAULT_RECEIPT = Path(
    "outputs/paper_a_statistics_independent/receipt.json")

# These identities were copied into this independent auditor after both
# protocols were sealed.  Checking only ``config == lock[protocol_sha256]``
# would let a coordinated edit of the config and lock redefine the experiment.
SEALED_IDENTITIES = {
    WAVE2_CONFIG: {
        "protocol_sha256":
            "b1af10f4bc243b9c22aee29e7f2c420905c3f4f38e45c6ea4d9457f819205178",
        "lock_sha256":
            "a3c030b54cdfe81dbd5379e58295f770f9033707973ae147eebaef5411412a79",
        "lock_schema": "dinowm_wave2_spatial_carrier_lock_v1",
        "grid": {"tasks": 2, "arms": 5, "seeds": 5, "cells": 50},
        "locked_flag": "locked_before_formal_metrics",
        "preoutcome_sha256": {
            "outputs/dinowm_wave2_spatial_carrier_v1_1/formal/admissions.json":
                "f5c71f5538fe8369aa5c6f88a92936440fff536091484cdd71fee75632f0c658",
            "outputs/dinowm_wave2_spatial_carrier_v1_1/cache/manifest.json":
                "b22274a1fe7b5be0acf8f8fa60a78a244d552f8d163fab4680edd3f428d381de",
        },
    },
    WAVE3_CONFIG: {
        "protocol_sha256":
            "bb070fa13892d5b0ab7f84efc02ccd3bdbc9bdff0f1000b3b160d60e68bef3a4",
        "lock_sha256":
            "d8998ce0cae30751f0d65b268863c332838e89fbfd19484858474d3fba2b560d",
        "lock_schema": "dinowm_pointmaze_wave3_lock_v1",
        "grid": {"tasks": 1, "arms": 5, "seeds": 5, "cells": 25},
        "locked_flag": "locked_before_semantic_metrics",
        "preoutcome_sha256": {
            "outputs/dinowm_pointmaze_wave3/formal/admission.json":
                "9af1395fdb0ca9ba8ebc24378c0f2024d60a79410c714b9ee4679e2a17c79fbe",
            "outputs/dinowm_pointmaze_wave3/formal/controller_gate.json":
                "47957f3dd8f67b66cd489dea925b661177d8fbf2f8788e3d4f53f4cc93c129ec",
            "outputs/dinowm_pointmaze_wave3/cache/manifest.json":
                "b613d7667e6bb57557bcc5820538b5723628132b183d3edc3b23743da2fd340a",
            "outputs/dinowm_pointmaze_wave3/cache/selection.json":
                "679b69f98fefa10b109783db3372aca528023fd66292f9530bfa0e2c82e0139d",
            "outputs/dinowm_pointmaze_wave3/cache/metadata.npz":
                "df666a9adc7216c56ce8cfdd150d27e55b14b0077469a55e2a4674a38d6f3a1f",
            "outputs/dinowm_pointmaze_wave3/cache/execution_deck.npz":
                "48242d4b8de22a4289c51da5c7fb0f041d836f2402694f06e3b6254e42c975ae",
        },
    },
}

ARM_PARAMETERS = {
    "none": 0,
    "gru": 298_368,
    "lstm": 299_632,
    "ssm": 299_520,
    "fixed_trust": 299_520,
}

# Independently frozen constants for D=384 and action dimension 10.  This is
# deliberately not obtained from ``parameter_report`` in the producer module.
PARAMETER_MATCHING = {
    "action_dim": 10,
    "arms": {
        "acgru": {
            "delta": -1152,
            "parameters": 298368,
            "relative_mismatch": 0.0038461538461538464,
            "width": 148,
            "width_name": "hidden_dim",
        },
        "aclstm": {
            "delta": 112,
            "parameters": 299632,
            "relative_mismatch": 0.00037393162393162396,
            "width": 122,
            "width_name": "hidden_dim",
        },
        "diag_ssm": {
            "delta": 0,
            "parameters": 299520,
            "relative_mismatch": 0.0,
            "width": 384,
            "width_name": "width",
        },
        "lkc_fixed_trust": {
            "delta": 0,
            "parameters": 299520,
            "relative_mismatch": 0.0,
            "width": 384,
            "width_name": "state_dim",
        },
        "none": {
            "delta": -299520,
            "parameters": 0,
            "relative_mismatch": 1.0,
            "width": None,
            "width_name": None,
        },
    },
    "embed_dim": 384,
    "target_parameters": 299520,
}


class AuditFailure(RuntimeError):
    """A completed artifact or reported statistic failed verification."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def stable_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"file changed while hashing: {path}")
    return digest.hexdigest()


def repository_path(root: Path, value: str | Path) -> Path:
    base = root.resolve()
    candidate = Path(value)
    result = candidate.resolve() if candidate.is_absolute() \
        else (base / candidate).resolve()
    try:
        result.relative_to(base)
    except ValueError as error:
        raise AuditFailure(f"path leaves repository: {value}") from error
    return result


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AuditFailure(f"invalid {label}: {path}: {error}") from error
    require(isinstance(value, dict), f"{label} is not a mapping: {path}")
    return value


def read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise AuditFailure(f"invalid {label}: {path}: {error}") from error
    require(isinstance(value, dict), f"{label} is not a mapping: {path}")
    return value


def compare_exact(actual: Any, expected: Any, path: str = "root") -> None:
    """Recursively reject any numeric, metadata, shape, or key mismatch."""

    if isinstance(expected, Mapping):
        require(isinstance(actual, Mapping), f"{path}: expected mapping")
        require(set(actual) == set(expected),
                f"{path}: keys differ: actual={sorted(actual)} "
                f"expected={sorted(expected)}")
        for key in expected:
            compare_exact(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        require(isinstance(actual, list), f"{path}: expected list")
        require(len(actual) == len(expected), f"{path}: length differs")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_exact(left, right, f"{path}[{index}]")
        return
    # bool is a subclass of int; require the exact JSON scalar kind.
    require(type(actual) is type(expected),
            f"{path}: type differs: {type(actual).__name__} != "
            f"{type(expected).__name__}")
    if isinstance(expected, float):
        require(math.isfinite(actual) and actual == expected,
                f"{path}: value differs: {actual!r} != {expected!r}")
    else:
        require(actual == expected,
                f"{path}: value differs: {actual!r} != {expected!r}")


def verify_manifest_artifact(path: Path, record: Mapping[str, Any],
                             label: str, *, require_path: bool = False) -> str:
    require(isinstance(record, Mapping), f"{label}: record is not a mapping")
    allowed = ({"size", "sha256"}, {"path", "size", "sha256"})
    require(set(record) in allowed,
            f"{label}: artifact metadata keys differ")
    require(not require_path or "path" in record,
            f"{label}: artifact path is missing")
    if "path" in record:
        require(Path(str(record["path"])).name == path.name,
                f"{label}: manifest basename differs")
    require(isinstance(record["size"], int)
            and not isinstance(record["size"], bool)
            and record["size"] >= 0,
            f"{label}: invalid artifact size")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"]))),
            f"{label}: invalid SHA-256")
    require(path.is_file(), f"{label}: missing artifact: {path}")
    require(path.stat().st_size == record["size"],
            f"{label}: artifact size differs")
    digest = sha256_file(path)
    require(digest == record["sha256"], f"{label}: SHA-256 differs")
    return digest


def verify_reference(root: Path, record: Mapping[str, Any], label: str,
                     *, expected_path: Path | None = None) -> tuple[Path, str]:
    """Verify a root-relative referenced artifact with optional size."""

    require(isinstance(record, Mapping) and "path" in record
            and "sha256" in record, f"{label}: reference is incomplete")
    path = repository_path(root, record["path"])
    if expected_path is not None:
        require(path == expected_path.resolve(),
                f"{label}: referenced path differs")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"]))),
            f"{label}: invalid SHA-256")
    if "size" in record:
        require(isinstance(record["size"], int)
                and not isinstance(record["size"], bool)
                and path.is_file() and path.stat().st_size == record["size"],
                f"{label}: referenced size differs")
    digest = sha256_file(path)
    require(digest == record["sha256"], f"{label}: referenced SHA-256 differs")
    return path, digest


def require_integer_labels(values: np.ndarray, classes: int, label: str,
                           *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    raw = np.asarray(values)
    require(np.issubdtype(raw.dtype, np.integer),
            f"{label}: labels are not stored as integers")
    if shape is not None:
        require(raw.shape == shape, f"{label}: shape differs")
    result = raw.astype(np.int64, copy=False)
    require(result.size > 0
            and np.all((result >= 0) & (result < classes)),
            f"{label}: contains an undeclared class")
    return result


def require_numeric_vector(values: np.ndarray, length: int,
                           label: str) -> np.ndarray:
    raw = np.asarray(values)
    require(np.issubdtype(raw.dtype, np.number) and raw.shape == (length,),
            f"{label}: expected a numeric vector of length {length}")
    result = raw.astype(np.float64, copy=False)
    require(np.isfinite(result).all(), f"{label}: contains non-finite values")
    return result


def classification_record(prediction: np.ndarray, truth: np.ndarray,
                          classes: int) -> dict[str, Any]:
    pred = require_integer_labels(
        prediction, classes, "classification prediction")
    target = require_integer_labels(
        truth, classes, "classification truth", shape=pred.shape)
    require(set(np.unique(target)) == set(range(classes)),
            "classification truth does not contain every class")
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (target, pred), 1)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    return {
        "balanced_accuracy": float(np.mean(recall)),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": matrix.tolist(),
        "count": int(len(target)),
    }


def class_balanced_accuracy(prediction: np.ndarray, truth: np.ndarray,
                            classes: int) -> float:
    """Independent equal-class accuracy with strict label coverage."""

    raw_pred = np.asarray(prediction)
    raw_target = np.asarray(truth)
    require(raw_pred.ndim == raw_target.ndim == 1
            and raw_pred.shape == raw_target.shape,
            "balanced accuracy arrays are not aligned vectors")
    pred = require_integer_labels(
        raw_pred, classes, "balanced-accuracy prediction")
    target = require_integer_labels(
        raw_target, classes, "balanced-accuracy truth", shape=pred.shape)
    require(classes >= 2 and set(np.unique(target)) == set(range(classes)),
            "truth does not contain every declared class")
    return float(np.mean([
        np.mean(pred[target == label] == label) for label in range(classes)
    ]))


def _validate_prediction_matrix(values: np.ndarray, truth: np.ndarray,
                                classes: int) -> tuple[np.ndarray, np.ndarray]:
    raw_matrix = np.asarray(values)
    raw_target = np.asarray(truth)
    require(raw_matrix.ndim == 2 and raw_target.ndim == 1
            and raw_matrix.shape[1] == len(raw_target)
            and raw_matrix.shape[0] >= 1,
            "prediction matrix is not (seed, episode)")
    matrix = require_integer_labels(
        raw_matrix, classes, "prediction matrix")
    target = require_integer_labels(
        raw_target, classes, "prediction truth",
        shape=(raw_matrix.shape[1],))
    require(set(np.unique(target)) == set(range(classes)),
            "truth does not contain every class")
    return matrix, target


def stratified_paired_bootstrap(
        left_prediction: np.ndarray, right_prediction: np.ndarray,
        truth: np.ndarray, *, classes: int, draws: int, seed: int,
        confidence: float = 0.95) -> dict[str, Any]:
    """Independent matched-seed × class-stratified episode bootstrap.

    The 128-draw batching is part of the sealed Wave-2 random-number schedule:
    seed indices are drawn before class indices inside each batch.
    """

    left, target = _validate_prediction_matrix(
        left_prediction, truth, classes)
    right, right_target = _validate_prediction_matrix(
        right_prediction, truth, classes)
    require(np.array_equal(target, right_target) and left.shape == right.shape,
            "paired prediction matrices are not aligned")
    require(isinstance(draws, int) and not isinstance(draws, bool)
            and draws >= 1, "draws must be a positive integer")
    require(0.0 < confidence < 1.0, "confidence must lie in (0,1)")

    class_rows = [np.flatnonzero(target == label)
                  for label in range(classes)]
    require(all(len(rows) > 0 for rows in class_rows),
            "empty class in bootstrap")
    left_correct = left == target[None]
    right_correct = right == target[None]
    per_seed_left = np.stack([
        left_correct[:, rows].mean(axis=1) for rows in class_rows
    ], axis=1).mean(axis=1)
    per_seed_right = np.stack([
        right_correct[:, rows].mean(axis=1) for rows in class_rows
    ], axis=1).mean(axis=1)
    point = float(np.mean(per_seed_left - per_seed_right))

    rng = np.random.default_rng(int(seed))
    samples = np.empty(draws, dtype=np.float64)
    cursor = 0
    seed_count = left.shape[0]
    while cursor < draws:
        stop = min(draws, cursor + 128)
        count = stop - cursor
        sampled_seeds = rng.integers(
            0, seed_count, size=(count, seed_count))
        left_selected = left_correct[sampled_seeds]
        right_selected = right_correct[sampled_seeds]
        left_class: list[np.ndarray] = []
        right_class: list[np.ndarray] = []
        for rows in class_rows:
            positions = rng.integers(0, len(rows),
                                     size=(count, len(rows)))
            episodes = rows[positions]
            episodes = np.broadcast_to(
                episodes[:, None, :],
                (count, seed_count, len(rows)))
            left_class.append(np.take_along_axis(
                left_selected, episodes, axis=2).mean(axis=(1, 2)))
            right_class.append(np.take_along_axis(
                right_selected, episodes, axis=2).mean(axis=(1, 2)))
        samples[cursor:stop] = (
            np.stack(left_class, axis=1).mean(axis=1)
            - np.stack(right_class, axis=1).mean(axis=1))
        cursor = stop
    alpha = (1.0 - confidence) / 2.0
    interval = np.quantile(samples, (alpha, 1.0 - alpha))
    return {
        "mean": point,
        "ci95": [float(interval[0]), float(interval[1])],
        "draws": draws,
        "seed": int(seed),
        "confidence": float(confidence),
        "paired": True,
        "units": ["matched carrier seed",
                  "class-stratified held-out episode"],
        "ci_excludes_zero": bool(interval[0] > 0 or interval[1] < 0),
    }


def stratified_absolute_bootstrap(
        prediction: np.ndarray, truth: np.ndarray, *, classes: int,
        draws: int, seed: int, confidence: float = 0.95) -> dict[str, Any]:
    matrix, _ = _validate_prediction_matrix(prediction, truth, classes)
    impossible = np.full_like(matrix, -1)
    # The paired implementation rejects undeclared classes, so construct the
    # absolute statistic directly with an all-false right correctness matrix.
    target = np.asarray(truth, dtype=np.int64)
    class_rows = [np.flatnonzero(target == label)
                  for label in range(classes)]
    left_correct = matrix == target[None]
    right_correct = impossible == target[None]
    per_seed = np.stack([
        left_correct[:, rows].mean(axis=1) for rows in class_rows
    ], axis=1).mean(axis=1)
    point = float(np.mean(per_seed))
    rng = np.random.default_rng(int(seed))
    samples = np.empty(draws, dtype=np.float64)
    cursor = 0
    seed_count = matrix.shape[0]
    while cursor < draws:
        stop = min(draws, cursor + 128)
        count = stop - cursor
        sampled_seeds = rng.integers(
            0, seed_count, size=(count, seed_count))
        selected = left_correct[sampled_seeds]
        selected_right = right_correct[sampled_seeds]
        left_class: list[np.ndarray] = []
        right_class: list[np.ndarray] = []
        for rows in class_rows:
            positions = rng.integers(0, len(rows),
                                     size=(count, len(rows)))
            episodes = np.broadcast_to(
                rows[positions][:, None, :],
                (count, seed_count, len(rows)))
            left_class.append(np.take_along_axis(
                selected, episodes, axis=2).mean(axis=(1, 2)))
            right_class.append(np.take_along_axis(
                selected_right, episodes, axis=2).mean(axis=(1, 2)))
        samples[cursor:stop] = (
            np.stack(left_class, axis=1).mean(axis=1)
            - np.stack(right_class, axis=1).mean(axis=1))
        cursor = stop
    alpha = (1.0 - confidence) / 2.0
    interval = np.quantile(samples, (alpha, 1.0 - alpha))
    return {
        "mean": point,
        "ci95": [float(interval[0]), float(interval[1])],
        "draws": draws,
        "seed": int(seed),
        "confidence": float(confidence),
        "paired": True,
        "units": ["matched carrier seed",
                  "class-stratified held-out episode"],
        "ci_excludes_zero": bool(interval[0] > 0 or interval[1] < 0),
        "metric": "balanced_accuracy",
    }


def native_episode_bootstrap(
        values: np.ndarray, episodes: np.ndarray, *, draws: int, seed: int,
        confidence: float = 0.95) -> dict[str, Any]:
    """Independent crossed seed × equal-native-episode bootstrap.

    Wave 3's sealed schedule draws seed and episode indices in 512-draw
    batches.  Counterfactual label rows remain clustered within episode.
    """

    raw_values = np.asarray(values)
    raw_cluster = np.asarray(episodes)
    require(np.issubdtype(raw_values.dtype, np.number),
            "cluster values are not numeric")
    require(np.issubdtype(raw_cluster.dtype, np.integer)
            and raw_cluster.ndim == 1,
            "native-episode cluster IDs are not stored as integers")
    matrix = raw_values.astype(np.float64, copy=False)
    cluster = raw_cluster.astype(np.int64, copy=False)
    require(matrix.ndim == 2 and matrix.shape[1] == len(cluster),
            "cluster values are not (seed, expanded-example)")
    require(np.isfinite(matrix).all(), "cluster values are non-finite")
    require(isinstance(draws, int) and not isinstance(draws, bool)
            and draws >= 1, "draws must be a positive integer")
    require(0.0 < confidence < 1.0, "confidence must lie in (0,1)")
    unique = np.unique(cluster)
    require(len(unique) >= 2, "cluster bootstrap needs at least two episodes")
    per_episode = np.stack([
        matrix[:, cluster == episode].mean(axis=1) for episode in unique
    ], axis=1)
    point = float(per_episode.mean())

    rng = np.random.default_rng(int(seed))
    samples = np.empty(draws, dtype=np.float64)
    cursor = 0
    while cursor < draws:
        stop = min(draws, cursor + 512)
        count = stop - cursor
        seed_rows = rng.integers(
            0, matrix.shape[0], size=(count, matrix.shape[0]))
        episode_rows = rng.integers(
            0, len(unique), size=(count, len(unique)))
        selected = per_episode[
            seed_rows[:, :, None], episode_rows[:, None, :]]
        samples[cursor:stop] = selected.mean(axis=(1, 2))
        cursor = stop
    alpha = (1.0 - confidence) / 2.0
    interval = np.quantile(samples, (alpha, 1.0 - alpha))
    return {
        "mean": point,
        "ci95": [float(interval[0]), float(interval[1])],
        "draws": draws,
        "seed": int(seed),
        "confidence": float(confidence),
        "paired": True,
        "equal_native_episode_weight": True,
        "native_episode_clusters": int(len(unique)),
        "carrier_seeds": int(matrix.shape[0]),
        "ci_excludes_zero": bool(interval[0] > 0 or interval[1] < 0),
    }


def correctness(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    raw_matrix = np.asarray(prediction)
    raw_target = np.asarray(truth)
    require(raw_matrix.ndim == 2 and raw_target.ndim == 1
            and raw_matrix.shape[1] == len(raw_target),
            "prediction matrix is not aligned")
    matrix = require_integer_labels(
        raw_matrix, 4, "correctness prediction")
    target = require_integer_labels(
        raw_target, 4, "correctness truth", shape=(raw_matrix.shape[1],))
    return (matrix == target[None]).astype(np.float64)


def executed_success(success_matrix: np.ndarray, prediction: np.ndarray,
                     truth: np.ndarray) -> np.ndarray:
    """Independently score selected goal against each true goal."""

    matrix = np.asarray(success_matrix)
    raw_pred = np.asarray(prediction)
    raw_target = np.asarray(truth)
    require(matrix.ndim == 3 and matrix.shape[1:] == (4, 4),
            "success matrix must be (native_episode, selected, true)")
    require(np.issubdtype(matrix.dtype, np.integer)
            or np.issubdtype(matrix.dtype, np.bool_),
            "success matrix is not stored as integer/bool")
    require(np.all((matrix == 0) | (matrix == 1)),
            "success matrix is not binary")
    require(raw_pred.ndim == 2 and raw_target.ndim == 1
            and raw_pred.shape[1] == len(raw_target),
            "execution predictions are not (seed, expanded-example)")
    pred = require_integer_labels(
        raw_pred, 4, "execution prediction")
    target = require_integer_labels(
        raw_target, 4, "execution truth", shape=(raw_pred.shape[1],))
    require(len(target) == matrix.shape[0] * 4,
            "execution truth does not contain four rows per native episode")
    require(np.array_equal(target, np.tile(np.arange(4), matrix.shape[0])),
            "execution truth is not base-major labels 0..3")
    base = np.arange(len(target), dtype=np.int64) // 4
    rows = np.arange(len(target), dtype=np.int64)
    output = np.stack([
        matrix[base, pred[index], target] for index in range(pred.shape[0])
    ])
    require(output.shape == pred.shape, "execution result shape differs")
    return output.astype(np.float64)


def _validate_locked_sources(root: Path, lock: Mapping[str, Any],
                             label: str) -> int:
    sources = lock.get("source_sha256")
    require(isinstance(sources, Mapping) and len(sources) >= 1,
            f"{label}: locked source ledger is missing")
    for relative, expected in sources.items():
        require(isinstance(relative, str)
                and bool(re.fullmatch(r"[0-9a-f]{64}", str(expected))),
                f"{label}: invalid locked source record: {relative!r}")
        path = repository_path(root, relative)
        require(sha256_file(path) == expected,
                f"{label}: locked source changed: {relative}")
    return len(sources)


def _validate_preoutcome_identities(root: Path, relative: Path,
                                    label: str) -> int:
    records = SEALED_IDENTITIES[relative]["preoutcome_sha256"]
    for artifact, expected in records.items():
        require(sha256_file(repository_path(root, artifact)) == expected,
                f"{label}: pre-outcome artifact changed: {artifact}")
    return len(records)


def _validate_provenance(provenance: Mapping[str, Any],
                         lock: Mapping[str, Any], *, label: str,
                         schema: str, gpu: int, paper_flag: str) -> None:
    require(provenance.get("schema") == schema
            and provenance.get("status") == "complete",
            f"{label}: formal provenance is not complete")
    require(provenance.get("protocol_sha256") == lock["protocol_sha256"],
            f"{label}: provenance protocol differs")
    compare_exact(provenance.get("source_sha256"), lock["source_sha256"],
                  f"{label}.provenance.source_sha256")
    require(provenance.get("physical_gpu") == gpu
            and provenance.get("cuda_visible_devices") == str(gpu),
            f"{label}: GPU provenance differs")
    require(provenance.get(paper_flag) is False,
            f"{label}: provenance reports a paper edit")
    before = provenance.get("runtime_host_digest")
    after = provenance.get("runtime_host_digest_after")
    require(isinstance(before, str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", before))
            and before == after,
            f"{label}: frozen host digest changed")


def _validate_progress(progress: Mapping[str, Any], *, label: str,
                       protocol_sha256: str,
                       completed: Sequence[Sequence[Any]]) -> None:
    require(progress.get("protocol_sha256") == protocol_sha256,
            f"{label}: progress protocol differs")
    require(progress.get("count") == progress.get("expected")
            == len(completed), f"{label}: completed cell count differs")
    require(progress.get("completed_cells") == [list(row) for row in completed],
            f"{label}: completed-cell ledger differs")


def _all_declared_passes(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "pass" in value and value["pass"] is not True:
            return False
        return all(_all_declared_passes(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_declared_passes(item) for item in value)
    return True


def _validate_wave2_gates(root: Path, output: Path, formal: Path,
                          cfg: Mapping[str, Any], lock: Mapping[str, Any]) \
        -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_preoutcome_identities(root, WAVE2_CONFIG, "Wave 2")
    admissions = read_json(formal / "admissions.json", "Wave 2 admissions")
    provenance = read_json(formal / "provenance.json", "Wave 2 provenance")
    _validate_provenance(
        provenance, lock, label="Wave 2",
        schema="dinowm_wave2_spatial_provenance_v1", gpu=1,
        paper_flag="paper_modified_by_wave2")
    compare_exact(provenance.get("admissions"), admissions,
                  "Wave 2 provenance admissions")
    task_keys = [str(task["key"]) for task in cfg["tasks"]]
    require(isinstance(admissions.get("tasks"), Mapping)
            and set(admissions["tasks"]) == set(task_keys),
            "Wave 2 admission task set differs")
    require(all(admissions["tasks"][key].get("admitted") is True
                for key in task_keys)
            and admissions.get("rollout_health", {}).get("admitted") is True
            and _all_declared_passes(admissions),
            "Wave 2 prerequisite admission failed")
    for key in task_keys:
        record = admissions["tasks"][key]
        verify_reference(root, record, f"Wave 2 admission/{key}")
    verify_reference(root, admissions["rollout_health"],
                     "Wave 2 rollout-health admission")
    prior_path = "outputs/dinowm_native_pusht_audit_v2r2/formal/verification.json"
    require(admissions.get("prior_verification_sha256")
            == lock["source_sha256"].get(prior_path),
            "Wave 2 prior verification identity differs")

    cache = read_json(output / "cache/manifest.json", "Wave 2 cache manifest")
    require(cache.get("schema") ==
            "dinowm_wave2_full_patch_cache_v1_1_hardlink"
            and cache.get("protocol_sha256") == lock["protocol_sha256"],
            "Wave 2 cache lock differs")
    amendment = cache.get("amendment", {})
    require(cache.get("carrier_outcomes_computed_before_gate") is False
            and amendment.get("carrier_outcomes_seen") is False
            and amendment.get("exact_original_layout_gate", {}).get("pass")
            is True
            and amendment.get("reusable_cache_layout_gate", {}).get("pass")
            is True
            and _all_declared_passes(amendment),
            "Wave 2 pre-outcome numerical amendment failed")
    exact_value = amendment["exact_original_layout_gate"].get("value")
    reusable_value = amendment["reusable_cache_layout_gate"].get("value")
    require(isinstance(exact_value, (int, float))
            and not isinstance(exact_value, bool)
            and isinstance(reusable_value, (int, float))
            and not isinstance(reusable_value, bool)
            and math.isfinite(float(exact_value))
            and math.isfinite(float(reusable_value))
            and exact_value
            <= float(cfg["cache"]["exact_teacher_replay_threshold"])
            and reusable_value
            <= float(cfg["cache"]["reusable_cache_layout_threshold"]),
            "Wave 2 numerical amendment threshold differs")
    require(not (formal / "stop_receipt.json").exists(),
            "Wave 2 formal stop receipt exists")
    return admissions, provenance


def _load_lock(root: Path, relative: Path, label: str) \
        -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    config_path = root / relative
    lock_path = config_path.with_suffix(".lock.json")
    cfg = read_yaml(config_path, f"{label} config")
    lock = read_json(lock_path, f"{label} lock")
    identity = SEALED_IDENTITIES[relative]
    require(sha256_file(config_path) == lock.get("protocol_sha256")
            == identity["protocol_sha256"],
            f"{label} config hash differs from lock")
    require(sha256_file(lock_path) == identity["lock_sha256"],
            f"{label} lock identity differs")
    require(lock.get("schema") == identity["lock_schema"]
            and lock.get(identity["locked_flag"]) is True,
            f"{label} lock metadata differs")
    compare_exact(lock.get("grid"), identity["grid"], f"{label}.lock.grid")
    compare_exact(lock.get("parameter_matching"), PARAMETER_MATCHING,
                  f"{label}.lock.parameter_matching")
    require(cfg.get("protocol_status") == "locked_before_formal_metrics"
            and cfg.get("artifacts", {}).get("paper_edits") is False,
            f"{label} protocol status differs")
    _validate_locked_sources(root, lock, label)
    output = repository_path(root, cfg["artifacts"]["root"])
    formal = output / cfg["artifacts"]["formal"]
    return cfg, lock, output, formal


def completion_preflight(root: Path) -> dict[str, Any]:
    """Return incomplete before any prediction file can be opened."""

    root = root.resolve()
    progress: dict[str, Any] = {}
    missing: list[str] = []
    specs = (
        ("wave2", WAVE2_CONFIG, ("summary.json",), 50),
        ("wave3", WAVE3_CONFIG,
         ("carrier_summary.json", "external_use_summary.json", "summary.json"),
         25),
    )
    for name, relative, summaries, expected in specs:
        cfg = read_yaml(root / relative, f"{name} config")
        output = repository_path(root, cfg["artifacts"]["root"])
        formal = output / cfg["artifacts"]["formal"]
        progress_path = formal / "progress.json"
        if progress_path.is_file():
            value = read_json(progress_path, f"{name} progress")
            progress[name] = {
                "count": value.get("count"),
                "expected": value.get("expected", expected),
            }
        else:
            progress[name] = {"count": 0, "expected": expected}
        for summary in summaries:
            path = formal / summary
            if not path.is_file():
                missing.append(str(path.relative_to(root)))
    if missing:
        return {
            "schema": "paper_a_statistics_independent_receipt_v1",
            "status": "incomplete",
            "read_only": True,
            "statistics_computed": False,
            "missing_completed_summaries": sorted(missing),
            "progress": progress,
        }
    for name, record in progress.items():
        require(record["count"] == record["expected"],
                f"{name} has summaries but incomplete progress: "
                f"{record['count']}/{record['expected']}")
    return {
        "schema": "paper_a_statistics_independent_receipt_v1",
        "status": "ready",
        "read_only": True,
        "statistics_computed": False,
        "progress": progress,
    }


def _load_cell_arrays(
        directory: Path, manifest: Mapping[str, Any], label: str, *,
        expected_artifacts: set[str]) \
        -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, str]]:
    records = manifest.get("artifacts")
    require(isinstance(records, Mapping), f"{label}: missing artifact ledger")
    require(set(records) == expected_artifacts,
            f"{label}: cell artifact set differs")
    digests = {
        name: verify_manifest_artifact(
            directory / name, records[name], f"{label}/{name}")
        for name in sorted(expected_artifacts)
    }
    metrics = read_json(directory / "metrics.json", f"{label} metrics")
    try:
        with np.load(directory / "validation_predictions.npz",
                     allow_pickle=False) as values:
            arrays = {name: values[name].copy() for name in values.files}
    except (OSError, ValueError, KeyError) as error:
        raise AuditFailure(f"{label}: invalid prediction artifact: {error}") \
            from error
    return metrics, arrays, digests


def _require_digest(value: Any, label: str) -> str:
    require(isinstance(value, str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", value)),
            f"{label}: invalid SHA-256")
    return value


def _validate_common_cell(
        manifest: Mapping[str, Any], metrics: Mapping[str, Any], *,
        label: str, protocol_sha256: str, manifest_schema: str,
        metrics_schema: str, task: str, arm: str, seed: int,
        physical_gpu: int) -> None:
    require(manifest.get("schema") == manifest_schema
            and manifest.get("protocol_sha256") == protocol_sha256
            and manifest.get("arm") == arm
            and manifest.get("seed") == seed,
            f"{label}: manifest identity differs")
    if "task" in manifest:
        require(manifest.get("task") == task,
                f"{label}: manifest task differs")
    require(metrics.get("schema") == metrics_schema
            and metrics.get("protocol_sha256") == protocol_sha256
            and metrics.get("task") == task
            and metrics.get("arm") == arm
            and metrics.get("seed") == seed,
            f"{label}: metrics identity differs")
    require(metrics.get("physical_gpu") == physical_gpu
            and metrics.get("cuda_visible_devices") == str(physical_gpu),
            f"{label}: GPU identity differs")
    elapsed = metrics.get("elapsed_seconds")
    peak = metrics.get("peak_vram_bytes")
    require(isinstance(metrics.get("gpu_name"), str)
            and bool(metrics["gpu_name"])
            and isinstance(elapsed, (int, float))
            and not isinstance(elapsed, bool) and math.isfinite(float(elapsed))
            and elapsed >= 0
            and isinstance(peak, int) and not isinstance(peak, bool)
            and peak >= 0,
            f"{label}: runtime metadata differs")
    before = metrics.get("host_digest_before")
    after = metrics.get("host_digest_after")
    require(metrics.get("host_unchanged") is True
            and _require_digest(before, f"{label} host before") == after,
            f"{label}: frozen host changed")
    require(metrics.get("training_labels_used") is False,
            f"{label}: carrier training used labels")
    _require_digest(metrics.get("common_schedule_sha256"),
                    f"{label} training schedule")
    compare_exact(metrics.get("parameter_matching"), PARAMETER_MATCHING,
                  f"{label}.parameter_matching")


def _validate_none_clone(
        reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray],
        label: str) -> None:
    require(set(reference) == set(candidate),
            f"{label}: no-carrier artifact keys differ")
    for name in reference:
        require(np.array_equal(reference[name], candidate[name], equal_nan=False),
                f"{label}: no-carrier clone differs in {name}")


def _validate_carrier_config(metrics: Mapping[str, Any], arm: str,
                             label: str) -> None:
    config = metrics.get("carrier_config")
    expected_name = {
        "none": "none", "gru": "acgru", "lstm": "aclstm",
        "ssm": "diag_ssm", "fixed_trust": "lkc_fixed_trust",
    }[arm]
    require(isinstance(config, Mapping)
            and config.get("carrier") == expected_name
            and config.get("embed_dim") == 384
            and config.get("action_dim") == 10
            and config.get("parameters") == ARM_PARAMETERS[arm],
            f"{label}: carrier configuration differs")


def audit_wave2(root: Path) -> dict[str, Any]:
    cfg, lock, output, formal = _load_lock(
        root, WAVE2_CONFIG, "Wave 2")
    summary_path = formal / "summary.json"
    actual = read_json(summary_path, "Wave 2 summary")
    arms = list(cfg["training"]["arms"])
    seeds = [int(value) for value in cfg["training"]["seeds"]]
    ages = [int(value) for value in cfg["sequence"]["evidence_ages"]]
    tasks = list(cfg["tasks"])
    require(arms == ["none", "gru", "lstm", "ssm", "fixed_trust"]
            and seeds == list(range(5)) and ages == [4, 8, 15]
            and len(tasks) == 2, "Wave 2 sealed grid metadata differs")
    compare_exact(cfg["inference"], {
        "method": "matched-seed x class-stratified held-out-episode bootstrap",
        "paired": True, "draws": 20_000, "confidence": 0.95,
        "seed": 907_300, "no_carrier_effective_independent_models": 1,
    }, "Wave 2 sealed inference")
    _validate_wave2_gates(root, output, formal, cfg, lock)
    _validate_progress(
        read_json(formal / "progress.json", "Wave 2 progress"),
        label="Wave 2", protocol_sha256=lock["protocol_sha256"],
        completed=[
            [str(task["key"]), arm, seed]
            for task in tasks for arm in arms for seed in seeds
        ])
    loaded: dict[tuple[str, str, int],
                 tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    artifact_count = 0
    feature_dim = sum(int(level) ** 2 for level in
                      cfg["evaluation"]["spatial_pool_levels"]) * 384
    expected_artifacts = {
        "history.csv", "validation_predictions.npz", "metrics.json",
        "carrier.pt",
    }
    for task in tasks:
        key = str(task["key"])
        classes = int(task["classes"])
        for arm in arms:
            for seed in seeds:
                directory = formal / "cells" / key / arm / f"s{seed}"
                manifest = read_json(
                    directory / "manifest.json",
                    f"Wave 2 {key}/{arm}/s{seed} manifest")
                require(set(manifest) == {
                    "schema", "protocol_sha256", "task", "arm", "seed",
                    "artifacts",
                }, f"Wave 2 manifest field set differs: {key}/{arm}/s{seed}")
                require(manifest.get("protocol_sha256")
                        == lock["protocol_sha256"],
                        f"Wave 2 cell protocol differs: {key}/{arm}/s{seed}")
                label = f"Wave 2 {key}/{arm}/s{seed}"
                metrics, arrays, digests = _load_cell_arrays(
                    directory, manifest, label,
                    expected_artifacts=expected_artifacts)
                expected_metric_keys = {
                    "schema", "protocol_sha256", "task", "semantic_name",
                    "classes", "arm", "seed", "adaptive_opened_bank",
                    "physical_gpu", "cuda_visible_devices", "gpu_name",
                    "host_digest_before", "host_digest_after",
                    "host_unchanged", "carrier_parameters",
                    "parameter_matching", "carrier_config", "carrier_scope",
                    "training_labels_used", "objective", "epochs",
                    "common_schedule_sha256", "final_train_loss",
                    "final_five_epoch_relative_change", "ages",
                    "elapsed_seconds", "peak_vram_bytes",
                }
                if arm == "none" and seed > 0:
                    expected_metric_keys.update({
                        "duplicated_deterministic_no_carrier_from_seed",
                        "effective_independent_models",
                    })
                require(set(metrics) == expected_metric_keys,
                        f"{label}: metrics field set differs")
                _validate_common_cell(
                    manifest, metrics, label=label,
                    protocol_sha256=lock["protocol_sha256"],
                    manifest_schema="dinowm_wave2_spatial_cell_manifest_v1",
                    metrics_schema="dinowm_wave2_spatial_cell_v1",
                    task=key, arm=arm, seed=seed, physical_gpu=1)
                require(metrics.get("carrier_parameters")
                        == ARM_PARAMETERS[arm],
                        f"Wave 2 parameters differ: {key}/{arm}/s{seed}")
                _validate_carrier_config(metrics, arm, label)
                require(metrics.get("semantic_name") == task["semantic_name"]
                        and metrics.get("classes") == classes
                        and metrics.get("adaptive_opened_bank") is True
                        and metrics.get("carrier_scope")
                        == cfg["adapter"]["carrier_scope"],
                        f"Wave 2 cell task metadata differs: {key}/{arm}/s{seed}")
                compare_exact(metrics.get("objective"), cfg["objective"],
                              f"{label}.objective")
                require(metrics.get("epochs")
                        == (0 if arm == "none"
                            else int(cfg["training"]["epochs"])),
                        f"Wave 2 epoch count differs: {key}/{arm}/s{seed}")
                if arm == "none" and seed > 0:
                    require(metrics.get(
                        "duplicated_deterministic_no_carrier_from_seed") == 0
                        and metrics.get("effective_independent_models") == 1,
                        f"Wave 2 no-carrier clone metadata differs: "
                        f"{key}/{arm}/s{seed}")
                require(isinstance(metrics.get("ages"), Mapping)
                        and set(metrics["ages"]) == {str(age) for age in ages},
                        f"Wave 2 cell age set differs: {key}/{arm}/s{seed}")
                expected_names = {"truth"}
                for age in ages:
                    expected_names.update({
                        f"age_{age}_full_prediction",
                        f"age_{age}_reset_prediction",
                        f"age_{age}_prior_prediction",
                        f"age_{age}_full_mse",
                        f"age_{age}_reset_mse",
                    })
                require(set(arrays) == expected_names,
                        f"Wave 2 prediction keys differ: {key}/{arm}/s{seed}")
                truth = require_integer_labels(
                    arrays["truth"], classes, f"{label} truth", shape=(480,))
                require(set(np.unique(truth)) == set(range(classes)),
                        f"Wave 2 truth coverage differs: {key}/{arm}/s{seed}")
                for age_index, age in enumerate(ages):
                    record = metrics["ages"][str(age)]
                    full = require_integer_labels(
                        arrays[f"age_{age}_full_prediction"], classes,
                        f"{label} age {age} full", shape=(480,))
                    reset = require_integer_labels(
                        arrays[f"age_{age}_reset_prediction"], classes,
                        f"{label} age {age} reset", shape=(480,))
                    prior = require_integer_labels(
                        arrays[f"age_{age}_prior_prediction"], classes,
                        f"{label} age {age} prior", shape=(480,))
                    full_mse = arrays[f"age_{age}_full_mse"]
                    reset_mse = arrays[f"age_{age}_reset_mse"]
                    require_numeric_vector(full_mse, 480,
                                           f"{label} age {age} full MSE")
                    require_numeric_vector(reset_mse, 480,
                                           f"{label} age {age} reset MSE")
                    expected_record = {
                        "endpoint_frame": cfg["sequence"]["endpoint_frames"][
                            age_index],
                        "predictor_context": cfg["sequence"][
                            "predictor_contexts"][age_index],
                        "target_observation_excluded": cfg["sequence"][
                            "endpoint_observation_excluded"],
                        "full_balanced_accuracy": class_balanced_accuracy(
                            full, truth, classes),
                        "reset_with_full_readout_balanced_accuracy":
                            class_balanced_accuracy(reset, truth, classes),
                        "prior_balanced_accuracy": class_balanced_accuracy(
                            prior, truth, classes),
                        "full_next_visual_mse": float(np.mean(full_mse)),
                        "reset_next_visual_mse": float(np.mean(reset_mse)),
                        "feature_dim": feature_dim,
                        "readout": cfg["evaluation"]["readout"],
                    }
                    compare_exact(record, expected_record,
                                  f"{label}.ages.{age}")
                    if arm == "none":
                        require(np.array_equal(full, reset)
                                and np.array_equal(full_mse, reset_mse),
                                f"Wave 2 no-carrier reset differs: "
                                f"{key}/s{seed}/age{age}")
                loaded[(key, arm, seed)] = (metrics, arrays)
                artifact_count += len(digests)

        reference_none = loaded[(key, "none", 0)][1]
        for seed in seeds[1:]:
            _validate_none_clone(
                reference_none, loaded[(key, "none", seed)][1],
                f"Wave 2 {key}/none/s{seed}")

    inference = cfg["inference"]
    results: dict[str, Any] = {}
    bootstrap_records = 0
    for task_index, task in enumerate(tasks):
        key = str(task["key"])
        classes = int(task["classes"])
        truth = loaded[(key, "none", 0)][1]["truth"]
        for arm in arms:
            for seed in seeds:
                require(np.array_equal(
                    truth, loaded[(key, arm, seed)][1]["truth"]),
                    f"Wave 2 truth alignment differs: {key}/{arm}/s{seed}")
        task_result: dict[str, Any] = {
            "semantic_name": task["semantic_name"],
            "classes": classes,
            "chance": 1.0 / classes,
            "ages": {},
        }
        for age_index, age in enumerate(ages):
            record: dict[str, Any] = {
                "arms": {}, "paired_vs_none": {},
                "full_vs_context_reset": {},
            }
            predictions: dict[str, np.ndarray] = {}
            resets: dict[str, np.ndarray] = {}
            for arm_index, arm in enumerate(arms):
                predictions[arm] = np.stack([
                    loaded[(key, arm, seed)][1][
                        f"age_{age}_full_prediction"] for seed in seeds])
                resets[arm] = np.stack([
                    loaded[(key, arm, seed)][1][
                        f"age_{age}_reset_prediction"] for seed in seeds])
                seed = (int(inference["seed"]) + task_index * 1000
                        + age_index * 100 + arm_index)
                absolute = stratified_absolute_bootstrap(
                    predictions[arm], truth, classes=classes,
                    draws=int(inference["draws"]), seed=seed,
                    confidence=float(inference["confidence"]))
                bootstrap_records += 1
                record["arms"][arm] = {
                    "balanced_accuracy": absolute,
                    "seed_values": [class_balanced_accuracy(
                        predictions[arm][index], truth, classes)
                        for index in range(len(seeds))],
                    "prior_seed_values": [
                        loaded[(key, arm, seed_value)][0]["ages"][str(age)][
                            "prior_balanced_accuracy"]
                        for seed_value in seeds],
                    "next_visual_mse_seed_values": [
                        loaded[(key, arm, seed_value)][0]["ages"][str(age)][
                            "full_next_visual_mse"]
                        for seed_value in seeds],
                    "parameters": ARM_PARAMETERS[arm],
                    "effective_independent_models": 1 if arm == "none" else 5,
                }
            for arm_index, arm in enumerate(arms):
                if arm != "none":
                    record["paired_vs_none"][arm] = \
                        stratified_paired_bootstrap(
                            predictions[arm], predictions["none"], truth,
                            classes=classes, draws=int(inference["draws"]),
                            seed=int(inference["seed"]) + 5000
                            + task_index * 1000 + age_index * 100 + arm_index,
                            confidence=float(inference["confidence"]))
                    bootstrap_records += 1
                record["full_vs_context_reset"][arm] = \
                    stratified_paired_bootstrap(
                        predictions[arm], resets[arm], truth,
                        classes=classes, draws=int(inference["draws"]),
                        seed=int(inference["seed"]) + 10_000
                        + task_index * 1000 + age_index * 100 + arm_index,
                        confidence=float(inference["confidence"]))
                bootstrap_records += 1
            task_result["ages"][str(age)] = record
        results[key] = task_result

    expected = {
        "schema": "dinowm_wave2_spatial_carrier_summary_v1",
        "status": "complete",
        "protocol_sha256": lock["protocol_sha256"],
        "study": cfg["study"],
        "scope": cfg["scope"],
        "host": cfg["checkpoint"]["display_name"],
        "adapter": cfg["adapter"],
        "parameter_matching": PARAMETER_MATCHING,
        "grid": {"tasks": 2, "arms": 5, "seeds": 5, "cells": 50},
        "inference": cfg["inference"],
        "results": results,
    }
    compare_exact(actual, expected, "wave2.summary")
    return {
        "status": "verified",
        "summary_sha256": sha256_file(summary_path),
        "cells": 50,
        "locked_sources_hashed": len(lock["source_sha256"]),
        "preoutcome_artifacts_hard_pinned": len(
            SEALED_IDENTITIES[WAVE2_CONFIG]["preoutcome_sha256"]),
        "all_cell_artifacts_hashed": artifact_count,
        "absolute_records": 2 * 3 * 5,
        "paired_vs_none_records": 2 * 3 * 4,
        "full_vs_reset_records": 2 * 3 * 5,
        "bootstrap_records_recomputed": bootstrap_records,
        "draws_per_record": 20_000,
        "seed_schedule": "907300 + sealed task/age/arm offsets",
    }


def _load_pointmaze_audit_inputs(
        root: Path, output: Path, formal: Path, cfg: Mapping[str, Any],
        lock: Mapping[str, Any], admission: Mapping[str, Any],
        controller: Mapping[str, Any]) \
        -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Validate gates/cache/deck and return exact native clusters + success."""

    cache_root = output / "cache"
    manifest = read_json(cache_root / "manifest.json",
                         "Wave 3 cache manifest")
    require(manifest.get("schema") == "dinowm_pointmaze_wave3_cache_v1"
            and manifest.get("status") == "admitted"
            and manifest.get("protocol_sha256") == lock["protocol_sha256"]
            and manifest.get("precarrier_gates_passed") is True
            and manifest.get("host_unchanged") is True,
            "Wave 3 cache/gate lock differs")
    require(repository_path(root, manifest.get("admission_path", ""))
            == (formal / "admission.json").resolve()
            and repository_path(root, manifest.get("controller_gate_path", ""))
            == (formal / "controller_gate.json").resolve()
            and manifest.get("admission_sha256")
            == sha256_file(formal / "admission.json")
            and manifest.get("controller_gate_sha256")
            == sha256_file(formal / "controller_gate.json"),
            "Wave 3 cache gate receipt differs")

    require(admission.get("schema") == "dinowm_pointmaze_wave3_admission_v1"
            and admission.get("status") == "admitted"
            and admission.get("admitted") is True
            and admission.get("all_gates_required") is True
            and _all_declared_passes(admission),
            "Wave 3 semantic admission failed")
    frozen = admission.get("frozen_host", {})
    require(frozen.get("pass") is True
            and _require_digest(frozen.get("digest_before"),
                                "Wave 3 admission host before")
            == frozen.get("digest_after"),
            "Wave 3 admission host changed")
    require(set(admission.get("shortcuts", {}))
            == {str(age) for age in cfg["sequence"]["evidence_ages"]},
            "Wave 3 shortcut age set differs")

    use = cfg["external_use"]
    expected_thresholds = {
        "oracle_success_minimum": use["oracle_success_minimum"],
        "oracle_per_class_success_minimum":
            use["oracle_per_class_success_minimum"],
        "off_diagonal_false_success_maximum":
            use["off_diagonal_false_success_maximum"],
        "deterministic_reset_replay_minimum":
            use["deterministic_reset_replay_minimum"],
    }
    require(controller.get("schema") ==
            "dinowm_pointmaze_wave3_controller_gate_v1"
            and controller.get("status") == "admitted"
            and controller.get("admitted") is True,
            "Wave 3 controller gate failed")
    compare_exact(controller.get("thresholds"), expected_thresholds,
                  "Wave 3 controller thresholds")
    try:
        version = tuple(int(value) for value in str(
            controller["current_mujoco_version"]).split(".")[:2])
    except (KeyError, ValueError) as error:
        raise AuditFailure("Wave 3 MuJoCo version is invalid") from error
    require(version >= (3, 0)
            and bool(re.fullmatch(r"[0-9a-f]{64}", str(
                controller.get("released_xml_sha256", "")))),
            "Wave 3 controller runtime identity differs")

    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, Mapping)
            and set(artifacts) == {
                "base_visual", "cue_visual", "metadata", "selection"},
            "Wave 3 cache artifact ledger differs")
    selection_path, selection_digest = verify_reference(
        root, artifacts["selection"], "Wave 3 selection",
        expected_path=cache_root / "selection.json")
    metadata_path, metadata_digest = verify_reference(
        root, artifacts["metadata"], "Wave 3 metadata",
        expected_path=cache_root / "metadata.npz")
    selection = read_json(selection_path, "Wave 3 selection")
    require(selection.get("schema") ==
            "dinowm_pointmaze_wave3_selection_v1"
            and selection.get("selection_sha256")
            == manifest.get("selection_sha256"),
            "Wave 3 selection metadata differs")
    train_count = int(cfg["dataset"]["train_base_windows"])
    validation_count = int(cfg["dataset"]["validation_base_windows"])
    base_count = train_count + validation_count
    try:
        with np.load(metadata_path, allow_pickle=False) as values:
            require(set(values.files) == {
                "actions", "proprio", "states", "split", "episode_index",
                "local_start",
            }, "Wave 3 metadata keys differ")
            require(values["actions"].shape == (base_count, 19, 10)
                    and values["proprio"].shape == (base_count, 20, 4)
                    and values["states"].shape == (base_count, 20, 4),
                    "Wave 3 metadata tensor shape differs")
            split = np.asarray(values["split"]).copy()
            episode = np.asarray(values["episode_index"]).copy()
            local_start = np.asarray(values["local_start"]).copy()
            states = np.asarray(values["states"]).copy()
    except (OSError, ValueError, KeyError) as error:
        raise AuditFailure(f"invalid Wave 3 metadata: {error}") from error
    require(np.issubdtype(split.dtype, np.integer)
            and split.shape == episode.shape == local_start.shape
            == (base_count,)
            and np.count_nonzero(split == 0) == train_count
            and np.count_nonzero(split == 1) == validation_count
            and np.all((split == 0) | (split == 1))
            and np.isfinite(states).all(),
            "Wave 3 metadata split/content differs")
    rows = selection.get("values")
    require(isinstance(rows, list) and len(rows) == base_count,
            "Wave 3 selection count differs")
    expected_rows = [
        {
            "split": "train" if int(split[index]) == 0 else "validation",
            "episode_index": int(episode[index]),
            "local_start": int(local_start[index]),
        }
        for index in range(base_count)
    ]
    compare_exact(rows, expected_rows, "Wave 3 selection/metadata alignment")
    validation_base = np.flatnonzero(split == 1)
    validation_episode = episode[validation_base].astype(np.int64, copy=False)
    require(len(np.unique(validation_episode)) == validation_count,
            "Wave 3 native validation episodes are not unique")

    deck_record = controller.get("artifact")
    deck_path, deck_digest = verify_reference(
        root, deck_record, "Wave 3 execution deck",
        expected_path=cache_root / "execution_deck.npz")
    try:
        with np.load(deck_path, allow_pickle=False) as values:
            require(set(values.files) == {
                "validation_base_index", "validation_episode",
                "initial_state", "goal_waypoints", "success_matrix",
                "distance_matrix", "final_state", "steps", "replay",
                "selected_goal_success",
            }, "Wave 3 execution-deck keys differ")
            deck = {name: values[name].copy() for name in values.files}
    except (OSError, ValueError, KeyError) as error:
        raise AuditFailure(f"invalid Wave 3 execution deck: {error}") from error
    success = np.asarray(deck["success_matrix"])
    replay = np.asarray(deck["replay"])
    selected_success = np.asarray(deck["selected_goal_success"])
    require(np.array_equal(deck["validation_base_index"], validation_base)
            and np.array_equal(deck["validation_episode"], validation_episode),
            "Wave 3 execution-deck selection differs")
    use_age = int(use["evidence_age"])
    use_age_index = [int(value) for value in
                     cfg["sequence"]["evidence_ages"]].index(use_age)
    endpoint = int(cfg["sequence"]["endpoint_frames"][use_age_index])
    require(np.array_equal(deck["initial_state"],
                           states[validation_base, endpoint])
            and deck["goal_waypoints"].shape == (4, 2)
            and np.isfinite(deck["goal_waypoints"]).all(),
            "Wave 3 execution-deck state/goal differs")
    require(success.shape == (validation_count, 4, 4)
            and (np.issubdtype(success.dtype, np.integer)
                 or np.issubdtype(success.dtype, np.bool_))
            and np.all((success == 0) | (success == 1))
            and deck["distance_matrix"].shape
            == deck["final_state"].shape == (validation_count, 4, 4)
            and np.isfinite(deck["distance_matrix"]).all()
            and np.isfinite(deck["final_state"]).all()
            and deck["steps"].shape == replay.shape
            == selected_success.shape == (validation_count, 4)
            and np.all((replay == 0) | (replay == 1))
            and np.all((selected_success == 0) | (selected_success == 1)),
            "Wave 3 execution-deck arrays are invalid")
    require(np.array_equal(
        success, deck["distance_matrix"] < float(use["success_radius"]))
        and np.array_equal(selected_success,
                           success[:, np.arange(4), np.arange(4)]),
            "Wave 3 execution-deck success derivation differs")
    oracle = success[:, np.arange(4), np.arange(4)]
    per_class = [float(oracle[:, label].mean()) for label in range(4)]
    off_diagonal = success[:, ~np.eye(4, dtype=np.bool_)].reshape(-1)
    require(controller.get("validation_base_windows") == validation_count
            and controller.get("executions") == validation_count * 4
            and controller.get("replayed_executions") == validation_count * 4
            and controller.get("oracle_executed_success")
            == float(oracle.mean())
            and controller.get("oracle_per_class_executed_success")
            == per_class
            and controller.get("off_diagonal_false_success")
            == float(off_diagonal.mean())
            and controller.get("deterministic_replay_fidelity")
            == float(replay.mean()),
            "Wave 3 controller summary differs from execution deck")
    require(float(oracle.mean()) >= float(use["oracle_success_minimum"])
            and min(per_class)
            >= float(use["oracle_per_class_success_minimum"])
            and float(off_diagonal.mean())
            <= float(use["off_diagonal_false_success_maximum"])
            and float(replay.mean())
            >= float(use["deterministic_reset_replay_minimum"]),
            "Wave 3 controller deck misses a registered threshold")
    return np.repeat(validation_episode, 4), success, {
        "selection": selection_digest,
        "metadata": metadata_digest,
        "execution_deck": deck_digest,
    }


def _load_use_features(path: Path, *, train_rows: int,
                       validation_rows: int, feature_dim: int,
                       classes: int, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as values:
            require(set(values.files) == {
                "train_feature", "validation_feature", "train_truth",
                "validation_truth",
            }, f"{label}: use-feature keys differ")
            arrays = {name: values[name].copy() for name in values.files}
    except (OSError, ValueError, KeyError) as error:
        raise AuditFailure(f"{label}: invalid use-feature artifact: {error}") \
            from error
    train_x = np.asarray(arrays["train_feature"])
    validation_x = np.asarray(arrays["validation_feature"])
    require(train_x.dtype == np.float32
            and validation_x.dtype == np.float32
            and train_x.shape == (train_rows, feature_dim)
            and validation_x.shape == (validation_rows, feature_dim)
            and np.isfinite(train_x).all() and np.isfinite(validation_x).all(),
            f"{label}: feature matrix differs")
    train_y = require_integer_labels(
        arrays["train_truth"], classes, f"{label} train truth",
        shape=(train_rows,))
    validation_y = require_integer_labels(
        arrays["validation_truth"], classes, f"{label} validation truth",
        shape=(validation_rows,))
    require(set(np.unique(train_y)) == set(range(classes))
            and set(np.unique(validation_y)) == set(range(classes)),
            f"{label}: use truth lacks a class")
    arrays["train_truth"] = train_y
    arrays["validation_truth"] = validation_y
    return arrays


def refit_shared_consumers(
        formal: Path, arms: Sequence[str], seeds: Sequence[int], *,
        train_rows_per_arm: int, validation_rows: int, feature_dim: int,
        classes: int = 4) \
        -> tuple[dict[str, np.ndarray], list[dict[str, Any]], np.ndarray]:
    """Refit the sealed arm-blind consumer without producer helpers."""

    predictions: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
    receipts: list[dict[str, Any]] = []
    reference_validation_truth: np.ndarray | None = None
    expected_train_truth = np.tile(
        np.arange(classes, dtype=np.int64), train_rows_per_arm // classes)
    expected_validation_truth = np.tile(
        np.arange(classes, dtype=np.int64), validation_rows // classes)
    require(len(expected_train_truth) == train_rows_per_arm
            and len(expected_validation_truth) == validation_rows,
            "consumer truth rows are not class-balanced")
    none_reference: dict[str, np.ndarray] | None = None
    for seed in seeds:
        sources = {
            arm: _load_use_features(
                formal / "cells" / arm / f"s{seed}" / "use_features.npz",
                train_rows=train_rows_per_arm,
                validation_rows=validation_rows, feature_dim=feature_dim,
                classes=classes, label=f"Wave 3 {arm}/s{seed}")
            for arm in arms
        }
        if arms and arms[0] == "none":
            if seed == seeds[0]:
                none_reference = sources["none"]
            else:
                assert none_reference is not None
                _validate_none_clone(
                    none_reference, sources["none"],
                    f"Wave 3 none/s{seed} use features")
        for arm, source in sources.items():
            require(np.array_equal(source["train_truth"], expected_train_truth)
                    and np.array_equal(source["validation_truth"],
                                       expected_validation_truth),
                    f"Wave 3 {arm}/s{seed}: expanded truth ordering differs")
        train_x = np.concatenate([
            sources[arm]["train_feature"] for arm in arms])
        train_y = np.concatenate([
            sources[arm]["train_truth"] for arm in arms])
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000,
                               random_state=0))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier.fit(train_x, train_y)
        except Exception as error:
            raise AuditFailure(
                f"Wave 3 consumer refit failed for seed {seed}: {error}") \
                from error
        coefficient = np.asarray(classifier[-1].coef_)
        digest = hashlib.sha256(coefficient.tobytes()).hexdigest()
        for arm in arms:
            prediction = classifier.predict(
                sources[arm]["validation_feature"])
            predictions[arm].append(require_integer_labels(
                prediction, classes,
                f"Wave 3 refit prediction {arm}/s{seed}",
                shape=(validation_rows,)))
        validation_truth = sources[arms[0]]["validation_truth"]
        if reference_validation_truth is None:
            reference_validation_truth = validation_truth.copy()
        else:
            require(np.array_equal(reference_validation_truth,
                                   validation_truth),
                    "Wave 3 validation truth differs between consumer seeds")
        receipts.append({
            "seed": seed,
            "arm_blind": True,
            "training_arms": list(arms),
            "arm_identifier_feature": False,
            "train_examples": int(len(train_y)),
            "feature_dim": int(train_x.shape[1]),
            "coefficient_sha256": digest,
        })
        del classifier, coefficient, train_x, train_y, sources
    assert reference_validation_truth is not None
    return ({arm: np.stack(values) for arm, values in predictions.items()},
            receipts, reference_validation_truth)


def audit_wave3(root: Path) -> dict[str, Any]:
    cfg, lock, output, formal = _load_lock(
        root, WAVE3_CONFIG, "Wave 3")
    carrier_path = formal / "carrier_summary.json"
    use_path = formal / "external_use_summary.json"
    combined_path = formal / "summary.json"
    carrier_actual = read_json(carrier_path, "Wave 3 carrier summary")
    use_actual = read_json(use_path, "Wave 3 external-use summary")
    combined_actual = read_json(combined_path, "Wave 3 combined summary")
    admission = read_json(formal / "admission.json", "Wave 3 admission")
    controller = read_json(
        formal / "controller_gate.json", "Wave 3 controller gate")

    arms = list(cfg["training"]["arms"])
    seeds = [int(value) for value in cfg["training"]["seeds"]]
    ages = [int(value) for value in cfg["sequence"]["evidence_ages"]]
    require(arms == ["none", "gru", "lstm", "ssm", "fixed_trust"]
            and seeds == list(range(5)) and ages == [4, 8, 15],
            "Wave 3 sealed grid metadata differs")
    compare_exact(cfg["inference"], {
        "method": ("matched carrier-seed x native-validation-episode "
                   "cluster bootstrap preserving four-label counterfactual "
                   "sets"),
        "paired": True, "draws": 20_000, "confidence": 0.95,
        "seed": 832_000, "none_effective_independent_models": 1,
        "resampling_units": ["matched carrier seed", "native episode"],
    }, "Wave 3 sealed inference")
    _validate_preoutcome_identities(root, WAVE3_CONFIG, "Wave 3")
    provenance = read_json(formal / "provenance.json", "Wave 3 provenance")
    _validate_provenance(
        provenance, lock, label="Wave 3",
        schema="dinowm_pointmaze_wave3_provenance_v1", gpu=2,
        paper_flag="paper_modified_by_wave3")
    require(provenance.get("admission_sha256")
            == sha256_file(formal / "admission.json")
            and provenance.get("controller_gate_sha256")
            == sha256_file(formal / "controller_gate.json"),
            "Wave 3 provenance gate hashes differ")
    _validate_progress(
        read_json(formal / "progress.json", "Wave 3 progress"),
        label="Wave 3", protocol_sha256=lock["protocol_sha256"],
        completed=[[arm, seed] for arm in arms for seed in seeds])
    require(not (formal / "stop_receipt.json").exists()
            and not (formal / "formal_stop_receipt.json").exists(),
            "Wave 3 formal stop receipt exists")
    episodes, deck_success, cache_digests = _load_pointmaze_audit_inputs(
        root, output, formal, cfg, lock, admission, controller)
    require(episodes.shape == (480,) and len(np.unique(episodes)) == 120,
            "Wave 3 native episode cluster count differs")
    loaded: dict[tuple[str, int],
                 tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    artifact_count = 0
    truth: np.ndarray | None = None
    expected_artifacts = {
        "history.csv", "validation_predictions.npz", "use_features.npz",
        "metrics.json", "carrier.pt",
    }
    for arm in arms:
        for seed in seeds:
            directory = formal / "cells" / arm / f"s{seed}"
            manifest = read_json(directory / "manifest.json",
                                 f"Wave 3 {arm}/s{seed} manifest")
            require(set(manifest) == {
                "schema", "protocol_sha256", "arm", "seed", "artifacts",
            }, f"Wave 3 manifest field set differs: {arm}/s{seed}")
            require(manifest.get("protocol_sha256")
                    == lock["protocol_sha256"],
                    f"Wave 3 cell protocol differs: {arm}/s{seed}")
            label = f"Wave 3 {arm}/s{seed}"
            metrics, arrays, digests = _load_cell_arrays(
                directory, manifest, label,
                expected_artifacts=expected_artifacts)
            expected_metric_keys = {
                "schema", "protocol_sha256", "task", "arm", "seed",
                "physical_gpu", "cuda_visible_devices", "gpu_name",
                "host_digest_before", "host_digest_after", "host_unchanged",
                "carrier_parameters", "carrier_config", "parameter_matching",
                "training_labels_used", "epochs", "common_schedule_sha256",
                "final_train_loss", "final_five_epoch_relative_change",
                "ages", "elapsed_seconds", "peak_vram_bytes",
            }
            if arm == "none" and seed > 0:
                expected_metric_keys.update({
                    "duplicated_deterministic_no_carrier_from_seed",
                    "effective_independent_models",
                })
            require(set(metrics) == expected_metric_keys,
                    f"{label}: metrics field set differs")
            _validate_common_cell(
                manifest, metrics, label=label,
                protocol_sha256=lock["protocol_sha256"],
                manifest_schema="dinowm_pointmaze_wave3_cell_manifest_v1",
                metrics_schema="dinowm_pointmaze_wave3_cell_v1",
                task=str(cfg["task"]["key"]), arm=arm, seed=seed,
                physical_gpu=2)
            require(metrics.get("carrier_parameters") == ARM_PARAMETERS[arm],
                    f"Wave 3 parameters differ: {arm}/s{seed}")
            _validate_carrier_config(metrics, arm, label)
            require(metrics.get("epochs")
                    == (0 if arm == "none"
                        else int(cfg["training"]["epochs"])),
                    f"Wave 3 epoch count differs: {arm}/s{seed}")
            if arm == "none" and seed > 0:
                require(metrics.get(
                    "duplicated_deterministic_no_carrier_from_seed") == 0
                    and metrics.get("effective_independent_models") == 1,
                    f"Wave 3 no-carrier clone metadata differs: {arm}/s{seed}")
            require(isinstance(metrics.get("ages"), Mapping)
                    and set(metrics["ages"]) == {str(age) for age in ages},
                    f"Wave 3 cell age set differs: {arm}/s{seed}")
            expected_names = {"truth"}
            for age in ages:
                expected_names.update({
                    f"age_{age}_full_prediction",
                    f"age_{age}_reset_prediction",
                    f"age_{age}_prior_prediction",
                    f"age_{age}_full_mse",
                    f"age_{age}_reset_mse",
                })
            require(set(arrays) == expected_names,
                    f"Wave 3 prediction keys differ: {arm}/s{seed}")
            current_truth = require_integer_labels(
                arrays["truth"], 4, f"{label} truth", shape=(480,))
            require(np.array_equal(current_truth, np.tile(np.arange(4), 120)),
                    f"Wave 3 truth ordering differs: {arm}/s{seed}")
            if truth is None:
                truth = current_truth.copy()
            else:
                require(np.array_equal(truth, current_truth),
                        f"Wave 3 truth alignment differs: {arm}/s{seed}")
            for age_index, age in enumerate(ages):
                record = metrics["ages"][str(age)]
                full = require_integer_labels(
                    arrays[f"age_{age}_full_prediction"], 4,
                    f"{label} age {age} full", shape=(480,))
                reset = require_integer_labels(
                    arrays[f"age_{age}_reset_prediction"], 4,
                    f"{label} age {age} reset", shape=(480,))
                prior = require_integer_labels(
                    arrays[f"age_{age}_prior_prediction"], 4,
                    f"{label} age {age} prior", shape=(480,))
                full_mse = arrays[f"age_{age}_full_mse"]
                reset_mse = arrays[f"age_{age}_reset_mse"]
                require_numeric_vector(full_mse, 480,
                                       f"{label} age {age} full MSE")
                require_numeric_vector(reset_mse, 480,
                                       f"{label} age {age} reset MSE")
                expected_record = {
                    "endpoint_frame": cfg["sequence"]["endpoint_frames"][
                        age_index],
                    "predictor_context": cfg["sequence"][
                        "predictor_contexts"][age_index],
                    "target_observation_excluded": cfg["sequence"][
                        "endpoint_observation_excluded"],
                    "full": classification_record(full, current_truth, 4),
                    "reset_with_full_readout": classification_record(
                        reset, current_truth, 4),
                    "prior": classification_record(prior, current_truth, 4),
                    "full_next_visual_mse": float(np.mean(full_mse)),
                    "reset_next_visual_mse": float(np.mean(reset_mse)),
                }
                compare_exact(record, expected_record,
                              f"{label}.ages.{age}")
                if arm == "none":
                    require(np.array_equal(full, reset)
                            and np.array_equal(full_mse, reset_mse),
                            f"Wave 3 no-carrier reset differs: "
                            f"s{seed}/age{age}")
            loaded[(arm, seed)] = (metrics, arrays)
            artifact_count += len(digests)
    assert truth is not None
    reference_none = loaded[("none", 0)][1]
    for seed in seeds[1:]:
        _validate_none_clone(
            reference_none, loaded[("none", seed)][1],
            f"Wave 3 none/s{seed}")

    inference = cfg["inference"]
    carrier_results: dict[str, Any] = {}
    bootstrap_records = 0
    for age_index, age in enumerate(ages):
        record: dict[str, Any] = {
            "arms": {}, "paired_vs_none": {},
            "full_vs_context_reset": {},
        }
        predictions: dict[str, np.ndarray] = {}
        resets: dict[str, np.ndarray] = {}
        for arm_index, arm in enumerate(arms):
            predictions[arm] = np.stack([
                loaded[(arm, seed)][1][f"age_{age}_full_prediction"]
                for seed in seeds])
            resets[arm] = np.stack([
                loaded[(arm, seed)][1][f"age_{age}_reset_prediction"]
                for seed in seeds])
            absolute = native_episode_bootstrap(
                correctness(predictions[arm], truth), episodes,
                draws=int(inference["draws"]),
                seed=int(inference["seed"]) + age_index * 100 + arm_index,
                confidence=float(inference["confidence"]))
            bootstrap_records += 1
            record["arms"][arm] = {
                "balanced_accuracy": absolute,
                "seed_values": [class_balanced_accuracy(
                    predictions[arm][index], truth, 4)
                    for index in range(len(seeds))],
                "parameters": ARM_PARAMETERS[arm],
                "effective_independent_models": 1 if arm == "none" else 5,
                "prior_seed_values": [loaded[(arm, seed)][0]["ages"][
                    str(age)]["prior"]["balanced_accuracy"] for seed in seeds],
                "next_visual_mse_seed_values": [loaded[(arm, seed)][0]["ages"][
                    str(age)]["full_next_visual_mse"] for seed in seeds],
            }
            if arm != "none":
                contrast = correctness(predictions[arm], truth) \
                    - correctness(predictions["none"], truth)
                record["paired_vs_none"][arm] = native_episode_bootstrap(
                    contrast, episodes, draws=int(inference["draws"]),
                    seed=int(inference["seed"]) + 5000
                    + age_index * 100 + arm_index,
                    confidence=float(inference["confidence"]))
                bootstrap_records += 1
            reset_contrast = correctness(predictions[arm], truth) \
                - correctness(resets[arm], truth)
            record["full_vs_context_reset"][arm] = native_episode_bootstrap(
                reset_contrast, episodes, draws=int(inference["draws"]),
                seed=int(inference["seed"]) + 10_000
                + age_index * 100 + arm_index,
                confidence=float(inference["confidence"]))
            bootstrap_records += 1
        carrier_results[str(age)] = record

    carrier_expected = {
        "schema": "dinowm_pointmaze_wave3_carrier_summary_v1",
        "status": "complete",
        "protocol_sha256": lock["protocol_sha256"],
        "study": cfg["study"],
        "task": cfg["task"],
        "host": cfg["checkpoint"]["display_name"],
        "adapter": cfg["adapter"],
        "grid": {"tasks": 1, "arms": 5, "seeds": 5, "cells": 25},
        "parameter_matching": PARAMETER_MATCHING,
        "inference": cfg["inference"],
        "results": carrier_results,
    }
    compare_exact(carrier_actual, carrier_expected, "wave3.carrier_summary")

    artifact_record = use_actual.get("artifact")
    require(isinstance(artifact_record, Mapping),
            "Wave 3 use artifact record missing")
    prediction_path = repository_path(root, artifact_record.get("path", ""))
    require(prediction_path == (formal / "external_use_predictions.npz").resolve(),
            "Wave 3 use prediction artifact path differs")
    verify_manifest_artifact(prediction_path, artifact_record,
                             "Wave 3 external-use predictions",
                             require_path=True)
    try:
        with np.load(prediction_path, allow_pickle=False) as values:
            use_arrays = {name: values[name].copy() for name in values.files}
    except (OSError, ValueError, KeyError) as error:
        raise AuditFailure(f"invalid Wave 3 use predictions: {error}") from error
    expected_keys = {
        "truth", "validation_episode", "success_matrix",
        "random_prediction", "random_executed_success",
    } | {f"prediction__{arm}" for arm in arms} \
      | {f"executed__{arm}" for arm in arms}
    require(set(use_arrays) == expected_keys,
            "Wave 3 external-use prediction keys differ")
    use_truth = require_integer_labels(
        use_arrays["truth"], 4, "Wave 3 use truth", shape=(480,))
    require(np.array_equal(use_truth, truth),
            "Wave 3 use and carrier truth differ")
    use_episodes = np.asarray(use_arrays["validation_episode"])
    require(np.issubdtype(use_episodes.dtype, np.integer)
            and np.array_equal(use_episodes, episodes),
            "Wave 3 use native episode clusters differ")
    success = np.asarray(use_arrays["success_matrix"])
    require(np.array_equal(success, deck_success),
            "Wave 3 use success matrix differs from controller deck")
    train_rows = int(cfg["dataset"]["train_base_windows"]) * 4
    validation_rows = int(cfg["dataset"]["validation_base_windows"]) * 4
    refit_predictions, receipts, refit_truth = refit_shared_consumers(
        formal, arms, seeds, train_rows_per_arm=train_rows,
        validation_rows=validation_rows,
        feature_dim=int(cfg["external_use"]["consumer_feature_dim"]),
        classes=4)
    require(np.array_equal(refit_truth, use_truth),
            "Wave 3 refit consumer truth differs")
    prediction_matrices: dict[str, np.ndarray] = {}
    executed_matrices: dict[str, np.ndarray] = {}
    for arm in arms:
        prediction_matrices[arm] = require_integer_labels(
            use_arrays[f"prediction__{arm}"], 4,
            f"Wave 3 external prediction {arm}",
            shape=(len(seeds), validation_rows))
        require(np.array_equal(prediction_matrices[arm],
                               refit_predictions[arm]),
                f"Wave 3 consumer refit prediction differs: {arm}")
        executed_matrices[arm] = executed_success(
            success, prediction_matrices[arm], use_truth)
        stored_executed = np.asarray(use_arrays[f"executed__{arm}"])
        require(stored_executed.dtype == np.float64
                and stored_executed.shape == (len(seeds), validation_rows)
                and np.isfinite(stored_executed).all()
                and np.all((stored_executed == 0) | (stored_executed == 1))
                and np.array_equal(executed_matrices[arm], stored_executed),
                f"Wave 3 stored execution differs: {arm}")

    random_prediction = require_integer_labels(
        use_arrays["random_prediction"], 4,
        "Wave 3 random-goal prediction",
        shape=(len(seeds), validation_rows))
    expected_random = np.stack([
        np.random.default_rng(
            int(cfg["external_use"]["random_goal_seed"]) + seed).integers(
                0, 4, size=len(use_truth), dtype=np.int64)
        for seed in seeds])
    require(np.array_equal(random_prediction, expected_random),
            "Wave 3 random-goal predictions differ from sealed schedule")
    random_executed = executed_success(success, random_prediction, use_truth)
    stored_random = np.asarray(use_arrays["random_executed_success"])
    require(stored_random.dtype == np.float64
            and stored_random.shape == (len(seeds), validation_rows)
            and np.isfinite(stored_random).all()
            and np.all((stored_random == 0) | (stored_random == 1))
            and np.array_equal(random_executed, stored_random),
            "Wave 3 stored random execution differs")

    use_results: dict[str, Any] = {}
    for arm_index, arm in enumerate(arms):
        goal = correctness(prediction_matrices[arm], use_truth)
        result = {
            "goal_accuracy": native_episode_bootstrap(
                goal, episodes, draws=int(inference["draws"]),
                seed=int(inference["seed"]) + 20_000 + arm_index,
                confidence=float(inference["confidence"])),
            "executed_success": native_episode_bootstrap(
                executed_matrices[arm], episodes,
                draws=int(inference["draws"]),
                seed=int(inference["seed"]) + 21_000 + arm_index,
                confidence=float(inference["confidence"])),
            "contrast_vs_none": native_episode_bootstrap(
                executed_matrices[arm] - executed_matrices["none"], episodes,
                draws=int(inference["draws"]),
                seed=int(inference["seed"]) + 22_000 + arm_index,
                confidence=float(inference["confidence"])),
            "contrast_vs_random": native_episode_bootstrap(
                executed_matrices[arm] - random_executed, episodes,
                draws=int(inference["draws"]),
                seed=int(inference["seed"]) + 23_000 + arm_index,
                confidence=float(inference["confidence"])),
        }
        bootstrap_records += 4
        result["resolved_execution_gain"] = bool(
            arm != "none"
            and result["contrast_vs_none"]["ci95"][0] > 0
            and result["contrast_vs_random"]["ci95"][0] > 0)
        use_results[arm] = result
    random_result = native_episode_bootstrap(
        random_executed, episodes, draws=int(inference["draws"]),
        seed=int(inference["seed"]) + 24_000,
        confidence=float(inference["confidence"]))
    bootstrap_records += 1
    expected_artifact = {
        "path": str(prediction_path.relative_to(root)),
        "size": prediction_path.stat().st_size,
        "sha256": sha256_file(prediction_path),
    }
    use_expected = {
        "schema": "dinowm_pointmaze_wave3_external_use_v1",
        "status": "complete",
        "protocol_sha256": lock["protocol_sha256"],
        "scope": cfg["external_use"],
        "controller_gate": controller,
        "consumer_receipts": receipts,
        "arms": use_results,
        "realized_random_goal": random_result,
        "oracle_executed_success": controller["oracle_executed_success"],
        "artifact": expected_artifact,
        "interpretation": (
            "External arm-blind goal selection plus released waypoint control "
            "in current MuJoCo; this is not native DINO-WM planning."),
    }
    compare_exact(use_actual, use_expected, "wave3.external_use_summary")

    resolved = [arm for arm, value in use_results.items()
                if value["resolved_execution_gain"]]
    combined_expected = {
        "schema": "dinowm_pointmaze_wave3_summary_v1",
        "status": "complete",
        "protocol_sha256": lock["protocol_sha256"],
        "scope": cfg["scope"],
        "admission": admission,
        "controller_gate": controller,
        "carrier_summary_path": "carrier_summary.json",
        "external_use_summary_path": "external_use_summary.json",
        "resolved_external_use_arms": resolved,
    }
    compare_exact(combined_actual, combined_expected, "wave3.summary")
    return {
        "status": "verified",
        "carrier_summary_sha256": sha256_file(carrier_path),
        "external_use_summary_sha256": sha256_file(use_path),
        "combined_summary_sha256": sha256_file(combined_path),
        "external_use_predictions_sha256": expected_artifact["sha256"],
        "relevant_cache_sha256": cache_digests,
        "cells": 25,
        "locked_sources_hashed": len(lock["source_sha256"]),
        "preoutcome_artifacts_hard_pinned": len(
            SEALED_IDENTITIES[WAVE3_CONFIG]["preoutcome_sha256"]),
        "all_cell_artifacts_hashed": artifact_count,
        "use_feature_artifacts_hashed": 25,
        "external_consumers_independently_refit": len(seeds),
        "native_episode_clusters": 120,
        "carrier_absolute_records": 3 * 5,
        "carrier_paired_vs_none_records": 3 * 4,
        "carrier_full_vs_reset_records": 3 * 5,
        "external_use_arm_records": 5 * 4,
        "external_use_random_records": 1,
        "bootstrap_records_recomputed": bootstrap_records,
        "draws_per_record": 20_000,
        "seed_schedule": "832000 + sealed carrier/use offsets; random 833000+seed",
    }


def audit_repository(root: Path) -> dict[str, Any]:
    root = root.resolve()
    preflight = completion_preflight(root)
    if preflight["status"] == "incomplete":
        return preflight
    wave2 = audit_wave2(root)
    wave3 = audit_wave3(root)
    require(wave2["status"] == wave3["status"] == "verified",
            "not every independent statistics audit verified")
    script = Path(__file__).resolve()
    return {
        "schema": "paper_a_statistics_independent_receipt_v1",
        "status": "verified",
        "read_only": True,
        "statistics_computed": True,
        "imports_producer_statistics": False,
        "experiment_roots_modified": False,
        "scientific_cross_family_pooling": False,
        "waves": {"wave2": wave2, "wave3": wave3},
        "totals": {
            "formal_cells": wave2["cells"] + wave3["cells"],
            "summary_records_recomputed": (
                wave2["bootstrap_records_recomputed"]
                + wave3["bootstrap_records_recomputed"]),
            "bootstrap_draws_per_record": 20_000,
        },
        "auditor": {
            "path": str(script.relative_to(root)),
            "sha256": sha256_file(script),
        },
        "claim_boundary": (
            "This receipt independently verifies preserved-prediction "
            "statistics, sealed resampling metadata, locked source/gate "
            "provenance, the PointMaze execution deck, and refitted arm-blind "
            "external consumers. It does not retrain a carrier or world model "
            "or pool scores across hosts, tasks, or families."),
    }


def emit_receipt(root: Path, destination: Path,
                 payload: Mapping[str, Any], *, execute: bool) -> bool:
    if not execute:
        return False
    require(payload.get("status") == "verified",
            "refusing to write an incomplete or failed audit receipt")
    root = root.resolve()
    target = repository_path(root, destination)
    experiment_roots = []
    for relative in (WAVE2_CONFIG, WAVE3_CONFIG):
        cfg = read_yaml(root / relative, "receipt boundary config")
        experiment_roots.append(repository_path(root, cfg["artifacts"]["root"]))
    for experiment in experiment_roots:
        try:
            target.relative_to(experiment)
        except ValueError:
            continue
        raise AuditFailure(f"receipt cannot enter experiment root: {target}")
    require(not target.exists(), f"receipt already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    content = stable_json(payload).encode()
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        require(not target.exists(), f"receipt appeared concurrently: {target}")
        os.link(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = audit_repository(args.root)
        print(stable_json(payload), end="")
        if args.execute and payload.get("status") != "verified":
            # Incomplete is an expected state while formal jobs are active.
            # Never turn that state into a receipt and never print a second,
            # contradictory JSON object.
            return 2
        if args.execute:
            emit_receipt(args.root, args.output, payload, execute=True)
        return 0
    except AuditFailure as error:
        print(stable_json({
            "schema": "paper_a_statistics_independent_receipt_v1",
            "status": "failed",
            "read_only": True,
            "reason": str(error),
        }), end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
