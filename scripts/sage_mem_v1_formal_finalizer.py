#!/usr/bin/env python3
"""Two-phase, fail-closed finalizer for the SAGE-Mem v1 formal grid.

Phase A is deliberately label-free with respect to ``formal_test``.  Every
worker writes predictions (or feature handles), identities, intervention
MSEs, and immutable provenance.  This module authenticates the *entire*
5 cohort x 12 arm x 10 seed grid before it writes a label-reveal receipt.
Only after that receipt is durable may the sealed label registry be opened.

The module is intentionally independent of development selection and does not
import the development auditor.  It never launches a worker.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sage_mem_v1_formal_finalizer_v1"
PHASE_A_SCHEMA = "sage_mem_v1_phase_a_cell_v1"
HISTORY_SCHEMA = "sage_mem_v1_phase_a_history_v1"
RESOURCE_SCHEMA = "sage_mem_v1_phase_a_resources_v1"
LABEL_REGISTRY_SCHEMA = "sage_mem_v1_sealed_label_registry_v1"
CUSTODY_REGISTRY_SCHEMA = "sage_mem_v1_custody_vault_registry_v1"
RAW_CONTEXT_SCHEMA = "sage_mem_v1_raw_context_reference_v1"
RAW_CONTEXT_FEATURE_CONTRACT = {
    "slots": 16,
    "short_observed_slots": 3,
    "long_observed_slots": 16,
    "padding": "left-zero",
    "flatten_order": "time-major",
    "lewm_frame_representation": "frozen-frame-embedding",
    "dino_frame_representation": "mean-pool-frozen-spatial-patches",
}
EXECUTION_DECK_REGISTRY_SCHEMA = "sage_mem_v1_execution_deck_registry_v2"
EXECUTION_UNAVAILABLE_RECEIPT_SCHEMA = \
    "sage_mem_v1_execution_deck_unavailable_v1"
EXECUTION_REPLAY_RECEIPT_SCHEMA = "sage_mem_v1_execution_replay_receipt_v1"
AGES = (4, 8, 15)
COHORTS = (
    "lewm_reacher_color",
    "lewm_pusht_color",
    "dinowm_pusht_token",
    "dinowm_pusht_binding",
    "dinowm_pointmaze_goal",
)
ARMS = (
    "none", "gru", "lstm", "ssm", "fixed_trust", "gdelta",
    "fixed_trust_aux", "ssm_aux", "sage_mem_full",
    "sage_mem_next_only", "sage_mem_no_exposure",
    "sage_mem_exposure_only",
)
SEEDS = tuple(range(10))
CLASSES = {
    "lewm_reacher_color": 4,
    "lewm_pusht_color": 4,
    "dinowm_pusht_token": 4,
    "dinowm_pusht_binding": 6,
    "dinowm_pointmaze_goal": 4,
}
# Counts below are expanded sequence rows, not native-base counts.
FORMAL_TEST_ROWS = {
    "lewm_reacher_color": 720,
    "lewm_pusht_color": 720,
    "dinowm_pusht_token": 960,
    "dinowm_pusht_binding": 960,
    "dinowm_pointmaze_goal": 360 * 4,
}
CONSUMER_TRAIN_ROWS = {
    "lewm_reacher_color": 480,
    "lewm_pusht_color": 480,
    "dinowm_pusht_token": 600,
    "dinowm_pusht_binding": 600,
    "dinowm_pointmaze_goal": 240 * 4,
}
VARIANTS_PER_NATIVE_CLUSTER = {
    cohort: (4 if cohort == "dinowm_pointmaze_goal" else 1)
    for cohort in COHORTS
}
PHYSICAL_GPUS = {
    "lewm_reacher_color": 0,
    "lewm_pusht_color": 0,
    "dinowm_pusht_token": 1,
    "dinowm_pusht_binding": 1,
    "dinowm_pointmaze_goal": 2,
}
RESOURCE_FIELDS = (
    "trainable_parameters",
    "forward_flops_per_episode",
    "persistent_state_floats",
    "peak_cuda_bytes",
    "wall_clock_train_seconds",
)


class SageMemFormalFinalizerError(RuntimeError):
    """A phase boundary, identity, or artifact invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SageMemFormalFinalizerError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _reject_json_constant(value: str) -> None:
    raise SageMemFormalFinalizerError(
        f"non-finite JSON constant is forbidden: {value}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SageMemFormalFinalizerError(
            f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} must be a JSON mapping")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return _sha256_file(path)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    _require(observed == expected,
             f"{label} schema differs; missing={sorted(expected - observed)}, "
             f"unexpected={sorted(observed - expected)}")


def _finite_nonnegative(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)) and float(value) >= 0.0)


def _safe_artifact(parent: Path, record: Mapping[str, Any], label: str) -> Path:
    _exact_keys(record, {"path", "sha256", "size"}, f"{label} handle")
    relative = record.get("path")
    _require(isinstance(relative, str) and relative,
             f"{label} path must be non-empty")
    part = Path(relative)
    _require(not part.is_absolute() and ".." not in part.parts
             and len(part.parts) == 1,
             f"{label} path is unsafe: {relative!r}")
    path = parent / part
    _require(path.is_file() and not path.is_symlink(),
             f"{label} artifact is absent or a symlink: {path}")
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as error:
        raise SageMemFormalFinalizerError(
            f"{label} leaves its owning directory") from error
    _require(isinstance(record.get("size"), int)
             and not isinstance(record["size"], bool)
             and record["size"] > 0 and path.stat().st_size == record["size"],
             f"{label} size differs")
    _require(_is_sha256(record.get("sha256"))
             and _sha256_file(path) == record["sha256"],
             f"{label} hash differs")
    return path


def _safe_registry_artifact(parent: Path, record: Mapping[str, Any],
                            label: str) -> Path:
    """Resolve a hashed registry artifact below the registry custody root."""

    _exact_keys(record, {"path", "sha256", "size"}, f"{label} handle")
    relative = record.get("path")
    _require(isinstance(relative, str) and relative,
             f"{label} path must be non-empty")
    part = Path(relative)
    _require(not part.is_absolute() and ".." not in part.parts,
             f"{label} path is unsafe: {relative!r}")
    path = parent / part
    _require(path.is_file() and not path.is_symlink(),
             f"{label} artifact is absent or a symlink: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as error:
        raise SageMemFormalFinalizerError(
            f"{label} leaves the custody root") from error
    cursor = parent
    for component in part.parts:
        cursor = cursor / component
        _require(not cursor.is_symlink(),
                 f"{label} path contains a symlink")
    _require(isinstance(record.get("size"), int)
             and not isinstance(record["size"], bool)
             and record["size"] > 0 and path.stat().st_size == record["size"],
             f"{label} size differs")
    _require(_is_sha256(record.get("sha256"))
             and _sha256_file(path) == record["sha256"],
             f"{label} hash differs")
    return path


@dataclass(frozen=True)
class _GridContract:
    cohorts: tuple[str, ...]
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    ages: tuple[int, ...]
    classes: Mapping[str, int]
    formal_test_rows: Mapping[str, int]
    consumer_train_rows: Mapping[str, int]
    variants_per_cluster: Mapping[str, int]
    physical_gpus: Mapping[str, int]
    protocol_fingerprint: str | None = None
    require_600: bool = False

    def validate(self) -> None:
        _require(bool(self.cohorts) and bool(self.arms) and bool(self.seeds),
                 "grid contract cannot be empty")
        _require(len(set(self.cohorts)) == len(self.cohorts)
                 and len(set(self.arms)) == len(self.arms)
                 and len(set(self.seeds)) == len(self.seeds),
                 "grid contract identities must be unique")
        _require(self.ages == AGES,
                 "SAGE-Mem v1 evidence ages must be exactly 4/8/15")
        for mapping, label in (
                (self.classes, "classes"),
                (self.formal_test_rows, "formal rows"),
                (self.consumer_train_rows, "consumer rows"),
                (self.variants_per_cluster, "cluster variants")):
            _require(set(mapping) == set(self.cohorts),
                     f"{label} registry differs from cohorts")
            _require(all(isinstance(value, int) and not isinstance(value, bool)
                         and value > 0 for value in mapping.values()),
                     f"{label} values must be positive integers")
        _require(set(self.physical_gpus) == set(self.cohorts)
                 and all(isinstance(value, int)
                         and not isinstance(value, bool)
                         and value in (0, 1, 2)
                         for value in self.physical_gpus.values()),
                 "physical GPU registry must assign exactly devices 0/1/2")
        _require(self.protocol_fingerprint is None
                 or _is_sha256(self.protocol_fingerprint),
                 "registered protocol fingerprint must be SHA-256")
        if self.require_600:
            _require(len(self.cohorts) * len(self.arms) * len(self.seeds) == 600,
                     "production formal grid must contain exactly 600 cells")

    @property
    def total_cells(self) -> int:
        return len(self.cohorts) * len(self.arms) * len(self.seeds)


def _registered_protocol_fingerprint() -> str:
    path = ROOT / "configs/sage_mem_v1.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "SAGE-Mem protocol must be a mapping")
    return _sha256_json(value)


PRODUCTION_CONTRACT = _GridContract(
    cohorts=COHORTS,
    arms=ARMS,
    seeds=SEEDS,
    ages=AGES,
    classes=CLASSES,
    formal_test_rows=FORMAL_TEST_ROWS,
    consumer_train_rows=CONSUMER_TRAIN_ROWS,
    variants_per_cluster=VARIANTS_PER_NATIVE_CLUSTER,
    physical_gpus=PHYSICAL_GPUS,
    protocol_fingerprint=_registered_protocol_fingerprint(),
    require_600=True,
)
PRODUCTION_CONTRACT.validate()


@dataclass(frozen=True)
class ValidatedCell:
    cohort: str
    arm: str
    seed: int
    directory: Path
    manifest_sha256: str
    bank_sha256: str
    representation: str
    shared_consumer_sha256: str | None
    measurements_path: Path
    feature_dimension: int | None
    protocol_fingerprint: str
    physical_gpu: int
    arrays: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ValidatedGrid:
    root: Path
    contract: _GridContract
    cells: Mapping[tuple[str, str, int], ValidatedCell]
    bank_hashes: Mapping[str, str]
    grid_sha256: str


@dataclass(frozen=True)
class LabelSet:
    formal_test_episode_id: np.ndarray
    formal_test_native_cluster_id: np.ndarray
    formal_test_label: np.ndarray
    consumer_train_episode_id: np.ndarray
    consumer_train_native_cluster_id: np.ndarray
    consumer_train_label: np.ndarray


@dataclass(frozen=True)
class ValidatedRawContextReference:
    cohort: str
    seed: int
    manifest_sha256: str
    bank_sha256: str
    measurements_path: Path
    feature_dimension: int
    arrays: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ExecutionDeck:
    cohort: str
    controller_identity_sha256: str
    threshold: float
    episode_id: np.ndarray
    native_cluster_id: np.ndarray
    class_conditioned_success: np.ndarray
    oracle_success: np.ndarray
    random_class: np.ndarray
    random_success: np.ndarray

    @property
    def oracle_rate(self) -> float:
        return float(np.mean(self.oracle_success))

    @property
    def random_rate(self) -> float:
        return float(np.mean(self.random_success))

    @property
    def eligible(self) -> bool:
        return self.oracle_rate >= self.threshold


def _validate_history(path: Path, bank_hash: str) -> None:
    value = _read_json(path, "phase-A training history")
    _exact_keys(value, {
        "schema", "study", "status", "formal_test_labels_read",
        "development_outcomes_read", "bank_manifest_sha256", "epochs",
    }, "phase-A history")
    _require(value["schema"] == HISTORY_SCHEMA
             and value["study"] == "sage-mem-v1"
             and value["status"] == "complete"
             and value["formal_test_labels_read"] is False
             and value["development_outcomes_read"] is False
             and value["bank_manifest_sha256"] == bank_hash,
             "phase-A history crosses a sealed data boundary")
    epochs = value["epochs"]
    _require(isinstance(epochs, list), "phase-A epochs must be a list")
    previous = -1
    for row in epochs:
        _require(isinstance(row, dict), "phase-A epoch must be a mapping")
        _exact_keys(row, {"epoch", "train_label_free_loss"},
                    "phase-A epoch")
        _require(isinstance(row["epoch"], int)
                 and not isinstance(row["epoch"], bool)
                 and row["epoch"] > previous
                 and _finite_nonnegative(row["train_label_free_loss"]),
                 "phase-A epoch/loss is malformed")
        previous = row["epoch"]


def _validate_resources(path: Path) -> None:
    value = _read_json(path, "phase-A resource report")
    _exact_keys(value, {"schema", "study", "status", "metrics"},
                "phase-A resource report")
    _require(value["schema"] == RESOURCE_SCHEMA
             and value["study"] == "sage-mem-v1"
             and value["status"] == "complete"
             and isinstance(value["metrics"], dict),
             "phase-A resource report identity differs")
    _exact_keys(value["metrics"], set(RESOURCE_FIELDS),
                "phase-A resource metrics")
    _require(all(_finite_nonnegative(value["metrics"][key])
                 for key in RESOURCE_FIELDS),
             "phase-A resources must be finite and non-negative")


_COMMON_ARRAY_KEYS = {
    "formal_test_episode_id",
    "formal_test_native_cluster_id",
    "formal_test_evidence_age",
    "consumer_train_episode_id",
    "consumer_train_native_cluster_id",
    "consumer_train_evidence_age",
    "formal_test_full_mse",
    "formal_test_reset_mse",
    "formal_test_prior_mse",
}
_PREDICTION_KEYS = {
    "formal_test_full_pred",
    "formal_test_reset_pred",
    "formal_test_prior_pred",
}
_FEATURE_KEYS = {
    "formal_test_full_features",
    "formal_test_reset_features",
    "formal_test_prior_features",
    "consumer_train_full_features",
}


def _validate_age_identity(arrays: Mapping[str, np.ndarray], prefix: str,
                           count: int, contract: _GridContract) -> None:
    episode = arrays[f"{prefix}_episode_id"]
    cluster = arrays[f"{prefix}_native_cluster_id"]
    age = arrays[f"{prefix}_evidence_age"]
    expected_shape = (len(contract.ages), count)
    for name, value in (("episode", episode), ("cluster", cluster),
                        ("age", age)):
        _require(value.shape == expected_shape
                 and np.issubdtype(value.dtype, np.integer),
                 f"{prefix} {name} must be an integer age-by-row matrix")
    _require(np.all(episode >= 0) and np.all(cluster >= 0),
             f"{prefix} identities must be non-negative")
    for index, evidence_age in enumerate(contract.ages):
        _require(np.all(age[index] == evidence_age),
                 f"{prefix} evidence-age row {index} is not age {evidence_age}")
        _require(np.array_equal(episode[index], episode[0])
                 and np.array_equal(cluster[index], cluster[0]),
                 f"{prefix} identities drift across evidence ages")
    _require(len(np.unique(episode[0])) == count,
             f"{prefix} episode IDs are not unique")


def _validate_cluster_multiplicity(episode: np.ndarray, cluster: np.ndarray,
                                   variants: int, label: str,
                                   labels: np.ndarray | None = None,
                                   classes: int | None = None) -> None:
    _require(episode.ndim == cluster.ndim == 1 and len(episode) == len(cluster),
             f"{label} cluster registry is unaligned")
    unique, counts = np.unique(cluster, return_counts=True)
    _require(len(unique) * variants == len(cluster)
             and np.all(counts == variants),
             f"{label} must preserve exactly x{variants} native clustering")
    if labels is not None:
        _require(classes is not None and labels.shape == episode.shape,
                 f"{label} label registry is unaligned")
        for native in unique:
            selected = labels[cluster == native]
            if variants == classes:
                _require(set(map(int, selected)) == set(range(classes)),
                         f"{label} counterfactual cluster omits a class")


def _load_measurements(path: Path, representation: str, *, cohort: str,
                       contract: _GridContract
                       ) -> tuple[dict[str, np.ndarray], int | None]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            expected = (_COMMON_ARRAY_KEYS | _PREDICTION_KEYS
                        if representation == "predicted_labels"
                        else _COMMON_ARRAY_KEYS | _FEATURE_KEYS)
            _require(keys == expected,
                     f"phase-A measurement schema differs for {cohort}")
            # Identities, MSEs, and predicted labels are small enough to keep.
            # High-dimensional feature tensors are validated one at a time and
            # re-opened lazily during centralized consumer fitting; retaining
            # 600 cells of d~=8,064 tensors would exceed host RAM.
            retained = (_COMMON_ARRAY_KEYS | _PREDICTION_KEYS
                        if representation == "predicted_labels"
                        else _COMMON_ARRAY_KEYS)
            arrays = {key: np.asarray(archive[key]).copy()
                      for key in retained}
            feature_dimension: int | None = None
            if representation == "feature_artifact":
                dimensions: set[int] = set()
                for key in _FEATURE_KEYS:
                    value = np.asarray(archive[key])
                    rows = (contract.consumer_train_rows[cohort]
                            if key.startswith("consumer_train") else
                            contract.formal_test_rows[cohort])
                    _require(value.ndim == 3
                             and value.shape[:2] ==
                             (len(contract.ages), rows)
                             and value.shape[2] > 0
                             and (np.issubdtype(value.dtype, np.floating)
                                  or np.issubdtype(value.dtype, np.integer))
                             and np.isfinite(value).all(),
                             f"{cohort} {key} must be a finite "
                             "age-by-row feature tensor")
                    dimensions.add(int(value.shape[2]))
                    del value
                _require(len(dimensions) == 1,
                         f"{cohort} feature dimensions differ across "
                         "interventions")
                feature_dimension = dimensions.pop()
    except (OSError, ValueError) as error:
        raise SageMemFormalFinalizerError(
            f"cannot load phase-A measurements: {path}") from error

    test_count = contract.formal_test_rows[cohort]
    consumer_count = contract.consumer_train_rows[cohort]
    _validate_age_identity(arrays, "formal_test", test_count, contract)
    _validate_age_identity(arrays, "consumer_train", consumer_count, contract)
    variants = contract.variants_per_cluster[cohort]
    _validate_cluster_multiplicity(
        arrays["formal_test_episode_id"][0],
        arrays["formal_test_native_cluster_id"][0], variants,
        f"{cohort}/formal_test")
    _validate_cluster_multiplicity(
        arrays["consumer_train_episode_id"][0],
        arrays["consumer_train_native_cluster_id"][0], variants,
        f"{cohort}/consumer_train")
    expected_test = (len(contract.ages), test_count)
    for key in ("formal_test_full_mse", "formal_test_reset_mse",
                "formal_test_prior_mse"):
        value = arrays[key]
        _require(value.shape == expected_test
                 and (np.issubdtype(value.dtype, np.floating)
                      or np.issubdtype(value.dtype, np.integer))
                 and np.isfinite(value).all() and np.all(value >= 0),
                 f"{cohort} {key} must be finite, non-negative, age-separated")
    classes = contract.classes[cohort]
    if representation == "predicted_labels":
        for key in _PREDICTION_KEYS:
            value = arrays[key]
            _require(value.shape == expected_test
                     and np.issubdtype(value.dtype, np.integer)
                     and np.all((value >= 0) & (value < classes)),
                     f"{cohort} {key} is not a valid per-age class prediction")
    return arrays, feature_dimension


def _validate_cell(cell: Path, cohort: str, arm: str, seed: int,
                   contract: _GridContract) -> ValidatedCell:
    _require(cell.is_dir() and not cell.is_symlink(),
             f"phase-A cell directory missing or unsafe: {cell}")
    manifest_path = cell / "manifest.json"
    _require(manifest_path.is_file() and not manifest_path.is_symlink(),
             f"phase-A manifest missing: {manifest_path}")
    value = _read_json(manifest_path, "phase-A manifest")
    _exact_keys(value, {
        "schema", "study", "stage", "status", "cohort", "arm", "seed",
        "physical_gpu", "cuda_visible_devices", "protocol_fingerprint",
        "completed_unix_ns",
        "ages", "formal_test_labels_read", "formal_test_labels_available",
        "development_outcomes_read", "labels_used_for_training",
        "bank_manifest_sha256", "host_hash_before", "host_hash_after",
        "prediction_representation", "consumer_contract",
        "shared_consumer_sha256", "artifacts",
    }, "phase-A manifest")
    _require(value["schema"] == PHASE_A_SCHEMA
             and value["study"] == "sage-mem-v1"
             and value["stage"] == "formal-phase-a"
             and value["status"] == "complete-label-free"
             and value["cohort"] == cohort and value["arm"] == arm
             and value["seed"] == seed and value["ages"] == list(contract.ages),
             f"phase-A cell identity changed: {cohort}/{arm}/seed-{seed}")
    expected_gpu = contract.physical_gpus[cohort]
    _require(value["physical_gpu"] == expected_gpu
             and value["cuda_visible_devices"] == str(expected_gpu)
             and _is_sha256(value["protocol_fingerprint"])
             and (contract.protocol_fingerprint is None
                  or value["protocol_fingerprint"] ==
                  contract.protocol_fingerprint)
             and isinstance(value["completed_unix_ns"], int)
             and not isinstance(value["completed_unix_ns"], bool)
             and value["completed_unix_ns"] > 0,
             f"phase-A GPU/protocol/completion identity differs: {cell}")
    _require(value["formal_test_labels_read"] is False
             and value["formal_test_labels_available"] is False
             and value["development_outcomes_read"] is False
             and value["labels_used_for_training"] is False,
             f"phase-A cell crossed a label/development boundary: {cell}")
    bank_hash = value["bank_manifest_sha256"]
    _require(_is_sha256(bank_hash), "phase-A exact bank hash is missing")
    _require(_is_sha256(value["host_hash_before"])
             and value["host_hash_before"] == value["host_hash_after"],
             "phase-A frozen host hash changed")
    representation = value["prediction_representation"]
    _require(representation in {"predicted_labels", "feature_artifact"},
             "unknown phase-A prediction representation")
    shared_hash = value["shared_consumer_sha256"]
    if representation == "predicted_labels":
        _require(value["consumer_contract"] == "precomputed-shared-arm-blind"
                 and _is_sha256(shared_hash),
                 "precomputed predictions lack a shared arm-blind consumer")
    else:
        _require(value["consumer_contract"] ==
                 "centralized-pooled-consumer-train-features"
                 and shared_hash is None,
                 "feature cells must defer consumer fitting to the finalizer")
    artifacts = value["artifacts"]
    _require(isinstance(artifacts, dict), "phase-A artifacts must be a mapping")
    _exact_keys(artifacts, {"measurements", "checkpoint", "history",
                            "resource_report"}, "phase-A artifacts")
    paths = {name: _safe_artifact(cell, artifacts[name], name)
             for name in artifacts}
    _validate_history(paths["history"], bank_hash)
    _validate_resources(paths["resource_report"])
    arrays, feature_dimension = _load_measurements(
        paths["measurements"], representation, cohort=cohort,
        contract=contract)
    observed = {item.name for item in cell.iterdir()}
    allowed = {"manifest.json", *(path.name for path in paths.values())}
    _require(observed == allowed,
             f"unexpected files in phase-A cell: {sorted(observed - allowed)}")
    return ValidatedCell(
        cohort=cohort, arm=arm, seed=seed, directory=cell,
        manifest_sha256=_sha256_file(manifest_path), bank_sha256=bank_hash,
        representation=representation, shared_consumer_sha256=shared_hash,
        measurements_path=paths["measurements"],
        feature_dimension=feature_dimension,
        protocol_fingerprint=value["protocol_fingerprint"],
        physical_gpu=expected_gpu,
        arrays=arrays)


def validate_phase_a_cell(directory: str | Path, cohort: str, arm: str,
                          seed: int) -> ValidatedCell:
    """Validate one runner-written Phase-A cell against production identity."""

    _require(cohort in COHORTS and arm in ARMS and seed in SEEDS,
             "phase-A cell is outside the production grid")
    path = Path(directory)
    _require(path.name == f"seed-{seed}",
             "phase-A production directory must use seed-{seed}")
    return _validate_cell(path, cohort, arm, seed, PRODUCTION_CONTRACT)


def _validate_complete_grid(root: str | Path,
                            contract: _GridContract) -> ValidatedGrid:
    contract.validate()
    root = Path(root).resolve()
    cells_root = root / "cells"
    _require(cells_root.is_dir() and not cells_root.is_symlink(),
             f"phase-A cells root is missing: {cells_root}")
    _require(all(item.is_dir() and not item.is_symlink()
                 for item in cells_root.iterdir())
             and {item.name for item in cells_root.iterdir()}
             == set(contract.cohorts),
             "phase-A cohort directory registry differs")
    for cohort in contract.cohorts:
        cohort_root = cells_root / cohort
        _require(all(item.is_dir() and not item.is_symlink()
                     for item in cohort_root.iterdir())
                 and {item.name for item in cohort_root.iterdir()}
                 == set(contract.arms),
                 f"phase-A arm directory registry differs: {cohort}")
        for arm in contract.arms:
            arm_root = cohort_root / arm
            _require(all(item.is_dir() and not item.is_symlink()
                         for item in arm_root.iterdir())
                     and {item.name for item in arm_root.iterdir()}
                     == {f"seed-{seed}" for seed in contract.seeds},
                     f"phase-A seed directory registry differs: "
                     f"{cohort}/{arm}")
    expected_paths = {
        (cells_root / cohort / arm / f"seed-{seed}").resolve()
        for cohort in contract.cohorts for arm in contract.arms
        for seed in contract.seeds
    }
    observed_paths = {
        path.parent.resolve() for path in cells_root.glob("*/*/s*/manifest.json")
    }
    _require(observed_paths == expected_paths,
             f"phase-A grid is incomplete or has extras: expected "
             f"{contract.total_cells}, observed {len(observed_paths)}")

    cells: dict[tuple[str, str, int], ValidatedCell] = {}
    canonical: dict[str, ValidatedCell] = {}
    bank_hashes: dict[str, str] = {}
    feature_dimensions: dict[str, int] = {}
    consumer_hashes: dict[tuple[str, int], str] = {}
    protocol_fingerprint: str | None = None
    digest_rows: list[dict[str, Any]] = []
    for cohort in contract.cohorts:
        for arm in contract.arms:
            for seed in contract.seeds:
                directory = cells_root / cohort / arm / f"seed-{seed}"
                cell = _validate_cell(directory, cohort, arm, seed, contract)
                key = (cohort, arm, seed)
                cells[key] = cell
                if protocol_fingerprint is None:
                    protocol_fingerprint = cell.protocol_fingerprint
                _require(cell.protocol_fingerprint == protocol_fingerprint,
                         "protocol fingerprint drifts across the formal grid")
                if cohort not in canonical:
                    canonical[cohort] = cell
                    bank_hashes[cohort] = cell.bank_sha256
                reference = canonical[cohort]
                _require(cell.bank_sha256 == bank_hashes[cohort],
                         f"exact bank hash drifts across {cohort} cells")
                _require(cell.representation == reference.representation,
                         f"prediction representation drifts across {cohort}")
                for name in (
                        "formal_test_episode_id",
                        "formal_test_native_cluster_id",
                        "formal_test_evidence_age",
                        "consumer_train_episode_id",
                        "consumer_train_native_cluster_id",
                        "consumer_train_evidence_age"):
                    _require(np.array_equal(cell.arrays[name],
                                            reference.arrays[name]),
                             f"cross-arm/seed identity drift: {cohort}/{name}")
                _require(cell.protocol_fingerprint ==
                         reference.protocol_fingerprint,
                         f"protocol fingerprint drifts across {cohort}")
                if cell.representation == "predicted_labels":
                    consumer_key = (cohort, seed)
                    previous = consumer_hashes.setdefault(
                        consumer_key, str(cell.shared_consumer_sha256))
                    _require(previous == cell.shared_consumer_sha256,
                             f"consumer is not shared across arms: "
                             f"{cohort}/seed-{seed}")
                else:
                    _require(cell.feature_dimension is not None,
                             f"feature dimension missing: {cohort}")
                    dimension = int(cell.feature_dimension)
                    previous_dimension = feature_dimensions.setdefault(
                        cohort, dimension)
                    _require(dimension == previous_dimension,
                             f"feature dimension drifts across {cohort}")
                digest_rows.append({
                    "cohort": cohort, "arm": arm, "seed": seed,
                    "manifest_sha256": cell.manifest_sha256,
                    "bank_manifest_sha256": cell.bank_sha256,
                    "measurement_identity_sha256": _sha256_json({
                        name: cell.arrays[name].astype(np.int64).tolist()
                        for name in (
                            "formal_test_episode_id",
                            "formal_test_native_cluster_id",
                            "formal_test_evidence_age",
                            "consumer_train_episode_id",
                            "consumer_train_native_cluster_id",
                            "consumer_train_evidence_age",
                        )
                    }),
                })
    _require(len(cells) == contract.total_cells,
             "validated phase-A cell count differs from contract")
    return ValidatedGrid(
        root=root, contract=contract, cells=cells,
        bank_hashes=bank_hashes,
        grid_sha256=_sha256_json(digest_rows))


def validate_complete_phase_a_grid(root: str | Path) -> ValidatedGrid:
    """Authenticate the exact 600-cell production grid without reading labels."""

    return _validate_complete_grid(root, PRODUCTION_CONTRACT)


_RAW_CONTEXT_ARRAY_KEYS = {
    "formal_test_episode_id",
    "formal_test_native_cluster_id",
    "formal_test_evidence_age",
    "consumer_train_episode_id",
    "consumer_train_native_cluster_id",
    "consumer_train_evidence_age",
    "formal_test_short_features",
    "formal_test_long_features",
    "consumer_train_short_features",
    "consumer_train_long_features",
}


def _validate_raw_context_references(
        root: str | Path, grid: ValidatedGrid
        ) -> tuple[dict[tuple[str, int], ValidatedRawContextReference], str]:
    """Validate the optional short-3/long-16 raw-context reference contract.

    The producer supplies only frozen, age-major short/long features.  One
    shared semantic consumer is fitted after the complete-grid label reveal.
    These references are not parameter-matched arms.
    """

    root = Path(root).resolve()
    _require(root.is_dir() and not root.is_symlink(),
             f"raw-context reference root is missing or unsafe: {root}")
    _require(all(item.is_dir() and not item.is_symlink()
                 for item in root.iterdir() if item.name != "summary.json")
             and {item.name for item in root.iterdir()}
             == set(grid.contract.cohorts) | {"summary.json"},
             "raw-context cohort directory registry differs")
    producer_summary = _read_json(
        root / "summary.json", "raw-context producer summary")
    _exact_keys(producer_summary, {
        "schema", "study", "status", "cells", "cohorts", "seeds",
        "feature_contract", "formal_labels_read",
        "development_outcomes_read", "mse_emitted", "records_sha256",
    }, "raw-context producer summary")
    _require(producer_summary["schema"] ==
             "sage_mem_v1_raw_context_producer_v1"
             and producer_summary["study"] == "sage-mem-v1"
             and producer_summary["status"] == "complete-label-free"
             and producer_summary["cells"] ==
             len(grid.contract.cohorts) * len(grid.contract.seeds)
             and producer_summary["cohorts"] == list(grid.contract.cohorts)
             and producer_summary["seeds"] == list(grid.contract.seeds)
             and producer_summary["feature_contract"] ==
             RAW_CONTEXT_FEATURE_CONTRACT
             and producer_summary["formal_labels_read"] is False
             and producer_summary["development_outcomes_read"] is False
             and producer_summary["mse_emitted"] is False
             and _is_sha256(producer_summary["records_sha256"]),
             "raw-context producer summary identity differs")
    for cohort in grid.contract.cohorts:
        cohort_root = root / cohort
        _require(all(item.is_dir() and not item.is_symlink()
                     for item in cohort_root.iterdir())
                 and {item.name for item in cohort_root.iterdir()}
                 == {f"seed-{seed}" for seed in grid.contract.seeds},
                 f"raw-context seed directory registry differs: {cohort}")
    expected = {
        (root / cohort / f"seed-{seed}").resolve()
        for cohort in grid.contract.cohorts for seed in grid.contract.seeds
    }
    observed = {
        path.parent.resolve()
        for path in root.glob("*/seed-*/manifest.json")
    }
    _require(observed == expected,
             "raw-context reference grid is incomplete or has extras")
    references: dict[tuple[str, int], ValidatedRawContextReference] = {}
    digest_rows: list[dict[str, Any]] = []
    producer_records: list[dict[str, Any]] = []
    for cohort in grid.contract.cohorts:
        canonical = grid.cells[(cohort, grid.contract.arms[0],
                                grid.contract.seeds[0])]
        for seed in grid.contract.seeds:
            directory = root / cohort / f"seed-{seed}"
            manifest_path = directory / "manifest.json"
            value = _read_json(manifest_path, "raw-context manifest")
            _exact_keys(value, {
                "schema", "study", "stage", "status", "cohort", "seed",
                "ages", "short_context_frames", "long_context_frames",
                "separate_from_parameter_matched_arms",
                "formal_test_labels_read", "development_outcomes_read",
                "bank_manifest_sha256", "host_hash_before", "host_hash_after",
                "consumer_contract", "shared_consumer_sha256",
                "feature_contract", "artifact",
            }, "raw-context manifest")
            _require(value["schema"] == RAW_CONTEXT_SCHEMA
                     and value["study"] == "sage-mem-v1"
                     and value["stage"] == "formal-raw-context-reference"
                     and value["status"] == "complete-label-free"
                     and value["cohort"] == cohort and value["seed"] == seed
                     and value["ages"] == list(grid.contract.ages)
                     and value["short_context_frames"] == 3
                     and value["long_context_frames"] == 16
                     and value["separate_from_parameter_matched_arms"] is True,
                     f"raw-context reference identity changed: "
                     f"{cohort}/seed-{seed}")
            _require(value["formal_test_labels_read"] is False
                     and value["development_outcomes_read"] is False,
                     "raw-context reference crossed a sealed data boundary")
            _require(value["bank_manifest_sha256"] == grid.bank_hashes[cohort]
                     and _is_sha256(value["host_hash_before"])
                     and value["host_hash_before"] == value["host_hash_after"],
                     "raw-context reference bank/host identity differs")
            _require(value["consumer_contract"] ==
                     "post-reveal-shared-short-long-arm-blind"
                     and value["shared_consumer_sha256"] is None
                     and value["feature_contract"] ==
                     RAW_CONTEXT_FEATURE_CONTRACT,
                     "raw-context Phase A must defer its shared consumer")
            artifact = _safe_artifact(
                directory, value["artifact"], "raw-context measurements")
            try:
                with np.load(artifact, allow_pickle=False) as archive:
                    _require(set(archive.files) == _RAW_CONTEXT_ARRAY_KEYS,
                             "raw-context measurement schema differs")
                    arrays = {key: np.asarray(archive[key]).copy()
                              for key in _RAW_CONTEXT_ARRAY_KEYS
                              if not key.endswith("_features")}
                    dimensions: set[int] = set()
                    for key in (
                            "formal_test_short_features",
                            "formal_test_long_features",
                            "consumer_train_short_features",
                            "consumer_train_long_features"):
                        features = np.asarray(archive[key])
                        count = (grid.contract.consumer_train_rows[cohort]
                                 if key.startswith("consumer_train") else
                                 grid.contract.formal_test_rows[cohort])
                        _require(features.ndim == 3
                                 and features.shape[:2] ==
                                 (len(grid.contract.ages), count)
                                 and features.shape[2] > 0
                                 and (np.issubdtype(
                                     features.dtype, np.floating)
                                      or np.issubdtype(
                                          features.dtype, np.integer))
                                 and np.isfinite(features).all(),
                                 f"raw-context feature is malformed: {key}")
                        dimensions.add(int(features.shape[2]))
                        del features
                    _require(len(dimensions) == 1,
                             "raw-context short/long feature dimensions differ")
                    feature_dimension = dimensions.pop()
            except (OSError, ValueError) as error:
                raise SageMemFormalFinalizerError(
                    f"cannot load raw-context reference: {artifact}") from error
            for split in ("formal_test", "consumer_train"):
                _validate_age_identity(
                    arrays, split,
                    (grid.contract.formal_test_rows[cohort]
                     if split == "formal_test" else
                     grid.contract.consumer_train_rows[cohort]),
                    grid.contract)
                for suffix in (
                        "episode_id", "native_cluster_id", "evidence_age"):
                    identity = f"{split}_{suffix}"
                    _require(np.array_equal(arrays[identity],
                                            canonical.arrays[identity]),
                             f"raw-context identity drifts from carriers: "
                             f"{cohort}/{identity}")
            _require({item.name for item in directory.iterdir()} == {
                "manifest.json", artifact.name},
                "raw-context reference directory contains unexpected files")
            reference = ValidatedRawContextReference(
                cohort=cohort, seed=seed,
                manifest_sha256=_sha256_file(manifest_path),
                bank_sha256=value["bank_manifest_sha256"],
                measurements_path=artifact,
                feature_dimension=feature_dimension,
                arrays=arrays)
            references[(cohort, seed)] = reference
            digest_rows.append({
                "cohort": cohort, "seed": seed,
                "manifest_sha256": reference.manifest_sha256,
                "bank_manifest_sha256": reference.bank_sha256,
                "feature_dimension": reference.feature_dimension,
            })
            producer_records.append({
                "cohort": cohort,
                "seed": seed,
                "manifest_sha256": reference.manifest_sha256,
                "artifact_sha256": _sha256_file(artifact),
                "bank_manifest_sha256": reference.bank_sha256,
            })
    _require(producer_summary["records_sha256"] ==
             _sha256_json(producer_records),
             "raw-context producer record digest differs")
    return references, _sha256_json(digest_rows)


def _record_label_reveal_receipt(grid: ValidatedGrid, registry_path: Path,
                                 output_root: Path, *,
                                 raw_context_sha256: str | None = None,
                                 execution_deck_registry: Path | None = None
                                 ) -> Path:
    _require(registry_path.is_file() and not registry_path.is_symlink(),
             "sealed label-registry manifest is missing or unsafe")
    if execution_deck_registry is not None:
        _require(execution_deck_registry.is_file()
                 and not execution_deck_registry.is_symlink(),
                 "sealed execution-deck registry is missing or unsafe")
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_path = output_root / "label_reveal_receipt.json"
    _require(not receipt_path.exists(),
             "label-reveal receipt already exists; refusing a second reveal")
    receipt = {
        "schema": SCHEMA,
        "study": "sage-mem-v1",
        "stage": "label-reveal",
        "status": "authorized-after-complete-phase-a-grid",
        "complete_grid_validated_before_label_reveal": True,
        "formal_test_labels_read_before_receipt": False,
        "development_outcomes_read": False,
        "phase_a_cells": grid.contract.total_cells,
        "phase_a_grid_sha256": grid.grid_sha256,
        "bank_manifest_sha256": dict(grid.bank_hashes),
        "raw_context_reference": {
            "status": ("validated" if raw_context_sha256 is not None
                       else "not-supplied"),
            "sha256": raw_context_sha256,
            "separate_from_parameter_matched_arms": True,
            "short_context_frames": 3,
            "long_context_frames": 16,
        },
        "execution_deck_registry": ({
            "status": "sealed-supplied",
            "path": str(execution_deck_registry.resolve()),
            "sha256": _sha256_file(execution_deck_registry),
            "size": execution_deck_registry.stat().st_size,
        } if execution_deck_registry is not None else {
            "status": "not-supplied", "path": None,
            "sha256": None, "size": None,
        }),
        "label_registry": {
            "path": str(registry_path.resolve()),
            "sha256": _sha256_file(registry_path),
            "size": registry_path.stat().st_size,
        },
        "recorded_unix_ns": time.time_ns(),
    }
    _atomic_json(receipt_path, receipt)
    return receipt_path


def _validate_reveal_receipt(path: Path, grid: ValidatedGrid,
                             registry_path: Path, *,
                             raw_context_sha256: str | None = None,
                             execution_deck_registry: Path | None = None,
                             ) -> dict[str, Any]:
    value = _read_json(path, "label-reveal receipt")
    _exact_keys(value, {
        "schema", "study", "stage", "status",
        "complete_grid_validated_before_label_reveal",
        "formal_test_labels_read_before_receipt", "development_outcomes_read",
        "phase_a_cells", "phase_a_grid_sha256", "bank_manifest_sha256",
        "raw_context_reference", "execution_deck_registry",
        "label_registry", "recorded_unix_ns",
    }, "label-reveal receipt")
    _require(value["schema"] == SCHEMA and value["study"] == "sage-mem-v1"
             and value["stage"] == "label-reveal"
             and value["status"] ==
             "authorized-after-complete-phase-a-grid"
             and value["complete_grid_validated_before_label_reveal"] is True
             and value["formal_test_labels_read_before_receipt"] is False
             and value["development_outcomes_read"] is False
             and value["phase_a_cells"] == grid.contract.total_cells
             and value["phase_a_grid_sha256"] == grid.grid_sha256
             and value["bank_manifest_sha256"] == dict(grid.bank_hashes),
             "label-reveal receipt is stale or malformed")
    raw = value["raw_context_reference"]
    _require(isinstance(raw, dict), "raw-context receipt is malformed")
    _exact_keys(raw, {
        "status", "sha256", "separate_from_parameter_matched_arms",
        "short_context_frames", "long_context_frames",
    }, "raw-context receipt")
    _require(raw == {
        "status": ("validated" if raw_context_sha256 is not None
                   else "not-supplied"),
        "sha256": raw_context_sha256,
        "separate_from_parameter_matched_arms": True,
        "short_context_frames": 3,
        "long_context_frames": 16,
    }, "raw-context receipt differs from the pre-reveal validation")
    deck = value["execution_deck_registry"]
    expected_deck = ({
        "status": "sealed-supplied",
        "path": str(execution_deck_registry.resolve()),
        "sha256": _sha256_file(execution_deck_registry),
        "size": execution_deck_registry.stat().st_size,
    } if execution_deck_registry is not None else {
        "status": "not-supplied", "path": None,
        "sha256": None, "size": None,
    })
    _require(deck == expected_deck,
             "execution-deck receipt differs from the sealed input")
    registry = value["label_registry"]
    _require(isinstance(registry, dict), "receipt label registry is malformed")
    _exact_keys(registry, {"path", "sha256", "size"},
                "receipt label registry")
    _require(registry["path"] == str(registry_path.resolve())
             and registry["sha256"] == _sha256_file(registry_path)
             and registry["size"] == registry_path.stat().st_size,
             "sealed label registry changed after reveal authorization")
    return value


def _load_consolidated_label_artifact(
        registry_path: Path, cohort: str, record: Mapping[str, Any]
        ) -> dict[str, np.ndarray]:
    artifact = _safe_registry_artifact(
        registry_path.parent, record["artifact"], f"{cohort} sealed labels")
    expected = {
        "formal_test_episode_id",
        "formal_test_native_cluster_id",
        "formal_test_label",
        "consumer_train_episode_id",
        "consumer_train_native_cluster_id",
        "consumer_train_label",
    }
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            _require(set(archive.files) == expected,
                     f"{cohort} sealed-label schema differs")
            return {key: np.asarray(archive[key]).copy() for key in expected}
    except (OSError, ValueError) as error:
        raise SageMemFormalFinalizerError(
            f"cannot open sealed labels for {cohort}") from error


def _normalize_custody_source(
        registry_path: Path, cohort: str, split: str,
        source: Mapping[str, Any], desired_episode: np.ndarray,
        desired_cluster: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select explicit identities from one generic custody vault."""

    _exact_keys(source, {"artifact", "keys"},
                f"{cohort}/{split} custody source")
    keys = source["keys"]
    _require(isinstance(keys, dict), "custody key map must be a mapping")
    _exact_keys(keys, {"episode_id", "native_cluster_id", "label"},
                f"{cohort}/{split} custody keys")
    _require(isinstance(keys["episode_id"], str) and keys["episode_id"]
             and isinstance(keys["label"], str) and keys["label"]
             and (keys["native_cluster_id"] is None
                  or (isinstance(keys["native_cluster_id"], str)
                      and keys["native_cluster_id"])),
             "custody key map is malformed")
    artifact = _safe_registry_artifact(
        registry_path.parent, source["artifact"],
        f"{cohort}/{split} custody vault")
    required = {keys["episode_id"], keys["label"]}
    if keys["native_cluster_id"] is not None:
        required.add(keys["native_cluster_id"])
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            _require(required.issubset(archive.files),
                     f"{cohort}/{split} custody vault omits declared keys")
            episode = np.asarray(archive[keys["episode_id"]]).copy()
            label = np.asarray(archive[keys["label"]]).copy()
            cluster = (episode.copy() if keys["native_cluster_id"] is None
                       else np.asarray(
                           archive[keys["native_cluster_id"]]).copy())
    except (OSError, ValueError) as error:
        raise SageMemFormalFinalizerError(
            f"cannot open {cohort}/{split} custody vault") from error
    _require(all(value.ndim == 1 and len(value) == len(episode)
                 and np.issubdtype(value.dtype, np.integer)
                 for value in (episode, cluster, label)),
             f"{cohort}/{split} custody arrays are not aligned integers")
    _require(len(np.unique(episode)) == len(episode),
             f"{cohort}/{split} custody episode IDs are not unique")
    lookup = {int(identity): index for index, identity in enumerate(episode)}
    _require(all(int(identity) in lookup for identity in desired_episode),
             f"{cohort}/{split} custody vault omits an explicit split ID")
    rows = np.asarray([lookup[int(identity)] for identity in desired_episode],
                      dtype=np.int64)
    selected_episode = episode[rows].astype(np.int64, copy=False)
    selected_cluster = cluster[rows].astype(np.int64, copy=False)
    selected_label = label[rows].astype(np.int64, copy=False)
    _require(np.array_equal(selected_episode, desired_episode)
             and np.array_equal(selected_cluster, desired_cluster),
             f"{cohort}/{split} custody IDs/clusters differ from phase A")
    return selected_episode, selected_cluster, selected_label


def _write_post_reveal_normalization(
        output_root: Path, labels: Mapping[str, LabelSet],
        source_registry: Path, receipt_path: Path,
        grid: ValidatedGrid) -> Path:
    _require(not output_root.exists() or not any(output_root.iterdir()),
             "post-reveal normalization output is not empty")
    output_root.mkdir(parents=True, exist_ok=True)
    cohorts: dict[str, Any] = {}
    for cohort in grid.contract.cohorts:
        path = output_root / f"{cohort}.npz"
        values = labels[cohort]
        digest = _atomic_npz(path, {
            name: np.asarray(getattr(values, name))
            for name in LabelSet.__dataclass_fields__
        })
        os.chmod(path, 0o400)
        cohorts[cohort] = {
            "bank_manifest_sha256": grid.bank_hashes[cohort],
            "classes": grid.contract.classes[cohort],
            "artifact": {"path": path.name, "sha256": digest,
                         "size": path.stat().st_size},
        }
    manifest = {
        "schema": "sage_mem_v1_post_reveal_consolidated_registry_v1",
        "study": "sage-mem-v1",
        "status": "normalized-after-complete-grid-reveal",
        "source_custody_registry_sha256": _sha256_file(source_registry),
        "label_reveal_receipt_sha256": _sha256_file(receipt_path),
        "development_outcomes_read": False,
        "cohorts": cohorts,
    }
    path = output_root / "manifest.json"
    _atomic_json(path, manifest)
    os.chmod(path, 0o400)
    return path


def _load_label_registry(registry_path: Path, receipt_path: Path,
                         grid: ValidatedGrid, *,
                         raw_context_sha256: str | None = None,
                         execution_deck_registry: Path | None = None,
                         normalization_output_root: Path | None = None,
                         ) -> dict[str, LabelSet]:
    # This receipt validation is deliberately the first operation that can be
    # followed by parsing semantic labels.  Before this point the registry was
    # only treated as opaque bytes for hashing.
    _validate_reveal_receipt(
        receipt_path, grid, registry_path,
        raw_context_sha256=raw_context_sha256,
        execution_deck_registry=execution_deck_registry)
    value = _read_json(registry_path, "sealed label registry")
    _exact_keys(value, {
        "schema", "study", "status",
        "labels_available_only_after_complete_phase_a_grid",
        "development_outcomes_read", "cohorts",
    }, "sealed label registry")
    _require(value["schema"] in {
                 LABEL_REGISTRY_SCHEMA, CUSTODY_REGISTRY_SCHEMA}
             and value["study"] == "sage-mem-v1"
             and value["status"] == "sealed"
             and value["labels_available_only_after_complete_phase_a_grid"]
             is True and value["development_outcomes_read"] is False
             and isinstance(value["cohorts"], dict)
             and set(value["cohorts"]) == set(grid.contract.cohorts),
             "sealed label registry identity/boundary differs")
    result: dict[str, LabelSet] = {}
    custody_mode = value["schema"] == CUSTODY_REGISTRY_SCHEMA
    for cohort in grid.contract.cohorts:
        record = value["cohorts"][cohort]
        _require(isinstance(record, dict), "cohort label record is malformed")
        _exact_keys(record, {
            "bank_manifest_sha256", "classes",
            ("sources" if custody_mode else "artifact")},
                    f"{cohort} label record")
        _require(record["bank_manifest_sha256"] == grid.bank_hashes[cohort]
                 and record["classes"] == grid.contract.classes[cohort],
                 f"{cohort} label registry does not match the exact bank")
        canonical = next(
            cell for key, cell in grid.cells.items() if key[0] == cohort)
        if custody_mode:
            sources = record["sources"]
            _require(isinstance(sources, dict),
                     "custody split sources must be a mapping")
            _exact_keys(sources, {"formal_test", "consumer_train"},
                        f"{cohort} custody sources")
            arrays: dict[str, np.ndarray] = {}
            for split in ("formal_test", "consumer_train"):
                episode, cluster, label = _normalize_custody_source(
                    registry_path, cohort, split, sources[split],
                    canonical.arrays[f"{split}_episode_id"][0],
                    canonical.arrays[f"{split}_native_cluster_id"][0])
                arrays[f"{split}_episode_id"] = episode
                arrays[f"{split}_native_cluster_id"] = cluster
                arrays[f"{split}_label"] = label
        else:
            arrays = _load_consolidated_label_artifact(
                registry_path, cohort, record)
        test_count = grid.contract.formal_test_rows[cohort]
        consumer_count = grid.contract.consumer_train_rows[cohort]
        for split, count in (("formal_test", test_count),
                             ("consumer_train", consumer_count)):
            episode = arrays[f"{split}_episode_id"]
            cluster = arrays[f"{split}_native_cluster_id"]
            labels = arrays[f"{split}_label"]
            _require(all(value.ndim == 1 and len(value) == count
                         and np.issubdtype(value.dtype, np.integer)
                         for value in (episode, cluster, labels)),
                     f"{cohort}/{split} sealed labels are unaligned")
            _require(np.all((labels >= 0)
                            & (labels < grid.contract.classes[cohort])),
                     f"{cohort}/{split} sealed class is out of range")
            _require(np.array_equal(
                episode, canonical.arrays[f"{split}_episode_id"][0])
                and np.array_equal(
                    cluster,
                    canonical.arrays[f"{split}_native_cluster_id"][0]),
                f"{cohort}/{split} labels drift from phase-A identities")
            _validate_cluster_multiplicity(
                episode, cluster, grid.contract.variants_per_cluster[cohort],
                f"{cohort}/{split}", labels=labels,
                classes=grid.contract.classes[cohort])
        result[cohort] = LabelSet(**arrays)
    if custody_mode and normalization_output_root is not None:
        _write_post_reveal_normalization(
            normalization_output_root, result, registry_path, receipt_path,
            grid)
    return result


def normalize_custody_vaults_after_reveal(
        custody_registry_manifest: str | Path,
        label_reveal_receipt: str | Path, grid: ValidatedGrid,
        output_root: str | Path, *,
        raw_context_sha256: str | None = None,
        execution_deck_registry: str | Path | None = None) -> Path:
    """Normalize generic custody vaults using explicit Phase-A split IDs.

    The reveal receipt is authenticated *before* the custody descriptor or any
    vault is parsed.  Vault key names and formal_test/consumer_train sources
    are explicit in the descriptor, so neither LeWM nor DINO host modules are
    imported here.
    """

    registry = Path(custody_registry_manifest).resolve()
    receipt = Path(label_reveal_receipt).resolve()
    output = Path(output_root).resolve()
    _load_label_registry(
        registry, receipt, grid, raw_context_sha256=raw_context_sha256,
        execution_deck_registry=(Path(execution_deck_registry).resolve()
                                 if execution_deck_registry is not None
                                 else None),
        normalization_output_root=output)
    manifest = output / "manifest.json"
    _require(manifest.is_file(), "custody normalization did not publish")
    return manifest


def _load_execution_decks(
        deck_registry_path: Path, label_registry_path: Path,
        reveal_receipt_path: Path, grid: ValidatedGrid,
        labels: Mapping[str, LabelSet], *,
        raw_context_sha256: str | None = None) -> dict[str, ExecutionDeck]:
    """Open label-free execution cubes and bind their target axis post-reveal.

    A pre-formal producer may execute every selected class against every
    possible physical target, but it must not know which target belongs to a
    formal row.  Consequently the sealed artifact is a three-dimensional
    ``row x selected-class x true-target-class`` cube.  Only this function,
    after the durable Phase-A reveal receipt and sealed-label validation,
    selects the final axis.  This prevents the execution producer from
    reconstructing or opening formal semantic labels.
    """

    _validate_reveal_receipt(
        reveal_receipt_path, grid, label_registry_path,
        raw_context_sha256=raw_context_sha256,
        execution_deck_registry=deck_registry_path)
    value = _read_json(deck_registry_path, "sealed execution-deck registry")
    _exact_keys(value, {
        "schema", "study", "status",
        "available_only_after_complete_phase_a_grid",
        "development_outcomes_read", "cohorts", "unavailable_cohorts",
    }, "execution-deck registry")
    _require(value["schema"] == EXECUTION_DECK_REGISTRY_SCHEMA
             and value["study"] == "sage-mem-v1"
             and value["status"] == "sealed"
             and value["available_only_after_complete_phase_a_grid"] is True
             and value["development_outcomes_read"] is False
             and isinstance(value["cohorts"], dict)
             and isinstance(value["unavailable_cohorts"], dict),
             "execution-deck registry identity/boundary differs")
    supplied = set(value["cohorts"])
    unavailable = set(value["unavailable_cohorts"])
    _require(not supplied.intersection(unavailable)
             and supplied.union(unavailable) == set(grid.contract.cohorts),
             "execution registry must classify every cohort exactly once")

    # Unsupported cohorts are explicit immutable facts, not silently omitted
    # optional analyses.  They can never enter the eligible-deck count.
    for cohort, record in value["unavailable_cohorts"].items():
        _require(isinstance(record, dict),
                 f"unavailable execution record is malformed: {cohort}")
        _exact_keys(record, {
            "status", "bank_manifest_sha256", "reason_code", "receipt",
        }, f"{cohort} unavailable execution record")
        reason = record["reason_code"]
        _require(record["status"] == "unavailable"
                 and record["bank_manifest_sha256"] ==
                 grid.bank_hashes[cohort]
                 and isinstance(reason, str) and reason
                 and len(reason) <= 160,
                 f"unavailable execution identity differs: {cohort}")
        receipt_path = _safe_registry_artifact(
            deck_registry_path.parent, record["receipt"],
            f"{cohort} unavailable execution receipt")
        receipt = _read_json(
            receipt_path, f"{cohort} unavailable execution receipt")
        _exact_keys(receipt, {
            "schema", "study", "status", "cohort", "reason_code",
            "bank_manifest_sha256", "formal_labels_read",
            "development_outcomes_read",
        }, f"{cohort} unavailable execution receipt")
        _require(receipt["schema"] == EXECUTION_UNAVAILABLE_RECEIPT_SCHEMA
                 and receipt["study"] == "sage-mem-v1"
                 and receipt["status"] == "unavailable"
                 and receipt["cohort"] == cohort
                 and receipt["reason_code"] == reason
                 and receipt["bank_manifest_sha256"] ==
                 grid.bank_hashes[cohort]
                 and receipt["formal_labels_read"] is False
                 and receipt["development_outcomes_read"] is False,
                 f"unavailable execution receipt differs: {cohort}")

    decks: dict[str, ExecutionDeck] = {}
    for cohort, record in value["cohorts"].items():
        _require(isinstance(record, dict),
                 f"execution deck record is malformed: {cohort}")
        _exact_keys(record, {
            "bank_manifest_sha256", "classes", "controller",
            "eligibility_gate", "artifact", "replay_receipt",
        }, f"{cohort} execution deck")
        _require(record["bank_manifest_sha256"] == grid.bank_hashes[cohort]
                 and record["classes"] == grid.contract.classes[cohort],
                 f"execution deck differs from exact bank: {cohort}")
        controller = record["controller"]
        _require(isinstance(controller, dict),
                 "execution controller record must be a mapping")
        _exact_keys(controller, {
            "controller_identity_sha256", "implementation_sha256",
            "physics_sha256", "pinned", "arm_identity_input", "input",
        }, f"{cohort} execution controller")
        _require(all(_is_sha256(controller[key]) for key in (
            "controller_identity_sha256", "implementation_sha256",
            "physics_sha256"))
            and controller["pinned"] is True
            and controller["arm_identity_input"] is False
            and controller["input"] == "predicted_class_only",
            f"execution controller is not pinned/arm-blind: {cohort}")
        gate = record["eligibility_gate"]
        _require(isinstance(gate, dict),
                 "execution eligibility gate must be a mapping")
        _exact_keys(gate, {"metric", "operator", "threshold",
                           "preregistered"},
                    f"{cohort} execution eligibility gate")
        threshold = gate["threshold"]
        _require(gate["metric"] == "mean_oracle_success"
                 and gate["operator"] == ">="
                 and gate["preregistered"] is True
                 and isinstance(threshold, (int, float))
                 and not isinstance(threshold, bool)
                 and math.isfinite(float(threshold))
                 and 0.0 <= float(threshold) <= 1.0,
                 f"execution eligibility gate changed: {cohort}")
        artifact = _safe_registry_artifact(
            deck_registry_path.parent, record["artifact"],
            f"{cohort} execution deck")
        replay_path = _safe_registry_artifact(
            deck_registry_path.parent, record["replay_receipt"],
            f"{cohort} execution replay receipt")
        replay = _read_json(replay_path, f"{cohort} execution replay receipt")
        _exact_keys(replay, {
            "schema", "study", "status", "cohort",
            "bank_manifest_sha256", "formal_labels_read",
            "development_outcomes_read", "controller_identity_sha256",
            "rows", "classes", "native_clusters", "executions",
            "replayed_executions", "deterministic_replay_fidelity",
            "execution_endpoint",
        }, f"{cohort} execution replay receipt")
        _require(replay["schema"] == EXECUTION_REPLAY_RECEIPT_SCHEMA
                 and replay["study"] == "sage-mem-v1"
                 and replay["status"] == "sealed-label-free"
                 and replay["cohort"] == cohort
                 and replay["bank_manifest_sha256"] ==
                 grid.bank_hashes[cohort]
                 and replay["formal_labels_read"] is False
                 and replay["development_outcomes_read"] is False
                 and replay["controller_identity_sha256"] == controller[
                     "controller_identity_sha256"]
                 and replay["rows"] == grid.contract.formal_test_rows[cohort]
                 and replay["classes"] == grid.contract.classes[cohort]
                 and isinstance(replay["native_clusters"], int)
                 and replay["native_clusters"] > 0
                 and isinstance(replay["executions"], int)
                 and replay["executions"] > 0
                 and replay["replayed_executions"] == replay["executions"]
                 and replay["deterministic_replay_fidelity"] == 1.0
                 and isinstance(replay["execution_endpoint"], str)
                 and replay["execution_endpoint"],
                 f"execution replay receipt differs: {cohort}")
        expected = {
            "formal_test_episode_id",
            "formal_test_native_cluster_id",
            "selected_class_by_true_target_success",
            "deterministic_random_class",
        }
        try:
            with np.load(artifact, allow_pickle=False) as archive:
                _require(set(archive.files) == expected,
                         f"execution deck array schema differs: {cohort}")
                arrays = {name: np.asarray(archive[name]).copy()
                          for name in expected}
        except (OSError, ValueError) as error:
            raise SageMemFormalFinalizerError(
                f"cannot open execution deck: {cohort}") from error
        count = grid.contract.formal_test_rows[cohort]
        classes = grid.contract.classes[cohort]
        episode = arrays["formal_test_episode_id"]
        cluster = arrays["formal_test_native_cluster_id"]
        cube = arrays["selected_class_by_true_target_success"]
        random_class = arrays["deterministic_random_class"]
        _require(episode.shape == cluster.shape == random_class.shape ==
                 (count,)
                 and cube.shape == (count, classes, classes)
                 and all(np.issubdtype(array.dtype, np.integer)
                         or np.issubdtype(array.dtype, np.bool_)
                         for array in (episode, cluster, cube, random_class)),
                 f"execution deck arrays are unaligned: {cohort}")
        _require(np.isin(cube, (0, 1)).all()
                 and np.all((random_class >= 0) & (random_class < classes)),
                 f"execution deck outcomes are malformed: {cohort}")
        canonical = grid.cells[(cohort, grid.contract.arms[0],
                                grid.contract.seeds[0])]
        _require(np.array_equal(
            episode, canonical.arrays["formal_test_episode_id"][0])
            and np.array_equal(
                cluster,
                canonical.arrays["formal_test_native_cluster_id"][0]),
                f"execution deck identities drift from phase A: {cohort}")
        _validate_cluster_multiplicity(
            episode, cluster, grid.contract.variants_per_cluster[cohort],
            f"{cohort}/execution-deck")
        variants = grid.contract.variants_per_cluster[cohort]
        if variants > 1:
            for native in np.unique(cluster):
                selected = np.flatnonzero(cluster == native)
                _require(all(np.array_equal(cube[selected[0]], cube[index])
                             for index in selected[1:])
                         and np.all(random_class[selected] ==
                                    random_class[selected[0]]),
                         f"execution cube/random policy varies within native "
                         f"cluster: {cohort}/{int(native)}")

        cohort_labels = labels.get(cohort)
        _require(cohort_labels is not None
                 and np.array_equal(
                     episode, cohort_labels.formal_test_episode_id)
                 and np.array_equal(
                     cluster, cohort_labels.formal_test_native_cluster_id),
                 f"execution deck does not align with revealed labels: "
                 f"{cohort}")
        truth = np.asarray(
            cohort_labels.formal_test_label, dtype=np.int64)
        _require(truth.shape == (count,)
                 and np.all((truth >= 0) & (truth < classes)),
                 f"revealed execution labels are malformed: {cohort}")
        row = np.arange(count, dtype=np.int64)
        # This is the first true-target-axis access.  The producer never emits
        # these target-conditioned two-dimensional outcomes.
        success = cube[row[:, None],
                       np.arange(classes, dtype=np.int64)[None, :],
                       truth[:, None]]
        oracle = cube[row, truth, truth]
        random_success = cube[row, random_class, truth]
        decks[cohort] = ExecutionDeck(
            cohort=cohort,
            controller_identity_sha256=controller[
                "controller_identity_sha256"],
            threshold=float(threshold),
            episode_id=episode.astype(np.int64, copy=False),
            native_cluster_id=cluster.astype(np.int64, copy=False),
            class_conditioned_success=success.astype(np.uint8, copy=False),
            oracle_success=oracle.astype(np.uint8, copy=False),
            random_class=random_class.astype(np.int64, copy=False),
            random_success=random_success.astype(np.uint8, copy=False),
        )
    return decks


def _evaluate_execution_decks(
        decks: Mapping[str, ExecutionDeck], grid: ValidatedGrid,
        predictions: Mapping[tuple[str, str, int], Mapping[str, np.ndarray]],
        output_root: Path) -> tuple[
            dict[tuple[str, str, int], dict[str, np.ndarray]],
            dict[str, dict[str, Any]], int]:
    results: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    cohort_status: dict[str, dict[str, Any]] = {}
    eligible_count = 0
    for cohort, deck in decks.items():
        receipt_path = output_root / "execution" / cohort / "receipt.json"
        common = {
            "schema": SCHEMA,
            "study": "sage-mem-v1",
            "stage": "external-execution",
            "cohort": cohort,
            "controller_identity_sha256": deck.controller_identity_sha256,
            "controller_pinned": True,
            "arm_identity_used": False,
            "input": "predicted_class_only",
            "oracle_success": deck.oracle_rate,
            "random_success": deck.random_rate,
            "eligibility_metric": "mean_oracle_success",
            "eligibility_operator": ">=",
            "eligibility_threshold": deck.threshold,
        }
        if not deck.eligible:
            receipt = {
                **common,
                "status": "skipped-oracle-gate",
                "eligible": False,
                "skip_reason": "oracle-success-below-preregistered-threshold",
                "computed_cells": 0,
            }
            _atomic_json(receipt_path, receipt)
            cohort_status[cohort] = {
                "status": receipt["status"], "eligible": False,
                "receipt_sha256": _sha256_file(receipt_path),
                "oracle_success": deck.oracle_rate,
                "random_success": deck.random_rate,
            }
            continue
        eligible_count += 1
        per_age_summary: dict[str, dict[str, list[float]]] = {}
        row = np.arange(len(deck.episode_id))[None, :]
        for arm in grid.contract.arms:
            for seed in grid.contract.seeds:
                key = (cohort, arm, seed)
                streams: dict[str, np.ndarray] = {}
                for stream in ("full", "reset", "prior"):
                    predicted = np.asarray(predictions[key][stream],
                                           dtype=np.int64)
                    _require(predicted.shape == (
                        len(grid.contract.ages), len(deck.episode_id)),
                        f"execution prediction shape differs: {key}/{stream}")
                    executed = deck.class_conditioned_success[
                        row, predicted].astype(np.uint8)
                    streams[stream] = executed
                    per_age_summary.setdefault(arm, {}).setdefault(
                        stream, []).extend(
                            map(float, np.mean(executed, axis=1)))
                results[key] = streams
        receipt = {
            **common,
            "status": "computed-class-conditioned-arm-blind",
            "eligible": True,
            "skip_reason": None,
            "computed_cells": len(grid.contract.arms) * len(
                grid.contract.seeds),
            "ages": list(grid.contract.ages),
            "per_arm_seed_age_values_sha256": _sha256_json(
                per_age_summary),
        }
        _atomic_json(receipt_path, receipt)
        cohort_status[cohort] = {
            "status": receipt["status"], "eligible": True,
            "receipt_sha256": _sha256_file(receipt_path),
            "oracle_success": deck.oracle_rate,
            "random_success": deck.random_rate,
        }
    return results, cohort_status, eligible_count


def _ridge_consumer(train_x: np.ndarray, train_y: np.ndarray,
                    test_features: Iterable[np.ndarray], classes: int,
                    ridge: float = 1e-3
                    ) -> tuple[list[np.ndarray], str]:
    # LSQR works through matrix-vector products and never materializes the
    # d-by-d Gram matrix.  This is essential for DINO spatial pyramids
    # (d ~= 8,064).  float32 plus in-place scaling also keeps the pooled
    # 12-arm consumer set within a bounded memory footprint.
    try:
        from sklearn.linear_model import RidgeClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise SageMemFormalFinalizerError(
            "scikit-learn is required for the bounded-memory shared consumer"
        ) from error
    x = np.asarray(train_x, dtype=np.float32)
    y = np.asarray(train_y, dtype=np.int64)
    _require(x.ndim == 2 and len(x) == len(y) and len(x) > 0
             and np.isfinite(x).all(),
             "pooled consumer training data is malformed")
    _require(set(np.unique(y).tolist()) == set(range(classes)),
             "pooled consumer training omits a class")
    scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
    normalized = scaler.fit_transform(x)
    classifier = RidgeClassifier(
        alpha=float(ridge), fit_intercept=True, solver="lsqr",
        tol=1e-6, max_iter=5_000)
    try:
        classifier.fit(normalized, y)
    except (FloatingPointError, ValueError) as error:
        raise SageMemFormalFinalizerError(
            "centralized shared consumer fit failed") from error
    _require(np.array_equal(classifier.classes_, np.arange(classes)),
             "shared consumer class ordering changed")
    predictions: list[np.ndarray] = []
    for features in test_features:
        values = np.array(features, dtype=np.float32, copy=True)
        _require(values.ndim == 2 and values.shape[1] == x.shape[1]
                 and np.isfinite(values).all(),
                 "consumer test feature shape differs")
        predictions.append(classifier.predict(
            scaler.transform(values)).astype(np.int64))
    digest = hashlib.sha256()
    for value in (scaler.mean_, scaler.scale_, classifier.coef_,
                  np.atleast_1d(classifier.intercept_)):
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(_canonical_json({
        "classes": classes, "ridge": ridge,
        "solver": "lsqr", "tol": 1e-6, "max_iter": 5_000,
        "arm_identity_used": False,
        "formal_test_labels_used": False,
    }).encode("utf-8"))
    return predictions, digest.hexdigest()


def _centralized_predictions(
        grid: ValidatedGrid, labels: Mapping[str, LabelSet],
        output_root: Path) -> tuple[
            dict[tuple[str, str, int], dict[str, np.ndarray]],
            dict[tuple[str, str, int], str]]:
    predictions: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    consumer_hash_for_cell: dict[tuple[str, str, int], str] = {}
    for cohort in grid.contract.cohorts:
        representation = grid.cells[(cohort, grid.contract.arms[0],
                                     grid.contract.seeds[0])].representation
        if representation == "predicted_labels":
            for arm in grid.contract.arms:
                for seed in grid.contract.seeds:
                    cell = grid.cells[(cohort, arm, seed)]
                    predictions[(cohort, arm, seed)] = {
                        "full": cell.arrays["formal_test_full_pred"],
                        "reset": cell.arrays["formal_test_reset_pred"],
                        "prior": cell.arrays["formal_test_prior_pred"],
                    }
                    consumer_hash_for_cell[(cohort, arm, seed)] = str(
                        cell.shared_consumer_sha256)
            continue

        cohort_labels = labels[cohort]

        def feature_age(cell: ValidatedCell, name: str,
                        age_index: int) -> np.ndarray:
            try:
                with np.load(cell.measurements_path,
                             allow_pickle=False) as archive:
                    return np.asarray(
                        archive[name][age_index], dtype=np.float32).copy()
            except (OSError, ValueError, KeyError, IndexError) as error:
                raise SageMemFormalFinalizerError(
                    f"cannot stream feature {name} from "
                    f"{cell.measurements_path}") from error

        for seed in grid.contract.seeds:
            age_hashes: list[str] = []
            per_arm = {
                arm: {name: [] for name in ("full", "reset", "prior")}
                for arm in grid.contract.arms
            }
            for age_index, evidence_age in enumerate(grid.contract.ages):
                rows = grid.contract.consumer_train_rows[cohort]
                dimension = grid.cells[(
                    cohort, grid.contract.arms[0], seed)].feature_dimension
                _require(dimension is not None,
                         "shared consumer feature dimension is absent")
                pooled_x = np.empty(
                    (rows * len(grid.contract.arms), int(dimension)),
                    dtype=np.float32)
                for arm_index, arm in enumerate(grid.contract.arms):
                    pooled_x[arm_index * rows:(arm_index + 1) * rows] = \
                        feature_age(
                            grid.cells[(cohort, arm, seed)],
                            "consumer_train_full_features", age_index)
                pooled_y = np.tile(
                    cohort_labels.consumer_train_label,
                    len(grid.contract.arms))
                locations: list[tuple[str, str]] = []
                for arm in grid.contract.arms:
                    for stream in ("full", "reset", "prior"):
                        locations.append((arm, stream))
                test_blocks = (
                    feature_age(
                        grid.cells[(cohort, arm, seed)],
                        f"formal_test_{stream}_features", age_index)
                    for arm, stream in locations)
                values, consumer_hash = _ridge_consumer(
                    pooled_x, pooled_y, test_blocks,
                    grid.contract.classes[cohort])
                age_hashes.append(consumer_hash)
                for (arm, stream), value in zip(locations, values):
                    per_arm[arm][stream].append(value)
            shared_hash = _sha256_json({
                "cohort": cohort, "seed": seed,
                "ages": list(grid.contract.ages),
                "age_model_sha256": age_hashes,
                "pooled_arms": list(grid.contract.arms),
                "arm_identity_used": False,
                "fit_split": "consumer_train",
                "formal_test_labels_used": False,
            })
            receipt = {
                "schema": SCHEMA,
                "study": "sage-mem-v1",
                "stage": "shared-consumer",
                "status": "fit-on-pooled-consumer-train",
                "cohort": cohort,
                "seed": seed,
                "ages": list(grid.contract.ages),
                "pooled_arms": list(grid.contract.arms),
                "training_rows_per_age": (
                    grid.contract.consumer_train_rows[cohort]
                    * len(grid.contract.arms)),
                "arm_identity_used": False,
                "formal_test_labels_used": False,
                "age_model_sha256": age_hashes,
                "shared_consumer_sha256": shared_hash,
            }
            receipt_path = (output_root / "consumers" / cohort
                            / f"seed-{seed}.json")
            _atomic_json(receipt_path, receipt)
            for arm in grid.contract.arms:
                key = (cohort, arm, seed)
                predictions[key] = {
                    stream: np.stack(per_arm[arm][stream], axis=0)
                    for stream in ("full", "reset", "prior")
                }
                consumer_hash_for_cell[key] = shared_hash
    return predictions, consumer_hash_for_cell


def _centralized_raw_context_predictions(
        references: Mapping[tuple[str, int], ValidatedRawContextReference],
        labels: Mapping[str, LabelSet], grid: ValidatedGrid,
        output_root: Path) -> tuple[
            dict[tuple[str, int], dict[str, np.ndarray]],
            dict[tuple[str, int], str]]:
    """Fit exactly one post-reveal short/long consumer per cohort and seed."""

    predictions: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    hashes: dict[tuple[str, int], str] = {}
    for key, reference in references.items():
        cohort, seed = key
        try:
            with np.load(reference.measurements_path,
                         allow_pickle=False) as archive:
                consumer_short = np.asarray(
                    archive["consumer_train_short_features"],
                    dtype=np.float32)
                consumer_long = np.asarray(
                    archive["consumer_train_long_features"],
                    dtype=np.float32)
                test_short = np.asarray(
                    archive["formal_test_short_features"],
                    dtype=np.float32)
                test_long = np.asarray(
                    archive["formal_test_long_features"],
                    dtype=np.float32)
        except (OSError, ValueError, KeyError) as error:
            raise SageMemFormalFinalizerError(
                f"cannot stream raw-context features: "
                f"{reference.measurements_path}") from error
        ages = len(grid.contract.ages)
        dimension = reference.feature_dimension
        _require(all(value.ndim == 3 and value.shape[0] == ages
                     and value.shape[2] == dimension
                     and np.isfinite(value).all()
                     for value in (
                         consumer_short, consumer_long, test_short, test_long)),
                 "raw-context feature shape changed after reveal")
        train_x = np.concatenate([
            consumer_short.reshape(-1, dimension),
            consumer_long.reshape(-1, dimension),
        ], axis=0)
        repeated_labels = np.tile(
            labels[cohort].consumer_train_label, ages)
        train_y = np.concatenate([repeated_labels, repeated_labels])
        test_count = grid.contract.formal_test_rows[cohort]
        values, consumer_hash = _ridge_consumer(
            train_x, train_y, (
                test_short.reshape(-1, dimension),
                test_long.reshape(-1, dimension),
            ), grid.contract.classes[cohort])
        predictions[key] = {
            "short": values[0].reshape(ages, test_count),
            "long": values[1].reshape(ages, test_count),
        }
        shared_hash = _sha256_json({
            "cohort": cohort,
            "seed": seed,
            "model_sha256": consumer_hash,
            "contexts": ["short-3", "long-16"],
            "ages_pooled": list(grid.contract.ages),
            "fit_split": "consumer_train",
            "arm_identity_used": False,
            "context_identity_used": False,
            "formal_test_labels_used": False,
        })
        hashes[key] = shared_hash
        receipt = {
            "schema": SCHEMA,
            "study": "sage-mem-v1",
            "stage": "raw-context-shared-consumer",
            "status": "fit-after-complete-grid-reveal",
            "cohort": cohort,
            "seed": seed,
            "contexts": ["short-3", "long-16"],
            "ages_pooled": list(grid.contract.ages),
            "training_rows": int(len(train_y)),
            "feature_dimension": dimension,
            "arm_identity_used": False,
            "context_identity_used": False,
            "formal_test_labels_used": False,
            "model_sha256": consumer_hash,
            "shared_consumer_sha256": shared_hash,
        }
        _atomic_json(
            output_root / "raw_context_consumers" / cohort
            / f"seed-{seed}.json", receipt)
    return predictions, hashes


def _finalize_with_contract(
        phase_a_root: str | Path, sealed_registry_manifest: str | Path,
        output_root: str | Path, contract: _GridContract, *,
        raw_context_root: str | Path | None = None,
        execution_deck_registry: str | Path | None = None,
        require_at_least_two_eligible_execution_cohorts: bool = False,
        ) -> dict[str, Any]:
    """Internal finalizer used by production and compact contract tests."""

    output_root = Path(output_root).resolve()
    _require(not output_root.exists() or not any(output_root.iterdir()),
             f"finalizer output is not empty: {output_root}")
    # Phase boundary: the following call validates every expected cell and no
    # label-registry content is opened before it returns successfully.
    grid = _validate_complete_grid(phase_a_root, contract)
    raw_references: dict[
        tuple[str, int], ValidatedRawContextReference] = {}
    raw_context_sha256: str | None = None
    if raw_context_root is not None:
        raw_references, raw_context_sha256 = _validate_raw_context_references(
            raw_context_root, grid)
    registry_path = Path(sealed_registry_manifest).resolve()
    deck_registry_path = (Path(execution_deck_registry).resolve()
                          if execution_deck_registry is not None else None)
    receipt_path = _record_label_reveal_receipt(
        grid, registry_path, output_root,
        raw_context_sha256=raw_context_sha256,
        execution_deck_registry=deck_registry_path)
    # Semantic labels may be parsed only below this durable receipt.
    labels = _load_label_registry(
        registry_path, receipt_path, grid,
        raw_context_sha256=raw_context_sha256,
        execution_deck_registry=deck_registry_path,
        normalization_output_root=(output_root / "normalized_label_registry"))
    predictions, consumer_hashes = _centralized_predictions(
        grid, labels, output_root)
    raw_predictions, raw_consumer_hashes = \
        _centralized_raw_context_predictions(
            raw_references, labels, grid, output_root)
    decks = (_load_execution_decks(
        deck_registry_path, registry_path, receipt_path, grid, labels,
        raw_context_sha256=raw_context_sha256)
        if deck_registry_path is not None else {})
    execution_results, execution_status, eligible_execution_cohorts = \
        _evaluate_execution_decks(decks, grid, predictions, output_root)
    if require_at_least_two_eligible_execution_cohorts:
        _require(eligible_execution_cohorts >= 2,
                 "program execution gate requires at least two eligible "
                 "cohorts")

    finalized_records: list[dict[str, Any]] = []
    for cohort in contract.cohorts:
        cohort_labels = labels[cohort]
        label_by_age = np.repeat(
            cohort_labels.formal_test_label[None, :],
            len(contract.ages), axis=0)
        for arm in contract.arms:
            for seed in contract.seeds:
                key = (cohort, arm, seed)
                cell = grid.cells[key]
                pred = predictions[key]
                arrays = {
                    "formal_test_episode_id":
                        cell.arrays["formal_test_episode_id"],
                    "formal_test_native_cluster_id":
                        cell.arrays["formal_test_native_cluster_id"],
                    "formal_test_evidence_age":
                        cell.arrays["formal_test_evidence_age"],
                    "formal_test_label": label_by_age,
                    "formal_test_full_pred": pred["full"],
                    "formal_test_reset_pred": pred["reset"],
                    "formal_test_prior_pred": pred["prior"],
                    "formal_test_full_correct":
                        (pred["full"] == label_by_age).astype(np.uint8),
                    "formal_test_reset_correct":
                        (pred["reset"] == label_by_age).astype(np.uint8),
                    "formal_test_prior_correct":
                        (pred["prior"] == label_by_age).astype(np.uint8),
                    "formal_test_full_mse":
                        cell.arrays["formal_test_full_mse"],
                    "formal_test_reset_mse":
                        cell.arrays["formal_test_reset_mse"],
                    "formal_test_prior_mse":
                        cell.arrays["formal_test_prior_mse"],
                }
                if key in execution_results:
                    for stream in ("full", "reset", "prior"):
                        arrays[f"formal_test_{stream}_execution_success"] = \
                            execution_results[key][stream]
                    execution_manifest = {
                        "status": execution_status[cohort]["status"],
                        "eligible": True,
                        "receipt_sha256": execution_status[cohort][
                            "receipt_sha256"],
                        "arm_identity_used": False,
                        "ages": list(contract.ages),
                        "per_age_success": {
                            stream: list(map(float, np.mean(
                                execution_results[key][stream], axis=1)))
                            for stream in ("full", "reset", "prior")
                        },
                    }
                elif cohort in execution_status:
                    execution_manifest = {
                        "status": execution_status[cohort]["status"],
                        "eligible": False,
                        "receipt_sha256": execution_status[cohort][
                            "receipt_sha256"],
                        "arm_identity_used": False,
                        "ages": list(contract.ages),
                        "per_age_success": None,
                    }
                else:
                    execution_manifest = {
                        "status": "not-supplied", "eligible": None,
                        "receipt_sha256": None,
                        "arm_identity_used": False,
                        "ages": list(contract.ages),
                        "per_age_success": None,
                    }
                destination = (output_root / "cells" / cohort / arm
                               / f"seed-{seed}")
                artifact = destination / "finalized_results.npz"
                artifact_hash = _atomic_npz(artifact, arrays)
                manifest = {
                    "schema": SCHEMA,
                    "study": "sage-mem-v1",
                    "stage": "formal-finalized",
                    "status": "complete",
                    "cohort": cohort,
                    "arm": arm,
                    "seed": seed,
                    "ages": list(contract.ages),
                    "phase_a_grid_sha256": grid.grid_sha256,
                    "phase_a_manifest_sha256": cell.manifest_sha256,
                    "bank_manifest_sha256": cell.bank_sha256,
                    "label_registry_sha256": _sha256_file(registry_path),
                    "label_reveal_receipt_sha256": _sha256_file(receipt_path),
                    "shared_arm_blind_consumer_sha256": consumer_hashes[key],
                    "native_cluster_id_preserved": True,
                    "counterfactual_variants_per_native_cluster":
                        contract.variants_per_cluster[cohort],
                    "execution": execution_manifest,
                    "artifact": {
                        "path": artifact.name,
                        "sha256": artifact_hash,
                        "size": artifact.stat().st_size,
                    },
                }
                _atomic_json(destination / "manifest.json", manifest)
                finalized_records.append({
                    "cohort": cohort, "arm": arm, "seed": seed,
                    "artifact_sha256": artifact_hash,
                    "consumer_sha256": consumer_hashes[key],
                })

    raw_records: list[dict[str, Any]] = []
    for (cohort, seed), reference in raw_references.items():
        label_by_age = np.repeat(
            labels[cohort].formal_test_label[None, :],
            len(contract.ages), axis=0)
        arrays = {
            "formal_test_episode_id":
                reference.arrays["formal_test_episode_id"],
            "formal_test_native_cluster_id":
                reference.arrays["formal_test_native_cluster_id"],
            "formal_test_evidence_age":
                reference.arrays["formal_test_evidence_age"],
            "formal_test_label": label_by_age,
            "formal_test_short_pred": raw_predictions[(cohort, seed)][
                "short"],
            "formal_test_long_pred": raw_predictions[(cohort, seed)][
                "long"],
            "formal_test_short_correct": (
                raw_predictions[(cohort, seed)]["short"] ==
                label_by_age).astype(np.uint8),
            "formal_test_long_correct": (
                raw_predictions[(cohort, seed)]["long"] ==
                label_by_age).astype(np.uint8),
        }
        destination = (output_root / "raw_context" / cohort
                       / f"seed-{seed}")
        artifact = destination / "finalized_results.npz"
        artifact_hash = _atomic_npz(artifact, arrays)
        manifest = {
            "schema": SCHEMA,
            "study": "sage-mem-v1",
            "stage": "formal-raw-context-finalized",
            "status": "complete",
            "cohort": cohort,
            "seed": seed,
            "ages": list(contract.ages),
            "short_context_frames": 3,
            "long_context_frames": 16,
            "separate_from_parameter_matched_arms": True,
            "phase_a_grid_sha256": grid.grid_sha256,
            "source_manifest_sha256": reference.manifest_sha256,
            "shared_arm_blind_consumer_sha256":
                raw_consumer_hashes[(cohort, seed)],
            "artifact": {
                "path": artifact.name,
                "sha256": artifact_hash,
                "size": artifact.stat().st_size,
            },
        }
        _atomic_json(destination / "manifest.json", manifest)
        raw_records.append({
            "cohort": cohort, "seed": seed,
            "artifact_sha256": artifact_hash,
            "consumer_sha256": raw_consumer_hashes[(cohort, seed)],
        })
    summary = {
        "schema": SCHEMA,
        "study": "sage-mem-v1",
        "stage": "formal-finalizer",
        "status": "complete",
        "phase_a_cells": contract.total_cells,
        "phase_a_grid_sha256": grid.grid_sha256,
        "label_reveal_receipt_sha256": _sha256_file(receipt_path),
        "label_registry_sha256": _sha256_file(registry_path),
        "development_outcomes_read": False,
        "per_age_results_preserved": True,
        "pointmaze_x4_native_clustering_preserved": (
            contract.variants_per_cluster.get("dinowm_pointmaze_goal") == 4),
        "raw_context_reference": {
            "status": ("complete" if raw_references else "not-supplied"),
            "short_context_frames": 3,
            "long_context_frames": 16,
            "separate_from_parameter_matched_arms": True,
            "references": len(raw_records),
            "records_sha256": (
                _sha256_json(raw_records) if raw_records else None),
        },
        "execution_decks": {
            "status": ("evaluated" if decks else "not-supplied"),
            "supplied_cohorts": sorted(decks),
            "eligible_cohorts": eligible_execution_cohorts,
            "program_requires_at_least_two_eligible_cohorts":
                require_at_least_two_eligible_execution_cohorts,
            "program_gate_passed": (
                eligible_execution_cohorts >= 2
                if require_at_least_two_eligible_execution_cohorts else None),
            "cohort_status": execution_status,
        },
        "finalized_cells_sha256": _sha256_json(finalized_records),
        "finalized_cells": len(finalized_records),
    }
    _atomic_json(output_root / "summary.json", summary)
    return summary


_FINAL_BASE_ARRAY_KEYS = {
    "formal_test_episode_id",
    "formal_test_native_cluster_id",
    "formal_test_evidence_age",
    "formal_test_label",
    "formal_test_full_pred",
    "formal_test_reset_pred",
    "formal_test_prior_pred",
    "formal_test_full_correct",
    "formal_test_reset_correct",
    "formal_test_prior_correct",
    "formal_test_full_mse",
    "formal_test_reset_mse",
    "formal_test_prior_mse",
}
_FINAL_EXECUTION_ARRAY_KEYS = {
    "formal_test_full_execution_success",
    "formal_test_reset_execution_success",
    "formal_test_prior_execution_success",
}


def _validate_feature_consumer_receipt(
        output_root: Path, grid: ValidatedGrid, cohort: str, seed: int) -> str:
    path = output_root / "consumers" / cohort / f"seed-{seed}.json"
    value = _read_json(path, "shared-consumer receipt")
    _exact_keys(value, {
        "schema", "study", "stage", "status", "cohort", "seed", "ages",
        "pooled_arms", "training_rows_per_age", "arm_identity_used",
        "formal_test_labels_used", "age_model_sha256",
        "shared_consumer_sha256",
    }, "shared-consumer receipt")
    _require(value["schema"] == SCHEMA
             and value["study"] == "sage-mem-v1"
             and value["stage"] == "shared-consumer"
             and value["status"] == "fit-on-pooled-consumer-train"
             and value["cohort"] == cohort and value["seed"] == seed
             and value["ages"] == list(grid.contract.ages)
             and value["pooled_arms"] == list(grid.contract.arms)
             and value["training_rows_per_age"] ==
             grid.contract.consumer_train_rows[cohort] * len(
                 grid.contract.arms)
             and value["arm_identity_used"] is False
             and value["formal_test_labels_used"] is False
             and isinstance(value["age_model_sha256"], list)
             and len(value["age_model_sha256"]) == len(grid.contract.ages)
             and all(_is_sha256(item)
                     for item in value["age_model_sha256"])
             and _is_sha256(value["shared_consumer_sha256"]),
             f"shared-consumer receipt differs: {cohort}/seed-{seed}")
    expected = _sha256_json({
        "cohort": cohort, "seed": seed,
        "ages": list(grid.contract.ages),
        "age_model_sha256": value["age_model_sha256"],
        "pooled_arms": list(grid.contract.arms),
        "arm_identity_used": False,
        "fit_split": "consumer_train",
        "formal_test_labels_used": False,
    })
    _require(value["shared_consumer_sha256"] == expected,
             "shared-consumer aggregate hash differs")
    return expected


def _validate_execution_receipts(
        output_root: Path, grid: ValidatedGrid,
        decks: Mapping[str, ExecutionDeck],
        finalized_arrays: Mapping[
            tuple[str, str, int], Mapping[str, np.ndarray]],
        summary_status: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    observed_status: dict[str, dict[str, Any]] = {}
    execution_root = output_root / "execution"
    if not decks:
        _require(not execution_root.exists(),
                 "execution outputs exist without a sealed deck")
        _require(summary_status == {},
                 "summary reports execution cohorts without decks")
        return observed_status
    _require(execution_root.is_dir() and not execution_root.is_symlink()
             and {item.name for item in execution_root.iterdir()}
             == set(decks),
             "execution receipt cohort registry differs")
    for cohort, deck in decks.items():
        cohort_root = execution_root / cohort
        _require(cohort_root.is_dir() and not cohort_root.is_symlink()
                 and {item.name for item in cohort_root.iterdir()} == {
                     "receipt.json"},
                 f"execution receipt directory differs: {cohort}")
        path = cohort_root / "receipt.json"
        value = _read_json(path, "execution receipt")
        common_keys = {
            "schema", "study", "stage", "status", "cohort",
            "controller_identity_sha256", "controller_pinned",
            "arm_identity_used", "input", "oracle_success",
            "random_success", "eligibility_metric", "eligibility_operator",
            "eligibility_threshold", "eligible", "skip_reason",
            "computed_cells",
        }
        expected_keys = (common_keys | {
            "ages", "per_arm_seed_age_values_sha256"}
            if deck.eligible else common_keys)
        _exact_keys(value, expected_keys, f"{cohort} execution receipt")
        _require(value["schema"] == SCHEMA
                 and value["study"] == "sage-mem-v1"
                 and value["stage"] == "external-execution"
                 and value["cohort"] == cohort
                 and value["controller_identity_sha256"] ==
                 deck.controller_identity_sha256
                 and value["controller_pinned"] is True
                 and value["arm_identity_used"] is False
                 and value["input"] == "predicted_class_only"
                 and value["oracle_success"] == deck.oracle_rate
                 and value["random_success"] == deck.random_rate
                 and value["eligibility_metric"] == "mean_oracle_success"
                 and value["eligibility_operator"] == ">="
                 and value["eligibility_threshold"] == deck.threshold,
                 f"execution receipt provenance differs: {cohort}")
        if deck.eligible:
            per_age: dict[str, dict[str, list[float]]] = {}
            for arm in grid.contract.arms:
                for seed in grid.contract.seeds:
                    arrays = finalized_arrays[(cohort, arm, seed)]
                    for stream in ("full", "reset", "prior"):
                        executed = arrays[
                            f"formal_test_{stream}_execution_success"]
                        per_age.setdefault(arm, {}).setdefault(
                            stream, []).extend(
                                map(float, np.mean(executed, axis=1)))
            _require(value["status"] ==
                     "computed-class-conditioned-arm-blind"
                     and value["eligible"] is True
                     and value["skip_reason"] is None
                     and value["computed_cells"] ==
                     len(grid.contract.arms) * len(grid.contract.seeds)
                     and value["ages"] == list(grid.contract.ages)
                     and value["per_arm_seed_age_values_sha256"] ==
                     _sha256_json(per_age),
                     f"eligible execution receipt differs: {cohort}")
        else:
            _require(value["status"] == "skipped-oracle-gate"
                     and value["eligible"] is False
                     and value["skip_reason"] ==
                     "oracle-success-below-preregistered-threshold"
                     and value["computed_cells"] == 0,
                     f"skipped execution receipt differs: {cohort}")
        observed_status[cohort] = {
            "status": value["status"], "eligible": bool(deck.eligible),
            "receipt_sha256": _sha256_file(path),
            "oracle_success": deck.oracle_rate,
            "random_success": deck.random_rate,
        }
    _require(dict(summary_status) == observed_status,
             "summary execution cohort status differs from receipts")
    return observed_status


def _validate_finalized_with_contract(
        phase_a_root: str | Path, sealed_registry_manifest: str | Path,
        output_root: str | Path, contract: _GridContract, *,
        raw_context_root: str | Path | None = None,
        execution_deck_registry: str | Path | None = None,
        ) -> dict[str, Any]:
    """Read-only, hash-complete validation used before safe resume."""

    output = Path(output_root).resolve()
    _require(output.is_dir() and not output.is_symlink(),
             "finalized output root is missing or unsafe")
    grid = _validate_complete_grid(phase_a_root, contract)
    raw_references: dict[
        tuple[str, int], ValidatedRawContextReference] = {}
    raw_context_sha256: str | None = None
    if raw_context_root is not None:
        raw_references, raw_context_sha256 = _validate_raw_context_references(
            raw_context_root, grid)
    registry = Path(sealed_registry_manifest).resolve()
    deck_registry = (Path(execution_deck_registry).resolve()
                     if execution_deck_registry is not None else None)
    receipt_path = output / "label_reveal_receipt.json"
    _validate_reveal_receipt(
        receipt_path, grid, registry,
        raw_context_sha256=raw_context_sha256,
        execution_deck_registry=deck_registry)
    labels = _load_label_registry(
        registry, receipt_path, grid,
        raw_context_sha256=raw_context_sha256,
        execution_deck_registry=deck_registry)
    decks = (_load_execution_decks(
        deck_registry, registry, receipt_path, grid, labels,
        raw_context_sha256=raw_context_sha256)
        if deck_registry is not None else {})

    summary_path = output / "summary.json"
    summary = _read_json(summary_path, "formal-finalizer summary")
    _exact_keys(summary, {
        "schema", "study", "stage", "status", "phase_a_cells",
        "phase_a_grid_sha256", "label_reveal_receipt_sha256",
        "label_registry_sha256", "development_outcomes_read",
        "per_age_results_preserved",
        "pointmaze_x4_native_clustering_preserved",
        "raw_context_reference", "execution_decks",
        "finalized_cells_sha256", "finalized_cells",
    }, "formal-finalizer summary")
    _require(summary["schema"] == SCHEMA
             and summary["study"] == "sage-mem-v1"
             and summary["stage"] == "formal-finalizer"
             and summary["status"] == "complete"
             and summary["phase_a_cells"] == contract.total_cells
             and summary["phase_a_grid_sha256"] == grid.grid_sha256
             and summary["label_reveal_receipt_sha256"] ==
             _sha256_file(receipt_path)
             and summary["label_registry_sha256"] == _sha256_file(registry)
             and summary["development_outcomes_read"] is False
             and summary["per_age_results_preserved"] is True
             and summary["pointmaze_x4_native_clustering_preserved"] ==
             (contract.variants_per_cluster.get(
                 "dinowm_pointmaze_goal") == 4)
             and summary["finalized_cells"] == contract.total_cells,
             "formal-finalizer summary identity differs")

    cells_root = output / "cells"
    _require(cells_root.is_dir() and not cells_root.is_symlink()
             and {item.name for item in cells_root.iterdir()}
             == set(contract.cohorts),
             "finalized cohort directory registry differs")
    expected_consumer_hash: dict[tuple[str, int], str] = {}
    feature_cohorts = {
        cohort for cohort in contract.cohorts
        if grid.cells[(cohort, contract.arms[0],
                       contract.seeds[0])].representation ==
        "feature_artifact"
    }
    consumer_root = output / "consumers"
    if feature_cohorts:
        _require(consumer_root.is_dir() and not consumer_root.is_symlink()
                 and {item.name for item in consumer_root.iterdir()} ==
                 feature_cohorts
                 and all(item.is_dir() and not item.is_symlink()
                         for item in consumer_root.iterdir()),
                 "shared-consumer cohort registry differs")
        for cohort in feature_cohorts:
            cohort_root = consumer_root / cohort
            _require({item.name for item in cohort_root.iterdir()} == {
                f"seed-{seed}.json" for seed in contract.seeds},
                f"shared-consumer seed registry differs: {cohort}")
    else:
        _require(not consumer_root.exists(),
                 "shared-consumer receipts exist for precomputed predictions")
    for cohort in contract.cohorts:
        representation = grid.cells[(cohort, contract.arms[0],
                                     contract.seeds[0])].representation
        for seed in contract.seeds:
            if representation == "feature_artifact":
                expected_consumer_hash[(cohort, seed)] = \
                    _validate_feature_consumer_receipt(
                        output, grid, cohort, seed)
            else:
                expected_consumer_hash[(cohort, seed)] = str(
                    grid.cells[(cohort, contract.arms[0], seed)]
                    .shared_consumer_sha256)

    finalized_records: list[dict[str, Any]] = []
    finalized_arrays: dict[
        tuple[str, str, int], dict[str, np.ndarray]] = {}
    for cohort in contract.cohorts:
        cohort_root = cells_root / cohort
        _require(all(item.is_dir() and not item.is_symlink()
                     for item in cohort_root.iterdir())
                 and {item.name for item in cohort_root.iterdir()}
                 == set(contract.arms),
                 f"finalized arm directory registry differs: {cohort}")
        labels_by_age = np.repeat(
            labels[cohort].formal_test_label[None, :],
            len(contract.ages), axis=0)
        deck = decks.get(cohort)
        for arm in contract.arms:
            arm_root = cohort_root / arm
            _require(all(item.is_dir() and not item.is_symlink()
                         for item in arm_root.iterdir())
                     and {item.name for item in arm_root.iterdir()}
                     == {f"seed-{seed}" for seed in contract.seeds},
                     f"finalized seed directory registry differs: "
                     f"{cohort}/{arm}")
            for seed in contract.seeds:
                key = (cohort, arm, seed)
                phase_cell = grid.cells[key]
                directory = arm_root / f"seed-{seed}"
                manifest_path = directory / "manifest.json"
                manifest = _read_json(manifest_path, "finalized manifest")
                _exact_keys(manifest, {
                    "schema", "study", "stage", "status", "cohort", "arm",
                    "seed", "ages", "phase_a_grid_sha256",
                    "phase_a_manifest_sha256", "bank_manifest_sha256",
                    "label_registry_sha256", "label_reveal_receipt_sha256",
                    "shared_arm_blind_consumer_sha256",
                    "native_cluster_id_preserved",
                    "counterfactual_variants_per_native_cluster",
                    "execution", "artifact",
                }, "finalized manifest")
                _require(manifest["schema"] == SCHEMA
                         and manifest["study"] == "sage-mem-v1"
                         and manifest["stage"] == "formal-finalized"
                         and manifest["status"] == "complete"
                         and manifest["cohort"] == cohort
                         and manifest["arm"] == arm
                         and manifest["seed"] == seed
                         and manifest["ages"] == list(contract.ages)
                         and manifest["phase_a_grid_sha256"] == grid.grid_sha256
                         and manifest["phase_a_manifest_sha256"] ==
                         phase_cell.manifest_sha256
                         and manifest["bank_manifest_sha256"] ==
                         phase_cell.bank_sha256
                         and manifest["label_registry_sha256"] ==
                         _sha256_file(registry)
                         and manifest["label_reveal_receipt_sha256"] ==
                         _sha256_file(receipt_path)
                         and manifest[
                             "shared_arm_blind_consumer_sha256"] ==
                         expected_consumer_hash[(cohort, seed)]
                         and manifest["native_cluster_id_preserved"] is True
                         and manifest[
                             "counterfactual_variants_per_native_cluster"] ==
                         contract.variants_per_cluster[cohort],
                         f"finalized manifest identity differs: {directory}")
                artifact = _safe_artifact(
                    directory, manifest["artifact"], "finalized results")
                _require({item.name for item in directory.iterdir()} == {
                    "manifest.json", artifact.name},
                    "finalized cell contains unexpected files")
                try:
                    with np.load(artifact, allow_pickle=False) as archive:
                        execution_expected = deck is not None and deck.eligible
                        expected_keys = (_FINAL_BASE_ARRAY_KEYS |
                                         (_FINAL_EXECUTION_ARRAY_KEYS
                                          if execution_expected else set()))
                        _require(set(archive.files) == expected_keys,
                                 "finalized result array schema differs")
                        arrays = {name: np.asarray(archive[name]).copy()
                                  for name in archive.files}
                except (OSError, ValueError) as error:
                    raise SageMemFormalFinalizerError(
                        f"cannot load finalized artifact: {artifact}") from error
                for name in ("formal_test_episode_id",
                             "formal_test_native_cluster_id",
                             "formal_test_evidence_age"):
                    _require(np.array_equal(arrays[name],
                                            phase_cell.arrays[name]),
                             f"finalized identity drifts: {key}/{name}")
                _require(np.array_equal(
                    arrays["formal_test_label"], labels_by_age),
                    f"finalized labels drift: {key}")
                for stream in ("full", "reset", "prior"):
                    prediction = arrays[f"formal_test_{stream}_pred"]
                    correctness = arrays[f"formal_test_{stream}_correct"]
                    _require(prediction.shape == labels_by_age.shape
                             and np.issubdtype(prediction.dtype, np.integer)
                             and np.all((prediction >= 0) & (prediction <
                                 contract.classes[cohort]))
                             and np.array_equal(
                                 correctness,
                                 (prediction == labels_by_age).astype(np.uint8))
                             and np.array_equal(
                                 arrays[f"formal_test_{stream}_mse"],
                                 phase_cell.arrays[
                                     f"formal_test_{stream}_mse"]),
                             f"finalized semantic metric differs: "
                             f"{key}/{stream}")
                execution_manifest = manifest["execution"]
                _require(isinstance(execution_manifest, dict),
                         "finalized execution manifest must be a mapping")
                _exact_keys(execution_manifest, {
                    "status", "eligible", "receipt_sha256",
                    "arm_identity_used", "ages", "per_age_success",
                }, "finalized execution manifest")
                if execution_expected:
                    row = np.arange(len(deck.episode_id))[None, :]
                    expected_per_age: dict[str, list[float]] = {}
                    for stream in ("full", "reset", "prior"):
                        expected_execution = deck.class_conditioned_success[
                            row, arrays[f"formal_test_{stream}_pred"]]
                        observed = arrays[
                            f"formal_test_{stream}_execution_success"]
                        _require(np.array_equal(observed, expected_execution),
                                 f"executed success differs: {key}/{stream}")
                        expected_per_age[stream] = list(map(
                            float, np.mean(observed, axis=1)))
                    execution_receipt = output / "execution" / cohort \
                        / "receipt.json"
                    _require(execution_manifest == {
                        "status": "computed-class-conditioned-arm-blind",
                        "eligible": True,
                        "receipt_sha256": _sha256_file(execution_receipt),
                        "arm_identity_used": False,
                        "ages": list(contract.ages),
                        "per_age_success": expected_per_age,
                    }, f"eligible execution link differs: {key}")
                elif deck is not None:
                    execution_receipt = output / "execution" / cohort \
                        / "receipt.json"
                    _require(execution_manifest == {
                        "status": "skipped-oracle-gate",
                        "eligible": False,
                        "receipt_sha256": _sha256_file(execution_receipt),
                        "arm_identity_used": False,
                        "ages": list(contract.ages),
                        "per_age_success": None,
                    }, f"skipped execution link differs: {key}")
                else:
                    _require(execution_manifest == {
                        "status": "not-supplied", "eligible": None,
                        "receipt_sha256": None,
                        "arm_identity_used": False,
                        "ages": list(contract.ages),
                        "per_age_success": None,
                    }, f"absent execution link differs: {key}")
                finalized_arrays[key] = arrays
                finalized_records.append({
                    "cohort": cohort, "arm": arm, "seed": seed,
                    "artifact_sha256": _sha256_file(artifact),
                    "consumer_sha256": expected_consumer_hash[(cohort, seed)],
                })
    _require(summary["finalized_cells_sha256"] ==
             _sha256_json(finalized_records),
             "summary finalized-cell digest differs")

    execution_summary = summary["execution_decks"]
    _require(isinstance(execution_summary, dict),
             "summary execution record must be a mapping")
    _exact_keys(execution_summary, {
        "status", "supplied_cohorts", "eligible_cohorts",
        "program_requires_at_least_two_eligible_cohorts",
        "program_gate_passed", "cohort_status",
    }, "summary execution record")
    eligible_count = sum(deck.eligible for deck in decks.values())
    observed_execution_status = _validate_execution_receipts(
        output, grid, decks, finalized_arrays,
        execution_summary["cohort_status"])
    program_required = execution_summary[
        "program_requires_at_least_two_eligible_cohorts"]
    _require(execution_summary["status"] ==
             ("evaluated" if decks else "not-supplied")
             and execution_summary["supplied_cohorts"] == sorted(decks)
             and execution_summary["eligible_cohorts"] == eligible_count
             and isinstance(program_required, bool)
             and execution_summary["program_gate_passed"] ==
             (eligible_count >= 2 if program_required else None)
             and (not program_required or eligible_count >= 2)
             and execution_summary["cohort_status"] ==
             observed_execution_status,
             "summary execution gate differs")

    raw_summary = summary["raw_context_reference"]
    _require(isinstance(raw_summary, dict),
             "summary raw-context record must be a mapping")
    _exact_keys(raw_summary, {
        "status", "short_context_frames", "long_context_frames",
        "separate_from_parameter_matched_arms", "references",
        "records_sha256",
    }, "summary raw-context record")
    raw_records: list[dict[str, Any]] = []
    raw_output = output / "raw_context"
    if raw_references:
        raw_consumer_root = output / "raw_context_consumers"
        _require(raw_consumer_root.is_dir()
                 and not raw_consumer_root.is_symlink()
                 and {item.name for item in raw_consumer_root.iterdir()} ==
                 set(contract.cohorts),
                 "raw-context consumer cohort registry differs")
        for cohort in contract.cohorts:
            _require({item.name for item in (
                raw_consumer_root / cohort).iterdir()} == {
                    f"seed-{seed}.json" for seed in contract.seeds},
                f"raw-context consumer seed registry differs: {cohort}")
        _require(raw_output.is_dir() and not raw_output.is_symlink(),
                 "finalized raw-context root is missing")
        _require({item.name for item in raw_output.iterdir()} ==
                 set(contract.cohorts)
                 and all(item.is_dir() and not item.is_symlink()
                         for item in raw_output.iterdir()),
                 "finalized raw-context cohort registry differs")
        for cohort in contract.cohorts:
            cohort_root = raw_output / cohort
            _require({item.name for item in cohort_root.iterdir()} ==
                     {f"seed-{seed}" for seed in contract.seeds}
                     and all(item.is_dir() and not item.is_symlink()
                             for item in cohort_root.iterdir()),
                     f"finalized raw-context seed registry differs: {cohort}")
        for (cohort, seed), reference in raw_references.items():
            consumer_path = (raw_consumer_root / cohort
                             / f"seed-{seed}.json")
            consumer = _read_json(
                consumer_path, "raw-context shared-consumer receipt")
            _exact_keys(consumer, {
                "schema", "study", "stage", "status", "cohort", "seed",
                "contexts", "ages_pooled", "training_rows",
                "feature_dimension", "arm_identity_used",
                "context_identity_used", "formal_test_labels_used",
                "model_sha256", "shared_consumer_sha256",
            }, "raw-context shared-consumer receipt")
            expected_consumer_hash = _sha256_json({
                "cohort": cohort,
                "seed": seed,
                "model_sha256": consumer.get("model_sha256"),
                "contexts": ["short-3", "long-16"],
                "ages_pooled": list(contract.ages),
                "fit_split": "consumer_train",
                "arm_identity_used": False,
                "context_identity_used": False,
                "formal_test_labels_used": False,
            })
            _require(consumer["schema"] == SCHEMA
                     and consumer["study"] == "sage-mem-v1"
                     and consumer["stage"] ==
                     "raw-context-shared-consumer"
                     and consumer["status"] ==
                     "fit-after-complete-grid-reveal"
                     and consumer["cohort"] == cohort
                     and consumer["seed"] == seed
                     and consumer["contexts"] == ["short-3", "long-16"]
                     and consumer["ages_pooled"] == list(contract.ages)
                     and consumer["training_rows"] ==
                     2 * len(contract.ages) *
                     contract.consumer_train_rows[cohort]
                     and consumer["feature_dimension"] ==
                     reference.feature_dimension
                     and consumer["arm_identity_used"] is False
                     and consumer["context_identity_used"] is False
                     and consumer["formal_test_labels_used"] is False
                     and _is_sha256(consumer["model_sha256"])
                     and consumer["shared_consumer_sha256"] ==
                     expected_consumer_hash,
                     "raw-context shared-consumer receipt differs")
            directory = raw_output / cohort / f"seed-{seed}"
            manifest_path = directory / "manifest.json"
            manifest = _read_json(manifest_path,
                                  "finalized raw-context manifest")
            _exact_keys(manifest, {
                "schema", "study", "stage", "status", "cohort", "seed",
                "ages", "short_context_frames", "long_context_frames",
                "separate_from_parameter_matched_arms",
                "phase_a_grid_sha256", "source_manifest_sha256",
                "shared_arm_blind_consumer_sha256", "artifact",
            }, "finalized raw-context manifest")
            _require(manifest["schema"] == SCHEMA
                     and manifest["study"] == "sage-mem-v1"
                     and manifest["stage"] ==
                     "formal-raw-context-finalized"
                     and manifest["status"] == "complete"
                     and manifest["cohort"] == cohort
                     and manifest["seed"] == seed
                     and manifest["ages"] == list(contract.ages)
                     and manifest["short_context_frames"] == 3
                     and manifest["long_context_frames"] == 16
                     and manifest[
                         "separate_from_parameter_matched_arms"] is True
                     and manifest["phase_a_grid_sha256"] == grid.grid_sha256
                     and manifest["source_manifest_sha256"] ==
                     reference.manifest_sha256
                     and manifest[
                         "shared_arm_blind_consumer_sha256"] ==
                     expected_consumer_hash,
                     "finalized raw-context identity differs")
            artifact = _safe_artifact(
                directory, manifest["artifact"],
                "finalized raw-context results")
            _require({item.name for item in directory.iterdir()} == {
                "manifest.json", artifact.name},
                "finalized raw-context cell contains unexpected files")
            try:
                with np.load(artifact, allow_pickle=False) as archive:
                    expected = {
                        "formal_test_episode_id",
                        "formal_test_native_cluster_id",
                        "formal_test_evidence_age", "formal_test_label",
                        "formal_test_short_pred", "formal_test_long_pred",
                        "formal_test_short_correct",
                        "formal_test_long_correct",
                    }
                    _require(set(archive.files) == expected,
                             "finalized raw-context schema differs")
                    arrays = {name: np.asarray(archive[name]).copy()
                              for name in archive.files}
            except (OSError, ValueError) as error:
                raise SageMemFormalFinalizerError(
                    "cannot load finalized raw-context artifact") from error
            labels_by_age = np.repeat(
                labels[cohort].formal_test_label[None, :],
                len(contract.ages), axis=0)
            for name in ("formal_test_episode_id",
                         "formal_test_native_cluster_id",
                         "formal_test_evidence_age"):
                _require(np.array_equal(arrays[name], reference.arrays[name]),
                         f"finalized raw-context data differs: {name}")
            _require(all(
                arrays[name].shape == labels_by_age.shape
                and np.issubdtype(arrays[name].dtype, np.integer)
                and np.all((arrays[name] >= 0) &
                           (arrays[name] < contract.classes[cohort]))
                for name in (
                    "formal_test_short_pred", "formal_test_long_pred"))
                     and np.array_equal(
                         arrays["formal_test_label"], labels_by_age)
                     and np.array_equal(
                         arrays["formal_test_short_correct"],
                         (arrays["formal_test_short_pred"] ==
                          labels_by_age).astype(np.uint8))
                     and np.array_equal(
                         arrays["formal_test_long_correct"],
                         (arrays["formal_test_long_pred"] ==
                          labels_by_age).astype(np.uint8)),
                     "finalized raw-context correctness differs")
            raw_records.append({
                "cohort": cohort, "seed": seed,
                "artifact_sha256": _sha256_file(artifact),
                "consumer_sha256": expected_consumer_hash,
            })
        _require(raw_summary == {
            "status": "complete", "short_context_frames": 3,
            "long_context_frames": 16,
            "separate_from_parameter_matched_arms": True,
            "references": len(raw_records),
            "records_sha256": _sha256_json(raw_records),
        }, "summary raw-context digest differs")
    else:
        _require(not raw_output.exists() and raw_summary == {
            "status": "not-supplied", "short_context_frames": 3,
            "long_context_frames": 16,
            "separate_from_parameter_matched_arms": True,
            "references": 0, "records_sha256": None,
        }, "summary reports raw-context output without a source")

    allowed_top = {"cells", "label_reveal_receipt.json", "summary.json"}
    if feature_cohorts:
        allowed_top.add("consumers")
    registry_value = _read_json(registry, "sealed label registry")
    if registry_value.get("schema") == CUSTODY_REGISTRY_SCHEMA:
        normalized_root = output / "normalized_label_registry"
        normalized_manifest_path = normalized_root / "manifest.json"
        normalized = _read_json(
            normalized_manifest_path, "normalized label registry")
        _exact_keys(normalized, {
            "schema", "study", "status",
            "source_custody_registry_sha256",
            "label_reveal_receipt_sha256", "development_outcomes_read",
            "cohorts",
        }, "normalized label registry")
        _require(normalized["schema"] ==
                 "sage_mem_v1_post_reveal_consolidated_registry_v1"
                 and normalized["study"] == "sage-mem-v1"
                 and normalized["status"] ==
                 "normalized-after-complete-grid-reveal"
                 and normalized["source_custody_registry_sha256"] ==
                 _sha256_file(registry)
                 and normalized["label_reveal_receipt_sha256"] ==
                 _sha256_file(receipt_path)
                 and normalized["development_outcomes_read"] is False
                 and isinstance(normalized["cohorts"], dict)
                 and set(normalized["cohorts"]) == set(contract.cohorts),
                 "normalized label registry identity differs")
        expected_normalized_files = {"manifest.json"}
        for cohort in contract.cohorts:
            record = normalized["cohorts"][cohort]
            _require(isinstance(record, dict),
                     "normalized cohort record must be a mapping")
            _exact_keys(record, {
                "bank_manifest_sha256", "classes", "artifact"},
                f"{cohort} normalized label record")
            _require(record["bank_manifest_sha256"] ==
                     grid.bank_hashes[cohort]
                     and record["classes"] == contract.classes[cohort],
                     "normalized label cohort identity differs")
            artifact = _safe_registry_artifact(
                normalized_root, record["artifact"],
                f"{cohort} normalized labels")
            expected_normalized_files.add(artifact.name)
            try:
                with np.load(artifact, allow_pickle=False) as archive:
                    _require(set(archive.files) ==
                             set(LabelSet.__dataclass_fields__),
                             "normalized label artifact schema differs")
                    for name in LabelSet.__dataclass_fields__:
                        _require(np.array_equal(
                            archive[name], getattr(labels[cohort], name)),
                            f"normalized label artifact differs: "
                            f"{cohort}/{name}")
            except (OSError, ValueError) as error:
                raise SageMemFormalFinalizerError(
                    "cannot load normalized label artifact") from error
        _require({item.name for item in normalized_root.iterdir()} ==
                 expected_normalized_files,
                 "normalized label registry contains unexpected files")
        allowed_top.add("normalized_label_registry")
    else:
        _require(not (output / "normalized_label_registry").exists(),
                 "normalized labels exist for a consolidated source")
    if decks:
        allowed_top.add("execution")
    if raw_references:
        allowed_top.add("raw_context")
        allowed_top.add("raw_context_consumers")
    _require({item.name for item in output.iterdir()} == allowed_top,
             "finalized output root contains missing or unexpected entries")
    return summary


def validate_finalized_output(
        phase_a_root: str | Path, sealed_registry_manifest: str | Path,
        output_root: str | Path, *,
        raw_context_root: str | Path | None = None,
        execution_deck_registry: str | Path | None = None,
        ) -> dict[str, Any]:
    """Validate an existing 600-cell finalized root before ``--resume``."""

    return _validate_finalized_with_contract(
        phase_a_root, sealed_registry_manifest, output_root,
        PRODUCTION_CONTRACT, raw_context_root=raw_context_root,
        execution_deck_registry=execution_deck_registry)


def finalize_formal_grid(
        phase_a_root: str | Path, sealed_registry_manifest: str | Path,
        output_root: str | Path, *,
        raw_context_root: str | Path | None = None,
        execution_deck_registry: str | Path | None = None,
        require_at_least_two_eligible_execution_cohorts: bool = False,
        ) -> dict[str, Any]:
    """Finalize the exact 600-cell production grid after a durable reveal."""

    return _finalize_with_contract(
        phase_a_root, sealed_registry_manifest, output_root,
        PRODUCTION_CONTRACT, raw_context_root=raw_context_root,
        execution_deck_registry=execution_deck_registry,
        require_at_least_two_eligible_execution_cohorts=
            require_at_least_two_eligible_execution_cohorts)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a-root", type=Path, required=True)
    parser.add_argument("--label-registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-context-root", type=Path)
    parser.add_argument("--execution-deck-registry", type=Path)
    parser.add_argument(
        "--require-at-least-two-eligible-execution-cohorts",
        action="store_true")
    parser.add_argument(
        "--validate-finalized-output", "--resume",
        dest="validate_finalized_output", action="store_true",
        help="read-only validation of an existing finalized root")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.validate_finalized_output:
        summary = validate_finalized_output(
            args.phase_a_root, args.label_registry, args.output_root,
            raw_context_root=args.raw_context_root,
            execution_deck_registry=args.execution_deck_registry)
        print(_canonical_json({
            "status": "validated-finalized-output",
            "phase_a_cells": summary["phase_a_cells"],
            "phase_a_grid_sha256": summary["phase_a_grid_sha256"],
            "finalized_cells_sha256": summary["finalized_cells_sha256"],
        }))
        return 0
    grid = validate_complete_phase_a_grid(args.phase_a_root)
    if not args.execute:
        print(_canonical_json({
            "status": "validated-labels-still-sealed",
            "phase_a_cells": grid.contract.total_cells,
            "phase_a_grid_sha256": grid.grid_sha256,
            "bank_manifest_sha256": dict(grid.bank_hashes),
        }))
        return 0
    summary = finalize_formal_grid(
        args.phase_a_root, args.label_registry, args.output_root,
        raw_context_root=args.raw_context_root,
        execution_deck_registry=args.execution_deck_registry,
        require_at_least_two_eligible_execution_cohorts=
            args.require_at_least_two_eligible_execution_cohorts)
    print(_canonical_json(summary))
    return 0


__all__ = [
    "AGES",
    "ARMS",
    "COHORTS",
    "EXECUTION_DECK_REGISTRY_SCHEMA",
    "EXECUTION_REPLAY_RECEIPT_SCHEMA",
    "EXECUTION_UNAVAILABLE_RECEIPT_SCHEMA",
    "LABEL_REGISTRY_SCHEMA",
    "PHASE_A_SCHEMA",
    "PRODUCTION_CONTRACT",
    "RAW_CONTEXT_SCHEMA",
    "SageMemFormalFinalizerError",
    "ValidatedCell",
    "ValidatedGrid",
    "finalize_formal_grid",
    "normalize_custody_vaults_after_reveal",
    "validate_complete_phase_a_grid",
    "validate_finalized_output",
    "validate_phase_a_cell",
]


if __name__ == "__main__":
    raise SystemExit(main())
