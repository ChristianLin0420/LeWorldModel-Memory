#!/usr/bin/env python3
"""Independent Phase-B reproduction and provenance audit for SAGE-Mem v1.

This program is intentionally separate from the formal finalizer.  In execute
mode it authenticates the sealed producer identities, the complete Phase-A
inventory, the post-reveal registries, and the finalized/report inventories;
then it independently refits every registered pooled ``RidgeClassifier`` and
requires byte-exact agreement with finalized predictions, correctness, raw-
context predictions, and eligible class-conditioned execution arrays.

The default mode is a true preview: it performs no filesystem reads.  Execute
mode is only valid after the complete 600-cell finalization and formal report
exist, and requires operator-supplied hashes for every outcome-bearing root.
It emits a value-free canonical receipt containing provenance and semantic
digests, never accuracies, effects, labels, or gate decisions.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import contextvars
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sage_mem_v1_phase_b_reproduction_v1"
FINALIZER_SCHEMA = "sage_mem_v1_formal_finalizer_v1"
FORMAL_REPORT_SCHEMA = "sage_mem_v1_formal_evidence_audit_v1"
PHASE_A_SCHEMA = "sage_mem_v1_phase_a_cell_v1"
RAW_SCHEMA = "sage_mem_v1_raw_context_reference_v1"
LABEL_SCHEMAS = {
    "sage_mem_v1_sealed_label_registry_v1",
    "sage_mem_v1_custody_vault_registry_v1",
}
EXECUTION_SCHEMA = "sage_mem_v1_execution_deck_registry_v2"
EXECUTION_REPLAY_SCHEMA = "sage_mem_v1_execution_replay_receipt_v1"
EXECUTION_UNAVAILABLE_SCHEMA = \
    "sage_mem_v1_execution_deck_unavailable_v1"

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
AGES = (4, 8, 15)
CLASSES = {
    "lewm_reacher_color": 4,
    "lewm_pusht_color": 4,
    "dinowm_pusht_token": 4,
    "dinowm_pusht_binding": 6,
    "dinowm_pointmaze_goal": 4,
}
FORMAL_ROWS = {
    "lewm_reacher_color": 720,
    "lewm_pusht_color": 720,
    "dinowm_pusht_token": 960,
    "dinowm_pusht_binding": 960,
    "dinowm_pointmaze_goal": 1440,
}
CONSUMER_ROWS = {
    "lewm_reacher_color": 480,
    "lewm_pusht_color": 480,
    "dinowm_pusht_token": 600,
    "dinowm_pusht_binding": 600,
    "dinowm_pointmaze_goal": 960,
}
VARIANTS = {cohort: (4 if cohort == "dinowm_pointmaze_goal" else 1)
            for cohort in COHORTS}
PHYSICAL_GPUS = {
    "lewm_reacher_color": 0,
    "lewm_pusht_color": 0,
    "dinowm_pusht_token": 1,
    "dinowm_pusht_binding": 1,
    "dinowm_pointmaze_goal": 2,
}
RAW_FEATURE_CONTRACT = {
    "slots": 16,
    "short_observed_slots": 3,
    "long_observed_slots": 16,
    "padding": "left-zero",
    "flatten_order": "time-major",
    "lewm_frame_representation": "frozen-frame-embedding",
    "dino_frame_representation": "mean-pool-frozen-spatial-patches",
}

PHASE_COMMON_KEYS = {
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
PHASE_FEATURE_KEYS = {
    "formal_test_full_features",
    "formal_test_reset_features",
    "formal_test_prior_features",
    "consumer_train_full_features",
}
RAW_KEYS = {
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
FINAL_KEYS = {
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
FINAL_EXECUTION_KEYS = {
    "formal_test_full_execution_success",
    "formal_test_reset_execution_success",
    "formal_test_prior_execution_success",
}
FINAL_RAW_KEYS = {
    "formal_test_episode_id",
    "formal_test_native_cluster_id",
    "formal_test_evidence_age",
    "formal_test_label",
    "formal_test_short_pred",
    "formal_test_long_pred",
    "formal_test_short_correct",
    "formal_test_long_correct",
}


class PhaseBReproductionError(RuntimeError):
    """A provenance, inventory, or independent-reproduction check failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhaseBReproductionError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


class _StableFile:
    """One no-follow file descriptor binding all reads to authenticated bytes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(os.path.abspath(path))
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
            | getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(self.path, flags)
            self.fd = descriptor
            self.initial_stat = os.fstat(self.fd)
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
            raise PhaseBReproductionError(
                f"cannot open stable artifact descriptor: {self.path}") \
                from error
        if not stat.S_ISREG(self.initial_stat.st_mode):
            os.close(self.fd)
            raise PhaseBReproductionError(
                f"stable artifact is not a regular file: {self.path}")
        self.initial_sha256 = self._hash_fd()

    @property
    def size(self) -> int:
        return int(self.initial_stat.st_size)

    def _hash_fd(self) -> str:
        digest = hashlib.sha256()
        offset = 0
        while True:
            try:
                block = os.pread(self.fd, 8 * 1024 * 1024, offset)
            except OSError as error:
                raise PhaseBReproductionError(
                    f"cannot hash stable artifact: {self.path}") from error
            if not block:
                break
            digest.update(block)
            offset += len(block)
        return digest.hexdigest()

    def read_bytes(self) -> bytes:
        chunks: list[bytes] = []
        offset = 0
        while True:
            try:
                block = os.pread(self.fd, 8 * 1024 * 1024, offset)
            except OSError as error:
                raise PhaseBReproductionError(
                    f"cannot read stable artifact: {self.path}") from error
            if not block:
                break
            chunks.append(block)
            offset += len(block)
        return b"".join(chunks)

    @contextmanager
    def open_binary(self):
        duplicate = os.dup(self.fd)
        try:
            os.lseek(duplicate, 0, os.SEEK_SET)
            with os.fdopen(duplicate, "rb", closefd=True) as stream:
                duplicate = -1
                yield stream
        finally:
            if duplicate >= 0:
                os.close(duplicate)

    def verify_unchanged(self) -> None:
        current = os.fstat(self.fd)
        identity = ("st_dev", "st_ino", "st_mode", "st_size",
                    "st_mtime_ns", "st_ctime_ns")
        _require(all(getattr(current, name) ==
                     getattr(self.initial_stat, name) for name in identity),
                 f"stable artifact metadata changed during audit: {self.path}")
        _require(self._hash_fd() == self.initial_sha256,
                 f"stable artifact bytes changed during audit: {self.path}")
        try:
            path_stat = os.lstat(self.path)
        except OSError as error:
            raise PhaseBReproductionError(
                f"stable artifact path disappeared during audit: {self.path}") \
                from error
        _require(not stat.S_ISLNK(path_stat.st_mode)
                 and path_stat.st_dev == self.initial_stat.st_dev
                 and path_stat.st_ino == self.initial_stat.st_ino,
                 f"stable artifact path identity changed during audit: "
                 f"{self.path}")

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


class _ArtifactSession:
    def __init__(self) -> None:
        self.files: dict[Path, _StableFile] = {}
        self.directories: dict[Path, _StableDirectory] = {}

    def get(self, path: Path) -> _StableFile:
        absolute = Path(os.path.abspath(path))
        stable = self.files.get(absolute)
        if stable is None:
            stable = _StableFile(absolute)
            self.files[absolute] = stable
        return stable

    def verify_all(self) -> None:
        for path in sorted(self.directories, key=lambda item: item.as_posix()):
            self.directories[path].verify_unchanged()
        for path in sorted(self.files, key=lambda item: item.as_posix()):
            self.files[path].verify_unchanged()

    def directory(self, path: Path) -> "_StableDirectory":
        absolute = Path(os.path.abspath(path))
        stable = self.directories.get(absolute)
        if stable is None:
            stable = _StableDirectory(absolute)
            self.directories[absolute] = stable
        return stable

    def close(self) -> None:
        for stable in self.files.values():
            stable.close()
        self.files.clear()
        self.directories.clear()


class _StableDirectory:
    """Stable metadata and no-follow entry snapshot for exact inventories."""

    def __init__(self, path: Path) -> None:
        self.path = Path(os.path.abspath(path))
        self.initial_stat, self.entries = self._snapshot()

    def _snapshot(self) -> tuple[os.stat_result,
                                 tuple[tuple[str, int, int, int], ...]]:
        try:
            info = os.lstat(self.path)
            _require(stat.S_ISDIR(info.st_mode)
                     and not stat.S_ISLNK(info.st_mode),
                     f"inventory directory is absent or unsafe: {self.path}")
            rows = []
            with os.scandir(self.path) as stream:
                for entry in stream:
                    child = entry.stat(follow_symlinks=False)
                    rows.append((entry.name, int(child.st_mode),
                                 int(child.st_dev), int(child.st_ino)))
        except OSError as error:
            raise PhaseBReproductionError(
                f"cannot snapshot inventory directory: {self.path}") \
                from error
        return info, tuple(sorted(rows))

    def items(self) -> tuple[Path, ...]:
        return tuple(self.path / row[0] for row in self.entries)

    def verify_unchanged(self) -> None:
        current, entries = self._snapshot()
        identity = ("st_dev", "st_ino", "st_mode", "st_mtime_ns",
                    "st_ctime_ns")
        _require(all(getattr(current, name) ==
                     getattr(self.initial_stat, name) for name in identity)
                 and entries == self.entries,
                 f"inventory directory changed during audit: {self.path}")


_ACTIVE_ARTIFACT_SESSION: contextvars.ContextVar[_ArtifactSession | None] = \
    contextvars.ContextVar("sage_mem_phase_b_artifact_session", default=None)


@contextmanager
def _temporary_stable(path: Path):
    stable = _StableFile(path)
    try:
        yield stable
        stable.verify_unchanged()
    finally:
        stable.close()


def _stable_file(path: Path) -> tuple[_StableFile, bool]:
    session = _ACTIVE_ARTIFACT_SESSION.get()
    if session is not None:
        return session.get(path), False
    return _StableFile(path), True


def _iterdir(path: Path) -> tuple[Path, ...]:
    session = _ACTIVE_ARTIFACT_SESSION.get()
    if session is None:
        return tuple(path.iterdir())
    return session.directory(path).items()


def _sha256_file(path: Path) -> str:
    stable, temporary = _stable_file(path)
    try:
        return stable.initial_sha256
    finally:
        if temporary:
            stable.verify_unchanged()
            stable.close()


def _stable_size(path: Path) -> int:
    stable, temporary = _stable_file(path)
    try:
        return stable.size
    finally:
        if temporary:
            stable.verify_unchanged()
            stable.close()


@contextmanager
def _open_npz(path: Path):
    stable, temporary = _stable_file(path)
    try:
        with stable.open_binary() as stream:
            with np.load(stream, allow_pickle=False) as archive:
                yield archive
    finally:
        if temporary:
            stable.verify_unchanged()
            stable.close()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _expected_sha(value: str, label: str) -> str:
    _require(_is_sha256(value), f"operator must supply a SHA-256 for {label}")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(),
             f"{label} is absent, non-regular, or a symlink")
    def reject(token: str) -> None:
        raise PhaseBReproductionError(
            f"non-finite JSON constant in {label}: {token}")
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PhaseBReproductionError(
                    f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result
    try:
        stable, temporary = _stable_file(path)
        raw = stable.read_bytes()
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(text, parse_constant=reject,
                           object_pairs_hook=unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PhaseBReproductionError(f"cannot read {label}: {path}") \
            from error
    finally:
        if "temporary" in locals() and temporary:
            stable.verify_unchanged()
            stable.close()
    _require(isinstance(value, dict), f"{label} is not a JSON mapping")
    _require(raw == (_canonical_json(value) + "\n").encode("utf-8"),
             f"{label} is not canonical JSON")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) \
        -> None:
    _require(set(value) == expected,
             f"{label} keys differ: expected {sorted(expected)}, "
             f"observed {sorted(value)}")


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(_canonical_json({
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _safe_existing(raw: str | Path, workspace: Path, label: str, *,
                   kind: str) -> Path:
    lexical = Path(raw)
    _require(".." not in lexical.parts, f"{label} contains path traversal")
    workspace = Path(os.path.abspath(workspace))
    candidate = lexical if lexical.is_absolute() else workspace / lexical
    absolute = Path(os.path.abspath(candidate))
    try:
        relative = absolute.relative_to(workspace)
    except ValueError as error:
        raise PhaseBReproductionError(f"{label} leaves the workspace") \
            from error
    _require(workspace.exists() and not workspace.is_symlink(),
             "workspace root is missing or a symlink")
    cursor = workspace
    for component in relative.parts:
        cursor = cursor / component
        _require(not cursor.is_symlink(), f"{label} path contains a symlink")
    if kind == "file":
        _require(absolute.is_file(), f"{label} is missing or not a file")
    elif kind == "dir":
        _require(absolute.is_dir(), f"{label} is missing or not a directory")
    else:
        raise AssertionError(kind)
    return absolute


def _safe_workspace(raw: str | Path) -> Path:
    """Reject a symlink in any lexical component before canonicalization."""
    lexical = Path(raw)
    _require(".." not in lexical.parts, "workspace contains path traversal")
    absolute = Path(os.path.abspath(lexical))
    anchor = Path(absolute.anchor)
    cursor = anchor
    for component in absolute.parts[1:]:
        cursor = cursor / component
        _require(cursor.exists(), "workspace component is missing")
        _require(not cursor.is_symlink(),
                 "workspace path contains a symlink")
    _require(absolute.is_dir(), "workspace is not a directory")
    return absolute


def _safe_output(raw: str | Path, workspace: Path) -> Path:
    lexical = Path(raw)
    _require(".." not in lexical.parts and lexical.suffix == ".json",
             "receipt output path is unsafe")
    workspace = workspace.resolve()
    candidate = lexical if lexical.is_absolute() else workspace / lexical
    absolute = Path(os.path.abspath(candidate))
    try:
        relative = absolute.relative_to(workspace)
    except ValueError as error:
        raise PhaseBReproductionError("receipt output leaves the workspace") \
            from error
    cursor = workspace
    for component in relative.parts[:-1]:
        cursor = cursor / component
        _require(cursor.exists() and cursor.is_dir() and not cursor.is_symlink(),
                 "receipt output parent is missing or unsafe")
    _require(not absolute.exists() and not absolute.is_symlink(),
             "refusing stale or partial reproduction receipt")
    return absolute


def _require_output_outside_inputs(output: Path,
                                   forbidden_roots: Iterable[Path]) -> None:
    for root in forbidden_roots:
        try:
            output.relative_to(root)
        except ValueError:
            continue
        raise PhaseBReproductionError(
            "receipt output overlaps an authenticated input inventory")


def _safe_artifact(parent: Path, record: Mapping[str, Any], label: str, *,
                   nested: bool = False) -> Path:
    _exact_keys(record, {"path", "sha256", "size"}, f"{label} handle")
    relative = Path(str(record.get("path", "")))
    _require(not relative.is_absolute() and ".." not in relative.parts
             and bool(relative.parts)
             and (nested or len(relative.parts) == 1),
             f"{label} artifact path is unsafe")
    path = Path(os.path.abspath(parent / relative))
    try:
        path.relative_to(parent.resolve())
    except ValueError as error:
        raise PhaseBReproductionError(f"{label} leaves its custody root") \
            from error
    cursor = parent
    for component in relative.parts:
        cursor = cursor / component
        _require(not cursor.is_symlink(), f"{label} path contains a symlink")
    _require(path.is_file() and not path.is_symlink(),
             f"{label} artifact is absent or unsafe")
    stable, temporary = _stable_file(path)
    _require(isinstance(record.get("size"), int)
             and not isinstance(record["size"], bool)
             and record["size"] > 0 and stable.size == record["size"],
             f"{label} artifact size differs")
    _require(_is_sha256(record.get("sha256"))
             and stable.initial_sha256 == record["sha256"],
             f"{label} artifact hash differs")
    if temporary:
        stable.verify_unchanged()
        stable.close()
    return path


def _atomic_receipt(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
            os.chmod(path, 0o400)
        except FileExistsError as error:
            raise PhaseBReproductionError(
                "refusing stale or partial reproduction receipt") from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) \
            | getattr(os, "O_CLOEXEC", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


@dataclass(frozen=True)
class ReproductionContract:
    cohorts: tuple[str, ...]
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    ages: tuple[int, ...]
    classes: Mapping[str, int]
    formal_rows: Mapping[str, int]
    consumer_rows: Mapping[str, int]
    variants: Mapping[str, int]
    physical_gpus: Mapping[str, int]
    require_600: bool = True

    @property
    def total_cells(self) -> int:
        return len(self.cohorts) * len(self.arms) * len(self.seeds)

    @property
    def raw_references(self) -> int:
        return len(self.cohorts) * len(self.seeds)

    def identity(self) -> dict[str, Any]:
        return {
            "cohorts": list(self.cohorts),
            "arms": list(self.arms),
            "seeds": list(self.seeds),
            "ages": list(self.ages),
            "classes": dict(self.classes),
            "formal_rows": dict(self.formal_rows),
            "consumer_rows": dict(self.consumer_rows),
            "variants": dict(self.variants),
            "physical_gpus": dict(self.physical_gpus),
        }

    @property
    def is_registered_production(self) -> bool:
        expected = {
            "cohorts": list(COHORTS), "arms": list(ARMS),
            "seeds": list(SEEDS), "ages": list(AGES),
            "classes": dict(CLASSES), "formal_rows": dict(FORMAL_ROWS),
            "consumer_rows": dict(CONSUMER_ROWS),
            "variants": dict(VARIANTS),
            "physical_gpus": dict(PHYSICAL_GPUS),
        }
        return _canonical_json(self.identity()) == _canonical_json(expected)

    def validate(self) -> None:
        _require(bool(self.cohorts) and bool(self.arms) and bool(self.seeds),
                 "reproduction contract is empty")
        _require(len(set(self.cohorts)) == len(self.cohorts)
                 and len(set(self.arms)) == len(self.arms)
                 and len(set(self.seeds)) == len(self.seeds)
                 and all(isinstance(value, str) and value
                         for value in (*self.cohorts, *self.arms))
                 and all(isinstance(value, int)
                         and not isinstance(value, bool) and value >= 0
                         for value in self.seeds),
                 "reproduction contract identities are malformed")
        _require(self.ages == AGES, "registered ages must be exactly 4/8/15")
        for mapping in (self.classes, self.formal_rows, self.consumer_rows,
                        self.variants, self.physical_gpus):
            _require(set(mapping) == set(self.cohorts),
                     "contract cohort registries differ")
        for mapping in (self.classes, self.formal_rows, self.consumer_rows,
                        self.variants):
            _require(all(isinstance(value, int)
                         and not isinstance(value, bool) and value > 0
                         for value in mapping.values()),
                     "contract row/class/variant counts are malformed")
        _require(all(isinstance(value, int) and not isinstance(value, bool)
                     and value in (0, 1, 2)
                     for value in self.physical_gpus.values()),
                 "contract GPU ownership is malformed")
        if self.require_600:
            _require(self.is_registered_production
                     and self.total_cells == 600
                     and self.raw_references == 50,
                     "production reproduction must use the exact registered "
                     "contract and cover 600 cells/50 refs")
        _require(not self.is_registered_production or self.require_600,
                 "the exact production contract cannot be downgraded to a "
                 "test contract")


PRODUCTION_CONTRACT = ReproductionContract(
    cohorts=COHORTS,
    arms=ARMS,
    seeds=SEEDS,
    ages=AGES,
    classes=CLASSES,
    formal_rows=FORMAL_ROWS,
    consumer_rows=CONSUMER_ROWS,
    variants=VARIANTS,
    physical_gpus=PHYSICAL_GPUS,
    require_600=True,
)


@dataclass(frozen=True)
class ExpectedHashes:
    verifier_source: str
    protocol_lock: str
    phase_a_grid: str
    raw_context_summary: str
    label_registry: str
    execution_registry: str
    finalizer_summary: str
    finalized_cells: str
    formal_report: str

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            _expected_sha(value, name)


@dataclass(frozen=True)
class InputPaths:
    protocol_lock: Path
    phase_a_root: Path
    raw_context_root: Path
    label_registry: Path
    execution_registry: Path
    finalized_root: Path
    prepare_root: Path
    formal_report: Path


@dataclass(frozen=True)
class PhaseCell:
    cohort: str
    arm: str
    seed: int
    manifest_sha256: str
    bank_sha256: str
    measurement: Path
    feature_dimension: int
    arrays: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class LabelSet:
    formal_test_episode_id: np.ndarray
    formal_test_native_cluster_id: np.ndarray
    formal_test_label: np.ndarray
    consumer_train_episode_id: np.ndarray
    consumer_train_native_cluster_id: np.ndarray
    consumer_train_label: np.ndarray


@dataclass(frozen=True)
class RawReference:
    cohort: str
    seed: int
    manifest_sha256: str
    measurement: Path
    feature_dimension: int
    arrays: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ExecutionDeck:
    cohort: str
    threshold: float
    controller_sha256: str
    class_success: np.ndarray
    oracle_success: np.ndarray
    random_success: np.ndarray

    @property
    def eligible(self) -> bool:
        return float(np.mean(self.oracle_success)) >= self.threshold


def _repo_relative(path: Path, workspace: Path) -> str:
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def _environment_identity() -> dict[str, Any]:
    """Return a stable numerical-runtime identity without host identifiers."""
    try:
        import scipy
        import sklearn
        from threadpoolctl import threadpool_info
    except ImportError as error:
        raise PhaseBReproductionError(
            "scipy, scikit-learn, and threadpoolctl are required") from error
    thread_variables = (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )
    def normalized_path(path: Path) -> str:
        resolved = path.resolve()
        prefix = Path(sys.prefix).resolve()
        try:
            return "$PYTHON_PREFIX/" + resolved.relative_to(prefix).as_posix()
        except ValueError:
            return "$SYSTEM/" + resolved.name

    def file_identity(path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        _require(resolved.is_file() and not resolved.is_symlink(),
                 f"runtime binary/source is absent or unsafe: {resolved}")
        return {"path": normalized_path(resolved),
                "sha256": _sha256_file(resolved),
                "size": _stable_size(resolved)}

    pools = []
    stable_fields = (
        "user_api", "internal_api", "prefix", "version", "threading_layer",
        "architecture", "num_threads",
    )
    blas_binaries: dict[str, dict[str, Any]] = {}
    for record in threadpool_info():
        normalized = {name: record.get(name) for name in stable_fields}
        raw_library = record.get("filepath")
        if isinstance(raw_library, str) and raw_library:
            identity = file_identity(Path(raw_library))
            normalized["binary_sha256"] = identity["sha256"]
            blas_binaries[identity["sha256"]] = identity
        else:
            normalized["binary_sha256"] = None
        pools.append(normalized)
    pools.sort(key=lambda value: _canonical_json(value))
    modules = {
        "numpy": np,
        "scipy": scipy,
        "sklearn": sklearn,
        "threadpoolctl": sys.modules["threadpoolctl"],
    }
    module_identities: dict[str, dict[str, Any]] = {}
    for name, module in modules.items():
        module_path = getattr(module, "__file__", None)
        _require(isinstance(module_path, str) and module_path,
                 f"runtime module path is absent: {name}")
        module_identities[name] = file_identity(Path(module_path))
    distributions: dict[str, dict[str, Any]] = {}
    for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl"):
        distribution = importlib_metadata.distribution(name)
        record_files = [item for item in (distribution.files or ())
                        if str(item).endswith(".dist-info/RECORD")]
        _require(len(record_files) == 1,
                 f"distribution RECORD is absent or ambiguous: {name}")
        record_path = Path(distribution.locate_file(record_files[0]))
        distributions[name] = {
            "version": distribution.version,
            "record": file_identity(record_path),
        }
    compiled: dict[str, dict[str, Any]] = {}
    extension_suffixes = (".so", ".pyd", ".dll", ".dylib")
    for module_name, module in sorted(sys.modules.items()):
        if module_name.split(".", 1)[0] not in {
                "numpy", "scipy", "sklearn", "threadpoolctl"}:
            continue
        module_path = getattr(module, "__file__", None)
        if not isinstance(module_path, str) or not module_path.endswith(
                extension_suffixes):
            continue
        identity = file_identity(Path(module_path))
        compiled.setdefault(identity["sha256"], identity)
    python_binary = file_identity(Path(sys.executable))
    return {
        "python_executable": os.path.abspath(sys.executable),
        "python_executable_realpath": str(Path(sys.executable).resolve()),
        "python_executable_identity": python_binary,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "blas_thread_environment": {
            name: os.environ.get(name, "unset") for name in thread_variables},
        "normalized_threadpools": pools,
        "blas_library_identities": [blas_binaries[key]
                                    for key in sorted(blas_binaries)],
        "module_source_identities": module_identities,
        "distribution_record_identities": distributions,
        "loaded_extension_identities": [compiled[key]
                                        for key in sorted(compiled)],
        "normalization": (
            "runtime paths normalized to $PYTHON_PREFIX or basename-only "
            "$SYSTEM; process IDs, hostnames, and timestamps excluded; "
            "cryptographic file identities retained"),
    }


def _verify_lock_record(workspace: Path, record: Mapping[str, Any],
                        label: str, *, path_override: str | None = None) \
        -> dict[str, Any]:
    expected = {"sha256", "size"} | ({"path"} if path_override is None
                                      else set())
    _exact_keys(record, expected, label)
    raw_path = record.get("path") if path_override is None else path_override
    path = _safe_existing(str(raw_path), workspace, label, kind="file")
    stable, temporary = _stable_file(path)
    _require(isinstance(record["size"], int)
             and not isinstance(record["size"], bool)
             and stable.size == record["size"],
             f"{label} size differs from protocol lock")
    _require(_is_sha256(record["sha256"])
             and stable.initial_sha256 == record["sha256"],
             f"{label} hash differs from protocol lock")
    if temporary:
        stable.verify_unchanged()
        stable.close()
    return {"path": _repo_relative(path, workspace),
            "sha256": record["sha256"], "size": record["size"]}


def _authenticate_protocol_lock(path: Path, expected_hash: str,
                                workspace: Path,
                                contract: ReproductionContract) \
        -> tuple[dict[str, Any], str]:
    digest = _sha256_file(path)
    _require(digest == expected_hash, "protocol-lock operator hash differs")
    lock = _read_json(path, "protocol lock")
    _exact_keys(lock, {
        "development_audit", "formal_amendment", "formal_execution_started",
        "integration_identities", "preselection_source_receipt",
        "producer_identities", "protocol_fingerprint", "schema_version",
        "seed_registry", "spec_sha256", "stage", "status", "study",
    }, "protocol lock")
    _require(lock["schema_version"] == 1 and lock["study"] == "sage-mem-v1"
             and lock["stage"] == "seal" and lock["status"] == "sealed"
             and lock["formal_execution_started"] is False
             and _is_sha256(lock["protocol_fingerprint"])
             and _is_sha256(lock["spec_sha256"]),
             "protocol lock identity/boundary differs")
    inventory: list[dict[str, Any]] = []
    inventory.append(_verify_lock_record(
        workspace, lock["development_audit"], "locked development audit"))
    amendment = lock["formal_amendment"]
    _require(isinstance(amendment, dict)
             and amendment.get("status") ==
             "locked-before-development-selection-or-formal-data",
             "formal amendment lock status differs")
    inventory.append(_verify_lock_record(
        workspace, {key: amendment[key] for key in ("path", "sha256", "size")},
        "locked formal amendment"))
    inventory.append(_verify_lock_record(
        workspace, lock["preselection_source_receipt"],
        "locked preselection receipt"))
    integrations = lock["integration_identities"]
    _require(isinstance(integrations, dict)
             and set(integrations) == {"host_adapter", "model"},
             "integration lock registry differs")
    for name in sorted(integrations):
        inventory.append(_verify_lock_record(
            workspace, integrations[name], f"locked integration {name}"))
    producers = lock["producer_identities"]
    _require(isinstance(producers, dict) and producers,
             "producer identity registry is absent")
    for name in sorted(producers):
        _require(isinstance(name, str) and name,
                 "producer path key is malformed")
        inventory.append(_verify_lock_record(
            workspace, producers[name], f"locked producer {name}",
            path_override=name))
    spec_path = _safe_existing("configs/sage_mem_v1.yaml", workspace,
                               "registered protocol spec", kind="file")
    _require(_sha256_file(spec_path) == lock["spec_sha256"],
             "registered protocol spec hash differs")
    try:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise PhaseBReproductionError("cannot parse registered protocol") \
            from error
    _require(_sha256_json(spec) == lock["protocol_fingerprint"],
             "protocol fingerprint differs from locked spec")
    if contract.is_registered_production:
        required_replay_sources = {
            "configs/sage_mem_v1.yaml",
            "scripts/sage_mem_v1_formal_finalizer.py",
            "scripts/audit_sage_mem_v1_formal.py",
            "scripts/sage_mem_v1_spec.py",
        }
        _require(required_replay_sources.issubset(producers),
                 "production lock omits finalizer/spec/formal-auditor "
                 "producer identities")
    _require(isinstance(lock["seed_registry"], dict)
             and all(isinstance(key, str)
                     and isinstance(value, int) and not isinstance(value, bool)
                     for key, value in lock["seed_registry"].items()),
             "locked seed registry is malformed")
    return lock, _sha256_json(inventory)


def _validate_age_identity(arrays: Mapping[str, np.ndarray], prefix: str,
                           count: int,
                           contract: ReproductionContract) -> None:
    shape = (len(contract.ages), count)
    episode = arrays[f"{prefix}_episode_id"]
    cluster = arrays[f"{prefix}_native_cluster_id"]
    age = arrays[f"{prefix}_evidence_age"]
    _require(all(value.shape == shape
                 and np.issubdtype(value.dtype, np.integer)
                 for value in (episode, cluster, age)),
             f"{prefix} identity tensor differs")
    _require(np.all(episode >= 0) and np.all(cluster >= 0)
             and len(np.unique(episode[0])) == count,
             f"{prefix} identities are malformed")
    for index, evidence_age in enumerate(contract.ages):
        _require(np.all(age[index] == evidence_age)
                 and np.array_equal(episode[index], episode[0])
                 and np.array_equal(cluster[index], cluster[0]),
                 f"{prefix} identities drift across age")


def _validate_clusters(episode: np.ndarray, cluster: np.ndarray,
                       variants: int, label: str,
                       semantic: np.ndarray | None = None,
                       classes: int | None = None) -> None:
    unique, counts = np.unique(cluster, return_counts=True)
    _require(episode.ndim == cluster.ndim == 1
             and len(episode) == len(cluster)
             and len(unique) * variants == len(cluster)
             and np.all(counts == variants),
             f"{label} native-cluster multiplicity differs")
    if semantic is not None and variants == classes:
        _require(all(set(map(int, semantic[cluster == native])) ==
                     set(range(int(classes))) for native in unique),
                 f"{label} counterfactual cluster omits a class")


def _load_phase_grid(root: Path, contract: ReproductionContract,
                     protocol_fingerprint: str,
                     expected_grid: str) \
        -> tuple[dict[tuple[str, str, int], PhaseCell], str, str]:
    cells_root = root / "cells"
    _require(cells_root.is_dir() and not cells_root.is_symlink(),
             "Phase-A cells root is absent or unsafe")
    _require({item.name for item in _iterdir(cells_root)} ==
             set(contract.cohorts)
             and all(item.is_dir() and not item.is_symlink()
                     for item in _iterdir(cells_root)),
             "Phase-A cohort inventory differs")
    cells: dict[tuple[str, str, int], PhaseCell] = {}
    digest_rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    canonical_identity: dict[str, dict[str, np.ndarray]] = {}
    canonical_bank: dict[str, str] = {}
    canonical_dimension: dict[str, int] = {}
    for cohort in contract.cohorts:
        cohort_root = cells_root / cohort
        _require({item.name for item in _iterdir(cohort_root)} ==
                 set(contract.arms)
                 and all(item.is_dir() and not item.is_symlink()
                         for item in _iterdir(cohort_root)),
                 f"Phase-A arm inventory differs: {cohort}")
        for arm in contract.arms:
            arm_root = cohort_root / arm
            _require({item.name for item in _iterdir(arm_root)} ==
                     {f"seed-{seed}" for seed in contract.seeds}
                     and all(item.is_dir() and not item.is_symlink()
                             for item in _iterdir(arm_root)),
                     f"Phase-A seed inventory differs: {cohort}/{arm}")
            for seed in contract.seeds:
                directory = arm_root / f"seed-{seed}"
                manifest_path = directory / "manifest.json"
                _require(manifest_path.is_file()
                         and not manifest_path.is_symlink(),
                         "Phase-A manifest is absent or unsafe")
                manifest = _read_json(manifest_path, "Phase-A manifest")
                _exact_keys(manifest, {
                    "schema", "study", "stage", "status", "cohort", "arm",
                    "seed", "physical_gpu", "cuda_visible_devices",
                    "protocol_fingerprint", "completed_unix_ns", "ages",
                    "formal_test_labels_read",
                    "formal_test_labels_available",
                    "development_outcomes_read", "labels_used_for_training",
                    "bank_manifest_sha256", "host_hash_before",
                    "host_hash_after", "prediction_representation",
                    "consumer_contract", "shared_consumer_sha256",
                    "artifacts",
                }, "Phase-A manifest")
                _require(manifest["schema"] == PHASE_A_SCHEMA
                         and manifest["study"] == "sage-mem-v1"
                         and manifest["stage"] == "formal-phase-a"
                         and manifest["status"] == "complete-label-free"
                         and manifest["cohort"] == cohort
                         and manifest["arm"] == arm
                         and manifest["seed"] == seed
                         and manifest["ages"] == list(contract.ages)
                         and manifest["physical_gpu"] ==
                         contract.physical_gpus[cohort]
                         and manifest["cuda_visible_devices"] ==
                         str(contract.physical_gpus[cohort])
                         and manifest["protocol_fingerprint"] ==
                         protocol_fingerprint,
                         "Phase-A manifest identity differs")
                _require(manifest["formal_test_labels_read"] is False
                         and manifest["formal_test_labels_available"] is False
                         and manifest["development_outcomes_read"] is False
                         and manifest["labels_used_for_training"] is False,
                         "Phase-A cell crossed the sealed-label boundary")
                _require(manifest["prediction_representation"] ==
                         "feature_artifact"
                         and manifest["consumer_contract"] ==
                         "centralized-pooled-consumer-train-features"
                         and manifest["shared_consumer_sha256"] is None,
                         "Phase-B reproduction requires deferred feature "
                         "artifacts for every cell")
                _require(_is_sha256(manifest["bank_manifest_sha256"])
                         and _is_sha256(manifest["host_hash_before"])
                         and manifest["host_hash_before"] ==
                         manifest["host_hash_after"],
                         "Phase-A bank/host identity differs")
                artifacts = manifest["artifacts"]
                _require(isinstance(artifacts, dict),
                         "Phase-A artifacts are malformed")
                _exact_keys(artifacts, {
                    "measurements", "checkpoint", "history",
                    "resource_report"}, "Phase-A artifacts")
                paths = {name: _safe_artifact(
                    directory, artifacts[name], f"Phase-A {name}")
                    for name in artifacts}
                _require({item.name for item in _iterdir(directory)} ==
                         {"manifest.json", *(path.name for path in paths.values())},
                         "Phase-A cell inventory has missing or extra files")
                try:
                    with _open_npz(paths["measurements"]) as z:
                        _require(set(z.files) ==
                                 PHASE_COMMON_KEYS | PHASE_FEATURE_KEYS,
                                 "Phase-A measurement schema differs")
                        arrays = {name: np.asarray(z[name]).copy()
                                  for name in PHASE_COMMON_KEYS}
                        dimensions: set[int] = set()
                        for name in PHASE_FEATURE_KEYS:
                            value = np.asarray(z[name])
                            rows = (contract.consumer_rows[cohort]
                                    if name.startswith("consumer_train")
                                    else contract.formal_rows[cohort])
                            _require(value.ndim == 3
                                     and value.shape[:2] ==
                                     (len(contract.ages), rows)
                                     and value.shape[2] > 0
                                     and np.issubdtype(value.dtype,
                                                      np.number)
                                     and np.isfinite(value).all(),
                                     f"Phase-A feature tensor differs: {name}")
                            dimensions.add(int(value.shape[2]))
                        _require(len(dimensions) == 1,
                                 "Phase-A feature dimensions differ")
                        dimension = dimensions.pop()
                except (OSError, ValueError) as error:
                    raise PhaseBReproductionError(
                        "cannot load Phase-A measurement artifact") from error
                _validate_age_identity(arrays, "formal_test",
                                       contract.formal_rows[cohort], contract)
                _validate_age_identity(arrays, "consumer_train",
                                       contract.consumer_rows[cohort], contract)
                _validate_clusters(
                    arrays["formal_test_episode_id"][0],
                    arrays["formal_test_native_cluster_id"][0],
                    contract.variants[cohort], f"{cohort}/formal_test")
                _validate_clusters(
                    arrays["consumer_train_episode_id"][0],
                    arrays["consumer_train_native_cluster_id"][0],
                    contract.variants[cohort], f"{cohort}/consumer_train")
                expected_shape = (len(contract.ages),
                                  contract.formal_rows[cohort])
                for name in ("formal_test_full_mse", "formal_test_reset_mse",
                             "formal_test_prior_mse"):
                    value = arrays[name]
                    _require(value.shape == expected_shape
                             and np.issubdtype(value.dtype, np.number)
                             and np.isfinite(value).all() and np.all(value >= 0),
                             f"Phase-A MSE differs: {name}")
                identity_names = (
                    "formal_test_episode_id",
                    "formal_test_native_cluster_id",
                    "formal_test_evidence_age",
                    "consumer_train_episode_id",
                    "consumer_train_native_cluster_id",
                    "consumer_train_evidence_age",
                )
                if cohort not in canonical_identity:
                    canonical_identity[cohort] = {
                        name: arrays[name] for name in identity_names}
                    canonical_bank[cohort] = manifest["bank_manifest_sha256"]
                    canonical_dimension[cohort] = dimension
                _require(manifest["bank_manifest_sha256"] ==
                         canonical_bank[cohort] and dimension ==
                         canonical_dimension[cohort]
                         and all(np.array_equal(arrays[name],
                                                canonical_identity[cohort][name])
                                 for name in identity_names),
                         f"Phase-A cross-cell identity drifts: {cohort}")
                manifest_hash = _sha256_file(manifest_path)
                cell = PhaseCell(
                    cohort=cohort, arm=arm, seed=seed,
                    manifest_sha256=manifest_hash,
                    bank_sha256=manifest["bank_manifest_sha256"],
                    measurement=paths["measurements"],
                    feature_dimension=dimension, arrays=arrays)
                cells[(cohort, arm, seed)] = cell
                digest_rows.append({
                    "cohort": cohort, "arm": arm, "seed": seed,
                    "manifest_sha256": manifest_hash,
                    "bank_manifest_sha256": cell.bank_sha256,
                    "measurement_identity_sha256": _sha256_json({
                        name: arrays[name].astype(np.int64).tolist()
                        for name in identity_names}),
                })
                inventory.append({
                    "cohort": cohort, "arm": arm, "seed": seed,
                    "manifest_sha256": manifest_hash,
                    "artifacts": {name: artifacts[name]["sha256"]
                                  for name in sorted(artifacts)},
                })
    _require(len(cells) == contract.total_cells,
             "Phase-A cell count differs from the exact contract")
    grid_digest = _sha256_json(digest_rows)
    _require(grid_digest == expected_grid,
             "Phase-A grid differs from operator-pinned digest")
    return cells, grid_digest, _sha256_json(inventory)


def _registry_source(parent: Path, source: Mapping[str, Any], label: str,
                     desired_episode: np.ndarray,
                     desired_cluster: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    _exact_keys(source, {"artifact", "keys"}, f"{label} source")
    keys = source["keys"]
    _require(isinstance(keys, dict), f"{label} key map is malformed")
    _exact_keys(keys, {"episode_id", "native_cluster_id", "label"},
                f"{label} key map")
    _require(all(isinstance(keys[name], str) and keys[name]
                 for name in ("episode_id", "label"))
             and (keys["native_cluster_id"] is None
                  or (isinstance(keys["native_cluster_id"], str)
                      and keys["native_cluster_id"])),
             f"{label} key map is malformed")
    artifact = _safe_artifact(parent, source["artifact"], label, nested=True)
    required = {keys["episode_id"], keys["label"]}
    if keys["native_cluster_id"] is not None:
        required.add(keys["native_cluster_id"])
    try:
        with _open_npz(artifact) as z:
            _require(required.issubset(z.files),
                     f"{label} custody vault omits declared keys")
            episode = np.asarray(z[keys["episode_id"]]).copy()
            semantic = np.asarray(z[keys["label"]]).copy()
            cluster = (episode.copy() if keys["native_cluster_id"] is None
                       else np.asarray(z[keys["native_cluster_id"]]).copy())
    except (OSError, ValueError) as error:
        raise PhaseBReproductionError(f"cannot open {label} custody vault") \
            from error
    _require(all(value.ndim == 1 and len(value) == len(episode)
                 and np.issubdtype(value.dtype, np.integer)
                 for value in (episode, cluster, semantic))
             and len(np.unique(episode)) == len(episode),
             f"{label} custody arrays are malformed")
    lookup = {int(identity): index for index, identity in enumerate(episode)}
    _require(all(int(identity) in lookup for identity in desired_episode),
             f"{label} custody vault omits split identities")
    rows = np.asarray([lookup[int(identity)] for identity in desired_episode],
                      dtype=np.int64)
    episode = episode[rows].astype(np.int64, copy=False)
    cluster = cluster[rows].astype(np.int64, copy=False)
    semantic = semantic[rows].astype(np.int64, copy=False)
    _require(np.array_equal(episode, desired_episode)
             and np.array_equal(cluster, desired_cluster),
             f"{label} custody identity differs from Phase A")
    return episode, cluster, semantic, source["artifact"]["sha256"]


def _load_labels(path: Path, expected_hash: str,
                 cells: Mapping[tuple[str, str, int], PhaseCell],
                 contract: ReproductionContract) \
        -> tuple[dict[str, LabelSet], str]:
    _require(_sha256_file(path) == expected_hash,
             "label-registry operator hash differs")
    registry = _read_json(path, "sealed label registry")
    _exact_keys(registry, {
        "schema", "study", "status",
        "labels_available_only_after_complete_phase_a_grid",
        "development_outcomes_read", "cohorts",
    }, "sealed label registry")
    _require(registry["schema"] in LABEL_SCHEMAS
             and registry["study"] == "sage-mem-v1"
             and registry["status"] == "sealed"
             and registry[
                 "labels_available_only_after_complete_phase_a_grid"] is True
             and registry["development_outcomes_read"] is False
             and isinstance(registry["cohorts"], dict)
             and set(registry["cohorts"]) == set(contract.cohorts),
             "sealed label registry identity differs")
    if contract.is_registered_production:
        _require(registry["schema"] ==
                 "sage_mem_v1_custody_vault_registry_v1",
                 "production reproduction requires the original custody "
                 "registry, not a normalized derivative")
    custody = registry["schema"] == "sage_mem_v1_custody_vault_registry_v1"
    result: dict[str, LabelSet] = {}
    artifacts: list[dict[str, Any]] = []
    for cohort in contract.cohorts:
        record = registry["cohorts"][cohort]
        _require(isinstance(record, dict), "label cohort record is malformed")
        _exact_keys(record, {
            "bank_manifest_sha256", "classes",
            ("sources" if custody else "artifact")},
            f"{cohort} label record")
        canonical = cells[(cohort, contract.arms[0], contract.seeds[0])]
        _require(record["bank_manifest_sha256"] == canonical.bank_sha256
                 and record["classes"] == contract.classes[cohort],
                 f"{cohort} label record differs from Phase A")
        arrays: dict[str, np.ndarray] = {}
        artifact_hashes: list[str] = []
        if custody:
            sources = record["sources"]
            _require(isinstance(sources, dict), "custody sources malformed")
            _exact_keys(sources, {"formal_test", "consumer_train"},
                        f"{cohort} custody sources")
            for split in ("formal_test", "consumer_train"):
                episode, cluster, semantic, artifact_hash = _registry_source(
                    path.parent, sources[split], f"{cohort}/{split}",
                    canonical.arrays[f"{split}_episode_id"][0],
                    canonical.arrays[f"{split}_native_cluster_id"][0])
                arrays[f"{split}_episode_id"] = episode
                arrays[f"{split}_native_cluster_id"] = cluster
                arrays[f"{split}_label"] = semantic
                artifact_hashes.append(artifact_hash)
        else:
            artifact = _safe_artifact(path.parent, record["artifact"],
                                      f"{cohort} labels", nested=True)
            expected = {
                "formal_test_episode_id", "formal_test_native_cluster_id",
                "formal_test_label", "consumer_train_episode_id",
                "consumer_train_native_cluster_id", "consumer_train_label",
            }
            try:
                with _open_npz(artifact) as z:
                    _require(set(z.files) == expected,
                             "consolidated label schema differs")
                    arrays = {name: np.asarray(z[name]).copy()
                              for name in expected}
            except (OSError, ValueError) as error:
                raise PhaseBReproductionError(
                    "cannot open consolidated label artifact") from error
            artifact_hashes.append(record["artifact"]["sha256"])
        for split, rows in (("formal_test", contract.formal_rows[cohort]),
                            ("consumer_train",
                             contract.consumer_rows[cohort])):
            episode = arrays[f"{split}_episode_id"]
            cluster = arrays[f"{split}_native_cluster_id"]
            semantic = arrays[f"{split}_label"]
            _require(all(value.shape == (rows,)
                         and np.issubdtype(value.dtype, np.integer)
                         for value in (episode, cluster, semantic))
                     and np.all((semantic >= 0)
                                & (semantic < contract.classes[cohort]))
                     and np.array_equal(
                         episode, canonical.arrays[f"{split}_episode_id"][0])
                     and np.array_equal(
                         cluster,
                         canonical.arrays[f"{split}_native_cluster_id"][0]),
                     f"{cohort}/{split} revealed labels are unaligned")
            _validate_clusters(
                episode, cluster, contract.variants[cohort],
                f"{cohort}/{split}", semantic,
                contract.classes[cohort])
        result[cohort] = LabelSet(**arrays)
        artifacts.append({
            "cohort": cohort,
            "artifact_sha256": sorted(set(artifact_hashes)),
            "semantic_sha256": _sha256_json({
                name: _array_digest(value) for name, value in sorted(arrays.items())
            }),
        })
    return result, _sha256_json(artifacts)


def _load_raw_references(root: Path, expected_summary_hash: str,
                         cells: Mapping[tuple[str, str, int], PhaseCell],
                         contract: ReproductionContract) \
        -> tuple[dict[tuple[str, int], RawReference], str]:
    summary_path = root / "summary.json"
    _require(summary_path.is_file() and not summary_path.is_symlink()
             and _sha256_file(summary_path) == expected_summary_hash,
             "raw-context summary operator hash differs or is absent")
    summary = _read_json(summary_path, "raw-context producer summary")
    _exact_keys(summary, {
        "schema", "study", "status", "cells", "cohorts", "seeds",
        "feature_contract", "formal_labels_read",
        "development_outcomes_read", "mse_emitted", "records_sha256",
    }, "raw-context producer summary")
    _require(summary["schema"] == "sage_mem_v1_raw_context_producer_v1"
             and summary["study"] == "sage-mem-v1"
             and summary["status"] == "complete-label-free"
             and summary["cells"] == contract.raw_references
             and summary["cohorts"] == list(contract.cohorts)
             and summary["seeds"] == list(contract.seeds)
             and summary["feature_contract"] == RAW_FEATURE_CONTRACT
             and summary["formal_labels_read"] is False
             and summary["development_outcomes_read"] is False
             and summary["mse_emitted"] is False,
             "raw-context summary is incomplete or changed")
    _require({item.name for item in _iterdir(root)} ==
             set(contract.cohorts) | {"summary.json"},
             "raw-context top-level inventory differs")
    references: dict[tuple[str, int], RawReference] = {}
    records: list[dict[str, Any]] = []
    semantic: list[dict[str, Any]] = []
    for cohort in contract.cohorts:
        cohort_root = root / cohort
        _require(cohort_root.is_dir() and not cohort_root.is_symlink()
                 and {item.name for item in _iterdir(cohort_root)} ==
                 {f"seed-{seed}" for seed in contract.seeds}
                 and all(item.is_dir() and not item.is_symlink()
                         for item in _iterdir(cohort_root)),
                 f"raw-context seed inventory differs: {cohort}")
        canonical = cells[(cohort, contract.arms[0], contract.seeds[0])]
        for seed in contract.seeds:
            directory = cohort_root / f"seed-{seed}"
            manifest_path = directory / "manifest.json"
            _require(manifest_path.is_file()
                     and not manifest_path.is_symlink(),
                     "raw-context manifest is absent or unsafe")
            manifest = _read_json(manifest_path, "raw-context manifest")
            _exact_keys(manifest, {
                "schema", "study", "stage", "status", "cohort", "seed",
                "ages", "short_context_frames", "long_context_frames",
                "separate_from_parameter_matched_arms",
                "formal_test_labels_read", "development_outcomes_read",
                "bank_manifest_sha256", "host_hash_before", "host_hash_after",
                "consumer_contract", "shared_consumer_sha256",
                "feature_contract", "artifact",
            }, "raw-context manifest")
            _require(manifest["schema"] == RAW_SCHEMA
                     and manifest["study"] == "sage-mem-v1"
                     and manifest["stage"] ==
                     "formal-raw-context-reference"
                     and manifest["status"] == "complete-label-free"
                     and manifest["cohort"] == cohort
                     and manifest["seed"] == seed
                     and manifest["ages"] == list(contract.ages)
                     and manifest["short_context_frames"] == 3
                     and manifest["long_context_frames"] == 16
                     and manifest[
                         "separate_from_parameter_matched_arms"] is True
                     and manifest["formal_test_labels_read"] is False
                     and manifest["development_outcomes_read"] is False
                     and manifest["bank_manifest_sha256"] ==
                     canonical.bank_sha256
                     and manifest["host_hash_before"] ==
                     manifest["host_hash_after"]
                     and _is_sha256(manifest["host_hash_before"])
                     and manifest["consumer_contract"] ==
                     "post-reveal-shared-short-long-arm-blind"
                     and manifest["shared_consumer_sha256"] is None
                     and manifest["feature_contract"] == RAW_FEATURE_CONTRACT,
                     "raw-context manifest identity differs")
            artifact = _safe_artifact(directory, manifest["artifact"],
                                      "raw-context measurement")
            _require({item.name for item in _iterdir(directory)} ==
                     {"manifest.json", artifact.name},
                     "raw-context cell inventory differs")
            try:
                with _open_npz(artifact) as z:
                    _require(set(z.files) == RAW_KEYS,
                             "raw-context measurement schema differs")
                    arrays = {name: np.asarray(z[name]).copy()
                              for name in RAW_KEYS
                              if not name.endswith("_features")}
                    dimensions: set[int] = set()
                    for name in RAW_KEYS:
                        if not name.endswith("_features"):
                            continue
                        value = np.asarray(z[name])
                        rows = (contract.consumer_rows[cohort]
                                if name.startswith("consumer_train")
                                else contract.formal_rows[cohort])
                        _require(value.ndim == 3
                                 and value.shape[:2] ==
                                 (len(contract.ages), rows)
                                 and value.shape[2] > 0
                                 and np.issubdtype(value.dtype, np.number)
                                 and np.isfinite(value).all(),
                                 f"raw-context feature differs: {name}")
                        dimensions.add(int(value.shape[2]))
                    _require(len(dimensions) == 1,
                             "raw-context feature dimensions differ")
                    dimension = dimensions.pop()
            except (OSError, ValueError) as error:
                raise PhaseBReproductionError(
                    "cannot load raw-context measurement") from error
            for split, rows in (("formal_test", contract.formal_rows[cohort]),
                                ("consumer_train",
                                 contract.consumer_rows[cohort])):
                _validate_age_identity(arrays, split, rows, contract)
                for suffix in ("episode_id", "native_cluster_id",
                               "evidence_age"):
                    name = f"{split}_{suffix}"
                    _require(np.array_equal(arrays[name],
                                            canonical.arrays[name]),
                             f"raw-context identity drifts: {cohort}/{name}")
            manifest_hash = _sha256_file(manifest_path)
            references[(cohort, seed)] = RawReference(
                cohort=cohort, seed=seed,
                manifest_sha256=manifest_hash,
                measurement=artifact, feature_dimension=dimension,
                arrays=arrays)
            records.append({
                "cohort": cohort, "seed": seed,
                "manifest_sha256": manifest_hash,
                "artifact_sha256": manifest["artifact"]["sha256"],
                "bank_manifest_sha256": canonical.bank_sha256,
            })
            semantic.append({
                "cohort": cohort, "seed": seed,
                "identity_sha256": _sha256_json({
                    name: _array_digest(value)
                    for name, value in sorted(arrays.items())}),
                "feature_artifact_sha256": manifest["artifact"]["sha256"],
            })
    _require(len(references) == contract.raw_references
             and summary["records_sha256"] == _sha256_json(records),
             "raw-context inventory digest differs")
    return references, _sha256_json(semantic)


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray,
               test_blocks: Iterable[np.ndarray], classes: int) \
        -> tuple[list[np.ndarray], str]:
    """Independent registered pooled consumer implementation.

    This intentionally does not import or call any finalizer prediction helper.
    """
    try:
        from sklearn.linear_model import RidgeClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise PhaseBReproductionError(
            "scikit-learn is required for Phase-B reproduction") from error
    x = np.asarray(train_x, dtype=np.float32)
    y = np.asarray(train_y, dtype=np.int64)
    _require(x.ndim == 2 and y.shape == (len(x),) and len(x) > 0
             and np.isfinite(x).all()
             and set(np.unique(y).tolist()) == set(range(classes)),
             "pooled Ridge training data are malformed")
    scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
    normalized = scaler.fit_transform(x)
    classifier = RidgeClassifier(
        alpha=1e-3, fit_intercept=True, solver="lsqr",
        tol=1e-6, max_iter=5_000)
    try:
        classifier.fit(normalized, y)
    except (FloatingPointError, ValueError) as error:
        raise PhaseBReproductionError("independent Ridge fit failed") \
            from error
    _require(np.array_equal(classifier.classes_, np.arange(classes)),
             "independent Ridge class ordering differs")
    predictions: list[np.ndarray] = []
    for block in test_blocks:
        value = np.array(block, dtype=np.float32, copy=True)
        _require(value.ndim == 2 and value.shape[1] == x.shape[1]
                 and np.isfinite(value).all(),
                 "independent Ridge test feature shape differs")
        predictions.append(classifier.predict(
            scaler.transform(value)).astype(np.int64))
    digest = hashlib.sha256()
    for value in (scaler.mean_, scaler.scale_, classifier.coef_,
                  np.atleast_1d(classifier.intercept_)):
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(_canonical_json({
        "classes": classes, "ridge": 1e-3,
        "solver": "lsqr", "tol": 1e-6, "max_iter": 5_000,
        "arm_identity_used": False,
        "formal_test_labels_used": False,
    }).encode("utf-8"))
    return predictions, digest.hexdigest()


def _feature_age(cell: PhaseCell, name: str, age_index: int) -> np.ndarray:
    try:
        with _open_npz(cell.measurement) as z:
            return np.asarray(z[name][age_index], dtype=np.float32).copy()
    except (OSError, ValueError, KeyError, IndexError) as error:
        raise PhaseBReproductionError(
            f"cannot stream Phase-A feature: {cell.cohort}/{cell.arm}/"
            f"{cell.seed}/{name}") from error


def _reproduce_carrier_predictions(
        cells: Mapping[tuple[str, str, int], PhaseCell],
        labels: Mapping[str, LabelSet], finalized_root: Path,
        contract: ReproductionContract) \
        -> tuple[dict[tuple[str, str, int], dict[str, np.ndarray]],
                 dict[tuple[str, int], str], str]:
    predictions: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    shared_hashes: dict[tuple[str, int], str] = {}
    semantic_models: list[dict[str, Any]] = []
    consumer_root = finalized_root / "consumers"
    _require(consumer_root.is_dir() and not consumer_root.is_symlink()
             and {item.name for item in _iterdir(consumer_root)} ==
             set(contract.cohorts)
             and all(item.is_dir() and not item.is_symlink()
                     for item in _iterdir(consumer_root)),
             "finalized shared-consumer cohort inventory differs")
    for cohort in contract.cohorts:
        cohort_root = consumer_root / cohort
        _require({item.name for item in _iterdir(cohort_root)} ==
                 {f"seed-{seed}.json" for seed in contract.seeds}
                 and all(item.is_file() and not item.is_symlink()
                         for item in _iterdir(cohort_root)),
                 f"shared-consumer receipt inventory differs: {cohort}")
        for seed in contract.seeds:
            per_arm = {arm: {stream: [] for stream in
                             ("full", "reset", "prior")}
                       for arm in contract.arms}
            age_hashes: list[str] = []
            for age_index, _age in enumerate(contract.ages):
                rows = contract.consumer_rows[cohort]
                dimension = cells[(cohort, contract.arms[0],
                                   seed)].feature_dimension
                pooled_x = np.empty(
                    (rows * len(contract.arms), dimension), dtype=np.float32)
                for arm_index, arm in enumerate(contract.arms):
                    pooled_x[arm_index * rows:(arm_index + 1) * rows] = \
                        _feature_age(cells[(cohort, arm, seed)],
                                     "consumer_train_full_features",
                                     age_index)
                pooled_y = np.tile(labels[cohort].consumer_train_label,
                                   len(contract.arms))
                locations = [(arm, stream) for arm in contract.arms
                             for stream in ("full", "reset", "prior")]
                values, model_hash = _fit_ridge(
                    pooled_x, pooled_y,
                    (_feature_age(
                        cells[(cohort, arm, seed)],
                        f"formal_test_{stream}_features", age_index)
                     for arm, stream in locations),
                    contract.classes[cohort])
                age_hashes.append(model_hash)
                for (arm, stream), value in zip(locations, values):
                    per_arm[arm][stream].append(value)
            shared_hash = _sha256_json({
                "cohort": cohort, "seed": seed,
                "ages": list(contract.ages),
                "age_model_sha256": age_hashes,
                "pooled_arms": list(contract.arms),
                "arm_identity_used": False,
                "fit_split": "consumer_train",
                "formal_test_labels_used": False,
            })
            receipt_path = cohort_root / f"seed-{seed}.json"
            receipt = _read_json(receipt_path, "shared-consumer receipt")
            _exact_keys(receipt, {
                "schema", "study", "stage", "status", "cohort", "seed",
                "ages", "pooled_arms", "training_rows_per_age",
                "arm_identity_used", "formal_test_labels_used",
                "age_model_sha256", "shared_consumer_sha256",
            }, "shared-consumer receipt")
            _require(receipt["schema"] == FINALIZER_SCHEMA
                     and receipt["study"] == "sage-mem-v1"
                     and receipt["stage"] == "shared-consumer"
                     and receipt["status"] ==
                     "fit-on-pooled-consumer-train"
                     and receipt["cohort"] == cohort
                     and receipt["seed"] == seed
                     and receipt["ages"] == list(contract.ages)
                     and receipt["pooled_arms"] == list(contract.arms)
                     and receipt["training_rows_per_age"] ==
                     contract.consumer_rows[cohort] * len(contract.arms)
                     and receipt["arm_identity_used"] is False
                     and receipt["formal_test_labels_used"] is False
                     and receipt["age_model_sha256"] == age_hashes
                     and receipt["shared_consumer_sha256"] == shared_hash,
                     "independent Ridge model differs from finalizer receipt")
            shared_hashes[(cohort, seed)] = shared_hash
            semantic_models.append({
                "cohort": cohort, "seed": seed,
                "age_model_sha256": age_hashes,
                "shared_consumer_sha256": shared_hash,
            })
            for arm in contract.arms:
                predictions[(cohort, arm, seed)] = {
                    stream: np.stack(per_arm[arm][stream], axis=0)
                    for stream in ("full", "reset", "prior")}
    return predictions, shared_hashes, _sha256_json(semantic_models)


def _load_execution_registry(
        path: Path, expected_hash: str,
        labels: Mapping[str, LabelSet],
        cells: Mapping[tuple[str, str, int], PhaseCell],
        contract: ReproductionContract) \
        -> tuple[dict[str, ExecutionDeck], str, dict[str, str]]:
    _require(_sha256_file(path) == expected_hash,
             "execution-registry operator hash differs")
    registry = _read_json(path, "execution-deck registry")
    _exact_keys(registry, {
        "schema", "study", "status",
        "available_only_after_complete_phase_a_grid",
        "development_outcomes_read", "cohorts", "unavailable_cohorts",
    }, "execution-deck registry")
    _require(registry["schema"] == EXECUTION_SCHEMA
             and registry["study"] == "sage-mem-v1"
             and registry["status"] == "sealed"
             and registry[
                 "available_only_after_complete_phase_a_grid"] is True
             and registry["development_outcomes_read"] is False
             and isinstance(registry["cohorts"], dict)
             and isinstance(registry["unavailable_cohorts"], dict),
             "execution-deck registry identity differs")
    supplied = set(registry["cohorts"])
    unavailable = set(registry["unavailable_cohorts"])
    _require(not supplied.intersection(unavailable)
             and supplied | unavailable == set(contract.cohorts),
             "execution registry must classify every cohort exactly once")
    inventory: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    for cohort in sorted(unavailable):
        record = registry["unavailable_cohorts"][cohort]
        _exact_keys(record, {"status", "bank_manifest_sha256",
                             "reason_code", "receipt"},
                    f"{cohort} unavailable execution record")
        canonical = cells[(cohort, contract.arms[0], contract.seeds[0])]
        _require(record["status"] == "unavailable"
                 and record["bank_manifest_sha256"] == canonical.bank_sha256
                 and isinstance(record["reason_code"], str)
                 and 0 < len(record["reason_code"]) <= 160,
                 f"{cohort} unavailable execution record differs")
        receipt_path = _safe_artifact(
            path.parent, record["receipt"],
            f"{cohort} unavailable receipt", nested=True)
        receipt = _read_json(receipt_path, "unavailable execution receipt")
        _exact_keys(receipt, {
            "schema", "study", "status", "cohort", "reason_code",
            "bank_manifest_sha256", "formal_labels_read",
            "development_outcomes_read",
        }, "unavailable execution receipt")
        _require(receipt["schema"] == EXECUTION_UNAVAILABLE_SCHEMA
                 and receipt["study"] == "sage-mem-v1"
                 and receipt["status"] == "unavailable"
                 and receipt["cohort"] == cohort
                 and receipt["reason_code"] == record["reason_code"]
                 and receipt["bank_manifest_sha256"] == canonical.bank_sha256
                 and receipt["formal_labels_read"] is False
                 and receipt["development_outcomes_read"] is False,
                 "unavailable execution receipt differs")
        inventory.append({"cohort": cohort, "status": "unavailable",
                          "receipt_sha256": record["receipt"]["sha256"]})
        statuses[cohort] = "unavailable"
    decks: dict[str, ExecutionDeck] = {}
    for cohort in sorted(supplied):
        record = registry["cohorts"][cohort]
        _exact_keys(record, {
            "bank_manifest_sha256", "classes", "controller",
            "eligibility_gate", "artifact", "replay_receipt",
        }, f"{cohort} execution deck")
        canonical = cells[(cohort, contract.arms[0], contract.seeds[0])]
        _require(record["bank_manifest_sha256"] == canonical.bank_sha256
                 and record["classes"] == contract.classes[cohort],
                 f"{cohort} execution deck differs from Phase A")
        controller = record["controller"]
        _exact_keys(controller, {
            "controller_identity_sha256", "implementation_sha256",
            "physics_sha256", "pinned", "arm_identity_input", "input",
        }, f"{cohort} execution controller")
        _require(all(_is_sha256(controller[name]) for name in
                     ("controller_identity_sha256", "implementation_sha256",
                      "physics_sha256"))
                 and controller["pinned"] is True
                 and controller["arm_identity_input"] is False
                 and controller["input"] == "predicted_class_only",
                 f"{cohort} execution controller differs")
        gate = record["eligibility_gate"]
        _exact_keys(gate, {"metric", "operator", "threshold",
                           "preregistered"}, f"{cohort} execution gate")
        _require(gate["metric"] == "mean_oracle_success"
                 and gate["operator"] == ">="
                 and gate["preregistered"] is True
                 and isinstance(gate["threshold"], (int, float))
                 and not isinstance(gate["threshold"], bool)
                 and math.isfinite(float(gate["threshold"]))
                 and 0 <= float(gate["threshold"]) <= 1,
                 f"{cohort} execution gate differs")
        artifact = _safe_artifact(path.parent, record["artifact"],
                                  f"{cohort} execution deck", nested=True)
        replay_path = _safe_artifact(
            path.parent, record["replay_receipt"],
            f"{cohort} execution replay", nested=True)
        replay = _read_json(replay_path, "execution replay receipt")
        _exact_keys(replay, {
            "schema", "study", "status", "cohort", "bank_manifest_sha256",
            "formal_labels_read", "development_outcomes_read",
            "controller_identity_sha256", "rows", "classes",
            "native_clusters", "executions", "replayed_executions",
            "deterministic_replay_fidelity", "execution_endpoint",
        }, "execution replay receipt")
        _require(replay["schema"] == EXECUTION_REPLAY_SCHEMA
                 and replay["study"] == "sage-mem-v1"
                 and replay["status"] == "sealed-label-free"
                 and replay["cohort"] == cohort
                 and replay["bank_manifest_sha256"] == canonical.bank_sha256
                 and replay["formal_labels_read"] is False
                 and replay["development_outcomes_read"] is False
                 and replay["controller_identity_sha256"] ==
                 controller["controller_identity_sha256"]
                 and replay["rows"] == contract.formal_rows[cohort]
                 and replay["classes"] == contract.classes[cohort]
                 and isinstance(replay["native_clusters"], int)
                 and not isinstance(replay["native_clusters"], bool)
                 and replay["native_clusters"] > 0
                 and isinstance(replay["executions"], int)
                 and not isinstance(replay["executions"], bool)
                 and replay["executions"] > 0
                 and replay["replayed_executions"] == replay["executions"]
                 and replay["deterministic_replay_fidelity"] == 1.0
                 and isinstance(replay["execution_endpoint"], str)
                 and replay["execution_endpoint"],
                 f"{cohort} execution replay differs")
        expected = {
            "formal_test_episode_id", "formal_test_native_cluster_id",
            "selected_class_by_true_target_success",
            "deterministic_random_class",
        }
        try:
            with _open_npz(artifact) as z:
                _require(set(z.files) == expected,
                         "execution deck array schema differs")
                arrays = {name: np.asarray(z[name]).copy()
                          for name in expected}
        except (OSError, ValueError) as error:
            raise PhaseBReproductionError("cannot open execution deck") \
                from error
        count = contract.formal_rows[cohort]
        classes = contract.classes[cohort]
        episode = arrays["formal_test_episode_id"]
        cluster = arrays["formal_test_native_cluster_id"]
        cube = arrays["selected_class_by_true_target_success"]
        random_class = arrays["deterministic_random_class"]
        _require(episode.shape == cluster.shape == random_class.shape ==
                 (count,) and cube.shape == (count, classes, classes)
                 and all(np.issubdtype(value.dtype, np.integer)
                         or np.issubdtype(value.dtype, np.bool_)
                         for value in (episode, cluster, cube, random_class))
                 and np.isin(cube, (0, 1)).all()
                 and np.all((random_class >= 0) & (random_class < classes))
                 and np.array_equal(
                     episode, labels[cohort].formal_test_episode_id)
                 and np.array_equal(
                     cluster, labels[cohort].formal_test_native_cluster_id),
                 f"{cohort} execution arrays are malformed or unaligned")
        _validate_clusters(episode, cluster, contract.variants[cohort],
                           f"{cohort}/execution-deck")
        if contract.variants[cohort] > 1:
            for native in np.unique(cluster):
                selected = np.flatnonzero(cluster == native)
                _require(all(np.array_equal(cube[selected[0]], cube[index])
                             for index in selected[1:])
                         and np.all(random_class[selected] ==
                                    random_class[selected[0]]),
                         f"execution cube/random policy varies within "
                         f"cluster: {cohort}/{int(native)}")
        truth = labels[cohort].formal_test_label
        row = np.arange(count, dtype=np.int64)
        class_success = cube[
            row[:, None], np.arange(classes)[None, :], truth[:, None]]
        oracle = cube[row, truth, truth]
        random_success = cube[row, random_class, truth]
        decks[cohort] = ExecutionDeck(
            cohort=cohort, threshold=float(gate["threshold"]),
            controller_sha256=controller["controller_identity_sha256"],
            class_success=class_success.astype(np.uint8, copy=False),
            oracle_success=oracle.astype(np.uint8, copy=False),
            random_success=random_success.astype(np.uint8, copy=False))
        inventory.append({
            "cohort": cohort, "status": "supplied",
            "artifact_sha256": record["artifact"]["sha256"],
            "replay_sha256": record["replay_receipt"]["sha256"],
            "class_success_sha256": _array_digest(class_success),
            "oracle_sha256": _array_digest(oracle),
            "random_sha256": _array_digest(random_success),
        })
        statuses[cohort] = "supplied"
    return decks, _sha256_json(inventory), statuses


def _validate_reveal_receipt(finalized_root: Path, summary: Mapping[str, Any],
                             phase_grid: str, label_path: Path,
                             label_hash: str, raw_digest: str,
                             execution_path: Path, execution_hash: str,
                             cells: Mapping[tuple[str, str, int], PhaseCell],
    contract: ReproductionContract) -> str:
    path = finalized_root / "label_reveal_receipt.json"
    _require(path.is_file() and not path.is_symlink(),
             "label-reveal receipt is absent or unsafe")
    receipt = _read_json(path, "label-reveal receipt")
    _exact_keys(receipt, {
        "schema", "study", "stage", "status",
        "complete_grid_validated_before_label_reveal",
        "formal_test_labels_read_before_receipt", "development_outcomes_read",
        "phase_a_cells", "phase_a_grid_sha256", "bank_manifest_sha256",
        "raw_context_reference", "execution_deck_registry",
        "label_registry", "recorded_unix_ns",
    }, "label-reveal receipt")
    bank = {cohort: cells[(cohort, contract.arms[0],
                           contract.seeds[0])].bank_sha256
            for cohort in contract.cohorts}
    _require(receipt["schema"] == FINALIZER_SCHEMA
             and receipt["study"] == "sage-mem-v1"
             and receipt["stage"] == "label-reveal"
             and receipt["status"] ==
             "authorized-after-complete-phase-a-grid"
             and receipt[
                 "complete_grid_validated_before_label_reveal"] is True
             and receipt["formal_test_labels_read_before_receipt"] is False
             and receipt["development_outcomes_read"] is False
             and receipt["phase_a_cells"] == contract.total_cells
             and receipt["phase_a_grid_sha256"] == phase_grid
             and receipt["bank_manifest_sha256"] == bank,
             "label-reveal receipt is stale or incomplete")
    raw = receipt["raw_context_reference"]
    _require(raw == {
        "status": "validated", "sha256": raw_digest,
        "separate_from_parameter_matched_arms": True,
        "short_context_frames": 3, "long_context_frames": 16,
    }, "label-reveal raw-context binding differs")
    registry = receipt["label_registry"]
    deck = receipt["execution_deck_registry"]
    _require(registry == {
        "path": str(label_path.resolve()), "sha256": label_hash,
        "size": _stable_size(label_path),
    } and deck == {
        "status": "sealed-supplied", "path": str(execution_path.resolve()),
        "sha256": execution_hash, "size": _stable_size(execution_path),
    }, "label/execution reveal binding differs")
    digest = _sha256_file(path)
    _require(summary["label_reveal_receipt_sha256"] == digest,
             "finalizer summary reveal-receipt hash differs")
    return digest


def _validate_normalized_labels(
        finalized_root: Path, source_registry_hash: str, reveal_hash: str,
        labels: Mapping[str, LabelSet],
        cells: Mapping[tuple[str, str, int], PhaseCell],
    contract: ReproductionContract) -> str:
    root = finalized_root / "normalized_label_registry"
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file() and not manifest_path.is_symlink(),
             "normalized label manifest is absent or unsafe")
    manifest = _read_json(manifest_path, "normalized label registry")
    _exact_keys(manifest, {
        "schema", "study", "status", "source_custody_registry_sha256",
        "label_reveal_receipt_sha256", "development_outcomes_read",
        "cohorts",
    }, "normalized label registry")
    _require(manifest["schema"] ==
             "sage_mem_v1_post_reveal_consolidated_registry_v1"
             and manifest["study"] == "sage-mem-v1"
             and manifest["status"] ==
             "normalized-after-complete-grid-reveal"
             and manifest["source_custody_registry_sha256"] ==
             source_registry_hash
             and manifest["label_reveal_receipt_sha256"] == reveal_hash
             and manifest["development_outcomes_read"] is False
             and isinstance(manifest["cohorts"], dict)
             and set(manifest["cohorts"]) == set(contract.cohorts),
             "normalized label registry provenance differs")
    inventory: list[dict[str, Any]] = []
    expected_files = {"manifest.json"}
    for cohort in contract.cohorts:
        record = manifest["cohorts"][cohort]
        _exact_keys(record, {"bank_manifest_sha256", "classes", "artifact"},
                    f"{cohort} normalized label record")
        _require(record["bank_manifest_sha256"] ==
                 cells[(cohort, contract.arms[0],
                        contract.seeds[0])].bank_sha256
                 and record["classes"] == contract.classes[cohort],
                 "normalized label cohort identity differs")
        artifact = _safe_artifact(root, record["artifact"],
                                  f"{cohort} normalized labels")
        expected_files.add(artifact.name)
        expected_names = set(LabelSet.__dataclass_fields__)
        try:
            with _open_npz(artifact) as z:
                _require(set(z.files) == expected_names,
                         "normalized label array schema differs")
                for name in expected_names:
                    _require(np.array_equal(z[name],
                                            getattr(labels[cohort], name)),
                             f"normalized label array differs: "
                             f"{cohort}/{name}")
        except (OSError, ValueError) as error:
            raise PhaseBReproductionError(
                "cannot open normalized label artifact") from error
        inventory.append({"cohort": cohort,
                          "artifact_sha256": record["artifact"]["sha256"]})
    _require({item.name for item in _iterdir(root)} == expected_files,
             "normalized label file inventory differs")
    return _sha256_json({
        "manifest_sha256": _sha256_file(manifest_path),
        "artifacts": inventory,
    })


def _validate_execution_receipts(
        finalized_root: Path, summary: Mapping[str, Any],
        decks: Mapping[str, ExecutionDeck],
        predictions: Mapping[tuple[str, str, int], Mapping[str, np.ndarray]],
        contract: ReproductionContract) -> str:
    root = finalized_root / "execution"
    _require(root.is_dir() and not root.is_symlink()
             and {item.name for item in _iterdir(root)} == set(decks),
             "execution receipt cohort inventory differs")
    observed_status: dict[str, dict[str, Any]] = {}
    semantic: list[dict[str, Any]] = []
    for cohort, deck in decks.items():
        directory = root / cohort
        _require(directory.is_dir() and not directory.is_symlink()
                 and {item.name for item in _iterdir(directory)} ==
                 {"receipt.json"},
                 f"execution receipt file inventory differs: {cohort}")
        path = directory / "receipt.json"
        receipt = _read_json(path, "execution receipt")
        common = {
            "schema", "study", "stage", "status", "cohort",
            "controller_identity_sha256", "controller_pinned",
            "arm_identity_used", "input", "oracle_success",
            "random_success", "eligibility_metric", "eligibility_operator",
            "eligibility_threshold", "eligible", "skip_reason",
            "computed_cells",
        }
        _exact_keys(receipt, common | ({
            "ages", "per_arm_seed_age_values_sha256"}
            if deck.eligible else set()), f"{cohort} execution receipt")
        oracle_rate = float(np.mean(deck.oracle_success))
        random_rate = float(np.mean(deck.random_success))
        _require(receipt["schema"] == FINALIZER_SCHEMA
                 and receipt["study"] == "sage-mem-v1"
                 and receipt["stage"] == "external-execution"
                 and receipt["cohort"] == cohort
                 and receipt["controller_identity_sha256"] ==
                 deck.controller_sha256
                 and receipt["controller_pinned"] is True
                 and receipt["arm_identity_used"] is False
                 and receipt["input"] == "predicted_class_only"
                 and receipt["oracle_success"] == oracle_rate
                 and receipt["random_success"] == random_rate
                 and receipt["eligibility_metric"] == "mean_oracle_success"
                 and receipt["eligibility_operator"] == ">="
                 and receipt["eligibility_threshold"] == deck.threshold,
                 f"execution receipt provenance differs: {cohort}")
        if deck.eligible:
            per_age: dict[str, dict[str, list[float]]] = {}
            for arm in contract.arms:
                for seed in contract.seeds:
                    key = (cohort, arm, seed)
                    for stream in ("full", "reset", "prior"):
                        predicted = predictions[key][stream]
                        row = np.arange(contract.formal_rows[cohort])[None, :]
                        executed = deck.class_success[row, predicted]
                        per_age.setdefault(arm, {}).setdefault(
                            stream, []).extend(
                                map(float, np.mean(executed, axis=1)))
            _require(receipt["status"] ==
                     "computed-class-conditioned-arm-blind"
                     and receipt["eligible"] is True
                     and receipt["skip_reason"] is None
                     and receipt["computed_cells"] ==
                     len(contract.arms) * len(contract.seeds)
                     and receipt["ages"] == list(contract.ages)
                     and receipt["per_arm_seed_age_values_sha256"] ==
                     _sha256_json(per_age),
                     f"eligible execution receipt differs: {cohort}")
        else:
            _require(receipt["status"] == "skipped-oracle-gate"
                     and receipt["eligible"] is False
                     and receipt["skip_reason"] ==
                     "oracle-success-below-preregistered-threshold"
                     and receipt["computed_cells"] == 0,
                     f"skipped execution receipt differs: {cohort}")
        receipt_hash = _sha256_file(path)
        observed_status[cohort] = {
            "status": receipt["status"], "eligible": deck.eligible,
            "receipt_sha256": receipt_hash,
            "oracle_success": oracle_rate, "random_success": random_rate,
        }
        semantic.append({
            "cohort": cohort, "eligible": deck.eligible,
            "receipt_sha256": receipt_hash,
            "oracle_array_sha256": _array_digest(deck.oracle_success),
            "random_array_sha256": _array_digest(deck.random_success),
        })
    execution = summary["execution_decks"]
    _exact_keys(execution, {
        "status", "supplied_cohorts", "eligible_cohorts",
        "program_requires_at_least_two_eligible_cohorts",
        "program_gate_passed", "cohort_status",
    }, "finalizer execution summary")
    required = execution["program_requires_at_least_two_eligible_cohorts"]
    eligible_count = sum(deck.eligible for deck in decks.values())
    _require(execution["status"] == "evaluated"
             and execution["supplied_cohorts"] == sorted(decks)
             and execution["eligible_cohorts"] == eligible_count
             and isinstance(required, bool)
             and execution["program_gate_passed"] ==
             (eligible_count >= 2 if required else None)
             and (not required or eligible_count >= 2)
             and execution["cohort_status"] == observed_status,
             "finalizer execution summary differs")
    return _sha256_json(semantic)


def _validate_finalized_cells(
        root: Path, summary: Mapping[str, Any], phase_grid: str,
        label_hash: str, reveal_hash: str,
        cells: Mapping[tuple[str, str, int], PhaseCell],
        labels: Mapping[str, LabelSet],
        predictions: Mapping[tuple[str, str, int], Mapping[str, np.ndarray]],
        consumer_hashes: Mapping[tuple[str, int], str],
        decks: Mapping[str, ExecutionDeck],
        contract: ReproductionContract) -> tuple[str, str, str]:
    cells_root = root / "cells"
    _require(cells_root.is_dir() and not cells_root.is_symlink()
             and {item.name for item in _iterdir(cells_root)} ==
             set(contract.cohorts)
             and all(item.is_dir() and not item.is_symlink()
                     for item in _iterdir(cells_root)),
             "finalized cohort inventory differs")
    records: list[dict[str, Any]] = []
    prediction_semantic: list[dict[str, Any]] = []
    correctness_semantic: list[dict[str, Any]] = []
    execution_semantic: list[dict[str, Any]] = []
    execution_receipt_hashes: dict[str, str] = {}
    for cohort, deck in decks.items():
        receipt_path = root / "execution" / cohort / "receipt.json"
        _require(receipt_path.is_file() and not receipt_path.is_symlink(),
                 f"eligible execution receipt absent: {cohort}")
        execution_receipt_hashes[cohort] = _sha256_file(receipt_path)
    supplied_execution = set(decks)
    if supplied_execution:
        execution_root = root / "execution"
        _require(execution_root.is_dir() and not execution_root.is_symlink()
                 and {item.name for item in _iterdir(execution_root)} ==
                 supplied_execution
                 and all(item.is_dir() and not item.is_symlink()
                         and {child.name for child in _iterdir(item)} ==
                         {"receipt.json"} for item in _iterdir(execution_root)),
                 "finalized execution-receipt inventory differs")
    for cohort in contract.cohorts:
        cohort_root = cells_root / cohort
        _require({item.name for item in _iterdir(cohort_root)} ==
                 set(contract.arms)
                 and all(item.is_dir() and not item.is_symlink()
                         for item in _iterdir(cohort_root)),
                 f"finalized arm inventory differs: {cohort}")
        label_by_age = np.repeat(
            labels[cohort].formal_test_label[None, :],
            len(contract.ages), axis=0)
        for arm in contract.arms:
            arm_root = cohort_root / arm
            _require({item.name for item in _iterdir(arm_root)} ==
                     {f"seed-{seed}" for seed in contract.seeds}
                     and all(item.is_dir() and not item.is_symlink()
                             for item in _iterdir(arm_root)),
                     f"finalized seed inventory differs: {cohort}/{arm}")
            for seed in contract.seeds:
                key = (cohort, arm, seed)
                phase_cell = cells[key]
                directory = arm_root / f"seed-{seed}"
                manifest_path = directory / "manifest.json"
                _require(manifest_path.is_file()
                         and not manifest_path.is_symlink(),
                         "finalized manifest is absent or unsafe")
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
                _require(manifest["schema"] == FINALIZER_SCHEMA
                         and manifest["study"] == "sage-mem-v1"
                         and manifest["stage"] == "formal-finalized"
                         and manifest["status"] == "complete"
                         and manifest["cohort"] == cohort
                         and manifest["arm"] == arm
                         and manifest["seed"] == seed
                         and manifest["ages"] == list(contract.ages)
                         and manifest["phase_a_grid_sha256"] == phase_grid
                         and manifest["phase_a_manifest_sha256"] ==
                         phase_cell.manifest_sha256
                         and manifest["bank_manifest_sha256"] ==
                         phase_cell.bank_sha256
                         and manifest["label_registry_sha256"] == label_hash
                         and manifest["label_reveal_receipt_sha256"] ==
                         reveal_hash
                         and manifest[
                             "shared_arm_blind_consumer_sha256"] ==
                         consumer_hashes[(cohort, seed)]
                         and manifest["native_cluster_id_preserved"] is True
                         and manifest[
                             "counterfactual_variants_per_native_cluster"] ==
                         contract.variants[cohort],
                         "finalized manifest provenance differs")
                artifact = _safe_artifact(directory, manifest["artifact"],
                                          "finalized result")
                _require({item.name for item in _iterdir(directory)} ==
                         {"manifest.json", artifact.name},
                         "finalized cell inventory differs")
                deck = decks.get(cohort)
                execution_expected = deck is not None and deck.eligible
                try:
                    with _open_npz(artifact) as z:
                        _require(set(z.files) == FINAL_KEYS |
                                 (FINAL_EXECUTION_KEYS
                                  if execution_expected else set()),
                                 "finalized array schema differs")
                        arrays = {name: np.asarray(z[name]).copy()
                                  for name in z.files}
                except (OSError, ValueError) as error:
                    raise PhaseBReproductionError(
                        "cannot open finalized result") from error
                for name in ("formal_test_episode_id",
                             "formal_test_native_cluster_id",
                             "formal_test_evidence_age"):
                    _require(np.array_equal(arrays[name],
                                            phase_cell.arrays[name]),
                             f"finalized identity differs: {key}/{name}")
                _require(np.array_equal(arrays["formal_test_label"],
                                        label_by_age),
                         f"finalized labels differ: {key}")
                pred_row: dict[str, Any] = {
                    "cohort": cohort, "arm": arm, "seed": seed}
                correct_row: dict[str, Any] = dict(pred_row)
                execution_row: dict[str, Any] = dict(pred_row)
                for stream in ("full", "reset", "prior"):
                    predicted = predictions[key][stream]
                    observed = arrays[f"formal_test_{stream}_pred"]
                    expected_correct = (predicted == label_by_age).astype(
                        np.uint8)
                    _require(np.array_equal(observed, predicted),
                             f"independent prediction mismatch: {key}/{stream}")
                    _require(np.array_equal(
                        arrays[f"formal_test_{stream}_correct"],
                        expected_correct),
                        f"independent correctness mismatch: {key}/{stream}")
                    _require(np.array_equal(
                        arrays[f"formal_test_{stream}_mse"],
                        phase_cell.arrays[f"formal_test_{stream}_mse"]),
                        f"finalized MSE differs from Phase A: {key}/{stream}")
                    pred_row[stream] = _array_digest(predicted)
                    correct_row[stream] = _array_digest(expected_correct)
                    if execution_expected:
                        row = np.arange(contract.formal_rows[cohort])[None, :]
                        expected_execution = deck.class_success[row, predicted]
                        _require(np.array_equal(
                            arrays[f"formal_test_{stream}_execution_success"],
                            expected_execution),
                            f"execution reproduction mismatch: {key}/{stream}")
                        execution_row[stream] = _array_digest(
                            expected_execution)
                execution = manifest["execution"]
                _exact_keys(execution, {
                    "status", "eligible", "receipt_sha256",
                    "arm_identity_used", "ages", "per_age_success",
                }, "finalized execution link")
                if execution_expected:
                    _require(execution["status"] ==
                             "computed-class-conditioned-arm-blind"
                             and execution["eligible"] is True
                             and execution["receipt_sha256"] ==
                             execution_receipt_hashes[cohort]
                             and execution["arm_identity_used"] is False
                             and execution["ages"] == list(contract.ages)
                             and execution["per_age_success"] == {
                                 stream: list(map(float, np.mean(
                                     arrays[
                                         f"formal_test_{stream}_execution_success"],
                                     axis=1)))
                                 for stream in ("full", "reset", "prior")},
                             "eligible execution manifest differs")
                    execution_semantic.append(execution_row)
                elif deck is not None:
                    _require(execution["status"] == "skipped-oracle-gate"
                             and execution["eligible"] is False
                             and execution["receipt_sha256"] ==
                             execution_receipt_hashes[cohort]
                             and execution["arm_identity_used"] is False
                             and execution["ages"] == list(contract.ages)
                             and execution["per_age_success"] is None,
                             "skipped execution manifest differs")
                else:
                    _require(execution == {
                        "status": "not-supplied", "eligible": None,
                        "receipt_sha256": None, "arm_identity_used": False,
                        "ages": list(contract.ages),
                        "per_age_success": None,
                    }, "unavailable execution manifest differs")
                prediction_semantic.append(pred_row)
                correctness_semantic.append(correct_row)
                records.append({
                    "cohort": cohort, "arm": arm, "seed": seed,
                    "artifact_sha256": manifest["artifact"]["sha256"],
                    "consumer_sha256": consumer_hashes[(cohort, seed)],
                })
    records_digest = _sha256_json(records)
    _require(len(records) == contract.total_cells
             and records_digest == summary["finalized_cells_sha256"],
             "finalized artifact inventory digest differs")
    return (records_digest, _sha256_json(prediction_semantic),
            _sha256_json({
                "correctness": correctness_semantic,
                "execution": execution_semantic,
            }))


def _reproduce_raw(
        references: Mapping[tuple[str, int], RawReference],
        labels: Mapping[str, LabelSet], root: Path,
        phase_grid: str, contract: ReproductionContract) -> tuple[str, str]:
    output_root = root / "raw_context"
    consumer_root = root / "raw_context_consumers"
    _require(output_root.is_dir() and not output_root.is_symlink()
             and consumer_root.is_dir() and not consumer_root.is_symlink()
             and {item.name for item in _iterdir(output_root)} ==
             set(contract.cohorts)
             and {item.name for item in _iterdir(consumer_root)} ==
             set(contract.cohorts)
             and all(item.is_dir() and not item.is_symlink()
                     for item in _iterdir(output_root))
             and all(item.is_dir() and not item.is_symlink()
                     for item in _iterdir(consumer_root)),
             "finalized raw-context inventory differs")
    records: list[dict[str, Any]] = []
    semantic: list[dict[str, Any]] = []
    for cohort in contract.cohorts:
        _require({item.name for item in _iterdir(output_root / cohort)} ==
                 {f"seed-{seed}" for seed in contract.seeds}
                 and all(item.is_dir() and not item.is_symlink()
                         for item in _iterdir(output_root / cohort))
                 and {item.name for item in
                      _iterdir(consumer_root / cohort)} ==
                 {f"seed-{seed}.json" for seed in contract.seeds}
                 and all(item.is_file() and not item.is_symlink()
                         for item in _iterdir(consumer_root / cohort)),
                 f"finalized raw-context seed inventory differs: {cohort}")
        for seed in contract.seeds:
            reference = references[(cohort, seed)]
            try:
                with _open_npz(reference.measurement) as z:
                    consumer_short = np.asarray(
                        z["consumer_train_short_features"], dtype=np.float32)
                    consumer_long = np.asarray(
                        z["consumer_train_long_features"], dtype=np.float32)
                    test_short = np.asarray(
                        z["formal_test_short_features"], dtype=np.float32)
                    test_long = np.asarray(
                        z["formal_test_long_features"], dtype=np.float32)
            except (OSError, ValueError, KeyError) as error:
                raise PhaseBReproductionError(
                    "cannot stream raw-context features") from error
            dimension = reference.feature_dimension
            train_x = np.concatenate([
                consumer_short.reshape(-1, dimension),
                consumer_long.reshape(-1, dimension)], axis=0)
            repeated = np.tile(labels[cohort].consumer_train_label,
                               len(contract.ages))
            train_y = np.concatenate([repeated, repeated])
            values, model_hash = _fit_ridge(
                train_x, train_y,
                (test_short.reshape(-1, dimension),
                 test_long.reshape(-1, dimension)),
                contract.classes[cohort])
            predicted = {
                "short": values[0].reshape(
                    len(contract.ages), contract.formal_rows[cohort]),
                "long": values[1].reshape(
                    len(contract.ages), contract.formal_rows[cohort]),
            }
            shared_hash = _sha256_json({
                "cohort": cohort, "seed": seed,
                "model_sha256": model_hash,
                "contexts": ["short-3", "long-16"],
                "ages_pooled": list(contract.ages),
                "fit_split": "consumer_train",
                "arm_identity_used": False,
                "context_identity_used": False,
                "formal_test_labels_used": False,
            })
            consumer = _read_json(
                consumer_root / cohort / f"seed-{seed}.json",
                "raw-context consumer receipt")
            _exact_keys(consumer, {
                "schema", "study", "stage", "status", "cohort", "seed",
                "contexts", "ages_pooled", "training_rows",
                "feature_dimension", "arm_identity_used",
                "context_identity_used", "formal_test_labels_used",
                "model_sha256", "shared_consumer_sha256",
            }, "raw-context consumer receipt")
            _require(consumer["schema"] == FINALIZER_SCHEMA
                     and consumer["study"] == "sage-mem-v1"
                     and consumer["stage"] ==
                     "raw-context-shared-consumer"
                     and consumer["status"] ==
                     "fit-after-complete-grid-reveal"
                     and consumer["cohort"] == cohort
                     and consumer["seed"] == seed
                     and consumer["contexts"] == ["short-3", "long-16"]
                     and consumer["ages_pooled"] == list(contract.ages)
                     and consumer["training_rows"] == len(train_y)
                     and consumer["feature_dimension"] == dimension
                     and consumer["arm_identity_used"] is False
                     and consumer["context_identity_used"] is False
                     and consumer["formal_test_labels_used"] is False
                     and consumer["model_sha256"] == model_hash
                     and consumer["shared_consumer_sha256"] == shared_hash,
                     "independent raw-context Ridge model differs")
            directory = output_root / cohort / f"seed-{seed}"
            manifest_path = directory / "manifest.json"
            _require(manifest_path.is_file()
                     and not manifest_path.is_symlink(),
                     "finalized raw-context manifest is absent or unsafe")
            manifest = _read_json(manifest_path,
                                  "finalized raw-context manifest")
            _exact_keys(manifest, {
                "schema", "study", "stage", "status", "cohort", "seed",
                "ages", "short_context_frames", "long_context_frames",
                "separate_from_parameter_matched_arms",
                "phase_a_grid_sha256", "source_manifest_sha256",
                "shared_arm_blind_consumer_sha256", "artifact",
            }, "finalized raw-context manifest")
            _require(manifest["schema"] == FINALIZER_SCHEMA
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
                     and manifest["phase_a_grid_sha256"] == phase_grid
                     and manifest["source_manifest_sha256"] ==
                     reference.manifest_sha256
                     and manifest[
                         "shared_arm_blind_consumer_sha256"] == shared_hash,
                     "finalized raw-context manifest differs")
            artifact = _safe_artifact(directory, manifest["artifact"],
                                      "finalized raw-context result")
            _require({item.name for item in _iterdir(directory)} ==
                     {"manifest.json", artifact.name},
                     "finalized raw-context cell inventory differs")
            try:
                with _open_npz(artifact) as z:
                    _require(set(z.files) == FINAL_RAW_KEYS,
                             "finalized raw-context array schema differs")
                    arrays = {name: np.asarray(z[name]).copy()
                              for name in z.files}
            except (OSError, ValueError) as error:
                raise PhaseBReproductionError(
                    "cannot open finalized raw-context result") from error
            label_by_age = np.repeat(
                labels[cohort].formal_test_label[None, :],
                len(contract.ages), axis=0)
            for name in ("formal_test_episode_id",
                         "formal_test_native_cluster_id",
                         "formal_test_evidence_age"):
                _require(np.array_equal(arrays[name], reference.arrays[name]),
                         f"finalized raw-context identity differs: {name}")
            _require(np.array_equal(arrays["formal_test_label"], label_by_age),
                     "finalized raw-context labels differ")
            row = {"cohort": cohort, "seed": seed,
                   "model_sha256": model_hash}
            for context in ("short", "long"):
                expected_correct = (predicted[context] ==
                                    label_by_age).astype(np.uint8)
                _require(np.array_equal(
                    arrays[f"formal_test_{context}_pred"], predicted[context]),
                    f"raw-context prediction mismatch: {cohort}/{seed}/"
                    f"{context}")
                _require(np.array_equal(
                    arrays[f"formal_test_{context}_correct"],
                    expected_correct),
                    f"raw-context correctness mismatch: {cohort}/{seed}/"
                    f"{context}")
                row[f"{context}_prediction_sha256"] = _array_digest(
                    predicted[context])
                row[f"{context}_correctness_sha256"] = _array_digest(
                    expected_correct)
            semantic.append(row)
            records.append({
                "cohort": cohort, "seed": seed,
                "artifact_sha256": manifest["artifact"]["sha256"],
                "consumer_sha256": shared_hash,
            })
    _require(len(records) == contract.raw_references,
             "raw-context reproduction count differs")
    return _sha256_json(records), _sha256_json(semantic)


def _validate_summary_and_report(
        finalized_root: Path, report_path: Path, expected: ExpectedHashes,
        contract: ReproductionContract) -> tuple[dict[str, Any], str]:
    summary_path = finalized_root / "summary.json"
    _require(summary_path.is_file() and not summary_path.is_symlink(),
             "formal-finalizer summary is absent or unsafe")
    _require(_sha256_file(summary_path) == expected.finalizer_summary,
             "finalizer-summary operator hash differs")
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
    _require(summary["schema"] == FINALIZER_SCHEMA
             and summary["study"] == "sage-mem-v1"
             and summary["stage"] == "formal-finalizer"
             and summary["status"] == "complete"
             and summary["phase_a_cells"] == contract.total_cells
             and summary["phase_a_grid_sha256"] == expected.phase_a_grid
             and summary["label_registry_sha256"] == expected.label_registry
             and summary["development_outcomes_read"] is False
             and summary["per_age_results_preserved"] is True
             and summary["pointmaze_x4_native_clustering_preserved"] ==
             (contract.variants.get("dinowm_pointmaze_goal") == 4)
             and summary["finalized_cells"] == contract.total_cells
             and summary["finalized_cells_sha256"] == expected.finalized_cells
             and isinstance(summary["raw_context_reference"], dict)
             and summary["raw_context_reference"].get("status") == "complete"
             and summary["raw_context_reference"].get("references") ==
             contract.raw_references
             and isinstance(summary["execution_decks"], dict)
             and summary["execution_decks"].get("status") == "evaluated",
             "formal finalization is incomplete or stale")
    _require(summary["raw_context_reference"] == {
        "status": "complete", "short_context_frames": 3,
        "long_context_frames": 16,
        "separate_from_parameter_matched_arms": True,
        "references": contract.raw_references,
        "records_sha256": summary["raw_context_reference"].get(
            "records_sha256"),
    } and _is_sha256(summary["raw_context_reference"].get(
        "records_sha256")), "formal raw-context summary differs")
    _require(_sha256_file(report_path) == expected.formal_report,
             "formal-report operator hash differs")
    report = _read_json(report_path, "formal evidence report")
    required = {
        "schema", "study", "stage", "status", "phase_a_cells_verified",
        "finalized_cells_verified", "phase_a_grid_sha256",
        "raw_context_references_verified",
    }
    _require(required.issubset(report), "formal evidence report is malformed")
    _require(report["schema"] == FORMAL_REPORT_SCHEMA
             and report["study"] == "sage-mem-v1"
             and report["stage"] == "formal-evidence-audit"
             and report["status"] == "complete"
             and report["phase_a_cells_verified"] == contract.total_cells
             and report["finalized_cells_verified"] == contract.total_cells
             and report["phase_a_grid_sha256"] == expected.phase_a_grid
             and report["raw_context_references_verified"] ==
             contract.raw_references,
             "formal evidence report is incomplete or stale")
    return summary, _sha256_file(report_path)


_REPORT_REPLAY_CHILD = r'''
import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import sys

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)

workspace = Path(sys.argv[1])
records = json.loads(base64.b64decode(sys.argv[2]).decode("utf-8"))
spec_path = Path(sys.argv[3])
phase_root = Path(sys.argv[4])
finalized_root = Path(sys.argv[5])
prepare_root = Path(sys.argv[6])
raw_root = Path(sys.argv[7])
required = {
    "configs/sage_mem_v1.yaml",
    "scripts/sage_mem_v1_spec.py",
    "scripts/sage_mem_v1_formal_finalizer.py",
    "scripts/audit_sage_mem_v1_formal.py",
}
if set(records) != required:
    raise RuntimeError("child locked replay source registry differs")
for relative, record in records.items():
    candidate = workspace / relative
    info = os.lstat(candidate)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise RuntimeError("child replay source is unsafe")
    payload = candidate.read_bytes()
    if len(payload) != record["size"] or hashlib.sha256(payload).hexdigest() \
            != record["sha256"]:
        raise RuntimeError("child replay source hash differs")
targets = (
    "scripts.sage_mem_v1_spec",
    "scripts.sage_mem_v1_formal_finalizer",
    "scripts.audit_sage_mem_v1_formal",
)
if any(name in sys.modules for name in targets):
    raise RuntimeError("locked replay module was preloaded in isolated child")
sys.path.insert(0, str(workspace))
spec_module = importlib.import_module("scripts.sage_mem_v1_spec")
finalizer_module = importlib.import_module(
    "scripts.sage_mem_v1_formal_finalizer")
auditor_module = importlib.import_module("scripts.audit_sage_mem_v1_formal")
for module, relative in zip((spec_module, finalizer_module, auditor_module),
                            ("scripts/sage_mem_v1_spec.py",
                             "scripts/sage_mem_v1_formal_finalizer.py",
                             "scripts/audit_sage_mem_v1_formal.py")):
    module_path = Path(module.__file__).resolve()
    expected_path = (workspace / relative).resolve()
    if module_path != expected_path:
        raise RuntimeError("child imported a shadow replay module")
    payload = module_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != records[relative]["sha256"]:
        raise RuntimeError("child imported replay module hash differs")
spec = spec_module.load_spec(spec_path, verify_parent_paths=False)
report = auditor_module.audit_formal_evidence(
    spec=spec, phase_a_root=phase_root, finalized_root=finalized_root,
    prepare_root=prepare_root, raw_context_root=raw_root)
sys.stdout.write(canonical(report) + "\n")
'''


def _run_locked_report_subprocess(
        *, workspace: Path, producers: Mapping[str, Mapping[str, Any]],
        spec_path: Path, phase_root: Path, finalized_root: Path,
        prepare_root: Path, raw_root: Path) -> bytes:
    required = (
        "configs/sage_mem_v1.yaml",
        "scripts/sage_mem_v1_spec.py",
        "scripts/sage_mem_v1_formal_finalizer.py",
        "scripts/audit_sage_mem_v1_formal.py",
    )
    records: dict[str, dict[str, Any]] = {}
    for relative in required:
        _require(relative in producers,
                 f"locked replay source is absent: {relative}")
        record = producers[relative]
        path = _safe_existing(relative, workspace,
                              f"locked replay source {relative}", kind="file")
        stable, temporary = _stable_file(path)
        try:
            _require(stable.size == record.get("size")
                     and stable.initial_sha256 == record.get("sha256"),
                     f"locked replay source identity differs: {relative}")
            records[relative] = {
                "size": stable.size, "sha256": stable.initial_sha256}
        finally:
            if temporary:
                stable.verify_unchanged()
                stable.close()
    encoded = base64.b64encode(
        _canonical_json(records).encode("utf-8")).decode("ascii")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONSTARTUP", None)
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _REPORT_REPLAY_CHILD,
             workspace.as_posix(), encoded, spec_path.as_posix(),
             phase_root.as_posix(), finalized_root.as_posix(),
             prepare_root.as_posix(), raw_root.as_posix()],
            cwd=workspace, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False)
    except OSError as error:
        raise PhaseBReproductionError(
            "cannot launch isolated formal-report replay") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise PhaseBReproductionError(
            f"isolated formal-report replay failed: {detail}")
    return completed.stdout


def _replay_formal_report(
        *, report_path: Path, spec_path: Path, phase_root: Path,
        finalized_root: Path, prepare_root: Path, raw_root: Path,
        contract: ReproductionContract,
        locked_producers: Mapping[str, Mapping[str, Any]],
        reproducer: Callable[[], Mapping[str, Any]] | None = None) \
        -> tuple[str, str]:
    """Recompute the locked report and require canonical byte equality.

    ``recorded_unix_ns`` belongs only to the label-reveal custody receipt and
    is not a field of the registered evidence-report schema.  Consequently no
    timestamp value is rewritten: the locked auditor deterministically omits
    it, and both original and replay are serialized by the same specified
    canonical JSON encoding plus one LF byte.

    The callable hook exists only for compact non-production fixtures.  The
    exact 600-cell contract always imports and executes the locked auditor.
    """
    if reproducer is not None:
        _require(not contract.is_registered_production,
                 "production report replay cannot be injected")
        replay = reproducer()
        _require(isinstance(replay, Mapping),
                 "formal-report replay is not a JSON mapping")
        replay_bytes = (_canonical_json(dict(replay)) + "\n").encode(
            "utf-8")
    else:
        replay_bytes = _run_locked_report_subprocess(
            workspace=spec_path.parents[1], producers=locked_producers,
            spec_path=spec_path, phase_root=phase_root,
            finalized_root=finalized_root, prepare_root=prepare_root,
            raw_root=raw_root)
        replay = _read_json(report_path, "original formal report")
    def contains_key(value: Any, forbidden: str) -> bool:
        if isinstance(value, Mapping):
            return forbidden in value or any(
                contains_key(child, forbidden) for child in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_key(child, forbidden) for child in value)
        return False
    _require(not contains_key(replay, "recorded_unix_ns")
             and not contains_key(replay, "label_reveal_receipt_sha256"),
             "formal-report replay improperly depends on reveal timestamp "
             "or receipt identity")
    stable, temporary = _stable_file(report_path)
    try:
        original_bytes = stable.read_bytes()
    finally:
        if temporary:
            stable.verify_unchanged()
            stable.close()
    _require(original_bytes == replay_bytes,
             "formal-report replay is not byte-for-byte identical")
    return _sha256_bytes(original_bytes), _sha256_bytes(replay_bytes)


def _validate_top_inventory(root: Path, contract: ReproductionContract) -> None:
    expected = {
        "cells", "consumers", "execution", "label_reveal_receipt.json",
        "normalized_label_registry", "raw_context",
        "raw_context_consumers", "summary.json",
    }
    observed = {item.name for item in _iterdir(root)}
    _require(observed == expected,
             f"finalized top-level inventory differs: {sorted(observed)}")
    for name in expected - {"label_reveal_receipt.json", "summary.json"}:
        path = root / name
        _require(path.is_dir() and not path.is_symlink(),
                 f"finalized directory is absent or unsafe: {name}")
    normalized = root / "normalized_label_registry"
    _require({item.name for item in _iterdir(normalized)} ==
             {"manifest.json", *(f"{cohort}.npz"
                                  for cohort in contract.cohorts)},
             "normalized-label inventory differs")


def _audit_phase_b_reproduction_session(
        *, workspace: str | Path, paths: InputPaths,
        expected: ExpectedHashes, output: str | Path,
        contract: ReproductionContract = PRODUCTION_CONTRACT,
        _report_reproducer: Callable[[], Mapping[str, Any]] | None = None,
        _verifier_source: str | Path | None = None) \
        -> tuple[dict[str, Any], Path]:
    """Run under one active stable-artifact session."""
    contract.validate()
    expected.validate()
    workspace_path = _safe_workspace(workspace)
    _require(_verifier_source is None
             or not contract.is_registered_production,
             "production verifier source path cannot be injected")
    verifier_path = _safe_existing(
        Path(__file__) if _verifier_source is None else _verifier_source,
        workspace_path, "Phase-B verifier source", kind="file")
    _require(_sha256_file(verifier_path) == expected.verifier_source,
             "Phase-B verifier source differs from operator pin")
    protocol_lock = _safe_existing(paths.protocol_lock, workspace_path,
                                   "protocol lock", kind="file")
    phase_root = _safe_existing(paths.phase_a_root, workspace_path,
                                "Phase-A root", kind="dir")
    raw_root = _safe_existing(paths.raw_context_root, workspace_path,
                              "raw-context root", kind="dir")
    label_registry = _safe_existing(paths.label_registry, workspace_path,
                                    "label registry", kind="file")
    execution_registry = _safe_existing(
        paths.execution_registry, workspace_path,
        "execution registry", kind="file")
    finalized_root = _safe_existing(paths.finalized_root, workspace_path,
                                    "finalized root", kind="dir")
    prepare_root = _safe_existing(paths.prepare_root, workspace_path,
                                  "formal prepare root", kind="dir")
    report_path = _safe_existing(paths.formal_report, workspace_path,
                                 "formal report", kind="file")
    output_path = _safe_output(output, workspace_path)
    _require_output_outside_inputs(output_path, (
        phase_root / "cells", raw_root, finalized_root,
        label_registry.parent, execution_registry.parent, prepare_root,
    ))

    lock, producer_inventory = _authenticate_protocol_lock(
        protocol_lock, expected.protocol_lock, workspace_path, contract)
    summary, report_hash = _validate_summary_and_report(
        finalized_root, report_path, expected, contract)
    cells, phase_grid, phase_inventory = _load_phase_grid(
        phase_root, contract, lock["protocol_fingerprint"],
        expected.phase_a_grid)
    raw_references, raw_source_semantic = _load_raw_references(
        raw_root, expected.raw_context_summary, cells, contract)
    labels, label_semantic = _load_labels(
        label_registry, expected.label_registry, cells, contract)
    decks, execution_source_semantic, execution_statuses = \
        _load_execution_registry(
            execution_registry, expected.execution_registry,
            labels, cells, contract)
    reveal_hash = _validate_reveal_receipt(
        finalized_root, summary, phase_grid, label_registry,
        expected.label_registry, _sha256_json([
            {
                "cohort": cohort, "seed": seed,
                "manifest_sha256": reference.manifest_sha256,
                "bank_manifest_sha256":
                    cells[(cohort, contract.arms[0],
                           contract.seeds[0])].bank_sha256,
                "feature_dimension": reference.feature_dimension,
            }
            for (cohort, seed), reference in raw_references.items()
        ]), execution_registry, expected.execution_registry,
        cells, contract)
    normalized_label_inventory = _validate_normalized_labels(
        finalized_root, expected.label_registry, reveal_hash,
        labels, cells, contract)
    predictions, consumer_hashes, carrier_model_semantic = \
        _reproduce_carrier_predictions(
            cells, labels, finalized_root, contract)
    finalized_digest, carrier_prediction_semantic, carrier_other_semantic = \
        _validate_finalized_cells(
            finalized_root, summary, phase_grid, expected.label_registry,
            reveal_hash, cells, labels, predictions, consumer_hashes,
            decks, contract)
    _require(finalized_digest == expected.finalized_cells,
             "reproduced finalized-cell digest differs from operator pin")
    execution_receipt_semantic = _validate_execution_receipts(
        finalized_root, summary, decks, predictions, contract)
    raw_records_digest, raw_prediction_semantic = _reproduce_raw(
        raw_references, labels, finalized_root, phase_grid, contract)
    _require(summary["raw_context_reference"]["records_sha256"] ==
             raw_records_digest,
             "finalizer raw-context record digest differs")
    _validate_top_inventory(finalized_root, contract)
    original_report_hash, replay_report_hash = _replay_formal_report(
        report_path=report_path,
        spec_path=_safe_existing("configs/sage_mem_v1.yaml", workspace_path,
                                 "registered protocol spec", kind="file"),
        phase_root=phase_root, finalized_root=finalized_root,
        prepare_root=prepare_root, raw_root=raw_root, contract=contract,
        locked_producers=lock["producer_identities"],
        reproducer=_report_reproducer)
    _require(original_report_hash == expected.formal_report
             and replay_report_hash == expected.formal_report,
             "formal-report replay hash differs from operator pin")

    receipt = {
        "schema": SCHEMA,
        "study": "sage-mem-v1",
        "stage": "phase-b-independent-reproduction",
        "status": "complete",
        "production_contract_verified": (
            contract.is_registered_production and contract.require_600),
        "report_reproducer_injected": _report_reproducer is not None,
        "verifier_source_injected": _verifier_source is not None,
        "contract_identity": contract.identity(),
        "contract_identity_sha256": _sha256_json(contract.identity()),
        "registered_contract_sha256": _sha256_json(
            PRODUCTION_CONTRACT.identity()),
        "outcome_values_emitted": False,
        "finalizer_prediction_helpers_called": False,
        "operator_pins": {
            "verifier_source_sha256": expected.verifier_source,
            "protocol_lock_sha256": expected.protocol_lock,
            "phase_a_grid_sha256": expected.phase_a_grid,
            "raw_context_summary_sha256": expected.raw_context_summary,
            "label_registry_sha256": expected.label_registry,
            "execution_registry_sha256": expected.execution_registry,
            "finalizer_summary_sha256": expected.finalizer_summary,
            "finalized_cells_sha256": expected.finalized_cells,
            "formal_report_sha256": expected.formal_report,
        },
        "authenticated_inventories": {
            "verifier_source": {
                "path": _repo_relative(verifier_path, workspace_path),
                "sha256": expected.verifier_source,
                "size": _stable_size(verifier_path),
            },
            "bound_input_files": {
                "protocol_lock": {
                    "path": _repo_relative(protocol_lock, workspace_path),
                    "sha256": expected.protocol_lock,
                    "size": _stable_size(protocol_lock),
                },
                "raw_context_summary": {
                    "path": _repo_relative(
                        raw_root / "summary.json", workspace_path),
                    "sha256": expected.raw_context_summary,
                    "size": _stable_size(raw_root / "summary.json"),
                },
                "label_registry": {
                    "path": _repo_relative(label_registry, workspace_path),
                    "sha256": expected.label_registry,
                    "size": _stable_size(label_registry),
                },
                "execution_registry": {
                    "path": _repo_relative(
                        execution_registry, workspace_path),
                    "sha256": expected.execution_registry,
                    "size": _stable_size(execution_registry),
                },
                "finalizer_summary": {
                    "path": _repo_relative(
                        finalized_root / "summary.json", workspace_path),
                    "sha256": expected.finalizer_summary,
                    "size": _stable_size(finalized_root / "summary.json"),
                },
                "formal_report": {
                    "path": _repo_relative(report_path, workspace_path),
                    "sha256": expected.formal_report,
                    "size": _stable_size(report_path),
                },
            },
            "numerical_environment": _environment_identity(),
            "locked_producers_sha256": producer_inventory,
            "phase_a_artifacts_sha256": phase_inventory,
            "normalized_label_artifacts_sha256":
                normalized_label_inventory,
            "phase_a_cells": contract.total_cells,
            "raw_context_references": contract.raw_references,
            "finalized_cells": contract.total_cells,
            "execution_registry_status_sha256": _sha256_json(
                execution_statuses),
            "formal_report_sha256": report_hash,
            "replayed_formal_report_sha256": replay_report_hash,
        },
        "independent_reproduction": {
            "registered_consumer": {
                "estimator": "sklearn.linear_model.RidgeClassifier",
                "alpha": 1e-3, "solver": "lsqr", "tol": 1e-6,
                "max_iter": 5000,
                "standardization": "StandardScaler(mean=True,std=True)",
                "carrier_models_refit": (
                    len(contract.cohorts) * len(contract.seeds)
                    * len(contract.ages)),
                "raw_context_models_refit": contract.raw_references,
            },
            "carrier_streams_reproduced": ["full", "reset", "prior"],
            "raw_context_streams_reproduced": ["short-3", "long-16"],
            "eligible_execution_arrays_recomputed": True,
            "all_arrays_exact": True,
            "formal_report_byte_exact": True,
            "report_timestamp_normalization": (
                "none; label-reveal recorded_unix_ns is outside the locked "
                "formal-report schema and is therefore deterministically "
                "excluded from both original and replay"),
        },
        "semantic_digests": {
            "revealed_labels_sha256": label_semantic,
            "raw_phase_a_sha256": raw_source_semantic,
            "execution_decks_sha256": execution_source_semantic,
            "execution_receipts_sha256": execution_receipt_semantic,
            "carrier_models_sha256": carrier_model_semantic,
            "carrier_predictions_sha256": carrier_prediction_semantic,
            "carrier_correctness_and_execution_sha256":
                carrier_other_semantic,
            "raw_predictions_and_correctness_sha256":
                raw_prediction_semantic,
        },
        "claim_boundary": (
            "provenance-and-reproduction-only; this receipt contains no "
            "accuracy, effect, interval, gate, or universal-success claim"),
    }
    return receipt, output_path


def audit_phase_b_reproduction(
        *, workspace: str | Path, paths: InputPaths,
        expected: ExpectedHashes, output: str | Path,
        contract: ReproductionContract = PRODUCTION_CONTRACT,
        _report_reproducer: Callable[[], Mapping[str, Any]] | None = None,
        _verifier_source: str | Path | None = None) -> dict[str, Any]:
    """Execute with stable descriptors, reverify, then publish one receipt."""
    session = _ArtifactSession()
    token = _ACTIVE_ARTIFACT_SESSION.set(session)
    try:
        receipt, output_path = _audit_phase_b_reproduction_session(
            workspace=workspace, paths=paths, expected=expected,
            output=output, contract=contract,
            _report_reproducer=_report_reproducer,
            _verifier_source=_verifier_source)
        # No receipt can exist until every descriptor used for provenance or
        # scientific computation still matches its initial bytes and pathname.
        session.verify_all()
        _atomic_receipt(output_path, receipt)
        return receipt
    finally:
        _ACTIVE_ARTIFACT_SESSION.reset(token)
        session.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--protocol-lock", type=Path,
                        default=Path("outputs/sage_mem_v1/protocol_lock.json"))
    parser.add_argument("--phase-a-root", type=Path,
                        default=Path("outputs/sage_mem_v1"))
    parser.add_argument("--raw-context-root", type=Path,
                        default=Path("outputs/sage_mem_v1/raw_context_phase_a"))
    parser.add_argument("--label-registry", type=Path,
                        default=Path("outputs/sage_mem_v1/formal_preparation/"
                                     "custody/registry.json"))
    parser.add_argument("--execution-registry", type=Path,
                        default=Path("outputs/sage_mem_v1/formal_preparation/"
                                     "execution_decks/registry.json"))
    parser.add_argument("--finalized-root", type=Path,
                        default=Path("outputs/sage_mem_v1/formal_finalized"))
    parser.add_argument("--prepare-root", type=Path,
                        default=Path("outputs/sage_mem_v1/formal_preparation"))
    parser.add_argument("--formal-report", type=Path,
                        default=Path("outputs/sage_mem_v1/formal_audit/"
                                     "report.json"))
    parser.add_argument("--output", type=Path)
    for name in (
            "verifier-source", "protocol-lock", "phase-a-grid",
            "raw-context-summary",
            "label-registry", "execution-registry", "finalizer-summary",
            "finalized-cells", "formal-report"):
        parser.add_argument(f"--expected-{name}-sha256")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.execute:
        # Do not resolve, stat, hash, or read any supplied path in preview.
        print(_canonical_json({
            "schema": SCHEMA,
            "preview": True,
            "phase_a_cells_required": 600,
            "raw_context_references_required": 50,
            "operator_hashes_required": 9,
            "filesystem_reads": 0,
            "outcomes_read": False,
            "receipt_written": False,
        }))
        return 0
    _require(args.output is not None, "--execute requires --output")
    expected = ExpectedHashes(
        verifier_source=args.expected_verifier_source_sha256,
        protocol_lock=args.expected_protocol_lock_sha256,
        phase_a_grid=args.expected_phase_a_grid_sha256,
        raw_context_summary=args.expected_raw_context_summary_sha256,
        label_registry=args.expected_label_registry_sha256,
        execution_registry=args.expected_execution_registry_sha256,
        finalizer_summary=args.expected_finalizer_summary_sha256,
        finalized_cells=args.expected_finalized_cells_sha256,
        formal_report=args.expected_formal_report_sha256,
    )
    receipt = audit_phase_b_reproduction(
        workspace=args.workspace,
        paths=InputPaths(
            protocol_lock=args.protocol_lock,
            phase_a_root=args.phase_a_root,
            raw_context_root=args.raw_context_root,
            label_registry=args.label_registry,
            execution_registry=args.execution_registry,
            finalized_root=args.finalized_root,
            prepare_root=args.prepare_root,
            formal_report=args.formal_report),
        expected=expected, output=args.output)
    print(_canonical_json(receipt))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PhaseBReproductionError as error:
        print(f"Phase-B reproduction stopped: {error}", file=os.sys.stderr)
        raise SystemExit(2) from error


__all__ = [
    "ExpectedHashes", "InputPaths", "PRODUCTION_CONTRACT",
    "PhaseBReproductionError", "ReproductionContract",
    "audit_phase_b_reproduction", "main",
]
