#!/usr/bin/env python3
"""Build the label-free SAGE-Mem short-3/long-16 formal reference.

Each prepared immutable bank exposes a true 20-frame cue-inserted frozen
feature sequence.  This producer converts the observed prefix at each evidence
age into one deterministic 16-slot representation:

* short context right-aligns the latest three observed frames and zero-pads 13;
* long context right-aligns up to the latest sixteen observed frames;
* DINO spatial patches are mean-pooled per frame before slotting;
* LeWM uses its frozen frame embedding directly.

Only consumer_train/formal_test identities and frozen features are emitted.
No semantic prediction, MSE, label vault, development outcome, simulator, or
checkpoint is opened.  The post-grid finalizer fits the single shared semantic
consumer only after label reveal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping
import uuid

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sage_mem_v1_formal_finalizer import (  # noqa: E402
    AGES,
    RAW_CONTEXT_FEATURE_CONTRACT,
    RAW_CONTEXT_SCHEMA,
)


DEFAULT_CONFIG = ROOT / "configs/sage_mem_v1.yaml"
DEFAULT_PREPARED_ROOT = ROOT / "outputs/sage_mem_v1/formal_preparation/banks"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/sage_mem_v1/raw_context_phase_a"
COHORTS = (
    "lewm_reacher_color",
    "lewm_pusht_color",
    "dinowm_pusht_token",
    "dinowm_pusht_binding",
    "dinowm_pointmaze_goal",
)
LEWM_COHORTS = COHORTS[:2]
SEEDS = tuple(range(10))
SPLITS = ("consumer_train", "formal_test")
ARRAY_KEYS = {
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
PRODUCER_SCHEMA = "sage_mem_v1_raw_context_producer_v1"


class RawContextReferenceError(RuntimeError):
    """A bank, feature, atomic-output, or resume invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RawContextReferenceError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
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
    return _sha256_file(path)


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


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=lambda token: (_ for _ in ()).throw(
                               RawContextReferenceError(
                                   f"non-finite JSON in {label}: {token}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RawContextReferenceError(
            f"cannot read {label}: {path}") from error
    _require(isinstance(value, dict), f"{label} must be a mapping")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "SAGE-Mem config must be a mapping")
    _require(tuple(value.get("cohorts", {})) == COHORTS,
             "registered cohort order changed")
    reference = value.get("long_context_reference")
    _require(isinstance(reference, dict)
             and reference.get("enabled") is True
             and reference.get("short_context_frames") == 3
             and reference.get("long_context_frames") == 16,
             "registered raw-context frame counts changed")
    _require(value.get("optimization", {}).get("formal_seeds") == list(SEEDS),
             "registered formal seeds changed")
    return value


@dataclass
class PreparedBankView:
    cohort: str
    spatial: bool
    bank_manifest_sha256: str
    host_hash: str
    split_banks: Mapping[str, Any]

    def identities(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        _require(split in SPLITS, f"unknown split: {split}")
        bank = self.split_banks[split]
        if self.spatial:
            identity = bank.identity(split)
            episode = np.asarray(identity["episode_id"], dtype=np.int64)
            cluster = np.asarray(
                identity["native_cluster_id"], dtype=np.int64)
        else:
            episode = np.asarray(bank.episode_ids, dtype=np.int64)
            cluster = episode.copy()
        _require(episode.ndim == cluster.ndim == 1
                 and len(episode) == len(cluster)
                 and len(np.unique(episode)) == len(episode)
                 and np.all(episode >= 0) and np.all(cluster >= 0),
                 f"prepared identity is malformed: {self.cohort}/{split}")
        return episode.copy(), cluster.copy()

    def sequence(self, split: str, age: int) -> np.ndarray:
        bank = self.split_banks[split]
        indices = (bank.indices(split) if self.spatial else bank.fit_indices)
        value = np.asarray(bank.features(int(age), indices), dtype=np.float32)
        _require(value.shape[0] == len(indices),
                 "prepared feature rows differ from split identity")
        return value


def _load_prepared_bank(
        cohort: str, prepared_root: Path,
        split_counts: Mapping[str, int]) -> PreparedBankView:
    bank_root = prepared_root / cohort
    _require(bank_root.is_dir() and not bank_root.is_symlink(),
             f"prepared bank root is missing or unsafe: {cohort}")
    for path in (bank_root, *bank_root.rglob("*")):
        _require(not path.is_symlink()
                 and (path.stat().st_mode & 0o222) == 0,
                 f"prepared bank is not immutable: {path}")
    manifest_path = bank_root / "manifest.json"
    _require(manifest_path.is_file() and not manifest_path.is_symlink(),
             f"prepared bank manifest is missing: {cohort}")
    bank_hash = _sha256_file(manifest_path)
    if cohort in LEWM_COHORTS:
        from scripts.sage_mem_v1_lewm_formal import (
            load_lewm_trajectory_banks,
        )
        handle, banks = load_lewm_trajectory_banks(
            manifest_path, expected_cohort=cohort,
            expected_counts={
                split: int(split_counts[split])
                for split in ("formal_train", *SPLITS)})
        _require(handle.get("formal_labels_hidden") is True
                 and handle.get("labels_accessible_through_handle") is False
                 and handle.get("manifest_sha256") == bank_hash
                 and _is_sha256(handle.get("host_digest")),
                 f"LeWM label-free bank boundary failed: {cohort}")
        return PreparedBankView(
            cohort=cohort, spatial=False,
            bank_manifest_sha256=bank_hash,
            host_hash=str(handle["host_digest"]),
            split_banks={split: banks[split] for split in SPLITS})
    from scripts.sage_mem_v1_dino_formal import DinoLabelFreeFormalBank
    bank = DinoLabelFreeFormalBank(bank_root, verify_artifacts=True)
    handle = bank.provenance_handle()
    _require(handle.get("labels_accessible_through_handle") is False
             and handle.get("semantic_label_vault_inside_bank") is False
             and _is_sha256(handle.get("host_hash_before"))
             and handle.get("host_hash_before") == handle.get(
                 "host_hash_after")
             and handle.get("manifest_sha256") == bank_hash,
             f"DINO label-free bank boundary failed: {cohort}")
    for split in SPLITS:
        _require(len(bank.indices(split)) == int(split_counts[split]) * (
            4 if cohort == "dinowm_pointmaze_goal" else 1),
            f"DINO prepared split count differs: {cohort}/{split}")
    return PreparedBankView(
        cohort=cohort, spatial=True,
        bank_manifest_sha256=bank_hash,
        host_hash=str(handle["host_hash_before"]),
        split_banks={split: bank for split in SPLITS})


def _slot_representation(sequence: np.ndarray, *, spatial: bool,
                         age: int) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-dimensional right-aligned short-3 and long-16 vectors."""

    value = np.asarray(sequence, dtype=np.float32)
    _require(value.ndim == (4 if spatial else 3)
             and value.shape[1] == 20,
             "true 20-frame frozen cue-inserted sequence is unavailable")
    if spatial:
        _require(value.shape[2] > 0 and value.shape[3] > 0,
                 "DINO spatial feature shape is malformed")
        frames = np.mean(value, axis=2, dtype=np.float32)
        endpoint = 3 + int(age)
    else:
        _require(value.shape[2] > 0,
                 "LeWM frame embedding is malformed")
        frames = value
        endpoint = 19
    _require(int(age) in AGES and 3 <= endpoint < 20,
             "evidence-age endpoint is invalid")
    observed = frames[:, :endpoint]
    dimension = frames.shape[2]
    short = np.zeros((len(frames), 16, dimension), dtype=np.float32)
    long = np.zeros_like(short)
    short_values = observed[:, -3:]
    long_values = observed[:, -16:]
    short[:, -len(short_values[0]):] = short_values
    long[:, -len(long_values[0]):] = long_values
    return short.reshape(len(frames), -1), long.reshape(len(frames), -1)


def build_feature_arrays(view: PreparedBankView) -> dict[str, np.ndarray]:
    """Build one label-free, age-major cohort artifact."""

    arrays: dict[str, np.ndarray] = {}
    for split in SPLITS:
        episode, cluster = view.identities(split)
        arrays[f"{split}_episode_id"] = np.repeat(
            episode[None], len(AGES), axis=0)
        arrays[f"{split}_native_cluster_id"] = np.repeat(
            cluster[None], len(AGES), axis=0)
        arrays[f"{split}_evidence_age"] = np.repeat(
            np.asarray(AGES, dtype=np.int64)[:, None], len(episode), axis=1)
        short_parts, long_parts = [], []
        for age in AGES:
            short, long = _slot_representation(
                view.sequence(split, age), spatial=view.spatial, age=age)
            short_parts.append(short)
            long_parts.append(long)
        arrays[f"{split}_short_features"] = np.stack(short_parts)
        arrays[f"{split}_long_features"] = np.stack(long_parts)
    validate_feature_arrays(arrays, view=view)
    return arrays


def validate_feature_arrays(arrays: Mapping[str, np.ndarray], *,
                            view: PreparedBankView) -> None:
    _require(set(arrays) == ARRAY_KEYS,
             "raw-context feature artifact schema changed")
    dimensions: set[int] = set()
    for split in SPLITS:
        episode, cluster = view.identities(split)
        expected_episode = np.repeat(episode[None], len(AGES), axis=0)
        expected_cluster = np.repeat(cluster[None], len(AGES), axis=0)
        expected_age = np.repeat(
            np.asarray(AGES, dtype=np.int64)[:, None], len(episode), axis=1)
        _require(np.array_equal(arrays[f"{split}_episode_id"],
                                expected_episode)
                 and np.array_equal(
                     arrays[f"{split}_native_cluster_id"], expected_cluster)
                 and np.array_equal(
                     arrays[f"{split}_evidence_age"], expected_age),
                 f"raw-context identity differs: {view.cohort}/{split}")
        for context in ("short", "long"):
            value = np.asarray(arrays[f"{split}_{context}_features"])
            _require(value.ndim == 3
                     and value.shape[:2] == (len(AGES), len(episode))
                     and value.shape[2] > 0 and np.isfinite(value).all(),
                     f"raw-context feature is malformed: {split}/{context}")
            dimensions.add(int(value.shape[2]))
    _require(len(dimensions) == 1,
             "short/long feature dimensions are not identical")


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "sha256": _sha256_file(path),
            "size": path.stat().st_size}


def _manifest_value(view: PreparedBankView, seed: int,
                    artifact: Path) -> dict[str, Any]:
    return {
        "schema": RAW_CONTEXT_SCHEMA,
        "study": "sage-mem-v1",
        "stage": "formal-raw-context-reference",
        "status": "complete-label-free",
        "cohort": view.cohort,
        "seed": int(seed),
        "ages": list(AGES),
        "short_context_frames": 3,
        "long_context_frames": 16,
        "separate_from_parameter_matched_arms": True,
        "formal_test_labels_read": False,
        "development_outcomes_read": False,
        "bank_manifest_sha256": view.bank_manifest_sha256,
        "host_hash_before": view.host_hash,
        "host_hash_after": view.host_hash,
        "consumer_contract": "post-reveal-shared-short-long-arm-blind",
        "shared_consumer_sha256": None,
        "feature_contract": dict(RAW_CONTEXT_FEATURE_CONTRACT),
        "artifact": _artifact_record(artifact),
    }


def _validate_cell(directory: Path, *, view: PreparedBankView, seed: int,
                   expected_arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    _require(directory.is_dir() and not directory.is_symlink()
             and manifest_path.is_file() and not manifest_path.is_symlink(),
             f"raw-context cell is incomplete: {directory}")
    manifest = _read_json(manifest_path, "raw-context manifest")
    artifact_record = manifest.get("artifact")
    _require(isinstance(artifact_record, dict)
             and set(artifact_record) == {"path", "sha256", "size"},
             "raw-context artifact handle is malformed")
    artifact = directory / str(artifact_record["path"])
    _require(not Path(str(artifact_record["path"])).is_absolute()
             and ".." not in Path(str(artifact_record["path"])).parts
             and artifact.is_file() and not artifact.is_symlink()
             and artifact.stat().st_size == artifact_record["size"]
             and _sha256_file(artifact) == artifact_record["sha256"],
             "raw-context artifact identity differs")
    _require(manifest == _manifest_value(view, seed, artifact),
             "raw-context manifest differs from prepared bank")
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            _require(set(archive.files) == ARRAY_KEYS,
                     "raw-context artifact arrays changed")
            observed = {name: np.asarray(archive[name])
                        for name in archive.files}
            validate_feature_arrays(observed, view=view)
            for name, expected in expected_arrays.items():
                _require(np.array_equal(observed[name], expected),
                         f"raw-context deterministic feature differs: {name}")
    except (OSError, ValueError) as error:
        raise RawContextReferenceError(
            f"cannot load raw-context artifact: {artifact}") from error
    _require({item.name for item in directory.iterdir()} == {
        "manifest.json", artifact.name},
        "raw-context cell contains unexpected files")
    return manifest


def _write_cell_atomic(directory: Path, *, view: PreparedBankView, seed: int,
                       arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    _require(not directory.exists(), f"refusing to overwrite cell: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / (
        f".{directory.name}.partial-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o750)
    try:
        artifact = staging / "raw_context_features.npz"
        _atomic_npz(artifact, arrays)
        _atomic_json(staging / "manifest.json",
                     _manifest_value(view, seed, artifact))
        os.rename(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return _validate_cell(
        directory, view=view, seed=seed, expected_arrays=arrays)


def preview_plan(*, config_path: str | Path = DEFAULT_CONFIG,
                 prepared_root: str | Path = DEFAULT_PREPARED_ROOT,
                 output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    prepared_root = Path(prepared_root).resolve()
    output_root = Path(output_root).resolve()
    _load_config(config_path)
    banks = {
        cohort: {
            "path": str((prepared_root / cohort / "manifest.json").resolve()),
            "available": (prepared_root / cohort / "manifest.json").is_file(),
        }
        for cohort in COHORTS
    }
    return {
        "schema": PRODUCER_SCHEMA,
        "study": "sage-mem-v1",
        "status": "ready" if all(value["available"]
                                  for value in banks.values()) else
                  "blocked-missing-prepared-banks",
        "execute_required": True,
        "planned_cells": len(COHORTS) * len(SEEDS),
        "cohorts": list(COHORTS),
        "seeds": list(SEEDS),
        "prepared_root": str(prepared_root),
        "output_root": str(output_root),
        "prepared_banks": banks,
        "feature_contract": dict(RAW_CONTEXT_FEATURE_CONTRACT),
        "formal_labels_read": False,
        "development_outcomes_read": False,
        "mse_endpoint_registered": False,
        "phase_a_arrays": sorted(ARRAY_KEYS),
    }


def _produce_grid(*, config_path: str | Path,
                  prepared_root: str | Path,
                  output_root: str | Path,
                  cohorts: tuple[str, ...], seeds: tuple[int, ...],
                  resume: bool = False) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    prepared_root = Path(prepared_root).resolve()
    output_root = Path(output_root).resolve()
    config = _load_config(config_path)
    _require(prepared_root.is_dir() and not prepared_root.is_symlink(),
             "prepared formal-bank root is missing")
    if output_root.exists():
        _require(resume and output_root.is_dir()
                 and not output_root.is_symlink(),
                 "output exists; use --resume for strict validation")
        _require(not any(".partial-" in path.name
                         for path in output_root.rglob("*")),
                 "partial raw-context output exists")
    else:
        output_root.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    _require(cohorts and set(cohorts).issubset(COHORTS)
             and seeds and set(seeds).issubset(SEEDS),
             "requested raw-context grid is outside the registered study")
    for cohort in cohorts:
        counts = config["cohorts"][cohort]["split_episodes"]
        view = _load_prepared_bank(cohort, prepared_root, counts)
        arrays = build_feature_arrays(view)
        cohort_root = output_root / cohort
        if cohort_root.exists():
            _require(cohort_root.is_dir() and not cohort_root.is_symlink(),
                     f"cohort output is unsafe: {cohort}")
            _require({item.name for item in cohort_root.iterdir()}.issubset({
                f"seed-{seed}" for seed in seeds}),
                f"unexpected raw-context cell: {cohort}")
        for seed in seeds:
            directory = cohort_root / f"seed-{seed}"
            if directory.exists():
                _require(resume, f"cell already exists: {directory}")
                manifest = _validate_cell(
                    directory, view=view, seed=seed,
                    expected_arrays=arrays)
            else:
                manifest = _write_cell_atomic(
                    directory, view=view, seed=seed, arrays=arrays)
            records.append({
                "cohort": cohort,
                "seed": seed,
                "manifest_sha256": _sha256_file(
                    directory / "manifest.json"),
                "artifact_sha256": manifest["artifact"]["sha256"],
                "bank_manifest_sha256": view.bank_manifest_sha256,
            })
        del arrays
    _require({item.name for item in output_root.iterdir()}.issubset(
        {*cohorts, "summary.json"}),
        "raw-context output root contains unexpected entries")
    summary = {
        "schema": PRODUCER_SCHEMA,
        "study": "sage-mem-v1",
        "status": "complete-label-free",
        "cells": len(records),
        "cohorts": list(cohorts),
        "seeds": list(seeds),
        "feature_contract": dict(RAW_CONTEXT_FEATURE_CONTRACT),
        "formal_labels_read": False,
        "development_outcomes_read": False,
        "mse_emitted": False,
        "records_sha256": hashlib.sha256(
            _canonical_json(records).encode("utf-8")).hexdigest(),
    }
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        _require(resume and _read_json(
            summary_path, "raw-context summary") == summary,
            "raw-context resume summary differs")
    else:
        _atomic_json(summary_path, summary)
    return summary


def produce_all(*, config_path: str | Path = DEFAULT_CONFIG,
                prepared_root: str | Path = DEFAULT_PREPARED_ROOT,
                output_root: str | Path = DEFAULT_OUTPUT_ROOT,
                resume: bool = False) -> dict[str, Any]:
    """Build the exact registered five-cohort by ten-seed formal grid."""

    return _produce_grid(
        config_path=config_path, prepared_root=prepared_root,
        output_root=output_root, cohorts=COHORTS, seeds=SEEDS,
        resume=resume)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepared-root", type=Path,
                        default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--output-root", type=Path,
                        default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.execute:
        print(_canonical_json(preview_plan(
            config_path=args.config, prepared_root=args.prepared_root,
            output_root=args.output_root)))
        return 0
    summary = produce_all(
        config_path=args.config, prepared_root=args.prepared_root,
        output_root=args.output_root, resume=args.resume)
    print(_canonical_json(summary))
    return 0


__all__ = [
    "ARRAY_KEYS",
    "PreparedBankView",
    "RawContextReferenceError",
    "build_feature_arrays",
    "preview_plan",
    "produce_all",
    "validate_feature_arrays",
]


if __name__ == "__main__":
    raise SystemExit(main())
