"""Immutable protocol loader for the formal official-PushT memory study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from lewm.models.official_lewm_pusht import OFFICIAL_PUSHT_CHECKPOINT
from lewm.official_tasks.artifacts import sha256_file
from lewm.official_tasks.pusht_hdf5 import (
    OFFICIAL_PUSHT_DATASET_ARCHIVE,
    OFFICIAL_PUSHT_EXTRACTED_HDF5,
)
from lewm.official_tasks.pusht_memory import PUSHT_MEMORY_TASKS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PUSHT_SPEC = ROOT / "configs/official_pusht_memory.yaml"
DEFAULT_PUSHT_LOCK = ROOT / "configs/official_pusht_memory.lock.json"
PUSHT_LOCK_SCHEMA = "official_pusht_memory_lock_v1"
ALLOWED_PUSHT_DEVICES = ("cuda:1", "cuda:2")


def resolve_pusht_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_pusht_device(device: str) -> str:
    if device not in ALLOWED_PUSHT_DEVICES:
        raise ValueError(
            f"formal PushT jobs permit only {ALLOWED_PUSHT_DEVICES}; "
            f"got {device!r}")
    return device


def _require_equal(mapping: dict[str, Any], expected: dict[str, Any],
                   path: str) -> None:
    for key, value in expected.items():
        if mapping.get(key) != value:
            raise ValueError(
                f"{path}.{key} must be {value!r}, got {mapping.get(key)!r}")


def _validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("PushT spec schema_version must be 1")
    if spec.get("protocol_status") != "locked_before_formal_run":
        raise ValueError("PushT formal spec is not locked before the run")
    archive = OFFICIAL_PUSHT_DATASET_ARCHIVE
    extracted = OFFICIAL_PUSHT_EXTRACTED_HDF5
    _require_equal(spec.get("dataset", {}), {
        "repo_id": archive.repo_id,
        "revision": archive.revision,
        "archive_filename": archive.filename,
        "archive_sha256": archive.sha256,
        "archive_size": archive.size,
        "archive_file_commit": archive.file_commit,
        "hdf5_path": (
            "outputs/paper_a_strengthening/data/pusht_expert_train.h5"),
        "hdf5_sha256": extracted.sha256,
        "hdf5_size": extracted.size,
        "required_rows": 2_336_736,
        "required_episodes": 18_685,
        "pixel_shape": [224, 224, 3],
        "pixel_filter_id": 32_001,
    }, "dataset")
    checkpoint = OFFICIAL_PUSHT_CHECKPOINT
    _require_equal(spec.get("official_host", {}), {
        "repo_id": checkpoint.repo_id,
        "revision": checkpoint.revision,
        "bundle_path": (
            "outputs/paper_a_strengthening/pretrained/lewm-pusht"),
        "config_sha256": checkpoint.config_sha256,
        "weights_sha256": checkpoint.weights_sha256,
        "weights_size": checkpoint.weights_size,
        "frozen_encoder": True,
        "frozen_predictor": True,
        "latent_dim": 192,
        "image_size": 224,
        "context": 3,
        "action_block_dim": 10,
    }, "official_host")
    expected_tasks = [
        {
            "key": "transient-visual-token-recall",
            "display_name": PUSHT_MEMORY_TASKS[0].semantic_name,
            "classes": PUSHT_MEMORY_TASKS[0].num_classes,
            "label_seed": 461_103,
        },
        {
            "key": "multi-item-visual-binding-recall",
            "display_name": PUSHT_MEMORY_TASKS[1].semantic_name,
            "classes": PUSHT_MEMORY_TASKS[1].num_classes,
            "label_seed": 461_203,
        },
    ]
    if spec.get("semantic_tasks") != expected_tasks:
        raise ValueError("semantic_tasks differ from the code contract")
    _require_equal(spec.get("sequence", {}), {
        "num_frames": 20,
        "frame_skip": 5,
        "raw_action_dim": 2,
        "cue_start": 1,
        "cue_length": 3,
        "decision_index": 19,
        "decision_observation_excluded": True,
        "final_context_indices": [16, 17, 18],
        "action_alignment": (
            "action[t] is the five raw controls after z[t] and before z[t+1]"),
        "context_cause_action_indices": [15, 16, 17],
        "decision_prior_action_index": 18,
        "shortcut_action_indices": [15, 16, 17, 18],
        "base_frames_stored": False,
        "task_cache_stores_only_cue_latents": True,
    }, "sequence")
    selection = spec.get("selection", {})
    _require_equal(selection, {
        "train": {"episodes": 1200},
        "validation": {"episodes": 480},
        "split_seed": 461_001,
        "start_seed": 461_002,
        "one_sequence_per_episode": True,
        "train_validation_episode_disjoint": True,
    }, "selection")
    _require_equal(spec.get("normalization", {}), {
        "raw_action_ddof": 1,
        "statistics_must_be_computed_from_dataset": True,
    }, "normalization")
    training = spec.get("carrier_training", {})
    if training.get("arms") != ["none", "gru", "lstm", "ssm", "fixed_trust"]:
        raise ValueError("carrier_training.arms differs from the formal grid")
    if training.get("seeds") != [0, 1, 2, 3, 4]:
        raise ValueError("carrier_training.seeds differs from the formal grid")
    launcher = spec.get("launcher", {})
    _require_equal(launcher, {
        "allowed_devices": list(ALLOWED_PUSHT_DEVICES),
        "preview_by_default": True,
        "explicit_execute_required": True,
        "jobs_per_gpu": 1,
    }, "launcher")


def load_locked_pusht_spec(
        spec_path: str | Path = DEFAULT_PUSHT_SPEC,
        lock_path: str | Path = DEFAULT_PUSHT_LOCK) -> dict[str, Any]:
    """Load the protocol only when spec and every producer hash match."""

    spec_path = resolve_pusht_path(spec_path)
    lock_path = resolve_pusht_path(lock_path)
    if not spec_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError("formal PushT spec and lock are both required")
    lock = json.loads(lock_path.read_text())
    if lock.get("schema") != PUSHT_LOCK_SCHEMA \
            or lock.get("immutable") is not True:
        raise ValueError("invalid or mutable formal PushT lock")
    actual_spec_hash = sha256_file(spec_path)
    if lock.get("spec_sha256") != actual_spec_hash:
        raise ValueError("formal PushT spec differs from its immutable lock")
    if resolve_pusht_path(lock.get("spec_path", "")).resolve() \
            != spec_path.resolve():
        raise ValueError("formal PushT lock points to a different spec")
    producers = lock.get("producer_sha256")
    if not isinstance(producers, dict) or not producers:
        raise ValueError("formal PushT lock has no producer hashes")
    for source, expected in producers.items():
        source_path = resolve_pusht_path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"locked PushT producer is missing: {source}")
        if sha256_file(source_path) != expected:
            raise ValueError(f"locked PushT producer changed: {source}")
    spec = yaml.safe_load(spec_path.read_text())
    if not isinstance(spec, dict):
        raise ValueError("formal PushT spec must be a YAML mapping")
    _validate_spec(spec)
    spec["_lock_record"] = {
        "path": str(lock_path.resolve()),
        "sha256": sha256_file(lock_path),
        "spec_sha256": actual_spec_hash,
        "producer_sha256": producers,
    }
    return spec


def pusht_lock_receipt(spec: dict[str, Any]) -> dict[str, Any]:
    if "_lock_record" not in spec:
        raise ValueError("PushT spec was not loaded through its immutable lock")
    return dict(spec["_lock_record"])


__all__ = [
    "ALLOWED_PUSHT_DEVICES",
    "DEFAULT_PUSHT_LOCK",
    "DEFAULT_PUSHT_SPEC",
    "load_locked_pusht_spec",
    "pusht_lock_receipt",
    "resolve_pusht_path",
    "validate_pusht_device",
]
