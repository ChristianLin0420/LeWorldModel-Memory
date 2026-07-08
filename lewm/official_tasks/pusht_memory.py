"""Semantic visual-memory overlays and cache contracts for official PushT.

The overlay is exogenous: every counterfactual shares the same PushT pixels,
states, proprioception, actions, and final legal observation window.  Only a
short, explicitly declared cue interval changes.  Constructing an overlay
always runs the counterfactual leakage audit before returning data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from lewm.models.official_lewm_pusht import OFFICIAL_PUSHT_CHECKPOINT
from lewm.official_tasks.pusht_hdf5 import (
    NativePushTSequence,
    OFFICIAL_FRAME_SKIP,
    OFFICIAL_PUSHT_DATASET_ARCHIVE,
    OFFICIAL_PUSHT_EXTRACTED_HDF5,
    OfficialPushTHDF5,
    PushTIdentityError,
    PushTSequenceSelection,
)


OFFICIAL_PUSHT_CONTEXT = 3
UPSTREAM_LEWM_REVISION = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
UPSTREAM_STABLE_WORLDMODEL_REVISION = (
    "0ef3856875e70a1283e637fcd2ab936eae6c4e6f"
)


class PushTLeakageError(ValueError):
    """Counterfactual task identity leaked outside the declared cue."""


@dataclass(frozen=True)
class PushTMemoryTask:
    semantic_name: str
    num_classes: int


PUSHT_MEMORY_TASKS = (
    PushTMemoryTask("PushT transient visual-token recall", 4),
    PushTMemoryTask("PushT multi-item visual-binding recall", 6),
)
_TASKS_BY_NAME = {task.semantic_name: task for task in PUSHT_MEMORY_TASKS}


def get_pusht_memory_task(semantic_name: str) -> PushTMemoryTask:
    try:
        return _TASKS_BY_NAME[semantic_name]
    except KeyError as exc:
        supported = ", ".join(repr(task.semantic_name)
                              for task in PUSHT_MEMORY_TASKS)
        raise ValueError(
            f"unknown semantic PushT memory task {semantic_name!r}; "
            f"expected one of {supported}") from exc


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
            isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be an object with string keys")
    return value


def _exact_keys(value: Mapping[str, Any], path: str,
                fields: set[str]) -> None:
    missing, extra = sorted(fields - set(value)), sorted(set(value) - fields)
    if missing or extra:
        raise ValueError(
            f"{path} fields differ from the frozen schema; "
            f"missing={missing}, unexpected={extra}")


def _integer(value: object, path: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _sha256(value: object, path: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PushTIdentityError(
            f"{path} must be a lowercase 64-character SHA-256")
    return value


def _require_identity(actual: object, expected: object, path: str) -> None:
    if actual != expected:
        raise PushTIdentityError(
            f"{path} does not match the pinned official identity: "
            f"expected {expected!r}, got {actual!r}")


@dataclass(frozen=True)
class PushTMemoryCacheConfig:
    """Fully resolved input and selection identity for a frozen-host cache."""

    cache_root: Path
    semantic_task_name: str
    extracted_hdf5_sha256: str
    extracted_hdf5_size: int
    num_frames: int
    cue_start: int
    cue_length: int
    train_count: int
    validation_count: int
    split_seed: int
    start_seed: int
    label_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.cache_root, Path) or not str(self.cache_root):
            raise ValueError("cache_root must be a non-empty pathlib.Path")
        task = get_pusht_memory_task(self.semantic_task_name)
        _sha256(self.extracted_hdf5_sha256, "extracted_hdf5_sha256")
        _integer(self.extracted_hdf5_size, "extracted_hdf5_size", 1)
        _integer(self.num_frames, "num_frames", 2)
        _integer(self.cue_start, "cue_start", 0)
        _integer(self.cue_length, "cue_length", 1)
        if self.cue_end > self.num_frames - OFFICIAL_PUSHT_CONTEXT:
            raise ValueError("cue must end before the final legal context")
        for name, value, minimum in (
                ("train_count", self.train_count, task.num_classes),
                ("validation_count", self.validation_count, task.num_classes),
                ("split_seed", self.split_seed, 0),
                ("start_seed", self.start_seed, 0),
                ("label_seed", self.label_seed, 0)):
            _integer(value, name, minimum)

    @property
    def task(self) -> PushTMemoryTask:
        return get_pusht_memory_task(self.semantic_task_name)

    @property
    def cue_end(self) -> int:
        return self.cue_start + self.cue_length

    @classmethod
    def from_json(cls, path: str | Path) -> "PushTMemoryCacheConfig":
        try:
            value = json.loads(Path(path).read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid PushT cache JSON: {exc}") from exc
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: object) -> "PushTMemoryCacheConfig":
        root = _mapping(value, "config")
        _exact_keys(root, "config", {
            "schema_version", "cache_root", "semantic_task_name",
            "dataset", "checkpoint", "sequence", "selection",
        })
        if root["schema_version"] != 1:
            raise ValueError("config.schema_version must be 1")
        if not isinstance(root["cache_root"], str) or not root["cache_root"].strip():
            raise ValueError("config.cache_root must be a non-empty path string")
        if not isinstance(root["semantic_task_name"], str):
            raise ValueError("config.semantic_task_name must be a string")
        task = get_pusht_memory_task(root["semantic_task_name"])

        dataset = _mapping(root["dataset"], "config.dataset")
        _exact_keys(dataset, "config.dataset", {
            "repo_id", "revision", "filename", "archive_sha256",
            "archive_size", "file_commit", "extracted_hdf5_sha256",
            "extracted_hdf5_size",
        })
        archive = OFFICIAL_PUSHT_DATASET_ARCHIVE
        for field, expected in (
                ("repo_id", archive.repo_id),
                ("revision", archive.revision),
                ("filename", archive.filename),
                ("archive_sha256", archive.sha256),
                ("archive_size", archive.size),
                ("file_commit", archive.file_commit)):
            _require_identity(dataset[field], expected,
                              f"config.dataset.{field}")
        hdf5_sha = _sha256(dataset["extracted_hdf5_sha256"],
                           "config.dataset.extracted_hdf5_sha256")
        hdf5_size = _integer(
            dataset["extracted_hdf5_size"],
            "config.dataset.extracted_hdf5_size", 1)

        checkpoint = _mapping(root["checkpoint"], "config.checkpoint")
        _exact_keys(checkpoint, "config.checkpoint", {
            "repo_id", "revision", "config_sha256", "weights_sha256",
            "weights_size",
        })
        ckpt = OFFICIAL_PUSHT_CHECKPOINT
        for field, expected in (
                ("repo_id", ckpt.repo_id),
                ("revision", ckpt.revision),
                ("config_sha256", ckpt.config_sha256),
                ("weights_sha256", ckpt.weights_sha256),
                ("weights_size", ckpt.weights_size)):
            _require_identity(checkpoint[field], expected,
                              f"config.checkpoint.{field}")

        sequence = _mapping(root["sequence"], "config.sequence")
        _exact_keys(sequence, "config.sequence", {
            "frame_skip", "num_frames", "cue_start", "cue_length",
        })
        if sequence["frame_skip"] != OFFICIAL_FRAME_SKIP:
            raise ValueError("config.sequence.frame_skip must be 5")
        num_frames = _integer(sequence["num_frames"],
                              "config.sequence.num_frames", 2)
        cue_start = _integer(sequence["cue_start"],
                             "config.sequence.cue_start", 0)
        cue_length = _integer(sequence["cue_length"],
                              "config.sequence.cue_length", 1)
        if cue_start + cue_length > num_frames - OFFICIAL_PUSHT_CONTEXT:
            raise ValueError(
                "the cue must end before the frozen host's final 3-frame "
                "legal decision window")

        selection = _mapping(root["selection"], "config.selection")
        _exact_keys(selection, "config.selection", {
            "train_count", "validation_count", "split_seed",
            "start_seed", "label_seed",
        })
        train_count = _integer(selection["train_count"],
                               "config.selection.train_count", 1)
        validation_count = _integer(selection["validation_count"],
                                    "config.selection.validation_count", 1)
        split_seed = _integer(selection["split_seed"],
                              "config.selection.split_seed", 0)
        start_seed = _integer(selection["start_seed"],
                              "config.selection.start_seed", 0)
        label_seed = _integer(selection["label_seed"],
                              "config.selection.label_seed", 0)
        if min(train_count, validation_count) < task.num_classes:
            raise ValueError(
                "each split must include at least one example per task class")

        return cls(
            cache_root=Path(root["cache_root"]),
            semantic_task_name=task.semantic_name,
            extracted_hdf5_sha256=hdf5_sha,
            extracted_hdf5_size=hdf5_size,
            num_frames=num_frames,
            cue_start=cue_start,
            cue_length=cue_length,
            train_count=train_count,
            validation_count=validation_count,
            split_seed=split_seed,
            start_seed=start_seed,
            label_seed=label_seed,
        )

    def select(self, dataset: OfficialPushTHDF5
               ) -> tuple[PushTSequenceSelection, ...]:
        if dataset.expected_hdf5_sha256 != self.extracted_hdf5_sha256:
            raise PushTIdentityError(
                "dataset object and cache config pin different extracted files")
        if dataset.hdf5_size != self.extracted_hdf5_size:
            raise PushTIdentityError(
                "dataset object and cache config pin different extracted sizes")
        return dataset.select_sequences(
            num_frames=self.num_frames,
            train_count=self.train_count,
            validation_count=self.validation_count,
            num_classes=self.task.num_classes,
            split_seed=self.split_seed,
            start_seed=self.start_seed,
            label_seed=self.label_seed,
        )


def validate_counterfactual_no_leakage(counterfactual_frames: np.ndarray,
                                       cue_start: int, cue_end: int) -> None:
    """Require all labels to differ only inside an explicit cue interval."""

    values = np.asarray(counterfactual_frames)
    if values.ndim != 5 or values.shape[-1] != 3 or values.shape[0] < 2:
        raise PushTLeakageError(
            "counterfactual_frames must be (classes, time, height, width, 3)")
    if values.dtype != np.uint8:
        raise PushTLeakageError("counterfactual frames must remain uint8")
    if isinstance(cue_start, bool) or isinstance(cue_end, bool) \
            or not isinstance(cue_start, int) or not isinstance(cue_end, int) \
            or not 0 <= cue_start < cue_end <= values.shape[1]:
        raise PushTLeakageError("invalid cue interval")
    reference = values[0]
    for label in range(1, values.shape[0]):
        if not np.array_equal(values[label, :cue_start],
                              reference[:cue_start]):
            raise PushTLeakageError(
                f"class {label} leaks before the declared cue")
        if not np.array_equal(values[label, cue_end:], reference[cue_end:]):
            raise PushTLeakageError(
                f"class {label} leaks after the declared cue")
    cue_signatures = {
        values[label, cue_start:cue_end].tobytes()
        for label in range(values.shape[0])
    }
    if len(cue_signatures) != values.shape[0]:
        raise PushTLeakageError("two or more class cues are visually identical")


_TOKEN_COLORS = np.asarray([
    (230, 57, 70),
    (40, 160, 84),
    (45, 108, 223),
    (239, 174, 45),
], dtype=np.uint8)
_BINDING_PERMUTATIONS = (
    (0, 1, 2), (0, 2, 1), (1, 0, 2),
    (1, 2, 0), (2, 0, 1), (2, 1, 0),
)


def _draw_transient_token(frame: np.ndarray, label: int) -> None:
    height, width = frame.shape[:2]
    thickness = max(1, min(height, width) // 24)
    color = _TOKEN_COLORS[label]
    frame[:thickness] = color
    frame[-thickness:] = color
    frame[:, :thickness] = color
    frame[:, -thickness:] = color
    side = max(2, min(height, width) // 8)
    frame[thickness:thickness + side,
          thickness:thickness + side] = color


def _draw_visual_binding(frame: np.ndarray, label: int) -> None:
    height, width = frame.shape[:2]
    margin = max(1, min(height, width) // 32)
    swatch_h = max(2, height // 10)
    available = width - 4 * margin
    swatch_w = max(1, available // 3)
    colors = _TOKEN_COLORS[np.asarray(_BINDING_PERMUTATIONS[label])]
    for slot, color in enumerate(colors):
        left = margin + slot * (swatch_w + margin)
        right = min(width - margin, left + swatch_w)
        frame[margin:margin + swatch_h, left:right] = color
    # A neutral anchor makes the ordered slots explicit without encoding label.
    anchor = max(1, swatch_h // 3)
    frame[margin + swatch_h:margin + swatch_h + anchor,
          margin:width - margin] = np.asarray((245, 245, 245), dtype=np.uint8)


def render_counterfactual_overlays(
        base_frames: np.ndarray, semantic_task_name: str,
        cue_start: int, cue_length: int) -> np.ndarray:
    """Render every label counterfactual and prove cue-only differences."""

    frames = np.asarray(base_frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 \
            or frames.dtype != np.uint8:
        raise ValueError("base_frames must be uint8 THWC RGB")
    if min(frames.shape[1:3]) < 8:
        raise ValueError("visual overlays require frames at least 8x8")
    if isinstance(cue_start, bool) or not isinstance(cue_start, int) \
            or isinstance(cue_length, bool) or not isinstance(cue_length, int) \
            or cue_length <= 0 or not 0 <= cue_start < cue_start + cue_length \
            <= frames.shape[0]:
        raise ValueError("invalid cue interval")
    task = get_pusht_memory_task(semantic_task_name)
    variants = np.repeat(frames[None], task.num_classes, axis=0)
    for label in range(task.num_classes):
        for step in range(cue_start, cue_start + cue_length):
            if task.semantic_name == "PushT transient visual-token recall":
                _draw_transient_token(variants[label, step], label)
            else:
                _draw_visual_binding(variants[label, step], label)
    validate_counterfactual_no_leakage(
        variants, cue_start, cue_start + cue_length)
    variants.setflags(write=False)
    return variants


def render_single_overlay(base_frames: np.ndarray, semantic_task_name: str,
                          label: int, cue_start: int,
                          cue_length: int) -> np.ndarray:
    """Render one cue without materializing copies for all task classes."""

    frames = np.asarray(base_frames)
    task = get_pusht_memory_task(semantic_task_name)
    if isinstance(label, bool) or not isinstance(label, int) \
            or not 0 <= label < task.num_classes:
        raise ValueError("label is outside the semantic task vocabulary")
    if frames.ndim != 4 or frames.shape[-1] != 3 \
            or frames.dtype != np.uint8 or min(frames.shape[1:3]) < 8:
        raise ValueError("base_frames must be uint8 THWC RGB and at least 8x8")
    cue_end = cue_start + cue_length
    if isinstance(cue_start, bool) or not isinstance(cue_start, int) \
            or isinstance(cue_length, bool) or not isinstance(cue_length, int) \
            or cue_length <= 0 or not 0 <= cue_start < cue_end <= len(frames):
        raise ValueError("invalid cue interval")
    result = frames.copy()
    for step in range(cue_start, cue_end):
        if task.semantic_name == "PushT transient visual-token recall":
            _draw_transient_token(result[step], label)
        else:
            _draw_visual_binding(result[step], label)
    if not np.array_equal(result[:cue_start], frames[:cue_start]) \
            or not np.array_equal(result[cue_end:], frames[cue_end:]):
        raise PushTLeakageError("single-label renderer modified pixels outside cue")
    if np.array_equal(result[cue_start:cue_end], frames[cue_start:cue_end]):
        raise PushTLeakageError("single-label renderer did not create a cue")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PushTMemorySequence:
    semantic_task_name: str
    label: int
    frames: np.ndarray
    native: NativePushTSequence
    selection: PushTSequenceSelection


def build_memory_sequence(native: NativePushTSequence,
                          selection: PushTSequenceSelection,
                          config: PushTMemoryCacheConfig,
                          ) -> PushTMemorySequence:
    """Apply a semantic cue after checking selection and counterfactual parity."""

    if native.episode_index != selection.episode_index \
            or native.local_start != selection.local_start:
        raise ValueError("native sequence does not match its selection record")
    if native.frames.shape[0] != config.num_frames:
        raise ValueError("native sequence length differs from cache config")
    if not 0 <= selection.label < config.task.num_classes:
        raise ValueError("selection label is outside the task vocabulary")
    frames = render_single_overlay(
        native.frames, config.semantic_task_name, selection.label,
        config.cue_start, config.cue_length)
    return PushTMemorySequence(
        semantic_task_name=config.semantic_task_name,
        label=selection.label, frames=frames, native=native,
        selection=selection)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def cache_relative_path(config: PushTMemoryCacheConfig) -> Path:
    """Return a deterministic namespace; this function never creates it."""

    return Path(
        "frozen-official-pusht",
        _slug(config.semantic_task_name),
        f"dataset-{config.extracted_hdf5_sha256[:16]}",
        f"checkpoint-{OFFICIAL_PUSHT_CHECKPOINT.weights_sha256[:16]}",
        f"frames-{config.num_frames}-skip-{OFFICIAL_FRAME_SKIP}",
        (f"split-{config.split_seed}-start-{config.start_seed}-"
         f"label-{config.label_seed}"),
    )


def cache_path(config: PushTMemoryCacheConfig) -> Path:
    """Resolve the configured cache location without touching the filesystem."""

    return config.cache_root / cache_relative_path(config)


def cache_manifest(config: PushTMemoryCacheConfig,
                   dataset: OfficialPushTHDF5,
                   selections: Sequence[PushTSequenceSelection],
                   ) -> dict[str, Any]:
    """Build JSON-safe provenance for a future cache writer."""

    if dataset.expected_hdf5_sha256 != config.extracted_hdf5_sha256:
        raise PushTIdentityError(
            "manifest dataset and config identities do not match")
    if dataset.hdf5_size != config.extracted_hdf5_size:
        raise PushTIdentityError(
            "manifest dataset and config sizes do not match")
    expected_count = config.train_count + config.validation_count
    if len(selections) != expected_count:
        raise ValueError(
            f"manifest expected {expected_count} selections, got {len(selections)}")
    train_eps = {item.episode_index for item in selections
                 if item.split == "train"}
    validation_eps = {item.episode_index for item in selections
                      if item.split == "validation"}
    if train_eps & validation_eps:
        raise PushTLeakageError(
            "train and validation selections share an episode")
    if len(train_eps) != config.train_count \
            or len(validation_eps) != config.validation_count:
        raise ValueError("selection split counts or episode uniqueness are invalid")
    if any(not 0 <= item.label < config.task.num_classes
           for item in selections):
        raise ValueError("manifest selection contains an invalid class label")

    archive = asdict(OFFICIAL_PUSHT_DATASET_ARCHIVE)
    checkpoint = asdict(OFFICIAL_PUSHT_CHECKPOINT)
    return {
        "schema": "frozen_official_pusht_memory_cache_v1",
        "semantic_task_name": config.semantic_task_name,
        "num_classes": config.task.num_classes,
        "cache_relative_path": str(cache_relative_path(config)),
        "dataset": {
            "archive": archive,
            "published_extracted_hdf5": asdict(
                OFFICIAL_PUSHT_EXTRACTED_HDF5),
            "extracted_hdf5_sha256": config.extracted_hdf5_sha256,
            "extracted_hdf5_size": config.extracted_hdf5_size,
            "required_schema_fingerprint": dataset.schema.fingerprint,
            "num_rows": dataset.schema.num_rows,
            "num_episodes": dataset.schema.num_episodes,
        },
        "checkpoint": checkpoint,
        "upstream_contracts": {
            "lewm_revision": UPSTREAM_LEWM_REVISION,
            "stable_worldmodel_revision": UPSTREAM_STABLE_WORLDMODEL_REVISION,
        },
        "sequence": {
            "frame_skip": OFFICIAL_FRAME_SKIP,
            "native_action_block_dim": 10,
            "num_frames": config.num_frames,
            "cue_start": config.cue_start,
            "cue_end_exclusive": config.cue_end,
            "legal_final_context": OFFICIAL_PUSHT_CONTEXT,
        },
        "selection": [asdict(item) for item in selections],
    }


__all__ = [
    "OFFICIAL_PUSHT_CONTEXT",
    "PUSHT_MEMORY_TASKS",
    "PushTLeakageError",
    "PushTMemoryCacheConfig",
    "PushTMemorySequence",
    "PushTMemoryTask",
    "build_memory_sequence",
    "cache_manifest",
    "cache_path",
    "cache_relative_path",
    "get_pusht_memory_task",
    "render_counterfactual_overlays",
    "render_single_overlay",
    "validate_counterfactual_no_leakage",
]
