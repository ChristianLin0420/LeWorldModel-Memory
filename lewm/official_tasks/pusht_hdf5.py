"""Strict reader for the HDF5 dataset used by the official PushT LeWM.

The upstream ``stable-worldmodel`` HDF5 format stores every column at the
root and records episode boundaries in ``ep_len``/``ep_offset``.  Its PushT
configuration samples observations every five raw environment steps while
keeping all five two-dimensional actions.  Consequently one model action is
the time-major flattening of a native ``(5, 2)`` action block, not a repeated
or interpolated two-dimensional action.

This reader intentionally has a smaller surface than the upstream training
dataset.  It verifies a caller-pinned SHA-256, validates the columns needed by
the released PushT checkpoint, and returns only sequences that remain inside
one episode.  No padding or boundary-frame duplication is permitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

import h5py
import hdf5plugin  # noqa: F401  (registers official pixels' Blosc filter 32001)
import numpy as np


OFFICIAL_FRAME_SKIP = 5
OFFICIAL_RAW_ACTION_DIM = 2
OFFICIAL_ACTION_BLOCK_DIM = OFFICIAL_FRAME_SKIP * OFFICIAL_RAW_ACTION_DIM
REQUIRED_COLUMNS = ("pixels", "action", "proprio", "state")


class PushTSchemaError(ValueError):
    """The HDF5 file does not implement the official PushT data contract."""


class PushTIdentityError(RuntimeError):
    """A local file does not match its explicitly pinned identity."""


@dataclass(frozen=True)
class OfficialPushTDatasetArchiveIdentity:
    """Published identity of the compressed official PushT dataset archive."""

    repo_id: str
    revision: str
    filename: str
    sha256: str
    size: int
    file_commit: str


@dataclass(frozen=True)
class OfficialPushTExtractedIdentity:
    """Verified identity of the decompressed official HDF5 payload."""

    filename: str
    sha256: str
    size: int


OFFICIAL_PUSHT_DATASET_ARCHIVE = OfficialPushTDatasetArchiveIdentity(
    repo_id="quentinll/lewm-pusht",
    revision="655cd446b9929369d7d406001da85c15d1457850",
    filename="pusht_expert_train.h5.zst",
    sha256="7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318",
    size=13_136_247_974,
    file_commit="6eebda9ccea0d55de7586c31145cac8dea60327b",
)
OFFICIAL_PUSHT_EXTRACTED_HDF5 = OfficialPushTExtractedIdentity(
    filename="pusht_expert_train.h5",
    sha256="b6ebd9ac94bbe9e383f6e7a9cd92d74e9aa665ea57b758ed3717b0ee7df8d4fb",
    size=46_300_921_856,
)


def _validate_sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PushTIdentityError(f"{field} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise PushTIdentityError(f"{field} must be hexadecimal") from exc
    if value != value.lower():
        raise PushTIdentityError(f"{field} must use lowercase hexadecimal")
    return value


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_native_action_blocks(
        action_blocks: np.ndarray, raw_mean: np.ndarray,
        raw_std_ddof1: np.ndarray) -> np.ndarray:
    """Normalize each raw 2-D action before flattening its native 5-step block."""

    blocks = np.asarray(action_blocks)
    mean = np.asarray(raw_mean, dtype=np.float64)
    std = np.asarray(raw_std_ddof1, dtype=np.float64)
    if blocks.ndim < 2 or blocks.shape[-1] != OFFICIAL_ACTION_BLOCK_DIM:
        raise ValueError("action_blocks must end in the native 10-D dimension")
    if mean.shape != (OFFICIAL_RAW_ACTION_DIM,) \
            or std.shape != (OFFICIAL_RAW_ACTION_DIM,):
        raise ValueError("raw action mean/std must each have shape (2,)")
    if not np.isfinite(blocks).all() or not np.isfinite(mean).all() \
            or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("action blocks and normalization statistics must be finite")
    raw = blocks.astype(np.float64).reshape(
        *blocks.shape[:-1], OFFICIAL_FRAME_SKIP, OFFICIAL_RAW_ACTION_DIM)
    normalized = (raw - mean) / std
    return normalized.reshape(blocks.shape).astype(np.float32)


@dataclass(frozen=True)
class PushTHDF5Schema:
    episode_lengths: tuple[int, ...]
    episode_offsets: tuple[int, ...]
    num_rows: int
    pixel_shape: tuple[int, int, int]
    pixel_filter_ids: tuple[int, ...]
    action_dtype: str
    proprio_dtype: str
    state_dtype: str
    fingerprint: str

    @property
    def num_episodes(self) -> int:
        return len(self.episode_lengths)


@dataclass(frozen=True)
class PushTSequenceSelection:
    """One deterministic, episode-disjoint sequence selection."""

    split: Literal["train", "validation"]
    episode_index: int
    local_start: int
    label: int


@dataclass(frozen=True)
class NativePushTSequence:
    """Native observations and non-overlapping 10-D action transitions."""

    frames: np.ndarray
    actions: np.ndarray
    proprio: np.ndarray
    state: np.ndarray
    episode_index: int
    local_start: int
    local_frame_indices: np.ndarray
    global_frame_indices: np.ndarray
    global_action_indices: np.ndarray

    def __post_init__(self) -> None:
        if self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise PushTSchemaError("sequence frames must be THWC RGB")
        steps = self.frames.shape[0]
        if self.actions.shape != (steps - 1, OFFICIAL_ACTION_BLOCK_DIM):
            raise PushTSchemaError(
                "sequence actions must contain one native 10-D block per transition")
        if self.proprio.shape != (steps, 4) or self.state.shape != (steps, 7):
            raise PushTSchemaError("sequence proprio/state shapes are invalid")
        if self.local_frame_indices.shape != (steps,) \
                or self.global_frame_indices.shape != (steps,):
            raise PushTSchemaError("sequence frame-index shapes are invalid")
        if self.global_action_indices.shape != (
                steps - 1, OFFICIAL_FRAME_SKIP):
            raise PushTSchemaError("sequence action-index shape is invalid")
        if steps > 1:
            if not np.all(np.diff(self.local_frame_indices) == OFFICIAL_FRAME_SKIP):
                raise PushTSchemaError("sequence frame cadence is not five raw steps")
            if len(np.unique(self.global_frame_indices)) != steps:
                raise PushTSchemaError("sequence contains duplicated frames")


class OfficialPushTHDF5:
    """Identity-pinned and schema-validated official PushT HDF5 reader."""

    def __init__(self, path: str | Path, *, expected_hdf5_sha256: str,
                 finite_check_chunk_rows: int = 65_536) -> None:
        self.path = Path(path)
        self.expected_hdf5_sha256 = _validate_sha256(
            expected_hdf5_sha256, "expected_hdf5_sha256")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.hdf5_size = self.path.stat().st_size
        if isinstance(finite_check_chunk_rows, bool) \
                or not isinstance(finite_check_chunk_rows, int) \
                or finite_check_chunk_rows <= 0:
            raise ValueError("finite_check_chunk_rows must be a positive integer")
        self.finite_check_chunk_rows = finite_check_chunk_rows
        actual = sha256_file(self.path)
        if actual != self.expected_hdf5_sha256:
            raise PushTIdentityError(
                f"PushT HDF5 SHA-256 mismatch: expected "
                f"{self.expected_hdf5_sha256}, got {actual}")
        self.schema = self._validate_schema()

    @staticmethod
    def _open(path: Path) -> h5py.File:
        return h5py.File(path, "r", swmr=True,
                         rdcc_nbytes=256 * 1024 * 1024)

    @staticmethod
    def _require_dataset(handle: h5py.File, key: str) -> h5py.Dataset:
        if key not in handle:
            raise PushTSchemaError(f"required root dataset {key!r} is missing")
        dataset = handle[key]
        if not isinstance(dataset, h5py.Dataset):
            raise PushTSchemaError(f"root object {key!r} must be a dataset")
        return dataset

    @staticmethod
    def _require_dtype(dataset: h5py.Dataset, dtype: np.dtype,
                       key: str) -> None:
        if dataset.dtype != np.dtype(dtype):
            raise PushTSchemaError(
                f"{key} dtype must be {np.dtype(dtype)}, got {dataset.dtype}")

    def _check_finite(self, dataset: h5py.Dataset, key: str,
                      terminal_rows: np.ndarray | None = None) -> None:
        row_count = dataset.shape[0]
        for start in range(0, row_count, self.finite_check_chunk_rows):
            stop = min(row_count, start + self.finite_check_chunk_rows)
            values = dataset[start:stop]
            finite = np.isfinite(values).reshape(len(values), -1)
            if terminal_rows is not None and np.any(
                    finite.any(axis=1) != finite.all(axis=1)):
                raise PushTSchemaError(
                    f"{key} has a partially non-finite row; action rows must "
                    "be entirely finite or entirely non-finite")
            bad_rows = np.flatnonzero(~finite.all(axis=1))
            if not len(bad_rows):
                continue
            global_rows = bad_rows.astype(np.int64) + start
            if terminal_rows is None or not np.all(
                    np.isin(global_rows, terminal_rows)):
                preview = ", ".join(map(str, global_rows[:5]))
                raise PushTSchemaError(
                    f"{key} contains non-finite values outside allowed "
                    f"episode-terminal rows (rows: {preview})")

    def _validate_schema(self) -> PushTHDF5Schema:
        try:
            with self._open(self.path) as handle:
                ep_len_ds = self._require_dataset(handle, "ep_len")
                ep_offset_ds = self._require_dataset(handle, "ep_offset")
                self._require_dtype(ep_len_ds, np.int32, "ep_len")
                self._require_dtype(ep_offset_ds, np.int64, "ep_offset")
                if ep_len_ds.ndim != 1 or ep_offset_ds.ndim != 1:
                    raise PushTSchemaError("ep_len and ep_offset must be one-dimensional")
                if ep_len_ds.shape != ep_offset_ds.shape or not ep_len_ds.shape[0]:
                    raise PushTSchemaError(
                        "ep_len and ep_offset must be non-empty and equally sized")

                lengths = ep_len_ds[:].astype(np.int64, copy=False)
                offsets = ep_offset_ds[:].astype(np.int64, copy=False)
                if np.any(lengths <= 0):
                    raise PushTSchemaError("every episode length must be positive")
                expected_offsets = np.concatenate((
                    np.asarray([0], dtype=np.int64),
                    np.cumsum(lengths[:-1], dtype=np.int64),
                ))
                if not np.array_equal(offsets, expected_offsets):
                    raise PushTSchemaError(
                        "ep_offset must be cumulative with no gaps or overlaps")
                num_rows = int(lengths.sum(dtype=np.int64))

                pixels = self._require_dataset(handle, "pixels")
                action = self._require_dataset(handle, "action")
                proprio = self._require_dataset(handle, "proprio")
                state = self._require_dataset(handle, "state")
                for key, dataset in (
                        ("pixels", pixels), ("action", action),
                        ("proprio", proprio), ("state", state)):
                    if dataset.shape[0] != num_rows:
                        raise PushTSchemaError(
                            f"{key} has {dataset.shape[0]} rows; expected {num_rows}")

                if pixels.ndim != 4 or pixels.shape[-1] != 3 \
                        or min(pixels.shape[1:3]) <= 0:
                    raise PushTSchemaError(
                        "pixels must have fixed per-step shape (height, width, 3)")
                self._require_dtype(pixels, np.uint8, "pixels")
                if action.shape != (num_rows, OFFICIAL_RAW_ACTION_DIM):
                    raise PushTSchemaError(
                        f"action shape must be ({num_rows}, 2), got {action.shape}")
                self._require_dtype(action, np.float32, "action")
                if proprio.shape != (num_rows, 4):
                    raise PushTSchemaError(
                        f"proprio shape must be ({num_rows}, 4), got {proprio.shape}")
                if state.shape != (num_rows, 7):
                    raise PushTSchemaError(
                        f"state shape must be ({num_rows}, 7), got {state.shape}")
                supported_float_dtypes = {np.dtype(np.float32), np.dtype(np.float64)}
                for key, dataset in (("proprio", proprio), ("state", state)):
                    if dataset.dtype not in supported_float_dtypes:
                        raise PushTSchemaError(
                            f"{key} dtype must be float32 or float64, got {dataset.dtype}")

                terminal_rows = offsets + lengths - 1
                self._check_finite(proprio, "proprio")
                self._check_finite(state, "state")
                self._check_finite(action, "action", terminal_rows)

                schema_record = {
                    "format": "stable-worldmodel-root-hdf5-v1",
                    "metadata": {
                        "ep_len": {"dtype": str(ep_len_ds.dtype), "rank": 1},
                        "ep_offset": {"dtype": str(ep_offset_ds.dtype), "rank": 1},
                    },
                    "required_columns": {
                        "pixels": {
                            "dtype": str(pixels.dtype),
                            "step_shape": list(pixels.shape[1:]),
                            "filter_ids": [
                                int(pixels.id.get_create_plist().get_filter(i)[0])
                                for i in range(pixels.id.get_create_plist(
                                ).get_nfilters())
                            ],
                        },
                        "action": {"dtype": str(action.dtype), "step_shape": [2]},
                        "proprio": {"dtype": str(proprio.dtype), "step_shape": [4]},
                        "state": {"dtype": str(state.dtype), "step_shape": [7]},
                    },
                }
                fingerprint = hashlib.sha256(json.dumps(
                    schema_record, sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest()
                pixel_shape = tuple(map(int, pixels.shape[1:]))
                pixel_filter_ids = tuple(schema_record[
                    "required_columns"]["pixels"]["filter_ids"])
                action_dtype = str(action.dtype)
                proprio_dtype = str(proprio.dtype)
                state_dtype = str(state.dtype)
        except OSError as exc:
            raise PushTSchemaError(f"cannot open PushT HDF5: {exc}") from exc

        return PushTHDF5Schema(
            episode_lengths=tuple(map(int, lengths)),
            episode_offsets=tuple(map(int, offsets)),
            num_rows=num_rows,
            pixel_shape=pixel_shape,
            pixel_filter_ids=pixel_filter_ids,
            action_dtype=action_dtype,
            proprio_dtype=proprio_dtype,
            state_dtype=state_dtype,
            fingerprint=fingerprint,
        )

    def select_sequences(
            self, *, num_frames: int, train_count: int,
            validation_count: int, num_classes: int, split_seed: int,
            start_seed: int, label_seed: int,
            frame_skip: int = OFFICIAL_FRAME_SKIP,
            ) -> tuple[PushTSequenceSelection, ...]:
        """Select one sequence per episode with disjoint train/validation sets."""

        values = {
            "num_frames": num_frames, "train_count": train_count,
            "validation_count": validation_count, "num_classes": num_classes,
            "split_seed": split_seed, "start_seed": start_seed,
            "label_seed": label_seed,
        }
        for key, value in values.items():
            minimum = 2 if key == "num_frames" else (1 if key in {
                "train_count", "validation_count", "num_classes"} else 0)
            if isinstance(value, bool) or not isinstance(value, int) \
                    or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if frame_skip != OFFICIAL_FRAME_SKIP:
            raise ValueError("official PushT sequences require frame_skip=5")

        raw_span = (num_frames - 1) * frame_skip + 1
        eligible = np.flatnonzero(
            np.asarray(self.schema.episode_lengths) >= raw_span)
        requested = train_count + validation_count
        if len(eligible) < requested:
            raise PushTSchemaError(
                f"only {len(eligible)} episodes can provide {num_frames} frames; "
                f"{requested} disjoint episodes requested")
        selected_episodes = np.random.default_rng(split_seed).permutation(
            eligible)[:requested]

        result: list[PushTSequenceSelection] = []
        split_specs = (
            ("train", selected_episodes[:train_count]),
            ("validation", selected_episodes[train_count:]),
        )
        for split_index, (split, episodes) in enumerate(split_specs):
            labels = np.arange(len(episodes), dtype=np.int64) % num_classes
            np.random.default_rng(np.random.SeedSequence(
                [label_seed, split_index])).shuffle(labels)
            for episode_index, label in zip(episodes, labels):
                max_start = (self.schema.episode_lengths[int(episode_index)]
                             - raw_span)
                start_rng = np.random.default_rng(np.random.SeedSequence(
                    [start_seed, int(episode_index)]))
                local_start = int(start_rng.integers(max_start + 1))
                result.append(PushTSequenceSelection(
                    split=split, episode_index=int(episode_index),
                    local_start=local_start, label=int(label)))
        return tuple(result)

    def raw_action_statistics(self, *, ddof: int = 1,
                              chunk_rows: int | None = None
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-coordinate action statistics from the pinned file.

        Entirely non-finite terminal rows are excluded.  Statistics are
        accumulated in float64 in two deterministic passes; normalization is
        intentionally performed on the raw 2-D controls before 5x2 flattening.
        Returns ``(mean, std, count)``.
        """

        if isinstance(ddof, bool) or not isinstance(ddof, int) or ddof < 0:
            raise ValueError("ddof must be a non-negative integer")
        rows = self.finite_check_chunk_rows if chunk_rows is None else chunk_rows
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError("chunk_rows must be a positive integer")
        count = np.zeros(OFFICIAL_RAW_ACTION_DIM, dtype=np.int64)
        total = np.zeros(OFFICIAL_RAW_ACTION_DIM, dtype=np.float64)
        with self._open(self.path) as handle:
            action = handle["action"]
            for start in range(0, action.shape[0], rows):
                values = np.asarray(action[start:start + rows], dtype=np.float64)
                finite = np.isfinite(values)
                count += finite.sum(axis=0, dtype=np.int64)
                total += np.where(finite, values, 0.0).sum(axis=0)
        if np.any(count <= ddof):
            raise PushTSchemaError(
                f"not enough finite raw actions for ddof={ddof}: {count.tolist()}")
        mean = total / count
        squared = np.zeros(OFFICIAL_RAW_ACTION_DIM, dtype=np.float64)
        with self._open(self.path) as handle:
            action = handle["action"]
            for start in range(0, action.shape[0], rows):
                values = np.asarray(action[start:start + rows], dtype=np.float64)
                finite = np.isfinite(values)
                delta = np.where(finite, values - mean, 0.0)
                squared += np.square(delta).sum(axis=0)
        std = np.sqrt(squared / (count - ddof))
        if not np.isfinite(mean).all() or not np.isfinite(std).all() \
                or np.any(std <= 0):
            raise PushTSchemaError("raw action statistics are non-finite or degenerate")
        for array in (mean, std, count):
            array.setflags(write=False)
        return mean, std, count

    def read_sequence(self, episode_index: int, local_start: int,
                      num_frames: int,
                      frame_skip: int = OFFICIAL_FRAME_SKIP,
                      ) -> NativePushTSequence:
        """Read strided observations and exact intervening action blocks."""

        for key, value, minimum in (
                ("episode_index", episode_index, 0),
                ("local_start", local_start, 0),
                ("num_frames", num_frames, 2)):
            if isinstance(value, bool) or not isinstance(value, int) \
                    or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if frame_skip != OFFICIAL_FRAME_SKIP:
            raise ValueError("official PushT sequences require frame_skip=5")
        if episode_index >= self.schema.num_episodes:
            raise IndexError(f"episode index {episode_index} is out of range")

        raw_span = (num_frames - 1) * frame_skip + 1
        episode_length = self.schema.episode_lengths[episode_index]
        if local_start + raw_span > episode_length:
            raise PushTSchemaError(
                "requested sequence crosses its episode boundary; padding is forbidden")
        local_frames = local_start + np.arange(
            num_frames, dtype=np.int64) * frame_skip
        episode_offset = self.schema.episode_offsets[episode_index]
        global_frames = local_frames + episode_offset
        global_actions = global_frames[:-1, None] + np.arange(
            frame_skip, dtype=np.int64)[None, :]

        with self._open(self.path) as handle:
            frames = np.asarray(handle["pixels"][global_frames]).copy()
            proprio = np.asarray(handle["proprio"][global_frames]).copy()
            state = np.asarray(handle["state"][global_frames]).copy()
            # h5py does not support this two-dimensional point-index pattern;
            # each slice is contiguous and preserves native temporal order.
            action_blocks = [np.asarray(handle["action"][
                int(indices[0]):int(indices[-1]) + 1]) for indices in global_actions]
        actions = np.stack(action_blocks).reshape(
            num_frames - 1, OFFICIAL_ACTION_BLOCK_DIM).copy()
        if not np.isfinite(actions).all():
            raise PushTSchemaError(
                "selected action block contains an episode-terminal non-finite value")

        arrays = (frames, actions, proprio, state, local_frames,
                  global_frames, global_actions)
        for array in arrays:
            array.setflags(write=False)
        return NativePushTSequence(
            frames=frames, actions=actions, proprio=proprio, state=state,
            episode_index=episode_index, local_start=local_start,
            local_frame_indices=local_frames,
            global_frame_indices=global_frames,
            global_action_indices=global_actions,
        )


__all__ = [
    "NativePushTSequence",
    "OFFICIAL_ACTION_BLOCK_DIM",
    "OFFICIAL_FRAME_SKIP",
    "OFFICIAL_PUSHT_DATASET_ARCHIVE",
    "OFFICIAL_PUSHT_EXTRACTED_HDF5",
    "OFFICIAL_RAW_ACTION_DIM",
    "OfficialPushTDatasetArchiveIdentity",
    "OfficialPushTExtractedIdentity",
    "OfficialPushTHDF5",
    "PushTHDF5Schema",
    "PushTIdentityError",
    "PushTSchemaError",
    "PushTSequenceSelection",
    "normalize_native_action_blocks",
    "sha256_file",
]
