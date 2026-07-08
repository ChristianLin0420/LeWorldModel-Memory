"""Identity-pinned reader for official LeWM root-column HDF5 datasets.

This narrow reader covers the common pixel/action/state contract needed by
the matched-host audit.  It deliberately does not emulate the upstream
training dataset or permit boundary padding.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from collections.abc import Iterable, Iterator

import h5py
import hdf5plugin  # noqa: F401 - registers the official Blosc filter
import numpy as np


class NativeSequenceError(ValueError):
    """A native LeWM dataset violates the registered sequence contract."""


@dataclass(frozen=True)
class SequenceSelection:
    split: str
    episode_index: int
    local_start: int


@dataclass(frozen=True)
class NativeSequence:
    frames: np.ndarray
    actions: np.ndarray
    state: np.ndarray
    episode_index: int
    local_start: int
    global_frame_indices: np.ndarray


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeSequenceHDF5:
    """Validate and read 5-step/2-D-action sequences from a pinned HDF5."""

    frame_skip = 5
    raw_action_dim = 2
    action_block_dim = 10

    def __init__(self, path: str | Path, *, expected_sha256: str,
                 expected_size: int, state_key: str = "state") -> None:
        self.path = Path(path)
        if not self.path.is_file() or self.path.stat().st_size != expected_size:
            raise NativeSequenceError(f"dataset file identity mismatch: {self.path}")
        if sha256_file(self.path) != expected_sha256:
            raise NativeSequenceError(f"dataset SHA-256 mismatch: {self.path}")
        self.expected_sha256 = expected_sha256
        self.expected_size = expected_size
        if not isinstance(state_key, str) or not state_key:
            raise NativeSequenceError("state_key must be a non-empty string")
        self.state_key = state_key
        self._validate()

    def _open(self) -> h5py.File:
        return h5py.File(self.path, "r", swmr=True,
                         rdcc_nbytes=256 * 1024 * 1024)

    def _validate(self) -> None:
        try:
            with self._open() as handle:
                required = {"ep_len", "ep_offset", "pixels", "action",
                            self.state_key}
                missing = sorted(required - set(handle))
                if missing:
                    raise NativeSequenceError(
                        f"native HDF5 is missing root columns: {missing}")
                lengths = np.asarray(handle["ep_len"][:], dtype=np.int64)
                offsets = np.asarray(handle["ep_offset"][:], dtype=np.int64)
                if lengths.ndim != 1 or offsets.shape != lengths.shape \
                        or not len(lengths) or np.any(lengths <= 0):
                    raise NativeSequenceError("episode metadata is invalid")
                expected_offsets = np.concatenate((
                    np.asarray([0], dtype=np.int64),
                    np.cumsum(lengths[:-1], dtype=np.int64)))
                if not np.array_equal(offsets, expected_offsets):
                    raise NativeSequenceError("episode offsets have gaps or overlaps")
                rows = int(lengths.sum())
                pixels, action, state = (
                    handle["pixels"], handle["action"], handle[self.state_key])
                if pixels.shape[0] != rows or action.shape[0] != rows \
                        or state.shape[0] != rows:
                    raise NativeSequenceError("root columns disagree on row count")
                if pixels.dtype != np.uint8 or pixels.ndim != 4 \
                        or pixels.shape[-1] != 3:
                    raise NativeSequenceError("pixels must be uint8 NHWC RGB")
                if action.dtype != np.float32 \
                        or action.shape != (rows, self.raw_action_dim):
                    raise NativeSequenceError("actions must be float32 Nx2")
                if state.ndim != 2 or state.shape[1] <= 0 \
                        or state.dtype not in (np.float32, np.float64):
                    raise NativeSequenceError("state must be a finite numeric matrix")
                self.episode_lengths = lengths
                self.episode_offsets = offsets
                self.num_rows = rows
                self.num_episodes = len(lengths)
                self.pixel_shape = tuple(map(int, pixels.shape[1:]))
                self.state_dim = int(state.shape[1])
                self.pixel_filters = tuple(
                    int(pixels.id.get_create_plist().get_filter(index)[0])
                    for index in range(
                        pixels.id.get_create_plist().get_nfilters()))
        except OSError as error:
            raise NativeSequenceError(f"cannot open native HDF5: {error}") from error

    def select_sequences(self, *, num_frames: int,
                         split_counts: tuple[tuple[str, int], ...],
                         split_seed: int, start_seed: int
                         ) -> tuple[SequenceSelection, ...]:
        if num_frames < 2 or any(count <= 0 for _, count in split_counts):
            raise NativeSequenceError("invalid selection size")
        names = tuple(name for name, _ in split_counts)
        if len(set(names)) != len(names):
            raise NativeSequenceError("split names must be unique")
        raw_span = (num_frames - 1) * self.frame_skip + 1
        eligible = np.flatnonzero(self.episode_lengths >= raw_span)
        requested = sum(count for _, count in split_counts)
        if len(eligible) < requested:
            raise NativeSequenceError(
                f"only {len(eligible)} eligible episodes for {requested} requests")
        episodes = np.random.default_rng(split_seed).permutation(
            eligible)[:requested]
        result: list[SequenceSelection] = []
        cursor = 0
        for split, count in split_counts:
            for episode in episodes[cursor:cursor + count]:
                maximum = int(self.episode_lengths[int(episode)] - raw_span)
                rng = np.random.default_rng(np.random.SeedSequence(
                    [start_seed, int(episode)]))
                result.append(SequenceSelection(
                    split=split, episode_index=int(episode),
                    local_start=int(rng.integers(maximum + 1))))
            cursor += count
        return tuple(result)

    def raw_action_statistics(self, *, ddof: int = 1,
                              chunk_rows: int = 65_536
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = np.zeros(self.raw_action_dim, dtype=np.int64)
        total = np.zeros(self.raw_action_dim, dtype=np.float64)
        with self._open() as handle:
            action = handle["action"]
            for start in range(0, len(action), chunk_rows):
                values = np.asarray(action[start:start + chunk_rows],
                                    dtype=np.float64)
                complete = np.isfinite(values).all(axis=1)
                terminal = np.isnan(values).all(axis=1)
                if not np.all(complete | terminal):
                    raise NativeSequenceError(
                        "actions contain partial NaNs or infinite values")
                finite = np.isfinite(values)
                count += finite.sum(axis=0)
                total += np.where(finite, values, 0.0).sum(axis=0)
        if np.any(count <= ddof):
            raise NativeSequenceError("insufficient finite actions")
        mean = total / count
        squared = np.zeros(self.raw_action_dim, dtype=np.float64)
        with self._open() as handle:
            action = handle["action"]
            for start in range(0, len(action), chunk_rows):
                values = np.asarray(action[start:start + chunk_rows],
                                    dtype=np.float64)
                delta = np.where(np.isfinite(values), values - mean, 0.0)
                squared += np.square(delta).sum(axis=0)
        std = np.sqrt(squared / (count - ddof))
        if not np.isfinite(mean).all() or not np.isfinite(std).all() \
                or np.any(std <= 0):
            raise NativeSequenceError("action normalization is degenerate")
        return mean, std, count

    def _read_sequence_from_handle(
            self, handle: h5py.File, selection: SequenceSelection,
            num_frames: int) -> NativeSequence:
        episode = selection.episode_index
        start = selection.local_start
        raw_span = (num_frames - 1) * self.frame_skip + 1
        if episode < 0 or episode >= self.num_episodes \
                or start < 0 \
                or start + raw_span > self.episode_lengths[episode]:
            raise NativeSequenceError("sequence crosses an episode boundary")
        local_frames = start + np.arange(num_frames) * self.frame_skip
        global_frames = local_frames + self.episode_offsets[episode]
        global_actions = global_frames[:-1, None] + np.arange(
            self.frame_skip)[None, :]
        frames = np.asarray(handle["pixels"][global_frames]).copy()
        state = np.asarray(handle[self.state_key][global_frames]).copy()
        action_parts = [np.asarray(handle["action"][
            int(indices[0]):int(indices[-1]) + 1])
            for indices in global_actions]
        actions = np.stack(action_parts).reshape(
            num_frames - 1, self.action_block_dim).astype(np.float32, copy=False)
        if not np.isfinite(actions).all() or not np.isfinite(state).all():
            raise NativeSequenceError("selected sequence is not finite")
        arrays = (frames, actions, state, global_frames)
        for value in arrays:
            value.setflags(write=False)
        return NativeSequence(
            frames=frames, actions=actions, state=state,
            episode_index=episode, local_start=start,
            global_frame_indices=global_frames.astype(np.int64, copy=False))

    def read_sequence(self, selection: SequenceSelection,
                      num_frames: int) -> NativeSequence:
        with self._open() as handle:
            return self._read_sequence_from_handle(
                handle, selection, num_frames)

    def read_sequences(self, selections: Iterable[SequenceSelection],
                       num_frames: int) -> Iterator[NativeSequence]:
        """Read a deck through one verified HDF5 handle.

        The yielded arrays own their bytes, so callers may retain them after
        iteration.  Keeping one handle open avoids thousands of expensive
        open/close cycles on the official compressed archives.
        """

        with self._open() as handle:
            for selection in selections:
                yield self._read_sequence_from_handle(
                    handle, selection, num_frames)


def normalize_action_blocks(actions: np.ndarray, mean: np.ndarray,
                            std: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if values.shape[-1] != 10 or mean.shape != (2,) or std.shape != (2,) \
            or np.any(std <= 0):
        raise NativeSequenceError("action normalization shapes are invalid")
    raw = values.reshape(*values.shape[:-1], 5, 2)
    return ((raw - mean) / std).reshape(values.shape).astype(np.float32)


__all__ = [
    "NativeSequence", "NativeSequenceError", "NativeSequenceHDF5",
    "SequenceSelection", "normalize_action_blocks", "sha256_file",
]
