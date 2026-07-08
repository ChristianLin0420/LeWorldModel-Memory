from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np

from lewm.official_tasks.native_sequence_hdf5 import (
    NativeSequenceHDF5,
    normalize_action_blocks,
)


def _fixture(path: Path) -> str:
    lengths = np.asarray([101, 106, 111, 116], dtype=np.int32)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64),
                              np.cumsum(lengths[:-1], dtype=np.int64)))
    rows = int(lengths.sum())
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_len", data=lengths)
        handle.create_dataset("ep_offset", data=offsets)
        handle.create_dataset("pixels", data=np.zeros(
            (rows, 16, 16, 3), dtype=np.uint8))
        action = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
        handle.create_dataset("action", data=action)
        handle.create_dataset("state", data=np.arange(
            rows * 3, dtype=np.float32).reshape(rows, 3))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_reader_selects_disjoint_boundary_safe_sequences(tmp_path) -> None:
    path = tmp_path / "native.h5"
    digest = _fixture(path)
    data = NativeSequenceHDF5(
        path, expected_sha256=digest, expected_size=path.stat().st_size)
    selected = data.select_sequences(
        num_frames=20, split_counts=(("train", 2), ("validation", 2)),
        split_seed=3, start_seed=4)
    assert len({item.episode_index for item in selected}) == 4
    sequence = data.read_sequence(selected[0], 20)
    streamed = tuple(data.read_sequences(selected[:2], 20))
    assert sequence.frames.shape == (20, 16, 16, 3)
    assert sequence.actions.shape == (19, 10)
    assert sequence.state.shape == (20, 3)
    assert np.all(np.diff(sequence.global_frame_indices) == 5)
    assert len(streamed) == 2
    assert np.array_equal(streamed[0].actions, sequence.actions)


def test_action_normalization_precedes_flattening() -> None:
    actions = np.arange(20, dtype=np.float32).reshape(2, 10)
    value = normalize_action_blocks(
        actions, np.asarray([1.0, 2.0]), np.asarray([2.0, 4.0]))
    expected = ((actions.reshape(2, 5, 2) - np.asarray([1.0, 2.0]))
                / np.asarray([2.0, 4.0])).reshape(2, 10)
    assert np.allclose(value, expected)
