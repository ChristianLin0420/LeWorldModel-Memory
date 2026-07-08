"""Focused tests for the released-LeWM latent cache contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cache_official_lewm import (
    OFFICIAL_ACTION_DIM,
    OFFICIAL_EMBED_DIM,
    action_statistics,
    categorical_availability,
    continuous_availability,
    encode_frames,
    sha256_file,
    transform_actions,
    write_npz_deterministic,
)


def test_native_and_legacy_action_contracts() -> None:
    native = np.arange(4 * 3 * 10, dtype=np.float32).reshape(4, 3, 10)
    mean, std = action_statistics(native)
    transformed = transform_actions(native, mean, std, "clean")
    assert transformed.shape == native.shape
    np.testing.assert_allclose(
        transformed.reshape(-1, 10).mean(axis=0), 0.0, atol=2e-7)
    np.testing.assert_allclose(
        transformed.reshape(-1, 10).std(axis=0), 1.0, atol=2e-7)

    legacy = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    legacy_mean, legacy_std = action_statistics(legacy)
    normalized = ((legacy - legacy_mean) / legacy_std).astype(np.float32)
    expanded = transform_actions(
        legacy, legacy_mean, legacy_std, "observed")
    assert expanded.shape == (4, 3, OFFICIAL_ACTION_DIM)
    np.testing.assert_array_equal(expanded, np.tile(normalized, (1, 1, 5)))

    with pytest.raises(ValueError, match="native 10-D"):
        transform_actions(legacy, legacy_mean, legacy_std, "clean")
    with pytest.raises(ValueError, match="expects 2-D"):
        transform_actions(native, mean, std, "observed")


class _MeanEncoder(torch.nn.Module):
    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        value = pixels.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        return value.repeat(1, OFFICIAL_EMBED_DIM)


def test_fixed_chunk_encoding_is_chunk_size_invariant() -> None:
    frames = np.arange(2 * 3 * 8 * 8 * 3, dtype=np.uint8).reshape(
        2, 3, 8, 8, 3)
    model = _MeanEncoder().eval()
    first = encode_frames(model, frames, torch.device("cpu"), 2)
    second = encode_frames(model, frames, torch.device("cpu"), 5)
    assert first.shape == (2, 3, OFFICIAL_EMBED_DIM)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-6)


def test_availability_probe_families_read_linear_signal() -> None:
    rng = np.random.default_rng(4)
    train_y = np.tile(np.arange(4, dtype=np.int64), 20)
    val_y = np.tile(np.arange(4, dtype=np.int64), 8)
    train_x = rng.normal(scale=0.01, size=(len(train_y), 12)).astype(np.float32)
    val_x = rng.normal(scale=0.01, size=(len(val_y), 12)).astype(np.float32)
    train_x[np.arange(len(train_y)), train_y] += 4.0
    val_x[np.arange(len(val_y)), val_y] += 4.0
    categorical = categorical_availability(
        train_x, train_y, val_x, val_y)
    assert categorical["value"] == 1.0
    assert categorical["chance"] == 0.25

    train_x = rng.normal(size=(100, 16)).astype(np.float32)
    val_x = rng.normal(size=(40, 16)).astype(np.float32)
    weights = rng.normal(size=(16, 2))
    train_y = (train_x @ weights).astype(np.float32)
    val_y = (val_x @ weights).astype(np.float32)
    continuous = continuous_availability(
        train_x, train_y, val_x, val_y)
    assert continuous["value"] > 0.999
    assert len(continuous["per_target"]) == 2


def test_npz_writer_is_byte_stable_and_pickle_free(tmp_path: Path) -> None:
    metadata = {"schema": "test", "label_training": False}
    arrays = {
        "z": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "actions": np.zeros((2, 2, 10), dtype=np.float32),
        "xi": np.asarray([0, 1], dtype=np.int64),
        "meta_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    first_hash = write_npz_deterministic(first, arrays)
    second_hash = write_npz_deterministic(second, arrays)
    assert first_hash == second_hash == sha256_file(first)
    with np.load(first, allow_pickle=False) as loaded:
        assert loaded.files == list(arrays)
        assert json.loads(str(loaded["meta_json"])) == metadata
        np.testing.assert_array_equal(loaded["z"], arrays["z"])

