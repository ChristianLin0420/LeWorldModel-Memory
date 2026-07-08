"""CPU-only tests for the official PushT HDF5 and memory-cache contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import h5py
import hdf5plugin
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.official_lewm_pusht import OFFICIAL_PUSHT_CHECKPOINT
from lewm.official_tasks.pusht_hdf5 import (
    OFFICIAL_PUSHT_DATASET_ARCHIVE,
    OFFICIAL_PUSHT_EXTRACTED_HDF5,
    OfficialPushTHDF5,
    PushTIdentityError,
    PushTSchemaError,
    normalize_native_action_blocks,
    sha256_file,
)
from lewm.official_tasks.pusht_memory import (
    PUSHT_MEMORY_TASKS,
    PushTLeakageError,
    PushTMemoryCacheConfig,
    build_memory_sequence,
    cache_manifest,
    cache_path,
    cache_relative_path,
    render_counterfactual_overlays,
    validate_counterfactual_no_leakage,
)


def _write_tiny_pusht(path: Path, *, episodes: int = 14,
                      episode_length: int = 31) -> None:
    """Write the upstream root-column HDF5 schema, including terminal NaNs."""

    lengths = np.full(episodes, episode_length, dtype=np.int32)
    offsets = np.arange(episodes, dtype=np.int64) * episode_length
    rows = int(lengths.sum())
    row = np.arange(rows, dtype=np.float32)
    pixels = np.empty((rows, 12, 14, 3), dtype=np.uint8)
    pixels[:] = (np.arange(rows, dtype=np.uint16) % 251).astype(
        np.uint8)[:, None, None, None]
    # A spatial pattern ensures cue overlays are tested against non-flat input.
    pixels[:, :, :, 1] ^= np.arange(14, dtype=np.uint8)[None, None, :]
    actions = np.stack((row, row + np.float32(0.25)), axis=1)
    terminal_rows = offsets + lengths.astype(np.int64) - 1
    actions[terminal_rows] = np.nan
    proprio = np.stack(tuple(row.astype(np.float64) + i
                              for i in range(4)), axis=1)
    state = np.stack(tuple(row.astype(np.float64) + i / 10
                           for i in range(7)), axis=1)
    with h5py.File(path, "w", libver="latest") as handle:
        handle.create_dataset("ep_len", data=lengths, dtype=np.int32)
        handle.create_dataset("ep_offset", data=offsets, dtype=np.int64)
        handle.create_dataset("pixels", data=pixels, dtype=np.uint8)
        handle.create_dataset("action", data=actions, dtype=np.float32)
        handle.create_dataset("proprio", data=proprio, dtype=np.float64)
        handle.create_dataset("state", data=state, dtype=np.float64)
        # Extra columns are legal because official configs choose keys_to_load;
        # the pinned whole-file hash still detects any change to them.
        handle.create_dataset("unused_reward", data=np.zeros(rows, np.float32))


def _open_fixture(path: Path) -> OfficialPushTHDF5:
    return OfficialPushTHDF5(
        path, expected_hdf5_sha256=sha256_file(path),
        finite_check_chunk_rows=7)


def _config_mapping(cache_root: Path, hdf5_sha: str,
                    task_name: str = "PushT transient visual-token recall",
                    hdf5_size: int = OFFICIAL_PUSHT_EXTRACTED_HDF5.size,
                    ) -> dict:
    archive = OFFICIAL_PUSHT_DATASET_ARCHIVE
    checkpoint = OFFICIAL_PUSHT_CHECKPOINT
    return {
        "schema_version": 1,
        "cache_root": str(cache_root),
        "semantic_task_name": task_name,
        "dataset": {
            "repo_id": archive.repo_id,
            "revision": archive.revision,
            "filename": archive.filename,
            "archive_sha256": archive.sha256,
            "archive_size": archive.size,
            "file_commit": archive.file_commit,
            "extracted_hdf5_sha256": hdf5_sha,
            "extracted_hdf5_size": hdf5_size,
        },
        "checkpoint": {
            "repo_id": checkpoint.repo_id,
            "revision": checkpoint.revision,
            "config_sha256": checkpoint.config_sha256,
            "weights_sha256": checkpoint.weights_sha256,
            "weights_size": checkpoint.weights_size,
        },
        "sequence": {
            "frame_skip": 5,
            "num_frames": 5,
            "cue_start": 0,
            "cue_length": 1,
        },
        "selection": {
            "train_count": 6,
            "validation_count": 6,
            "split_seed": 4101,
            "start_seed": 4102,
            "label_seed": 4103,
        },
    }


def test_valid_schema_and_native_action_blocks(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _write_tiny_pusht(path)
    dataset = _open_fixture(path)
    assert dataset.schema.num_episodes == 14
    assert dataset.schema.num_rows == 14 * 31
    assert dataset.schema.pixel_shape == (12, 14, 3)
    assert dataset.schema.pixel_filter_ids == ()
    assert len(dataset.schema.fingerprint) == 64

    sequence = dataset.read_sequence(episode_index=2, local_start=3,
                                     num_frames=4)
    np.testing.assert_array_equal(sequence.local_frame_indices,
                                  np.asarray([3, 8, 13, 18]))
    assert sequence.frames.shape == (4, 12, 14, 3)
    assert sequence.actions.shape == (3, 10)
    assert len(np.unique(sequence.global_frame_indices)) == 4
    np.testing.assert_array_equal(np.diff(sequence.global_frame_indices), 5)
    offset = dataset.schema.episode_offsets[2]
    expected = np.stack([
        np.stack((np.arange(offset + start, offset + start + 5,
                            dtype=np.float32),
                  np.arange(offset + start, offset + start + 5,
                            dtype=np.float32) + np.float32(0.25)), axis=1
        ).reshape(10)
        for start in (3, 8, 13)
    ])
    np.testing.assert_array_equal(sequence.actions, expected)
    assert not sequence.frames.flags.writeable
    assert not sequence.actions.flags.writeable


def test_blosc_filter_32001_pixels_decode_like_official_file(
        tmp_path: Path) -> None:
    path = tmp_path / "tiny-blosc.h5"
    _write_tiny_pusht(path, episodes=1, episode_length=21)
    with h5py.File(path, "r+") as handle:
        pixels = handle["pixels"][:]
        del handle["pixels"]
        handle.create_dataset(
            "pixels", data=pixels, chunks=(5, *pixels.shape[1:]),
            **hdf5plugin.Blosc(
                cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
        assert handle["pixels"].id.get_create_plist().get_filter(0)[0] == 32001
    dataset = _open_fixture(path)
    assert dataset.schema.pixel_filter_ids == (32001,)
    sequence = dataset.read_sequence(0, 0, num_frames=5)
    np.testing.assert_array_equal(sequence.frames, pixels[[0, 5, 10, 15, 20]])


def test_deterministic_episode_disjoint_balanced_selection(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _write_tiny_pusht(path)
    dataset = _open_fixture(path)
    kwargs = dict(num_frames=5, train_count=6, validation_count=6,
                  num_classes=3, split_seed=19, start_seed=23,
                  label_seed=29)
    first = dataset.select_sequences(**kwargs)
    second = dataset.select_sequences(**kwargs)
    assert first == second
    train = [item for item in first if item.split == "train"]
    validation = [item for item in first if item.split == "validation"]
    assert {item.episode_index for item in train}.isdisjoint(
        {item.episode_index for item in validation})
    for split in (train, validation):
        counts = np.bincount([item.label for item in split], minlength=3)
        assert counts.max() - counts.min() <= 1
        for item in split:
            sequence = dataset.read_sequence(
                item.episode_index, item.local_start, num_frames=5)
            assert np.all(np.diff(sequence.local_frame_indices) == 5)
            assert np.isfinite(sequence.actions).all()


def test_raw_ddof1_statistics_then_normalize_before_flatten(
        tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _write_tiny_pusht(path, episodes=3, episode_length=21)
    dataset = _open_fixture(path)
    mean, std, count = dataset.raw_action_statistics(ddof=1, chunk_rows=4)
    with h5py.File(path, "r") as handle:
        raw = handle["action"][:]
    expected_mean = np.nanmean(raw.astype(np.float64), axis=0)
    expected_std = np.nanstd(raw.astype(np.float64), axis=0, ddof=1)
    np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=1e-13)
    np.testing.assert_allclose(std, expected_std, rtol=0, atol=1e-13)
    np.testing.assert_array_equal(count, np.asarray([60, 60]))

    native = dataset.read_sequence(0, 0, num_frames=3)
    normalized = normalize_native_action_blocks(native.actions, mean, std)
    expected = ((native.actions.reshape(2, 5, 2).astype(np.float64)
                 - mean) / std).reshape(2, 10).astype(np.float32)
    np.testing.assert_array_equal(normalized, expected)


def test_boundary_crossing_is_rejected_without_padding(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _write_tiny_pusht(path, episodes=1, episode_length=21)
    dataset = _open_fixture(path)
    terminal_sequence = dataset.read_sequence(0, 0, num_frames=5)
    assert terminal_sequence.local_frame_indices[-1] == 20
    assert np.isfinite(terminal_sequence.actions).all()
    with pytest.raises(PushTSchemaError, match="crosses.*boundary"):
        dataset.read_sequence(0, 1, num_frames=5)


def _delete_action(handle: h5py.File) -> None:
    del handle["action"]


def _wrong_action_shape(handle: h5py.File) -> None:
    rows = handle["action"].shape[0]
    del handle["action"]
    handle.create_dataset("action", data=np.zeros((rows, 3), np.float32))


def _wrong_action_dtype(handle: h5py.File) -> None:
    values = handle["action"][:].astype(np.float64)
    del handle["action"]
    handle.create_dataset("action", data=values)


def _wrong_episode_dtype(handle: h5py.File) -> None:
    values = handle["ep_len"][:].astype(np.int64)
    del handle["ep_len"]
    handle.create_dataset("ep_len", data=values)


def _wrong_pixel_layout(handle: h5py.File) -> None:
    values = handle["pixels"][:].transpose(0, 3, 1, 2)
    del handle["pixels"]
    handle.create_dataset("pixels", data=values)


def _integer_proprio(handle: h5py.File) -> None:
    values = handle["proprio"][:].astype(np.int32)
    del handle["proprio"]
    handle.create_dataset("proprio", data=values)


def _bad_offset(handle: h5py.File) -> None:
    handle["ep_offset"][1] += 1


def _nonterminal_nan(handle: h5py.File) -> None:
    handle["action"][1, 0] = np.nan


def _row_count_mismatch(handle: h5py.File) -> None:
    values = handle["state"][:-1]
    del handle["state"]
    handle.create_dataset("state", data=values)


@pytest.mark.parametrize(("mutation", "message"), [
    (_delete_action, "action.*missing"),
    (_wrong_action_shape, "action shape"),
    (_wrong_action_dtype, "action dtype"),
    (_wrong_episode_dtype, "ep_len dtype"),
    (_wrong_pixel_layout, "pixels must have.*height, width, 3"),
    (_integer_proprio, "proprio dtype"),
    (_bad_offset, "no gaps or overlaps"),
    (_nonterminal_nan, "partially non-finite"),
    (_row_count_mismatch, "state has.*rows"),
])
def test_schema_drift_fails_closed(tmp_path: Path, mutation, message: str) -> None:
    path = tmp_path / "drift.h5"
    _write_tiny_pusht(path)
    with h5py.File(path, "r+") as handle:
        mutation(handle)
    with pytest.raises(PushTSchemaError, match=message):
        _open_fixture(path)


def test_wrong_extracted_identity_fails_before_schema_use(tmp_path: Path) -> None:
    path = tmp_path / "tiny.h5"
    _write_tiny_pusht(path)
    wrong = hashlib.sha256(b"not this file").hexdigest()
    with pytest.raises(PushTIdentityError, match="mismatch"):
        OfficialPushTHDF5(path, expected_hdf5_sha256=wrong)


@pytest.mark.parametrize("task", PUSHT_MEMORY_TASKS,
                         ids=lambda task: task.semantic_name)
def test_semantic_overlays_are_cue_only(tmp_path: Path, task) -> None:
    path = tmp_path / "tiny.h5"
    _write_tiny_pusht(path)
    native = _open_fixture(path).read_sequence(0, 0, num_frames=5)
    variants = render_counterfactual_overlays(
        native.frames, task.semantic_name, cue_start=1, cue_length=2)
    assert variants.shape[0] == task.num_classes
    validate_counterfactual_no_leakage(variants, 1, 3)
    for label in range(task.num_classes):
        np.testing.assert_array_equal(variants[label, 0], native.frames[0])
        np.testing.assert_array_equal(variants[label, 3:], native.frames[3:])

    leaked = variants.copy()
    leaked[1, 4, 0, 0, 0] ^= np.uint8(1)
    with pytest.raises(PushTLeakageError, match="after"):
        validate_counterfactual_no_leakage(leaked, 1, 3)


def test_pinned_config_cache_and_manifest_create_no_outputs(
        tmp_path: Path) -> None:
    hdf5_path = tmp_path / "tiny.h5"
    _write_tiny_pusht(hdf5_path)
    dataset = _open_fixture(hdf5_path)
    output_root = tmp_path / "must-not-be-created"
    mapping = _config_mapping(
        output_root, dataset.expected_hdf5_sha256,
        hdf5_size=dataset.hdf5_size)
    config_path = tmp_path / "cache.json"
    config_path.write_text(json.dumps(mapping))
    config = PushTMemoryCacheConfig.from_json(config_path)
    assert not output_root.exists()
    relative = cache_relative_path(config)
    assert not relative.is_absolute()
    assert "transient-visual-token-recall" in str(relative)
    assert cache_path(config) == output_root / relative
    assert not output_root.exists()

    selections = config.select(dataset)
    first = selections[0]
    native = dataset.read_sequence(
        first.episode_index, first.local_start, config.num_frames)
    memory = build_memory_sequence(native, first, config)
    assert memory.semantic_task_name == config.semantic_task_name
    assert memory.frames.shape == native.frames.shape
    np.testing.assert_array_equal(
        memory.frames[config.cue_end:], native.frames[config.cue_end:])
    np.testing.assert_array_equal(memory.native.actions, native.actions)
    manifest = cache_manifest(config, dataset, selections)
    assert manifest["dataset"]["archive"]["sha256"] \
        == OFFICIAL_PUSHT_DATASET_ARCHIVE.sha256
    assert manifest["checkpoint"]["weights_sha256"] \
        == OFFICIAL_PUSHT_CHECKPOINT.weights_sha256
    assert manifest["sequence"]["native_action_block_dim"] == 10
    assert not output_root.exists()


@pytest.mark.parametrize(("mutator", "message"), [
    (lambda value: value["dataset"].__setitem__(
        "archive_sha256", "0" * 64), "dataset.archive_sha256"),
    (lambda value: value["checkpoint"].__setitem__(
        "weights_sha256", "0" * 64), "checkpoint.weights_sha256"),
    (lambda value: value.__setitem__("semantic_task_name", "T1"),
     "unknown semantic"),
    (lambda value: value["sequence"].__setitem__("frame_skip", 1),
     "frame_skip must be 5"),
    (lambda value: value["dataset"].__setitem__(
        "extracted_hdf5_size", 0), "extracted_hdf5_size"),
])
def test_cache_config_rejects_identity_or_contract_drift(
        tmp_path: Path, mutator, message: str) -> None:
    mapping = _config_mapping(tmp_path / "cache", "a" * 64)
    mutator(mapping)
    with pytest.raises((PushTIdentityError, ValueError), match=message):
        PushTMemoryCacheConfig.from_mapping(mapping)
