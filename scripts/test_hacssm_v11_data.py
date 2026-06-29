#!/usr/bin/env python3
"""Focused cache, corruption, IID-action, and real-DMC tests for V11 data."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.hacssm_v11_data import (
    ACTION_PROCESS,
    DEFAULT_CORRUPTION_SEED,
    DEFAULT_ROOT,
    DEFAULT_SMOOTH_RHO,
    DEFAULT_TRAIN_SEED,
    DEFAULT_VAL_SEED,
    SCHEMA_VERSION,
    TASK_SCHEMA_FIELDS,
    VIEWS,
    V11TrajectoryDataset,
    collect_clean_dmc,
    content_sha256,
    decode_observation_schema,
    iid_bounded_action,
    load_cache,
    observation_schema_arrays,
    parse_args,
    sidecar_path,
    write_cache,
)


def _arrays(episodes: int = 4, length: int = 24, size: int = 16,
            action_dim: int = 2, state_dim: int = 5,
            seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    schema = observation_schema_arrays(
        ("position", "orientation", "speed"), ((2,), (2, 2), ()))
    task_dim = 2 + 4 + 1
    return {
        "obs": rng.integers(
            0, 256, (episodes, length, size, size, 3), dtype=np.uint8),
        "actions": rng.uniform(
            -0.95, 0.95, (episodes, length - 1, action_dim)).astype(np.float32),
        "task_observation": rng.normal(
            size=(episodes, length, task_dim)).astype(np.float64),
        **schema,
        "physics_state": rng.normal(
            size=(episodes, length, state_dim)).astype(np.float64),
        "rewards": rng.normal(size=(episodes, length - 1)).astype(np.float32),
        "action_min": np.full(action_dim, -1, dtype=np.float32),
        "action_max": np.full(action_dim, 1, dtype=np.float32),
    }


def _write_tiny(root: Path, arrays: dict[str, np.ndarray] | None = None) -> Path:
    path = root / "tiny_train.npz"
    write_cache(
        path, env_id="walker.walk", split="train", seed=DEFAULT_TRAIN_SEED,
        length=24, img_size=16, smooth_rho=0.0,
        arrays=_arrays() if arrays is None else arrays)
    return path


def test_defaults_and_iid_action_contract() -> None:
    args = parse_args([])
    assert args.root == DEFAULT_ROOT
    assert args.train_seed == 37_100 == DEFAULT_TRAIN_SEED
    assert args.val_seed == 103_710 == DEFAULT_VAL_SEED
    assert args.smooth_rho == 0.0 == DEFAULT_SMOOTH_RHO
    assert DEFAULT_CORRUPTION_SEED == 11_012
    assert ACTION_PROCESS == "bounded_tanh_iid_gaussian"

    action_min = np.asarray((-2.0, -0.5), dtype=np.float32)
    action_max = np.asarray((2.0, 1.5), dtype=np.float32)
    actual_rng = np.random.default_rng(123)
    expected_rng = np.random.default_rng(123)
    actual = iid_bounded_action(actual_rng, action_min, action_max)
    latent = expected_rng.standard_normal(action_min.shape).astype(np.float32)
    expected = (action_min + action_max) * 0.5
    expected += (action_max - action_min) * 0.5 * np.tanh(latent)
    assert np.array_equal(actual, expected.astype(np.float32))
    # The next action consumes the next independent Gaussian draw, not the previous action.
    second = iid_bounded_action(actual_rng, action_min, action_max)
    latent = expected_rng.standard_normal(action_min.shape).astype(np.float32)
    expected = ((action_min + action_max) * 0.5
                + (action_max - action_min) * 0.5 * np.tanh(latent))
    assert np.array_equal(second, expected.astype(np.float32))


def test_cache_schema_roundtrip_and_corruptions() -> None:
    with tempfile.TemporaryDirectory() as directory:
        arrays = _arrays()
        path = _write_tiny(Path(directory), arrays)
        metadata = load_cache(path)
        assert metadata.episodes == 4 and metadata.length == 24
        assert metadata.action_dim == 2 and metadata.state_dim == 5
        assert metadata.task_observation_dim == 7
        assert metadata.task_observation_keys == ("position", "orientation", "speed")
        assert metadata.task_observation_shapes == ((2,), (2, 2), ())
        assert metadata.smooth_rho == 0.0
        assert sidecar_path(path).is_file()

        with np.load(path, allow_pickle=False) as source:
            assert int(source["schema_version"]) == SCHEMA_VERSION
            assert str(source["action_process"]) == ACTION_PROCESS
            assert str(source["content_sha256"]) == content_sha256(
                {name: source[name] for name in source.files})
            for name in TASK_SCHEMA_FIELDS:
                assert source[name].dtype.kind != "O"

        clean = V11TrajectoryDataset(path, "clean")
        assert torch.equal(clean[0]["task_observation"], torch.from_numpy(
            arrays["task_observation"][0].astype(np.float32)))
        assert clean[0]["task_observation"].shape == (24, 7)
        assert clean[0]["physics_state"].shape == (24, 5)
        for view in VIEWS:
            left = V11TrajectoryDataset(path, view)[0]
            right = V11TrajectoryDataset(path, view)[0]
            assert torch.equal(left["observed"], right["observed"])
            assert torch.equal(left["clean"], right["clean"])
            assert torch.equal(left["task_observation"], right["task_observation"])
            assert int(left["gap_start"]) < int(left["gap_end"])
            if view == "clean":
                assert torch.equal(left["observed"], left["clean"])
                assert not bool(left["corruption_mask"].any())
            else:
                assert bool(left["corruption_mask"].any())
                assert not torch.equal(left["observed"], left["clean"])


def test_validation_rejects_shortcuts_and_malformed_payloads() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        arrays = _arrays()
        arrays["actions"][0, 0, 0] = 2.0
        try:
            _write_tiny(root, arrays)
        except ValueError as error:
            assert "action outside native bounds" in str(error)
        else:
            raise AssertionError("out-of-bounds V11 action was accepted")
        assert not (root / "tiny_train.npz").exists()

        arrays = _arrays()
        arrays["task_observation_slices"][-1, 1] += 1
        try:
            _write_tiny(root, arrays)
        except ValueError as error:
            assert "task-observation slices" in str(error)
        else:
            raise AssertionError("malformed V11 task-observation schema was accepted")

        try:
            write_cache(
                root / "rho.npz", env_id="walker.walk", split="train", seed=1,
                length=24, img_size=16, smooth_rho=0.85, arrays=_arrays())
        except ValueError as error:
            assert "smooth_rho=0.0" in str(error)
        else:
            raise AssertionError("smooth-action V11 cache was accepted")


def test_real_fish_dmc_smoke_includes_task_target() -> None:
    arrays = collect_clean_dmc(
        "fish.swim", episodes=1, length=6, img_size=16, seed=4242,
        smooth_rho=0.0)
    keys, shapes, task_dim = decode_observation_schema(arrays, label="fish smoke")
    assert keys == ("joint_angles", "upright", "target", "velocity")
    assert shapes == ((7,), (), (3,), (13,))
    assert task_dim == 24
    assert arrays["task_observation"].shape == (1, 6, 24)
    assert arrays["physics_state"].shape == (1, 6, 27)
    assert arrays["actions"].shape == (1, 5, 5)
    assert np.isfinite(arrays["task_observation"]).all()
    assert np.all(arrays["actions"] >= arrays["action_min"])
    assert np.all(arrays["actions"] <= arrays["action_max"])
    # The task target is explicitly represented in flattened columns 8:11.
    target_slice = arrays["task_observation_slices"][keys.index("target")]
    assert tuple(target_slice) == (8, 11)


def main() -> None:
    test_defaults_and_iid_action_contract()
    test_cache_schema_roundtrip_and_corruptions()
    test_validation_rejects_shortcuts_and_malformed_payloads()
    test_real_fish_dmc_smoke_includes_task_target()
    print("V11 data tests passed.")


if __name__ == "__main__":
    main()
