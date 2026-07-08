from __future__ import annotations

import numpy as np

from scripts.run_dinowm_native_pusht_audit_v2 import OfficialDinoWMPushT


def test_native_audit_split_is_episode_disjoint_and_balanced() -> None:
    dataset = object.__new__(OfficialDinoWMPushT)
    dataset.splits = {
        "train": {"lengths": tuple([120] * 40)},
        "val": {"lengths": tuple([120] * 8)},
    }
    selections = dataset.select(
        train_count=18, validation_count=12,
        num_frames=20, frame_skip=5, classes=6,
        split_seed=11, start_seed=12, label_seed=13,
        source_split="train")
    train = [item for item in selections if item.split == "train"]
    validation = [item for item in selections if item.split == "validation"]
    assert len(train) == 18
    assert len(validation) == 12
    assert {item.episode_index for item in train}.isdisjoint(
        {item.episode_index for item in validation})
    assert all(item.source_split == "train" for item in selections)
    assert np.array_equal(
        np.bincount([item.label for item in train], minlength=6),
        np.full(6, 3))
    assert np.array_equal(
        np.bincount([item.label for item in validation], minlength=6),
        np.full(6, 2))
    assert all(0 <= item.local_start <= 24 for item in selections)


def test_short_native_official_val_pool_fails_closed() -> None:
    dataset = object.__new__(OfficialDinoWMPushT)
    dataset.splits = {
        "train": {"lengths": tuple([120] * 20)},
        "val": {"lengths": tuple([120] * 4)},
    }
    try:
        dataset.select(
            train_count=3, validation_count=3,
            num_frames=20, frame_skip=5, classes=3,
            split_seed=1, start_seed=2, label_seed=3,
            source_split="val")
    except RuntimeError as error:
        assert "eligible episodes" in str(error)
    else:
        raise AssertionError("undersized native pool must fail")
