"""CPU contracts for the locked PushT downstream-use extension."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.pusht_downstream import (
    deterministic_partition,
    goal_directions,
    interior_state_mask,
    pose_success,
    target_block_poses,
)


SPEC = ROOT / "configs/paper_a_pusht_downstream_use_v1.yaml"


def test_protocol_sidecar_is_locked() -> None:
    expected, name = SPEC.with_suffix(".sha256").read_text().split()
    assert name == SPEC.name
    assert hashlib.sha256(SPEC.read_bytes()).hexdigest() == expected
    spec = yaml.safe_load(SPEC.read_text())
    assert spec["compute"]["only_device"] == "cuda:1"
    assert spec["protocol_status"] == "locked_before_heldout_evaluation"
    assert spec["consumer"]["arm_identity_feature"] == "forbidden"
    assert spec["metrics"]["task_pooling"] == "forbidden"


def test_label_blind_partition_is_unique_deterministic_and_disjoint() -> None:
    episodes = np.arange(1200) * 3 + 11
    train_a, dev_a = deterministic_partition(
        episodes, seed=791301, train_count=800)
    train_b, dev_b = deterministic_partition(
        episodes, seed=791301, train_count=800)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(dev_a, dev_b)
    assert len(train_a) == 800 and len(dev_a) == 400
    assert not set(train_a).intersection(set(dev_a))


def test_eligibility_uses_only_native_block_position() -> None:
    states = np.zeros((4, 7), dtype=np.float64)
    states[:, 2:4] = [[170, 170], [342, 342], [169.9, 256], [256, 342.1]]
    assert interior_state_mask(states, 170, 342).tolist() == [
        True, True, False, False]


def test_goal_decks_and_pose_receipt_have_registered_shapes() -> None:
    states = np.array([
        [250, 300, 250, 250, 0.2, 10, -5],
        [260, 310, 260, 260, 0.4, -2, 3],
    ], dtype=np.float64)
    for classes in (4, 6):
        directions = goal_directions(classes)
        assert directions.shape == (classes, 2)
        assert np.allclose(np.linalg.norm(directions, axis=1), 1)
        targets = target_block_poses(states, classes, displacement=50)
        assert targets.shape == (2, classes, 3)
        assert np.allclose(targets[:, :, 2], states[:, None, 4])


def test_pose_success_requires_both_position_and_angle() -> None:
    target = np.array([100.0, 100.0, 0.0])
    final = np.array([
        [107.9, 100.0, 0.11],
        [108.1, 100.0, 0.0],
        [100.0, 100.0, 0.13],
    ])
    assert pose_success(final, target, 8.0, 0.12).tolist() == [
        True, False, False]


def test_prepare_process_never_loads_validation_task_labels() -> None:
    source = (ROOT / "scripts/prepare_paper_a_pusht_downstream_use.py").read_text()
    # The validation task cache is loaded only to form carrier features; its
    # labels are never indexed or assigned in the preparation process.
    assert 'task_test["labels"]' not in source
    assert "prepared_without_heldout_metrics" in source
