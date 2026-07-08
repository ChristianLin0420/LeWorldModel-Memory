from __future__ import annotations

import numpy as np
import pytest

from lewm.official_tasks.dinowm_native_audit import (
    RolloutHealthThresholds,
    endpoint_frame_for_age,
    frozen_linear_probe,
    paired_transport_summary,
    pairwise_counterfactual_separation,
    spatial_pyramid_pool,
    summarize_rollout_health,
    temporal_spatial_pyramid_pool,
)


def test_age_mapping_is_anchored_after_last_cue_frame() -> None:
    assert [endpoint_frame_for_age(last_cue_frame=3, age=age)
            for age in (1, 4, 8, 15)] == [4, 7, 11, 18]
    with pytest.raises(ValueError):
        endpoint_frame_for_age(last_cue_frame=3, age=0)


def test_spatial_pyramid_pool_has_fixed_order_and_shape() -> None:
    grid = np.arange(14 * 14, dtype=np.float32).reshape(1, 196, 1)
    pooled = spatial_pyramid_pool(grid, levels=(1, 2, 4))
    assert pooled.shape == (1, 21)
    assert pooled[0, 0] == pytest.approx(grid.mean())
    # The first 2x2 bin spans rows/columns [0,7).
    assert pooled[0, 1] == pytest.approx(
        grid.reshape(14, 14)[:7, :7].mean())


def test_temporal_pool_averages_frames_after_spatial_pool() -> None:
    patches = np.stack([
        np.zeros((196, 2), dtype=np.float32),
        np.full((196, 2), 2.0, dtype=np.float32),
    ], axis=0)[None]
    pooled = temporal_spatial_pyramid_pool(patches)
    assert pooled.shape == (1, 42)
    assert np.allclose(pooled, 1.0)


def test_pairwise_counterfactual_separation_is_episode_paired() -> None:
    features = np.asarray([
        [[0.0, 0.0], [2.0, 0.0]],
        [[1.0, 1.0], [1.0, 3.0]],
    ])
    separation = pairwise_counterfactual_separation(features)
    assert np.allclose(separation, np.sqrt(2.0))
    summary = paired_transport_summary(
        separation, separation / 2.0, draws=50, seed=4)
    assert summary["transport_ratio"] == pytest.approx(0.5)


def test_frozen_probe_recovers_balanced_separable_labels() -> None:
    train_y = np.tile(np.arange(3), 12)
    validation_y = np.tile(np.arange(3), 6)
    train_x = np.eye(3, dtype=np.float32)[train_y]
    validation_x = np.eye(3, dtype=np.float32)[validation_y]
    record = frozen_linear_probe(
        train_x, train_y, validation_x, validation_y, classes=3)
    assert record["balanced_accuracy"] == pytest.approx(1.0)
    assert np.array_equal(record["prediction"], validation_y)


def test_rollout_health_requires_copy_and_action_controls() -> None:
    true = np.full((5, 4), 0.5)
    copy = np.full((5, 4), 1.0)
    shuffled = np.full((5, 4), 0.8)
    passed = summarize_rollout_health(true, copy, shuffled)
    assert passed["admitted"] is True
    failed = summarize_rollout_health(
        true, copy, np.full((5, 4), 0.4),
        thresholds=RolloutHealthThresholds())
    assert failed["admitted"] is False
    assert failed["gates"]["horizon_1_to_H_uses_actions"]["pass"] is False
