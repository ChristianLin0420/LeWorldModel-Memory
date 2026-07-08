from __future__ import annotations

from pathlib import Path

import numpy as np

from lewm.official_tasks.dinowm_pointmaze import (
    CUE_LEFT,
    CUE_SIZE,
    CUE_TOP,
    CurrentMujocoPointMaze,
    crossed_execution_arrays,
    endpoint_frame,
    goal_card,
    predictor_context_for_endpoint,
    released_pointmaze_xml,
    render_transient_goal_cue,
    select_native_windows,
    verify_cue_only_counterfactual,
)
from lewm.official_tasks.dinowm_native_audit import spatial_pyramid_pool


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "outputs/dinowm_native_pusht_audit_v1/vendor/dino_wm"


def test_transient_goal_card_is_a_strict_cue_only_intervention() -> None:
    rng = np.random.default_rng(4)
    base = rng.integers(0, 256, size=(20, 224, 224, 3), dtype=np.uint8)
    variants = np.stack([
        render_transient_goal_cue(base, label) for label in range(4)
    ])
    audit = verify_cue_only_counterfactual(base, variants)
    assert audit["passed"] is True
    assert np.array_equal(variants[:, :1], np.broadcast_to(base[:1], variants[:, :1].shape))
    assert np.array_equal(variants[:, 4:], np.broadcast_to(base[4:], variants[:, 4:].shape))
    for label in range(4):
        np.testing.assert_array_equal(
            variants[label, 1, CUE_TOP:CUE_TOP + CUE_SIZE,
                     CUE_LEFT:CUE_LEFT + CUE_SIZE], goal_card(label))
    assert len({goal_card(label).tobytes() for label in range(4)}) == 4


def test_native_window_selection_keeps_episode_split_disjoint() -> None:
    values = select_native_windows(
        [130] * 10, train_episodes=range(8), validation_episodes=range(8, 10),
        train_count=20, validation_count=8, num_frames=20, frame_skip=5,
        seed=3)
    train = values[:20]
    validation = values[20:]
    assert len({(value.episode_index, value.local_start) for value in values}) == 28
    assert {value.episode_index for value in train}.isdisjoint(
        {value.episode_index for value in validation})
    assert all(value.local_start + 100 <= 130 for value in values)


def test_evidence_age_endpoint_excludes_target_frame() -> None:
    assert [endpoint_frame(3, age) for age in (4, 8, 15)] == [7, 11, 18]
    assert predictor_context_for_endpoint(18) == (15, 16, 17)


def test_released_xml_loads_in_current_mujoco_and_replays() -> None:
    xml = released_pointmaze_xml(VENDOR)
    assert '<option timestep="0.01" gravity="0 0 0"' in xml
    simulator = CurrentMujocoPointMaze(VENDOR)
    initial = np.asarray([3.0, 1.0, 0.0, 0.0])
    first = simulator.reset(initial)
    after = simulator.step(np.asarray([0.25, 0.0]))
    assert after[0] > first[0]
    repeat = simulator.reset(initial)
    again = simulator.step(np.asarray([0.25, 0.0]))
    np.testing.assert_array_equal(first, repeat)
    np.testing.assert_array_equal(after, again)


def test_crossed_execution_scores_true_goal_not_selected_goal() -> None:
    matrix = np.zeros((2, 4, 4), dtype=np.int8)
    for base in range(2):
        matrix[base, np.arange(4), np.arange(4)] = 1
    truth = np.tile(np.arange(4), 2)
    prediction = truth.copy()
    prediction[1] = 0
    result = crossed_execution_arrays(matrix, prediction, truth)
    assert result["goal_correct"].tolist() == [1, 0, 1, 1, 1, 1, 1, 1]
    assert result["executed_success"].tolist() == result["goal_correct"].tolist()
    assert result["oracle_success"].tolist() == [1] * 8


def test_registered_spatial_pyramid_read_preserves_layout_and_is_8064d() -> None:
    patches = np.zeros((2, 196, 384), dtype=np.float32)
    patches[0, 0] = 1.0
    patches[1, -1] = 1.0
    pooled = spatial_pyramid_pool(patches)
    assert pooled.shape == (2, 8064)
    assert not np.array_equal(pooled[0], pooled[1])
