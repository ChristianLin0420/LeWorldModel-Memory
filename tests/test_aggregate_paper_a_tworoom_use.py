from __future__ import annotations

import numpy as np
import pytest

from scripts.aggregate_paper_a_tworoom_use import (
    ARMS,
    EPISODES,
    SEEDS,
    _stratified_episode_weights,
    _summarize,
    hierarchical_paired_bootstrap,
)


def _balanced_joint() -> np.ndarray:
    return np.repeat(np.arange(16, dtype=np.int64), 30)


def _grid(offsets: tuple[float, ...]) -> np.ndarray:
    result = np.empty((len(ARMS), len(SEEDS), EPISODES), dtype=np.float64)
    episode = np.linspace(-0.02, 0.02, EPISODES)
    for arm, offset in enumerate(offsets):
        for seed in range(len(SEEDS)):
            result[arm, seed] = offset + 0.003 * seed + episode
    return result


def test_use_bootstrap_is_deterministic_and_fully_paired() -> None:
    offsets = (0.10, 0.20, 0.25, 0.30, 0.35)
    raw = _grid(offsets)
    values = {
        "goal_correct": raw,
        "executed_success": raw + 0.01,
        "selected_distance": 10.0 - raw,
        "distance_regret": 2.0 - raw,
    }
    baseline = np.linspace(0.03, 0.07, EPISODES)
    baselines = {
        "realized_random_goal_correct": baseline,
        "realized_random_success": baseline + 0.01,
        "realized_random_distance": 12.0 - baseline,
        "realized_random_regret": 4.0 - baseline,
        "oracle_goal_correct": np.ones(EPISODES),
        "oracle_success": np.full(EPISODES, 0.95),
        "oracle_distance": np.full(EPISODES, 1.0),
        "oracle_regret": np.zeros(EPISODES),
    }
    first = hierarchical_paired_bootstrap(
        values, _balanced_joint(), baselines, draws=128, seed=17)
    second = hierarchical_paired_bootstrap(
        values, _balanced_joint(), baselines, draws=128, seed=17)
    for first_group, second_group in zip(first, second):
        for key in first_group:
            assert np.array_equal(
                np.asarray(first_group[key]), np.asarray(second_group[key]))

    points, samples, baseline_points, baseline_samples = first
    # Arm offsets are constant within every seed and episode.  Joint
    # resampling must consequently preserve their paired difference exactly.
    assert np.allclose(
        samples["executed_success"][:, ARMS.index("fixed_trust")]
        - samples["executed_success"][:, ARMS.index("none")],
        offsets[-1] - offsets[0], atol=1e-12)
    summary = _summarize(
        values, points, samples, baseline_points, baseline_samples)
    assert summary["external_use_claims"]["fixed_trust"][
        "resolved_external_use"] is True
    assert summary["fixed_trust_minus_ssm"][
        "executed_success_rate"]["mean"] == pytest.approx(
            offsets[-1] - offsets[-2])


def test_use_bootstrap_rejects_nonbalanced_episode_strata() -> None:
    labels = _balanced_joint()
    labels[0] = 1
    with pytest.raises(ValueError, match="exactly balanced"):
        _stratified_episode_weights(
            labels, 2, np.random.default_rng(0))
