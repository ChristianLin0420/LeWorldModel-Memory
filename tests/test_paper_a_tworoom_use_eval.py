from __future__ import annotations

import numpy as np

from scripts.evaluate_paper_a_tworoom_use import (
    _execution_result,
    _fit_consumer,
)


def test_execution_result_indexes_each_registered_choice() -> None:
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    joint = np.asarray([0, 5, 10, 15], dtype=np.int64)
    prediction = np.asarray([0, 2, 2, 1], dtype=np.int64)
    random_choice = np.asarray([3, 2, 1, 0], dtype=np.int64)
    # Axes are episode x selected controller goal x true cued goal.
    distance = np.full((4, 4, 4), 20.0)
    distance[:, np.arange(4), np.arange(4)] = 1.0
    distance[0, 0, 0] = 1.0
    distance[1, 2, 1] = 2.0
    distance[2, 2, 2] = 1.0
    distance[3, 1, 3] = 8.0
    distance[0, 3, 0] = 18.0
    distance[1, 2, 1] = 2.0
    distance[2, 1, 2] = 19.0
    distance[3, 0, 3] = 2.0
    success = (distance < 16.0).astype(np.int8)
    result = _execution_result(
        prediction, labels, joint, success, distance, random_choice)
    assert result["goal_correct"] == [1, 0, 1, 0]
    assert result["executed_success"] == [1, 1, 1, 1]
    assert result["oracle_success"] == [1, 1, 1, 1]
    assert result["random_success"] == [0, 1, 0, 1]
    assert result["selected_distance"] == [1.0, 2.0, 1.0, 8.0]
    assert result["oracle_distance"] == [1.0, 1.0, 1.0, 1.0]
    assert result["distance_regret"] == [0.0, 1.0, 0.0, 7.0]
    assert result["goal_selection_accuracy"] == 0.5
    assert result["executed_success_rate"] == 1.0


def test_locked_consumer_is_arm_id_free_and_deterministic() -> None:
    rng = np.random.default_rng(9)
    labels = np.tile(np.arange(4, dtype=np.int64), 8)
    x = rng.normal(size=(32, 768)).astype(np.float32)
    x[:, :4] += np.eye(4, dtype=np.float32)[labels] * 4.0
    protocol = {
        "logistic_c": 1.0,
        "solver": "lbfgs",
        "max_iter": 4000,
        "random_state": 0,
    }
    pooled_x = np.concatenate([x + 0.01 * arm for arm in range(5)], axis=0)
    pooled_labels = np.tile(labels, 5)
    first, first_record = _fit_consumer(pooled_x, pooled_labels, x, protocol)
    second, second_record = _fit_consumer(
        pooled_x, pooled_labels, x[::-1].copy(), protocol)
    assert np.array_equal(first, second[::-1])
    assert first_record["parameter_sha256"] == second_record["parameter_sha256"]
    assert first_record["arm_id_feature_present"] is False
    assert first_record["feature_dimension"] == 768
    assert first_record["fit_episodes"] == 160
