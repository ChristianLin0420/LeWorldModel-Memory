#!/usr/bin/env python3
"""Shared feature, consumer, and executed-choice helpers."""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.cache_official_lewm import _spaced_indices


def carrier_interface(data: Mapping[str, Any], prior: np.ndarray) -> np.ndarray:
    z = np.asarray(data["z"], dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)
    if z.shape != prior.shape or z.ndim != 3 or z.shape[1] != 64:
        raise ValueError("carrier interface requires matching (E,64,D) arrays")
    return np.concatenate([
        z[:, 60:63].reshape(len(z), -1), prior[:, 63]], axis=1)


def long_context_interface(data: Mapping[str, Any], history: int = 56
                           ) -> np.ndarray:
    z = np.asarray(data["z"], dtype=np.float32)
    decision = z.shape[1] - 1
    window = z[:, decision - history:decision]
    if history != 56 or window.shape[1] != 56:
        raise ValueError("locked long-context interface requires exactly 56 tokens")
    return np.concatenate(
        [chunk.mean(axis=1) for chunk in np.split(window, 4, axis=1)], axis=1)


def cue_window_interface(data: Mapping[str, Any]) -> np.ndarray:
    z = np.asarray(data["z"], dtype=np.float32)
    cue_on = np.asarray(data["event_cue_on"], dtype=np.int64)
    cue_off = np.asarray(data["event_cue_off"], dtype=np.int64)
    indices = _spaced_indices(cue_on, cue_off - 1)
    selected = z[np.arange(len(z))[:, None], indices]
    return selected.reshape(len(z), -1)


def cue_repair_target(data: Mapping[str, Any], *, decision_index: int = 63
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return a label-free early-cue target and its audited frame indices."""

    z = np.asarray(data["z"], dtype=np.float32)
    cue_on = np.asarray(data["event_cue_on"], dtype=np.int64)
    cue_off = np.asarray(data["event_cue_off"], dtype=np.int64)
    if z.ndim != 3 or z.shape[1] != decision_index + 1:
        raise ValueError("repair target requires the locked 64-frame bank")
    indices = _spaced_indices(cue_on, cue_off - 1)
    if np.any(indices < 0) or np.any(indices >= decision_index):
        raise ValueError("repair target touches a final or future frame")
    if np.any(indices < cue_on[:, None]) \
            or np.any(indices >= cue_off[:, None]):
        raise ValueError("repair target leaves the registered cue window")
    selected = z[np.arange(len(z))[:, None], indices]
    return selected.reshape(len(z), -1), indices


def action_time_interface(data: Mapping[str, Any], dimension: int = 768
                          ) -> np.ndarray:
    actions = np.asarray(data["actions"], dtype=np.float32)
    if actions.ndim != 3 or actions.shape[2] != 10:
        raise ValueError("action-time control requires (E,T,10) actions")
    cue_on = np.asarray(data["event_cue_on"], dtype=np.float32)
    cue_off = np.asarray(data["event_cue_off"], dtype=np.float32)
    decision = float(np.asarray(data["z"]).shape[1] - 1)
    summary = np.concatenate([
        actions.mean(axis=1),
        actions.sum(axis=1),
        actions[:, -1],
        (cue_on / decision)[:, None],
        (cue_off / decision)[:, None],
        np.full((len(actions), 1), 1.0, dtype=np.float32),
    ], axis=1)
    if summary.shape[1] > dimension:
        raise ValueError("action-time summary exceeds common interface")
    return np.pad(summary, ((0, 0), (0, dimension - summary.shape[1])))


class SharedGoalConsumer:
    def __init__(self, scaler: StandardScaler,
                 model: LogisticRegression) -> None:
        self.scaler = scaler
        self.model = model

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(features)).astype(np.int64)

    def digest(self) -> str:
        payload = {
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "classes": self.model.classes_.tolist(),
            "coef": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fit_shared_consumer(feature_sets: Mapping[str, np.ndarray],
                        labels: np.ndarray, source_order: list[str],
                        *, c: float, solver: str, max_iter: int,
                        random_state: int,
                        label_permutation: np.ndarray | None = None
                        ) -> SharedGoalConsumer:
    labels = np.asarray(labels, dtype=np.int64)
    if sorted(np.unique(labels).tolist()) != [0, 1, 2, 3]:
        raise ValueError("consumer requires all four registered classes")
    matrices = []
    for source in source_order:
        value = np.asarray(feature_sets[source], dtype=np.float32)
        if value.shape != (len(labels), 768):
            raise ValueError(
                f"source {source} has shape {value.shape}, expected "
                f"{(len(labels), 768)}")
        matrices.append(value)
    x = np.concatenate(matrices, axis=0)
    fit_labels = labels if label_permutation is None else labels[label_permutation]
    y = np.tile(fit_labels, len(source_order))
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=c, solver=solver, max_iter=max_iter, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(x), y)
    return SharedGoalConsumer(scaler, model)


def fit_shortcut_consumer(train_x: np.ndarray, labels: np.ndarray,
                          consumer: Mapping[str, Any]) -> SharedGoalConsumer:
    return fit_shared_consumer(
        {"shortcut": train_x}, labels, ["shortcut"],
        c=float(consumer["logistic_c"]),
        solver=str(consumer["solver"]),
        max_iter=int(consumer["max_iter"]),
        random_state=int(consumer["random_state"]),
    )


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def wrapped_rms(position: np.ndarray, goal: np.ndarray) -> float:
    difference = wrap_angle(np.asarray(position) - np.asarray(goal))
    return float(np.sqrt(np.mean(np.square(difference))))


def pd_action(position: np.ndarray, velocity: np.ndarray, goal: np.ndarray,
              kp: float, kd: float, low: np.ndarray,
              high: np.ndarray) -> np.ndarray:
    command = kp * wrap_angle(np.asarray(goal) - np.asarray(position)) \
        - kd * np.asarray(velocity)
    return np.clip(command, low, high).astype(np.float32)


def execute_reacher_choices(initial_states: np.ndarray, labels: np.ndarray,
                            protocol: Mapping[str, Any]
                            ) -> dict[str, np.ndarray]:
    """Execute every candidate goal from every cached decision state."""

    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from dm_control import suite

    states = np.asarray(initial_states, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    goals = np.asarray(protocol["joint_goals"], dtype=np.float64)
    if states.shape != (len(labels), 4) or goals.shape != (4, 2):
        raise ValueError("unexpected Reacher state or goal shape")
    horizon = int(protocol["executed_horizon"])
    kp = float(protocol["proportional_gain"])
    kd = float(protocol["derivative_gain"])
    tolerance = float(protocol["success_tolerance_radians"])
    scale = float(protocol["return_scale_radians"])
    environment = suite.load("reacher", "easy", task_kwargs={"random": 0})
    spec = environment.action_spec()
    low = np.asarray(spec.minimum, dtype=np.float64)
    high = np.asarray(spec.maximum, dtype=np.float64)
    distance = np.empty((len(labels), 4), dtype=np.float64)
    for episode, state in enumerate(states):
        true_goal = goals[labels[episode]]
        for decision, selected_goal in enumerate(goals):
            environment.reset()
            with environment.physics.reset_context():
                environment.physics.set_state(state)
            for _ in range(horizon):
                position = np.asarray(environment.physics.data.qpos, dtype=np.float64)
                velocity = np.asarray(environment.physics.data.qvel, dtype=np.float64)
                timestep = environment.step(pd_action(
                    position, velocity, selected_goal, kp, kd, low, high))
                if timestep.last():
                    break
            distance[episode, decision] = wrapped_rms(
                environment.physics.data.qpos, true_goal)
    returns = np.exp(-0.5 * np.square(distance / scale))
    success = distance <= tolerance
    return {
        "distance": distance,
        "return": returns,
        "success": success,
    }


def decision_metrics(prediction: np.ndarray, labels: np.ndarray,
                     execution: Mapping[str, np.ndarray]) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    index = np.arange(len(labels))
    distance = np.asarray(execution["distance"])[index, prediction]
    returns = np.asarray(execution["return"])[index, prediction]
    success = np.asarray(execution["success"])[index, prediction]
    oracle_return = np.asarray(execution["return"])[index, labels]
    regret = oracle_return - returns
    return {
        "episodes": int(len(labels)),
        "goal_decision_accuracy": float(np.mean(prediction == labels)),
        "executed_success_rate": float(success.mean()),
        "mean_executed_return": float(returns.mean()),
        "mean_regret_to_label_oracle": float(regret.mean()),
        "prediction": prediction.tolist(),
        "correct": (prediction == labels).astype(np.int8).tolist(),
        "success": success.astype(np.int8).tolist(),
        "distance": distance.tolist(),
        "executed_return": returns.tolist(),
        "regret_to_label_oracle": regret.tolist(),
    }
