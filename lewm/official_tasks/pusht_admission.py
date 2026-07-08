"""CPU-only frozen-feature admissions for semantic PushT memory tasks."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any, Mapping

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PushTAdmissionThresholds:
    cue_accuracy_min: float = 0.75
    shortcut_margin_above_chance: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.cue_accuracy_min <= 1:
            raise ValueError("cue_accuracy_min must be in (0,1]")
        if not 0 <= self.shortcut_margin_above_chance < 1:
            raise ValueError("shortcut margin must be in [0,1)")


def _validate_labels(train: np.ndarray, validation: np.ndarray,
                     classes: int) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.int64)
    validation = np.asarray(validation, dtype=np.int64)
    expected = set(range(classes))
    if train.ndim != 1 or validation.ndim != 1:
        raise ValueError("PushT admission labels must be one-dimensional")
    if set(np.unique(train)) != expected or set(np.unique(validation)) != expected:
        raise ValueError("every task class must occur in both formal splits")
    for labels in (train, validation):
        counts = np.bincount(labels, minlength=classes)
        if counts.max() - counts.min() > 1:
            raise ValueError("formal task labels must be deterministically balanced")
    return train, validation


def _probe(train_x: np.ndarray, train_y: np.ndarray,
           validation_x: np.ndarray, validation_y: np.ndarray) -> dict[str, Any]:
    train_x = np.asarray(train_x, dtype=np.float64).reshape(len(train_y), -1)
    validation_x = np.asarray(validation_x, dtype=np.float64).reshape(
        len(validation_y), -1)
    if train_x.shape[1] != validation_x.shape[1] \
            or not np.isfinite(train_x).all() \
            or not np.isfinite(validation_x).all():
        raise ValueError("admission features are non-finite or dimensionally unequal")
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=3000, random_state=0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(train_x, train_y)
    prediction = classifier.predict(validation_x)
    return {
        "accuracy": float(np.mean(prediction == validation_y)),
        "balanced_accuracy": float(balanced_accuracy_score(
            validation_y, prediction)),
        "feature_dim": int(train_x.shape[1]),
        "probe": "StandardScaler+LogisticRegression(C=1,lbfgs)",
    }


def _gate(value: Any, threshold: float, direction: str,
          passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "threshold": threshold,
        "direction": direction,
        "pass": bool(passed),
    }


def evaluate_pusht_admission(
        *, task_key: str, semantic_name: str, classes: int,
        train_base: Mapping[str, np.ndarray],
        train_task: Mapping[str, np.ndarray],
        validation_base: Mapping[str, np.ndarray],
        validation_task: Mapping[str, np.ndarray],
        cue_start: int = 1, cue_length: int = 3,
        final_context_indices: tuple[int, ...] = (16, 17, 18),
        shortcut_action_indices: tuple[int, ...] = (15, 16, 17, 18),
        thresholds: PushTAdmissionThresholds = PushTAdmissionThresholds(),
        ) -> dict[str, Any]:
    """Gate cue readability and latent/action/state shortcut absence."""

    if classes < 2:
        raise ValueError("formal memory tasks require at least two classes")
    for split, base, task in (
            ("train", train_base, train_task),
            ("validation", validation_base, validation_task)):
        for key in ("episode_index", "local_start"):
            if not np.array_equal(base[key], task[key]):
                raise ValueError(f"{split} base/task selection mismatch in {key}")
        if task["z_cue"].shape[1] != cue_length:
            raise ValueError(f"{split} task cache stores the wrong cue length")
        if np.asarray(base["z_base"]).shape[1] <= max(final_context_indices):
            raise ValueError(f"{split} base cache is too short for final context")
        if np.asarray(base["actions"]).shape[1] <= max(shortcut_action_indices):
            raise ValueError(f"{split} action cache is too short for shortcut probe")
    train_y, validation_y = _validate_labels(
        train_task["labels"], validation_task["labels"], classes)

    cue = _probe(train_task["z_cue"], train_y,
                 validation_task["z_cue"], validation_y)
    final_context = _probe(
        np.asarray(train_base["z_base"])[:, final_context_indices], train_y,
        np.asarray(validation_base["z_base"])[:, final_context_indices],
        validation_y)
    action_shortcut = _probe(
        np.asarray(train_base["actions"])[:, shortcut_action_indices], train_y,
        np.asarray(validation_base["actions"])[:, shortcut_action_indices],
        validation_y)
    state_shortcut = _probe(
        np.asarray(train_base["state"])[:, final_context_indices], train_y,
        np.asarray(validation_base["state"])[:, final_context_indices],
        validation_y)
    chance = 1.0 / classes
    shortcut_ceiling = chance + thresholds.shortcut_margin_above_chance
    gates = {
        "cue_availability": _gate(
            cue, thresholds.cue_accuracy_min, ">= balanced accuracy",
            cue["balanced_accuracy"] >= thresholds.cue_accuracy_min),
        "final_context_latent_shortcut": _gate(
            final_context, shortcut_ceiling, "<= balanced accuracy",
            final_context["balanced_accuracy"] <= shortcut_ceiling),
        "final_context_action_shortcut": _gate(
            action_shortcut, shortcut_ceiling, "<= balanced accuracy",
            action_shortcut["balanced_accuracy"] <= shortcut_ceiling),
        "final_context_state_shortcut": _gate(
            state_shortcut, shortcut_ceiling, "<= balanced accuracy",
            state_shortcut["balanced_accuracy"] <= shortcut_ceiling),
    }
    return {
        "schema": "official_pusht_frozen_admission_v1",
        "task_key": task_key,
        "semantic_name": semantic_name,
        "classes": classes,
        "chance": chance,
        "train_episodes": int(len(train_y)),
        "validation_episodes": int(len(validation_y)),
        "representation_frozen": True,
        "world_model_training_performed": False,
        "task_cache_contract": (
            "only cue latents are stored; final context, actions, and state "
            "come from the shared label-independent base cache"),
        "thresholds": {
            "cue_accuracy_min": thresholds.cue_accuracy_min,
            "shortcut_margin_above_chance":
                thresholds.shortcut_margin_above_chance,
            "shortcut_ceiling": shortcut_ceiling,
        },
        "gates": gates,
        "admitted": all(gate["pass"] for gate in gates.values()),
    }


__all__ = ["PushTAdmissionThresholds", "evaluate_pusht_admission"]
