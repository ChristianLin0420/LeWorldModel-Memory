"""Frozen utilities for the native DINO-WM PushT portability audit.

The audit intentionally does not add a carrier or persistent state.  It reads
out the released model's frozen patch representation and its native open-loop
predictions.  Functions in this module are CPU-testable; the released model
adapter lives in the execution script because it depends on the pinned vendor
checkout and checkpoint pickle classes.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


NATIVE_CONTEXT = 3
NATIVE_PATCHES = 196
NATIVE_VISUAL_DIM = 384
NATIVE_ACTION_DIM = 10
NATIVE_PROPRIO_DIM = 4


def endpoint_frame_for_age(*, last_cue_frame: int, age: int) -> int:
    """Map an evidence age to the corresponding predicted frame index."""

    if isinstance(last_cue_frame, bool) or not isinstance(last_cue_frame, int) \
            or last_cue_frame < 0:
        raise ValueError("last_cue_frame must be a non-negative integer")
    if isinstance(age, bool) or not isinstance(age, int) or age < 1:
        raise ValueError("age must be a positive integer")
    return last_cue_frame + age


def spatial_pyramid_pool(
        patches: np.ndarray, *, levels: Sequence[int] = (1, 2, 4),
        ) -> np.ndarray:
    """Average a square patch grid into fixed spatial-pyramid bins.

    ``patches`` may have any leading dimensions and must end in ``(P, D)``.
    Adaptive integer boundaries make the declared 1x1/2x2/4x4 pyramid valid
    for the native 14x14 DINO grid without resizing or learning a readout.
    Bins are flattened in level, row, column, feature order.
    """

    values = np.asarray(patches)
    if values.ndim < 2 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("patches must be a numeric array ending in (P,D)")
    patch_count, feature_dim = values.shape[-2:]
    side = int(round(np.sqrt(patch_count)))
    if side * side != patch_count or feature_dim < 1:
        raise ValueError("patches must form a non-empty square grid")
    clean_levels: list[int] = []
    for level in levels:
        if isinstance(level, bool) or not isinstance(level, int) \
                or level < 1 or level > side:
            raise ValueError("each pyramid level must be in [1, grid_side]")
        clean_levels.append(level)
    if not clean_levels or len(set(clean_levels)) != len(clean_levels):
        raise ValueError("pyramid levels must be non-empty and unique")
    if not np.isfinite(values).all():
        raise ValueError("patch features must be finite")

    grid = values.reshape(*values.shape[:-2], side, side, feature_dim)
    pooled: list[np.ndarray] = []
    for level in clean_levels:
        row_edges = np.floor(np.linspace(0, side, level + 1)).astype(int)
        col_edges = np.floor(np.linspace(0, side, level + 1)).astype(int)
        for row in range(level):
            for col in range(level):
                cell = grid[..., row_edges[row]:row_edges[row + 1],
                            col_edges[col]:col_edges[col + 1], :]
                if cell.shape[-3] == 0 or cell.shape[-2] == 0:
                    raise ValueError("a spatial-pyramid bin is empty")
                pooled.append(cell.mean(axis=(-3, -2), dtype=np.float64))
    result = np.concatenate(pooled, axis=-1).astype(np.float32)
    return result


def temporal_spatial_pyramid_pool(
        patches: np.ndarray, *, levels: Sequence[int] = (1, 2, 4),
        ) -> np.ndarray:
    """Pool each frame spatially, then average the declared time axis."""

    values = np.asarray(patches)
    if values.ndim < 3 or values.shape[-3] < 1:
        raise ValueError("patches must end in (time,P,D) with non-empty time")
    return spatial_pyramid_pool(values, levels=levels).mean(
        axis=-2, dtype=np.float64).astype(np.float32)


def validate_balanced_labels(labels: np.ndarray, classes: int,
                             name: str) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
        raise ValueError("classes must be an integer >= 2")
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} labels must be a non-empty vector")
    if set(np.unique(values)) != set(range(classes)):
        raise ValueError(f"{name} does not contain every declared class")
    counts = np.bincount(values, minlength=classes)
    if counts.max() - counts.min() > 1:
        raise ValueError(f"{name} labels are not deterministically balanced")
    return values


def frozen_linear_probe(
        train_x: np.ndarray, train_y: np.ndarray,
        validation_x: np.ndarray, validation_y: np.ndarray, *, classes: int,
        c: float = 1.0, max_iter: int = 3000,
        ) -> dict[str, Any]:
    """Fit the preregistered standardized multinomial linear probe."""

    train_y = validate_balanced_labels(train_y, classes, "train")
    validation_y = validate_balanced_labels(
        validation_y, classes, "validation")
    train_x = np.asarray(train_x, dtype=np.float64).reshape(len(train_y), -1)
    validation_x = np.asarray(validation_x, dtype=np.float64).reshape(
        len(validation_y), -1)
    if train_x.shape[1] != validation_x.shape[1] or train_x.shape[1] < 1 \
            or not np.isfinite(train_x).all() \
            or not np.isfinite(validation_x).all():
        raise ValueError("probe features are non-finite or dimensionally unequal")
    if not np.isfinite(c) or c <= 0 or isinstance(max_iter, bool) \
            or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("invalid frozen probe hyperparameters")
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c), solver="lbfgs", max_iter=max_iter,
            random_state=0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(train_x, train_y)
    prediction = classifier.predict(validation_x).astype(np.int64)
    return {
        "accuracy": float(np.mean(prediction == validation_y)),
        "balanced_accuracy": float(balanced_accuracy_score(
            validation_y, prediction)),
        "feature_dim": int(train_x.shape[1]),
        "prediction": prediction,
        "truth": validation_y,
        "readout": (
            "StandardScaler+LogisticRegression(C=1,solver=lbfgs,"
            f"max_iter={max_iter},random_state=0)"),
    }


def bootstrap_mean_ci(values: np.ndarray, *, draws: int, seed: int,
                      confidence: float = 0.95) -> dict[str, float]:
    """Episode bootstrap confidence interval for a scalar mean."""

    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all():
        raise ValueError("bootstrap values must be a finite non-empty vector")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    # Chunk draws so formal 10k bootstraps do not allocate draws x episodes.
    cursor = 0
    while cursor < draws:
        stop = min(draws, cursor + 512)
        indices = rng.integers(0, len(data), size=(stop - cursor, len(data)))
        samples[cursor:stop] = data[indices].mean(axis=1)
        cursor = stop
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(data.mean()),
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1.0 - alpha)),
        "confidence": float(confidence),
        "draws": int(draws),
    }


def bootstrap_accuracy_ci(prediction: np.ndarray, truth: np.ndarray, *,
                          draws: int, seed: int,
                          confidence: float = 0.95) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    if prediction.shape != truth.shape or prediction.ndim != 1:
        raise ValueError("prediction and truth must be equal-length vectors")
    return bootstrap_mean_ci(
        (prediction == truth).astype(np.float64), draws=draws, seed=seed,
        confidence=confidence)


def pairwise_counterfactual_separation(features: np.ndarray) -> np.ndarray:
    """Per-episode RMS separation across every within-episode label pair.

    Input shape is ``(episodes, classes, ...)``.  The output is one scalar per
    episode, retaining the paired counterfactual unit for bootstrap inference.
    """

    values = np.asarray(features, dtype=np.float64)
    if values.ndim < 3 or values.shape[0] < 1 or values.shape[1] < 2 \
            or not np.isfinite(values).all():
        raise ValueError("features must be finite (episodes,classes,...)")
    flat = values.reshape(values.shape[0], values.shape[1], -1)
    pairs: list[np.ndarray] = []
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            delta = flat[:, left] - flat[:, right]
            pairs.append(np.sqrt(np.mean(np.square(delta), axis=1)))
    return np.stack(pairs, axis=1).mean(axis=1)


def paired_transport_summary(
        cue_separation: np.ndarray, age_separation: np.ndarray, *,
        draws: int, seed: int, confidence: float = 0.95,
        ) -> dict[str, Any]:
    """Paired separation and ratio-of-means interval for one evidence age."""

    cue = np.asarray(cue_separation, dtype=np.float64)
    age = np.asarray(age_separation, dtype=np.float64)
    if cue.shape != age.shape or cue.ndim != 1 or not len(cue) \
            or not np.isfinite(cue).all() or not np.isfinite(age).all() \
            or np.any(cue <= 0) or np.any(age < 0):
        raise ValueError("paired separations must be finite vectors with cue > 0")
    rng = np.random.default_rng(seed)
    ratios = np.empty(draws, dtype=np.float64)
    cursor = 0
    while cursor < draws:
        stop = min(draws, cursor + 512)
        indices = rng.integers(0, len(cue), size=(stop - cursor, len(cue)))
        cue_mean = cue[indices].mean(axis=1)
        age_mean = age[indices].mean(axis=1)
        ratios[cursor:stop] = age_mean / cue_mean
        cursor = stop
    alpha = (1.0 - confidence) / 2.0
    return {
        "cue_rms_mean": float(cue.mean()),
        "age_rms_mean": float(age.mean()),
        "transport_ratio": float(age.mean() / cue.mean()),
        "transport_ratio_lower": float(np.quantile(ratios, alpha)),
        "transport_ratio_upper": float(np.quantile(ratios, 1.0 - alpha)),
        "confidence": float(confidence),
        "draws": int(draws),
    }


@dataclass(frozen=True)
class RolloutHealthThresholds:
    one_step_copy_ratio_max: float = 1.0
    integrated_copy_ratio_max: float = 1.0
    integrated_action_advantage_min: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.one_step_copy_ratio_max,
            self.integrated_copy_ratio_max,
            self.integrated_action_advantage_min,
        )
        if not all(np.isfinite(values)) \
                or min(self.one_step_copy_ratio_max,
                       self.integrated_copy_ratio_max) <= 0:
            raise ValueError("invalid rollout-health thresholds")


def summarize_rollout_health(
        true_mse: np.ndarray, copy_last_mse: np.ndarray,
        shuffled_mse: np.ndarray, *,
        thresholds: RolloutHealthThresholds = RolloutHealthThresholds(),
        ) -> dict[str, Any]:
    """Summarize native rollout health at every horizon and evaluate gates.

    Arrays must be ``(episodes, horizons)``.  The integrated criteria use the
    mean over all horizons 1..H and therefore cannot hide a terminal horizon.
    """

    true = np.asarray(true_mse, dtype=np.float64)
    copy = np.asarray(copy_last_mse, dtype=np.float64)
    shuffled = np.asarray(shuffled_mse, dtype=np.float64)
    if true.shape != copy.shape or true.shape != shuffled.shape \
            or true.ndim != 2 or min(true.shape) < 1 \
            or not np.isfinite(true).all() or not np.isfinite(copy).all() \
            or not np.isfinite(shuffled).all() \
            or np.any(true < 0) or np.any(copy <= 0) \
            or np.any(shuffled <= 0):
        raise ValueError("health MSE arrays must be finite positive E x H arrays")
    true_h = true.mean(axis=0)
    copy_h = copy.mean(axis=0)
    shuffled_h = shuffled.mean(axis=0)
    copy_ratio_h = true_h / copy_h
    advantage_h = (shuffled_h - true_h) / shuffled_h
    one_step_ratio = float(copy_ratio_h[0])
    integrated_ratio = float(true_h.mean() / copy_h.mean())
    integrated_advantage = float(
        (shuffled_h.mean() - true_h.mean()) / shuffled_h.mean())
    gates = {
        "one_step_beats_copy_last": {
            "value": one_step_ratio,
            "threshold": thresholds.one_step_copy_ratio_max,
            "direction": "<",
            "pass": one_step_ratio < thresholds.one_step_copy_ratio_max,
        },
        "horizon_1_to_H_beats_copy_last": {
            "value": integrated_ratio,
            "threshold": thresholds.integrated_copy_ratio_max,
            "direction": "<",
            "pass": integrated_ratio < thresholds.integrated_copy_ratio_max,
        },
        "horizon_1_to_H_uses_actions": {
            "value": integrated_advantage,
            "threshold": thresholds.integrated_action_advantage_min,
            "direction": ">",
            "pass": integrated_advantage \
                    > thresholds.integrated_action_advantage_min,
        },
    }
    return {
        "episodes": int(true.shape[0]),
        "horizons": list(range(1, true.shape[1] + 1)),
        "true_action_mse": true_h.tolist(),
        "copy_last_mse": copy_h.tolist(),
        "shuffled_action_mse": shuffled_h.tolist(),
        "model_to_copy_ratio": copy_ratio_h.tolist(),
        "true_action_advantage_over_shuffled": advantage_h.tolist(),
        "integrated_model_to_copy_ratio": integrated_ratio,
        "integrated_true_action_advantage": integrated_advantage,
        "gates": gates,
        "admitted": all(record["pass"] for record in gates.values()),
    }


def strip_probe_arrays(record: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe probe record without per-example arrays."""

    return {key: value for key, value in record.items()
            if key not in {"prediction", "truth"}}


__all__ = [
    "NATIVE_ACTION_DIM",
    "NATIVE_CONTEXT",
    "NATIVE_PATCHES",
    "NATIVE_PROPRIO_DIM",
    "NATIVE_VISUAL_DIM",
    "RolloutHealthThresholds",
    "bootstrap_accuracy_ci",
    "bootstrap_mean_ci",
    "endpoint_frame_for_age",
    "frozen_linear_probe",
    "paired_transport_summary",
    "pairwise_counterfactual_separation",
    "spatial_pyramid_pool",
    "strip_probe_arrays",
    "summarize_rollout_health",
    "temporal_spatial_pyramid_pool",
    "validate_balanced_labels",
]
