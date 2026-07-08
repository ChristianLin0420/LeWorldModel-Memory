#!/usr/bin/env python3
"""Label-free cue-residual objective and deterministic V2 training plans."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


LATENT_DIM = 192
ACTION_DIM = 10
SEQUENCE_LENGTH = 64
DECISION_INDEX = 63
LABEL_ARRAY_NAMES = frozenset({"xi", "label", "labels", "target_class"})


@dataclass(frozen=True)
class TargetStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    scale_floor: float

    def __post_init__(self) -> None:
        if self.mean.shape != (LATENT_DIM,) \
                or self.scale.shape != (LATENT_DIM,):
            raise ValueError("cue-residual standardizer must have 192 coordinates")
        floor32 = np.float32(self.scale_floor)
        if self.mean.dtype != np.float32 or self.scale.dtype != np.float32 \
                or not np.isfinite(self.mean).all() \
                or not np.isfinite(self.scale).all() \
                or np.any(self.scale < floor32):
            raise ValueError("invalid cue-residual standardizer")

    def transform(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != LATENT_DIM:
            raise ValueError("cue-residual values must be (episodes,192)")
        return ((value - self.mean) / self.scale).astype(np.float32)

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.mean.tobytes())
        digest.update(self.scale.tobytes())
        digest.update(np.float64(self.scale_floor).tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class EpochPlan:
    order: np.ndarray
    starts_by_batch: tuple[np.ndarray, ...]


def load_label_free_bank(path: str | Path, *, require_actions: bool = True
                         ) -> dict[str, Any]:
    """Load only z/actions/cue timing; label arrays are never indexed."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as source:
        required = ["z", "event_cue_on", "event_cue_off"]
        if require_actions:
            required.append("actions")
        missing = [key for key in required if key not in source.files]
        if missing:
            raise ValueError(f"{path} misses label-free arrays {missing}")
        label_arrays_present = sorted(LABEL_ARRAY_NAMES.intersection(source.files))
        data = {key: np.asarray(source[key]) for key in required}
    z = np.asarray(data["z"])
    if z.dtype != np.float32 or z.shape[1:] != (SEQUENCE_LENGTH, LATENT_DIM):
        raise ValueError("label-free bank requires float32 (E,64,192) z")
    episodes = len(z)
    if require_actions:
        actions = np.asarray(data["actions"])
        if actions.dtype != np.float32 \
                or actions.shape != (episodes, 63, ACTION_DIM):
            raise ValueError("label-free bank requires float32 (E,63,10) actions")
    for key in ("event_cue_on", "event_cue_off"):
        value = np.asarray(data[key])
        if value.dtype != np.int64 or value.shape != (episodes,):
            raise ValueError(f"{key} must be int64 (episodes,)")
    data["label_arrays_present_but_not_loaded"] = label_arrays_present
    data["label_arrays_loaded"] = False
    return data


def cue_residual_target(
        data: Mapping[str, Any], *, decision_index: int = DECISION_INDEX,
        ) -> tuple[np.ndarray, dict[str, Any]]:
    """Return cue mean minus the immediate endpoint interpolation mean."""

    z = np.asarray(data["z"], dtype=np.float32)
    cue_on = np.asarray(data["event_cue_on"], dtype=np.int64)
    cue_off = np.asarray(data["event_cue_off"], dtype=np.int64)
    if z.ndim != 3 or z.shape[1:] != (decision_index + 1, LATENT_DIM) \
            or cue_on.shape != (len(z),) or cue_off.shape != (len(z),):
        raise ValueError("cue-residual target has an invalid bank shape")
    pre = cue_on - 1
    post = cue_off
    duration = cue_off - cue_on
    if np.any(pre < 0) or np.any(duration <= 0) \
            or np.any(post >= decision_index):
        raise ValueError("cue-residual target touches an illegal time index")
    cue_mean = np.empty((len(z), LATENT_DIM), dtype=np.float32)
    for episode, (start, stop) in enumerate(zip(cue_on, cue_off, strict=True)):
        cue_mean[episode] = z[episode, int(start):int(stop)].mean(
            axis=0, dtype=np.float64).astype(np.float32)
    episodes = np.arange(len(z))
    baseline = 0.5 * (z[episodes, pre] + z[episodes, post])
    residual = (cue_mean - baseline).astype(np.float32)
    if not np.isfinite(residual).all():
        raise ValueError("cue-residual target contains non-finite values")
    audit = {
        "schema": "delayed_repair_cue_residual_target_v2",
        "episodes": int(len(z)),
        "dimension": LATENT_DIM,
        "formula": (
            "mean(z[cue_on:cue_off]) - "
            "0.5*(z[cue_on-1]+z[cue_off])"),
        "cue_index_min": int(cue_on.min()),
        "cue_index_max": int((cue_off - 1).max()),
        "pre_index_min": int(pre.min()),
        "pre_index_max": int(pre.max()),
        "post_index_min": int(post.min()),
        "post_index_max": int(post.max()),
        "cue_duration_min": int(duration.min()),
        "cue_duration_max": int(duration.max()),
        "decision_index": int(decision_index),
        "decision_frame_excluded": bool(post.max() < decision_index),
        "future_frame_consumed": False,
        "labels_consumed": False,
    }
    return residual, audit


def fit_target_standardizer(target: np.ndarray, scale_floor: float
                            ) -> TargetStandardizer:
    """Fit population statistics on the formal training target only."""

    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 2 or target.shape[1] != LATENT_DIM \
            or not np.isfinite(target).all():
        raise ValueError("training cue-residual target must be finite (E,192)")
    if not 0 < scale_floor < 1:
        raise ValueError("scale floor must lie in (0,1)")
    mean = target.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = target.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, np.float32(scale_floor))
    return TargetStandardizer(mean, scale, float(scale_floor))


def development_health(
        target: np.ndarray, audit: Mapping[str, Any],
        protocol: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Apply the locked unlabeled target non-degeneracy/causality gate."""

    target = np.asarray(target, dtype=np.float32)
    coordinate_std = target.std(axis=0, dtype=np.float64)
    episode_rms = np.sqrt(np.mean(np.square(target), axis=1))
    std_threshold = float(protocol["coordinate_std_min"])
    std_fraction = float(np.mean(coordinate_std >= std_threshold))
    median_rms = float(np.median(episode_rms))
    checks = {
        "episode_count": {
            "value": int(len(target)),
            "expected": int(protocol["required_episodes"]),
            "pass": len(target) == int(protocol["required_episodes"]),
        },
        "cue_duration": {
            "value": [audit["cue_duration_min"], audit["cue_duration_max"]],
            "expected": [protocol["cue_duration_min"],
                         protocol["cue_duration_max"]],
            "pass": (
                audit["cue_duration_min"] >= protocol["cue_duration_min"]
                and audit["cue_duration_max"] <= protocol["cue_duration_max"]),
        },
        "coordinate_std_fraction": {
            "value": std_fraction,
            "threshold": float(protocol["coordinate_std_fraction_min"]),
            "direction": ">=",
            "pass": std_fraction >= float(
                protocol["coordinate_std_fraction_min"]),
        },
        "median_episode_residual_rms": {
            "value": median_rms,
            "interval": [float(protocol["median_episode_residual_rms_min"]),
                         float(protocol["median_episode_residual_rms_max"])],
            "pass": (
                float(protocol["median_episode_residual_rms_min"])
                <= median_rms
                <= float(protocol["median_episode_residual_rms_max"])),
        },
        "causal_indices": {
            "value": {
                "post_index_max": audit["post_index_max"],
                "decision_index": audit["decision_index"],
                "decision_frame_excluded": audit["decision_frame_excluded"],
            },
            "expected": "all target indices < decision index",
            "pass": bool(
                audit["decision_frame_excluded"]
                and audit["post_index_max"] < audit["decision_index"]),
        },
    }
    return {
        "schema": "delayed_repair_cue_residual_development_health_v2",
        "target_audit": dict(audit),
        "labels_loaded": False,
        "formal_training_performed": False,
        "coordinate_std_minimum": float(coordinate_std.min()),
        "coordinate_std_median": float(np.median(coordinate_std)),
        "episode_residual_rms_minimum": float(episode_rms.min()),
        "episode_residual_rms_median": median_rms,
        "checks": checks,
        "passed": all(check["pass"] for check in checks.values()),
    }


def make_epoch_plans(
        episodes: int, *, epochs: int, batch_size: int,
        windows_per_batch: int, sequence_length: int,
        history: int, seed: int,
        ) -> tuple[tuple[EpochPlan, ...], str]:
    """Freeze shared twin batch orders and next-latent windows."""

    if min(episodes, epochs, batch_size, windows_per_batch, history) <= 0 \
            or sequence_length <= history:
        raise ValueError("invalid deterministic training-plan dimensions")
    candidates = sequence_length - history
    if windows_per_batch > candidates:
        raise ValueError("too many next-latent windows requested")
    rng = np.random.default_rng(seed)
    plans = []
    digest = hashlib.sha256()
    for _ in range(epochs):
        order = rng.permutation(episodes).astype(np.int64)
        starts = []
        digest.update(order.tobytes())
        for offset in range(0, episodes, batch_size):
            index = order[offset:offset + batch_size]
            if len(index) < 4:
                continue
            value = np.sort(rng.choice(
                candidates, size=windows_per_batch,
                replace=False)).astype(np.int64)
            starts.append(value)
            digest.update(value.tobytes())
        plans.append(EpochPlan(order, tuple(starts)))
    return tuple(plans), digest.hexdigest()


def reconstruction_metrics(
        prediction: np.ndarray, normalized_target: np.ndarray,
        ) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(normalized_target, dtype=np.float32)
    if prediction.shape != target.shape \
            or prediction.ndim != 2 or prediction.shape[1] != LATENT_DIM:
        raise ValueError("prediction/target must be matching (E,192) arrays")
    model_error = np.mean(np.square(prediction - target), axis=1)
    zero_error = np.mean(np.square(target), axis=1)
    model_mse = float(model_error.mean())
    zero_mse = float(zero_error.mean())
    if zero_mse <= 0:
        raise ValueError("zero-predictor baseline is degenerate")
    return {
        "mse": model_mse,
        "zero_predictor_mse": zero_mse,
        "normalized_mse_to_zero": model_mse / zero_mse,
        "r2_vs_training_mean": 1.0 - model_mse / zero_mse,
        "per_episode_mse": model_error.astype(np.float32),
        "per_episode_zero_mse": zero_error.astype(np.float32),
    }


__all__ = [
    "ACTION_DIM",
    "DECISION_INDEX",
    "EpochPlan",
    "LATENT_DIM",
    "SEQUENCE_LENGTH",
    "TargetStandardizer",
    "cue_residual_target",
    "development_health",
    "fit_target_standardizer",
    "load_label_free_bank",
    "make_epoch_plans",
    "reconstruction_metrics",
]
