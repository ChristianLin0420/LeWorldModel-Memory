"""Spatial-token carrier plumbing for the frozen native DINO-WM host.

The official PushT predictor consumes three frames of 196 spatial tokens.  A
token has 404 channels: 384 frozen DINOv2 visual features followed by ten
proprioceptive and ten action channels.  This module keeps that contract
unchanged.  One parameter-shared carrier is applied independently to every
384-D patch stream, and only the resulting visual residual is injected before
the frozen predictor.

All helpers here are deliberately independent of the released checkpoint so
their causality, shape, parameter-count, and bootstrap contracts can be tested
on CPU before a metric-bearing run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from lewm.models.frozen_swap_carriers import Carrier


PATCHES = 196
VISUAL_DIM = 384
ACTION_DIM = 10
CONTEXT = 3


@dataclass(frozen=True)
class SpatialCarrierOutput:
    """Carrier outputs restored to ``(batch,time,patch,feature)`` order."""

    fused_visual: torch.Tensor
    prior_visual: torch.Tensor


def _validate_spatial_inputs(visual: torch.Tensor,
                             actions: torch.Tensor) -> tuple[int, int]:
    if visual.ndim != 4 or visual.shape[2:] != (PATCHES, VISUAL_DIM):
        raise ValueError(
            "visual must be (B,T,196,384), got " + str(tuple(visual.shape)))
    batch, time = visual.shape[:2]
    if time < 1 or actions.shape != (batch, time - 1, ACTION_DIM):
        raise ValueError(
            f"actions must be ({batch},{time - 1},10), got "
            f"{tuple(actions.shape)}")
    if visual.device != actions.device:
        raise ValueError("visual and actions must share a device")
    return batch, time


def spatial_carrier_forward(carrier: Carrier, visual: torch.Tensor,
                            actions: torch.Tensor) -> SpatialCarrierOutput:
    """Apply one tied carrier to all native DINO patch trajectories.

    Patch position is folded into the batch dimension, never pooled or
    permuted.  The same action stream is broadcast to every patch.
    """

    batch, time = _validate_spatial_inputs(visual, actions)
    flat_visual = visual.permute(0, 2, 1, 3).reshape(
        batch * PATCHES, time, VISUAL_DIM)
    flat_actions = actions[:, None].expand(
        batch, PATCHES, time - 1, ACTION_DIM).reshape(
            batch * PATCHES, time - 1, ACTION_DIM)
    output = carrier(flat_visual, flat_actions)

    expected = (batch * PATCHES, time, VISUAL_DIM)
    if output.z_tilde.shape != expected or output.prior_read.shape != expected:
        raise RuntimeError("carrier violated the spatial-token output contract")

    def restore(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch, PATCHES, time, VISUAL_DIM).permute(
            0, 2, 1, 3).contiguous()

    return SpatialCarrierOutput(
        fused_visual=restore(output.z_tilde),
        prior_visual=restore(output.prior_read),
    )


def endpoint_frame(last_cue_frame: int, evidence_age: int) -> int:
    if isinstance(last_cue_frame, bool) or last_cue_frame < 0:
        raise ValueError("last_cue_frame must be a non-negative integer")
    if isinstance(evidence_age, bool) or evidence_age < 1:
        raise ValueError("evidence_age must be a positive integer")
    return int(last_cue_frame + evidence_age)


def predictor_context_for_endpoint(endpoint: int) -> tuple[int, int, int]:
    """Return the three observed frames that predict ``endpoint``."""

    if isinstance(endpoint, bool) or endpoint < CONTEXT:
        raise ValueError("endpoint must be an integer >= 3")
    return tuple(range(endpoint - CONTEXT, endpoint))


def balanced_accuracy_from_predictions(prediction: np.ndarray,
                                       truth: np.ndarray,
                                       classes: int) -> float:
    prediction = np.asarray(prediction, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    if prediction.shape != truth.shape or prediction.ndim != 1:
        raise ValueError("prediction and truth must be equal-length vectors")
    if set(np.unique(truth)) != set(range(classes)):
        raise ValueError("truth does not contain every declared class")
    return float(np.mean([
        np.mean(prediction[truth == label] == label)
        for label in range(classes)
    ]))


def crossed_paired_bootstrap(
        left_prediction: np.ndarray, right_prediction: np.ndarray,
        truth: np.ndarray, *, classes: int, draws: int, seed: int,
        confidence: float = 0.95) -> dict[str, Any]:
    """Matched seed × class-stratified episode bootstrap.

    ``left_prediction`` and ``right_prediction`` are ``(seed,episode)``.
    Both seed indices and held-out episodes inside each class are resampled.
    The statistic is the equal-class balanced-accuracy contrast, averaged over
    matched carrier seeds.
    """

    left = np.asarray(left_prediction, dtype=np.int64)
    right = np.asarray(right_prediction, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 2 \
            or left.shape[1] != len(truth) or left.shape[0] < 1:
        raise ValueError("predictions must be aligned (seed,episode) arrays")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    class_rows = [np.flatnonzero(truth == label) for label in range(classes)]
    if any(len(rows) == 0 for rows in class_rows):
        raise ValueError("truth does not contain every declared class")

    left_correct = left == truth[None]
    right_correct = right == truth[None]
    per_seed_left = np.stack([
        left_correct[:, rows].mean(axis=1) for rows in class_rows], axis=1
    ).mean(axis=1)
    per_seed_right = np.stack([
        right_correct[:, rows].mean(axis=1) for rows in class_rows], axis=1
    ).mean(axis=1)
    point = float(np.mean(per_seed_left - per_seed_right))

    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    cursor = 0
    seed_count = left.shape[0]
    while cursor < draws:
        stop = min(draws, cursor + 128)
        count = stop - cursor
        sampled_seeds = rng.integers(
            0, seed_count, size=(count, seed_count))
        left_selected = left_correct[sampled_seeds]
        right_selected = right_correct[sampled_seeds]
        left_class, right_class = [], []
        for rows in class_rows:
            positions = rng.integers(0, len(rows), size=(count, len(rows)))
            episode_rows = rows[positions]
            episode_rows = np.broadcast_to(
                episode_rows[:, None, :],
                (count, seed_count, len(rows)))
            left_class.append(np.take_along_axis(
                left_selected, episode_rows, axis=2).mean(axis=(1, 2)))
            right_class.append(np.take_along_axis(
                right_selected, episode_rows, axis=2).mean(axis=(1, 2)))
        samples[cursor:stop] = (
            np.stack(left_class, axis=1).mean(axis=1)
            - np.stack(right_class, axis=1).mean(axis=1))
        cursor = stop

    alpha = (1.0 - confidence) / 2.0
    interval = np.quantile(samples, (alpha, 1.0 - alpha))
    return {
        "mean": point,
        "ci95": [float(interval[0]), float(interval[1])],
        "draws": int(draws),
        "seed": int(seed),
        "confidence": float(confidence),
        "paired": True,
        "units": ["matched carrier seed", "class-stratified held-out episode"],
        "ci_excludes_zero": bool(interval[0] > 0 or interval[1] < 0),
    }


def absolute_bootstrap(prediction: np.ndarray, truth: np.ndarray, *,
                       classes: int, draws: int, seed: int,
                       confidence: float = 0.95) -> dict[str, Any]:
    """Crossed bootstrap for an absolute balanced accuracy."""

    values = np.asarray(prediction, dtype=np.int64)
    impossible = np.full_like(values, fill_value=-1)
    result = crossed_paired_bootstrap(
        values, impossible, truth, classes=classes, draws=draws,
        seed=seed, confidence=confidence)
    result["metric"] = "balanced_accuracy"
    return result


__all__ = [
    "PATCHES", "VISUAL_DIM", "ACTION_DIM", "CONTEXT",
    "SpatialCarrierOutput", "spatial_carrier_forward", "endpoint_frame",
    "predictor_context_for_endpoint", "balanced_accuracy_from_predictions",
    "crossed_paired_bootstrap", "absolute_bootstrap",
]
