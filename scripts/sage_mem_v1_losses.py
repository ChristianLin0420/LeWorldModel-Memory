#!/usr/bin/env python3
"""Small, auditable label-free objectives for SAGE-Mem v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


class LabelFreeLossError(ValueError):
    """Loss inputs violate the preregistered label-free boundary."""


@dataclass(frozen=True)
class LossWeights:
    next_feature: float = 1.0
    exposure_alignment: float = 0.10
    past_feature_replay: float = 0.10

    def __post_init__(self) -> None:
        if any(not isinstance(value, (int, float)) or value < 0
               for value in (self.next_feature, self.exposure_alignment,
                             self.past_feature_replay)):
            raise LabelFreeLossError("loss weights must be finite and non-negative")


def _pair(prediction: torch.Tensor, target: torch.Tensor,
          label: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(prediction, torch.Tensor) \
            or not isinstance(target, torch.Tensor):
        raise LabelFreeLossError(f"{label} inputs must be tensors")
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise LabelFreeLossError(f"{label} shapes must be equal and non-empty")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise LabelFreeLossError(f"{label} inputs must be finite")
    return prediction, target.detach()


def next_feature_mse(prediction: torch.Tensor,
                     frozen_next_feature: torch.Tensor) -> torch.Tensor:
    prediction, target = _pair(prediction, frozen_next_feature, "next-feature")
    return F.mse_loss(prediction, target)


def exposure_alignment_loss(exposure: torch.Tensor,
                            frozen_host_output: torch.Tensor) -> torch.Tensor:
    exposure, target = _pair(exposure, frozen_host_output, "exposure")
    exposure = F.normalize(exposure, dim=-1, eps=1e-8)
    target = F.normalize(target, dim=-1, eps=1e-8)
    return (1.0 - (exposure * target).sum(dim=-1)).mean()


def past_feature_replay_loss(replayed: torch.Tensor,
                             frozen_past_feature: torch.Tensor) -> torch.Tensor:
    replayed, target = _pair(replayed, frozen_past_feature, "past-feature")
    return F.smooth_l1_loss(replayed, target)


def compose_label_free_loss(*, next_prediction: torch.Tensor,
                            frozen_next_feature: torch.Tensor,
                            exposure: torch.Tensor,
                            frozen_host_output: torch.Tensor,
                            replayed_past: torch.Tensor,
                            frozen_past_feature: torch.Tensor,
                            weights: LossWeights = LossWeights(),
                            metadata: Mapping[str, Any] | None = None
                            ) -> tuple[torch.Tensor, dict[str, float]]:
    """Compose losses; reject any semantic-label metadata fail-closed."""
    forbidden = {"label", "labels", "semantic_label", "oracle_state",
                 "class_id", "goal_id", "color_id", "token_id"}
    keys = set(metadata or {})
    found = forbidden.intersection(keys)
    if found:
        raise LabelFreeLossError(
            f"semantic metadata reached training loss: {sorted(found)}")
    next_loss = next_feature_mse(next_prediction, frozen_next_feature)
    exposure_loss = exposure_alignment_loss(exposure, frozen_host_output)
    replay_loss = past_feature_replay_loss(replayed_past, frozen_past_feature)
    total = (weights.next_feature * next_loss
             + weights.exposure_alignment * exposure_loss
             + weights.past_feature_replay * replay_loss)
    if not torch.isfinite(total):
        raise LabelFreeLossError("composed loss is not finite")
    metrics = {
        "next_feature_mse": float(next_loss.detach()),
        "exposure_alignment": float(exposure_loss.detach()),
        "past_feature_replay": float(replay_loss.detach()),
        "total": float(total.detach()),
        "labels_used": 0.0,
    }
    return total, metrics
