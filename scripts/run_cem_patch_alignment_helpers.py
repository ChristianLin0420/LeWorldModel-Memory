"""Evaluation helpers for patch masking and causal alignment."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from lewm.models.spatial_memory_conditioner import (
    SpatialTokenBatch,
    masked_spatial_tokens,
)
from scripts.run_cem_spatial_conditioner import evaluate, rollout


@torch.no_grad()
def generate_deletion_targets(
    state: Any,
    model: torch.nn.Module,
    data: Any,
    batch_size: int,
) -> torch.Tensor:
    full, _ = evaluate(
        state, model, data, data.memory, batch_size,
    )
    targets = torch.zeros(
        (len(data), data.memory.valid.shape[1]),
        device=full.device,
    )
    for slot in range(data.memory.valid.shape[1]):
        valid = data.memory.valid.clone()
        valid[:, slot] = False
        deleted_memory = SpatialTokenBatch(
            feature=data.memory.feature,
            delta=data.memory.delta,
            coordinates=data.memory.coordinates,
            extent=data.memory.extent,
            metadata=data.memory.metadata,
            valid=valid,
            kind=data.memory.kind,
        )
        deleted, _ = evaluate(
            state, model, data, deleted_memory, batch_size,
        )
        targets[:, slot] = (
            deleted - full
        ) / data.recent_loss.clamp_min(1e-8)
    return targets.detach()


def _diagnostic_mask(
    variant: Any,
    memory: SpatialTokenBatch,
    query: SpatialTokenBatch,
    seed: int,
) -> torch.Tensor:
    mask = torch.zeros_like(memory.valid)
    if variant.mask_mode == "none":
        return mask
    count = max(1, int(round(memory.valid.shape[1] * variant.ratio)))
    if variant.mask_mode == "semantic":
        score = (memory.feature - query.feature).square().mean(-1)
    else:
        generator = torch.Generator(device=memory.feature.device)
        generator.manual_seed(seed)
        score = torch.rand(
            memory.valid.shape,
            generator=generator,
            device=memory.feature.device,
        )
    index = torch.topk(score, count, dim=1).indices
    mask.scatter_(1, index, True)
    return mask


@torch.no_grad()
def evaluate_variant(
    state: Any,
    model: torch.nn.Module,
    auxiliary: torch.nn.Module,
    data: Any,
    variant: Any,
    args: Any,
    causal_target: torch.Tensor,
    device: torch.device,
    *,
    validation: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    recent, _ = evaluate(
        state, model, data, data.recent, args.batch_size,
    )
    memory, telemetry = evaluate(
        state,
        model,
        data,
        data.memory,
        args.batch_size,
        telemetry=True,
    )
    opportunity = data.opportunity
    ordinary = ~opportunity
    gain = (recent - memory).cpu().numpy()
    raw_gain = (
        data.recent_loss - data.oracle_loss
    ).cpu().numpy()
    recovery = (
        float(
            gain[opportunity].mean()
            / max(raw_gain[opportunity].mean(), 1e-12)
        )
        if opportunity.any()
        else 0.0
    )
    degradation = (
        float(
            recent[ordinary].mean()
            / data.recent_loss[ordinary].mean().clamp_min(1e-8)
            - 1.0
        )
        if ordinary.any()
        else 0.0
    )
    mask = _diagnostic_mask(
        variant, data.memory, data.query, args.seed + 99_001,
    )
    if bool(mask.any()):
        masked = masked_spatial_tokens(data.memory, mask)
        rows = []
        for start in range(0, len(data), args.batch_size):
            stop = min(len(data), start + args.batch_size)
            index = torch.arange(start, stop, device=device)
            part = data.index(index)
            _, info = rollout(
                state.host,
                model,
                part.split.batch,
                part.query,
                masked.index(index),
            )
            prediction = auxiliary(info["query_code"])
            rows.append(
                F.mse_loss(
                    prediction[mask[index]],
                    data.memory.feature[index][mask[index]],
                    reduction="mean",
                ).detach()
            )
        reconstruction = float(torch.stack(rows).mean())
    else:
        reconstruction = 0.0
    diagnostics = ranking_diagnostics(
        telemetry["attention"].sum(1),
        causal_target.cpu().numpy(),
    )
    metrics = {
        "recovery": recovery,
        "ordinary_recent_degradation": degradation,
        "memory_mse_opportunity": float(memory[opportunity].mean()),
        "recent_mse_opportunity": float(recent[opportunity].mean()),
        "memory_gain": float(gain[opportunity].mean()),
        "reconstruction_mse": reconstruction,
        "attention_entropy": float(
            telemetry["attention_entropy"].mean()
        ),
        "attention_overlap": float(
            telemetry["attention_overlap"].mean()
        ),
        "patch_utilization": float(
            telemetry["patch_utilization"].mean()
        ),
        "identity_preservation": float(
            telemetry["identity_preservation"].mean()
        ),
        **diagnostics,
    }
    arrays = {
        "episode_id": data.split.episode_ids,
        "gap": data.split.gaps,
        "opportunity": opportunity,
        "loss_recent": recent.cpu().numpy(),
        "loss_memory": memory.cpu().numpy(),
        "loss_raw_recent": data.recent_loss.cpu().numpy(),
        "loss_raw_oracle": data.oracle_loss.cpu().numpy(),
    }
    return metrics, arrays, telemetry


def ranking_diagnostics(
    score: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | None]:
    from scipy.stats import spearmanr

    score = np.asarray(score)
    target = np.asarray(target)
    correlation = float(
        spearmanr(score.reshape(-1), target.reshape(-1)).statistic
    )
    correct = 0
    count = 0
    for row in range(len(score)):
        for first in range(score.shape[1]):
            for second in range(first + 1, score.shape[1]):
                truth = target[row, first] - target[row, second]
                if abs(float(truth)) <= 1e-8:
                    continue
                correct += int(
                    (score[row, first] - score[row, second]) * truth > 0
                )
                count += 1
    high = target[
        np.arange(len(target)),
        np.argmax(score, axis=1),
    ]
    rng = np.random.default_rng(73_001)
    random_index = rng.integers(0, score.shape[1], size=len(score))
    random_value = target[np.arange(len(target)), random_index]
    return {
        "patch_spearman": (
            correlation if np.isfinite(correlation) else None
        ),
        "patch_pairwise_accuracy": (
            float(correct / count) if count else None
        ),
        "high_effect_deletion": float(high.mean()),
        "random_deletion": float(random_value.mean()),
        "high_minus_random_deletion": float(
            (high - random_value).mean()
        ),
    }
