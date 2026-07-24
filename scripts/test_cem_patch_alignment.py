"""Focused tests for patch masking and causal alignment."""
from __future__ import annotations

import torch

from lewm.models.spatial_memory_conditioner import (
    HISTORY,
    SpatialTokenBatch,
    masked_spatial_tokens,
)
from scripts.run_cem_patch_alignment import Variant, mask_for_batch
from scripts.run_cem_spatial_conditioner import FEATURE_DIM, META_DIM, TOKENS


def tokens(rows: int = 4) -> SpatialTokenBatch:
    return SpatialTokenBatch(
        feature=torch.randn(rows, TOKENS, FEATURE_DIM),
        delta=torch.randn(rows, TOKENS, FEATURE_DIM),
        coordinates=torch.randn(rows, TOKENS, 2),
        extent=torch.ones(rows, TOKENS, 2),
        metadata=torch.randn(rows, TOKENS, META_DIM),
        valid=torch.ones(rows, TOKENS, dtype=torch.bool),
        kind=torch.full((rows,), HISTORY, dtype=torch.long),
    )


def test_random_masks_are_reproducible_and_budget_preserving() -> None:
    memory = tokens()
    query = tokens()
    variant = Variant("random", "random", 0.5, False)
    first = mask_for_batch(variant, memory, query, seed=91)
    second = mask_for_batch(variant, memory, query, seed=91)
    assert torch.equal(first, second)
    assert first.sum(1).tolist() == [TOKENS // 2] * len(memory)
    masked = masked_spatial_tokens(memory, first)
    assert masked.feature.shape == memory.feature.shape
    assert torch.equal(masked.coordinates, memory.coordinates)
    assert torch.equal(masked.valid, memory.valid)


def test_semantic_mask_uses_patch_change_without_labels() -> None:
    memory = tokens()
    query = tokens()
    variant = Variant("semantic", "semantic", 0.25, False)
    mask = mask_for_batch(variant, memory, query, seed=0)
    assert mask.sum(1).tolist() == [TOKENS // 4] * len(memory)


def test_test_path_is_unmasked_by_contract() -> None:
    variant = Variant("causal", "none", 0.0, True)
    memory = tokens()
    assert not bool(mask_for_batch(variant, memory, memory, seed=0).any())
