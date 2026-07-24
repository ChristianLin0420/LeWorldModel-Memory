"""Focused tests for the spatial Gate-B conditioner."""
from __future__ import annotations

import torch

from lewm.models.spatial_memory_conditioner import (
    HISTORY,
    QUERY,
    SpatialMemoryConditioner,
    SpatialTokenBatch,
    serialized_spatial_bytes,
)
from scripts.run_cem_spatial_conditioner import (
    FEATURE_DIM,
    META_DIM,
    TOKENS,
    audit_source,
)


def token_batch(rows: int, kind: int) -> SpatialTokenBatch:
    return SpatialTokenBatch(
        feature=torch.randn(rows, TOKENS, FEATURE_DIM),
        delta=torch.zeros(rows, TOKENS, FEATURE_DIM),
        coordinates=torch.rand(rows, TOKENS, 2) * 2.0 - 1.0,
        extent=torch.full((rows, TOKENS, 2), 0.25),
        metadata=torch.randn(rows, TOKENS, META_DIM),
        valid=torch.ones(rows, TOKENS, dtype=torch.bool),
        kind=torch.full((rows,), kind, dtype=torch.long),
    )


def test_spatial_shapes_and_empty_identity() -> None:
    rows, host_dim, action_dim = 3, 24, 5
    model = SpatialMemoryConditioner(
        host_dim,
        action_dim,
        FEATURE_DIM,
        META_DIM,
        TOKENS,
        code_dim=32,
        hidden=48,
        heads=4,
    )
    base = torch.randn(rows, host_dim)
    context = torch.randn(rows, 4, host_dim)
    action = torch.randn(rows, action_dim)
    query = token_batch(rows, QUERY)
    memory = token_batch(rows, HISTORY)
    empty, _ = model(base, context, action, query, None)
    initialized, telemetry = model(base, context, action, query, memory)
    assert torch.equal(empty, base)
    assert torch.equal(initialized, base)
    assert telemetry["attention"].shape == (rows, TOKENS, TOKENS)


def test_fixed_serialized_budget() -> None:
    expected = TOKENS * (2 * FEATURE_DIM + 2 + 2 + META_DIM) * 4
    assert serialized_spatial_bytes(TOKENS, FEATURE_DIM, META_DIM) == expected


def test_source_contract() -> None:
    audit = audit_source()
    assert audit["passed"]
    assert not audit["manual_labels"]
    assert not audit["event_discovery_modified"]
    assert not audit["gate_a_modified"]
