"""Focused tests for the non-default flat conditional-CE diagnostic."""
from __future__ import annotations

import numpy as np
import torch

from scripts.run_cem_conditional_ce import (
    ConditionalCEHead,
    ReplayScores,
    conditional_source_audit,
    normalized_conditional_effect,
    replay_store_masks,
)
from scripts.run_cem_raw_ogbench import (
    Event,
    QuerySample,
    QueryTensors,
    StoreConfig,
)


def test_normalized_conditional_effect_uses_full_store_loss() -> None:
    deleted = torch.tensor([5.0, 1.0])
    full = torch.tensor([3.0, 2.0])
    empty = torch.tensor([4.0, 2.0])
    value = normalized_conditional_effect(deleted, full, empty)
    assert torch.allclose(value, torch.tensor([0.5, -0.5]))


def test_replay_after_version_deletion_restores_fallback() -> None:
    first = Event(
        event_id=0,
        start=1,
        end=1,
        peak_t=1,
        proposal_score=1.0,
        surprise=1.0,
        semantic_change=1.0,
        vector=np.zeros(3, dtype=np.float32),
        key_id=4,
    )
    second = Event(
        event_id=1,
        start=4,
        end=4,
        peak_t=4,
        proposal_score=2.0,
        surprise=2.0,
        semantic_change=2.0,
        vector=np.ones(3, dtype=np.float32),
        key_id=4,
    )
    query = QuerySample(
        episode_id=0,
        query_t=8,
        context_z=np.zeros((2, 3), dtype=np.float32),
        action_history=np.zeros((2, 1), dtype=np.float32),
        future_actions=np.zeros((2, 1), dtype=np.float32),
        targets=np.zeros((2, 3), dtype=np.float32),
        events=[first, second],
        recent_event=np.zeros(3, dtype=np.float32),
    )
    batch = QueryTensors(
        context_z=torch.zeros(1, 2, 3),
        action_history=torch.zeros(1, 2, 1),
        future_actions=torch.zeros(1, 2, 1),
        targets=torch.zeros(1, 2, 3),
        events=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]),
        metadata=torch.zeros(1, 2, 3),
        valid=torch.ones(1, 2, dtype=torch.bool),
        recent_event=torch.zeros(1, 1, 3),
    )
    scores = ReplayScores(
        ce_hat=torch.tensor([[0.4, 0.8]]),
        route=torch.tensor([[0.1, 0.2]]),
        combined=torch.tensor([[0.4, 0.8]]),
    )
    config = StoreConfig(
        promotion_threshold=0.0,
        hysteresis=0.0,
        budget=2,
        topk=1,
        verification_delay=1,
    )
    full, _, _ = replay_store_masks(batch, [query], config, scores)
    assert full.tolist() == [[False, True]]
    excluded = torch.tensor([[False, True]])
    deleted, occupied, telemetry = replay_store_masks(
        batch, [query], config, scores, excluded
    )
    assert deleted.tolist() == [[True, False]]
    assert occupied.tolist() == [[True, False]]
    assert telemetry["fallback_versions_after_deletion"] == 1
    assert telemetry["dependency_payload_leaks"] == 0


def test_pair_head_is_symmetric() -> None:
    head = ConditionalCEHead(input_dim=7, hidden=12)
    _, _, hidden = head(torch.randn(3, 4, 7))
    mean, variance = head.pair(hidden)
    assert mean.shape == (3, 4, 4)
    assert variance.shape == (3, 4, 4)
    assert torch.allclose(mean, mean.transpose(1, 2), atol=1e-6)


def test_conditional_source_contract_is_label_free() -> None:
    audit = conditional_source_audit()
    assert audit["passed"]
    assert audit["forbidden_loaded_names"] == []
    assert not audit["test_future_used_for_training"]
