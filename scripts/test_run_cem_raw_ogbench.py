"""Focused tests for the raw OGBench CEM protocol."""
from __future__ import annotations

import numpy as np
import torch

from lewm.models.cem_controller import VersionedEventStore
from scripts.run_cem_raw_ogbench import (
    ActionConditionedHost,
    QueryTensors,
    RawMemoryConditioner,
    rollout,
    source_contract_audit,
    split_indices,
)


def test_no_manual_cue_contract_has_no_forbidden_calls() -> None:
    contract = source_contract_audit()
    assert contract["passed"]
    assert contract["forbidden_call_sites"] == []
    assert contract["input_keys_consumed"] == ["frames", "actions"]
    assert contract["cue_window"] is None
    assert not contract["cue_window_used_by_model"]


def test_trajectory_split_is_disjoint_and_complete() -> None:
    train, validation, test = split_indices(97)
    assert len(set(train) & set(validation)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(validation) & set(test)) == 0
    assert sorted(np.concatenate([train, validation, test]).tolist()) == list(
        range(97)
    )


def test_version_store_keeps_old_event_until_new_version_passes() -> None:
    store = VersionedEventStore(budget=2, hysteresis=0.1)
    store.propose(event_id=0, key_id=3, event_timestamp=2, proposed_at=2)
    first = store.verify(
        event_id=0, verified_at=4, ce_hat=0.5, threshold=0.0
    )
    assert first["transition"] == "promoted"
    store.propose(event_id=1, key_id=3, event_timestamp=7, proposed_at=7)
    second = store.verify(
        event_id=1, verified_at=9, ce_hat=0.55, threshold=0.0
    )
    assert second["transition"] == "rejected"
    assert second["fallback_event_id"] == 0
    assert [record.event_id for record in store.active()] == [0]


def test_rollout_shapes_and_reset_matches_host() -> None:
    batch_size, context, horizon, slots = 3, 2, 2, 4
    latent_dim, action_dim = 8, 2
    host = ActionConditionedHost(latent_dim, action_dim, context, hidden=16)
    memory = RawMemoryConditioner(
        latent_dim, action_dim, hidden=16, budget=slots
    )
    query = QueryTensors(
        context_z=torch.randn(batch_size, context, latent_dim),
        action_history=torch.randn(batch_size, context, action_dim),
        future_actions=torch.randn(batch_size, horizon, action_dim),
        targets=torch.randn(batch_size, horizon, latent_dim),
        events=torch.randn(batch_size, slots, latent_dim),
        metadata=torch.rand(batch_size, slots, 3),
        valid=torch.ones(batch_size, slots, dtype=torch.bool),
        recent_event=torch.randn(batch_size, 1, latent_dim),
    )
    host_prediction, _ = rollout(host, None, query)
    reset_prediction, _ = rollout(
        host, memory, query, mask=torch.zeros_like(query.valid)
    )
    assert host_prediction.shape == (batch_size, horizon, latent_dim)
    assert torch.allclose(host_prediction, reset_prediction)
