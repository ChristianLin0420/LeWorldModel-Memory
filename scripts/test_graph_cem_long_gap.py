"""Focused tests for the controlled suffix-collision oracle ladder."""
from __future__ import annotations

import numpy as np
import torch

from scripts.run_cem_raw_ogbench import QueryTensors, RawMemoryConditioner
from scripts.run_graph_cem_long_gap import (
    MEMORY_TOKENS,
    build_raw_gap,
    route_topk,
)


def synthetic_sources() -> tuple[np.ndarray, np.ndarray]:
    latents = np.zeros((4, 22, 5), dtype=np.float32)
    actions = np.zeros((4, 21, 2), dtype=np.float32)
    for episode in range(4):
        latents[episode] = episode + np.arange(22)[:, None] * 0.01
        actions[episode] = episode + np.arange(21)[:, None] * 0.001
    return latents, actions


def test_paired_branches_have_exact_recent_suffixes() -> None:
    latents, actions = synthetic_sources()
    raw = build_raw_gap(
        latents,
        actions,
        sources=np.asarray([[0, 1]], dtype=np.int64),
        donors=np.asarray([2], dtype=np.int64),
        split="test",
        gap=32,
    )
    assert np.array_equal(raw.history[0, -6:], raw.history[1, -6:])
    assert np.array_equal(
        raw.history_actions[0, -6:], raw.history_actions[1, -6:]
    )
    assert not np.array_equal(raw.history[0, 2:4], raw.history[1, 2:4])
    assert not np.array_equal(raw.targets[0], raw.targets[1])
    assert raw.pair_ids.tolist() == [0, 0]
    assert raw.branches.tolist() == [0, 1]


def test_route_topk_reallocates_full_read_budget_after_deletion() -> None:
    rows, slots, latent_dim = 2, 7, 8
    action_dim = 3
    memory = RawMemoryConditioner(
        latent_dim, action_dim, hidden=16, budget=slots
    )
    batch = QueryTensors(
        context_z=torch.randn(rows, 2, latent_dim),
        action_history=torch.randn(rows, 2, action_dim),
        future_actions=torch.randn(rows, 4, action_dim),
        targets=torch.randn(rows, 4, latent_dim),
        events=torch.randn(rows, slots, latent_dim),
        metadata=torch.rand(rows, slots, 3),
        valid=torch.ones(rows, slots, dtype=torch.bool),
        recent_event=torch.randn(rows, 1, latent_dim),
    )
    selected = route_topk(memory, batch, MEMORY_TOKENS)
    assert selected.sum(1).tolist() == [MEMORY_TOKENS, MEMORY_TOKENS]
    excluded = torch.zeros_like(selected)
    excluded[:, 0] = True
    rerouted = route_topk(memory, batch, MEMORY_TOKENS, excluded)
    assert rerouted.sum(1).tolist() == [MEMORY_TOKENS, MEMORY_TOKENS]
    assert not bool((rerouted & excluded).any())
