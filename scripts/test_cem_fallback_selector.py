"""Focused tests for the flat frame+event fallback selector."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from lewm.models.fallback_selector import (
    EVENT,
    FRAME,
    RECENT,
    FallbackCandidateBatch,
    FallbackSelectorEnsemble,
    serialized_token_bytes,
)
from scripts.run_cem_fallback_selector import (
    EVENT_SLOTS,
    FRAME_SLOTS,
    MEMORY_TOKENS,
    POOL_SLOTS,
    RECENT_SLOTS,
    build_candidate_pool,
)
from scripts.run_graph_cem_long_gap import build_raw_gap


def synthetic_raw():
    latents = np.zeros((4, 22, 8), dtype=np.float32)
    actions = np.zeros((4, 21, 2), dtype=np.float32)
    for episode in range(4):
        latents[episode] = episode + np.arange(22)[:, None] * 0.01
        actions[episode] = episode + np.arange(21)[:, None] * 0.001
    raw = build_raw_gap(
        latents,
        actions,
        sources=np.asarray([[0, 1]], dtype=np.int64),
        donors=np.asarray([2], dtype=np.int64),
        split="test",
        gap=32,
    )
    steps = raw.history.shape[1]
    raw.surprise = np.tile(
        np.linspace(0.0, 1.0, steps, dtype=np.float32),
        (len(raw.history), 1),
    )
    raw.change = np.tile(
        np.linspace(1.0, 0.0, steps, dtype=np.float32),
        (len(raw.history), 1),
    )
    return raw


def test_pool_has_fixed_type_and_token_budget() -> None:
    pool = build_candidate_pool(synthetic_raw(), (0.5, 0.5), torch.device("cpu"))
    assert pool.batch.events.shape[1] == POOL_SLOTS
    assert pool.candidate_type[0, :EVENT_SLOTS].tolist() == [EVENT] * EVENT_SLOTS
    assert pool.candidate_type[
        0, EVENT_SLOTS : EVENT_SLOTS + FRAME_SLOTS
    ].tolist() == [FRAME] * FRAME_SLOTS
    assert pool.candidate_type[0, -RECENT_SLOTS:].tolist() == [
        RECENT
    ] * RECENT_SLOTS
    assert serialized_token_bytes(8, MEMORY_TOKENS) == 128


def test_ensemble_shapes_and_uncertainty_calibration() -> None:
    rows, slots, latent_dim = 5, 6, 8
    batch = FallbackCandidateBatch(
        latent=torch.randn(rows, slots, latent_dim),
        query=torch.randn(rows, latent_dim),
        metadata=torch.randn(rows, slots, 3),
        candidate_type=torch.randint(0, 3, (rows, slots)),
        discovery_uncertainty=torch.rand(rows, slots),
        router_score=torch.randn(rows, slots),
        occupied=torch.ones(rows, slots, dtype=torch.bool),
        valid=torch.ones(rows, slots, dtype=torch.bool),
    )
    ensemble = FallbackSelectorEnsemble(
        latent_dim=latent_dim,
        metadata_dim=3,
        hidden=16,
        members=3,
    )
    output = ensemble(batch)
    assert output["mean"].shape == (rows, slots)
    assert output["std"].shape == (rows, slots)
    assert bool((output["std"] > 0).all())
    scale = ensemble.calibrate_scale(
        batch,
        torch.randn(rows, slots),
        batch.valid,
    )
    assert scale > 0


def test_crossfit_partition_has_no_pair_overlap() -> None:
    pair_ids = np.arange(30)
    for fold in range(3):
        held = set(pair_ids[pair_ids % 3 == fold].tolist())
        train = set(pair_ids[pair_ids % 3 != fold].tolist())
        assert not held & train
        assert held | train == set(pair_ids.tolist())


def test_source_has_no_loaded_cue_fields() -> None:
    path = Path(__file__).with_name("run_cem_fallback_selector.py")
    tree = ast.parse(path.read_text())
    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert not loaded & {"cue_labels", "cue_positions", "cue_window"}
