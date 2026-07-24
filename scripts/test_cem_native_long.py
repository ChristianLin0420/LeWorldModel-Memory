"""Focused tests for native long-trajectory memory and abstaining activation."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from lewm.models.native_long_memory import (
    FRAME,
    RECENT,
    MemoryTokenBatch,
    NativeMemoryConditioner,
    UtilityGateEnsemble,
)
from scripts.build_cem_native_long import (
    CONTEXT,
    HORIZON,
    _query_records,
    source_audit,
)
from scripts.run_cem_native_long import MEMORY_TOKENS, choose_memory


def test_query_mining_uses_only_causal_prefix() -> None:
    latents = np.random.default_rng(0).normal(size=(3, 140, 8)).astype(
        np.float32
    )
    actions = np.random.default_rng(1).normal(size=(3, 139, 2)).astype(
        np.float32
    )
    records = _query_records(
        latents,
        actions,
        np.asarray([0, 1], dtype=np.int64),
        gap=128,
    )
    assert records
    assert all(int(row["query_t"]) + HORIZON < 140 for row in records)
    assert all(int(row["query_t"]) - int(row["gap"]) >= 0 for row in records)
    for row in records:
        modified = latents.copy()
        episode = int(row["episode_id"])
        query_t = int(row["query_t"])
        modified[episode, query_t + 1 :] += 1000.0
        repeated = _query_records(
            modified,
            actions,
            np.asarray([0, 1], dtype=np.int64),
            gap=128,
        )
        matched = next(
            value
            for value in repeated
            if int(value["episode_id"]) == episode
            and int(value["query_t"]) == query_t
        )
        assert row == matched


def test_conditioner_has_exact_zero_memory_path_and_zero_init() -> None:
    rows, latent_dim, action_dim = 4, 12, 3
    conditioner = NativeMemoryConditioner(
        latent_dim,
        action_dim,
        metadata_dim=7,
        code_dim=16,
        hidden=24,
        heads=4,
    )
    base = torch.randn(rows, latent_dim)
    context = torch.randn(rows, CONTEXT, latent_dim)
    action = torch.randn(rows, action_dim)
    memory = MemoryTokenBatch(
        values=torch.randn(rows, MEMORY_TOKENS, latent_dim),
        metadata=torch.randn(rows, MEMORY_TOKENS, 7),
        token_type=torch.randint(0, 3, (rows, MEMORY_TOKENS)),
        valid=torch.ones(rows, MEMORY_TOKENS, dtype=torch.bool),
    )
    no_memory, _ = conditioner(base, context, action, None)
    initialized, telemetry = conditioner(base, context, action, memory)
    assert torch.equal(no_memory, base)
    assert torch.equal(initialized, base)
    assert telemetry["token_code"].shape == (rows, MEMORY_TOKENS, 16)


def test_recent_and_memory_choices_have_identical_budget() -> None:
    rows, latent_dim = 5, 8

    def tokens(type_id: int) -> MemoryTokenBatch:
        return MemoryTokenBatch(
            values=torch.randn(rows, MEMORY_TOKENS, latent_dim),
            metadata=torch.randn(rows, MEMORY_TOKENS, 7),
            token_type=torch.full((rows, MEMORY_TOKENS), type_id),
            valid=torch.ones(rows, MEMORY_TOKENS, dtype=torch.bool),
        )

    recent = tokens(RECENT)
    historical = tokens(FRAME)
    selected = choose_memory(
        torch.tensor([True, False, True, False, True]),
        historical,
        recent,
    )
    assert selected.values.shape == recent.values.shape
    assert selected.values.element_size() * selected.values[0].numel() == (
        recent.values.element_size() * recent.values[0].numel()
    )
    assert torch.equal(selected.values[1], recent.values[1])
    assert torch.equal(selected.values[0], historical.values[0])


def test_utility_gate_produces_abstaining_lower_bound() -> None:
    ensemble = UtilityGateEnsemble(9, hidden=16, members=3)
    features = torch.randn(7, 9)
    lower, output = ensemble.lower_confidence_bound(features)
    assert lower.shape == (7,)
    assert output["probability"].shape == (7,)
    assert bool((output["std"] > 0).all())
    assert bool(((output["probability"] >= 0) & (output["probability"] <= 1)).all())


def test_source_contract_and_no_graph_import() -> None:
    assert source_audit()["passed"]
    path = Path(__file__).with_name("run_cem_native_long.py")
    tree = ast.parse(path.read_text())
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any("graph_cem" in name for name in imports)


def test_gpu3_rejected_by_contract() -> None:
    from scripts.run_cem_native_long import resolve_device

    try:
        resolve_device(3)
    except ValueError as error:
        assert "GPU3" in str(error)
    else:
        raise AssertionError("GPU3 must be rejected")
