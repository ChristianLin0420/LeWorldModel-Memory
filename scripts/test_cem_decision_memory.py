"""Focused tests for the decision-conditioned oracle task."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from scripts.build_cem_decision_task import source_audit
from scripts.run_cem_decision_memory import margin_and_accuracy
from scripts.run_graph_cem_long_gap import build_raw_gap


def test_paired_suffix_is_exact() -> None:
    latents = np.random.default_rng(0).normal(
        size=(4, 22, 8)
    ).astype(np.float32)
    actions = np.random.default_rng(1).normal(
        size=(4, 21, 2)
    ).astype(np.float32)
    raw = build_raw_gap(
        latents,
        actions,
        np.asarray([[0, 1]], dtype=np.int64),
        np.asarray([2], dtype=np.int64),
        "test",
        64,
    )
    assert np.array_equal(raw.history[0, -6:], raw.history[1, -6:])
    assert np.array_equal(
        raw.history_actions[0, -6:],
        raw.history_actions[1, -6:],
    )


def test_action_ranking_is_deterministic() -> None:
    losses = torch.tensor(
        [[0.4, 0.1, 0.5], [0.2, 0.3, 0.1]]
    )
    margin, accuracy, rank = margin_and_accuracy(
        losses, np.asarray([1, 0])
    )
    assert accuracy.tolist() == [1.0, 0.0]
    assert rank.tolist() == [1, 2]
    assert margin[0] > 0 and margin[1] < 0


def test_no_manual_or_graph_inputs() -> None:
    assert source_audit()["passed"]
    tree = ast.parse(
        Path(__file__).with_name("run_cem_decision_memory.py").read_text()
    )
    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert not loaded & {
        "cue_labels", "cue_positions", "cue_window", "goal_state",
    }
