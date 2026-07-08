"""CPU-only frozen-feature admission tests for shell-game capacity tasks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.shell_game_admission import evaluate_frozen_admission
from lewm.official_tasks.shell_game_capacity import (
    OfficialHostBaseBatch,
    ShellGameCapacityContract,
    build_admission_inputs,
    get_capacity_stage,
    paired_counterfactual_batches,
)


def _pair(episodes: int, seed: int):
    rng = np.random.default_rng(seed)
    base = OfficialHostBaseBatch(
        frames=np.zeros((episodes, 64, 64, 64, 3), dtype=np.uint8),
        actions=rng.normal(size=(episodes, 63, 10)).astype(np.float32),
        endo_state=rng.normal(size=(episodes, 64, 4)).astype(np.float32),
    )
    contract = ShellGameCapacityContract(get_capacity_stage("four-item"))
    return paired_counterfactual_batches(base, contract, seed=seed)


def _synthetic_frozen_latents(batch, *, cue: bool = True,
                              swap: bool = True,
                              final_leak: bool = False) -> np.ndarray:
    """Linearly expose selected registered coordinates and nothing else."""

    inputs = build_admission_inputs(batch)
    latents = np.zeros((batch.num_episodes, 64, 24), dtype=np.float32)
    for episode in range(batch.num_episodes):
        if cue:
            for item in range(inputs.capacity):
                slot = int(batch.initial_slots[episode, item])
                latents[episode, inputs.cue_indices[episode, item], slot] = 5.0
        if swap:
            for swap_index, time in enumerate(inputs.swap_indices[episode]):
                pair = int(batch.swap_pairs[episode, swap_index])
                latents[episode, time, 3 + pair] = 5.0
        if final_leak:
            for item in range(inputs.capacity):
                slot = int(batch.final_slots[episode, item])
                coordinate = 6 + item * 3 + slot
                latents[
                    episode, inputs.final_context_indices[episode], coordinate
                ] = 5.0
                latents[
                    episode, inputs.post_shuffle_indices[episode], coordinate
                ] = 5.0
    return latents


def _evaluate(train_pair, validation_pair, train_latents, validation_latents):
    return evaluate_frozen_admission(
        train_latents=train_latents,
        train_primary=train_pair[0],
        train_counterfactual=train_pair[1],
        validation_latents=validation_latents,
        validation_primary=validation_pair[0],
        validation_counterfactual=validation_pair[1],
    )


def test_frozen_admission_passes_with_cue_and_swap_signal_only() -> None:
    train_pair = _pair(60, seed=41)
    validation_pair = _pair(30, seed=42)
    report = _evaluate(
        train_pair,
        validation_pair,
        _synthetic_frozen_latents(train_pair[0]),
        _synthetic_frozen_latents(validation_pair[0]),
    )
    assert report["admitted"] is True
    assert report["representation_frozen"] is True
    assert report["world_model_training_performed"] is False
    assert report["availability_target"] == "initial slot for each cued item"
    assert report["per_item_chance"] == 1 / 3
    assert report["exact_set_chance"] == 1 / 81
    assert all(gate["pass"] for gate in report["gates"].values())
    cue = report["gates"]["cue_initial_slot_availability"]["value"]
    assert cue["per_item_accuracy"] == [1.0, 1.0, 1.0, 1.0]
    assert report["gates"]["swap_pair_visibility"]["value"] == 1.0


def test_frozen_admission_fails_closed_without_cue_availability() -> None:
    train_pair = _pair(60, seed=43)
    validation_pair = _pair(30, seed=44)
    report = _evaluate(
        train_pair,
        validation_pair,
        _synthetic_frozen_latents(train_pair[0], cue=False),
        _synthetic_frozen_latents(validation_pair[0], cue=False),
    )
    assert report["admitted"] is False
    assert report["gates"]["paired_counterfactual_construction"]["pass"] is True
    assert report["gates"]["cue_initial_slot_availability"]["pass"] is False
    assert report["gates"]["swap_pair_visibility"]["pass"] is True


def test_frozen_admission_rejects_final_context_label_leakage() -> None:
    train_pair = _pair(60, seed=45)
    validation_pair = _pair(30, seed=46)
    report = _evaluate(
        train_pair,
        validation_pair,
        _synthetic_frozen_latents(train_pair[0], final_leak=True),
        _synthetic_frozen_latents(validation_pair[0], final_leak=True),
    )
    assert report["admitted"] is False
    assert report["gates"]["cue_initial_slot_availability"]["pass"] is True
    assert report["gates"]["swap_pair_visibility"]["pass"] is True
    assert report["gates"]["post_shuffle_target_leakage"]["pass"] is False
    assert report["gates"]["final_context_target_leakage"]["pass"] is False
