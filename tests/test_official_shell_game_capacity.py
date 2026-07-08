"""Construction tests for the official-host shell-game capacity contract."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.shell_game_capacity import (
    CAPACITY_STAGES,
    CounterfactualAuditError,
    OfficialHostBaseBatch,
    ShellGameCapacityContract,
    audit_paired_counterfactual,
    build_admission_inputs,
    gather_sequence,
    get_capacity_stage,
    paired_counterfactual_batches,
    require_paired_counterfactual,
)


def _base(episodes: int = 6, seed: int = 0) -> OfficialHostBaseBatch:
    rng = np.random.default_rng(seed)
    return OfficialHostBaseBatch(
        frames=rng.integers(
            0, 256, size=(episodes, 64, 64, 64, 3), dtype=np.uint8),
        actions=rng.normal(size=(episodes, 63, 10)).astype(np.float32),
        endo_state=rng.normal(size=(episodes, 64, 4)).astype(np.float32),
    )


def test_capacity_stages_are_semantic_and_have_exact_set_chance() -> None:
    assert [stage.capacity for stage in CAPACITY_STAGES] == [1, 2, 4]
    assert [stage.exact_set_chance for stage in CAPACITY_STAGES] == [
        1 / 3, 1 / 9, 1 / 81]
    for stage in CAPACITY_STAGES:
        assert not re.fullmatch(r"[tT]\d+", stage.key)
        assert "shell-game recall" in stage.display_name
        assert get_capacity_stage(stage.key) is stage
        assert get_capacity_stage(stage.display_name) is stage


def test_official_timing_and_action_contract() -> None:
    contract = ShellGameCapacityContract(get_capacity_stage("four-item"))
    description = contract.describe()
    assert description["cue_windows"] == [[4, 7], [8, 11], [12, 15], [16, 19]]
    assert description["swap_times"] == [24, 32, 40, 48]
    assert description["shuffle_off"] == 52
    assert description["decision_index"] == 63
    assert description["decision_observation_excluded"] is True
    assert description["final_context_indices"] == [60, 61, 62]
    assert description["official_action_block_dim"] == 10

    base = _base(3)
    with pytest.raises(ValueError, match="native float32 official 5x2"):
        OfficialHostBaseBatch(
            frames=base.frames,
            actions=np.zeros((3, 63, 2), dtype=np.float32),
            endo_state=base.endo_state,
        )


@pytest.mark.parametrize("stage", CAPACITY_STAGES, ids=lambda stage: stage.key)
def test_paired_counterfactual_is_deterministic_and_leakage_free(stage) -> None:
    base = _base(6, seed=11)
    contract = ShellGameCapacityContract(stage)
    first = paired_counterfactual_batches(base, contract, seed=29)
    second = paired_counterfactual_batches(base, contract, seed=29)
    for first_branch, second_branch in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_branch.frames, second_branch.frames)
        np.testing.assert_array_equal(
            first_branch.initial_slots, second_branch.initial_slots)
        np.testing.assert_array_equal(
            first_branch.final_slots, second_branch.final_slots)

    primary, counterfactual = first
    report = require_paired_counterfactual(primary, counterfactual)
    assert report["overall_pass"] is True
    assert report["display_name"] == stage.display_name
    assert all(check["pass"] for check in report["checks"].values())
    assert np.all(primary.initial_slots != counterfactual.initial_slots)
    assert np.all(primary.final_slots != counterfactual.final_slots)
    np.testing.assert_array_equal(primary.actions, counterfactual.actions)
    np.testing.assert_array_equal(primary.entity_x, counterfactual.entity_x)


def test_capacity_stages_are_nested_for_a_shared_seed() -> None:
    base = _base(6, seed=17)
    batches = {}
    for stage in CAPACITY_STAGES:
        contract = ShellGameCapacityContract(stage)
        batches[stage.capacity] = paired_counterfactual_batches(
            base, contract, seed=31)[0]
    np.testing.assert_array_equal(
        batches[1].initial_slots, batches[4].initial_slots[:, :1])
    np.testing.assert_array_equal(
        batches[2].initial_slots, batches[4].initial_slots[:, :2])
    np.testing.assert_array_equal(
        batches[1].final_slots, batches[4].final_slots[:, :1])
    np.testing.assert_array_equal(
        batches[2].final_slots, batches[4].final_slots[:, :2])
    np.testing.assert_array_equal(batches[1].swap_pairs, batches[4].swap_pairs)


def test_primary_final_targets_are_coordinate_balanced() -> None:
    base = _base(12, seed=5)
    contract = ShellGameCapacityContract(get_capacity_stage("four-item"))
    primary, _ = paired_counterfactual_batches(base, contract, seed=7)
    for item in range(contract.stage.capacity):
        assert np.bincount(primary.final_slots[:, item], minlength=3).tolist() \
            == [4, 4, 4]


def test_admission_coordinates_are_legal_and_gatherable() -> None:
    base = _base(6, seed=3)
    contract = ShellGameCapacityContract(get_capacity_stage("two-item"))
    primary, _ = paired_counterfactual_batches(base, contract, seed=4)
    inputs = build_admission_inputs(primary)
    assert inputs.cue_indices.shape == (6, 2, 3)
    np.testing.assert_array_equal(inputs.cue_indices[0, 0], [4, 5, 6])
    np.testing.assert_array_equal(inputs.cue_indices[0, 1], [8, 9, 10])
    np.testing.assert_array_equal(inputs.swap_indices[0], [26, 34, 42, 50])
    np.testing.assert_array_equal(inputs.final_context_indices[0], [60, 61, 62])
    assert inputs.final_context_indices.max() < contract.decision_index
    assert inputs.post_shuffle_indices.min() >= contract.shuffle_off + 2
    assert inputs.post_shuffle_indices.max() < contract.decision_index

    time_code = np.broadcast_to(
        np.arange(64, dtype=np.float32)[None, :, None], (6, 64, 1))
    gathered = gather_sequence(time_code, inputs.cue_indices)
    np.testing.assert_array_equal(gathered[..., 0], inputs.cue_indices)


def test_counterfactual_audit_detects_one_postcue_pixel_leak() -> None:
    base = _base(3, seed=9)
    contract = ShellGameCapacityContract(get_capacity_stage("single-item"))
    primary, counterfactual = paired_counterfactual_batches(
        base, contract, seed=10)
    counterfactual.frames[0, 62, 20, 20, 0] ^= np.uint8(1)
    report = audit_paired_counterfactual(primary, counterfactual)
    assert report["overall_pass"] is False
    assert report["checks"]["off_cue_pixel_leakage"]["pass"] is False
    assert report["checks"]["final_legal_context_leakage"]["pass"] is False
    with pytest.raises(CounterfactualAuditError, match="audit failed"):
        require_paired_counterfactual(primary, counterfactual)
