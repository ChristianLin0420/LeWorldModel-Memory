"""CPU construction tests for the cue-only shell-game V3 amendment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.shell_game_capacity import (  # noqa: E402
    CAPACITY_STAGES,
    OfficialHostBaseBatch,
    ShellGameCapacityContract,
    audit_paired_counterfactual,
    build_admission_inputs,
    paired_counterfactual_batches,
)
from lewm.official_tasks.shell_game_capacity_v3 import (  # noqa: E402
    V3_SALIENCE,
    paired_counterfactual_batches_v3,
    v3_contract_description,
)
from lewm.official_tasks.shell_game_capacity_v2 import (  # noqa: E402
    paired_counterfactual_batches_v2,
)
from lewm.official_tasks.shell_game_pipeline_v3 import (  # noqa: E402
    development_selection_decision_v3,
    require_all_selected_salience_v3,
    require_selected_salience_v3,
    slice_admission_inputs_v3,
)
from lewm.tasks_v19.overlays import CUE_COLORS  # noqa: E402


def _base(episodes: int = 6, seed: int = 0) -> OfficialHostBaseBatch:
    rng = np.random.default_rng(seed)
    return OfficialHostBaseBatch(
        frames=rng.integers(
            0, 256, size=(episodes, 64, 64, 64, 3), dtype=np.uint8),
        actions=rng.normal(size=(episodes, 63, 10)).astype(np.float32),
        endo_state=rng.normal(size=(episodes, 64, 4)).astype(np.float32),
    )


@pytest.mark.parametrize("stage", CAPACITY_STAGES, ids=lambda item: item.key)
def test_v3_changes_only_cue_pixels_and_preserves_exact_audit(stage) -> None:
    base = _base(seed=81)
    contract = ShellGameCapacityContract(stage)
    v1_primary, v1_counterfactual = paired_counterfactual_batches(
        base, contract, seed=82)
    v2_primary, v2_counterfactual = paired_counterfactual_batches_v2(
        base, contract, seed=82)
    v3_primary, v3_counterfactual = paired_counterfactual_batches_v3(
        base, contract, seed=82)

    for name in (
            "actions", "endo_state", "initial_slots", "final_slots",
            "entity_x", "cue_on", "cue_off", "swap_pairs", "shuffle_off"):
        np.testing.assert_array_equal(
            getattr(v3_primary, name), getattr(v1_primary, name))
        np.testing.assert_array_equal(
            getattr(v3_counterfactual, name), getattr(v1_counterfactual, name))

    cue_mask = np.zeros(v3_primary.frames.shape[:2], dtype=bool)
    for episode in range(v3_primary.num_episodes):
        for start, stop in zip(
                v3_primary.cue_on[episode], v3_primary.cue_off[episode],
                strict=True):
            cue_mask[episode, int(start):int(stop)] = True
    np.testing.assert_array_equal(
        v3_primary.frames[~cue_mask], v1_primary.frames[~cue_mask])
    np.testing.assert_array_equal(
        v3_counterfactual.frames[~cue_mask],
        v1_counterfactual.frames[~cue_mask])
    np.testing.assert_array_equal(
        v3_primary.frames[~cue_mask], v2_primary.frames[~cue_mask])
    np.testing.assert_array_equal(
        v3_counterfactual.frames[~cue_mask],
        v2_counterfactual.frames[~cue_mask])
    assert np.any(v3_primary.frames[cue_mask] != v1_primary.frames[cue_mask])

    report = audit_paired_counterfactual(v3_primary, v3_counterfactual)
    assert report["overall_pass"] is True
    assert report["checks"]["off_cue_pixel_leakage"]["value"] == 0
    assert report["checks"]["final_legal_context_leakage"]["value"] == 0
    assert report["checks"]["all_item_targets_change"]["pass"] is True


def test_v3_cup_ball_and_three_slot_card_are_unambiguous() -> None:
    base = _base(3, seed=83)
    stage = CAPACITY_STAGES[-1]
    contract = ShellGameCapacityContract(stage)
    v1, _ = paired_counterfactual_batches(base, contract, seed=84)
    v3, _ = paired_counterfactual_batches_v3(base, contract, seed=84)
    width, _ = contract.cup_size

    for episode in range(v3.num_episodes):
        for item, (start, _) in enumerate(contract.cue_windows):
            color = np.asarray(CUE_COLORS[item], dtype=np.uint8)
            slot = int(v3.initial_slots[episode, item])
            center = contract.slot_x[slot]
            cup = v3.frames[
                episode, start, 0:contract.cup_y - contract.cue_lift
                + contract.cup_size[1],
                center - width // 2:center - width // 2 + width,
            ]
            assert np.all(cup == color), "the whole visible cued cup must be colored"
            np.testing.assert_array_equal(
                v3.frames[episode, start, contract.ball_y, center + 5], color)
            assert not np.array_equal(
                v1.frames[episode, start, contract.ball_y, center + 5], color)
            np.testing.assert_array_equal(
                v3.frames[episode, start, 40, 32], color)
            np.testing.assert_array_equal(
                v3.frames[episode, start, 50, center], color)
            for other_slot in set(range(3)) - {slot}:
                np.testing.assert_array_equal(
                    v3.frames[
                        episode, start, contract.cup_y + 4,
                        contract.slot_x[other_slot]],
                    np.asarray(contract.cup_color, dtype=np.uint8),
                )
                np.testing.assert_array_equal(
                    v3.frames[
                        episode, start, 50, contract.slot_x[other_slot]],
                    np.asarray(V3_SALIENCE.neutral_cell_color, dtype=np.uint8),
                )


def test_v3_retains_semantic_capacity_and_final_endpoint_contract() -> None:
    assert V3_SALIENCE.ball_radius == 6
    assert V3_SALIENCE.cue_card == "three_slot_spatial_panel"
    assert V3_SALIENCE.describe()["selection_candidates"] == 1
    for stage in CAPACITY_STAGES:
        contract = ShellGameCapacityContract(stage)
        description = v3_contract_description(contract)
        semantic = description["semantic_contract"]
        assert description["semantic_contract_changed_from_v1_or_v2"] is False
        assert semantic["stage"] == stage.key
        assert semantic["capacity"] == stage.capacity
        assert semantic["decision_index"] == 63
        assert semantic["decision_observation_excluded"] is True
        assert semantic["final_context_indices"] == [60, 61, 62]
        assert semantic["exact_set_chance"] == pytest.approx(3 ** -stage.capacity)


def test_development_partition_is_disjoint_and_complete() -> None:
    base = _base(6, seed=85)
    batch, _ = paired_counterfactual_batches_v3(
        base, ShellGameCapacityContract(CAPACITY_STAGES[1]), seed=86)
    inputs = build_admission_inputs(batch)
    fit = slice_admission_inputs_v3(inputs, 0, 3)
    check = slice_admission_inputs_v3(inputs, 3, 6)
    assert fit.cue_indices.shape[0] == check.cue_indices.shape[0] == 3
    np.testing.assert_array_equal(
        fit.cue_initial_slot_targets, inputs.cue_initial_slot_targets[:3])
    np.testing.assert_array_equal(
        check.cue_initial_slot_targets, inputs.cue_initial_slot_targets[3:])
    with pytest.raises(ValueError, match="invalid admission slice"):
        slice_admission_inputs_v3(inputs, 3, 3)


def test_formal_data_gate_fails_closed_without_development_receipt(tmp_path) -> None:
    spec = {
        "artifacts": {
            "root": str(tmp_path / "v3"),
            "development_cache": "development-cache",
        },
        "_lock_record": {
            "sha256": "lock",
            "spec_sha256": "spec",
        },
    }
    with pytest.raises(FileNotFoundError, match="formal data is blocked"):
        require_selected_salience_v3(spec, "single-item")
    assert not (tmp_path / "v3").exists()


def test_all_stage_gate_checks_every_semantic_stage(monkeypatch) -> None:
    visited = []

    def selected(_spec, stage):
        visited.append(stage)
        return {"stage": stage, "minimum_item_accuracy": 0.75}

    monkeypatch.setattr(
        "lewm.official_tasks.shell_game_pipeline_v3."
        "require_selected_salience_v3", selected)
    result = require_all_selected_salience_v3({
        "development_selection": {"all_stages_must_pass": True},
    })
    assert visited == [stage.key for stage in CAPACITY_STAGES]
    assert set(result) == set(visited)


def test_development_selection_retains_point_seven_five_threshold() -> None:
    diagnostic = {
        "gates": {
            "cue_initial_slot_availability": {
                "threshold": 0.75,
                "pass": True,
                "value": {
                    "minimum_item_accuracy": 0.8,
                    "per_item_accuracy": [0.82, 0.8],
                },
            },
            "paired_counterfactual_construction": {"pass": True},
        },
    }
    decision = development_selection_decision_v3(diagnostic, 0.75)
    assert decision["selected"] is True
    assert decision["threshold"] == 0.75
    with pytest.raises(ValueError, match="exactly 0.75"):
        development_selection_decision_v3(diagnostic, 0.74)
    diagnostic["gates"]["cue_initial_slot_availability"].update({
        "pass": False,
        "value": {
            "minimum_item_accuracy": 0.74,
            "per_item_accuracy": [0.9, 0.74],
        },
    })
    assert development_selection_decision_v3(
        diagnostic, 0.75)["selected"] is False
