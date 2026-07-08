"""Cue-only V3 salience amendment for the official shell-game contract.

V2 preserved the exact task and leakage contract but its cue remained below
the frozen-encoder availability threshold on the development bank.  V3 has
exactly one predeclared visual candidate: during an item's existing cue
window, a large three-cell spatial cue card is shown in the lower image.  The
left/middle/right cells correspond to the unchanged three initial cup slots;
the target cell and a header band use the item's color.  The lifted cup and
ball retain the V2 item coloring.

The renderer starts from the locked V1 batch and changes only pixels in the
already registered cue windows.  All actions, simulator state, timing,
nuisance, targets, post-cue pixels, and legal final context are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from lewm.official_tasks.shell_game_capacity import (
    OfficialHostBaseBatch,
    ShellGameCapacityBatch,
    ShellGameCapacityContract,
    paired_counterfactual_batches,
)
from lewm.tasks_v19.overlays import CUE_COLORS, draw_disc, draw_rect


SALIENCE_SCHEMA = "official_shell_game_cue_salience_v3"


@dataclass(frozen=True)
class ShellGameCueSalienceV3:
    """The single immutable V3 development candidate."""

    cued_cup_fill: str = "item_color"
    ball_radius: int = 6
    cue_card: str = "three_slot_spatial_panel"
    card_bounds: tuple[int, int, int, int] = (2, 36, 62, 62)
    header_bounds: tuple[int, int, int, int] = (4, 38, 60, 43)
    cell_width: int = 16
    cell_bounds_y: tuple[int, int] = (45, 60)
    neutral_cell_color: tuple[int, int, int] = (92, 94, 104)
    card_background: tuple[int, int, int] = (25, 28, 34)
    other_cups: str = "unchanged_neutral_identical"
    off_cue_pixels: str = "byte_identical_to_v1_and_v2"

    def __post_init__(self) -> None:
        expected = {
            "cued_cup_fill": "item_color",
            "ball_radius": 6,
            "cue_card": "three_slot_spatial_panel",
            "card_bounds": (2, 36, 62, 62),
            "header_bounds": (4, 38, 60, 43),
            "cell_width": 16,
            "cell_bounds_y": (45, 60),
            "neutral_cell_color": (92, 94, 104),
            "card_background": (25, 28, 34),
            "other_cups": "unchanged_neutral_identical",
            "off_cue_pixels": "byte_identical_to_v1_and_v2",
        }
        for key, value in expected.items():
            if getattr(self, key) != value:
                raise ValueError(f"V3 freezes {key} at {value!r}")

    def describe(self) -> dict[str, Any]:
        return {
            "schema": SALIENCE_SCHEMA,
            "cued_cup_fill": self.cued_cup_fill,
            "ball_radius": self.ball_radius,
            "cue_card": self.cue_card,
            "card_bounds": list(self.card_bounds),
            "header_bounds": list(self.header_bounds),
            "cell_width": self.cell_width,
            "cell_bounds_y": list(self.cell_bounds_y),
            "neutral_cell_color": list(self.neutral_cell_color),
            "card_background": list(self.card_background),
            "other_cups": self.other_cups,
            "off_cue_pixels": self.off_cue_pixels,
            "selection_candidates": 1,
            "selection_policy": (
                "development-only pass/fail; a failed candidate requires "
                "a new version"),
        }


V3_SALIENCE = ShellGameCueSalienceV3()


def _draw_slot_card(frame: np.ndarray, contract: ShellGameCapacityContract,
                    item: int, initial_slot: int,
                    salience: ShellGameCueSalienceV3) -> None:
    """Draw one conventional left/middle/right cue card."""

    item_color = CUE_COLORS[item]
    draw_rect(frame, *salience.card_bounds, salience.card_background)
    draw_rect(frame, *salience.header_bounds, item_color)
    cell_y0, cell_y1 = salience.cell_bounds_y
    half = salience.cell_width // 2
    for slot, center_x in enumerate(contract.slot_x):
        color = item_color if slot == initial_slot else salience.neutral_cell_color
        draw_rect(frame, center_x - half, cell_y0,
                  center_x - half + salience.cell_width, cell_y1, color)


def _apply_salience(
        batch: ShellGameCapacityBatch,
        salience: ShellGameCueSalienceV3 = V3_SALIENCE,
        ) -> ShellGameCapacityBatch:
    """Return a V3 batch while preserving every non-frame field exactly."""

    frames = np.array(batch.frames, copy=True)
    contract = batch.contract
    cup_width, cup_height = contract.cup_size
    for episode in range(batch.num_episodes):
        for item, (start, stop) in enumerate(contract.cue_windows):
            color = CUE_COLORS[item]
            cued_entity = int(batch.initial_slots[episode, item])
            initial_slot = cued_entity
            for time in range(start, stop):
                frame = frames[episode, time]
                center_x = int(round(float(
                    batch.entity_x[episode, time, cued_entity])))
                cup_y0 = contract.cup_y - contract.cue_lift
                draw_rect(
                    frame,
                    center_x - cup_width // 2,
                    cup_y0,
                    center_x - cup_width // 2 + cup_width,
                    cup_y0 + cup_height,
                    color,
                )
                draw_disc(
                    frame,
                    contract.slot_x[initial_slot],
                    contract.ball_y,
                    salience.ball_radius,
                    color,
                )
                _draw_slot_card(
                    frame, contract, item, initial_slot, salience)
    return replace(batch, frames=frames)


def paired_counterfactual_batches_v3(
        base: OfficialHostBaseBatch,
        contract: ShellGameCapacityContract,
        seed: int,
        salience: ShellGameCueSalienceV3 = V3_SALIENCE,
        ) -> tuple[ShellGameCapacityBatch, ShellGameCapacityBatch]:
    """Render unchanged V1 scripts under the V3 cue-only amendment."""

    primary, counterfactual = paired_counterfactual_batches(
        base, contract, seed)
    return (_apply_salience(primary, salience),
            _apply_salience(counterfactual, salience))


def v3_contract_description(
        contract: ShellGameCapacityContract,
        salience: ShellGameCueSalienceV3 = V3_SALIENCE,
        ) -> dict[str, Any]:
    return {
        "schema": "official_shell_game_capacity_v3",
        "semantic_contract": contract.describe(),
        "semantic_contract_changed_from_v1_or_v2": False,
        "cue_salience": salience.describe(),
        "amendment_scope": "cue-window pixels only",
    }


__all__ = [
    "SALIENCE_SCHEMA",
    "ShellGameCueSalienceV3",
    "V3_SALIENCE",
    "paired_counterfactual_batches_v3",
    "v3_contract_description",
]
