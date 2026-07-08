"""Versioned cue-salience amendment for the official shell-game contract.

V1 passed exact counterfactual construction, swap visibility, and both
post-cue leakage gates, but its small slot-specific ball was not reliably
available in the frozen official encoder.  V2 changes only pixels inside an
item's cue window: the lifted cup is filled with that item's color and the
ball radius is increased from three to six pixels.  Timing, nuisance,
targets, semantic capacity stages, and all legal final coordinates are the
unchanged V1 objects.

The implementation intentionally starts from the locked V1 renderer and
applies an integer-only cue overlay.  Thus every off-cue V2 pixel is not only
counterfactually identical but byte-identical to V1 as well.
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


SALIENCE_SCHEMA = "official_shell_game_cue_salience_v2"


@dataclass(frozen=True)
class ShellGameCueSalienceV2:
    """The single precommitted V2 salience candidate."""

    cued_cup_fill: str = "item_color"
    ball_radius: int = 6
    other_cups: str = "unchanged_neutral_identical"
    border_pixels: int = 2
    off_cue_pixels: str = "byte_identical_to_v1"

    def __post_init__(self) -> None:
        if self.cued_cup_fill != "item_color":
            raise ValueError("V2 requires the entire cued cup in item color")
        if self.ball_radius != 6:
            raise ValueError("V2 freezes the larger cue ball radius at six")
        if self.other_cups != "unchanged_neutral_identical":
            raise ValueError("V2 may not alter non-cued cups")
        if self.border_pixels != 2:
            raise ValueError("V2 retains the V1 two-pixel item-color border")
        if self.off_cue_pixels != "byte_identical_to_v1":
            raise ValueError("V2 may not alter any off-cue pixel")

    def describe(self) -> dict[str, Any]:
        return {
            "schema": SALIENCE_SCHEMA,
            "cued_cup_fill": self.cued_cup_fill,
            "ball_radius": self.ball_radius,
            "other_cups": self.other_cups,
            "border_pixels": self.border_pixels,
            "off_cue_pixels": self.off_cue_pixels,
            "selection_candidates": 1,
            "selection_policy": (
                "development-only pass/fail; a failed candidate requires "
                "a new version"),
        }


V2_SALIENCE = ShellGameCueSalienceV2()


def _apply_salience(
        batch: ShellGameCapacityBatch,
        salience: ShellGameCueSalienceV2 = V2_SALIENCE,
        ) -> ShellGameCapacityBatch:
    """Return a V2 batch while preserving every non-frame field exactly."""

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
                # Draw after the cup so the enlarged ball remains an explicit,
                # high-area slot marker rather than being partially occluded.
                draw_disc(
                    frame,
                    contract.slot_x[initial_slot],
                    contract.ball_y,
                    salience.ball_radius,
                    color,
                )
    return replace(batch, frames=frames)


def paired_counterfactual_batches_v2(
        base: OfficialHostBaseBatch,
        contract: ShellGameCapacityContract,
        seed: int,
        salience: ShellGameCueSalienceV2 = V2_SALIENCE,
        ) -> tuple[ShellGameCapacityBatch, ShellGameCapacityBatch]:
    """Render the unchanged V1 scripts under the V2 cue-only amendment."""

    primary, counterfactual = paired_counterfactual_batches(
        base, contract, seed)
    return (_apply_salience(primary, salience),
            _apply_salience(counterfactual, salience))


def v2_contract_description(
        contract: ShellGameCapacityContract,
        salience: ShellGameCueSalienceV2 = V2_SALIENCE,
        ) -> dict[str, Any]:
    """Describe the unchanged semantic contract plus its visual amendment."""

    return {
        "schema": "official_shell_game_capacity_v2",
        "semantic_contract": contract.describe(),
        "semantic_contract_changed_from_v1": False,
        "cue_salience": salience.describe(),
        "amendment_scope": "cue-window pixels only",
    }


__all__ = [
    "SALIENCE_SCHEMA",
    "ShellGameCueSalienceV2",
    "V2_SALIENCE",
    "paired_counterfactual_batches_v2",
    "v2_contract_description",
]
