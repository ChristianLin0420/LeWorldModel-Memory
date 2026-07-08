"""Leakage-auditable shell-game capacity tasks for the official LeWM host.

The task presents one, two, or four colored items in consecutive cue windows.
For each item, a ball appears below one of three visually identical cups.
After all cues disappear, the same visible cup-swap script is replayed in
every counterfactual branch.  At the final decision the target is the final
slot of every item.  Items are sampled independently and may share a cup, so
the exact-set chance rate is ``3 ** -capacity`` rather than being capped by a
three-object permutation.

The paired construction is the binding demand/leakage contract.  Branches
share base pixels, native 10-D official action blocks, simulator state, cue
timing, and swap nuisance.  They differ only in hidden initial item slots.
Consequently every non-cue pixel, including the final legal context, must be
byte-identical while every item target changes.  This is stronger than a
finite-sample statistical leakage test and is available before any encoder or
carrier is trained.

This module deliberately performs no model loading or training.  It reuses
the repository's integer overlay primitives so exact equality is attainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from lewm.tasks_v19.base import IMG_SIZE
from lewm.tasks_v19.overlays import CUE_COLORS, draw_border, draw_disc, draw_rect


SCHEMA = "official_shell_game_capacity_v1"
OFFICIAL_EPISODE_LENGTH = 64
OFFICIAL_ACTION_BLOCK_DIM = 10
NUM_SLOTS = 3
MAX_CAPACITY = len(CUE_COLORS)
COUNTERFACTUAL_SEED_SALT = 0x5E11CA9E
SLOT_PAIRS = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class ShellGameCapacityStage:
    """One visible capacity stage; names are suitable for paper artifacts."""

    key: str
    display_name: str
    capacity: int

    def __post_init__(self) -> None:
        if not self.key or self.key.lower().startswith("t") and self.key[1:].isdigit():
            raise ValueError("capacity stage requires a semantic, non-T key")
        if not self.display_name:
            raise ValueError("capacity stage requires a display name")
        if not 1 <= self.capacity <= MAX_CAPACITY:
            raise ValueError(
                f"capacity must be in [1,{MAX_CAPACITY}], got {self.capacity}")

    @property
    def per_item_chance(self) -> float:
        return 1.0 / NUM_SLOTS

    @property
    def exact_set_chance(self) -> float:
        return float(NUM_SLOTS ** -self.capacity)


CAPACITY_STAGES = (
    ShellGameCapacityStage(
        key="single-item",
        display_name="Single-item shell-game recall",
        capacity=1,
    ),
    ShellGameCapacityStage(
        key="two-item",
        display_name="Two-item shell-game recall",
        capacity=2,
    ),
    ShellGameCapacityStage(
        key="four-item",
        display_name="Four-item shell-game recall",
        capacity=4,
    ),
)


def get_capacity_stage(name: str) -> ShellGameCapacityStage:
    """Resolve a stage by semantic key or exact display name."""

    for stage in CAPACITY_STAGES:
        if name in (stage.key, stage.display_name):
            return stage
    choices = [stage.key for stage in CAPACITY_STAGES]
    raise KeyError(f"unknown shell-game capacity stage {name!r}; expected {choices}")


@dataclass(frozen=True)
class ShellGameCapacityContract:
    """Frozen timing, geometry, and legal-read contract for one stage."""

    stage: ShellGameCapacityStage
    episode_length: int = OFFICIAL_EPISODE_LENGTH
    decision_index: int = OFFICIAL_EPISODE_LENGTH - 1
    final_history: int = 3
    cue_start: int = 4
    cue_frames: int = 3
    cue_stride: int = 4
    swap_times: tuple[int, ...] = (24, 32, 40, 48)
    swap_frames: int = 4
    slot_x: tuple[int, int, int] = (12, 32, 52)
    table_height: int = 22
    cup_y: int = 4
    cup_size: tuple[int, int] = (12, 14)
    cue_lift: int = 5
    ball_y: int = 17
    ball_radius: int = 3
    cue_border_px: int = 2
    table_color: tuple[int, int, int] = (70, 55, 40)
    cup_color: tuple[int, int, int] = (150, 150, 150)

    def __post_init__(self) -> None:
        if self.episode_length != OFFICIAL_EPISODE_LENGTH:
            raise ValueError(
                f"official host contract requires length {OFFICIAL_EPISODE_LENGTH}")
        if self.decision_index != self.episode_length - 1:
            raise ValueError("decision must be the final observation")
        if not 1 <= self.final_history < self.decision_index:
            raise ValueError("invalid final-history length")
        if len(self.slot_x) != NUM_SLOTS or tuple(sorted(self.slot_x)) != self.slot_x:
            raise ValueError("slot_x must contain three increasing slots")
        cue_windows = self.cue_windows
        if cue_windows[-1][1] > self.swap_times[0]:
            raise ValueError("all item cues must finish before the first swap")
        if self.swap_frames < 2:
            raise ValueError("swap animation requires at least two frames")
        for first, second in zip(self.swap_times, self.swap_times[1:]):
            if second < first + self.swap_frames:
                raise ValueError("swap animations may not overlap")
        if self.shuffle_off + 2 >= self.final_context_indices[0]:
            raise ValueError(
                "post-shuffle evidence must expire before the final legal context")

    @property
    def cue_windows(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (self.cue_start + item * self.cue_stride,
             self.cue_start + item * self.cue_stride + self.cue_frames)
            for item in range(self.stage.capacity)
        )

    @property
    def shuffle_off(self) -> int:
        return self.swap_times[-1] + self.swap_frames

    @property
    def final_context_indices(self) -> tuple[int, ...]:
        return tuple(range(
            self.decision_index - self.final_history,
            self.decision_index,
        ))

    def describe(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "display_name": self.stage.display_name,
            "stage": self.stage.key,
            "capacity": self.stage.capacity,
            "per_item_classes": NUM_SLOTS,
            "per_item_chance": self.stage.per_item_chance,
            "exact_set_chance": self.stage.exact_set_chance,
            "episode_length": self.episode_length,
            "official_action_block_dim": OFFICIAL_ACTION_BLOCK_DIM,
            "decision_index": self.decision_index,
            "decision_observation_excluded": True,
            "final_context_indices": list(self.final_context_indices),
            "cue_windows": [list(window) for window in self.cue_windows],
            "swap_times": list(self.swap_times),
            "swap_frames": self.swap_frames,
            "shuffle_off": self.shuffle_off,
            "targets": "ordered final slot for every cued item",
            "items_may_share_slot": True,
            "capacity_stages_are_nested_by_seed": True,
        }


@dataclass(frozen=True)
class OfficialHostBaseBatch:
    """Base Reacher stream before visual shell-game compositing."""

    frames: np.ndarray
    actions: np.ndarray
    endo_state: np.ndarray

    def __post_init__(self) -> None:
        if self.frames.ndim != 5:
            raise ValueError(f"frames must be (E,L,H,W,3), got {self.frames.shape}")
        episodes, length = self.frames.shape[:2]
        if self.frames.dtype != np.uint8 \
                or self.frames.shape[2:] != (IMG_SIZE, IMG_SIZE, 3):
            raise ValueError(
                f"frames must be uint8 (E,L,{IMG_SIZE},{IMG_SIZE},3), got "
                f"{self.frames.dtype} {self.frames.shape}")
        if length != OFFICIAL_EPISODE_LENGTH:
            raise ValueError(
                f"official task requires {OFFICIAL_EPISODE_LENGTH} observations")
        if self.actions.dtype != np.float32 \
                or self.actions.shape != (
                    episodes, length - 1, OFFICIAL_ACTION_BLOCK_DIM):
            raise ValueError(
                "actions must be native float32 official 5x2 blocks with shape "
                f"(E,L-1,{OFFICIAL_ACTION_BLOCK_DIM}), got "
                f"{self.actions.dtype} {self.actions.shape}")
        if self.endo_state.dtype != np.float32 \
                or self.endo_state.ndim != 3 \
                or self.endo_state.shape[:2] != (episodes, length):
            raise ValueError(
                f"endo_state must be float32 (E,L,S), got "
                f"{self.endo_state.dtype} {self.endo_state.shape}")

    @property
    def num_episodes(self) -> int:
        return int(self.frames.shape[0])


@dataclass(frozen=True)
class ShellGameCapacityBatch:
    """Rendered multi-item task bank for one capacity stage and branch."""

    contract: ShellGameCapacityContract
    frames: np.ndarray
    actions: np.ndarray
    endo_state: np.ndarray
    initial_slots: np.ndarray
    final_slots: np.ndarray
    entity_x: np.ndarray
    cue_on: np.ndarray
    cue_off: np.ndarray
    swap_pairs: np.ndarray
    shuffle_off: np.ndarray
    seed: int
    branch: str

    def __post_init__(self) -> None:
        episodes, length = self.frames.shape[:2]
        capacity = self.contract.stage.capacity
        if self.frames.dtype != np.uint8 \
                or self.frames.shape[2:] != (IMG_SIZE, IMG_SIZE, 3) \
                or length != self.contract.episode_length:
            raise ValueError(f"invalid rendered frame contract {self.frames.shape}")
        if self.actions.dtype != np.float32 \
                or self.actions.shape != (
                    episodes, length - 1, OFFICIAL_ACTION_BLOCK_DIM):
            raise ValueError(f"invalid action contract {self.actions.shape}")
        if self.endo_state.dtype != np.float32 \
                or self.endo_state.shape[:2] != (episodes, length):
            raise ValueError(f"invalid endo_state contract {self.endo_state.shape}")
        for name, value in (
                ("initial_slots", self.initial_slots),
                ("final_slots", self.final_slots)):
            if value.dtype != np.int64 or value.shape != (episodes, capacity):
                raise ValueError(f"{name} must be int64 (E,K), got {value.shape}")
            if value.min() < 0 or value.max() >= NUM_SLOTS:
                raise ValueError(f"{name} contains an invalid slot")
        if self.entity_x.dtype != np.float32 \
                or self.entity_x.shape != (episodes, length, NUM_SLOTS):
            raise ValueError(f"invalid entity trace {self.entity_x.shape}")
        if self.cue_on.dtype != np.int64 \
                or self.cue_on.shape != (episodes, capacity):
            raise ValueError("cue_on must be int64 (E,K)")
        if self.cue_off.dtype != np.int64 \
                or self.cue_off.shape != (episodes, capacity):
            raise ValueError("cue_off must be int64 (E,K)")
        if self.swap_pairs.dtype != np.int64 \
                or self.swap_pairs.shape != (
                    episodes, len(self.contract.swap_times)):
            raise ValueError("swap_pairs has the wrong shape or dtype")
        if self.shuffle_off.dtype != np.int64 \
                or self.shuffle_off.shape != (episodes,):
            raise ValueError("shuffle_off must be int64 (E,)")
        if self.branch not in ("primary", "counterfactual"):
            raise ValueError("branch must be primary or counterfactual")

    @property
    def num_episodes(self) -> int:
        return int(self.frames.shape[0])

    @property
    def display_name(self) -> str:
        return self.contract.stage.display_name

    @property
    def per_item_chance(self) -> float:
        return self.contract.stage.per_item_chance

    @property
    def exact_set_chance(self) -> float:
        return self.contract.stage.exact_set_chance

    @property
    def events(self) -> Mapping[str, np.ndarray]:
        return {
            "cue_on": self.cue_on,
            "cue_off": self.cue_off,
            "swap_pairs": self.swap_pairs,
            "shuffle_off": self.shuffle_off,
        }


def _balanced_labels(num_episodes: int, count: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Nearly exact per-coordinate class balance, shuffled independently."""

    labels = np.empty((num_episodes, count), dtype=np.int64)
    template = np.resize(np.arange(NUM_SLOTS, dtype=np.int64), num_episodes)
    for coordinate in range(count):
        labels[:, coordinate] = rng.permutation(template)
    return labels


def _sample_scripts(num_episodes: int, seed: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Nested desired final slots and nuisance shared by every capacity stage.

    Primary final targets are balanced coordinate-wise before their cue slots
    are obtained by inverting each episode's visible cup permutation.  This
    prevents a constant leakage probe from exploiting finite-bank label
    imbalance.  It does not couple target values to base pixels or actions.
    """

    root = np.random.SeedSequence((COUNTERFACTUAL_SEED_SALT, int(seed)))
    nuisance_seed, target_seed = root.spawn(2)
    nuisance_rng = np.random.default_rng(nuisance_seed)
    target_rng = np.random.default_rng(target_seed)
    desired_final_slots = _balanced_labels(
        num_episodes, MAX_CAPACITY, target_rng)
    swap_pairs = _balanced_labels(
        num_episodes, len(ShellGameCapacityContract(
            CAPACITY_STAGES[0]).swap_times), nuisance_rng)
    return desired_final_slots, swap_pairs


def _entity_positions(contract: ShellGameCapacityContract,
                      swap_pairs: np.ndarray) -> np.ndarray:
    """Visible x traces of the three identical cup entities."""

    episodes = swap_pairs.shape[0]
    output = np.empty(
        (episodes, contract.episode_length, NUM_SLOTS), dtype=np.float32)
    slots = np.asarray(contract.slot_x, dtype=np.float32)
    for episode in range(episodes):
        current = slots.copy()  # entity -> x; entity id is its initial slot.
        slot_entity = np.arange(NUM_SLOTS)
        cursor = 0
        for start, pair_index in zip(
                contract.swap_times, swap_pairs[episode], strict=True):
            output[episode, cursor:start] = current
            slot_a, slot_b = SLOT_PAIRS[int(pair_index)]
            entity_a, entity_b = slot_entity[slot_a], slot_entity[slot_b]
            for offset in range(contract.swap_frames):
                progress = offset / (contract.swap_frames - 1)
                output[episode, start + offset] = current
                output[episode, start + offset, entity_a] = (
                    contract.slot_x[slot_a]
                    + progress * (contract.slot_x[slot_b]
                                  - contract.slot_x[slot_a]))
                output[episode, start + offset, entity_b] = (
                    contract.slot_x[slot_b]
                    + progress * (contract.slot_x[slot_a]
                                  - contract.slot_x[slot_b]))
            current = current.copy()
            current[entity_a], current[entity_b] = (
                contract.slot_x[slot_b], contract.slot_x[slot_a])
            slot_entity[slot_a], slot_entity[slot_b] = entity_b, entity_a
            cursor = start + contract.swap_frames
        output[episode, cursor:] = current
    return output


def _final_slots(contract: ShellGameCapacityContract, entity_x: np.ndarray,
                 initial_slots: np.ndarray) -> np.ndarray:
    slots = np.asarray(contract.slot_x, dtype=np.float32)
    final_entity_slot = np.argmin(
        np.abs(entity_x[:, -1, :, None] - slots[None, None, :]), axis=-1)
    return np.take_along_axis(final_entity_slot, initial_slots, axis=1).astype(
        np.int64)


def _cue_arrays(contract: ShellGameCapacityContract,
                episodes: int) -> tuple[np.ndarray, np.ndarray]:
    windows = np.asarray(contract.cue_windows, dtype=np.int64)
    return (np.broadcast_to(windows[None, :, 0],
                            (episodes, len(windows))).copy(),
            np.broadcast_to(windows[None, :, 1],
                            (episodes, len(windows))).copy())


def _render(base_frames: np.ndarray, contract: ShellGameCapacityContract,
            initial_slots: np.ndarray, entity_x: np.ndarray) -> np.ndarray:
    frames = np.array(base_frames, copy=True)
    width, height = contract.cup_size
    windows = contract.cue_windows
    for episode in range(frames.shape[0]):
        for time in range(contract.episode_length):
            frame = frames[episode, time]
            draw_rect(frame, 0, 0, IMG_SIZE, contract.table_height,
                      contract.table_color)
            active_item = next(
                (item for item, (start, stop) in enumerate(windows)
                 if start <= time < stop),
                None,
            )
            lifted_entity = None
            if active_item is not None:
                color = CUE_COLORS[active_item]
                initial_slot = int(initial_slots[episode, active_item])
                lifted_entity = initial_slot
                draw_border(frame, contract.cue_border_px, color)
                draw_disc(frame, contract.slot_x[initial_slot],
                          contract.ball_y, contract.ball_radius, color)
            for entity in range(NUM_SLOTS):
                center = int(round(float(entity_x[episode, time, entity])))
                y0 = contract.cup_y - (
                    contract.cue_lift if entity == lifted_entity else 0)
                draw_rect(frame, center - width // 2, y0,
                          center - width // 2 + width, y0 + height,
                          contract.cup_color)
    return frames


def _make_batch(base: OfficialHostBaseBatch,
                contract: ShellGameCapacityContract,
                initial_slots: np.ndarray, swap_pairs: np.ndarray,
                entity_x: np.ndarray, seed: int,
                branch: str) -> ShellGameCapacityBatch:
    capacity = contract.stage.capacity
    initial_slots = np.array(initial_slots[:, :capacity], copy=True)
    cue_on, cue_off = _cue_arrays(contract, base.num_episodes)
    return ShellGameCapacityBatch(
        contract=contract,
        frames=_render(base.frames, contract, initial_slots, entity_x),
        actions=np.array(base.actions, copy=True),
        endo_state=np.array(base.endo_state, copy=True),
        initial_slots=initial_slots,
        final_slots=_final_slots(contract, entity_x, initial_slots),
        entity_x=np.array(entity_x, copy=True),
        cue_on=cue_on,
        cue_off=cue_off,
        swap_pairs=np.array(swap_pairs, copy=True),
        shuffle_off=np.full(
            base.num_episodes, contract.shuffle_off, dtype=np.int64),
        seed=int(seed),
        branch=branch,
    )


def paired_counterfactual_batches(
        base: OfficialHostBaseBatch,
        contract: ShellGameCapacityContract,
        seed: int) -> tuple[ShellGameCapacityBatch, ShellGameCapacityBatch]:
    """Render paired branches differing only in every hidden item slot."""

    desired_final_slots, swap_pairs = _sample_scripts(base.num_episodes, seed)
    entity_x = _entity_positions(contract, swap_pairs)
    slots = np.asarray(contract.slot_x, dtype=np.float32)
    final_entity_slot = np.argmin(
        np.abs(entity_x[:, -1, :, None] - slots[None, None, :]), axis=-1)
    # argsort maps a desired final slot back to the cup entity that started in
    # the corresponding initial slot.
    final_slot_to_initial = np.argsort(final_entity_slot, axis=1)
    all_initial_slots = np.take_along_axis(
        final_slot_to_initial, desired_final_slots, axis=1).astype(np.int64)
    primary = _make_batch(
        base, contract, all_initial_slots, swap_pairs, entity_x, seed,
        "primary")
    counterfactual = _make_batch(
        base, contract, (all_initial_slots + 1) % NUM_SLOTS,
        swap_pairs, entity_x, seed, "counterfactual")
    return primary, counterfactual


def _check(value: Any, expected: str, passed: bool) -> dict[str, Any]:
    if isinstance(value, np.generic):
        value = value.item()
    return {"value": value, "expected": expected, "pass": bool(passed)}


def audit_paired_counterfactual(
        primary: ShellGameCapacityBatch,
        counterfactual: ShellGameCapacityBatch) -> dict[str, Any]:
    """Exact construction audit for leakage, demand, and cue non-vacuity."""

    if primary.contract != counterfactual.contract:
        raise ValueError("counterfactual branches use different contracts")
    if primary.frames.shape != counterfactual.frames.shape:
        raise ValueError("counterfactual branches have different shapes")
    contract = primary.contract
    episodes = primary.num_episodes
    cue_mask = np.zeros((episodes, contract.episode_length), dtype=bool)
    for episode in range(episodes):
        for start, stop in zip(
                primary.cue_on[episode], primary.cue_off[episode], strict=True):
            cue_mask[episode, int(start):int(stop)] = True

    difference = np.abs(
        primary.frames.astype(np.int16)
        - counterfactual.frames.astype(np.int16))
    off_cue_max = int(difference[~cue_mask].max(initial=0))
    per_cue_max = []
    for episode in range(episodes):
        for start, stop in zip(
                primary.cue_on[episode], primary.cue_off[episode], strict=True):
            per_cue_max.append(int(
                difference[episode, int(start):int(stop)].max(initial=0)))
    minimum_cue_difference = min(per_cue_max, default=0)
    final_indices = np.asarray(contract.final_context_indices, dtype=np.int64)
    final_context_max = int(difference[:, final_indices].max(initial=0))

    nuisance_equal = bool(
        np.array_equal(primary.actions, counterfactual.actions)
        and np.array_equal(primary.endo_state, counterfactual.endo_state)
        and np.array_equal(primary.entity_x, counterfactual.entity_x)
        and np.array_equal(primary.cue_on, counterfactual.cue_on)
        and np.array_equal(primary.cue_off, counterfactual.cue_off)
        and np.array_equal(primary.swap_pairs, counterfactual.swap_pairs)
        and np.array_equal(primary.shuffle_off, counterfactual.shuffle_off)
    )
    every_target_changes = bool(
        np.all(primary.initial_slots != counterfactual.initial_slots)
        and np.all(primary.final_slots != counterfactual.final_slots))
    checks = {
        "shared_nuisance": _check(nuisance_equal, "true", nuisance_equal),
        "off_cue_pixel_leakage": _check(off_cue_max, "== 0", off_cue_max == 0),
        "final_legal_context_leakage": _check(
            final_context_max, "== 0", final_context_max == 0),
        "cue_counterfactual_nonvacuity": _check(
            minimum_cue_difference, "> 0", minimum_cue_difference > 0),
        "all_item_targets_change": _check(
            every_target_changes, "true", every_target_changes),
    }
    demand_pass = bool(
        checks["shared_nuisance"]["pass"]
        and checks["off_cue_pixel_leakage"]["pass"]
        and checks["all_item_targets_change"]["pass"])
    checks["paired_counterfactual_demand"] = _check(
        demand_pass,
        "identical legal non-cue input with different targets",
        demand_pass,
    )
    return {
        "schema": "official_shell_game_counterfactual_audit_v1",
        "display_name": contract.stage.display_name,
        "capacity": contract.stage.capacity,
        "episodes": episodes,
        "decision_index": contract.decision_index,
        "decision_observation_excluded": True,
        "final_context_indices": list(contract.final_context_indices),
        "checks": checks,
        "overall_pass": all(check["pass"] for check in checks.values()),
    }


class CounterfactualAuditError(RuntimeError):
    """Raised when an exact paired task contract is violated."""


def require_paired_counterfactual(
        primary: ShellGameCapacityBatch,
        counterfactual: ShellGameCapacityBatch) -> dict[str, Any]:
    """Run the audit and fail with all violated clauses."""

    report = audit_paired_counterfactual(primary, counterfactual)
    failed = [
        name for name, check in report["checks"].items()
        if not check["pass"]
    ]
    if failed:
        raise CounterfactualAuditError(
            f"{report['display_name']} counterfactual audit failed: {failed}")
    return report


def _spaced_indices(start: np.ndarray, stop: np.ndarray,
                    count: int) -> np.ndarray:
    """Evenly spaced integer indices over inclusive ``[start, stop]``."""

    if count <= 0:
        raise ValueError("probe frame count must be positive")
    start = np.asarray(start, dtype=np.float64)
    stop = np.asarray(stop, dtype=np.float64)
    if start.shape != stop.shape or np.any(stop < start):
        raise ValueError("invalid probe windows")
    weights = np.linspace(0.0, 1.0, count)
    return np.rint(
        start[..., None] + (stop - start)[..., None] * weights
    ).astype(np.int64)


@dataclass(frozen=True)
class ShellGameAdmissionInputs:
    """Indices and targets consumed by frozen-representation admission probes."""

    display_name: str
    capacity: int
    cue_indices: np.ndarray
    cue_initial_slot_targets: np.ndarray
    swap_indices: np.ndarray
    swap_pair_targets: np.ndarray
    post_shuffle_indices: np.ndarray
    final_context_indices: np.ndarray
    final_slot_targets: np.ndarray
    per_item_chance: float
    exact_set_chance: float

    def __post_init__(self) -> None:
        episodes, capacity, _ = self.cue_indices.shape
        if capacity != self.capacity:
            raise ValueError("cue index capacity mismatch")
        if self.cue_indices.dtype != np.int64:
            raise ValueError("cue indices must be int64")
        if self.cue_initial_slot_targets.shape != (episodes, capacity):
            raise ValueError("cue targets must be (E,K)")
        if self.final_slot_targets.shape != (episodes, capacity):
            raise ValueError("final targets must be (E,K)")
        if self.swap_indices.shape != self.swap_pair_targets.shape:
            raise ValueError("swap indices and targets must share shape")
        if self.post_shuffle_indices.shape[0] != episodes \
                or self.final_context_indices.shape[0] != episodes:
            raise ValueError("post-shuffle/final indices need one row per episode")


def build_admission_inputs(
        batch: ShellGameCapacityBatch,
        cue_probe_frames: int = 3,
        post_shuffle_probe_frames: int = 4) -> ShellGameAdmissionInputs:
    """Build legal cue, swap, and final-context coordinates without encoding."""

    contract = batch.contract
    episodes = batch.num_episodes
    cue_indices = _spaced_indices(
        batch.cue_on, batch.cue_off - 1, cue_probe_frames)
    swap_indices = np.broadcast_to(
        np.asarray(contract.swap_times, dtype=np.int64)[None, :]
        + contract.swap_frames // 2,
        (episodes, len(contract.swap_times)),
    ).copy()
    post_start = batch.shuffle_off + 2
    post_stop = np.full(episodes, contract.decision_index - 1, dtype=np.int64)
    post_shuffle_indices = _spaced_indices(
        post_start, post_stop, post_shuffle_probe_frames)
    final_context_indices = np.broadcast_to(
        np.asarray(contract.final_context_indices, dtype=np.int64)[None, :],
        (episodes, contract.final_history),
    ).copy()
    if np.any(final_context_indices >= contract.decision_index):
        raise AssertionError("decision observation leaked into final context")
    return ShellGameAdmissionInputs(
        display_name=batch.display_name,
        capacity=contract.stage.capacity,
        cue_indices=cue_indices,
        cue_initial_slot_targets=np.array(batch.initial_slots, copy=True),
        swap_indices=swap_indices,
        swap_pair_targets=np.array(batch.swap_pairs, copy=True),
        post_shuffle_indices=post_shuffle_indices,
        final_context_indices=final_context_indices,
        final_slot_targets=np.array(batch.final_slots, copy=True),
        per_item_chance=batch.per_item_chance,
        exact_set_chance=batch.exact_set_chance,
    )


def gather_sequence(sequence: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Gather per-episode coordinates of arbitrary rank from ``(E,L,...)``."""

    sequence = np.asarray(sequence)
    indices = np.asarray(indices)
    if sequence.ndim < 2 or indices.ndim < 2:
        raise ValueError("sequence and indices require episode/time axes")
    if sequence.shape[0] != indices.shape[0]:
        raise ValueError("sequence and indices episode counts differ")
    if indices.dtype.kind not in "iu":
        raise ValueError("indices must be integer")
    if indices.min() < 0 or indices.max() >= sequence.shape[1]:
        raise IndexError("gather index outside sequence")
    episode_shape = (sequence.shape[0],) + (1,) * (indices.ndim - 1)
    episodes = np.arange(sequence.shape[0]).reshape(episode_shape)
    return sequence[episodes, indices]
