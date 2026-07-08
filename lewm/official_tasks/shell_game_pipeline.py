"""Shared paths and serialization for the formal shell-game pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lewm.official_tasks.artifacts import load_verified_npz
from lewm.official_tasks.shell_game_capacity import (
    OfficialHostBaseBatch,
    ShellGameCapacityBatch,
    ShellGameCapacityContract,
    get_capacity_stage,
)
from lewm.official_tasks.shell_game_spec import resolve_path


SPLITS = ("train", "validation")


def artifact_root(spec: Mapping[str, Any]) -> Path:
    return resolve_path(spec["artifacts"]["root"])


def base_path(spec: Mapping[str, Any], split: str) -> Path:
    require_split(split)
    return artifact_root(spec) / spec["artifacts"]["base"] / f"{split}.npz"


def stage_path(spec: Mapping[str, Any], stage: str, split: str) -> Path:
    require_split(split)
    get_capacity_stage(stage)
    return (artifact_root(spec) / spec["artifacts"]["stages"]
            / stage / f"{split}.npz")


def audit_path(spec: Mapping[str, Any], stage: str, split: str) -> Path:
    return stage_path(spec, stage, split).with_name(
        f"{split}.counterfactual_audit.json")


def cache_path(spec: Mapping[str, Any], stage: str, split: str) -> Path:
    require_split(split)
    get_capacity_stage(stage)
    return (artifact_root(spec) / spec["artifacts"]["cache"]
            / stage / f"{split}.npz")


def admission_path(spec: Mapping[str, Any], stage: str) -> Path:
    return cache_path(spec, stage, "train").with_name("admission.json")


def cache_manifest_path(spec: Mapping[str, Any], stage: str) -> Path:
    return cache_path(spec, stage, "train").with_name("manifest.json")


def carrier_directory(spec: Mapping[str, Any], stage: str,
                      arm: str, seed: int) -> Path:
    get_capacity_stage(stage)
    return (artifact_root(spec) / spec["artifacts"]["carriers"]
            / stage / arm / f"seed-{int(seed)}")


def log_root(spec: Mapping[str, Any]) -> Path:
    return artifact_root(spec) / spec["artifacts"]["logs"]


def require_split(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return split


def split_spec(spec: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    require_split(split)
    return spec["data"][split]


def stage_contract(stage: str) -> ShellGameCapacityContract:
    return ShellGameCapacityContract(get_capacity_stage(stage))


def lock_receipt(spec: Mapping[str, Any]) -> dict[str, Any]:
    record = spec["_lock_record"]
    return {
        "lock_sha256": record["sha256"],
        "spec_sha256": record["spec_sha256"],
    }


def batch_arrays(batch: ShellGameCapacityBatch) -> dict[str, np.ndarray]:
    return {
        "frames": batch.frames,
        "actions": batch.actions,
        "endo_state": batch.endo_state,
        "initial_slots": batch.initial_slots,
        "final_slots": batch.final_slots,
        "entity_x": batch.entity_x,
        "cue_on": batch.cue_on,
        "cue_off": batch.cue_off,
        "swap_pairs": batch.swap_pairs,
        "shuffle_off": batch.shuffle_off,
    }


def load_base(spec: Mapping[str, Any], split: str
              ) -> tuple[OfficialHostBaseBatch, dict]:
    arrays, sidecar = load_verified_npz(base_path(spec, split))
    required = {"frames", "actions", "endo_state"}
    if set(arrays) != required:
        raise ValueError(
            f"base artifact fields differ: {sorted(arrays)} != {sorted(required)}")
    if sidecar.get("schema") != "official_shell_game_base_v1" \
            or sidecar.get("split") != split:
        raise ValueError("base artifact metadata contract differs")
    if sidecar.get("formal_lock") != lock_receipt(spec):
        raise ValueError("base artifact was not produced under the active lock")
    return OfficialHostBaseBatch(**arrays), sidecar


def load_stage(spec: Mapping[str, Any], stage: str, split: str
               ) -> tuple[ShellGameCapacityBatch, dict]:
    arrays, sidecar = load_verified_npz(stage_path(spec, stage, split))
    expected = set(batch_arrays(_empty_batch_for_fields(stage)))
    if set(arrays) != expected:
        raise ValueError(
            f"stage artifact fields differ: {sorted(arrays)} != {sorted(expected)}")
    if sidecar.get("schema") != "official_shell_game_stage_v1" \
            or sidecar.get("stage") != stage \
            or sidecar.get("split") != split:
        raise ValueError("stage artifact metadata contract differs")
    if sidecar.get("formal_lock") != lock_receipt(spec):
        raise ValueError("stage artifact was not produced under the active lock")
    counter_seed = int(split_spec(spec, split)["counterfactual_seed"])
    batch = ShellGameCapacityBatch(
        contract=stage_contract(stage),
        frames=arrays["frames"],
        actions=arrays["actions"],
        endo_state=arrays["endo_state"],
        initial_slots=arrays["initial_slots"],
        final_slots=arrays["final_slots"],
        entity_x=arrays["entity_x"],
        cue_on=arrays["cue_on"],
        cue_off=arrays["cue_off"],
        swap_pairs=arrays["swap_pairs"],
        shuffle_off=arrays["shuffle_off"],
        seed=counter_seed,
        branch="primary",
    )
    return batch, sidecar


def _empty_batch_for_fields(stage: str) -> ShellGameCapacityBatch:
    """Tiny valid batch used only to keep the serialized field list singular."""

    contract = stage_contract(stage)
    capacity = contract.stage.capacity
    return ShellGameCapacityBatch(
        contract=contract,
        frames=np.zeros((1, 64, 64, 64, 3), dtype=np.uint8),
        actions=np.zeros((1, 63, 10), dtype=np.float32),
        endo_state=np.zeros((1, 64, 1), dtype=np.float32),
        initial_slots=np.zeros((1, capacity), dtype=np.int64),
        final_slots=np.zeros((1, capacity), dtype=np.int64),
        entity_x=np.broadcast_to(
            np.asarray(contract.slot_x, dtype=np.float32)[None, None, :],
            (1, 64, 3)).copy(),
        cue_on=np.asarray(contract.cue_windows, dtype=np.int64)[None, :, 0],
        cue_off=np.asarray(contract.cue_windows, dtype=np.int64)[None, :, 1],
        swap_pairs=np.zeros((1, len(contract.swap_times)), dtype=np.int64),
        shuffle_off=np.asarray([contract.shuffle_off], dtype=np.int64),
        seed=0,
        branch="primary",
    )
