"""Paths, integrity checks, and admission partitions for shell-game V3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lewm.official_tasks.artifacts import (
    load_verified_npz,
    sha256_arrays,
    sha256_file,
)
from lewm.official_tasks.shell_game_capacity import (
    CAPACITY_STAGES,
    OfficialHostBaseBatch,
    ShellGameAdmissionInputs,
    ShellGameCapacityBatch,
    ShellGameCapacityContract,
    get_capacity_stage,
)
from lewm.official_tasks.shell_game_capacity_v3 import V3_SALIENCE
from lewm.official_tasks.shell_game_spec_v3 import (
    ALL_SPLITS_V3,
    FORMAL_SPLITS_V3,
    resolve_path_v3,
)


STAGE_FIELDS_V3 = {
    "frames", "actions", "endo_state", "initial_slots", "final_slots",
    "entity_x", "cue_on", "cue_off", "swap_pairs", "shuffle_off",
}


def artifact_root_v3(spec: Mapping[str, Any]) -> Path:
    return resolve_path_v3(spec["artifacts"]["root"])


def require_split_v3(split: str) -> str:
    if split not in ALL_SPLITS_V3:
        raise ValueError(f"V3 split must be one of {ALL_SPLITS_V3}, got {split!r}")
    return split


def split_spec_v3(spec: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    require_split_v3(split)
    return spec["data"][split]


def stage_contract_v3(stage: str) -> ShellGameCapacityContract:
    return ShellGameCapacityContract(get_capacity_stage(stage))


def lock_receipt_v3(spec: Mapping[str, Any]) -> dict[str, Any]:
    record = spec["_lock_record"]
    return {
        "lock_sha256": record["sha256"],
        "spec_sha256": record["spec_sha256"],
        "amendment": "pre-formal_salience_amendment_v3",
    }


def base_path_v3(spec: Mapping[str, Any], split: str) -> Path:
    require_split_v3(split)
    return artifact_root_v3(spec) / spec["artifacts"]["base"] / f"{split}.npz"


def stage_path_v3(spec: Mapping[str, Any], stage: str, split: str) -> Path:
    require_split_v3(split)
    get_capacity_stage(stage)
    return (artifact_root_v3(spec) / spec["artifacts"]["stages"]
            / stage / f"{split}.npz")


def audit_path_v3(spec: Mapping[str, Any], stage: str, split: str) -> Path:
    return stage_path_v3(spec, stage, split).with_name(
        f"{split}.counterfactual_audit.json")


def development_cache_path_v3(spec: Mapping[str, Any], stage: str) -> Path:
    get_capacity_stage(stage)
    return (artifact_root_v3(spec) / spec["artifacts"]["development_cache"]
            / stage / "development.npz")


def development_receipt_path_v3(spec: Mapping[str, Any], stage: str) -> Path:
    return development_cache_path_v3(spec, stage).with_name(
        "salience_selection.json")


def development_manifest_path_v3(spec: Mapping[str, Any], stage: str) -> Path:
    return development_cache_path_v3(spec, stage).with_name("manifest.json")


def cache_path_v3(spec: Mapping[str, Any], stage: str, split: str) -> Path:
    if split not in FORMAL_SPLITS_V3:
        raise ValueError(f"formal V3 cache split must be {FORMAL_SPLITS_V3}")
    get_capacity_stage(stage)
    return (artifact_root_v3(spec) / spec["artifacts"]["cache"]
            / stage / f"{split}.npz")


def admission_path_v3(spec: Mapping[str, Any], stage: str) -> Path:
    return cache_path_v3(spec, stage, "train").with_name("admission.json")


def cache_manifest_path_v3(spec: Mapping[str, Any], stage: str) -> Path:
    return cache_path_v3(spec, stage, "train").with_name("manifest.json")


def carrier_directory_v3(spec: Mapping[str, Any], stage: str,
                         arm: str, seed: int) -> Path:
    get_capacity_stage(stage)
    return (artifact_root_v3(spec) / spec["artifacts"]["carriers"]
            / stage / arm / f"seed-{int(seed)}")


def log_root_v3(spec: Mapping[str, Any]) -> Path:
    return artifact_root_v3(spec) / spec["artifacts"]["logs"]


def batch_arrays_v3(batch: ShellGameCapacityBatch) -> dict[str, np.ndarray]:
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


def load_base_v3(spec: Mapping[str, Any], split: str
                 ) -> tuple[OfficialHostBaseBatch, dict[str, Any]]:
    arrays, sidecar = load_verified_npz(base_path_v3(spec, split))
    required = {"frames", "actions", "endo_state"}
    if set(arrays) != required:
        raise ValueError(f"V3 base fields differ: {sorted(arrays)}")
    split_data = split_spec_v3(spec, split)
    if sidecar.get("schema") != "official_shell_game_base_v3" \
            or sidecar.get("split") != split \
            or sidecar.get("episodes") != split_data["episodes"] \
            or sidecar.get("base_seed") != split_data["base_seed"] \
            or sidecar.get("formal_lock") != lock_receipt_v3(spec):
        raise ValueError("V3 base metadata differs from the locked contract")
    return OfficialHostBaseBatch(**arrays), sidecar


def load_stage_v3(spec: Mapping[str, Any], stage: str, split: str
                  ) -> tuple[ShellGameCapacityBatch, dict[str, Any]]:
    arrays, sidecar = load_verified_npz(stage_path_v3(spec, stage, split))
    if set(arrays) != STAGE_FIELDS_V3:
        raise ValueError(f"V3 stage fields differ: {sorted(arrays)}")
    if sidecar.get("schema") != "official_shell_game_stage_v3" \
            or sidecar.get("stage") != stage \
            or sidecar.get("split") != split \
            or sidecar.get("formal_lock") != lock_receipt_v3(spec) \
            or sidecar.get("cue_salience") != V3_SALIENCE.describe():
        raise ValueError("V3 stage metadata differs from the locked contract")
    contract = stage_contract_v3(stage)
    batch = ShellGameCapacityBatch(
        contract=contract,
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
        seed=int(split_spec_v3(spec, split)["counterfactual_seed"]),
        branch="primary",
    )
    return batch, sidecar


def load_counterfactual_audit_v3(
        spec: Mapping[str, Any], stage: str, split: str,
        batch: ShellGameCapacityBatch, sidecar: Mapping[str, Any],
        ) -> dict[str, Any]:
    path = audit_path_v3(spec, stage, split)
    expected = sidecar.get("counterfactual_receipt", {})
    if path.name != expected.get("path") \
            or not path.is_file() \
            or sha256_file(path) != expected.get("sha256"):
        raise ValueError(f"V3 counterfactual receipt mismatch for {stage}/{split}")
    receipt = json.loads(path.read_text())
    if receipt.get("schema") \
            != "official_shell_game_counterfactual_receipt_v3" \
            or receipt.get("formal_lock") != lock_receipt_v3(spec) \
            or receipt.get("cue_salience") != V3_SALIENCE.describe() \
            or receipt.get("audit", {}).get("overall_pass") is not True:
        raise ValueError(f"invalid V3 counterfactual audit for {stage}/{split}")
    if receipt.get("primary_content_sha256") \
            != sha256_arrays(batch_arrays_v3(batch)):
        raise ValueError(f"V3 primary content differs from audit for {stage}/{split}")
    return receipt


def require_selected_salience_v3(
        spec: Mapping[str, Any], stage: str) -> dict[str, Any]:
    """Require a hash-bound, development-only salience pass for ``stage``."""

    receipt_path = development_receipt_path_v3(spec, stage)
    manifest_path = development_manifest_path_v3(spec, stage)
    if not receipt_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"V3 formal data is blocked until development salience passes: {stage}")
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    selection_record = manifest.get("salience_selection", {})
    if manifest.get("schema") != "official_shell_game_development_manifest_v3" \
            or manifest.get("formal_lock") != lock_receipt_v3(spec) \
            or manifest.get("stage") != stage \
            or selection_record.get("path") != receipt_path.name \
            or selection_record.get("sha256") != sha256_file(receipt_path):
        raise ValueError(f"V3 development manifest mismatch for {stage}")
    selection = spec["development_selection"]
    criterion = receipt.get("criterion", {})
    per_item = criterion.get("per_item_accuracy")
    threshold = selection["threshold"]
    if receipt.get("schema") != "official_shell_game_salience_selection_v3" \
            or receipt.get("formal_lock") != lock_receipt_v3(spec) \
            or receipt.get("stage") != stage \
            or receipt.get("selected") is not True \
            or selection_record.get("selected") is not True \
            or receipt.get("formal_data_read") is not False \
            or receipt.get("threshold_changed_from_v1_or_v2") is not False \
            or criterion.get("threshold") != threshold \
            or criterion.get("pass") is not True \
            or not isinstance(per_item, list) or not per_item \
            or criterion.get("value") != min(per_item) \
            or any(not isinstance(value, (int, float)) or value < threshold
                   for value in per_item):
        raise RuntimeError(f"V3 development salience did not pass for {stage}")
    cache_record = manifest.get("development_cache", {})
    cache_path = development_cache_path_v3(spec, stage)
    sidecar_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if cache_record.get("sha256") != sha256_file(cache_path) \
            or cache_record.get("sidecar_sha256") != sha256_file(sidecar_path):
        raise ValueError(f"V3 development cache mismatch for {stage}")
    return {
        "stage": stage,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "selection_path": str(receipt_path),
        "selection_sha256": sha256_file(receipt_path),
        "threshold": threshold,
        "minimum_item_accuracy": float(criterion["value"]),
        "per_item_accuracy": [float(value) for value in per_item],
        "threshold_changed_from_v1_or_v2": False,
    }


def require_all_selected_salience_v3(
        spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Fail closed unless every registered stage passes the locked 0.75 gate."""

    if spec["development_selection"].get("all_stages_must_pass") is not True:
        raise ValueError("V3 all-stage formal gate is not enabled")
    return {
        stage.key: require_selected_salience_v3(spec, stage.key)
        for stage in CAPACITY_STAGES
    }


def slice_admission_inputs_v3(
        inputs: ShellGameAdmissionInputs, start: int, stop: int,
        ) -> ShellGameAdmissionInputs:
    """Take a deterministic contiguous development partition."""

    episodes = inputs.cue_indices.shape[0]
    if not 0 <= start < stop <= episodes:
        raise ValueError(f"invalid admission slice [{start},{stop})/{episodes}")
    return ShellGameAdmissionInputs(
        display_name=inputs.display_name,
        capacity=inputs.capacity,
        cue_indices=np.array(inputs.cue_indices[start:stop], copy=True),
        cue_initial_slot_targets=np.array(
            inputs.cue_initial_slot_targets[start:stop], copy=True),
        swap_indices=np.array(inputs.swap_indices[start:stop], copy=True),
        swap_pair_targets=np.array(
            inputs.swap_pair_targets[start:stop], copy=True),
        post_shuffle_indices=np.array(
            inputs.post_shuffle_indices[start:stop], copy=True),
        final_context_indices=np.array(
            inputs.final_context_indices[start:stop], copy=True),
        final_slot_targets=np.array(
            inputs.final_slot_targets[start:stop], copy=True),
        per_item_chance=inputs.per_item_chance,
        exact_set_chance=inputs.exact_set_chance,
    )


def development_selection_decision_v3(
        diagnostic: Mapping[str, Any], threshold: float,
        ) -> dict[str, Any]:
    """Apply the immutable development rule without consulting formal data."""

    if threshold != 0.75:
        raise ValueError("V3 development threshold must remain exactly 0.75")
    gates = diagnostic.get("gates", {})
    cue = gates.get("cue_initial_slot_availability", {})
    exact = gates.get("paired_counterfactual_construction", {})
    value = cue.get("value", {})
    minimum = value.get("minimum_item_accuracy")
    per_item = value.get("per_item_accuracy")
    if cue.get("threshold") != threshold \
            or not isinstance(minimum, (int, float)) \
            or not isinstance(per_item, list) or not per_item:
        raise ValueError("malformed V3 development cue diagnostic")
    cue_pass = bool(minimum >= threshold)
    if cue.get("pass") is not cue_pass:
        raise ValueError("V3 cue diagnostic pass flag disagrees with threshold")
    exact_pass = exact.get("pass") is True
    return {
        "metric": "minimum per-item frozen cue initial-slot accuracy",
        "value": float(minimum),
        "per_item_accuracy": [float(item) for item in per_item],
        "threshold": threshold,
        "direction": ">= for every item",
        "cue_pass": cue_pass,
        "exact_counterfactual_pass": exact_pass,
        "selected": bool(cue_pass and exact_pass),
    }


__all__ = [
    "STAGE_FIELDS_V3",
    "admission_path_v3",
    "artifact_root_v3",
    "audit_path_v3",
    "base_path_v3",
    "batch_arrays_v3",
    "cache_manifest_path_v3",
    "cache_path_v3",
    "carrier_directory_v3",
    "development_cache_path_v3",
    "development_manifest_path_v3",
    "development_receipt_path_v3",
    "development_selection_decision_v3",
    "load_base_v3",
    "load_counterfactual_audit_v3",
    "load_stage_v3",
    "lock_receipt_v3",
    "log_root_v3",
    "require_selected_salience_v3",
    "require_all_selected_salience_v3",
    "require_split_v3",
    "slice_admission_inputs_v3",
    "split_spec_v3",
    "stage_contract_v3",
    "stage_path_v3",
]
