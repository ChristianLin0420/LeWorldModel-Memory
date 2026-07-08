"""Immutable loader for the V3 pre-formal cue-salience amendment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from lewm.official_tasks.artifacts import sha256_file
from lewm.official_tasks.shell_game_capacity import (
    CAPACITY_STAGES,
    ShellGameCapacityContract,
)
from lewm.official_tasks.shell_game_capacity_v3 import V3_SALIENCE
from lewm.official_tasks.shell_game_spec_v2 import load_locked_spec_v2


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_V3 = ROOT / "configs/official_shell_game_capacity_v3.yaml"
DEFAULT_LOCK_V3 = ROOT / "configs/official_shell_game_capacity_v3.lock.json"
LOCK_SCHEMA_V3 = "official_shell_game_capacity_lock_v3"
ALLOWED_DEVICES_V3 = ("cuda:1", "cuda:2")
V1_FORMAL_SEEDS = {380701, 380702, 381701, 381702}
FORMAL_SPLITS_V3 = ("train", "validation")
ALL_SPLITS_V3 = ("development", *FORMAL_SPLITS_V3)


def resolve_path_v3(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_device_v3(device: str) -> str:
    if device not in ALLOWED_DEVICES_V3:
        raise ValueError(
            f"formal shell-game V3 jobs permit only {ALLOWED_DEVICES_V3}; "
            f"got {device!r}")
    return device


def _expected_task_contract() -> dict[str, Any]:
    reference = ShellGameCapacityContract(CAPACITY_STAGES[-1]).describe()
    return {
        "num_slots": 3,
        "decision_index": reference["decision_index"],
        "decision_observation_excluded": True,
        "final_context_indices": reference["final_context_indices"],
        "cue_start": 4,
        "cue_frames": 3,
        "cue_stride": 4,
        "swap_times": reference["swap_times"],
        "swap_frames": reference["swap_frames"],
        "shuffle_off": reference["shuffle_off"],
        "items_may_share_slot": True,
        "capacity_stages_nested_by_seed": True,
        "target": "ordered final slot for every cued item",
    }


def _validate_parent_evidence(spec: dict[str, Any]) -> None:
    """Bind V3 to V2's development failure without reading formal data."""

    parent = spec["amendment"]["parent_v2"]
    parent_paths = {}
    for kind in ("spec", "lock"):
        path = resolve_path_v3(parent[f"{kind}_path"])
        parent_paths[kind] = path
        if not path.is_file():
            raise FileNotFoundError(f"missing parent V2 {kind}: {path}")
        if sha256_file(path) != parent[f"{kind}_sha256"]:
            raise ValueError(f"parent V2 {kind} hash differs from V3 record")
    parent_spec = load_locked_spec_v2(
        parent_paths["spec"], parent_paths["lock"])
    for section in (
            "official_host", "semantic_stages", "task_contract",
            "admission", "carrier_training"):
        if spec.get(section) != parent_spec.get(section):
            raise ValueError(
                f"V3 unexpectedly changes the locked V2 {section} section")
    for key in (
            "frame_skip", "raw_action_dim", "source_stream",
            "compression_level"):
        if spec.get("data", {}).get(key) != parent_spec.get("data", {}).get(key):
            raise ValueError(f"V3 changes formal data setting {key}")
    for split in FORMAL_SPLITS_V3:
        if spec["data"][split]["episodes"] \
                != parent_spec["data"][split]["episodes"]:
            raise ValueError(f"V3 changes formal split size for {split}")
    expected_parent_lock = {
        "lock_sha256": parent["lock_sha256"],
        "spec_sha256": parent["spec_sha256"],
        "amendment": "pre-formal_salience_amendment_v2",
    }
    for stage in (item.key for item in CAPACITY_STAGES):
        record = parent["evidence"][stage]
        path = resolve_path_v3(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"V2 salience evidence differs for {stage}")
        manifest_path = resolve_path_v3(record["manifest_path"])
        if not manifest_path.is_file() \
                or sha256_file(manifest_path) != record["manifest_sha256"]:
            raise ValueError(f"V2 salience manifest differs for {stage}")
        receipt = json.loads(path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if receipt.get("formal_lock") != expected_parent_lock:
            raise ValueError(f"V2 selection has wrong formal lock for {stage}")
        criterion = receipt.get("criterion", {})
        if receipt.get("schema") != "official_shell_game_salience_selection_v2" \
                or receipt.get("selected") is not False \
                or record.get("selected") is not False \
                or receipt.get("formal_data_read") is not False \
                or receipt.get("exact_counterfactual_pass") is not True \
                or criterion.get("threshold") != parent["unchanged_threshold"] \
                or criterion.get("pass") is not False \
                or criterion.get("per_item_accuracy") \
                != record.get("per_item_accuracy"):
            raise ValueError(f"V2 cue failure record differs for {stage}")
        selection = manifest.get("salience_selection", {})
        if manifest.get("schema") \
                != "official_shell_game_development_manifest_v2" \
                or manifest.get("formal_data_read") is not False \
                or selection.get("selected") is not False \
                or selection.get("path") != path.name \
                or selection.get("sha256") != record["sha256"]:
            raise ValueError(f"V2 development manifest is invalid for {stage}")
    parent_output = resolve_path_v3(parent["output_root"])
    formal_paths = [
        parent_output / "base" / f"{split}.npz"
        for split in FORMAL_SPLITS_V3
    ]
    if parent.get("formal_data_read") is not False \
            or any(path.exists() for path in formal_paths):
        raise ValueError("V2 formal data exists or was read before V3 locking")


def _validate_spec_v3(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 3:
        raise ValueError("shell-game V3 spec schema_version must be 3")
    if spec.get("protocol_status") \
            != "locked_before_v3_development_and_formal_run":
        raise ValueError("V3 is not marked locked before development/formal use")
    amendment = spec.get("amendment", {})
    required_flags = {
        "kind": "pre-formal_salience_amendment_v3",
        "threshold_changed_from_v1_or_v2": False,
        "semantic_capacity_contract_changed_from_v1_or_v2": False,
        "carrier_definitions_changed_from_v1_or_v2": False,
        "formal_protocol_changed_from_v2": False,
    }
    for key, value in required_flags.items():
        if amendment.get(key) != value:
            raise ValueError(f"V3 amendment.{key} must be {value!r}")

    expected_stages = [
        {"key": stage.key, "display_name": stage.display_name,
         "capacity": stage.capacity}
        for stage in CAPACITY_STAGES
    ]
    if spec.get("semantic_stages") != expected_stages:
        raise ValueError("V3 semantic stages differ from the V1/V2 contract")
    if spec.get("task_contract") != _expected_task_contract():
        raise ValueError("V3 task contract differs from V1/V2")
    if spec.get("cue_salience") != V3_SALIENCE.describe():
        raise ValueError("V3 cue salience differs from the frozen renderer")

    host = spec.get("official_host", {})
    expected_host = {
        "latent_dim": 192,
        "observation_length": 64,
        "context": 3,
        "action_block_dim": 10,
    }
    for key, value in expected_host.items():
        if host.get(key) != value:
            raise ValueError(f"official_host.{key} must remain {value}")
    if host.get("frozen_encoder") is not True \
            or host.get("frozen_predictor") is not True:
        raise ValueError("V3 official encoder and predictor must remain frozen")

    selection = spec.get("development_selection", {})
    admission = spec.get("admission", {})
    if selection.get("threshold_changed_from_v1_or_v2") is not False \
            or selection.get("threshold") != 0.75 \
            or admission.get("cue_initial_slot_accuracy_min") != 0.75:
        raise ValueError("V3 must retain the V1/V2 cue threshold of 0.75")
    development = spec.get("data", {}).get("development", {})
    if development.get("episodes") != (
            selection.get("fit_episodes", 0)
            + selection.get("check_episodes", 0)):
        raise ValueError("development fit/check partition must cover the bank")
    expected_partition = {
        "fit_indices": [0, int(selection.get("fit_episodes", 0))],
        "check_indices": [int(selection.get("fit_episodes", 0)),
                          int(development.get("episodes", 0))],
    }
    if selection.get("immutable_partition") != "contiguous_by_episode_index" \
            or any(selection.get(key) != value
                   for key, value in expected_partition.items()):
        raise ValueError("V3 development partition differs from its lock")
    if selection.get("all_stages_must_pass") is not True \
            or selection.get(
                "formal_data_must_not_be_collected_before_all_stages_pass") \
            is not True:
        raise ValueError("every formal wave must be gated by all V3 stages")

    seeds = []
    for split in ALL_SPLITS_V3:
        split_data = spec.get("data", {}).get(split, {})
        if int(split_data.get("episodes", 0)) <= 0:
            raise ValueError(f"V3 {split} episodes must be positive")
        seeds.extend((split_data.get("base_seed"),
                      split_data.get("counterfactual_seed")))
    if any(not isinstance(seed, int) for seed in seeds) \
            or len(seeds) != len(set(seeds)):
        raise ValueError("every V3 bank seed must be a distinct integer")
    parent_spec = yaml.safe_load(resolve_path_v3(
        amendment["parent_v2"]["spec_path"]).read_text())
    parent_seeds = {
        parent_spec["data"][split][kind]
        for split in ALL_SPLITS_V3
        for kind in ("base_seed", "counterfactual_seed")
    }
    if set(seeds) & (V1_FORMAL_SEEDS | parent_seeds):
        raise ValueError("V3 may not reuse a V1 or V2 bank seed")

    if resolve_path_v3(spec["artifacts"]["root"]).resolve() \
            == resolve_path_v3(
                amendment["parent_v2"]["output_root"]).resolve():
        raise ValueError("V3 must use a new output root")
    launcher = spec.get("launcher", {})
    if launcher.get("allowed_devices") != list(ALLOWED_DEVICES_V3) \
            or launcher.get("preview_by_default") is not True \
            or launcher.get("explicit_execute_required") is not True \
            or launcher.get("jobs_per_gpu") != 1:
        raise ValueError("V3 launcher safety policy differs")
    _validate_parent_evidence(spec)


def load_locked_spec_v3(
        spec_path: str | Path = DEFAULT_SPEC_V3,
        lock_path: str | Path = DEFAULT_LOCK_V3,
        ) -> dict[str, Any]:
    """Load V3 only when its spec, producers, and V2 evidence all match."""

    spec_path = resolve_path_v3(spec_path)
    lock_path = resolve_path_v3(lock_path)
    if not spec_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError("formal shell-game V3 spec and lock are required")
    lock = json.loads(lock_path.read_text())
    if lock.get("schema") != LOCK_SCHEMA_V3 \
            or lock.get("immutable") is not True:
        raise ValueError("invalid or mutable shell-game V3 lock")
    actual_spec_hash = sha256_file(spec_path)
    if lock.get("spec_sha256") != actual_spec_hash:
        raise ValueError("shell-game V3 spec hash differs from its lock")
    if resolve_path_v3(lock.get("spec_path", "")).resolve() \
            != spec_path.resolve():
        raise ValueError("V3 lock points to a different specification")
    for source, expected_hash in lock.get("producer_sha256", {}).items():
        source_path = resolve_path_v3(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"locked V3 producer missing: {source}")
        if sha256_file(source_path) != expected_hash:
            raise ValueError(f"locked V3 producer changed: {source}")
    for evidence, expected_hash in lock.get(
            "parent_evidence_sha256", {}).items():
        evidence_path = resolve_path_v3(evidence)
        if not evidence_path.is_file() \
                or sha256_file(evidence_path) != expected_hash:
            raise ValueError(f"locked V2 failure evidence changed: {evidence}")
    spec = yaml.safe_load(spec_path.read_text())
    if not isinstance(spec, dict):
        raise ValueError("shell-game V3 spec must be a YAML mapping")
    _validate_spec_v3(spec)
    spec["_lock_record"] = {
        "path": str(lock_path.resolve()),
        "sha256": sha256_file(lock_path),
        "spec_sha256": actual_spec_hash,
        "producer_sha256": lock["producer_sha256"],
    }
    return spec


__all__ = [
    "ALL_SPLITS_V3",
    "ALLOWED_DEVICES_V3",
    "DEFAULT_LOCK_V3",
    "DEFAULT_SPEC_V3",
    "FORMAL_SPLITS_V3",
    "load_locked_spec_v3",
    "resolve_path_v3",
    "validate_device_v3",
]
