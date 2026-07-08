"""Immutable loader for the V2 pre-formal cue-salience amendment."""

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
from lewm.official_tasks.shell_game_capacity_v2 import V2_SALIENCE


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_V2 = ROOT / "configs/official_shell_game_capacity_v2.yaml"
DEFAULT_LOCK_V2 = ROOT / "configs/official_shell_game_capacity_v2.lock.json"
LOCK_SCHEMA_V2 = "official_shell_game_capacity_lock_v2"
ALLOWED_DEVICES_V2 = ("cuda:1", "cuda:2")
V1_FORMAL_SEEDS = {380701, 380702, 381701, 381702}
FORMAL_SPLITS_V2 = ("train", "validation")
ALL_SPLITS_V2 = ("development", *FORMAL_SPLITS_V2)


def resolve_path_v2(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_device_v2(device: str) -> str:
    if device not in ALLOWED_DEVICES_V2:
        raise ValueError(
            f"formal shell-game V2 jobs permit only {ALLOWED_DEVICES_V2}; "
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
    """Bind V2 to the immutable V1 failure without modifying V1."""

    parent = spec["amendment"]["parent_v1"]
    parent_paths = {}
    for kind in ("spec", "lock"):
        path = resolve_path_v2(parent[f"{kind}_path"])
        parent_paths[kind] = path
        if not path.is_file():
            raise FileNotFoundError(f"missing parent V1 {kind}: {path}")
        if sha256_file(path) != parent[f"{kind}_sha256"]:
            raise ValueError(f"parent V1 {kind} hash differs from V2 record")
    parent_spec = yaml.safe_load(parent_paths["spec"].read_text())
    for section in (
            "official_host", "semantic_stages", "task_contract",
            "admission", "carrier_training"):
        if spec.get(section) != parent_spec.get(section):
            raise ValueError(
                f"V2 unexpectedly changes the locked V1 {section} section")
    expected_parent_lock = {
        "lock_sha256": parent["lock_sha256"],
        "spec_sha256": parent["spec_sha256"],
    }
    expected_other = {
        "paired_counterfactual_construction",
        "swap_pair_visibility",
        "post_shuffle_target_leakage",
        "final_context_target_leakage",
    }
    for stage in (item.key for item in CAPACITY_STAGES):
        record = parent["evidence"][stage]
        path = resolve_path_v2(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"V1 admission evidence differs for {stage}")
        receipt = json.loads(path.read_text())
        if receipt.get("formal_lock") != expected_parent_lock:
            raise ValueError(f"V1 admission has wrong formal lock for {stage}")
        if receipt.get("admitted") is not False \
                or record.get("admitted") is not False:
            raise ValueError(f"V1 {stage} must be recorded as not admitted")
        gates = receipt.get("gates", {})
        cue = gates.get(parent["failed_gate"], {})
        if cue.get("pass") is not False \
                or record.get("cue_gate_pass") is not False \
                or cue.get("threshold") != parent["unchanged_threshold"] \
                or cue.get("value", {}).get("per_item_accuracy") \
                != record.get("per_item_accuracy"):
            raise ValueError(f"V1 cue failure record differs for {stage}")
        if set(record.get("other_gate_pass", {})) != expected_other:
            raise ValueError(f"V1 other-gate record is incomplete for {stage}")
        for gate in expected_other:
            if gates.get(gate, {}).get("pass") is not True \
                    or record["other_gate_pass"].get(gate) is not True:
                raise ValueError(
                    f"V1 did not pass recorded non-cue gate {gate}/{stage}")


def _validate_spec_v2(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 2:
        raise ValueError("shell-game V2 spec schema_version must be 2")
    if spec.get("protocol_status") \
            != "locked_before_v2_development_and_formal_run":
        raise ValueError("V2 is not marked locked before development/formal use")
    amendment = spec.get("amendment", {})
    required_flags = {
        "kind": "pre-formal_salience_amendment",
        "threshold_changed_from_v1": False,
        "semantic_capacity_contract_changed_from_v1": False,
        "carrier_definitions_changed_from_v1": False,
    }
    for key, value in required_flags.items():
        if amendment.get(key) != value:
            raise ValueError(f"V2 amendment.{key} must be {value!r}")

    expected_stages = [
        {"key": stage.key, "display_name": stage.display_name,
         "capacity": stage.capacity}
        for stage in CAPACITY_STAGES
    ]
    if spec.get("semantic_stages") != expected_stages:
        raise ValueError("V2 semantic stages differ from the V1 contract")
    if spec.get("task_contract") != _expected_task_contract():
        raise ValueError("V2 task contract differs from V1")
    if spec.get("cue_salience") != V2_SALIENCE.describe():
        raise ValueError("V2 cue salience differs from the frozen renderer")

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
        raise ValueError("V2 official encoder and predictor must remain frozen")

    selection = spec.get("development_selection", {})
    admission = spec.get("admission", {})
    if selection.get("threshold_changed_from_v1") is not False \
            or selection.get("threshold") != 0.75 \
            or admission.get("cue_initial_slot_accuracy_min") != 0.75:
        raise ValueError("V2 must retain the V1 cue threshold of 0.75")
    development = spec.get("data", {}).get("development", {})
    if development.get("episodes") != (
            selection.get("fit_episodes", 0)
            + selection.get("check_episodes", 0)):
        raise ValueError("development fit/check partition must cover the bank")
    if selection.get("formal_data_must_not_be_collected_before_pass") is not True:
        raise ValueError("formal collection must be gated by development salience")

    seeds = []
    for split in ALL_SPLITS_V2:
        split_data = spec.get("data", {}).get(split, {})
        if int(split_data.get("episodes", 0)) <= 0:
            raise ValueError(f"V2 {split} episodes must be positive")
        seeds.extend((split_data.get("base_seed"),
                      split_data.get("counterfactual_seed")))
    if any(not isinstance(seed, int) for seed in seeds) \
            or len(seeds) != len(set(seeds)):
        raise ValueError("every V2 bank seed must be a distinct integer")
    if set(seeds) & V1_FORMAL_SEEDS:
        raise ValueError("V2 may not reuse a V1 formal seed")

    if resolve_path_v2(spec["artifacts"]["root"]).resolve() \
            == resolve_path_v2(
                amendment["parent_v1"]["output_root"]).resolve():
        raise ValueError("V2 must use a new output root")
    launcher = spec.get("launcher", {})
    if launcher.get("allowed_devices") != list(ALLOWED_DEVICES_V2) \
            or launcher.get("preview_by_default") is not True \
            or launcher.get("explicit_execute_required") is not True \
            or launcher.get("jobs_per_gpu") != 1:
        raise ValueError("V2 launcher safety policy differs")
    _validate_parent_evidence(spec)


def load_locked_spec_v2(
        spec_path: str | Path = DEFAULT_SPEC_V2,
        lock_path: str | Path = DEFAULT_LOCK_V2,
        ) -> dict[str, Any]:
    """Load V2 only when its spec, producers, and V1 evidence all match."""

    spec_path = resolve_path_v2(spec_path)
    lock_path = resolve_path_v2(lock_path)
    if not spec_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError("formal shell-game V2 spec and lock are required")
    lock = json.loads(lock_path.read_text())
    if lock.get("schema") != LOCK_SCHEMA_V2 \
            or lock.get("immutable") is not True:
        raise ValueError("invalid or mutable shell-game V2 lock")
    actual_spec_hash = sha256_file(spec_path)
    if lock.get("spec_sha256") != actual_spec_hash:
        raise ValueError("shell-game V2 spec hash differs from its lock")
    if resolve_path_v2(lock.get("spec_path", "")).resolve() \
            != spec_path.resolve():
        raise ValueError("V2 lock points to a different specification")
    for source, expected_hash in lock.get("producer_sha256", {}).items():
        source_path = resolve_path_v2(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"locked V2 producer missing: {source}")
        if sha256_file(source_path) != expected_hash:
            raise ValueError(f"locked V2 producer changed: {source}")
    for evidence, expected_hash in lock.get(
            "parent_evidence_sha256", {}).items():
        evidence_path = resolve_path_v2(evidence)
        if not evidence_path.is_file() \
                or sha256_file(evidence_path) != expected_hash:
            raise ValueError(f"locked V1 failure evidence changed: {evidence}")
    spec = yaml.safe_load(spec_path.read_text())
    if not isinstance(spec, dict):
        raise ValueError("shell-game V2 spec must be a YAML mapping")
    _validate_spec_v2(spec)
    spec["_lock_record"] = {
        "path": str(lock_path.resolve()),
        "sha256": sha256_file(lock_path),
        "spec_sha256": actual_spec_hash,
        "producer_sha256": lock["producer_sha256"],
    }
    return spec


__all__ = [
    "ALL_SPLITS_V2",
    "ALLOWED_DEVICES_V2",
    "DEFAULT_LOCK_V2",
    "DEFAULT_SPEC_V2",
    "FORMAL_SPLITS_V2",
    "load_locked_spec_v2",
    "resolve_path_v2",
    "validate_device_v2",
]
