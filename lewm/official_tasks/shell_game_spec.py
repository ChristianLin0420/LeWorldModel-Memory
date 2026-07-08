"""Immutable specification loader for the formal shell-game study."""

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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "configs/official_shell_game_capacity.yaml"
DEFAULT_LOCK = ROOT / "configs/official_shell_game_capacity.lock.json"
LOCK_SCHEMA = "official_shell_game_capacity_lock_v1"
ALLOWED_DEVICES = ("cuda:1", "cuda:2")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_device(device: str) -> str:
    if device not in ALLOWED_DEVICES:
        raise ValueError(
            f"formal shell-game jobs permit only {ALLOWED_DEVICES}; got {device!r}")
    return device


def _validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("shell-game spec schema_version must be 1")
    if spec.get("protocol_status") != "locked_before_formal_run":
        raise ValueError("formal spec is not marked locked_before_formal_run")
    configured = spec.get("semantic_stages")
    expected = [
        {"key": stage.key, "display_name": stage.display_name,
         "capacity": stage.capacity}
        for stage in CAPACITY_STAGES
    ]
    if configured != expected:
        raise ValueError("semantic capacity stages differ from the code contract")
    task = spec.get("task_contract", {})
    reference = ShellGameCapacityContract(CAPACITY_STAGES[-1]).describe()
    exact = {
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
    if task != exact:
        raise ValueError("task_contract differs from the frozen code contract")
    host = spec.get("official_host", {})
    expected_host = {
        "latent_dim": 192,
        "observation_length": 64,
        "context": 3,
        "action_block_dim": 10,
    }
    for key, expected_value in expected_host.items():
        if host.get(key) != expected_value:
            raise ValueError(f"official_host.{key} must be {expected_value}")
    if host.get("frozen_encoder") is not True \
            or host.get("frozen_predictor") is not True:
        raise ValueError("official encoder and predictor must both be frozen")
    launcher = spec.get("launcher", {})
    if launcher.get("allowed_devices") != list(ALLOWED_DEVICES):
        raise ValueError("launcher device allowlist differs from formal policy")
    if launcher.get("preview_by_default") is not True \
            or launcher.get("explicit_execute_required") is not True:
        raise ValueError("launcher must default to preview and require --execute")
    if launcher.get("jobs_per_gpu") != 1:
        raise ValueError("formal launcher requires exactly one job per GPU")


def load_locked_spec(spec_path: str | Path = DEFAULT_SPEC,
                     lock_path: str | Path = DEFAULT_LOCK) -> dict[str, Any]:
    """Load a spec only when its bytes and producer sources match the lock."""

    spec_path, lock_path = resolve_path(spec_path), resolve_path(lock_path)
    if not spec_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError("formal shell-game spec and lock are both required")
    lock = json.loads(lock_path.read_text())
    if lock.get("schema") != LOCK_SCHEMA or lock.get("immutable") is not True:
        raise ValueError("invalid or mutable shell-game lock")
    actual_spec_hash = sha256_file(spec_path)
    if lock.get("spec_sha256") != actual_spec_hash:
        raise ValueError("shell-game spec hash differs from its immutable lock")
    if resolve_path(lock.get("spec_path", "")).resolve() != spec_path.resolve():
        raise ValueError("lock points to a different shell-game spec")
    for source, expected_hash in lock.get("producer_sha256", {}).items():
        source_path = resolve_path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"locked producer source is missing: {source}")
        if sha256_file(source_path) != expected_hash:
            raise ValueError(f"locked producer source changed: {source}")
    spec = yaml.safe_load(spec_path.read_text())
    if not isinstance(spec, dict):
        raise ValueError("shell-game spec must be a YAML mapping")
    _validate_spec(spec)
    spec["_lock_record"] = {
        "path": str(lock_path.resolve()),
        "sha256": sha256_file(lock_path),
        "spec_sha256": actual_spec_hash,
        "producer_sha256": lock["producer_sha256"],
    }
    return spec
