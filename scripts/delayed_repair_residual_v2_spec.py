#!/usr/bin/env python3
"""Fail-closed specification and provenance loader for delayed repair V2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts.delayed_repair_residual_v2_objective import (
    cue_residual_target,
    development_health,
    load_label_free_bank,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/paper_a_delayed_repair_residual_v2.yaml"
DEFAULT_LOCK = ROOT / "configs/paper_a_delayed_repair_residual_v2.lock.json"
LOCK_SCHEMA = "paper_a_delayed_repair_residual_lock_v2"
TASKS = ("transient-marker-recall", "drifting-color-recall")
ARMS = ("gru", "ssm")
CONDITIONS = ("objective-off", "cue-residual-repair")
SEEDS = (0, 1, 2, 3, 4)
ALLOWED_DEVICES = ("cuda:1", "cuda:2")
FORBIDDEN_DEVICES = ("cuda:0", "cuda:3")


class ResidualRepairSpecError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def resolve_path(value: Any, *, root: Path = ROOT) -> Path:
    if not isinstance(value, str) or not value:
        raise ResidualRepairSpecError("artifact path must be repository-relative")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ResidualRepairSpecError(f"unsafe artifact path {value!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ResidualRepairSpecError(
            f"artifact path leaves repository: {value}") from error
    return path


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ResidualRepairSpecError(f"{label} must be a mapping")
    return value


def _tree_digest(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _glob_tree_digest(root: Path, pattern: str
                      ) -> tuple[int, str]:
    paths = sorted(root.glob(pattern))
    return len(paths), _tree_digest(root, paths)


def _validate_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("schema_version") != 2 \
            or spec.get("study") \
            != "paper-a-delayed-repair-cue-residual-v2" \
            or spec.get("protocol_status") \
            != "locked_before_development_health_and_gpu_training":
        raise ResidualRepairSpecError("unexpected residual-repair V2 identity")
    role = _mapping(spec.get("scientific_role"), "scientific_role")
    if role.get("classification") != "post_v1_diagnostic_repair" \
            or role.get("preregistered_primary_result") is not False \
            or role.get("downstream_label_use_claim") is not False:
        raise ResidualRepairSpecError(
            "V2 must remain an explicitly post-V1 non-primary diagnostic")
    if tuple(spec.get("tasks", {})) != TASKS:
        raise ResidualRepairSpecError("semantic task grid changed")
    expected_parent_ids = {
        "transient-marker-recall": "t1",
        "drifting-color-recall": "t3",
    }
    for task, parent_id in expected_parent_ids.items():
        record = _mapping(spec["tasks"][task], f"tasks.{task}")
        if record.get("parent_task_id") != parent_id:
            raise ResidualRepairSpecError(f"parent identity changed for {task}")
        states = _mapping(
            record.get("parent_carrier_state_sha256"),
            f"tasks.{task}.parent_carrier_state_sha256")
        if tuple(states) != ARMS:
            raise ResidualRepairSpecError(f"parent arm ledger changed for {task}")
        for arm in ARMS:
            if set(states[arm]) != set(SEEDS) or any(
                    not isinstance(value, str) or len(value) != 64
                    for value in states[arm].values()):
                raise ResidualRepairSpecError(
                    f"parent state ledger is incomplete for {task}/{arm}")

    target = _mapping(spec.get("cue_residual_target"), "cue_residual_target")
    expected_target = {
        "dimension": 192,
        "cue_summary": "mean frozen z[t] for cue_on <= t < cue_off",
        "scene_baseline": "0.5 * (z[cue_on-1] + z[cue_off])",
        "residual": "cue_summary - scene_baseline",
        "interpolation_equivalence": (
            "mean linear interpolation between immediate pre-cue and "
            "post-cue endpoints"),
        "standardization": (
            "per-coordinate training-cache mean and population standard deviation"),
        "scale_floor": 0.000001,
        "target_gradient": "stop_gradient",
        "repair_head": "Linear(192,192)",
        "read": "prior_read[:,63] before consuming z[:,63]",
        "forbidden_inputs": [
            "xi", "z[:,63]", "validation optimization",
            "development normalization in formal training"],
    }
    if target != expected_target:
        raise ResidualRepairSpecError("cue-residual target contract changed")

    health = _mapping(spec.get("development_health"), "development_health")
    expected_health = {
        "required_episodes": 240,
        "cue_duration_min": 4,
        "cue_duration_max": 6,
        "coordinate_std_min": 0.05,
        "coordinate_std_fraction_min": 1.0,
        "median_episode_residual_rms_min": 0.1,
        "median_episode_residual_rms_max": 10.0,
        "all_target_indices_before_decision": True,
        "labels_loaded": False,
        "formal_training_performed": False,
    }
    if any(health.get(key) != value for key, value in expected_health.items()):
        raise ResidualRepairSpecError("development health rule changed")

    repair = _mapping(spec.get("formal_repair"), "formal_repair")
    expected_repair = {
        "arms": list(ARMS),
        "conditions": list(CONDITIONS),
        "checkpoint_seeds": list(SEEDS),
        "initialization": "authenticated original parent carrier checkpoint",
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "next_latent_weight": 1.0,
        "cue_residual_weight": {
            "objective-off": 0.0, "cue-residual-repair": 1.0},
        "next_latent_windows_per_batch": 8,
        "gradient_clip_norm": 1.0,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "torch_seed_base": 170000,
        "batch_plan_seed_base": 270000,
    }
    if any(repair.get(key) != value
           for key, value in expected_repair.items()):
        raise ResidualRepairSpecError("formal repair/twin protocol changed")

    endpoints = _mapping(spec.get("formal_endpoints"), "formal_endpoints")
    if endpoints.get("bootstrap_draws") != 20000 \
            or endpoints.get("bootstrap_seed") != 20260708 \
            or endpoints.get("primary_diagnostic") != (
                "validation cue-residual MSE normalized by the "
                "training-mean zero-predictor MSE"):
        raise ResidualRepairSpecError("formal diagnostic endpoint changed")
    execution = _mapping(spec.get("execution"), "execution")
    if execution.get("allowed_devices") != list(ALLOWED_DEVICES) \
            or execution.get("forbidden_devices") != list(FORBIDDEN_DEVICES) \
            or execution.get("preview_by_default") is not True \
            or execution.get("require_explicit_execute") is not True \
            or execution.get("jobs_per_gpu") != 1:
        raise ResidualRepairSpecError("execution safety contract changed")

    output = _mapping(spec.get("output"), "output")
    output_root = resolve_path(output.get("root"))
    if output_root != resolve_path(
            "outputs/paper_a_delayed_repair_residual_v2"):
        raise ResidualRepairSpecError("V2 output root changed")
    forbidden_roots = (
        resolve_path(spec["parent"]["expansion_root"]),
        resolve_path(spec["parent_v1_diagnostic"]["output_root"]),
    )
    if any(output_root == parent or parent in output_root.parents
           for parent in forbidden_roots):
        raise ResidualRepairSpecError("V2 output overlaps a parent study")
    for key in ("development", "repairs", "summary", "logs"):
        path = resolve_path(output[key])
        if output_root not in path.parents:
            raise ResidualRepairSpecError(f"output.{key} leaves the V2 root")


def _verify_record(record: Mapping[str, Any], label: str) -> None:
    path = resolve_path(record.get("path"))
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ResidualRepairSpecError(f"parent artifact changed: {label}")


def _verify_parent(spec: Mapping[str, Any]) -> None:
    parent = spec["parent"]
    for key in (
            "expansion_config", "expansion_summary", "robustness_config",
            "official_weights"):
        _verify_record(parent[key], f"parent.{key}")
    v1 = spec["parent_v1_diagnostic"]
    _verify_record(v1["spec"], "parent_v1.spec")
    _verify_record(v1["lock"], "parent_v1.lock")
    v1_root = resolve_path(v1["output_root"])
    for key in ("repair_metrics_tree", "evaluation_metrics_tree"):
        record = v1[key]
        count, digest = _glob_tree_digest(v1_root, record["pattern"])
        if count != record["count"] or digest != record["sha256"]:
            raise ResidualRepairSpecError(f"V1 diagnostic evidence changed: {key}")

    for task in TASKS:
        record = spec["tasks"][task]
        for split in ("training_cache", "validation_cache"):
            _verify_record(record[split], f"tasks.{task}.{split}")
        development = record["development_cache"]
        _verify_record(development, f"tasks.{task}.development_cache")
        path = resolve_path(development["path"])
        sidecar = path.with_suffix(path.suffix + ".json")
        manifest = path.with_name("manifest.json")
        if sha256_file(sidecar) != development["sidecar_sha256"] \
                or sha256_file(manifest) != development["manifest_sha256"]:
            raise ResidualRepairSpecError(
                f"development cache provenance changed for {task}")

    checkpoint_root = resolve_path(parent["checkpoint_root"])
    carrier_paths, metric_paths = [], []
    for task in TASKS:
        parent_id = spec["tasks"][task]["parent_task_id"]
        for arm in ARMS:
            for seed in SEEDS:
                directory = checkpoint_root / parent_id / arm / f"s{seed}"
                carrier_paths.append(directory / "carrier.pt")
                metric_paths.append(directory / "metrics.json")
    tree = parent["checkpoint_tree"]
    if len(carrier_paths) != tree["carrier_files"] \
            or _tree_digest(checkpoint_root, carrier_paths) \
            != tree["carrier_sha256"] \
            or len(metric_paths) != tree["metrics_files"] \
            or _tree_digest(checkpoint_root, metric_paths) \
            != tree["metrics_sha256"]:
        raise ResidualRepairSpecError("authenticated parent checkpoint tree changed")


def load_locked_spec(
        spec_path: str | Path = DEFAULT_SPEC,
        lock_path: str | Path = DEFAULT_LOCK,
        *, verify_parent: bool = True,
        ) -> dict[str, Any]:
    spec_path, lock_path = Path(spec_path).resolve(), Path(lock_path).resolve()
    if not spec_path.is_file() or not lock_path.is_file():
        raise ResidualRepairSpecError("missing delayed-repair V2 spec or lock")
    lock = json.loads(lock_path.read_text())
    if lock.get("schema") != LOCK_SCHEMA or lock.get("immutable") is not True:
        raise ResidualRepairSpecError("invalid delayed-repair V2 lock")
    actual = sha256_file(spec_path)
    if actual != lock.get("spec_sha256") \
            or resolve_path(lock.get("spec_path")).resolve() != spec_path:
        raise ResidualRepairSpecError("delayed-repair V2 spec hash mismatch")
    for source, expected in lock.get("producer_sha256", {}).items():
        path = resolve_path(source)
        if not path.is_file() or sha256_file(path) != expected:
            raise ResidualRepairSpecError(f"locked producer changed: {source}")
    spec = yaml.safe_load(spec_path.read_text())
    spec = dict(_mapping(spec, "delayed-repair V2 spec"))
    _validate_spec(spec)
    if verify_parent:
        _verify_parent(spec)
    spec["_lock_record"] = {
        "path": str(lock_path),
        "sha256": sha256_file(lock_path),
        "spec_sha256": actual,
        "producer_sha256": lock["producer_sha256"],
    }
    return spec


def validate_device(device: str) -> str:
    if device in FORBIDDEN_DEVICES:
        raise ResidualRepairSpecError(f"device {device} is explicitly forbidden")
    if device not in ALLOWED_DEVICES:
        raise ResidualRepairSpecError(f"device {device} is not allowed")
    return device


def lock_receipt(spec: Mapping[str, Any]) -> dict[str, str]:
    return {
        "lock_sha256": spec["_lock_record"]["sha256"],
        "spec_sha256": spec["_lock_record"]["spec_sha256"],
    }


def development_receipt_path(spec: Mapping[str, Any], task: str) -> Path:
    if task not in TASKS:
        raise ResidualRepairSpecError(f"unknown semantic task {task!r}")
    return resolve_path(spec["output"]["development"]) / task / "health.json"


def repair_directory(spec: Mapping[str, Any], task: str, arm: str,
                     seed: int, condition: str) -> Path:
    if task not in TASKS or arm not in ARMS or seed not in SEEDS \
            or condition not in CONDITIONS:
        raise ResidualRepairSpecError("formal repair cell is outside the grid")
    return (resolve_path(spec["output"]["repairs"]) / task / arm
            / f"checkpoint-seed-{seed}" / condition)


def build_development_receipt(
        spec: Mapping[str, Any], task: str) -> dict[str, Any]:
    record = spec["tasks"][task]["development_cache"]
    bank = load_label_free_bank(resolve_path(record["path"]),
                                require_actions=False)
    target, audit = cue_residual_target(bank)
    health = development_health(target, audit, spec["development_health"])
    return {
        "schema": "paper_a_delayed_repair_residual_development_v2",
        "study": spec["study"],
        "scientific_role": spec["scientific_role"]["classification"],
        "preregistered_primary_result": False,
        "task": task,
        "display_name": spec["tasks"][task]["display_name"],
        "formal_lock": lock_receipt(spec),
        "source_cache": record,
        "label_arrays_present_but_not_loaded":
            bank["label_arrays_present_but_not_loaded"],
        "label_arrays_loaded": False,
        "formal_training_performed": False,
        "development_statistics_used_for_formal_normalization": False,
        "health": health,
    }


def require_development_health(
        spec: Mapping[str, Any], task: str) -> dict[str, Any]:
    path = development_receipt_path(spec, task)
    if not path.is_file():
        raise FileNotFoundError(
            f"formal delayed repair blocked until development health: {task}")
    actual = json.loads(path.read_text())
    expected = build_development_receipt(spec, task)
    if actual != expected or actual.get("health", {}).get("passed") is not True:
        raise ResidualRepairSpecError(
            f"development health receipt failed or changed for {task}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "passed": True,
        "task": task,
    }


__all__ = [
    "ALLOWED_DEVICES",
    "ARMS",
    "CONDITIONS",
    "DEFAULT_LOCK",
    "DEFAULT_SPEC",
    "FORBIDDEN_DEVICES",
    "ROOT",
    "ResidualRepairSpecError",
    "SEEDS",
    "TASKS",
    "build_development_receipt",
    "development_receipt_path",
    "load_locked_spec",
    "lock_receipt",
    "repair_directory",
    "require_development_health",
    "resolve_path",
    "sha256_file",
    "stable_json",
    "validate_device",
]
