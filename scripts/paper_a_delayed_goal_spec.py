#!/usr/bin/env python3
"""Fail-closed loader for the delayed-goal consumer-use specification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/paper_a_delayed_goal_use.yaml"
TASKS = ("t1", "t3")
SEEDS = (0, 1, 2, 3, 4)
CARRIER_ARMS = ("none", "gru", "ssm", "fixed_trust")
REPAIR_ARMS = ("gru", "ssm")
REPAIR_CONDITIONS = ("objective_off", "cue_repair")
REPAIR_SOURCE_IDS = tuple(
    f"{arm}_{condition}" for arm in REPAIR_ARMS
    for condition in REPAIR_CONDITIONS)
SOURCE_IDS = (*CARRIER_ARMS, *REPAIR_SOURCE_IDS,
              "long_context_56", "cue_window")
ALLOWED_DEVICES = ("cuda:1", "cuda:2")
FORBIDDEN_DEVICES = ("cuda:0", "cuda:3")


class DelayedGoalSpecError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DelayedGoalSpecError(f"{label} must be a mapping")
    return value


def resolve_path(value: Any, *, root: Path = ROOT) -> Path:
    if not isinstance(value, str) or not value:
        raise DelayedGoalSpecError("artifact path must be repository-relative")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise DelayedGoalSpecError(f"unsafe artifact path {value!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise DelayedGoalSpecError(f"artifact path leaves repository: {value}") from error
    return path


def validate_spec(spec: Mapping[str, Any], *, root: Path = ROOT) -> None:
    if spec.get("schema_version") != 1 \
            or spec.get("study") != "paper-a-delayed-goal-use-v1":
        raise DelayedGoalSpecError("unexpected delayed-goal study identity")
    parent = _mapping(spec.get("parent"), "parent")
    output = _mapping(spec.get("output"), "output")
    tasks = _mapping(spec.get("tasks"), "tasks")
    consumer = _mapping(spec.get("consumer"), "consumer")
    choice = _mapping(spec.get("executed_choice"), "executed_choice")
    controls = _mapping(spec.get("controls"), "controls")
    endpoints = _mapping(spec.get("endpoints"), "endpoints")
    execution = _mapping(spec.get("execution"), "execution")
    repair = _mapping(spec.get("repair"), "repair")

    if tuple(tasks) != TASKS or spec.get("checkpoint_seeds") != list(SEEDS):
        raise DelayedGoalSpecError("task or checkpoint-seed grid changed")
    expected_tasks = {
        "t1": ("Transient-marker recall", "transient-marker-recall", 4),
        "t3": ("Drifting-color recall", "drifting-color-recall", 4),
    }
    for task, (name, slug, classes) in expected_tasks.items():
        if tasks[task] != {"name": name, "slug": slug, "classes": classes}:
            raise DelayedGoalSpecError("task semantic identity changed")
    sources = spec.get("representation_sources")
    if not isinstance(sources, list) or tuple(
            source.get("id") for source in sources) != SOURCE_IDS:
        raise DelayedGoalSpecError("representation-source grid changed")
    slugs = [source.get("slug") for source in sources]
    if any(not isinstance(slug, str) or not slug for slug in slugs) \
            or len(slugs) != len(set(slugs)):
        raise DelayedGoalSpecError("representation-source slugs are invalid")
    if consumer.get("fitting_sources") != list(SOURCE_IDS):
        raise DelayedGoalSpecError("shared-consumer source deck changed")
    if consumer.get("interface_dimension") != 768 \
            or consumer.get("decision_index") != 63:
        raise DelayedGoalSpecError("consumer interface changed")
    if consumer.get("scope") != \
            "one arm-blind consumer per task and checkpoint seed":
        raise DelayedGoalSpecError("consumer is no longer shared across arms")
    if consumer.get("labels_available_during_fit") is not True \
            or consumer.get("validation_labels_available_during_fit") is not False:
        raise DelayedGoalSpecError("validation labels may not enter consumer fitting")
    if consumer.get("model") != "StandardScaler+LogisticRegression" \
            or consumer.get("solver") != "lbfgs" \
            or consumer.get("logistic_c") != 1.0 \
            or consumer.get("max_iter") != 2000:
        raise DelayedGoalSpecError("shared-consumer fitting protocol changed")
    if choice.get("executed_horizon") != 80 \
            or len(choice.get("joint_goals", [])) != 4:
        raise DelayedGoalSpecError("executed-choice protocol changed")
    if float(choice.get("oracle_success_min", -1)) != 0.90:
        raise DelayedGoalSpecError("oracle health gate changed")
    if choice.get("role") != "external downstream delayed-goal choice task" \
            or choice.get("environment") != "dm_control/reacher/easy":
        raise DelayedGoalSpecError("executed-choice interpretation changed")
    if controls.get("label_shuffle", {}).get("enabled") is not True \
            or controls.get("action_time_shortcut", {}).get("enabled") is not True:
        raise DelayedGoalSpecError("mandatory shortcut controls are disabled")
    bootstrap = _mapping(endpoints.get("bootstrap"), "endpoints.bootstrap")
    if bootstrap.get("draws") != 20000 \
            or bootstrap.get("method") != (
                "crossed paired checkpoint-seed and validation-episode "
                "percentile bootstrap"):
        raise DelayedGoalSpecError("bootstrap contract changed")
    if repair.get("implemented") is not True \
            or repair.get("eligible_arms") != list(REPAIR_ARMS) \
            or repair.get("conditions") != list(REPAIR_CONDITIONS):
        raise DelayedGoalSpecError("repair grid changed")
    if repair.get("initialization") != \
            "authenticated parent carrier checkpoint" \
            or repair.get("target_gradient") != "stop_gradient" \
            or repair.get("read") != (
                "prior_read[:,63] computed before consuming z[:,63]"):
        raise DelayedGoalSpecError("repair causal contract changed")
    if repair.get("repair_objective_forbidden_inputs") != [
            "xi", "z[:,63]", "validation data"]:
        raise DelayedGoalSpecError("repair leakage guard changed")
    weights = _mapping(
        repair.get("cue_repair_weight"), "repair.cue_repair_weight")
    if weights != {"objective_off": 0.0, "cue_repair": 1.0}:
        raise DelayedGoalSpecError("repair objective-off ablation changed")
    expected_repair = {
        "epochs": 20, "batch_size": 64, "learning_rate": 0.0001,
        "weight_decay": 0.00001, "next_latent_weight": 1.0,
        "repair_head": "Linear(192,768)",
        "target_normalization": (
            "per-coordinate training-bank mean and standard deviation"),
    }
    if any(repair.get(key) != value
           for key, value in expected_repair.items()):
        raise DelayedGoalSpecError("repair optimization protocol changed")
    if execution.get("allowed_devices") != list(ALLOWED_DEVICES) \
            or execution.get("forbidden_devices") != list(FORBIDDEN_DEVICES) \
            or execution.get("require_explicit_execute") is not True:
        raise DelayedGoalSpecError("execution safety contract changed")

    parent_root = resolve_path(parent.get("root"), root=root)
    for key in ("cache_root", "checkpoint_root"):
        path = resolve_path(parent.get(key), root=root)
        if parent_root not in path.parents:
            raise DelayedGoalSpecError(f"parent.{key} leaves parent.root")
    output_root = resolve_path(output.get("root"), root=root)
    if parent_root == output_root or parent_root in output_root.parents:
        raise DelayedGoalSpecError("use outputs overlap parent artifacts")
    for key in ("repairs", "evaluations", "summary", "logs"):
        path = resolve_path(output.get(key), root=root)
        if output_root not in path.parents:
            raise DelayedGoalSpecError(f"output.{key} is outside output.root")

    records = [parent.get("config"), parent.get("summary"),
               parent.get("official_weights")]
    for split in ("train_caches", "validation_caches"):
        values = _mapping(parent.get(split), f"parent.{split}")
        if set(values) != set(TASKS):
            raise DelayedGoalSpecError(f"parent.{split} task grid changed")
        records.extend(values.values())
    for raw in records:
        record = _mapping(raw, "parent artifact record")
        resolve_path(record.get("path"), root=root)
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise DelayedGoalSpecError("invalid parent artifact digest")


def load_locked_spec(path: Path = DEFAULT_SPEC, *, verify_parent: bool = True,
                     root: Path = ROOT) -> dict[str, Any]:
    path = path.resolve()
    lock = path.with_suffix(".sha256")
    if not path.is_file() or not lock.is_file():
        raise DelayedGoalSpecError(f"missing locked use spec: {path}")
    tokens = lock.read_text().strip().split()
    actual = sha256_file(path)
    if tokens != [actual, path.name]:
        raise DelayedGoalSpecError("delayed-goal spec hash mismatch")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise DelayedGoalSpecError(f"cannot parse use spec: {error}") from error
    spec = dict(_mapping(raw, "delayed-goal spec"))
    validate_spec(spec, root=root)
    spec["_spec_record"] = {
        "path": str(path.relative_to(root.resolve())),
        "sha256": actual,
    }
    if verify_parent:
        parent = spec["parent"]
        records = [parent["config"], parent["summary"],
                   parent["official_weights"]]
        records.extend(parent["train_caches"].values())
        records.extend(parent["validation_caches"].values())
        for record in records:
            source = resolve_path(record["path"], root=root)
            if not source.is_file() or sha256_file(source) != record["sha256"]:
                raise DelayedGoalSpecError(
                    f"parent artifact changed: {record['path']}")
    return spec


def validate_device(spec: Mapping[str, Any], device: str,
                    *, allow_cpu: bool = False) -> None:
    if allow_cpu and device == "cpu":
        return
    if device in spec["execution"]["forbidden_devices"]:
        raise DelayedGoalSpecError(f"device {device} is explicitly forbidden")
    if device not in spec["execution"]["allowed_devices"]:
        raise DelayedGoalSpecError(f"device {device} is not allowed")


def task_slug(spec: Mapping[str, Any], task: str) -> str:
    try:
        return str(spec["tasks"][task]["slug"])
    except KeyError as error:
        raise DelayedGoalSpecError(f"unknown task {task!r}") from error


def source_slug(spec: Mapping[str, Any], source: str) -> str:
    for record in spec["representation_sources"]:
        if record["id"] == source:
            return str(record["slug"])
    raise DelayedGoalSpecError(f"unknown representation source {source!r}")


def repair_directory(spec: Mapping[str, Any], task: str, arm: str,
                     seed: int, condition: str, *, root: Path = ROOT) -> Path:
    if arm not in REPAIR_ARMS or condition not in REPAIR_CONDITIONS \
            or seed not in SEEDS:
        raise DelayedGoalSpecError("repair cell is outside the locked grid")
    return (resolve_path(spec["output"]["repairs"], root=root)
            / task_slug(spec, task) / source_slug(spec, arm)
            / f"checkpoint-seed-{seed}" / condition.replace("_", "-"))


def evaluation_directory(spec: Mapping[str, Any], task: str, seed: int,
                         *, root: Path = ROOT) -> Path:
    if seed not in SEEDS:
        raise DelayedGoalSpecError("evaluation seed is outside the locked grid")
    return (resolve_path(spec["output"]["evaluations"], root=root)
            / task_slug(spec, task) / f"checkpoint-seed-{seed}")
