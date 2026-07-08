#!/usr/bin/env python3
"""Fail-closed contracts for the delayed-goal V2 controller amendment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/paper_a_delayed_goal_use_v2.yaml"
TASKS = ("t1", "t3")
SEEDS = (0, 1, 2, 3, 4)
ALLOWED_DEVICES = ("cuda:1", "cuda:2")
FORBIDDEN_DEVICES = ("cuda:0", "cuda:3")
CANDIDATE_IDS = (
    "unchanged-gains-horizon-120",
    "unchanged-gains-horizon-160",
    "unchanged-gains-horizon-200",
    "damped-gains-horizon-160",
)
SOURCE_IDS = (
    "none", "gru", "ssm", "fixed_trust", "gru_objective_off",
    "gru_cue_repair", "ssm_objective_off", "ssm_cue_repair",
    "long_context_56", "cue_window",
)


class DelayedGoalV2SpecError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DelayedGoalV2SpecError(f"{label} must be a mapping")
    return value


def resolve_path(value: Any, *, root: Path = ROOT) -> Path:
    if not isinstance(value, str) or not value:
        raise DelayedGoalV2SpecError("artifact path must be repository-relative")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise DelayedGoalV2SpecError(f"unsafe artifact path {value!r}")
    result = (root / relative).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise DelayedGoalV2SpecError(
            f"artifact path leaves repository: {value}") from error
    return result


def _artifact_record(value: Any, label: str, *, root: Path) -> Mapping[str, Any]:
    record = _mapping(value, label)
    resolve_path(record.get("path"), root=root)
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise DelayedGoalV2SpecError(f"{label} has an invalid digest")
    return record


def validate_spec(spec: Mapping[str, Any], *, root: Path = ROOT) -> None:
    if spec.get("schema_version") != 1 or spec.get("study") != (
            "paper-a-delayed-goal-use-v2-controller-health-amendment"):
        raise DelayedGoalV2SpecError("unexpected V2 study identity")
    if spec.get("amendment_scope") != "controller-health-only":
        raise DelayedGoalV2SpecError("V2 scope is not controller-health-only")
    v1 = _mapping(spec.get("v1"), "v1")
    parent = _mapping(spec.get("parent"), "parent")
    output = _mapping(spec.get("output"), "output")
    tasks = _mapping(spec.get("tasks"), "tasks")
    development = _mapping(spec.get("development"), "development")
    controller = _mapping(spec.get("executed_choice"), "executed_choice")
    controls = _mapping(spec.get("controls"), "controls")
    endpoints = _mapping(spec.get("endpoints"), "endpoints")
    execution = _mapping(spec.get("execution"), "execution")

    if tuple(tasks) != TASKS or spec.get("checkpoint_seeds") != list(SEEDS):
        raise DelayedGoalV2SpecError("V2 task/seed grid changed")
    expected_tasks = {
        "t1": {"name": "Transient-marker recall",
               "slug": "transient-marker-recall", "classes": 4},
        "t3": {"name": "Drifting-color recall",
               "slug": "drifting-color-recall", "classes": 4},
    }
    if tasks != expected_tasks:
        raise DelayedGoalV2SpecError("V2 semantic task identity changed")
    sources = spec.get("representation_sources")
    if not isinstance(sources, list) or tuple(
            source.get("id") for source in sources) != SOURCE_IDS \
            or len({source.get("slug") for source in sources}) != len(SOURCE_IDS):
        raise DelayedGoalV2SpecError("V2 representation-source grid changed")
    if v1.get("repair_retraining_permitted") is not False \
            or v1.get("reuse_all_v1_repairs") is not True:
        raise DelayedGoalV2SpecError("V2 no-retraining contract changed")
    _artifact_record(v1.get("spec"), "v1.spec", root=root)
    _artifact_record(
        v1.get("provenance_manifest"), "v1.provenance_manifest", root=root)
    _artifact_record(parent.get("config"), "parent.config", root=root)
    resolve_path(parent.get("checkpoint_root"), root=root)
    for split in ("train_caches", "validation_caches"):
        records = _mapping(parent.get(split), f"parent.{split}")
        if tuple(records) != TASKS:
            raise DelayedGoalV2SpecError(f"parent.{split} task grid changed")
        for task, record in records.items():
            _artifact_record(record, f"parent.{split}.{task}", root=root)

    if development.get("source_split") != "parent training bank only" \
            or development.get("validation_data_permitted") is not False \
            or development.get("episodes_per_class") != 60 \
            or development.get("selection_seed_base") != 20260708:
        raise DelayedGoalV2SpecError("controller development split changed")
    index_hashes = _mapping(
        development.get("index_sha256"), "development.index_sha256")
    if tuple(index_hashes) != TASKS or any(
            not isinstance(value, str) or len(value) != 64
            for value in index_hashes.values()):
        raise DelayedGoalV2SpecError("development index hashes changed")

    fixed = _mapping(controller.get("fixed"), "executed_choice.fixed")
    reference = _mapping(
        controller.get("v1_reference"), "executed_choice.v1_reference")
    if fixed.get("environment") != "dm_control/reacher/easy" \
            or fixed.get("decision_index") != 63 \
            or fixed.get("success_tolerance_radians") != 0.35 \
            or fixed.get("return_scale_radians") != 0.50 \
            or fixed.get("joint_goals") != [
                [-1.2, -0.6], [-0.4, 1.2], [0.4, -1.2], [1.2, 0.6]]:
        raise DelayedGoalV2SpecError("fixed goal/cost protocol changed")
    if reference != {"executed_horizon": 80, "proportional_gain": 1.5,
                     "derivative_gain": 0.25}:
        raise DelayedGoalV2SpecError("V1 controller reference changed")
    candidates = controller.get("candidates")
    if not isinstance(candidates, list) or tuple(
            item.get("id") for item in candidates) != CANDIDATE_IDS:
        raise DelayedGoalV2SpecError("controller candidate deck changed")
    expected_candidates = (
        (120, 1.5, 0.25), (160, 1.5, 0.25),
        (200, 1.5, 0.25), (160, 1.25, 0.35),
    )
    for priority, (candidate, expected) in enumerate(
            zip(candidates, expected_candidates, strict=True), start=1):
        actual = (candidate.get("executed_horizon"),
                  candidate.get("proportional_gain"),
                  candidate.get("derivative_gain"))
        if candidate.get("priority") != priority or actual != expected:
            raise DelayedGoalV2SpecError("controller candidate changed")
    selection = _mapping(
        controller.get("development_selection"),
        "executed_choice.development_selection")
    if selection.get("per_task_oracle_success_min") != 0.925 \
            or selection.get("per_class_oracle_success_min") != 0.85 \
            or selection.get("rule") != (
                "select first priority-ordered candidate passing every "
                "task and class gate; otherwise stop without validation"):
        raise DelayedGoalV2SpecError("controller selection rule changed")
    lock = _mapping(controller.get("lock"), "executed_choice.lock")
    for key in ("path", "sha256_path"):
        resolve_path(lock.get(key), root=root)
    if lock.get("must_precede_validation") is not True \
            or lock.get("refuse_overwrite") is not True:
        raise DelayedGoalV2SpecError("controller lock contract changed")

    if controller.get("validation_oracle_success_min") != 0.90:
        raise DelayedGoalV2SpecError("V2 oracle validation gate changed")
    if controls.get("shortcut_accuracy_max") != 0.35 \
            or controls.get("label_shuffle", {}).get("enabled") is not True \
            or controls.get("action_time_shortcut", {}).get("enabled") is not True:
        raise DelayedGoalV2SpecError("V2 shortcut gates changed")
    bootstrap = _mapping(endpoints.get("bootstrap"), "endpoints.bootstrap")
    if bootstrap.get("draws") != 20000 or bootstrap.get("method") != (
            "crossed paired checkpoint-seed and validation-episode "
            "percentile bootstrap"):
        raise DelayedGoalV2SpecError("V2 inference contract changed")
    if execution.get("allowed_devices") != list(ALLOWED_DEVICES) \
            or execution.get("forbidden_devices") != list(FORBIDDEN_DEVICES) \
            or execution.get("require_explicit_execute") is not True \
            or execution.get("controller_selection_gl_backend") != "glfw":
        raise DelayedGoalV2SpecError("V2 execution safety changed")

    v1_root = resolve_path(v1.get("output_root"), root=root)
    output_root = resolve_path(output.get("root"), root=root)
    if output_root == v1_root or v1_root in output_root.parents \
            or output_root in v1_root.parents:
        raise DelayedGoalV2SpecError("V2 outputs overlap V1")
    for key in ("controller_selection", "evaluations", "summary", "logs"):
        path = resolve_path(output.get(key), root=root)
        if output_root not in path.parents:
            raise DelayedGoalV2SpecError(f"output.{key} leaves V2 root")


def load_locked_spec(path: Path = DEFAULT_SPEC, *, verify_artifacts: bool = True,
                     root: Path = ROOT) -> dict[str, Any]:
    path = path.resolve()
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise DelayedGoalV2SpecError(f"missing locked V2 spec: {path}")
    actual = sha256_file(path)
    if sidecar.read_text().strip().split() != [actual, path.name]:
        raise DelayedGoalV2SpecError("V2 spec hash mismatch")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise DelayedGoalV2SpecError(f"cannot parse V2 spec: {error}") from error
    spec = dict(_mapping(raw, "V2 spec"))
    validate_spec(spec, root=root)
    spec["_spec_record"] = {
        "path": str(path.relative_to(root.resolve())), "sha256": actual}
    if verify_artifacts:
        records = [spec["v1"]["spec"], spec["v1"]["provenance_manifest"],
                   spec["parent"]["config"]]
        for task in TASKS:
            records.extend((spec["parent"]["train_caches"][task],
                            spec["parent"]["validation_caches"][task]))
        for record in records:
            artifact = resolve_path(record["path"], root=root)
            if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
                raise DelayedGoalV2SpecError(
                    f"authenticated V2 input changed: {record['path']}")
        load_v1_provenance(spec, root=root, verify_artifacts=True)
    return spec


def load_v1_provenance(spec: Mapping[str, Any], *, root: Path = ROOT,
                       verify_artifacts: bool = True) -> dict[str, Any]:
    record = spec["v1"]["provenance_manifest"]
    path = resolve_path(record["path"], root=root)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DelayedGoalV2SpecError(
            f"cannot read V1 provenance manifest: {error}") from error
    if value.get("schema_version") != 1 \
            or value.get("study") != "paper-a-delayed-goal-v2-v1-provenance" \
            or len(value.get("repair_checkpoints", [])) != 40:
        raise DelayedGoalV2SpecError("V1 provenance manifest identity changed")
    failure = _mapping(value.get("v1_failure"), "v1_failure")
    if failure.get("failed_gate") != "label_oracle_success" \
            or failure.get("observed_oracle_success") != 0.8875 \
            or failure.get("registered_oracle_success_min") != 0.90 \
            or failure.get("shortcut_gates_passed") is not True \
            or failure.get("valid_for_use_claim") is not False:
        raise DelayedGoalV2SpecError("V1 failure evidence changed")
    if verify_artifacts:
        records = value.get("repair_checkpoints", []) + failure.get(
            "completed_evaluations", []) + value.get("v1_code", [])
        for record in records:
            artifact = resolve_path(record.get("path"), root=root)
            if not artifact.is_file() or sha256_file(artifact) != record.get(
                    "sha256"):
                raise DelayedGoalV2SpecError(
                    f"sealed V1 artifact changed: {record.get('path')}")
    return value


def validate_device(spec: Mapping[str, Any], device: str) -> None:
    if device in spec["execution"]["forbidden_devices"]:
        raise DelayedGoalV2SpecError(f"device {device} is explicitly forbidden")
    if device not in spec["execution"]["allowed_devices"]:
        raise DelayedGoalV2SpecError(f"device {device} is not allowed")


def task_slug(spec: Mapping[str, Any], task: str) -> str:
    if task not in TASKS:
        raise DelayedGoalV2SpecError(f"unknown task {task!r}")
    return str(spec["tasks"][task]["slug"])


def evaluation_directory(spec: Mapping[str, Any], task: str, seed: int,
                         *, root: Path = ROOT) -> Path:
    if task not in TASKS or seed not in SEEDS:
        raise DelayedGoalV2SpecError("V2 evaluation cell leaves locked grid")
    return (resolve_path(spec["output"]["evaluations"], root=root)
            / task_slug(spec, task) / f"checkpoint-seed-{seed}")


def development_indices(labels: np.ndarray, task: str,
                        spec: Mapping[str, Any]) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (1200,) or sorted(np.unique(labels).tolist()) != [0, 1, 2, 3]:
        raise DelayedGoalV2SpecError("unexpected parent training labels")
    task_offset = TASKS.index(task)
    rng = np.random.default_rng(
        int(spec["development"]["selection_seed_base"]) + task_offset)
    per_class = int(spec["development"]["episodes_per_class"])
    selected = np.concatenate([
        rng.permutation(np.flatnonzero(labels == category))[:per_class]
        for category in range(4)
    ])
    selected = np.sort(selected).astype("<i8")
    digest = hashlib.sha256(selected.tobytes()).hexdigest()
    if digest != spec["development"]["index_sha256"][task]:
        raise DelayedGoalV2SpecError(
            f"development subset digest mismatch for {task}")
    return selected


def controller_protocol(spec: Mapping[str, Any], candidate_id: str
                        ) -> dict[str, Any]:
    candidates = spec["executed_choice"]["candidates"]
    try:
        candidate = next(item for item in candidates
                         if item["id"] == candidate_id)
    except StopIteration as error:
        raise DelayedGoalV2SpecError(
            f"unknown controller candidate {candidate_id!r}") from error
    fixed = spec["executed_choice"]["fixed"]
    return {
        "environment": fixed["environment"],
        "initialization": fixed["initialization"],
        "executed_horizon": int(candidate["executed_horizon"]),
        "action_policy": fixed["action_policy"],
        "proportional_gain": float(candidate["proportional_gain"]),
        "derivative_gain": float(candidate["derivative_gain"]),
        "joint_goals": fixed["joint_goals"],
        "distance": fixed["distance"],
        "success_tolerance_radians": float(
            fixed["success_tolerance_radians"]),
        "return": fixed["return"],
        "return_scale_radians": float(fixed["return_scale_radians"]),
        "oracle": fixed["oracle"],
        "oracle_success_min": float(
            spec["executed_choice"]["validation_oracle_success_min"]),
    }


def select_candidate_id(spec: Mapping[str, Any],
                        candidate_results: list[Mapping[str, Any]]) -> str | None:
    by_id = {result.get("id"): result for result in candidate_results}
    if set(by_id) != set(CANDIDATE_IDS):
        raise DelayedGoalV2SpecError("candidate result grid is incomplete")
    for candidate in spec["executed_choice"]["candidates"]:
        result = by_id[candidate["id"]]
        if result.get("development_gate_pass") is True:
            return str(candidate["id"])
    return None


def controller_lock_paths(spec: Mapping[str, Any], *, root: Path = ROOT
                          ) -> tuple[Path, Path]:
    lock = spec["executed_choice"]["lock"]
    return (resolve_path(lock["path"], root=root),
            resolve_path(lock["sha256_path"], root=root))


def validate_controller_lock_payload(payload: Mapping[str, Any],
                                     spec: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1 \
            or payload.get("study") != spec["study"] \
            or payload.get("spec") != spec["_spec_record"]:
        raise DelayedGoalV2SpecError("controller lock identity changed")
    if payload.get("development_source") != "parent training bank only" \
            or payload.get("validation_data_accessed") is not False \
            or payload.get("validation_artifacts_absent_at_lock") is not True:
        raise DelayedGoalV2SpecError("controller lock used validation state")
    results = payload.get("candidate_results")
    if not isinstance(results, list) or len(results) != len(CANDIDATE_IDS):
        raise DelayedGoalV2SpecError("controller lock candidate grid changed")
    gate = spec["executed_choice"]["development_selection"]
    for result in results:
        if result.get("id") not in CANDIDATE_IDS:
            raise DelayedGoalV2SpecError("unknown candidate in controller lock")
        if result.get("protocol") != controller_protocol(spec, result["id"]):
            raise DelayedGoalV2SpecError("candidate protocol receipt changed")
        task_results = result.get("tasks")
        if not isinstance(task_results, dict) or tuple(task_results) != TASKS:
            raise DelayedGoalV2SpecError("controller task health grid changed")
        recomputed = True
        for task_result in task_results.values():
            success = task_result.get("oracle_success")
            class_success = task_result.get("per_class_oracle_success")
            if not isinstance(success, (int, float)) \
                    or not isinstance(class_success, list) \
                    or len(class_success) != 4 \
                    or not all(isinstance(value, (int, float))
                               for value in class_success):
                raise DelayedGoalV2SpecError("invalid controller health metric")
            recomputed &= success >= gate["per_task_oracle_success_min"]
            recomputed &= min(class_success) >= gate[
                "per_class_oracle_success_min"]
        if result.get("development_gate_pass") is not bool(recomputed):
            raise DelayedGoalV2SpecError("controller health gate receipt changed")
    selected = select_candidate_id(spec, results)
    if selected is None or payload.get("status") != "controller_locked" \
            or payload.get("selected_candidate_id") != selected \
            or payload.get("selected_protocol") != controller_protocol(
                spec, selected):
        raise DelayedGoalV2SpecError("controller selection is not locked/healthy")
    subsets = payload.get("development_subsets")
    if not isinstance(subsets, dict) or tuple(subsets) != TASKS:
        raise DelayedGoalV2SpecError("controller development subsets changed")
    for task, record in subsets.items():
        if record.get("episodes") != 240 \
                or record.get("index_sha256") != spec[
                    "development"]["index_sha256"][task] \
                or record.get("class_counts") != [60, 60, 60, 60]:
            raise DelayedGoalV2SpecError("controller subset receipt changed")


def load_controller_lock(spec: Mapping[str, Any], *, root: Path = ROOT
                         ) -> tuple[dict[str, Any], dict[str, str]]:
    path, sidecar = controller_lock_paths(spec, root=root)
    if not path.is_file() or not sidecar.is_file():
        raise DelayedGoalV2SpecError(
            "V2 validation is forbidden before the controller lock exists")
    actual = sha256_file(path)
    if sidecar.read_text().strip().split() != [actual, path.name]:
        raise DelayedGoalV2SpecError("controller lock hash mismatch")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise DelayedGoalV2SpecError(
            f"cannot parse controller lock: {error}") from error
    validate_controller_lock_payload(payload, spec)
    return payload, {"path": str(path.relative_to(root.resolve())),
                     "sha256": actual}


def seal_json(path: Path, sidecar: Path, payload: Mapping[str, Any]) -> None:
    """Write one immutable JSON+hash pair without an overwrite path."""

    if path.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite controller lock {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary_sidecar = sidecar.with_name(
        f".{sidecar.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text)
        digest = sha256_file(temporary)
        temporary_sidecar.write_text(f"{digest} {path.name}\n")
        os.replace(temporary, path)
        os.replace(temporary_sidecar, sidecar)
        path.chmod(0o444)
        sidecar.chmod(0o444)
    except BaseException:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        raise
