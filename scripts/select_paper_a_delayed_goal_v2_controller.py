#!/usr/bin/env python3
"""Select and seal the V2 executed-choice controller on training data only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.paper_a_delayed_goal_use import pd_action, wrapped_rms
from scripts.paper_a_delayed_goal_v2_spec import (
    CANDIDATE_IDS,
    DEFAULT_SPEC,
    TASKS,
    controller_lock_paths,
    controller_protocol,
    development_indices,
    load_locked_spec,
    load_v1_provenance,
    resolve_path,
    seal_json,
    select_candidate_id,
    sha256_file,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _load_development(path: Path, task: str,
                      spec: Mapping[str, Any]) -> dict[str, np.ndarray]:
    expected = spec["parent"]["train_caches"][task]
    if sha256_file(path) != expected["sha256"]:
        raise ValueError(f"parent training bank changed: {path}")
    with np.load(path) as source:
        labels = np.asarray(source["xi"], dtype=np.int64)
        states = np.asarray(source["endo_state"], dtype=np.float64)
    indices = development_indices(labels, task, spec)
    decision = int(spec["executed_choice"]["fixed"]["decision_index"])
    if states.shape != (1200, 64, 4):
        raise ValueError(f"unexpected training physics states for {task}")
    return {
        "labels": labels[indices],
        "states": states[indices, decision],
        "indices": indices,
    }


def execute_oracle_choices(states: np.ndarray, labels: np.ndarray,
                           protocol: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Execute only the true-label choice; no representation model is loaded."""

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    from dm_control import suite

    states = np.asarray(states, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    goals = np.asarray(protocol["joint_goals"], dtype=np.float64)
    if states.shape != (len(labels), 4) or goals.shape != (4, 2):
        raise ValueError("unexpected development state/goal shape")
    environment = suite.load("reacher", "easy", task_kwargs={"random": 0})
    action_spec = environment.action_spec()
    low = np.asarray(action_spec.minimum, dtype=np.float64)
    high = np.asarray(action_spec.maximum, dtype=np.float64)
    distance = np.empty(len(labels), dtype=np.float64)
    for episode, (state, label) in enumerate(zip(states, labels, strict=True)):
        goal = goals[label]
        environment.reset()
        with environment.physics.reset_context():
            environment.physics.set_state(state)
        for _ in range(int(protocol["executed_horizon"])):
            position = np.asarray(environment.physics.data.qpos,
                                  dtype=np.float64)
            velocity = np.asarray(environment.physics.data.qvel,
                                  dtype=np.float64)
            timestep = environment.step(pd_action(
                position, velocity, goal,
                float(protocol["proportional_gain"]),
                float(protocol["derivative_gain"]), low, high))
            if timestep.last():
                break
        distance[episode] = wrapped_rms(environment.physics.data.qpos, goal)
    scale = float(protocol["return_scale_radians"])
    tolerance = float(protocol["success_tolerance_radians"])
    return {
        "distance": distance,
        "return": np.exp(-0.5 * np.square(distance / scale)),
        "success": distance <= tolerance,
    }


def summarize_oracle(execution: Mapping[str, np.ndarray], labels: np.ndarray
                     ) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    success = np.asarray(execution["success"], dtype=bool)
    returns = np.asarray(execution["return"], dtype=np.float64)
    distance = np.asarray(execution["distance"], dtype=np.float64)
    if any(value.shape != labels.shape
           for value in (success, returns, distance)):
        raise ValueError("oracle execution arrays have inconsistent shapes")
    return {
        "episodes": int(len(labels)),
        "oracle_success": float(success.mean()),
        "oracle_return": float(returns.mean()),
        "mean_distance": float(distance.mean()),
        "per_class_oracle_success": [
            float(success[labels == category].mean()) for category in range(4)],
    }


def candidate_receipt(spec: Mapping[str, Any], candidate_id: str,
                      task_data: Mapping[str, Mapping[str, np.ndarray]]
                      ) -> dict[str, Any]:
    protocol = controller_protocol(spec, candidate_id)
    tasks: dict[str, Any] = {}
    gate = spec["executed_choice"]["development_selection"]
    passed = True
    for task in TASKS:
        data = task_data[task]
        result = summarize_oracle(execute_oracle_choices(
            data["states"], data["labels"], protocol), data["labels"])
        tasks[task] = result
        passed &= result["oracle_success"] >= gate[
            "per_task_oracle_success_min"]
        passed &= min(result["per_class_oracle_success"]) >= gate[
            "per_class_oracle_success_min"]
    return {
        "id": candidate_id,
        "protocol": protocol,
        "tasks": tasks,
        "equal_task_oracle_success": float(np.mean([
            result["oracle_success"] for result in tasks.values()])),
        "equal_task_oracle_return": float(np.mean([
            result["oracle_return"] for result in tasks.values()])),
        "development_gate_pass": bool(passed),
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit(
            "refusing controller development without explicit --execute")
    spec = load_locked_spec(args.spec, verify_artifacts=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_GL"] = spec["execution"][
        "controller_selection_gl_backend"]
    # Authenticate the sealed amendment inputs without opening a parent
    # validation cache.  Historical V1 failure files are provenance only.
    for record in (spec["v1"]["spec"], spec["v1"]["provenance_manifest"]):
        path = resolve_path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"sealed V1 input changed: {record['path']}")
    load_v1_provenance(spec, verify_artifacts=True)
    lock_path, lock_sidecar = controller_lock_paths(spec)
    evaluations = resolve_path(spec["output"]["evaluations"])
    summary = resolve_path(spec["output"]["summary"])
    if lock_path.exists() or lock_sidecar.exists():
        raise FileExistsError("V2 controller is already locked")
    if evaluations.exists() or summary.exists():
        raise RuntimeError(
            "validation artifacts exist before V2 controller selection")

    task_data = {
        task: _load_development(
            resolve_path(spec["parent"]["train_caches"][task]["path"]),
            task, spec)
        for task in TASKS
    }
    results = [candidate_receipt(spec, candidate_id, task_data)
               for candidate_id in CANDIDATE_IDS]
    selected = select_candidate_id(spec, results)
    subsets = {
        task: {
            "episodes": int(len(data["indices"])),
            "class_counts": [int(np.sum(data["labels"] == category))
                             for category in range(4)],
            "index_sha256": spec["development"]["index_sha256"][task],
            "source_cache": spec["parent"]["train_caches"][task],
        }
        for task, data in task_data.items()
    }
    if selected is None:
        failure_path = lock_path.with_name("controller-selection-failed.json")
        failure_sidecar = failure_path.with_suffix(".sha256")
        seal_json(failure_path, failure_sidecar, {
            "schema_version": 1, "study": spec["study"],
            "spec": spec["_spec_record"],
            "status": "no_development_healthy_controller",
            "development_source": "parent training bank only",
            "validation_data_accessed": False,
            "validation_artifacts_absent_at_lock": True,
            "development_subsets": subsets,
            "candidate_results": results,
        })
        raise SystemExit(
            "no controller passed development health; validation forbidden")
    payload = {
        "schema_version": 1,
        "study": spec["study"],
        "spec": spec["_spec_record"],
        "status": "controller_locked",
        "development_source": "parent training bank only",
        "validation_data_accessed": False,
        "validation_artifacts_absent_at_lock": True,
        "development_subsets": subsets,
        "candidate_results": results,
        "selected_candidate_id": selected,
        "selected_protocol": controller_protocol(spec, selected),
        "v1_failure_provenance": spec["v1"]["provenance_manifest"],
        "v1_repairs_reused": True,
        "repair_retraining_performed": False,
    }
    seal_json(lock_path, lock_sidecar, payload)
    print(f"[delayed-goal-v2] sealed {selected} at {lock_path}", flush=True)


if __name__ == "__main__":
    main()
