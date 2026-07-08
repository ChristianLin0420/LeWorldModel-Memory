#!/usr/bin/env python3
"""Run the pinned native PushT controller gate or held-out physical deck."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "lewm/official_tasks/pusht_downstream.py"


def _load_primitives():
    spec = importlib.util.spec_from_file_location("pusht_downstream_primitives", MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.execute:
        raise RuntimeError("physical simulation writes formal artifacts; pass --execute")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    warnings.filterwarnings("ignore", message=".*Casting input x to numpy array.*")
    pd = _load_primitives()

    protocol = ROOT / "configs/paper_a_pusht_downstream_use_v1.yaml"
    receipt_path = ROOT / "outputs/pusht_downstream_use_v1/protocol_receipt.json"
    gates_path = ROOT / "outputs/pusht_downstream_use_v1/gates.json"
    receipt = json.loads(receipt_path.read_text())
    gates = json.loads(gates_path.read_text())
    expected_sha = receipt["protocol"]["sha256"]
    if pd.sha256_file(protocol) != expected_sha:
        raise RuntimeError("locked downstream protocol changed after preparation")
    if gates["protocol_receipt"]["sha256"] != pd.sha256_file(receipt_path):
        raise RuntimeError("protocol receipt changed after preparation")

    if args.split == "development":
        cache = np.load(ROOT / "outputs/official_pusht_memory/cache/base/train.npz")
        source_rows = np.asarray(
            receipt["split"]["development_rows"], dtype=np.int64)
        states = cache["state"][source_rows, 19].astype(np.float64)
        episode_ids = cache["episode_index"][source_rows].astype(np.int64)
    else:
        if gates.get("physical_development_pending", True):
            raise RuntimeError("held-out physical deck is blocked before development gate")
        cache = np.load(ROOT / "outputs/official_pusht_memory/cache/base/validation.npz")
        source_rows = np.arange(len(cache["state"]), dtype=np.int64)
        states = cache["state"][:, 19].astype(np.float64)
        episode_ids = cache["episode_index"].astype(np.int64)

    low, high = receipt["simulator_contract"]["eligibility_interval"]
    keep = pd.interior_state_mask(states, float(low), float(high))
    states = states[keep]
    episode_ids = episode_ids[keep]
    source_rows = source_rows[keep]
    if len(states) < 200:
        raise RuntimeError("label-blind physical eligibility retained fewer than 200 episodes")

    checkout = ROOT / receipt["simulator_contract"]["checkout"]
    PushT, commit = pd.load_pinned_pusht(
        checkout, receipt["simulator_contract"]["commit"])
    env = pd.make_native_env(PushT)
    controller_spec = receipt["simulator_contract"]["controller"]
    controller = pd.NativePushTController(
        env,
        orbit_radius=float(controller_spec["orbit_radius_pixels"]),
        orbit_points=int(controller_spec["orbit_points"]),
        waypoint_steps=int(controller_spec["steps_per_waypoint"]),
        push_steps=int(controller_spec["push_steps"]),
        push_distance=float(controller_spec["push_distance_pixels"]),
    )
    # Apply the already-locked reset contract as a fail-closed, label-blind
    # state-validity check.  A cached contact state that the pinned simulator
    # moves by more than one pixel cannot define a faithful physical replay.
    reset_contract_exclusions = []
    reset_valid = np.ones(len(states), dtype=bool)
    for index, state in enumerate(states):
        observation, _ = env.reset(
            seed=0, options={"variation": [], "state": state,
                             "goal_state": state})
        reset = np.asarray(observation["state"], dtype=np.float64)
        agent_residual = float(np.max(np.abs(
            reset[:2] - (state[:2] + state[5:] * float(env.dt)))))
        block_adjustment = float(np.max(np.abs(reset[2:5] - state[2:5])))
        velocity_residual = float(np.max(np.abs(reset[5:] - state[5:])))
        if (agent_residual > 2e-4 or block_adjustment > 1.0
                or velocity_residual > 2e-5):
            reset_valid[index] = False
            reset_contract_exclusions.append({
                "source_row": int(source_rows[index]),
                "episode_id": int(episode_ids[index]),
                "agent_equation_residual": agent_residual,
                "block_contact_adjustment": block_adjustment,
                "velocity_residual": velocity_residual,
            })
    states = states[reset_valid]
    episode_ids = episode_ids[reset_valid]
    source_rows = source_rows[reset_valid]
    if len(states) < 200:
        raise RuntimeError("reset-contract filtering retained fewer than 200 episodes")
    pos_tol = float(receipt["simulator_contract"]["position_tolerance_pixels"])
    angle_tol = float(receipt["simulator_contract"]["angle_tolerance_radians"])
    reset_agent_residual = []
    reset_block_adjustment = []
    reset_velocity_residual = []
    health: dict[str, dict] = {}

    tasks = (("transient-visual-token-recall", 4),
             ("multi-item-visual-binding-recall", 6))
    for task, classes in tasks:
        directions = pd.goal_directions(classes)
        goals = np.empty((len(states), classes, 3), dtype=np.float64)
        finals = np.empty_like(goals)
        for row, state in enumerate(states):
            reference_state = state.copy()
            reference_state[5:] = 0.0
            for selected, direction in enumerate(directions):
                reference = controller.execute(reference_state, direction)
                execution = controller.execute(state, direction)
                goals[row, selected] = reference["final_block_pose"]
                finals[row, selected] = execution["final_block_pose"]
                if selected == 0:
                    reset = execution["reset_state"]
                    reset_agent_residual.append(float(np.max(np.abs(
                        reset[:2] - (state[:2] + state[5:] * float(env.dt))))))
                    reset_block_adjustment.append(float(np.max(np.abs(
                        reset[2:5] - state[2:5]))))
                    reset_velocity_residual.append(float(np.max(np.abs(
                        reset[5:] - state[5:]))))
        success = pd.pose_success(
            finals[:, :, None, :], goals[:, None, :, :], pos_tol, angle_tol)
        position_cost = np.linalg.norm(
            finals[:, :, None, :2] - goals[:, None, :, :2], axis=-1)
        angle_cost = pd.angular_error(
            finals[:, :, None, 2], goals[:, None, :, 2])
        cost = position_cost + 20.0 * angle_cost
        oracle_cost = np.stack(
            [cost[:, index, index] for index in range(classes)], axis=1)
        regret = cost - oracle_cost[:, None, :]
        oracle = np.stack(
            [success[:, index, index] for index in range(classes)], axis=1)
        per_class = oracle.mean(axis=0)
        physical_spec = receipt["simulator_contract"]["physical_gate"]
        passed = (float(oracle.mean())
                  >= float(physical_spec["development_success_minimum"])
                  and float(per_class.min())
                  >= float(physical_spec["development_per_class_minimum"]))
        output = (ROOT / "outputs/pusht_downstream_use_v1/simulator"
                  / args.split / f"{task}.npz")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        np.savez_compressed(
            output,
            source_rows=source_rows,
            episode_ids=episode_ids,
            cached_states=states.astype(np.float32),
            goal_pose=goals.astype(np.float32),
            final_pose=finals.astype(np.float32),
            success_matrix=success,
            cost_matrix=cost.astype(np.float32),
            regret_matrix=regret.astype(np.float32),
        )
        health[task] = {
            "classes": classes,
            "eligible_episodes": int(len(states)),
            "oracle_success": float(oracle.mean()),
            "oracle_success_per_class": [float(value) for value in per_class],
            "wrong_goal_success": float(success[:, ~np.eye(
                classes, dtype=bool)].mean()),
            "passed": bool(passed),
            "artifact": {
                "path": str(output.relative_to(ROOT)),
                "sha256": pd.sha256_file(output),
            },
        }

    state_receipt = {
        "schema": "paper_a_pusht_state_reset_receipt_v1",
        "upstream_commit": commit,
        "split": args.split,
        "eligible_episodes": int(len(states)),
        "reset_contract_exclusions": reset_contract_exclusions,
        "state_order": ["agent_x", "agent_y", "block_x", "block_y",
                        "block_theta", "agent_vx", "agent_vy"],
        "upstream_dt": float(env.dt),
        "max_agent_equation_residual": float(max(reset_agent_residual)),
        "max_block_contact_adjustment": float(max(reset_block_adjustment)),
        "max_velocity_residual": float(max(reset_velocity_residual)),
        "health": health,
    }
    if args.split == "development":
        output_receipt = ROOT / "outputs/pusht_downstream_use_v1/state_reset_receipt.json"
        if output_receipt.exists():
            raise FileExistsError(f"refusing to overwrite {output_receipt}")
        output_receipt.write_text(pd.stable_json(state_receipt))
        for task, record in health.items():
            gates["tasks"][task]["physical_oracle"] = record
            gates["tasks"][task]["physical_oracle_pass"] = record["passed"]
        gates["physical_development_pending"] = False
        gates["state_reset_receipt"] = {
            "path": str(output_receipt.relative_to(ROOT)),
            "sha256": pd.sha256_file(output_receipt),
        }
        gates_path.write_text(pd.stable_json(gates))
    else:
        output_receipt = (ROOT / "outputs/pusht_downstream_use_v1/simulator"
                          / "test_receipt.json")
        if output_receipt.exists():
            raise FileExistsError(f"refusing to overwrite {output_receipt}")
        output_receipt.write_text(pd.stable_json(state_receipt))
    env.close()
    print(pd.stable_json(state_receipt))


if __name__ == "__main__":
    main()
