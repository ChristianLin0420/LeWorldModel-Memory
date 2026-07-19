#!/usr/bin/env python3
"""Build a native DINO-WM PushT execution deck for checkpointed downstream use."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PRIMITIVES = ROOT / "lewm/official_tasks/pusht_downstream.py"
DEFAULT_OUTPUT = ROOT / "outputs/pusht_checkpointed_downstream_use_v1/dinowm_simulator/test"
DEFAULT_DINO_CACHE = ROOT / "outputs/dinowm_wave2_spatial_carrier_v1_1/cache/metadata.npz"
DEFAULT_DINO_DATA = ROOT / "outputs/dinowm_native_pusht_audit_v2/data/pusht_noise/train"
DEFAULT_VENDOR = ROOT / "outputs/pusht_downstream_use_v1/vendor/stable-worldmodel"


def load_primitives():
    spec = importlib.util.spec_from_file_location("pusht_downstream_primitives", PRIMITIVES)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {PRIMITIVES}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--task", default="multi-item-visual-binding-recall",
                        choices=["multi-item-visual-binding-recall",
                                 "transient-visual-token-recall"])
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--dino-cache", default=str(DEFAULT_DINO_CACHE))
    parser.add_argument("--dino-data", default=str(DEFAULT_DINO_DATA))
    parser.add_argument("--vendor", default=str(DEFAULT_VENDOR))
    parser.add_argument("--endpoint-frame", type=int, default=18)
    parser.add_argument("--low", type=float, default=170.0)
    parser.add_argument("--high", type=float, default=342.0)
    parser.add_argument("--position-tolerance", type=float, default=8.0)
    parser.add_argument("--angle-tolerance", type=float, default=0.12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_validation_states(metadata_path: Path, data_root: Path,
                           endpoint_frame: int) -> dict[str, np.ndarray]:
    metadata = np.load(metadata_path, allow_pickle=False)
    split = np.asarray(metadata["split"], dtype=np.uint8)
    validation_indices = np.flatnonzero(split == 1)
    episode_index = np.asarray(metadata["episode_index"], dtype=np.int64)[validation_indices]
    local_start = np.asarray(metadata["local_start"], dtype=np.int64)[validation_indices]
    states = torch.load(data_root / "states.pth", map_location="cpu", weights_only=False).numpy()
    velocities = torch.load(data_root / "velocities.pth", map_location="cpu", weights_only=False).numpy()
    with (data_root / "seq_lengths.pkl").open("rb") as stream:
        lengths = np.asarray(pickle.load(stream), dtype=np.int64)
    source_time = local_start + int(endpoint_frame)
    if np.any(source_time >= lengths[episode_index]):
        raise RuntimeError("DINO validation endpoint exceeds native episode length")
    state5 = states[episode_index, source_time].astype(np.float64)
    velocity2 = velocities[episode_index, source_time].astype(np.float64)
    state7 = np.concatenate((state5, velocity2), axis=1)
    return {
        "source_rows": np.arange(len(validation_indices), dtype=np.int64),
        "metadata_rows": validation_indices.astype(np.int64),
        "episode_ids": episode_index.astype(np.int64),
        "local_start": local_start.astype(np.int64),
        "source_time": source_time.astype(np.int64),
        "states": state7.astype(np.float64),
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    warnings.filterwarnings("ignore", message=".*Casting input x to numpy array.*")
    output_dir = Path(args.output)
    output_path = output_dir / f"{args.task}.npz"
    receipt_path = output_dir / f"{args.task}.receipt.json"
    if (output_path.exists() or receipt_path.exists()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path} / {receipt_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pd = load_primitives()
    source = load_validation_states(
        Path(args.dino_cache), Path(args.dino_data), int(args.endpoint_frame))
    keep = pd.interior_state_mask(source["states"], float(args.low), float(args.high))
    states = source["states"][keep]
    source_rows = source["source_rows"][keep]
    metadata_rows = source["metadata_rows"][keep]
    episode_ids = source["episode_ids"][keep]
    local_start = source["local_start"][keep]
    source_time = source["source_time"][keep]
    if len(states) < 200:
        raise RuntimeError(
            f"DINO native physical eligibility retained only {len(states)} episodes")

    PushT, commit = pd.load_pinned_pusht(Path(args.vendor))
    env = pd.make_native_env(PushT)
    controller = pd.NativePushTController(env)

    reset_valid = np.ones(len(states), dtype=bool)
    reset_exclusions = []
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
            reset_exclusions.append({
                "source_row": int(source_rows[index]),
                "episode_id": int(episode_ids[index]),
                "agent_equation_residual": agent_residual,
                "block_contact_adjustment": block_adjustment,
                "velocity_residual": velocity_residual,
            })
    states = states[reset_valid]
    source_rows = source_rows[reset_valid]
    metadata_rows = metadata_rows[reset_valid]
    episode_ids = episode_ids[reset_valid]
    local_start = local_start[reset_valid]
    source_time = source_time[reset_valid]
    if len(states) < 200:
        raise RuntimeError(
            f"DINO reset-contract filtering retained only {len(states)} episodes")

    directions = pd.goal_directions(int(args.classes))
    goals = np.empty((len(states), int(args.classes), 3), dtype=np.float64)
    finals = np.empty_like(goals)
    for row, state in enumerate(states):
        reference_state = state.copy()
        reference_state[5:] = 0.0
        if row and row % 25 == 0:
            print(f"[dinowm-exec-deck] rows={row}/{len(states)}", flush=True)
        for selected, direction in enumerate(directions):
            reference = controller.execute(reference_state, direction)
            execution = controller.execute(state, direction)
            goals[row, selected] = reference["final_block_pose"]
            finals[row, selected] = execution["final_block_pose"]
    env.close()

    success = pd.pose_success(
        finals[:, :, None, :], goals[:, None, :, :],
        float(args.position_tolerance), float(args.angle_tolerance))
    position_cost = np.linalg.norm(
        finals[:, :, None, :2] - goals[:, None, :, :2], axis=-1)
    angle_cost = pd.angular_error(finals[:, :, None, 2], goals[:, None, :, 2])
    cost = position_cost + 20.0 * angle_cost
    oracle_cost = np.stack(
        [cost[:, index, index] for index in range(int(args.classes))], axis=1)
    regret = cost - oracle_cost[:, None, :]
    oracle = np.stack(
        [success[:, index, index] for index in range(int(args.classes))], axis=1)
    np.savez_compressed(
        output_path,
        source_rows=source_rows.astype(np.int64),
        metadata_rows=metadata_rows.astype(np.int64),
        episode_ids=episode_ids.astype(np.int64),
        local_start=local_start.astype(np.int64),
        source_time=source_time.astype(np.int64),
        cached_states=states.astype(np.float32),
        goal_pose=goals.astype(np.float32),
        final_pose=finals.astype(np.float32),
        success_matrix=success,
        cost_matrix=cost.astype(np.float32),
        regret_matrix=regret.astype(np.float32),
    )
    receipt = {
        "schema": "dinowm_pusht_native_execution_deck_v1",
        "task": args.task,
        "classes": int(args.classes),
        "endpoint_frame": int(args.endpoint_frame),
        "eligible_episodes": int(len(states)),
        "interior_interval": [float(args.low), float(args.high)],
        "reset_contract_exclusions": reset_exclusions,
        "oracle_success": float(oracle.mean()),
        "oracle_success_per_class": [float(value) for value in oracle.mean(axis=0)],
        "wrong_goal_success": float(success[:, ~np.eye(int(args.classes), dtype=bool)].mean()),
        "stable_worldmodel_commit": commit,
        "artifact": {
            "path": str(output_path.resolve().relative_to(ROOT)),
            "sha256": pd.sha256_file(output_path),
        },
    }
    receipt_path.write_text(stable_json(receipt))
    print(stable_json(receipt))


if __name__ == "__main__":
    main()
