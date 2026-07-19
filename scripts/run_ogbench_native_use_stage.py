#!/usr/bin/env python3
"""Native-success use stage for OGBench feature-host predictions.

This stage converts memory-selected cue labels into environment execution
success.  It is intentionally limited to OGBench environments with a fixed,
auditable controller available in the local package:

* PointMaze: BFS subgoal controller over the native point-mass action space.
* Cube-single: OGBench CubeMarkovOracle, with the controller target overridden
  by the selected label while the environment success target remains the true
  cue label.

The stage does not claim native world-model planning.  It reports whether the
memory readout can select a target that a fixed controller converts into native
``info["success"]`` / reward.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_ROOT = ROOT / "outputs" / "ogbench_feature_host_stage_v1"
DEFAULT_OUTPUT = ROOT / "outputs" / "ogbench_native_use_stage_v1"
ENVS = ("pointmaze-large-navigate-v0", "cube-single-play-v0")
AGES = (4, 8, 15)
SEEDS = (0, 1, 2, 3, 4)
CLASSES = 4


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def env_key(env_name: str) -> str:
    return env_name.replace("/", "_")


def metric(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "count": int(arr.size),
        "values": arr.astype(float).tolist(),
    }


def pointmaze_selected_success(env_name: str, true_label: int, selected_label: int,
                               row_index: int, horizon: int) -> dict[str, Any]:
    import ogbench  # noqa: WPS433

    def teleport_aware_subgoal(unwrapped: Any, xy: np.ndarray,
                               goal: np.ndarray) -> np.ndarray | None:
        teleport = getattr(unwrapped, "_teleport_info", None)
        if not teleport:
            return None
        start_ij = unwrapped.xy_to_ij(xy)
        goal_ij = unwrapped.xy_to_ij(goal)
        maze_map = unwrapped.maze_map.copy()
        for ij in teleport.get("teleport_in_ijs", []):
            if tuple(ij) not in {tuple(start_ij), tuple(goal_ij)}:
                maze_map[ij[0], ij[1]] = 1

        bfs_map = np.full_like(maze_map, -1)
        if maze_map[goal_ij[0], goal_ij[1]] != 0 or maze_map[start_ij[0], start_ij[1]] != 0:
            return None
        bfs_map[goal_ij[0], goal_ij[1]] = 0
        queue = [goal_ij]
        while queue:
            i, j = queue.pop(0)
            for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                ni, nj = i + di, j + dj
                if (
                    0 <= ni < maze_map.shape[0]
                    and 0 <= nj < maze_map.shape[1]
                    and maze_map[ni, nj] == 0
                    and bfs_map[ni, nj] == -1
                ):
                    bfs_map[ni, nj] = bfs_map[i, j] + 1
                    queue.append((ni, nj))
        if bfs_map[start_ij[0], start_ij[1]] < 0:
            return None
        subgoal_ij = start_ij
        for di, dj in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            ni, nj = start_ij[0] + di, start_ij[1] + dj
            if (
                0 <= ni < maze_map.shape[0]
                and 0 <= nj < maze_map.shape[1]
                and maze_map[ni, nj] == 0
                and 0 <= bfs_map[ni, nj] < bfs_map[subgoal_ij[0], subgoal_ij[1]]
            ):
                subgoal_ij = (ni, nj)
        return np.asarray(unwrapped.ij_to_xy(subgoal_ij), dtype=np.float64)

    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    try:
        ob, info = env.reset(
            seed=880_000 + int(row_index),
            options={"task_id": int(true_label) + 1},
        )
        del ob, info
        unwrapped = env.unwrapped
        if int(selected_label) == int(true_label):
            selected_goal = np.asarray(unwrapped.cur_goal_xy, dtype=np.float64)
        else:
            selected_goal = np.asarray(
                unwrapped.task_infos[int(selected_label)]["goal_xy"],
                dtype=np.float64,
            )
        total_reward = 0.0
        success = False
        steps = 0
        for steps in range(1, int(horizon) + 1):
            xy = np.asarray(unwrapped.get_xy(), dtype=np.float64)
            subgoal = teleport_aware_subgoal(unwrapped, xy, selected_goal)
            if subgoal is None:
                subgoal, _ = unwrapped.get_oracle_subgoal(xy, selected_goal)
            target = selected_goal if np.linalg.norm(subgoal - xy) < 0.7 else subgoal
            diff = np.asarray(target, dtype=np.float64) - xy
            teleport = getattr(unwrapped, "_teleport_info", None)
            if teleport:
                for inbound in teleport.get("teleport_in_xys", []):
                    away = xy - np.asarray(inbound, dtype=np.float64)
                    dist = float(np.linalg.norm(away))
                    if 1e-6 < dist < float(teleport.get("teleport_radius", 1.0)) * 2.4:
                        diff = diff + away / dist * (2.4 - dist)
            norm = np.linalg.norm(diff)
            action = diff / (norm + 1e-6) * min(1.0, norm / 0.2)
            action = np.clip(action, -1.0, 1.0)
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            success = success or bool(info.get("success", False))
            if terminated or truncated:
                break
        final_xy = np.asarray(unwrapped.get_xy(), dtype=np.float64)
        true_goal = np.asarray(unwrapped.cur_goal_xy, dtype=np.float64)
        return {
            "success": bool(success),
            "reward_sum": float(total_reward),
            "steps": int(steps),
            "final_distance": float(np.linalg.norm(final_xy - true_goal)),
        }
    finally:
        env.close()


def cube_augmented_info(info: dict[str, Any], env: Any,
                        selected_label: int) -> dict[str, Any]:
    out = dict(info)
    goal = np.asarray(
        env.unwrapped.task_infos[int(selected_label)]["goal_xyzs"][0],
        dtype=np.float64,
    )
    out["privileged/target_block"] = 0
    out["privileged/target_block_pos"] = goal.copy()
    out["privileged/target_block_yaw"] = np.asarray([0.0], dtype=np.float64)
    return out


def cube_selected_success(env_name: str, true_label: int, selected_label: int,
                          row_index: int, horizon: int) -> dict[str, Any]:
    import ogbench  # noqa: WPS433
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle  # noqa: WPS433

    if env_name != "cube-single-play-v0":
        raise NotImplementedError(
            "native-use cube oracle is currently audited only for cube-single-play-v0")
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    try:
        ob, info = env.reset(
            seed=990_000 + int(row_index),
            options={"task_id": int(true_label) + 1},
        )
        oracle = CubeMarkovOracle(env=env)
        oracle.reset(ob, cube_augmented_info(info, env, selected_label))
        total_reward = 0.0
        success = False
        steps = 0
        for steps in range(1, int(horizon) + 1):
            action = oracle.select_action(
                ob, cube_augmented_info(info, env, selected_label))
            ob, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            success = success or bool(info.get("success", False))
            if terminated or truncated or oracle.done:
                break
        final_pos = np.asarray(
            env.unwrapped._data.joint("object_joint_0").qpos[:3],
            dtype=np.float64,
        )
        true_goal = np.asarray(
            env.unwrapped.task_infos[int(true_label)]["goal_xyzs"][0],
            dtype=np.float64,
        )
        return {
            "success": bool(success),
            "reward_sum": float(total_reward),
            "steps": int(steps),
            "final_distance": float(np.linalg.norm(final_pos - true_goal)),
        }
    finally:
        env.close()


def feature_path(feature_root: Path, env_name: str, age: int, seed: int) -> Path:
    return feature_root / env_key(env_name) / f"age_{age}" / f"s{seed}" / "features.npz"


def load_reference_labels(feature_root: Path, env_name: str) -> np.ndarray:
    for age in AGES:
        for seed in SEEDS:
            path = feature_path(feature_root, env_name, age, seed)
            if path.is_file():
                with np.load(path) as data:
                    return np.asarray(data["val_y"], dtype=np.int64)
    raise FileNotFoundError(f"no feature predictions found for {env_name}")


def build_success_cube(args: argparse.Namespace, env_name: str) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    labels = load_reference_labels(args.feature_root, env_name)
    rows = min(int(args.max_rows), int(labels.shape[0]))
    labels = labels[:rows]
    cube_success = np.zeros((rows, CLASSES), dtype=np.uint8)
    cube_reward = np.zeros((rows, CLASSES), dtype=np.float32)
    cube_steps = np.zeros((rows, CLASSES), dtype=np.int32)
    cube_distance = np.zeros((rows, CLASSES), dtype=np.float32)
    runner = pointmaze_selected_success if env_name.startswith("pointmaze") else cube_selected_success
    horizon = args.pointmaze_horizon if env_name.startswith("pointmaze") else args.cube_horizon
    for row_index, true_label in enumerate(labels):
        for selected_label in range(CLASSES):
            result = runner(
                env_name, int(true_label), int(selected_label), row_index, horizon)
            cube_success[row_index, selected_label] = int(result["success"])
            cube_reward[row_index, selected_label] = float(result["reward_sum"])
            cube_steps[row_index, selected_label] = int(result["steps"])
            cube_distance[row_index, selected_label] = float(result["final_distance"])
        if (row_index + 1) % 20 == 0:
            print(f"[ogbench-use] {env_name} cube {row_index + 1}/{rows}", flush=True)
    oracle = cube_success[np.arange(rows), labels]
    rng = np.random.default_rng(int(args.random_seed))
    random_choice = rng.integers(0, CLASSES, size=rows, dtype=np.int64)
    random = cube_success[np.arange(rows), random_choice]
    return {
        "labels": labels,
        "success": cube_success,
        "reward": cube_reward,
        "steps": cube_steps,
        "distance": cube_distance,
        "oracle_success": oracle.astype(np.float64),
        "random_choice": random_choice,
        "random_success": random.astype(np.float64),
    }


def evaluate_cell(feature_root: Path, env_name: str, age: int, seed: int,
                  cube: dict[str, np.ndarray]) -> dict[str, Any]:
    path = feature_path(feature_root, env_name, age, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        labels = np.asarray(data["val_y"], dtype=np.int64)[:len(cube["labels"])]
        pred_full = np.asarray(data["pred_full"], dtype=np.int64)[:len(labels)]
        pred_reset = np.asarray(data["pred_reset"], dtype=np.int64)[:len(labels)]
        pred_no_state = np.asarray(data["pred_no_state"], dtype=np.int64)[:len(labels)]
    if not np.array_equal(labels, cube["labels"]):
        raise ValueError(f"label mismatch for {env_name} age={age} seed={seed}")
    rows = np.arange(len(labels), dtype=np.int64)
    success = cube["success"]
    reward = cube["reward"]

    def arm(pred: np.ndarray) -> dict[str, Any]:
        pred = np.asarray(pred, dtype=np.int64)
        executed = success[rows, pred].astype(np.float64)
        rewards = reward[rows, pred].astype(np.float64)
        correct = (pred == labels).astype(np.float64)
        return {
            "goal_accuracy": float(correct.mean()),
            "executed_success": float(executed.mean()),
            "reward_sum_mean": float(rewards.mean()),
        }

    return {
        "env_name": env_name,
        "age": int(age),
        "seed": int(seed),
        "rows": int(len(labels)),
        "arms": {
            "full": arm(pred_full),
            "reset": arm(pred_reset),
            "no_state": arm(pred_no_state),
            "oracle": {
                "goal_accuracy": 1.0,
                "executed_success": float(cube["oracle_success"].mean()),
                "reward_sum_mean": float(
                    reward[rows, labels].astype(np.float64).mean()),
            },
            "random": {
                "goal_accuracy": float(
                    (cube["random_choice"][:len(labels)] == labels).mean()),
                "executed_success": float(cube["random_success"].mean()),
                "reward_sum_mean": float(
                    reward[rows, cube["random_choice"][:len(labels)]]
                    .astype(np.float64).mean()),
            },
        },
    }


def summarize(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault((cell["env_name"], int(cell["age"])), []).append(cell)
    rows = []
    for (env_name, age), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "env_name": env_name,
            "age": int(age),
            "seed_count": int(len(values)),
            "seeds": [int(v["seed"]) for v in values],
        }
        for arm_name in ("full", "reset", "no_state", "oracle", "random"):
            row[arm_name] = {
                "goal_accuracy": metric([
                    v["arms"][arm_name]["goal_accuracy"] for v in values]),
                "executed_success": metric([
                    v["arms"][arm_name]["executed_success"] for v in values]),
                "reward_sum_mean": metric([
                    v["arms"][arm_name]["reward_sum_mean"] for v in values]),
            }
        row["full_vs_no_state_success"] = float(
            row["full"]["executed_success"]["mean"]
            - row["no_state"]["executed_success"]["mean"])
        row["full_vs_random_success"] = float(
            row["full"]["executed_success"]["mean"]
            - row["random"]["executed_success"]["mean"])
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--envs", nargs="*", default=list(ENVS))
    parser.add_argument("--ages", type=int, nargs="*", default=list(AGES))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--max-rows", type=int, default=320)
    parser.add_argument("--pointmaze-horizon", type=int, default=600)
    parser.add_argument("--cube-horizon", type=int, default=220)
    parser.add_argument("--random-seed", type=int, default=4471)
    parser.add_argument("--reuse-cubes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.feature_root.is_absolute():
        args.feature_root = ROOT / args.feature_root
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    args.output.mkdir(parents=True, exist_ok=True)
    cells = []
    cube_receipts = {}
    for env_name in args.envs:
        cube_path = args.output / env_key(env_name) / "execution_cube.npz"
        cube_path.parent.mkdir(parents=True, exist_ok=True)
        if args.reuse_cubes and cube_path.is_file():
            with np.load(cube_path) as data:
                cube = {name: data[name] for name in data.files}
        else:
            cube = build_success_cube(args, env_name)
            np.savez_compressed(cube_path, **cube)
        cube_receipts[env_name] = {
            "path": str(cube_path.relative_to(ROOT)),
            "rows": int(len(cube["labels"])),
            "oracle_success": float(np.asarray(cube["oracle_success"]).mean()),
            "random_success": float(np.asarray(cube["random_success"]).mean()),
        }
        for age in args.ages:
            for seed in args.seeds:
                cells.append(evaluate_cell(
                    args.feature_root, env_name, int(age), int(seed), cube))
    summary = {
        "schema": "ogbench_native_use_stage_v1",
        "status": "completed",
        "claim_boundary": (
            "Memory-selected target execution under fixed OGBench controllers; "
            "reports native env success, not native world-model planning."),
        "feature_root": str(args.feature_root.relative_to(ROOT)),
        "cell_count": int(len(cells)),
        "cube_receipts": cube_receipts,
        "rows": summarize(cells),
    }
    (args.output / "cells.json").write_text(stable_json(cells))
    (args.output / "summary.json").write_text(stable_json(summary))
    print(stable_json(summary), flush=True)


if __name__ == "__main__":
    main()
