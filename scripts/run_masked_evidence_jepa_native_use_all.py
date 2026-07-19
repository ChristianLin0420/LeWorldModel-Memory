#!/usr/bin/env python3
"""Native-use evaluation for ME-JEPA feature roots across tested OGBench envs.

This extends the original native-use script beyond PointMaze/Cube-single.  It
still stays conservative: a row is reported only when a fixed controller/oracle
is available locally.  AntMaze/HumanoidMaze are recorded as unavailable because
OGBench exposes maze subgoals but no valid ant/humanoid low-level controller in
this repository.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASSES = 4
AGES = (4, 8, 15)
SEEDS = (0, 1, 2)
SUPPORTED_PREFIXES = ("pointmaze",)
SUPPORTED_EXACT = {
    "cube-single-play-v0",
    "cube-double-play-v0",
    "cube-triple-play-v0",
    "scene-play-v0",
    "puzzle-3x3-play-v0",
}
HTML_PATH = ROOT / "docs" / "mesm_nvidia_plan.html"
HTML_START = "<!-- NATIVE_USE_ALL_STATUS_START -->"
HTML_END = "<!-- NATIVE_USE_ALL_STATUS_END -->"


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def env_key(env_name: str) -> str:
    return env_name.replace("/", "_")


def feature_path(feature_root: Path, env_name: str, age: int, seed: int) -> Path:
    return feature_root / env_key(env_name) / f"age_{age}" / f"s{seed}" / "features.npz"


def load_cell(feature_root: Path, env_name: str, age: int, seed: int) -> dict[str, np.ndarray]:
    path = feature_path(feature_root, env_name, age, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def metric(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "count": int(arr.size),
        "values": arr.astype(float).tolist(),
    }


def predict_arms(cell: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    readout = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    readout.fit(cell["train_memory"], cell["train_labels"].astype(np.int64))
    return {
        "full": readout.predict(cell["val_full_memory"]).astype(np.int64),
        "reset": readout.predict(cell["val_reset_memory"]).astype(np.int64),
        "recent": readout.predict(cell["val_no_state_memory"]).astype(np.int64),
    }


def is_supported(env_name: str) -> bool:
    return env_name.startswith(SUPPORTED_PREFIXES) or env_name in SUPPORTED_EXACT


def pointmaze_success(env_name: str, true_label: int, selected_label: int,
                      row_index: int, horizon: int) -> dict[str, Any]:
    from scripts.run_ogbench_native_use_stage import pointmaze_selected_success  # noqa: WPS433

    return pointmaze_selected_success(env_name, true_label, selected_label, row_index, horizon)


def cube_augmented_info(info: dict[str, Any], env: Any, target_block: int,
                        target_pos: np.ndarray) -> dict[str, Any]:
    out = dict(info)
    out["privileged/target_task"] = "cube"
    out["privileged/target_block"] = int(target_block)
    out["privileged/target_block_pos"] = np.asarray(target_pos, dtype=np.float64).copy()
    out["privileged/target_block_yaw"] = np.asarray([0.0], dtype=np.float64)
    return out


def cube_goal_xyzs(env: Any, true_label: int, selected_label: int) -> np.ndarray:
    if int(selected_label) == int(true_label):
        return np.asarray([
            env.unwrapped._data.mocap_pos[mocap_id].copy()
            for mocap_id in env.unwrapped._cube_target_mocap_ids
        ], dtype=np.float64)
    return np.asarray(
        env.unwrapped.task_infos[int(selected_label)]["goal_xyzs"],
        dtype=np.float64,
    )


def cube_block_xyzs(env: Any) -> np.ndarray:
    return np.asarray([
        env.unwrapped._data.joint(f"object_joint_{block}").qpos[:3].copy()
        for block in range(env.unwrapped._num_cubes)
    ], dtype=np.float64)


def cube_block_order(env: Any, goals: np.ndarray) -> list[int]:
    current = cube_block_xyzs(env)
    unsolved = [
        block for block in range(goals.shape[0])
        if np.linalg.norm(current[block] - goals[block]) > 0.035
    ]
    if not unsolved:
        return []

    current_is_stack = float(np.ptp(current[:, 2])) > 0.03
    goal_is_stack = float(np.ptp(goals[:, 2])) > 0.03
    if current_is_stack and not goal_is_stack:
        return sorted(unsolved, key=lambda block: float(current[block, 2]), reverse=True)
    return sorted(unsolved, key=lambda block: float(goals[block, 2]))


def cube_success(env_name: str, true_label: int, selected_label: int,
                 row_index: int, horizon: int) -> dict[str, Any]:
    import ogbench  # noqa: WPS433
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle  # noqa: WPS433

    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    try:
        ob, info = env.reset(seed=990_000 + int(row_index), options={"task_id": int(true_label) + 1})
        goals = cube_goal_xyzs(env, true_label, selected_label)
        oracle = CubeMarkovOracle(env=env)
        total_reward = 0.0
        success = False
        steps = 0
        for block in cube_block_order(env, goals):
            for _ in range(2):
                if np.linalg.norm(cube_block_xyzs(env)[block] - goals[block]) <= 0.035:
                    break
                oracle.reset(ob, cube_augmented_info(info, env, block, goals[block]))
                while not oracle.done and steps < int(horizon):
                    action = oracle.select_action(ob, cube_augmented_info(info, env, block, goals[block]))
                    ob, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    success = success or bool(info.get("success", False))
                    steps += 1
                    if terminated or truncated:
                        break
                if success or steps >= int(horizon):
                    break
            if steps >= int(horizon):
                break
        return {"success": bool(success), "reward_sum": float(total_reward), "steps": int(steps)}
    finally:
        env.close()


def button_info(info: dict[str, Any], env: Any, target_button: int,
                target_state: int) -> dict[str, Any]:
    out = dict(info)
    out["privileged/target_task"] = "button"
    out["privileged/target_button"] = int(target_button)
    out["privileged/target_button_state"] = int(target_state)
    out["privileged/target_button_top_pos"] = env.unwrapped._data.site_xpos[
        env.unwrapped._button_site_ids[int(target_button)]
    ].copy()
    return out


def drawer_info(info: dict[str, Any], env: Any, target_pos: float) -> dict[str, Any]:
    out = dict(info)
    out["privileged/target_task"] = "drawer"
    out["privileged/target_drawer_pos"] = np.asarray([float(target_pos)], dtype=np.float64)
    env.unwrapped._model.site("drawer_handle_center_target").pos[1] = float(target_pos)
    out["privileged/target_drawer_handle_pos"] = env.unwrapped._data.site_xpos[
        env.unwrapped._drawer_target_site_id
    ].copy()
    return out


def window_info(info: dict[str, Any], env: Any, target_pos: float) -> dict[str, Any]:
    out = dict(info)
    out["privileged/target_task"] = "window"
    out["privileged/target_window_pos"] = np.asarray([float(target_pos)], dtype=np.float64)
    env.unwrapped._model.site("window_handle_center_target").pos[0] = float(target_pos)
    out["privileged/target_window_handle_pos"] = env.unwrapped._data.site_xpos[
        env.unwrapped._window_target_site_id
    ].copy()
    return out


def run_oracle(env: Any, ob: np.ndarray, info: dict[str, Any], oracle: Any,
               info_builder: Any, horizon: int, steps: int,
               total_reward: float, success: bool) -> tuple[np.ndarray, dict[str, Any], int, float, bool]:
    oracle.reset(ob, info_builder(info))
    while not oracle.done and steps < int(horizon):
        aug = info_builder(info)
        action = oracle.select_action(ob, aug)
        ob, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        success = success or bool(info.get("success", False))
        steps += 1
        if terminated or truncated:
            break
    return ob, info, steps, total_reward, success


def scene_goal(env: Any, true_label: int, selected_label: int) -> dict[str, Any]:
    if int(selected_label) == int(true_label):
        return {
            "block_xyzs": np.asarray([
                env.unwrapped._data.mocap_pos[mocap_id].copy()
                for mocap_id in env.unwrapped._cube_target_mocap_ids
            ], dtype=np.float64),
            "button_states": np.asarray(env.unwrapped._target_button_states, dtype=np.int64).copy(),
            "drawer_pos": float(env.unwrapped._target_drawer_pos),
            "window_pos": float(env.unwrapped._target_window_pos),
        }
    selected = env.unwrapped.task_infos[int(selected_label)]["goal"]
    return {
        "block_xyzs": np.asarray(selected["block_xyzs"], dtype=np.float64),
        "button_states": np.asarray(selected["button_states"], dtype=np.int64),
        "drawer_pos": float(selected["drawer_pos"]),
        "window_pos": float(selected["window_pos"]),
    }


def scene_success(env_name: str, true_label: int, selected_label: int,
                  row_index: int, horizon: int) -> dict[str, Any]:
    import ogbench  # noqa: WPS433
    from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle  # noqa: WPS433
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle  # noqa: WPS433
    from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle  # noqa: WPS433
    from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle  # noqa: WPS433

    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    try:
        ob, info = env.reset(seed=880_000 + int(row_index), options={"task_id": int(true_label) + 1})
        selected = scene_goal(env, true_label, selected_label)
        total_reward = 0.0
        success = False
        steps = 0

        def press_button(button: int, target: int, attempts: int = 2) -> None:
            nonlocal ob, info, steps, total_reward, success
            for _ in range(attempts):
                if steps >= int(horizon) or int(env.unwrapped._cur_button_states[button]) == int(target):
                    return
                oracle = ButtonMarkovOracle(env=env)
                builder = lambda latest, b=button, t=int(target): button_info(latest, env, b, t)
                ob, info, steps, total_reward, success = run_oracle(
                    env, ob, info, oracle, builder, horizon, steps, total_reward, success)

        def move_drawer(target: float, attempts: int = 2) -> None:
            nonlocal ob, info, steps, total_reward, success
            for _ in range(attempts):
                current = float(env.unwrapped._data.joint("drawer_slide").qpos[0])
                if steps >= int(horizon) or abs(current - float(target)) <= 0.04:
                    return
                oracle = DrawerMarkovOracle(env=env)
                builder = lambda latest, t=float(target): drawer_info(latest, env, t)
                ob, info, steps, total_reward, success = run_oracle(
                    env, ob, info, oracle, builder, horizon, steps, total_reward, success)

        def move_window(target: float, attempts: int = 2) -> None:
            nonlocal ob, info, steps, total_reward, success
            for _ in range(attempts):
                current = float(env.unwrapped._data.joint("window_slide").qpos[0])
                if steps >= int(horizon) or abs(current - float(target)) <= 0.04:
                    return
                oracle = WindowMarkovOracle(env=env)
                builder = lambda latest, t=float(target): window_info(latest, env, t)
                ob, info, steps, total_reward, success = run_oracle(
                    env, ob, info, oracle, builder, horizon, steps, total_reward, success)

        def move_cube(block: int, target: np.ndarray, attempts: int = 2) -> None:
            nonlocal ob, info, steps, total_reward, success
            for _ in range(attempts):
                current = env.unwrapped._data.joint(f"object_joint_{block}").qpos[:3].copy()
                if steps >= int(horizon) or np.linalg.norm(current - target) <= 0.04:
                    return
                oracle = CubeMarkovOracle(env=env)
                builder = lambda latest, b=block, g=target.copy(): cube_augmented_info(latest, env, b, g)
                ob, info, steps, total_reward, success = run_oracle(
                    env, ob, info, oracle, builder, horizon, steps, total_reward, success)

        goal_buttons = np.asarray(selected["button_states"], dtype=np.int64)
        drawer_target = float(selected["drawer_pos"])
        window_target = float(selected["window_pos"])
        block_goals = np.asarray(selected["block_xyzs"], dtype=np.float64)
        block_in_drawer = bool(block_goals.size and float(block_goals[0, 1]) < -0.25)
        drawer_needed = (
            abs(float(env.unwrapped._data.joint("drawer_slide").qpos[0]) - drawer_target) > 0.04
            or block_in_drawer
        )
        window_needed = abs(float(env.unwrapped._data.joint("window_slide").qpos[0]) - window_target) > 0.04

        if drawer_needed and int(env.unwrapped._cur_button_states[0]) == 0:
            press_button(0, 1)
        if window_needed and int(env.unwrapped._cur_button_states[1]) == 0:
            press_button(1, 1)

        if block_in_drawer:
            move_drawer(-0.16, attempts=3)
            for block, target in enumerate(block_goals):
                move_cube(block, target, attempts=3)
            move_drawer(drawer_target, attempts=3)
            for block, target in enumerate(block_goals):
                move_cube(block, target, attempts=1)
        else:
            move_window(window_target, attempts=3)
            move_drawer(drawer_target, attempts=3)
            for block, target in enumerate(block_goals):
                move_cube(block, target, attempts=2)

        if drawer_needed and int(env.unwrapped._cur_button_states[0]) == 0:
            press_button(0, 1)
        move_drawer(drawer_target, attempts=3)
        if window_needed and int(env.unwrapped._cur_button_states[1]) == 0:
            press_button(1, 1)
        move_window(window_target, attempts=3)
        for button, target in enumerate(goal_buttons):
            press_button(button, int(target), attempts=3)

        for _ in range(10):
            if success or steps >= int(horizon):
                break
            ob, reward, terminated, truncated, info = env.step(np.zeros(5))
            total_reward += float(reward)
            success = success or bool(info.get("success", False))
            steps += 1
            if terminated or truncated:
                break

        return {"success": bool(success), "reward_sum": float(total_reward), "steps": int(steps)}
    finally:
        env.close()


def solve_lights_out(current: np.ndarray, goal: np.ndarray, rows: int, cols: int) -> list[int]:
    current = np.asarray(current, dtype=np.int64) % 2
    goal = np.asarray(goal, dtype=np.int64) % 2
    delta = (goal - current) % 2
    n = rows * cols
    mat = np.zeros((n, n), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            press = r * cols + c
            for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < rows and 0 <= cc < cols:
                    mat[rr * cols + cc, press] = 1
    aug = np.concatenate([mat, delta.reshape(-1, 1).astype(np.uint8)], axis=1)
    pivot_cols: list[int] = []
    row = 0
    for col in range(n):
        pivot = next((r for r in range(row, n) if aug[r, col]), None)
        if pivot is None:
            continue
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(n):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        pivot_cols.append(col)
        row += 1
    x = np.zeros(n, dtype=np.uint8)
    for r, col in enumerate(pivot_cols):
        x[col] = aug[r, -1]
    return [int(i) for i, value in enumerate(x) if value]


def puzzle_success(env_name: str, true_label: int, selected_label: int,
                   row_index: int, horizon: int) -> dict[str, Any]:
    import ogbench  # noqa: WPS433
    from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle  # noqa: WPS433

    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    try:
        ob, info = env.reset(seed=770_000 + int(row_index), options={"task_id": int(true_label) + 1})
        selected = env.unwrapped.task_infos[int(selected_label)]
        current = env.unwrapped._cur_button_states.copy()
        goal = np.asarray(selected["goal_button_states"], dtype=np.int64)
        presses = solve_lights_out(current, goal, env.unwrapped._num_rows, env.unwrapped._num_cols)
        total_reward = 0.0
        success = False
        steps = 0
        for button in presses:
            target = int((env.unwrapped._cur_button_states[button] + 1) % 2)
            oracle = ButtonMarkovOracle(env=env, gripper_always_closed=True)
            builder = lambda latest, b=button, t=target: button_info(latest, env, b, t)
            ob, info, steps, total_reward, success = run_oracle(
                env, ob, info, oracle, builder, horizon, steps, total_reward, success)
            if steps >= int(horizon):
                break
        return {"success": bool(success), "reward_sum": float(total_reward), "steps": int(steps)}
    finally:
        env.close()


def selected_success(env_name: str, true_label: int, selected_label: int,
                     row_index: int, args: argparse.Namespace) -> dict[str, Any]:
    if env_name.startswith("pointmaze"):
        return pointmaze_success(env_name, true_label, selected_label, row_index, args.pointmaze_horizon)
    if env_name.startswith("cube-"):
        return cube_success(env_name, true_label, selected_label, row_index, args.cube_horizon)
    if env_name == "scene-play-v0":
        return scene_success(env_name, true_label, selected_label, row_index, args.scene_horizon)
    if env_name == "puzzle-3x3-play-v0":
        return puzzle_success(env_name, true_label, selected_label, row_index, args.puzzle_horizon)
    raise NotImplementedError(f"no audited fixed controller for {env_name}")


def build_success_cube(env_name: str, labels: np.ndarray,
                       args: argparse.Namespace) -> dict[str, np.ndarray]:
    if not is_supported(env_name):
        raise NotImplementedError(f"no audited fixed controller for {env_name}")
    os.environ.setdefault("MUJOCO_GL", "egl")
    rows = min(int(args.max_rows), int(len(labels)))
    labels = np.asarray(labels[:rows], dtype=np.int64)
    success = np.zeros((rows, CLASSES), dtype=np.uint8)
    reward = np.zeros((rows, CLASSES), dtype=np.float32)
    steps = np.zeros((rows, CLASSES), dtype=np.int32)
    for row_index, true_label in enumerate(labels):
        for selected_label in range(CLASSES):
            result = selected_success(env_name, int(true_label), selected_label, row_index, args)
            success[row_index, selected_label] = int(result["success"])
            reward[row_index, selected_label] = float(result["reward_sum"])
            steps[row_index, selected_label] = int(result["steps"])
        if (row_index + 1) % max(1, int(args.progress_every)) == 0:
            print(f"[native-use-all] {env_name} cube {row_index + 1}/{rows}", flush=True)
    rng = np.random.default_rng(int(args.random_seed))
    random_choice = rng.integers(0, CLASSES, size=rows, dtype=np.int64)
    return {
        "labels": labels,
        "success": success,
        "reward": reward,
        "steps": steps,
        "random_choice": random_choice,
    }


def evaluate_cell(feature_root: Path, env_name: str, age: int, seed: int,
                  cube: dict[str, np.ndarray]) -> dict[str, Any]:
    cell = load_cell(feature_root, env_name, age, seed)
    labels = cell["val_labels"].astype(np.int64)[:len(cube["labels"])]
    if not np.array_equal(labels, cube["labels"]):
        raise ValueError(f"label mismatch for {env_name} age={age} seed={seed}")
    pred = predict_arms(cell)
    rows = np.arange(len(labels), dtype=np.int64)

    def arm(choice: np.ndarray) -> dict[str, float]:
        choice = np.asarray(choice[:len(labels)], dtype=np.int64)
        executed = cube["success"][rows, choice].astype(np.float64)
        rewards = cube["reward"][rows, choice].astype(np.float64)
        return {
            "goal_accuracy": float((choice == labels).mean()),
            "executed_success": float(executed.mean()),
            "reward_sum_mean": float(rewards.mean()),
        }

    return {
        "env_name": env_name,
        "age": int(age),
        "seed": int(seed),
        "rows": int(len(labels)),
        "arms": {
            "full": arm(pred["full"]),
            "reset": arm(pred["reset"]),
            "recent": arm(pred["recent"]),
            "random": arm(cube["random_choice"]),
            "oracle": arm(labels),
        },
    }


def summarize(cells: list[dict[str, Any]], unavailable: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault((cell["env_name"], int(cell["age"])), []).append(cell)
    rows: list[dict[str, Any]] = []
    for (env_name, age), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "env_name": env_name,
            "age": int(age),
            "status": "completed",
            "seed_count": int(len(values)),
            "seeds": [int(v["seed"]) for v in values],
        }
        for arm_name in ("full", "reset", "recent", "random", "oracle"):
            row[arm_name] = {
                "goal_accuracy": metric([v["arms"][arm_name]["goal_accuracy"] for v in values]),
                "executed_success": metric([v["arms"][arm_name]["executed_success"] for v in values]),
                "reward_sum_mean": metric([v["arms"][arm_name]["reward_sum_mean"] for v in values]),
            }
        row["full_vs_recent_success"] = float(
            row["full"]["executed_success"]["mean"] - row["recent"]["executed_success"]["mean"])
        row["full_vs_random_success"] = float(
            row["full"]["executed_success"]["mean"] - row["random"]["executed_success"]["mean"])
        rows.append(row)
    rows.extend(unavailable)
    return rows


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def mean_value(row: dict[str, Any], arm_name: str) -> float | None:
    try:
        return float(row[arm_name]["executed_success"]["mean"])
    except (KeyError, TypeError, ValueError):
        return None


def fmt_mean(row: dict[str, Any], arm_name: str) -> str:
    value = mean_value(row, arm_name)
    return "n/a" if value is None else f"{value:.3f}"


def row_status_html(row: dict[str, Any]) -> str:
    status = str(row.get("status", "unknown"))
    if status == "completed":
        return '<span class="status-pill pass">Completed</span>'
    if status == "controller_unavailable":
        return '<span class="status-pill partial">No local controller</span>'
    return f'<span class="status-pill fail">{escape(status.replace("_", " ").title())}</span>'


def collapse_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "completed":
            continue
        key = (
            str(row.get("env_name", "unknown")),
            int(row.get("age", -1)),
            str(row.get("status", "unknown")),
        )
        grouped.setdefault(key, []).append(row)

    collapsed: list[dict[str, Any]] = []
    for (env_name, age, status), values in grouped.items():
        item: dict[str, Any] = {
            "env_name": env_name,
            "age": int(age),
            "status": status,
            "reason": str(values[0].get("reason", "")),
        }
        seeds = sorted(int(value["seed"]) for value in values if "seed" in value)
        if seeds:
            item["seeds"] = seeds
            item["seed_count"] = len(seeds)
        if status == "controller_validation_failed":
            oracle_values = [
                float(value["oracle_success"])
                for value in values
                if "oracle_success" in value
            ]
            if oracle_values:
                item["oracle_success_mean"] = float(sum(oracle_values) / len(oracle_values))
                item["oracle_success_min_seen"] = float(min(oracle_values))
                item["oracle_success_max_seen"] = float(max(oracle_values))
            if "oracle_success_min" in values[0]:
                item["oracle_success_min"] = float(values[0]["oracle_success_min"])
        collapsed.append(item)
    return sorted(
        completed + collapsed,
        key=lambda item: (
            str(item.get("env_name", "")),
            int(item.get("age", -1)),
            str(item.get("status", "")),
        ),
    )


def seed_cell(row: dict[str, Any]) -> str:
    status = row.get("status")
    if status == "completed":
        return escape(str(row.get("seed_count", "n/a")))
    if status == "controller_validation_failed":
        seeds = row.get("seeds", [])
        seed_text = ",".join(str(seed) for seed in seeds) if seeds else "n/a"
        return f"failed {escape(seed_text)}"
    return "all"


def metric_cell(row: dict[str, Any], arm_name: str) -> str:
    status = row.get("status")
    if status == "completed":
        return fmt_mean(row, arm_name)
    if status == "controller_validation_failed":
        return "not scored" if arm_name != "oracle" else oracle_validation_cell(row)
    return "n/a"


def oracle_validation_cell(row: dict[str, Any]) -> str:
    if row.get("status") != "controller_validation_failed":
        return fmt_mean(row, "oracle")
    mean = row.get("oracle_success_mean")
    low = row.get("oracle_success_min_seen")
    high = row.get("oracle_success_max_seen")
    threshold = row.get("oracle_success_min")
    if mean is None or threshold is None:
        return "validation failed"
    span = f"{float(mean):.3f}"
    if low is not None and high is not None and abs(float(low) - float(high)) > 0.0005:
        span = f"{float(low):.3f}-{float(high):.3f}"
    return f"{span} &lt; {float(threshold):.3f}"


def native_use_html(summary: dict[str, Any], output: Path) -> str:
    raw_rows = list(summary.get("rows", []))
    rows = collapse_rows(raw_rows)
    completed = [row for row in rows if row.get("status") == "completed"]
    validation_failed = [row for row in rows if row.get("status") == "controller_validation_failed"]
    controller_unavailable = [row for row in rows if row.get("status") == "controller_unavailable"]
    full_values = [value for row in completed if (value := mean_value(row, "full")) is not None]
    random_values = [value for row in completed if (value := mean_value(row, "random")) is not None]
    recent_values = [value for row in completed if (value := mean_value(row, "recent")) is not None]
    supported_envs = sorted({str(row.get("env_name", "unknown")) for row in completed})
    validation_failed_envs = sorted({str(row.get("env_name", "unknown")) for row in validation_failed})
    controller_unavailable_envs = sorted({str(row.get("env_name", "unknown")) for row in controller_unavailable})

    mean_full = sum(full_values) / len(full_values) if full_values else None
    mean_recent = sum(recent_values) / len(recent_values) if recent_values else None
    mean_random = sum(random_values) / len(random_values) if random_values else None
    lift_recent = None if mean_full is None or mean_recent is None else mean_full - mean_recent
    lift_random = None if mean_full is None or mean_random is None else mean_full - mean_random
    worst_full = min(full_values) if full_values else None

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td><code>{escape(str(row.get('env_name', 'unknown')))}</code></td>"
            f"<td class=\"num\">{escape(str(row.get('age', 'n/a')))}</td>"
            f"<td class=\"num\">{seed_cell(row)}</td>"
            f"<td class=\"num\">{metric_cell(row, 'full')}</td>"
            f"<td class=\"num\">{metric_cell(row, 'recent')}</td>"
            f"<td class=\"num\">{metric_cell(row, 'random')}</td>"
            f"<td class=\"num\">{metric_cell(row, 'oracle')}</td>"
            f"<td>{row_status_html(row)}</td>"
            "</tr>"
        )

    output_rel = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output
    return f"""
        {HTML_START}
        <section class="section-block" id="native-use-all-status">
          <div class="section-kicker">Native success-rate extension</div>
          <h2>All-env fixed-controller success rates</h2>
          <p><b>{escape(str(summary.get('status', 'unknown')))}</b> — {len(completed)}/{len(rows)} env-age rows have audited fixed-controller native success. Rows that fail controller validation or lack a local controller are reported as scope limits, not method failures.</p>
          <div class="matrix" style="margin-top:14px">
            <div class="cell"><b>{len(completed)}/{len(rows)}</b><p>env-age rows with native success-rate estimates.</p></div>
            <div class="cell"><b>{fmt(mean_full)}</b><p>mean full-memory executed success over supported rows.</p></div>
            <div class="cell"><b>{fmt(worst_full)}</b><p>worst supported full-memory executed success.</p></div>
            <div class="cell"><b>{fmt(lift_recent)} / {fmt(lift_random)}</b><p>mean full-memory lift over recent and random baselines.</p></div>
          </div>
          <div class="table-wrap" style="margin-top:14px">
            <table>
              <caption>All-env temporal-coverage current-method native use. Completed environments: {escape(', '.join(supported_envs) or 'none')}. Controller-validation failed: {escape(', '.join(validation_failed_envs) or 'none')}. No local controller: {escape(', '.join(controller_unavailable_envs) or 'none')}.</caption>
              <thead><tr><th>Environment</th><th>Age</th><th>Seeds</th><th>Full</th><th>Recent</th><th>Random</th><th>Oracle</th><th>Status</th></tr></thead>
              <tbody>{''.join(table_rows)}</tbody>
            </table>
          </div>
          <p><small>Last update: {now()}. Machine-readable status: <code>{escape(str(output_rel))}/summary.json</code>. {escape(str(summary.get('claim_boundary', '')))}</small></p>
        </section>
        {HTML_END}
"""


def update_html(summary: dict[str, Any], output: Path) -> None:
    if not HTML_PATH.exists():
        return
    block = native_use_html(summary, output)
    text = HTML_PATH.read_text()
    if HTML_START in text and HTML_END in text:
        before, rest = text.split(HTML_START, 1)
        _, after = rest.split(HTML_END, 1)
        HTML_PATH.write_text(before + block + after)
        return
    if "</main>" in text:
        HTML_PATH.write_text(text.replace("</main>", block + "\n</main>", 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", nargs="*", required=True)
    parser.add_argument("--ages", type=int, nargs="*", default=list(AGES))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--max-rows", type=int, default=120)
    parser.add_argument("--pointmaze-horizon", type=int, default=600)
    parser.add_argument("--cube-horizon", type=int, default=650)
    parser.add_argument("--scene-horizon", type=int, default=800)
    parser.add_argument("--puzzle-horizon", type=int, default=900)
    parser.add_argument("--random-seed", type=int, default=1187)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--oracle-success-min", type=float, default=0.85)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.feature_root = args.feature_root if args.feature_root.is_absolute() else ROOT / args.feature_root
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.output.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    cubes: dict[str, dict[str, np.ndarray]] = {}
    for env_name in args.envs:
        if not is_supported(env_name):
            for age in args.ages:
                unavailable.append({
                    "env_name": env_name,
                    "age": int(age),
                    "status": "controller_unavailable",
                    "reason": "No audited low-level controller/policy is available locally for this OGBench env.",
                })
            continue
        for age in args.ages:
            for seed in args.seeds:
                path = feature_path(args.feature_root, env_name, int(age), int(seed))
                if not path.is_file():
                    if args.allow_missing:
                        unavailable.append({
                            "env_name": env_name,
                            "age": int(age),
                            "seed": int(seed),
                            "status": "feature_missing",
                            "path": str(path.relative_to(ROOT)),
                        })
                        continue
                    raise FileNotFoundError(path)
                cell = load_cell(args.feature_root, env_name, int(age), int(seed))
                labels = cell["val_labels"].astype(np.int64)
                cube_key = f"{env_name}/seed-{seed}"
                if cube_key not in cubes:
                    cubes[cube_key] = build_success_cube(env_name, labels, args)
                cube = cubes[cube_key]
                oracle_success = float(
                    cube["success"][
                        np.arange(len(cube["labels"]), dtype=np.int64),
                        cube["labels"],
                    ].astype(np.float64).mean()
                )
                if oracle_success < float(args.oracle_success_min):
                    unavailable.append({
                        "env_name": env_name,
                        "age": int(age),
                        "seed": int(seed),
                        "status": "controller_validation_failed",
                        "oracle_success": oracle_success,
                        "oracle_success_min": float(args.oracle_success_min),
                        "reason": "Fixed controller does not solve the oracle-label condition reliably enough.",
                    })
                    continue
                cells.append(evaluate_cell(args.feature_root, env_name, int(age), int(seed), cube))
    summary = {
        "schema": "masked_evidence_jepa_native_use_all_summary_v1",
        "status": "completed",
        "feature_root": str(args.feature_root.relative_to(ROOT)),
        "cell_count": int(len(cells)),
        "unavailable_count": int(len(unavailable)),
        "rows": summarize(cells, unavailable),
        "claim_boundary": (
            "Fixed-controller native use from ME-JEPA memory readout where a local audited controller exists. "
            "Controller-unavailable rows are not counted as success or failure."
        ),
    }
    (args.output / "cells.json").write_text(stable_json(cells))
    (args.output / "summary.json").write_text(stable_json(summary))
    update_html(summary, args.output)
    print(stable_json(summary), flush=True)


if __name__ == "__main__":
    main()
