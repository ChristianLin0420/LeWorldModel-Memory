#!/usr/bin/env python3
"""Render native OGBench success/failure cases for the memory-use report.

The videos are qualitative evidence only.  Quantitative native-use numbers come
from ``run_ogbench_native_use_stage.py``; this script renders representative
fixed-controller executions for the same target-selection interface.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "ogbench_renders"
POINTMAZE_CASES = ("pointmaze-large-navigate-v0",)
CLASSES = 4


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def safe_stem(text: str) -> str:
    return text.replace("/", "_").replace("-", "_")


def overlay(frame: np.ndarray, title: str, subtitle: str,
            scale: int = 1) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    if int(scale) > 1:
        image = image.resize(
            (image.size[0] * int(scale), image.size[1] * int(scale)),
            Image.Resampling.NEAREST,
        )
    width, height = image.size
    header = Image.new("RGB", (width, 38), (17, 24, 39))
    canvas = Image.new("RGB", (width, height + 38), (17, 24, 39))
    canvas.paste(header, (0, 0))
    canvas.paste(image, (0, 38))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill=(251, 212, 91))
    draw.text((8, 21), subtitle, fill=(245, 244, 239))
    return np.asarray(canvas)


def write_gif(frames: list[np.ndarray], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / float(fps)
    imageio.mimsave(path, frames, duration=duration, loop=0)


def pointmaze_run(env_name: str, true_label: int, selected_label: int,
                  row_index: int, horizon: int, sample_every: int,
                  render: bool, render_scale: int = 1) -> dict[str, Any]:
    import ogbench  # noqa: WPS433

    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    frames: list[np.ndarray] = []
    try:
        _, info = env.reset(
            seed=880_000 + int(row_index),
            options={"task_id": int(true_label) + 1},
        )
        del info
        unwrapped = env.unwrapped
        selected_goal = (
            np.asarray(unwrapped.cur_goal_xy, dtype=np.float64)
            if int(selected_label) == int(true_label)
            else np.asarray(
                unwrapped.task_infos[int(selected_label)]["goal_xy"],
                dtype=np.float64,
            )
        )
        success = False
        total_reward = 0.0
        steps = 0
        if render:
            frames.append(overlay(
                env.render(),
                f"{env_name} | selected={selected_label} true={true_label}",
                "start",
                render_scale,
            ))
        for steps in range(1, int(horizon) + 1):
            xy = np.asarray(unwrapped.get_xy(), dtype=np.float64)
            subgoal, _ = unwrapped.get_oracle_subgoal(xy, selected_goal)
            target = selected_goal if np.linalg.norm(subgoal - xy) < 0.7 else subgoal
            diff = np.asarray(target, dtype=np.float64) - xy
            norm = np.linalg.norm(diff)
            action = diff / (norm + 1e-6) * min(1.0, norm / 0.2)
            action = np.clip(action, -1.0, 1.0)
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            success = success or bool(info.get("success", False))
            if render and (steps % int(sample_every) == 0 or terminated or truncated):
                frames.append(overlay(
                    env.render(),
                    f"{env_name} | selected={selected_label} true={true_label}",
                    f"step={steps} success={int(success)} reward={total_reward:.1f}",
                    render_scale,
                ))
            if terminated or truncated:
                break
        final_xy = np.asarray(unwrapped.get_xy(), dtype=np.float64)
        true_goal = np.asarray(unwrapped.cur_goal_xy, dtype=np.float64)
        return {
            "success": bool(success),
            "reward_sum": float(total_reward),
            "steps": int(steps),
            "final_distance": float(np.linalg.norm(final_xy - true_goal)),
            "frames": frames,
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


def cube_run(true_label: int, selected_label: int, row_index: int,
             horizon: int, sample_every: int, render: bool,
             render_scale: int = 1) -> dict[str, Any]:
    import ogbench  # noqa: WPS433
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle  # noqa: WPS433

    env_name = "cube-single-play-v0"
    env = ogbench.make_env_and_datasets(env_name, env_only=True)
    frames: list[np.ndarray] = []
    try:
        ob, info = env.reset(
            seed=990_000 + int(row_index),
            options={"task_id": int(true_label) + 1},
        )
        oracle = CubeMarkovOracle(env=env)
        oracle.reset(ob, cube_augmented_info(info, env, selected_label))
        success = False
        total_reward = 0.0
        steps = 0
        if render:
            frames.append(overlay(
                env.render(),
                f"{env_name} | selected={selected_label} true={true_label}",
                "start",
                render_scale,
            ))
        for steps in range(1, int(horizon) + 1):
            action = oracle.select_action(
                ob, cube_augmented_info(info, env, selected_label))
            ob, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            success = success or bool(info.get("success", False))
            if render and (steps % int(sample_every) == 0
                           or terminated or truncated or oracle.done):
                frames.append(overlay(
                    env.render(),
                    f"{env_name} | selected={selected_label} true={true_label}",
                    f"step={steps} success={int(success)} reward={total_reward:.1f}",
                    render_scale,
                ))
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
            "frames": frames,
        }
    finally:
        env.close()


def find_pointmaze_case(env_name: str, want_success: bool,
                        horizon: int) -> tuple[int, int, int, dict[str, Any]]:
    for row_index in range(12):
        for true_label in range(CLASSES):
            labels = [true_label] if want_success else [
                value for value in range(CLASSES) if value != true_label]
            for selected_label in labels:
                result = pointmaze_run(
                    env_name, true_label, selected_label, row_index,
                    horizon, sample_every=20, render=False)
                if bool(result["success"]) == bool(want_success):
                    return true_label, selected_label, row_index, result
    raise RuntimeError(f"no pointmaze case found for {env_name} success={want_success}")


def find_cube_case(want_success: bool,
                   horizon: int) -> tuple[int, int, int, dict[str, Any]]:
    for row_index in range(12):
        for true_label in range(CLASSES):
            labels = [true_label] if want_success else [
                value for value in range(CLASSES) if value != true_label]
            for selected_label in labels:
                result = cube_run(
                    true_label, selected_label, row_index, horizon,
                    sample_every=4, render=False)
                if bool(result["success"]) == bool(want_success):
                    return true_label, selected_label, row_index, result
    raise RuntimeError(f"no cube case found for success={want_success}")


def render_pointmaze(env_name: str, case_name: str, want_success: bool,
                     args: argparse.Namespace) -> dict[str, Any]:
    result = None
    true_label = selected_label = row_index = -1
    for candidate_row in range(16):
        for candidate_true in range(CLASSES):
            labels = [candidate_true] if want_success else [
                value for value in range(CLASSES) if value != candidate_true]
            for candidate_selected in labels:
                candidate = pointmaze_run(
                    env_name, candidate_true, candidate_selected, candidate_row,
                    args.pointmaze_horizon, args.pointmaze_sample_every,
                    render=True, render_scale=args.scale)
                if bool(candidate["success"]) == bool(want_success):
                    result = candidate
                    true_label = candidate_true
                    selected_label = candidate_selected
                    row_index = candidate_row
                    break
            if result is not None:
                break
        if result is not None:
            break
    if result is None:
        raise RuntimeError(
            f"no rendered pointmaze case found for {env_name} success={want_success}")
    out_path = args.output / f"{safe_stem(env_name)}_{case_name}.gif"
    write_gif(result["frames"], out_path, args.fps)
    return {
        "env_name": env_name,
        "case": case_name,
        "path": str(out_path.relative_to(ROOT)),
        "true_label": int(true_label),
        "selected_label": int(selected_label),
        "row_index": int(row_index),
        "success": bool(result["success"]),
        "probe_success": bool(result["success"]),
        "reward_sum": float(result["reward_sum"]),
        "steps": int(result["steps"]),
        "final_distance": float(result["final_distance"]),
    }


def render_cube(case_name: str, want_success: bool,
                args: argparse.Namespace) -> dict[str, Any]:
    true_label, selected_label, row_index, probe = find_cube_case(
        want_success, args.cube_horizon)
    result = cube_run(
        true_label, selected_label, row_index, args.cube_horizon,
        args.cube_sample_every, render=True, render_scale=args.scale)
    env_name = "cube-single-play-v0"
    out_path = args.output / f"{safe_stem(env_name)}_{case_name}.gif"
    write_gif(result["frames"], out_path, args.fps)
    return {
        "env_name": env_name,
        "case": case_name,
        "path": str(out_path.relative_to(ROOT)),
        "true_label": int(true_label),
        "selected_label": int(selected_label),
        "row_index": int(row_index),
        "success": bool(result["success"]),
        "probe_success": bool(probe["success"]),
        "reward_sum": float(result["reward_sum"]),
        "steps": int(result["steps"]),
        "final_distance": float(result["final_distance"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pointmaze-envs", nargs="*", default=list(POINTMAZE_CASES))
    parser.add_argument("--pointmaze-horizon", type=int, default=600)
    parser.add_argument("--cube-horizon", type=int, default=220)
    parser.add_argument("--pointmaze-sample-every", type=int, default=12)
    parser.add_argument("--cube-sample-every", type=int, default=3)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--skip-cube", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for env_name in args.pointmaze_envs:
        records.append(render_pointmaze(
            env_name, "full_memory_success", True, args))
        records.append(render_pointmaze(
            env_name, "wrong_target_failure", False, args))
    if not args.skip_cube:
        records.append(render_cube("full_memory_success", True, args))
        records.append(render_cube("wrong_target_failure", False, args))
    manifest = {
        "schema": "ogbench_native_render_cases_v1",
        "claim_boundary": (
            "Representative fixed-controller videos for qualitative inspection; "
            "not a substitute for aggregate native-use metrics."),
        "case_count": int(len(records)),
        "cases": records,
    }
    (args.output / "manifest.json").write_text(stable_json(manifest))
    print(stable_json(manifest), flush=True)


if __name__ == "__main__":
    main()
