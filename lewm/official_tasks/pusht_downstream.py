"""Pinned PushT downstream-control primitives.

This module deliberately imports the upstream simulator lazily.  The formal
experiment runs it from an isolated environment while the carrier readout runs
from the repository environment on CUDA.  Keeping the two processes separate
prevents simulator-only dependencies from changing the frozen LeWM software
stack.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


STABLE_WORLDMODEL_COMMIT = "0ef3856875e70a1283e637fcd2ab936eae6c4e6f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def deterministic_partition(
    episode_ids: Sequence[int], *, seed: int, train_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return label-blind train/development rows from unique episode ids."""

    ids = np.asarray(episode_ids, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("partition input repeats an episode")
    if not 0 < train_count < len(ids):
        raise ValueError("train_count must leave non-empty train and development")
    keys = np.asarray([
        int.from_bytes(hashlib.sha256(f"{seed}:{int(value)}".encode()).digest()[:8],
                       "little")
        for value in ids
    ], dtype=np.uint64)
    order = np.argsort(keys, kind="stable")
    return np.sort(order[:train_count]), np.sort(order[train_count:])


def interior_state_mask(states: np.ndarray, low: float, high: float) -> np.ndarray:
    """Label-blind eligibility rule using only the native block x/y state."""

    values = np.asarray(states)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("PushT states must have shape (episodes, 7)")
    return np.all((values[:, 2:4] >= low) & (values[:, 2:4] <= high), axis=1)


def goal_directions(classes: int) -> np.ndarray:
    """Registered direction deck, beginning at north and proceeding clockwise."""

    if classes not in (4, 6):
        raise ValueError("formal PushT use tasks have four or six classes")
    angles = -np.pi / 2 + np.arange(classes) * (2 * np.pi / classes)
    return np.stack((np.cos(angles), np.sin(angles)), axis=1)


def angular_error(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return np.abs((np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi)


def target_block_poses(states: np.ndarray, classes: int,
                       displacement: float) -> np.ndarray:
    """Return per-episode native block-pose goals, shape ``(N,K,3)``."""

    values = np.asarray(states, dtype=np.float64)
    directions = goal_directions(classes)
    pose = np.empty((len(values), classes, 3), dtype=np.float64)
    pose[:, :, :2] = values[:, None, 2:4] + displacement * directions[None]
    pose[:, :, 2] = values[:, None, 4]
    return pose


def pose_success(final_pose: np.ndarray, target_pose: np.ndarray,
                 position_tolerance: float,
                 angle_tolerance: float) -> np.ndarray:
    final = np.asarray(final_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    return ((np.linalg.norm(final[..., :2] - target[..., :2], axis=-1)
             <= position_tolerance)
            & (angular_error(final[..., 2], target[..., 2])
               <= angle_tolerance))


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_pinned_pusht(vendor_repo: Path,
                      expected_commit: str = STABLE_WORLDMODEL_COMMIT):
    """Load only the pinned upstream PushT modules, not its training stack."""

    vendor_repo = vendor_repo.resolve()
    actual = subprocess.run(
        ["git", "-C", str(vendor_repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(
            f"stable-worldmodel commit differs: {actual} != {expected_commit}")
    root = vendor_repo / "stable_worldmodel"
    package = types.ModuleType("stable_worldmodel")
    package.__path__ = [str(root)]
    sys.modules["stable_worldmodel"] = package
    package.utils = _load_module("stable_worldmodel.utils", root / "utils.py")
    package.spaces = _load_module("stable_worldmodel.spaces", root / "spaces.py")
    envs = types.ModuleType("stable_worldmodel.envs")
    envs.__path__ = [str(root / "envs")]
    sys.modules["stable_worldmodel.envs"] = envs
    _load_module("stable_worldmodel.envs.utils", root / "envs" / "utils.py")
    pusht = types.ModuleType("stable_worldmodel.envs.pusht")
    pusht.__path__ = [str(root / "envs" / "pusht")]
    sys.modules["stable_worldmodel.envs.pusht"] = pusht
    module = _load_module(
        "stable_worldmodel.envs.pusht.env", root / "envs" / "pusht" / "env.py")
    return module.PushT, actual


class NativePushTController:
    """One deterministic, goal-conditioned controller shared by every arm.

    The controller first retreats radially, follows a collision-free circular
    arc to the back of the block, then pushes through its centre.  No carrier
    information enters after the categorical goal has been selected.
    """

    def __init__(self, env: Any, *, orbit_radius: float = 118.0,
                 orbit_points: int = 8, waypoint_steps: int = 5,
                 push_steps: int = 9, push_distance: float = 90.0) -> None:
        self.env = env
        self.orbit_radius = float(orbit_radius)
        self.orbit_points = int(orbit_points)
        self.waypoint_steps = int(waypoint_steps)
        self.push_steps = int(push_steps)
        self.push_distance = float(push_distance)

    @staticmethod
    def _action(agent: np.ndarray, waypoint: np.ndarray) -> np.ndarray:
        return np.clip((waypoint - agent) / 100.0, -1.0, 1.0).astype(np.float32)

    def _track(self, waypoint: np.ndarray, steps: int) -> np.ndarray:
        observation = None
        for _ in range(steps):
            state = np.asarray(self.env._get_obs(), dtype=np.float64)
            observation, _, _, _, _ = self.env.step(
                self._action(state[:2], waypoint))
        assert observation is not None
        return np.asarray(observation["state"], dtype=np.float64)

    def execute(self, state: np.ndarray, direction: np.ndarray) -> dict[str, Any]:
        state = np.asarray(state, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        observation, _ = self.env.reset(
            seed=0, options={"variation": [], "state": state,
                             "goal_state": state})
        reset_state = np.asarray(observation["state"], dtype=np.float64)
        # Upstream ``_set_state`` intentionally advances physics by one 0.01 s
        # substep.  Its exact state contract is therefore x' = x + v*dt,
        # unchanged stored velocity, and at most a sub-pixel contact-solver
        # adjustment of the block pose.
        expected_agent = state[:2] + state[5:] * float(self.env.dt)
        if (not np.allclose(reset_state[:2], expected_agent,
                            atol=2e-4, rtol=0)
                or np.max(np.abs(reset_state[2:5] - state[2:5])) > 1.0
                or not np.allclose(reset_state[5:], state[5:],
                                   atol=2e-5, rtol=0)):
            raise RuntimeError("upstream PushT reset changed cached state semantics")

        block = reset_state[2:4]
        agent = reset_state[:2]
        radial = agent - block
        if np.linalg.norm(radial) < 1e-6:
            radial = np.array([0.0, 1.0])
        start_angle = float(np.arctan2(radial[1], radial[0]))
        behind = -direction
        end_angle = float(np.arctan2(behind[1], behind[0]))
        delta = (end_angle - start_angle + np.pi) % (2 * np.pi) - np.pi

        # Retreat before orbiting so the route does not sweep through the block.
        state_now = self._track(
            block + self.orbit_radius * radial / np.linalg.norm(radial),
            self.waypoint_steps)
        path = []
        for fraction in np.linspace(0.0, 1.0, self.orbit_points + 1)[1:]:
            angle = start_angle + fraction * delta
            waypoint = block + self.orbit_radius * np.array(
                [np.cos(angle), np.sin(angle)])
            state_now = self._track(waypoint, self.waypoint_steps)
            path.append(state_now[:2].copy())

        # Approach the contact point, then push through the block centre.
        state_now = self._track(block - direction * 72.0,
                                self.waypoint_steps)
        push_target = block + direction * self.push_distance
        state_now = self._track(push_target, self.push_steps)
        return {
            "reset_state": reset_state,
            "final_state": state_now,
            "final_block_pose": state_now[2:5],
            "orbit_path": np.asarray(path),
        }


def make_native_env(PushT: Any, resolution: int = 64) -> Any:
    init = {
        "agent.start_position": np.array([256.0, 400.0]),
        "block.start_position": np.array([256.0, 256.0]),
        "block.angle": 0.0,
        "rendering.render_goal": 0,
    }
    return PushT(
        resolution=resolution, with_target=False, relative=True,
        render_mode="rgb_array", init_value=init)


__all__ = [
    "NativePushTController",
    "STABLE_WORLDMODEL_COMMIT",
    "angular_error",
    "deterministic_partition",
    "goal_directions",
    "interior_state_mask",
    "load_pinned_pusht",
    "make_native_env",
    "pose_success",
    "sha256_file",
    "stable_json",
    "target_block_poses",
]
