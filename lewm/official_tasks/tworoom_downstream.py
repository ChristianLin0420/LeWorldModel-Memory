"""Pinned TwoRoom simulator and deterministic waypoint controller."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


STABLE_WORLDMODEL_COMMIT = "0ef3856875e70a1283e637fcd2ab936eae6c4e6f"


class _NoOpLogger:
    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_pinned_tworoom(vendor_repo: Path,
                        expected_commit: str = STABLE_WORLDMODEL_COMMIT
                        ) -> tuple[type, type, str]:
    """Load only the pinned environment, spaces, and expert policy modules."""

    vendor_repo = vendor_repo.resolve()
    actual = subprocess.run(
        ["git", "-C", str(vendor_repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    if actual != expected_commit:
        raise RuntimeError(
            f"stable-worldmodel commit differs: {actual} != {expected_commit}")
    dirty = subprocess.run(
        ["git", "-C", str(vendor_repo), "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise RuntimeError("stable-worldmodel checkout is not clean")
    root = vendor_repo / "stable_worldmodel"
    package = types.ModuleType("stable_worldmodel")
    package.__path__ = [str(root)]
    sys.modules["stable_worldmodel"] = package
    if "loguru" not in sys.modules:
        loguru = types.ModuleType("loguru")
        loguru.logger = _NoOpLogger()
        sys.modules["loguru"] = loguru
    package.utils = _load_module("stable_worldmodel.utils", root / "utils.py")
    package.spaces = _load_module("stable_worldmodel.spaces", root / "spaces.py")
    package.policy = _load_module("stable_worldmodel.policy", root / "policy.py")
    envs = types.ModuleType("stable_worldmodel.envs")
    envs.__path__ = [str(root / "envs")]
    sys.modules["stable_worldmodel.envs"] = envs
    tworoom = types.ModuleType("stable_worldmodel.envs.two_room")
    tworoom.__path__ = [str(root / "envs" / "two_room")]
    sys.modules["stable_worldmodel.envs.two_room"] = tworoom
    env_module = _load_module(
        "stable_worldmodel.envs.two_room.env",
        root / "envs" / "two_room" / "env.py")
    policy_module = _load_module(
        "stable_worldmodel.envs.two_room.expert_policy",
        root / "envs" / "two_room" / "expert_policy.py")
    return env_module.TwoRoomEnv, policy_module.ExpertPolicy, actual


def make_tworoom_env(TwoRoomEnv: type) -> Any:
    """Construct the native hidden-target environment used by the checkpoint."""

    return TwoRoomEnv(render_mode="rgb_array", render_target=False)


def execute_waypoint(env: Any, ExpertPolicy: type, *, state: np.ndarray,
                     target: np.ndarray, max_steps: int,
                     physics_seed: int = 0) -> dict[str, Any]:
    """Reset exactly to a decision state and execute one fixed controller."""

    state = np.asarray(state, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    _, info = env.reset(
        seed=physics_seed, options={"state": state, "target_state": target})
    reset = np.asarray(info["state"], dtype=np.float32)
    if not np.array_equal(reset, state):
        raise RuntimeError("TwoRoom reset did not preserve decision state exactly")
    policy = ExpertPolicy(action_noise=0.0, action_repeat_prob=0.0, seed=0)
    policy.set_env(env)
    terminated = False
    steps = 0
    for steps in range(1, max_steps + 1):
        action = policy.get_action(info)
        _, _, terminated, _, info = env.step(action)
        if terminated:
            break
    final = np.asarray(info["state"], dtype=np.float32)
    distance = float(np.linalg.norm(final - target))
    return {
        "reset_state": reset,
        "final_state": final,
        "steps": steps,
        "distance": distance,
        "success": bool(terminated),
    }


__all__ = [
    "STABLE_WORLDMODEL_COMMIT", "execute_waypoint", "load_pinned_tworoom",
    "make_tworoom_env",
]
