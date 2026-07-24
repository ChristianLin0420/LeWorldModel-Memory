"""Canonical MIKASA-Robo-VLA long-memory integration.

The environment dependency is intentionally imported lazily so the rest of
LeWorldModel can be tested without the isolated MIKASA environment.  Policy
inputs are restricted to the wrapped RGB observation, proprioception, and the
task-provided language instruction.  Privileged task fields are used only by
the evaluator and by the explicitly named oracle memory conditions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
from typing import Any, Mapping, Sequence

import numpy as np


OFFICIAL_REPOSITORY = "https://github.com/CognitiveAISystems/MIKASA-Robo"
OFFICIAL_PROJECT = "CognitiveAISystems/MIKASA-Robo"
OFFICIAL_RELEASE = "v1.0.0"
OFFICIAL_COMMIT = "16634db18bef08128ed79346469c86fc12169aed"
OFFICIAL_LICENSE = "MIT"
OFFICIAL_PAPER = "https://openreview.net/forum?id=9cLPurIZMj"
OFFICIAL_ARXIV = "https://arxiv.org/abs/2502.10550"
OFFICIAL_DATASET = "mikasa-robo/mikasa-robo-vla-lerobot"

CANONICAL_OBS_MODE = "rgb"
CANONICAL_CONTROL_MODE = "pd_ee_delta_pose"
CANONICAL_REWARD_MODE = "sparse"
CANONICAL_CAMERA_SHAPE = (128, 128, 6)
ACTION_DIM = 7
ACTION_CANDIDATES = 3
DEFAULT_MEMORY_BUDGET = 8
MATCHED_MEMORY_CONDITIONS = (
    "no_memory",
    "recent_only",
    "random_event",
    "oracle_event",
    "oracle_full_event",
)
ALL_MEMORY_CONDITIONS = MATCHED_MEMORY_CONDITIONS + ("full_history",)
ORACLE_ONLY_FIELDS = frozenset(
    {
        "oracle_info",
        "task_cue",
        "flash_active",
        "flash_color",
        "flash_triggered",
        "flash_start_step",
        "flash_duration",
    }
)


@dataclass(frozen=True)
class MikasaTaskSpec:
    env_id: str
    n_cubes: int
    horizon: int
    horizon_split: str
    memory_type: str = "Prospective"


TASK_SPECS: dict[str, MikasaTaskSpec] = {
    f"GatherAndRecall{n}-VLA-v0": MikasaTaskSpec(
        env_id=f"GatherAndRecall{n}-VLA-v0",
        n_cubes=n,
        horizon=100 * (n + 1),
        horizon_split="Medium" if n in (3, 5) else "Long",
    )
    for n in (3, 5, 7, 9)
}


@dataclass(frozen=True)
class GateThresholds:
    chance_accuracy: float = 1.0 / ACTION_CANDIDATES
    recent_ceiling: float = 0.45
    minimum_oracle_gain: float = 0.10
    minimum_closed_gap: float = 0.25
    minimum_oracle_execution: float = 0.90
    confidence: float = 0.95


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    clauses: dict[str, bool]
    metrics: dict[str, float]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayPrefix:
    """One raw canonical rollout prefix ending at the recall decision."""

    seed: int
    label: int
    instruction: str
    rgb: np.ndarray
    proprio: np.ndarray
    flash_mask: np.ndarray
    actions_used: int
    all_on_disc: bool
    final_obs: Mapping[str, Any]
    final_info: Mapping[str, Any]


def stable_digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def deterministic_episode_split(
    env_id: str,
    episode_seeds: Sequence[int],
    *,
    validation_episodes: int = 35,
    test_episodes: int = 35,
) -> dict[str, list[int]]:
    """Return exact-size, seed-disjoint train/validation/test episode indices."""

    count = len(episode_seeds)
    if validation_episodes <= 0 or test_episodes <= 0:
        raise ValueError("validation_episodes and test_episodes must be positive")
    if validation_episodes + test_episodes >= count:
        raise ValueError("split leaves no training episodes")

    ranked = sorted(
        range(count),
        key=lambda index: stable_digest(
            f"mikasa-admission-v1|{env_id}|{int(episode_seeds[index])}"
        ),
    )
    test = sorted(ranked[:test_episodes])
    validation = sorted(
        ranked[test_episodes : test_episodes + validation_episodes]
    )
    train = sorted(ranked[test_episodes + validation_episodes :])
    result = {"train": train, "validation": validation, "test": test}
    assert_split_disjoint(result, episode_seeds)
    return result


def assert_split_disjoint(
    split_indices: Mapping[str, Sequence[int]],
    episode_seeds: Sequence[int],
) -> None:
    index_sets = {
        name: {int(index) for index in indices}
        for name, indices in split_indices.items()
    }
    names = tuple(index_sets)
    for left_pos, left in enumerate(names):
        for right in names[left_pos + 1 :]:
            overlap = index_sets[left] & index_sets[right]
            if overlap:
                raise AssertionError(
                    f"episode index leakage between {left} and {right}: "
                    f"{sorted(overlap)}"
                )
    flat = [index for indices in index_sets.values() for index in indices]
    if len(flat) != len(set(flat)):
        raise AssertionError("duplicate episode indices across splits")
    if any(index < 0 or index >= len(episode_seeds) for index in flat):
        raise AssertionError("split index outside episode seed table")

    seed_sets = {
        name: {int(episode_seeds[index]) for index in indices}
        for name, indices in index_sets.items()
    }
    for left_pos, left in enumerate(names):
        for right in names[left_pos + 1 :]:
            overlap = seed_sets[left] & seed_sets[right]
            if overlap:
                raise AssertionError(
                    f"seed leakage between {left} and {right}: "
                    f"{sorted(overlap)}"
                )


def _fixed_length(indices: np.ndarray, budget: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return np.full(budget, -1, dtype=np.int64)
    if indices.size == budget:
        return indices
    if indices.size > budget:
        positions = np.linspace(0, indices.size - 1, budget)
        return indices[np.rint(positions).astype(np.int64)]
    repeats = np.resize(indices, budget)
    return repeats.astype(np.int64, copy=False)


def select_memory_indices(
    condition: str,
    history_length: int,
    *,
    budget: int = DEFAULT_MEMORY_BUDGET,
    flash_mask: Sequence[bool] | None = None,
    random_seed: int = 0,
    full_history_tokens: int = 96,
) -> np.ndarray:
    """Select raw observation indices for one named memory condition.

    ``flash_mask`` is consulted only by the two explicitly oracle conditions.
    The random and recent conditions are selected without looking at evaluator
    metadata.  A single "event" is represented by one contiguous ``budget``
    frame group so every matched condition exposes exactly the same number of
    raw observation tokens to the shared decision head.
    """

    if condition not in ALL_MEMORY_CONDITIONS:
        raise ValueError(f"unknown memory condition: {condition}")
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    if budget <= 0:
        raise ValueError("budget must be positive")

    if condition == "no_memory":
        return np.full(budget, -1, dtype=np.int64)

    if condition == "recent_only":
        start = max(0, history_length - budget)
        return _fixed_length(np.arange(start, history_length), budget)

    if condition == "random_event":
        latest_start = max(0, history_length - 2 * budget)
        rng = np.random.default_rng(int(random_seed))
        start = int(rng.integers(0, latest_start + 1))
        stop = min(history_length, start + budget)
        return _fixed_length(np.arange(start, stop), budget)

    if condition == "full_history":
        count = min(int(full_history_tokens), history_length)
        return np.rint(
            np.linspace(0, history_length - 1, count)
        ).astype(np.int64)

    if flash_mask is None:
        raise ValueError(f"{condition} requires evaluator-only flash_mask")
    mask = np.asarray(flash_mask, dtype=np.bool_).reshape(-1)
    if mask.shape[0] != history_length:
        raise ValueError("flash_mask length does not match history")
    relevant = np.flatnonzero(mask)
    if relevant.size == 0:
        raise ValueError("oracle condition requested but no flash event exists")

    if condition == "oracle_full_event":
        return _fixed_length(relevant, budget)

    scores = np.convolve(
        mask.astype(np.int64),
        np.ones(budget, dtype=np.int64),
        mode="full",
    )
    valid_starts = np.arange(max(1, history_length - budget + 1))
    # Full-convolution offset for the sum over [start, start + budget).
    window_scores = scores[valid_starts + budget - 1]
    start = int(valid_starts[int(np.argmax(window_scores))])
    stop = min(history_length, start + budget)
    return _fixed_length(np.arange(start, stop), budget)


def materialize_memory(
    values: np.ndarray,
    indices: Sequence[int],
    *,
    null_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize selected items and a validity mask without hidden metadata."""

    values = np.asarray(values)
    indices_np = np.asarray(indices, dtype=np.int64)
    output = np.full(
        (indices_np.size, *values.shape[1:]),
        null_value,
        dtype=values.dtype,
    )
    valid = indices_np >= 0
    output[valid] = values[indices_np[valid]]
    return output, valid


def assert_matched_budget(
    selections: Mapping[str, Sequence[int]],
    *,
    budget: int = DEFAULT_MEMORY_BUDGET,
) -> None:
    missing = set(MATCHED_MEMORY_CONDITIONS) - set(selections)
    if missing:
        raise AssertionError(f"missing matched conditions: {sorted(missing)}")
    lengths = {
        name: len(np.asarray(selections[name]).reshape(-1))
        for name in MATCHED_MEMORY_CONDITIONS
    }
    if set(lengths.values()) != {int(budget)}:
        raise AssertionError(f"memory token budget mismatch: {lengths}")


def recent_suffix_audit(
    flash_mask: Sequence[bool],
    recent_indices: Sequence[int],
) -> dict[str, Any]:
    mask = np.asarray(flash_mask, dtype=np.bool_).reshape(-1)
    indices = np.asarray(recent_indices, dtype=np.int64).reshape(-1)
    valid = indices >= 0
    overlap = int(mask[indices[valid]].sum()) if bool(valid.any()) else 0
    last_flash = int(np.flatnonzero(mask)[-1]) if bool(mask.any()) else -1
    first_recent = int(indices[valid].min()) if bool(valid.any()) else -1
    return {
        "passed": overlap == 0 and last_flash < first_recent,
        "flash_overlap_frames": overlap,
        "last_flash_index": last_flash,
        "first_recent_index": first_recent,
        "gap_frames": (
            first_recent - last_flash - 1
            if last_flash >= 0 and first_recent >= 0
            else None
        ),
    }


def paired_bootstrap_ci(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    confidence: float = 0.95,
    samples: int = 20_000,
    seed: int = 94731,
) -> tuple[float, float, float]:
    """Paired episode bootstrap after averaging repeated model seeds."""

    treatment_np = np.asarray(treatment, dtype=np.float64)
    control_np = np.asarray(control, dtype=np.float64)
    if treatment_np.shape != control_np.shape:
        raise ValueError("paired arrays must have identical shape")
    if treatment_np.ndim == 1:
        paired = treatment_np - control_np
    elif treatment_np.ndim == 2:
        paired = (treatment_np - control_np).mean(axis=0)
    else:
        raise ValueError("paired arrays must be [episodes] or [seeds, episodes]")
    if paired.size < 2:
        raise ValueError("at least two paired episodes are required")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, paired.size, size=(int(samples), paired.size))
    estimates = paired[draws].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(paired.mean()), float(low), float(high)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    samples: int = 20_000,
    seed: int = 4937,
) -> tuple[float, float, float]:
    values_np = np.asarray(values, dtype=np.float64)
    if values_np.ndim == 2:
        values_np = values_np.mean(axis=0)
    values_np = values_np.reshape(-1)
    if values_np.size < 2:
        raise ValueError("at least two values are required")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(
        0, values_np.size, size=(int(samples), values_np.size)
    )
    estimates = values_np[draws].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(values_np.mean()), float(low), float(high)


def decide_admission_gate(
    *,
    recent_success: np.ndarray,
    oracle_success: np.ndarray,
    no_memory_success: np.ndarray,
    recent_probe_accuracy: np.ndarray,
    thresholds: GateThresholds = GateThresholds(),
) -> GateDecision:
    recent = np.asarray(recent_success, dtype=np.float64)
    oracle = np.asarray(oracle_success, dtype=np.float64)
    no_memory = np.asarray(no_memory_success, dtype=np.float64)
    probe = np.asarray(recent_probe_accuracy, dtype=np.float64)
    for name, value in (
        ("oracle", oracle),
        ("no_memory", no_memory),
        ("probe", probe),
    ):
        if value.shape != recent.shape:
            raise ValueError(f"{name} result shape differs from recent")

    gain, gain_low, gain_high = paired_bootstrap_ci(
        oracle,
        recent,
        confidence=thresholds.confidence,
    )
    recent_mean, recent_low, recent_high = bootstrap_mean_ci(
        recent, confidence=thresholds.confidence
    )
    oracle_mean, oracle_low, oracle_high = bootstrap_mean_ci(
        oracle, confidence=thresholds.confidence
    )
    probe_mean, probe_low, probe_high = bootstrap_mean_ci(
        probe, confidence=thresholds.confidence
    )
    no_memory_mean = float(no_memory.mean())

    recoverable_gap = max(
        1e-12, thresholds.minimum_oracle_execution - no_memory_mean
    )
    closed_gap = gain / recoverable_gap
    clauses = {
        "recent_is_weak": (
            recent_mean <= thresholds.recent_ceiling
            or oracle_mean - recent_mean >= 0.20
        ),
        "oracle_gain": (
            (
                gain >= thresholds.minimum_oracle_gain
                and gain_low > 0.0
            )
            or (
                closed_gap >= thresholds.minimum_closed_gap
                and gain_low > 0.0
            )
        ),
        "oracle_execution": (
            oracle_mean >= thresholds.minimum_oracle_execution
        ),
        "recent_suffix_probe": (
            probe_mean <= thresholds.recent_ceiling
            and probe_high <= thresholds.recent_ceiling + 0.05
        ),
    }
    reasons = tuple(name for name, passed in clauses.items() if not passed)
    metrics = {
        "recent_success": recent_mean,
        "recent_success_ci_low": recent_low,
        "recent_success_ci_high": recent_high,
        "oracle_success": oracle_mean,
        "oracle_success_ci_low": oracle_low,
        "oracle_success_ci_high": oracle_high,
        "oracle_minus_recent": gain,
        "oracle_minus_recent_ci_low": gain_low,
        "oracle_minus_recent_ci_high": gain_high,
        "no_memory_success": no_memory_mean,
        "oracle_control_gap_closed": closed_gap,
        "recent_probe_accuracy": probe_mean,
        "recent_probe_ci_low": probe_low,
        "recent_probe_ci_high": probe_high,
    }
    return GateDecision(
        passed=all(clauses.values()),
        clauses=clauses,
        metrics=metrics,
        reasons=reasons,
    )


def _scalar(value: Any, *, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1)
            return value[0].item() if value.numel() else default
    except ImportError:
        pass
    array = np.asarray(value).reshape(-1)
    return array[0].item() if array.size else default


def policy_view(
    obs: Mapping[str, Any],
    info: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete and only observation surface allowed to a policy."""

    if not isinstance(obs, Mapping):
        raise TypeError("canonical MIKASA observation must be a mapping")
    keys = set(obs)
    forbidden = keys & ORACLE_ONLY_FIELDS
    if forbidden:
        raise AssertionError(
            f"privileged keys leaked into policy observation: {sorted(forbidden)}"
        )
    if set(obs) - {"rgb", "proprio"}:
        raise AssertionError(
            f"unexpected policy observation keys: {sorted(set(obs))}"
        )
    if "rgb" not in obs or "proprio" not in obs:
        raise AssertionError("policy observation requires rgb and proprio")
    instruction = info.get("language_instruction")
    if not isinstance(instruction, str) or not instruction:
        raise AssertionError("missing task-provided language instruction")
    return {
        "rgb": obs["rgb"],
        "proprio": obs["proprio"],
        "language_instruction": instruction,
    }


def canonical_environment(
    env_id: str,
    *,
    sim_backend: str = "gpu",
    render_mode: str = "all",
) -> Any:
    """Construct the official wrapped RGB evaluation environment."""

    if env_id not in TASK_SPECS:
        raise ValueError(f"unsupported admission task: {env_id}")
    import gymnasium as gym
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401
    from mikasa_robo_suite.vla.utils.apply_wrappers import (
        apply_mikasa_vla_wrappers,
    )

    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode=CANONICAL_OBS_MODE,
        control_mode=CANONICAL_CONTROL_MODE,
        reward_mode=CANONICAL_REWARD_MODE,
        render_mode=render_mode,
        sim_backend=sim_backend,
    )
    env = apply_mikasa_vla_wrappers(env, include_overlays=False)
    return env


def _numpy_observation(value: Any) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    array = np.asarray(value)
    if array.shape[0] == 1:
        array = array[0]
    return array


def replay_to_recall(
    env: Any,
    actions: np.ndarray,
    *,
    seed: int,
    capture_rgb: bool = True,
) -> ReplayPrefix:
    """Replay canonical actions until all cubes are on the disc.

    Privileged fields are copied only into this evaluator-owned receipt.  The
    returned ``final_obs`` remains the wrapped policy observation.
    """

    import torch

    obs, info = env.reset(seed=int(seed))
    view = policy_view(obs, info)
    label = int(_scalar(env.unwrapped.flash_color))
    rgb_history: list[np.ndarray] = []
    proprio_history: list[np.ndarray] = []
    flash_history: list[bool] = []

    def append(current_obs: Mapping[str, Any], current_info: Mapping[str, Any]) -> None:
        current_view = policy_view(current_obs, current_info)
        if capture_rgb:
            rgb = _numpy_observation(current_view["rgb"]).astype(
                np.uint8, copy=True
            )
            if rgb.shape != CANONICAL_CAMERA_SHAPE:
                raise AssertionError(
                    f"unexpected RGB shape {rgb.shape}, "
                    f"expected {CANONICAL_CAMERA_SHAPE}"
                )
            rgb_history.append(rgb)
        proprio_history.append(
            _numpy_observation(current_view["proprio"]).astype(
                np.float32, copy=True
            )
        )
        flash_history.append(bool(_scalar(current_info.get("flash_active"), default=False)))

    append(obs, info)
    all_on_disc = bool(_scalar(info.get("all_on_disc"), default=False))
    used = 0
    for action in np.asarray(actions, dtype=np.float32):
        tensor = torch.as_tensor(
            action,
            dtype=torch.float32,
            device=env.unwrapped.device,
        )
        obs, _, terminated, truncated, info = env.step(tensor)
        used += 1
        append(obs, info)
        all_on_disc = bool(_scalar(info.get("all_on_disc"), default=False))
        if all_on_disc:
            break
        if bool(_scalar(terminated, default=False)) or bool(
            _scalar(truncated, default=False)
        ):
            break

    rgb_array = (
        np.stack(rgb_history)
        if capture_rgb
        else np.empty((0, *CANONICAL_CAMERA_SHAPE), dtype=np.uint8)
    )
    return ReplayPrefix(
        seed=int(seed),
        label=label,
        instruction=str(view["language_instruction"]),
        rgb=rgb_array,
        proprio=np.stack(proprio_history),
        flash_mask=np.asarray(flash_history, dtype=np.bool_),
        actions_used=used,
        all_on_disc=all_on_disc,
        final_obs=obs,
        final_info=info,
    )


class ButtonPressController:
    """One fixed candidate-conditioned motor controller for every baseline.

    The memory policy supplies only an integer candidate in ``{0, 1, 2}``.
    Environment geometry is consumed by this common low-level controller, just
    as a motion planner consumes a selected goal pose.  Geometry never enters
    the memory decision head.
    """

    def __init__(
        self,
        *,
        position_scale_m: float = 0.1,
        approach_height_m: float = 0.10,
        tolerance_m: float = 0.012,
        max_approach_steps: int = 30,
        max_press_steps: int = 20,
        hold_steps: int = 5,
    ) -> None:
        self.position_scale_m = float(position_scale_m)
        self.approach_height_m = float(approach_height_m)
        self.tolerance_m = float(tolerance_m)
        self.max_approach_steps = int(max_approach_steps)
        self.max_press_steps = int(max_press_steps)
        self.hold_steps = int(hold_steps)

    def digest(self) -> str:
        return stable_digest(inspect.getsource(type(self)))

    def _action(
        self,
        proprio: np.ndarray,
        target_xyz: np.ndarray,
        *,
        gripper: float = -1.0,
    ) -> np.ndarray:
        error = np.asarray(target_xyz, dtype=np.float32) - np.asarray(
            proprio[:3], dtype=np.float32
        )
        translation = np.clip(
            error / self.position_scale_m,
            -1.0,
            1.0,
        )
        return np.asarray(
            [
                translation[0],
                translation[1],
                translation[2],
                0.0,
                0.0,
                0.0,
                float(gripper),
            ],
            dtype=np.float32,
        )

    def execute(
        self,
        env: Any,
        obs: Mapping[str, Any],
        candidate: int,
    ) -> dict[str, Any]:
        import torch

        candidate = int(candidate)
        if not 0 <= candidate < ACTION_CANDIDATES:
            raise ValueError(f"candidate outside [0, 2]: {candidate}")
        env_u = env.unwrapped
        button_xy = _numpy_observation(env_u.buttons_xy[candidate]).reshape(-1, 2)[0]
        button_top = float(_scalar(env_u.button_top_z))
        travel = float(env_u.BUTTON_CAP_TRAVEL)
        approach = np.asarray(
            [
                float(button_xy[0]),
                float(button_xy[1]),
                button_top + self.approach_height_m,
            ],
            dtype=np.float32,
        )
        current_xyz = _numpy_observation(obs["proprio"]).astype(np.float32)[:3]
        safe_lift = current_xyz.copy()
        safe_lift[2] = approach[2]
        press = approach.copy()
        press[2] = button_top - 0.70 * travel

        info: Mapping[str, Any] = {}
        terminated = truncated = False
        step_count = 0

        def step(action: np.ndarray) -> None:
            nonlocal obs, info, terminated, truncated, step_count
            tensor = torch.as_tensor(
                action,
                dtype=torch.float32,
                device=env_u.device,
            )
            obs, _, terminated_value, truncated_value, info = env.step(tensor)
            terminated = bool(_scalar(terminated_value, default=False))
            truncated = bool(_scalar(truncated_value, default=False))
            step_count += 1

        for target, limit in (
            (safe_lift, self.max_approach_steps),
            (approach, self.max_approach_steps),
        ):
            for _ in range(limit):
                if terminated or truncated:
                    break
                proprio = _numpy_observation(obs["proprio"]).astype(np.float32)
                distance = float(np.linalg.norm(target - proprio[:3]))
                if distance <= self.tolerance_m:
                    break
                step(self._action(proprio, target, gripper=1.0))

        for _ in range(3):
            step(np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            if terminated or truncated:
                break

        for _ in range(self.max_press_steps):
            if terminated or truncated:
                break
            proprio = _numpy_observation(obs["proprio"]).astype(np.float32)
            distance = float(np.linalg.norm(press - proprio[:3]))
            if distance <= self.tolerance_m:
                break
            step(self._action(proprio, press))

        for _ in range(self.hold_steps):
            if terminated or truncated:
                break
            proprio = _numpy_observation(obs["proprio"]).astype(np.float32)
            step(self._action(proprio, press))

        final = env_u.evaluate()
        return {
            "candidate": candidate,
            "success": bool(_scalar(final.get("success"), default=False)),
            "failed": bool(_scalar(final.get("failed"), default=False)),
            "pressed_button": int(
                _scalar(final.get("pressed_button"), default=-1)
            ),
            "steps": step_count,
            "terminated": terminated,
            "truncated": truncated,
        }


def controller_receipt(controller: ButtonPressController) -> dict[str, Any]:
    return {
        "class": f"{type(controller).__module__}.{type(controller).__qualname__}",
        "source_sha256": controller.digest(),
        "action_candidates": ACTION_CANDIDATES,
        "action_dim": ACTION_DIM,
        "control_mode": CANONICAL_CONTROL_MODE,
        "parameters": dict(controller.__dict__),
    }


def source_receipt() -> dict[str, Any]:
    return {
        "project": OFFICIAL_PROJECT,
        "repository": OFFICIAL_REPOSITORY,
        "release": OFFICIAL_RELEASE,
        "commit": OFFICIAL_COMMIT,
        "license": OFFICIAL_LICENSE,
        "paper": OFFICIAL_PAPER,
        "arxiv": OFFICIAL_ARXIV,
        "dataset": OFFICIAL_DATASET,
        "tasks": {name: asdict(spec) for name, spec in TASK_SPECS.items()},
        "canonical_api": {
            "obs_mode": CANONICAL_OBS_MODE,
            "control_mode": CANONICAL_CONTROL_MODE,
            "reward_mode": CANONICAL_REWARD_MODE,
            "rgb_shape": CANONICAL_CAMERA_SHAPE,
            "action_dim": ACTION_DIM,
            "wrapper": "apply_mikasa_vla_wrappers(include_overlays=False)",
        },
    }


__all__ = [
    "ACTION_CANDIDATES",
    "ACTION_DIM",
    "ALL_MEMORY_CONDITIONS",
    "ButtonPressController",
    "CANONICAL_CAMERA_SHAPE",
    "DEFAULT_MEMORY_BUDGET",
    "GateDecision",
    "GateThresholds",
    "MATCHED_MEMORY_CONDITIONS",
    "MikasaTaskSpec",
    "ReplayPrefix",
    "TASK_SPECS",
    "assert_matched_budget",
    "assert_split_disjoint",
    "bootstrap_mean_ci",
    "canonical_environment",
    "controller_receipt",
    "decide_admission_gate",
    "deterministic_episode_split",
    "materialize_memory",
    "paired_bootstrap_ci",
    "policy_view",
    "recent_suffix_audit",
    "replay_to_recall",
    "select_memory_indices",
    "source_receipt",
    "stable_digest",
]
