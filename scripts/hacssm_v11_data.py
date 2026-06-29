#!/usr/bin/env python3
"""Deterministic clean-pixel data and corruption views for HACSSM-v11.

V11 removes the smooth-action shortcut in the V10 cache: every control is an
independent bounded tanh-Gaussian draw.  It also stores the flattened native
``timestep.observation`` as the primary *evaluation-only* state target.  Raw
``physics.get_state()`` remains available as a secondary diagnostic.  Neither
target is consumed by the self-supervised model objective.

The cache contains only clean simulator trajectories.  Training and held-out
corruptions are deterministic dataset views and retain the V10 definitions in
a disjoint corruption-seed namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SCHEMA_VERSION = 2
ACTION_PROCESS = "bounded_tanh_iid_gaussian"
TASKS = (
    "walker.walk",
    "hopper.hop",
    "cartpole.swingup",
    "pendulum.swingup",
    "fish.swim",
)
VIEWS = (
    "clean",
    "train",
    "freeze",
    "gaussian_noise",
    "checkerboard",
    "long_freeze",
)
DEFAULT_ROOT = "outputs/hacssm_v11_data"
DEFAULT_TRAIN_EPISODES = 1_200
DEFAULT_VAL_EPISODES = 240
DEFAULT_LENGTH = 48
DEFAULT_IMG_SIZE = 64
DEFAULT_TRAIN_SEED = 37_100
DEFAULT_VAL_SEED = 103_710
DEFAULT_SMOOTH_RHO = 0.0
DEFAULT_CORRUPTION_SEED = 11_012

TASK_SCHEMA_FIELDS = (
    "task_observation_keys",
    "task_observation_shape_offsets",
    "task_observation_shape_values",
    "task_observation_slices",
)
CONTENT_FIELDS = (
    "schema_version",
    "action_process",
    "env_id",
    "split",
    "seed",
    "length",
    "img_size",
    "smooth_rho",
    "obs",
    "actions",
    "task_observation",
    *TASK_SCHEMA_FIELDS,
    "physics_state",
    "rewards",
    "action_min",
    "action_max",
)
REQUIRED_FIELDS = frozenset((*CONTENT_FIELDS, "content_sha256"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(name: str, value: Any) -> bytes:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.digest()


def content_sha256(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in CONTENT_FIELDS:
        if name not in values:
            raise ValueError(f"missing content field {name!r}")
        digest.update(_array_digest(name, values[name]))
    return digest.hexdigest()


def cache_name(env_id: str, split: str, episodes: int, length: int,
               img_size: int, seed: int) -> str:
    safe = f"dmc_{env_id.replace('.', '_')}"
    return f"{safe}_{split}_n{episodes}_L{length}_s{img_size}_seed{seed}.npz"


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _canonical_env_id(value: str) -> str:
    value = value.removeprefix("dmc:")
    if value not in TASKS:
        raise ValueError(f"unsupported V11 environment {value!r}; expected one of {TASKS}")
    return value


def observation_schema_arrays(
        keys: Sequence[str], shapes: Sequence[Sequence[int]]) -> dict[str, np.ndarray]:
    """Encode an ordered observation dictionary schema without object arrays."""
    if len(keys) != len(shapes) or not keys:
        raise ValueError("task-observation keys and shapes must be non-empty and aligned")
    normalized_keys = tuple(str(key) for key in keys)
    if any(not key for key in normalized_keys) or len(set(normalized_keys)) != len(keys):
        raise ValueError("task-observation keys must be non-empty and unique")
    offsets = [0]
    shape_values: list[int] = []
    slices = []
    cursor = 0
    for raw_shape in shapes:
        shape = tuple(int(value) for value in raw_shape)
        if any(value <= 0 for value in shape):
            raise ValueError(f"task-observation shape must be positive, got {shape}")
        shape_values.extend(shape)
        offsets.append(len(shape_values))
        width = int(np.prod(shape, dtype=np.int64)) if shape else 1
        slices.append((cursor, cursor + width))
        cursor += width
    return {
        "task_observation_keys": np.asarray(normalized_keys, dtype=np.str_),
        "task_observation_shape_offsets": np.asarray(offsets, dtype=np.int64),
        "task_observation_shape_values": np.asarray(shape_values, dtype=np.int64),
        "task_observation_slices": np.asarray(slices, dtype=np.int64),
    }


def decode_observation_schema(
        values: Mapping[str, np.ndarray], *, label: str = "cache"
        ) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...], int]:
    keys_array = np.asarray(values["task_observation_keys"])
    offsets = np.asarray(values["task_observation_shape_offsets"])
    shape_values = np.asarray(values["task_observation_shape_values"])
    slices = np.asarray(values["task_observation_slices"])
    if keys_array.ndim != 1 or keys_array.dtype.kind not in {"U", "S"} or len(keys_array) < 1:
        raise ValueError(f"{label}: task-observation keys must be a non-empty string array")
    keys = tuple(str(key) for key in keys_array)
    if any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError(f"{label}: task-observation keys must be non-empty and unique")
    if not np.issubdtype(offsets.dtype, np.integer) or offsets.shape != (len(keys) + 1,):
        raise ValueError(f"{label}: invalid task-observation shape offsets")
    if not np.issubdtype(shape_values.dtype, np.integer) or shape_values.ndim != 1:
        raise ValueError(f"{label}: invalid task-observation shape values")
    if not np.issubdtype(slices.dtype, np.integer) or slices.shape != (len(keys), 2):
        raise ValueError(f"{label}: invalid task-observation slices")
    if offsets[0] != 0 or offsets[-1] != len(shape_values) or np.any(np.diff(offsets) < 0):
        raise ValueError(f"{label}: inconsistent task-observation shape offsets")
    shapes: list[tuple[int, ...]] = []
    cursor = 0
    for index in range(len(keys)):
        shape = tuple(int(value) for value in shape_values[offsets[index]:offsets[index + 1]])
        if any(value <= 0 for value in shape):
            raise ValueError(f"{label}: non-positive task-observation shape {shape}")
        width = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if tuple(int(value) for value in slices[index]) != (cursor, cursor + width):
            raise ValueError(f"{label}: non-contiguous task-observation slices")
        shapes.append(shape)
        cursor += width
    return keys, tuple(shapes), cursor


def _observation_schema(observation: Mapping[str, Any]
                        ) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    if not isinstance(observation, Mapping) or not observation:
        raise ValueError("DMC timestep observation must be a non-empty mapping")
    keys = tuple(str(key) for key in observation.keys())
    shapes = tuple(np.asarray(observation[key]).shape for key in observation)
    # Reuse the serialized-schema validator for uniqueness and positive dimensions.
    observation_schema_arrays(keys, shapes)
    return keys, shapes


def _flatten_task_observation(observation: Mapping[str, Any],
                              keys: Sequence[str],
                              shapes: Sequence[Sequence[int]]) -> np.ndarray:
    if tuple(str(key) for key in observation.keys()) != tuple(keys):
        raise ValueError("DMC task-observation keys changed within a cache")
    chunks = []
    for key, raw_shape in zip(keys, shapes, strict=True):
        value = np.asarray(observation[key], dtype=np.float64)
        shape = tuple(raw_shape)
        if value.shape != shape:
            raise ValueError(
                f"DMC task-observation shape changed for {key!r}: {value.shape} != {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"DMC task observation {key!r} contains non-finite values")
        chunks.append(value.reshape(-1))
    return np.concatenate(chunks).astype(np.float64, copy=False)


def iid_bounded_action(rng: np.random.Generator, action_min: np.ndarray,
                       action_max: np.ndarray) -> np.ndarray:
    """Draw one independent tanh-Gaussian control within native DMC bounds."""
    latent = rng.standard_normal(action_min.shape).astype(np.float32)
    center = (action_min + action_max) * 0.5
    half_range = (action_max - action_min) * 0.5
    return np.clip(center + half_range * np.tanh(latent), action_min, action_max).astype(
        np.float32)


def collect_clean_dmc(env_id: str, episodes: int, length: int, img_size: int,
                      seed: int, smooth_rho: float = DEFAULT_SMOOTH_RHO
                      ) -> dict[str, np.ndarray]:
    """Collect deterministic DMC trajectories under IID bounded continuous actions."""
    env_id = _canonical_env_id(env_id)
    if episodes < 1 or length < 4 or img_size < 16:
        raise ValueError("episodes, length, and img_size must be positive and nontrivial")
    if not math.isfinite(smooth_rho) or smooth_rho != 0.0:
        raise ValueError("V11 requires smooth_rho=0.0 (IID actions)")

    os.environ.setdefault("MUJOCO_GL", "egl")
    from dm_control import suite

    domain, task = env_id.split(".", 1)
    env = suite.load(domain, task, task_kwargs={"random": int(seed)})
    action_spec = env.action_spec()
    action_min = np.broadcast_to(action_spec.minimum, action_spec.shape).astype(np.float32)
    action_max = np.broadcast_to(action_spec.maximum, action_spec.shape).astype(np.float32)
    action_min = np.nan_to_num(action_min, neginf=-1.0)
    action_max = np.nan_to_num(action_max, posinf=1.0)
    if not (action_min.ndim == 1 and np.isfinite(action_min).all()
            and np.isfinite(action_max).all() and np.all(action_max > action_min)):
        raise ValueError(f"invalid action bounds for {env_id}")

    rng = np.random.default_rng(seed)
    observations, actions, task_observations = [], [], []
    states, rewards = [], []
    schema_keys: tuple[str, ...] | None = None
    schema_shapes: tuple[tuple[int, ...], ...] | None = None

    for _episode in range(episodes):
        timestep = env.reset()
        if schema_keys is None:
            schema_keys, schema_shapes = _observation_schema(timestep.observation)
        assert schema_shapes is not None
        frames = [env.physics.render(img_size, img_size, camera_id=0)]
        episode_states = [np.asarray(env.physics.get_state(), dtype=np.float64).copy()]
        episode_task = [
            _flatten_task_observation(timestep.observation, schema_keys, schema_shapes)]
        episode_actions, episode_rewards = [], []
        for _step in range(length - 1):
            action = iid_bounded_action(rng, action_min, action_max)
            timestep = env.step(action)
            if timestep.last():
                raise RuntimeError(
                    f"{env_id} terminated before the requested length {length}; "
                    "refusing to splice simulator episodes")
            frames.append(env.physics.render(img_size, img_size, camera_id=0))
            episode_states.append(
                np.asarray(env.physics.get_state(), dtype=np.float64).copy())
            episode_task.append(
                _flatten_task_observation(timestep.observation, schema_keys, schema_shapes))
            episode_actions.append(action)
            episode_rewards.append(0.0 if timestep.reward is None else float(timestep.reward))
        observations.append(np.stack(frames).astype(np.uint8))
        actions.append(np.stack(episode_actions).astype(np.float32))
        task_observations.append(np.stack(episode_task).astype(np.float64))
        states.append(np.stack(episode_states).astype(np.float64))
        rewards.append(np.asarray(episode_rewards, dtype=np.float32))

    assert schema_keys is not None and schema_shapes is not None
    return {
        "obs": np.stack(observations),
        "actions": np.stack(actions),
        "task_observation": np.stack(task_observations),
        **observation_schema_arrays(schema_keys, schema_shapes),
        "physics_state": np.stack(states),
        "rewards": np.stack(rewards),
        "action_min": action_min,
        "action_max": action_max,
    }


def _validate_cache_values(values: Mapping[str, np.ndarray], *, label: str) -> dict[str, Any]:
    if int(values["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"{label}: unsupported V11 schema version")
    if str(values["action_process"]) != ACTION_PROCESS:
        raise ValueError(f"{label}: invalid V11 action process")
    if float(values["smooth_rho"]) != 0.0:
        raise ValueError(f"{label}: V11 cache must use smooth_rho=0.0")
    if str(values["split"]) not in {"train", "val"}:
        raise ValueError(f"{label}: invalid split")
    obs = np.asarray(values["obs"])
    actions = np.asarray(values["actions"])
    task_observation = np.asarray(values["task_observation"])
    physics_state = np.asarray(values["physics_state"])
    rewards = np.asarray(values["rewards"])
    action_min = np.asarray(values["action_min"])
    action_max = np.asarray(values["action_max"])
    if obs.dtype != np.uint8 or obs.ndim != 5 or obs.shape[-1] != 3:
        raise ValueError(f"{label}: invalid RGB observations {obs.shape}/{obs.dtype}")
    episodes, length, height, width, _ = obs.shape
    action_dim = actions.shape[-1] if actions.ndim == 3 else -1
    state_dim = physics_state.shape[-1] if physics_state.ndim == 3 else -1
    task_dim = task_observation.shape[-1] if task_observation.ndim == 3 else -1
    keys, shapes, schema_dim = decode_observation_schema(values, label=label)
    if (
        actions.shape != (episodes, length - 1, action_dim)
        or task_observation.shape != (episodes, length, task_dim)
        or physics_state.shape != (episodes, length, state_dim)
        or rewards.shape != (episodes, length - 1)
        or action_min.shape != (action_dim,)
        or action_max.shape != (action_dim,)
        or height != width
        or length != int(values["length"])
        or height != int(values["img_size"])
        or action_dim < 1 or state_dim < 1 or task_dim < 1
        or task_dim != schema_dim
    ):
        raise ValueError(f"{label}: inconsistent V11 trajectory shapes")
    for name in ("actions", "task_observation", "physics_state", "rewards",
                 "action_min", "action_max"):
        if not np.issubdtype(np.asarray(values[name]).dtype, np.number):
            raise ValueError(f"{label}: non-numeric {name}")
        if not np.isfinite(values[name]).all():
            raise ValueError(f"{label}: non-finite {name}")
    if not np.all(action_max > action_min):
        raise ValueError(f"{label}: invalid action bounds")
    tolerance = 1e-6
    if np.any(actions < action_min.reshape(1, 1, -1) - tolerance) or np.any(
            actions > action_max.reshape(1, 1, -1) + tolerance):
        raise ValueError(f"{label}: action outside native bounds")
    return {
        "episodes": episodes,
        "length": length,
        "img_size": height,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "task_observation_dim": task_dim,
        "task_observation_keys": keys,
        "task_observation_shapes": shapes,
    }


def write_cache(path: Path, *, env_id: str, split: str, seed: int,
                length: int, img_size: int, smooth_rho: float,
                arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path = path.resolve()
    if path.exists() or sidecar_path(path).exists():
        raise FileExistsError(f"refusing to overwrite V11 data cache {path}")
    values: dict[str, Any] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "action_process": np.asarray(ACTION_PROCESS),
        "env_id": np.asarray(_canonical_env_id(env_id)),
        "split": np.asarray(split),
        "seed": np.asarray(seed, dtype=np.int64),
        "length": np.asarray(length, dtype=np.int64),
        "img_size": np.asarray(img_size, dtype=np.int64),
        "smooth_rho": np.asarray(smooth_rho, dtype=np.float64),
        **arrays,
    }
    if set(values) != set(CONTENT_FIELDS):
        raise ValueError(
            f"V11 cache fields {sorted(values)} != {sorted(CONTENT_FIELDS)}")
    _validate_cache_values(values, label=str(path))
    values["content_sha256"] = np.asarray(content_sha256(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    file_hash = sha256_file(path)
    _atomic_text(sidecar_path(path), f"{file_hash}  {path.name}\n")
    return {
        "path": str(path),
        "sha256": file_hash,
        "content_sha256": str(values["content_sha256"]),
        "bytes": path.stat().st_size,
    }


@dataclass(frozen=True)
class CacheMetadata:
    path: Path
    env_id: str
    split: str
    seed: int
    length: int
    img_size: int
    smooth_rho: float
    episodes: int
    action_dim: int
    state_dim: int
    task_observation_dim: int
    task_observation_keys: tuple[str, ...]
    task_observation_shapes: tuple[tuple[int, ...], ...]
    file_sha256: str
    content_sha256: str


def load_cache(path: str | Path, *, verify: bool = True,
               return_values: bool = False
               ) -> CacheMetadata | tuple[dict[str, np.ndarray], CacheMetadata]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = sidecar_path(path)
    if verify:
        if not sidecar.is_file():
            raise FileNotFoundError(f"missing V11 cache sidecar {sidecar}")
        file_hash = sha256_file(path)
        if sidecar.read_text() != f"{file_hash}  {path.name}\n":
            raise ValueError(f"V11 cache sidecar mismatch for {path}")
    else:
        file_hash = sha256_file(path)
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != REQUIRED_FIELDS:
            raise ValueError(
                f"{path}: fields {sorted(source.files)} != {sorted(REQUIRED_FIELDS)}")
        values = {name: np.array(source[name], copy=True) for name in source.files}
    actual_content_hash = content_sha256(values)
    if str(values["content_sha256"]) != actual_content_hash:
        raise ValueError(f"{path}: content hash mismatch")
    details = _validate_cache_values(values, label=str(path))
    metadata = CacheMetadata(
        path=path,
        env_id=str(values["env_id"]),
        split=str(values["split"]),
        seed=int(values["seed"]),
        length=details["length"],
        img_size=details["img_size"],
        smooth_rho=float(values["smooth_rho"]),
        episodes=details["episodes"],
        action_dim=details["action_dim"],
        state_dim=details["state_dim"],
        task_observation_dim=details["task_observation_dim"],
        task_observation_keys=details["task_observation_keys"],
        task_observation_shapes=details["task_observation_shapes"],
        file_sha256=file_hash,
        content_sha256=actual_content_hash,
    )
    return (values, metadata) if return_values else metadata


def _episode_rng(base_seed: int, episode: int, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence((int(base_seed), int(episode), int(salt))))


def corruption_interval(length: int, view: str, seed: int, episode: int,
                        history_len: int = 3) -> tuple[int, int]:
    """The V10 interval contract, retained with the V11 corruption seed."""
    if view not in VIEWS:
        raise ValueError(f"unknown corruption view {view!r}")
    rng = _episode_rng(seed, episode, 17)
    if view == "long_freeze":
        low, high = max(history_len + 3, length // 3), max(history_len + 4, length // 2)
    else:
        low, high = max(4, length // 8), max(5, length // 4)
    gap = int(rng.integers(low, high + 1))
    earliest = history_len + 2
    latest = length - gap - 2
    if latest < earliest:
        raise ValueError(f"length {length} is too short for {view} corruption")
    start = int(rng.integers(earliest, latest + 1))
    return start, start + gap


class V11TrajectoryDataset(Dataset):
    """Validated V11 clean cache exposed through one deterministic corruption view."""

    def __init__(self, path: str | Path, view: str,
                 corruption_seed: int = DEFAULT_CORRUPTION_SEED,
                 history_len: int = 3, verify: bool = True):
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
        values, metadata = load_cache(path, verify=verify, return_values=True)
        self.obs = values["obs"]
        self.actions = values["actions"].astype(np.float32, copy=False)
        self.task_observation = values["task_observation"].astype(np.float32)
        self.physics_state = values["physics_state"].astype(np.float32)
        self.metadata = metadata
        self.view = view
        self.corruption_seed = int(corruption_seed)
        self.history_len = int(history_len)
        self.mean_frame = np.rint(self.obs.mean(axis=(0, 1))).clip(0, 255).astype(np.uint8)

    def __len__(self) -> int:
        return self.metadata.episodes

    def _corrupt(self, clean: np.ndarray, index: int,
                 start: int, end: int) -> tuple[np.ndarray, str]:
        observed = clean.copy()
        rng = _episode_rng(self.corruption_seed, index, 101)
        effective_view = self.view
        if self.view == "train":
            effective_view = "cutout" if int(rng.integers(0, 2)) == 0 else "meanframe"
        if self.view == "clean":
            return observed, "clean"
        if effective_view == "cutout":
            height, width = clean.shape[1:3]
            cut_h = max(1, int(round(height * 0.55)))
            cut_w = max(1, int(round(width * 0.55)))
            top = int(rng.integers(0, height - cut_h + 1))
            left = int(rng.integers(0, width - cut_w + 1))
            fill = np.rint(self.mean_frame.mean(axis=(0, 1))).astype(np.uint8)
            observed[start:end, top:top + cut_h, left:left + cut_w] = fill
        elif effective_view == "meanframe":
            observed[start:end] = self.mean_frame
        elif effective_view in {"freeze", "long_freeze"}:
            observed[start:end] = clean[start - 1]
        elif effective_view == "gaussian_noise":
            noise = rng.normal(127.5, 63.75, size=observed[start:end].shape)
            observed[start:end] = np.rint(noise).clip(0, 255).astype(np.uint8)
        elif effective_view == "checkerboard":
            height, width = clean.shape[1:3]
            yy, xx = np.indices((height, width))
            checker = ((yy // 8 + xx // 8) % 2).astype(bool)
            fill = np.rint(self.mean_frame.mean(axis=(0, 1))).astype(np.uint8)
            segment = observed[start:end]
            segment[:, checker] = fill
            observed[start:end] = segment
        else:
            raise AssertionError(f"unhandled corruption {effective_view}")
        return observed, effective_view

    def __getitem__(self, index: int) -> dict[str, Any]:
        clean = self.obs[index]
        start, end = corruption_interval(
            self.metadata.length, self.view, self.corruption_seed, index,
            history_len=self.history_len)
        observed, effective_view = self._corrupt(clean, index, start, end)
        corruption_mask = np.zeros(self.metadata.length, dtype=np.bool_)
        if self.view != "clean":
            corruption_mask[start:end] = True
        return {
            "observed": torch.from_numpy(
                observed.astype(np.float32) / 255.0).permute(0, 3, 1, 2),
            "clean": torch.from_numpy(
                clean.astype(np.float32) / 255.0).permute(0, 3, 1, 2),
            "actions": torch.from_numpy(self.actions[index]),
            "task_observation": torch.from_numpy(self.task_observation[index]),
            "physics_state": torch.from_numpy(self.physics_state[index]),
            "corruption_mask": torch.from_numpy(corruption_mask),
            "gap_start": torch.tensor(start, dtype=torch.long),
            "gap_end": torch.tensor(end, dtype=torch.long),
            "condition": effective_view,
            "episode_index": torch.tensor(index, dtype=torch.long),
        }


def _collect_one(root: Path, env_id: str, split: str, episodes: int,
                 length: int, img_size: int, seed: int,
                 smooth_rho: float) -> dict[str, Any]:
    name = cache_name(env_id, split, episodes, length, img_size, seed)
    path = root / name
    if path.exists():
        metadata = load_cache(path)
        expected = (env_id, split, seed, length, img_size, episodes, 0.0)
        actual = (metadata.env_id, metadata.split, metadata.seed, metadata.length,
                  metadata.img_size, metadata.episodes, metadata.smooth_rho)
        if actual != expected:
            raise ValueError(f"existing V11 cache contract mismatch: {actual} != {expected}")
        return {
            "path": str(path.resolve()),
            "sha256": metadata.file_sha256,
            "content_sha256": metadata.content_sha256,
            "bytes": path.stat().st_size,
        }
    arrays = collect_clean_dmc(env_id, episodes, length, img_size, seed, smooth_rho)
    return write_cache(
        path, env_id=env_id, split=split, seed=seed, length=length,
        img_size=img_size, smooth_rho=smooth_rho, arrays=arrays)


def _write_manifest(root: Path, records: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    manifest_path = root / "manifest.json"
    sidecar = root / "manifest.sha256"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": protocol,
        "artifacts": sorted(records, key=lambda item: item["path"]),
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text() != payload:
            raise ValueError(f"existing V11 manifest differs: {manifest_path}")
    else:
        _atomic_text(manifest_path, payload)
    digest = sha256_file(manifest_path)
    wanted = f"{digest}  {manifest_path.name}\n"
    if sidecar.exists() and sidecar.read_text() != wanted:
        raise ValueError(f"existing V11 manifest sidecar differs: {sidecar}")
    if not sidecar.exists():
        _atomic_text(sidecar, wanted)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--all", action="store_true", help="collect all five V11 tasks")
    parser.add_argument("--env", action="append", choices=TASKS)
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--val-episodes", type=int, default=DEFAULT_VAL_EPISODES)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED)
    parser.add_argument("--smooth-rho", type=float, default=DEFAULT_SMOOTH_RHO)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not math.isfinite(args.smooth_rho) or args.smooth_rho != 0.0:
        raise ValueError("V11 collection requires --smooth-rho 0.0")
    environments = TASKS if args.all else tuple(args.env or ())
    if not environments:
        raise ValueError("select --all or at least one --env")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    splits = ("train", "val") if args.split == "both" else (args.split,)
    records = []
    for env_id in environments:
        for split in splits:
            episodes = args.train_episodes if split == "train" else args.val_episodes
            seed = args.train_seed if split == "train" else args.val_seed
            print(f"collect/validate V11 {env_id} {split}: n={episodes} L={args.length}",
                  flush=True)
            records.append(_collect_one(
                root, env_id, split, episodes, args.length, args.img_size,
                seed, args.smooth_rho))
    protocol = {
        "tasks": list(environments),
        "splits": list(splits),
        "train_episodes": args.train_episodes,
        "val_episodes": args.val_episodes,
        "length": args.length,
        "img_size": args.img_size,
        "train_seed": args.train_seed,
        "val_seed": args.val_seed,
        "smooth_rho": 0.0,
        "action_process": ACTION_PROCESS,
        "primary_evaluation_target": "flattened_native_task_observation",
        "secondary_evaluation_target": "raw_physics_state",
        "evaluation_targets_used_for_training": False,
        "cache_role": "clean_only_corruptions_are_deterministic_dataset_views",
        "corruption_seed": DEFAULT_CORRUPTION_SEED,
    }
    _write_manifest(root, records, protocol)
    print(json.dumps({"root": str(root), "artifacts": records}, indent=2), flush=True)


if __name__ == "__main__":
    main()
