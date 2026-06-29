#!/usr/bin/env python3
"""Deterministic clean-pixel data and corruption views for HACSSM-v10.

The cache contains only clean simulator trajectories.  Training and held-out
corruptions are deterministic dataset views, so every architecture sees exactly
the same pixels, actions, gap positions, and physics-state targets.
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
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


SCHEMA_VERSION = 1
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
DEFAULT_TRAIN_EPISODES = 1_200
DEFAULT_VAL_EPISODES = 240
DEFAULT_LENGTH = 48
DEFAULT_IMG_SIZE = 64
DEFAULT_TRAIN_SEED = 27_100
DEFAULT_VAL_SEED = 92_710
DEFAULT_SMOOTH_RHO = 0.85
DEFAULT_CORRUPTION_SEED = 10_012

CONTENT_FIELDS = (
    "schema_version",
    "env_id",
    "split",
    "seed",
    "length",
    "img_size",
    "smooth_rho",
    "obs",
    "actions",
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


def content_sha256(values: dict[str, Any]) -> str:
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
        raise ValueError(f"unsupported V10 environment {value!r}; expected one of {TASKS}")
    return value


def collect_clean_dmc(env_id: str, episodes: int, length: int, img_size: int,
                      seed: int, smooth_rho: float = DEFAULT_SMOOTH_RHO
                      ) -> dict[str, np.ndarray]:
    """Collect deterministic DMC trajectories with smooth continuous actions."""
    env_id = _canonical_env_id(env_id)
    if episodes < 1 or length < 4 or img_size < 16:
        raise ValueError("episodes, length, and img_size must be positive and nontrivial")
    if not math.isfinite(smooth_rho) or not 0.0 <= smooth_rho < 1.0:
        raise ValueError("smooth_rho must be finite in [0,1)")

    os.environ.setdefault("MUJOCO_GL", "egl")
    from dm_control import suite

    domain, task = env_id.split(".", 1)
    env = suite.load(domain, task, task_kwargs={"random": int(seed)})
    action_spec = env.action_spec()
    action_min = np.broadcast_to(action_spec.minimum, action_spec.shape).astype(np.float32)
    action_max = np.broadcast_to(action_spec.maximum, action_spec.shape).astype(np.float32)
    action_min = np.nan_to_num(action_min, neginf=-1.0)
    action_max = np.nan_to_num(action_max, posinf=1.0)
    if not (np.isfinite(action_min).all() and np.isfinite(action_max).all()
            and np.all(action_max > action_min)):
        raise ValueError(f"invalid action bounds for {env_id}")

    rng = np.random.default_rng(seed)
    center = (action_min + action_max) * 0.5
    half_range = (action_max - action_min) * 0.5
    innovation_scale = math.sqrt(1.0 - smooth_rho * smooth_rho)
    observations, actions, states, rewards = [], [], [], []

    for _episode in range(episodes):
        timestep = env.reset()
        frames = [env.physics.render(img_size, img_size, camera_id=0)]
        episode_states = [np.asarray(env.physics.get_state(), dtype=np.float64).copy()]
        episode_actions, episode_rewards = [], []
        latent_action = rng.standard_normal(action_spec.shape).astype(np.float32)
        for _step in range(length - 1):
            innovation = rng.standard_normal(action_spec.shape).astype(np.float32)
            latent_action = smooth_rho * latent_action + innovation_scale * innovation
            action = center + half_range * np.tanh(latent_action)
            action = np.clip(action, action_min, action_max).astype(np.float32)
            timestep = env.step(action)
            if timestep.last():
                raise RuntimeError(
                    f"{env_id} terminated before the requested length {length}; "
                    "refusing to splice simulator episodes")
            frames.append(env.physics.render(img_size, img_size, camera_id=0))
            episode_states.append(
                np.asarray(env.physics.get_state(), dtype=np.float64).copy())
            episode_actions.append(action)
            episode_rewards.append(0.0 if timestep.reward is None else float(timestep.reward))
        observations.append(np.stack(frames).astype(np.uint8))
        actions.append(np.stack(episode_actions).astype(np.float32))
        states.append(np.stack(episode_states).astype(np.float64))
        rewards.append(np.asarray(episode_rewards, dtype=np.float32))

    return {
        "obs": np.stack(observations),
        "actions": np.stack(actions),
        "physics_state": np.stack(states),
        "rewards": np.stack(rewards),
        "action_min": action_min,
        "action_max": action_max,
    }


def write_cache(path: Path, *, env_id: str, split: str, seed: int,
                length: int, img_size: int, smooth_rho: float,
                arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    path = path.resolve()
    if path.exists() or sidecar_path(path).exists():
        raise FileExistsError(f"refusing to overwrite V10 data cache {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    values: dict[str, Any] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "env_id": np.asarray(_canonical_env_id(env_id)),
        "split": np.asarray(split),
        "seed": np.asarray(seed, dtype=np.int64),
        "length": np.asarray(length, dtype=np.int64),
        "img_size": np.asarray(img_size, dtype=np.int64),
        "smooth_rho": np.asarray(smooth_rho, dtype=np.float64),
        **arrays,
    }
    values["content_sha256"] = np.asarray(content_sha256(values))
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
            raise FileNotFoundError(f"missing V10 cache sidecar {sidecar}")
        file_hash = sha256_file(path)
        expected = f"{file_hash}  {path.name}\n"
        if sidecar.read_text() != expected:
            raise ValueError(f"V10 cache sidecar mismatch for {path}")
    else:
        file_hash = sha256_file(path)
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != REQUIRED_FIELDS:
            raise ValueError(
                f"{path}: fields {sorted(source.files)} != {sorted(REQUIRED_FIELDS)}")
        values = {name: np.array(source[name], copy=True) for name in source.files}
    if int(values["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema version")
    actual_content_hash = content_sha256(values)
    if str(values["content_sha256"]) != actual_content_hash:
        raise ValueError(f"{path}: content hash mismatch")

    obs = values["obs"]
    actions = values["actions"]
    states = values["physics_state"]
    rewards = values["rewards"]
    if obs.dtype != np.uint8 or obs.ndim != 5 or obs.shape[-1] != 3:
        raise ValueError(f"{path}: invalid RGB observations {obs.shape}/{obs.dtype}")
    episodes, length, height, width, _ = obs.shape
    action_dim = actions.shape[-1] if actions.ndim == 3 else -1
    state_dim = states.shape[-1] if states.ndim == 3 else -1
    if (
        actions.shape != (episodes, length - 1, action_dim)
        or states.shape != (episodes, length, state_dim)
        or rewards.shape != (episodes, length - 1)
        or height != width
        or length != int(values["length"])
        or height != int(values["img_size"])
        or action_dim < 1
        or state_dim < 1
    ):
        raise ValueError(f"{path}: inconsistent trajectory shapes")
    for name in ("actions", "physics_state", "rewards", "action_min", "action_max"):
        if not np.isfinite(values[name]).all():
            raise ValueError(f"{path}: non-finite {name}")
    if str(values["split"]) not in {"train", "val"}:
        raise ValueError(f"{path}: invalid split")
    metadata = CacheMetadata(
        path=path,
        env_id=str(values["env_id"]),
        split=str(values["split"]),
        seed=int(values["seed"]),
        length=length,
        img_size=height,
        smooth_rho=float(values["smooth_rho"]),
        episodes=episodes,
        action_dim=action_dim,
        state_dim=state_dim,
        file_sha256=file_hash,
        content_sha256=actual_content_hash,
    )
    return (values, metadata) if return_values else metadata


def _episode_rng(base_seed: int, episode: int, salt: int = 0) -> np.random.Generator:
    sequence = np.random.SeedSequence((int(base_seed), int(episode), int(salt)))
    return np.random.default_rng(sequence)


def corruption_interval(length: int, view: str, seed: int, episode: int,
                        history_len: int = 3) -> tuple[int, int]:
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


class V10TrajectoryDataset(Dataset):
    """Validated clean cache exposed through one deterministic corruption view."""

    def __init__(self, path: str | Path, view: str,
                 corruption_seed: int = DEFAULT_CORRUPTION_SEED,
                 history_len: int = 3, verify: bool = True):
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
        values, metadata = load_cache(path, verify=verify, return_values=True)
        self.obs = values["obs"]
        self.actions = values["actions"].astype(np.float32, copy=False)
        self.physics_state = values["physics_state"].astype(np.float32)
        self.metadata = metadata
        self.view = view
        self.corruption_seed = int(corruption_seed)
        self.history_len = int(history_len)
        # A train-only, data-derived replacement image.  It is constant across episodes and
        # cannot reveal the clean target of the corrupted frame.
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
            "observed": torch.from_numpy(observed.astype(np.float32) / 255.0).permute(0, 3, 1, 2),
            "clean": torch.from_numpy(clean.astype(np.float32) / 255.0).permute(0, 3, 1, 2),
            "actions": torch.from_numpy(self.actions[index]),
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
        expected = (env_id, split, seed, length, img_size, episodes)
        actual = (metadata.env_id, metadata.split, metadata.seed,
                  metadata.length, metadata.img_size, metadata.episodes)
        if actual != expected:
            raise ValueError(f"existing cache contract mismatch: {actual} != {expected}")
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
            raise ValueError(f"existing manifest differs: {manifest_path}")
    else:
        _atomic_text(manifest_path, payload)
    digest = sha256_file(manifest_path)
    wanted = f"{digest}  {manifest_path.name}\n"
    if sidecar.exists() and sidecar.read_text() != wanted:
        raise ValueError(f"existing manifest sidecar differs: {sidecar}")
    if not sidecar.exists():
        _atomic_text(sidecar, wanted)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/hacssm_v10_data")
    parser.add_argument("--all", action="store_true", help="collect all five locked tasks")
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
            print(f"collect/validate {env_id} {split}: n={episodes} L={args.length}", flush=True)
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
        "smooth_rho": args.smooth_rho,
        "action_process": "bounded_tanh_ar1",
        "cache_role": "clean_only_corruptions_are_deterministic_dataset_views",
    }
    _write_manifest(root, records, protocol)
    print(json.dumps({"root": str(root), "artifacts": records}, indent=2), flush=True)


if __name__ == "__main__":
    main()
