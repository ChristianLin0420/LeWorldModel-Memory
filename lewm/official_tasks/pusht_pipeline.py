"""Paths and fail-closed cache loading for the formal PushT pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lewm.official_tasks.artifacts import load_verified_npz
from lewm.official_tasks.pusht_spec import (
    pusht_lock_receipt,
    resolve_pusht_path,
)


PUSHT_SPLITS = ("train", "validation")


def require_pusht_split(split: str) -> str:
    if split not in PUSHT_SPLITS:
        raise ValueError(f"unknown PushT split {split!r}")
    return split


def pusht_task_spec(spec: Mapping[str, Any], task_key: str) -> dict[str, Any]:
    matches = [task for task in spec["semantic_tasks"]
               if task["key"] == task_key]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate semantic task {task_key!r}")
    return dict(matches[0])


def pusht_artifact_root(spec: Mapping[str, Any]) -> Path:
    return resolve_pusht_path(spec["artifacts"]["root"])


def pusht_base_cache_path(spec: Mapping[str, Any], split: str) -> Path:
    require_pusht_split(split)
    return (pusht_artifact_root(spec) / spec["artifacts"]["base_cache"]
            / f"{split}.npz")


def pusht_base_manifest_path(spec: Mapping[str, Any]) -> Path:
    return (pusht_artifact_root(spec) / spec["artifacts"]["base_cache"]
            / "manifest.json")


def pusht_task_cache_path(spec: Mapping[str, Any], task_key: str,
                          split: str) -> Path:
    require_pusht_split(split)
    pusht_task_spec(spec, task_key)
    return (pusht_artifact_root(spec) / spec["artifacts"]["task_cache"]
            / task_key / f"{split}.npz")


def pusht_task_manifest_path(spec: Mapping[str, Any], task_key: str) -> Path:
    pusht_task_spec(spec, task_key)
    return (pusht_artifact_root(spec) / spec["artifacts"]["task_cache"]
            / task_key / "manifest.json")


def pusht_admission_path(spec: Mapping[str, Any], task_key: str) -> Path:
    pusht_task_spec(spec, task_key)
    return (pusht_artifact_root(spec) / spec["artifacts"]["admissions"]
            / f"{task_key}.json")


def pusht_carrier_directory(spec: Mapping[str, Any], task_key: str,
                            arm: str, seed: int) -> Path:
    pusht_task_spec(spec, task_key)
    return (pusht_artifact_root(spec) / spec["artifacts"]["carriers"]
            / task_key / arm / f"seed-{seed}")


def pusht_log_root(spec: Mapping[str, Any]) -> Path:
    return pusht_artifact_root(spec) / spec["artifacts"]["logs"]


def _load_cache(path: Path, schema: str, spec: dict[str, Any],
                split: str, task_key: str | None = None
                ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, sidecar = load_verified_npz(path)
    if sidecar.get("schema") != schema or sidecar.get("split") != split:
        raise ValueError(f"cache metadata contract differs for {path}")
    if sidecar.get("formal_lock") != pusht_lock_receipt(spec):
        raise ValueError(f"cache was produced under a different lock: {path}")
    if task_key is not None and sidecar.get("task_key") != task_key:
        raise ValueError(f"cache task differs for {path}")
    return arrays, sidecar


def load_pusht_base_cache(spec: dict[str, Any], split: str
                          ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, sidecar = _load_cache(
        pusht_base_cache_path(spec, split),
        "official_pusht_base_cache_v1", spec, split)
    required = {
        "z_base", "actions", "state", "proprio", "episode_index",
        "local_start", "global_frame_indices",
    }
    if set(arrays) != required:
        raise ValueError(f"base cache fields differ: {sorted(arrays)}")
    episodes = spec["selection"][split]["episodes"]
    frames = spec["sequence"]["num_frames"]
    if arrays["z_base"].shape != (episodes, frames, 192):
        raise ValueError("base cache latent shape differs from formal spec")
    if arrays["actions"].shape != (episodes, frames - 1, 10):
        raise ValueError("base cache action shape differs from formal spec")
    if arrays["state"].shape != (episodes, frames, 7) \
            or arrays["proprio"].shape != (episodes, frames, 4):
        raise ValueError("base cache state/proprio shape differs from formal spec")
    if len(np.unique(arrays["episode_index"])) != episodes:
        raise ValueError("base cache repeats an episode")
    return arrays, sidecar


def load_pusht_task_cache(spec: dict[str, Any], task_key: str, split: str
                          ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, sidecar = _load_cache(
        pusht_task_cache_path(spec, task_key, split),
        "official_pusht_task_cue_cache_v1", spec, split, task_key)
    required = {"z_cue", "labels", "episode_index", "local_start"}
    if set(arrays) != required:
        raise ValueError(f"task cache fields differ: {sorted(arrays)}")
    episodes = spec["selection"][split]["episodes"]
    cue_length = spec["sequence"]["cue_length"]
    classes = pusht_task_spec(spec, task_key)["classes"]
    if arrays["z_cue"].shape != (episodes, cue_length, 192):
        raise ValueError("task cue latent shape differs from formal spec")
    if arrays["labels"].shape != (episodes,) \
            or not set(np.unique(arrays["labels"])).issubset(range(classes)):
        raise ValueError("task labels differ from formal vocabulary")
    return arrays, sidecar


def aligned_pusht_latents(base: Mapping[str, np.ndarray],
                          task: Mapping[str, np.ndarray],
                          cue_start: int, cue_length: int) -> np.ndarray:
    """Materialize task latents in RAM; base frames are never duplicated on disk."""

    for key in ("episode_index", "local_start"):
        if not np.array_equal(base[key], task[key]):
            raise ValueError(f"base/task selection mismatch in {key}")
    z = np.asarray(base["z_base"], dtype=np.float32).copy()
    cue_end = cue_start + cue_length
    if task["z_cue"].shape != z[:, cue_start:cue_end].shape:
        raise ValueError("cue latent replacement has the wrong shape")
    z[:, cue_start:cue_end] = task["z_cue"]
    return z


__all__ = [
    "PUSHT_SPLITS",
    "aligned_pusht_latents",
    "load_pusht_base_cache",
    "load_pusht_task_cache",
    "pusht_admission_path",
    "pusht_artifact_root",
    "pusht_base_cache_path",
    "pusht_base_manifest_path",
    "pusht_carrier_directory",
    "pusht_log_root",
    "pusht_task_cache_path",
    "pusht_task_manifest_path",
    "pusht_task_spec",
]
