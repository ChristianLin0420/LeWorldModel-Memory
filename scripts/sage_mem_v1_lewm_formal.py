#!/usr/bin/env python3
"""Fresh, parent-disjoint LeWM formal banks for SAGE-Mem v1.

Only the two LeWM cohorts are covered here.  The module is intentionally
isolated from the generic runner and development adapter so that it can be
reviewed before either integration is enabled.

Freshness rules
---------------
* Reacher uses one newly derived simulator seed per formal split.  Those seeds
  must be pairwise distinct and absent from every seed in the three parent
  Reacher registries.
* PushT selects one sequence per eligible HDF episode after excluding the union
  of every parent matched-host, matched-token, and matched-color episode.
  Formal splits are selected sequentially from the remaining set and are
  therefore mutually episode-disjoint.
* Color/location labels are generated only after base selection and are
  written to a chmod-0400 custody vault outside the public bank.  The
  trajectory API never receives the vault path or exposes semantic labels.

This file deliberately does *not* train a carrier, fit a readout, compute
correctness, or execute a controller.  Formal execution needs two phases:
label-free carrier cells first, followed by a post-grid finalizer that may open
the sealed consumer/test labels and fit one shared arm-blind consumer.  Keeping
that boundary out of the one-cell API makes label hiding inspectable instead
of relying on a caller promise.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping
import uuid

import numpy as np
import torch
import yaml

from lewm.official_tasks.matched_memory import (
    NUM_COMBINATIONS,
    balanced_joint_labels,
    render_joint_cue,
    validate_joint_counterfactuals,
)
from lewm.official_tasks.native_sequence_hdf5 import (
    NativeSequenceHDF5,
    SequenceSelection,
    normalize_action_blocks,
)
from scripts.paper_a_matched_color_v1_1_spec import (
    DEFAULT_SHA as PARENT_SHA,
    load_locked_spec as load_parent_spec,
    prior_excluded_episode_indices,
    resolve_input_path,
)
from scripts.sage_mem_v1_spec import AGES, canonical_json


ROOT = Path(__file__).resolve().parents[1]
LEWM_FORMAL_SCHEMA = "sage_mem_v1_lewm_formal_bank_v1"
LEWM_LABEL_CUSTODY_SCHEMA = "sage_mem_v1_lewm_label_custody_v1"
LEWM_COHORTS = ("lewm_reacher_color", "lewm_pusht_color")
FORMAL_SPLITS = ("formal_train", "consumer_train", "formal_test")
FEATURE_SPLITS = ("consumer_train", "formal_test")
POST_GRID_FINALIZER_PHASE = "post-grid-finalizer"
PUSHT_PARENT_BASE_CACHES = (
    "outputs/paper_a_matched_host_v1/cache/pusht/base/train.npz",
    "outputs/paper_a_matched_host_v1/cache/pusht/base/validation.npz",
    "outputs/paper_a_matched_token_v1/cache/pusht/base/train.npz",
    "outputs/paper_a_matched_token_v1/cache/pusht/base/validation.npz",
    "outputs/paper_a_matched_color_v1_1/cache/pusht/base/train.npz",
    "outputs/paper_a_matched_color_v1_1/cache/pusht/base/validation.npz",
)
REACHER_PARENT_BASE_CACHES = tuple(
    value.replace("/pusht/", "/reacher/")
    for value in PUSHT_PARENT_BASE_CACHES
)
PARENT_RNG_PROTOCOLS = (
    "configs/paper_a_matched_host_v1.yaml",
    "configs/paper_a_matched_token_v1.yaml",
    "configs/paper_a_matched_color_v1_1.yaml",
)


class LeWMFormalError(RuntimeError):
    """A fresh-bank or provenance invariant failed."""


@dataclass(frozen=True)
class FormalSeedPlan:
    base_seed: int
    label_seed: int
    loader_seed: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        canonical = np.ascontiguousarray(value)
        digest.update(str(canonical.dtype).encode("ascii"))
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(value) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return _sha256_file(path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return _sha256_file(path)


def formal_seed_plan(spec: Mapping[str, Any], cohort: str
                     ) -> dict[str, FormalSeedPlan]:
    if cohort not in LEWM_COHORTS:
        raise LeWMFormalError(f"unsupported LeWM cohort {cohort!r}")
    registry = spec.get("_seed_registry")
    if not isinstance(registry, Mapping):
        raise LeWMFormalError("SAGE-Mem seed registry is missing")
    result = {}
    for split in FORMAL_SPLITS:
        prefix = f"{cohort}/{split}"
        result[split] = FormalSeedPlan(
            base_seed=int(registry[f"{prefix}/episode_selection"]),
            label_seed=int(registry[f"{prefix}/cue_labels"]),
            loader_seed=int(registry[f"{prefix}/loader"]),
        )
    values = [number for plan in result.values()
              for number in (plan.base_seed, plan.label_seed, plan.loader_seed)]
    if len(set(values)) != len(values) or any(value < 0 for value in values):
        raise LeWMFormalError("formal seed plan is not unique and non-negative")
    return result


def reacher_parent_seed_receipt(parent_spec: Mapping[str, Any]
                                ) -> dict[str, Any]:
    registry = parent_spec.get("_lock", {}).get("implementation", {}).get(
        "reacher_rng_exclusion")
    if not isinstance(registry, Mapping):
        raise LeWMFormalError("parent Reacher RNG registry is missing")
    source = registry.get("registry")
    if not isinstance(source, Mapping):
        raise LeWMFormalError("parent Reacher seed ledger is malformed")
    seeds: list[int] = []
    prior = source.get("prior_screens")
    if not isinstance(prior, Mapping):
        raise LeWMFormalError("parent prior-screen seed ledger is malformed")
    for values in prior.values():
        seeds.extend(map(int, values))
    seeds.extend(map(int, source.get("wave1_1", [])))
    if len(seeds) != 6 or len(set(seeds)) != 6:
        raise LeWMFormalError("expected six distinct parent Reacher seeds")
    canonical = sorted(seeds)
    cache_records = []
    for relative in REACHER_PARENT_BASE_CACHES:
        path = ROOT / relative
        sidecar = path.with_suffix(path.suffix + ".json")
        if not path.is_file() or not sidecar.is_file():
            raise LeWMFormalError(
                f"parent Reacher cache is missing: {relative}")
        with np.load(path, allow_pickle=False) as archive:
            if "episode_index" not in archive.files:
                raise LeWMFormalError(
                    f"parent Reacher cache lacks episode_index: {relative}")
            episode_index = np.asarray(
                archive["episode_index"], dtype=np.int64)
        if episode_index.ndim != 1 \
                or len(np.unique(episode_index)) != len(episode_index):
            raise LeWMFormalError(
                f"parent Reacher episode registry is malformed: {relative}")
        sidecar_value = json.loads(sidecar.read_text())
        if sidecar_value.get("host") != "reacher" \
                or sidecar_value.get("split") not in {"train", "validation"}:
            raise LeWMFormalError(
                f"parent Reacher sidecar identity differs: {relative}")
        cache_records.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
            "sidecar": str(sidecar.relative_to(ROOT)),
            "sidecar_sha256": _sha256_file(sidecar),
            "episodes": len(episode_index),
            "episode_indices_sha256": _sha256_arrays(
                episode_index.astype("<i8")),
            "study": sidecar_value.get("study"),
            "split": sidecar_value.get("split"),
        })
    digest = hashlib.sha256(canonical_json({
        "parent_reacher_seeds": canonical}).encode("utf-8")).hexdigest()
    return {
        "parent_reacher_seeds": canonical,
        "count": len(canonical),
        "sha256": digest,
        "parent_registry_sha256": registry.get("registry_sha256"),
        "cache_records": cache_records,
        "cache_registry_sha256": hashlib.sha256(canonical_json(
            cache_records).encode("utf-8")).hexdigest(),
    }


def pusht_parent_exclusion_receipt(parent_spec: Mapping[str, Any]
                                   ) -> tuple[set[int], dict[str, Any]]:
    """Union all three parent LeWM PushT studies with file identities."""

    locked_prior = set(map(int, prior_excluded_episode_indices(
        parent_spec, "pusht")))
    union: set[int] = set()
    records = []
    for relative in PUSHT_PARENT_BASE_CACHES:
        path = ROOT / relative
        sidecar = path.with_suffix(path.suffix + ".json")
        if not path.is_file() or not sidecar.is_file():
            raise LeWMFormalError(f"parent PushT cache is missing: {relative}")
        with np.load(path, allow_pickle=False) as archive:
            if "episode_index" not in archive.files:
                raise LeWMFormalError(
                    f"parent PushT cache lacks episode_index: {relative}")
            episodes = np.asarray(archive["episode_index"], dtype=np.int64)
        if episodes.ndim != 1 or len(np.unique(episodes)) != len(episodes):
            raise LeWMFormalError(
                f"parent PushT episode registry is malformed: {relative}")
        sidecar_value = json.loads(sidecar.read_text())
        if sidecar_value.get("host") != "pusht" \
                or sidecar_value.get("split") not in {"train", "validation"}:
            raise LeWMFormalError(
                f"parent PushT sidecar identity differs: {relative}")
        values = set(map(int, episodes))
        union.update(values)
        records.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
            "sidecar": str(sidecar.relative_to(ROOT)),
            "sidecar_sha256": _sha256_file(sidecar),
            "episodes": len(values),
            "indices_sha256": _sha256_arrays(
                np.sort(episodes).astype("<i8")),
            "study": sidecar_value.get("study"),
            "split": sidecar_value.get("split"),
        })
    if not locked_prior.issubset(union):
        raise LeWMFormalError(
            "parent locked PushT exclusion is not covered by cache union")
    sorted_union = np.asarray(sorted(union), dtype="<i8")
    return union, {
        "policy": (
            "union of matched-host, matched-token, and matched-color "
            "train/validation episode registries"),
        "records": records,
        "locked_prior_count": len(locked_prior),
        "union_count": len(union),
        "union_indices_sha256": _sha256_arrays(sorted_union),
        "includes_current_development_parent": True,
    }


def forbidden_parent_artifact_receipt(
        spec: Mapping[str, Any], cohort: str) -> dict[str, Any]:
    """Hash every preregistered parent artifact without trusting filenames."""

    if cohort not in LEWM_COHORTS:
        raise LeWMFormalError(f"unsupported LeWM cohort {cohort!r}")
    relative_paths = spec["cohorts"][cohort].get(
        "forbidden_parent_artifacts")
    if not isinstance(relative_paths, list) or not relative_paths:
        raise LeWMFormalError("forbidden parent artifact registry is missing")
    records = []
    for value in relative_paths:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise LeWMFormalError("forbidden parent artifact path is unsafe")
        path = (ROOT / relative).resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError as error:
            raise LeWMFormalError(
                "forbidden parent artifact leaves repository") from error
        if not path.is_file():
            raise LeWMFormalError(
                f"forbidden parent artifact is missing: {value}")
        records.append({
            "path": str(relative),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    return {
        "records": records,
        "registry_sha256": hashlib.sha256(canonical_json(
            records).encode("utf-8")).hexdigest(),
    }


def parent_rng_registry_receipt() -> dict[str, Any]:
    """Hash and enumerate every seed-valued parent protocol field."""

    values: set[int] = set()
    records = []

    def collect(value: Any, *, seed_context: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                collect(item, seed_context=(
                    seed_context or "seed" in str(key).lower()))
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item, seed_context=seed_context)
        elif seed_context and isinstance(value, int) \
                and not isinstance(value, bool):
            values.add(int(value))

    for relative in PARENT_RNG_PROTOCOLS:
        path = ROOT / relative
        sidecar = path.with_suffix(".sha256")
        if not path.is_file() or not sidecar.is_file():
            raise LeWMFormalError(
                f"parent RNG protocol identity is missing: {relative}")
        payload = yaml.safe_load(path.read_text())
        if not isinstance(payload, Mapping):
            raise LeWMFormalError(
                f"parent RNG protocol is malformed: {relative}")
        before = len(values)
        collect(payload)
        records.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
            "sidecar": str(sidecar.relative_to(ROOT)),
            "sidecar_sha256": _sha256_file(sidecar),
            "new_unique_seed_values": len(values) - before,
        })
    sorted_values = sorted(values)
    if len(sorted_values) < 20 or any(value < 0 for value in sorted_values):
        raise LeWMFormalError("parent RNG registry extraction is implausible")
    return {
        "policy": "all integer values under seed-named parent fields",
        "records": records,
        "seed_values": sorted_values,
        "seed_count": len(sorted_values),
        "seed_values_sha256": _sha256_arrays(
            np.asarray(sorted_values, dtype="<i8")),
    }


def select_fresh_hdf_splits(
        episode_lengths: np.ndarray, *, excluded: Iterable[int],
        split_counts: Mapping[str, int], seed_plan: Mapping[str, FormalSeedPlan],
        num_frames: int = 20, frame_skip: int = 5
) -> dict[str, tuple[SequenceSelection, ...]]:
    """Pure deterministic selection used by the real PushT builder and tests."""

    lengths = np.asarray(episode_lengths, dtype=np.int64)
    if lengths.ndim != 1 or not len(lengths) or np.any(lengths <= 0):
        raise LeWMFormalError("HDF episode lengths are malformed")
    if tuple(split_counts) != FORMAL_SPLITS or tuple(seed_plan) != FORMAL_SPLITS:
        raise LeWMFormalError("formal split ordering changed")
    raw_span = (num_frames - 1) * frame_skip + 1
    eligible = set(map(int, np.flatnonzero(lengths >= raw_span)))
    exclusion = set(map(int, excluded))
    if not exclusion.issubset(set(range(len(lengths)))):
        raise LeWMFormalError("PushT exclusion leaves dataset range")
    remaining = np.asarray(sorted(eligible.difference(exclusion)), dtype=np.int64)
    if len(remaining) < sum(map(int, split_counts.values())):
        raise LeWMFormalError("insufficient parent-disjoint PushT episodes")
    result: dict[str, tuple[SequenceSelection, ...]] = {}
    used: set[int] = set()
    for split in FORMAL_SPLITS:
        count = int(split_counts[split])
        if count <= 0:
            raise LeWMFormalError("formal split count must be positive")
        plan = seed_plan[split]
        available = remaining[~np.isin(remaining, np.fromiter(
            used, dtype=np.int64, count=len(used)))] if used else remaining
        chosen = np.random.default_rng(plan.base_seed).permutation(
            available)[:count]
        values = []
        for episode_value in chosen:
            episode = int(episode_value)
            maximum = int(lengths[episode] - raw_span)
            rng = np.random.default_rng(np.random.SeedSequence(
                [plan.loader_seed, episode]))
            values.append(SequenceSelection(
                split=split, episode_index=episode,
                local_start=int(rng.integers(maximum + 1))))
        result[split] = tuple(values)
        used.update(map(int, chosen))
    selected = {item.episode_index for values in result.values()
                for item in values}
    if selected.intersection(exclusion) \
            or len(selected) != sum(map(int, split_counts.values())):
        raise LeWMFormalError("PushT formal selection is not disjoint")
    return result


def _selection_record(
        split: str, plan: FormalSeedPlan,
        episode_index: np.ndarray, local_start: np.ndarray
) -> dict[str, Any]:
    episodes = np.asarray(episode_index, dtype=np.int64)
    starts = np.asarray(local_start, dtype=np.int64)
    if episodes.shape != starts.shape or episodes.ndim != 1 \
            or len(np.unique(episodes)) != len(episodes):
        raise LeWMFormalError(f"{split} selection arrays are malformed")
    return {
        "count": int(len(episodes)),
        "base_seed": plan.base_seed,
        "label_seed": plan.label_seed,
        "loader_seed": plan.loader_seed,
        "selection_sha256": _sha256_arrays(
            episodes.astype("<i8"), starts.astype("<i8")),
        "episode_indices_sha256": _sha256_arrays(
            np.sort(episodes).astype("<i8")),
    }


def _split_arrays(
        *, model: torch.nn.Module, device: torch.device,
        frames: np.ndarray, raw_actions: np.ndarray, state: np.ndarray,
        episode_index: np.ndarray, local_start: np.ndarray,
        global_frame_indices: np.ndarray, plan: FormalSeedPlan,
        action_mean: np.ndarray, action_std: np.ndarray,
        generated_actions: bool, frame_batch: int
) -> dict[str, np.ndarray]:
    from scripts.prepare_paper_a_matched_host import (
        _encode_stream, _normalize_generated,
    )

    frames = np.asarray(frames)
    episodes, length = frames.shape[:2]
    if length != 20 or episodes % NUM_COMBINATIONS:
        raise LeWMFormalError("formal split must be 16-way balanced at length 20")
    labels = balanced_joint_labels(episodes, plan.label_seed)

    def base_stream():
        for row in range(episodes):
            for step in range(length):
                yield row * length + step, frames[row, step]

    z_base = _encode_stream(
        model, base_stream(), episodes * length, frame_batch,
        device).reshape(episodes, length, 192)
    actions = (_normalize_generated(raw_actions, action_mean, action_std)
               if generated_actions else
               normalize_action_blocks(raw_actions, action_mean, action_std))
    result: dict[str, np.ndarray] = {
        "z_base": z_base.astype(np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "state": np.asarray(state, dtype=np.float32),
        "episode_index": np.asarray(episode_index, dtype=np.int64),
        "local_start": np.asarray(local_start, dtype=np.int64),
        "global_frame_indices": np.asarray(
            global_frame_indices, dtype=np.int64),
        "color_label": labels.color,
        "location_label": labels.location,
        "combination_label": labels.combination,
    }
    for age in AGES:
        cue_off = 19 - int(age)
        cue_on = cue_off - 3
        validate_joint_counterfactuals(frames[0], cue_on, 3)

        def cue_stream():
            for row in range(episodes):
                rendered = render_joint_cue(
                    frames[row], int(labels.color[row]),
                    int(labels.location[row]), cue_on, 3)
                for offset in range(3):
                    yield row * 3 + offset, rendered[cue_on + offset]

        result[f"z_cue_age_{age}"] = _encode_stream(
            model, cue_stream(), episodes * 3, frame_batch,
            device).reshape(episodes, 3, 192).astype(np.float32)
        result[f"cue_on_age_{age}"] = np.full(
            episodes, cue_on, dtype=np.int64)
        result[f"cue_off_age_{age}"] = np.full(
            episodes, cue_off, dtype=np.int64)
    validate_formal_split_arrays(result, expected_count=episodes)
    return result


def trajectory_split_keys() -> set[str]:
    """Array names visible to label-free carrier cells."""

    base = {
        "z_base", "actions", "state", "episode_index", "local_start",
        "global_frame_indices",
    }
    for age in AGES:
        base.update({f"z_cue_age_{age}", f"cue_on_age_{age}",
                     f"cue_off_age_{age}"})
    return base


def label_split_keys() -> set[str]:
    """Array names reserved for the post-grid finalizer."""

    return {
        "episode_index", "color_label", "location_label",
        "combination_label",
    }


def formal_split_keys() -> set[str]:
    return trajectory_split_keys().union(label_split_keys())


def partition_formal_split_arrays(
        arrays: Mapping[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Separate carrier-visible trajectories from sealed semantic labels."""

    validate_formal_split_arrays(
        arrays, expected_count=len(np.asarray(arrays["episode_index"])))
    trajectories = {
        name: np.asarray(arrays[name]) for name in trajectory_split_keys()}
    labels = {name: np.asarray(arrays[name]) for name in label_split_keys()}
    if trajectory_split_keys().intersection(
            set(labels).difference({"episode_index"})):
        raise LeWMFormalError("semantic label leaked into trajectory artifact")
    return trajectories, labels


def validate_trajectory_split_arrays(
        arrays: Mapping[str, np.ndarray], *, expected_count: int) -> None:
    if set(arrays) != trajectory_split_keys():
        raise LeWMFormalError("formal trajectory schema changed")
    count = int(expected_count)
    if count <= 0 or count % NUM_COMBINATIONS:
        raise LeWMFormalError("formal count must be a positive multiple of 16")
    expected_shapes = {
        "z_base": (count, 20, 192), "actions": (count, 19, 10),
        "episode_index": (count,), "local_start": (count,),
        "global_frame_indices": (count, 20),
    }
    for age in AGES:
        expected_shapes[f"z_cue_age_{age}"] = (count, 3, 192)
        expected_shapes[f"cue_on_age_{age}"] = (count,)
        expected_shapes[f"cue_off_age_{age}"] = (count,)
    for name, shape in expected_shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise LeWMFormalError(f"formal trajectory is invalid: {name}")
    state = np.asarray(arrays["state"])
    if state.ndim != 3 or state.shape[:2] != (count, 20) \
            or not np.isfinite(state).all():
        raise LeWMFormalError("formal state trajectory is invalid")
    episodes = np.asarray(arrays["episode_index"], dtype=np.int64)
    if len(np.unique(episodes)) != count:
        raise LeWMFormalError("formal trajectory reuses a native episode")
    for age in AGES:
        cue_on, cue_off = 19 - int(age) - 3, 19 - int(age)
        if not np.all(arrays[f"cue_on_age_{age}"] == cue_on) \
                or not np.all(arrays[f"cue_off_age_{age}"] == cue_off):
            raise LeWMFormalError("formal cue timing changed")


def validate_label_split_arrays(
        arrays: Mapping[str, np.ndarray], *, expected_count: int,
        expected_episode_index: np.ndarray | None = None) -> None:
    if set(arrays) != label_split_keys():
        raise LeWMFormalError("formal sealed-label schema changed")
    count = int(expected_count)
    if count <= 0 or count % NUM_COMBINATIONS:
        raise LeWMFormalError("formal label count must be a multiple of 16")
    for name in label_split_keys():
        value = np.asarray(arrays[name])
        if value.shape != (count,) or not np.isfinite(value).all():
            raise LeWMFormalError(f"formal sealed label is invalid: {name}")
    episodes = np.asarray(arrays["episode_index"], dtype=np.int64)
    if len(np.unique(episodes)) != count:
        raise LeWMFormalError("formal sealed labels reuse an episode")
    if expected_episode_index is not None and not np.array_equal(
            episodes, np.asarray(expected_episode_index, dtype=np.int64)):
        raise LeWMFormalError("sealed labels do not align with trajectories")
    color = np.asarray(arrays["color_label"], dtype=np.int64)
    location = np.asarray(arrays["location_label"], dtype=np.int64)
    joint = np.asarray(arrays["combination_label"], dtype=np.int64)
    if not np.array_equal(joint, color * 4 + location) \
            or not np.array_equal(
                np.bincount(joint, minlength=16),
                np.full(16, count // 16)):
        raise LeWMFormalError("formal labels are not exactly 16-way balanced")


def validate_formal_split_arrays(
        arrays: Mapping[str, np.ndarray], *, expected_count: int) -> None:
    if set(arrays) != formal_split_keys():
        raise LeWMFormalError("formal LeWM split schema changed")
    trajectories, labels = (
        {name: np.asarray(arrays[name]) for name in trajectory_split_keys()},
        {name: np.asarray(arrays[name]) for name in label_split_keys()},
    )
    validate_trajectory_split_arrays(
        trajectories, expected_count=expected_count)
    validate_label_split_arrays(
        labels, expected_count=expected_count,
        expected_episode_index=trajectories["episode_index"])


def _write_split_artifacts(
        staging: Path, split: str, plan: FormalSeedPlan,
        arrays: Mapping[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    trajectories, labels = partition_formal_split_arrays(arrays)
    count = len(trajectories["episode_index"])
    trajectory_path = staging / f"{split}.trajectories.npz"
    trajectory_sha = _atomic_npz(trajectory_path, trajectories)
    if _sha256_file(trajectory_path) != trajectory_sha:
        raise LeWMFormalError("formal split changed immediately after writing")
    record = _selection_record(
        split, plan, trajectories["episode_index"],
        trajectories["local_start"])
    record.update({
        "trajectory_artifact": {
            "path": trajectory_path.name,
            "sha256": trajectory_sha,
            "size": trajectory_path.stat().st_size,
            "contains_semantic_labels": False,
        },
        "sealed_label_arrays_sha256": _sha256_arrays(*(
            np.asarray(labels[name]) for name in sorted(labels))),
        "sealed_label_episode_indices_sha256": _sha256_arrays(
            np.asarray(labels["episode_index"], dtype="<i8")),
        "trajectory_label_episode_alignment_sha256": _sha256_arrays(
            np.asarray(trajectories["episode_index"], dtype="<i8"),
            np.asarray(labels["episode_index"], dtype="<i8")),
    })
    if record["count"] != count:
        raise LeWMFormalError("formal split count changed while publishing")
    return record, labels


def _label_vault_arrays(
        labels_by_split: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    if set(labels_by_split) != set(FORMAL_SPLITS):
        raise LeWMFormalError("sealed-label split registry changed")
    result = {}
    for split in FORMAL_SPLITS:
        labels = labels_by_split[split]
        validate_label_split_arrays(
            labels, expected_count=len(labels["episode_index"]))
        for name in sorted(label_split_keys()):
            result[f"{split}__{name}"] = np.asarray(labels[name])
    return result


def _labels_from_vault_arrays(
        arrays: Mapping[str, np.ndarray], *,
        split_counts: Mapping[str, int]) -> dict[str, dict[str, np.ndarray]]:
    expected = {
        f"{split}__{name}"
        for split in FORMAL_SPLITS for name in label_split_keys()
    }
    if set(arrays) != expected:
        raise LeWMFormalError("sealed-label vault schema changed")
    result = {}
    for split in FORMAL_SPLITS:
        labels = {
            name: np.asarray(arrays[f"{split}__{name}"])
            for name in label_split_keys()
        }
        validate_label_split_arrays(
            labels, expected_count=int(split_counts[split]))
        result[split] = labels
    return result


def prepare_lewm_formal_bank(
        *, cohort: str, spec: Mapping[str, Any], output_directory: Path,
        label_vault_path: Path, label_custody_receipt_path: Path,
        device_name: str = "cuda:0") -> dict[str, Any]:
    """Generate and atomically publish one fresh LeWM formal bank."""

    if cohort not in LEWM_COHORTS:
        raise LeWMFormalError(f"unsupported LeWM cohort {cohort!r}")
    if device_name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0" \
            or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise LeWMFormalError(
            "LeWM formal bank generation requires physical GPU 0 only")
    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError(f"formal bank already exists: {destination}")
    vault = Path(label_vault_path).resolve()
    custody_path = Path(label_custody_receipt_path).resolve()
    if vault.exists() or custody_path.exists():
        raise FileExistsError("formal label vault or custody receipt exists")
    if vault == custody_path:
        raise LeWMFormalError("label vault and custody receipt must differ")
    for protected, label in ((vault, "label vault"),
                             (custody_path, "label custody receipt")):
        try:
            protected.relative_to(destination.resolve())
        except ValueError:
            pass
        else:
            raise LeWMFormalError(f"{label} must live outside formal bank")
        protected.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (
        f".{destination.name}.partial-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(mode=0o750)
    started = time.time()
    published_vault = False
    published_custody = False
    published_bank = False
    try:
        study_path = Path(spec.get("_spec_path", "")).resolve()
        try:
            study_relative = study_path.relative_to(ROOT.resolve())
        except (ValueError, OSError) as error:
            raise LeWMFormalError(
                "loaded SAGE-Mem protocol path is not repository-local") from error
        if not study_path.is_file() or spec.get("_spec_sha256") != \
                _sha256_file(study_path):
            raise LeWMFormalError("loaded SAGE-Mem protocol identity changed")
        parent_path = ROOT / spec["cohorts"][cohort]["parent_protocol"]
        parent = load_parent_spec(parent_path, PARENT_SHA, verify_inputs=False)
        plans = formal_seed_plan(spec, cohort)
        formal_rng_seeds = sorted({
            value for plan in plans.values()
            for value in (plan.base_seed, plan.label_seed, plan.loader_seed)
        })
        parent_rng = parent_rng_registry_receipt()
        parent_rng_overlap = sorted(set(formal_rng_seeds).intersection(
            parent_rng["seed_values"]))
        if parent_rng_overlap:
            raise LeWMFormalError(
                "formal RNG registry overlaps a parent protocol")
        counts = {
            split: int(spec["cohorts"][cohort]["split_episodes"][split])
            for split in FORMAL_SPLITS
        }
        if any(count % 16 for count in counts.values()):
            raise LeWMFormalError("formal split counts are not 16-way balanced")
        from scripts.paper_a_evidence_age import configure_determinism
        from scripts.prepare_paper_a_matched_host import (
            _load_host, _raw_action_statistics,
        )
        from scripts.train_frozen_official_swap import state_digest

        configure_determinism(0)
        device = torch.device(device_name)
        parent_host = "reacher" if cohort == "lewm_reacher_color" else "pusht"
        model = _load_host(parent, parent_host, device)
        host_before = state_digest(model)
        frame_batch = int(parent["cache"]["frame_batch_size"])
        splits: dict[str, Any] = {}
        labels_by_split: dict[str, dict[str, np.ndarray]] = {}
        selected_by_split: dict[str, set[int]] = {}
        forbidden_receipt = forbidden_parent_artifact_receipt(spec, cohort)

        if parent_host == "reacher":
            from scripts.make_official_lewm_memory_data import collect_base
            parent_seeds = reacher_parent_seed_receipt(parent)
            formal_base_seeds = [plans[split].base_seed for split in FORMAL_SPLITS]
            if set(formal_rng_seeds).intersection(
                    parent_seeds["parent_reacher_seeds"]):
                raise LeWMFormalError("formal Reacher RNG overlaps a parent")
            raw_train = collect_base(
                counts["formal_train"], 20,
                plans["formal_train"].base_seed)
            action_mean, action_std, action_count = _raw_action_statistics(
                raw_train[1])
            for split in FORMAL_SPLITS:
                raw = (raw_train if split == "formal_train" else collect_base(
                    counts[split], 20, plans[split].base_seed))
                number = counts[split]
                # A generated Reacher episode is identified by the simulator
                # seed and its ordinal within that deterministic collection.
                episode = (np.int64(plans[split].base_seed) * 10_000
                           + np.arange(number, dtype=np.int64))
                local = np.zeros(number, dtype=np.int64)
                global_frames = np.arange(
                    number * 20, dtype=np.int64).reshape(number, 20)
                arrays = _split_arrays(
                    model=model, device=device, frames=raw[0],
                    raw_actions=raw[1], state=raw[2], episode_index=episode,
                    local_start=local, global_frame_indices=global_frames,
                    plan=plans[split], action_mean=action_mean,
                    action_std=action_std, generated_actions=True,
                    frame_batch=frame_batch)
                splits[split], labels_by_split[split] = \
                    _write_split_artifacts(
                        staging, split, plans[split], arrays)
                selected_by_split[split] = set(map(int, episode))
                if split == "formal_train":
                    del raw_train
                del raw, arrays
            freshness = {
                "kind": "seed-disjoint fresh dm_control trajectories",
                "parent": parent_seeds,
                "formal_base_seeds": formal_base_seeds,
                "formal_rng_seeds": formal_rng_seeds,
                "formal_parent_rng_overlap": [],
                "parent_rng_registry": parent_rng,
                "all_parent_rng_overlap": parent_rng_overlap,
                "forbidden_parent_artifacts": forbidden_receipt,
                "zero_overlap_proven": True,
            }
        else:
            source = parent["inputs"]["pusht"]
            dataset_record = source["dataset"]
            dataset = NativeSequenceHDF5(
                resolve_input_path(dataset_record),
                expected_sha256=dataset_record["sha256"],
                expected_size=int(dataset_record["size"]),
                state_key=source["state_key"])
            excluded, exclusion_receipt = pusht_parent_exclusion_receipt(parent)
            selections = select_fresh_hdf_splits(
                dataset.episode_lengths, excluded=excluded,
                split_counts=counts, seed_plan=plans)
            action_mean, action_std, action_count = (
                dataset.raw_action_statistics(ddof=1))
            for split in FORMAL_SPLITS:
                selected = selections[split]
                number = len(selected)
                frames = np.empty(
                    (number, 20, *dataset.pixel_shape), dtype=np.uint8)
                raw_actions = np.empty((number, 19, 10), dtype=np.float32)
                state = np.empty(
                    (number, 20, dataset.state_dim), dtype=np.float32)
                global_frames = np.empty((number, 20), dtype=np.int64)
                for row, native in enumerate(
                        dataset.read_sequences(selected, 20)):
                    frames[row] = native.frames
                    raw_actions[row] = native.actions
                    state[row] = native.state
                    global_frames[row] = native.global_frame_indices
                episode = np.asarray(
                    [item.episode_index for item in selected], dtype=np.int64)
                local = np.asarray(
                    [item.local_start for item in selected], dtype=np.int64)
                arrays = _split_arrays(
                    model=model, device=device, frames=frames,
                    raw_actions=raw_actions, state=state,
                    episode_index=episode, local_start=local,
                    global_frame_indices=global_frames, plan=plans[split],
                    action_mean=action_mean, action_std=action_std,
                    generated_actions=False, frame_batch=frame_batch)
                splits[split], labels_by_split[split] = \
                    _write_split_artifacts(
                        staging, split, plans[split], arrays)
                selected_by_split[split] = set(map(int, episode))
                del frames, raw_actions, state, global_frames, arrays
            all_selected = {
                item.episode_index for values in selections.values()
                for item in values}
            freshness = {
                "kind": "episode-disjoint official HDF sequences",
                "parent": exclusion_receipt,
                "formal_rng_seeds": formal_rng_seeds,
                "parent_rng_registry": parent_rng,
                "all_parent_rng_overlap": parent_rng_overlap,
                "selected_episode_count": len(all_selected),
                "selected_parent_overlap": sorted(all_selected.intersection(
                    excluded)),
                "forbidden_parent_artifacts": forbidden_receipt,
                "zero_overlap_proven": not bool(
                    all_selected.intersection(excluded)),
            }

        host_after = state_digest(model)
        if host_before != host_after:
            raise LeWMFormalError("formal-bank encoding mutated frozen host")
        split_hashes = {record["selection_sha256"]
                        for record in splits.values()}
        if len(split_hashes) != len(FORMAL_SPLITS):
            raise LeWMFormalError("formal split selection hashes collide")
        pairwise_overlap = {
            f"{left}::{right}": sorted(
                selected_by_split[left].intersection(selected_by_split[right]))
            for index, left in enumerate(FORMAL_SPLITS)
            for right in FORMAL_SPLITS[index + 1:]
        }
        if any(pairwise_overlap.values()) or not freshness["zero_overlap_proven"]:
            raise LeWMFormalError("formal-parent or formal-split overlap detected")
        vault_arrays = _label_vault_arrays(labels_by_split)
        vault_sha256 = _atomic_npz(vault, vault_arrays)
        published_vault = True
        os.chmod(vault, 0o400)
        for split in FORMAL_SPLITS:
            expected_label_hash = _sha256_arrays(*(
                np.asarray(labels_by_split[split][name])
                for name in sorted(labels_by_split[split])))
            if splits[split]["sealed_label_arrays_sha256"] != \
                    expected_label_hash:
                raise LeWMFormalError(
                    f"sealed-label split identity changed: {split}")
        custody = {
            "schema": LEWM_LABEL_CUSTODY_SCHEMA,
            "study": "sage-mem-v1",
            "status": "sealed-for-post-grid-finalizer",
            "cohort": cohort,
            "path": str(vault),
            "size": vault.stat().st_size,
            "sha256": vault_sha256,
            "mode": oct(vault.stat().st_mode & 0o777),
            "study_protocol_sha256": _sha256_file(study_path),
            "split_label_hashes": {
                split: splits[split]["sealed_label_arrays_sha256"]
                for split in FORMAL_SPLITS
            },
            "per_cell_api_access": False,
            "available_only_to": POST_GRID_FINALIZER_PHASE,
        }
        _atomic_json(custody_path, custody)
        published_custody = True
        os.chmod(custody_path, 0o400)
        artifact_hashes = {
            f"{split}/trajectory_artifact": splits[split][
                "trajectory_artifact"]["sha256"]
            for split in FORMAL_SPLITS
        }
        if len(set(artifact_hashes.values())) != len(artifact_hashes):
            raise LeWMFormalError("formal artifact hashes collide")
        manifest = {
            "schema_version": 1,
            "schema": LEWM_FORMAL_SCHEMA,
            "study": "sage-mem-v1",
            "status": "prepared",
            "cohort": cohort,
            "formal_only": True,
            "parent_disjoint": True,
            "development_formal_disjoint": True,
            "selection_uses_semantic_labels": False,
            "labels_generated_after_selection": True,
            "labels_used_for_carrier_training": False,
            "trajectory_artifacts_contain_semantic_labels": False,
            "sealed_labels_available_only_to": POST_GRID_FINALIZER_PHASE,
            "semantic_label_vault_inside_bank": False,
            "sealed_label_vault_sha256": vault_sha256,
            "ages": list(AGES),
            "splits": splits,
            "freshness": freshness,
            "formal_split_overlap": pairwise_overlap,
            "artifact_hashes": artifact_hashes,
            "action_normalization": {
                "source": ("formal_train generated actions" if parent_host ==
                           "reacher" else "complete official HDF action column"),
                "mean": np.asarray(action_mean).tolist(),
                "std_ddof1": np.asarray(action_std).tolist(),
                "count": np.asarray(action_count).tolist(),
            },
            "study_protocol": str(study_relative),
            "study_protocol_sha256": _sha256_file(study_path),
            "parent_protocol": spec["cohorts"][cohort]["parent_protocol"],
            "parent_protocol_sha256": _sha256_file(parent_path),
            "host_hash_before": host_before,
            "host_hash_after": host_after,
            "host_unchanged": True,
            "admissions": {
                "registered_split_counts_exact": True,
                "exact_joint_label_balance": True,
                "parent_overlap_zero": True,
                "formal_split_overlap_zero": True,
                "trajectory_label_artifacts_separated": True,
                "trajectory_artifacts_label_free": True,
                "host_digest_unchanged": True,
                "artifact_hashes_reverified": True,
            },
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(staging / "manifest.json", manifest)
        os.rename(staging, destination)
        published_bank = True
        return {**manifest, "manifest": {
            "path": str((destination / "manifest.json").resolve()),
            "sha256": _sha256_file(destination / "manifest.json"),
        }}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if published_bank:
            shutil.rmtree(destination, ignore_errors=True)
        if published_custody:
            try:
                os.chmod(custody_path, 0o600)
                custody_path.unlink()
            except FileNotFoundError:
                pass
        if published_vault:
            try:
                os.chmod(vault, 0o600)
                vault.unlink()
            except FileNotFoundError:
                pass
        raise


class FormalLeWMTrajectoryBank:
    """Verified, label-free latent trajectories with lazy cue insertion."""

    spatial = False

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        values = {name: np.asarray(value) for name, value in arrays.items()}
        count = len(values.get("episode_index", ()))
        validate_trajectory_split_arrays(values, expected_count=count)
        self._values = values
        self.count = count
        self.fit_indices = np.arange(count, dtype=np.int64)
        self.episode_ids = np.asarray(
            values["episode_index"], dtype=np.int64).copy()

    @property
    def array_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._values))

    def features(self, age: int, indices: np.ndarray) -> np.ndarray:
        age = int(age)
        if age not in AGES:
            raise LeWMFormalError(f"unregistered evidence age {age}")
        local = np.asarray(indices, dtype=np.int64)
        result = np.asarray(
            self._values["z_base"][local], dtype=np.float32).copy()
        cue_on = 19 - age - 3
        result[:, cue_on:cue_on + 3] = self._values[
            f"z_cue_age_{age}"][local]
        return result

    def actions(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._values["actions"][indices], dtype=np.float32).copy()

    def native_state(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._values["state"][indices], dtype=np.float32).copy()

    def trajectory_identity(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        local = np.asarray(indices, dtype=np.int64)
        return {
            "episode_index": np.asarray(
                self._values["episode_index"][local], dtype=np.int64).copy(),
            "local_start": np.asarray(
                self._values["local_start"][local], dtype=np.int64).copy(),
            "global_frame_indices": np.asarray(
                self._values["global_frame_indices"][local],
                dtype=np.int64).copy(),
        }

    def proprio(self, indices: np.ndarray) -> None:
        del indices
        return None


@dataclass(frozen=True)
class SealedLeWMLabels:
    """Labels available only to a separately invoked post-grid finalizer."""

    episode_ids: np.ndarray
    color: np.ndarray
    location: np.ndarray
    combination: np.ndarray


def _safe_artifact_path(root: Path, record: Mapping[str, Any],
                        *, label: str) -> Path:
    relative = Path(record.get("path", ""))
    if relative.is_absolute() or ".." in relative.parts \
            or len(relative.parts) != 1:
        raise LeWMFormalError(f"{label} artifact path is unsafe")
    path = root / relative
    if not path.is_file() or path.is_symlink() \
            or path.stat().st_size != record.get("size") \
            or _sha256_file(path) != record.get("sha256"):
        raise LeWMFormalError(f"{label} artifact identity failed")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise LeWMFormalError(f"{label} artifact leaves bank") from error
    return path


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def validate_lewm_formal_manifest(
        manifest_path: Path, *, expected_cohort: str | None = None,
        expected_counts: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Rehash every artifact and replay all immutable bank admissions."""

    raw_manifest_path = Path(manifest_path)
    if not raw_manifest_path.is_file() or raw_manifest_path.is_symlink():
        raise LeWMFormalError("LeWM formal manifest is missing or a symlink")
    path = raw_manifest_path.resolve()
    manifest = json.loads(path.read_text())
    required_admissions = {
        "registered_split_counts_exact", "exact_joint_label_balance",
        "parent_overlap_zero", "formal_split_overlap_zero",
        "trajectory_label_artifacts_separated",
        "trajectory_artifacts_label_free", "host_digest_unchanged",
        "artifact_hashes_reverified",
    }
    admissions = manifest.get("admissions")
    if manifest.get("schema") != LEWM_FORMAL_SCHEMA \
            or manifest.get("status") != "prepared" \
            or manifest.get("cohort") not in LEWM_COHORTS \
            or (expected_cohort is not None
                and manifest.get("cohort") != expected_cohort) \
            or manifest.get("formal_only") is not True \
            or manifest.get("parent_disjoint") is not True \
            or manifest.get("development_formal_disjoint") is not True \
            or manifest.get("selection_uses_semantic_labels") is not False \
            or manifest.get("labels_used_for_carrier_training") is not False \
            or manifest.get(
                "trajectory_artifacts_contain_semantic_labels") is not False \
            or manifest.get("sealed_labels_available_only_to") != \
                POST_GRID_FINALIZER_PHASE \
            or manifest.get("semantic_label_vault_inside_bank") is not False \
            or not _is_sha256(manifest.get("sealed_label_vault_sha256")) \
            or manifest.get("host_hash_before") != manifest.get(
                "host_hash_after") \
            or not _is_sha256(manifest.get("host_hash_before")) \
            or manifest.get("host_unchanged") is not True \
            or not isinstance(admissions, Mapping) \
            or set(admissions) != required_admissions \
            or any(value is not True for value in admissions.values()):
        raise LeWMFormalError("LeWM formal manifest identity is invalid")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(FORMAL_SPLITS):
        raise LeWMFormalError("LeWM formal split registry changed")
    if expected_counts is not None \
            and set(expected_counts) != set(FORMAL_SPLITS):
        raise LeWMFormalError("expected formal count registry changed")

    trajectory_sets: dict[str, set[int]] = {}
    trajectory_ids: dict[str, np.ndarray] = {}
    observed_hashes: dict[str, str] = {}
    for split in FORMAL_SPLITS:
        record = splits[split]
        count = int(record.get("count", -1))
        if expected_counts is not None \
                and count != int(expected_counts[split]):
            raise LeWMFormalError(f"formal split count differs: {split}")
        trajectory_record = record.get("trajectory_artifact")
        if not isinstance(trajectory_record, Mapping) \
                or trajectory_record.get(
                    "contains_semantic_labels") is not False \
                or not _is_sha256(record.get(
                    "sealed_label_arrays_sha256")) \
                or not _is_sha256(record.get(
                    "sealed_label_episode_indices_sha256")):
            raise LeWMFormalError(f"formal artifact boundary failed: {split}")
        trajectory_path = _safe_artifact_path(
            path.parent, trajectory_record, label=f"{split} trajectory")
        trajectories = _load_npz(trajectory_path)
        validate_trajectory_split_arrays(
            trajectories, expected_count=count)
        if record.get("sealed_label_episode_indices_sha256") != \
                _sha256_arrays(np.asarray(
                    trajectories["episode_index"], dtype="<i8")) \
                or record.get(
                    "trajectory_label_episode_alignment_sha256") != \
                _sha256_arrays(
                    np.asarray(trajectories["episode_index"], dtype="<i8"),
                    np.asarray(trajectories["episode_index"], dtype="<i8")):
            raise LeWMFormalError(
                f"sealed-label opaque episode binding differs: {split}")
        selection = _selection_record(
            split, FormalSeedPlan(
                int(record["base_seed"]), int(record["label_seed"]),
                int(record["loader_seed"])),
            trajectories["episode_index"], trajectories["local_start"])
        if selection["selection_sha256"] != record.get("selection_sha256") \
                or selection["episode_indices_sha256"] != record.get(
                    "episode_indices_sha256"):
            raise LeWMFormalError(f"formal selection identity differs: {split}")
        trajectory_sets[split] = set(map(
            int, trajectories["episode_index"]))
        trajectory_ids[split] = np.asarray(
            trajectories["episode_index"], dtype=np.int64)
        observed_hashes[f"{split}/trajectory_artifact"] = trajectory_record[
            "sha256"]

    observed_overlap = {
        f"{left}::{right}": sorted(
            trajectory_sets[left].intersection(trajectory_sets[right]))
        for index, left in enumerate(FORMAL_SPLITS)
        for right in FORMAL_SPLITS[index + 1:]
    }
    if any(observed_overlap.values()) \
            or manifest.get("formal_split_overlap") != observed_overlap:
        raise LeWMFormalError("formal LeWM splits reuse an episode")
    if manifest.get("artifact_hashes") != observed_hashes:
        raise LeWMFormalError("formal artifact hash ledger differs")
    expected_files = {
        "manifest.json",
        *(manifest["splits"][split]["trajectory_artifact"]["path"]
          for split in FORMAL_SPLITS),
    }
    observed_files = {item.name for item in path.parent.iterdir()}
    if observed_files != expected_files:
        raise LeWMFormalError(
            "label-free bank contains unexpected files: "
            f"{sorted(observed_files.difference(expected_files))}")
    freshness = manifest.get("freshness")
    if not isinstance(freshness, Mapping) \
            or freshness.get("zero_overlap_proven") is not True:
        raise LeWMFormalError("parent-disjoint admission is absent")

    # Reopen the exact registered protocols and replay the exclusion proof.
    # This turns the overlap fields into checked receipts rather than booleans
    # that a caller could assert without evidence.
    study_relative = Path(manifest.get("study_protocol", ""))
    parent_relative = Path(manifest.get("parent_protocol", ""))
    for relative, label in (
            (study_relative, "study"), (parent_relative, "parent")):
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise LeWMFormalError(f"{label} protocol path is unsafe")
    study_path = (ROOT / study_relative).resolve()
    parent_path = (ROOT / parent_relative).resolve()
    try:
        study_path.relative_to(ROOT.resolve())
        parent_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise LeWMFormalError("protocol path leaves repository") from error
    if not study_path.is_file() or _sha256_file(study_path) != manifest.get(
            "study_protocol_sha256") \
            or not parent_path.is_file() \
            or _sha256_file(parent_path) != manifest.get(
                "parent_protocol_sha256"):
        raise LeWMFormalError("registered protocol identity changed")
    from scripts.sage_mem_v1_spec import load_spec
    study = load_spec(study_path, verify_parent_paths=False)
    if study.get("_spec_sha256") != manifest.get("study_protocol_sha256"):
        raise LeWMFormalError("registered SAGE-Mem protocol hash differs")
    if study["cohorts"][manifest["cohort"]]["parent_protocol"] != str(
            parent_relative):
        raise LeWMFormalError("cohort parent protocol differs")
    registered_counts = {
        split: int(study["cohorts"][manifest["cohort"]][
            "split_episodes"][split])
        for split in FORMAL_SPLITS
    }
    if any(int(splits[split]["count"]) != registered_counts[split]
           for split in FORMAL_SPLITS):
        raise LeWMFormalError("formal split count differs from protocol")
    if expected_counts is not None and any(
            int(expected_counts[split]) != registered_counts[split]
            for split in FORMAL_SPLITS):
        raise LeWMFormalError("expected split count differs from protocol")
    plans = formal_seed_plan(study, manifest["cohort"])
    for split in FORMAL_SPLITS:
        record = splits[split]
        expected_plan = plans[split]
        if (int(record["base_seed"]), int(record["label_seed"]),
                int(record["loader_seed"])) != (
                    expected_plan.base_seed, expected_plan.label_seed,
                    expected_plan.loader_seed):
            raise LeWMFormalError(f"registered seed differs: {split}")
    expected_parent_rng = parent_rng_registry_receipt()
    formal_rng = sorted({
        value for plan in plans.values()
        for value in (plan.base_seed, plan.label_seed, plan.loader_seed)
    })
    all_parent_rng_overlap = sorted(set(formal_rng).intersection(
        expected_parent_rng["seed_values"]))
    if freshness.get("formal_rng_seeds") != formal_rng \
            or freshness.get("parent_rng_registry") != expected_parent_rng \
            or freshness.get(
                "all_parent_rng_overlap") != all_parent_rng_overlap \
            or all_parent_rng_overlap:
        raise LeWMFormalError("all-parent RNG overlap proof failed")
    expected_forbidden = forbidden_parent_artifact_receipt(
        study, manifest["cohort"])
    if freshness.get("forbidden_parent_artifacts") != expected_forbidden:
        raise LeWMFormalError("forbidden parent artifact receipt differs")

    parent = load_parent_spec(parent_path, PARENT_SHA, verify_inputs=False)
    if manifest["cohort"] == "lewm_reacher_color":
        expected_parent = reacher_parent_seed_receipt(parent)
        overlap = sorted(set(formal_rng).intersection(
            expected_parent["parent_reacher_seeds"]))
        if freshness.get("parent") != expected_parent \
                or freshness.get("formal_rng_seeds") != formal_rng \
                or freshness.get("formal_parent_rng_overlap") != overlap \
                or overlap:
            raise LeWMFormalError("Reacher parent RNG overlap proof failed")
        for split in FORMAL_SPLITS:
            expected_ids = (np.int64(plans[split].base_seed) * 10_000
                            + np.arange(
                                int(splits[split]["count"]), dtype=np.int64))
            if not np.array_equal(trajectory_ids[split], expected_ids):
                raise LeWMFormalError(
                    f"Reacher generated episode identity differs: {split}")
    else:
        excluded, expected_parent = pusht_parent_exclusion_receipt(parent)
        selected = set().union(*trajectory_sets.values())
        overlap = sorted(selected.intersection(excluded))
        if freshness.get("parent") != expected_parent \
                or freshness.get("selected_episode_count") != len(selected) \
                or freshness.get("selected_parent_overlap") != overlap \
                or overlap:
            raise LeWMFormalError("PushT parent episode overlap proof failed")
    return manifest


def load_lewm_trajectory_banks(
        manifest_path: Path, *, expected_cohort: str | None = None,
        expected_counts: Mapping[str, int] | None = None
) -> tuple[dict[str, Any], dict[str, FormalLeWMTrajectoryBank]]:
    """Open only label-free artifacts for carrier training/evaluation."""

    path = Path(manifest_path).resolve()
    manifest = validate_lewm_formal_manifest(
        path, expected_cohort=expected_cohort,
        expected_counts=expected_counts)
    banks = {}
    for split in FORMAL_SPLITS:
        artifact = _safe_artifact_path(
            path.parent,
            manifest["splits"][split]["trajectory_artifact"],
            label=f"{split} trajectory")
        bank = FormalLeWMTrajectoryBank(_load_npz(artifact))
        if set(bank.array_names).intersection({
                "color_label", "location_label", "combination_label"}):
            raise LeWMFormalError("semantic label leaked through carrier API")
        banks[split] = bank
    return _trajectory_handle_from_manifest(path, manifest), banks


def sealed_label_vault_handle(
        manifest_path: Path, custody_receipt_path: Path) -> dict[str, Any]:
    """Authenticate external label custody without returning label values."""

    manifest = validate_lewm_formal_manifest(manifest_path)
    receipt_path = Path(custody_receipt_path).resolve()
    if not receipt_path.is_file() or receipt_path.is_symlink() \
            or oct(receipt_path.stat().st_mode & 0o777) != "0o400":
        raise LeWMFormalError("sealed-label custody receipt is not protected")
    custody = json.loads(receipt_path.read_text())
    required = {
        "schema", "study", "status", "cohort", "path", "size", "sha256",
        "mode", "study_protocol_sha256", "split_label_hashes",
        "per_cell_api_access", "available_only_to",
    }
    if set(custody) != required \
            or custody.get("schema") != LEWM_LABEL_CUSTODY_SCHEMA \
            or custody.get("study") != "sage-mem-v1" \
            or custody.get("status") != "sealed-for-post-grid-finalizer" \
            or custody.get("cohort") != manifest["cohort"] \
            or custody.get("per_cell_api_access") is not False \
            or custody.get("available_only_to") != POST_GRID_FINALIZER_PHASE \
            or custody.get("study_protocol_sha256") != manifest.get(
                "study_protocol_sha256") \
            or custody.get("split_label_hashes") != {
                split: manifest["splits"][split][
                    "sealed_label_arrays_sha256"]
                for split in FORMAL_SPLITS}:
        raise LeWMFormalError("sealed-label custody identity failed")
    vault = Path(custody.get("path", "")).resolve()
    try:
        vault.relative_to(Path(manifest_path).resolve().parent)
    except ValueError:
        pass
    else:
        raise LeWMFormalError("sealed-label vault is inside label-free bank")
    if not vault.is_file() or vault.is_symlink() \
            or vault.stat().st_size != custody.get("size") \
            or _sha256_file(vault) != custody.get("sha256") \
            or custody.get("sha256") != manifest.get(
                "sealed_label_vault_sha256") \
            or oct(vault.stat().st_mode & 0o777) != custody.get("mode") \
            or custody.get("mode") != "0o400":
        raise LeWMFormalError("sealed-label vault artifact identity failed")
    return {
        "custody_receipt": {
            "path": str(receipt_path),
            "sha256": _sha256_file(receipt_path),
            "size": receipt_path.stat().st_size,
        },
        "vault_sha256": custody["sha256"],
        "cohort": manifest["cohort"],
        "available_only_to": POST_GRID_FINALIZER_PHASE,
        "per_cell_api_access": False,
    }


def finalizer_custody_record(
        manifest_path: Path, custody_receipt_path: Path, *,
        registry_root: Path) -> dict[str, Any]:
    """Build the finalizer's opaque custody mapping without parsing labels."""

    sealed_label_vault_handle(manifest_path, custody_receipt_path)
    custody = json.loads(Path(custody_receipt_path).read_text())
    vault = Path(custody["path"]).resolve()
    root = Path(registry_root).resolve()
    try:
        relative = vault.relative_to(root)
    except ValueError as error:
        raise LeWMFormalError(
            "LeWM label vault must be below finalizer registry root") from error
    artifact = {
        "path": str(relative),
        "sha256": custody["sha256"],
        "size": custody["size"],
    }
    sources = {}
    for split in ("formal_test", "consumer_train"):
        sources[split] = {
            "artifact": dict(artifact),
            "keys": {
                "episode_id": f"{split}__episode_index",
                "native_cluster_id": None,
                "label": f"{split}__color_label",
            },
        }
    return {
        "bank_manifest_sha256": _sha256_file(Path(manifest_path)),
        "classes": 4,
        "sources": sources,
    }


def load_lewm_sealed_labels(
        manifest_path: Path, custody_receipt_path: Path, *, phase: str
) -> dict[str, SealedLeWMLabels]:
    """Open labels only from the explicit post-grid finalizer phase."""

    if phase != POST_GRID_FINALIZER_PHASE:
        raise LeWMFormalError(
            "sealed labels are unavailable to carrier-cell execution")
    sealed_label_vault_handle(manifest_path, custody_receipt_path)
    path = Path(manifest_path).resolve()
    manifest = validate_lewm_formal_manifest(path)
    custody = json.loads(Path(custody_receipt_path).read_text())
    vault_arrays = _load_npz(Path(custody["path"]))
    split_counts = {
        split: int(manifest["splits"][split]["count"])
        for split in FORMAL_SPLITS
    }
    labels_by_split = _labels_from_vault_arrays(
        vault_arrays, split_counts=split_counts)
    result = {}
    for split in FORMAL_SPLITS:
        trajectory = _load_npz(_safe_artifact_path(
            path.parent,
            manifest["splits"][split]["trajectory_artifact"],
            label=f"{split} trajectory"))
        arrays = labels_by_split[split]
        validate_label_split_arrays(
            arrays, expected_count=split_counts[split],
            expected_episode_index=trajectory["episode_index"])
        if _sha256_arrays(*(
                np.asarray(arrays[name]) for name in sorted(arrays))) != \
                manifest["splits"][split]["sealed_label_arrays_sha256"]:
            raise LeWMFormalError(f"sealed-label split hash differs: {split}")
        result[split] = SealedLeWMLabels(
            episode_ids=np.asarray(
                arrays["episode_index"], dtype=np.int64).copy(),
            color=np.asarray(arrays["color_label"], dtype=np.int64).copy(),
            location=np.asarray(
                arrays["location_label"], dtype=np.int64).copy(),
            combination=np.asarray(
                arrays["combination_label"], dtype=np.int64).copy(),
        )
    return result


def trajectory_bank_handle(manifest_path: Path) -> dict[str, Any]:
    """Minimal label-free handle suitable for later carrier-cell payloads."""

    path = Path(manifest_path).resolve()
    manifest = validate_lewm_formal_manifest(path)
    return _trajectory_handle_from_manifest(path, manifest)


def _trajectory_handle_from_manifest(
        path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitize a full bank manifest for the label-free process boundary."""

    return {
        "schema": LEWM_FORMAL_SCHEMA,
        "cohort": manifest["cohort"],
        # The external custody receipt/path is intentionally absent.  The
        # digest is enough for a carrier feature artifact to bind to this bank.
        "manifest_sha256": _sha256_file(path),
        "formal_labels_hidden": True,
        "labels_used_for_carrier_training": False,
        "sealed_label_vault_sha256": manifest[
            "sealed_label_vault_sha256"],
        "labels_accessible_through_handle": False,
        "host_digest": manifest["host_hash_before"],
        "admissions": dict(manifest["admissions"]),
        "splits": {
            split: {
                "count": int(manifest["splits"][split]["count"]),
                "selection_sha256": manifest["splits"][split][
                    "selection_sha256"],
                "trajectory_artifact": {
                    **dict(manifest["splits"][split][
                        "trajectory_artifact"]),
                    "path": str((path.parent / manifest["splits"][split][
                        "trajectory_artifact"]["path"]).resolve()),
                },
                # Only an opaque split identity is exposed; no vault path.
                "sealed_label_sha256": manifest["splits"][split][
                    "sealed_label_arrays_sha256"],
            }
            for split in FORMAL_SPLITS
        },
    }


def label_free_feature_keys() -> set[str]:
    """Exact feature representation consumed by the shared finalizer."""

    return {
        "formal_test_episode_id",
        "formal_test_native_cluster_id",
        "formal_test_evidence_age",
        "consumer_train_episode_id",
        "consumer_train_native_cluster_id",
        "consumer_train_evidence_age",
        "formal_test_full_mse",
        "formal_test_reset_mse",
        "formal_test_prior_mse",
        "formal_test_full_features",
        "formal_test_reset_features",
        "formal_test_prior_features",
        "consumer_train_full_features",
    }


def phase_a_identity_arrays(
        *, consumer_train_episode_ids: np.ndarray,
        formal_test_episode_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Build the finalizer's age-major identity matrices for LeWM.

    LeWM has one counterfactual row per native trajectory, so the stable native
    cluster identifier is exactly the episode identifier.
    """

    result = {}
    for prefix, raw_ids in (
            ("consumer_train", consumer_train_episode_ids),
            ("formal_test", formal_test_episode_ids)):
        ids = np.asarray(raw_ids, dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids) \
                or np.any(ids < 0):
            raise LeWMFormalError(f"{prefix} episode identity is malformed")
        repeated = np.repeat(ids[None, :], len(AGES), axis=0)
        result[f"{prefix}_episode_id"] = repeated
        result[f"{prefix}_native_cluster_id"] = repeated.copy()
        result[f"{prefix}_evidence_age"] = np.repeat(
            np.asarray(AGES, dtype=np.int64)[:, None], len(ids), axis=1)
    return result


def validate_label_free_feature_arrays(
        arrays: Mapping[str, np.ndarray], *,
        expected_consumer_train_episode_ids: np.ndarray,
        expected_formal_test_episode_ids: np.ndarray) -> None:
    """Validate the exact finalizer feature schema without opening labels."""

    if set(arrays) != label_free_feature_keys():
        raise LeWMFormalError("label-free feature artifact schema changed")
    expected_identity = phase_a_identity_arrays(
        consumer_train_episode_ids=expected_consumer_train_episode_ids,
        formal_test_episode_ids=expected_formal_test_episode_ids)
    for name, expected in expected_identity.items():
        value = np.asarray(arrays[name])
        if not np.array_equal(value, expected):
            raise LeWMFormalError(f"feature identity differs: {name}")
    test_count = len(expected_formal_test_episode_ids)
    consumer_count = len(expected_consumer_train_episode_ids)
    for name in ("formal_test_full_mse", "formal_test_reset_mse",
                 "formal_test_prior_mse"):
        value = np.asarray(arrays[name])
        if value.shape != (len(AGES), test_count) \
                or not np.issubdtype(value.dtype, np.number) \
                or not np.isfinite(value).all() or np.any(value < 0):
            raise LeWMFormalError(f"invalid label-free health value: {name}")
    feature_shapes = {
        "formal_test_full_features": (len(AGES), test_count, 192),
        "formal_test_reset_features": (len(AGES), test_count, 192),
        "formal_test_prior_features": (len(AGES), test_count, 192),
        "consumer_train_full_features": (
            len(AGES), consumer_count, 192),
    }
    for name, shape in feature_shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or not np.issubdtype(value.dtype, np.number) \
                or not np.isfinite(value).all():
            raise LeWMFormalError(f"invalid label-free feature: {name}")


def validate_lewm_phase_a_measurement_artifact(
        artifact_path: Path, *, trajectory_handle: Mapping[str, Any]
) -> dict[str, Any]:
    """Hash and validate one finalizer-compatible phase-A measurement NPZ."""

    if trajectory_handle.get("schema") != LEWM_FORMAL_SCHEMA \
            or trajectory_handle.get("formal_labels_hidden") is not True:
        raise LeWMFormalError("trajectory handle is not label-free")
    identities = {}
    for split in FEATURE_SPLITS:
        record = trajectory_handle["splits"][split]["trajectory_artifact"]
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["size"] \
                or _sha256_file(path) != record["sha256"]:
            raise LeWMFormalError(
                f"trajectory handle identity failed: {split}")
        trajectory = _load_npz(path)
        validate_trajectory_split_arrays(
            trajectory,
            expected_count=int(trajectory_handle["splits"][split]["count"]))
        identities[split] = trajectory["episode_index"]
    path = Path(artifact_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise LeWMFormalError("phase-A measurement artifact is missing")
    arrays = _load_npz(path)
    validate_label_free_feature_arrays(
        arrays,
        expected_consumer_train_episode_ids=identities["consumer_train"],
        expected_formal_test_episode_ids=identities["formal_test"])
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
        "bank_manifest_sha256": trajectory_handle["manifest_sha256"],
        "prediction_representation": "feature_artifact",
        "consumer_contract": "centralized-pooled-consumer-train-features",
        "formal_test_labels_read": False,
    }


__all__ = [
    "LEWM_FORMAL_SCHEMA", "LEWM_LABEL_CUSTODY_SCHEMA", "LEWM_COHORTS",
    "FORMAL_SPLITS", "FEATURE_SPLITS",
    "POST_GRID_FINALIZER_PHASE", "LeWMFormalError", "FormalSeedPlan",
    "formal_seed_plan", "reacher_parent_seed_receipt",
    "pusht_parent_exclusion_receipt", "forbidden_parent_artifact_receipt",
    "parent_rng_registry_receipt", "select_fresh_hdf_splits",
    "trajectory_split_keys", "label_split_keys",
    "formal_split_keys", "partition_formal_split_arrays",
    "validate_trajectory_split_arrays", "validate_label_split_arrays",
    "validate_formal_split_arrays", "prepare_lewm_formal_bank",
    "FormalLeWMTrajectoryBank", "SealedLeWMLabels",
    "validate_lewm_formal_manifest", "load_lewm_trajectory_banks",
    "sealed_label_vault_handle", "finalizer_custody_record",
    "load_lewm_sealed_labels",
    "trajectory_bank_handle", "label_free_feature_keys",
    "phase_a_identity_arrays", "validate_label_free_feature_arrays",
    "validate_lewm_phase_a_measurement_artifact",
]
