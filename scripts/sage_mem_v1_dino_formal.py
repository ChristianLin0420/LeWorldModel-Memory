#!/usr/bin/env python3
"""Fresh-bank and formal-artifact support for the DINO-WM SAGE-Mem cohorts.

This module is intentionally independent of the development host adapter.  It
has three narrow responsibilities:

* authenticate every declared parent episode/RNG registry;
* select and materialize fresh, parent-disjoint native DINO-WM banks; and
* expose immutable label-free feature/trajectory handles for a later
  two-phase formal executor.

No development outcome is opened here.  Carrier training, label unlock,
shared-consumer fitting, and episode-level result finalization are deliberately
outside this API.  In particular, this module cannot produce a per-cell
correctness artifact: doing so would violate the preregistered post-grid label
boundary and shared arm-blind consumer contract.

PointMaze counts are native base windows.  Each selected base is expanded into
its intact four-cue counterfactual set.  Fresh banks store both a unique
``episode_id`` and ``native_cluster_id`` so a later finalizer can preserve the
native-episode resampling unit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DINO_FORMAL_API_VERSION = "sage_mem_v1_dino_formal_v1"
DINO_COHORTS = (
    "dinowm_pusht_token",
    "dinowm_pusht_binding",
    "dinowm_pointmaze_goal",
)
FORMAL_SPLITS = ("formal_train", "consumer_train", "formal_test")
AGES = (4, 8, 15)

_COHORT_INFO: dict[str, dict[str, Any]] = {
    "dinowm_pusht_token": {
        "family": "pusht",
        "classes": 4,
        "semantic_task": "PushT transient visual-token recall",
        "parent_task": "transient-visual-token-recall",
        "namespace": 11,
        "variants_per_native_episode": 1,
    },
    "dinowm_pusht_binding": {
        "family": "pusht",
        "classes": 6,
        "semantic_task": "PushT multi-item visual-binding recall",
        "parent_task": "multi-item-visual-binding-recall",
        "namespace": 12,
        "variants_per_native_episode": 1,
    },
    "dinowm_pointmaze_goal": {
        "family": "pointmaze",
        "classes": 4,
        "semantic_task": "PointMaze transient four-goal recall",
        "parent_task": "transient-four-goal-recall",
        "namespace": 13,
        "variants_per_native_episode": 4,
    },
}


class DinoFormalError(RuntimeError):
    """A DINO formal freshness, identity, or execution invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DinoFormalError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


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
    return sha256_file(path)


def _repo_path(value: str | Path, *, root: Path = ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    result = (root / path).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise DinoFormalError(f"path leaves repository: {value}") from error
    return result


def _configured_path(value: str | Path, *, root: Path = ROOT) -> Path:
    """Resolve configuration syntax without following an in-repo symlink."""

    path = Path(value)
    if path.is_absolute():
        return path
    _require(".." not in path.parts,
             f"configured path leaves repository: {value}")
    return root.absolute() / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def activate_parent_dependency_environment(
        config: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Authenticate and activate the parent's pinned dependency/shim paths.

    Native DINO-WM checkpoints contain pickled vendor classes.  Their modules
    must be importable before ``torch.load`` opens a frozen host.  This mirrors
    the development adapter's activation contract while keeping formal bank
    preparation independent of that adapter.
    """

    execution = config.get("execution")
    _require(isinstance(execution, Mapping),
             "parent execution registry is missing")
    python_value = execution.get("isolated_python")
    manifest_value = execution.get("dependency_manifest_path")
    identity = execution.get("dependency_manifest_identity")
    _require(isinstance(python_value, str) and python_value
             and isinstance(manifest_value, str) and manifest_value
             and isinstance(identity, Mapping),
             "parent dependency registry is malformed")
    # The venv Python is normally an in-repository symlink to /usr/bin/python.
    # Preserve the configured path so its sibling lib/site-packages tree can
    # be authenticated; resolving the symlink here would lose that tree.
    isolated = _configured_path(python_value, root=root)
    manifest = _configured_path(manifest_value, root=root)
    _require(isolated.is_file(),
             f"parent isolated Python is unavailable: {isolated}")
    _require(manifest.is_file()
             and manifest.stat().st_size == int(identity.get("size", -1))
             and sha256_file(manifest) == identity.get("sha256"),
             "parent dependency manifest identity changed")
    candidates = sorted(
        isolated.parent.parent.glob("lib/python*/site-packages/*.pth"))
    _require(len(candidates) == 1,
             "parent isolated dependency shim is ambiguous")
    pth = candidates[0]
    paths = [Path(line.strip()).resolve()
             for line in pth.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    _require(paths and all(path.is_dir() for path in paths),
             "parent dependency shim path is unavailable")
    # Reverse insertion preserves the path priority declared by the .pth.
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return {
        "status": "activated-before-native-host-import",
        "isolated_python": str(python_value),
        "isolated_python_resolved_target": str(isolated.resolve()),
        "dependency_manifest": {
            "path": _display_path(manifest, root),
            "size": manifest.stat().st_size,
            "sha256": sha256_file(manifest),
        },
        "pth": {
            "path": _display_path(pth, root),
            "size": pth.stat().st_size,
            "sha256": sha256_file(pth),
        },
        "activated_paths": [str(path) for path in paths],
    }


def _numeric_seed_values(value: Any, key_path: tuple[str, ...] = ()) \
        -> set[int]:
    result: set[int] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.update(_numeric_seed_values(child, key_path + (str(key),)))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.update(_numeric_seed_values(child, key_path))
    elif key_path and "seed" in key_path[-1].lower() \
            and isinstance(value, int) and not isinstance(value, bool):
        result.add(int(value))
    return result


def _episode_rows(value: Any) -> list[tuple[int, int | None]]:
    """Extract only explicit selection rows, never provenance episode lists."""

    result: list[tuple[int, int | None]] = []
    if isinstance(value, Mapping):
        episode = value.get("episode_index")
        if isinstance(episode, int) and not isinstance(episode, bool):
            start = value.get("local_start")
            result.append((int(episode), int(start) if isinstance(
                start, int) and not isinstance(start, bool) else None))
        for child in value.values():
            result.extend(_episode_rows(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_episode_rows(child))
    return result


def _registry_references(value: Any, key_path: tuple[str, ...] = ()) \
        -> set[str]:
    """Find selection/metadata artifacts referenced by a config or manifest."""

    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.update(_registry_references(child, key_path + (str(key),)))
    elif isinstance(value, list):
        for child in value:
            result.update(_registry_references(child, key_path))
    elif isinstance(value, str) and key_path:
        context = "/".join(key_path).lower()
        basename = Path(value).name.lower()
        registry_name = basename == "selection.json" or basename == \
            "metadata.npz"
        if registry_name and ("selection" in context or "metadata" in context):
            result.add(value)
    return result


@dataclass(frozen=True)
class ParentExclusionRegistry:
    """Authenticated union of parent native episodes and RNG seeds."""

    episodes: frozenset[int]
    episode_starts: frozenset[tuple[int, int]]
    seeds: frozenset[int]
    artifacts: tuple[Mapping[str, Any], ...]
    registry_sha256: str

    def public_receipt(self) -> dict[str, Any]:
        return {
            "schema": "sage_mem_v1_dino_parent_exclusion_v1",
            "api_version": DINO_FORMAL_API_VERSION,
            "episode_count": len(self.episodes),
            "episode_start_count": len(self.episode_starts),
            "seed_count": len(self.seeds),
            "episode_union_sha256": sha256_json(sorted(self.episodes)),
            "episode_start_union_sha256": sha256_json(
                sorted([list(value) for value in self.episode_starts])),
            "seed_union_sha256": sha256_json(sorted(self.seeds)),
            "artifacts": [dict(record) for record in self.artifacts],
            "registry_sha256": self.registry_sha256,
        }


def collect_parent_exclusion_registry(
        *, parent_protocol: str | Path,
        forbidden_parent_artifacts: Sequence[str | Path],
        root: Path = ROOT) -> ParentExclusionRegistry:
    """Authenticate and union all reachable parent selection/RNG registries.

    The declared forbidden artifacts are mandatory.  Their JSON manifests may
    point at a selection JSON or metadata NPZ; those references are followed
    recursively.  The parent YAML is inspected only for RNG fields and
    selection/metadata references.  No result/cell directory is discovered or
    opened.
    """

    protocol = _repo_path(parent_protocol, root=root)
    _require(protocol.is_file(), f"parent protocol missing: {protocol}")
    queue = [protocol] + [
        _repo_path(value, root=root) for value in forbidden_parent_artifacts]
    declared = {path.resolve() for path in queue[1:]}
    visited: set[Path] = set()
    episodes: set[int] = set()
    starts: set[tuple[int, int]] = set()
    seeds: set[int] = set()
    artifacts: list[dict[str, Any]] = []

    while queue:
        path = queue.pop(0).resolve()
        if path in visited:
            continue
        _require(path.is_file(), f"parent registry missing: {path}")
        visited.add(path)
        suffix = path.suffix.lower()
        references: set[str] = set()
        row_count = 0
        if suffix in (".yaml", ".yml"):
            value = yaml.safe_load(path.read_text())
            _require(isinstance(value, Mapping),
                     f"parent YAML is not a mapping: {path}")
            seeds.update(_numeric_seed_values(value))
            references.update(_registry_references(value))
            kind = "parent-protocol"
        elif suffix == ".json":
            value = json.loads(path.read_text())
            seeds.update(_numeric_seed_values(value))
            rows = _episode_rows(value)
            row_count = len(rows)
            for episode, start in rows:
                episodes.add(episode)
                if start is not None:
                    starts.add((episode, start))
            references.update(_registry_references(value))
            kind = "json-registry"
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                _require("episode_index" in archive.files,
                         f"metadata registry has no episode_index: {path}")
                raw_episode = np.asarray(archive["episode_index"])
                _require(raw_episode.ndim == 1
                         and np.issubdtype(raw_episode.dtype, np.integer),
                         f"malformed episode_index registry: {path}")
                raw_start = (np.asarray(archive["local_start"])
                             if "local_start" in archive.files else None)
                if raw_start is not None:
                    _require(raw_start.shape == raw_episode.shape
                             and np.issubdtype(raw_start.dtype, np.integer),
                             f"malformed local_start registry: {path}")
                for index, raw in enumerate(raw_episode.tolist()):
                    episode = int(raw)
                    episodes.add(episode)
                    if raw_start is not None:
                        starts.add((episode, int(raw_start[index])))
                row_count = int(raw_episode.size)
            kind = "npz-registry"
        else:
            raise DinoFormalError(
                f"unsupported parent registry format: {path}")
        artifacts.append({
            "path": _display_path(path, root),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "kind": kind,
            "explicit_selection_rows": row_count,
        })
        for reference in sorted(references):
            referenced = _repo_path(reference, root=root)
            # A relative cache filename such as ``cache/metadata.npz`` lacks
            # enough context in a standalone config.  A pinned manifest must
            # provide its repository-qualified counterpart; ignore only this
            # unresolved shorthand, never a declared forbidden artifact.
            if referenced.is_file():
                queue.append(referenced)

    missing_declared = declared.difference(visited)
    _require(not missing_declared,
             f"declared forbidden registry was not visited: {missing_declared}")
    _require(episodes, "parent registry union contains no selected episodes")
    artifacts.sort(key=lambda record: str(record["path"]))
    fingerprint = sha256_json({
        "episodes": sorted(episodes),
        "episode_starts": sorted([list(value) for value in starts]),
        "seeds": sorted(seeds),
        "artifacts": artifacts,
    })
    return ParentExclusionRegistry(
        episodes=frozenset(episodes),
        episode_starts=frozenset(starts),
        seeds=frozenset(seeds),
        artifacts=tuple(artifacts),
        registry_sha256=fingerprint,
    )


@dataclass(frozen=True)
class FormalSelectionRow:
    split: str
    source_split: str
    episode_index: int
    local_start: int
    class_id: int
    episode_id: int
    native_cluster_id: int

    def public_value(self) -> dict[str, Any]:
        """Return the label-hidden identity written to the public receipt."""

        return {
            "split": self.split,
            "source_split": self.source_split,
            "episode_index": self.episode_index,
            "local_start": self.local_start,
            "episode_id": self.episode_id,
            "native_cluster_id": self.native_cluster_id,
        }


@dataclass(frozen=True)
class FormalSelectionPlan:
    cohort: str
    parent_protocol: str
    parent_protocol_sha256: str
    parent_registry: ParentExclusionRegistry
    rows: tuple[FormalSelectionRow, ...]
    requested_counts: Mapping[str, int]
    used_seeds: Mapping[str, Mapping[str, int]]
    sequence: Mapping[str, int]

    @property
    def info(self) -> Mapping[str, Any]:
        return _COHORT_INFO[self.cohort]

    def rows_for(self, split: str) -> tuple[FormalSelectionRow, ...]:
        if split not in FORMAL_SPLITS:
            raise DinoFormalError(f"unknown formal split: {split}")
        return tuple(row for row in self.rows if row.split == split)

    @property
    def native_episodes(self) -> frozenset[int]:
        return frozenset(row.episode_index for row in self.rows)

    @property
    def plan_sha256(self) -> str:
        return sha256_json({
            "cohort": self.cohort,
            "parent_protocol_sha256": self.parent_protocol_sha256,
            "parent_registry_sha256": self.parent_registry.registry_sha256,
            "counts": dict(self.requested_counts),
            "seeds": {key: dict(value) for key, value in self.used_seeds.items()},
            "sequence": dict(self.sequence),
            "rows": [dict(row.public_value(), class_id=row.class_id)
                     for row in self.rows],
        })

    def split_record(self, split: str) -> dict[str, Any]:
        rows = self.rows_for(split)
        public = [row.public_value() for row in rows]
        labels = [row.class_id for row in rows]
        native = sorted({row.episode_index for row in rows})
        return {
            # The registered SAGE-Mem count is a native episode/base-window
            # count.  PointMaze expands each base into four immutable cue
            # variants, while PushT has one sequence per native episode.
            "count": int(self.requested_counts[split]),
            "expanded_sequence_count": len(rows),
            "native_episode_count": len(native),
            "selection_sha256": sha256_json(public),
            "native_episode_sha256": sha256_json(native),
            "label_vault_sha256": sha256_json(labels),
            "class_histogram": {
                str(label): labels.count(label)
                for label in range(int(self.info["classes"]))
            },
        }

    def public_receipt(self) -> dict[str, Any]:
        public_rows = [row.public_value() for row in self.rows]
        return {
            "schema": "sage_mem_v1_dino_formal_selection_v1",
            "api_version": DINO_FORMAL_API_VERSION,
            "study": "sage-mem-v1",
            "status": "selected-parent-disjoint",
            "cohort": self.cohort,
            "family": self.info["family"],
            "parent_protocol": self.parent_protocol,
            "parent_protocol_sha256": self.parent_protocol_sha256,
            "parent_exclusion": self.parent_registry.public_receipt(),
            "split_count_unit": "native episode/base window",
            "counterfactual_variants_per_native_episode": int(
                self.info["variants_per_native_episode"]),
            "splits": {split: self.split_record(split)
                       for split in FORMAL_SPLITS},
            "used_seeds": {
                split: dict(self.used_seeds[split]) for split in FORMAL_SPLITS},
            "sequence": dict(self.sequence),
            "selection_rows": public_rows,
            "semantic_labels_in_public_receipt": False,
            "labels_used_for_selection": False,
            "labels_used_for_carrier_training": False,
            "parent_episode_overlap": 0,
            "cross_split_native_episode_overlap": 0,
            "plan_sha256": self.plan_sha256,
        }


def _length_vector(value: Sequence[int] | Mapping[int, int]) -> dict[int, int]:
    if isinstance(value, Mapping):
        result = {int(key): int(length) for key, length in value.items()}
    else:
        result = {index: int(length) for index, length in enumerate(value)}
    _require(result and all(index >= 0 and length > 0
                            for index, length in result.items()),
             "episode_lengths must contain positive native lengths")
    return result


def _validate_registry_seeds(
        cohort: str, seed_registry: Mapping[str, int],
        parent_seeds: frozenset[int]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    flat: list[int] = []
    for split in FORMAL_SPLITS:
        values: dict[str, int] = {}
        for purpose in ("episode_selection", "cue_labels", "loader"):
            key = f"{cohort}/{split}/{purpose}"
            seed = seed_registry.get(key)
            _require(isinstance(seed, int) and not isinstance(seed, bool)
                     and 0 <= int(seed) < 2_147_483_647,
                     f"missing or invalid seed registry entry: {key}")
            values[purpose] = int(seed)
            flat.append(int(seed))
        result[split] = values
    _require(len(flat) == len(set(flat)),
             f"formal seed registry collides within {cohort}")
    overlap = set(flat).intersection(parent_seeds)
    _require(not overlap,
             f"formal seeds collide with parent RNG registry: {sorted(overlap)}")
    return result


def _episode_id(namespace: int, episode: int, label: int,
                variants: int) -> int:
    # The namespace prevents cross-cohort accidental identity collisions while
    # remaining exactly representable by int64.
    return namespace * 1_000_000_000 + episode * variants + (
        label if variants > 1 else 0)


def plan_fresh_formal_selection(
        *, cohort: str, split_counts: Mapping[str, int],
        seed_registry: Mapping[str, int],
        episode_lengths: Sequence[int] | Mapping[int, int],
        parent_registry: ParentExclusionRegistry,
        eligible_episodes: Sequence[int] | None = None,
        additional_excluded_episodes: Iterable[int] = (),
        parent_protocol: str | Path,
        root: Path = ROOT,
        source_split: str = "train",
        num_frames: int = 20,
        frame_skip: int = 5) -> FormalSelectionPlan:
    """Create an exact, deterministic, parent-disjoint formal selection.

    SAGE-Mem split counts denote native episodes/base windows.  For PointMaze
    every selected native base contributes its intact four-label
    counterfactual set, so the materialized sequence count is four times the
    registered split count.
    """

    _require(cohort in DINO_COHORTS, f"unsupported DINO cohort: {cohort}")
    info = _COHORT_INFO[cohort]
    _require(isinstance(source_split, str) and source_split,
             "native source_split must be a non-empty string")
    counts = {split: split_counts.get(split) for split in FORMAL_SPLITS}
    _require(all(isinstance(value, int) and not isinstance(value, bool)
                 and int(value) > 0 for value in counts.values()),
             f"formal split counts are incomplete for {cohort}")
    counts = {key: int(value) for key, value in counts.items()}
    variants = int(info["variants_per_native_episode"])
    used_seeds = _validate_registry_seeds(
        cohort, seed_registry, parent_registry.seeds)
    lengths = _length_vector(episode_lengths)
    source = (list(map(int, eligible_episodes)) if eligible_episodes is not None
              else sorted(lengths))
    _require(len(source) == len(set(source)),
             "eligible native episode registry contains duplicates")
    # PointMaze's native TrajSlicer owns the complete final action block.  The
    # PushT reader needs only the final sampled observation.
    span = (num_frames * frame_skip if info["family"] == "pointmaze" else
            (num_frames - 1) * frame_skip + 1)
    excluded = set(parent_registry.episodes)
    excluded.update(map(int, additional_excluded_episodes))
    remaining = [episode for episode in source
                 if episode in lengths and lengths[episode] >= span
                 and episode not in excluded]
    needed_native = sum(counts.values())
    _require(len(remaining) >= needed_native,
             f"{cohort} has {len(remaining)} eligible fresh episodes; "
             f"needs {needed_native}")

    rows: list[FormalSelectionRow] = []
    available = set(remaining)
    classes = int(info["classes"])
    namespace = int(info["namespace"])
    for split in FORMAL_SPLITS:
        requested = counts[split]
        native_count = requested
        candidates = np.asarray(sorted(available), dtype=np.int64)
        permutation = np.random.default_rng(
            used_seeds[split]["episode_selection"]).permutation(candidates)
        chosen = list(map(int, permutation[:native_count]))
        _require(len(chosen) == native_count, f"short selection: {split}")
        available.difference_update(chosen)
        if variants == 1:
            labels = np.arange(requested, dtype=np.int64) % classes
            np.random.default_rng(used_seeds[split]["cue_labels"]).shuffle(
                labels)
            label_rows = [[int(label)] for label in labels.tolist()]
        else:
            # Every base carries every label.  The cue-label seed controls only
            # public row order and cannot alter membership or class balance.
            label_rows = []
            for episode in chosen:
                order = np.random.default_rng(np.random.SeedSequence([
                    used_seeds[split]["cue_labels"], episode])).permutation(
                        classes)
                label_rows.append(list(map(int, order)))
        for chosen_index, episode in enumerate(chosen):
            max_start = lengths[episode] - span
            start = int(np.random.default_rng(np.random.SeedSequence([
                used_seeds[split]["loader"], episode])).integers(
                    max_start + 1))
            cluster = namespace * 1_000_000_000 + episode
            for label in label_rows[chosen_index]:
                rows.append(FormalSelectionRow(
                    split=split,
                    source_split=source_split,
                    episode_index=episode,
                    local_start=start,
                    class_id=label,
                    episode_id=_episode_id(
                        namespace, episode, label, variants),
                    native_cluster_id=cluster,
                ))

    _require(len(rows) == sum(counts.values()) * variants,
             "formal selection count changed during planning")
    selected = {row.episode_index for row in rows}
    _require(not selected.intersection(parent_registry.episodes),
             "fresh selection overlaps a parent episode")
    split_sets = [{row.episode_index for row in rows if row.split == split}
                  for split in FORMAL_SPLITS]
    _require(all(not split_sets[left].intersection(split_sets[right])
                 for left in range(len(split_sets))
                 for right in range(left + 1, len(split_sets))),
             "formal native episodes overlap across splits")
    for split in FORMAL_SPLITS:
        labels = [row.class_id for row in rows if row.split == split]
        histogram = [labels.count(label) for label in range(classes)]
        _require(max(histogram) - min(histogram) <= 1,
                 f"class imbalance changed for {cohort}/{split}")
    protocol = _repo_path(parent_protocol, root=root)
    _require(protocol.is_file(), f"parent protocol missing: {protocol}")
    return FormalSelectionPlan(
        cohort=cohort,
        parent_protocol=_display_path(protocol, root),
        parent_protocol_sha256=sha256_file(protocol),
        parent_registry=parent_registry,
        rows=tuple(rows),
        requested_counts=counts,
        used_seeds=used_seeds,
        sequence={"num_frames": int(num_frames),
                  "frame_skip": int(frame_skip),
                  "cue_start": 1, "cue_length": 3},
    )


def plan_pusht_formal_pair(
        *, split_counts_by_cohort: Mapping[str, Mapping[str, int]],
        seed_registry: Mapping[str, int],
        episode_lengths: Sequence[int] | Mapping[int, int],
        parent_registry: ParentExclusionRegistry,
        parent_protocol: str | Path,
        eligible_episodes: Sequence[int] | None = None,
        source_split: str = "train",
        root: Path = ROOT) -> dict[str, FormalSelectionPlan]:
    """Plan token then binding banks with native episodes disjoint across both."""

    token = plan_fresh_formal_selection(
        cohort="dinowm_pusht_token",
        split_counts=split_counts_by_cohort["dinowm_pusht_token"],
        seed_registry=seed_registry,
        episode_lengths=episode_lengths,
        parent_registry=parent_registry,
        eligible_episodes=eligible_episodes,
        parent_protocol=parent_protocol,
        root=root,
        source_split=source_split,
    )
    binding = plan_fresh_formal_selection(
        cohort="dinowm_pusht_binding",
        split_counts=split_counts_by_cohort["dinowm_pusht_binding"],
        seed_registry=seed_registry,
        episode_lengths=episode_lengths,
        parent_registry=parent_registry,
        eligible_episodes=eligible_episodes,
        additional_excluded_episodes=token.native_episodes,
        parent_protocol=parent_protocol,
        root=root,
        source_split=source_split,
    )
    _require(not token.native_episodes.intersection(binding.native_episodes),
             "paired PushT plans share native episodes")
    return {token.cohort: token, binding.cohort: binding}


def _spec_seed_registry(spec: Mapping[str, Any]) -> Mapping[str, int]:
    value = spec.get("_seed_registry")
    if isinstance(value, Mapping):
        return value
    # Seed derivation is protocol metadata, not a development outcome.
    from scripts.sage_mem_v1_spec import seed_registry
    return seed_registry(spec)


def plan_pusht_formal_pair_from_spec(
        spec: Mapping[str, Any], *, root: Path = ROOT
        ) -> dict[str, FormalSelectionPlan]:
    """Inspect the authenticated native PushT catalog and plan both cohorts."""

    cohorts = spec.get("cohorts")
    _require(isinstance(cohorts, Mapping), "SAGE-Mem cohort registry missing")
    token = cohorts["dinowm_pusht_token"]
    binding = cohorts["dinowm_pusht_binding"]
    _require(token["parent_protocol"] == binding["parent_protocol"],
             "DINO PushT cohorts no longer share one parent protocol")
    forbidden = list(dict.fromkeys(
        list(token["forbidden_parent_artifacts"])
        + list(binding["forbidden_parent_artifacts"])))
    parent = token["parent_protocol"]
    registry = collect_parent_exclusion_registry(
        parent_protocol=parent,
        forbidden_parent_artifacts=forbidden,
        root=root,
    )
    cfg = yaml.safe_load(_repo_path(parent, root=root).read_text())
    activate_parent_dependency_environment(cfg, root=root)
    from scripts.run_dinowm_native_pusht_audit_v2 import OfficialDinoWMPushT
    dataset_cfg = cfg["dataset"]
    dataset = OfficialDinoWMPushT(
        _repo_path(dataset_cfg["root"], root=root),
        _repo_path(dataset_cfg["manifest_path"], root=root),
        str(dataset_cfg["manifest_identity"]["sha256"]),
    )
    source_split = str(dataset_cfg["source_split"])
    lengths = dataset.splits[source_split]["lengths"]
    return plan_pusht_formal_pair(
        split_counts_by_cohort={
            name: cohorts[name]["split_episodes"]
            for name in ("dinowm_pusht_token", "dinowm_pusht_binding")},
        seed_registry=_spec_seed_registry(spec),
        episode_lengths=lengths,
        parent_registry=registry,
        parent_protocol=parent,
        eligible_episodes=range(len(lengths)),
        source_split=source_split,
        root=root,
    )


def plan_pointmaze_formal_from_spec(
        spec: Mapping[str, Any], *, root: Path = ROOT
        ) -> FormalSelectionPlan:
    """Inspect the official native-train PointMaze split and plan fresh bases."""

    cohort = spec.get("cohorts", {}).get("dinowm_pointmaze_goal")
    _require(isinstance(cohort, Mapping),
             "PointMaze SAGE-Mem cohort registry missing")
    parent = cohort["parent_protocol"]
    registry = collect_parent_exclusion_registry(
        parent_protocol=parent,
        forbidden_parent_artifacts=cohort["forbidden_parent_artifacts"],
        root=root,
    )
    cfg = yaml.safe_load(_repo_path(parent, root=root).read_text())
    activate_parent_dependency_environment(cfg, root=root)
    from scripts.run_dinowm_pointmaze_wave3 import NativePointMazeData
    dataset = NativePointMazeData(cfg)
    native_train, _ = dataset.official_split()
    sequence = cfg["sequence"]
    return plan_fresh_formal_selection(
        cohort="dinowm_pointmaze_goal",
        split_counts=cohort["split_episodes"],
        seed_registry=_spec_seed_registry(spec),
        episode_lengths=dataset.lengths.tolist(),
        parent_registry=registry,
        eligible_episodes=native_train,
        parent_protocol=parent,
        root=root,
        source_split="train",
        num_frames=int(sequence["num_frames"]),
        frame_skip=int(sequence["native_frame_skip"]),
    )


def write_selection_receipt(plan: FormalSelectionPlan, path: str | Path) \
        -> dict[str, Any]:
    value = plan.public_receipt()
    _atomic_json(Path(path), value)
    return value


def _artifact(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _identity_record(path: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path, ROOT),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _parent_admission_proof(
        plan: FormalSelectionPlan, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the parent pre-carrier admission used by this cohort."""

    artifact_root = _repo_path(cfg["artifacts"]["root"])
    manifest_path = artifact_root / "cache" / "manifest.json"
    _require(manifest_path.is_file(),
             f"parent cache admission manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    proof: dict[str, Any] = {
        "parent_cache_manifest": _identity_record(manifest_path),
        "cohort": plan.cohort,
    }
    if plan.info["family"] == "pusht":
        admissions = manifest.get("admissions")
        _require(isinstance(admissions, Mapping),
                 "parent PushT admission registry is missing")
        tasks = admissions.get("tasks")
        task = (tasks.get(plan.info["parent_task"])
                if isinstance(tasks, Mapping) else None)
        rollout = admissions.get("rollout_health")
        _require(isinstance(task, Mapping) and task.get("admitted") is True,
                 f"parent task admission failed: {plan.cohort}")
        _require(isinstance(rollout, Mapping)
                 and rollout.get("admitted") is True,
                 "parent DINO PushT rollout-health admission failed")
        task_path = _repo_path(task["path"])
        rollout_path = _repo_path(rollout["path"])
        _require(sha256_file(task_path) == task.get("sha256")
                 and sha256_file(rollout_path) == rollout.get("sha256"),
                 "parent PushT admission artifact identity changed")
        task_value = json.loads(task_path.read_text())
        rollout_value = json.loads(rollout_path.read_text())
        _require(task_value.get("admitted") is True
                 and rollout_value.get("admitted") is True,
                 "parent PushT admission artifact is no longer admitted")
        proof.update({
            "status": "admitted",
            "task_admission": _identity_record(task_path),
            "rollout_health": _identity_record(rollout_path),
        })
        return proof

    _require(manifest.get("precarrier_gates_passed") is True,
             "parent PointMaze pre-carrier admission failed")
    admission_path = _repo_path(manifest["admission_path"])
    controller_path = _repo_path(manifest["controller_gate_path"])
    _require(sha256_file(admission_path) == manifest.get("admission_sha256")
             and sha256_file(controller_path)
             == manifest.get("controller_gate_sha256"),
             "parent PointMaze admission/controller identity changed")
    admission = json.loads(admission_path.read_text())
    controller = json.loads(controller_path.read_text())
    _require(admission.get("admitted") is True
             and controller.get("admitted") is True,
             "parent PointMaze admission or controller gate failed")
    proof.update({
        "status": "admitted",
        "task_admission": _identity_record(admission_path),
        "controller_gate": _identity_record(controller_path),
    })
    return proof


def _actual_overlap_proof(
        plan: FormalSelectionPlan, metadata_path: Path) -> dict[str, Any]:
    """Re-open materialized metadata and prove native-episode disjointness."""

    with np.load(metadata_path, allow_pickle=False) as archive:
        episode = np.asarray(archive["episode_index"], dtype=np.int64)
        split = np.asarray(archive["split"], dtype=np.uint8)
        start = np.asarray(archive["local_start"], dtype=np.int64)
        cluster = np.asarray(archive["native_cluster_id"], dtype=np.int64)
    _require(all(value.shape == episode.shape for value in (
        split, start, cluster)), "materialized identity arrays are unaligned")
    expected_episode = np.asarray(
        [row.episode_index for row in plan.rows], dtype=np.int64)
    expected_start = np.asarray(
        [row.local_start for row in plan.rows], dtype=np.int64)
    expected_split = np.asarray([
        FORMAL_SPLITS.index(row.split) for row in plan.rows], dtype=np.uint8)
    expected_cluster = np.asarray(
        [row.native_cluster_id for row in plan.rows], dtype=np.int64)
    _require(np.array_equal(episode, expected_episode)
             and np.array_equal(start, expected_start)
             and np.array_equal(split, expected_split)
             and np.array_equal(cluster, expected_cluster),
             "materialized identity registry differs from sealed selection")
    actual_parent_overlap = set(map(int, episode)).intersection(
        plan.parent_registry.episodes)
    _require(not actual_parent_overlap,
             "materialized bank overlaps an authenticated parent episode")
    split_native: dict[str, set[int]] = {}
    for code, name in enumerate(FORMAL_SPLITS):
        selected = episode[split == code]
        split_native[name] = set(map(int, selected))
        _require(len(split_native[name]) ==
                 int(plan.requested_counts[name]),
                 f"materialized native count changed: {name}")
        _require(len(selected) == int(plan.requested_counts[name])
                 * int(plan.info["variants_per_native_episode"]),
                 f"materialized expanded count changed: {name}")
    intersections = {
        f"{left}:{right}": sorted(split_native[left].intersection(
            split_native[right]))
        for left_index, left in enumerate(FORMAL_SPLITS)
        for right in FORMAL_SPLITS[left_index + 1:]
    }
    _require(not any(intersections.values()),
             "materialized formal splits share native episodes")
    return {
        "checked_from_artifact": metadata_path.name,
        "metadata_sha256": sha256_file(metadata_path),
        "parent_episode_overlap_count": 0,
        "cross_split_native_episode_overlap_count": 0,
        "split_native_episode_counts": {
            key: len(value) for key, value in split_native.items()},
        "selected_native_episode_union_sha256": sha256_json(
            sorted(set(map(int, episode)))),
        "selected_episode_start_union_sha256": sha256_json(sorted({
            (int(raw_episode), int(raw_start))
            for raw_episode, raw_start in zip(episode, start, strict=True)
        })),
        "native_cluster_registry_sha256": sha256_json(
            cluster.astype(np.int64).tolist()),
    }


def _label_vault_arrays(plan: FormalSelectionPlan) -> dict[str, np.ndarray]:
    return {
        "episode_id": np.asarray(
            [row.episode_id for row in plan.rows], dtype=np.int64),
        "class_id": np.asarray(
            [row.class_id for row in plan.rows], dtype=np.int64),
        "native_cluster_id": np.asarray(
            [row.native_cluster_id for row in plan.rows], dtype=np.int64),
    }


def _public_metadata(plan: FormalSelectionPlan,
                     base_index: np.ndarray) -> dict[str, np.ndarray]:
    split_code = {split: index for index, split in enumerate(FORMAL_SPLITS)}
    return {
        "split": np.asarray(
            [split_code[row.split] for row in plan.rows], dtype=np.uint8),
        "episode_index": np.asarray(
            [row.episode_index for row in plan.rows], dtype=np.int64),
        "local_start": np.asarray(
            [row.local_start for row in plan.rows], dtype=np.int64),
        "episode_id": np.asarray(
            [row.episode_id for row in plan.rows], dtype=np.int64),
        "native_cluster_id": np.asarray(
            [row.native_cluster_id for row in plan.rows], dtype=np.int64),
        "base_index": np.asarray(base_index, dtype=np.int64),
    }


def _materialize_pusht(
        plan: FormalSelectionPlan, cfg: Mapping[str, Any], staging: Path,
        *, progress: Callable[[str], None]) -> tuple[dict[str, Path], str, str]:
    from scripts.run_dinowm_native_pusht_audit_v1 import (
        _fixed_normalize_actions, _fixed_normalize_proprio,
    )
    from scripts.run_dinowm_native_pusht_audit_v2 import (
        NativeSelection, OfficialDinoWMPushT,
    )
    from scripts.run_dinowm_wave2_spatial_carrier import FrozenNativeHost
    from lewm.official_tasks.pusht_memory import render_single_overlay

    sequence = cfg["sequence"]
    dataset_cfg = cfg["dataset"]
    dataset = OfficialDinoWMPushT(
        _repo_path(dataset_cfg["root"]),
        _repo_path(dataset_cfg["manifest_path"]),
        str(dataset_cfg["manifest_identity"]["sha256"]),
    )
    selections = [NativeSelection(
        split=row.split,
        source_split=row.source_split,
        episode_index=row.episode_index,
        local_start=row.local_start,
        label=row.class_id,
    ) for row in plan.rows]
    count = len(selections)
    base_path = staging / "base_visual.npy"
    cue_path = staging / "cue_visual.npy"
    base = np.lib.format.open_memmap(
        base_path, mode="w+", dtype=np.float32,
        shape=(count, 20, 196, 384))
    cue = np.lib.format.open_memmap(
        cue_path, mode="w+", dtype=np.float32,
        shape=(count, 3, 196, 384))
    actions = np.empty((count, 19, 10), dtype=np.float32)
    proprio = np.empty((count, 20, 4), dtype=np.float32)
    host = FrozenNativeHost(cfg, load_encoder=True)
    before = str(host.digest())
    batch = int(cfg.get("cache", {}).get("build_episode_batch", 8))
    frame_batch = int(cfg.get("cache", {}).get("frame_batch_size", 64))
    task_name = str(plan.info["semantic_task"])
    for offset in range(0, count, batch):
        stop = min(count, offset + batch)
        selected = selections[offset:stop]
        native = [dataset.read(
            value, num_frames=int(sequence["num_frames"]),
            frame_skip=int(sequence["frame_skip"])) for value in selected]
        frames = np.stack([value.frames for value in native])
        frame_shape = frames.shape[2:]
        base[offset:stop] = host.encode_visual(
            frames.reshape(-1, *frame_shape), batch_size=frame_batch).reshape(
                len(native), 20, 196, 384)
        overlays = np.stack([
            render_single_overlay(
                value.frames, task_name, selection.label,
                int(sequence["cue_start"]), int(sequence["cue_length"]))[
                    int(sequence["cue_start"]):
                    int(sequence["cue_start"]) + int(sequence["cue_length"])]
            for value, selection in zip(native, selected, strict=True)
        ])
        cue[offset:stop] = host.encode_visual(
            overlays.reshape(-1, *frame_shape), batch_size=frame_batch).reshape(
                len(native), 3, 196, 384)
        actions[offset:stop] = _fixed_normalize_actions(
            np.stack([value.actions for value in native]))
        proprio[offset:stop] = _fixed_normalize_proprio(
            np.stack([value.proprio for value in native]))
        progress(f"[{plan.cohort}] encoded {stop}/{count}")
    base.flush()
    cue.flush()
    after = str(host.digest())
    _require(before == after, "fresh PushT cache build mutated frozen host")
    metadata_path = staging / "metadata.npz"
    metadata = _public_metadata(plan, np.arange(count, dtype=np.int64))
    metadata.update(base_actions=actions, base_proprio=proprio)
    _atomic_npz(metadata_path, metadata)
    del base, cue
    return ({"base_visual": base_path, "cue_visual": cue_path,
             "metadata": metadata_path}, before, after)


def _materialize_pointmaze(
        plan: FormalSelectionPlan, cfg: Mapping[str, Any], staging: Path,
        *, progress: Callable[[str], None]) -> tuple[dict[str, Path], str, str]:
    from lewm.official_tasks.dinowm_pointmaze import (
        MazeSelection, render_transient_goal_cue,
        verify_cue_only_counterfactual,
    )
    from scripts.run_dinowm_pointmaze_wave3 import (
        FrozenPointMazeHost, NativePointMazeData,
    )

    _require({row.source_split for row in plan.rows} == {"train"},
             "PointMaze formal bank must use the official native TRAIN split")

    unique: list[FormalSelectionRow] = []
    seen: set[int] = set()
    for row in plan.rows:
        if row.native_cluster_id not in seen:
            unique.append(row)
            seen.add(row.native_cluster_id)
    base_count = len(unique)
    base_lookup = {row.native_cluster_id: index
                   for index, row in enumerate(unique)}
    sequence_rows_by_cluster: dict[int, list[int]] = {}
    for sequence_index, row in enumerate(plan.rows):
        sequence_rows_by_cluster.setdefault(
            row.native_cluster_id, []).append(sequence_index)
    base_index = np.asarray(
        [base_lookup[row.native_cluster_id] for row in plan.rows],
        dtype=np.int64)
    base_path = staging / "base_visual.npy"
    cue_path = staging / "cue_visual.npy"
    base = np.lib.format.open_memmap(
        base_path, mode="w+", dtype=np.float32,
        shape=(base_count, 20, 196, 384))
    cue = np.lib.format.open_memmap(
        cue_path, mode="w+", dtype=np.float32,
        # Cue rows follow the expanded, label-hidden sequence order.  This
        # lets a label-free worker assemble inputs without opening the vault
        # or receiving a semantic class/variant index.
        shape=(len(plan.rows), 3, 196, 384))
    actions = np.empty((base_count, 19, 10), dtype=np.float32)
    proprio = np.empty((base_count, 20, 4), dtype=np.float32)
    states = np.empty((base_count, 20, 4), dtype=np.float32)
    dataset = NativePointMazeData(cfg)
    host = FrozenPointMazeHost(cfg, load_encoder=True)
    before = str(host.digest())
    batch = int(cfg.get("cache", {}).get("build_base_batch", 4))
    frame_batch = int(cfg.get("cache", {}).get("frame_batch_size", 64))
    for offset in range(0, base_count, batch):
        stop = min(base_count, offset + batch)
        selected = [MazeSelection(
            # ``split`` is the authenticated native parent partition, not the
            # SAGE-Mem formal split (formal_train/consumer_train/formal_test).
            split=row.source_split,
            episode_index=row.episode_index,
            local_start=row.local_start,
        ) for row in unique[offset:stop]]
        native = [dataset.read(value) for value in selected]
        frames = np.stack([value["frames"] for value in native])
        base[offset:stop] = host.encode_visual(
            frames.reshape(-1, 224, 224, 3),
            batch_size=frame_batch).reshape(stop - offset, 20, 196, 384)
        variants: list[np.ndarray] = []
        for value in native:
            rendered = np.stack([
                render_transient_goal_cue(value["frames"], label)
                for label in range(4)])
            audit = verify_cue_only_counterfactual(value["frames"], rendered)
            _require(bool(audit.get("passed")),
                     "PointMaze cue-only intervention failed")
            variants.append(rendered[:, 1:4])
        encoded_variants = host.encode_visual(
            np.stack(variants).reshape(-1, 224, 224, 3),
            batch_size=frame_batch).reshape(stop - offset, 4, 3, 196, 384)
        for local, base_row in enumerate(unique[offset:stop]):
            sequence_rows = sequence_rows_by_cluster[base_row.native_cluster_id]
            _require(len(sequence_rows) == 4,
                     "PointMaze native base lost a counterfactual variant")
            for sequence_row in sequence_rows:
                cue[sequence_row] = encoded_variants[
                    local, plan.rows[int(sequence_row)].class_id]
        actions[offset:stop] = np.stack([value["actions"] for value in native])
        proprio[offset:stop] = np.stack([value["proprio"] for value in native])
        states[offset:stop] = np.stack([value["state"] for value in native])
        progress(f"[{plan.cohort}] encoded {stop}/{base_count} native bases")
    base.flush()
    cue.flush()
    after = str(host.digest())
    _require(before == after,
             "fresh PointMaze cache build mutated frozen host")
    metadata_path = staging / "metadata.npz"
    metadata = _public_metadata(plan, base_index)
    metadata.update(base_actions=actions, base_proprio=proprio,
                    base_states=states)
    _atomic_npz(metadata_path, metadata)
    del base, cue
    return ({"base_visual": base_path, "cue_visual": cue_path,
             "metadata": metadata_path}, before, after)


def materialize_dino_formal_bank(
        plan: FormalSelectionPlan, destination: str | Path, *,
        label_vault_destination: str | Path,
        label_vault_receipt_destination: str | Path | None = None,
        progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Encode one planned bank with the released native reader and frozen host.

    This is the expensive GPU entry point.  It is never called on import and
    has no command-line auto-execution path.  The destination is atomically
    renamed only after every artifact and frozen-host invariant is hashed.
    """

    destination = Path(destination)
    label_vault_destination = Path(label_vault_destination)
    label_vault_receipt_destination = Path(
        label_vault_receipt_destination) if \
        label_vault_receipt_destination is not None else \
        label_vault_destination.with_name(
            f"{label_vault_destination.name}.custody.json")
    _require(not destination.exists(),
             f"refusing to overwrite formal bank: {destination}")
    _require(not label_vault_destination.exists(),
             f"refusing to overwrite formal label vault: "
             f"{label_vault_destination}")
    _require(not label_vault_receipt_destination.exists(),
             f"refusing to overwrite label custody receipt: "
             f"{label_vault_receipt_destination}")
    _require(label_vault_destination.resolve()
             != label_vault_receipt_destination.resolve(),
             "label vault and custody receipt must be distinct paths")
    for protected_path, label in (
            (label_vault_destination, "formal label vault"),
            (label_vault_receipt_destination, "label custody receipt")):
        try:
            protected_path.resolve().relative_to(destination.resolve())
        except ValueError:
            pass
        else:
            raise DinoFormalError(
                f"{label} must live outside the label-free bank")
    destination.parent.mkdir(parents=True, exist_ok=True)
    label_vault_destination.parent.mkdir(parents=True, exist_ok=True)
    label_vault_receipt_destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent))
    try:
        parent_path = _repo_path(plan.parent_protocol)
        _require(sha256_file(parent_path) == plan.parent_protocol_sha256,
                 "parent protocol changed after fresh selection")
        cfg = yaml.safe_load(parent_path.read_text())
        _require(isinstance(cfg, Mapping), "parent protocol is not a mapping")
        dependency_activation = activate_parent_dependency_environment(cfg)
        admission_proof = _parent_admission_proof(plan, cfg)
        selection_path = staging / "selection.json"
        _atomic_json(selection_path, plan.public_receipt())
        label_hash = _atomic_npz(
            label_vault_destination, _label_vault_arrays(plan))
        os.chmod(label_vault_destination, 0o400)
        if plan.info["family"] == "pusht":
            artifacts, before, after = _materialize_pusht(
                plan, cfg, staging, progress=progress)
        else:
            artifacts, before, after = _materialize_pointmaze(
                plan, cfg, staging, progress=progress)
        overlap_proof = _actual_overlap_proof(plan, artifacts["metadata"])
        artifacts.update(selection=selection_path)
        identities = {name: _artifact(path, root=staging)
                      for name, path in artifacts.items()}
        manifest = {
            "schema": "sage_mem_v1_dino_formal_bank_v1",
            "api_version": DINO_FORMAL_API_VERSION,
            "study": "sage-mem-v1",
            "status": "prepared",
            "cohort": plan.cohort,
            "plan_sha256": plan.plan_sha256,
            "parent_protocol_sha256": plan.parent_protocol_sha256,
            "parent_exclusion_registry_sha256":
                plan.parent_registry.registry_sha256,
            "host_hash_before": before,
            "host_hash_after": after,
            "host_unchanged": before == after,
            "dependency_activation": dependency_activation,
            "admission_proof": admission_proof,
            "freshness_proof": overlap_proof,
            "parent_episode_overlap":
                overlap_proof["parent_episode_overlap_count"],
            "cross_split_native_episode_overlap":
                overlap_proof["cross_split_native_episode_overlap_count"],
            "formal_outcomes_read": False,
            "semantic_labels_in_public_selection": False,
            "semantic_label_vault_inside_bank": False,
            "sealed_label_vault_sha256": label_hash,
            "labels_used_for_carrier_training": False,
            "split_count_unit": "native episode/base window",
            "splits": {split: plan.split_record(split)
                       for split in FORMAL_SPLITS},
            "artifacts": identities,
        }
        _atomic_json(staging / "manifest.json", manifest)
        custody = {
            "schema": "sage_mem_v1_dino_formal_label_custody_v1",
            "api_version": DINO_FORMAL_API_VERSION,
            "status": "sealed-for-post-grid-finalizer",
            "cohort": plan.cohort,
            "plan_sha256": plan.plan_sha256,
            "path": str(label_vault_destination.resolve()),
            "size": label_vault_destination.stat().st_size,
            "sha256": label_hash,
            "mode": oct(label_vault_destination.stat().st_mode & 0o777),
            "per_cell_api_access": False,
        }
        _atomic_json(label_vault_receipt_destination, custody)
        os.chmod(label_vault_receipt_destination, 0o400)
        custody_receipt = {
            "path": str(label_vault_receipt_destination.resolve()),
            "size": label_vault_receipt_destination.stat().st_size,
            "sha256": sha256_file(label_vault_receipt_destination),
        }
        for artifact in staging.iterdir():
            if artifact.is_file():
                os.chmod(artifact, 0o444)
        os.chmod(staging, 0o555)
        os.rename(staging, destination)
        return {
            "bank_manifest": manifest,
            "label_vault_custody": custody,
            "label_vault_custody_receipt": custody_receipt,
        }
    except BaseException:
        try:
            os.chmod(staging, 0o750)
            for artifact in staging.iterdir():
                if artifact.is_file():
                    os.chmod(artifact, 0o600)
        except FileNotFoundError:
            pass
        shutil.rmtree(staging, ignore_errors=True)
        try:
            label_vault_destination.unlink()
        except FileNotFoundError:
            pass
        try:
            label_vault_receipt_destination.unlink()
        except FileNotFoundError:
            pass
        raise


class DinoLabelFreeFormalBank:
    """Read-only label-free feature/trajectory view of a fresh formal bank.

    There is deliberately no label-opening method.  A future post-grid
    finalizer must authenticate the vault independently after every label-free
    cell has completed.
    """

    def __init__(self, root: str | Path, *, verify_artifacts: bool = True) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        _require(manifest_path.is_file(), f"formal bank manifest missing: {root}")
        self.manifest = json.loads(manifest_path.read_text())
        _require(self.manifest.get("schema") ==
                 "sage_mem_v1_dino_formal_bank_v1"
                 and self.manifest.get("status") == "prepared",
                 "formal bank manifest schema/status changed")
        self.cohort = str(self.manifest["cohort"])
        _require(self.cohort in DINO_COHORTS, "formal bank cohort changed")
        if verify_artifacts:
            for record in self.manifest["artifacts"].values():
                path = self.root / record["path"]
                _require(path.is_file() and path.stat().st_size == record["size"]
                         and sha256_file(path) == record["sha256"],
                         f"formal bank artifact identity failed: {path}")
        self.base_visual = np.load(
            self.root / self.manifest["artifacts"]["base_visual"]["path"],
            mmap_mode="r")
        self.cue_visual = np.load(
            self.root / self.manifest["artifacts"]["cue_visual"]["path"],
            mmap_mode="r")
        with np.load(
                self.root / self.manifest["artifacts"]["metadata"]["path"],
                allow_pickle=False) as metadata:
            self.split = np.asarray(metadata["split"], dtype=np.uint8)
            self.episode_index = np.asarray(
                metadata["episode_index"], dtype=np.int64)
            self.local_start = np.asarray(
                metadata["local_start"], dtype=np.int64)
            self.episode_ids = np.asarray(
                metadata["episode_id"], dtype=np.int64)
            self.native_cluster_ids = np.asarray(
                metadata["native_cluster_id"], dtype=np.int64)
            self.base_index = np.asarray(
                metadata["base_index"], dtype=np.int64)
            self.base_actions = np.asarray(
                metadata["base_actions"], dtype=np.float32)
            self.base_proprio = np.asarray(
                metadata["base_proprio"], dtype=np.float32)
            self.base_states = (
                np.asarray(metadata["base_states"], dtype=np.float32)
                if "base_states" in metadata.files else None)
        self.spatial = True
        self.count = len(self.split)
        _require(all(len(value) == self.count for value in (
            self.episode_index, self.local_start, self.episode_ids,
            self.native_cluster_ids, self.base_index)),
            "formal bank public metadata is unaligned")

    def indices(self, split: str) -> np.ndarray:
        _require(split in FORMAL_SPLITS, f"unknown split: {split}")
        return np.flatnonzero(self.split == FORMAL_SPLITS.index(split))

    def identity(self, split: str) -> dict[str, np.ndarray]:
        rows = self.indices(split)
        return {
            "row_index": rows.copy(),
            "episode_id": self.episode_ids[rows].copy(),
            "native_episode_index": self.episode_index[rows].copy(),
            "native_cluster_id": self.native_cluster_ids[rows].copy(),
            "local_start": self.local_start[rows].copy(),
        }

    def phase_a_identity_inputs(self) -> dict[str, np.ndarray]:
        """Emit the finalizer's label-free age-major identity matrices.

        The Phase-A worker adds intervention features/predictions and MSEs to
        these six arrays.  Semantic class IDs are intentionally absent.
        """

        result: dict[str, np.ndarray] = {}
        for split in ("formal_test", "consumer_train"):
            rows = self.indices(split)
            episode = self.episode_ids[rows]
            cluster = self.native_cluster_ids[rows]
            result[f"{split}_episode_id"] = np.tile(
                episode[None], (len(AGES), 1))
            result[f"{split}_native_cluster_id"] = np.tile(
                cluster[None], (len(AGES), 1))
            result[f"{split}_evidence_age"] = np.repeat(
                np.asarray(AGES, dtype=np.int64)[:, None], len(rows), axis=1)
        return result

    def phase_a_split_inputs(
            self, split: str, age: int) -> dict[str, np.ndarray]:
        """Return one label-free split/age deck for a Phase-A worker."""

        rows = self.indices(split)
        value = self.trajectory(rows)
        value.update({
            "evidence_age": np.full(len(rows), int(age), dtype=np.int64),
            "features": self.features(age, rows),
        })
        return value

    def features(self, age: int, indices: np.ndarray) -> np.ndarray:
        _require(int(age) in AGES, f"unregistered evidence age: {age}")
        rows = np.asarray(indices, dtype=np.int64)
        bases = self.base_index[rows]
        values = np.asarray(self.base_visual[bases], dtype=np.float32).copy()
        values[:, 1:4] = np.asarray(
            self.cue_visual[rows], dtype=np.float32)
        return values

    def actions(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.base_actions[self.base_index[np.asarray(indices, dtype=np.int64)]],
            dtype=np.float32)

    def proprio(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.base_proprio[self.base_index[np.asarray(indices, dtype=np.int64)]],
            dtype=np.float32)

    def states(self, indices: np.ndarray) -> np.ndarray:
        _require(self.base_states is not None,
                 "native states are available only for PointMaze")
        return np.asarray(
            self.base_states[self.base_index[np.asarray(indices, dtype=np.int64)]],
            dtype=np.float32)

    def trajectory(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Return label-free native trajectory inputs and stable identities."""

        rows = np.asarray(indices, dtype=np.int64)
        value = {
            "row_index": rows.copy(),
            "episode_id": self.episode_ids[rows].copy(),
            "native_episode_index": self.episode_index[rows].copy(),
            "native_cluster_id": self.native_cluster_ids[rows].copy(),
            "local_start": self.local_start[rows].copy(),
            "actions": self.actions(rows),
            "proprio": self.proprio(rows),
        }
        if self.base_states is not None:
            value["states"] = self.states(rows)
        return value

    def provenance_handle(self) -> dict[str, Any]:
        return {
            "api_version": DINO_FORMAL_API_VERSION,
            "cohort": self.cohort,
            "bank_root": str(self.root.resolve()),
            "manifest_sha256": sha256_file(self.root / "manifest.json"),
            "plan_sha256": self.manifest["plan_sha256"],
            "host_hash_before": self.manifest["host_hash_before"],
            "host_hash_after": self.manifest["host_hash_after"],
            "dependency_activation": dict(
                self.manifest["dependency_activation"]),
            "admission_proof": dict(self.manifest["admission_proof"]),
            "freshness_proof": dict(self.manifest["freshness_proof"]),
            "splits": dict(self.manifest["splits"]),
            "sealed_label_vault_sha256":
                self.manifest["sealed_label_vault_sha256"],
            "semantic_label_vault_inside_bank": False,
            "labels_accessible_through_handle": False,
        }


def validate_materialized_bank_provenance(
        root: str | Path, *, verify_artifacts: bool = True) -> dict[str, Any]:
    """Revalidate an immutable bank without opening its semantic label vault."""

    bank = DinoLabelFreeFormalBank(root, verify_artifacts=verify_artifacts)
    manifest = bank.manifest
    before, after = manifest.get("host_hash_before"), manifest.get(
        "host_hash_after")
    _require(isinstance(before, str) and len(before) == 64
             and before == after,
             "materialized bank host digest is missing or changed")
    admission = manifest.get("admission_proof")
    freshness = manifest.get("freshness_proof")
    dependency = manifest.get("dependency_activation")
    _require(isinstance(dependency, Mapping)
             and dependency.get("status") ==
             "activated-before-native-host-import",
             "materialized bank lacks pinned dependency activation proof")
    _require(isinstance(admission, Mapping)
             and admission.get("status") == "admitted",
             "materialized bank lacks an admitted parent proof")
    _require(isinstance(freshness, Mapping)
             and freshness.get("parent_episode_overlap_count") == 0
             and freshness.get("cross_split_native_episode_overlap_count") == 0,
             "materialized bank freshness proof failed")
    for split in FORMAL_SPLITS:
        record = manifest["splits"][split]
        rows = bank.indices(split)
        native = set(map(int, bank.episode_index[rows]))
        _require(len(native) == int(record["count"])
                 and len(rows) == int(record["expanded_sequence_count"]),
                 f"materialized split shape changed: {split}")
    return bank.provenance_handle()


def open_label_free_bank(
        root: str | Path, *, verify_artifacts: bool = True
        ) -> DinoLabelFreeFormalBank:
    """Open the only bank handle permitted during label-free cell execution."""

    return DinoLabelFreeFormalBank(root, verify_artifacts=verify_artifacts)


__all__ = [
    "DINO_FORMAL_API_VERSION",
    "DINO_COHORTS",
    "FORMAL_SPLITS",
    "AGES",
    "DinoFormalError",
    "ParentExclusionRegistry",
    "FormalSelectionRow",
    "FormalSelectionPlan",
    "activate_parent_dependency_environment",
    "collect_parent_exclusion_registry",
    "plan_fresh_formal_selection",
    "plan_pusht_formal_pair",
    "plan_pusht_formal_pair_from_spec",
    "plan_pointmaze_formal_from_spec",
    "write_selection_receipt",
    "materialize_dino_formal_bank",
    "DinoLabelFreeFormalBank",
    "validate_materialized_bank_provenance",
    "open_label_free_bank",
]
