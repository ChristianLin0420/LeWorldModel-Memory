#!/usr/bin/env python3
"""Prepare fresh adaptive Wave 1.1 color-only caches and admission receipts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    stable_json,
    write_npz_with_sidecar,
)
from lewm.official_tasks.matched_memory import (  # noqa: E402
    MatchedLabels,
    balanced_joint_labels,
    render_joint_cue,
    validate_joint_counterfactuals,
)
from lewm.official_tasks.native_sequence_hdf5 import (  # noqa: E402
    NativeSequenceHDF5,
    SequenceSelection,
    normalize_action_blocks,
)
from scripts.make_official_lewm_memory_data import collect_base  # noqa: E402
from scripts.paper_a_evidence_age import configure_determinism  # noqa: E402
from scripts.paper_a_matched_color_v1_1_spec import (  # noqa: E402
    AGES,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    load_locked_spec,
    output_path,
    resolve_input_path,
    prior_excluded_episode_indices,
    validate_device,
)
from scripts.prepare_paper_a_matched_host import (  # noqa: E402
    _encode_stream,
    _full_latent,
    _load_host,
    _normalize_generated,
    _probe,
    _raw_action_statistics,
)
from scripts.train_frozen_official_swap import state_digest  # noqa: E402


SPLITS = ("train", "validation")
BASE_SCHEMA = "paper_a_matched_color_v1_1_base_cache_v1"
CUE_SCHEMA = "paper_a_matched_color_v1_1_cue_cache_v1"
BASE_KEYS = (
    "z_base", "actions", "state", "episode_index", "local_start",
    "global_frame_indices",
)
CUE_KEYS = (
    "z_cue", "combination_label", "color_label", "location_label",
    "episode_index", "local_start", "cue_on", "cue_off",
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, choices=HOSTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def host_cache_root(spec: dict[str, Any], host: str) -> Path:
    return output_path(spec, "cache") / host


def base_cache_path(spec: dict[str, Any], host: str, split: str) -> Path:
    return host_cache_root(spec, host) / "base" / f"{split}.npz"


def cue_cache_path(spec: dict[str, Any], host: str, split: str,
                   age: int) -> Path:
    return host_cache_root(spec, host) / "cue" / split / f"age-{age}.npz"


def host_manifest_path(spec: dict[str, Any], host: str) -> Path:
    return host_cache_root(spec, host) / "manifest.json"


def _split_labels(spec: dict[str, Any], split: str) -> MatchedLabels:
    count = int(spec["selection"][f"{split}_episodes"])
    seed = int(spec["selection"][f"{split}_label_seed"])
    return balanced_joint_labels(count, seed)


def _fresh_hdf_selections(
        dataset: NativeSequenceHDF5, spec: dict[str, Any], host: str
        ) -> tuple[SequenceSelection, ...]:
    """Select only episodes outside both prior cache unions locked at seal."""

    source = spec["inputs"][host]
    num_frames = int(spec["sequence"]["num_frames"])
    raw_span = (num_frames - 1) * dataset.frame_skip + 1
    eligible = np.flatnonzero(dataset.episode_lengths >= raw_span)
    excluded = np.asarray(
        prior_excluded_episode_indices(spec, host), dtype=np.int64)
    if len(excluded) and (excluded.min() < 0 or
                          excluded.max() >= dataset.num_episodes):
        raise RuntimeError(f"{host} locked prior exclusion is outside dataset")
    if len(excluded) and not np.isin(excluded, eligible).all():
        raise RuntimeError(f"{host} locked prior cache contains ineligible episodes")
    available = eligible[~np.isin(eligible, excluded)]
    split_counts = tuple(
        (split, int(spec["selection"][f"{split}_episodes"]))
        for split in SPLITS)
    requested = sum(count for _, count in split_counts)
    if len(available) < requested:
        raise RuntimeError(
            f"{host} has {len(available)} fresh eligible episodes for "
            f"{requested} requests after excluding {len(excluded)} prior episodes")
    selected_episodes = np.random.default_rng(
        int(source["split_seed"])).permutation(available)[:requested]
    result: list[SequenceSelection] = []
    cursor = 0
    for split, count in split_counts:
        for episode in selected_episodes[cursor:cursor + count]:
            maximum = int(dataset.episode_lengths[int(episode)] - raw_span)
            rng = np.random.default_rng(np.random.SeedSequence(
                [int(source["start_seed"]), int(episode)]))
            result.append(SequenceSelection(
                split=split, episode_index=int(episode),
                local_start=int(rng.integers(maximum + 1))))
        cursor += count
    selected_set = {item.episode_index for item in result}
    overlap = sorted(selected_set.intersection(map(int, excluded)))
    if overlap:
        raise RuntimeError(f"{host} fresh-data exclusion failed: {overlap[:8]}")
    if len(selected_set) != requested:
        raise RuntimeError(f"{host} selection is not episode-disjoint")
    return tuple(result)


def _reacher_arrays(spec: dict[str, Any], model: torch.nn.Module,
                    device: torch.device
                    ) -> tuple[dict[str, dict[str, np.ndarray]],
                               dict[str, dict[int, dict[str, np.ndarray]]],
                               dict[str, Any]]:
    source = spec["inputs"]["reacher"]
    length = int(spec["sequence"]["num_frames"])
    frame_batch = int(spec["cache"]["frame_batch_size"])
    raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        count = int(spec["selection"][f"{split}_episodes"])
        raw[split] = collect_base(
            count, length, int(source[f"{split}_base_seed"]))
    mean, std, count = _raw_action_statistics(raw["train"][1])

    base: dict[str, dict[str, np.ndarray]] = {}
    cue: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    checks = 0
    for split in SPLITS:
        frames, raw_actions, state = raw[split]
        episodes = len(frames)
        labels = _split_labels(spec, split)

        def base_stream() -> Iterator[tuple[int, np.ndarray]]:
            for row in range(episodes):
                for step in range(length):
                    yield row * length + step, frames[row, step]

        z_base = _encode_stream(
            model, base_stream(), episodes * length, frame_batch,
            device).reshape(episodes, length, 192)
        base[split] = {
            "z_base": z_base,
            "actions": _normalize_generated(raw_actions, mean, std),
            "state": np.asarray(state, dtype=np.float32),
            "episode_index": np.arange(episodes, dtype=np.int64),
            "local_start": np.zeros(episodes, dtype=np.int64),
            "global_frame_indices": np.arange(
                episodes * length, dtype=np.int64).reshape(episodes, length),
        }
        cue[split] = {}
        for age in AGES:
            cue_off = int(spec["sequence"]["decision_index"]) - age
            cue_on = cue_off - int(spec["sequence"]["cue_length"])
            validate_joint_counterfactuals(
                frames[0], cue_on, int(spec["sequence"]["cue_length"]))
            checks += 1

            def cue_stream() -> Iterator[tuple[int, np.ndarray]]:
                for row in range(episodes):
                    rendered = render_joint_cue(
                        frames[row], int(labels.color[row]),
                        int(labels.location[row]), cue_on,
                        int(spec["sequence"]["cue_length"]))
                    for offset in range(int(spec["sequence"]["cue_length"])):
                        yield (row * int(spec["sequence"]["cue_length"]) + offset,
                               rendered[cue_on + offset])

            z_cue = _encode_stream(
                model, cue_stream(),
                episodes * int(spec["sequence"]["cue_length"]),
                frame_batch, device).reshape(
                    episodes, int(spec["sequence"]["cue_length"]), 192)
            cue[split][age] = {
                "z_cue": z_cue,
                "combination_label": labels.combination,
                "color_label": labels.color,
                "location_label": labels.location,
                "episode_index": base[split]["episode_index"],
                "local_start": base[split]["local_start"],
                "cue_on": np.full(episodes, cue_on, dtype=np.int64),
                "cue_off": np.full(episodes, cue_off, dtype=np.int64),
            }
    provenance = {
        "source": "fresh dm_control reacher/easy replay",
        "freshness": {
            "prior_seed_pairs": source["prior_seed_pairs"],
            "wave1_1_base_seeds": [
                int(source["train_base_seed"]),
                int(source["validation_base_seed"])],
            "new_vs_all_prior_seeds_disjoint": True,
            "seed_registry_sha256": spec["_lock"]["implementation"][
                "reacher_rng_exclusion"]["registry_sha256"],
            "prior_trajectories_reused": False,
            "wave1_trajectories_reused": False,
        },
        "raw_action_mean": mean.tolist(),
        "raw_action_std_ddof1": std.tolist(),
        "raw_action_count": count.tolist(),
        "exhaustive_16_way_counterfactual_checks": checks,
    }
    return base, cue, provenance


def _hdf_arrays(spec: dict[str, Any], host: str,
                 model: torch.nn.Module, device: torch.device
                 ) -> tuple[dict[str, dict[str, np.ndarray]],
                            dict[str, dict[int, dict[str, np.ndarray]]],
                            dict[str, Any]]:
    source = spec["inputs"][host]
    dataset_record = source["dataset"]
    dataset = NativeSequenceHDF5(
        resolve_input_path(dataset_record),
        expected_sha256=dataset_record["sha256"],
        expected_size=int(dataset_record["size"]),
        state_key=source["state_key"])
    selected = _fresh_hdf_selections(dataset, spec, host)
    selections = {
        split: tuple(value for value in selected if value.split == split)
        for split in SPLITS}
    excluded = set(prior_excluded_episode_indices(spec, host))
    selected_indices = {value.episode_index for value in selected}
    overlap = sorted(selected_indices.intersection(excluded))
    if overlap:
        raise RuntimeError(f"{host} overlaps locked prior-screen data")

    mean, std, count = dataset.raw_action_statistics(ddof=1)
    frame_batch = int(spec["cache"]["frame_batch_size"])
    length = int(spec["sequence"]["num_frames"])
    base: dict[str, dict[str, np.ndarray]] = {}
    cue: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    checks = 0
    for split in SPLITS:
        items = selections[split]
        episodes = len(items)
        labels = _split_labels(spec, split)
        frames = np.empty(
            (episodes, length, *dataset.pixel_shape), dtype=np.uint8)
        actions = np.empty((episodes, length - 1, 10), dtype=np.float32)
        state = np.empty((episodes, length, dataset.state_dim), dtype=np.float32)
        global_indices = np.empty((episodes, length), dtype=np.int64)
        for row, native in enumerate(dataset.read_sequences(items, length)):
            frames[row] = native.frames
            actions[row] = normalize_action_blocks(native.actions, mean, std)
            state[row] = native.state
            global_indices[row] = native.global_frame_indices

        def base_stream() -> Iterator[tuple[int, np.ndarray]]:
            for row in range(episodes):
                for step, frame in enumerate(frames[row]):
                    yield row * length + step, frame

        z_base = _encode_stream(
            model, base_stream(), episodes * length, frame_batch,
            device).reshape(episodes, length, 192)
        if len(np.unique(global_indices)) != episodes * length:
            raise RuntimeError(f"{host}/{split} duplicates source frames")
        base[split] = {
            "z_base": z_base, "actions": actions, "state": state,
            "episode_index": np.asarray(
                [value.episode_index for value in items], dtype=np.int64),
            "local_start": np.asarray(
                [value.local_start for value in items], dtype=np.int64),
            "global_frame_indices": global_indices,
        }
        cue[split] = {}
        for age in AGES:
            cue_off = int(spec["sequence"]["decision_index"]) - age
            cue_on = cue_off - int(spec["sequence"]["cue_length"])
            validate_joint_counterfactuals(
                frames[0], cue_on, int(spec["sequence"]["cue_length"]))
            checks += 1

            def cue_stream() -> Iterator[tuple[int, np.ndarray]]:
                for row in range(episodes):
                    rendered = render_joint_cue(
                        frames[row], int(labels.color[row]),
                        int(labels.location[row]), cue_on,
                        int(spec["sequence"]["cue_length"]))
                    for offset in range(int(spec["sequence"]["cue_length"])):
                        yield (row * int(spec["sequence"]["cue_length"]) + offset,
                               rendered[cue_on + offset])

            z_cue = _encode_stream(
                model, cue_stream(),
                episodes * int(spec["sequence"]["cue_length"]),
                frame_batch, device).reshape(
                    episodes, int(spec["sequence"]["cue_length"]), 192)
            cue[split][age] = {
                "z_cue": z_cue,
                "combination_label": labels.combination,
                "color_label": labels.color,
                "location_label": labels.location,
                "episode_index": base[split]["episode_index"],
                "local_start": base[split]["local_start"],
                "cue_on": np.full(episodes, cue_on, dtype=np.int64),
                "cue_off": np.full(episodes, cue_off, dtype=np.int64),
            }
    locked_exclusion = spec["_lock"]["implementation"][
        "prior_hdf_exclusions"][host]
    provenance = {
        "source": str(resolve_input_path(dataset_record)),
        "dataset_sha256": dataset_record["sha256"],
        "dataset_size": dataset_record["size"],
        "dataset_rows": dataset.num_rows,
        "dataset_episodes": dataset.num_episodes,
        "eligible_20_frame_episodes": int(np.sum(
            dataset.episode_lengths >= 96)),
        "pixel_shape": list(dataset.pixel_shape),
        "pixel_filters": list(dataset.pixel_filters),
        "state_key": source["state_key"], "state_dim": dataset.state_dim,
        "raw_action_mean": mean.tolist(),
        "raw_action_std_ddof1": std.tolist(),
        "raw_action_count": count.tolist(),
        "exhaustive_16_way_counterfactual_checks": checks,
        "prior_screen_fresh_data_exclusion": {
            "policy": spec["adaptive_origin"]["hdf_exclusion_policy"],
            "locked_prior_episode_count": len(excluded),
            "locked_prior_indices_sha256": locked_exclusion["indices_sha256"],
            "selected_episode_count": len(selected_indices),
            "overlap_count": len(overlap),
            "overlap_episode_indices": overlap,
            "zero_overlap_proven": len(overlap) == 0,
            "prior_cache_receipts": locked_exclusion["cache_candidates"],
            "prior_host_manifests": locked_exclusion["host_manifests"],
        },
        # Compatibility name consumed by the reused, implementation-locked
        # grid auditor; its content explicitly names both prior screens.
        "v1_fresh_data_exclusion": {
            "zero_overlap_proven": len(overlap) == 0,
        },
    }
    return base, cue, provenance


def _admission(spec: dict[str, Any],
               base: dict[str, dict[str, np.ndarray]],
               cue: dict[str, dict[int, dict[str, np.ndarray]]]
               ) -> tuple[dict[str, Any], bool]:
    threshold = float(spec["admission"]["cue_balanced_accuracy_min"])
    recall_threshold = float(spec["admission"]["cue_min_class_recall_min"])
    shortcut_ceiling = float(spec["admission"]["shortcut_ceiling"])
    endpoint = int(spec["sequence"]["decision_index"])
    history = int(spec["sequence"]["endpoint_history"])
    result: dict[str, Any] = {}
    passed = True
    for age in AGES:
        train_cue, validation_cue = cue["train"][age], cue["validation"][age]
        train_z = _full_latent(base["train"], train_cue)
        validation_z = _full_latent(base["validation"], validation_cue)
        train_y = train_cue["color_label"]
        validation_y = validation_cue["color_label"]
        cue_probe = _probe(
            train_cue["z_cue"].reshape(len(train_y), -1), train_y,
            validation_cue["z_cue"].reshape(len(validation_y), -1),
            validation_y, spec)
        latent = _probe(
            train_z[:, endpoint - history:endpoint].reshape(len(train_y), -1),
            train_y,
            validation_z[:, endpoint - history:endpoint].reshape(
                len(validation_y), -1), validation_y, spec)
        action = _probe(
            base["train"]["actions"][:, endpoint - 4:endpoint].reshape(
                len(train_y), -1), train_y,
            base["validation"]["actions"][:, endpoint - 4:endpoint].reshape(
                len(validation_y), -1), validation_y, spec)
        state = _probe(
            base["train"]["state"][:, endpoint], train_y,
            base["validation"]["state"][:, endpoint], validation_y, spec)
        admitted = (
            cue_probe["balanced_accuracy"] >= threshold
            and cue_probe["min_class_recall"] >= recall_threshold
            and max(latent["balanced_accuracy"], action["balanced_accuracy"],
                    state["balanced_accuracy"]) <= shortcut_ceiling)
        result[f"age-{age}"] = {
            "color": {
                "admitted": bool(admitted), "chance": 0.25,
                "location_role": "exact-balanced randomized nuisance",
                "requirement_by_construction": (
                    "color is randomized after fresh base selection; location "
                    "is a balanced nuisance; only registered cue pixels change"),
                "cue_balanced_accuracy_min": threshold,
                "cue_min_class_recall_min": recall_threshold,
                "shortcut_ceiling": shortcut_ceiling,
                "cue_probe": cue_probe,
                "final_context_latent_shortcut": latent,
                "final_action_shortcut": action,
                "final_state_shortcut": state,
            }
        }
        passed = passed and bool(admitted)
    return result, passed


def _validate_array_contract(
        base: dict[str, dict[str, np.ndarray]],
        cue: dict[str, dict[int, dict[str, np.ndarray]]]) -> None:
    for split in SPLITS:
        if tuple(base[split]) != BASE_KEYS:
            raise RuntimeError(f"{split} base key contract changed")
        for age in AGES:
            if tuple(cue[split][age]) != CUE_KEYS:
                raise RuntimeError(f"{split}/age-{age} cue key contract changed")
            if not np.array_equal(
                    base[split]["episode_index"],
                    cue[split][age]["episode_index"]):
                raise RuntimeError("base/cue episode alignment changed")
            labels = cue[split][age]
            if not np.array_equal(
                    labels["combination_label"],
                    labels["color_label"] * 4 + labels["location_label"]):
                raise RuntimeError("joint nuisance labels are inconsistent")


def _write_caches(spec: dict[str, Any], host: str,
                  base: dict[str, dict[str, np.ndarray]],
                  cue: dict[str, dict[int, dict[str, np.ndarray]]]
                  ) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in SPLITS:
        records.append(write_npz_with_sidecar(
            base_cache_path(spec, host, split), base[split], {
                "schema": BASE_SCHEMA, "study": spec["study"],
                "lock": spec["_lock"], "host": host, "split": split,
                "num_frames": 20, "decision_index": 19,
                "label_independent_base": True,
                "fresh_never_opened_selection": True,
            }, compression_level=int(spec["cache"]["compression_level"])))
        for age in AGES:
            records.append(write_npz_with_sidecar(
                cue_cache_path(spec, host, split, age), cue[split][age], {
                    "schema": CUE_SCHEMA, "study": spec["study"],
                    "lock": spec["_lock"], "host": host, "split": split,
                    "age": age, "decision_index": 19, "cue_length": 3,
                    "target": "color",
                    "location_role": "exact-balanced randomized nuisance",
                }, compression_level=int(spec["cache"]["compression_level"])))
    return records


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing adaptive matched-color writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=True)
    validate_device(spec, args.device)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "0"):
        raise RuntimeError("Wave 1.1 preparation requires physical GPU 0")
    if not torch.cuda.is_available():
        raise RuntimeError("adaptive matched-color preparation requires GPU 0")
    configure_determinism(0)
    device = torch.device(args.device)
    manifest_path = host_manifest_path(spec, args.host)
    stop_path = host_cache_root(spec, args.host) / "prerequisite-stopped.json"
    if manifest_path.exists() or stop_path.exists():
        raise FileExistsError(f"Wave 1.1 host branch already resolved: {args.host}")
    try:
        model = _load_host(spec, args.host, device)
        host_before = state_digest(model)
        if args.host == "reacher":
            base, cue, provenance = _reacher_arrays(spec, model, device)
        else:
            base, cue, provenance = _hdf_arrays(
                spec, args.host, model, device)
        _validate_array_contract(base, cue)
        host_after = state_digest(model)
        if host_before != host_after:
            raise RuntimeError("frozen host changed during Wave 1.1 encoding")
        admission, admitted = _admission(spec, base, cue)
        records = _write_caches(spec, args.host, base, cue)
        payload = {
            "schema_version": 1, "study": spec["study"],
            "branch": "admission-informed-matched-color-v1-1",
            "adaptive_origin": {
                "deterministic_selection_rule": spec["adaptive_origin"][
                    "deterministic_selection_rule"],
                "selected_target": "color", "retained_nuisance": "location",
                "prior_admission_metrics_used_for_wave1_1_inference": False,
                "prior_carrier_outcomes_observed": False,
                "preserve_both_prior_failures": True,
                "limitation": spec["adaptive_origin"]["limitation"],
            },
            "lock": spec["_lock"], "host": args.host,
            "status": "admitted" if admitted else "stopped-admission-failure",
            "all_color_ages_admitted": bool(admitted),
            "target": "color",
            "location_role": "exact-balanced randomized nuisance",
            "requirement_by_construction": True,
            "device": str(device), "physical_gpu": 0,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "frozen_host_sha256_before": host_before,
            "frozen_host_sha256_after": host_after,
            "frozen_host_unchanged": True,
            "released_checkpoint_predictor_history": 3,
            "history_contract_source": "authenticated released config.json",
            "provenance": provenance, "admission": admission,
            "artifacts": records,
        }
        atomic_text(manifest_path, stable_json(payload))
        if not admitted:
            raise SystemExit(
                f"adaptive matched-color {args.host} failed formal admission")
        print(f"[matched-color-v1.1] admitted {manifest_path}", flush=True)
    except SystemExit:
        raise
    except BaseException as error:
        atomic_text(stop_path, stable_json({
            "schema_version": 1, "study": spec["study"],
            "lock": spec["_lock"], "host": args.host,
            "status": "stopped-prerequisite-failure",
            "device": str(device), "physical_gpu": 0,
            "error_type": type(error).__name__, "error": str(error),
            "traceback_tail": traceback.format_exc().splitlines()[-16:],
        }))
        raise


if __name__ == "__main__":
    main()


__all__ = [
    "BASE_KEYS", "BASE_SCHEMA", "CUE_KEYS", "CUE_SCHEMA",
    "base_cache_path", "cue_cache_path", "host_manifest_path",
]
