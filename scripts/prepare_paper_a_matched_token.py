#!/usr/bin/env python3
"""Prepare fresh composite-token caches and formal admission receipts."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text, stable_json, write_npz_with_sidecar,
)
from lewm.official_tasks.matched_token import (  # noqa: E402
    balanced_token_labels, render_token_cue, validate_token_counterfactuals,
)
from lewm.official_tasks.native_sequence_hdf5 import (  # noqa: E402
    NativeSequenceHDF5, SequenceSelection, normalize_action_blocks,
)
from scripts.make_official_lewm_memory_data import collect_base  # noqa: E402
from scripts.paper_a_evidence_age import configure_determinism, fit_readout  # noqa: E402
from scripts.paper_a_matched_token_spec import (  # noqa: E402
    AGES, DEFAULT_SHA, DEFAULT_SPEC, HOSTS, load_locked_spec, output_path,
    resolve_input_path, validate_device, v1_excluded_episode_indices,
)
from scripts.prepare_paper_a_matched_host import _encode_stream, _load_host  # noqa: E402
from scripts.train_frozen_official_swap import state_digest  # noqa: E402


SPLITS = ("train", "validation")
BASE_SCHEMA = "paper_a_matched_token_base_cache_v1"
CUE_SCHEMA = "paper_a_matched_token_cue_cache_v1"
BASE_KEYS = ("z_base", "actions", "state", "episode_index", "local_start",
             "global_frame_indices")
CUE_KEYS = ("z_cue", "combination_label", "token_label", "location_label",
            "episode_index", "local_start", "cue_on", "cue_off")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, choices=HOSTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def host_cache_root(spec: dict, host: str) -> Path:
    return output_path(spec, "cache") / host


def base_cache_path(spec: dict, host: str, split: str) -> Path:
    return host_cache_root(spec, host) / "base" / f"{split}.npz"


def cue_cache_path(spec: dict, host: str, split: str, age: int) -> Path:
    return host_cache_root(spec, host) / "cue" / split / f"age-{age}.npz"


def host_manifest_path(spec: dict, host: str) -> Path:
    return host_cache_root(spec, host) / "manifest.json"


def _labels(spec: dict, split: str):
    return balanced_token_labels(
        int(spec["selection"][f"{split}_episodes"]),
        int(spec["selection"][f"{split}_label_seed"]))


def _action_statistics(actions: np.ndarray):
    raw = np.asarray(actions, dtype=np.float64).reshape(-1, 2)
    mean, std = raw.mean(0), raw.std(0, ddof=1)
    if not np.isfinite(raw).all() or np.any(std <= 0):
        raise ValueError("Reacher action statistics are invalid")
    return mean, std, np.full(2, len(raw), dtype=np.int64)


def _fresh_hdf_selections(dataset: NativeSequenceHDF5, spec: dict,
                           host: str) -> tuple[SequenceSelection, ...]:
    excluded = set(v1_excluded_episode_indices(spec, host))
    eligible = np.flatnonzero(dataset.episode_lengths >= 96)
    available = np.asarray(
        [value for value in eligible if int(value) not in excluded],
        dtype=np.int64)
    requested = 1680
    if len(available) < requested:
        raise RuntimeError(f"{host} lacks fresh eligible episodes")
    source = spec["inputs"][host]
    selected = np.random.default_rng(int(source["split_seed"])).permutation(
        available)[:requested]
    result = []
    cursor = 0
    for split, count in (("train", 1200), ("validation", 480)):
        for episode in selected[cursor:cursor + count]:
            maximum = int(dataset.episode_lengths[int(episode)] - 96)
            rng = np.random.default_rng(np.random.SeedSequence(
                [int(source["start_seed"]), int(episode)]))
            result.append(SequenceSelection(
                split=split, episode_index=int(episode),
                local_start=int(rng.integers(maximum + 1))))
        cursor += count
    if set(map(int, selected)).intersection(excluded):
        raise RuntimeError(f"{host} selection overlaps V1")
    return tuple(result)


def _reacher_arrays(spec: dict, model: torch.nn.Module,
                    device: torch.device):
    source = spec["inputs"]["reacher"]
    raw = {split: collect_base(
        int(spec["selection"][f"{split}_episodes"]), 20,
        int(source[f"{split}_base_seed"])) for split in SPLITS}
    mean, std, count = _action_statistics(raw["train"][1])
    base, cue = {}, {}
    checks = 0
    for split in SPLITS:
        frames, raw_actions, state = raw[split]
        episodes, labels = len(frames), _labels(spec, split)

        def base_stream() -> Iterator[tuple[int, np.ndarray]]:
            for row in range(episodes):
                for step in range(20):
                    yield row * 20 + step, frames[row, step]

        z_base = _encode_stream(
            model, base_stream(), episodes * 20,
            int(spec["cache"]["frame_batch_size"]), device).reshape(
                episodes, 20, 192)
        base[split] = {
            "z_base": z_base,
            "actions": normalize_action_blocks(raw_actions, mean, std),
            "state": np.asarray(state, dtype=np.float32),
            "episode_index": np.arange(episodes, dtype=np.int64),
            "local_start": np.zeros(episodes, dtype=np.int64),
            "global_frame_indices": np.arange(
                episodes * 20, dtype=np.int64).reshape(episodes, 20),
        }
        cue[split] = {}
        for age in AGES:
            cue_off, cue_on = 19 - age, 19 - age - 3
            validate_token_counterfactuals(frames[0], cue_on, 3)
            checks += 1

            def cue_stream() -> Iterator[tuple[int, np.ndarray]]:
                for row in range(episodes):
                    rendered = render_token_cue(
                        frames[row], int(labels.token[row]),
                        int(labels.location[row]), cue_on, 3)
                    for offset in range(3):
                        yield row * 3 + offset, rendered[cue_on + offset]

            z_cue = _encode_stream(
                model, cue_stream(), episodes * 3,
                int(spec["cache"]["frame_batch_size"]), device).reshape(
                    episodes, 3, 192)
            cue[split][age] = {
                "z_cue": z_cue, "combination_label": labels.combination,
                "token_label": labels.token, "location_label": labels.location,
                "episode_index": base[split]["episode_index"],
                "local_start": base[split]["local_start"],
                "cue_on": np.full(episodes, cue_on, dtype=np.int64),
                "cue_off": np.full(episodes, cue_off, dtype=np.int64),
            }
    return base, cue, {
        "source": "fresh dm_control reacher/easy replay",
        "freshness": {"v1_base_seeds": [20260741, 20260742],
                      "matched_token_base_seeds": [20260941, 20260942],
                      "v1_trajectories_reused": False},
        "raw_action_mean": mean.tolist(),
        "raw_action_std_ddof1": std.tolist(),
        "raw_action_count": count.tolist(),
        "counterfactual_checks": checks,
    }


def _hdf_arrays(spec: dict, host: str, model: torch.nn.Module,
                 device: torch.device):
    source, record = spec["inputs"][host], spec["inputs"][host]["dataset"]
    dataset = NativeSequenceHDF5(
        resolve_input_path(record), expected_sha256=record["sha256"],
        expected_size=int(record["size"]), state_key=source["state_key"])
    selected = _fresh_hdf_selections(dataset, spec, host)
    selections = {split: tuple(x for x in selected if x.split == split)
                  for split in SPLITS}
    excluded = set(v1_excluded_episode_indices(spec, host))
    mean, std, count = dataset.raw_action_statistics(ddof=1)
    base, cue, checks = {}, {}, 0
    for split in SPLITS:
        items, labels = selections[split], _labels(spec, split)
        episodes = len(items)
        frames = np.empty((episodes, 20, *dataset.pixel_shape), dtype=np.uint8)
        actions = np.empty((episodes, 19, 10), dtype=np.float32)
        state = np.empty((episodes, 20, dataset.state_dim), dtype=np.float32)
        global_indices = np.empty((episodes, 20), dtype=np.int64)
        for row, native in enumerate(dataset.read_sequences(items, 20)):
            frames[row] = native.frames
            actions[row] = normalize_action_blocks(native.actions, mean, std)
            state[row] = native.state
            global_indices[row] = native.global_frame_indices

        def base_stream() -> Iterator[tuple[int, np.ndarray]]:
            for row in range(episodes):
                for step in range(20):
                    yield row * 20 + step, frames[row, step]

        z_base = _encode_stream(
            model, base_stream(), episodes * 20,
            int(spec["cache"]["frame_batch_size"]), device).reshape(
                episodes, 20, 192)
        base[split] = {
            "z_base": z_base, "actions": actions, "state": state,
            "episode_index": np.asarray(
                [x.episode_index for x in items], dtype=np.int64),
            "local_start": np.asarray(
                [x.local_start for x in items], dtype=np.int64),
            "global_frame_indices": global_indices,
        }
        cue[split] = {}
        for age in AGES:
            cue_off, cue_on = 19 - age, 19 - age - 3
            validate_token_counterfactuals(frames[0], cue_on, 3)
            checks += 1

            def cue_stream() -> Iterator[tuple[int, np.ndarray]]:
                for row in range(episodes):
                    rendered = render_token_cue(
                        frames[row], int(labels.token[row]),
                        int(labels.location[row]), cue_on, 3)
                    for offset in range(3):
                        yield row * 3 + offset, rendered[cue_on + offset]

            z_cue = _encode_stream(
                model, cue_stream(), episodes * 3,
                int(spec["cache"]["frame_batch_size"]), device).reshape(
                    episodes, 3, 192)
            cue[split][age] = {
                "z_cue": z_cue, "combination_label": labels.combination,
                "token_label": labels.token, "location_label": labels.location,
                "episode_index": base[split]["episode_index"],
                "local_start": base[split]["local_start"],
                "cue_on": np.full(episodes, cue_on, dtype=np.int64),
                "cue_off": np.full(episodes, cue_off, dtype=np.int64),
            }
    selected_ids = {x.episode_index for x in selected}
    return base, cue, {
        "source": str(resolve_input_path(record)),
        "dataset_sha256": record["sha256"],
        "eligible_20_frame_episodes": int(np.sum(dataset.episode_lengths >= 96)),
        "raw_action_mean": mean.tolist(),
        "raw_action_std_ddof1": std.tolist(),
        "raw_action_count": count.tolist(),
        "counterfactual_checks": checks,
        "v1_fresh_data_exclusion": {
            "locked_v1_episode_count": len(excluded),
            "selected_episode_count": len(selected_ids),
            "overlap_count": len(selected_ids.intersection(excluded)),
            "zero_overlap_proven": not bool(selected_ids.intersection(excluded)),
        },
    }


def _full_latent(base: dict, cue: dict) -> np.ndarray:
    value = np.asarray(base["z_base"], dtype=np.float32).copy()
    start, stop = int(cue["cue_on"][0]), int(cue["cue_off"][0])
    value[:, start:stop] = cue["z_cue"]
    return value


def _probe(train_x, train_y, val_x, val_y, spec):
    result = fit_readout(
        train_x, train_y, val_x, val_y, spec["readout"], balanced=True)
    prediction = np.asarray(result.pop("prediction"), dtype=np.int64)
    result.pop("correct")
    matrix = confusion_matrix(val_y, prediction, labels=np.arange(4))
    recall = np.diag(matrix) / np.maximum(matrix.sum(1), 1)
    result["per_class_recall"] = recall.tolist()
    result["min_class_recall"] = float(recall.min())
    return result


def _admission(spec: dict, base: dict, cue: dict):
    result, passed = {}, True
    for age in AGES:
        train_cue, val_cue = cue["train"][age], cue["validation"][age]
        train_z = _full_latent(base["train"], train_cue)
        val_z = _full_latent(base["validation"], val_cue)
        train_y, val_y = train_cue["token_label"], val_cue["token_label"]
        probes = {
            "cue_probe": _probe(
                train_cue["z_cue"].reshape(len(train_y), -1), train_y,
                val_cue["z_cue"].reshape(len(val_y), -1), val_y, spec),
            "final_context_latent_shortcut": _probe(
                train_z[:, 16:19].reshape(len(train_y), -1), train_y,
                val_z[:, 16:19].reshape(len(val_y), -1), val_y, spec),
            "final_action_shortcut": _probe(
                base["train"]["actions"][:, 15:19].reshape(len(train_y), -1),
                train_y,
                base["validation"]["actions"][:, 15:19].reshape(len(val_y), -1),
                val_y, spec),
            "final_state_shortcut": _probe(
                base["train"]["state"][:, 19], train_y,
                base["validation"]["state"][:, 19], val_y, spec),
        }
        admitted = (
            probes["cue_probe"]["balanced_accuracy"] >= 0.75
            and probes["cue_probe"]["min_class_recall"] >= 0.70
            and max(probes[key]["balanced_accuracy"] for key in probes
                    if key != "cue_probe") <= 0.30)
        result[f"age-{age}"] = {
            "admitted": bool(admitted), "chance": 0.25,
            "target": "composite token identity",
            "location_role": "exact-balanced randomized nuisance",
            "cue_balanced_accuracy_min": 0.75,
            "cue_min_class_recall_min": 0.70,
            "shortcut_ceiling": 0.30, **probes,
        }
        passed = passed and admitted
    return result, bool(passed)


def _write(spec: dict, host: str, base: dict, cue: dict):
    records = []
    for split in SPLITS:
        records.append(write_npz_with_sidecar(
            base_cache_path(spec, host, split), base[split], {
                "schema": BASE_SCHEMA, "study": spec["study"],
                "lock": spec["_lock"], "host": host, "split": split,
                "num_frames": 20, "decision_index": 19,
                "fresh_from_v1": True, "label_independent_base": True,
            }, compression_level=int(spec["cache"]["compression_level"])))
        for age in AGES:
            records.append(write_npz_with_sidecar(
                cue_cache_path(spec, host, split, age), cue[split][age], {
                    "schema": CUE_SCHEMA, "study": spec["study"],
                    "lock": spec["_lock"], "host": host, "split": split,
                    "age": age, "decision_index": 19, "cue_length": 3,
                    "target": "token", "nuisance": "location",
                }, compression_level=int(spec["cache"]["compression_level"])))
    return records


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing matched-token writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=True)
    validate_device(spec, args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("matched-token prep requires GPU0")
    configure_determinism(0)
    device = torch.device(args.device)
    manifest_path = host_manifest_path(spec, args.host)
    stop_path = host_cache_root(spec, args.host) / "prerequisite-stopped.json"
    if manifest_path.exists() or stop_path.exists():
        raise FileExistsError("matched-token host already resolved")
    try:
        model = _load_host(spec, args.host, device)
        before = state_digest(model)
        if args.host == "reacher":
            base, cue, provenance = _reacher_arrays(spec, model, device)
        else:
            base, cue, provenance = _hdf_arrays(spec, args.host, model, device)
        after = state_digest(model)
        if before != after:
            raise RuntimeError("matched-token encoding changed frozen host")
        admission, admitted = _admission(spec, base, cue)
        artifacts = _write(spec, args.host, base, cue)
        payload = {
            "schema_version": 1, "study": spec["study"],
            "branch": "adaptive-matched-composite-token",
            "lock": spec["_lock"], "host": args.host,
            "status": "admitted" if admitted else "stopped-admission-failure",
            "all_token_ages_admitted": admitted,
            "target": "token", "nuisance": "location",
            "physical_gpu": 0, "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "frozen_host_sha256_before": before,
            "frozen_host_sha256_after": after,
            "frozen_host_unchanged": True,
            "provenance": provenance, "admission": admission,
            "artifacts": artifacts,
        }
        atomic_text(manifest_path, stable_json(payload))
        if not admitted:
            raise SystemExit(f"matched-token {args.host} failed admission")
        print(f"[matched-token] admitted {manifest_path}", flush=True)
    except SystemExit:
        raise
    except BaseException as error:
        atomic_text(stop_path, stable_json({
            "schema_version": 1, "study": spec["study"],
            "lock": spec["_lock"], "host": args.host,
            "status": "stopped-prerequisite-failure",
            "error_type": type(error).__name__, "error": str(error),
            "traceback_tail": traceback.format_exc().splitlines()[-16:],
        }))
        raise


if __name__ == "__main__":
    main()
