#!/usr/bin/env python3
"""Prepare paired strict fixed-endpoint cue-offset caches and admissions."""

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

from lewm.models.official_lewm import load_official_reacher_checkpoint
from lewm.official_tasks.artifacts import (
    atomic_text,
    sha256_array,
    stable_json,
    write_npz_with_sidecar,
)
from lewm.official_tasks.pusht_admission import (
    PushTAdmissionThresholds,
    evaluate_pusht_admission,
)
from lewm.official_tasks.pusht_memory import render_single_overlay
from lewm.official_tasks.pusht_pipeline import (
    load_pusht_base_cache,
    pusht_task_spec,
)
from lewm.official_tasks.pusht_spec import load_locked_pusht_spec
from lewm.tasks_v19 import make_task
from lewm.tasks_v19.base import EpisodeBatch, load_bank
from scripts.cache_official_lewm import (
    action_statistics,
    categorical_availability,
    discover_bank,
    encode_frames,
    transform_actions,
)
from scripts.cache_official_pusht_memory import (
    _encode_frame_stream,
    _load_inputs,
    _selection_by_split,
)
from scripts.make_official_lewm_memory_data import collect_base
from scripts.paper_a_evidence_age import configure_determinism, fit_readout
from scripts.paper_a_evidence_age_spec import (
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    host_tasks,
    load_locked_spec,
    output_root,
    resolve_path,
    sha256_file,
    validate_device,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--sha", type=Path, default=DEFAULT_SHA)
    parser.add_argument("--host", required=True, choices=HOSTS)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def strict_task_root(spec: dict[str, Any], host: str, task: str) -> Path:
    return output_root(spec, "strict") / "cache" / host / task


def strict_cache_path(spec: dict[str, Any], host: str, task: str,
                      split: str, age: int) -> Path:
    return strict_task_root(spec, host, task) / split / f"age-{age}.npz"


def _all_equal(left: EpisodeBatch, right: EpisodeBatch) -> dict[str, bool]:
    return {
        "frames": bool(np.array_equal(left.frames, right.frames)),
        "actions": bool(np.array_equal(left.actions, right.actions)),
        "xi": bool(np.array_equal(left.xi, right.xi)),
        "endo_state": bool(np.array_equal(left.endo_state, right.endo_state)),
        "exo_state": bool(np.array_equal(left.exo_state, right.exo_state)),
        "events": (set(left.events) == set(right.events)
                   and all(np.array_equal(left.events[key], right.events[key])
                           for key in left.events)),
    }


def _score_shortcut(train_x: np.ndarray, train_y: np.ndarray,
                    validation_x: np.ndarray, validation_y: np.ndarray,
                    spec: dict[str, Any], *, balanced: bool) -> dict[str, Any]:
    result = fit_readout(
        train_x, train_y, validation_x, validation_y,
        spec["read_time"]["readout"], balanced=balanced)
    for key in ("prediction", "correct"):
        result.pop(key)
    return result


def _reacher_shifted_script(script: dict[str, np.ndarray], age: int,
                             decision: int) -> dict[str, np.ndarray]:
    shifted = {key: np.array(value, copy=True) for key, value in script.items()}
    duration = np.asarray(script["cue_off"] - script["cue_on"], dtype=np.int64)
    cue_off = np.full(len(duration), decision - age, dtype=np.int64)
    cue_on = cue_off - duration
    if np.any(cue_on < 0) or np.any(cue_off > decision):
        raise ValueError(f"shifted Reacher cue leaves sequence at age {age}")
    shifted["cue_on"] = cue_on
    shifted["cue_off"] = cue_off
    return shifted


def _verify_reacher_counterfactual(task, base_frames: np.ndarray,
                                   script: dict[str, np.ndarray]) -> None:
    alternate = {key: np.array(value, copy=True) for key, value in script.items()}
    alternate["xi"] = (alternate["xi"] + 1) % task.n_classes
    primary_frames = np.array(base_frames, copy=True)
    alternate_frames = np.array(base_frames, copy=True)
    task._render(primary_frames, script)
    task._render(alternate_frames, alternate)
    steps = np.arange(primary_frames.shape[1])[None]
    cue = ((steps >= script["cue_on"][:, None])
           & (steps < script["cue_off"][:, None]))
    outside = ~cue
    if not np.array_equal(primary_frames[outside], alternate_frames[outside]):
        raise ValueError("Reacher shifted counterfactual leaks outside cue")
    if np.array_equal(primary_frames[cue], alternate_frames[cue]):
        raise ValueError("Reacher shifted counterfactual did not alter cue")


def _reacher_prepare_split(spec: dict[str, Any], task_key: str, split: str,
                           task, model: torch.nn.Module,
                           device: torch.device, ages: list[int],
                           action_mean: np.ndarray, action_std: np.ndarray
                           ) -> tuple[dict[int, dict[str, np.ndarray]], dict]:
    parent = spec["parents"]["reacher"]
    episodes = 1200 if split == "train" else 240
    seed = int(parent["train_seed"] if split == "train"
               else parent["validation_seed"])
    source_path = discover_bank(
        resolve_path(parent["source_data_root"]), task_key, split, "clean")
    archived = load_bank(source_path)
    base_seed, nuisance_rng, xi_rng = task._rngs(seed)
    script = task._sample_script(episodes, nuisance_rng, xi_rng, xi_shift=0)
    base_frames, actions, endo_state = collect_base(episodes, 64, base_seed)
    original_frames = np.array(base_frames, copy=True)
    original_exo = task._render(original_frames, script)
    regenerated = EpisodeBatch(
        frames=original_frames, actions=actions, xi=script["xi"],
        xi_kind=task.xi_kind, n_classes=task.n_classes,
        endo_state=endo_state, exo_state=original_exo,
        events={key: script[key] for key in task.event_keys},
        stream="iid", task=task_key, seed=seed)
    parity = _all_equal(archived, regenerated)
    if not all(parity.values()):
        raise RuntimeError(
            f"Reacher exact replay failed for {task_key}/{split}: {parity}")
    replay = {
        "source_path": str(source_path.relative_to(ROOT)),
        "source_sha256": sha256_file(source_path),
        "array_parity": parity,
        "base_frames_sha256": sha256_array(base_frames),
        "actions_sha256": sha256_array(actions),
        "endo_state_sha256": sha256_array(endo_state),
    }
    normalized_actions = transform_actions(
        actions, action_mean, action_std, "clean")
    caches: dict[int, dict[str, np.ndarray]] = {}
    for age in ages:
        shifted = _reacher_shifted_script(script, age, 63)
        _verify_reacher_counterfactual(task, base_frames, shifted)
        shifted_frames = np.array(base_frames, copy=True)
        shifted_exo = task._render(shifted_frames, shifted)
        z = encode_frames(
            model, shifted_frames, device, 128,
            f"strict/{task_key}/{split}/age-{age}")
        caches[age] = {
            "z": z,
            "actions": normalized_actions,
            "labels": np.asarray(shifted["xi"], dtype=np.int64),
            "cue_on": np.asarray(shifted["cue_on"], dtype=np.int64),
            "cue_off": np.asarray(shifted["cue_off"], dtype=np.int64),
            "endo_state": endo_state,
            "exo_state": shifted_exo,
            "episode_index": np.arange(episodes, dtype=np.int64),
        }
        del shifted_frames, shifted_exo, z
    del base_frames, original_frames, original_exo, archived, regenerated
    return caches, replay


def _reacher_prepare(spec: dict[str, Any], task_key: str,
                     device: torch.device) -> dict[str, Any]:
    parent = spec["parents"]["reacher"]
    ages = [int(value) for value in
            spec["strict_fixed_endpoint"]["reacher"]["ages"]]
    task = make_task(task_key)
    train_source = discover_bank(
        resolve_path(parent["source_data_root"]), task_key, "train", "clean")
    train_bank = load_bank(train_source)
    action_mean, action_std = action_statistics(train_bank.actions)
    del train_bank
    model = load_official_reacher_checkpoint(
        resolve_path(parent["weights"]["path"]), device).eval()
    train, train_replay = _reacher_prepare_split(
        spec, task_key, "train", task, model, device, ages,
        action_mean, action_std)
    validation, validation_replay = _reacher_prepare_split(
        spec, task_key, "val", task, model, device, ages,
        action_mean, action_std)
    classes = task.n_classes
    chance = 1.0 / classes
    ceiling = chance + float(spec["strict_fixed_endpoint"]["admission"][
        "shortcut_margin_above_chance"])
    age_admission = {}
    records = []
    for age in ages:
        train_value, validation_value = train[age], validation[age]
        train_indices = np.rint(np.linspace(
            train_value["cue_on"], train_value["cue_off"] - 1, 4,
            axis=-1)).astype(np.int64)
        validation_indices = np.rint(np.linspace(
            validation_value["cue_on"], validation_value["cue_off"] - 1, 4,
            axis=-1)).astype(np.int64)
        train_rows = np.arange(len(train_value["labels"]))[:, None]
        validation_rows = np.arange(len(validation_value["labels"]))[:, None]
        cue = categorical_availability(
            train_value["z"][train_rows, train_indices].reshape(len(train_rows), -1),
            train_value["labels"],
            validation_value["z"][validation_rows, validation_indices].reshape(
                len(validation_rows), -1), validation_value["labels"])
        final_latent = _score_shortcut(
            train_value["z"][:, 60:63].reshape(len(train_rows), -1),
            train_value["labels"],
            validation_value["z"][:, 60:63].reshape(len(validation_rows), -1),
            validation_value["labels"], spec, balanced=False)
        final_action = _score_shortcut(
            train_value["actions"][:, 59:63].reshape(len(train_rows), -1),
            train_value["labels"],
            validation_value["actions"][:, 59:63].reshape(
                len(validation_rows), -1), validation_value["labels"],
            spec, balanced=False)
        final_state = _score_shortcut(
            train_value["endo_state"][:, 63], train_value["labels"],
            validation_value["endo_state"][:, 63], validation_value["labels"],
            spec, balanced=False)
        admitted = (float(cue["value"]) >= float(
            spec["strict_fixed_endpoint"]["admission"]["cue_accuracy_min"])
            and max(final_latent["value"], final_action["value"],
                    final_state["value"]) <= ceiling)
        age_admission[str(age)] = {
            "admitted": bool(admitted), "chance": chance,
            "shortcut_ceiling": ceiling, "cue_availability": cue,
            "final_latent_shortcut": final_latent,
            "final_action_shortcut": final_action,
            "final_state_shortcut": final_state,
        }
        for split, values in (("train", train_value),
                              ("validation", validation_value)):
            path = strict_cache_path(spec, "reacher", task_key, split, age)
            payload = {key: value for key, value in values.items()
                       if key not in ("endo_state", "exo_state")}
            metadata = {
                "schema": "paper_a_strict_age_cache_v1",
                "study": spec["study"], "lock": spec["_lock"],
                "host": "reacher", "task": task_key, "split": split,
                "age": age, "decision_index": 63,
                "paired_base_episode": True,
                "current_observation_excluded": True,
            }
            records.append(write_npz_with_sidecar(
                path, payload, metadata, compression_level=1))
    all_admitted = all(value["admitted"] for value in age_admission.values())
    return {
        "host": "reacher", "task": task_key, "ages": ages,
        "replay": {"train": train_replay, "validation": validation_replay},
        "admission": age_admission, "all_ages_admitted": all_admitted,
        "artifacts": records,
        "official_host_frozen": True,
    }


def _pusht_age_arrays(dataset, selections, task: dict[str, Any],
                      parent: dict[str, Any], model: torch.nn.Module,
                      device: torch.device, base: dict[str, np.ndarray],
                      age: int) -> dict[str, np.ndarray]:
    episodes = len(selections)
    length = int(parent["sequence"]["num_frames"])
    decision = int(parent["sequence"]["decision_index"])
    cue_length = int(parent["sequence"]["cue_length"])
    cue_off = decision - age
    cue_on = cue_off - cue_length
    labels = np.asarray([item.label for item in selections], dtype=np.int64)
    episode_index = np.asarray(
        [item.episode_index for item in selections], dtype=np.int64)
    local_start = np.asarray(
        [item.local_start for item in selections], dtype=np.int64)
    if not np.array_equal(episode_index, base["episode_index"]) \
            or not np.array_equal(local_start, base["local_start"]):
        raise RuntimeError("strict PushT selection differs from paired base cache")

    def stream() -> Iterator[tuple[int, np.ndarray]]:
        for row, item in enumerate(selections):
            native = dataset.read_sequence(
                item.episode_index, item.local_start, length)
            overlaid = render_single_overlay(
                native.frames, task["display_name"], item.label,
                cue_on, cue_length)
            if not np.array_equal(overlaid[:cue_on], native.frames[:cue_on]) \
                    or not np.array_equal(overlaid[cue_off:],
                                          native.frames[cue_off:]):
                raise RuntimeError("PushT strict overlay changed non-cue pixels")
            for offset in range(cue_length):
                yield row * cue_length + offset, overlaid[cue_on + offset]

    z_cue = _encode_frame_stream(
        model, stream(), episodes * cue_length,
        int(parent["cache"]["frame_batch_size"]),
        int(parent["official_host"]["image_size"]), device,
    ).reshape(episodes, cue_length, 192)
    z = np.asarray(base["z_base"], dtype=np.float32).copy()
    z[:, cue_on:cue_off] = z_cue
    return {
        "z": z, "actions": np.asarray(base["actions"], dtype=np.float32),
        "labels": labels,
        "cue_on": np.full(episodes, cue_on, dtype=np.int64),
        "cue_off": np.full(episodes, cue_off, dtype=np.int64),
        "episode_index": episode_index, "local_start": local_start,
        "z_cue": z_cue,
    }


def _pusht_prepare(spec: dict[str, Any], task_key: str,
                   device: torch.device) -> dict[str, Any]:
    parent_record = spec["parents"]["pusht"]
    parent = load_locked_pusht_spec(
        resolve_path(parent_record["config"]["path"]),
        resolve_path(parent_record["lock"]["path"]))
    task = pusht_task_spec(parent, task_key)
    ages = [int(value) for value in
            spec["strict_fixed_endpoint"]["pusht"]["ages"]]
    dataset, model, loaded_device = _load_inputs(parent, str(device))
    if loaded_device != device:
        raise RuntimeError("PushT strict encoder loaded on wrong device")
    selections = _selection_by_split(
        dataset, parent, classes=int(task["classes"]),
        label_seed=int(task["label_seed"]))
    base = {}
    arrays: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for split in ("train", "validation"):
        base[split], _ = load_pusht_base_cache(parent, split)
        arrays[split] = {
            age: _pusht_age_arrays(
                dataset, selections[split], task, parent, model, device,
                base[split], age)
            for age in ages
        }
    thresholds = PushTAdmissionThresholds(
        cue_accuracy_min=float(spec["strict_fixed_endpoint"]["admission"][
            "cue_accuracy_min"]),
        shortcut_margin_above_chance=float(
            spec["strict_fixed_endpoint"]["admission"]
            ["shortcut_margin_above_chance"]))
    admission = {}
    records = []
    for age in ages:
        cue_on = int(arrays["train"][age]["cue_on"][0])
        train_task = {
            "z_cue": arrays["train"][age]["z_cue"],
            "labels": arrays["train"][age]["labels"],
            "episode_index": arrays["train"][age]["episode_index"],
            "local_start": arrays["train"][age]["local_start"],
        }
        validation_task = {
            "z_cue": arrays["validation"][age]["z_cue"],
            "labels": arrays["validation"][age]["labels"],
            "episode_index": arrays["validation"][age]["episode_index"],
            "local_start": arrays["validation"][age]["local_start"],
        }
        result = evaluate_pusht_admission(
            task_key=task_key, semantic_name=task["display_name"],
            classes=int(task["classes"]), train_base=base["train"],
            train_task=train_task, validation_base=base["validation"],
            validation_task=validation_task, cue_start=cue_on,
            cue_length=int(parent["sequence"]["cue_length"]),
            final_context_indices=tuple(parent["sequence"]["final_context_indices"]),
            shortcut_action_indices=tuple(parent["sequence"]["shortcut_action_indices"]),
            thresholds=thresholds)
        admission[str(age)] = result
        for split in ("train", "validation"):
            values = arrays[split][age]
            path = strict_cache_path(spec, "pusht", task_key, split, age)
            payload = {key: value for key, value in values.items()
                       if key != "z_cue"}
            metadata = {
                "schema": "paper_a_strict_age_cache_v1",
                "study": spec["study"], "lock": spec["_lock"],
                "host": "pusht", "task": task_key, "split": split,
                "age": age, "decision_index": 19,
                "paired_base_episode": True,
                "current_observation_excluded": True,
            }
            records.append(write_npz_with_sidecar(
                path, payload, metadata, compression_level=1))
    return {
        "host": "pusht", "task": task_key, "ages": ages,
        "replay": {
            split: {
                "episode_index_sha256": sha256_array(base[split]["episode_index"]),
                "local_start_sha256": sha256_array(base[split]["local_start"]),
                "paired_selection_exact": True,
                "cue_only_pixel_mutation_checked": True,
            } for split in ("train", "validation")
        },
        "admission": admission,
        "all_ages_admitted": all(value["admitted"]
                                  for value in admission.values()),
        "artifacts": records, "official_host_frozen": True,
    }


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing strict cache writes without --execute")
    spec = load_locked_spec(args.spec, args.sha)
    validate_device(spec, args.device)
    if args.task not in host_tasks(spec, args.host):
        raise ValueError(f"task {args.task!r} is not registered for {args.host}")
    if not torch.cuda.is_available():
        raise RuntimeError("strict evidence-age preparation requires cuda:0")
    device = torch.device(args.device)
    configure_determinism(0)
    root = strict_task_root(spec, args.host, args.task)
    manifest_path = root / "manifest.json"
    stop_path = root / "prerequisite-stopped.json"
    if manifest_path.exists() or stop_path.exists():
        raise FileExistsError(f"strict task already resolved: {root}")
    try:
        result = (_reacher_prepare(spec, args.task, device)
                  if args.host == "reacher"
                  else _pusht_prepare(spec, args.task, device))
        payload = {
            "schema_version": 1, "study": spec["study"],
            "branch": "strict-fixed-endpoint-cue-offset",
            "lock": spec["_lock"], "status": (
                "admitted" if result["all_ages_admitted"] else
                "stopped-admission-failure"),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(0),
            **result,
        }
        atomic_text(manifest_path, stable_json(payload))
        if not result["all_ages_admitted"]:
            raise SystemExit(
                f"strict {args.host}/{args.task} failed formal admission")
        print(f"[evidence-age/strict] admitted {manifest_path}", flush=True)
    except SystemExit:
        raise
    except BaseException as error:
        payload = {
            "schema_version": 1, "study": spec["study"],
            "branch": "strict-fixed-endpoint-cue-offset",
            "lock": spec["_lock"], "status": "stopped-prerequisite-failure",
            "host": args.host, "task": args.task,
            "device": str(device), "error_type": type(error).__name__,
            "error": str(error),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
        }
        atomic_text(stop_path, stable_json(payload))
        print(f"[evidence-age/strict] stopped {stop_path}: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
