#!/usr/bin/env python3
"""Prepare matched color-by-location caches and formal admission receipts."""

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

from lewm.models.official_lewm import (  # noqa: E402
    load_official_reacher_checkpoint,
    preprocess_frames,
)
from lewm.models.official_lewm_pusht import (  # noqa: E402
    load_official_pusht_checkpoint,
)
from lewm.models.official_lewm_tworoom import (  # noqa: E402
    load_official_tworoom_checkpoint,
)
from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    sha256_array,
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
from scripts.paper_a_evidence_age import (  # noqa: E402
    configure_determinism,
    fit_readout,
)
from scripts.paper_a_matched_host_spec import (  # noqa: E402
    AGES,
    DEFAULT_SHA,
    DEFAULT_SPEC,
    HOSTS,
    TARGETS,
    load_locked_spec,
    output_path,
    resolve_input_path,
    resolve_path,
    sha256_file,
    validate_device,
)
from scripts.train_frozen_official_swap import state_digest  # noqa: E402


SPLITS = ("train", "validation")


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


@torch.inference_mode()
def _encode_stream(model: torch.nn.Module,
                   indexed_frames: Iterator[tuple[int, np.ndarray]],
                   total: int, batch_size: int,
                   device: torch.device) -> np.ndarray:
    output = np.empty((total, 192), dtype=np.float32)
    seen = np.zeros(total, dtype=np.bool_)
    indices: list[int] = []
    frames: list[np.ndarray] = []

    def flush() -> None:
        if not frames:
            return
        batch = np.stack(frames)
        pixels = torch.from_numpy(batch).permute(0, 3, 1, 2).to(
            device, non_blocking=True)
        pixels = preprocess_frames(pixels, image_size=224)
        latent = model.encode_pixels(pixels).float().cpu().numpy()
        if latent.shape != (len(indices), 192) or not np.isfinite(latent).all():
            raise RuntimeError("official encoder produced invalid matched latents")
        output[np.asarray(indices, dtype=np.int64)] = latent
        frames.clear()
        indices.clear()

    for index, frame in indexed_frames:
        if not 0 <= index < total or seen[index]:
            raise ValueError(f"invalid or duplicate frame index {index}")
        value = np.asarray(frame)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError("encoder stream requires uint8 HWC RGB")
        seen[index] = True
        indices.append(index)
        frames.append(value)
        if len(frames) == batch_size:
            flush()
    flush()
    if not seen.all():
        raise ValueError(f"encoder stream omitted {int((~seen).sum())} frames")
    return output


def _load_host(spec: dict[str, Any], host: str,
               device: torch.device) -> torch.nn.Module:
    value = spec["inputs"][host]
    if host == "reacher":
        model = load_official_reacher_checkpoint(
            resolve_path(value["weights"]["path"]), device)
    elif host == "pusht":
        model = load_official_pusht_checkpoint(
            resolve_path(value["bundle_path"]), device)
    else:
        model = load_official_tworoom_checkpoint(
            resolve_path(value["bundle_path"]), device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError(f"{host} official loader did not freeze the model")
    return model


def _raw_action_statistics(actions: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(actions, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(raw).all():
        raise ValueError("generated Reacher actions are not finite")
    mean = raw.mean(axis=0)
    std = raw.std(axis=0, ddof=1)
    count = np.full(2, len(raw), dtype=np.int64)
    if np.any(std <= 0):
        raise ValueError("generated Reacher action normalization is degenerate")
    return mean, std, count


def _normalize_generated(actions: np.ndarray, mean: np.ndarray,
                         std: np.ndarray) -> np.ndarray:
    return normalize_action_blocks(actions, mean, std)


def _split_labels(spec: dict[str, Any], split: str) -> MatchedLabels:
    count = int(spec["selection"][f"{split}_episodes"])
    seed = int(spec["selection"][f"{split}_label_seed"])
    return balanced_joint_labels(count, seed)


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
        seed = int(source[f"{split}_base_seed"])
        raw[split] = collect_base(count, length, seed)
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
        "raw_action_mean": mean.tolist(), "raw_action_std_ddof1": std.tolist(),
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
    split_counts = tuple(
        (split, int(spec["selection"][f"{split}_episodes"]))
        for split in SPLITS)
    selected = dataset.select_sequences(
        num_frames=int(spec["sequence"]["num_frames"]),
        split_counts=split_counts,
        split_seed=int(source["split_seed"]),
        start_seed=int(source["start_seed"]))
    selections = {
        split: tuple(value for value in selected if value.split == split)
        for split in SPLITS}
    if len({value.episode_index for value in selected}) != len(selected):
        raise RuntimeError(f"{host} selections are not episode-disjoint")
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
            raise RuntimeError(f"{host}/{split} duplicates selected source frames")
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
        "raw_action_mean": mean.tolist(), "raw_action_std_ddof1": std.tolist(),
        "raw_action_count": count.tolist(),
        "exhaustive_16_way_counterfactual_checks": checks,
    }
    return base, cue, provenance


def _full_latent(base: dict[str, np.ndarray],
                 cue: dict[str, np.ndarray]) -> np.ndarray:
    value = np.asarray(base["z_base"], dtype=np.float32).copy()
    cue_on = int(cue["cue_on"][0])
    cue_off = int(cue["cue_off"][0])
    if not np.all(cue["cue_on"] == cue_on) \
            or not np.all(cue["cue_off"] == cue_off):
        raise ValueError("matched cue timing must be constant within an age")
    value[:, cue_on:cue_off] = cue["z_cue"]
    return value


def _probe(train_x: np.ndarray, train_y: np.ndarray,
           validation_x: np.ndarray, validation_y: np.ndarray,
           spec: dict[str, Any]) -> dict[str, Any]:
    result = fit_readout(
        train_x, train_y, validation_x, validation_y,
        spec["readout"], balanced=True)
    prediction = np.asarray(result.pop("prediction"), dtype=np.int64)
    result.pop("correct")
    matrix = confusion_matrix(validation_y, prediction, labels=np.arange(4))
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    result["per_class_recall"] = recall.tolist()
    result["min_class_recall"] = float(recall.min())
    return result


def _admission(spec: dict[str, Any],
               base: dict[str, dict[str, np.ndarray]],
               cue: dict[str, dict[int, dict[str, np.ndarray]]]
               ) -> tuple[dict[str, Any], bool]:
    threshold = float(spec["admission"]["cue_balanced_accuracy_min"])
    recall_threshold = float(spec["admission"]["cue_min_class_recall_min"])
    shortcut_ceiling = 0.25 + float(
        spec["admission"]["shortcut_margin_above_chance"])
    endpoint = int(spec["sequence"]["decision_index"])
    history = int(spec["sequence"]["endpoint_history"])
    result: dict[str, Any] = {}
    passed = True
    for age in AGES:
        train_cue, validation_cue = cue["train"][age], cue["validation"][age]
        train_z = _full_latent(base["train"], train_cue)
        validation_z = _full_latent(base["validation"], validation_cue)
        age_result: dict[str, Any] = {}
        for target in TARGETS:
            train_y = train_cue[f"{target}_label"]
            validation_y = validation_cue[f"{target}_label"]
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
            target_pass = (
                cue_probe["balanced_accuracy"] >= threshold
                and cue_probe["min_class_recall"] >= recall_threshold
                and max(latent["balanced_accuracy"], action["balanced_accuracy"],
                        state["balanced_accuracy"]) <= shortcut_ceiling)
            age_result[target] = {
                "admitted": bool(target_pass), "chance": 0.25,
                "requirement_by_construction": (
                    "label is independently randomized after base-trajectory "
                    "selection and only changes the registered cue pixels"),
                "cue_balanced_accuracy_min": threshold,
                "cue_min_class_recall_min": recall_threshold,
                "shortcut_ceiling": shortcut_ceiling,
                "cue_probe": cue_probe,
                "final_context_latent_shortcut": latent,
                "final_action_shortcut": action,
                "final_state_shortcut": state,
            }
            passed = passed and bool(target_pass)
        result[f"age-{age}"] = age_result
    return result, passed


def _write_caches(spec: dict[str, Any], host: str,
                  base: dict[str, dict[str, np.ndarray]],
                  cue: dict[str, dict[int, dict[str, np.ndarray]]]
                  ) -> list[dict[str, Any]]:
    records = []
    for split in SPLITS:
        records.append(write_npz_with_sidecar(
            base_cache_path(spec, host, split), base[split], {
                "schema": "paper_a_matched_base_cache_v1",
                "study": spec["study"], "lock": spec["_lock"],
                "host": host, "split": split,
                "num_frames": 20, "decision_index": 19,
                "label_independent_base": True,
            }, compression_level=int(spec["cache"]["compression_level"])))
        for age in AGES:
            records.append(write_npz_with_sidecar(
                cue_cache_path(spec, host, split, age), cue[split][age], {
                    "schema": "paper_a_matched_cue_cache_v1",
                    "study": spec["study"], "lock": spec["_lock"],
                    "host": host, "split": split, "age": age,
                    "decision_index": 19, "cue_length": 3,
                    "same_rendered_episode_for_both_targets": True,
                }, compression_level=int(spec["cache"]["compression_level"])))
    return records


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing matched-host writes without --execute")
    spec = load_locked_spec(args.spec, args.sha, verify_inputs=True)
    validate_device(spec, args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("matched-host preparation requires physical GPU 0")
    configure_determinism(0)
    device = torch.device(args.device)
    manifest_path = host_manifest_path(spec, args.host)
    stop_path = host_cache_root(spec, args.host) / "prerequisite-stopped.json"
    if manifest_path.exists() or stop_path.exists():
        raise FileExistsError(f"host branch already resolved: {args.host}")
    try:
        model = _load_host(spec, args.host, device)
        host_before = state_digest(model)
        if args.host == "reacher":
            base, cue, provenance = _reacher_arrays(spec, model, device)
        else:
            base, cue, provenance = _hdf_arrays(
                spec, args.host, model, device)
        host_after = state_digest(model)
        if host_before != host_after:
            raise RuntimeError("frozen host changed during matched encoding")
        admission, admitted = _admission(spec, base, cue)
        records = _write_caches(spec, args.host, base, cue)
        payload = {
            "schema_version": 1, "study": spec["study"],
            "branch": "matched-color-location-fixed-endpoint",
            "lock": spec["_lock"], "host": args.host,
            "status": "admitted" if admitted else "stopped-admission-failure",
            "all_targets_ages_admitted": bool(admitted),
            "requirement_by_construction": True,
            "requirement_contract": (
                "balanced joint labels are independent of base episodes and "
                "counterfactual renderings differ only inside three cue frames"),
            "device": str(device), "physical_gpu": 0,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "frozen_host_sha256_before": host_before,
            "frozen_host_sha256_after": host_after,
            "frozen_host_unchanged": True,
            "released_checkpoint_predictor_history": 3,
            "paper_appendix_reported_tworoom_history": 1 if args.host == "tworoom" else None,
            "history_contract_source": "authenticated released config.json",
            "provenance": provenance, "admission": admission,
            "artifacts": records,
        }
        atomic_text(manifest_path, stable_json(payload))
        if not admitted:
            raise SystemExit(f"matched-host {args.host} failed formal admission")
        print(f"[matched-host] admitted {manifest_path}", flush=True)
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
