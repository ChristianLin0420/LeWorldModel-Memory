#!/usr/bin/env python3
"""Stream the pinned PushT HDF5 through the frozen official encoder.

``--phase base`` stores one shared latent/action/state cache.  ``--phase task``
stores only the three cue latents and labels for one semantic task, then runs
CPU admission probes against the shared label-independent cache.  Rendered
base frames are never persisted or copied into task artifacts.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.official_lewm import preprocess_frames  # noqa: E402
from lewm.models.official_lewm_pusht import (  # noqa: E402
    OFFICIAL_PUSHT_CHECKPOINT,
    load_official_pusht_checkpoint,
)
from lewm.official_tasks.artifacts import (  # noqa: E402
    atomic_text,
    sha256_file,
    stable_json,
    write_npz_with_sidecar,
)
from lewm.official_tasks.pusht_admission import (  # noqa: E402
    PushTAdmissionThresholds,
    evaluate_pusht_admission,
)
from lewm.official_tasks.pusht_hdf5 import (  # noqa: E402
    OfficialPushTHDF5,
    PushTSequenceSelection,
    normalize_native_action_blocks,
)
from lewm.official_tasks.pusht_memory import (  # noqa: E402
    render_single_overlay,
)
from lewm.official_tasks.pusht_pipeline import (  # noqa: E402
    PUSHT_SPLITS,
    load_pusht_base_cache,
    pusht_admission_path,
    pusht_base_cache_path,
    pusht_base_manifest_path,
    pusht_task_cache_path,
    pusht_task_manifest_path,
    pusht_task_spec,
)
from lewm.official_tasks.pusht_spec import (  # noqa: E402
    DEFAULT_PUSHT_LOCK,
    DEFAULT_PUSHT_SPEC,
    load_locked_pusht_spec,
    pusht_lock_receipt,
    resolve_pusht_path,
    validate_pusht_device,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("base", "task"))
    parser.add_argument("--task")
    parser.add_argument("--device", required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_PUSHT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_PUSHT_LOCK)
    parser.add_argument(
        "--execute", action="store_true",
        help="required acknowledgement before any cache artifact is written")
    return parser.parse_args(argv)


def _configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _preflight(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite formal PushT artifacts: "
            + ", ".join(map(str, existing)))


def _selection_by_split(
        dataset: OfficialPushTHDF5, spec: dict[str, Any],
        *, classes: int, label_seed: int,
        ) -> dict[str, tuple[PushTSequenceSelection, ...]]:
    selected = dataset.select_sequences(
        num_frames=spec["sequence"]["num_frames"],
        train_count=spec["selection"]["train"]["episodes"],
        validation_count=spec["selection"]["validation"]["episodes"],
        num_classes=classes,
        split_seed=spec["selection"]["split_seed"],
        start_seed=spec["selection"]["start_seed"],
        label_seed=label_seed,
    )
    return {
        split: tuple(item for item in selected if item.split == split)
        for split in PUSHT_SPLITS
    }


@torch.inference_mode()
def _encode_frame_stream(
        model: torch.nn.Module,
        indexed_frames: Iterator[tuple[int, np.ndarray]],
        total_frames: int, frame_batch_size: int,
        image_size: int, device: torch.device) -> np.ndarray:
    """Encode a single-pass indexed stream and reject repeats or omissions."""

    output = np.empty((total_frames, 192), dtype=np.float32)
    seen = np.zeros(total_frames, dtype=np.bool_)
    indices: list[int] = []
    frames: list[np.ndarray] = []

    def flush() -> None:
        if not frames:
            return
        batch = np.stack(frames)
        pixels = torch.from_numpy(batch).permute(0, 3, 1, 2).to(
            device, non_blocking=True)
        pixels = preprocess_frames(pixels, image_size=image_size)
        encoded = model.encode_pixels(pixels).float().cpu().numpy()
        if encoded.shape != (len(indices), 192) \
                or not np.isfinite(encoded).all():
            raise RuntimeError("official PushT encoder returned invalid latents")
        output[np.asarray(indices)] = encoded
        indices.clear()
        frames.clear()

    for index, frame in indexed_frames:
        if not 0 <= index < total_frames or seen[index]:
            raise ValueError(f"duplicate or invalid streamed frame index {index}")
        value = np.asarray(frame)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError("streamed frames must be uint8 HWC RGB")
        seen[index] = True
        indices.append(index)
        frames.append(value)
        if len(frames) == frame_batch_size:
            flush()
    flush()
    if not seen.all():
        raise ValueError(
            f"frame stream omitted {int((~seen).sum())} required positions")
    return output


def _verify_dataset_contract(dataset: OfficialPushTHDF5,
                             spec: dict[str, Any]) -> None:
    expected = spec["dataset"]
    if dataset.hdf5_size != expected["hdf5_size"]:
        raise ValueError("official PushT HDF5 byte size differs from formal spec")
    if dataset.schema.num_rows != expected["required_rows"] \
            or dataset.schema.num_episodes != expected["required_episodes"]:
        raise ValueError("official PushT row/episode count differs from formal spec")
    if list(dataset.schema.pixel_shape) != expected["pixel_shape"]:
        raise ValueError("official PushT pixel shape differs from formal spec")
    if dataset.schema.pixel_filter_ids != (expected["pixel_filter_id"],):
        raise ValueError(
            "official PushT pixels do not use the locked Blosc filter 32001")


def _base_arrays(
        dataset: OfficialPushTHDF5, selections: tuple[PushTSequenceSelection, ...],
        spec: dict[str, Any], model: torch.nn.Module, device: torch.device,
        raw_mean: np.ndarray, raw_std: np.ndarray,
        ) -> dict[str, np.ndarray]:
    episodes = len(selections)
    frames = spec["sequence"]["num_frames"]
    actions = np.empty((episodes, frames - 1, 10), dtype=np.float32)
    state = np.empty(
        (episodes, frames, 7), dtype=np.dtype(dataset.schema.state_dtype))
    proprio = np.empty(
        (episodes, frames, 4), dtype=np.dtype(dataset.schema.proprio_dtype))
    global_indices = np.empty((episodes, frames), dtype=np.int64)
    episode_index = np.asarray(
        [item.episode_index for item in selections], dtype=np.int64)
    local_start = np.asarray(
        [item.local_start for item in selections], dtype=np.int64)

    def stream() -> Iterator[tuple[int, np.ndarray]]:
        for row, item in enumerate(selections):
            native = dataset.read_sequence(
                item.episode_index, item.local_start, frames)
            actions[row] = normalize_native_action_blocks(
                native.actions, raw_mean, raw_std)
            state[row] = native.state
            proprio[row] = native.proprio
            global_indices[row] = native.global_frame_indices
            for step, frame in enumerate(native.frames):
                yield row * frames + step, frame

    z_base = _encode_frame_stream(
        model, stream(), episodes * frames,
        spec["cache"]["frame_batch_size"],
        spec["official_host"]["image_size"], device,
    ).reshape(episodes, frames, 192)
    if len(np.unique(global_indices)) != episodes * frames:
        raise ValueError("formal base selection duplicates a source frame")
    return {
        "z_base": z_base,
        "actions": actions,
        "state": state,
        "proprio": proprio,
        "episode_index": episode_index,
        "local_start": local_start,
        "global_frame_indices": global_indices,
    }


def _task_arrays(
        dataset: OfficialPushTHDF5, selections: tuple[PushTSequenceSelection, ...],
        task: dict[str, Any], spec: dict[str, Any], model: torch.nn.Module,
        device: torch.device) -> dict[str, np.ndarray]:
    episodes = len(selections)
    num_frames = spec["sequence"]["num_frames"]
    cue_start = spec["sequence"]["cue_start"]
    cue_length = spec["sequence"]["cue_length"]
    labels = np.asarray([item.label for item in selections], dtype=np.int64)
    episode_index = np.asarray(
        [item.episode_index for item in selections], dtype=np.int64)
    local_start = np.asarray(
        [item.local_start for item in selections], dtype=np.int64)

    def stream() -> Iterator[tuple[int, np.ndarray]]:
        for row, item in enumerate(selections):
            native = dataset.read_sequence(
                item.episode_index, item.local_start, num_frames)
            overlaid = render_single_overlay(
                native.frames, task["display_name"], item.label,
                cue_start, cue_length)
            for cue_offset in range(cue_length):
                yield (row * cue_length + cue_offset,
                       overlaid[cue_start + cue_offset])

    z_cue = _encode_frame_stream(
        model, stream(), episodes * cue_length,
        spec["cache"]["frame_batch_size"],
        spec["official_host"]["image_size"], device,
    ).reshape(episodes, cue_length, 192)
    return {
        "z_cue": z_cue,
        "labels": labels,
        "episode_index": episode_index,
        "local_start": local_start,
    }


def _load_inputs(spec: dict[str, Any], device: str
                 ) -> tuple[OfficialPushTHDF5, torch.nn.Module, torch.device]:
    dataset_path = resolve_pusht_path(spec["dataset"]["hdf5_path"])
    if dataset_path.stat().st_size != spec["dataset"]["hdf5_size"]:
        raise ValueError("official PushT HDF5 size differs before hashing")
    dataset = OfficialPushTHDF5(
        dataset_path,
        expected_hdf5_sha256=spec["dataset"]["hdf5_sha256"])
    _verify_dataset_contract(dataset, spec)
    bundle = resolve_pusht_path(spec["official_host"]["bundle_path"])
    model = load_official_pusht_checkpoint(bundle, device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("official PushT model was not frozen by its loader")
    return dataset, model, torch.device(device)


def _base_phase(spec: dict[str, Any], dataset: OfficialPushTHDF5,
                model: torch.nn.Module, device: torch.device) -> None:
    selections = _selection_by_split(
        dataset, spec, classes=1, label_seed=0)
    raw_mean, raw_std, raw_count = dataset.raw_action_statistics(ddof=1)
    records = []
    for split in PUSHT_SPLITS:
        arrays = _base_arrays(
            dataset, selections[split], spec, model, device,
            raw_mean, raw_std)
        metadata = {
            "schema": "official_pusht_base_cache_v1",
            "split": split,
            "formal_lock": pusht_lock_receipt(spec),
            "dataset_hdf5_sha256": spec["dataset"]["hdf5_sha256"],
            "official_checkpoint": {
                "repo_id": OFFICIAL_PUSHT_CHECKPOINT.repo_id,
                "revision": OFFICIAL_PUSHT_CHECKPOINT.revision,
                "weights_sha256": OFFICIAL_PUSHT_CHECKPOINT.weights_sha256,
            },
            "base_frames_stored": False,
            "one_sequence_per_episode": True,
            "action_normalization": {
                "source": "all finite raw actions in pinned official HDF5",
                "mean": raw_mean.tolist(),
                "std_ddof1": raw_std.tolist(),
                "finite_count": raw_count.tolist(),
                "order": "normalize raw (5,2), then time-major flatten to 10",
            },
        }
        record = write_npz_with_sidecar(
            pusht_base_cache_path(spec, split), arrays, metadata,
            compression_level=spec["cache"]["compression_level"])
        records.append(record)
    manifest = {
        "schema": "official_pusht_base_cache_manifest_v1",
        "formal_lock": pusht_lock_receipt(spec),
        "dataset_hdf5_sha256": spec["dataset"]["hdf5_sha256"],
        "dataset_hdf5_size": spec["dataset"]["hdf5_size"],
        "schema_fingerprint": dataset.schema.fingerprint,
        "base_frames_stored": False,
        "task_data_stored": False,
        "action_mean": raw_mean.tolist(),
        "action_std_ddof1": raw_std.tolist(),
        "artifacts": records,
    }
    atomic_text(pusht_base_manifest_path(spec), stable_json(manifest))


def _task_phase(spec: dict[str, Any], task_key: str,
                dataset: OfficialPushTHDF5, model: torch.nn.Module,
                device: torch.device) -> None:
    task = pusht_task_spec(spec, task_key)
    selections = _selection_by_split(
        dataset, spec, classes=task["classes"],
        label_seed=task["label_seed"])
    base: dict[str, dict[str, np.ndarray]] = {}
    task_arrays: dict[str, dict[str, np.ndarray]] = {}
    records = []
    for split in PUSHT_SPLITS:
        base[split], _ = load_pusht_base_cache(spec, split)
        task_arrays[split] = _task_arrays(
            dataset, selections[split], task, spec, model, device)
        for key in ("episode_index", "local_start"):
            if not np.array_equal(base[split][key], task_arrays[split][key]):
                raise ValueError(
                    f"{task_key}/{split} differs from shared base selection")
        metadata = {
            "schema": "official_pusht_task_cue_cache_v1",
            "split": split,
            "task_key": task_key,
            "semantic_name": task["display_name"],
            "classes": task["classes"],
            "formal_lock": pusht_lock_receipt(spec),
            "stores_only_cue_latents": True,
            "base_frames_stored": False,
            "cue_start": spec["sequence"]["cue_start"],
            "cue_length": spec["sequence"]["cue_length"],
            "cue_only_overlay_checks": len(selections[split]),
        }
        records.append(write_npz_with_sidecar(
            pusht_task_cache_path(spec, task_key, split),
            task_arrays[split], metadata,
            compression_level=spec["cache"]["compression_level"]))
    thresholds = PushTAdmissionThresholds(
        cue_accuracy_min=spec["admission"]["cue_accuracy_min"],
        shortcut_margin_above_chance=
            spec["admission"]["shortcut_margin_above_chance"])
    admission = evaluate_pusht_admission(
        task_key=task_key, semantic_name=task["display_name"],
        classes=task["classes"],
        train_base=base["train"], train_task=task_arrays["train"],
        validation_base=base["validation"],
        validation_task=task_arrays["validation"],
        cue_start=spec["sequence"]["cue_start"],
        cue_length=spec["sequence"]["cue_length"],
        final_context_indices=tuple(
            spec["sequence"]["final_context_indices"]),
        shortcut_action_indices=tuple(
            spec["sequence"]["shortcut_action_indices"]),
        thresholds=thresholds,
    )
    admission["formal_lock"] = pusht_lock_receipt(spec)
    admission_file = pusht_admission_path(spec, task_key)
    admission_hash = atomic_text(admission_file, stable_json(admission))
    manifest = {
        "schema": "official_pusht_task_cache_manifest_v1",
        "task_key": task_key,
        "semantic_name": task["display_name"],
        "formal_lock": pusht_lock_receipt(spec),
        "shared_base_manifest": {
            "path": str(pusht_base_manifest_path(spec)),
            "sha256": sha256_file(pusht_base_manifest_path(spec)),
        },
        "stores_only_cue_latents": True,
        "base_frames_stored": False,
        "artifacts": records,
        "admission": {
            "path": str(admission_file),
            "sha256": admission_hash,
            "admitted": admission["admitted"],
        },
    }
    atomic_text(pusht_task_manifest_path(spec, task_key), stable_json(manifest))


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    spec = load_locked_pusht_spec(args.spec, args.lock)
    validate_pusht_device(args.device)
    task_keys = [task["key"] for task in spec["semantic_tasks"]]
    if args.phase == "base" and args.task is not None:
        raise ValueError("--task is invalid for --phase base")
    if args.phase == "task" and args.task not in task_keys:
        raise ValueError(f"--phase task requires --task in {task_keys}")
    if not args.execute:
        raise RuntimeError("formal PushT cache writes require explicit --execute")
    paths = ([pusht_base_cache_path(spec, split) for split in PUSHT_SPLITS]
             + [pusht_base_manifest_path(spec)]
             if args.phase == "base" else
             [pusht_task_cache_path(spec, args.task, split)
              for split in PUSHT_SPLITS]
             + [pusht_task_manifest_path(spec, args.task),
                pusht_admission_path(spec, args.task)])
    _preflight([path for item in paths for path in (
        item, item.with_suffix(item.suffix + ".json"))])
    if not torch.cuda.is_available():
        raise RuntimeError("formal PushT frozen encoding requires CUDA")
    _configure_determinism()
    dataset, model, device = _load_inputs(spec, args.device)
    if args.phase == "base":
        _base_phase(spec, dataset, model, device)
    else:
        _task_phase(spec, args.task, dataset, model, device)


if __name__ == "__main__":
    main()
