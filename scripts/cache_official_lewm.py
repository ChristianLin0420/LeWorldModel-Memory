#!/usr/bin/env python3
"""Cache frozen latents from the released SIGReg LeWM Reacher encoder.

The default input is the clean, five-simulator-step bank produced by
``scripts/make_official_lewm_memory_data.py``.  Its actions are already the
official 10-D flattened 5x2 blocks, so each column is standardized with
statistics from the selected task's training bank.  Compatibility with the
older V19 2-D corrupted banks is deliberately opt-in via
``--source-stream observed``; only that mode repeats a standardized 2-D
action five times.

For every selected task this script writes::

    <output>/<task>/train.npz
    <output>/<task>/val.npz
    <output>/<task>/availability.json
    <output>/<task>/manifest.json

Each split archive contains ``z``, ``actions``, ``xi``, ``endo_state``,
``exo_state``, every ``event_*`` array, and a scalar JSON string in
``meta_json``.  The encoder and projector are frozen; labels are used only by
the post-hoc availability probe after all latents have been cached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.official_lewm import (  # noqa: E402
    OFFICIAL_ACTION_DIM,
    OFFICIAL_EMBED_DIM,
    OFFICIAL_IMAGE_SIZE,
    load_official_reacher_checkpoint,
    preprocess_frames,
)
from lewm.tasks_v19.base import EpisodeBatch, load_bank  # noqa: E402


TASKS = ("t1", "t3", "t4")
SPLITS = ("train", "val")
SOURCE_STREAMS = ("clean", "observed")
DEFAULT_DATA_ROOT = ROOT / "outputs/paper_a_expansion/data"
DEFAULT_OUTPUT = ROOT / "outputs/paper_a_expansion/cache"
DEFAULT_WEIGHTS = (
    ROOT / "outputs/paper_a_expansion/pretrained/lewm-reacher/weights.pt"
)
OFFICIAL_WEIGHTS_SHA256 = (
    "eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469"
)
OFFICIAL_SOURCE = "quentinll/lewm-reacher"
OFFICIAL_SOURCE_COMMIT = "62adae4b71dc474ddf8f794c476ebfe737a743ca"
FRAME_BATCH_SIZE = 128
PROBE_FRAMES = 4
RIDGE_ALPHAS = np.logspace(-3, 3, 7)
PROBE_RANDOM_STATE = 0
SCHEMA = "official_lewm_reacher_latents_v1"


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _atomic_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)
    return sha256_file(path)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def configure_determinism(seed: int = 0) -> None:
    """Fix inference ordering and disable reduced-precision CUDA paths."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def discover_bank(data_root: Path, task: str, split: str,
                  source_stream: str) -> Path:
    """Resolve exactly one bank, refusing an ambiguous cache directory."""
    directory = data_root / task
    matches = sorted(directory.glob(f"{split}_{source_stream}_*.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {split}_{source_stream}_*.npz below "
            f"{directory}, found {len(matches)}: "
            f"{[path.name for path in matches]}")
    sidecar = matches[0].with_suffix(matches[0].suffix + ".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing bank sidecar {sidecar}")
    return matches[0]


def source_record(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(sidecar.read_text())
    actual = sha256_file(path)
    if actual != metadata.get("npz_sha256"):
        raise ValueError(
            f"source bank hash mismatch for {path}: {actual} != "
            f"{metadata.get('npz_sha256')}")
    return {
        "path": _display_path(path),
        "npz_sha256": actual,
        "sidecar_sha256": sha256_file(sidecar),
        "bank_format": metadata.get("format"),
        "seed": metadata.get("seed"),
        "num_episodes": metadata.get("num_episodes"),
        "length": metadata.get("length"),
    }


def action_statistics(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Population mean/std for each raw training-bank action column."""
    if actions.ndim != 3:
        raise ValueError(f"actions must be (E,T,A), got {actions.shape}")
    flattened = actions.reshape(-1, actions.shape[-1]).astype(np.float64)
    mean = flattened.mean(axis=0)
    std = flattened.std(axis=0, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("non-finite training action statistics")
    if (std <= 1e-12).any():
        raise ValueError(f"degenerate training action columns: std={std}")
    return mean, std


def transform_actions(actions: np.ndarray, mean: np.ndarray,
                      std: np.ndarray, source_stream: str) -> np.ndarray:
    """Apply the official action contract using training-only statistics.

    Native clean banks contain five independently executed 2-D controls in
    temporal order and therefore already have width 10.  The explicit legacy
    observed mode has width 2 and repeats that one standardized control block
    five times as ``[a0,a1,a0,a1,...]``.
    """
    actions = np.asarray(actions)
    if actions.ndim != 3 or actions.shape[-1] != len(mean) \
            or mean.shape != std.shape:
        raise ValueError(
            f"action/statistic shape mismatch: actions={actions.shape}, "
            f"mean={mean.shape}, std={std.shape}")
    normalized = ((actions.astype(np.float64) - mean) / std).astype(np.float32)
    if source_stream == "clean":
        if normalized.shape[-1] != OFFICIAL_ACTION_DIM:
            raise ValueError(
                "clean source must contain native 10-D (5x2) action blocks; "
                "use --source-stream observed explicitly for legacy 2-D banks")
        return normalized
    if source_stream == "observed":
        if normalized.shape[-1] != 2:
            raise ValueError(
                "legacy observed compatibility expects 2-D actions, got "
                f"{normalized.shape[-1]}")
        return np.tile(normalized, (1, 1, OFFICIAL_ACTION_DIM // 2))
    raise ValueError(f"unknown source_stream {source_stream!r}")


@torch.inference_mode()
def encode_frames(model: torch.nn.Module, frames: np.ndarray,
                  device: torch.device, frame_batch_size: int,
                  progress_label: str | None = None) -> np.ndarray:
    """Encode ``(E,L,H,W,3)`` uint8 frames in a fixed flattened order."""
    if frames.ndim != 5 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (E,L,H,W,3), got {frames.shape}")
    if frame_batch_size <= 0:
        raise ValueError("frame_batch_size must be positive")
    episodes, length = frames.shape[:2]
    flattened = frames.reshape(-1, *frames.shape[2:])
    latents = np.empty((len(flattened), OFFICIAL_EMBED_DIM), dtype=np.float32)
    chunks = (len(flattened) + frame_batch_size - 1) // frame_batch_size
    for chunk_index, start in enumerate(
            range(0, len(flattened), frame_batch_size), start=1):
        stop = min(start + frame_batch_size, len(flattened))
        pixels = torch.from_numpy(flattened[start:stop]).permute(0, 3, 1, 2)
        pixels = preprocess_frames(pixels.to(device, non_blocking=True))
        encoded = model.encode_pixels(pixels)
        if encoded.shape != (stop - start, OFFICIAL_EMBED_DIM):
            raise ValueError(
                f"official encoder returned {tuple(encoded.shape)}, expected "
                f"{(stop - start, OFFICIAL_EMBED_DIM)}")
        latents[start:stop] = encoded.float().cpu().numpy()
        if progress_label and (chunk_index == chunks
                               or chunk_index % max(chunks // 10, 1) == 0):
            print(f"[official-cache] {progress_label}: "
                  f"{chunk_index}/{chunks} frame chunks", flush=True)
    if not np.isfinite(latents).all():
        raise ValueError("official encoder produced non-finite latents")
    return latents.reshape(episodes, length, OFFICIAL_EMBED_DIM)


def _spaced_indices(start: np.ndarray, stop: np.ndarray,
                    count: int = PROBE_FRAMES) -> np.ndarray:
    """Evenly spaced inclusive integer indices for per-episode windows."""
    start = np.asarray(start, dtype=np.float64)
    stop = np.broadcast_to(np.asarray(stop, dtype=np.float64), start.shape)
    if (stop < start).any():
        raise ValueError("probe window stop precedes start")
    return np.rint(np.linspace(start, stop, count, axis=-1)).astype(np.int64)


def availability_features(latents: np.ndarray, bank: EpisodeBatch
                          ) -> np.ndarray:
    """Frozen sighted coordinate used by the task-specific probe."""
    if bank.xi_kind == "cat":
        indices = _spaced_indices(
            bank.events["cue_on"], bank.events["cue_off"] - 1)
    else:
        gap_on = np.asarray(bank.events["gap_on"], dtype=np.int64)
        indices = gap_on[:, None] + np.arange(-PROBE_FRAMES, 0)[None, :]
    selected = latents[np.arange(bank.num_episodes)[:, None], indices]
    return selected.reshape(bank.num_episodes, -1).astype(np.float32)


def categorical_availability(train_x: np.ndarray, train_y: np.ndarray,
                             val_x: np.ndarray, val_y: np.ndarray
                             ) -> dict[str, Any]:
    """Standardized cue-window multinomial logistic accuracy."""
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=2000,
        random_state=PROBE_RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(scaler.transform(train_x), train_y)
    prediction = model.predict(scaler.transform(val_x))
    classes = np.unique(train_y)
    return {
        "metric": "accuracy",
        "value": float(np.mean(prediction == val_y)),
        "chance": float(1.0 / len(classes)),
        "n_classes": int(len(classes)),
        "feature": "four_evenly_spaced_cue_window_latents_concatenated",
        "probe": "StandardScaler+LogisticRegression(C=1,lbfgs)",
        "iterations": [int(value) for value in model.n_iter_],
    }


def continuous_availability(train_x: np.ndarray, train_y: np.ndarray,
                            val_x: np.ndarray, val_y: np.ndarray
                            ) -> dict[str, Any]:
    """Target-standardized pre-gap RidgeCV R2 for the continuous task."""
    x_scaler = StandardScaler().fit(train_x)
    y_scaler = StandardScaler().fit(train_y)
    model = RidgeCV(alphas=RIDGE_ALPHAS)
    model.fit(x_scaler.transform(train_x), y_scaler.transform(train_y))
    prediction = y_scaler.inverse_transform(
        model.predict(x_scaler.transform(val_x)))
    per_target = r2_score(val_y, prediction, multioutput="raw_values")
    alpha = np.asarray(model.alpha_)
    return {
        "metric": "r2",
        "value": float(r2_score(val_y, prediction)),
        "per_target": [float(value) for value in per_target],
        "selected_alpha": (float(alpha) if alpha.ndim == 0
                           else [float(value) for value in alpha]),
        "alphas": [float(value) for value in RIDGE_ALPHAS],
        "feature": "last_four_pre_gap_latents_concatenated",
        "probe": "StandardScaler(X)+StandardScaler(y)+RidgeCV",
    }


def _zip_info(name: str, compression_level: int) -> zipfile.ZipInfo:
    """Create a ZIP member with fixed metadata for byte-stable archives."""
    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    info._compresslevel = compression_level  # Python exposes no public setter.
    return info


def write_npz_deterministic(path: Path, arrays: Mapping[str, np.ndarray],
                            compression_level: int = 1,
                            overwrite: bool = False) -> str:
    """Atomically write an allow-pickle-free NPZ with stable member metadata."""
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be in [0, 9]")
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(temporary, mode="w", allowZip64=True) as archive:
            for name, value in arrays.items():
                if not name or "/" in name or name.endswith(".npy"):
                    raise ValueError(f"invalid NPZ key {name!r}")
                array = np.asanyarray(value)
                if array.dtype.hasobject:
                    raise ValueError(f"object arrays are forbidden: {name}")
                with archive.open(
                        _zip_info(name, compression_level), mode="w",
                        force_zip64=True) as member:
                    np.lib.format.write_array(member, array,
                                              allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def _array_metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {"shape": list(np.asarray(value).shape),
               "dtype": str(np.asarray(value).dtype)}
        for name, value in arrays.items()
    }


def _action_method(source_stream: str) -> str:
    if source_stream == "clean":
        return "per_column_train_zscore_of_native_5x2_action_block"
    return "per_column_train_zscore_then_repeat_2d_block_five_times"


def cache_split(*, model: torch.nn.Module, device: torch.device,
                source_path: Path, destination: Path, task: str, split: str,
                source_stream: str, action_mean: np.ndarray,
                action_std: np.ndarray, weights_sha256: str,
                frame_batch_size: int, compression_level: int,
                overwrite: bool, quiet: bool) -> tuple[dict[str, Any],
                                                       np.ndarray,
                                                       np.ndarray]:
    """Encode and persist one split, returning probe features and targets."""
    source = source_record(source_path)
    bank = load_bank(source_path, verify=False)  # source_record already verified.
    if bank.task != task:
        raise ValueError(f"{source_path} has task={bank.task!r}, expected {task}")
    if split == "train" and bank.num_episodes != source["num_episodes"]:
        raise ValueError("bank sidecar episode count mismatch")
    expected_dim = OFFICIAL_ACTION_DIM if source_stream == "clean" else 2
    if bank.actions.shape[-1] != expected_dim:
        raise ValueError(
            f"{task}/{split}/{source_stream}: expected action width "
            f"{expected_dim}, got {bank.actions.shape[-1]}")
    z = encode_frames(
        model, bank.frames, device, frame_batch_size,
        None if quiet else f"{task}/{split}/{source_stream}")
    probe_x = availability_features(z, bank)
    probe_y = np.array(bank.xi, copy=True)
    actions = transform_actions(
        bank.actions, action_mean, action_std, source_stream)
    payload: dict[str, np.ndarray] = {
        "z": z,
        "actions": actions,
        "xi": np.array(bank.xi, copy=False),
        "endo_state": np.array(bank.endo_state, copy=False),
        "exo_state": np.array(bank.exo_state, copy=False),
    }
    for name in sorted(bank.events):
        payload[f"event_{name}"] = np.array(bank.events[name], copy=False)
    metadata: dict[str, Any] = {
        "schema": SCHEMA,
        "task": task,
        "split": split,
        "source_stream": source_stream,
        "z_semantics": f"frozen_official_encoder({source_stream}_frames)",
        "representation_label_training": False,
        "labels_used_only_for": "post_hoc_availability_probe",
        "source_bank": source,
        "official_checkpoint": {
            "source": OFFICIAL_SOURCE,
            "source_commit": OFFICIAL_SOURCE_COMMIT,
            "sha256": weights_sha256,
        },
        "preprocessing": {
            "function": "lewm.models.official_lewm.preprocess_frames",
            "resize": [OFFICIAL_IMAGE_SIZE, OFFICIAL_IMAGE_SIZE],
            "normalization": "ImageNet_mean_std",
            "dtype": "float32",
        },
        "action_transform": {
            "method": _action_method(source_stream),
            "training_mean": [float(value) for value in action_mean],
            "training_std_ddof0": [float(value) for value in action_std],
            "output_dim": OFFICIAL_ACTION_DIM,
        },
        "frame_batch_size": frame_batch_size,
        "arrays": _array_metadata(payload),
    }
    payload["meta_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    artifact_sha256 = write_npz_deterministic(
        destination, payload, compression_level, overwrite)
    sidecar = {
        **metadata,
        "artifact": {
            "path": destination.name,
            "sha256": artifact_sha256,
            "compression": f"zip_deflate_level_{compression_level}",
            "sidecar": destination.name + ".json",
        },
    }
    sidecar_path = destination.with_suffix(destination.suffix + ".json")
    sidecar_sha256 = _atomic_text(sidecar_path, _stable_json(sidecar))
    record = {
        "split": split,
        "path": destination.name,
        "sha256": artifact_sha256,
        "sidecar": sidecar_path.name,
        "sidecar_sha256": sidecar_sha256,
        "source_bank": source,
        "shape": list(z.shape),
    }
    return record, probe_x, probe_y


def _preflight_output(task_dir: Path, overwrite: bool) -> None:
    paths = [task_dir / f"{split}.npz" for split in SPLITS]
    paths += [task_dir / f"{split}.npz.json" for split in SPLITS]
    paths += [task_dir / "availability.json", task_dir / "manifest.json",
              task_dir / "manifest.sha256"]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing cache files: "
            + ", ".join(str(path) for path in existing))


def cache_task(*, task: str, args: argparse.Namespace,
               model: torch.nn.Module, device: torch.device,
               weights_sha256: str) -> dict[str, Any]:
    task_dir = Path(args.output) / task
    _preflight_output(task_dir, args.overwrite)
    paths = {
        split: discover_bank(Path(args.data_root), task, split,
                             args.source_stream)
        for split in SPLITS
    }

    # Only the selected training bank determines normalization.  Validation
    # data and task labels never influence the action transform.
    training_bank = load_bank(paths["train"])
    expected_dim = OFFICIAL_ACTION_DIM if args.source_stream == "clean" else 2
    if training_bank.actions.shape[-1] != expected_dim:
        raise ValueError(
            f"{task}/train/{args.source_stream}: expected action width "
            f"{expected_dim}, got {training_bank.actions.shape[-1]}")
    action_mean, action_std = action_statistics(training_bank.actions)
    del training_bank

    records: list[dict[str, Any]] = []
    features: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for split in SPLITS:
        record, features[split], targets[split] = cache_split(
            model=model, device=device, source_path=paths[split],
            destination=task_dir / f"{split}.npz", task=task, split=split,
            source_stream=args.source_stream, action_mean=action_mean,
            action_std=action_std, weights_sha256=weights_sha256,
            frame_batch_size=args.frame_batch_size,
            compression_level=args.compression_level,
            overwrite=args.overwrite, quiet=args.quiet)
        records.append(record)

    if task in ("t1", "t3"):
        availability = categorical_availability(
            features["train"], targets["train"],
            features["val"], targets["val"])
    else:
        availability = continuous_availability(
            features["train"], targets["train"],
            features["val"], targets["val"])
    availability.update({
        "task": task,
        "train_episodes": int(len(targets["train"])),
        "val_episodes": int(len(targets["val"])),
        "representation_frozen": True,
        "representation_label_training": False,
    })
    availability_path = task_dir / "availability.json"
    availability_sha256 = _atomic_text(
        availability_path, _stable_json(availability))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "schema": SCHEMA,
        "task": task,
        "source_stream": args.source_stream,
        "official_checkpoint": {
            "path": _display_path(Path(args.weights)),
            "source": OFFICIAL_SOURCE,
            "source_commit": OFFICIAL_SOURCE_COMMIT,
            "sha256": weights_sha256,
            "latent_dim": OFFICIAL_EMBED_DIM,
            "action_dim": OFFICIAL_ACTION_DIM,
        },
        "determinism": {
            "seed": 0,
            "fixed_flattened_frame_order": True,
            "frame_batch_size": args.frame_batch_size,
            "float32_inference": True,
            "tf32": False,
            "torch_deterministic_algorithms": True,
        },
        "action_transform": {
            "method": _action_method(args.source_stream),
            "training_mean": [float(value) for value in action_mean],
            "training_std_ddof0": [float(value) for value in action_std],
        },
        "artifacts": records,
        "availability": availability,
        "availability_file": {
            "path": availability_path.name,
            "sha256": availability_sha256,
        },
        "software": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    manifest_path = task_dir / "manifest.json"
    manifest_sha256 = _atomic_text(manifest_path, _stable_json(manifest))
    _atomic_text(task_dir / "manifest.sha256",
                 f"{manifest_sha256}  manifest.json\n")
    if not args.quiet:
        print(f"[official-cache] {task}: availability "
              f"{availability['metric']}={availability['value']:.4f}; "
              f"manifest={manifest_sha256}", flush=True)
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS,
                        default=list(TASKS))
    parser.add_argument("--data-root", type=Path,
                        default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-stream", choices=SOURCE_STREAMS,
                        default="clean",
                        help=("clean: native 10-D official-timescale banks; "
                              "observed: explicit legacy 2-D fallback"))
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--expected-weights-sha256",
                        default=OFFICIAL_WEIGHTS_SHA256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frame-batch-size", type=int,
                        default=FRAME_BATCH_SIZE)
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.frame_batch_size <= 0:
        raise ValueError("--frame-batch-size must be positive")
    if not 0 <= args.compression_level <= 9:
        raise ValueError("--compression-level must be in [0,9]")
    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"missing official checkpoint {args.weights}")
    # Fail before loading a GPU model or encoding frames if any task directory
    # would be ambiguous or overwritten.
    for task in args.tasks:
        _preflight_output(Path(args.output) / task, args.overwrite)
        for split in SPLITS:
            discover_bank(Path(args.data_root), task, split,
                          args.source_stream)

    weights_sha256 = sha256_file(args.weights)
    if args.expected_weights_sha256 \
            and weights_sha256 != args.expected_weights_sha256:
        raise ValueError(
            f"official checkpoint hash mismatch: {weights_sha256} != "
            f"{args.expected_weights_sha256}")
    configure_determinism(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    model = load_official_reacher_checkpoint(args.weights, device)
    model.eval()
    for parameter in model.parameters():
        if parameter.requires_grad:
            raise AssertionError("official model must be fully frozen")
    if not args.quiet:
        print(f"[official-cache] checkpoint={weights_sha256} device={device} "
              f"tasks={','.join(args.tasks)} stream={args.source_stream}",
              flush=True)
    for task in args.tasks:
        cache_task(task=task, args=args, model=model, device=device,
                   weights_sha256=weights_sha256)


if __name__ == "__main__":
    main()
