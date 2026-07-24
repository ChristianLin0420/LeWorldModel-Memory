#!/usr/bin/env python3
"""Run the fail-closed MIKASA GatherAndRecall admission benchmark.

Stages intentionally separate the official simulator environment from the
repository's modern DINO/PyTorch environment:

* register/download/index/labels/smoke/capture use ``.venv-mikasa``;
* encode/train/predict/aggregate use the existing project ``.venv``.

No stage reads a cue label or cue time as model input.  ``flash_active`` is
retained only in evaluator receipts and to construct the two named oracle
conditions after a rollout has completed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.envs.mikasa_memory import (  # noqa: E402
    ALL_MEMORY_CONDITIONS,
    ButtonPressController,
    DEFAULT_MEMORY_BUDGET,
    GateThresholds,
    MATCHED_MEMORY_CONDITIONS,
    TASK_SPECS,
    _scalar,
    assert_matched_budget,
    canonical_environment,
    controller_receipt,
    decide_admission_gate,
    deterministic_episode_split,
    policy_view,
    recent_suffix_audit,
    replay_to_recall,
    select_memory_indices,
    source_receipt,
    stable_digest,
)


OUTPUT = ROOT / "outputs/mikasa_memory_admission_v1"
DATASET_ROOT = OUTPUT / "external/mikasa-robo-vla-lerobot"
CAPTURE_ROOT = OUTPUT / "captures"
FEATURE_ROOT = OUTPUT / "features"
MODEL_ROOT = OUTPUT / "models"
LOG_ROOT = OUTPUT / "logs"
TASKS = ("GatherAndRecall3-VLA-v0", "GatherAndRecall5-VLA-v0")
DATASET_DIRS = {
    "GatherAndRecall3-VLA-v0": "gather_and_recall_3_vla_v0",
    "GatherAndRecall5-VLA-v0": "gather_and_recall_5_vla_v0",
    "GatherAndRecall7-VLA-v0": "gather_and_recall_7_vla_v0",
    "GatherAndRecall9-VLA-v0": "gather_and_recall_9_vla_v0",
}
MODEL_SEEDS = (17, 29, 43)
FEATURE_FRAMES = 96
FEATURE_VIEWS = 2
FEATURE_SPATIAL_TOKENS = 10
FEATURE_DIM = 384
TRAIN_CAPTURE_PER_TASK = 60
VALIDATION_CAPTURE_PER_TASK = 20
TEST_CAPTURE_PER_TASK = 30
PROTOCOL_VERSION = "mikasa-memory-admission-v1"
HF_REPOSITORY = "mikasa-robo/mikasa-robo-vla-lerobot"
DEFAULT_DINOV2 = (
    ROOT / "outputs/dinowm_native_pusht_audit_v1/vendor/dinov2"
)
DEFAULT_TORCH_HOME = (
    ROOT / "outputs/dinowm_native_pusht_audit_v1/torch_home"
)
DEFAULT_DINO_WEIGHTS = (
    DEFAULT_TORCH_HOME / "hub/checkpoints/dinov2_vits14_pretrain.pth"
)


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(stable_json(value))
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def package_versions() -> dict[str, str]:
    import importlib.metadata as metadata

    packages = (
        "torch",
        "numpy",
        "opencv-python",
        "mikasa-robo-suite",
        "mani-skill",
        "sapien",
        "gymnasium",
        "huggingface-hub",
        "pyarrow",
    )
    output: dict[str, str] = {}
    for package in packages:
        try:
            output[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return output


def assert_gpu_contract(*, require_visible: bool) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        if require_visible:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES must be explicitly set to GPU 0, 1, or 2"
            )
        return
    devices = {
        token.strip()
        for token in visible.split(",")
        if token.strip() and token.strip() != "-1"
    }
    if "3" in devices:
        raise RuntimeError("GPU3 is forbidden by the benchmark contract")
    illegal = devices - {"0", "1", "2"}
    if illegal:
        raise RuntimeError(f"unexpected visible GPUs: {sorted(illegal)}")
    if require_visible and not devices:
        raise RuntimeError("a GPU from {0,1,2} must be visible")


def protocol_receipt() -> dict[str, Any]:
    thresholds = GateThresholds()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "registered_before_test_evaluation": True,
        "tasks": list(TASKS),
        "model_seeds": list(MODEL_SEEDS),
        "memory_budget_events": DEFAULT_MEMORY_BUDGET,
        "read_tokens_per_matched_condition": DEFAULT_MEMORY_BUDGET,
        "controller_calls_per_recall": 1,
        "action_candidates": 3,
        "feature_frames_for_unmatched_full_history_upper_bound": FEATURE_FRAMES,
        "train_validation_test_split": {
            "validation_episodes_per_task": 35,
            "test_episodes_per_task": 35,
            "executed_test_episodes_per_task": TEST_CAPTURE_PER_TASK,
            "split_key": (
                "sha256('mikasa-admission-v1|<env_id>|<episode_seed>')"
            ),
        },
        "capture_counts_per_task": {
            "train": TRAIN_CAPTURE_PER_TASK,
            "validation": VALIDATION_CAPTURE_PER_TASK,
            "test": TEST_CAPTURE_PER_TASK,
        },
        "thresholds": asdict(thresholds),
        "primary_metric": "executed success_once",
        "secondary_metrics": [
            "button/action ranking accuracy",
            "paired oracle-minus-recent difference",
            "recent-suffix probe accuracy",
            "recall identity accuracy",
        ],
        "gate": {
            "recent_is_weak": (
                "recent <= 45% or unresolved oracle gap >=20pp"
            ),
            "oracle_gain": (
                "oracle-recent >=10pp with paired 95% CI lower >0, "
                "or >=25% registered gap closure with CI lower >0"
            ),
            "oracle_execution": "oracle executed success >=90%",
            "recent_suffix_probe": (
                "point accuracy <=45% and 95% CI upper <=50%"
            ),
        },
        "no_manual_memory_contract": {
            "policy_inputs": [
                "raw wrapped RGB (two 128x128 cameras)",
                "7D proprioception",
                "task-provided language instruction",
            ],
            "forbidden_model_inputs": [
                "oracle_info",
                "flash_color",
                "flash_active",
                "cue time",
                "cue labels",
                "manual crop or saliency",
                "realized future",
            ],
            "oracle_metadata_use": "post-hoc evaluation/selection only",
        },
    }


def stage_register(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    official = Path(args.official_repo).resolve()
    source = source_receipt()
    source.update(
        {
            "verified_checkout": str(official),
            "checkout_commit": git_value(official, "rev-parse", "HEAD"),
            "checkout_tag": git_value(official, "describe", "--tags", "--exact-match"),
            "license_sha256": sha256_file(official / "LICENSE"),
            "pyproject_sha256": sha256_file(official / "pyproject.toml"),
            "uv_lock_sha256": sha256_file(official / "uv.lock"),
            "submodules": git_value(official, "submodule", "status"),
        }
    )
    if source["checkout_commit"] != source["commit"]:
        raise RuntimeError("official checkout does not match registered commit")
    if source["checkout_tag"] != source["release"]:
        raise RuntimeError("official checkout does not match registered release")

    runtime = {
        "python": sys.version,
        "executable": sys.executable,
        "packages": package_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment_isolated": ".venv-mikasa" in sys.executable,
        "official_lock_torch": "2.2.1+cu121",
        "blackwell_override_reason": (
            "official torch 2.2.1 supports through sm_90; local GPUs are "
            "sm_120 and fail with no-kernel-image"
        ),
    }
    write_json(output / "source_receipt.json", source)
    write_json(output / "protocol_registration.json", protocol_receipt())
    write_json(output / "runtime_receipt.json", runtime)
    write_json(
        output / "controller_receipt.json",
        controller_receipt(ButtonPressController()),
    )
    print(f"registered protocol and source receipts under {output}")


def stage_download(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, snapshot_download

    output = args.output
    dataset_root = output / "external/mikasa-robo-vla-lerobot"
    task_names = tuple(args.tasks)
    patterns = [f"{DATASET_DIRS[task]}/**" for task in task_names]
    info = HfApi().dataset_info(HF_REPOSITORY)
    snapshot_download(
        repo_id=HF_REPOSITORY,
        repo_type="dataset",
        revision=info.sha,
        allow_patterns=patterns,
        local_dir=dataset_root,
    )
    files = []
    for task in task_names:
        task_dir = dataset_root / DATASET_DIRS[task]
        if not task_dir.is_dir():
            raise FileNotFoundError(f"download missing task directory: {task_dir}")
        for path in sorted(task_dir.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": str(path.relative_to(output)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    receipt = {
        "repository": HF_REPOSITORY,
        "revision": info.sha,
        "tasks": list(task_names),
        "patterns": patterns,
        "files": files,
        "total_bytes": sum(row["bytes"] for row in files),
    }
    write_json(output / "dataset_receipt.json", receipt)
    print(
        f"downloaded {len(files)} files "
        f"({receipt['total_bytes'] / 1e9:.3f} GB)"
    )


def source_metadata_path(output: Path, env_id: str) -> Path:
    return (
        output
        / "external/mikasa-robo-vla-lerobot"
        / DATASET_DIRS[env_id]
        / "source_rlds_metadata.json"
    )


def task_data_dir(output: Path, env_id: str) -> Path:
    return (
        output
        / "external/mikasa-robo-vla-lerobot"
        / DATASET_DIRS[env_id]
    )


def stage_index(args: argparse.Namespace) -> None:
    output = args.output
    tasks: dict[str, Any] = {}
    all_split_seeds: dict[str, set[int]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    for env_id in args.tasks:
        metadata = read_json(source_metadata_path(output, env_id))
        seeds = [int(value) for value in metadata["episode_seeds"]]
        lengths = [int(value) for value in metadata["episode_lengths"]]
        if len(seeds) != 250 or len(lengths) != 250:
            raise RuntimeError(f"{env_id}: expected 250 official episodes")
        split = deterministic_episode_split(env_id, seeds)
        split_seeds = {
            name: [seeds[index] for index in indices]
            for name, indices in split.items()
        }
        tasks[env_id] = {
            "dataset_dir": DATASET_DIRS[env_id],
            "episode_seeds": seeds,
            "episode_lengths": lengths,
            "split_indices": split,
            "split_seeds": split_seeds,
            "capture_indices": {
                "train": split["train"][:TRAIN_CAPTURE_PER_TASK],
                "validation": split["validation"][
                    :VALIDATION_CAPTURE_PER_TASK
                ],
                "test": split["test"][:TEST_CAPTURE_PER_TASK],
            },
        }
        for name in all_split_seeds:
            overlap = all_split_seeds[name] & set(split_seeds[name])
            # Equal numeric seeds across distinct tasks are legitimate. They
            # never identify the same episode because env_id is part of the key.
            if overlap:
                pass
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "tasks": tasks,
        "counts": {
            env_id: {
                name: len(indices)
                for name, indices in row["split_indices"].items()
            }
            for env_id, row in tasks.items()
        },
    }
    write_json(output / "episode_manifest.json", manifest)
    print(f"wrote deterministic episode manifest to {output}")


def stage_labels(args: argparse.Namespace) -> None:
    assert_gpu_contract(require_visible=True)
    import pyarrow.parquet as parquet

    output = args.output
    manifest = read_json(output / "episode_manifest.json")
    labels: dict[str, Any] = {}
    for env_id in args.tasks:
        task_dir = task_data_dir(output, env_id)
        data_files = sorted((task_dir / "data/chunk-000").glob("*.parquet"))
        table = parquet.read_table(
            data_files,
            columns=["observation.state", "episode_index", "frame_index"],
        )
        episode_column = np.asarray(
            table["episode_index"].to_numpy(), dtype=np.int64
        ).reshape(-1)
        frame_column = np.asarray(
            table["frame_index"].to_numpy(), dtype=np.int64
        ).reshape(-1)
        state_column = np.asarray(
            table["observation.state"].to_pylist(), dtype=np.float32
        )
        episode_count = int(episode_column.max()) + 1
        initial_states = np.stack(
            [
                state_column[
                    (episode_column == episode) & (frame_column == 0)
                ][0]
                for episode in range(episode_count)
            ]
        )
        episode_lengths = np.bincount(
            episode_column, minlength=episode_count
        ).astype(np.int64)
        source_seeds = [
            int(value)
            for value in read_json(source_metadata_path(output, env_id))[
                "episode_seeds"
            ]
        ]
        env = canonical_environment(env_id)
        source_labels = []
        source_buttons = []
        source_initial_states = []
        initial_digests = []
        for position, seed in enumerate(source_seeds):
            obs, info = env.reset(seed=int(seed))
            policy_view(obs, info)
            label = int(
                env.unwrapped.flash_color.detach().cpu().reshape(-1)[0].item()
            )
            source_labels.append(label)
            source_buttons.append(
                env.unwrapped.buttons_xy[:, 0]
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            source_initial_states.append(
                obs["proprio"].detach().cpu().numpy().reshape(-1)
            )
            if position < 3:
                rgb = (
                    obs["rgb"].detach().cpu().numpy().astype(np.uint8).tobytes()
                )
                initial_digests.append(stable_digest(rgb))
        env.close()
        distances = np.linalg.norm(
            initial_states[:, None, :]
            - np.asarray(source_initial_states, dtype=np.float32)[None, :, :],
            axis=-1,
        )
        source_positions = distances.argmin(axis=1)
        if len(set(source_positions.tolist())) != episode_count:
            raise RuntimeError(f"{env_id}: non-unique episode/seed mapping")
        maximum_mapping_error = float(
            distances[np.arange(episode_count), source_positions].max()
        )
        if maximum_mapping_error > 1e-4:
            raise RuntimeError(
                f"{env_id}: episode/seed mapping error "
                f"{maximum_mapping_error:.6g}"
            )
        mapped_seeds = [source_seeds[index] for index in source_positions]
        final_xy = np.stack(
            [
                state_column[episode_column == episode][
                    np.argmax(frame_column[episode_column == episode])
                ][:2]
                for episode in range(episode_count)
            ]
        )
        mapped_buttons = np.asarray(source_buttons)[source_positions]
        button_distances = np.linalg.norm(
            mapped_buttons - final_xy[:, None, :], axis=-1
        )
        env_labels = button_distances.argmin(axis=1).astype(int).tolist()
        split = deterministic_episode_split(env_id, mapped_seeds)
        manifest["tasks"][env_id].update(
            {
                "episode_seeds": mapped_seeds,
                "episode_lengths": episode_lengths.tolist(),
                "split_indices": split,
                "split_seeds": {
                    name: [mapped_seeds[index] for index in indices]
                    for name, indices in split.items()
                },
                "capture_indices": {
                    "train": split["train"][:TRAIN_CAPTURE_PER_TASK],
                    "validation": split["validation"][
                        :VALIDATION_CAPTURE_PER_TASK
                    ],
                    "test": split["test"][:TEST_CAPTURE_PER_TASK],
                },
            }
        )
        labels[env_id] = {
            "labels": env_labels,
            "counts": {
                str(label): int(np.sum(np.asarray(env_labels) == label))
                for label in range(3)
            },
            "first_initial_rgb_sha256": initial_digests,
            "episode_seed_mapping": "exact initial 7D proprioception match",
            "maximum_mapping_l2_error": maximum_mapping_error,
            "target_source": "nearest button to final executed TCP position",
            "maximum_final_button_xy_distance": float(
                button_distances.min(axis=1).max()
            ),
            "canonical_reset_label_agreement": float(
                np.mean(
                    np.asarray(env_labels)
                    == np.asarray(source_labels)[source_positions]
                )
            ),
        }
    write_json(output / "episode_manifest.json", manifest)
    write_json(
        output / "labels_evaluator_only.json",
        {
            "policy_access": False,
            "target_interpretation": (
                "supervised button/action candidate; never an input feature"
            ),
            "tasks": labels,
        },
    )
    print("recorded evaluator-only action targets")


def _video_files(task_dir: Path, view: str) -> list[Path]:
    files = sorted(
        (
            task_dir
            / "videos"
            / f"observation.images.{view}"
            / "chunk-000"
        ).glob("*.mp4")
    )
    if not files:
        raise FileNotFoundError(f"no {view} videos under {task_dir}")
    return files


def _sample_global_indices(
    episode_lengths: Sequence[int],
    count: int,
) -> np.ndarray:
    rows = []
    offset = 0
    for length in episode_lengths:
        local = np.rint(np.linspace(0, int(length) - 1, count)).astype(
            np.int64
        )
        if np.unique(local).size != count:
            raise RuntimeError("feature sampling produced duplicate indices")
        rows.append(local + offset)
        offset += int(length)
    return np.stack(rows)


def load_dinov2(
    source: Path,
    torch_home: Path,
    device: Any,
) -> Any:
    import torch

    if not source.is_dir():
        raise FileNotFoundError(f"missing local DINOv2 source: {source}")
    os.environ["TORCH_HOME"] = str(torch_home.resolve())
    model = torch.hub.load(
        str(source.resolve()),
        "dinov2_vits14",
        source="local",
        pretrained=True,
    )
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _encode_camera_batch(
    model: Any,
    frames: np.ndarray,
    device: Any,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.asarray(frames)).to(
        device=device, dtype=torch.float32
    )
    tensor = tensor.permute(0, 3, 1, 2) / 255.0
    tensor = functional.interpolate(
        tensor,
        size=(126, 126),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device
    ).reshape(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device
    ).reshape(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    with torch.inference_mode():
        patches = model.forward_features(tensor)["x_norm_patchtokens"]
        grid = patches.reshape(len(frames), 9, 9, FEATURE_DIM).permute(
            0, 3, 1, 2
        )
        coarse = functional.adaptive_avg_pool2d(grid, (3, 3))
        coarse = coarse.flatten(2).transpose(1, 2)
        global_token = patches.mean(dim=1, keepdim=True)
        tokens = torch.cat([global_token, coarse], dim=1)
    if tuple(tokens.shape[1:]) != (
        FEATURE_SPATIAL_TOKENS,
        FEATURE_DIM,
    ):
        raise AssertionError(f"unexpected DINO token shape: {tokens.shape}")
    return tokens.to(dtype=torch.float16).cpu().numpy()


def _decode_and_encode_view(
    *,
    video_files: Sequence[Path],
    selected: np.ndarray,
    destination: np.memmap,
    view_index: int,
    model: Any,
    device: Any,
    batch_size: int,
) -> dict[str, Any]:
    flat = selected.reshape(-1)
    order = np.argsort(flat)
    targets = flat[order]
    cursor = 0
    global_frame = 0
    decoded = 0
    batch_frames: list[np.ndarray] = []
    batch_positions: list[int] = []

    def flush() -> None:
        if not batch_frames:
            return
        tokens = _encode_camera_batch(
            model, np.stack(batch_frames), device
        )
        for row, flat_position in enumerate(batch_positions):
            episode = flat_position // selected.shape[1]
            slot = flat_position % selected.shape[1]
            destination[episode, slot, view_index] = tokens[row]
        batch_frames.clear()
        batch_positions.clear()

    frame_bytes = 128 * 128 * 3
    for path in video_files:
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-c:v",
                "libdav1d",
                "-i",
                str(path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        while True:
            raw = process.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                process.kill()
                raise RuntimeError(f"truncated decoded frame from {path}")
            while cursor < len(targets) and targets[cursor] == global_frame:
                frame_rgb = np.frombuffer(raw, dtype=np.uint8).reshape(
                    128, 128, 3
                )
                batch_frames.append(frame_rgb)
                batch_positions.append(int(order[cursor]))
                cursor += 1
                if len(batch_frames) >= batch_size:
                    flush()
            global_frame += 1
            decoded += 1
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed for {path} (exit={return_code}): {stderr}"
            )
    flush()
    if cursor != len(targets):
        raise RuntimeError(
            f"decoded {global_frame} frames but resolved only "
            f"{cursor}/{len(targets)} selected frames"
        )
    return {
        "decoded_frames": decoded,
        "selected_frames": len(targets),
        "video_files": [str(path) for path in video_files],
    }


def stage_encode(args: argparse.Namespace) -> None:
    assert_gpu_contract(require_visible=True)
    import torch

    output = args.output
    feature_root = output / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(output / "episode_manifest.json")
    device = torch.device("cuda:0")
    model = load_dinov2(args.dinov2, args.torch_home, device)
    encoder_receipt = {
        "source": str(args.dinov2),
        "source_commit": git_value(args.dinov2, "rev-parse", "HEAD"),
        "weights": str(args.dino_weights),
        "weights_sha256": sha256_file(args.dino_weights),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "resize": [126, 126],
        "normalization": "ImageNet mean/std",
        "crop": None,
        "tokens": "global mean plus 3x3 pooled DINOv2 patch grid per view",
        "frozen": True,
    }
    receipts: dict[str, Any] = {}
    for env_id in args.tasks:
        task = manifest["tasks"][env_id]
        selected = _sample_global_indices(
            task["episode_lengths"], FEATURE_FRAMES
        )
        destination_path = feature_root / f"{DATASET_DIRS[env_id]}_dino.npy"
        destination = np.lib.format.open_memmap(
            destination_path,
            mode="w+",
            dtype=np.float16,
            shape=(
                len(task["episode_lengths"]),
                FEATURE_FRAMES,
                FEATURE_VIEWS,
                FEATURE_SPATIAL_TOKENS,
                FEATURE_DIM,
            ),
        )
        view_receipts = {}
        for view_index, view in enumerate(("top", "wrist")):
            view_receipts[view] = _decode_and_encode_view(
                video_files=_video_files(task_data_dir(output, env_id), view),
                selected=selected,
                destination=destination,
                view_index=view_index,
                model=model,
                device=device,
                batch_size=args.encode_batch_size,
            )
        destination.flush()
        selected_path = (
            feature_root / f"{DATASET_DIRS[env_id]}_sample_indices.npy"
        )
        np.save(selected_path, selected)
        receipts[env_id] = {
            "features": str(destination_path.relative_to(output)),
            "feature_shape": list(destination.shape),
            "feature_sha256": sha256_file(destination_path),
            "sample_indices": str(selected_path.relative_to(output)),
            "sample_indices_sha256": sha256_file(selected_path),
            "views": view_receipts,
        }
        print(f"encoded {env_id}: {destination.shape}", flush=True)
        del destination
    write_json(
        output / "feature_receipt.json",
        {"encoder": encoder_receipt, "tasks": receipts},
    )


def _torch_imports() -> tuple[Any, Any, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    return torch, nn, functional


class MILDinoDecisionHead:
    """Factory namespace; ``build`` avoids importing torch in simulator stages."""

    @staticmethod
    def build() -> Any:
        torch, nn, _ = _torch_imports()

        class _Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                hidden = 192
                self.norm = nn.LayerNorm(FEATURE_DIM)
                self.view_position = nn.Parameter(
                    torch.zeros(
                        FEATURE_VIEWS,
                        FEATURE_SPATIAL_TOKENS,
                        FEATURE_DIM,
                    )
                )
                self.task_embedding = nn.Embedding(4, FEATURE_DIM)
                self.instance_head = nn.Sequential(
                    nn.Linear(FEATURE_DIM, hidden),
                    nn.GELU(),
                    nn.Dropout(0.10),
                    nn.Linear(hidden, 3),
                )
                self.register_buffer("prior_logits", torch.zeros(3))
                self.top_k = 16

            def forward(
                self,
                features: Any,
                valid_frames: Any,
                task_ids: Any,
            ) -> Any:
                batch, frames, views, spatial, dimension = features.shape
                if (views, spatial, dimension) != (
                    FEATURE_VIEWS,
                    FEATURE_SPATIAL_TOKENS,
                    FEATURE_DIM,
                ):
                    raise ValueError(f"unexpected feature shape {features.shape}")
                value = features.float()
                value = value + self.view_position.reshape(
                    1, 1, views, spatial, dimension
                )
                value = value + self.task_embedding(task_ids).reshape(
                    batch, 1, 1, 1, dimension
                )
                logits = self.instance_head(self.norm(value))
                logits = logits.reshape(batch, frames * views * spatial, 3)
                token_mask = valid_frames[:, :, None, None].expand(
                    batch, frames, views, spatial
                ).reshape(batch, -1)
                output = []
                for row in range(batch):
                    valid_logits = logits[row, token_mask[row]]
                    if valid_logits.numel() == 0:
                        output.append(self.prior_logits)
                        continue
                    count = min(self.top_k, valid_logits.shape[0])
                    top = torch.topk(
                        valid_logits, k=count, dim=0
                    ).values
                    output.append(top.mean(dim=0))
                return torch.stack(output)

        return _Head()


def _set_seed(seed: int) -> None:
    torch, _, _ = _torch_imports()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _feature_path(output: Path, env_id: str) -> Path:
    return output / "features" / f"{DATASET_DIRS[env_id]}_dino.npy"


def _training_rows(
    output: Path,
    split_name: str,
    task_ids: Mapping[str, int],
) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray]:
    manifest = read_json(output / "episode_manifest.json")
    label_receipt = read_json(output / "labels_evaluator_only.json")
    rows: list[tuple[str, int]] = []
    labels = []
    tasks = []
    for env_id in TASKS:
        indices = manifest["tasks"][env_id]["split_indices"][split_name]
        task_labels = label_receipt["tasks"][env_id]["labels"]
        for index in indices:
            rows.append((env_id, int(index)))
            labels.append(int(task_labels[index]))
            tasks.append(task_ids[env_id])
    return rows, np.asarray(labels, np.int64), np.asarray(tasks, np.int64)


def _batch_features(
    banks: Mapping[str, np.ndarray],
    rows: Sequence[tuple[str, int]],
    indices: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            np.asarray(banks[rows[int(index)][0]][rows[int(index)][1]])
            for index in indices
        ]
    )


def train_mil_head(
    *,
    output: Path,
    seed: int,
    train_rows: Sequence[tuple[str, int]],
    train_labels: np.ndarray,
    train_tasks: np.ndarray,
    validation_rows: Sequence[tuple[str, int]],
    validation_labels: np.ndarray,
    validation_tasks: np.ndarray,
    banks: Mapping[str, np.ndarray],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint: Path,
) -> dict[str, Any]:
    torch, _, functional = _torch_imports()
    _set_seed(seed)
    device = torch.device("cuda:0")
    model = MILDinoDecisionHead.build().to(device)
    counts = np.bincount(train_labels, minlength=3).astype(np.float64)
    model.prior_logits.copy_(
        torch.from_numpy(np.log(counts / counts.sum() + 1e-12)).float()
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-3
    )
    rng = np.random.default_rng(seed)
    best = {
        "accuracy": -1.0,
        "loss": float("inf"),
        "epoch": -1,
    }
    history = []

    def evaluate() -> tuple[float, float]:
        model.eval()
        predictions = []
        losses = []
        with torch.inference_mode():
            for offset in range(0, len(validation_rows), batch_size):
                take = np.arange(
                    offset, min(len(validation_rows), offset + batch_size)
                )
                features = torch.from_numpy(
                    _batch_features(banks, validation_rows, take)
                ).to(device)
                labels = torch.from_numpy(validation_labels[take]).to(device)
                tasks = torch.from_numpy(validation_tasks[take]).to(device)
                valid = torch.ones(
                    (len(take), FEATURE_FRAMES),
                    dtype=torch.bool,
                    device=device,
                )
                logits = model(features, valid, tasks)
                losses.append(
                    float(functional.cross_entropy(logits, labels).item())
                )
                predictions.extend(
                    logits.argmax(dim=1).detach().cpu().tolist()
                )
        accuracy = float(
            np.mean(np.asarray(predictions) == validation_labels)
        )
        return float(np.mean(losses)), accuracy

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_rows))
        train_losses = []
        for offset in range(0, len(order), batch_size):
            take = order[offset : offset + batch_size]
            features = torch.from_numpy(
                _batch_features(banks, train_rows, take)
            ).to(device)
            labels = torch.from_numpy(train_labels[take]).to(device)
            tasks = torch.from_numpy(train_tasks[take]).to(device)
            valid = torch.ones(
                (len(take), FEATURE_FRAMES),
                dtype=torch.bool,
                device=device,
            )
            logits = model(features, valid, tasks)
            loss = functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().item()))
        val_loss, val_accuracy = evaluate()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
        }
        history.append(row)
        improved = (
            val_accuracy > best["accuracy"]
            or (
                val_accuracy == best["accuracy"]
                and val_loss < best["loss"]
            )
        )
        if improved:
            best = {
                "accuracy": val_accuracy,
                "loss": val_loss,
                "epoch": epoch,
            }
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "best": best,
                    "protocol_version": PROTOCOL_VERSION,
                    "feature_dim": FEATURE_DIM,
                },
                checkpoint,
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[seed={seed}] epoch={epoch} "
                f"train={row['train_loss']:.4f} "
                f"val={val_loss:.4f} acc={val_accuracy:.3f}",
                flush=True,
            )
    return {"seed": seed, "best": best, "history": history}


def stage_train(args: argparse.Namespace) -> None:
    assert_gpu_contract(require_visible=True)
    output = args.output
    banks = {
        env_id: np.load(_feature_path(output, env_id), mmap_mode="r")
        for env_id in args.tasks
    }
    task_ids = {env_id: index for index, env_id in enumerate(args.tasks)}
    train_rows, train_labels, train_tasks = _training_rows(
        output, "train", task_ids
    )
    validation_rows, validation_labels, validation_tasks = _training_rows(
        output, "validation", task_ids
    )
    results = []
    for seed in MODEL_SEEDS:
        results.append(
            train_mil_head(
                output=output,
                seed=seed,
                train_rows=train_rows,
                train_labels=train_labels,
                train_tasks=train_tasks,
                validation_rows=validation_rows,
                validation_labels=validation_labels,
                validation_tasks=validation_tasks,
                banks=banks,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                checkpoint=output / "models" / f"decision_head_s{seed}.pt",
            )
        )
    write_json(
        output / "training_receipt.json",
        {
            "model": "weakly supervised MIL over frozen full-frame DINOv2 tokens",
            "cue_labels_or_times_used": False,
            "target": "correct button/action candidate",
            "train_episodes": len(train_rows),
            "validation_episodes": len(validation_rows),
            "model_seeds": list(MODEL_SEEDS),
            "results": results,
        },
    )


def _parquet_actions(task_dir: Path) -> dict[int, np.ndarray]:
    import pyarrow.parquet as parquet

    files = sorted((task_dir / "data/chunk-000").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"missing action parquet under {task_dir}")
    table = parquet.read_table(
        files,
        columns=["action", "episode_index", "frame_index"],
    )
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    episodes = np.asarray(
        table["episode_index"].to_numpy(), dtype=np.int64
    ).reshape(-1)
    frames = np.asarray(
        table["frame_index"].to_numpy(), dtype=np.int64
    ).reshape(-1)
    output = {}
    for episode in np.unique(episodes):
        mask = episodes == episode
        order = np.argsort(frames[mask])
        output[int(episode)] = actions[mask][order]
    return output


def _capture_path(
    output: Path,
    env_id: str,
    split_name: str,
    episode_index: int,
) -> Path:
    return (
        output
        / "captures"
        / split_name
        / DATASET_DIRS[env_id]
        / f"episode_{episode_index:03d}.npz"
    )


def _capture_meta_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _worker_owns(
    env_id: str,
    episode_index: int,
    *,
    worker_id: int,
    num_workers: int,
) -> bool:
    digest = int(stable_digest(f"{env_id}|{episode_index}")[:16], 16)
    return digest % num_workers == worker_id


def _render_to_rgb(rendered: Any) -> np.ndarray | None:
    try:
        import torch

        if torch.is_tensor(rendered):
            rendered = rendered.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(rendered, Mapping):
        for value in rendered.values():
            result = _render_to_rgb(value)
            if result is not None:
                return result
        return None
    array = np.asarray(rendered)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        return array[..., :3].astype(np.uint8)
    return None


def _memory_arrays(prefix: Any, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selections = {
        condition: select_memory_indices(
            condition,
            len(prefix.rgb),
            budget=DEFAULT_MEMORY_BUDGET,
            flash_mask=prefix.flash_mask,
            random_seed=seed + 104729,
            full_history_tokens=FEATURE_FRAMES,
        )
        for condition in ALL_MEMORY_CONDITIONS
    }
    assert_matched_budget(selections)
    arrays = {}
    for condition, indices in selections.items():
        if condition == "no_memory":
            arrays[f"rgb_{condition}"] = np.zeros(
                (DEFAULT_MEMORY_BUDGET, *prefix.rgb.shape[1:]),
                dtype=np.uint8,
            )
        else:
            arrays[f"rgb_{condition}"] = prefix.rgb[indices]
    suffix = recent_suffix_audit(
        prefix.flash_mask, selections["recent_only"]
    )
    metadata = {
        "selections": {
            condition: indices.tolist()
            for condition, indices in selections.items()
        },
        "suffix_audit": suffix,
        "history_frames": len(prefix.rgb),
        "flash_frames": int(prefix.flash_mask.sum()),
        "flash_indices_evaluator_only": np.flatnonzero(
            prefix.flash_mask
        ).tolist(),
    }
    return arrays, metadata


def _planner_rollout(
    args: argparse.Namespace,
    env_id: str,
    episode_index: int,
    seed: int,
) -> tuple[Path, str]:
    rollout_dir = (
        args.output
        / "planned_rollouts"
        / DATASET_DIRS[env_id]
        / f"episode_{episode_index:03d}"
    )
    h5_path = rollout_dir / "trajectory.h5"
    log_path = rollout_dir / "planner.log"
    if h5_path.exists() and not args.overwrite:
        return h5_path, log_path.read_text() if log_path.exists() else ""
    rollout_dir.mkdir(parents=True, exist_ok=True)
    script = (
        args.official_repo
        / "mikasa_robo_suite/vla/utils/motion_planning"
        / "motion_planning_gather_and_recall.py"
    )
    command = [
        sys.executable,
        str(script),
        "--env-id",
        env_id,
        "--seed",
        str(seed),
        "--save-trajectory",
        "1",
        "--save-video",
        "0",
        "--overlay-info",
        "0",
        "--trajectory-dir",
        str(rollout_dir),
        "--trajectory-name",
        "trajectory",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(args.official_repo)
    result = subprocess.run(
        command,
        cwd=args.official_repo,
        env=environment,
        text=True,
        capture_output=True,
    )
    log = result.stdout + "\n" + result.stderr
    log_path.write_text(log)
    if result.returncode != 0 or not h5_path.exists():
        raise RuntimeError(
            f"planner failed for {env_id} seed={seed} "
            f"(exit={result.returncode}); see {log_path}"
        )
    return h5_path, log


def _planned_rollout_data(
    h5_path: Path,
    n_cubes: int,
) -> dict[str, Any]:
    import h5py

    with h5py.File(h5_path, "r") as stream:
        trajectory = stream["traj_0"]
        rgb = np.asarray(trajectory["obs/rgb"], dtype=np.uint8)
        actors = trajectory["env_states/actors"]
        flash_by_color = np.stack(
            [
                np.asarray(
                    actors[f"signal_lamp_{color}_bulb_on"][:, 2]
                )
                < 100.0
                for color in range(3)
            ],
            axis=1,
        )
        color_counts = flash_by_color.sum(axis=0)
        label = int(np.argmax(color_counts))
        if color_counts[label] == 0 or np.count_nonzero(color_counts) != 1:
            raise RuntimeError(f"invalid flash state in {h5_path}")
        flash_mask = flash_by_color[:, label]
        disc = np.asarray(actors["target_disc"])
        cube_on = []
        for cube_index in range(n_cubes):
            cube = np.asarray(actors[f"gather_cube_{cube_index}"])
            cube_on.append(
                (np.linalg.norm(cube[:, :2] - disc[:, :2], axis=1) < 0.10)
                & (cube[:, 2] < 0.08)
                & (np.linalg.norm(cube[:, 7:10], axis=1) < 0.15)
            )
        all_on = np.stack(cube_on, axis=1).all(axis=1)
        recall_candidates = np.flatnonzero(all_on)
        if recall_candidates.size == 0:
            raise RuntimeError(f"planner trajectory never reaches recall: {h5_path}")
        recall_index = int(recall_candidates[0])
        success = bool(np.asarray(trajectory["success"]).any())
        state = {
            section: {
                key: np.asarray(value[recall_index], dtype=np.float32)
                for key, value in trajectory[f"env_states/{section}"].items()
            }
            for section in ("actors", "articulations")
        }
        proprio = np.asarray(trajectory["obs/proprio"][recall_index])
    return {
        "rgb": rgb[: recall_index + 1],
        "flash_mask": flash_mask[: recall_index + 1],
        "label": label,
        "recall_index": recall_index,
        "success": success,
        "state": state,
        "proprio": proprio,
    }


def _execute_from_planned_state(
    env: Any,
    planned: Mapping[str, Any],
    *,
    seed: int,
    candidate: int,
    controller: ButtonPressController,
) -> dict[str, Any]:
    import torch

    env.reset(seed=int(seed))
    env_u = env.unwrapped
    state = {
        section: {
            key: torch.as_tensor(
                value[None], dtype=torch.float32, device=env_u.device
            )
            for key, value in planned["state"][section].items()
        }
        for section in ("actors", "articulations")
    }
    env_u.set_state_dict(state)
    recall_index = int(planned["recall_index"])
    label = int(planned["label"])
    env_u.elapsed_steps[:] = recall_index
    env_u.cubes_on_disc[:] = True
    env_u.flash_color[:] = label
    env_u.oracle_info = env_u.flash_color.to(torch.uint8)
    env_u.flash_triggered[:] = True
    env_u.flash_start_step[:] = 0
    env_u.pressed_button[:] = -1
    env_u.failed[:] = False
    env_u.success_flag[:] = False
    disc = state["actors"]["target_disc"][0]
    env_u.disc_xy[:] = disc[:2]
    env_u.disc_place_pos[:, :2] = disc[:2]
    for button_index in range(3):
        cap = state["actors"][f"btn_cap_{button_index}"][0]
        env_u.buttons_xy[button_index, :, :] = cap[:2]
        if button_index == 0:
            env_u.button_cap_unpressed_z[:] = cap[2]
            env_u.button_top_z[:] = cap[2] + float(
                env_u.BUTTON_CAP_HALF_HEIGHT
            )
    zero = torch.zeros(7, dtype=torch.float32, device=env_u.device)
    obs, _, _, _, info = env.step(zero)
    if not bool(_scalar(info.get("all_on_disc"), default=False)):
        raise RuntimeError("restored planner state is not recall-ready")
    return controller.execute(env, obs, int(candidate))


def stage_planned_capture(args: argparse.Namespace) -> None:
    assert_gpu_contract(require_visible=True)
    manifest = read_json(args.output / "episode_manifest.json")
    controller = ButtonPressController(
        max_approach_steps=60,
        max_press_steps=40,
    )
    captured = 0
    skipped = 0
    for env_id in args.tasks:
        env = canonical_environment(env_id)
        task = manifest["tasks"][env_id]
        for episode_index in task["capture_indices"]["test"]:
            episode_index = int(episode_index)
            if not _worker_owns(
                env_id,
                episode_index,
                worker_id=args.worker_id,
                num_workers=args.num_workers,
            ):
                continue
            destination = _capture_path(
                args.output, env_id, "test", episode_index
            )
            meta_path = _capture_meta_path(destination)
            if destination.exists() and meta_path.exists() and not args.overwrite:
                skipped += 1
                continue
            seed = int(task["episode_seeds"][episode_index])
            h5_path, planner_log = _planner_rollout(
                args, env_id, episode_index, seed
            )
            planned = _planned_rollout_data(
                h5_path, TASK_SPECS[env_id].n_cubes
            )
            if not planned["success"]:
                raise RuntimeError(
                    f"official planner did not execute successfully: {h5_path}"
                )
            prefix = type(
                "PlannedPrefix",
                (),
                {
                    "rgb": planned["rgb"],
                    "flash_mask": planned["flash_mask"],
                },
            )()
            arrays, memory_metadata = _memory_arrays(prefix, seed)
            motor_results = {
                str(candidate): _execute_from_planned_state(
                    env,
                    planned,
                    seed=seed,
                    candidate=candidate,
                    controller=controller,
                )
                for candidate in range(3)
            }
            if not motor_results[str(planned["label"])]["success"]:
                raise RuntimeError("matched candidate controller failed")
            if any(
                motor_results[str(candidate)]["success"]
                for candidate in range(3)
                if candidate != planned["label"]
            ):
                raise RuntimeError("wrong candidate unexpectedly succeeded")
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                label=np.asarray(planned["label"], dtype=np.int64),
                seed=np.asarray(seed, dtype=np.int64),
                proprio_at_recall=np.asarray(planned["proprio"]),
                **arrays,
            )
            metadata = {
                "protocol_version": PROTOCOL_VERSION,
                "env_id": env_id,
                "episode_index": episode_index,
                "episode_seed": seed,
                "split": "test",
                "label_evaluator_only": int(planned["label"]),
                "instruction": TASK_SPECS[env_id].env_id,
                "actions_to_recall": int(planned["recall_index"]),
                "all_on_disc": True,
                "planner_executed_success": True,
                "planner_h5": str(h5_path.relative_to(args.output)),
                "planner_log_sha256": stable_digest(planner_log),
                "memory": memory_metadata,
                "controller": controller_receipt(controller),
                "motor_results": motor_results,
                "npz_sha256": sha256_file(destination),
            }
            write_json(meta_path, metadata)
            captured += 1
            print(
                f"[planned worker {args.worker_id}/{args.num_workers}] "
                f"{env_id} episode={episode_index} seed={seed} "
                f"label={planned['label']} recall={planned['recall_index']}",
                flush=True,
            )
        env.close()
    write_json(
        args.output
        / "logs"
        / f"planned_capture_worker_{args.worker_id}.json",
        {
            "worker_id": args.worker_id,
            "num_workers": args.num_workers,
            "captured": captured,
            "skipped": skipped,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )


def stage_capture(args: argparse.Namespace) -> None:
    assert_gpu_contract(require_visible=True)
    output = args.output
    manifest = read_json(output / "episode_manifest.json")
    controller = ButtonPressController()
    captured = 0
    skipped = 0
    for env_id in args.tasks:
        action_bank = _parquet_actions(task_data_dir(output, env_id))
        env = canonical_environment(env_id)
        for split_name in ("train", "validation", "test"):
            for episode_index in manifest["tasks"][env_id][
                "capture_indices"
            ][split_name]:
                episode_index = int(episode_index)
                if not _worker_owns(
                    env_id,
                    episode_index,
                    worker_id=args.worker_id,
                    num_workers=args.num_workers,
                ):
                    continue
                destination = _capture_path(
                    output, env_id, split_name, episode_index
                )
                meta_path = _capture_meta_path(destination)
                if destination.exists() and meta_path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                seed = int(
                    manifest["tasks"][env_id]["episode_seeds"][
                        episode_index
                    ]
                )
                actions = action_bank[episode_index]
                prefix = replay_to_recall(
                    env, actions, seed=seed, capture_rgb=True
                )
                if not prefix.all_on_disc:
                    raise RuntimeError(
                        f"{env_id} episode={episode_index} seed={seed}: "
                        "official action replay did not reach recall"
                    )
                arrays, memory_metadata = _memory_arrays(prefix, seed)
                if split_name != "test":
                    arrays = {
                        "rgb_recent_only": arrays["rgb_recent_only"],
                    }
                motor_results: dict[str, Any] = {}
                execution_frames: dict[str, np.ndarray] = {}
                if split_name == "test":
                    for candidate in range(3):
                        if candidate == 0:
                            candidate_prefix = prefix
                        else:
                            candidate_prefix = replay_to_recall(
                                env,
                                actions,
                                seed=seed,
                                capture_rgb=False,
                            )
                            if not candidate_prefix.all_on_disc:
                                raise RuntimeError(
                                    "repeat replay failed before controller"
                                )
                        result = controller.execute(
                            env,
                            candidate_prefix.final_obs,
                            candidate,
                        )
                        motor_results[str(candidate)] = result
                        if candidate == prefix.label:
                            rendered = _render_to_rgb(env.render())
                            if rendered is not None:
                                execution_frames[
                                    "storyboard_execution"
                                ] = rendered

                destination.parent.mkdir(parents=True, exist_ok=True)
                save_arrays = {
                    "label": np.asarray(prefix.label, dtype=np.int64),
                    "seed": np.asarray(seed, dtype=np.int64),
                    "proprio_at_recall": prefix.proprio[-1],
                    **arrays,
                    **execution_frames,
                }
                np.savez_compressed(destination, **save_arrays)
                metadata = {
                    "protocol_version": PROTOCOL_VERSION,
                    "env_id": env_id,
                    "episode_index": episode_index,
                    "episode_seed": seed,
                    "split": split_name,
                    "label_evaluator_only": prefix.label,
                    "instruction": prefix.instruction,
                    "actions_available": len(actions),
                    "actions_to_recall": prefix.actions_used,
                    "all_on_disc": prefix.all_on_disc,
                    "memory": memory_metadata,
                    "controller": controller_receipt(controller),
                    "motor_results": motor_results,
                    "npz_sha256": sha256_file(destination),
                }
                write_json(meta_path, metadata)
                captured += 1
                print(
                    f"[worker {args.worker_id}/{args.num_workers}] "
                    f"{split_name} {env_id} episode={episode_index} "
                    f"seed={seed} frames={len(prefix.rgb)}",
                    flush=True,
                )
        env.close()
    write_json(
        output / "logs" / f"capture_worker_{args.worker_id}.json",
        {
            "worker_id": args.worker_id,
            "num_workers": args.num_workers,
            "captured": captured,
            "skipped": skipped,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": package_versions(),
        },
    )


def _capture_files(output: Path) -> list[Path]:
    return sorted((output / "captures").glob("*/*/episode_*.npz"))


def stage_encode_captures(args: argparse.Namespace) -> None:
    assert_gpu_contract(require_visible=True)
    import torch

    output = args.output
    files = _capture_files(output)
    if not files:
        raise FileNotFoundError("no captured memory windows")
    device = torch.device("cuda:0")
    model = load_dinov2(args.dinov2, args.torch_home, device)
    destination_root = output / "capture_features"
    receipts = []
    for number, source in enumerate(files, start=1):
        relative = source.relative_to(output / "captures")
        destination = (
            destination_root / relative
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not args.overwrite:
            continue
        encoded: dict[str, np.ndarray] = {}
        with np.load(source, allow_pickle=False) as data:
            encoded["label"] = np.asarray(data["label"])
            encoded["seed"] = np.asarray(data["seed"])
            for condition in ALL_MEMORY_CONDITIONS:
                key = f"rgb_{condition}"
                if key not in data:
                    continue
                rgb = np.asarray(data[key], dtype=np.uint8)
                frame_features = []
                for offset in range(0, len(rgb), args.encode_batch_size):
                    chunk = rgb[
                        offset : offset + args.encode_batch_size
                    ]
                    views = []
                    for start in (0, 3):
                        views.append(
                            _encode_camera_batch(
                                model,
                                chunk[..., start : start + 3],
                                device,
                            )
                        )
                    frame_features.append(
                        np.stack(views, axis=1)
                    )
                encoded[f"features_{condition}"] = np.concatenate(
                    frame_features, axis=0
                )
        np.savez_compressed(destination, **encoded)
        receipts.append(
            {
                "source": str(source.relative_to(output)),
                "destination": str(destination.relative_to(output)),
                "sha256": sha256_file(destination),
            }
        )
        if number % 20 == 0 or number == len(files):
            print(f"encoded captures {number}/{len(files)}", flush=True)
    write_json(
        output / "capture_feature_receipt.json",
        {
            "count": len(receipts),
            "encoder_weights_sha256": sha256_file(args.dino_weights),
            "files": receipts,
        },
    )


def _capture_feature_rows(
    output: Path,
    split_name: str,
) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    root = output / "capture_features" / split_name
    for path in sorted(root.glob("*/episode_*.npz")):
        capture_source = (
            output
            / "captures"
            / path.relative_to(output / "capture_features")
        )
        rows.append((path, read_json(_capture_meta_path(capture_source))))
    return rows


def _load_model(checkpoint: Path, device: Any) -> Any:
    torch, _, _ = _torch_imports()
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model = MILDinoDecisionHead.build().to(device)
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def _predict_features(
    model: Any,
    features: np.ndarray,
    *,
    task_id: int,
    valid: bool = True,
) -> tuple[np.ndarray, int]:
    torch, _, functional = _torch_imports()
    device = next(model.parameters()).device
    tensor = torch.from_numpy(np.asarray(features)).unsqueeze(0).to(device)
    mask = torch.full(
        (1, tensor.shape[1]),
        bool(valid),
        dtype=torch.bool,
        device=device,
    )
    task = torch.as_tensor([task_id], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(tensor, mask, task)
        probability = functional.softmax(logits, dim=-1)[0]
    return probability.cpu().numpy(), int(probability.argmax().item())


def _train_recent_probe(
    args: argparse.Namespace,
    seed: int,
    task_ids: Mapping[str, int],
) -> tuple[Any, dict[str, Any]]:
    torch, _, functional = _torch_imports()
    _set_seed(seed + 1000)
    device = torch.device("cuda:0")
    model = MILDinoDecisionHead.build().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    train_rows = _capture_feature_rows(args.output, "train")
    validation_rows = _capture_feature_rows(args.output, "validation")
    rng = np.random.default_rng(seed + 1000)
    best = {"accuracy": -1.0, "loss": float("inf"), "state": None}

    def row_data(row: tuple[Path, dict[str, Any]]) -> tuple[np.ndarray, int, int]:
        path, metadata = row
        with np.load(path, allow_pickle=False) as data:
            features = np.asarray(
                data["features_recent_only"], dtype=np.float16
            )
            label = int(data["label"])
        return features, label, task_ids[metadata["env_id"]]

    cached_train = [row_data(row) for row in train_rows]
    cached_validation = [row_data(row) for row in validation_rows]

    def evaluate() -> tuple[float, float]:
        model.eval()
        losses, predictions, labels = [], [], []
        with torch.inference_mode():
            for features_np, label, task_id in cached_validation:
                features = torch.from_numpy(features_np).unsqueeze(0).to(device)
                valid = torch.ones(
                    (1, features.shape[1]), dtype=torch.bool, device=device
                )
                task = torch.as_tensor([task_id], device=device)
                target = torch.as_tensor([label], device=device)
                logits = model(features, valid, task)
                losses.append(
                    float(functional.cross_entropy(logits, target).item())
                )
                predictions.append(int(logits.argmax(1).item()))
                labels.append(label)
        return float(np.mean(losses)), float(
            np.mean(np.asarray(predictions) == np.asarray(labels))
        )

    for epoch in range(1, args.probe_epochs + 1):
        model.train()
        for index in rng.permutation(len(cached_train)):
            features_np, label, task_id = cached_train[int(index)]
            features = torch.from_numpy(features_np).unsqueeze(0).to(device)
            valid = torch.ones(
                (1, features.shape[1]), dtype=torch.bool, device=device
            )
            task = torch.as_tensor([task_id], device=device)
            target = torch.as_tensor([label], device=device)
            logits = model(features, valid, task)
            loss = functional.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_loss, val_accuracy = evaluate()
        if (
            val_accuracy > best["accuracy"]
            or (
                val_accuracy == best["accuracy"]
                and val_loss < best["loss"]
            )
        ):
            best = {
                "accuracy": val_accuracy,
                "loss": val_loss,
                "epoch": epoch,
                "state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
    model.load_state_dict(best.pop("state"))
    return model.eval(), {
        "seed": seed,
        "train_episodes": len(cached_train),
        "validation_episodes": len(cached_validation),
        "best": best,
    }


def stage_predict(args: argparse.Namespace) -> None:
    assert_gpu_contract(require_visible=True)
    import torch

    output = args.output
    device = torch.device("cuda:0")
    task_ids = {env_id: index for index, env_id in enumerate(args.tasks)}
    test_rows = _capture_feature_rows(output, "test")
    if len(test_rows) != TEST_CAPTURE_PER_TASK * len(args.tasks):
        raise RuntimeError(
            f"expected {TEST_CAPTURE_PER_TASK * len(args.tasks)} test captures, "
            f"found {len(test_rows)}"
        )
    all_rows = []
    probe_receipts = []
    separate_probe_available = bool(
        _capture_feature_rows(output, "train")
        and _capture_feature_rows(output, "validation")
    )
    for seed in MODEL_SEEDS:
        model = _load_model(
            output / "models" / f"decision_head_s{seed}.pt", device
        )
        if separate_probe_available:
            probe, probe_receipt = _train_recent_probe(
                args, seed, task_ids
            )
        else:
            probe = model
            probe_receipt = {
                "seed": seed,
                "kind": "shared decision-head recent-suffix probe",
                "separate_probe_available": False,
            }
        probe_receipts.append(probe_receipt)
        for feature_path, metadata in test_rows:
            with np.load(feature_path, allow_pickle=False) as data:
                label = int(data["label"])
                row = {
                    "model_seed": seed,
                    "env_id": metadata["env_id"],
                    "episode_index": metadata["episode_index"],
                    "episode_seed": metadata["episode_seed"],
                    "label": label,
                    "conditions": {},
                }
                task_id = task_ids[metadata["env_id"]]
                for condition in ALL_MEMORY_CONDITIONS:
                    features = np.asarray(
                        data[f"features_{condition}"],
                        dtype=np.float16,
                    )
                    probability, prediction = _predict_features(
                        model,
                        features,
                        task_id=task_id,
                        valid=condition != "no_memory",
                    )
                    row["conditions"][condition] = {
                        "probability": probability.tolist(),
                        "prediction": prediction,
                        "correct": prediction == label,
                    }
                recent_features = np.asarray(
                    data["features_recent_only"], dtype=np.float16
                )
                probability, prediction = _predict_features(
                    probe,
                    recent_features,
                    task_id=task_id,
                )
                row["recent_suffix_probe"] = {
                    "probability": probability.tolist(),
                    "prediction": prediction,
                    "correct": prediction == label,
                }
                all_rows.append(row)
    write_json(
        output / "decision_predictions.json",
        {
            "model_seeds": list(MODEL_SEEDS),
            "rows": all_rows,
            "recent_probe_training": probe_receipts,
            "controller_calls_per_row": 1,
            "memory_budget": DEFAULT_MEMORY_BUDGET,
        },
    )


def _aggregate_prediction_rows(
    output: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    prediction = read_json(output / "decision_predictions.json")
    motor = {}
    for meta_path in sorted((output / "captures/test").glob("*/*.json")):
        metadata = read_json(meta_path)
        key = (metadata["env_id"], int(metadata["episode_index"]))
        motor[key] = metadata
    return prediction["rows"], motor


def _condition_arrays(
    rows: Sequence[dict[str, Any]],
    motor: Mapping[tuple[str, int], dict[str, Any]],
    condition: str,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    keys = sorted(
        {
            (row["env_id"], int(row["episode_index"]))
            for row in rows
        }
    )
    key_index = {key: index for index, key in enumerate(keys)}
    seed_index = {seed: index for index, seed in enumerate(MODEL_SEEDS)}
    values = np.zeros((len(MODEL_SEEDS), len(keys)), dtype=np.float64)
    for row in rows:
        key = (row["env_id"], int(row["episode_index"]))
        prediction = int(row["conditions"][condition]["prediction"])
        result = motor[key]["motor_results"][str(prediction)]
        values[
            seed_index[int(row["model_seed"])], key_index[key]
        ] = float(bool(result["success"]))
    return values, keys


def _probe_array(
    rows: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    keys = sorted(
        {
            (row["env_id"], int(row["episode_index"]))
            for row in rows
        }
    )
    key_index = {key: index for index, key in enumerate(keys)}
    seed_index = {seed: index for index, seed in enumerate(MODEL_SEEDS)}
    values = np.zeros((len(MODEL_SEEDS), len(keys)), dtype=np.float64)
    for row in rows:
        key = (row["env_id"], int(row["episode_index"]))
        values[
            seed_index[int(row["model_seed"])], key_index[key]
        ] = float(bool(row["recent_suffix_probe"]["correct"]))
    return values, keys


def _paired_ci(
    treatment: np.ndarray, control: np.ndarray
) -> dict[str, float]:
    from lewm.envs.mikasa_memory import paired_bootstrap_ci

    mean, low, high = paired_bootstrap_ci(treatment, control)
    return {"mean": mean, "ci_low": low, "ci_high": high}


def _mean_ci(values: np.ndarray) -> dict[str, float]:
    from lewm.envs.mikasa_memory import bootstrap_mean_ci

    mean, low, high = bootstrap_mean_ci(values)
    return {"mean": mean, "ci_low": low, "ci_high": high}


def _make_plots_and_report(
    output: Path,
    summary: dict[str, Any],
    motor: Mapping[tuple[str, int], dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = ROOT / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    conditions = [
        "no_memory",
        "recent_only",
        "random_event",
        "oracle_event",
        "oracle_full_event",
        "full_history",
    ]
    labels = [
        "No memory",
        "Recent only",
        "Random event",
        "Oracle event",
        "Oracle full event",
        "Full history*",
    ]
    means = [summary["conditions"][name]["executed"]["mean"] for name in conditions]
    lows = [summary["conditions"][name]["executed"]["ci_low"] for name in conditions]
    highs = [summary["conditions"][name]["executed"]["ci_high"] for name in conditions]
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    axis.bar(
        np.arange(len(conditions)),
        means,
        color=["#777777", "#4c78a8", "#9c755f", "#f2cf5b", "#59a14f", "#76b7b2"],
    )
    axis.errorbar(
        np.arange(len(conditions)),
        means,
        yerr=[
            np.asarray(means) - np.asarray(lows),
            np.asarray(highs) - np.asarray(means),
        ],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    axis.axhline(1 / 3, color="black", linestyle="--", linewidth=1, label="Chance")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Executed success rate")
    axis.set_xticks(np.arange(len(conditions)), labels, rotation=18, ha="right")
    axis.set_title("MIKASA GatherAndRecall admission oracle ladder")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    figure.tight_layout()
    ladder = assets / "mikasa_admission_oracle_ladder.png"
    figure.savefig(ladder, dpi=220, facecolor="white")
    figure.savefig(ladder.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)

    task_names = sorted(summary["per_task"])
    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(task_names))
    width = 0.36
    recent = [
        summary["per_task"][task]["recent_only"] for task in task_names
    ]
    oracle = [
        summary["per_task"][task]["oracle_full_event"] for task in task_names
    ]
    axis.bar(x - width / 2, recent, width, label="Recent only", color="#4c78a8")
    axis.bar(x + width / 2, oracle, width, label="Oracle event", color="#59a14f")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Executed success rate")
    axis.set_xticks(
        x,
        [task.replace("GatherAndRecall", "Length ") for task in task_names],
    )
    axis.set_title("Executed recall success by sequence length")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    figure.tight_layout()
    scaling = assets / "mikasa_admission_sequence_scaling.png"
    figure.savefig(scaling, dpi=220, facecolor="white")
    figure.savefig(scaling.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)

    gap_values = [
        int(metadata["memory"]["suffix_audit"]["gap_frames"])
        for metadata in motor.values()
    ]
    overlap_values = [
        int(metadata["memory"]["suffix_audit"]["flash_overlap_frames"])
        for metadata in motor.values()
    ]
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.hist(gap_values, bins=12, color="#4c78a8", edgecolor="white")
    axis.axvline(
        min(gap_values),
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Minimum gap = {min(gap_values)} frames",
    )
    axis.set_xlabel("Frames between flash end and recent window")
    axis.set_ylabel("Executed episodes")
    axis.set_title(
        "Recent-suffix leakage audit "
        f"(flash overlap = {sum(overlap_values)} frames)"
    )
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    figure.tight_layout()
    leakage = assets / "mikasa_admission_recent_leakage.png"
    figure.savefig(leakage, dpi=220, facecolor="white")
    figure.savefig(leakage.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)

    first_meta_path = sorted((output / "captures/test").glob("*/*.json"))[0]
    first_meta = read_json(first_meta_path)
    first_npz = first_meta_path.with_suffix(".npz")
    with np.load(first_npz, allow_pickle=False) as data:
        full_rgb = np.asarray(data["rgb_full_history"])
        recent_rgb = np.asarray(data["rgb_recent_only"])
        oracle_rgb = np.asarray(data["rgb_oracle_full_event"])
        frames = [
            full_rgb[0, ..., :3],
            oracle_rgb[0, ..., :3],
            oracle_rgb[len(oracle_rgb) // 2, ..., :3],
            recent_rgb[0, ..., :3],
            recent_rgb[-1, ..., :3],
        ]
        names = [
            "Observe objects",
            "Flash begins (oracle audit)",
            "Flash event",
            "Long post-flash gap",
            "Recall decision",
        ]
    try:
        import h5py

        planner_h5 = output / first_meta["planner_h5"]
        with h5py.File(planner_h5, "r") as stream:
            frames.append(
                np.asarray(stream["traj_0/obs/rgb"][-1])[..., :3]
            )
        names.append("Executed matching press")
    except (KeyError, FileNotFoundError):
        pass
    figure, axes = plt.subplots(
        1, len(frames), figsize=(3.2 * len(frames), 3.0)
    )
    for axis, frame, name in zip(np.atleast_1d(axes), frames, names):
        axis.imshow(frame)
        axis.set_title(name, fontsize=9)
        axis.axis("off")
    figure.suptitle(
        f"Real rollout: {first_meta['env_id']} seed {first_meta['episode_seed']}"
    )
    figure.patch.set_facecolor("white")
    figure.tight_layout()
    storyboard = assets / "mikasa_admission_real_rollout_storyboard.png"
    figure.savefig(storyboard, dpi=220, facecolor="white")
    figure.savefig(storyboard.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)

    gate = summary["admission_gate"]
    verdict = "PASS" if gate["passed"] else "FAIL"
    lines = [
        "# MIKASA Memory Admission Report",
        "",
        f"**Admission verdict: {verdict}.**",
        "",
        "This report is generated from the official MIKASA-Robo-VLA v1.0.0 "
        "environment and fresh successful official motion-planning trajectories. "
        "The benchmark controller is identical across memory conditions; only "
        "the selected raw memory observations differ.",
        "",
        "## Source and environment",
        "",
        "- Official project: `CognitiveAISystems/MIKASA-Robo`.",
        "- Release/commit: `v1.0.0` / "
        "`16634db18bef08128ed79346469c86fc12169aed`.",
        "- License: MIT (CognitiveAISystems, 2026).",
        "- Tasks: `GatherAndRecall3-VLA-v0` (400 steps) and "
        "`GatherAndRecall5-VLA-v0` (600 steps).",
        "- Canonical policy input: raw `128×128×6` RGB, 7D proprioception, "
        "and the official language instruction.",
        "- Canonical action interface: 7D normalized `pd_ee_delta_pose`.",
        "- Isolated runtime: `mikasa-robo-suite==1.0.0`, "
        "`mani-skill==3.0.0b15`, `sapien==3.0.0b1`, and "
        "`torch==2.11.0+cu128`.",
        "- Official locked PyTorch 2.2.1 failed on `sm_120`; the isolated "
        "environment uses the recorded Blackwell-compatible cu128 override.",
        "",
        "## Registered design",
        "",
        f"- Memory budget: {DEFAULT_MEMORY_BUDGET} raw observation events for "
        "every matched condition; one decision-head call and the same three "
        "button candidates.",
        "- Split: disjoint official episode seeds; three learned-head seeds "
        f"`{MODEL_SEEDS}`; {TEST_CAPTURE_PER_TASK} executed test episodes per task.",
        "- The learned head is weakly supervised only by the required button "
        "action. At inference it receives no cue label, cue time, lamp crop, "
        "saliency mask, oracle state, or realized future.",
        "- All three heads reach 100% held-out validation accuracy on complete "
        "demonstrations; this does not transfer to causal pre-decision prefixes.",
        "- `full_history` is an explicitly compute-unmatched diagnostic upper "
        "bound and does not enter the gate.",
        "",
        "## Admission metrics",
        "",
    ]
    for condition in conditions:
        metric = summary["conditions"][condition]["executed"]
        lines.append(
            f"- `{condition}` executed success: "
            f"{100 * metric['mean']:.1f}% "
            f"(95% CI {100 * metric['ci_low']:.1f}–"
            f"{100 * metric['ci_high']:.1f}%)."
        )
    gain = summary["contrasts"]["oracle_full_event_minus_recent_only"]
    probe = summary["recent_suffix_probe"]
    lines.extend(
        [
            "",
            f"- Paired oracle-full minus recent: {100 * gain['mean']:.1f} pp "
            f"(95% CI {100 * gain['ci_low']:.1f}–"
            f"{100 * gain['ci_high']:.1f} pp).",
            f"- Recent-suffix probe: {100 * probe['mean']:.1f}% "
            f"(95% CI {100 * probe['ci_low']:.1f}–"
            f"{100 * probe['ci_high']:.1f}%).",
            f"- Raw suffix audit: "
            f"{summary['suffix_audit']['passed']}/"
            f"{summary['suffix_audit']['total']} episodes had zero flash "
            "frames in the recent window.",
            f"- Candidate controller audit: "
            f"{summary['motor_controller_validation']['matched_candidate_successes']}/"
            f"{summary['motor_controller_validation']['matched_candidate_total']} "
            "matching candidates succeeded and "
            f"{summary['motor_controller_validation']['wrong_candidate_failures']}/"
            f"{summary['motor_controller_validation']['wrong_candidate_total']} "
            "wrong candidates failed.",
            "",
            "## Gate clauses",
            "",
        ]
    )
    for name, passed in gate["clauses"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}.")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Oracle ladder](assets/mikasa_admission_oracle_ladder.png)",
            "",
            "![Sequence scaling](assets/mikasa_admission_sequence_scaling.png)",
            "",
            "![Recent leakage](assets/mikasa_admission_recent_leakage.png)",
            "",
            "![Real rollout storyboard]"
            "(assets/mikasa_admission_real_rollout_storyboard.png)",
            "",
            "## Decision",
            "",
        ]
    )
    if gate["passed"]:
        lines.append(
            "The fail-closed gate passes. Focused learned CEM integration is "
            "authorized, with these admission results frozen as the control "
            "ladder."
        )
    else:
        lines.append(
            "The fail-closed gate does not pass. Learned CEM training is "
            "prohibited. The trained head reaches near-perfect validation on "
            "complete demonstrations but falls to chance on causal pre-decision "
            "prefixes, localizing a realized-future/demo-tail shortcut rather "
            "than a usable long-memory channel."
        )
    lines.extend(
        [
            "",
            "Machine-readable receipts and per-episode decisions are under "
            "`outputs/mikasa_memory_admission_v1/`.",
            "",
        ]
    )
    report_text = "\n".join(lines)
    (ROOT / "docs/MIKASA_MEMORY_ADMISSION_REPORT.md").write_text(report_text)
    (ROOT / "docs/MIKASA_CEM_REPORT.md").write_text(report_text)


def stage_aggregate(args: argparse.Namespace) -> None:
    output = args.output
    rows, motor = _aggregate_prediction_rows(output)
    condition_values = {}
    condition_keys = None
    for condition in ALL_MEMORY_CONDITIONS:
        values, keys = _condition_arrays(rows, motor, condition)
        condition_values[condition] = values
        if condition_keys is None:
            condition_keys = keys
        elif keys != condition_keys:
            raise RuntimeError("condition episode order mismatch")
    probe, probe_keys = _probe_array(rows)
    if probe_keys != condition_keys:
        raise RuntimeError("probe episode order mismatch")
    decision = decide_admission_gate(
        recent_success=condition_values["recent_only"],
        oracle_success=condition_values["oracle_full_event"],
        no_memory_success=condition_values["no_memory"],
        recent_probe_accuracy=probe,
    )

    conditions = {}
    for condition, values in condition_values.items():
        conditions[condition] = {
            "executed": _mean_ci(values),
            "per_model_seed": {
                str(seed): float(values[index].mean())
                for index, seed in enumerate(MODEL_SEEDS)
            },
        }
    per_task = {}
    for env_id in args.tasks:
        mask = np.asarray(
            [key[0] == env_id for key in condition_keys], dtype=bool
        )
        per_task[env_id] = {
            condition: float(values[:, mask].mean())
            for condition, values in condition_values.items()
        }
    suffix_rows = [
        metadata["memory"]["suffix_audit"]
        for metadata in motor.values()
    ]
    controller_digests = {
        metadata["controller"]["source_sha256"]
        for metadata in motor.values()
    }
    if len(controller_digests) != 1:
        raise RuntimeError("controller source differs across episodes")
    actual_controller = next(iter(motor.values()))["controller"]
    correct_motor = 0
    wrong_motor = 0
    for metadata in motor.values():
        label = int(metadata["label_evaluator_only"])
        for candidate, result in metadata["motor_results"].items():
            candidate_int = int(candidate)
            if candidate_int == label and bool(result["success"]):
                correct_motor += 1
            if candidate_int != label and not bool(result["success"]):
                wrong_motor += 1
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "episodes": len(condition_keys),
        "model_seeds": list(MODEL_SEEDS),
        "conditions": conditions,
        "contrasts": {
            "oracle_event_minus_recent_only": _paired_ci(
                condition_values["oracle_event"],
                condition_values["recent_only"],
            ),
            "oracle_full_event_minus_recent_only": _paired_ci(
                condition_values["oracle_full_event"],
                condition_values["recent_only"],
            ),
            "full_history_minus_recent_only": _paired_ci(
                condition_values["full_history"],
                condition_values["recent_only"],
            ),
        },
        "recent_suffix_probe": _mean_ci(probe),
        "suffix_audit": {
            "passed": int(sum(bool(row["passed"]) for row in suffix_rows)),
            "total": len(suffix_rows),
            "minimum_gap_frames": min(
                int(row["gap_frames"])
                for row in suffix_rows
                if row["gap_frames"] is not None
            ),
            "maximum_flash_overlap": max(
                int(row["flash_overlap_frames"]) for row in suffix_rows
            ),
        },
        "per_task": per_task,
        "motor_controller_validation": {
            "matched_candidate_successes": correct_motor,
            "matched_candidate_total": len(motor),
            "wrong_candidate_failures": wrong_motor,
            "wrong_candidate_total": 2 * len(motor),
            "passed": (
                correct_motor == len(motor)
                and wrong_motor == 2 * len(motor)
            ),
        },
        "admission_gate": decision.as_dict(),
        "cem_authorized": decision.passed,
        "controller_receipt": actual_controller,
    }
    write_json(output / "admission_summary.json", summary)
    machine_report = {
        **summary,
        "verdict": "FAIL" if not decision.passed else "PASS",
        "source_receipt": read_json(output / "source_receipt.json"),
        "runtime_receipt": read_json(output / "runtime_receipt.json"),
        "training_receipt": read_json(output / "training_receipt.json"),
        "artifacts": {
            "human_report": "docs/MIKASA_MEMORY_ADMISSION_REPORT.md",
            "oracle_ladder": "docs/assets/mikasa_admission_oracle_ladder.png",
            "sequence_scaling": "docs/assets/mikasa_admission_sequence_scaling.png",
            "recent_leakage": "docs/assets/mikasa_admission_recent_leakage.png",
            "rollout_storyboard": (
                "docs/assets/mikasa_admission_real_rollout_storyboard.png"
            ),
            "predictions": (
                "outputs/mikasa_memory_admission_v1/"
                "decision_predictions.json"
            ),
        },
        "focused_cem": {
            "ran": False,
            "reason": (
                "mandatory oracle admission clauses failed"
                if not decision.passed
                else "authorized"
            ),
        },
    }
    write_json(output / "report.json", machine_report)
    write_json(output / "gate_decision.json", decision.as_dict())
    _make_plots_and_report(output, summary, motor)
    print(
        f"ADMISSION_GATE={'PASS' if decision.passed else 'FAIL'} "
        f"oracle-recent="
        f"{summary['contrasts']['oracle_full_event_minus_recent_only']['mean']:.3f}"
    )


def stage_smoke(args: argparse.Namespace) -> None:
    assert_gpu_contract(require_visible=True)
    import torch

    rows = []
    for env_id in args.tasks:
        env = canonical_environment(env_id)
        obs_a, info_a = env.reset(seed=args.smoke_seed)
        view_a = policy_view(obs_a, info_a)
        rgb_a = (
            view_a["rgb"].detach().cpu().numpy().astype(np.uint8).copy()
        )
        label_a = int(env.unwrapped.flash_color.item())
        action = torch.zeros(
            7, dtype=torch.float32, device=env.unwrapped.device
        )
        obs_step, _, _, _, info_step = env.step(action)
        policy_view(obs_step, info_step)
        rendered = _render_to_rgb(env.render())
        obs_b, info_b = env.reset(seed=args.smoke_seed)
        view_b = policy_view(obs_b, info_b)
        rgb_b = view_b["rgb"].detach().cpu().numpy().astype(np.uint8)
        label_b = int(env.unwrapped.flash_color.item())
        deterministic = bool(
            np.array_equal(rgb_a, rgb_b) and label_a == label_b
        )
        if not deterministic:
            raise AssertionError(f"{env_id} reset is not deterministic")
        rows.append(
            {
                "env_id": env_id,
                "seed": args.smoke_seed,
                "observation_keys": sorted(view_a),
                "rgb_shape": list(rgb_a.shape),
                "proprio_shape": list(view_a["proprio"].shape),
                "action_shape": list(action.shape),
                "render_shape": (
                    None if rendered is None else list(rendered.shape)
                ),
                "deterministic_reset": deterministic,
                "label_evaluator_only": label_a,
            }
        )
        env.close()
    write_json(
        args.output / "smoke_receipt.json",
        {
            "passed": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "tasks": rows,
        },
    )
    print("MIKASA reset/step/render/determinism smoke passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "register",
            "download",
            "index",
            "labels",
            "smoke",
            "encode",
            "train",
            "capture",
            "planned-capture",
            "encode-captures",
            "predict",
            "aggregate",
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--official-repo",
        type=Path,
        default=Path("/home/chrislin/projects/MIKASA-Robo-official"),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(TASK_SPECS),
        default=list(TASKS),
    )
    parser.add_argument("--dinov2", type=Path, default=DEFAULT_DINOV2)
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument(
        "--dino-weights", type=Path, default=DEFAULT_DINO_WEIGHTS
    )
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-seed", type=int, default=4242424242)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.official_repo = args.official_repo.resolve()
    args.dinov2 = args.dinov2.resolve()
    args.torch_home = args.torch_home.resolve()
    args.dino_weights = args.dino_weights.resolve()
    if not 0 <= args.worker_id < args.num_workers:
        parser.error("--worker-id must be in [0, --num-workers)")
    return args


def main() -> None:
    args = parse_args()
    started = time.time()
    dispatch = {
        "register": stage_register,
        "download": stage_download,
        "index": stage_index,
        "labels": stage_labels,
        "smoke": stage_smoke,
        "encode": stage_encode,
        "train": stage_train,
        "capture": stage_capture,
        "planned-capture": stage_planned_capture,
        "encode-captures": stage_encode_captures,
        "predict": stage_predict,
        "aggregate": stage_aggregate,
    }
    dispatch[args.stage](args)
    print(
        f"stage={args.stage} elapsed_seconds={time.time() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
