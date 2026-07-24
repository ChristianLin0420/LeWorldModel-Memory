#!/usr/bin/env python3
"""Run the fail-closed RoboTwin-MeM oracle-admission benchmark.

The official simulator path is attempted only when all files referenced by the
official installation instructions exist.  The 2026-07-02 checkout omits its
ignored ``task_config/`` tree and both asset/data downloader implementations,
so this harness uses the official LeRobot 2.1 release and labels execution as
unavailable.  Dataset evaluation predicts action-derived candidate sequences;
task state labels and keyframe times are evaluator-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.envs.robotwin_mem import (  # noqa: E402
    ALL_MEMORY_CONDITIONS,
    CAMERA_KEYS,
    CAMERA_SHAPE,
    DEFAULT_MEMORY_BUDGET,
    FULL_HISTORY_FRAMES,
    MATCHED_MEMORY_CONDITIONS,
    OFFICIAL_COMMIT,
    OFFICIAL_DATASET_REVISION,
    PROTOCOL_VERSION,
    TASK_SPECS,
    assert_matched_budget,
    bootstrap_mean_ci,
    decide_admission_gate,
    deterministic_episode_split,
    frame_union_for_encoding,
    load_episode_records,
    paired_bootstrap_ci,
    raw_memory_bytes,
    recent_suffix_audit,
    select_memory_indices,
    source_receipt,
    stable_digest,
)


OUTPUT = ROOT / "outputs/robotwin_mem_admission_v1"
DATASET_ROOT = OUTPUT / "external/RoboTwin-MeM"
FEATURE_ROOT = OUTPUT / "features"
MODEL_ROOT = OUTPUT / "models"
LOG_ROOT = OUTPUT / "logs"
DEFAULT_OFFICIAL_REPO = Path("/home/chrislin/projects/EventVLA-official")
DEFAULT_DINOV2 = ROOT / "outputs/dinowm_native_pusht_audit_v1/vendor/dinov2"
DEFAULT_TORCH_HOME = ROOT / "outputs/dinowm_native_pusht_audit_v1/torch_home"
DEFAULT_DINO_WEIGHTS = (
    DEFAULT_TORCH_HOME / "hub/checkpoints/dinov2_vits14_pretrain.pth"
)
DEFAULT_QWEN_MODEL = OUTPUT / "external/Qwen3-VL-4B-Instruct"

TASKS = (
    "pick_the_unhidden_block",
    "pick_objects_in_order",
    "cover_blocks_hard",
)
MODEL_SEEDS = (17, 29, 43)
ACTION_HORIZON = 50
DINO_DIM = 384
DINO_SPATIAL_TOKENS = 10
DINO_VIEWS = 3
TRAINING_MODES = ("full_history", "auto_surprise", "random_event")
VALIDATION_EPISODES = 10
TEST_EPISODES = 10


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
        "torchvision",
        "numpy",
        "h5py",
        "opencv-python-headless",
        "av",
        "pyarrow",
        "huggingface-hub",
        "scikit-learn",
        "transformers",
        "accelerate",
        "qwen-vl-utils",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    return versions


def assert_gpu_contract(*, require_visible: bool) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        if require_visible:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES must explicitly select GPU 0, 1, or 2"
            )
        return
    devices = {
        token.strip()
        for token in visible.split(",")
        if token.strip() and token.strip() != "-1"
    }
    if "3" in devices:
        raise RuntimeError("GPU3 is forbidden by the RoboTwin-MeM contract")
    illegal = devices - {"0", "1", "2"}
    if illegal:
        raise RuntimeError(f"unexpected visible GPUs: {sorted(illegal)}")
    if require_visible and not devices:
        raise RuntimeError("one of GPU 0, 1, or 2 must be visible")


def simulator_blockers(official_repo: Path) -> list[dict[str, str]]:
    simulator = official_repo / "RoboTwin-Mem"
    required = {
        "task configuration tree": simulator / "task_config",
        "asset downloader implementation": simulator / "assets/_download.py",
        "data downloader implementation": simulator / "data/_download.py",
    }
    return [
        {
            "missing": label,
            "path": str(path),
            "referenced_by": (
                "official README/install scripts at repository commit "
                f"{OFFICIAL_COMMIT}"
            ),
        }
        for label, path in required.items()
        if not path.exists()
    ]


def local_dataset_receipt(dataset_root: Path) -> dict[str, Any]:
    files = []
    task_summaries = {}
    for task_id in TASKS:
        task_files = []
        for root in (
            dataset_root / "lerobot_2.1" / task_id,
            dataset_root / "hdf5" / task_id / "demo_clean",
        ):
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or ".cache" in path.parts:
                    continue
                relative = str(path.relative_to(dataset_root))
                row = {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                files.append(row)
                task_files.append(row)
        task_summaries[task_id] = {
            "files": len(task_files),
            "bytes": sum(row["bytes"] for row in task_files),
        }
    return {
        "repository": "ganlinyang/RoboTwin-MeM",
        "revision": OFFICIAL_DATASET_REVISION,
        "license": "Apache-2.0",
        "scope": (
            "complete LeRobot 2.1 target tasks plus HDF5 metadata/"
            "instructions; large duplicate HDF5 RGB trajectories not downloaded"
        ),
        "tasks": task_summaries,
        "files": files,
        "total_files": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
    }


def local_control_model_receipt(model_root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(model_root.iterdir()):
        if not path.is_file():
            continue
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "repository": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "license": "Apache-2.0",
        "files": files,
        "total_bytes": sum(row["bytes"] for row in files),
    }


def protocol_receipt() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "registered_before_test_evaluation": True,
        "tasks": list(TASKS),
        "model_seeds": list(MODEL_SEEDS),
        "episode_split": {
            "train": 50 - VALIDATION_EPISODES - TEST_EPISODES,
            "validation": VALIDATION_EPISODES,
            "test": TEST_EPISODES,
            "key": (
                "sha256(protocol|task|official_episode_index|official_seed)"
            ),
        },
        "memory": {
            "matched_raw_frame_budget": DEFAULT_MEMORY_BUDGET,
            "matched_raw_bytes": raw_memory_bytes(),
            "cameras": list(CAMERA_KEYS),
            "full_history_frames_unmatched_upper_bound": FULL_HISTORY_FRAMES,
            "conditions": list(ALL_MEMORY_CONDITIONS),
        },
        "controller": {
            "kind": "shared frozen-DINO action-sequence decision head",
            "action_candidates": "KMeans over official demonstrated 50-step action chunks",
            "action_horizon": ACTION_HORIZON,
            "same_head_across_conditions": True,
            "executed_success_available": False,
        },
        "primary_metric": (
            "exact delayed-query action-candidate sequence accuracy"
        ),
        "secondary_metrics": [
            "per-query action ranking accuracy",
            "mean reciprocal rank",
            "cross-entropy",
            "paired oracle-minus-recent confidence interval",
            "recent-suffix probe accuracy",
        ],
        "gate": {
            "oracle_gain": (
                "oracle event set minus recent >=10pp with paired 95% CI >0 "
                "or closes >=25% no-memory-to-perfect gap with CI >0"
            ),
            "recent_suffix": (
                "trained recent-only probe <= chance+10pp ceiling "
                "(CI upper <= ceiling+5pp)"
            ),
            "oracle_control": "oracle exact sequence accuracy >75%",
            "benchmark_suitable": "at least two tasks pass",
        },
        "no_manual_contract": {
            "policy_inputs": [
                "official raw 640x480 RGB from head/left-wrist/right-wrist",
                "official 14D proprioception",
                "official generic task instruction",
            ],
            "training_supervision": [
                "official demonstrated 14D actions",
                "generic task instruction",
            ],
            "forbidden_model_inputs": [
                "keyframe_steps",
                "scene_info",
                "event labels/times",
                "manual crop or frame selection",
                "handcrafted saliency",
                "realized future at inference",
            ],
            "oracle_metadata_use": "post-hoc evaluator selection only",
            "automatic_training_proposal": (
                "DINO temporal surprise over uniformly sampled history; "
                "no event labels or times"
            ),
        },
    }


def stage_register(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    official = args.official_repo
    if not (official / ".git").exists():
        raise FileNotFoundError(f"official checkout missing: {official}")
    checkout_commit = git_value(official, "rev-parse", "HEAD")
    if checkout_commit != OFFICIAL_COMMIT:
        raise RuntimeError(
            f"official checkout {checkout_commit} != registered {OFFICIAL_COMMIT}"
        )
    code_license = official / "RoboTwin-Mem/LICENSE"
    dataset_readme = DATASET_ROOT / "README.md"
    blockers = simulator_blockers(official)
    source = source_receipt()
    source.update(
        {
            "verified_checkout": str(official),
            "checkout_commit": checkout_commit,
            "checkout_clean": not bool(git_value(official, "status", "--short")),
            "robotwin_mem_license_sha256": sha256_file(code_license),
            "dataset_readme_sha256": (
                sha256_file(dataset_readme) if dataset_readme.exists() else None
            ),
            "simulator_execution": {
                "available": not blockers,
                "blockers": blockers,
                "fallback": (
                    "official LeRobot 2.1 action-ranking evaluation"
                    if blockers
                    else None
                ),
            },
        }
    )
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "environment_isolated": ".venv-robotwin-mem" in sys.executable,
        "packages": package_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "main_venv_modified": False,
    }
    write_json(output / "source_receipt.json", source)
    write_json(output / "runtime_receipt.json", runtime)
    write_json(output / "protocol_registration.json", protocol_receipt())
    if args.dataset_root.exists():
        write_json(
            output / "dataset_receipt.json",
            local_dataset_receipt(args.dataset_root),
        )
    if DEFAULT_QWEN_MODEL.exists():
        write_json(
            output / "control_model_receipt.json",
            local_control_model_receipt(DEFAULT_QWEN_MODEL),
        )
    print(
        "registered official source; simulator execution "
        f"{'available' if not blockers else 'blocked'}"
    )


def _episode_parquet(dataset_root: Path, task_id: str, episode: int) -> Path:
    return (
        dataset_root
        / "lerobot_2.1"
        / task_id
        / "data/chunk-000"
        / f"episode_{episode:06d}.parquet"
    )


def _episode_video(
    dataset_root: Path,
    task_id: str,
    episode: int,
    camera_key: str,
) -> Path:
    return (
        dataset_root
        / "lerobot_2.1"
        / task_id
        / "videos/chunk-000"
        / camera_key
        / f"episode_{episode:06d}.mp4"
    )


def _decode_selected_frames(
    path: Path, selected_indices: Sequence[int]
) -> dict[int, np.ndarray]:
    import av

    requested = {int(value) for value in selected_indices}
    if not requested:
        return {}
    output: dict[int, np.ndarray] = {}
    maximum = max(requested)
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in requested:
                array = frame.to_ndarray(format="rgb24")
                if array.shape != CAMERA_SHAPE:
                    raise AssertionError(
                        f"{path}: decoded shape {array.shape} != {CAMERA_SHAPE}"
                    )
                output[index] = array
            if index >= maximum:
                break
    missing = requested - set(output)
    if missing:
        raise RuntimeError(f"{path}: missing decoded frames {sorted(missing)}")
    return output


def stage_smoke(args: argparse.Namespace) -> None:
    assert_gpu_contract(require_visible=True)
    import pyarrow.parquet as parquet
    import torch

    task_id = args.tasks[0]
    records = load_episode_records(args.dataset_root, task_id)
    record = records[0]
    table = parquet.read_table(_episode_parquet(args.dataset_root, task_id, 0))
    if table.num_rows != record.length:
        raise AssertionError("parquet length differs from official metadata")
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if state.shape != (record.length, 14) or action.shape != (record.length, 14):
        raise AssertionError("unexpected official state/action arrays")
    cameras = {}
    for camera in CAMERA_KEYS:
        path = _episode_video(args.dataset_root, task_id, 0, camera)
        first_a = _decode_selected_frames(path, [0])[0]
        first_b = _decode_selected_frames(path, [0])[0]
        if not np.array_equal(first_a, first_b):
            raise AssertionError(f"{camera}: deterministic decode failed")
        cameras[camera] = {
            "shape": list(first_a.shape),
            "sha256": stable_digest(first_a.tobytes()),
            "file_sha256": sha256_file(path),
        }

    os.environ["TORCH_HOME"] = str(args.torch_home)
    model = torch.hub.load(
        str(args.dinov2),
        "dinov2_vits14",
        source="local",
        pretrained=True,
    ).eval().cuda()
    frame = _decode_selected_frames(
        _episode_video(args.dataset_root, task_id, 0, CAMERA_KEYS[0]), [0]
    )[0]
    tensor = (
        torch.from_numpy(frame.copy())
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .cuda()
    )
    tensor = torch.nn.functional.interpolate(
        tensor, (224, 224), mode="bilinear", align_corners=False
    )
    with torch.inference_mode():
        patches = model.forward_features(tensor)["x_norm_patchtokens"]
    receipt = {
        "passed": True,
        "mode": "official trajectory fallback",
        "simulator_execution_available": False,
        "task": task_id,
        "episode": 0,
        "length": record.length,
        "parquet_columns": table.column_names,
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "cameras": cameras,
        "deterministic_decode": True,
        "dino_patch_shape": list(patches.shape),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    write_json(args.output / "smoke_receipt.json", receipt)
    print("official LeRobot trajectory/DINO smoke passed")


def _load_actions(path: Path) -> np.ndarray:
    import pyarrow.parquet as parquet

    table = parquet.read_table(path, columns=["action"])
    return np.asarray(table["action"].to_pylist(), dtype=np.float32)


def _action_chunks(
    actions: np.ndarray,
    query_steps: Sequence[int],
    *,
    horizon: int = ACTION_HORIZON,
) -> np.ndarray:
    chunks = []
    for query in query_steps:
        indices = np.clip(
            np.arange(int(query), int(query) + int(horizon)),
            0,
            len(actions) - 1,
        )
        chunks.append(actions[indices])
    return np.stack(chunks)


def _fit_action_candidates(
    chunks_by_episode: Sequence[np.ndarray],
    train_positions: Sequence[int],
    *,
    candidates: int,
    seed: int = 7301,
) -> tuple[dict[str, np.ndarray], list[list[int]], dict[str, Any]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    train = np.concatenate(
        [chunks_by_episode[int(position)] for position in train_positions], axis=0
    )
    flattened = train.reshape(len(train), -1).astype(np.float64)
    mean = flattened.mean(axis=0)
    scale = flattened.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (flattened - mean) / scale
    cluster = KMeans(
        n_clusters=int(candidates),
        random_state=int(seed),
        n_init=50,
    ).fit(normalized)
    labels: list[list[int]] = []
    distances = []
    for chunks in chunks_by_episode:
        values = chunks.reshape(len(chunks), -1).astype(np.float64)
        transformed = (values - mean) / scale
        labels.append(cluster.predict(transformed).astype(int).tolist())
        distances.extend(cluster.transform(transformed).min(axis=1).tolist())
    receipt = {
        "kind": "KMeans over standard demonstrated action chunks",
        "fit_episodes": [int(value) for value in train_positions],
        "candidates": int(candidates),
        "action_horizon": ACTION_HORIZON,
        "silhouette_train": float(silhouette_score(normalized, cluster.labels_)),
        "cluster_counts_train": np.bincount(
            cluster.labels_, minlength=int(candidates)
        ).astype(int).tolist(),
        "mean_assignment_distance_all": float(np.mean(distances)),
        "task_state_labels_used": False,
    }
    arrays = {
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "centers": cluster.cluster_centers_.astype(np.float32),
    }
    return arrays, labels, receipt


def stage_index(args: argparse.Namespace) -> None:
    tasks = {}
    candidate_receipts = {}
    for task_id in args.tasks:
        spec = TASK_SPECS[task_id]
        records = load_episode_records(args.dataset_root, task_id)
        split = deterministic_episode_split(
            task_id,
            records,
            validation_episodes=VALIDATION_EPISODES,
            test_episodes=TEST_EPISODES,
        )
        chunks = [
            _action_chunks(
                _load_actions(
                    _episode_parquet(
                        args.dataset_root, task_id, record.episode_index
                    )
                ),
                record.query_steps,
            )
            for record in records
        ]
        candidate_arrays, labels, candidate_receipt = _fit_action_candidates(
            chunks,
            split["train"],
            candidates=spec.action_candidates,
        )
        candidate_path = (
            args.output / "action_candidates" / f"{task_id}.npz"
        )
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(candidate_path, **candidate_arrays)
        candidate_receipt["npz_sha256"] = sha256_file(candidate_path)
        candidate_receipts[task_id] = candidate_receipt

        rows = []
        leakage = []
        for position, record in enumerate(records):
            random_seed = int(
                stable_digest(
                    f"{PROTOCOL_VERSION}|random|{task_id}|{record.episode_index}"
                )[:16],
                16,
            )
            selections = {
                condition: select_memory_indices(
                    condition,
                    record,
                    random_seed=random_seed,
                )
                for condition in ALL_MEMORY_CONDITIONS
                if condition != "oracle_best_event"
            }
            selections["oracle_best_event"] = select_memory_indices(
                "oracle_best_event",
                record,
                random_seed=random_seed,
                oracle_event_position=0,
            )
            assert_matched_budget(selections)
            audit = recent_suffix_audit(record)
            leakage.append(audit)
            split_name = next(
                name for name, positions in split.items() if position in positions
            )
            rows.append(
                {
                    **asdict(record),
                    "keyframe_steps": list(record.keyframe_steps),
                    "query_steps": list(record.query_steps),
                    "position": position,
                    "split": split_name,
                    "action_candidate_sequence": labels[position],
                    "random_seed": random_seed,
                    "selections": {
                        name: values.astype(int).tolist()
                        for name, values in selections.items()
                    },
                    "encoding_frame_indices": frame_union_for_encoding(
                        record,
                        random_seed=random_seed,
                    ).astype(int).tolist(),
                    "recent_suffix_audit": audit,
                }
            )
        if not all(row["passed"] for row in leakage):
            raise RuntimeError(f"{task_id}: recent suffix contains event frames")
        tasks[task_id] = {
            "spec": asdict(spec),
            "split_positions": split,
            "split_episode_indices": {
                name: [records[position].episode_index for position in positions]
                for name, positions in split.items()
            },
            "episodes": rows,
            "leakage": {
                "passed": len(leakage),
                "total": len(leakage),
                "minimum_gap_frames": min(
                    int(row["gap_frames"]) for row in leakage
                ),
                "maximum_event_overlap": max(
                    int(row["event_overlap_frames"]) for row in leakage
                ),
            },
        }
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_revision": OFFICIAL_DATASET_REVISION,
        "tasks": tasks,
    }
    write_json(args.output / "episode_manifest.json", manifest)
    write_json(args.output / "action_candidate_receipt.json", candidate_receipts)
    print("indexed deterministic splits, queries, action candidates, and leakage")


def _load_dinov2(args: argparse.Namespace) -> Any:
    assert_gpu_contract(require_visible=True)
    import torch

    os.environ["TORCH_HOME"] = str(args.torch_home)
    model = torch.hub.load(
        str(args.dinov2),
        "dinov2_vits14",
        source="local",
        pretrained=True,
    )
    model = model.eval().cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _encode_rgb_batch(model: Any, frames: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.asarray(frames)).cuda().float()
    tensor = tensor.permute(0, 3, 1, 2) / 255.0
    tensor = functional.interpolate(
        tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=tensor.device
    ).reshape(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=tensor.device
    ).reshape(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    with torch.inference_mode():
        patches = model.forward_features(tensor)["x_norm_patchtokens"]
        side = int(round(float(patches.shape[1]) ** 0.5))
        if side * side != patches.shape[1]:
            raise AssertionError("DINO patch tokens do not form a square grid")
        grid = patches.reshape(len(frames), side, side, DINO_DIM).permute(
            0, 3, 1, 2
        )
        spatial = functional.adaptive_avg_pool2d(grid, (3, 3))
        spatial = spatial.flatten(2).transpose(1, 2)
        global_token = patches.mean(dim=1, keepdim=True)
        tokens = torch.cat([global_token, spatial], dim=1)
    if tuple(tokens.shape[1:]) != (DINO_SPATIAL_TOKENS, DINO_DIM):
        raise AssertionError(f"unexpected DINO token shape {tokens.shape}")
    return tokens.half().cpu().numpy()


def stage_encode(args: argparse.Namespace) -> None:
    assert_gpu_contract(require_visible=True)
    import pyarrow.parquet as parquet
    import torch

    if len(args.tasks) != 1:
        raise ValueError("encode stage requires exactly one --tasks entry")
    task_id = args.tasks[0]
    manifest = read_json(args.output / "episode_manifest.json")
    task = manifest["tasks"][task_id]
    model = _load_dinov2(args)
    receipts = []
    destination_root = args.output / "features" / task_id
    destination_root.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(task["episodes"], start=1):
        episode = int(row["episode_index"])
        destination = destination_root / f"episode_{episode:06d}.npz"
        if destination.exists() and not args.overwrite:
            receipts.append(
                {
                    "episode_index": episode,
                    "path": str(destination.relative_to(args.output)),
                    "sha256": sha256_file(destination),
                    "skipped": True,
                }
            )
            continue
        selected = np.asarray(row["encoding_frame_indices"], dtype=np.int64)
        view_features = []
        for camera in CAMERA_KEYS:
            decoded = _decode_selected_frames(
                _episode_video(args.dataset_root, task_id, episode, camera),
                selected,
            )
            frames = np.stack([decoded[int(index)] for index in selected])
            chunks = []
            for offset in range(0, len(frames), args.encode_batch_size):
                chunks.append(
                    _encode_rgb_batch(
                        model, frames[offset : offset + args.encode_batch_size]
                    )
                )
            view_features.append(np.concatenate(chunks, axis=0))
        features = np.stack(view_features, axis=1)
        table = parquet.read_table(
            _episode_parquet(args.dataset_root, task_id, episode),
            columns=["observation.state"],
        )
        states = np.asarray(
            table["observation.state"].to_pylist(), dtype=np.float32
        )
        current_state = states[min(int(row["query_steps"][0]), len(states) - 1)]
        np.savez_compressed(
            destination,
            frame_indices=selected,
            features=features,
            current_state=current_state,
        )
        receipts.append(
            {
                "episode_index": episode,
                "path": str(destination.relative_to(args.output)),
                "shape": list(features.shape),
                "sha256": sha256_file(destination),
                "skipped": False,
            }
        )
        print(
            f"[{task_id}] encoded {number}/{len(task['episodes'])} "
            f"frames={len(selected)}",
            flush=True,
        )
    receipt = {
        "task": task_id,
        "encoder": {
            "name": "DINOv2 ViT-S/14",
            "source": str(args.dinov2),
            "source_commit": git_value(args.dinov2, "rev-parse", "HEAD"),
            "weights": str(args.dino_weights),
            "weights_sha256": sha256_file(args.dino_weights),
            "frozen": True,
            "manual_crop": False,
            "resize": [224, 224],
            "tokens": "global mean plus 3x3 pooled full-frame patch grid",
            "views": list(CAMERA_KEYS),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "episodes": receipts,
    }
    write_json(
        args.output / "features" / f"{task_id}_receipt.json", receipt
    )
    print(f"encoded official multiview RGB for {task_id}")


def _torch_imports() -> tuple[Any, Any, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    return torch, nn, functional


class VisualDecisionHead:
    @staticmethod
    def build(*, query_count: int, candidates: int) -> Any:
        torch, nn, _ = _torch_imports()

        class _Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                hidden = 96
                self.input_norm = nn.LayerNorm(DINO_DIM)
                self.projection = nn.Linear(DINO_DIM, hidden)
                self.view_embedding = nn.Parameter(
                    torch.zeros(DINO_VIEWS, hidden)
                )
                self.spatial_embedding = nn.Parameter(
                    torch.zeros(DINO_SPATIAL_TOKENS, hidden)
                )
                self.frame_embedding = nn.Parameter(
                    torch.zeros(FULL_HISTORY_FRAMES + 1, hidden)
                )
                self.role_embedding = nn.Parameter(torch.zeros(2, hidden))
                self.query_embedding = nn.Parameter(
                    torch.zeros(int(query_count), hidden)
                )
                self.proprio = nn.Sequential(
                    nn.LayerNorm(14),
                    nn.Linear(14, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, hidden),
                )
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=4,
                    dim_feedforward=192,
                    dropout=0.10,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=1)
                self.cross_attention = nn.MultiheadAttention(
                    hidden, 4, dropout=0.10, batch_first=True
                )
                self.output = nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(0.10),
                    nn.Linear(hidden, int(candidates)),
                )
                nn.init.normal_(self.view_embedding, std=0.02)
                nn.init.normal_(self.spatial_embedding, std=0.02)
                nn.init.normal_(self.frame_embedding, std=0.02)
                nn.init.normal_(self.role_embedding, std=0.02)
                nn.init.normal_(self.query_embedding, std=0.02)

            def forward(
                self,
                features: Any,
                frame_valid: Any,
                proprio: Any,
            ) -> Any:
                batch, frames, views, spatial, dimension = features.shape
                if (views, spatial, dimension) != (
                    DINO_VIEWS,
                    DINO_SPATIAL_TOKENS,
                    DINO_DIM,
                ):
                    raise ValueError(f"unexpected feature shape {features.shape}")
                if frames > self.frame_embedding.shape[0]:
                    raise ValueError("too many history frames")
                value = self.projection(self.input_norm(features.float()))
                value = value + self.view_embedding.reshape(
                    1, 1, views, 1, -1
                )
                value = value + self.spatial_embedding.reshape(
                    1, 1, 1, spatial, -1
                )
                value = value + self.frame_embedding[:frames].reshape(
                    1, frames, 1, 1, -1
                )
                roles = torch.zeros(
                    frames, dtype=torch.long, device=value.device
                )
                roles[-1] = 1
                value = value + self.role_embedding[roles].reshape(
                    1, frames, 1, 1, -1
                )
                tokens = value.reshape(batch, frames * views * spatial, -1)
                token_valid = (
                    frame_valid[:, :, None, None]
                    .expand(batch, frames, views, spatial)
                    .reshape(batch, -1)
                )
                encoded = self.encoder(
                    tokens, src_key_padding_mask=~token_valid
                )
                queries = self.query_embedding.reshape(
                    1, int(query_count), -1
                ).expand(batch, -1, -1)
                queries = queries + self.proprio(proprio.float()).unsqueeze(1)
                pooled, _ = self.cross_attention(
                    queries,
                    encoded,
                    encoded,
                    key_padding_mask=~token_valid,
                    need_weights=False,
                )
                return self.output(pooled)

        return _Head()


def _set_seed(seed: int) -> None:
    torch, _, _ = _torch_imports()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _feature_bank(
    output: Path, task_id: str, episode: int
) -> dict[str, np.ndarray]:
    path = (
        output / "features" / task_id / f"episode_{episode:06d}.npz"
    )
    with np.load(path, allow_pickle=False) as data:
        indices = np.asarray(data["frame_indices"], dtype=np.int64)
        features = np.asarray(data["features"], dtype=np.float16)
        state = np.asarray(data["current_state"], dtype=np.float32)
    return {
        "indices": indices,
        "features": features,
        "state": state,
    }


def _lookup_features(
    bank: Mapping[str, np.ndarray], indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    encoded_indices = np.asarray(bank["indices"], dtype=np.int64)
    encoded = np.asarray(bank["features"], dtype=np.float16)
    index_map = {
        int(frame): position for position, frame in enumerate(encoded_indices)
    }
    output = np.zeros(
        (
            len(indices),
            DINO_VIEWS,
            DINO_SPATIAL_TOKENS,
            DINO_DIM,
        ),
        dtype=np.float16,
    )
    valid = np.zeros(len(indices), dtype=np.bool_)
    for position, frame in enumerate(indices):
        frame = int(frame)
        if frame < 0:
            continue
        if frame not in index_map:
            raise KeyError(f"frame {frame} absent from encoded bank")
        output[position] = encoded[index_map[frame]]
        valid[position] = True
    return output, valid


def _automatic_surprise_indices(
    bank: Mapping[str, np.ndarray],
    full_history_indices: Sequence[int],
    *,
    budget: int = DEFAULT_MEMORY_BUDGET,
) -> np.ndarray:
    features, valid = _lookup_features(bank, full_history_indices)
    if not bool(valid.all()):
        raise AssertionError("full history contains invalid encoded frame")
    global_tokens = features[:, :, 0].astype(np.float32).mean(axis=1)
    global_tokens /= np.linalg.norm(global_tokens, axis=1, keepdims=True) + 1e-8
    surprise = np.zeros(len(global_tokens), dtype=np.float32)
    surprise[1:] = np.linalg.norm(
        global_tokens[1:] - global_tokens[:-1], axis=1
    )
    # Provisional temporal groups prevent one high-motion interaction from
    # consuming every write. Group boundaries are uniform and independent of
    # evaluator keyframes, labels, or known event times.
    groups = np.array_split(np.arange(len(surprise)), int(budget))
    selected = [
        int(group[int(np.argmax(surprise[group]))])
        for group in groups
        if len(group)
    ]
    return np.asarray(
        sorted(int(full_history_indices[position]) for position in selected),
        dtype=np.int64,
    )


def _condition_indices(
    row: Mapping[str, Any],
    bank: Mapping[str, np.ndarray],
    condition: str,
) -> np.ndarray:
    if condition == "auto_surprise":
        return _automatic_surprise_indices(
            bank, row["selections"]["full_history"]
        )
    return np.asarray(row["selections"][condition], dtype=np.int64)


def _example(
    row: Mapping[str, Any],
    bank: Mapping[str, np.ndarray],
    condition: str,
    *,
    oracle_event_position: int | None = None,
) -> dict[str, np.ndarray]:
    if condition == "oracle_best_event" and oracle_event_position is not None:
        event = int(row["keyframe_steps"][int(oracle_event_position)])
        memory_indices = np.full(
            DEFAULT_MEMORY_BUDGET, event, dtype=np.int64
        )
    else:
        memory_indices = _condition_indices(row, bank, condition)
    memory, memory_valid = _lookup_features(bank, memory_indices)
    current, current_valid = _lookup_features(
        bank, [int(row["query_steps"][0])]
    )
    return {
        "features": np.concatenate([memory, current], axis=0),
        "valid": np.concatenate([memory_valid, current_valid], axis=0),
        "proprio": np.asarray(bank["state"], dtype=np.float32),
        "labels": np.asarray(
            row["action_candidate_sequence"], dtype=np.int64
        ),
        "memory_indices": np.asarray(memory_indices, dtype=np.int64),
    }


def _collate(examples: Sequence[Mapping[str, np.ndarray]]) -> dict[str, Any]:
    torch, _, _ = _torch_imports()
    maximum = FULL_HISTORY_FRAMES + 1
    features = np.zeros(
        (
            len(examples),
            maximum,
            DINO_VIEWS,
            DINO_SPATIAL_TOKENS,
            DINO_DIM,
        ),
        dtype=np.float16,
    )
    valid = np.zeros((len(examples), maximum), dtype=np.bool_)
    for row, example in enumerate(examples):
        count = len(example["features"])
        if count > maximum:
            raise ValueError("example exceeds registered full-history budget")
        # Keep memory slots at the front and the current observation in the
        # final fixed slot. This makes frame/role embeddings identical across
        # every matched condition and the full-history upper bound.
        memory_count = count - 1
        features[row, :memory_count] = example["features"][:-1]
        valid[row, :memory_count] = example["valid"][:-1]
        features[row, -1] = example["features"][-1]
        valid[row, -1] = example["valid"][-1]
    return {
        "features": torch.from_numpy(features).cuda(),
        "valid": torch.from_numpy(valid).cuda(),
        "proprio": torch.from_numpy(
            np.stack([example["proprio"] for example in examples])
        ).cuda(),
        "labels": torch.from_numpy(
            np.stack([example["labels"] for example in examples])
        ).cuda(),
    }


def _evaluate_examples(
    model: Any,
    examples: Sequence[Mapping[str, np.ndarray]],
    *,
    batch_size: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    torch, _, functional = _torch_imports()
    model.eval()
    rows = []
    losses = []
    with torch.inference_mode():
        for offset in range(0, len(examples), int(batch_size)):
            chunk = examples[offset : offset + int(batch_size)]
            batch = _collate(chunk)
            logits = model(
                batch["features"], batch["valid"], batch["proprio"]
            )
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch["labels"].reshape(-1),
                reduction="none",
            ).reshape(logits.shape[:2])
            probabilities = functional.softmax(logits, dim=-1)
            predictions = probabilities.argmax(dim=-1)
            ranks = (
                torch.argsort(
                    probabilities, dim=-1, descending=True
                )
                == batch["labels"].unsqueeze(-1)
            ).float().argmax(dim=-1) + 1
            for position in range(len(chunk)):
                correct = predictions[position] == batch["labels"][position]
                rows.append(
                    {
                        "probabilities": probabilities[position]
                        .cpu()
                        .numpy()
                        .tolist(),
                        "prediction": predictions[position]
                        .cpu()
                        .numpy()
                        .astype(int)
                        .tolist(),
                        "target": batch["labels"][position]
                        .cpu()
                        .numpy()
                        .astype(int)
                        .tolist(),
                        "per_query_correct": correct.cpu().numpy().tolist(),
                        "per_query_accuracy": float(
                            correct.float().mean().item()
                        ),
                        "exact_success": bool(correct.all().item()),
                        "mean_reciprocal_rank": float(
                            (1.0 / ranks[position].float()).mean().item()
                        ),
                        "cross_entropy": float(loss[position].mean().item()),
                    }
                )
                losses.append(float(loss[position].mean().item()))
    metrics = {
        "loss": float(np.mean(losses)),
        "exact_success": float(
            np.mean([row["exact_success"] for row in rows])
        ),
        "per_query_accuracy": float(
            np.mean([row["per_query_accuracy"] for row in rows])
        ),
        "mean_reciprocal_rank": float(
            np.mean([row["mean_reciprocal_rank"] for row in rows])
        ),
    }
    return metrics, rows


def _train_head(
    *,
    task_id: str,
    seed: int,
    train_examples: Sequence[Mapping[str, np.ndarray]],
    validation_examples: Sequence[Mapping[str, np.ndarray]],
    query_count: int,
    candidates: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint: Path,
) -> tuple[Any, dict[str, Any]]:
    torch, _, functional = _torch_imports()
    _set_seed(seed)
    model = VisualDecisionHead.build(
        query_count=query_count, candidates=candidates
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-3
    )
    rng = np.random.default_rng(int(seed))
    best = {
        "exact_success": -1.0,
        "per_query_accuracy": -1.0,
        "loss": float("inf"),
        "epoch": -1,
        "state": None,
    }
    history = []
    patience = 50
    last_improvement = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = rng.permutation(len(train_examples))
        train_losses = []
        for offset in range(0, len(order), int(batch_size)):
            examples = [
                train_examples[int(index)]
                for index in order[offset : offset + int(batch_size)]
            ]
            batch = _collate(examples)
            logits = model(
                batch["features"], batch["valid"], batch["proprio"]
            )
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch["labels"].reshape(-1),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().item()))
        validation, _ = _evaluate_examples(
            model, validation_examples, batch_size=batch_size
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation": validation,
        }
        history.append(row)
        improved = (
            validation["exact_success"] > best["exact_success"]
            or (
                validation["exact_success"] == best["exact_success"]
                and validation["per_query_accuracy"]
                > best["per_query_accuracy"]
            )
            or (
                validation["exact_success"] == best["exact_success"]
                and validation["per_query_accuracy"]
                == best["per_query_accuracy"]
                and validation["loss"] < best["loss"]
            )
        )
        if improved:
            best = {
                "exact_success": validation["exact_success"],
                "per_query_accuracy": validation["per_query_accuracy"],
                "loss": validation["loss"],
                "epoch": epoch,
                "state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
            last_improvement = epoch
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"[{task_id} seed={seed}] epoch={epoch} "
                f"train={row['train_loss']:.4f} "
                f"val_exact={validation['exact_success']:.3f} "
                f"val_query={validation['per_query_accuracy']:.3f}",
                flush=True,
            )
        if epoch - last_improvement >= patience:
            break
    state = best.pop("state")
    if state is None:
        raise RuntimeError("training failed to produce a checkpoint")
    model.load_state_dict(state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state,
            "task": task_id,
            "seed": seed,
            "query_count": query_count,
            "candidates": candidates,
            "best": best,
            "protocol_version": PROTOCOL_VERSION,
        },
        checkpoint,
    )
    return model.eval(), {
        "task": task_id,
        "seed": seed,
        "best": best,
        "epochs_ran": len(history),
        "history": history,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }


def _rows_and_banks(
    output: Path, task_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    manifest = read_json(output / "episode_manifest.json")
    task = manifest["tasks"][task_id]
    rows = task["episodes"]
    banks = {
        int(row["episode_index"]): _feature_bank(
            output, task_id, int(row["episode_index"])
        )
        for row in rows
    }
    return task, rows, banks


def _split_examples(
    rows: Sequence[Mapping[str, Any]],
    banks: Mapping[int, Mapping[str, np.ndarray]],
    split_name: str,
    modes: Sequence[str],
) -> list[dict[str, np.ndarray]]:
    output = []
    for row in rows:
        if row["split"] != split_name:
            continue
        bank = banks[int(row["episode_index"])]
        for mode in modes:
            output.append(_example(row, bank, mode))
    return output


def _oracle_best_rows(
    model: Any,
    row: Mapping[str, Any],
    bank: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> tuple[dict[str, Any], int]:
    examples = [
        _example(
            row,
            bank,
            "oracle_best_event",
            oracle_event_position=position,
        )
        for position in range(len(row["keyframe_steps"]))
    ]
    _, predictions = _evaluate_examples(
        model, examples, batch_size=batch_size
    )
    scored = []
    for position, prediction in enumerate(predictions):
        target = np.asarray(prediction["target"], dtype=np.int64)
        probabilities = np.asarray(
            prediction["probabilities"], dtype=np.float64
        )
        true_probability = float(
            probabilities[np.arange(len(target)), target].sum()
        )
        scored.append(
            (
                int(prediction["exact_success"]),
                prediction["per_query_accuracy"],
                true_probability,
                -position,
            )
        )
    best = int(max(range(len(scored)), key=lambda position: scored[position]))
    return predictions[best], best


def stage_admission(args: argparse.Namespace) -> None:
    assert_gpu_contract(require_visible=True)
    import torch

    if len(args.tasks) != 1:
        raise ValueError("admission stage requires exactly one --tasks entry")
    task_id = args.tasks[0]
    spec = TASK_SPECS[task_id]
    task, rows, banks = _rows_and_banks(args.output, task_id)
    train = _split_examples(rows, banks, "train", TRAINING_MODES)
    validation = _split_examples(
        rows, banks, "validation", ("full_history",)
    )
    test_rows = [row for row in rows if row["split"] == "test"]
    results = []
    training_receipts = []
    for seed in MODEL_SEEDS:
        model, training = _train_head(
            task_id=task_id,
            seed=seed,
            train_examples=train,
            validation_examples=validation,
            query_count=spec.query_count,
            candidates=spec.action_candidates,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint=(
                args.output / "models" / task_id / f"decision_head_s{seed}.pt"
            ),
        )
        probe_train = _split_examples(
            rows, banks, "train", ("recent_only",)
        )
        probe_validation = _split_examples(
            rows, banks, "validation", ("recent_only",)
        )
        probe, probe_training = _train_head(
            task_id=f"{task_id}:recent_probe",
            seed=seed + 10_000,
            train_examples=probe_train,
            validation_examples=probe_validation,
            query_count=spec.query_count,
            candidates=spec.action_candidates,
            epochs=args.probe_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint=(
                args.output
                / "models"
                / task_id
                / f"recent_probe_s{seed}.pt"
            ),
        )
        training_receipts.append(
            {
                "seed": seed,
                "main": training,
                "recent_probe": probe_training,
            }
        )
        for row in test_rows:
            bank = banks[int(row["episode_index"])]
            episode_result = {
                "task": task_id,
                "model_seed": seed,
                "episode_index": int(row["episode_index"]),
                "episode_seed": int(row["episode_seed"]),
                "conditions": {},
            }
            for condition in ALL_MEMORY_CONDITIONS:
                if condition == "oracle_best_event":
                    prediction, event_position = _oracle_best_rows(
                        model, row, bank, batch_size=args.batch_size
                    )
                    prediction["oracle_event_position"] = event_position
                else:
                    _, predictions = _evaluate_examples(
                        model,
                        [_example(row, bank, condition)],
                        batch_size=args.batch_size,
                    )
                    prediction = predictions[0]
                episode_result["conditions"][condition] = prediction
            _, probe_prediction = _evaluate_examples(
                probe,
                [_example(row, bank, "recent_only")],
                batch_size=args.batch_size,
            )
            episode_result["recent_suffix_probe"] = probe_prediction[0]
            results.append(episode_result)
        del model, probe
        torch.cuda.empty_cache()
    receipt = {
        "task": task_id,
        "model_seeds": list(MODEL_SEEDS),
        "training_modes": list(TRAINING_MODES),
        "training_event_labels_or_times_used": False,
        "action_supervision": "official 14D demonstrated action chunks only",
        "oracle_metadata_policy_access": False,
        "training": training_receipts,
        "rows": results,
    }
    write_json(args.output / f"predictions_{task_id}.json", receipt)
    print(f"completed three-seed oracle admission for {task_id}")


def _condition_array(
    rows: Sequence[Mapping[str, Any]],
    task_id: str,
    condition: str,
    metric: str,
) -> tuple[np.ndarray, list[int]]:
    task_rows = [row for row in rows if row["task"] == task_id]
    episodes = sorted({int(row["episode_index"]) for row in task_rows})
    episode_index = {episode: index for index, episode in enumerate(episodes)}
    seed_index = {seed: index for index, seed in enumerate(MODEL_SEEDS)}
    values = np.zeros((len(MODEL_SEEDS), len(episodes)), dtype=np.float64)
    for row in task_rows:
        value = row["conditions"][condition][metric]
        values[
            seed_index[int(row["model_seed"])],
            episode_index[int(row["episode_index"])],
        ] = float(value)
    return values, episodes


def _probe_array(
    rows: Sequence[Mapping[str, Any]], task_id: str
) -> tuple[np.ndarray, list[int]]:
    task_rows = [row for row in rows if row["task"] == task_id]
    episodes = sorted({int(row["episode_index"]) for row in task_rows})
    episode_index = {episode: index for index, episode in enumerate(episodes)}
    seed_index = {seed: index for index, seed in enumerate(MODEL_SEEDS)}
    values = np.zeros((len(MODEL_SEEDS), len(episodes)), dtype=np.float64)
    for row in task_rows:
        values[
            seed_index[int(row["model_seed"])],
            episode_index[int(row["episode_index"])],
        ] = float(row["recent_suffix_probe"]["per_query_accuracy"])
    return values, episodes


def _mean_ci(values: np.ndarray) -> dict[str, float]:
    mean, low, high = bootstrap_mean_ci(values)
    return {"mean": mean, "ci_low": low, "ci_high": high}


def _paired_ci(
    treatment: np.ndarray, control: np.ndarray
) -> dict[str, float]:
    mean, low, high = paired_bootstrap_ci(treatment, control)
    return {"mean": mean, "ci_low": low, "ci_high": high}


def _make_storyboard(
    output: Path,
    dataset_root: Path,
    manifest: Mapping[str, Any],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task_id = "pick_the_unhidden_block"
    row = manifest["tasks"][task_id]["episodes"][0]
    keyframes = [int(value) for value in row["keyframe_steps"]]
    recent = [int(value) for value in row["selections"]["recent_only"]]
    recall = int(row["query_steps"][0])
    indices = [
        keyframes[0],
        keyframes[1],
        keyframes[2],
        (keyframes[-1] + recent[0]) // 2,
        recall,
        min(int(row["length"]) - 1, recall + 35),
    ]
    frames = _decode_selected_frames(
        _episode_video(
            dataset_root,
            task_id,
            int(row["episode_index"]),
            CAMERA_KEYS[0],
        ),
        indices,
    )
    labels = [
        "Transient event 1",
        "Transient event 2",
        "Transient event 3",
        "Post-event gap",
        "Delayed recall query",
        "Demonstrated action",
    ]
    figure, axes = plt.subplots(1, len(indices), figsize=(18, 3.6))
    for axis, index, label in zip(axes, indices, labels):
        axis.imshow(frames[index])
        axis.set_title(f"{label}\nframe {index}", fontsize=9)
        axis.axis("off")
    figure.suptitle(
        "Official RoboTwin-MeM rollout: pick_the_unhidden_block",
        fontsize=12,
    )
    figure.patch.set_facecolor("white")
    figure.tight_layout()
    path = ROOT / "docs/assets/robotwin_mem_real_rollout_storyboard.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    return path


def _make_admission_plots(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = ROOT / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    conditions = [
        "no_memory",
        "recent_only",
        "random_event",
        "oracle_best_event",
        "oracle_event_set",
        "full_history",
    ]
    labels = [
        "No memory",
        "Recent only",
        "Random event",
        "Oracle best",
        "Oracle event set",
        "Full history*",
    ]
    figure, axes = plt.subplots(
        1, len(TASKS), figsize=(15, 4.6), sharey=True
    )
    colors = ["#777777", "#4c78a8", "#9c755f", "#f2cf5b", "#59a14f", "#76b7b2"]
    for axis, task_id in zip(axes, TASKS):
        task = summary["tasks"][task_id]
        means = [
            task["conditions"][condition]["exact_sequence"]["mean"]
            for condition in conditions
        ]
        lows = [
            task["conditions"][condition]["exact_sequence"]["ci_low"]
            for condition in conditions
        ]
        highs = [
            task["conditions"][condition]["exact_sequence"]["ci_high"]
            for condition in conditions
        ]
        x = np.arange(len(conditions))
        axis.bar(x, means, color=colors)
        axis.errorbar(
            x,
            means,
            yerr=[
                np.asarray(means) - np.asarray(lows),
                np.asarray(highs) - np.asarray(means),
            ],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        axis.set_title(task_id.replace("_", " "))
        axis.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.2)
        axis.set_facecolor("white")
    axes[0].set_ylabel("Exact action-sequence accuracy")
    axes[0].set_ylim(0, 1.05)
    figure.suptitle("RoboTwin-MeM oracle admission ladder")
    figure.patch.set_facecolor("white")
    figure.tight_layout()
    ladder = assets / "robotwin_mem_admission_oracle_ladder.png"
    figure.savefig(ladder, dpi=220, facecolor="white")
    figure.savefig(ladder.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)

    vlm_ladder = None
    vlm_tasks = summary.get("confirmatory_vlm", {}).get("tasks", {})
    if vlm_tasks:
        names = list(vlm_tasks)
        figure, axes = plt.subplots(
            1, len(names), figsize=(5.2 * len(names), 4.6), sharey=True
        )
        axes = np.atleast_1d(axes)
        for axis, task_id in zip(axes, names):
            task = vlm_tasks[task_id]
            means = [
                task["conditions"][condition]["exact_sequence"]["mean"]
                for condition in conditions
            ]
            lows = [
                task["conditions"][condition]["exact_sequence"]["ci_low"]
                for condition in conditions
            ]
            highs = [
                task["conditions"][condition]["exact_sequence"]["ci_high"]
                for condition in conditions
            ]
            x = np.arange(len(conditions))
            axis.bar(x, means, color=colors)
            axis.errorbar(
                x,
                means,
                yerr=[
                    np.asarray(means) - np.asarray(lows),
                    np.asarray(highs) - np.asarray(means),
                ],
                fmt="none",
                ecolor="black",
                capsize=3,
            )
            axis.set_title(task_id.replace("_", " "))
            axis.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
            axis.grid(axis="y", alpha=0.2)
            axis.set_facecolor("white")
        axes[0].set_ylabel("Exact action-sequence accuracy")
        axes[0].set_ylim(0, 1.05)
        figure.suptitle("RoboTwin-MeM preregistered frozen-VLM control")
        figure.patch.set_facecolor("white")
        figure.tight_layout()
        vlm_ladder = assets / "robotwin_mem_vlm_oracle_ladder.png"
        figure.savefig(vlm_ladder, dpi=220, facecolor="white")
        figure.savefig(vlm_ladder.with_suffix(".pdf"), facecolor="white")
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    task_names = list(TASKS)
    gaps = [
        manifest["tasks"][task]["leakage"]["minimum_gap_frames"]
        for task in task_names
    ]
    axis.bar(
        np.arange(len(task_names)),
        gaps,
        color=["#4c78a8", "#59a14f", "#9c755f"],
    )
    axis.set_xticks(
        np.arange(len(task_names)),
        [name.replace("_", " ") for name in task_names],
        rotation=15,
        ha="right",
    )
    axis.set_ylabel("Minimum frames from last event to recent suffix")
    axis.set_title("Recent-suffix leakage audit (zero event overlap)")
    axis.grid(axis="y", alpha=0.2)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    figure.tight_layout()
    leakage = assets / "robotwin_mem_recent_suffix_leakage.png"
    figure.savefig(leakage, dpi=220, facecolor="white")
    figure.savefig(leakage.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    result = {
        "oracle_ladder": str(ladder.relative_to(ROOT)),
        "recent_suffix_leakage": str(leakage.relative_to(ROOT)),
    }
    if vlm_ladder is not None:
        result["vlm_oracle_ladder"] = str(vlm_ladder.relative_to(ROOT))
    return result


def _write_report(summary: Mapping[str, Any]) -> None:
    suitable = bool(summary["benchmark_suitable"])
    lines = [
        "# RoboTwin-MeM CEM Evaluation Report",
        "",
        f"**Benchmark admission: {'PASS' if suitable else 'FAIL'}. "
        f"Learned CEM: {'authorized but not yet run' if suitable else 'not reached'}.**",
        "",
        "## Scope and official source",
        "",
        "- Paper: EventVLA, arXiv `2606.20092` (v1).",
        "- Official repository: `InternRobotics/EventVLA`, commit "
        f"`{OFFICIAL_COMMIT}`.",
        "- Official dataset: `ganlinyang/RoboTwin-MeM`, revision "
        f"`{OFFICIAL_DATASET_REVISION}`.",
        "- Licenses: simulator/code MIT; dataset Apache-2.0.",
        "- Tasks: `pick_the_unhidden_block`, `pick_objects_in_order`, and "
        "`cover_blocks_hard` under official `demo_clean`.",
        "",
        "## Setup status",
        "",
        "- The isolated `.venv-robotwin-mem` trajectory runtime passed "
        "deterministic LeRobot 2.1 loading and AV1 decoding.",
        "- Policy inputs are the three official 640×480 RGB views, 14D "
        "proprioception, and generic task instruction. Frozen DINOv2 uses "
        "full frames without crops or saliency.",
        "- Simulator execution is unavailable in the published checkout: "
        "`RoboTwin-Mem/task_config/`, `assets/_download.py`, and "
        "`data/_download.py` are referenced but absent. No proxy simulator "
        "was substituted; all metrics below are official action ranking.",
        "- The release exposes only a 50-episode `train: 0:50` split. This "
        "evaluation preregisters a deterministic 30/10/10 train/validation/test "
        "split by official episode index and seed.",
        "",
        "## Oracle admission",
        "",
    ]
    for task_id in TASKS:
        task = summary["tasks"][task_id]
        lines.append(
            f"### `{task_id}` — "
            f"{'PASS' if task['admission_gate']['passed'] else 'FAIL'}"
        )
        lines.append("")
        for condition in (
            "no_memory",
            "recent_only",
            "random_event",
            "oracle_best_event",
            "oracle_event_set",
            "full_history",
        ):
            metric = task["conditions"][condition]["exact_sequence"]
            lines.append(
                f"- `{condition}` exact sequence: "
                f"{100 * metric['mean']:.1f}% "
                f"(95% CI {100 * metric['ci_low']:.1f}–"
                f"{100 * metric['ci_high']:.1f}%)."
            )
        gain = task["contrasts"]["oracle_event_set_minus_recent_only"]
        probe = task["recent_suffix_probe"]
        lines.extend(
            [
                f"- Paired oracle-set minus recent: "
                f"{100 * gain['mean']:.1f} pp "
                f"(95% CI {100 * gain['ci_low']:.1f}–"
                f"{100 * gain['ci_high']:.1f} pp).",
                f"- Recent-only trained probe, per-query: "
                f"{100 * probe['mean']:.1f}% "
                f"(95% CI {100 * probe['ci_low']:.1f}–"
                f"{100 * probe['ci_high']:.1f}%).",
                f"- Minimum event-to-recent gap: "
                f"{task['recent_suffix_audit']['minimum_gap_frames']} frames; "
                "event overlap 0.",
                "",
            ]
        )
    vlm_tasks = summary.get("confirmatory_vlm", {}).get("tasks", {})
    if vlm_tasks:
        lines.extend(
            [
                "## Preregistered strong-control check",
                "",
                "After the frozen-DINO head failed, the final frozen "
                "Qwen3-VL-4B control was preregistered on untouched "
                "confirmatory episodes. One prompt-development episode per "
                "task was excluded before this run; the remaining nine "
                "episodes and controller source hash are frozen in "
                "`vlm_control_protocol_registration.json`.",
                "",
            ]
        )
        for task_id, task in vlm_tasks.items():
            recent = task["conditions"]["recent_only"]["exact_sequence"]
            oracle = task["conditions"]["oracle_event_set"]["exact_sequence"]
            no_memory = task["conditions"]["no_memory"]["exact_sequence"]
            full = task["conditions"]["full_history"]["exact_sequence"]
            gain = task["contrasts"]["oracle_event_set_minus_recent_only"]
            lines.extend(
                [
                    f"- `{task_id}`: "
                    f"{'PASS' if task['admission_gate']['passed'] else 'FAIL'}; "
                    f"no memory {100 * no_memory['mean']:.1f}%, "
                    f"recent {100 * recent['mean']:.1f}%, "
                    f"oracle set {100 * oracle['mean']:.1f}%, "
                    f"full history {100 * full['mean']:.1f}%; "
                    f"paired oracle−recent {100 * gain['mean']:.1f} pp "
                    f"(95% CI {100 * gain['ci_low']:.1f}–"
                    f"{100 * gain['ci_high']:.1f} pp)."
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            (
                "At least two tasks pass the fail-closed gate, so a focused "
                "learned CEM run is authorized. Admission controls remain frozen."
                if suitable
                else
                "Fewer than two tasks pass the fail-closed gate. Per protocol, "
                "learned CEM is not run and no benchmark-suitability claim is made."
            ),
            "",
            "Executed success is **not available**; the exact next step is for "
            "the official authors to publish the referenced `task_config/` and "
            "asset/data downloader implementations (or a complete simulator "
            "artifact), after which the same seeds and controller can be replayed.",
            "",
            "## Reproduction",
            "",
            "Run from the repository root with the isolated environment:",
            "",
            "```bash",
            ".venv-robotwin-mem/bin/python scripts/run_robotwin_mem_admission.py --stage register",
            "CUDA_VISIBLE_DEVICES=0 .venv-robotwin-mem/bin/python scripts/run_robotwin_mem_admission.py --stage smoke --tasks pick_the_unhidden_block",
            ".venv-robotwin-mem/bin/python scripts/run_robotwin_mem_admission.py --stage index",
            "CUDA_VISIBLE_DEVICES=0 .venv-robotwin-mem/bin/python scripts/run_robotwin_mem_admission.py --stage encode --tasks pick_the_unhidden_block",
            "CUDA_VISIBLE_DEVICES=1 .venv-robotwin-mem/bin/python scripts/run_robotwin_mem_admission.py --stage encode --tasks pick_objects_in_order",
            "CUDA_VISIBLE_DEVICES=2 .venv-robotwin-mem/bin/python scripts/run_robotwin_mem_admission.py --stage encode --tasks cover_blocks_hard",
            "CUDA_VISIBLE_DEVICES=0 .venv-robotwin-mem/bin/python scripts/run_robotwin_mem_admission.py --stage admission --tasks pick_the_unhidden_block",
            "CUDA_VISIBLE_DEVICES=1 .venv-robotwin-mem/bin/python scripts/run_robotwin_mem_admission.py --stage admission --tasks pick_objects_in_order",
            "CUDA_VISIBLE_DEVICES=2 .venv-robotwin-mem/bin/python scripts/run_robotwin_mem_admission.py --stage admission --tasks cover_blocks_hard",
            ".venv-robotwin-mem/bin/python scripts/run_robotwin_mem_admission.py --stage aggregate",
            ".venv-robotwin-mem/bin/python -m pytest -q scripts/test_robotwin_mem_admission.py",
            "```",
            "",
            "The secondary VLM commands and untouched episode lists are frozen "
            "in `outputs/robotwin_mem_admission_v1/"
            "vlm_control_protocol_registration.json`.",
            "",
            "## Figures",
            "",
            "![Oracle ladder](assets/robotwin_mem_admission_oracle_ladder.png)",
            "",
            *(
                [
                    "![Frozen-VLM oracle ladder]"
                    "(assets/robotwin_mem_vlm_oracle_ladder.png)",
                    "",
                ]
                if vlm_tasks
                else []
            ),
            "![Recent suffix leakage](assets/robotwin_mem_recent_suffix_leakage.png)",
            "",
            "![Real rollout storyboard](assets/robotwin_mem_real_rollout_storyboard.png)",
            "",
            "Machine-readable receipts and per-episode decisions are under "
            "`outputs/robotwin_mem_admission_v1/`.",
            "",
        ]
    )
    (ROOT / "docs/ROBOTWIN_MEM_CEM_REPORT.md").write_text("\n".join(lines))


def stage_aggregate(args: argparse.Namespace) -> None:
    manifest = read_json(args.output / "episode_manifest.json")
    prediction_rows = []
    tasks = {}
    for task_id in args.tasks:
        prediction = read_json(args.output / f"predictions_{task_id}.json")
        prediction_rows.extend(prediction["rows"])
    for task_id in args.tasks:
        conditions = {}
        condition_values = {}
        condition_episodes = None
        for condition in ALL_MEMORY_CONDITIONS:
            exact, episodes = _condition_array(
                prediction_rows, task_id, condition, "exact_success"
            )
            query, query_episodes = _condition_array(
                prediction_rows, task_id, condition, "per_query_accuracy"
            )
            reciprocal, reciprocal_episodes = _condition_array(
                prediction_rows, task_id, condition, "mean_reciprocal_rank"
            )
            if episodes != query_episodes or episodes != reciprocal_episodes:
                raise RuntimeError("condition episode order mismatch")
            condition_values[condition] = exact
            condition_episodes = episodes
            conditions[condition] = {
                "exact_sequence": _mean_ci(exact),
                "per_query_accuracy": _mean_ci(query),
                "mean_reciprocal_rank": _mean_ci(reciprocal),
                "per_model_seed_exact": {
                    str(seed): float(exact[index].mean())
                    for index, seed in enumerate(MODEL_SEEDS)
                },
            }
        probe, probe_episodes = _probe_array(prediction_rows, task_id)
        if probe_episodes != condition_episodes:
            raise RuntimeError("probe episode order mismatch")
        gate = decide_admission_gate(
            recent_success=condition_values["recent_only"],
            oracle_success=condition_values["oracle_event_set"],
            no_memory_success=condition_values["no_memory"],
            recent_probe_accuracy=probe,
            candidate_count=TASK_SPECS[task_id].action_candidates,
        )
        tasks[task_id] = {
            "conditions": conditions,
            "contrasts": {
                "oracle_best_event_minus_recent_only": _paired_ci(
                    condition_values["oracle_best_event"],
                    condition_values["recent_only"],
                ),
                "oracle_event_set_minus_recent_only": _paired_ci(
                    condition_values["oracle_event_set"],
                    condition_values["recent_only"],
                ),
                "full_history_minus_recent_only": _paired_ci(
                    condition_values["full_history"],
                    condition_values["recent_only"],
                ),
            },
            "recent_suffix_probe": _mean_ci(probe),
            "recent_suffix_audit": manifest["tasks"][task_id]["leakage"],
            "test_episodes": condition_episodes,
            "admission_gate": gate.as_dict(),
        }
    admitted = [
        task_id
        for task_id, task in tasks.items()
        if task["admission_gate"]["passed"]
    ]
    vlm_tasks = {}
    vlm_admitted = []
    for task_id in args.tasks:
        path = args.output / f"predictions_vlm_{task_id}.json"
        if not path.exists():
            continue
        prediction = read_json(path)
        rows = prediction["rows"]
        observed_seeds = sorted({int(row["model_seed"]) for row in rows})
        observed_episodes = sorted(
            {int(row["episode_index"]) for row in rows}
        )
        if observed_seeds != sorted(MODEL_SEEDS) or len(observed_episodes) < 2:
            continue
        if len(rows) != len(MODEL_SEEDS) * len(observed_episodes):
            continue
        condition_values = {}
        conditions = {}
        condition_episodes = None
        for condition in ALL_MEMORY_CONDITIONS:
            exact, episodes = _condition_array(
                rows, task_id, condition, "exact_success"
            )
            query, query_episodes = _condition_array(
                rows, task_id, condition, "per_query_accuracy"
            )
            reciprocal, reciprocal_episodes = _condition_array(
                rows, task_id, condition, "mean_reciprocal_rank"
            )
            if episodes != query_episodes or episodes != reciprocal_episodes:
                raise RuntimeError("VLM condition episode order mismatch")
            condition_values[condition] = exact
            condition_episodes = episodes
            conditions[condition] = {
                "exact_sequence": _mean_ci(exact),
                "per_query_accuracy": _mean_ci(query),
                "mean_reciprocal_rank": _mean_ci(reciprocal),
                "per_model_seed_exact": {
                    str(seed): float(exact[index].mean())
                    for index, seed in enumerate(MODEL_SEEDS)
                },
            }
        probe, probe_episodes = _probe_array(rows, task_id)
        if probe_episodes != condition_episodes:
            raise RuntimeError("VLM probe episode order mismatch")
        gate = decide_admission_gate(
            recent_success=condition_values["recent_only"],
            oracle_success=condition_values["oracle_event_set"],
            no_memory_success=condition_values["no_memory"],
            recent_probe_accuracy=probe,
            candidate_count=TASK_SPECS[task_id].action_candidates,
        )
        vlm_tasks[task_id] = {
            "conditions": conditions,
            "contrasts": {
                "oracle_best_event_minus_recent_only": _paired_ci(
                    condition_values["oracle_best_event"],
                    condition_values["recent_only"],
                ),
                "oracle_event_set_minus_recent_only": _paired_ci(
                    condition_values["oracle_event_set"],
                    condition_values["recent_only"],
                ),
                "full_history_minus_recent_only": _paired_ci(
                    condition_values["full_history"],
                    condition_values["recent_only"],
                ),
            },
            "recent_suffix_probe": _mean_ci(probe),
            "recent_suffix_audit": manifest["tasks"][task_id]["leakage"],
            "test_episodes": condition_episodes,
            "admission_gate": gate.as_dict(),
            "controller": prediction["controller"],
        }
        if gate.passed:
            vlm_admitted.append(task_id)
    if len(admitted) >= 2:
        final_admitted = admitted
        admission_basis = "registered frozen-DINO action head"
    else:
        final_admitted = vlm_admitted
        admission_basis = (
            "preregistered frozen-Qwen confirmatory control"
            if vlm_tasks
            else "registered frozen-DINO action head"
        )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_mode": "official dataset action ranking",
        "executed_success_available": False,
        "tasks": tasks,
        "primary_dino_admitted_tasks": admitted,
        "confirmatory_vlm": {
            "tasks": vlm_tasks,
            "admitted_tasks": vlm_admitted,
            "protocol_registration": (
                "outputs/robotwin_mem_admission_v1/"
                "vlm_control_protocol_registration.json"
            ),
        },
        "admission_basis": admission_basis,
        "admitted_tasks": final_admitted,
        "benchmark_suitable": len(final_admitted) >= 2,
        "cem_reached": False,
        "cem_reason": (
            "admission passed; learned CEM stage not implemented in this run"
            if len(final_admitted) >= 2
            else "fewer than two tasks passed mandatory oracle admission"
        ),
    }
    write_json(args.output / "admission_summary.json", summary)
    write_json(
        args.output / "gate_decisions.json",
        {
            "primary_dino": {
                task_id: task["admission_gate"]
                for task_id, task in tasks.items()
            },
            "confirmatory_vlm": {
                task_id: task["admission_gate"]
                for task_id, task in vlm_tasks.items()
            },
            "benchmark_suitable": summary["benchmark_suitable"],
            "cem_reached": False,
        },
    )
    storyboard = _make_storyboard(
        args.output, args.dataset_root, manifest
    )
    figures = _make_admission_plots(summary, manifest)
    figures["storyboard"] = str(storyboard.relative_to(ROOT))
    report = {
        **summary,
        "source_receipt": read_json(args.output / "source_receipt.json"),
        "runtime_receipt": read_json(args.output / "runtime_receipt.json"),
        "protocol_registration": read_json(
            args.output / "protocol_registration.json"
        ),
        "smoke_receipt": read_json(args.output / "smoke_receipt.json"),
        "action_candidate_receipt": read_json(
            args.output / "action_candidate_receipt.json"
        ),
        "dataset_receipt": read_json(args.output / "dataset_receipt.json"),
        "control_model_receipt": (
            read_json(args.output / "control_model_receipt.json")
            if (args.output / "control_model_receipt.json").exists()
            else None
        ),
        "vlm_control_protocol_registration": (
            read_json(
                args.output / "vlm_control_protocol_registration.json"
            )
            if (args.output / "vlm_control_protocol_registration.json").exists()
            else None
        ),
        "figures": figures,
        "human_report": "docs/ROBOTWIN_MEM_CEM_REPORT.md",
        "decision_log": (
            "outputs/robotwin_mem_admission_v1/gate_decisions.json"
        ),
        "per_episode_predictions": {
            task_id: {
                "dino": (
                    f"outputs/robotwin_mem_admission_v1/"
                    f"predictions_{task_id}.json"
                ),
                "vlm": (
                    f"outputs/robotwin_mem_admission_v1/"
                    f"predictions_vlm_{task_id}.json"
                ),
            }
            for task_id in args.tasks
        },
        "jobs": {
            "status": "completed",
            "logs": "outputs/robotwin_mem_admission_v1/logs/",
            "active_benchmark_processes": 0,
            "allowed_physical_gpus": [0, 1, 2],
            "forbidden_gpu_used": False,
        },
    }
    write_json(args.output / "report.json", report)
    _write_report(summary)
    print(
        f"BENCHMARK_ADMISSION={'PASS' if summary['benchmark_suitable'] else 'FAIL'} "
        f"admitted={admitted}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "register",
            "smoke",
            "index",
            "encode",
            "admission",
            "aggregate",
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO
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
    parser.add_argument("--encode-batch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.official_repo = args.official_repo.resolve()
    args.dinov2 = args.dinov2.resolve()
    args.torch_home = args.torch_home.resolve()
    args.dino_weights = args.dino_weights.resolve()
    return args


def main() -> None:
    args = parse_args()
    started = time.time()
    dispatch = {
        "register": stage_register,
        "smoke": stage_smoke,
        "index": stage_index,
        "encode": stage_encode,
        "admission": stage_admission,
        "aggregate": stage_aggregate,
    }
    dispatch[args.stage](args)
    print(
        f"stage={args.stage} elapsed_seconds={time.time() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
