#!/usr/bin/env python3
"""Audit original DINO-WM on its own official PushT data distribution.

V2 is deliberately separate from the failed same-bank V1 audit.  Its train and
validation episodes come from the original DINO-WM OSF artifact, so these
results test native-distribution portability and are not bank-matched to LeWM.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml

try:
    import decord
    from decord import VideoReader
except ModuleNotFoundError:  # CPU unit tests do not decode official videos.
    decord = None
    VideoReader = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.dinowm_native_audit import (  # noqa: E402
    NATIVE_PATCHES,
    NATIVE_VISUAL_DIM,
    RolloutHealthThresholds,
    endpoint_frame_for_age,
    paired_transport_summary,
    pairwise_counterfactual_separation,
    spatial_pyramid_pool,
    summarize_rollout_health,
    temporal_spatial_pyramid_pool,
)
from lewm.official_tasks.pusht_hdf5 import sha256_file  # noqa: E402
from lewm.official_tasks.pusht_memory import render_single_overlay  # noqa: E402
from scripts.run_dinowm_native_pusht_audit_v1 import (  # noqa: E402
    AuditStop,
    NativeDinoWM,
    _canonical_sha256,
    _check_file_identity,
    _chunks,
    _fixed_normalize_actions,
    _fixed_normalize_proprio,
    _git_archive_sha256,
    _git_output,
    _json_dump,
    _load_locked_config,
    _probe_record,
    _require,
    _resolve,
)


if decord is not None:
    decord.bridge.set_bridge("native")


@dataclass(frozen=True)
class NativeSelection:
    split: str
    source_split: str
    episode_index: int
    local_start: int
    label: int


@dataclass(frozen=True)
class NativeSequence:
    frames: np.ndarray
    actions: np.ndarray
    proprio: np.ndarray
    state: np.ndarray
    split: str
    episode_index: int
    local_start: int


class OfficialDinoWMPushT:
    """Identity-verified reader for the original OSF PushT artifact."""

    def __init__(self, root: Path, manifest_path: Path,
                 manifest_sha256: str) -> None:
        _require(decord is not None and VideoReader is not None,
                 "decord==0.6.0 is required for the formal native dataset")
        self.root = root
        self.manifest_path = manifest_path
        _require(root.is_dir(), f"official PushT root missing: {root}")
        _require(sha256_file(manifest_path) == manifest_sha256,
                 "official PushT extracted manifest SHA-256 mismatch")
        manifest = json.loads(manifest_path.read_text())
        _require(manifest.get("schema") == "dinowm_pusht_extracted_manifest_v1",
                 "unexpected extracted manifest schema")
        records = manifest.get("files")
        _require(isinstance(records, list) and records,
                 "extracted manifest has no files")
        for record in records:
            path = root / record["relative_path"]
            _require(path.is_file(), f"manifest file missing: {path}")
            _require(path.stat().st_size == int(record["size"]),
                     f"manifest size mismatch: {path}")
            _require(sha256_file(path) == record["sha256"],
                     f"manifest SHA-256 mismatch: {path}")
        self.manifest = manifest
        self.splits: dict[str, dict[str, Any]] = {}
        for split in ("train", "val"):
            directory = root / split
            states = torch.load(
                directory / "states.pth", map_location="cpu",
                weights_only=False).float()
            actions = torch.load(
                directory / "rel_actions.pth", map_location="cpu",
                weights_only=False).float()
            velocities = torch.load(
                directory / "velocities.pth", map_location="cpu",
                weights_only=False).float()
            import pickle
            with (directory / "seq_lengths.pkl").open("rb") as handle:
                lengths = tuple(map(int, pickle.load(handle)))
            _require(states.ndim == 3 and states.shape[-1] >= 5,
                     f"{split} states have unexpected shape")
            _require(actions.ndim == 3 and actions.shape[-1] == 2,
                     f"{split} actions have unexpected shape")
            _require(velocities.ndim == 3 and velocities.shape[-1] == 2,
                     f"{split} velocities have unexpected shape")
            _require(states.shape[:2] == actions.shape[:2]
                     == velocities.shape[:2],
                     f"{split} tensor leading shapes disagree")
            _require(len(lengths) == states.shape[0]
                     and all(0 < length <= states.shape[1]
                             for length in lengths),
                     f"{split} sequence lengths are invalid")
            videos = directory / "obses"
            for episode in range(len(lengths)):
                _require((videos / f"episode_{episode:03d}.mp4").is_file(),
                         f"{split} video {episode} is missing")
            self.splits[split] = {
                "directory": directory,
                "states": states,
                "actions": actions,
                "velocities": velocities,
                "lengths": lengths,
                "videos": videos,
            }

    @property
    def schema(self) -> dict[str, Any]:
        return {
            split: {
                "episodes": len(value["lengths"]),
                "length_min": min(value["lengths"]),
                "length_max": max(value["lengths"]),
                "states_shape": list(value["states"].shape),
                "actions_shape": list(value["actions"].shape),
                "velocities_shape": list(value["velocities"].shape),
            }
            for split, value in self.splits.items()
        }

    def select(self, *, train_count: int, validation_count: int,
               num_frames: int, frame_skip: int, classes: int,
               split_seed: int, start_seed: int,
               label_seed: int, source_split: str) -> list[NativeSelection]:
        raw_span = (num_frames - 1) * frame_skip + 1
        _require(source_split in self.splits,
                 f"unknown native source split {source_split}")
        result: list[NativeSelection] = []
        lengths = np.asarray(
            self.splits[source_split]["lengths"], dtype=np.int64)
        eligible = np.flatnonzero(lengths >= raw_span)
        requested = train_count + validation_count
        _require(len(eligible) >= requested,
                 f"{source_split} has {len(eligible)} eligible episodes, "
                 f"needs {requested}")
        selected = np.random.default_rng(split_seed).permutation(
            eligible)[:requested]
        specifications = (
            ("train", selected[:train_count]),
            ("validation", selected[train_count:]),
        )
        for split_index, (split, episodes) in enumerate(specifications):
            count = len(episodes)
            labels = np.arange(count, dtype=np.int64) % classes
            np.random.default_rng(np.random.SeedSequence(
                [label_seed, split_index])).shuffle(labels)
            for episode, label in zip(episodes, labels):
                max_start = int(lengths[episode] - raw_span)
                start = int(np.random.default_rng(np.random.SeedSequence(
                    [start_seed, split_index, int(episode)])).integers(
                        max_start + 1))
                result.append(NativeSelection(
                    split=split, source_split=source_split,
                    episode_index=int(episode), local_start=start,
                    label=int(label)))
        return result

    def read(self, selection: NativeSelection, *, num_frames: int,
             frame_skip: int) -> NativeSequence:
        split = selection.source_split
        values = self.splits[split]
        raw_frames = selection.local_start + np.arange(
            num_frames, dtype=np.int64) * frame_skip
        _require(raw_frames[-1] < values["lengths"][selection.episode_index],
                 "selected native sequence crosses an episode boundary")
        assert VideoReader is not None
        reader = VideoReader(
            str(values["videos"] /
                f"episode_{selection.episode_index:03d}.mp4"),
            num_threads=1)
        _require(len(reader) >= values["lengths"][selection.episode_index],
                 "video is shorter than its declared sequence")
        frames = reader.get_batch(raw_frames.tolist()).asnumpy()
        _require(frames.ndim == 4 and frames.shape[-1] == 3
                 and frames.dtype == np.uint8,
                 "decoded native video is not uint8 THWC RGB")
        action_tensor = values["actions"][selection.episode_index]
        blocks = []
        for frame in raw_frames[:-1]:
            block = action_tensor[int(frame):int(frame) + frame_skip].numpy()
            _require(block.shape == (frame_skip, 2),
                     "native action block has the wrong shape")
            blocks.append(block)
        # PushTDataset divides stored relative pixel displacements by 100
        # before applying the fixed checkpoint normalization constants.
        actions = np.stack(blocks).reshape(num_frames - 1, -1).astype(
            np.float32) / 100.0
        states = values["states"][selection.episode_index, raw_frames].numpy()
        velocities = values["velocities"][
            selection.episode_index, raw_frames].numpy()
        state = np.concatenate((states, velocities), axis=-1).astype(np.float32)
        proprio = np.concatenate((states[:, :2], velocities), axis=-1).astype(
            np.float32)
        _require(state.shape == (num_frames, 7)
                 and proprio.shape == (num_frames, 4)
                 and actions.shape == (num_frames - 1, 10),
                 "native sequence violates state/proprio/action contract")
        _require(np.isfinite(state).all() and np.isfinite(proprio).all()
                 and np.isfinite(actions).all(),
                 "native sequence contains non-finite values")
        return NativeSequence(
            frames=frames, actions=actions, proprio=proprio, state=state,
            split=selection.split, episode_index=selection.episode_index,
            local_start=selection.local_start)


class NativeDistributionDinoWM(NativeDinoWM):
    """V1 model adapter with the original variable-resolution video transform."""

    def encode_visual(self, frames: np.ndarray, *, batch_size: int) -> np.ndarray:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        values = np.asarray(frames)
        if values.ndim != 4 or values.shape[-1] != 3 \
                or values.dtype != np.uint8 or min(values.shape[1:3]) < 1:
            raise ValueError("native frames must be uint8 BHWC RGB")
        outputs: list[np.ndarray] = []
        with self.torch.inference_mode():
            for batch in _chunks(values, batch_size):
                tensor = self.torch.from_numpy(np.asarray(batch).copy()).to(
                    self.device)
                tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
                # Original dataset default_transform, including integer Resize
                # semantics (short edge 224) before its center crop.
                tensor = TF.resize(
                    tensor, 224, interpolation=InterpolationMode.BILINEAR,
                    antialias=True)
                tensor = TF.center_crop(tensor, [224, 224])
                tensor = tensor.sub_(0.5).div_(0.5)
                # Released VWorldModel.encoder_transform.
                tensor = TF.resize(
                    tensor, [196, 196],
                    interpolation=InterpolationMode.BILINEAR, antialias=True)
                patches = self.encoder.forward_features(tensor)[
                    "x_norm_patchtokens"]
                _require(tuple(patches.shape[1:]) ==
                         (NATIVE_PATCHES, NATIVE_VISUAL_DIM),
                         "DINO native patch output violates contract")
                outputs.append(patches.float().cpu().numpy())
        return np.concatenate(outputs)


def _stack(values: Sequence[NativeSequence]) -> dict[str, np.ndarray]:
    return {
        "frames": np.stack([value.frames for value in values]),
        "actions": np.stack([value.actions for value in values]),
        "proprio": np.stack([value.proprio for value in values]),
        "state": np.stack([value.state for value in values]),
    }


def _read(dataset: OfficialDinoWMPushT,
          selections: Sequence[NativeSelection], cfg: Mapping[str, Any]
          ) -> list[NativeSequence]:
    return [dataset.read(
        selection, num_frames=int(cfg["sequence"]["num_frames"]),
        frame_skip=int(cfg["sequence"]["frame_skip"]))
        for selection in selections]


def _split(values: np.ndarray, selections: Sequence[NativeSelection]
           ) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray([item.split == "train" for item in selections])
    array = np.asarray(values)
    return array[mask], array[~mask]


def _select_tasks(dataset: OfficialDinoWMPushT,
                  cfg: Mapping[str, Any]) -> dict[str, list[NativeSelection]]:
    result = {}
    for task in cfg["tasks"]:
        result[task["key"]] = dataset.select(
            train_count=int(cfg["selection"]["train_episodes"]),
            validation_count=int(cfg["selection"]["validation_episodes"]),
            num_frames=int(cfg["sequence"]["num_frames"]),
            frame_skip=int(cfg["sequence"]["frame_skip"]),
            classes=int(task["classes"]),
            split_seed=int(cfg["selection"]["split_seed"]),
            start_seed=int(cfg["selection"]["start_seed"]),
            label_seed=int(task["label_seed"]),
            source_split=str(cfg["selection"]["source_split"]),
        )
    reference = next(iter(result.values()))
    identity = [(x.split, x.episode_index, x.local_start) for x in reference]
    for key, values in result.items():
        _require([(x.split, x.episode_index, x.local_start) for x in values]
                 == identity, f"task {key} does not share the native base bank")
    return result


def _teacher_features(*, model: NativeDistributionDinoWM,
                      dataset: OfficialDinoWMPushT,
                      selections: Sequence[NativeSelection],
                      cfg: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {
        "observed_context": [], "predicted_endpoint": [],
        "action_shortcut": [], "proprio_shortcut": [],
    }
    for selected in _chunks(
            selections, int(cfg["execution"]["teacher_batch_size"])):
        native = _read(dataset, selected, cfg)
        arrays = _stack(native)
        frame_shape = arrays["frames"].shape[2:]
        frames = arrays["frames"][:, 16:19]
        patches = model.encode_visual(
            frames.reshape(-1, *frame_shape),
            batch_size=int(cfg["execution"]["frame_batch_size"]),
        ).reshape(len(native), 3, NATIVE_PATCHES, NATIVE_VISUAL_DIM)
        endpoint = model.teacher_endpoint(
            patches, arrays["proprio"], arrays["actions"])
        result["observed_context"].append(
            temporal_spatial_pyramid_pool(patches))
        result["predicted_endpoint"].append(spatial_pyramid_pool(endpoint))
        result["action_shortcut"].append(
            _fixed_normalize_actions(arrays["actions"])[:, 15:19].reshape(
                len(native), -1))
        result["proprio_shortcut"].append(
            _fixed_normalize_proprio(arrays["proprio"])[:, 16:19].reshape(
                len(native), -1))
    return {key: np.concatenate(values) for key, values in result.items()}


def _cue_features(*, model: NativeDistributionDinoWM,
                  dataset: OfficialDinoWMPushT,
                  selections: Sequence[NativeSelection],
                  task: Mapping[str, Any], cfg: Mapping[str, Any]) -> np.ndarray:
    outputs = []
    for selected in _chunks(
            selections, int(cfg["execution"]["cue_batch_size"])):
        native = _read(dataset, selected, cfg)
        cues = [render_single_overlay(
            value.frames, task["semantic_name"], selection.label,
            int(cfg["sequence"]["cue_start"]),
            int(cfg["sequence"]["cue_length"]))[1:4]
            for value, selection in zip(native, selected)]
        cue = np.stack(cues)
        frame_shape = cue.shape[2:]
        patches = model.encode_visual(
            cue.reshape(-1, *frame_shape),
            batch_size=int(cfg["execution"]["frame_batch_size"]),
        ).reshape(len(native), 3, NATIVE_PATCHES, NATIVE_VISUAL_DIM)
        outputs.append(temporal_spatial_pyramid_pool(patches))
    return np.concatenate(outputs)


def _health(*, model: NativeDistributionDinoWM,
            dataset: OfficialDinoWMPushT,
            validation: Sequence[NativeSelection], cfg: Mapping[str, Any]
            ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    health_cfg = cfg["rollout_health"]
    count = int(health_cfg["episodes"])
    selected = list(validation[:count])
    _require(len(selected) == count, "not enough native health episodes")
    native = _read(dataset, selected, cfg)
    actions = np.stack([value.actions for value in native])
    permutation = np.random.default_rng(int(health_cfg["shuffle_seed"])).permutation(
        count)
    true_rows, copy_rows, shuffle_rows = [], [], []
    for indices in _chunks(list(range(count)), int(health_cfg["batch_size"])):
        batch = [native[index] for index in indices]
        arrays = _stack(batch)
        frames = arrays["frames"][:, 1:20]
        frame_shape = frames.shape[2:]
        visual = model.encode_visual(
            frames.reshape(-1, *frame_shape),
            batch_size=int(cfg["execution"]["frame_batch_size"]),
        ).reshape(len(batch), 19, NATIVE_PATCHES, NATIVE_VISUAL_DIM)
        true, copy, shuffled = model.health(
            visual, arrays["proprio"], arrays["actions"],
            actions[permutation[np.asarray(indices)]])
        true_rows.append(true)
        copy_rows.append(copy)
        shuffle_rows.append(shuffled)
    arrays = {
        "true_action_mse": np.concatenate(true_rows),
        "copy_last_mse": np.concatenate(copy_rows),
        "shuffled_action_mse": np.concatenate(shuffle_rows),
        "shuffle_permutation": permutation.astype(np.int64),
        "episode_index": np.asarray(
            [item.episode_index for item in selected], dtype=np.int64),
    }
    thresholds = RolloutHealthThresholds(
        one_step_copy_ratio_max=float(health_cfg["one_step_copy_ratio_max"]),
        integrated_copy_ratio_max=float(
            health_cfg["integrated_copy_ratio_max"]),
        integrated_action_advantage_min=float(
            health_cfg["integrated_action_advantage_min"]),
    )
    summary = summarize_rollout_health(
        arrays["true_action_mse"], arrays["copy_last_mse"],
        arrays["shuffled_action_mse"], thresholds=thresholds)
    summary.update({
        "native_official_dataset": True,
        "target": "frozen DINOv2 x_norm_patchtokens",
        "shuffle_seed": int(health_cfg["shuffle_seed"]),
    })
    return summary, arrays


def _assigned_rollout(*, model: NativeDistributionDinoWM,
                      dataset: OfficialDinoWMPushT,
                      selections: Sequence[NativeSelection],
                      task: Mapping[str, Any], cfg: Mapping[str, Any]
                      ) -> dict[int, np.ndarray]:
    ages = tuple(map(int, cfg["evaluation"]["evidence_ages"]))
    outputs: dict[int, list[np.ndarray]] = {age: [] for age in ages}
    for selected in _chunks(
            selections, int(cfg["execution"]["rollout_batch_size"])):
        native = _read(dataset, selected, cfg)
        arrays = _stack(native)
        cues = [render_single_overlay(
            value.frames, task["semantic_name"], selection.label,
            int(cfg["sequence"]["cue_start"]),
            int(cfg["sequence"]["cue_length"]))[1:4]
            for value, selection in zip(native, selected)]
        cue = np.stack(cues)
        frame_shape = cue.shape[2:]
        patches = model.encode_visual(
            cue.reshape(-1, *frame_shape),
            batch_size=int(cfg["execution"]["frame_batch_size"]),
        ).reshape(len(native), 3, NATIVE_PATCHES, NATIVE_VISUAL_DIM)
        rollout = model.rollout(
            patches, arrays["proprio"], arrays["actions"], ages=ages)
        for age in ages:
            outputs[age].append(spatial_pyramid_pool(rollout[age]))
    return {age: np.concatenate(values) for age, values in outputs.items()}


def _counterfactual(*, model: NativeDistributionDinoWM,
                    dataset: OfficialDinoWMPushT,
                    validation: Sequence[NativeSelection],
                    task: Mapping[str, Any], cfg: Mapping[str, Any]
                    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    count = int(cfg["evaluation"]["counterfactual_episodes"])
    selected = list(validation[:count])
    classes = int(task["classes"])
    ages = tuple(map(int, cfg["evaluation"]["evidence_ages"]))
    cue_rows: list[np.ndarray] = []
    age_rows: dict[int, list[np.ndarray]] = {age: [] for age in ages}
    for batch_selection in _chunks(
            selected,
            int(cfg["execution"]["counterfactual_episode_batch_size"])):
        native = _read(dataset, batch_selection, cfg)
        frames, proprio, actions = [], [], []
        for value in native:
            for label in range(classes):
                frames.append(render_single_overlay(
                    value.frames, task["semantic_name"], label,
                    int(cfg["sequence"]["cue_start"]),
                    int(cfg["sequence"]["cue_length"]))[1:4])
                proprio.append(value.proprio)
                actions.append(value.actions)
        cue = np.stack(frames)
        frame_shape = cue.shape[2:]
        patches = model.encode_visual(
            cue.reshape(-1, *frame_shape),
            batch_size=int(cfg["execution"]["frame_batch_size"]),
        ).reshape(len(native) * classes, 3,
                  NATIVE_PATCHES, NATIVE_VISUAL_DIM)
        cue_rows.append(pairwise_counterfactual_separation(
            patches.mean(axis=1).reshape(
                len(native), classes, NATIVE_PATCHES, NATIVE_VISUAL_DIM)))
        rollout = model.rollout(
            patches, np.stack(proprio), np.stack(actions), ages=ages)
        for age in ages:
            age_rows[age].append(pairwise_counterfactual_separation(
                rollout[age].reshape(
                    len(native), classes, NATIVE_PATCHES, NATIVE_VISUAL_DIM)))
    arrays = {"cue_separation": np.concatenate(cue_rows)}
    summary: dict[str, Any] = {
        "episodes": count, "classes": classes,
        "unit": "within-episode all-label pair mean RMS in DINO feature units",
        "ages": {},
    }
    bootstrap = cfg["bootstrap"]
    for offset, age in enumerate(ages):
        values = np.concatenate(age_rows[age])
        arrays[f"age_{age}_separation"] = values
        summary["ages"][str(age)] = paired_transport_summary(
            arrays["cue_separation"], values,
            draws=int(bootstrap["draws"]),
            seed=int(bootstrap["seed"]) + 800 + offset,
            confidence=float(bootstrap["confidence"]),
        )
    return summary, arrays


def execute(config_path: Path) -> dict[str, Any]:
    cfg, lock = _load_locked_config(config_path)
    output = _resolve(ROOT, cfg["artifacts"]["root"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    provenance: dict[str, Any] = {
        "schema": "dinowm_native_distribution_audit_provenance_v2",
        "protocol_path": str(config_path),
        "protocol_sha256": lock["protocol_sha256"],
        "started_unix": started,
        "v1_same_bank_status": "failed_preserved",
        "bank_matched_to_lewm": False,
        "paper_modified": False,
        "carrier_injection": False,
        "persistent_state_test": "not_applicable",
        "downstream_use_test": "not_applicable",
    }
    try:
        archive = _resolve(ROOT, cfg["dataset"]["archive_path"])
        manifest = _resolve(ROOT, cfg["dataset"]["manifest_path"])
        dataset_root = _resolve(ROOT, cfg["dataset"]["root"])
        checkpoint = _resolve(ROOT, cfg["checkpoint"]["weights_path"])
        checkpoint_config = _resolve(ROOT, cfg["checkpoint"]["config_path"])
        dino_weights = _resolve(ROOT, cfg["dino_encoder"]["weights_path"])
        dependency_manifest = _resolve(
            ROOT, cfg["execution"]["dependency_manifest_path"])
        identities = {
            "dataset_archive": _check_file_identity(
                archive, cfg["dataset"]["archive_identity"]),
            "checkpoint": _check_file_identity(
                checkpoint, cfg["checkpoint"]["weights_identity"]),
            "checkpoint_config": _check_file_identity(
                checkpoint_config, cfg["checkpoint"]["config_identity"]),
            "dino_weights": _check_file_identity(
                dino_weights, cfg["dino_encoder"]["weights_identity"]),
            "extracted_manifest": _check_file_identity(
                manifest, cfg["dataset"]["manifest_identity"]),
            "dependency_manifest": _check_file_identity(
                dependency_manifest,
                cfg["execution"]["dependency_manifest_identity"]),
        }
        dependency_record = json.loads(dependency_manifest.read_text())
        _require(dependency_record.get("package") == "decord==0.6.0",
                 "isolated dependency manifest does not pin decord==0.6.0")
        dependency_root = _resolve(
            ROOT, cfg["execution"]["decord_dependency"])
        for record in dependency_record["files"]:
            path = dependency_root / record["relative_path"]
            _require(path.is_file() and path.stat().st_size == record["size"]
                     and sha256_file(path) == record["sha256"],
                     f"isolated dependency mismatch: {path}")
        source_records = {}
        for key, section in (("dino_wm", cfg["source"]),
                             ("dinov2", cfg["dino_encoder"])):
            repo = _resolve(ROOT, section["repo_path"])
            head = _git_output(repo, "rev-parse", "HEAD")
            status = _git_output(repo, "status", "--porcelain")
            tree = _git_archive_sha256(repo)
            _require(head == section["revision"] and not status
                     and tree == section["source_tree_sha256"],
                     f"{key} source identity mismatch")
            source_records[key] = {
                "path": str(repo), "revision": head, "clean": True,
                "git_archive_sha256": tree,
            }
        provenance.update({
            "identities": identities,
            "sources": source_records,
            "preprocessing_contract_sha256": _canonical_sha256(
                cfg["preprocessing"]),
            "environment": {
                "python": sys.version, "executable": sys.executable,
                "platform": platform.platform(),
                "torch": torch.__version__, "numpy": np.__version__,
                "decord": decord.__version__ if decord is not None else None,
                "disk_before": shutil.disk_usage(ROOT)._asdict(),
            },
        })
        _json_dump(output / "provenance.pre_model.json", provenance)

        dataset = OfficialDinoWMPushT(
            dataset_root, manifest,
            cfg["dataset"]["manifest_identity"]["sha256"])
        selections_by_task = _select_tasks(dataset, cfg)
        selection_payload = {
            key: [asdict(item) for item in values]
            for key, values in selections_by_task.items()}
        provenance["dataset_schema"] = dataset.schema
        provenance["selection_sha256"] = _canonical_sha256(selection_payload)
        _json_dump(output / "selection.json", {
            "sha256": provenance["selection_sha256"],
            "bank_matched_to_lewm": False,
            "tasks": selection_payload,
        })

        model = NativeDistributionDinoWM(
            checkpoint=checkpoint,
            dino_repo=_resolve(ROOT, cfg["dino_encoder"]["repo_path"]),
            torch_home=_resolve(ROOT, cfg["dino_encoder"]["torch_home"]),
            vendor_repo=_resolve(ROOT, cfg["source"]["repo_path"]),
            device=cfg["execution"]["device"])
        provenance["model_schema"] = model.schema
        provenance["environment"].update({
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(2),
            "device": str(model.device),
            "deterministic_algorithms":
                torch.are_deterministic_algorithms_enabled(),
        })
        _json_dump(output / "provenance.json", provenance)

        reference = selections_by_task[cfg["tasks"][0]["key"]]
        teacher = _teacher_features(
            model=model, dataset=dataset, selections=reference, cfg=cfg)
        np.savez_compressed(output / "teacher_features.npz", **teacher)
        cue_by_task: dict[str, np.ndarray] = {}
        for task in cfg["tasks"]:
            key = task["key"]
            cue_by_task[key] = _cue_features(
                model=model, dataset=dataset,
                selections=selections_by_task[key], task=task, cfg=cfg)
            np.savez_compressed(
                output / "admission" / f"{key}_features.npz",
                cue=cue_by_task[key], **teacher,
                labels=np.asarray(
                    [item.label for item in selections_by_task[key]],
                    dtype=np.int64))

        admissions: dict[str, Any] = {}
        for task_index, task in enumerate(cfg["tasks"]):
            key = task["key"]
            selections = selections_by_task[key]
            labels = np.asarray([item.label for item in selections], dtype=np.int64)
            train_y, validation_y = _split(labels, selections)
            probes: dict[str, Any] = {}
            for probe_index, (name, features) in enumerate(
                    {"cue": cue_by_task[key], **teacher}.items()):
                train_x, validation_x = _split(features, selections)
                probes[name] = _probe_record(
                    train_x, train_y, validation_x, validation_y,
                    classes=int(task["classes"]),
                    bootstrap_cfg=cfg["bootstrap"],
                    seed_offset=task_index * 100 + probe_index)
            chance = 1.0 / int(task["classes"])
            ceiling = chance + float(
                cfg["admission"]["shortcut_margin_above_chance"])
            gates = {
                "cue_availability": {
                    "value": probes["cue"]["balanced_accuracy"],
                    "threshold": float(cfg["admission"]["cue_accuracy_min"]),
                    "direction": ">=",
                    "pass": probes["cue"]["balanced_accuracy"] >=
                            float(cfg["admission"]["cue_accuracy_min"]),
                }}
            for name in ("observed_context", "predicted_endpoint",
                         "action_shortcut", "proprio_shortcut"):
                gates[f"{name}_no_shortcut"] = {
                    "value": probes[name]["balanced_accuracy"],
                    "threshold": ceiling, "direction": "<=",
                    "pass": probes[name]["balanced_accuracy"] <= ceiling,
                }
            admissions[key] = {
                "task": task, "chance": chance,
                "shortcut_ceiling": ceiling, "probes": probes,
                "gates": gates,
                "admitted": all(value["pass"] for value in gates.values()),
            }
            _json_dump(output / "admission" / f"{key}.json", admissions[key])

        validation = [item for item in reference
                      if item.split == "validation"]
        health, health_arrays = _health(
            model=model, dataset=dataset, validation=validation, cfg=cfg)
        np.savez_compressed(output / "rollout_health.npz", **health_arrays)
        _json_dump(output / "rollout_health.json", health)
        failed = [key for key, value in admissions.items()
                  if not value["admitted"]]
        if failed:
            raise AuditStop("native task admission failed: " + ", ".join(failed))
        if not health["admitted"]:
            raise AuditStop("native-distribution rollout-health gate failed")

        task_results: dict[str, Any] = {}
        for task_index, task in enumerate(cfg["tasks"]):
            key = task["key"]
            selections = selections_by_task[key]
            labels = np.asarray([item.label for item in selections], dtype=np.int64)
            train_y, validation_y = _split(labels, selections)
            rollout = _assigned_rollout(
                model=model, dataset=dataset, selections=selections,
                task=task, cfg=cfg)
            age_results = {}
            for age_index, age in enumerate(cfg["evaluation"]["evidence_ages"]):
                age = int(age)
                train_x, validation_x = _split(rollout[age], selections)
                probe = _probe_record(
                    train_x, train_y, validation_x, validation_y,
                    classes=int(task["classes"]),
                    bootstrap_cfg=cfg["bootstrap"],
                    seed_offset=400 + task_index * 20 + age_index)
                chance = 1.0 / int(task["classes"])
                probe.update({
                    "chance": chance,
                    "chance_normalized_accuracy":
                        (probe["balanced_accuracy"] - chance) / (1.0 - chance),
                    "endpoint_frame": endpoint_frame_for_age(
                        last_cue_frame=3, age=age),
                })
                age_results[str(age)] = probe
            task_validation = [item for item in selections
                               if item.split == "validation"]
            counterfactual, counterfactual_arrays = _counterfactual(
                model=model, dataset=dataset, validation=task_validation,
                task=task, cfg=cfg)
            np.savez_compressed(
                output / "results" / f"{key}.npz",
                labels=labels,
                **{f"age_{age}_features": features
                   for age, features in rollout.items()},
                **counterfactual_arrays)
            task_results[key] = {
                "task": task,
                "open_loop_decodability": age_results,
                "paired_counterfactual_separation": counterfactual,
                "bank_matched_to_lewm": False,
                "claim_scope": (
                    "native-distribution cue-anchored imagination transport; "
                    "not persistent real-observation memory"),
            }
            _json_dump(output / "results" / f"{key}.json", task_results[key])

        summary = {
            "schema": "dinowm_native_distribution_pusht_audit_v2",
            "status": "complete",
            "same_bank_v1_status": "failed_preserved",
            "bank_matched_to_lewm": False,
            "host": cfg["checkpoint"]["display_name"],
            "host_is_dinowm_noprop": False,
            "admissions": admissions,
            "rollout_health": health,
            "tasks": task_results,
            "claim_ledger": {
                "cue_availability": "tested",
                "teacher_forced_endpoint_exposure": "tested",
                "native_open_loop_decodability": "tested",
                "paired_counterfactual_feature_separation": "tested",
                "rollout_health_through_16": "tested",
                "persistent_state_retention": "not_applicable",
                "downstream_use": "not_applicable",
            },
            "elapsed_seconds": time.time() - started,
        }
        _json_dump(output / "summary.json", summary)
        provenance["completed_unix"] = time.time()
        provenance["elapsed_seconds"] = provenance["completed_unix"] - started
        provenance["environment"]["disk_after"] = shutil.disk_usage(ROOT)._asdict()
        _json_dump(output / "provenance.json", provenance)
        return summary
    except AuditStop as exc:
        receipt = {
            "schema": "dinowm_native_distribution_audit_stop_v2",
            "status": "stopped_fail_closed",
            "reason": str(exc),
            "same_bank_v1_status": "failed_preserved",
            "bank_matched_to_lewm": False,
            "no_post_hoc_adaptation": True,
            "elapsed_seconds": time.time() - started,
        }
        _json_dump(output / "stop_receipt.json", receipt)
        provenance["stop_receipt"] = receipt
        provenance["environment"]["disk_after"] = shutil.disk_usage(ROOT)._asdict()
        _json_dump(output / "provenance.json", provenance)
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/dinowm_native_pusht_audit_v2.yaml")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing metric-bearing execution without --execute")
    print(json.dumps(execute(args.config.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
