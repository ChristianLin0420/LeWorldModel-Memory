#!/usr/bin/env python3
"""Execute the locked native DINO-WM PushT portability audit on GPU 2.

This is a frozen-model audit.  It never trains or injects a carrier.  The
original official DINO-WM PushT checkpoint contains proprioception and is kept
strictly separate from LeWM's unavailable ``dinowm_noprop`` release.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.dinowm_native_audit import (  # noqa: E402
    NATIVE_ACTION_DIM,
    NATIVE_CONTEXT,
    NATIVE_PATCHES,
    NATIVE_PROPRIO_DIM,
    NATIVE_VISUAL_DIM,
    RolloutHealthThresholds,
    bootstrap_accuracy_ci,
    endpoint_frame_for_age,
    frozen_linear_probe,
    paired_transport_summary,
    pairwise_counterfactual_separation,
    spatial_pyramid_pool,
    strip_probe_arrays,
    summarize_rollout_health,
    temporal_spatial_pyramid_pool,
)
from lewm.official_tasks.pusht_hdf5 import (  # noqa: E402
    NativePushTSequence,
    OfficialPushTHDF5,
    PushTSequenceSelection,
    sha256_file,
)
from lewm.official_tasks.pusht_memory import (  # noqa: E402
    render_single_overlay,
)


ACTION_MEAN = np.asarray([-0.0087, 0.0068], dtype=np.float32)
ACTION_STD = np.asarray([0.2019, 0.2002], dtype=np.float32)
PROPRIO_MEAN = np.asarray(
    [236.6155, 264.5674, -2.93032027, 2.54307914], dtype=np.float32)
PROPRIO_STD = np.asarray(
    [101.1202, 87.0112, 74.84556075, 74.14009094], dtype=np.float32)


class AuditStop(RuntimeError):
    """A preregistered gate failed; downstream metric computation must stop."""


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True).strip()


def _git_archive_sha256(repo: Path) -> str:
    process = subprocess.Popen(
        ["git", "-C", str(repo), "archive", "--format=tar", "HEAD"],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    status = process.wait()
    if status != 0:
        raise RuntimeError(f"git archive failed for {repo}")
    return digest.hexdigest()


def _check_file_identity(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.is_file(), f"pinned file is missing: {path}")
    size = path.stat().st_size
    _require(size == int(identity["size"]),
             f"size mismatch for {path}: expected {identity['size']}, got {size}")
    digest = sha256_file(path)
    _require(digest == identity["sha256"],
             f"SHA-256 mismatch for {path}: expected {identity['sha256']}, got {digest}")
    return {"path": str(path), "size": size, "sha256": digest}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_locked_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise RuntimeError("audit config must be a YAML mapping")
    lock_path = path.with_suffix(".lock.json")
    lock = json.loads(lock_path.read_text())
    digest = hashlib.sha256(raw).hexdigest()
    _require(lock.get("protocol_sha256") == digest,
             "protocol YAML differs from the pre-metric lock")
    _require(config.get("protocol_status") == "locked_before_metrics",
             "protocol is not marked locked_before_metrics")
    for relpath, expected in lock.get("code_sha256", {}).items():
        actual = sha256_file(_resolve(ROOT, relpath))
        _require(actual == expected,
                 f"locked code identity mismatch for {relpath}")
    return config, lock


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _stack_native(values: Sequence[NativePushTSequence]) -> dict[str, np.ndarray]:
    return {
        "frames": np.stack([value.frames for value in values]),
        "actions": np.stack([value.actions for value in values]),
        "proprio": np.stack([value.proprio for value in values]),
        "state": np.stack([value.state for value in values]),
    }


def _read_batch(dataset: OfficialPushTHDF5,
                selections: Sequence[PushTSequenceSelection],
                num_frames: int) -> list[NativePushTSequence]:
    return [dataset.read_sequence(
        selection.episode_index, selection.local_start, num_frames)
        for selection in selections]


def _fixed_normalize_actions(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if values.shape[-1] != NATIVE_ACTION_DIM:
        raise ValueError("native action blocks must be 10-D")
    raw = values.reshape(*values.shape[:-1], 5, 2)
    return ((raw - ACTION_MEAN) / ACTION_STD).reshape(values.shape).astype(
        np.float32)


def _fixed_normalize_proprio(proprio: np.ndarray) -> np.ndarray:
    values = np.asarray(proprio, dtype=np.float32)
    if values.shape[-1] != NATIVE_PROPRIO_DIM:
        raise ValueError("native proprioception must be 4-D")
    return ((values - PROPRIO_MEAN) / PROPRIO_STD).astype(np.float32)


def _column_stats(path: Path, key: str, *, chunk_rows: int = 131_072,
                  ddof: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r", swmr=True) as handle:
        dataset = handle[key]
        width = int(np.prod(dataset.shape[1:]))
        count = np.zeros(width, dtype=np.int64)
        total = np.zeros(width, dtype=np.float64)
        for start in range(0, dataset.shape[0], chunk_rows):
            values = np.asarray(dataset[start:start + chunk_rows],
                                dtype=np.float64).reshape(-1, width)
            finite = np.isfinite(values)
            count += finite.sum(axis=0)
            total += np.where(finite, values, 0.0).sum(axis=0)
        mean = total / count
        squared = np.zeros(width, dtype=np.float64)
        for start in range(0, dataset.shape[0], chunk_rows):
            values = np.asarray(dataset[start:start + chunk_rows],
                                dtype=np.float64).reshape(-1, width)
            finite = np.isfinite(values)
            delta = np.where(finite, values - mean, 0.0)
            squared += np.square(delta).sum(axis=0)
    std = np.sqrt(squared / (count - ddof))
    return mean, std, count


def _distribution_gate(mean: np.ndarray, std: np.ndarray,
                       reference_mean: np.ndarray,
                       reference_std: np.ndarray, *,
                       mean_shift_max: float, std_ratio_min: float,
                       std_ratio_max: float) -> dict[str, Any]:
    standardized_mean_shift = (mean - reference_mean) / reference_std
    std_ratio = std / reference_std
    mean_pass = bool(np.max(np.abs(standardized_mean_shift)) <= mean_shift_max)
    std_pass = bool(np.min(std_ratio) >= std_ratio_min
                    and np.max(std_ratio) <= std_ratio_max)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "reference_mean": reference_mean.astype(float).tolist(),
        "reference_std": reference_std.astype(float).tolist(),
        "standardized_mean_shift": standardized_mean_shift.tolist(),
        "std_ratio": std_ratio.tolist(),
        "gates": {
            "absolute_standardized_mean_shift": {
                "value": float(np.max(np.abs(standardized_mean_shift))),
                "threshold": mean_shift_max,
                "direction": "<=",
                "pass": mean_pass,
            },
            "std_ratio_range": {
                "value": [float(np.min(std_ratio)), float(np.max(std_ratio))],
                "threshold": [std_ratio_min, std_ratio_max],
                "direction": "inside_closed_interval",
                "pass": std_pass,
            },
        },
        "admitted": mean_pass and std_pass,
    }


class NativeDinoWM:
    """Thin, inference-only adapter preserving the released rollout contract."""

    def __init__(self, *, checkpoint: Path, dino_repo: Path,
                 torch_home: Path, vendor_repo: Path, device: str) -> None:
        if device != "cuda:2":
            raise RuntimeError("formal DINO-WM execution is restricted to cuda:2")
        os.environ["TORCH_HOME"] = str(torch_home)
        sys.path.insert(0, str(vendor_repo))
        import torch

        self.torch = torch
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        _require(torch.cuda.device_count() > 2, "physical CUDA device 2 is unavailable")
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        torch.manual_seed(9070)
        torch.cuda.manual_seed_all(9070)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _require(set(payload) == {
            "epoch", "predictor", "predictor_optimizer", "decoder",
            "decoder_optimizer", "action_encoder", "proprio_encoder",
        }, "released checkpoint has an unexpected top-level schema")
        self.epoch = int(payload["epoch"])
        self.predictor = payload["predictor"].eval().to(self.device)
        self.action_encoder = payload["action_encoder"].eval().to(self.device)
        self.proprio_encoder = payload["proprio_encoder"].eval().to(self.device)
        # Attention.bias is an upstream plain tensor, not a registered buffer.
        # Native loading maps it to the execution device; reproduce that here
        # after CPU-only unpickling so optimizer tensors never enter GPU memory.
        moved_biases = 0
        for module in self.predictor.modules():
            bias = getattr(module, "bias", None)
            if torch.is_tensor(bias) and bias.ndim == 4:
                module.bias = bias.to(self.device)
                moved_biases += 1
        _require(moved_biases == 6,
                 f"expected six causal attention masks, moved {moved_biases}")
        del payload

        self.encoder = torch.hub.load(
            str(dino_repo), "dinov2_vits14", source="local", pretrained=True
        ).eval().to(self.device)
        for module in (self.encoder, self.predictor,
                       self.action_encoder, self.proprio_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self._verify_schema()

    def _verify_schema(self) -> None:
        torch = self.torch
        pos = self.predictor.pos_embedding
        _require(tuple(pos.shape) == (1, NATIVE_CONTEXT * NATIVE_PATCHES, 404),
                 f"unexpected predictor position shape {tuple(pos.shape)}")
        _require(self.encoder.num_features == NATIVE_VISUAL_DIM
                 and self.encoder.patch_size == 14,
                 "unexpected DINOv2 encoder contract")
        action_weight = self.action_encoder.patch_embed.weight
        proprio_weight = self.proprio_encoder.patch_embed.weight
        _require(tuple(action_weight.shape) == (10, 10, 1),
                 "unexpected action encoder shape")
        _require(tuple(proprio_weight.shape) == (10, 4, 1),
                 "unexpected proprio encoder shape")
        _require(all(not parameter.requires_grad for parameter in
                     list(self.encoder.parameters())
                     + list(self.predictor.parameters())
                     + list(self.action_encoder.parameters())
                     + list(self.proprio_encoder.parameters())),
                 "frozen-model invariant failed")
        _require(next(self.predictor.parameters()).device == self.device,
                 "predictor is on the wrong CUDA device")
        _require(torch.cuda.current_device() == 2,
                 "formal process is not bound to physical CUDA device 2")

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "checkpoint_epoch": self.epoch,
            "context": NATIVE_CONTEXT,
            "patches": NATIVE_PATCHES,
            "visual_dim": NATIVE_VISUAL_DIM,
            "proprio_dim": NATIVE_PROPRIO_DIM,
            "action_block_dim": NATIVE_ACTION_DIM,
            "predictor_tokens": int(self.predictor.pos_embedding.shape[1]),
            "predictor_dim": int(self.predictor.pos_embedding.shape[2]),
            "persistent_state": False,
            "carrier_injection": False,
        }

    def encode_visual(self, frames: np.ndarray, *, batch_size: int) -> np.ndarray:
        """Apply the released 224->196 DINO preprocessing exactly."""

        torch = self.torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        values = np.asarray(frames)
        if values.ndim != 4 or values.shape[1:] != (224, 224, 3) \
                or values.dtype != np.uint8:
            raise ValueError("DINO-WM frames must be uint8 Bx224x224x3")
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in _chunks(values, batch_size):
                tensor = torch.from_numpy(np.asarray(batch).copy()).to(
                    self.device, non_blocking=False)
                tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
                # dataset default_transform: Resize(224), CenterCrop(224),
                # Normalize(.5,.5); the square 224 bank makes the first two no-op.
                tensor = tensor.sub_(0.5).div_(0.5)
                # VWorldModel.encoder_transform: 224//16 * DINO patch 14 = 196.
                tensor = TF.resize(
                    tensor, [196, 196], interpolation=InterpolationMode.BILINEAR,
                    antialias=True)
                patches = self.encoder.forward_features(tensor)[
                    "x_norm_patchtokens"]
                _require(tuple(patches.shape[1:]) ==
                         (NATIVE_PATCHES, NATIVE_VISUAL_DIM),
                         "DINO patch output violates the locked contract")
                outputs.append(patches.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _compose(self, visual: Any, proprio: Any, action: Any) -> Any:
        torch = self.torch
        prop_emb = self.proprio_encoder(proprio)
        action_emb = self.action_encoder(action)
        prop_tiled = prop_emb.unsqueeze(2).expand(-1, -1, visual.shape[2], -1)
        action_tiled = action_emb.unsqueeze(2).expand(
            -1, -1, visual.shape[2], -1)
        return torch.cat((visual, prop_tiled, action_tiled), dim=-1)

    def _predict(self, context: Any) -> Any:
        batch, time_steps, patches, dim = context.shape
        prediction = self.predictor(context.reshape(batch, time_steps * patches,
                                                     dim))
        return prediction.reshape(batch, time_steps, patches, dim)

    def rollout(self, cue_visual: np.ndarray, proprio: np.ndarray,
                actions: np.ndarray, *, ages: Sequence[int]) -> dict[int, np.ndarray]:
        """Native cue-anchored open loop from frames 1,2,3 through frame 19."""

        torch = self.torch
        cue_visual = np.asarray(cue_visual, dtype=np.float32)
        if cue_visual.ndim != 4 or cue_visual.shape[1:] != (
                3, NATIVE_PATCHES, NATIVE_VISUAL_DIM):
            raise ValueError("cue_visual must be Bx3x196x384")
        prop = _fixed_normalize_proprio(proprio)
        act = _fixed_normalize_actions(actions)
        if prop.shape[:2] != (len(cue_visual), 20) \
                or act.shape[:2] != (len(cue_visual), 19):
            raise ValueError("native rollout requires 20 observations/19 actions")
        requested = set(map(int, ages))
        if not requested or min(requested) < 1 or max(requested) > 16:
            raise ValueError("rollout ages must lie in [1,16]")
        with torch.inference_mode():
            visual = torch.from_numpy(cue_visual).to(self.device)
            prop_t = torch.from_numpy(prop[:, 1:4]).to(self.device)
            act_t = torch.from_numpy(act[:, 1:4]).to(self.device)
            context = self._compose(visual, prop_t, act_t)
            result: dict[int, np.ndarray] = {}
            for age in range(1, 17):
                predicted = self._predict(context)
                new = predicted[:, -1:, :, :]
                if age in requested:
                    result[age] = new[..., :NATIVE_VISUAL_DIM].squeeze(
                        1).float().cpu().numpy()
                if age < 16:
                    # The predicted frame is 3+age.  Native rollout replaces
                    # only that frame's attached action; proprio remains model-
                    # predicted exactly as in VWorldModel.replace_actions_from_z.
                    action_index = 3 + age
                    action_tensor = torch.from_numpy(
                        act[:, action_index:action_index + 1]).to(self.device)
                    action_emb = self.action_encoder(action_tensor)
                    action_tiled = action_emb.unsqueeze(2).expand(
                        -1, -1, NATIVE_PATCHES, -1)
                    new[..., -10:] = action_tiled
                    context = torch.cat((context[:, -2:], new), dim=1)
        _require(set(result) == requested, "native rollout omitted a requested age")
        return result

    def teacher_endpoint(self, context_visual: np.ndarray,
                         proprio: np.ndarray, actions: np.ndarray) -> np.ndarray:
        torch = self.torch
        visual = np.asarray(context_visual, dtype=np.float32)
        if visual.ndim != 4 or visual.shape[1:] != (
                3, NATIVE_PATCHES, NATIVE_VISUAL_DIM):
            raise ValueError("teacher context must be Bx3x196x384")
        prop = _fixed_normalize_proprio(proprio)[:, 16:19]
        act = _fixed_normalize_actions(actions)[:, 16:19]
        with torch.inference_mode():
            context = self._compose(
                torch.from_numpy(visual).to(self.device),
                torch.from_numpy(prop).to(self.device),
                torch.from_numpy(act).to(self.device),
            )
            predicted = self._predict(context)
            return predicted[:, -1, :, :NATIVE_VISUAL_DIM].float().cpu().numpy()

    def health(self, base_visual: np.ndarray, proprio: np.ndarray,
               true_actions: np.ndarray,
               shuffled_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                      np.ndarray]:
        """Return per-episode per-horizon MSE for native/copy/shuffled arms."""

        visual = np.asarray(base_visual, dtype=np.float32)
        if visual.ndim != 4 or visual.shape[1:] != (
                19, NATIVE_PATCHES, NATIVE_VISUAL_DIM):
            raise ValueError("health visual features must cover frames 1..19")
        true = self.rollout(
            visual[:, :3], proprio, true_actions, ages=tuple(range(1, 17)))
        shuffled = self.rollout(
            visual[:, :3], proprio, shuffled_actions,
            ages=tuple(range(1, 17)))
        true_stack = np.stack([true[age] for age in range(1, 17)], axis=1)
        shuffled_stack = np.stack(
            [shuffled[age] for age in range(1, 17)], axis=1)
        target = visual[:, 3:19]
        copy = np.repeat(visual[:, 2:3], 16, axis=1)
        axes = (2, 3)
        return (
            np.mean(np.square(true_stack - target), axis=axes, dtype=np.float64),
            np.mean(np.square(copy - target), axis=axes, dtype=np.float64),
            np.mean(np.square(shuffled_stack - target),
                    axis=axes, dtype=np.float64),
        )


def _task_selections(dataset: OfficialPushTHDF5,
                     cfg: Mapping[str, Any]) -> dict[str, list[PushTSequenceSelection]]:
    common = cfg["selection"]
    result: dict[str, list[PushTSequenceSelection]] = {}
    for task in cfg["tasks"]:
        selections = dataset.select_sequences(
            num_frames=int(cfg["sequence"]["num_frames"]),
            train_count=int(common["train_episodes"]),
            validation_count=int(common["validation_episodes"]),
            num_classes=int(task["classes"]),
            split_seed=int(common["split_seed"]),
            start_seed=int(common["start_seed"]),
            label_seed=int(task["label_seed"]),
        )
        result[task["key"]] = list(selections)
    reference = next(iter(result.values()))
    reference_identity = [(item.split, item.episode_index, item.local_start)
                          for item in reference]
    for key, values in result.items():
        identity = [(item.split, item.episode_index, item.local_start)
                    for item in values]
        _require(identity == reference_identity,
                 f"task {key} does not share the locked base bank")
    return result


def _split(values: Sequence[Any], selections: Sequence[PushTSequenceSelection]
           ) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values)
    mask = np.asarray([item.split == "train" for item in selections])
    return array[mask], array[~mask]


def _probe_record(train_x: np.ndarray, train_y: np.ndarray,
                  validation_x: np.ndarray, validation_y: np.ndarray, *,
                  classes: int, bootstrap_cfg: Mapping[str, Any],
                  seed_offset: int) -> dict[str, Any]:
    record = frozen_linear_probe(
        train_x, train_y, validation_x, validation_y, classes=classes,
        c=1.0, max_iter=3000)
    interval = bootstrap_accuracy_ci(
        record["prediction"], record["truth"],
        draws=int(bootstrap_cfg["draws"]),
        seed=int(bootstrap_cfg["seed"]) + seed_offset,
        confidence=float(bootstrap_cfg["confidence"]),
    )
    clean = strip_probe_arrays(record)
    clean["validation_accuracy_episode_bootstrap"] = interval
    return clean


def _encode_teacher_features(
        *, model: NativeDinoWM, dataset: OfficialPushTHDF5,
        selections: Sequence[PushTSequenceSelection], num_frames: int,
        batch_size: int, frame_batch_size: int,
        ) -> dict[str, np.ndarray]:
    observed: list[np.ndarray] = []
    endpoint: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    proprio: list[np.ndarray] = []
    for batch_selection in _chunks(selections, batch_size):
        native = _read_batch(dataset, batch_selection, num_frames)
        arrays = _stack_native(native)
        frames = arrays["frames"][:, 16:19]
        patch = model.encode_visual(
            frames.reshape(-1, 224, 224, 3), batch_size=frame_batch_size
        ).reshape(len(native), 3, NATIVE_PATCHES, NATIVE_VISUAL_DIM)
        predicted = model.teacher_endpoint(
            patch, arrays["proprio"], arrays["actions"])
        observed.append(temporal_spatial_pyramid_pool(patch))
        endpoint.append(spatial_pyramid_pool(predicted))
        actions.append(_fixed_normalize_actions(arrays["actions"])[:, 15:19]
                       .reshape(len(native), -1))
        proprio.append(_fixed_normalize_proprio(arrays["proprio"])[:, 16:19]
                        .reshape(len(native), -1))
    return {
        "observed_context": np.concatenate(observed),
        "predicted_endpoint": np.concatenate(endpoint),
        "action_shortcut": np.concatenate(actions),
        "proprio_shortcut": np.concatenate(proprio),
    }


def _encode_task_cues(
        *, model: NativeDinoWM, dataset: OfficialPushTHDF5,
        selections: Sequence[PushTSequenceSelection], task: Mapping[str, Any],
        sequence_cfg: Mapping[str, Any], batch_size: int,
        frame_batch_size: int,
        ) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for batch_selection in _chunks(selections, batch_size):
        native = _read_batch(
            dataset, batch_selection, int(sequence_cfg["num_frames"]))
        cue_frames = []
        for value, selection in zip(native, batch_selection):
            overlaid = render_single_overlay(
                value.frames, task["semantic_name"], int(selection.label),
                int(sequence_cfg["cue_start"]), int(sequence_cfg["cue_length"]))
            cue_frames.append(overlaid[1:4])
        cue = np.stack(cue_frames)
        patch = model.encode_visual(
            cue.reshape(-1, 224, 224, 3), batch_size=frame_batch_size
        ).reshape(len(native), 3, NATIVE_PATCHES, NATIVE_VISUAL_DIM)
        outputs.append(temporal_spatial_pyramid_pool(patch))
    return np.concatenate(outputs)


def _run_health(
        *, model: NativeDinoWM, dataset: OfficialPushTHDF5,
        validation_selections: Sequence[PushTSequenceSelection],
        cfg: Mapping[str, Any], num_frames: int,
        ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    health_cfg = cfg["rollout_health"]
    count = int(health_cfg["episodes"])
    selected = list(validation_selections[:count])
    _require(len(selected) == count, "insufficient validation health episodes")
    natives = _read_batch(dataset, selected, num_frames)
    permutation = np.random.default_rng(int(health_cfg["shuffle_seed"])).permutation(
        count)
    all_actions = np.stack([value.actions for value in natives])
    true_rows: list[np.ndarray] = []
    copy_rows: list[np.ndarray] = []
    shuffled_rows: list[np.ndarray] = []
    for indices in _chunks(list(range(count)), int(health_cfg["batch_size"])):
        batch = [natives[index] for index in indices]
        arrays = _stack_native(batch)
        frames = arrays["frames"][:, 1:20]
        visual = model.encode_visual(
            frames.reshape(-1, 224, 224, 3),
            batch_size=int(cfg["execution"]["frame_batch_size"]),
        ).reshape(len(batch), 19, NATIVE_PATCHES, NATIVE_VISUAL_DIM)
        shuffled_actions = all_actions[permutation[np.asarray(indices)]]
        true, copy, shuffled = model.health(
            visual, arrays["proprio"], arrays["actions"], shuffled_actions)
        true_rows.append(true)
        copy_rows.append(copy)
        shuffled_rows.append(shuffled)
    arrays = {
        "true_action_mse": np.concatenate(true_rows),
        "copy_last_mse": np.concatenate(copy_rows),
        "shuffled_action_mse": np.concatenate(shuffled_rows),
        "episode_index": np.asarray(
            [selection.episode_index for selection in selected], dtype=np.int64),
        "shuffle_permutation": permutation.astype(np.int64),
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
    summary["shuffle_seed"] = int(health_cfg["shuffle_seed"])
    summary["clean_base_frames"] = True
    summary["target"] = "frozen DINOv2 x_norm_patchtokens"
    return summary, arrays


def _run_assigned_rollouts(
        *, model: NativeDinoWM, dataset: OfficialPushTHDF5,
        selections: Sequence[PushTSequenceSelection], task: Mapping[str, Any],
        cfg: Mapping[str, Any],
        ) -> dict[int, np.ndarray]:
    ages = tuple(map(int, cfg["evaluation"]["evidence_ages"]))
    result: dict[int, list[np.ndarray]] = {age: [] for age in ages}
    sequence_cfg = cfg["sequence"]
    for batch_selection in _chunks(
            selections, int(cfg["execution"]["rollout_batch_size"])):
        native = _read_batch(
            dataset, batch_selection, int(sequence_cfg["num_frames"]))
        arrays = _stack_native(native)
        cue_frames = []
        for value, selection in zip(native, batch_selection):
            overlaid = render_single_overlay(
                value.frames, task["semantic_name"], int(selection.label),
                int(sequence_cfg["cue_start"]), int(sequence_cfg["cue_length"]))
            cue_frames.append(overlaid[1:4])
        patch = model.encode_visual(
            np.stack(cue_frames).reshape(-1, 224, 224, 3),
            batch_size=int(cfg["execution"]["frame_batch_size"]),
        ).reshape(len(native), 3, NATIVE_PATCHES, NATIVE_VISUAL_DIM)
        rollout = model.rollout(
            patch, arrays["proprio"], arrays["actions"], ages=ages)
        for age in ages:
            result[age].append(spatial_pyramid_pool(rollout[age]))
    return {age: np.concatenate(values) for age, values in result.items()}


def _run_counterfactual_separation(
        *, model: NativeDinoWM, dataset: OfficialPushTHDF5,
        validation_selections: Sequence[PushTSequenceSelection],
        task: Mapping[str, Any], cfg: Mapping[str, Any],
        ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    evaluation = cfg["evaluation"]
    sequence_cfg = cfg["sequence"]
    ages = tuple(map(int, evaluation["evidence_ages"]))
    count = int(evaluation["counterfactual_episodes"])
    selected = list(validation_selections[:count])
    classes = int(task["classes"])
    cue_rows: list[np.ndarray] = []
    age_rows: dict[int, list[np.ndarray]] = {age: [] for age in ages}
    for batch_selection in _chunks(
            selected, int(cfg["execution"]["counterfactual_episode_batch_size"])):
        native = _read_batch(
            dataset, batch_selection, int(sequence_cfg["num_frames"]))
        frames: list[np.ndarray] = []
        proprio: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        for value in native:
            for label in range(classes):
                frames.append(render_single_overlay(
                    value.frames, task["semantic_name"], label,
                    int(sequence_cfg["cue_start"]),
                    int(sequence_cfg["cue_length"]))[1:4])
                proprio.append(value.proprio)
                actions.append(value.actions)
        patch = model.encode_visual(
            np.stack(frames).reshape(-1, 224, 224, 3),
            batch_size=int(cfg["execution"]["frame_batch_size"]),
        ).reshape(len(native) * classes, 3,
                  NATIVE_PATCHES, NATIVE_VISUAL_DIM)
        # Counterfactual cue separation uses the same temporal mean as the
        # availability probe, but retains the full 196x384 patch tensor.
        cue_mean = patch.mean(axis=1).reshape(
            len(native), classes, NATIVE_PATCHES, NATIVE_VISUAL_DIM)
        cue_rows.append(pairwise_counterfactual_separation(cue_mean))
        rollout = model.rollout(
            patch, np.stack(proprio), np.stack(actions), ages=ages)
        for age in ages:
            shaped = rollout[age].reshape(
                len(native), classes, NATIVE_PATCHES, NATIVE_VISUAL_DIM)
            age_rows[age].append(pairwise_counterfactual_separation(shaped))
    arrays = {"cue_separation": np.concatenate(cue_rows)}
    summary: dict[str, Any] = {
        "episodes": count,
        "classes": classes,
        "unit": "within-episode all-label pair mean RMS in DINO feature units",
        "ages": {},
    }
    bootstrap_cfg = cfg["bootstrap"]
    for index, age in enumerate(ages):
        values = np.concatenate(age_rows[age])
        arrays[f"age_{age}_separation"] = values
        summary["ages"][str(age)] = paired_transport_summary(
            arrays["cue_separation"], values,
            draws=int(bootstrap_cfg["draws"]),
            seed=int(bootstrap_cfg["seed"]) + 800 + index,
            confidence=float(bootstrap_cfg["confidence"]),
        )
    return summary, arrays


def execute(config_path: Path) -> dict[str, Any]:
    cfg, lock = _load_locked_config(config_path)
    output = _resolve(ROOT, cfg["artifacts"]["root"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stop_reason: str | None = None
    provenance: dict[str, Any] = {
        "schema": "dinowm_native_pusht_audit_provenance_v1",
        "protocol_path": str(config_path),
        "protocol_sha256": lock["protocol_sha256"],
        "started_unix": started,
        "requested_missing_release": cfg["requested_missing_release"],
        "fallback_identity": cfg["fallback_checkpoint"]["display_name"],
        "fallback_is_not_dinowm_noprop": True,
        "paper_modified": False,
        "carrier_injection": False,
        "persistent_state_test": "not_applicable",
        "downstream_use_test": "not_applicable",
    }
    try:
        dataset_path = _resolve(ROOT, cfg["dataset"]["path"])
        checkpoint = _resolve(ROOT, cfg["fallback_checkpoint"]["weights_path"])
        hydra_yaml = _resolve(ROOT, cfg["fallback_checkpoint"]["config_path"])
        archive = _resolve(ROOT, cfg["fallback_checkpoint"]["archive_path"])
        dino_weights = _resolve(ROOT, cfg["dino_encoder"]["weights_path"])
        missing_status = _resolve(
            ROOT, cfg["requested_missing_release"]["status_receipt"])
        identities = {
            "dataset": _check_file_identity(dataset_path, cfg["dataset"]),
            "checkpoint_archive": _check_file_identity(
                archive, cfg["fallback_checkpoint"]["archive_identity"]),
            "checkpoint_weights": _check_file_identity(
                checkpoint, cfg["fallback_checkpoint"]["weights_identity"]),
            "checkpoint_config": _check_file_identity(
                hydra_yaml, cfg["fallback_checkpoint"]["config_identity"]),
            "dino_weights": _check_file_identity(
                dino_weights, cfg["dino_encoder"]["weights_identity"]),
            "missing_release_status": {
                "path": str(missing_status),
                "sha256": sha256_file(missing_status),
                "content": missing_status.read_text().splitlines(),
            },
        }
        _require(identities["missing_release_status"]["content"][0] == "404",
                 "missing dinowm_noprop receipt no longer records HTTP 404")
        source_records = {}
        for key, section in (
                ("dino_wm", cfg["fallback_source"]),
                ("dinov2", cfg["dino_encoder"])):
            repo = _resolve(ROOT, section["repo_path"])
            head = _git_output(repo, "rev-parse", "HEAD")
            status = _git_output(repo, "status", "--porcelain")
            archive_sha = _git_archive_sha256(repo)
            _require(head == section["revision"], f"{key} revision mismatch")
            _require(not status, f"{key} vendor checkout is dirty")
            _require(archive_sha == section["source_tree_sha256"],
                     f"{key} source tree SHA-256 mismatch")
            source_records[key] = {
                "path": str(repo), "revision": head, "clean": True,
                "git_archive_sha256": archive_sha,
            }
        provenance["identities"] = identities
        provenance["sources"] = source_records
        provenance["preprocessing_contract_sha256"] = _canonical_sha256(
            cfg["preprocessing"])
        provenance["environment"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "venv_path": str(Path(sys.executable).resolve()),
            "disk_before": shutil.disk_usage(ROOT)._asdict(),
        }
        _json_dump(output / "provenance.pre_model.json", provenance)

        dataset = OfficialPushTHDF5(
            dataset_path, expected_hdf5_sha256=cfg["dataset"]["sha256"])
        action_mean, action_std, action_count = dataset.raw_action_statistics(
            ddof=0)
        proprio_mean, proprio_std, proprio_count = _column_stats(
            dataset_path, "proprio", ddof=0)
        gate_cfg = cfg["same_bank_distribution_gate"]
        action_gate = _distribution_gate(
            action_mean, action_std, ACTION_MEAN, ACTION_STD,
            mean_shift_max=float(gate_cfg["mean_shift_std_units_max"]),
            std_ratio_min=float(gate_cfg["std_ratio_min"]),
            std_ratio_max=float(gate_cfg["std_ratio_max"]),
        )
        proprio_gate = _distribution_gate(
            proprio_mean, proprio_std, PROPRIO_MEAN, PROPRIO_STD,
            mean_shift_max=float(gate_cfg["mean_shift_std_units_max"]),
            std_ratio_min=float(gate_cfg["std_ratio_min"]),
            std_ratio_max=float(gate_cfg["std_ratio_max"]),
        )
        data_contract = {
            "schema": "dinowm_same_bank_contract_v1",
            "hdf5_schema": asdict(dataset.schema),
            "action_count": action_count.tolist(),
            "proprio_count": proprio_count.tolist(),
            "action_distribution": action_gate,
            "proprio_distribution": proprio_gate,
            "preprocessing_contract_sha256":
                provenance["preprocessing_contract_sha256"],
            "admitted": action_gate["admitted"] and proprio_gate["admitted"],
        }
        _json_dump(output / "data_contract.json", data_contract)
        if not data_contract["admitted"]:
            raise AuditStop("same-bank action/proprio distribution gate failed")

        selections_by_task = _task_selections(dataset, cfg)
        selection_payload = {
            key: [asdict(item) for item in values]
            for key, values in selections_by_task.items()
        }
        provenance["selection_sha256"] = _canonical_sha256(selection_payload)
        _json_dump(output / "selection.json", {
            "sha256": provenance["selection_sha256"],
            "tasks": selection_payload,
        })

        model = NativeDinoWM(
            checkpoint=checkpoint,
            dino_repo=_resolve(ROOT, cfg["dino_encoder"]["repo_path"]),
            torch_home=_resolve(ROOT, cfg["dino_encoder"]["torch_home"]),
            vendor_repo=_resolve(ROOT, cfg["fallback_source"]["repo_path"]),
            device=cfg["execution"]["device"],
        )
        import sklearn
        import torch
        provenance["environment"].update({
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(model.device),
            "gpu_name": torch.cuda.get_device_name(2),
            "gpu_capability": list(torch.cuda.get_device_capability(2)),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        })
        provenance["model_schema"] = model.schema
        _json_dump(output / "provenance.json", provenance)

        reference_task = cfg["tasks"][0]
        reference_selection = selections_by_task[reference_task["key"]]
        teacher = _encode_teacher_features(
            model=model, dataset=dataset, selections=reference_selection,
            num_frames=int(cfg["sequence"]["num_frames"]),
            batch_size=int(cfg["execution"]["teacher_batch_size"]),
            frame_batch_size=int(cfg["execution"]["frame_batch_size"]),
        )
        np.savez_compressed(output / "teacher_features.npz", **teacher)

        cue_features: dict[str, np.ndarray] = {}
        for task in cfg["tasks"]:
            key = task["key"]
            cue_features[key] = _encode_task_cues(
                model=model, dataset=dataset,
                selections=selections_by_task[key], task=task,
                sequence_cfg=cfg["sequence"],
                batch_size=int(cfg["execution"]["cue_batch_size"]),
                frame_batch_size=int(cfg["execution"]["frame_batch_size"]),
            )
            np.savez_compressed(
                output / "admission" / f"{key}_features.npz",
                cue=cue_features[key], **teacher,
                labels=np.asarray(
                    [item.label for item in selections_by_task[key]],
                    dtype=np.int64),
            )

        admissions: dict[str, Any] = {}
        shortcut_margin = float(cfg["admission"]["shortcut_margin_above_chance"])
        for task_index, task in enumerate(cfg["tasks"]):
            key = task["key"]
            selections = selections_by_task[key]
            labels = np.asarray([item.label for item in selections], dtype=np.int64)
            train_y, validation_y = _split(labels, selections)
            probes: dict[str, Any] = {}
            sources = {"cue": cue_features[key], **teacher}
            for probe_index, (name, values) in enumerate(sources.items()):
                train_x, validation_x = _split(values, selections)
                probes[name] = _probe_record(
                    train_x, train_y, validation_x, validation_y,
                    classes=int(task["classes"]),
                    bootstrap_cfg=cfg["bootstrap"],
                    seed_offset=100 * task_index + probe_index,
                )
            chance = 1.0 / int(task["classes"])
            ceiling = chance + shortcut_margin
            gates = {
                "cue_availability": {
                    "value": probes["cue"]["balanced_accuracy"],
                    "threshold": float(cfg["admission"]["cue_accuracy_min"]),
                    "direction": ">=",
                    "pass": probes["cue"]["balanced_accuracy"] >=
                            float(cfg["admission"]["cue_accuracy_min"]),
                }
            }
            for name in (
                    "observed_context", "predicted_endpoint",
                    "action_shortcut", "proprio_shortcut"):
                gates[f"{name}_no_shortcut"] = {
                    "value": probes[name]["balanced_accuracy"],
                    "threshold": ceiling,
                    "direction": "<=",
                    "pass": probes[name]["balanced_accuracy"] <= ceiling,
                }
            admissions[key] = {
                "schema": "dinowm_native_pusht_admission_v1",
                "task": task,
                "chance": chance,
                "shortcut_ceiling": ceiling,
                "probes": probes,
                "gates": gates,
                "admitted": all(gate["pass"] for gate in gates.values()),
            }
            _json_dump(output / "admission" / f"{key}.json", admissions[key])

        validation_reference = [item for item in reference_selection
                                if item.split == "validation"]
        health, health_arrays = _run_health(
            model=model, dataset=dataset,
            validation_selections=validation_reference, cfg=cfg,
            num_frames=int(cfg["sequence"]["num_frames"]),
        )
        np.savez_compressed(output / "rollout_health.npz", **health_arrays)
        _json_dump(output / "rollout_health.json", health)

        failed_admissions = [key for key, value in admissions.items()
                             if not value["admitted"]]
        if failed_admissions:
            raise AuditStop(
                "task admission failed: " + ", ".join(failed_admissions))
        if not health["admitted"]:
            raise AuditStop("same-bank native rollout-health gate failed")

        task_results: dict[str, Any] = {}
        for task_index, task in enumerate(cfg["tasks"]):
            key = task["key"]
            selections = selections_by_task[key]
            labels = np.asarray([item.label for item in selections], dtype=np.int64)
            train_y, validation_y = _split(labels, selections)
            rollout = _run_assigned_rollouts(
                model=model, dataset=dataset, selections=selections,
                task=task, cfg=cfg)
            age_results: dict[str, Any] = {}
            for age_index, age in enumerate(cfg["evaluation"]["evidence_ages"]):
                age = int(age)
                train_x, validation_x = _split(rollout[age], selections)
                probe = _probe_record(
                    train_x, train_y, validation_x, validation_y,
                    classes=int(task["classes"]),
                    bootstrap_cfg=cfg["bootstrap"],
                    seed_offset=400 + task_index * 20 + age_index,
                )
                chance = 1.0 / int(task["classes"])
                probe["chance"] = chance
                probe["chance_normalized_accuracy"] = (
                    (probe["balanced_accuracy"] - chance) / (1.0 - chance))
                probe["endpoint_frame"] = endpoint_frame_for_age(
                    last_cue_frame=int(cfg["sequence"]["cue_start"])
                    + int(cfg["sequence"]["cue_length"]) - 1,
                    age=age)
                age_results[str(age)] = probe
            validation_selections = [item for item in selections
                                     if item.split == "validation"]
            counterfactual, counterfactual_arrays = _run_counterfactual_separation(
                model=model, dataset=dataset,
                validation_selections=validation_selections,
                task=task, cfg=cfg)
            np.savez_compressed(
                output / "results" / f"{key}.npz",
                labels=labels,
                **{f"age_{age}_features": values
                   for age, values in rollout.items()},
                **counterfactual_arrays,
            )
            task_results[key] = {
                "task": task,
                "open_loop_decodability": age_results,
                "paired_counterfactual_separation": counterfactual,
                "claim_scope": (
                    "cue-anchored native imagination transport; not persistent "
                    "real-observation memory"),
                "carrier_injection": False,
                "persistent_state": "not_applicable",
                "downstream_use": "not_applicable",
            }
            _json_dump(output / "results" / f"{key}.json", task_results[key])

        summary = {
            "schema": "dinowm_native_pusht_portability_audit_v1",
            "status": "complete",
            "host": cfg["fallback_checkpoint"]["display_name"],
            "host_is_requested_dinowm_noprop": False,
            "missing_requested_release_http_status": 404,
            "data_contract": data_contract,
            "admissions": admissions,
            "rollout_health": health,
            "tasks": task_results,
            "claim_ledger": {
                "cue_availability": "tested",
                "teacher_forced_endpoint_exposure": "tested",
                "native_open_loop_decodability": "tested",
                "paired_counterfactual_feature_separation": "tested",
                "copy_last_and_shuffled_action_health_through_16": "tested",
                "persistent_state_retention": "not_applicable_no_persistent_state",
                "downstream_use": "not_applicable_no_carrier_injection",
            },
            "elapsed_seconds": time.time() - started,
        }
        _json_dump(output / "summary.json", summary)
        provenance["completed_unix"] = time.time()
        provenance["elapsed_seconds"] = provenance["completed_unix"] - started
        provenance["environment"]["disk_after"] = shutil.disk_usage(ROOT)._asdict()
        provenance["artifact_sha256"] = {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name not in {
                "provenance.json", "provenance.pre_model.json"}
        }
        _json_dump(output / "provenance.json", provenance)
        return summary
    except AuditStop as exc:
        stop_reason = str(exc)
        receipt = {
            "schema": "dinowm_native_pusht_audit_stop_v1",
            "status": "stopped_fail_closed",
            "reason": stop_reason,
            "no_post_hoc_adaptation": True,
            "no_downstream_metric_after_failed_gate": True,
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
        default=ROOT / "configs/dinowm_native_pusht_audit_v1.yaml")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("refusing metric-bearing execution without --execute")
    result = execute(args.config.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
