#!/usr/bin/env python3
"""Run the locked DINO-WM PointMaze persistent-memory audit on GPU 2.

Stages are intentionally irreversible and ordered: ``--smoke`` performs only
shape/runtime checks, ``--seal`` freezes protocol and source hashes,
``--prepare`` builds the frozen feature bank and resolves every pre-carrier
admission/controller gate, and ``--formal`` trains/evaluates the complete
locked carrier grid.  Formal stages require ``CUDA_VISIBLE_DEVICES=2`` and
never modify ``paper_a``.
"""

from __future__ import annotations

import argparse
import csv
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
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import (  # noqa: E402
    make_frozen_carrier,
    parameter_report,
)
from lewm.official_tasks.dinowm_native_audit import (  # noqa: E402
    spatial_pyramid_pool,
)
from lewm.official_tasks.dinowm_pointmaze import (  # noqa: E402
    GOAL_WAYPOINTS,
    CurrentMujocoPointMaze,
    MazeSelection,
    crossed_execution_arrays,
    endpoint_frame,
    execute_released_waypoint,
    predictor_context_for_endpoint,
    render_transient_goal_cue,
    select_native_windows,
    verify_cue_only_counterfactual,
)
from lewm.official_tasks.dinowm_spatial_carrier import (  # noqa: E402
    balanced_accuracy_from_predictions,
    spatial_carrier_forward,
)


DEFAULT_CONFIG = ROOT / "configs/dinowm_pointmaze_wave3.yaml"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True).strip()


def git_archive_sha256(repo: Path) -> str:
    process = subprocess.Popen(
        ["git", "-C", str(repo), "archive", "--format=tar", "HEAD"],
        stdout=subprocess.PIPE)
    digest = hashlib.sha256()
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    require(process.wait() == 0, f"git archive failed: {repo}")
    return digest.hexdigest()


def check_identity(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    require(path.is_file(), f"missing pinned file: {path}")
    size, digest = path.stat().st_size, sha256_file(path)
    require(size == int(identity["size"]), f"size mismatch: {path}")
    require(digest == identity["sha256"], f"SHA-256 mismatch: {path}")
    display = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return {"path": display, "size": size,
            "sha256": digest}


def verify_extracted_contents(cfg: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = resolve(cfg["dataset"]["extracted_manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    root = resolve(cfg["dataset"]["root"])
    require(manifest.get("schema") == "official_dinowm_pointmaze_extraction_v1"
            and Path(manifest["root"]) == root,
            "extracted PointMaze manifest root/schema changed")
    require(manifest.get("file_count") == 2003
            and len(manifest.get("files", [])) == 2003,
            "extracted PointMaze file count changed")
    total = 0
    for record in manifest["files"]:
        path = root / record["path"]
        require(path.is_file() and path.stat().st_size == record["size"]
                and sha256_file(path) == record["sha256"],
                f"extracted PointMaze file changed: {path}")
        total += int(record["size"])
    require(total == int(manifest["total_bytes"]),
            "extracted PointMaze byte total changed")
    return {"root": str(root), "file_count": 2003,
            "total_bytes": total, "all_file_sha256_verified": True}


def load_config(path: Path, *, locked: bool) \
        -> tuple[dict[str, Any], dict[str, Any] | None]:
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw)
    require(isinstance(cfg, dict), "Wave 3 config must be a mapping")
    if not locked:
        return cfg, None
    lock_path = path.with_suffix(".lock.json")
    require(lock_path.is_file(), "Wave 3 lock is missing")
    lock = json.loads(lock_path.read_text())
    require(lock.get("locked_before_semantic_metrics") is True,
            "Wave 3 lock is not formal")
    require(lock["protocol_sha256"] == hashlib.sha256(raw).hexdigest(),
            "Wave 3 protocol changed after sealing")
    for relative, expected in lock["source_sha256"].items():
        path_value = resolve(relative)
        require(path_value.is_file() and sha256_file(path_value) == expected,
                f"locked source changed: {relative}")
    return cfg, lock


def verify_pins(cfg: Mapping[str, Any]) -> dict[str, Any]:
    identities = {
        "archive": check_identity(resolve(cfg["dataset"]["archive_path"]),
                                  cfg["dataset"]["archive_identity"]),
        "extracted_manifest": check_identity(
            resolve(cfg["dataset"]["extracted_manifest_path"]),
            cfg["dataset"]["extracted_manifest_identity"]),
        "checkpoint": check_identity(resolve(cfg["checkpoint"]["weights_path"]),
                                     cfg["checkpoint"]["weights_identity"]),
        "checkpoint_config": check_identity(
            resolve(cfg["checkpoint"]["config_path"]),
            cfg["checkpoint"]["config_identity"]),
        "dinov2_weights": check_identity(
            resolve(cfg["source"]["dinov2"]["weights_path"]),
            cfg["source"]["dinov2"]["weights_identity"]),
        "dependencies": check_identity(
            resolve(cfg["execution"]["dependency_manifest_path"]),
            cfg["execution"]["dependency_manifest_identity"]),
        "controller_development": check_identity(
            resolve(cfg["external_use"]["prelock_controller_development"][
                "receipt"]),
            cfg["external_use"]["prelock_controller_development"][
                "receipt_identity"]),
    }
    sources: dict[str, Any] = {}
    for key in ("dino_wm", "dinov2"):
        record = cfg["source"][key]
        repo = resolve(record["repo_path"])
        revision = git_output(repo, "rev-parse", "HEAD")
        status = git_output(repo, "status", "--porcelain")
        archive = git_archive_sha256(repo)
        require(revision == record["revision"] and not status
                and archive == record["git_archive_sha256"],
                f"pinned {key} repository changed")
        sources[key] = {"revision": revision, "clean": True,
                        "git_archive_sha256": archive}
    vendor = resolve(cfg["source"]["dino_wm"]["repo_path"])
    file_pins = {}
    for relative, expected in cfg["source"]["dino_wm"][
            "pointmaze_files"].items():
        path = vendor / relative
        actual = sha256_file(path)
        require(actual == expected, f"released PointMaze source changed: {relative}")
        file_pins[relative] = actual
    sources["dino_wm"]["pointmaze_files"] = file_pins
    return {"identities": identities, "sources": sources,
            "extracted_contents": verify_extracted_contents(cfg)}


def configure_cuda(cfg: Mapping[str, Any], seed: int) -> torch.device:
    execution = cfg["execution"]
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "2",
            "Wave 3 requires CUDA_VISIBLE_DEVICES=2")
    require(int(execution["physical_gpu"]) == 2
            and execution["required_cuda_visible_devices"] == "2"
            and bool(execution["never_gpu3"]),
            "Wave 3 protocol is not pinned to physical GPU 2")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Wave 3 must see exactly one CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    return device


def module_digest(modules: Mapping[str, torch.nn.Module]) -> str:
    digest = hashlib.sha256()
    for module_name, module in sorted(modules.items()):
        for name, value in sorted(module.state_dict().items()):
            digest.update(module_name.encode())
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenPointMazeHost:
    """Frozen official PointMaze checkpoint under the native token contract."""

    def __init__(self, cfg: Mapping[str, Any], *, load_encoder: bool) -> None:
        self.cfg = cfg
        self.device = configure_cuda(cfg, 9070)
        vendor = resolve(cfg["source"]["dino_wm"]["repo_path"])
        dino_repo = resolve(cfg["source"]["dinov2"]["repo_path"])
        os.environ["TORCH_HOME"] = str(resolve(
            cfg["source"]["dinov2"]["torch_home"]))
        sys.path.insert(0, str(vendor))
        payload = torch.load(resolve(cfg["checkpoint"]["weights_path"]),
                             map_location="cpu", weights_only=False)
        require(set(payload) == {
            "epoch", "predictor", "predictor_optimizer", "decoder",
            "decoder_optimizer", "action_encoder", "proprio_encoder",
        }, "released checkpoint schema changed")
        self.epoch = int(payload["epoch"])
        require(self.epoch == int(cfg["checkpoint"]["expected_epoch_field"]),
                "released checkpoint epoch changed")
        self.predictor = payload["predictor"].eval().to(self.device)
        self.action_encoder = payload["action_encoder"].eval().to(self.device)
        self.proprio_encoder = payload["proprio_encoder"].eval().to(self.device)
        del payload
        moved = 0
        for module in self.predictor.modules():
            bias = getattr(module, "bias", None)
            if torch.is_tensor(bias) and bias.ndim == 4:
                module.bias = bias.to(self.device)
                moved += 1
        require(moved == 6, f"expected six attention masks, moved {moved}")
        self.encoder = None
        if load_encoder:
            self.encoder = torch.hub.load(
                str(dino_repo), "dinov2_vits14", source="local",
                pretrained=True).eval().to(self.device)
        for module in self.modules.values():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.verify_schema(load_encoder=load_encoder)

    @property
    def modules(self) -> dict[str, torch.nn.Module]:
        result = {"predictor": self.predictor,
                  "action_encoder": self.action_encoder,
                  "proprio_encoder": self.proprio_encoder}
        if self.encoder is not None:
            result["dinov2_encoder"] = self.encoder
        return result

    def verify_schema(self, *, load_encoder: bool) -> None:
        require(tuple(self.predictor.pos_embedding.shape) == (1, 588, 404),
                "PointMaze predictor token shape changed")
        require(tuple(self.action_encoder.patch_embed.weight.shape)
                == (10, 10, 1), "PointMaze action encoder changed")
        require(tuple(self.proprio_encoder.patch_embed.weight.shape)
                == (10, 4, 1), "PointMaze proprio encoder changed")
        if load_encoder:
            require(self.encoder is not None and self.encoder.num_features == 384
                    and self.encoder.patch_size == 14,
                    "DINOv2 encoder contract changed")
        require(all(not p.requires_grad for m in self.modules.values()
                    for p in m.parameters()), "host is not frozen")
        require(torch.cuda.current_device() == 0,
                "logical GPU is not cuda:0 under GPU-2 isolation")

    def digest(self) -> str:
        return module_digest(self.modules)

    @torch.no_grad()
    def encode_visual(self, frames: np.ndarray, *, batch_size: int) -> np.ndarray:
        require(self.encoder is not None, "DINO encoder is not loaded")
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        values = np.asarray(frames)
        require(values.ndim == 4 and values.shape[1:] == (224, 224, 3)
                and values.dtype == np.uint8,
                "native frames must be uint8 Bx224x224x3")
        outputs = []
        for start in range(0, len(values), int(batch_size)):
            rows = values[start:start + int(batch_size)]
            tensor = torch.from_numpy(rows.copy()).to(self.device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            tensor = tensor.sub_(0.5).div_(0.5)
            tensor = TF.resize(tensor, [196, 196],
                               interpolation=InterpolationMode.BILINEAR,
                               antialias=True)
            patches = self.encoder.forward_features(tensor)[
                "x_norm_patchtokens"]
            require(tuple(patches.shape[1:]) == (196, 384),
                    "DINOv2 patch shape changed")
            outputs.append(patches.float().cpu().numpy())
        return np.concatenate(outputs)

    def compose(self, visual: torch.Tensor, proprio: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        require(visual.ndim == 4 and visual.shape[2:] == (196, 384),
                "visual context violates native shape")
        require(proprio.shape == (*visual.shape[:2], 4),
                "proprio context violates native shape")
        require(actions.shape == (*visual.shape[:2], 10),
                "action context violates native shape")
        prop = self.proprio_encoder(proprio).unsqueeze(2).expand(
            -1, -1, 196, -1)
        action = self.action_encoder(actions).unsqueeze(2).expand(
            -1, -1, 196, -1)
        return torch.cat((visual, prop, action), dim=-1)

    def predict(self, visual: torch.Tensor, proprio: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        context = self.compose(visual, proprio, actions)
        batch, steps, patches, dim = context.shape
        require((steps, patches, dim) == (3, 196, 404),
                "native predictor requires 3x196x404")
        return self.predictor(context.reshape(
            batch, steps * patches, dim)).reshape(batch, steps, patches, dim)

    @torch.no_grad()
    def target_nonaction(self, visual: torch.Tensor,
                         proprio: torch.Tensor) -> torch.Tensor:
        prop = self.proprio_encoder(proprio).unsqueeze(2).expand(
            -1, -1, 196, -1)
        return torch.cat((visual, prop), dim=-1)


class NativePointMazeData:
    """Authenticated reader implementing the released PointMaze transforms."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.root = resolve(cfg["dataset"]["root"])
        self.states = torch.load(self.root / "states.pth", map_location="cpu",
                                 weights_only=True).float()
        self.raw_actions = torch.load(
            self.root / "actions.pth", map_location="cpu",
            weights_only=True).float()
        self.lengths = torch.load(
            self.root / "seq_lengths.pth", map_location="cpu",
            weights_only=True).long()
        require(self.states.ndim == 3 and self.states.shape[-1] == 4,
                "native PointMaze state tensor changed")
        require(self.raw_actions.shape[:2] == self.states.shape[:2]
                and self.raw_actions.shape[-1] == 2,
                "native PointMaze action tensor changed")
        require(self.lengths.shape == (len(self.states),),
                "native PointMaze length tensor changed")
        valid_actions, valid_states = [], []
        for episode, raw_length in enumerate(self.lengths.tolist()):
            length = int(raw_length)
            require(0 < length <= self.states.shape[1],
                    "native sequence length is invalid")
            valid_actions.append(self.raw_actions[episode, :length])
            valid_states.append(self.states[episode, :length])
        actions = torch.cat(valid_actions)
        states = torch.cat(valid_states)
        # Released PointMazeDataset uses torch.std's unbiased default.
        self.action_mean = actions.mean(0)
        self.action_std = actions.std(0, unbiased=True)
        self.proprio_mean = states.mean(0)
        self.proprio_std = states.std(0, unbiased=True)
        require(torch.all(self.action_std > 0).item()
                and torch.all(self.proprio_std > 0).item(),
                "native normalization has zero variance")

    def official_split(self) -> tuple[list[int], list[int]]:
        fraction = float(self.cfg["dataset"]["native_split_fraction"])
        generator = torch.Generator().manual_seed(
            int(self.cfg["dataset"]["native_split_seed"]))
        order = torch.randperm(len(self.lengths), generator=generator).tolist()
        count = int(fraction * len(order))
        return order[:count], order[count:]

    def selections(self) -> list[MazeSelection]:
        train, validation = self.official_split()
        sequence, dataset = self.cfg["sequence"], self.cfg["dataset"]
        return select_native_windows(
            self.lengths.tolist(), train_episodes=train,
            validation_episodes=validation,
            train_count=int(dataset["train_base_windows"]),
            validation_count=int(dataset["validation_base_windows"]),
            num_frames=int(sequence["num_frames"]),
            frame_skip=int(sequence["native_frame_skip"]),
            seed=int(dataset["selection_seed"]))

    def read(self, selection: MazeSelection) -> dict[str, np.ndarray]:
        skip = int(self.cfg["sequence"]["native_frame_skip"])
        frames_count = int(self.cfg["sequence"]["num_frames"])
        start = int(selection.local_start)
        indices = start + np.arange(frames_count) * skip
        episode = int(selection.episode_index)
        image_path = self.root / "obses" / f"episode_{episode:03d}.pth"
        images = torch.load(image_path, map_location="cpu", weights_only=True)
        frames = images[torch.from_numpy(indices)].cpu().numpy()
        require(frames.shape == (20, 224, 224, 3)
                and frames.dtype == np.uint8,
                f"native frame tensor changed: {image_path}")
        state = self.states[episode, torch.from_numpy(indices)]
        proprio = ((state - self.proprio_mean) / self.proprio_std).numpy()
        controls = self.raw_actions[episode, start:start + 19 * skip]
        controls = (controls - self.action_mean) / self.action_std
        actions = controls.reshape(19, skip * 2).numpy()
        require(actions.shape == (19, 10), "action block shape changed")
        return {"frames": frames, "state": state.numpy(),
                "proprio": proprio.astype(np.float32),
                "actions": actions.astype(np.float32)}

    def provenance(self) -> dict[str, Any]:
        train, validation = self.official_split()
        return {
            "episodes": len(self.lengths),
            "padded_steps": int(self.states.shape[1]),
            "sequence_length_min": int(self.lengths.min()),
            "sequence_length_max": int(self.lengths.max()),
            "sequence_length_sum": int(self.lengths.sum()),
            "native_train_episodes": train,
            "native_validation_episodes": validation,
            "action_mean": self.action_mean.tolist(),
            "action_std_unbiased": self.action_std.tolist(),
            "proprio_mean": self.proprio_mean.tolist(),
            "proprio_std_unbiased": self.proprio_std.tolist(),
        }


def selection_digest(values: Sequence[MazeSelection]) -> str:
    payload = [{"split": value.split,
                "episode_index": value.episode_index,
                "local_start": value.local_start} for value in values]
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expanded_labels(base_count: int) -> np.ndarray:
    return np.tile(np.arange(4, dtype=np.int64), int(base_count))


def fit_classifier(train_x: np.ndarray, train_y: np.ndarray,
                   validation_x: np.ndarray) -> np.ndarray:
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000,
                           random_state=0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(train_x, train_y)
    return classifier.predict(validation_x).astype(np.int64)


def classification_record(prediction: np.ndarray, truth: np.ndarray) \
        -> dict[str, Any]:
    matrix = confusion_matrix(truth, prediction, labels=np.arange(4))
    recall = np.diag(matrix) / np.maximum(matrix.sum(1), 1)
    return {
        "balanced_accuracy": balanced_accuracy_from_predictions(
            prediction, truth, 4),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": matrix.tolist(),
        "count": len(truth),
    }


def seal_protocol(config_path: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    lock_path = config_path.with_suffix(".lock.json")
    require(not lock_path.exists(), "refusing to overwrite Wave 3 lock")
    require(cfg.get("protocol_status") == "locked_before_formal_metrics",
            "set protocol_status to locked_before_formal_metrics before seal")
    smoke = resolve(cfg["artifacts"]["root"]) / cfg["artifacts"]["smoke"] \
        / "receipt.json"
    require(smoke.is_file(), "successful Wave 3 smoke is required")
    smoke_value = json.loads(smoke.read_text())
    require(smoke_value.get("status") == "passed_no_semantic_metric"
            and smoke_value.get("semantic_readout_fitted") is False,
            "Wave 3 smoke receipt is not admissible")
    formal = resolve(cfg["artifacts"]["root"]) / "formal"
    require(not formal.exists(), "formal directory already exists")
    pins = verify_pins(cfg)
    development_path = resolve(cfg["external_use"][
        "prelock_controller_development"]["receipt"])
    development = json.loads(development_path.read_text())
    require(development.get("status") == "passed"
            and development.get("validation_opened") is False
            and development.get("carrier_metric_computed") is False
            and development.get("chosen_horizon")
            == int(cfg["external_use"]["execution_horizon"]),
            "pre-lock controller selection does not match the protocol")
    sources = {}
    for relative in cfg["lock"]["source_paths"]:
        path = resolve(relative)
        require(path.is_file(), f"cannot seal missing source {relative}")
        sources[str(relative)] = sha256_file(path)
    value = {
        "schema": "dinowm_pointmaze_wave3_lock_v1",
        "locked_before_semantic_metrics": True,
        "protocol_sha256": sha256_file(config_path),
        "source_sha256": sources,
        "smoke_receipt_sha256": sha256_file(smoke),
        "controller_development_receipt_sha256": sha256_file(development_path),
        "parameter_matching": parameter_report(384, 10),
        "grid": {"tasks": 1, "arms": 5, "seeds": 5, "cells": 25},
        "pins": pins,
        "sealed_unix": time.time(),
    }
    atomic_json(lock_path, value)
    return value


def run_smoke(cfg: Mapping[str, Any]) -> dict[str, Any]:
    root = resolve(cfg["artifacts"]["root"]) / cfg["artifacts"]["smoke"]
    require(not root.exists(), "refusing to overwrite Wave 3 smoke")
    root.mkdir(parents=True)
    started = time.time()
    try:
        pins = verify_pins(cfg)
        dataset = NativePointMazeData(cfg)
        selections = dataset.selections()
        native = dataset.read(selections[0])
        variants = np.stack([
            render_transient_goal_cue(native["frames"], label)
            for label in range(4)])
        cue_audit = verify_cue_only_counterfactual(native["frames"], variants)
        require(cue_audit["passed"], "cue-only intervention smoke failed")
        host = FrozenPointMazeHost(cfg, load_encoder=True)
        host_before = host.digest()
        encoded = host.encode_visual(
            variants[:, :4].reshape(-1, 224, 224, 3), batch_size=16).reshape(
                4, 4, 196, 384)
        z = torch.from_numpy(encoded[:1]).to(host.device)
        actions = torch.from_numpy(native["actions"][:3][None]).to(host.device)
        proprio = torch.from_numpy(native["proprio"][:3][None]).to(host.device)
        cells = {}
        for arm in cfg["training"]["arms"]:
            carrier = make_frozen_carrier(arm, 384, 10).to(host.device)
            output = spatial_carrier_forward(carrier, z, actions)
            zero_error = float((output.fused_visual - z).abs().max().cpu())
            require(zero_error == 0.0, f"{arm} zero-init identity failed")
            if carrier.parameter_count():
                prediction = host.predict(
                    output.fused_visual[:, :3], proprio,
                    torch.from_numpy(native["actions"][:3][None]).to(
                        host.device))
                target = host.target_nonaction(
                    z[:, 1:4], torch.from_numpy(
                        native["proprio"][1:4][None]).to(host.device))
                loss = F.mse_loss(prediction[..., :394].float(), target.float())
                loss.backward()
                require(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in carrier.parameters()),
                        f"{arm} smoke gradient failed")
                loss_value = float(loss.detach())
            else:
                loss_value = None
            cells[arm] = {"parameters": carrier.parameter_count(),
                          "zero_init_max_abs": zero_error,
                          "one_step_loss": loss_value}
            del carrier
        simulator = CurrentMujocoPointMaze(
            resolve(cfg["source"]["dino_wm"]["repo_path"]))
        initial = native["state"][18]
        reset = simulator.reset(initial)
        first = simulator.step(np.zeros(2))
        replay = simulator.reset(initial)
        require(np.array_equal(reset, replay) and np.isfinite(first).all(),
                "current-MuJoCo smoke replay failed")
        host_after = host.digest()
        require(host_before == host_after, "smoke mutated frozen host")
        value = {
            "schema": "dinowm_pointmaze_wave3_smoke_v1",
            "status": "passed_no_semantic_metric",
            "semantic_readout_fitted": False,
            "task_accuracy_computed": False,
            "physical_gpu": 2,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "pins": pins,
            "dataset": dataset.provenance(),
            "selection_count": len(selections),
            "selection_sha256": selection_digest(selections),
            "cue_audit": cue_audit,
            "host_digest_before": host_before,
            "host_digest_after": host_after,
            "host_unchanged": True,
            "current_mujoco_version": simulator.mujoco.__version__,
            "released_xml_sha256": simulator.xml_sha256,
            "cells": cells,
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(root / "receipt.json", value)
        return value
    except Exception as error:
        atomic_json(root / "stop_receipt.json", {
            "schema": "dinowm_pointmaze_wave3_smoke_stop_v1",
            "status": "failed_preserved", "reason": repr(error),
            "semantic_readout_fitted": False,
            "elapsed_seconds": time.time() - started})
        raise


class FeatureBank:
    """Counterfactual expansion over a compact base/cue feature cache."""

    def __init__(self, cfg: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.root = resolve(cfg["artifacts"]["root"]) / "cache"
        manifest_path = self.root / "manifest.json"
        require(manifest_path.is_file(), "Wave 3 cache manifest is missing")
        self.manifest = json.loads(manifest_path.read_text())
        require(self.manifest.get("protocol_sha256") == lock["protocol_sha256"],
                "Wave 3 cache belongs to another lock")
        for record in self.manifest["artifacts"].values():
            path = resolve(record["path"])
            require(path.is_file() and path.stat().st_size == record["size"]
                    and sha256_file(path) == record["sha256"],
                    f"Wave 3 cache identity failed: {path}")
        self.base_visual = np.load(self.root / "base_visual.npy", mmap_mode="r")
        self.cue_visual = np.load(self.root / "cue_visual.npy", mmap_mode="r")
        metadata = np.load(self.root / "metadata.npz")
        self.actions = np.asarray(metadata["actions"], dtype=np.float32)
        self.proprio = np.asarray(metadata["proprio"], dtype=np.float32)
        self.states = np.asarray(metadata["states"], dtype=np.float32)
        self.split = np.asarray(metadata["split"], dtype=np.uint8)
        self.episode = np.asarray(metadata["episode_index"], dtype=np.int64)
        self.local_start = np.asarray(metadata["local_start"], dtype=np.int64)
        base_count = int(cfg["dataset"]["train_base_windows"]) + int(
            cfg["dataset"]["validation_base_windows"])
        require(self.base_visual.shape == (base_count, 20, 196, 384),
                "base visual cache shape changed")
        require(self.cue_visual.shape == (base_count, 4, 3, 196, 384),
                "cue visual cache shape changed")
        require(self.actions.shape == (base_count, 19, 10)
                and self.proprio.shape == (base_count, 20, 4)
                and self.states.shape == (base_count, 20, 4),
                "metadata cache shape changed")
        require(np.count_nonzero(self.split == 0)
                == int(cfg["dataset"]["train_base_windows"])
                and np.count_nonzero(self.split == 1)
                == int(cfg["dataset"]["validation_base_windows"]),
                "cache split count changed")

    def base_indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == {"train": 0, "validation": 1}[split])

    def expanded_indices(self, split: str) -> np.ndarray:
        bases = self.base_indices(split)
        return (bases[:, None] * 4 + np.arange(4)[None]).reshape(-1)

    @staticmethod
    def decode_expanded(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(indices, dtype=np.int64)
        return values // 4, values % 4

    def visual(self, expanded: np.ndarray) -> np.ndarray:
        bases, labels = self.decode_expanded(expanded)
        values = np.asarray(self.base_visual[bases], dtype=np.float32).copy()
        values[:, 1:4] = np.asarray(
            self.cue_visual[bases, labels], dtype=np.float32)
        return values

    def expanded_metadata(self, expanded: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bases, labels = self.decode_expanded(expanded)
        return bases, labels, self.episode[bases]


def _cache_artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "size": path.stat().st_size,
            "sha256": sha256_file(path)}


def _pooled(values: np.ndarray) -> np.ndarray:
    return spatial_pyramid_pool(np.asarray(values, dtype=np.float32))


def resolve_admission(bank: FeatureBank, cfg: Mapping[str, Any],
                      host_digest_before: str,
                      host_digest_after: str) -> dict[str, Any]:
    train_base = bank.base_indices("train")
    validation_base = bank.base_indices("validation")
    train_y = expanded_labels(len(train_base))
    validation_y = expanded_labels(len(validation_base))
    # Cue features retain all spatial patches at the last visible cue frame.
    cue_train = _pooled(np.asarray(bank.cue_visual[train_base, :, 2]).reshape(
        -1, 196, 384))
    cue_validation = _pooled(np.asarray(
        bank.cue_visual[validation_base, :, 2]).reshape(-1, 196, 384))
    cue_prediction = fit_classifier(cue_train, train_y, cue_validation)
    cue_record = classification_record(cue_prediction, validation_y)
    cue_cfg = cfg["admission"]["cue_encoding"]
    cue_record["thresholds"] = {
        "balanced_accuracy_minimum": cue_cfg["minimum"],
        "per_class_recall_minimum": cue_cfg["minimum_per_class_recall"],
    }
    cue_record["pass"] = bool(
        cue_record["balanced_accuracy"] >= float(cue_cfg["minimum"])
        and min(cue_record["per_class_recall"])
        >= float(cue_cfg["minimum_per_class_recall"]))

    shortcut_cfg = cfg["admission"]["shortcuts"]
    visual_limit = float(shortcut_cfg[
        "no_cue_visual_endpoint_balanced_accuracy_maximum"])
    action_limit = float(shortcut_cfg["action_only_balanced_accuracy_maximum"])
    proprio_limit = float(shortcut_cfg[
        "proprio_only_balanced_accuracy_maximum"])
    shortcuts: dict[str, Any] = {}
    for age in map(int, cfg["sequence"]["evidence_ages"]):
        endpoint = endpoint_frame(3, age)
        # Every class gets the exact same no-cue feature for each base window.
        visual_train = np.repeat(
            _pooled(np.asarray(bank.base_visual[train_base, endpoint])), 4,
            axis=0)
        visual_validation = np.repeat(
            _pooled(np.asarray(bank.base_visual[validation_base, endpoint])), 4,
            axis=0)
        action_train = np.repeat(
            bank.actions[train_base, :endpoint].reshape(len(train_base), -1),
            4, axis=0)
        action_validation = np.repeat(
            bank.actions[validation_base, :endpoint].reshape(
                len(validation_base), -1), 4, axis=0)
        proprio_train = np.repeat(
            bank.proprio[train_base, :endpoint + 1].reshape(
                len(train_base), -1), 4, axis=0)
        proprio_validation = np.repeat(
            bank.proprio[validation_base, :endpoint + 1].reshape(
                len(validation_base), -1), 4, axis=0)
        age_result = {}
        for name, x_train, x_validation, limit in (
                ("no_cue_visual", visual_train, visual_validation, visual_limit),
                ("action_only", action_train, action_validation, action_limit),
                ("proprio_only", proprio_train, proprio_validation,
                 proprio_limit)):
            prediction = fit_classifier(x_train, train_y, x_validation)
            record = classification_record(prediction, validation_y)
            record.update({"maximum": limit,
                           "pass": bool(record["balanced_accuracy"] <= limit)})
            age_result[name] = record
        shortcuts[str(age)] = age_result

    cue_audit = bank.manifest["cue_only_counterfactual"]
    counterfactual_pass = bool(
        cue_audit["outside_declared_mask_changed_pixels"] == 0
        and cue_audit["pairwise_outside_mask_changed_pixels"] == 0
        and cue_audit["post_cue_differing_pixels"] == 0
        and cue_audit["actions_proprio_states_max_abs_difference"] == 0.0)
    host_pass = host_digest_before == host_digest_after
    shortcut_pass = all(
        record["pass"] for age in shortcuts.values() for record in age.values())
    admitted = bool(cue_record["pass"] and shortcut_pass
                    and counterfactual_pass and host_pass)
    return {
        "schema": "dinowm_pointmaze_wave3_admission_v1",
        "status": "admitted" if admitted else "stopped_fail_closed",
        "admitted": admitted,
        "requirement": {
            "label_source": "four-label cue-only counterfactual repetition",
            "all_four_labels_per_base": True,
            "post_cue_target_leakage": False,
            "pass": counterfactual_pass,
        },
        "cue_encoding": cue_record,
        "shortcuts": shortcuts,
        "cue_only_counterfactual": {**cue_audit, "pass": counterfactual_pass},
        "frozen_host": {"digest_before": host_digest_before,
                        "digest_after": host_digest_after,
                        "pass": host_pass},
        "all_gates_required": True,
    }


def build_execution_deck(bank: FeatureBank, cfg: Mapping[str, Any],
                         admission: Mapping[str, Any]) -> dict[str, Any]:
    require(admission.get("admitted") is True,
            "controller deck is gated by semantic admission")
    output = bank.root / "execution_deck.npz"
    require(not output.exists(), "refusing to overwrite execution deck")
    use = cfg["external_use"]
    validation = bank.base_indices("validation")
    endpoint = endpoint_frame(3, int(use["evidence_age"]))
    initial_states = bank.states[validation, endpoint]
    vendor = resolve(cfg["source"]["dino_wm"]["repo_path"])
    simulator = CurrentMujocoPointMaze(vendor)
    version = tuple(int(value) for value in simulator.mujoco.__version__.split(".")[:2])
    require(version >= (3, 0), "current MuJoCo >=3.0 is required")
    base_count = len(validation)
    success = np.empty((base_count, 4, 4), dtype=np.int8)
    distance = np.empty((base_count, 4, 4), dtype=np.float32)
    final_state = np.empty((base_count, 4, 4), dtype=np.float64)
    steps = np.empty((base_count, 4), dtype=np.int32)
    replay = np.empty((base_count, 4), dtype=np.int8)
    selected_success = np.empty((base_count, 4), dtype=np.int8)
    for local, state in enumerate(initial_states):
        for selected in range(4):
            seed = 8_340_000 + local * 4 + selected
            kwargs = {
                "initial_state": state, "target": GOAL_WAYPOINTS[selected],
                "horizon": int(use["execution_horizon"]),
                "controller_seed": seed,
                "success_radius": float(use["success_radius"]),
            }
            first = execute_released_waypoint(simulator, vendor, **kwargs)
            second = execute_released_waypoint(simulator, vendor, **kwargs)
            final = np.asarray(first["final_state"])
            final_state[local, selected] = final
            steps[local, selected] = int(first["steps"])
            distance[local, selected] = np.linalg.norm(
                GOAL_WAYPOINTS - final[:2], axis=1)
            success[local, selected] = (
                distance[local, selected] < float(use["success_radius"]))
            selected_success[local, selected] = success[local, selected, selected]
            replay[local, selected] = int(
                np.array_equal(first["reset_state"], second["reset_state"])
                and np.array_equal(first["final_state"], second["final_state"])
                and first["steps"] == second["steps"])
        if (local + 1) % 10 == 0:
            print(f"[wave3-controller] {local + 1}/{base_count}", flush=True)
    oracle = success[:, np.arange(4), np.arange(4)]
    oracle_rate = float(oracle.mean())
    per_class = [float(oracle[:, label].mean()) for label in range(4)]
    off_diagonal = success[:, ~np.eye(4, dtype=np.bool_)].reshape(-1)
    replay_rate = float(replay.mean())
    admitted = bool(
        oracle_rate >= float(use["oracle_success_minimum"])
        and min(per_class) >= float(use["oracle_per_class_success_minimum"])
        and float(off_diagonal.mean())
        <= float(use["off_diagonal_false_success_maximum"])
        and replay_rate >= float(use["deterministic_reset_replay_minimum"]))
    np.savez_compressed(
        output, validation_base_index=validation,
        validation_episode=bank.episode[validation],
        initial_state=initial_states, goal_waypoints=GOAL_WAYPOINTS,
        success_matrix=success, distance_matrix=distance,
        final_state=final_state, steps=steps, replay=replay,
        selected_goal_success=selected_success)
    return {
        "schema": "dinowm_pointmaze_wave3_controller_gate_v1",
        "status": "admitted" if admitted else "stopped_fail_closed",
        "admitted": admitted,
        "current_mujoco_version": simulator.mujoco.__version__,
        "released_xml_sha256": simulator.xml_sha256,
        "validation_base_windows": base_count,
        "executions": base_count * 4,
        "replayed_executions": base_count * 4,
        "oracle_executed_success": oracle_rate,
        "oracle_per_class_executed_success": per_class,
        "off_diagonal_false_success": float(off_diagonal.mean()),
        "deterministic_replay_fidelity": replay_rate,
        "thresholds": {
            "oracle_success_minimum": use["oracle_success_minimum"],
            "oracle_per_class_success_minimum": use[
                "oracle_per_class_success_minimum"],
            "off_diagonal_false_success_maximum": use[
                "off_diagonal_false_success_maximum"],
            "deterministic_reset_replay_minimum": use[
                "deterministic_reset_replay_minimum"],
        },
        "artifact": _cache_artifact(output),
    }


def prepare_cache(cfg: Mapping[str, Any], lock: Mapping[str, Any]) \
        -> dict[str, Any]:
    root = resolve(cfg["artifacts"]["root"]) / "cache"
    require(not root.exists(), "refusing to overwrite Wave 3 cache")
    root.mkdir(parents=True)
    started = time.time()
    try:
        pins = verify_pins(cfg)
        dataset = NativePointMazeData(cfg)
        selections = dataset.selections()
        train_count = int(cfg["dataset"]["train_base_windows"])
        validation_count = int(cfg["dataset"]["validation_base_windows"])
        count = train_count + validation_count
        require(len(selections) == count, "locked base selection count changed")
        selection_path = root / "selection.json"
        atomic_json(selection_path, {
            "schema": "dinowm_pointmaze_wave3_selection_v1",
            "selection_sha256": selection_digest(selections),
            "values": [{"split": x.split, "episode_index": x.episode_index,
                        "local_start": x.local_start} for x in selections],
            "dataset": dataset.provenance(),
        })
        base_path, cue_path = root / "base_visual.npy", root / "cue_visual.npy"
        base = np.lib.format.open_memmap(
            base_path, mode="w+", dtype=np.float32,
            shape=(count, 20, 196, 384))
        cue = np.lib.format.open_memmap(
            cue_path, mode="w+", dtype=np.float32,
            shape=(count, 4, 3, 196, 384))
        actions = np.empty((count, 19, 10), dtype=np.float32)
        proprio = np.empty((count, 20, 4), dtype=np.float32)
        states = np.empty((count, 20, 4), dtype=np.float32)
        split = np.empty(count, dtype=np.uint8)
        episode = np.empty(count, dtype=np.int64)
        local_start = np.empty(count, dtype=np.int64)
        host = FrozenPointMazeHost(cfg, load_encoder=True)
        host_before = host.digest()
        outside, pairwise, post_cue, cue_min = 0, 0, 0, None
        batch_size = int(cfg["cache"]["build_base_batch"])
        for offset in range(0, count, batch_size):
            stop = min(count, offset + batch_size)
            native = [dataset.read(value) for value in selections[offset:stop]]
            frames = np.stack([value["frames"] for value in native])
            flat = frames.reshape(-1, 224, 224, 3)
            base[offset:stop] = host.encode_visual(
                flat, batch_size=int(cfg["cache"]["frame_batch_size"])).reshape(
                    stop - offset, 20, 196, 384)
            cue_frames = []
            for row, value in enumerate(native):
                variants = np.stack([
                    render_transient_goal_cue(value["frames"], label)
                    for label in range(4)])
                audit = verify_cue_only_counterfactual(value["frames"], variants)
                require(audit["passed"], "generated cue violated its mask")
                outside += audit["outside_declared_mask_changed_pixels"]
                pairwise += audit["pairwise_outside_mask_changed_pixels"]
                changed = min(audit["cue_changed_pixels_per_label"])
                cue_min = changed if cue_min is None else min(cue_min, changed)
                post_cue += int(np.count_nonzero(variants[:, 4:] != value[
                    "frames"][None, 4:]))
                cue_frames.append(variants[:, 1:4])
            cue_values = np.stack(cue_frames)
            cue[offset:stop] = host.encode_visual(
                cue_values.reshape(-1, 224, 224, 3),
                batch_size=int(cfg["cache"]["frame_batch_size"])).reshape(
                    stop - offset, 4, 3, 196, 384)
            actions[offset:stop] = np.stack([value["actions"] for value in native])
            proprio[offset:stop] = np.stack([value["proprio"] for value in native])
            states[offset:stop] = np.stack([value["state"] for value in native])
            split[offset:stop] = [0 if value.split == "train" else 1
                                  for value in selections[offset:stop]]
            episode[offset:stop] = [value.episode_index
                                    for value in selections[offset:stop]]
            local_start[offset:stop] = [value.local_start
                                        for value in selections[offset:stop]]
            print(f"[wave3-cache] {stop}/{count}", flush=True)
        base.flush()
        cue.flush()
        metadata_path = root / "metadata.npz"
        np.savez_compressed(
            metadata_path, actions=actions, proprio=proprio, states=states,
            split=split, episode_index=episode, local_start=local_start)
        host_after = host.digest()
        require(host_before == host_after, "cache build mutated frozen host")
        del base, cue
        cue_only = {
            "outside_declared_mask_changed_pixels": outside,
            "pairwise_outside_mask_changed_pixels": pairwise,
            "post_cue_differing_pixels": post_cue,
            "minimum_changed_cue_pixels_any_label": cue_min,
            "actions_proprio_states_max_abs_difference": 0.0,
            "all_four_labels_share_each_base": True,
        }
        artifacts = {name: _cache_artifact(path) for name, path in {
            "base_visual": base_path, "cue_visual": cue_path,
            "metadata": metadata_path, "selection": selection_path}.items()}
        partial = {
            "schema": "dinowm_pointmaze_wave3_cache_v1",
            "protocol_sha256": lock["protocol_sha256"],
            "selection_sha256": selection_digest(selections),
            "base_windows": count, "expanded_sequences": count * 4,
            "cue_only_counterfactual": cue_only,
            "host_digest_before": host_before,
            "host_digest_after": host_after,
            "host_unchanged": True,
            "pins": pins, "artifacts": artifacts,
            "status": "resolving_precarrier_gates",
        }
        # FeatureBank validates the immutable cache, so first materialize the
        # manifest before opening any semantic readout.
        atomic_json(root / "manifest.json", partial)
        bank = FeatureBank(cfg, lock)
        admission = resolve_admission(
            bank, cfg, host_before, host_after)
        formal = resolve(cfg["artifacts"]["root"]) / "formal"
        formal.mkdir(parents=True, exist_ok=False)
        atomic_json(formal / "admission.json", admission)
        if admission["admitted"]:
            controller = build_execution_deck(bank, cfg, admission)
        else:
            controller = {
                "schema": "dinowm_pointmaze_wave3_controller_not_run_v1",
                "status": "not_run_upstream_admission_failed",
                "admitted": False,
            }
        atomic_json(formal / "controller_gate.json", controller)
        complete = bool(admission["admitted"] and controller["admitted"])
        partial.update({
            "status": "admitted" if complete else "stopped_fail_closed",
            "precarrier_gates_passed": complete,
            "admission_path": str((formal / "admission.json").relative_to(ROOT)),
            "controller_gate_path": str((formal / "controller_gate.json").relative_to(ROOT)),
            "admission_sha256": sha256_file(formal / "admission.json"),
            "controller_gate_sha256": sha256_file(formal / "controller_gate.json"),
            "elapsed_seconds": time.time() - started,
        })
        atomic_json(root / "manifest.json", partial)
        if not complete:
            atomic_json(formal / "stop_receipt.json", {
                "schema": "dinowm_pointmaze_wave3_precarrier_stop_v1",
                "status": "stopped_fail_closed",
                "admission_passed": admission["admitted"],
                "controller_passed": controller["admitted"],
                "no_carrier_trained": True,
                "protocol_sha256": lock["protocol_sha256"],
            })
        return partial
    except Exception as error:
        atomic_json(root / "stop_receipt.json", {
            "schema": "dinowm_pointmaze_wave3_cache_stop_v1",
            "status": "failed_preserved", "reason": repr(error),
            "elapsed_seconds": time.time() - started})
        raise


def common_schedule_digest(cfg: Mapping[str, Any], seed: int,
                           episode_count: int) -> str:
    training = cfg["training"]
    rng = np.random.default_rng(int(training["common_schedule_seed_base"]) + seed)
    digest = hashlib.sha256()
    for _ in range(int(training["epochs"])):
        order = rng.permutation(episode_count).astype(np.int32)
        digest.update(order.tobytes())
        for offset in range(0, episode_count, int(training["batch_size"])):
            starts = rng.choice(
                17, int(training["windows_per_batch"]),
                replace=False).astype(np.int16)
            digest.update(starts.tobytes())
    return digest.hexdigest()


def shifted_objective(host: FrozenPointMazeHost, carrier: torch.nn.Module,
                      visual: torch.Tensor, actions: torch.Tensor,
                      proprio: torch.Tensor, starts: Sequence[int]) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = spatial_carrier_forward(carrier, visual, actions)
    fused_windows, prop_windows, action_windows = [], [], []
    target_visual, target_prop = [], []
    for raw_start in starts:
        start = int(raw_start)
        require(0 <= start and start + 3 < visual.shape[1],
                f"illegal objective window {start}")
        fused_windows.append(output.fused_visual[:, start:start + 3])
        prop_windows.append(proprio[:, start:start + 3])
        action_windows.append(actions[:, start:start + 3])
        target_visual.append(visual[:, start + 1:start + 4])
        target_prop.append(proprio[:, start + 1:start + 4])
    fused = torch.cat(fused_windows)
    prop = torch.cat(prop_windows)
    action = torch.cat(action_windows)
    target = host.target_nonaction(
        torch.cat(target_visual), torch.cat(target_prop))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = host.predict(fused, prop, action)[..., :394]
        visual_loss = F.mse_loss(
            prediction[..., :384].float(), target[..., :384].float())
        proprio_loss = F.mse_loss(
            prediction[..., 384:].float(), target[..., 384:].float())
        loss = F.mse_loss(prediction.float(), target.float())
    return loss, visual_loss, proprio_loss


def train_carrier(host: FrozenPointMazeHost, carrier: torch.nn.Module,
                  bank: FeatureBank, seed: int,
                  cfg: Mapping[str, Any]) \
        -> tuple[list[dict[str, Any]], str]:
    training = cfg["training"]
    train_indices = bank.expanded_indices("train")
    rng = np.random.default_rng(int(training["common_schedule_seed_base"]) + seed)
    schedule_hash = hashlib.sha256()
    optimizer = torch.optim.AdamW(
        carrier.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["epochs"]))
    carrier.train()
    history = []
    for epoch in range(1, int(training["epochs"]) + 1):
        order = rng.permutation(len(train_indices)).astype(np.int32)
        schedule_hash.update(order.tobytes())
        losses, visual_losses, proprio_losses = [], [], []
        epoch_started = time.time()
        for offset in range(0, len(order), int(training["batch_size"])):
            selected = train_indices[order[
                offset:offset + int(training["batch_size"])]]
            bases, _ = bank.decode_expanded(selected)
            starts = rng.choice(
                17, int(training["windows_per_batch"]),
                replace=False).astype(np.int16)
            schedule_hash.update(starts.tobytes())
            visual = torch.from_numpy(bank.visual(selected)).to(host.device)
            actions = torch.from_numpy(bank.actions[bases]).to(host.device)
            proprio = torch.from_numpy(bank.proprio[bases]).to(host.device)
            optimizer.zero_grad(set_to_none=True)
            loss, visual_loss, proprio_loss = shifted_objective(
                host, carrier, visual, actions, proprio, starts)
            require(torch.isfinite(loss).item(), "non-finite carrier loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                carrier.parameters(), float(training["gradient_clip_norm"]))
            require(all(p.grad is None or torch.isfinite(p.grad).all()
                        for p in carrier.parameters()),
                    "non-finite carrier gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            visual_losses.append(float(visual_loss.detach()))
            proprio_losses.append(float(proprio_loss.detach()))
            del visual, actions, proprio, loss, visual_loss, proprio_loss
        scheduler.step()
        record = {
            "epoch": epoch, "loss": float(np.mean(losses)),
            "visual_loss": float(np.mean(visual_losses)),
            "proprio_loss": float(np.mean(proprio_losses)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - epoch_started,
        }
        history.append(record)
        print(f"[wave3-train] {carrier.name}/s{seed} "
              f"epoch {epoch}/{training['epochs']} loss={record['loss']:.6f} "
              f"sec={record['seconds']:.1f}", flush=True)
    expected = common_schedule_digest(cfg, seed, len(train_indices))
    require(schedule_hash.hexdigest() == expected,
            "carrier training schedule changed")
    return history, expected


@torch.no_grad()
def collect_features(host: FrozenPointMazeHost, carrier: torch.nn.Module,
                     bank: FeatureBank, split: str,
                     cfg: Mapping[str, Any]) \
        -> dict[int, dict[str, np.ndarray]]:
    carrier.eval()
    indices = bank.expanded_indices(split)
    ages = list(map(int, cfg["sequence"]["evidence_ages"]))
    rows: dict[int, dict[str, list[np.ndarray]]] = {
        age: {"full": [], "reset": [], "prior": [],
              "full_mse": [], "reset_mse": []} for age in ages}
    batch_size = int(cfg["training"]["batch_size"])
    for offset in range(0, len(indices), batch_size):
        expanded = indices[offset:offset + batch_size]
        bases, _ = bank.decode_expanded(expanded)
        visual = torch.from_numpy(bank.visual(expanded)).to(host.device)
        actions = torch.from_numpy(bank.actions[bases]).to(host.device)
        proprio = torch.from_numpy(bank.proprio[bases]).to(host.device)
        full = spatial_carrier_forward(carrier, visual, actions)
        for age in ages:
            endpoint = endpoint_frame(3, age)
            context = predictor_context_for_endpoint(endpoint)
            start, stop = context[0], context[-1] + 1
            full_prediction = host.predict(
                full.fused_visual[:, start:stop], proprio[:, start:stop],
                actions[:, start:stop])[:, -1, :, :384]
            reset = spatial_carrier_forward(
                carrier, visual[:, start:stop], actions[:, start:stop - 1])
            reset_prediction = host.predict(
                reset.fused_visual, proprio[:, start:stop],
                actions[:, start:stop])[:, -1, :, :384]
            target = visual[:, endpoint]
            rows[age]["full"].append(_pooled(
                full_prediction.float().cpu().numpy()))
            rows[age]["reset"].append(_pooled(
                reset_prediction.float().cpu().numpy()))
            rows[age]["prior"].append(_pooled(
                full.prior_visual[:, endpoint].float().cpu().numpy()))
            rows[age]["full_mse"].append(torch.mean(
                torch.square(full_prediction - target), dim=(1, 2)).cpu().numpy())
            rows[age]["reset_mse"].append(torch.mean(
                torch.square(reset_prediction - target), dim=(1, 2)).cpu().numpy())
        del visual, actions, proprio, full
    return {age: {name: np.concatenate(values)
                  for name, values in record.items()}
            for age, record in rows.items()}


def evaluate_cell(host: FrozenPointMazeHost, carrier: torch.nn.Module,
                  bank: FeatureBank, cfg: Mapping[str, Any]) \
        -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    train = collect_features(host, carrier, bank, "train", cfg)
    validation = collect_features(host, carrier, bank, "validation", cfg)
    train_y = expanded_labels(int(cfg["dataset"]["train_base_windows"]))
    validation_y = expanded_labels(
        int(cfg["dataset"]["validation_base_windows"]))
    arrays: dict[str, np.ndarray] = {"truth": validation_y}
    metrics = {}
    for age in map(int, cfg["sequence"]["evidence_ages"]):
        prediction = fit_classifier(
            train[age]["full"], train_y, validation[age]["full"])
        reset_prediction = fit_classifier(
            train[age]["full"], train_y, validation[age]["reset"])
        prior_prediction = fit_classifier(
            train[age]["prior"], train_y, validation[age]["prior"])
        arrays[f"age_{age}_full_prediction"] = prediction
        arrays[f"age_{age}_reset_prediction"] = reset_prediction
        arrays[f"age_{age}_prior_prediction"] = prior_prediction
        arrays[f"age_{age}_full_mse"] = validation[age]["full_mse"]
        arrays[f"age_{age}_reset_mse"] = validation[age]["reset_mse"]
        metrics[str(age)] = {
            "endpoint_frame": endpoint_frame(3, age),
            "predictor_context": list(predictor_context_for_endpoint(
                endpoint_frame(3, age))),
            "target_observation_excluded": True,
            "full": classification_record(prediction, validation_y),
            "reset_with_full_readout": classification_record(
                reset_prediction, validation_y),
            "prior": classification_record(prior_prediction, validation_y),
            "full_next_visual_mse": float(np.mean(
                validation[age]["full_mse"])),
            "reset_next_visual_mse": float(np.mean(
                validation[age]["reset_mse"])),
        }
    use_age = int(cfg["external_use"]["evidence_age"])
    use = {
        "train_feature": train[use_age]["full"].astype(np.float32),
        "validation_feature": validation[use_age]["full"].astype(np.float32),
        "train_truth": train_y, "validation_truth": validation_y,
    }
    require(use["train_feature"].shape[1]
            == int(cfg["external_use"]["consumer_feature_dim"]),
            "external-use feature differs from the registered primary read")
    return metrics, arrays, use


def write_cell(formal: Path, *, arm: str, seed: int,
               cfg: Mapping[str, Any], lock: Mapping[str, Any],
               carrier: torch.nn.Module, history: list[dict[str, Any]],
               metrics: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
               use: Mapping[str, np.ndarray]) -> Path:
    final = formal / "cells" / arm / f"s{seed}"
    require(not final.exists(), f"refusing to overwrite {final}")
    stage = formal / ".staging" / arm / f"s{seed}"
    stage.mkdir(parents=True, exist_ok=False)
    history_path = stage / "history.csv"
    with history_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "epoch", "loss", "visual_loss", "proprio_loss", "lr", "seconds"))
        writer.writeheader()
        writer.writerows(history)
    predictions_path = stage / "validation_predictions.npz"
    np.savez_compressed(predictions_path, **arrays)
    use_path = stage / "use_features.npz"
    np.savez_compressed(use_path, **use)
    metrics_path = stage / "metrics.json"
    atomic_json(metrics_path, metrics)
    checkpoint_path = stage / "carrier.pt"
    torch.save({"carrier_state_dict": carrier.state_dict(),
                "metrics": dict(metrics)}, checkpoint_path)
    artifacts = {path.name: _cache_artifact(path) for path in (
        history_path, predictions_path, use_path, metrics_path, checkpoint_path)}
    # Paths in the sidecar are root-relative; validation checks identity only.
    atomic_json(stage / "manifest.json", {
        "schema": "dinowm_pointmaze_wave3_cell_manifest_v1",
        "protocol_sha256": lock["protocol_sha256"],
        "arm": arm, "seed": seed, "artifacts": artifacts})
    final.parent.mkdir(parents=True, exist_ok=True)
    os.rename(stage, final)
    return final


def run_cell(host: FrozenPointMazeHost, bank: FeatureBank, arm: str, seed: int,
             cfg: Mapping[str, Any], lock: Mapping[str, Any], formal: Path) -> Path:
    configure_cuda(cfg, seed)
    host.verify_schema(load_encoder=False)
    host_before = host.digest()
    started = time.time()
    torch.cuda.reset_peak_memory_stats(host.device)
    carrier = make_frozen_carrier(arm, 384, 10).to(host.device)
    if carrier.parameter_count():
        history, schedule = train_carrier(host, carrier, bank, seed, cfg)
    else:
        history = []
        schedule = common_schedule_digest(
            cfg, seed, int(cfg["dataset"]["train_expanded_sequences"]))
    readout, arrays, use = evaluate_cell(host, carrier, bank, cfg)
    host_after = host.digest()
    require(host_before == host_after, f"frozen host changed in {arm}/s{seed}")
    if arm == "none":
        for age in cfg["sequence"]["evidence_ages"]:
            require(np.array_equal(arrays[f"age_{age}_full_prediction"],
                                   arrays[f"age_{age}_reset_prediction"]),
                    "none full/reset predictions differ")
    losses = [float(row["loss"]) for row in history]
    convergence = None if len(losses) < 5 else float(
        (losses[-1] - losses[-5]) / max(abs(losses[-5]), 1e-12))
    metrics = {
        "schema": "dinowm_pointmaze_wave3_cell_v1",
        "protocol_sha256": lock["protocol_sha256"],
        "task": cfg["task"]["key"], "arm": arm, "seed": seed,
        "physical_gpu": 2,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "host_digest_before": host_before, "host_digest_after": host_after,
        "host_unchanged": True,
        "carrier_parameters": carrier.parameter_count(),
        "carrier_config": carrier.describe(),
        "parameter_matching": parameter_report(384, 10),
        "training_labels_used": False,
        "epochs": int(cfg["training"]["epochs"])
            if carrier.parameter_count() else 0,
        "common_schedule_sha256": schedule,
        "final_train_loss": losses[-1] if losses else None,
        "final_five_epoch_relative_change": convergence,
        "ages": readout,
        "elapsed_seconds": time.time() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(host.device)),
    }
    return write_cell(
        formal, arm=arm, seed=seed, cfg=cfg, lock=lock, carrier=carrier,
        history=history, metrics=metrics, arrays=arrays, use=use)


def clone_none_cell(source: Path, host: FrozenPointMazeHost, seed: int,
                    cfg: Mapping[str, Any], lock: Mapping[str, Any],
                    formal: Path) -> Path:
    metrics = json.loads((source / "metrics.json").read_text())
    with np.load(source / "validation_predictions.npz") as values:
        arrays = {name: values[name] for name in values.files}
    with np.load(source / "use_features.npz") as values:
        use = {name: values[name] for name in values.files}
    metrics.update({
        "seed": seed,
        "common_schedule_sha256": common_schedule_digest(
            cfg, seed, int(cfg["dataset"]["train_expanded_sequences"])),
        "duplicated_deterministic_no_carrier_from_seed": 0,
        "effective_independent_models": 1,
    })
    carrier = make_frozen_carrier("none", 384, 10).to(host.device)
    return write_cell(
        formal, arm="none", seed=seed, cfg=cfg, lock=lock, carrier=carrier,
        history=[], metrics=metrics, arrays=arrays, use=use)


def validate_cell(directory: Path, arm: str, seed: int,
                  lock: Mapping[str, Any]) \
        -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    require(manifest["protocol_sha256"] == lock["protocol_sha256"]
            and metrics["protocol_sha256"] == lock["protocol_sha256"],
            f"cell lock mismatch: {directory}")
    require((metrics["arm"], int(metrics["seed"])) == (arm, seed),
            f"cell identity mismatch: {directory}")
    require(metrics["host_unchanged"] is True
            and metrics["host_digest_before"] == metrics["host_digest_after"],
            f"cell changed host: {directory}")
    for name, record in manifest["artifacts"].items():
        path = directory / name
        require(path.is_file() and path.stat().st_size == record["size"]
                and sha256_file(path) == record["sha256"],
                f"cell artifact mismatch: {path}")
    with np.load(directory / "validation_predictions.npz") as values:
        arrays = {name: values[name] for name in values.files}
    require(arrays["truth"].shape == (480,),
            "validation prediction count changed")
    return metrics, arrays


def episode_cluster_bootstrap(values: np.ndarray, episodes: np.ndarray, *,
                              draws: int, seed: int,
                              confidence: float = 0.95) -> dict[str, Any]:
    """Crossed carrier-seed × equal-native-episode cluster bootstrap."""

    values = np.asarray(values, dtype=np.float64)
    episodes = np.asarray(episodes, dtype=np.int64)
    require(values.ndim == 2 and values.shape[1] == len(episodes),
            "cluster-bootstrap values must be (seed,expanded-example)")
    unique = np.unique(episodes)
    require(len(unique) >= 2, "cluster bootstrap needs >=2 native episodes")
    per_episode = np.stack([
        values[:, episodes == episode].mean(axis=1) for episode in unique
    ], axis=1)
    point = float(per_episode.mean())
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(draws), dtype=np.float64)
    cursor = 0
    while cursor < int(draws):
        stop = min(int(draws), cursor + 512)
        count = stop - cursor
        seed_rows = rng.integers(
            0, values.shape[0], size=(count, values.shape[0]))
        episode_rows = rng.integers(
            0, len(unique), size=(count, len(unique)))
        selected = per_episode[
            seed_rows[:, :, None], episode_rows[:, None, :]]
        samples[cursor:stop] = selected.mean(axis=(1, 2))
        cursor = stop
    alpha = (1.0 - float(confidence)) / 2.0
    interval = np.quantile(samples, (alpha, 1.0 - alpha))
    return {
        "mean": point, "ci95": interval.astype(float).tolist(),
        "draws": int(draws), "seed": int(seed),
        "confidence": float(confidence), "paired": True,
        "equal_native_episode_weight": True,
        "native_episode_clusters": len(unique),
        "carrier_seeds": values.shape[0],
        "ci_excludes_zero": bool(interval[0] > 0 or interval[1] < 0),
    }


def prediction_correct(predictions: np.ndarray, truth: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    require(values.ndim == 2 and values.shape[1] == len(truth),
            "prediction matrix is not aligned")
    return (values == truth[None]).astype(np.float64)


def aggregate_carriers(formal: Path, bank: FeatureBank,
                       cfg: Mapping[str, Any], lock: Mapping[str, Any]) \
        -> dict[str, Any]:
    arms = list(cfg["training"]["arms"])
    seeds = list(map(int, cfg["training"]["seeds"]))
    loaded = {}
    for arm in arms:
        for seed in seeds:
            directory = formal / "cells" / arm / f"s{seed}"
            loaded[(arm, seed)] = validate_cell(directory, arm, seed, lock)
    truth = loaded[("none", 0)][1]["truth"]
    validation_base = bank.base_indices("validation")
    episodes = np.repeat(bank.episode[validation_base], 4)
    inference = cfg["inference"]
    results = {}
    for age_index, age in enumerate(map(int, cfg["sequence"]["evidence_ages"])):
        predictions, resets = {}, {}
        record = {"arms": {}, "paired_vs_none": {},
                  "full_vs_context_reset": {}}
        for arm_index, arm in enumerate(arms):
            predictions[arm] = np.stack([
                loaded[(arm, seed)][1][f"age_{age}_full_prediction"]
                for seed in seeds])
            resets[arm] = np.stack([
                loaded[(arm, seed)][1][f"age_{age}_reset_prediction"]
                for seed in seeds])
            absolute = episode_cluster_bootstrap(
                prediction_correct(predictions[arm], truth), episodes,
                draws=int(inference["draws"]),
                seed=int(inference["seed"]) + age_index * 100 + arm_index,
                confidence=float(inference["confidence"]))
            seed_values = [balanced_accuracy_from_predictions(
                predictions[arm][index], truth, 4) for index in range(len(seeds))]
            record["arms"][arm] = {
                "balanced_accuracy": absolute,
                "seed_values": seed_values,
                "parameters": loaded[(arm, 0)][0]["carrier_parameters"],
                "effective_independent_models": 1 if arm == "none" else 5,
                "prior_seed_values": [loaded[(arm, seed)][0]["ages"][str(age)][
                    "prior"]["balanced_accuracy"] for seed in seeds],
                "next_visual_mse_seed_values": [loaded[(arm, seed)][0]["ages"][
                    str(age)]["full_next_visual_mse"] for seed in seeds],
            }
            if arm != "none":
                contrast = prediction_correct(predictions[arm], truth) \
                    - prediction_correct(predictions["none"], truth)
                record["paired_vs_none"][arm] = episode_cluster_bootstrap(
                    contrast, episodes, draws=int(inference["draws"]),
                    seed=int(inference["seed"]) + 5000
                    + age_index * 100 + arm_index,
                    confidence=float(inference["confidence"]))
            reset_contrast = prediction_correct(predictions[arm], truth) \
                - prediction_correct(resets[arm], truth)
            record["full_vs_context_reset"][arm] = episode_cluster_bootstrap(
                reset_contrast, episodes, draws=int(inference["draws"]),
                seed=int(inference["seed"]) + 10_000
                + age_index * 100 + arm_index,
                confidence=float(inference["confidence"]))
        results[str(age)] = record
    summary = {
        "schema": "dinowm_pointmaze_wave3_carrier_summary_v1",
        "status": "complete", "protocol_sha256": lock["protocol_sha256"],
        "study": cfg["study"], "task": cfg["task"],
        "host": cfg["checkpoint"]["display_name"],
        "adapter": cfg["adapter"],
        "grid": {"tasks": 1, "arms": 5, "seeds": 5, "cells": 25},
        "parameter_matching": parameter_report(384, 10),
        "inference": cfg["inference"], "results": results,
    }
    atomic_json(formal / "carrier_summary.json", summary)
    return summary


def load_use_features(formal: Path, arm: str, seed: int) \
        -> dict[str, np.ndarray]:
    with np.load(formal / "cells" / arm / f"s{seed}" / "use_features.npz") \
            as values:
        return {name: values[name] for name in values.files}


def evaluate_external_use(formal: Path, bank: FeatureBank,
                          cfg: Mapping[str, Any],
                          lock: Mapping[str, Any]) -> dict[str, Any]:
    gate = json.loads((formal / "controller_gate.json").read_text())
    require(gate.get("admitted") is True, "controller gate did not pass")
    deck_path = bank.root / "execution_deck.npz"
    with np.load(deck_path) as values:
        success_matrix = values["success_matrix"]
        deck_episodes = values["validation_episode"]
    arms = list(cfg["training"]["arms"])
    seeds = list(map(int, cfg["training"]["seeds"]))
    predictions: dict[str, list[np.ndarray]] = {arm: [] for arm in arms}
    truth = None
    consumer_receipts = []
    for seed in seeds:
        sources = {arm: load_use_features(formal, arm, seed) for arm in arms}
        reference = sources[arms[0]]
        if truth is None:
            truth = reference["validation_truth"].astype(np.int64)
        train_x = np.concatenate([sources[arm]["train_feature"] for arm in arms])
        train_y = np.concatenate([sources[arm]["train_truth"] for arm in arms])
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000,
                               random_state=0))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(train_x, train_y)
        coefficient = classifier[-1].coef_
        digest = hashlib.sha256(coefficient.tobytes()).hexdigest()
        for arm in arms:
            require(np.array_equal(sources[arm]["validation_truth"], truth),
                    "use validation truth differs across arms")
            predictions[arm].append(classifier.predict(
                sources[arm]["validation_feature"]).astype(np.int64))
        consumer_receipts.append({
            "seed": seed, "arm_blind": True,
            "training_arms": arms, "arm_identifier_feature": False,
            "train_examples": len(train_y), "feature_dim": train_x.shape[1],
            "coefficient_sha256": digest,
        })
    assert truth is not None
    prediction_matrices = {arm: np.stack(values)
                           for arm, values in predictions.items()}
    executed, goal_correct = {}, {}
    for arm in arms:
        arm_executed, arm_correct = [], []
        for seed_index in range(len(seeds)):
            crossed = crossed_execution_arrays(
                success_matrix, prediction_matrices[arm][seed_index], truth)
            arm_executed.append(crossed["executed_success"])
            arm_correct.append(crossed["goal_correct"])
        executed[arm] = np.stack(arm_executed).astype(np.float64)
        goal_correct[arm] = np.stack(arm_correct).astype(np.float64)
    random_predictions = []
    random_executed = []
    for seed in seeds:
        rng = np.random.default_rng(int(cfg["external_use"]["random_goal_seed"]) + seed)
        prediction = rng.integers(0, 4, size=len(truth), dtype=np.int64)
        random_predictions.append(prediction)
        random_executed.append(crossed_execution_arrays(
            success_matrix, prediction, truth)["executed_success"])
    random_executed_matrix = np.stack(random_executed).astype(np.float64)
    episodes = np.repeat(deck_episodes, 4)
    inference = cfg["inference"]
    arm_results = {}
    for arm_index, arm in enumerate(arms):
        result = {
            "goal_accuracy": episode_cluster_bootstrap(
                goal_correct[arm], episodes, draws=int(inference["draws"]),
                seed=int(inference["seed"]) + 20_000 + arm_index,
                confidence=float(inference["confidence"])),
            "executed_success": episode_cluster_bootstrap(
                executed[arm], episodes, draws=int(inference["draws"]),
                seed=int(inference["seed"]) + 21_000 + arm_index,
                confidence=float(inference["confidence"])),
            "contrast_vs_none": episode_cluster_bootstrap(
                executed[arm] - executed["none"], episodes,
                draws=int(inference["draws"]),
                seed=int(inference["seed"]) + 22_000 + arm_index,
                confidence=float(inference["confidence"])),
            "contrast_vs_random": episode_cluster_bootstrap(
                executed[arm] - random_executed_matrix, episodes,
                draws=int(inference["draws"]),
                seed=int(inference["seed"]) + 23_000 + arm_index,
                confidence=float(inference["confidence"])),
        }
        result["resolved_execution_gain"] = bool(
            arm != "none"
            and result["contrast_vs_none"]["ci95"][0] > 0
            and result["contrast_vs_random"]["ci95"][0] > 0)
        arm_results[arm] = result
    random_result = episode_cluster_bootstrap(
        random_executed_matrix, episodes, draws=int(inference["draws"]),
        seed=int(inference["seed"]) + 24_000,
        confidence=float(inference["confidence"]))
    prediction_path = formal / "external_use_predictions.npz"
    np.savez_compressed(
        prediction_path, truth=truth, validation_episode=episodes,
        success_matrix=success_matrix,
        random_prediction=np.stack(random_predictions),
        random_executed_success=random_executed_matrix,
        **{f"prediction__{arm}": value
           for arm, value in prediction_matrices.items()},
        **{f"executed__{arm}": value for arm, value in executed.items()})
    result = {
        "schema": "dinowm_pointmaze_wave3_external_use_v1",
        "status": "complete", "protocol_sha256": lock["protocol_sha256"],
        "scope": cfg["external_use"], "controller_gate": gate,
        "consumer_receipts": consumer_receipts,
        "arms": arm_results, "realized_random_goal": random_result,
        "oracle_executed_success": gate["oracle_executed_success"],
        "artifact": _cache_artifact(prediction_path),
        "interpretation": (
            "External arm-blind goal selection plus released waypoint control "
            "in current MuJoCo; this is not native DINO-WM planning."),
    }
    atomic_json(formal / "external_use_summary.json", result)
    return result


def write_markdown_summary(formal: Path, carrier: Mapping[str, Any],
                           use: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    arms = cfg["training"]["arms"]
    lines = [
        "# DINO-WM PointMaze Wave 3", "",
        "All intervals: 20,000 matched carrier-seed × native-episode cluster bootstrap draws.",
        "", "## Persistent carrier read", "",
        "| age | arm | balanced accuracy [95% CI] | Δ vs none [95% CI] | Δ full-reset [95% CI] |",
        "|---:|---|---:|---:|---:|",
    ]
    for age in cfg["sequence"]["evidence_ages"]:
        record = carrier["results"][str(age)]
        for arm in arms:
            absolute = record["arms"][arm]["balanced_accuracy"]
            if arm == "none":
                delta = "--"
            else:
                value = record["paired_vs_none"][arm]
                delta = f"{value['mean']:+.3f} [{value['ci95'][0]:+.3f},{value['ci95'][1]:+.3f}]"
            reset = record["full_vs_context_reset"][arm]
            lines.append(
                f"| {age} | {arm} | {absolute['mean']:.3f} "
                f"[{absolute['ci95'][0]:.3f},{absolute['ci95'][1]:.3f}] | "
                f"{delta} | {reset['mean']:+.3f} "
                f"[{reset['ci95'][0]:+.3f},{reset['ci95'][1]:+.3f}] |")
    lines += ["", "## External executed use (age 15)", "",
              "| arm | goal accuracy | executed success | Δ vs none [95% CI] | Δ vs random [95% CI] |",
              "|---|---:|---:|---:|---:|"]
    for arm in arms:
        record = use["arms"][arm]
        goal = record["goal_accuracy"]
        success = record["executed_success"]
        none = record["contrast_vs_none"]
        random = record["contrast_vs_random"]
        lines.append(
            f"| {arm} | {goal['mean']:.3f} | {success['mean']:.3f} | "
            f"{none['mean']:+.3f} [{none['ci95'][0]:+.3f},{none['ci95'][1]:+.3f}] | "
            f"{random['mean']:+.3f} [{random['ci95'][0]:+.3f},{random['ci95'][1]:+.3f}] |")
    lines += ["", "Execution is external: a shared arm-blind linear goal selector feeds the released waypoint controller in current MuJoCo; no native planner claim is made.", ""]
    (formal / "summary.md").write_text("\n".join(lines))


def run_formal(cfg: Mapping[str, Any], lock: Mapping[str, Any], *,
               resume: bool) -> dict[str, Any]:
    root = resolve(cfg["artifacts"]["root"])
    formal = root / "formal"
    require(formal.is_dir(), "prepare stage did not create formal gates")
    admission = json.loads((formal / "admission.json").read_text())
    controller = json.loads((formal / "controller_gate.json").read_text())
    require(admission.get("admitted") is True
            and controller.get("admitted") is True,
            "Wave 3 stopped before carriers because a locked gate failed")
    require(not (formal / "stop_receipt.json").exists(),
            "Wave 3 has a preserved pre-carrier stop receipt")
    provenance_path = formal / "provenance.json"
    if resume:
        require(provenance_path.is_file(), "resume requested before formal launch")
        provenance = json.loads(provenance_path.read_text())
        require(provenance["protocol_sha256"] == lock["protocol_sha256"],
                "resume lock differs")
    else:
        require(not provenance_path.exists(), "formal exists; use --resume")
        (formal / ".staging").mkdir(exist_ok=False)
        (formal / "cells").mkdir(exist_ok=False)
        (formal / "failures").mkdir(exist_ok=False)
        provenance = {
            "schema": "dinowm_pointmaze_wave3_provenance_v1",
            "status": "running", "protocol_sha256": lock["protocol_sha256"],
            "source_sha256": lock["source_sha256"],
            "physical_gpu": 2,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "paper_modified_by_wave3": False,
            "started_unix": time.time(), "pins": verify_pins(cfg),
            "admission_sha256": sha256_file(formal / "admission.json"),
            "controller_gate_sha256": sha256_file(
                formal / "controller_gate.json"),
            "environment": {"python": sys.version, "torch": torch.__version__,
                            "numpy": np.__version__,
                            "platform": platform.platform(),
                            "disk_before": shutil.disk_usage(ROOT)._asdict()},
        }
        atomic_json(provenance_path, provenance)
    bank = FeatureBank(cfg, lock)
    host = FrozenPointMazeHost(cfg, load_encoder=False)
    initial_host = host.digest()
    provenance["runtime_host_digest"] = initial_host
    provenance["environment"].update({
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "logical_device": str(host.device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    })
    atomic_json(provenance_path, provenance)
    expected = [(arm, seed) for arm in cfg["training"]["arms"]
                for seed in cfg["training"]["seeds"]]
    completed = []
    try:
        for arm, raw_seed in expected:
            seed = int(raw_seed)
            final = formal / "cells" / arm / f"s{seed}"
            if final.exists():
                validate_cell(final, arm, seed, lock)
                completed.append([arm, seed])
                continue
            print(f"[wave3-formal] start {arm}/s{seed}", flush=True)
            if arm == "none" and seed > 0:
                path = clone_none_cell(
                    formal / "cells/none/s0", host, seed, cfg, lock, formal)
            else:
                path = run_cell(host, bank, arm, seed, cfg, lock, formal)
            validate_cell(path, arm, seed, lock)
            require(host.digest() == initial_host,
                    "frozen host changed between formal cells")
            completed.append([arm, seed])
            atomic_json(formal / "progress.json", {
                "protocol_sha256": lock["protocol_sha256"],
                "completed_cells": completed, "count": len(completed),
                "expected": 25, "updated_unix": time.time()})
            print(f"[wave3-formal] complete {arm}/s{seed} "
                  f"({len(completed)}/25)", flush=True)
        require(len(completed) == 25, "Wave 3 carrier grid is incomplete")
        carrier = aggregate_carriers(formal, bank, cfg, lock)
        use = evaluate_external_use(formal, bank, cfg, lock)
        summary = {
            "schema": "dinowm_pointmaze_wave3_summary_v1",
            "status": "complete", "protocol_sha256": lock["protocol_sha256"],
            "scope": cfg["scope"], "admission": admission,
            "controller_gate": controller,
            "carrier_summary_path": "carrier_summary.json",
            "external_use_summary_path": "external_use_summary.json",
            "resolved_external_use_arms": [arm for arm, value in use["arms"].items()
                                           if value["resolved_execution_gain"]],
        }
        atomic_json(formal / "summary.json", summary)
        write_markdown_summary(formal, carrier, use, cfg)
        provenance.update({
            "status": "complete", "completed_unix": time.time(),
            "elapsed_seconds": time.time() - provenance["started_unix"],
            "runtime_host_digest_after": host.digest(),
        })
        require(provenance["runtime_host_digest_after"] == initial_host,
                "completed formal run changed host")
        provenance["environment"]["disk_after"] = shutil.disk_usage(ROOT)._asdict()
        atomic_json(provenance_path, provenance)
        return summary
    except Exception as error:
        failure = {
            "schema": "dinowm_pointmaze_wave3_formal_stop_v1",
            "status": "stopped_fail_closed", "reason": repr(error),
            "completed_cells": completed, "count": len(completed),
            "protocol_sha256": lock["protocol_sha256"],
            "stopped_unix": time.time(), "no_post_hoc_adaptation": True,
        }
        atomic_json(formal / "formal_stop_receipt.json", failure)
        provenance["status"] = "stopped_fail_closed"
        provenance["formal_stop_receipt"] = failure
        atomic_json(provenance_path, provenance)
        raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--smoke", action="store_true")
    stage.add_argument("--seal", action="store_true")
    stage.add_argument("--prepare", action="store_true")
    stage.add_argument("--formal", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config.resolve()
    if args.smoke:
        require(args.execute, "smoke requires --execute")
        cfg, _ = load_config(config_path, locked=False)
        result = run_smoke(cfg)
    elif args.seal:
        require(not args.execute, "seal is non-metric")
        cfg, _ = load_config(config_path, locked=False)
        result = seal_protocol(config_path, cfg)
    elif args.prepare:
        require(args.execute, "prepare requires --execute")
        cfg, lock = load_config(config_path, locked=True)
        assert lock is not None
        result = prepare_cache(cfg, lock)
    else:
        require(args.execute, "formal requires --execute")
        cfg, lock = load_config(config_path, locked=True)
        assert lock is not None
        result = run_formal(cfg, lock, resume=bool(args.resume))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
