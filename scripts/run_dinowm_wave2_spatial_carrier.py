#!/usr/bin/env python3
"""Wave 2 persistent carriers inside the frozen official DINO-WM predictor.

Stages are deliberately separated:

``--smoke`` performs shape, gradient, native-equivalence, and runtime checks
without fitting a semantic readout. ``--seal`` hashes the final protocol and
all implementation inputs. ``--prepare`` builds the frozen full-patch cache
only after the lock exists. ``--formal`` validates the reused V2R2 admissions,
trains the locked 2x5x5 grid on physical GPU 1, evaluates it, and writes the
20,000-draw paired inference.  No stage edits ``paper_a``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import (  # noqa: E402
    FROZEN_CARRIER_NAMES,
    make_frozen_carrier,
    parameter_report,
)
from lewm.official_tasks.dinowm_native_audit import (  # noqa: E402
    NATIVE_ACTION_DIM,
    NATIVE_CONTEXT,
    NATIVE_PATCHES,
    NATIVE_PROPRIO_DIM,
    NATIVE_VISUAL_DIM,
    spatial_pyramid_pool,
)
from lewm.official_tasks.dinowm_spatial_carrier import (  # noqa: E402
    absolute_bootstrap,
    balanced_accuracy_from_predictions,
    crossed_paired_bootstrap,
    endpoint_frame,
    predictor_context_for_endpoint,
    spatial_carrier_forward,
)
from lewm.official_tasks.pusht_memory import render_single_overlay  # noqa: E402
from scripts.run_dinowm_native_pusht_audit_v1 import (  # noqa: E402
    _canonical_sha256,
    _chunks,
    _fixed_normalize_actions,
    _fixed_normalize_proprio,
    _git_archive_sha256,
    _git_output,
)
from scripts.run_dinowm_native_pusht_audit_v2 import (  # noqa: E402
    NativeSelection,
    OfficialDinoWMPushT,
    _read,
)


DEFAULT_CONFIG = ROOT / "configs/dinowm_wave2_spatial_carrier.yaml"


class Wave2Stop(RuntimeError):
    """A locked fail-closed condition stopped downstream computation."""


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


def load_config(path: Path, *, locked: bool) -> tuple[dict[str, Any], dict[str, Any] | None]:
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw)
    require(isinstance(cfg, dict), "Wave 2 config must be a YAML mapping")
    if not locked:
        return cfg, None
    lock_path = path.with_suffix(".lock.json")
    require(lock_path.is_file(), "Wave 2 formal lock does not exist")
    lock = json.loads(lock_path.read_text())
    require(lock.get("locked_before_formal_metrics") is True,
            "Wave 2 lock is not formal")
    require(lock.get("protocol_sha256") == hashlib.sha256(raw).hexdigest(),
            "Wave 2 protocol changed after sealing")
    for relative, expected in lock.get("source_sha256", {}).items():
        source = resolve(relative)
        require(source.is_file() and sha256_file(source) == expected,
                f"locked source changed: {relative}")
    return cfg, lock


def check_identity(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    require(path.is_file(), f"missing pinned file: {path}")
    size = path.stat().st_size
    digest = sha256_file(path)
    require(size == int(record["size"]), f"size mismatch: {path}")
    require(digest == record["sha256"], f"SHA-256 mismatch: {path}")
    return {"path": str(path), "size": size, "sha256": digest}


def module_digest(modules: Mapping[str, torch.nn.Module]) -> str:
    digest = hashlib.sha256()
    for module_name, module in sorted(modules.items()):
        for name, value in sorted(module.state_dict().items()):
            digest.update(module_name.encode())
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configure_cuda(cfg: Mapping[str, Any], seed: int) -> torch.device:
    execution = cfg["execution"]
    expected = str(execution["required_cuda_visible_devices"])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == expected,
            f"CUDA_VISIBLE_DEVICES must be exactly {expected!r}")
    require(int(execution["physical_gpu"]) == 1
            and bool(execution["never_gpu3"]),
            "Wave 2 is pinned to physical GPU 1 and must never use GPU 3")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "GPU visibility must expose exactly one CUDA device")
    device = torch.device(str(execution["logical_device"]))
    require(str(device) == "cuda:0", "logical device must be cuda:0")
    torch.cuda.set_device(device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    return device


class FrozenNativeHost:
    """GPU-1 loader preserving the released DINO-WM token contract."""

    def __init__(self, cfg: Mapping[str, Any], *, load_encoder: bool) -> None:
        self.cfg = cfg
        self.device = configure_cuda(cfg, seed=9070)
        vendor = resolve(cfg["source"]["dino_wm"]["repo_path"])
        dino_repo = resolve(cfg["source"]["dinov2"]["repo_path"])
        os.environ["TORCH_HOME"] = str(resolve(
            cfg["source"]["dinov2"]["torch_home"]))
        sys.path.insert(0, str(vendor))
        payload = torch.load(
            resolve(cfg["checkpoint"]["weights_path"]), map_location="cpu",
            weights_only=False)
        require(set(payload) == {
            "epoch", "predictor", "predictor_optimizer", "decoder",
            "decoder_optimizer", "action_encoder", "proprio_encoder",
        }, "released checkpoint schema changed")
        self.epoch = int(payload["epoch"])
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
        require(moved == 6, f"expected six native attention masks, moved {moved}")
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
        values = {
            "predictor": self.predictor,
            "action_encoder": self.action_encoder,
            "proprio_encoder": self.proprio_encoder,
        }
        if self.encoder is not None:
            values["dinov2_encoder"] = self.encoder
        return values

    def verify_schema(self, *, load_encoder: bool) -> None:
        require(tuple(self.predictor.pos_embedding.shape) == (1, 588, 404),
                "unexpected predictor positional embedding")
        require(tuple(self.action_encoder.patch_embed.weight.shape) == (10, 10, 1),
                "unexpected native action encoder")
        require(tuple(self.proprio_encoder.patch_embed.weight.shape) == (10, 4, 1),
                "unexpected native proprio encoder")
        if load_encoder:
            assert self.encoder is not None
            require(self.encoder.num_features == 384
                    and self.encoder.patch_size == 14,
                    "unexpected DINOv2 encoder")
        require(all(not p.requires_grad for module in self.modules.values()
                    for p in module.parameters()), "host is not frozen")
        require(not self.predictor.training and not self.action_encoder.training
                and not self.proprio_encoder.training,
                "host modules must stay in eval mode")
        require(torch.cuda.current_device() == 0,
                "logical CUDA device is not zero under GPU-1 isolation")

    def digest(self) -> str:
        # The released predictor stores its six causal attention masks as
        # ordinary tensor attributes rather than registered buffers, so they
        # are absent from ``state_dict``.  Cover both the frozen weights and
        # those execution-critical masks in the before/after host identity.
        digest = hashlib.sha256()
        digest.update(b"frozen-module-state-dicts\0")
        digest.update(module_digest(self.modules).encode())
        mask_count = 0
        for name, module in sorted(self.predictor.named_modules()):
            bias = getattr(module, "bias", None)
            if torch.is_tensor(bias) and bias.ndim == 4:
                digest.update(b"attention-mask\0")
                digest.update(name.encode())
                digest.update(bias.detach().cpu().contiguous().numpy().tobytes())
                mask_count += 1
        require(mask_count == 6,
                f"expected six attention masks in host digest, got {mask_count}")
        return digest.hexdigest()

    @torch.no_grad()
    def encode_visual(self, frames: np.ndarray, *, batch_size: int) -> np.ndarray:
        require(self.encoder is not None, "DINO encoder was not loaded")
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        values = np.asarray(frames)
        require(values.ndim == 4 and values.shape[-1] == 3
                and values.dtype == np.uint8,
                "native frames must be uint8 BHWC")
        outputs = []
        for rows in _chunks(values, batch_size):
            tensor = torch.from_numpy(np.asarray(rows).copy()).to(self.device)
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            tensor = tensor.sub_(0.5).div_(0.5)
            tensor = TF.resize(
                tensor, [196, 196], interpolation=InterpolationMode.BILINEAR,
                antialias=True)
            patches = self.encoder.forward_features(tensor)["x_norm_patchtokens"]
            require(tuple(patches.shape[1:]) == (196, 384),
                    "DINO patch output changed")
            outputs.append(patches.float().cpu().numpy())
        return np.concatenate(outputs)

    def compose(self, visual: torch.Tensor, proprio: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        require(visual.ndim == 4 and visual.shape[2:] == (196, 384),
                "visual context violates native shape")
        require(proprio.shape[:2] == visual.shape[:2]
                and proprio.shape[-1] == 4,
                "proprio context violates native shape")
        require(actions.shape[:2] == visual.shape[:2]
                and actions.shape[-1] == 10,
                "action context violates native shape")
        prop = self.proprio_encoder(proprio)
        act = self.action_encoder(actions)
        prop = prop.unsqueeze(2).expand(-1, -1, 196, -1)
        act = act.unsqueeze(2).expand(-1, -1, 196, -1)
        return torch.cat((visual, prop, act), dim=-1)

    def predict(self, visual: torch.Tensor, proprio: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        context = self.compose(visual, proprio, actions)
        batch, steps, patches, dim = context.shape
        require(steps == 3 and patches == 196 and dim == 404,
                "native predictor requires exactly 3x196x404")
        predicted = self.predictor(context.reshape(batch, steps * patches, dim))
        return predicted.reshape(batch, steps, patches, dim)

    @torch.no_grad()
    def target_nonaction(self, visual: torch.Tensor,
                         proprio: torch.Tensor) -> torch.Tensor:
        prop = self.proprio_encoder(proprio)
        prop = prop.unsqueeze(2).expand(-1, -1, 196, -1)
        return torch.cat((visual, prop), dim=-1)


def select_tasks(dataset: OfficialDinoWMPushT,
                 cfg: Mapping[str, Any]) -> dict[str, list[NativeSelection]]:
    data_cfg, sequence = cfg["dataset"], cfg["sequence"]
    result: dict[str, list[NativeSelection]] = {}
    for task in cfg["tasks"]:
        result[task["key"]] = dataset.select(
            train_count=int(data_cfg["train_episodes"]),
            validation_count=int(data_cfg["validation_episodes"]),
            num_frames=int(sequence["num_frames"]),
            frame_skip=int(sequence["frame_skip"]),
            classes=int(task["classes"]),
            split_seed=int(data_cfg["split_seed"]),
            start_seed=int(data_cfg["start_seed"]),
            label_seed=int(task["label_seed"]),
            source_split=str(data_cfg["source_split"]),
        )
    reference = next(iter(result.values()))
    identity = [(x.split, x.episode_index, x.local_start) for x in reference]
    for key, selections in result.items():
        require([(x.split, x.episode_index, x.local_start)
                 for x in selections] == identity,
                f"task {key} does not share the native base bank")
    return result


def dataset_and_selections(cfg: Mapping[str, Any]
                           ) -> tuple[OfficialDinoWMPushT,
                                      dict[str, list[NativeSelection]]]:
    manifest = resolve(cfg["dataset"]["manifest_path"])
    dataset = OfficialDinoWMPushT(
        resolve(cfg["dataset"]["root"]), manifest,
        cfg["dataset"]["manifest_identity"]["sha256"])
    selections = select_tasks(dataset, cfg)
    prior_path = ROOT / "outputs/dinowm_native_pusht_audit_v2r2/formal/selection.json"
    prior = json.loads(prior_path.read_text())["tasks"]
    for key, values in selections.items():
        require([asdict(value) for value in values] == prior[key],
                f"Wave 2 {key} selection differs from locked V2R2")
    return dataset, selections


def verify_pins(cfg: Mapping[str, Any]) -> dict[str, Any]:
    identities = {
        "dataset_manifest": check_identity(
            resolve(cfg["dataset"]["manifest_path"]),
            cfg["dataset"]["manifest_identity"]),
        "dataset_archive": check_identity(
            resolve(cfg["dataset"]["archive_path"]),
            cfg["dataset"]["archive_identity"]),
        "checkpoint": check_identity(
            resolve(cfg["checkpoint"]["weights_path"]),
            cfg["checkpoint"]["weights_identity"]),
        "checkpoint_config": check_identity(
            resolve(cfg["checkpoint"]["config_path"]),
            cfg["checkpoint"]["config_identity"]),
        "dinov2_weights": check_identity(
            resolve(cfg["source"]["dinov2"]["weights_path"]),
            cfg["source"]["dinov2"]["weights_identity"]),
        "dependency_manifest": check_identity(
            resolve(cfg["execution"]["dependency_manifest_path"]),
            cfg["execution"]["dependency_manifest_identity"]),
    }
    sources = {}
    for key in ("dino_wm", "dinov2"):
        record = cfg["source"][key]
        repo = resolve(record["repo_path"])
        revision = _git_output(repo, "rev-parse", "HEAD")
        status = _git_output(repo, "status", "--porcelain")
        archive = _git_archive_sha256(repo)
        require(revision == record["revision"] and not status
                and archive == record["git_archive_sha256"],
                f"pinned {key} source changed")
        sources[key] = {"revision": revision, "clean": True,
                        "git_archive_sha256": archive}
    return {"identities": identities, "sources": sources}


def seal_protocol(config_path: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    lock_path = config_path.with_suffix(".lock.json")
    require(not lock_path.exists(), "refusing to overwrite Wave 2 lock")
    output = resolve(cfg["artifacts"]["root"])
    smoke = output / cfg["artifacts"]["smoke"] / "receipt.json"
    require(smoke.is_file(), "successful smoke receipt is required before sealing")
    smoke_value = json.loads(smoke.read_text())
    require(smoke_value.get("status") == "passed_no_semantic_metric",
            "smoke did not pass")
    formal = output / cfg["artifacts"]["formal"]
    require(not formal.exists(), "formal artifact directory already exists")
    sources = {}
    for relative in cfg["lock"]["source_paths"]:
        path = resolve(relative)
        require(path.is_file(), f"cannot seal missing source {relative}")
        sources[str(relative)] = sha256_file(path)
    value = {
        "schema": "dinowm_wave2_spatial_carrier_lock_v1",
        "locked_before_formal_metrics": True,
        "protocol_sha256": sha256_file(config_path),
        "source_sha256": sources,
        "smoke_receipt": {
            "path": str(smoke.relative_to(ROOT)),
            "sha256": sha256_file(smoke),
            "semantic_readout_fitted": False,
        },
        "parameter_matching": parameter_report(384, 10),
        "grid": {"tasks": 2, "arms": 5, "seeds": 5, "cells": 50},
    }
    atomic_json(lock_path, value)
    return value


def make_visual_batch(base: np.ndarray, cue: np.ndarray,
                      indices: np.ndarray, cue_start: int,
                      cue_length: int) -> np.ndarray:
    values = np.asarray(base[indices], dtype=np.float32).copy()
    values[:, cue_start:cue_start + cue_length] = np.asarray(
        cue[indices], dtype=np.float32)
    return values


def shifted_objective(host: FrozenNativeHost, carrier: torch.nn.Module,
                      visual: torch.Tensor, actions: torch.Tensor,
                      proprio: torch.Tensor, starts: Sequence[int]
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official three-position shifted feature loss for selected windows."""

    output = spatial_carrier_forward(carrier, visual, actions)
    fused_windows, prop_windows, action_windows = [], [], []
    target_visual, target_prop = [], []
    for raw_start in starts:
        start = int(raw_start)
        if start < 0 or start + 3 >= visual.shape[1]:
            raise ValueError(f"illegal training window start {start}")
        fused_windows.append(output.fused_visual[:, start:start + 3])
        prop_windows.append(proprio[:, start:start + 3])
        action_windows.append(actions[:, start:start + 3])
        target_visual.append(visual[:, start + 1:start + 4])
        target_prop.append(proprio[:, start + 1:start + 4])
    fused = torch.cat(fused_windows, dim=0)
    prop = torch.cat(prop_windows, dim=0)
    action = torch.cat(action_windows, dim=0)
    target = host.target_nonaction(
        torch.cat(target_visual, dim=0), torch.cat(target_prop, dim=0))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = host.predict(fused, prop, action)[..., :394]
        visual_loss = F.mse_loss(
            prediction[..., :384].float(), target[..., :384].float())
        proprio_loss = F.mse_loss(
            prediction[..., 384:].float(), target[..., 384:].float())
        loss = F.mse_loss(prediction.float(), target.float())
    return loss, visual_loss, proprio_loss


def run_smoke(config_path: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    output = resolve(cfg["artifacts"]["root"]) / cfg["artifacts"]["smoke"]
    require(not output.exists(), "refusing to overwrite Wave 2 smoke")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        pins = verify_pins(cfg)
        dataset, selections = dataset_and_selections(cfg)
        reference = selections[cfg["tasks"][0]["key"]][:2]
        native = _read(dataset, reference, cfg)
        task = cfg["tasks"][0]
        frames = np.stack([
            render_single_overlay(
                value.frames, task["semantic_name"], selection.label,
                int(cfg["sequence"]["cue_start"]),
                int(cfg["sequence"]["cue_length"]))
            for value, selection in zip(native, reference)
        ])
        actions = _fixed_normalize_actions(
            np.stack([value.actions for value in native]))
        proprio = _fixed_normalize_proprio(
            np.stack([value.proprio for value in native]))

        host = FrozenNativeHost(cfg, load_encoder=True)
        shape = frames.shape[2:]
        visual = host.encode_visual(
            frames.reshape(-1, *shape),
            batch_size=int(cfg["cache"]["frame_batch_size"])).reshape(
                2, 20, 196, 384)
        z = torch.from_numpy(visual[:1]).to(host.device)
        a = torch.from_numpy(actions[:1]).to(host.device)
        q = torch.from_numpy(proprio[:1]).to(host.device)
        host_before = host.digest()
        cells: dict[str, Any] = {}
        native_prediction = None
        for arm in cfg["training"]["arms"]:
            configure_cuda(cfg, seed=123)
            carrier = make_frozen_carrier(arm, 384, 10).to(host.device)
            initial = spatial_carrier_forward(carrier, z, a)
            zero_max = float((initial.fused_visual - z).abs().max().detach())
            torch.cuda.reset_peak_memory_stats(host.device)
            before = time.time()
            if carrier.parameter_count():
                optimizer = torch.optim.AdamW(
                    carrier.parameters(),
                    lr=float(cfg["training"]["learning_rate"]),
                    weight_decay=float(cfg["training"]["weight_decay"]))
                optimizer.zero_grad(set_to_none=True)
                loss, visual_loss, proprio_loss = shifted_objective(
                    host, carrier, z, a, q, starts=[0])
                loss.backward()
                gradients = [p.grad for p in carrier.parameters()
                             if p.requires_grad]
                require(gradients and all(g is not None and torch.isfinite(g).all()
                                          for g in gradients),
                        f"{arm} smoke gradients are missing or non-finite")
                torch.nn.utils.clip_grad_norm_(carrier.parameters(), 1.0)
                optimizer.step()
                scalar_loss = float(loss.detach())
                component = [float(visual_loss.detach()),
                             float(proprio_loss.detach())]
            else:
                with torch.no_grad():
                    prediction = host.predict(
                        initial.fused_visual[:, :3], q[:, :3], a[:, :3])
                native_prediction = prediction.detach().clone()
                scalar_loss, component = None, None
            elapsed = time.time() - before
            cells[arm] = {
                "parameters": carrier.parameter_count(),
                "description": carrier.describe(),
                "zero_init_fused_max_abs": zero_max,
                "one_step_loss_finite": scalar_loss is None
                    or bool(np.isfinite(scalar_loss)),
                "one_step_loss": scalar_loss,
                "visual_proprio_loss": component,
                "elapsed_seconds": elapsed,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated(
                    host.device)),
            }
            require(zero_max == 0.0,
                    f"{arm} zero initialization does not reproduce native tokens")
            del carrier
            torch.cuda.empty_cache()
        require(native_prediction is not None, "none smoke was not executed")
        with torch.no_grad():
            direct = host.predict(z[:, :3], q[:, :3], a[:, :3])
        none_error = float((native_prediction - direct).abs().max().cpu())
        require(none_error == 0.0, "none adapter differs from native predictor")
        host_after = host.digest()
        require(host_before == host_after, "smoke mutated the frozen host")
        receipt = {
            "schema": "dinowm_wave2_smoke_v1",
            "status": "passed_no_semantic_metric",
            "semantic_readout_fitted": False,
            "task_accuracy_computed": False,
            "physical_gpu": 1,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "host_digest_before": host_before,
            "host_digest_after": host_after,
            "host_unchanged": True,
            "none_native_prediction_max_abs": none_error,
            "parameter_matching": parameter_report(384, 10),
            "cells": cells,
            "pins": pins,
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(output / "receipt.json", receipt)
        return receipt
    except Exception as error:
        receipt = {
            "schema": "dinowm_wave2_smoke_stop_v1",
            "status": "failed_preserved",
            "reason": repr(error),
            "semantic_readout_fitted": False,
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(output / "stop_receipt.json", receipt)
        raise


def validate_prior_admissions(cfg: Mapping[str, Any]) -> dict[str, Any]:
    prior_root = ROOT / "outputs/dinowm_native_pusht_audit_v2r2/formal"
    verification = json.loads((prior_root / "verification.json").read_text())
    require(verification.get("verified") is True,
            "prior DINO-WM audit is not independently verified")
    admissions = {}
    for task in cfg["tasks"]:
        path = prior_root / "admission" / f"{task['key']}.json"
        value = json.loads(path.read_text())
        require(value.get("admitted") is True
                and all(gate.get("pass") is True
                        for gate in value.get("gates", {}).values()),
                f"prior admission failed for {task['key']}")
        admissions[task["key"]] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "admitted": True,
            "gates": value["gates"],
        }
    health_path = prior_root / "rollout_health.json"
    health = json.loads(health_path.read_text())
    require(health.get("admitted") is True
            and all(gate.get("pass") is True
                    for gate in health.get("gates", {}).values()),
            "prior native rollout-health gate failed")
    return {
        "prior_verification_sha256": sha256_file(
            prior_root / "verification.json"),
        "tasks": admissions,
        "rollout_health": {
            "path": str(health_path.relative_to(ROOT)),
            "sha256": sha256_file(health_path),
            "admitted": True,
            "gates": health["gates"],
        },
    }


def prepare_cache(config_path: Path, cfg: Mapping[str, Any],
                  lock: Mapping[str, Any]) -> dict[str, Any]:
    root = resolve(cfg["artifacts"]["root"])
    cache_root = root / "cache"
    require(not cache_root.exists(), "refusing to overwrite Wave 2 cache")
    cache_root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        pins = verify_pins(cfg)
        admissions = validate_prior_admissions(cfg)
        dataset, selections_by_task = dataset_and_selections(cfg)
        reference = selections_by_task[cfg["tasks"][0]["key"]]
        count = len(reference)
        shape = (count, 20, 196, 384)
        base_path = cache_root / "base_visual.npy"
        base = np.lib.format.open_memmap(
            base_path, mode="w+", dtype=np.float32, shape=shape)
        cue_maps = {}
        cue_paths = {}
        for task in cfg["tasks"]:
            path = cache_root / f"{task['key']}_cue_visual.npy"
            cue_paths[task["key"]] = path
            cue_maps[task["key"]] = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32,
                shape=(count, 3, 196, 384))
        actions = np.empty((count, 19, 10), dtype=np.float32)
        proprio = np.empty((count, 20, 4), dtype=np.float32)
        split = np.empty(count, dtype=np.uint8)
        episode = np.empty(count, dtype=np.int64)
        local_start = np.empty(count, dtype=np.int64)
        labels = {task["key"]: np.empty(count, dtype=np.int64)
                  for task in cfg["tasks"]}
        host = FrozenNativeHost(cfg, load_encoder=True)
        host_before = host.digest()
        batch_size = int(cfg["cache"]["build_episode_batch"])
        for offset in range(0, count, batch_size):
            stop = min(count, offset + batch_size)
            selected = reference[offset:stop]
            native = _read(dataset, selected, cfg)
            frames = np.stack([value.frames for value in native])
            frame_shape = frames.shape[2:]
            base[offset:stop] = host.encode_visual(
                frames.reshape(-1, *frame_shape),
                batch_size=int(cfg["cache"]["frame_batch_size"])).reshape(
                    len(native), 20, 196, 384)
            actions[offset:stop] = _fixed_normalize_actions(
                np.stack([value.actions for value in native]))
            proprio[offset:stop] = _fixed_normalize_proprio(
                np.stack([value.proprio for value in native]))
            split[offset:stop] = np.asarray(
                [0 if item.split == "train" else 1 for item in selected],
                dtype=np.uint8)
            episode[offset:stop] = [item.episode_index for item in selected]
            local_start[offset:stop] = [item.local_start for item in selected]
            for task in cfg["tasks"]:
                task_selected = selections_by_task[task["key"]][offset:stop]
                labels[task["key"]][offset:stop] = [
                    item.label for item in task_selected]
                overlays = np.stack([
                    render_single_overlay(
                        value.frames, task["semantic_name"], selection.label,
                        int(cfg["sequence"]["cue_start"]),
                        int(cfg["sequence"]["cue_length"]))[1:4]
                    for value, selection in zip(native, task_selected)
                ])
                cue_maps[task["key"]][offset:stop] = host.encode_visual(
                    overlays.reshape(-1, *frame_shape),
                    batch_size=int(cfg["cache"]["frame_batch_size"])).reshape(
                        len(native), 3, 196, 384)
            print(f"[wave2-cache] {stop}/{count}", flush=True)
        base.flush()
        for value in cue_maps.values():
            value.flush()
        metadata_path = cache_root / "metadata.npz"
        np.savez_compressed(
            metadata_path, actions=actions, proprio=proprio, split=split,
            episode_index=episode, local_start=local_start,
            **{f"labels__{key}": value for key, value in labels.items()})

        # Exact none-adapter equivalence to preserved V2R2 teacher endpoints.
        indices = np.asarray([0, 1, 1199, 1200, count - 1], dtype=np.int64)
        z = torch.from_numpy(np.asarray(base[indices, 16:19])).to(host.device)
        q = torch.from_numpy(proprio[indices, 16:19]).to(host.device)
        a = torch.from_numpy(actions[indices, 16:19]).to(host.device)
        with torch.no_grad():
            predicted = host.predict(z, q, a)[:, -1, :, :384].float().cpu().numpy()
        pooled = spatial_pyramid_pool(predicted)
        prior_teacher = np.load(
            ROOT / "outputs/dinowm_native_pusht_audit_v2r2/formal/teacher_features.npz")
        expected = prior_teacher["predicted_endpoint"][indices]
        equivalence = float(np.max(np.abs(pooled - expected)))
        require(equivalence <= 2e-6,
                f"none adapter differs from preserved V2R2 by {equivalence}")
        host_after = host.digest()
        require(host_before == host_after, "cache build mutated the frozen host")
        del base, cue_maps

        artifacts = {
            "base_visual": {"path": str(base_path.relative_to(ROOT)),
                            "size": base_path.stat().st_size,
                            "sha256": sha256_file(base_path)},
            "metadata": {"path": str(metadata_path.relative_to(ROOT)),
                         "size": metadata_path.stat().st_size,
                         "sha256": sha256_file(metadata_path)},
        }
        for key, path in cue_paths.items():
            artifacts[f"cue__{key}"] = {
                "path": str(path.relative_to(ROOT)), "size": path.stat().st_size,
                "sha256": sha256_file(path)}
        manifest = {
            "schema": "dinowm_wave2_full_patch_cache_v1",
            "protocol_sha256": lock["protocol_sha256"],
            "shape": list(shape),
            "dtype": "float32",
            "selection_sha256": _canonical_sha256({
                key: [asdict(item) for item in values]
                for key, values in selections_by_task.items()}),
            "none_v2r2_teacher_endpoint_max_abs": equivalence,
            "none_equivalence_threshold": 2e-6,
            "host_digest_before": host_before,
            "host_digest_after": host_after,
            "host_unchanged": True,
            "admissions": admissions,
            "pins": pins,
            "artifacts": artifacts,
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(cache_root / "manifest.json", manifest)
        return manifest
    except Exception as error:
        atomic_json(cache_root / "stop_receipt.json", {
            "schema": "dinowm_wave2_cache_stop_v1",
            "status": "failed_preserved", "reason": repr(error),
            "elapsed_seconds": time.time() - started,
        })
        raise


class FeatureBank:
    def __init__(self, cfg: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.root = resolve(cfg["artifacts"]["root"]) / "cache"
        manifest_path = self.root / "manifest.json"
        require(manifest_path.is_file(), "Wave 2 cache is incomplete")
        self.manifest = json.loads(manifest_path.read_text())
        require(self.manifest.get("protocol_sha256") == lock["protocol_sha256"],
                "cache belongs to another Wave 2 protocol")
        for record in self.manifest["artifacts"].values():
            path = resolve(record["path"])
            require(path.is_file() and path.stat().st_size == record["size"]
                    and sha256_file(path) == record["sha256"],
                    f"cache artifact identity failed: {path}")
        self.base = np.load(self.root / "base_visual.npy", mmap_mode="r")
        self.cues = {
            task["key"]: np.load(
                self.root / f"{task['key']}_cue_visual.npy", mmap_mode="r")
            for task in cfg["tasks"]
        }
        metadata = np.load(self.root / "metadata.npz")
        self.actions = np.asarray(metadata["actions"], dtype=np.float32)
        self.proprio = np.asarray(metadata["proprio"], dtype=np.float32)
        self.split = np.asarray(metadata["split"], dtype=np.uint8)
        self.labels = {
            task["key"]: np.asarray(
                metadata[f"labels__{task['key']}"], dtype=np.int64)
            for task in cfg["tasks"]
        }
        require(self.base.shape == (1680, 20, 196, 384)
                and self.actions.shape == (1680, 19, 10)
                and self.proprio.shape == (1680, 20, 4),
                "cache arrays violate the locked shape")
        require(np.count_nonzero(self.split == 0) == 1200
                and np.count_nonzero(self.split == 1) == 480,
                "cache split counts changed")

    def indices(self, split: str) -> np.ndarray:
        code = {"train": 0, "validation": 1}[split]
        return np.flatnonzero(self.split == code)

    def visual(self, task: str, indices: np.ndarray) -> np.ndarray:
        return make_visual_batch(
            self.base, self.cues[task], np.asarray(indices, dtype=np.int64),
            int(self.cfg["sequence"]["cue_start"]),
            int(self.cfg["sequence"]["cue_length"]))


def common_schedule_digest(cfg: Mapping[str, Any], seed: int,
                           episode_count: int) -> str:
    training = cfg["training"]
    rng = np.random.default_rng(int(training["common_schedule_seed_base"]) + seed)
    digest = hashlib.sha256()
    valid_starts = 20 - 3
    for _ in range(int(training["epochs"])):
        order = rng.permutation(episode_count).astype(np.int32)
        digest.update(order.tobytes())
        for offset in range(0, episode_count, int(training["batch_size"])):
            starts = rng.choice(
                valid_starts, int(training["windows_per_batch"]),
                replace=False).astype(np.int16)
            digest.update(starts.tobytes())
    return digest.hexdigest()


def train_carrier(host: FrozenNativeHost, carrier: torch.nn.Module,
                  bank: FeatureBank, task: str, seed: int,
                  cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    training = cfg["training"]
    train_indices = bank.indices("train")
    rng = np.random.default_rng(int(training["common_schedule_seed_base"]) + seed)
    schedule_hash = hashlib.sha256()
    carrier.train()
    optimizer = torch.optim.AdamW(
        carrier.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["epochs"]))
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training["epochs"]) + 1):
        order = rng.permutation(len(train_indices)).astype(np.int32)
        schedule_hash.update(order.tobytes())
        losses, visual_losses, proprio_losses = [], [], []
        epoch_started = time.time()
        for offset in range(0, len(order), int(training["batch_size"])):
            local = order[offset:offset + int(training["batch_size"])]
            rows = train_indices[local]
            starts = rng.choice(
                17, int(training["windows_per_batch"]),
                replace=False).astype(np.int16)
            schedule_hash.update(starts.tobytes())
            visual = torch.from_numpy(bank.visual(task, rows)).to(host.device)
            actions = torch.from_numpy(bank.actions[rows]).to(host.device)
            proprio = torch.from_numpy(bank.proprio[rows]).to(host.device)
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
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "visual_loss": float(np.mean(visual_losses)),
            "proprio_loss": float(np.mean(proprio_losses)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - epoch_started,
        })
        print(f"[wave2-train] {task}/{carrier.name}/s{seed} "
              f"epoch {epoch}/{training['epochs']} loss={history[-1]['loss']:.6f} "
              f"sec={history[-1]['seconds']:.1f}", flush=True)
    expected = common_schedule_digest(cfg, seed, len(train_indices))
    require(schedule_hash.hexdigest() == expected,
            "training schedule differs from the locked common schedule")
    return history, expected


@torch.no_grad()
def collect_features(host: FrozenNativeHost, carrier: torch.nn.Module,
                     bank: FeatureBank, task: str, split: str,
                     cfg: Mapping[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    carrier.eval()
    indices = bank.indices(split)
    ages = [int(value) for value in cfg["sequence"]["evidence_ages"]]
    rows: dict[int, dict[str, list[np.ndarray]]] = {
        age: {"full": [], "reset": [], "prior": [],
              "full_mse": [], "reset_mse": []}
        for age in ages
    }
    batch_size = int(cfg["evaluation"]["batch_size"])
    for offset in range(0, len(indices), batch_size):
        selected = indices[offset:offset + batch_size]
        visual = torch.from_numpy(bank.visual(task, selected)).to(host.device)
        actions = torch.from_numpy(bank.actions[selected]).to(host.device)
        proprio = torch.from_numpy(bank.proprio[selected]).to(host.device)
        full_output = spatial_carrier_forward(carrier, visual, actions)
        for age in ages:
            endpoint = endpoint_frame(
                int(cfg["sequence"]["last_cue_frame"]), age)
            context = predictor_context_for_endpoint(endpoint)
            start, stop = context[0], context[-1] + 1
            full_prediction = host.predict(
                full_output.fused_visual[:, start:stop],
                proprio[:, start:stop], actions[:, start:stop])[
                    :, -1, :, :384]
            reset_output = spatial_carrier_forward(
                carrier, visual[:, start:stop], actions[:, start:stop - 1])
            reset_prediction = host.predict(
                reset_output.fused_visual, proprio[:, start:stop],
                actions[:, start:stop])[:, -1, :, :384]
            target = visual[:, endpoint]
            full_mse = torch.mean(
                torch.square(full_prediction - target), dim=(1, 2))
            reset_mse = torch.mean(
                torch.square(reset_prediction - target), dim=(1, 2))
            rows[age]["full"].append(spatial_pyramid_pool(
                full_prediction.float().cpu().numpy()))
            rows[age]["reset"].append(spatial_pyramid_pool(
                reset_prediction.float().cpu().numpy()))
            rows[age]["prior"].append(spatial_pyramid_pool(
                full_output.prior_visual[:, endpoint].float().cpu().numpy()))
            rows[age]["full_mse"].append(full_mse.float().cpu().numpy())
            rows[age]["reset_mse"].append(reset_mse.float().cpu().numpy())
        del visual, actions, proprio, full_output
    return {
        age: {name: np.concatenate(values)
              for name, values in record.items()}
        for age, record in rows.items()
    }


def fit_readouts(train: Mapping[int, Mapping[str, np.ndarray]],
                 validation: Mapping[int, Mapping[str, np.ndarray]],
                 train_y: np.ndarray, validation_y: np.ndarray,
                 cfg: Mapping[str, Any], classes: int
                 ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics, arrays = {}, {"truth": validation_y.astype(np.int64)}
    for age in [int(value) for value in cfg["sequence"]["evidence_ages"]]:
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=4000, random_state=0))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(train[age]["full"], train_y)
        full_prediction = classifier.predict(
            validation[age]["full"]).astype(np.int64)
        reset_prediction = classifier.predict(
            validation[age]["reset"]).astype(np.int64)
        prior_classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=4000, random_state=0))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            prior_classifier.fit(train[age]["prior"], train_y)
        prior_prediction = prior_classifier.predict(
            validation[age]["prior"]).astype(np.int64)
        arrays[f"age_{age}_full_prediction"] = full_prediction
        arrays[f"age_{age}_reset_prediction"] = reset_prediction
        arrays[f"age_{age}_prior_prediction"] = prior_prediction
        arrays[f"age_{age}_full_mse"] = validation[age]["full_mse"]
        arrays[f"age_{age}_reset_mse"] = validation[age]["reset_mse"]
        metrics[str(age)] = {
            "endpoint_frame": endpoint_frame(3, age),
            "predictor_context": list(predictor_context_for_endpoint(
                endpoint_frame(3, age))),
            "target_observation_excluded": True,
            "full_balanced_accuracy": balanced_accuracy_from_predictions(
                full_prediction, validation_y, classes),
            "reset_with_full_readout_balanced_accuracy":
                balanced_accuracy_from_predictions(
                    reset_prediction, validation_y, classes),
            "prior_balanced_accuracy": balanced_accuracy_from_predictions(
                prior_prediction, validation_y, classes),
            "full_next_visual_mse": float(
                np.mean(validation[age]["full_mse"])),
            "reset_next_visual_mse": float(
                np.mean(validation[age]["reset_mse"])),
            "feature_dim": int(train[age]["full"].shape[1]),
            "readout": cfg["evaluation"]["readout"],
        }
    return metrics, arrays


def write_cell(final_root: Path, staging: Path, *, task: str, arm: str,
               seed: int, cfg: Mapping[str, Any], lock: Mapping[str, Any],
               carrier: torch.nn.Module, history: list[dict[str, Any]],
               metrics: dict[str, Any], arrays: Mapping[str, np.ndarray]) -> Path:
    final = final_root / "cells" / task / arm / f"s{seed}"
    require(not final.exists(), f"refusing to overwrite formal cell {final}")
    stage = staging / task / arm / f"s{seed}"
    require(not stage.exists(), f"stale staging cell exists: {stage}")
    stage.mkdir(parents=True, exist_ok=False)
    history_path = stage / "history.csv"
    fields = ("epoch", "loss", "visual_loss", "proprio_loss", "lr", "seconds")
    with history_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    predictions_path = stage / "validation_predictions.npz"
    np.savez_compressed(predictions_path, **arrays)
    metrics_path = stage / "metrics.json"
    atomic_json(metrics_path, metrics)
    checkpoint_path = stage / "carrier.pt"
    torch.save({"carrier_state_dict": carrier.state_dict(),
                "metrics": metrics}, checkpoint_path)
    manifest = {
        "schema": "dinowm_wave2_spatial_cell_manifest_v1",
        "protocol_sha256": lock["protocol_sha256"],
        "task": task, "arm": arm, "seed": seed,
        "artifacts": {
            path.name: {"size": path.stat().st_size,
                        "sha256": sha256_file(path)}
            for path in (history_path, predictions_path, metrics_path,
                         checkpoint_path)
        },
    }
    atomic_json(stage / "manifest.json", manifest)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.rename(stage, final)
    return final


def evaluate_cell(host: FrozenNativeHost, carrier: torch.nn.Module,
                  bank: FeatureBank, task_record: Mapping[str, Any],
                  cfg: Mapping[str, Any]
                  ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    task = str(task_record["key"])
    train = collect_features(host, carrier, bank, task, "train", cfg)
    validation = collect_features(
        host, carrier, bank, task, "validation", cfg)
    train_indices = bank.indices("train")
    validation_indices = bank.indices("validation")
    return fit_readouts(
        train, validation, bank.labels[task][train_indices],
        bank.labels[task][validation_indices], cfg, int(task_record["classes"]))


def run_cell(host: FrozenNativeHost, bank: FeatureBank,
             task_record: Mapping[str, Any], arm: str, seed: int,
             cfg: Mapping[str, Any], lock: Mapping[str, Any],
             formal: Path) -> Path:
    task = str(task_record["key"])
    configure_cuda(cfg, seed)
    host.verify_schema(load_encoder=False)
    host_before = host.digest()
    torch.cuda.reset_peak_memory_stats(host.device)
    started = time.time()
    carrier = make_frozen_carrier(arm, 384, 10).to(host.device)
    if carrier.parameter_count():
        history, schedule = train_carrier(
            host, carrier, bank, task, seed, cfg)
    else:
        history = []
        schedule = common_schedule_digest(cfg, seed, 1200)
    readout, arrays = evaluate_cell(
        host, carrier, bank, task_record, cfg)
    host_after = host.digest()
    require(host_before == host_after,
            f"frozen official host changed in {task}/{arm}/s{seed}")
    if arm == "none":
        for age in cfg["sequence"]["evidence_ages"]:
            require(np.array_equal(
                arrays[f"age_{age}_full_prediction"],
                arrays[f"age_{age}_reset_prediction"]),
                "none full/reset predictions differ")
            require(abs(readout[str(age)]["full_next_visual_mse"]
                        - readout[str(age)]["reset_next_visual_mse"]) <= 1e-12,
                    "none full/reset MSE differs")
    losses = [float(row["loss"]) for row in history]
    convergence = None
    if len(losses) >= 5:
        denominator = max(abs(losses[-5]), 1e-12)
        convergence = float((losses[-1] - losses[-5]) / denominator)
    metrics = {
        "schema": "dinowm_wave2_spatial_cell_v1",
        "protocol_sha256": lock["protocol_sha256"],
        "task": task,
        "semantic_name": task_record["semantic_name"],
        "classes": int(task_record["classes"]),
        "arm": arm,
        "seed": seed,
        "adaptive_opened_bank": True,
        "physical_gpu": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "host_digest_before": host_before,
        "host_digest_after": host_after,
        "host_unchanged": True,
        "carrier_parameters": carrier.parameter_count(),
        "parameter_matching": parameter_report(384, 10),
        "carrier_config": carrier.describe(),
        "carrier_scope": cfg["adapter"]["carrier_scope"],
        "training_labels_used": False,
        "objective": cfg["objective"],
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
        formal, formal / ".staging", task=task, arm=arm, seed=seed,
        cfg=cfg, lock=lock, carrier=carrier, history=history,
        metrics=metrics, arrays=arrays)


def clone_none_cell(source: Path, host: FrozenNativeHost,
                    task_record: Mapping[str, Any], seed: int,
                    cfg: Mapping[str, Any], lock: Mapping[str, Any],
                    formal: Path) -> Path:
    task = str(task_record["key"])
    source_metrics = json.loads((source / "metrics.json").read_text())
    with np.load(source / "validation_predictions.npz") as values:
        arrays = {name: values[name] for name in values.files}
    metrics = dict(source_metrics)
    metrics.update({
        "seed": seed,
        "common_schedule_sha256": common_schedule_digest(cfg, seed, 1200),
        "duplicated_deterministic_no_carrier_from_seed": 0,
        "effective_independent_models": 1,
    })
    carrier = make_frozen_carrier("none", 384, 10).to(host.device)
    return write_cell(
        formal, formal / ".staging", task=task, arm="none", seed=seed,
        cfg=cfg, lock=lock, carrier=carrier, history=[], metrics=metrics,
        arrays=arrays)


def validate_cell(directory: Path, task: str, arm: str, seed: int,
                  lock: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    paths = {name: directory / name for name in (
        "manifest.json", "metrics.json", "carrier.pt", "history.csv",
        "validation_predictions.npz")}
    require(all(path.is_file() for path in paths.values()),
            f"incomplete formal cell {directory}")
    manifest = json.loads(paths["manifest.json"].read_text())
    metrics = json.loads(paths["metrics.json"].read_text())
    require(manifest.get("protocol_sha256") == lock["protocol_sha256"]
            and metrics.get("protocol_sha256") == lock["protocol_sha256"],
            f"cell lock mismatch: {directory}")
    require((metrics["task"], metrics["arm"], int(metrics["seed"]))
            == (task, arm, seed), f"cell identity mismatch: {directory}")
    require(metrics.get("host_unchanged") is True
            and metrics["host_digest_before"] == metrics["host_digest_after"],
            f"cell changed host: {directory}")
    for name, record in manifest["artifacts"].items():
        path = directory / name
        require(path.stat().st_size == record["size"]
                and sha256_file(path) == record["sha256"],
                f"cell artifact hash mismatch: {path}")
    checkpoint = torch.load(paths["carrier.pt"], map_location="cpu",
                            weights_only=True)
    require(checkpoint.get("metrics") == metrics
            and isinstance(checkpoint.get("carrier_state_dict"), Mapping),
            f"checkpoint content mismatch: {directory}")
    with np.load(paths["validation_predictions.npz"]) as values:
        arrays = {name: values[name] for name in values.files}
    require(arrays["truth"].shape == (480,),
            f"validation truth shape changed: {directory}")
    return metrics, arrays


def aggregate(formal: Path, cfg: Mapping[str, Any],
              lock: Mapping[str, Any]) -> dict[str, Any]:
    arms = list(cfg["training"]["arms"])
    seeds = [int(value) for value in cfg["training"]["seeds"]]
    ages = [int(value) for value in cfg["sequence"]["evidence_ages"]]
    loaded: dict[tuple[str, str, int], tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    for task in cfg["tasks"]:
        key = task["key"]
        for arm in arms:
            for seed in seeds:
                directory = formal / "cells" / key / arm / f"s{seed}"
                loaded[(key, arm, seed)] = validate_cell(
                    directory, key, arm, seed, lock)

    results: dict[str, Any] = {}
    bootstrap = cfg["inference"]
    for task_index, task in enumerate(cfg["tasks"]):
        key, classes = task["key"], int(task["classes"])
        task_result = {
            "semantic_name": task["semantic_name"],
            "classes": classes, "chance": 1.0 / classes,
            "ages": {},
        }
        truth = loaded[(key, "none", 0)][1]["truth"]
        for age_index, age in enumerate(ages):
            age_result = {"arms": {}, "paired_vs_none": {},
                          "full_vs_context_reset": {}}
            predictions = {}
            resets = {}
            for arm_index, arm in enumerate(arms):
                predictions[arm] = np.stack([
                    loaded[(key, arm, seed)][1][
                        f"age_{age}_full_prediction"] for seed in seeds])
                resets[arm] = np.stack([
                    loaded[(key, arm, seed)][1][
                        f"age_{age}_reset_prediction"] for seed in seeds])
                seed_offset = task_index * 1000 + age_index * 100 + arm_index
                absolute = absolute_bootstrap(
                    predictions[arm], truth, classes=classes,
                    draws=int(bootstrap["draws"]),
                    seed=int(bootstrap["seed"]) + seed_offset,
                    confidence=float(bootstrap["confidence"]))
                mse = [loaded[(key, arm, seed)][0]["ages"][str(age)][
                    "full_next_visual_mse"] for seed in seeds]
                prior = [loaded[(key, arm, seed)][0]["ages"][str(age)][
                    "prior_balanced_accuracy"] for seed in seeds]
                age_result["arms"][arm] = {
                    "balanced_accuracy": absolute,
                    "seed_values": [balanced_accuracy_from_predictions(
                        predictions[arm][index], truth, classes)
                        for index in range(len(seeds))],
                    "prior_seed_values": prior,
                    "next_visual_mse_seed_values": mse,
                    "parameters": loaded[(key, arm, 0)][0][
                        "carrier_parameters"],
                    "effective_independent_models": 1 if arm == "none" else 5,
                }
            for arm_index, arm in enumerate(arms):
                if arm != "none":
                    age_result["paired_vs_none"][arm] = crossed_paired_bootstrap(
                        predictions[arm], predictions["none"], truth,
                        classes=classes, draws=int(bootstrap["draws"]),
                        seed=int(bootstrap["seed"]) + 5000
                        + task_index * 1000 + age_index * 100 + arm_index,
                        confidence=float(bootstrap["confidence"]))
                age_result["full_vs_context_reset"][arm] = \
                    crossed_paired_bootstrap(
                        predictions[arm], resets[arm], truth,
                        classes=classes, draws=int(bootstrap["draws"]),
                        seed=int(bootstrap["seed"]) + 10000
                        + task_index * 1000 + age_index * 100 + arm_index,
                        confidence=float(bootstrap["confidence"]))
            task_result["ages"][str(age)] = age_result
        results[key] = task_result

    summary = {
        "schema": "dinowm_wave2_spatial_carrier_summary_v1",
        "status": "complete",
        "protocol_sha256": lock["protocol_sha256"],
        "study": cfg["study"],
        "scope": cfg["scope"],
        "host": cfg["checkpoint"]["display_name"],
        "adapter": cfg["adapter"],
        "parameter_matching": parameter_report(384, 10),
        "grid": {"tasks": 2, "arms": 5, "seeds": 5, "cells": 50},
        "inference": cfg["inference"],
        "results": results,
    }
    atomic_json(formal / "summary.json", summary)
    lines = [
        "# DINO-WM Wave 2 spatial-carrier summary", "",
        "Adaptive extension on the opened native PushT bank; all intervals use "
        "20,000 matched-seed × class-stratified held-out-episode bootstrap draws.",
        "",
    ]
    for task in cfg["tasks"]:
        record = results[task["key"]]
        lines += [f"## {task['semantic_name']}", "",
                  "| age | arm | full balanced accuracy [95% CI] | Δ vs none [95% CI] | Δ full-reset [95% CI] |",
                  "|---:|---|---:|---:|---:|"]
        for age in ages:
            age_record = record["ages"][str(age)]
            for arm in arms:
                value = age_record["arms"][arm]["balanced_accuracy"]
                if arm == "none":
                    delta = "--"
                else:
                    pair = age_record["paired_vs_none"][arm]
                    delta = (f"{pair['mean']:+.3f} "
                             f"[{pair['ci95'][0]:+.3f},{pair['ci95'][1]:+.3f}]")
                reset = age_record["full_vs_context_reset"][arm]
                lines.append(
                    f"| {age} | {arm} | {value['mean']:.3f} "
                    f"[{value['ci95'][0]:.3f},{value['ci95'][1]:.3f}] | "
                    f"{delta} | {reset['mean']:+.3f} "
                    f"[{reset['ci95'][0]:+.3f},{reset['ci95'][1]:+.3f}] |")
        lines.append("")
    (formal / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def run_formal(config_path: Path, cfg: Mapping[str, Any],
               lock: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    output_root = resolve(cfg["artifacts"]["root"])
    formal = output_root / cfg["artifacts"]["formal"]
    if resume:
        require(formal.is_dir() and (formal / "provenance.json").is_file(),
                "resume requested without an existing formal run")
        provenance = json.loads((formal / "provenance.json").read_text())
        require(provenance.get("protocol_sha256") == lock["protocol_sha256"],
                "existing formal run belongs to another lock")
    else:
        require(not formal.exists(), "formal directory exists; use --resume")
        formal.mkdir(parents=True, exist_ok=False)
        (formal / ".staging").mkdir()
        (formal / "failures").mkdir()
        provenance = {
            "schema": "dinowm_wave2_spatial_provenance_v1",
            "status": "running",
            "protocol_path": str(config_path.relative_to(ROOT)),
            "protocol_sha256": lock["protocol_sha256"],
            "source_sha256": lock["source_sha256"],
            "started_unix": time.time(),
            "physical_gpu": 1,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "paper_modified_by_wave2": False,
            "adaptive_opened_bank": True,
            "pins": verify_pins(cfg),
            "admissions": validate_prior_admissions(cfg),
            "environment": {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "disk_before": shutil.disk_usage(ROOT)._asdict(),
            },
        }
        atomic_json(formal / "provenance.json", provenance)
        atomic_json(formal / "admissions.json", provenance["admissions"])

    bank = FeatureBank(cfg, lock)
    host = FrozenNativeHost(cfg, load_encoder=False)
    initial_host = host.digest()
    provenance["runtime_host_digest"] = initial_host
    provenance["environment"].update({
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "logical_device": str(host.device),
        "deterministic_algorithms":
            torch.are_deterministic_algorithms_enabled(),
    })
    atomic_json(formal / "provenance.json", provenance)
    expected = [(task, arm, seed)
                for task in cfg["tasks"]
                for arm in cfg["training"]["arms"]
                for seed in cfg["training"]["seeds"]]
    completed = []
    try:
        for task_record, arm, seed in expected:
            task = task_record["key"]
            final = formal / "cells" / task / arm / f"s{seed}"
            if final.exists():
                validate_cell(final, task, arm, int(seed), lock)
                completed.append([task, arm, int(seed)])
                continue
            print(f"[wave2-formal] start {task}/{arm}/s{seed}", flush=True)
            if arm == "none" and int(seed) > 0:
                source = formal / "cells" / task / "none" / "s0"
                require(source.is_dir(), "none seed 0 must precede duplicates")
                run_path = clone_none_cell(
                    source, host, task_record, int(seed), cfg, lock, formal)
            else:
                run_path = run_cell(
                    host, bank, task_record, arm, int(seed), cfg, lock, formal)
            validate_cell(run_path, task, arm, int(seed), lock)
            require(host.digest() == initial_host,
                    "frozen host changed between formal cells")
            completed.append([task, arm, int(seed)])
            atomic_json(formal / "progress.json", {
                "protocol_sha256": lock["protocol_sha256"],
                "completed_cells": completed,
                "count": len(completed), "expected": 50,
                "updated_unix": time.time(),
            })
            print(f"[wave2-formal] complete {task}/{arm}/s{seed} "
                  f"({len(completed)}/50)", flush=True)
        require(len(completed) == 50, "formal grid is incomplete")
        summary = aggregate(formal, cfg, lock)
        provenance.update({
            "status": "complete",
            "completed_unix": time.time(),
            "elapsed_seconds": time.time() - provenance["started_unix"],
            "runtime_host_digest_after": host.digest(),
        })
        require(provenance["runtime_host_digest_after"] == initial_host,
                "host changed by the completed formal grid")
        provenance["environment"]["disk_after"] = \
            shutil.disk_usage(ROOT)._asdict()
        atomic_json(formal / "provenance.json", provenance)
        return summary
    except Exception as error:
        failure = {
            "schema": "dinowm_wave2_formal_stop_v1",
            "status": "stopped_fail_closed",
            "reason": repr(error),
            "completed_cells": completed,
            "count": len(completed),
            "protocol_sha256": lock["protocol_sha256"],
            "no_post_hoc_adaptation": True,
            "stopped_unix": time.time(),
        }
        atomic_json(formal / "stop_receipt.json", failure)
        provenance["status"] = "stopped_fail_closed"
        provenance["stop_receipt"] = failure
        atomic_json(formal / "provenance.json", provenance)
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
        value = run_smoke(config_path, cfg)
    elif args.seal:
        require(not args.execute, "seal is non-metric and does not take --execute")
        cfg, _ = load_config(config_path, locked=False)
        value = seal_protocol(config_path, cfg)
    elif args.prepare:
        require(args.execute, "cache preparation requires --execute")
        cfg, lock = load_config(config_path, locked=True)
        assert lock is not None
        value = prepare_cache(config_path, cfg, lock)
    else:
        require(args.execute, "formal run requires --execute")
        cfg, lock = load_config(config_path, locked=True)
        assert lock is not None
        value = run_formal(
            config_path, cfg, lock, resume=bool(args.resume))
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
