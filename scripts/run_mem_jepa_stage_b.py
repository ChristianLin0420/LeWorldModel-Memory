#!/usr/bin/env python3
"""Stage-B host-exposure smoke test for Masked-Evidence JEPA memory.

This runner uses the locked DINO-WM PointMaze cache/checkpoint from Wave 3 but
keeps the exploratory Mem-JEPA implementation isolated.  The current Stage-B
loss is supervised by the counterfactual class label, so the result is an
interface-capacity smoke test, not yet a label-free paper claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.dinowm_pointmaze import (  # noqa: E402
    endpoint_frame,
    predictor_context_for_endpoint,
)
from lewm.official_tasks.dinowm_spatial_carrier import (  # noqa: E402
    balanced_accuracy_from_predictions,
)


DEFAULT_CONFIG = ROOT / "configs/dinowm_pointmaze_wave3.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/mem_jepa_stage_b"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    require(isinstance(cfg, dict), "config did not parse as a mapping")
    return cfg


def expanded_labels(base_count: int) -> np.ndarray:
    return np.tile(np.arange(4, dtype=np.int64), int(base_count))


def classification_record(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(truth, prediction, labels=np.arange(4))
    recall = np.diag(matrix) / np.maximum(matrix.sum(1), 1)
    return {
        "balanced_accuracy": balanced_accuracy_from_predictions(prediction, truth, 4),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": matrix.tolist(),
        "count": int(len(truth)),
    }


def compact_pyramid_pool(patches: np.ndarray) -> np.ndarray:
    """Fast 1x1 + 2x2 patch-grid pooling for exploratory Stage-B reads."""

    values = np.asarray(patches, dtype=np.float32)
    require(values.ndim >= 3 and values.shape[-2:] == (196, 384),
            "patches must end in 196x384")
    grid = values.reshape(*values.shape[:-2], 14, 14, 384)
    cells = [grid.mean(axis=(-3, -2))]
    for row in (slice(0, 7), slice(7, 14)):
        for col in (slice(0, 7), slice(7, 14)):
            cells.append(grid[..., row, col, :].mean(axis=(-3, -2)))
    return np.concatenate(cells, axis=-1)


def fit_classifier(train_x: np.ndarray, train_y: np.ndarray,
                   validation_x: np.ndarray) -> np.ndarray:
    classifier = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=1.0),
    )
    classifier.fit(train_x, train_y)
    return classifier.predict(validation_x).astype(np.int64)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


class FeatureBank:
    """Compact reader for the locked DINO-WM PointMaze feature cache."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.root = resolve(cfg["artifacts"]["root"]) / "cache"
        require((self.root / "manifest.json").is_file(),
                f"missing DINO-WM PointMaze cache: {self.root}")
        self.base_visual = np.load(self.root / "base_visual.npy", mmap_mode="r")
        self.cue_visual = np.load(self.root / "cue_visual.npy", mmap_mode="r")
        metadata = np.load(self.root / "metadata.npz")
        self.actions = np.asarray(metadata["actions"], dtype=np.float32)
        self.proprio = np.asarray(metadata["proprio"], dtype=np.float32)
        self.split = np.asarray(metadata["split"], dtype=np.uint8)
        train_count = int(cfg["dataset"]["train_base_windows"])
        val_count = int(cfg["dataset"]["validation_base_windows"])
        base_count = train_count + val_count
        require(self.base_visual.shape == (base_count, 20, 196, 384),
                "base visual cache shape changed")
        require(self.cue_visual.shape == (base_count, 4, 3, 196, 384),
                "cue visual cache shape changed")
        require(self.actions.shape == (base_count, 19, 10),
                "action cache shape changed")
        require(self.proprio.shape == (base_count, 20, 4),
                "proprio cache shape changed")
        require(np.count_nonzero(self.split == 0) == train_count
                and np.count_nonzero(self.split == 1) == val_count,
                "cache split counts changed")

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


class FrozenPointMazeHost:
    """Relaxed-GPU loader for the frozen official DINO-WM PointMaze predictor."""

    def __init__(self, cfg: Mapping[str, Any], device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        vendor = resolve(cfg["source"]["dino_wm"]["repo_path"])
        shim = ROOT / "outputs/dinowm_native_pusht_audit_v1/shims"
        if shim.is_dir():
            sys.path.insert(0, str(shim))
        sys.path.insert(0, str(vendor))
        weights = resolve(cfg["checkpoint"]["weights_path"])
        require(weights.is_file(), f"missing frozen DINO-WM checkpoint: {weights}")
        payload = torch.load(weights, map_location="cpu", weights_only=False)
        require(set(payload) == {
            "epoch", "predictor", "predictor_optimizer", "decoder",
            "decoder_optimizer", "action_encoder", "proprio_encoder",
        }, "released checkpoint schema changed")
        require(int(payload["epoch"]) == int(cfg["checkpoint"]["expected_epoch_field"]),
                "released checkpoint epoch changed")
        self.predictor = payload["predictor"].eval().to(device)
        self.action_encoder = payload["action_encoder"].eval().to(device)
        self.proprio_encoder = payload["proprio_encoder"].eval().to(device)
        del payload
        moved = 0
        for module in self.predictor.modules():
            bias = getattr(module, "bias", None)
            if torch.is_tensor(bias) and bias.ndim == 4:
                module.bias = bias.to(device)
                moved += 1
        require(moved == 6, f"expected six attention masks, moved {moved}")
        for module in self.modules.values():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.verify_schema()

    @property
    def modules(self) -> dict[str, torch.nn.Module]:
        return {
            "predictor": self.predictor,
            "action_encoder": self.action_encoder,
            "proprio_encoder": self.proprio_encoder,
        }

    def verify_schema(self) -> None:
        require(tuple(self.predictor.pos_embedding.shape) == (1, 588, 404),
                "PointMaze predictor token shape changed")
        require(tuple(self.action_encoder.patch_embed.weight.shape) == (10, 10, 1),
                "PointMaze action encoder changed")
        require(tuple(self.proprio_encoder.patch_embed.weight.shape) == (10, 4, 1),
                "PointMaze proprio encoder changed")
        require(all(not p.requires_grad for m in self.modules.values()
                    for p in m.parameters()), "host is not frozen")

    def compose(self, visual: torch.Tensor, proprio: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        require(visual.ndim == 4 and visual.shape[2:] == (196, 384),
                "visual context violates native shape")
        require(proprio.shape == (*visual.shape[:2], 4),
                "proprio context violates native shape")
        require(actions.shape == (*visual.shape[:2], 10),
                "action context violates native shape")
        prop = self.proprio_encoder(proprio).unsqueeze(2).expand(-1, -1, 196, -1)
        action = self.action_encoder(actions).unsqueeze(2).expand(-1, -1, 196, -1)
        return torch.cat((visual, prop, action), dim=-1)

    def predict(self, visual: torch.Tensor, proprio: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        context = self.compose(visual, proprio, actions)
        batch, steps, patches, dim = context.shape
        require((steps, patches, dim) == (3, 196, 404),
                "native predictor requires 3x196x404")
        return self.predictor(context.reshape(batch, steps * patches, dim)).reshape(
            batch, steps, patches, dim)

    def digest(self) -> str:
        digest = hashlib.sha256()
        for module_name, module in sorted(self.modules.items()):
            for name, value in sorted(module.state_dict().items()):
                digest.update(module_name.encode())
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


class MemJepaExposureAdapter(nn.Module):
    """Compact slot memory that writes a residual into the frozen host context."""

    def __init__(self, *, dim: int, slots: int, heads: int, max_frames: int = 20,
                 residual_scale: float = 1.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.slots = int(slots)
        self.visual_proj = nn.Linear(384, dim)
        self.action_proj = nn.Linear(10, dim)
        self.time = nn.Embedding(max_frames, dim)
        self.patch = nn.Parameter(torch.empty(196, dim))
        self.assign = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, slots),
        )
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.residual = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 384),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )
        self.memory_classifier = nn.Sequential(
            nn.LayerNorm(slots * dim),
            nn.Linear(slots * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )
        self.context_classifier = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        nn.init.normal_(self.patch, std=0.02)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def _tokens(self, visual: torch.Tensor, actions: torch.Tensor,
                times: torch.Tensor) -> torch.Tensor:
        batch, steps, patches, _ = visual.shape
        require((patches, actions.shape[:2]) == (196, (batch, steps)),
                "visual/action token shapes are inconsistent")
        time = self.time(times.to(visual.device))[None, :, None, :]
        patch = self.patch[None, None, :, :]
        action = self.action_proj(actions)[:, :, None, :]
        return self.visual_proj(visual) + time + patch + action

    def encode_memory(self, visual: torch.Tensor, actions: torch.Tensor,
                      times: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(visual, actions, times).reshape(
            visual.shape[0], -1, self.dim)
        weights = torch.softmax(self.assign(tokens).transpose(1, 2), dim=-1)
        return weights @ tokens

    def inject(self, prefix_visual: torch.Tensor, prefix_actions: torch.Tensor,
               prefix_times: torch.Tensor, context_visual: torch.Tensor,
               context_actions: torch.Tensor,
               context_times: torch.Tensor, *,
               return_slots: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        slots = self.encode_memory(prefix_visual, prefix_actions, prefix_times)
        queries = self._tokens(context_visual, context_actions, context_times).reshape(
            context_visual.shape[0], -1, self.dim)
        read, _ = self.cross(queries, slots, slots, need_weights=False)
        residual = self.residual(queries + read).reshape_as(context_visual)
        fused = context_visual + self.residual_scale * residual
        return (fused, slots) if return_slots else fused

    def classify(self, host_visual: torch.Tensor) -> torch.Tensor:
        return self.classifier(host_visual.mean(dim=1))

    def classify_memory(self, slots: torch.Tensor) -> torch.Tensor:
        return self.memory_classifier(slots.reshape(slots.shape[0], -1))

    def classify_context(self, context_visual: torch.Tensor) -> torch.Tensor:
        return self.context_classifier(context_visual.mean(dim=(1, 2)))


def tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)


def batch_arrays(bank: FeatureBank, expanded: np.ndarray, age: int,
                 condition: str, device: torch.device) -> dict[str, torch.Tensor]:
    endpoint = endpoint_frame(3, int(age))
    context = predictor_context_for_endpoint(endpoint)
    start, stop = context[0], context[-1] + 1
    bases, labels = bank.decode_expanded(expanded)
    visual = bank.visual(expanded)
    actions = np.asarray(bank.actions[bases], dtype=np.float32)
    if condition == "full":
        prefix_visual = visual[:, :stop]
        prefix_actions = actions[:, :stop]
        prefix_times = np.arange(stop, dtype=np.int64)
    elif condition == "reset":
        prefix_visual = visual[:, start:stop]
        prefix_actions = actions[:, start:stop]
        prefix_times = np.arange(start, stop, dtype=np.int64)
    else:
        raise ValueError(f"unknown memory condition {condition}")
    return {
        "prefix_visual": tensor(prefix_visual, device),
        "prefix_actions": tensor(prefix_actions, device),
        "prefix_times": torch.from_numpy(prefix_times).long().to(device),
        "context_visual": tensor(visual[:, start:stop], device),
        "context_actions": tensor(actions[:, start:stop], device),
        "context_times": torch.arange(start, stop, device=device).long(),
        "proprio_context": tensor(bank.proprio[bases, start:stop], device),
        "labels": torch.from_numpy(labels.astype(np.int64)).to(device),
    }


def no_state_arrays(bank: FeatureBank, expanded: np.ndarray, age: int,
                    device: torch.device) -> dict[str, torch.Tensor]:
    endpoint = endpoint_frame(3, int(age))
    context = predictor_context_for_endpoint(endpoint)
    start, stop = context[0], context[-1] + 1
    bases, labels = bank.decode_expanded(expanded)
    visual = bank.visual(expanded)
    return {
        "context_visual": tensor(visual[:, start:stop], device),
        "context_actions": tensor(bank.actions[bases, start:stop], device),
        "proprio_context": tensor(bank.proprio[bases, start:stop], device),
        "labels": torch.from_numpy(labels.astype(np.int64)).to(device),
    }


def train_one_age(model: MemJepaExposureAdapter, host: FrozenPointMazeHost,
                  bank: FeatureBank, *, age: int, seed: int, epochs: int,
                  batch_size: int, lr: float, weight_decay: float,
                  output_dir: Path) -> list[dict[str, Any]]:
    train_indices = bank.expanded_indices("train").copy()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs))
    history: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    csv_path = output_dir / f"age_{age}_history.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["epoch", "loss", "host_ce", "memory_ce",
                                "context_ce", "residual_l2",
                                "host_accuracy", "memory_accuracy",
                                "context_accuracy", "lr", "seconds"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            started = time.time()
            rng.shuffle(train_indices)
            losses, host_ces, memory_ces, context_ces, reg_values = [], [], [], [], []
            host_correct, memory_correct, context_correct, total = 0, 0, 0, 0
            model.train()
            for offset in range(0, len(train_indices), batch_size):
                expanded = train_indices[offset:offset + batch_size]
                batch = batch_arrays(bank, expanded, age, "full", host.device)
                fused, slots = model.inject(
                    batch["prefix_visual"], batch["prefix_actions"],
                    batch["prefix_times"], batch["context_visual"],
                    batch["context_actions"], batch["context_times"],
                    return_slots=True)
                predicted = host.predict(
                    fused, batch["proprio_context"],
                    batch["context_actions"])[:, -1, :, :384]
                logits = model.classify(predicted)
                memory_logits = model.classify_memory(slots)
                context_logits = model.classify_context(fused)
                host_ce = F.cross_entropy(logits, batch["labels"])
                memory_ce = F.cross_entropy(memory_logits, batch["labels"])
                context_ce = F.cross_entropy(context_logits, batch["labels"])
                residual_l2 = torch.mean(torch.square(fused - batch["context_visual"]))
                loss = host_ce + 0.5 * memory_ce + 0.5 * context_ce \
                    + 1.0e-4 * residual_l2
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                host_ces.append(float(host_ce.detach().cpu()))
                memory_ces.append(float(memory_ce.detach().cpu()))
                context_ces.append(float(context_ce.detach().cpu()))
                reg_values.append(float(residual_l2.detach().cpu()))
                host_correct += int((logits.argmax(dim=1) == batch["labels"]).sum().cpu())
                memory_correct += int(
                    (memory_logits.argmax(dim=1) == batch["labels"]).sum().cpu())
                context_correct += int(
                    (context_logits.argmax(dim=1) == batch["labels"]).sum().cpu())
                total += int(batch["labels"].numel())
            scheduler.step()
            record = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "host_ce": float(np.mean(host_ces)),
                "memory_ce": float(np.mean(memory_ces)),
                "context_ce": float(np.mean(context_ces)),
                "residual_l2": float(np.mean(reg_values)),
                "host_accuracy": float(host_correct / max(total, 1)),
                "memory_accuracy": float(memory_correct / max(total, 1)),
                "context_accuracy": float(context_correct / max(total, 1)),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": float(time.time() - started),
            }
            history.append(record)
            writer.writerow(record)
            stream.flush()
            print(
                f"[mem-jepa-stage-b] age={age} epoch={epoch}/{epochs} "
                f"loss={record['loss']:.4f} "
                f"host={record['host_accuracy']:.3f} "
                f"mem={record['memory_accuracy']:.3f} "
                f"ctx={record['context_accuracy']:.3f} "
                f"sec={record['seconds']:.1f}",
                flush=True,
            )
    return history


@torch.no_grad()
def collect_features(model: MemJepaExposureAdapter, host: FrozenPointMazeHost,
                     bank: FeatureBank, *, split: str, age: int,
                     condition: str, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    indices = bank.expanded_indices(split)
    features, direct_predictions, truth = [], [], []
    for offset in range(0, len(indices), batch_size):
        expanded = indices[offset:offset + batch_size]
        if condition == "no_state":
            batch = no_state_arrays(bank, expanded, age, host.device)
            predicted = host.predict(
                batch["context_visual"], batch["proprio_context"],
                batch["context_actions"])[:, -1, :, :384]
        else:
            batch = batch_arrays(bank, expanded, age, condition, host.device)
            fused = model.inject(
                batch["prefix_visual"], batch["prefix_actions"],
                batch["prefix_times"], batch["context_visual"],
                batch["context_actions"], batch["context_times"])
            predicted = host.predict(
                fused, batch["proprio_context"],
                batch["context_actions"])[:, -1, :, :384]
        logits = model.classify(predicted)
        direct_predictions.append(logits.argmax(dim=1).cpu().numpy())
        truth.append(batch["labels"].cpu().numpy())
        features.append(compact_pyramid_pool(predicted.float().cpu().numpy()))
    return {
        "features": np.concatenate(features),
        "direct_prediction": np.concatenate(direct_predictions).astype(np.int64),
        "truth": np.concatenate(truth).astype(np.int64),
    }


def evaluate_one_age(model: MemJepaExposureAdapter, host: FrozenPointMazeHost,
                     bank: FeatureBank, *, age: int,
                     batch_size: int) -> dict[str, Any]:
    train_full = collect_features(
        model, host, bank, split="train", age=age, condition="full",
        batch_size=batch_size)
    truth = collect_features(
        model, host, bank, split="validation", age=age, condition="full",
        batch_size=batch_size)
    reset = collect_features(
        model, host, bank, split="validation", age=age, condition="reset",
        batch_size=batch_size)
    no_state = collect_features(
        model, host, bank, split="validation", age=age, condition="no_state",
        batch_size=batch_size)
    train_y = train_full["truth"]
    validation_y = truth["truth"]
    records = {}
    for name, pack in {"full": truth, "reset": reset,
                       "no_state": no_state}.items():
        prediction = fit_classifier(
            train_full["features"], train_y, pack["features"])
        records[name] = classification_record(prediction, validation_y)
        records[name]["direct_classifier"] = classification_record(
            pack["direct_prediction"], validation_y)
    gate = {
        "full_minimum": 0.75,
        "control_maximum": 0.40,
        "passed": bool(
            records["full"]["balanced_accuracy"] >= 0.75
            and records["reset"]["balanced_accuracy"] <= 0.40
            and records["no_state"]["balanced_accuracy"] <= 0.40
        ),
    }
    return {"records": records, "gate": gate}


def run_age(args: argparse.Namespace) -> dict[str, Any]:
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(resolve(args.config))
    set_determinism(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    bank = FeatureBank(cfg)
    host = FrozenPointMazeHost(cfg, device)
    host_before = host.digest()
    model = MemJepaExposureAdapter(
        dim=args.dim, slots=args.slots, heads=args.heads,
        residual_scale=args.residual_scale).to(device)
    started = time.time()
    history = train_one_age(
        model, host, bank, age=args.age, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        output_dir=output)
    metrics = evaluate_one_age(
        model, host, bank, age=args.age, batch_size=args.eval_batch_size)
    host_after = host.digest()
    require(host_before == host_after, "frozen host digest changed")
    result = {
        "schema": "mem_jepa_stage_b_age_v1",
        "status": "completed",
        "claim_boundary": (
            "supervised host-exposure capacity smoke; labels are used in the "
            "adapter loss, so this is not yet a label-free paper claim"),
        "labels_used_for_adapter_training": True,
        "age": int(args.age),
        "endpoint_frame": endpoint_frame(3, int(args.age)),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "config": str(resolve(args.config).relative_to(ROOT)),
        "config_sha256": sha256_file(resolve(args.config)),
        "host_digest_unchanged": True,
        "host_digest": host_after,
        "model": {
            "dim": int(args.dim),
            "slots": int(args.slots),
            "heads": int(args.heads),
            "residual_scale_init": float(args.residual_scale),
            "parameters": int(sum(p.numel() for p in model.parameters())),
        },
        "training": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "final": history[-1] if history else None,
        },
        "metrics": metrics,
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_json(output / f"age_{args.age}.json", result)
    print(json.dumps({
        "age": args.age,
        "passed": metrics["gate"]["passed"],
        "full": metrics["records"]["full"]["balanced_accuracy"],
        "reset": metrics["records"]["reset"]["balanced_accuracy"],
        "no_state": metrics["records"]["no_state"]["balanced_accuracy"],
    }, indent=2), flush=True)
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = resolve(args.output)
    ages = [int(age) for age in args.ages]
    cells = {}
    all_passed = True
    for age in ages:
        path = output / f"age_{age}.json"
        require(path.is_file(), f"missing age result: {path}")
        value = json.loads(path.read_text())
        cells[str(age)] = value
        all_passed = all_passed and bool(value["metrics"]["gate"]["passed"])
    summary = {
        "schema": "mem_jepa_stage_b_summary_v1",
        "status": "completed" if all_passed else "completed_with_failed_gate",
        "claim_boundary": (
            "Stage B currently demonstrates supervised exposure capacity only. "
            "Proceed to a label-free masked-evidence objective before using it "
            "as a paper-level memory claim."),
        "labels_used_for_adapter_training": True,
        "all_gates_passed": bool(all_passed),
        "ages": ages,
        "cells": cells,
        "updated_unix": time.time(),
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "all_gates_passed": summary["all_gates_passed"],
        "ages": ages,
    }, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--age", type=int, choices=[4, 8, 15])
    parser.add_argument("--ages", type=int, nargs="*", default=[4, 8, 15])
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--seed", type=int, default=9400)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.aggregate and args.age is None:
        parser.error("--age is required unless --aggregate is used")
    return args


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_age(args)


if __name__ == "__main__":
    main()
