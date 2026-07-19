#!/usr/bin/env python3
"""PushT Stage-F label-free Masked-Evidence JEPA memory test.

This is the next gated experiment after the positive PointMaze Mem-JEPA
stages.  It reuses the locked DINO-WM PushT cache but does not train on
semantic labels.  The adapter is trained to preserve masked old evidence from
the cue frames using batch negatives; labels are used only for the final
retention audit readout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.official_tasks.dinowm_spatial_carrier import (  # noqa: E402
    balanced_accuracy_from_predictions,
    endpoint_frame,
    predictor_context_for_endpoint,
)
from scripts.run_mem_jepa_stage_b import (  # noqa: E402
    atomic_json,
    fit_classifier,
    require,
    resolve,
    set_determinism,
    sha256_file,
)
from scripts.run_mem_jepa_stage_c import (  # noqa: E402
    MemJepaLabelFreeAdapter,
    contrastive_loss,
    positive_cosine_loss,
)
from scripts.run_dinowm_wave2_spatial_carrier import make_visual_batch  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/dinowm_wave2_spatial_carrier_v1_1.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/mem_jepa_pusht_stage_f"
TARGET_DIM = 5 * 384
BINDING_ROW_SLOTS = (slice(0, 2), slice(0, 2), slice(0, 2))
BINDING_COL_SLOTS = (slice(0, 4), slice(5, 9), slice(10, 14))
BINDING_ANCHOR_ROWS = slice(2, 3)
BINDING_TOP_ROWS = slice(0, 4)
BINDING_SLOT_PERMUTATIONS = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    require(isinstance(cfg, dict), "config did not parse as a mapping")
    return cfg


def tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)


def compact_pyramid_pool_torch(patches: torch.Tensor) -> torch.Tensor:
    """1x1 + 2x2 patch-grid pooling; returns 5*384 features."""

    require(patches.ndim >= 3 and tuple(patches.shape[-2:]) == (196, 384),
            "patch tensor must end in 196x384")
    grid = patches.reshape(*patches.shape[:-2], 14, 14, 384)
    cells = [grid.mean(dim=(-3, -2))]
    for row in (slice(0, 7), slice(7, 14)):
        for col in (slice(0, 7), slice(7, 14)):
            cells.append(grid[..., row, col, :].mean(dim=(-3, -2)))
    return torch.cat(cells, dim=-1)


def compact_pyramid_pool_np(patches: np.ndarray) -> np.ndarray:
    values = np.asarray(patches, dtype=np.float32)
    require(values.ndim >= 3 and values.shape[-2:] == (196, 384),
            "patch array must end in 196x384")
    grid = values.reshape(*values.shape[:-2], 14, 14, 384)
    cells = [grid.mean(axis=(-3, -2))]
    for row in (slice(0, 7), slice(7, 14)):
        for col in (slice(0, 7), slice(7, 14)):
            cells.append(grid[..., row, col, :].mean(axis=(-3, -2)))
    return np.concatenate(cells, axis=-1)


def binding_slot_pool_torch(patches: torch.Tensor) -> torch.Tensor:
    """Order-preserving top-band pool for the 6-way PushT binding cue.

    The binding renderer writes three colored swatches across the top of the
    frame plus a neutral anchor strip.  Coarse 2x2 pooling can erase the left /
    middle / right ordering; this pool keeps those slots separate while staying
    at the existing 5*384 target dimensionality.
    """

    require(patches.ndim >= 3 and tuple(patches.shape[-2:]) == (196, 384),
            "patch tensor must end in 196x384")
    grid = patches.reshape(*patches.shape[:-2], 14, 14, 384)
    cells = [
        grid[..., row, col, :].mean(dim=(-3, -2))
        for row, col in zip(BINDING_ROW_SLOTS, BINDING_COL_SLOTS)
    ]
    cells.append(grid[..., BINDING_ANCHOR_ROWS, :, :].mean(dim=(-3, -2)))
    cells.append(grid[..., BINDING_TOP_ROWS, :, :].mean(dim=(-3, -2)))
    return torch.cat(cells, dim=-1)


def binding_slot_pool_np(patches: np.ndarray) -> np.ndarray:
    values = np.asarray(patches, dtype=np.float32)
    require(values.ndim >= 3 and values.shape[-2:] == (196, 384),
            "patch array must end in 196x384")
    grid = values.reshape(*values.shape[:-2], 14, 14, 384)
    cells = [
        grid[..., row, col, :].mean(axis=(-3, -2))
        for row, col in zip(BINDING_ROW_SLOTS, BINDING_COL_SLOTS)
    ]
    cells.append(grid[..., BINDING_ANCHOR_ROWS, :, :].mean(axis=(-3, -2)))
    cells.append(grid[..., BINDING_TOP_ROWS, :, :].mean(axis=(-3, -2)))
    return np.concatenate(cells, axis=-1)


def pool_torch(patches: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "binding_slots":
        return binding_slot_pool_torch(patches)
    return compact_pyramid_pool_torch(patches)


def pool_np(patches: np.ndarray, mode: str) -> np.ndarray:
    if mode == "binding_slots":
        return binding_slot_pool_np(patches)
    return compact_pyramid_pool_np(patches)


def classification_record(prediction: np.ndarray, truth: np.ndarray,
                          classes: int) -> dict[str, Any]:
    labels = np.arange(classes, dtype=np.int64)
    matrix = confusion_matrix(truth, prediction, labels=labels)
    recall = np.diag(matrix) / np.maximum(matrix.sum(1), 1)
    return {
        "balanced_accuracy": balanced_accuracy_from_predictions(
            prediction, truth, classes),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": matrix.tolist(),
        "count": int(len(truth)),
    }


class HostAlignedEvidenceWriterAdapter(nn.Module):
    """Slot memory with a gated writer aimed at frozen-host exposure.

    The baseline adapter stores evidence well but loses ordered PushT binding
    when the stored signal is written into the host-visible context.  This
    variant keeps the same memory encoder and evidence decoder contract, then
    uses a patch-level gate to write only the memory-conditioned residual that
    the frozen host should be able to expose.
    """

    def __init__(self, *, dim: int, slots: int, heads: int,
                 max_frames: int = 20, residual_scale: float = 1.0) -> None:
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
        self.writer = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, 384),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.evidence_decoder = nn.Sequential(
            nn.LayerNorm(slots * dim),
            nn.Linear(slots * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, TARGET_DIM),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        nn.init.normal_(self.patch, std=0.02)
        nn.init.zeros_(self.writer[-1].weight)
        nn.init.zeros_(self.writer[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.0)

    def _tokens(self, visual: torch.Tensor, actions: torch.Tensor,
                times: torch.Tensor) -> torch.Tensor:
        batch, steps, patches, _ = visual.shape
        require((patches, actions.shape[:2]) == (196, (batch, steps)),
                "visual/action token shapes are inconsistent")
        return (
            self.visual_proj(visual)
            + self.time(times.to(visual.device))[None, :, None, :]
            + self.patch[None, None, :, :]
            + self.action_proj(actions)[:, :, None, :]
        )

    def encode_memory(self, visual: torch.Tensor, actions: torch.Tensor,
                      times: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens(visual, actions, times).reshape(
            visual.shape[0], -1, self.dim)
        weights = torch.softmax(self.assign(tokens).transpose(1, 2), dim=-1)
        return weights @ tokens

    def inject(self, prefix_visual: torch.Tensor, prefix_actions: torch.Tensor,
               prefix_times: torch.Tensor, context_visual: torch.Tensor,
               context_actions: torch.Tensor,
               context_times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self.encode_memory(prefix_visual, prefix_actions, prefix_times)
        queries = self._tokens(context_visual, context_actions, context_times).reshape(
            context_visual.shape[0], -1, self.dim)
        read, _ = self.cross(queries, slots, slots, need_weights=False)
        joint = torch.cat((queries, read), dim=-1)
        gate = torch.sigmoid(self.gate(joint))
        residual = (gate * self.writer(joint)).reshape_as(context_visual)
        return context_visual + self.residual_scale * residual, slots

    def decode_evidence(self, slots: torch.Tensor) -> torch.Tensor:
        return self.evidence_decoder(slots.reshape(slots.shape[0], -1))


class PushTFeatureBank:
    """Reader for the locked DINO-WM PushT full-patch cache."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.root = resolve(cfg["artifacts"]["root"]) / "cache"
        manifest = self.root / "manifest.json"
        require(manifest.is_file(), f"missing PushT cache manifest: {manifest}")
        self.manifest = json.loads(manifest.read_text())
        for record in self.manifest.get("artifacts", {}).values():
            path = resolve(record["path"])
            require(path.is_file(), f"missing cached artifact: {path}")
            require(path.stat().st_size == int(record["size"]),
                    f"cache artifact size changed: {path}")
            require(sha256_file(path) == record["sha256"],
                    f"cache artifact hash changed: {path}")
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
        require(self.base.shape == (1680, 20, 196, 384),
                "PushT base feature shape changed")
        require(self.actions.shape == (1680, 19, 10),
                "PushT action feature shape changed")
        require(self.proprio.shape == (1680, 20, 4),
                "PushT proprio feature shape changed")
        require(np.count_nonzero(self.split == 0) == 1200
                and np.count_nonzero(self.split == 1) == 480,
                "PushT split counts changed")

    def task_record(self, task_key: str) -> Mapping[str, Any]:
        for task in self.cfg["tasks"]:
            if task["key"] == task_key:
                return task
        raise KeyError(task_key)

    def indices(self, split: str) -> np.ndarray:
        code = {"train": 0, "validation": 1}[split]
        return np.flatnonzero(self.split == code)

    def visual(self, task: str, indices: np.ndarray) -> np.ndarray:
        return make_visual_batch(
            self.base, self.cues[task], np.asarray(indices, dtype=np.int64),
            int(self.cfg["sequence"]["cue_start"]),
            int(self.cfg["sequence"]["cue_length"]))


class FrozenPushTHost:
    """Relaxed-GPU loader for the frozen official DINO-WM PushT predictor."""

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
        require(int(payload["epoch"]) == int(cfg["checkpoint"]["checkpoint_epoch_field"]),
                "released PushT checkpoint epoch changed")
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
                "PushT predictor token shape changed")
        require(tuple(self.action_encoder.patch_embed.weight.shape) == (10, 10, 1),
                "PushT action encoder changed")
        require(tuple(self.proprio_encoder.patch_embed.weight.shape) == (10, 4, 1),
                "PushT proprio encoder changed")
        require(all(not p.requires_grad for module in self.modules.values()
                    for p in module.parameters()), "host is not frozen")

    def digest(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        for module_name, module in sorted(self.modules.items()):
            for name, value in sorted(module.state_dict().items()):
                digest.update(module_name.encode())
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        mask_count = 0
        for name, module in sorted(self.predictor.named_modules()):
            bias = getattr(module, "bias", None)
            if torch.is_tensor(bias) and bias.ndim == 4:
                digest.update(b"attention-mask")
                digest.update(name.encode())
                digest.update(bias.detach().cpu().contiguous().numpy().tobytes())
                mask_count += 1
        require(mask_count == 6, f"expected six attention masks, got {mask_count}")
        return digest.hexdigest()

    def compose(self, visual: torch.Tensor, proprio: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        require(visual.ndim == 4 and tuple(visual.shape[2:]) == (196, 384),
                "visual context violates native shape")
        require(proprio.shape[:2] == visual.shape[:2] and proprio.shape[-1] == 4,
                "proprio context violates native shape")
        require(actions.shape[:2] == visual.shape[:2] and actions.shape[-1] == 10,
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
        predicted = self.predictor(context.reshape(batch, steps * patches, dim))
        return predicted.reshape(batch, steps, patches, dim)


def batch_arrays(bank: PushTFeatureBank, task: str, indices: np.ndarray,
                 age: int, condition: str,
                 device: torch.device) -> dict[str, torch.Tensor]:
    endpoint = endpoint_frame(3, int(age))
    context = predictor_context_for_endpoint(endpoint)
    start, stop = context[0], context[-1] + 1
    visual = bank.visual(task, indices)
    actions = np.asarray(bank.actions[indices], dtype=np.float32)
    if condition == "full":
        prefix_visual = visual[:, :stop]
        prefix_actions = actions[:, :stop]
        prefix_times = np.arange(stop, dtype=np.int64)
    elif condition == "reset":
        prefix_visual = visual[:, start:stop]
        prefix_actions = actions[:, start:stop]
        prefix_times = np.arange(start, stop, dtype=np.int64)
    else:
        raise ValueError(f"unknown condition {condition}")
    return {
        "prefix_visual": tensor(prefix_visual, device),
        "prefix_actions": tensor(prefix_actions, device),
        "prefix_times": torch.from_numpy(prefix_times).long().to(device),
        "context_visual": tensor(visual[:, start:stop], device),
        "context_actions": tensor(actions[:, start:stop], device),
        "context_times": torch.arange(start, stop, device=device).long(),
        "proprio_context": tensor(bank.proprio[indices, start:stop], device),
    }


def no_state_arrays(bank: PushTFeatureBank, task: str, indices: np.ndarray,
                    age: int, device: torch.device) -> dict[str, torch.Tensor]:
    endpoint = endpoint_frame(3, int(age))
    context = predictor_context_for_endpoint(endpoint)
    start, stop = context[0], context[-1] + 1
    visual = bank.visual(task, indices)
    return {
        "context_visual": tensor(visual[:, start:stop], device),
        "context_actions": tensor(bank.actions[indices, start:stop], device),
        "proprio_context": tensor(bank.proprio[indices, start:stop], device),
    }


def evidence_targets(bank: PushTFeatureBank, task: str, indices: np.ndarray,
                     device: torch.device, *,
                     target_mode: str, candidate_count: int,
                     shuffle_targets: bool,
                     negative_mode: str = "batch_roll") -> torch.Tensor:
    cue = torch.from_numpy(
        np.asarray(bank.cues[task][indices], dtype=np.float32)).to(device)
    if target_mode == "cue_compact":
        target = compact_pyramid_pool_torch(cue.reshape(-1, 196, 384)).reshape(
            len(indices), 3, TARGET_DIM).mean(dim=1)
    elif target_mode == "delta_compact":
        base = torch.from_numpy(np.asarray(
            bank.base[indices, 1:4], dtype=np.float32)).to(device)
        delta = cue - base
        target = compact_pyramid_pool_torch(delta.reshape(-1, 196, 384)).reshape(
            len(indices), 3, TARGET_DIM).mean(dim=1)
    elif target_mode == "delta_binding_slots":
        base = torch.from_numpy(np.asarray(
            bank.base[indices, 1:4], dtype=np.float32)).to(device)
        delta = cue - base
        target = binding_slot_pool_torch(delta.reshape(-1, 196, 384)).reshape(
            len(indices), 3, TARGET_DIM).mean(dim=1)
    elif target_mode == "cue_binding_slots":
        target = binding_slot_pool_torch(cue.reshape(-1, 196, 384)).reshape(
            len(indices), 3, TARGET_DIM).mean(dim=1)
    elif target_mode == "binding_slots":
        # Backward-compatible alias for the first attempted repair.
        base = torch.from_numpy(np.asarray(
            bank.base[indices, 1:4], dtype=np.float32)).to(device)
        delta = cue - base
        target = binding_slot_pool_torch(delta.reshape(-1, 196, 384)).reshape(
            len(indices), 3, TARGET_DIM).mean(dim=1)
    else:
        raise ValueError(f"unknown target mode: {target_mode}")
    if shuffle_targets:
        target = target[torch.randperm(len(target), device=device)]
    max_count = max(2, int(candidate_count))
    if negative_mode == "binding_permutation" and "binding_slots" in target_mode:
        chunks = target.reshape(len(indices), 5, 384)
        candidates = [
            torch.cat((chunks[:, list(order)].reshape(len(indices), 3 * 384),
                       chunks[:, 3:].reshape(len(indices), 2 * 384)), dim=1)
            for order in BINDING_SLOT_PERMUTATIONS[:max_count]
        ]
        while len(candidates) < max_count:
            candidates.append(target.roll(len(candidates), dims=0))
        return torch.stack(candidates, dim=1)
    require(negative_mode == "batch_roll",
            f"unsupported negative mode for {target_mode}: {negative_mode}")
    candidates = [target]
    for shift in range(1, max_count):
        candidates.append(target.roll(shift, dims=0))
    return torch.stack(candidates, dim=1)


def train_one_cell(model: MemJepaLabelFreeAdapter, host: FrozenPushTHost,
                   bank: PushTFeatureBank, *, task: str, age: int,
                   seed: int, epochs: int, batch_size: int, lr: float,
                   weight_decay: float, temperature: float,
                   candidate_count: int, target_mode: str, variant: str,
                   negative_mode: str, host_weight: float,
                   context_weight: float, memory_weight: float,
                   residual_l2_weight: float,
                   output_dir: Path) -> list[dict[str, Any]]:
    train_indices = bank.indices("train").copy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs))
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    history_path = output_dir / "history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "epoch", "loss", "host_loss", "context_loss", "memory_loss",
            "host_match", "context_match", "memory_match", "host_cos",
            "context_cos", "memory_cos", "residual_l2", "lr", "seconds",
        ])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            started = time.time()
            rng.shuffle(train_indices)
            losses, host_losses, context_losses, memory_losses, regs = [], [], [], [], []
            host_matches, context_matches, memory_matches = [], [], []
            host_cos_values, context_cos_values, memory_cos_values = [], [], []
            model.train()
            for offset in range(0, len(train_indices), batch_size):
                rows = train_indices[offset:offset + batch_size]
                batch = batch_arrays(bank, task, rows, age, "full", host.device)
                candidates = evidence_targets(
                    bank, task, rows, host.device, target_mode=target_mode,
                    candidate_count=candidate_count,
                    shuffle_targets=variant == "shuffle_targets",
                    negative_mode=negative_mode)
                fused, slots = model.inject(
                    batch["prefix_visual"], batch["prefix_actions"],
                    batch["prefix_times"], batch["context_visual"],
                    batch["context_actions"], batch["context_times"])
                predicted = host.predict(
                    fused, batch["proprio_context"],
                    batch["context_actions"])[:, -1, :, :384]
                feature_mode = "binding_slots" if "binding_slots" in target_mode \
                    else "compact"
                host_query = pool_torch(predicted, feature_mode)
                context_query = pool_torch(fused[:, -1], feature_mode)
                memory_query = model.decode_evidence(slots)
                host_loss, host_match = contrastive_loss(
                    host_query, candidates, temperature)
                context_loss, context_match = contrastive_loss(
                    context_query, candidates, temperature)
                memory_loss, memory_match = contrastive_loss(
                    memory_query, candidates, temperature)
                host_cos = positive_cosine_loss(host_query, candidates)
                context_cos = positive_cosine_loss(context_query, candidates)
                memory_cos = positive_cosine_loss(memory_query, candidates)
                residual_l2 = torch.mean(torch.square(
                    fused - batch["context_visual"]))
                terms = [
                    0.5 * float(memory_weight) * memory_loss,
                    0.25 * float(memory_weight) * memory_cos,
                ]
                if variant != "no_host":
                    terms.extend([
                        float(host_weight) * host_loss,
                        0.5 * float(host_weight) * host_cos,
                    ])
                if variant != "no_context":
                    terms.extend([
                        float(context_weight) * context_loss,
                        float(context_weight) * context_cos,
                    ])
                terms.append(float(residual_l2_weight) * residual_l2)
                loss = sum(terms)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                host_losses.append(float(host_loss.detach().cpu()))
                context_losses.append(float(context_loss.detach().cpu()))
                memory_losses.append(float(memory_loss.detach().cpu()))
                host_matches.append(host_match)
                context_matches.append(context_match)
                memory_matches.append(memory_match)
                host_cos_values.append(float(host_cos.detach().cpu()))
                context_cos_values.append(float(context_cos.detach().cpu()))
                memory_cos_values.append(float(memory_cos.detach().cpu()))
                regs.append(float(residual_l2.detach().cpu()))
            scheduler.step()
            record = {
                "epoch": int(epoch),
                "loss": float(np.mean(losses)),
                "host_loss": float(np.mean(host_losses)),
                "context_loss": float(np.mean(context_losses)),
                "memory_loss": float(np.mean(memory_losses)),
                "host_match": float(np.mean(host_matches)),
                "context_match": float(np.mean(context_matches)),
                "memory_match": float(np.mean(memory_matches)),
                "host_cos": float(np.mean(host_cos_values)),
                "context_cos": float(np.mean(context_cos_values)),
                "memory_cos": float(np.mean(memory_cos_values)),
                "residual_l2": float(np.mean(regs)),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": float(time.time() - started),
            }
            history.append(record)
            writer.writerow(record)
            stream.flush()
            print(
                f"[mem-jepa-pusht-f] task={task} age={age} seed={seed} "
                f"epoch={epoch}/{epochs} loss={record['loss']:.4f} "
                f"host_match={record['host_match']:.3f} "
                f"ctx_match={record['context_match']:.3f} "
                f"mem_match={record['memory_match']:.3f} "
                f"sec={record['seconds']:.1f}",
                flush=True,
            )
    return history


@torch.no_grad()
def collect_level_features(model: MemJepaLabelFreeAdapter, host: FrozenPushTHost,
                           bank: PushTFeatureBank, *, task: str, split: str,
                           age: int, condition: str, level: str,
                           batch_size: int,
                           feature_mode: str) -> dict[str, np.ndarray]:
    model.eval()
    indices = bank.indices(split)
    features = []
    for offset in range(0, len(indices), batch_size):
        rows = indices[offset:offset + batch_size]
        if condition == "no_state":
            require(level in {"host_output", "injected_context"},
                    "no_state is defined only for host_output/context features")
            batch = no_state_arrays(bank, task, rows, age, host.device)
            if level == "host_output":
                predicted = host.predict(
                    batch["context_visual"], batch["proprio_context"],
                    batch["context_actions"])[:, -1, :, :384]
                value = pool_torch(predicted, feature_mode)
            else:
                value = pool_torch(batch["context_visual"][:, -1],
                                   feature_mode)
        else:
            batch = batch_arrays(bank, task, rows, age, condition, host.device)
            fused, slots = model.inject(
                batch["prefix_visual"], batch["prefix_actions"],
                batch["prefix_times"], batch["context_visual"],
                batch["context_actions"], batch["context_times"])
            if level == "memory_prior":
                value = model.decode_evidence(slots)
            elif level == "injected_context":
                value = pool_torch(fused[:, -1], feature_mode)
            elif level == "host_output":
                predicted = host.predict(
                    fused, batch["proprio_context"],
                    batch["context_actions"])[:, -1, :, :384]
                value = pool_torch(predicted, feature_mode)
            else:
                raise ValueError(f"unknown diagnostic level: {level}")
        features.append(value.float().cpu().numpy())
    return {"features": np.concatenate(features)}


def collect_features(model: MemJepaLabelFreeAdapter, host: FrozenPushTHost,
                     bank: PushTFeatureBank, *, task: str, split: str,
                     age: int, condition: str,
                     batch_size: int, feature_mode: str) -> dict[str, np.ndarray]:
    return collect_level_features(
        model, host, bank, task=task, split=split, age=age,
        condition=condition, level="host_output", batch_size=batch_size,
        feature_mode=feature_mode)


def evaluate_one_cell(model: MemJepaLabelFreeAdapter, host: FrozenPushTHost,
                      bank: PushTFeatureBank, *, task: str, age: int,
                      classes: int, batch_size: int,
                      feature_mode: str) -> dict[str, Any]:
    train_indices = bank.indices("train")
    validation_indices = bank.indices("validation")
    train_y = bank.labels[task][train_indices]
    validation_y = bank.labels[task][validation_indices]
    train_full = collect_features(
        model, host, bank, task=task, split="train", age=age,
        condition="full", batch_size=batch_size, feature_mode=feature_mode)
    packs = {
        name: collect_features(
            model, host, bank, task=task, split="validation", age=age,
            condition=name, batch_size=batch_size, feature_mode=feature_mode)
        for name in ("full", "reset", "no_state")
    }
    records = {}
    for name, pack in packs.items():
        prediction = fit_classifier(
            train_full["features"], train_y, pack["features"])
        records[name] = classification_record(prediction, validation_y, classes)
    control_maximum = 1.0 / float(classes) + 0.05
    gate = {
        "full_minimum": 0.75,
        "control_maximum": control_maximum,
        "passed": bool(
            records["full"]["balanced_accuracy"] >= 0.75
            and records["reset"]["balanced_accuracy"] <= control_maximum
            and records["no_state"]["balanced_accuracy"] <= control_maximum),
    }
    return {"records": records, "gate": gate}


def evaluate_diagnostics(model: MemJepaLabelFreeAdapter, host: FrozenPushTHost,
                         bank: PushTFeatureBank, *, task: str, age: int,
                         classes: int, batch_size: int,
                         feature_mode: str) -> dict[str, Any]:
    """Read the same label-free adapter at three interface levels.

    Levels:
    - memory_prior: what the carrier slots alone decode before host exposure.
    - injected_context: what is present in the residual-written context token.
    - host_output: what survives the frozen host predictor.

    The classifier is trained only on train/full features for each level, then
    applied to validation full/reset/no-state controls where the control is
    meaningful. Semantic labels are still used only for audit readout.
    """

    train_indices = bank.indices("train")
    validation_indices = bank.indices("validation")
    train_y = bank.labels[task][train_indices]
    validation_y = bank.labels[task][validation_indices]
    levels = {
        "memory_prior": ("full", "reset"),
        "injected_context": ("full", "reset", "no_state"),
        "host_output": ("full", "reset", "no_state"),
    }
    diagnostics: dict[str, Any] = {
        "schema": "mem_jepa_stage_f_diagnostics_v1",
        "readout": "train_on_train_full_apply_to_validation_conditions",
        "levels": {},
    }
    for level, conditions in levels.items():
        train_full = collect_level_features(
            model, host, bank, task=task, split="train", age=age,
            condition="full", level=level, batch_size=batch_size,
            feature_mode=feature_mode)
        records = {}
        for condition in conditions:
            pack = collect_level_features(
                model, host, bank, task=task, split="validation", age=age,
                condition=condition, level=level, batch_size=batch_size,
                feature_mode=feature_mode)
            prediction = fit_classifier(
                train_full["features"], train_y, pack["features"])
            records[condition] = classification_record(
                prediction, validation_y, classes)
        full_bacc = float(records["full"]["balanced_accuracy"])
        reset_bacc = float(records["reset"]["balanced_accuracy"])
        no_state_bacc = records.get("no_state", {}).get("balanced_accuracy")
        diagnostics["levels"][level] = {
            "feature_dim": int(train_full["features"].shape[1]),
            "records": records,
            "full_minus_reset_pp": float(100.0 * (full_bacc - reset_bacc)),
            "full_minus_no_state_pp": (
                None if no_state_bacc is None
                else float(100.0 * (full_bacc - float(no_state_bacc)))
            ),
        }
    return diagnostics


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    output_root = resolve(args.output)
    output = output_root / args.task / f"s{args.seed}" / f"age_{args.age}"
    require(not (output / "result.json").exists() or args.overwrite,
            f"result already exists: {output / 'result.json'}")
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(resolve(args.config))
    set_determinism(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    bank = PushTFeatureBank(cfg)
    task_record = bank.task_record(args.task)
    classes = int(task_record["classes"])
    host = FrozenPushTHost(cfg, device)
    host_before = host.digest()
    adapter_class = (
        HostAlignedEvidenceWriterAdapter
        if args.adapter == "host_writer"
        else MemJepaLabelFreeAdapter
    )
    model = adapter_class(
        dim=args.dim, slots=args.slots, heads=args.heads,
        residual_scale=args.residual_scale).to(device)
    started = time.time()
    history = train_one_cell(
        model, host, bank, task=args.task, age=args.age, seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, temperature=args.temperature,
        candidate_count=args.candidate_count, target_mode=args.target_mode,
        variant=args.variant, negative_mode=args.negative_mode,
        host_weight=args.host_weight, context_weight=args.context_weight,
        memory_weight=args.memory_weight,
        residual_l2_weight=args.residual_l2_weight, output_dir=output)
    metrics = evaluate_one_cell(
        model, host, bank, task=args.task, age=args.age, classes=classes,
        batch_size=args.eval_batch_size,
        feature_mode="binding_slots" if "binding_slots" in args.target_mode
        else "compact")
    diagnostics = None
    if args.diagnostics:
        diagnostics = evaluate_diagnostics(
            model, host, bank, task=args.task, age=args.age, classes=classes,
            batch_size=args.eval_batch_size,
            feature_mode="binding_slots" if "binding_slots" in args.target_mode
            else "compact")
    host_after = host.digest()
    require(host_before == host_after, "frozen PushT host digest changed")
    checkpoint_path = output / "adapter.pt"
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {checkpoint_path}")
    torch.save({
        "schema": "mem_jepa_pusht_stage_f_adapter_checkpoint_v1",
        "adapter": args.adapter,
        "model_state_dict": model.state_dict(),
        "model": {
            "dim": int(args.dim),
            "slots": int(args.slots),
            "heads": int(args.heads),
            "residual_scale": float(args.residual_scale),
            "target_dim": TARGET_DIM,
        },
        "task": args.task,
        "age": int(args.age),
        "seed": int(args.seed),
        "target_mode": args.target_mode,
        "negative_mode": args.negative_mode,
        "candidate_count": int(args.candidate_count),
        "feature_mode": "binding_slots" if "binding_slots" in args.target_mode
            else "compact",
        "config": str(resolve(args.config).relative_to(ROOT)),
        "config_sha256": sha256_file(resolve(args.config)),
        "host_digest": host_after,
        "labels_used_for_adapter_training": False,
    }, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    result = {
        "schema": "mem_jepa_pusht_stage_f_cell_v1",
        "status": "completed",
        "claim_boundary": (
            "PushT label-free batch-negative masked-evidence retention test; "
            "semantic labels are used only for the final audit readout."),
        "labels_used_for_adapter_training": False,
        "task": args.task,
        "classes": classes,
        "age": int(args.age),
        "seed": int(args.seed),
        "endpoint_frame": endpoint_frame(3, int(args.age)),
        "predictor_context": list(predictor_context_for_endpoint(
            endpoint_frame(3, int(args.age)))),
        "target_mode": args.target_mode,
        "feature_mode": "binding_slots" if "binding_slots" in args.target_mode
            else "compact",
        "candidate_count": int(args.candidate_count),
        "negative_mode": args.negative_mode,
        "variant": args.variant,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "config": str(resolve(args.config).relative_to(ROOT)),
        "config_sha256": sha256_file(resolve(args.config)),
        "host_digest_unchanged": True,
        "host_digest": host_after,
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": checkpoint_sha256,
            "schema": "mem_jepa_pusht_stage_f_adapter_checkpoint_v1",
        },
        "model": {
            "adapter": args.adapter,
            "dim": int(args.dim),
            "slots": int(args.slots),
            "heads": int(args.heads),
            "target_dim": TARGET_DIM,
            "parameters": int(sum(p.numel() for p in model.parameters())),
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "temperature": float(args.temperature),
            "loss_weights": {
                "host": float(args.host_weight),
                "context": float(args.context_weight),
                "memory": float(args.memory_weight),
                "residual_l2": float(args.residual_l2_weight),
            },
            "final": history[-1] if history else None,
        },
        "metrics": metrics,
        "elapsed_seconds": float(time.time() - started),
    }
    if diagnostics is not None:
        result["diagnostics"] = diagnostics
    atomic_json(output / "result.json", result)
    print(json.dumps({
        "task": args.task,
        "age": args.age,
        "seed": args.seed,
        "passed": metrics["gate"]["passed"],
        "full": metrics["records"]["full"]["balanced_accuracy"],
        "reset": metrics["records"]["reset"]["balanced_accuracy"],
        "no_state": metrics["records"]["no_state"]["balanced_accuracy"],
    }, indent=2), flush=True)
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = resolve(args.output)
    summary: dict[str, Any] = {
        "schema": "mem_jepa_pusht_stage_f_summary_v1",
        "status": "completed",
        "claim_boundary": (
            "Aggregates PushT Stage-F label-free masked-evidence runs. "
            "This is exploratory until every registered cell exists."),
        "labels_used_for_adapter_training": False,
        "target_mode": args.target_mode,
        "negative_mode": args.negative_mode,
        "adapter": args.adapter,
        "tasks": {},
        "updated_unix": time.time(),
    }
    all_exist = True
    all_passed = True
    for task in args.tasks:
        summary["tasks"][task] = {}
        for age in args.ages:
            age_records = []
            for seed in args.seeds:
                path = output / task / f"s{seed}" / f"age_{age}" / "result.json"
                if not path.is_file():
                    all_exist = False
                    continue
                value = json.loads(path.read_text())
                age_records.append(value)
            if not age_records:
                continue
            full = [r["metrics"]["records"]["full"]["balanced_accuracy"]
                    for r in age_records]
            reset = [r["metrics"]["records"]["reset"]["balanced_accuracy"]
                     for r in age_records]
            no_state = [r["metrics"]["records"]["no_state"]["balanced_accuracy"]
                        for r in age_records]
            passed = [bool(r["metrics"]["gate"]["passed"])
                      for r in age_records]
            all_passed = all_passed and all(passed)
            summary["tasks"][task][str(age)] = {
                "seeds": [int(r["seed"]) for r in age_records],
                "all_seed_gates_passed": bool(all(passed)),
                "full_mean": float(np.mean(full)),
                "full_seed_values": full,
                "reset_mean": float(np.mean(reset)),
                "reset_seed_values": reset,
                "no_state_mean": float(np.mean(no_state)),
                "no_state_seed_values": no_state,
                "gate": age_records[0]["metrics"]["gate"],
            }
    summary["all_registered_cells_present"] = bool(all_exist)
    summary["all_gates_passed"] = bool(all_exist and all_passed)
    if not summary["all_gates_passed"]:
        summary["status"] = "completed_with_failed_or_missing_gate"
    atomic_json(output / "summary.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "all_registered_cells_present": summary["all_registered_cells_present"],
        "all_gates_passed": summary["all_gates_passed"],
    }, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--task", choices=[
        "transient-visual-token-recall",
        "multi-item-visual-binding-recall",
    ])
    parser.add_argument("--tasks", nargs="*", default=[
        "transient-visual-token-recall",
        "multi-item-visual-binding-recall",
    ])
    parser.add_argument("--age", type=int, choices=[4, 8, 15])
    parser.add_argument("--ages", type=int, nargs="*", default=[4, 8, 15])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--target-mode", default="delta_compact",
                        choices=["cue_compact", "delta_compact",
                                 "binding_slots", "delta_binding_slots",
                                 "cue_binding_slots"])
    parser.add_argument("--negative-mode", default="batch_roll",
                        choices=["batch_roll", "binding_permutation"])
    parser.add_argument("--adapter", default="residual",
                        choices=["residual", "host_writer"])
    parser.add_argument("--variant", default="full", choices=[
        "full", "no_host", "no_context", "shuffle_targets",
    ])
    parser.add_argument("--host-weight", type=float, default=1.0)
    parser.add_argument("--context-weight", type=float, default=1.0)
    parser.add_argument("--memory-weight", type=float, default=1.0)
    parser.add_argument("--residual-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--diagnostics", action="store_true",
                        help="also audit memory-prior and injected-context levels")
    args = parser.parse_args()
    if not args.aggregate and (args.task is None or args.age is None):
        parser.error("--task and --age are required unless --aggregate is used")
    return args


def main() -> None:
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_cell(args)


if __name__ == "__main__":
    main()
