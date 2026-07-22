#!/usr/bin/env python3
"""Label-free CEM on unmodified OGBench render trajectories.

The breadth host is a DINO-feature action-conditioned world model trained on
raw trajectories. It is DINO-WM-style, not an official DINO-WM checkpoint.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.cem_controller import (  # noqa: E402
    CEMController,
    VersionedEventStore,
)


DEFAULT_OUTPUT = ROOT / "outputs/cem_raw_ogbench"
DEFAULT_CACHE_ROOT = (
    ROOT / "outputs/multiview_patchset_color_jepa_native_v1/cache"
)
DEFAULT_DINOV2 = (
    ROOT / "outputs/dinowm_native_pusht_audit_v1/vendor/dinov2"
)
DEFAULT_TORCH_HOME = (
    ROOT / "outputs/dinowm_native_pusht_audit_v1/torch_home"
)
DEFAULT_DINO_WEIGHTS = (
    DEFAULT_TORCH_HOME / "hub/checkpoints/dinov2_vits14_pretrain.pth"
)
OFFICIAL_REPORT = (
    ROOT / "outputs/cem_event_versioning_dinowm_official_v1/report.json"
)

SPLIT_SEED = 20_260_721
FAMILY = {
    "pointmaze": "navigation",
    "antmaze": "navigation",
    "humanoidmaze": "navigation",
    "cube": "manipulation",
    "puzzle": "puzzle",
    "scene": "scene",
}
FORBIDDEN_CALLS = {
    "draw_cue",
    "inject_cue_sequence",
    "inject_cue_sequence_mode",
    "_saliency_map",
}
RAW_PROTOCOL = (
    "DINO-feature action-conditioned world model "
    "(DINO-WM-style breadth host; not official DINO-WM)"
)


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_family(env_name: str) -> str:
    lower = env_name.lower()
    for prefix, family in FAMILY.items():
        if lower.startswith(prefix):
            return family
    return "other"


def source_contract_audit() -> dict[str, Any]:
    tree = ast.parse(Path(__file__).read_text())
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in FORBIDDEN_CALLS:
            sites.append({"call": name, "line": int(node.lineno)})
    return {
        "passed": not sites,
        "forbidden_calls": sorted(FORBIDDEN_CALLS),
        "forbidden_call_sites": sites,
        "input_keys_consumed": ["frames", "actions"],
        "event_write_inputs": [
            "unmodified rendered frames",
            "actions",
            "timestamps",
            "frozen DINOv2 patch tokens",
            "frozen-host prediction surprise",
        ],
        "event_write_targets": (
            "adaptive host-surprise and DINO patch-token change points; "
            "future host-loss reduction under group deletion"
        ),
        "cue_window": None,
        "cue_window_used_by_model": False,
        "cue_metadata_used_by_model": False,
        "reward_goal_state_used_for_event_labels": False,
        "manual_event_labels": False,
        "manual_key_frames": False,
        "handcrafted_saliency_or_color_target": False,
    }


def raw_cache_path(cache_root: Path, env_name: str) -> Path:
    return cache_root / env_name.replace("/", "_") / "render_cache.npz"


def feature_path(output: Path, env_name: str) -> Path:
    return output / "features" / env_name.replace("/", "_") / "features.npz"


def cell_dir(output: Path, env_name: str, seed: int) -> Path:
    return output / "cells" / env_name.replace("/", "_") / f"s{seed}"


def split_indices(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(count)
    train_stop = int(round(0.70 * count))
    val_stop = train_stop + int(round(0.15 * count))
    return order[:train_stop], order[train_stop:val_stop], order[val_stop:]


def load_raw_cache(
    cache_root: Path,
    env_name: str,
    max_episodes: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = raw_cache_path(cache_root, env_name)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        available_keys = list(data.files)
        frames = np.asarray(data["frames"])
        actions = np.asarray(data["actions"], dtype=np.float32)
    if max_episodes is not None:
        frames = frames[:max_episodes]
        actions = actions[:max_episodes]
    if frames.ndim != 5 or actions.ndim != 3:
        raise ValueError(
            f"unexpected raw cache shapes: frames={frames.shape}, "
            f"actions={actions.shape}"
        )
    if frames.shape[:2] != (actions.shape[0], actions.shape[1] + 1):
        raise ValueError("frame/action trajectory alignment is invalid")
    receipt = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "available_keys": available_keys,
        "consumed_keys": ["frames", "actions"],
        "ignored_keys": [
            key for key in available_keys if key not in {"frames", "actions"}
        ],
        "frame_shape": list(frames.shape),
        "action_shape": list(actions.shape),
        "frames_modified_before_encoding": False,
    }
    return frames, actions, receipt


def load_dinov2(
    source: Path,
    torch_home: Path,
    device: torch.device,
) -> nn.Module:
    os.environ["TORCH_HOME"] = str(torch_home.resolve())
    model = torch.hub.load(
        str(source.resolve()),
        "dinov2_vits14",
        source="local",
        pretrained=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)


@torch.no_grad()
def encode_dino_patch_pyramid(
    model: nn.Module,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Encode a 1x1 + 2x2 spatial pyramid of DINOv2 patch tokens."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    flat = frames.reshape(-1, *frames.shape[-3:])
    outputs = []
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device
    ).view(1, 3, 1, 1)
    for start in range(0, len(flat), batch_size):
        rows = torch.from_numpy(flat[start:start + batch_size].copy()).to(device)
        rows = rows.permute(0, 3, 1, 2).float().div_(255.0)
        rows = TF.resize(
            rows,
            [196, 196],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        rows = (rows - mean) / std
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            patch = model.forward_features(rows)["x_norm_patchtokens"]
        grid = patch.float().reshape(len(rows), 14, 14, 384)
        pieces = [grid.mean(dim=(1, 2))]
        for y in range(2):
            for x in range(2):
                pieces.append(
                    grid[:, y * 7:(y + 1) * 7, x * 7:(x + 1) * 7]
                    .mean(dim=(1, 2))
                )
        outputs.append(torch.cat(pieces, dim=-1).cpu().numpy())
        print(
            f"[dino] {min(len(flat), start + batch_size)}/{len(flat)}",
            flush=True,
        )
    return np.concatenate(outputs).reshape(
        frames.shape[0], frames.shape[1], -1
    )


def resolve_device(gpu: int) -> torch.device:
    if gpu == 3:
        raise ValueError("GPU3 is prohibited for this campaign")
    if gpu not in (0, 1, 2):
        raise ValueError(f"GPU must be one of 0,1,2; received {gpu}")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


def prepare_features(args: argparse.Namespace) -> dict[str, Any]:
    from sklearn.decomposition import PCA

    path = feature_path(args.output, args.env_name)
    receipt_path = path.parent / "receipt.json"
    if path.is_file() and receipt_path.is_file() and not args.overwrite:
        return json.loads(receipt_path.read_text())
    path.parent.mkdir(parents=True, exist_ok=True)
    frames, actions, raw_receipt = load_raw_cache(
        args.cache_root, args.env_name, args.max_episodes
    )
    train_idx, val_idx, test_idx = split_indices(len(frames))
    device = resolve_device(args.gpu)
    set_seed(SPLIT_SEED)
    model = load_dinov2(args.dinov2, args.torch_home, device)
    started = time.time()
    pyramid = encode_dino_patch_pyramid(
        model, frames, device, args.feature_batch_size
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    raw_dim = int(pyramid.shape[-1])
    components = min(
        int(args.latent_dim),
        raw_dim,
        int(len(train_idx) * frames.shape[1] - 1),
    )
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        iterated_power=3,
        random_state=SPLIT_SEED,
    )
    train_flat = pyramid[train_idx].reshape(-1, raw_dim).astype(np.float32)
    pca.fit(train_flat)
    latent = pca.transform(
        pyramid.reshape(-1, raw_dim).astype(np.float32)
    ).reshape(len(frames), frames.shape[1], components)
    latent_scale = latent[train_idx].reshape(-1, components).std(0)
    latent_scale = np.maximum(latent_scale, 1e-4)
    latent = (latent / latent_scale).astype(np.float32)
    np.savez_compressed(
        path,
        latents=latent,
        actions=actions.astype(np.float32),
        train_indices=train_idx.astype(np.int64),
        val_indices=val_idx.astype(np.int64),
        test_indices=test_idx.astype(np.int64),
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        latent_scale=latent_scale.astype(np.float32),
    )
    contract = source_contract_audit()
    if not contract["passed"]:
        raise RuntimeError(f"no-manual-cue source audit failed: {contract}")
    receipt = {
        "schema": "cem_raw_ogbench_feature_receipt",
        "status": "completed",
        "environment": args.env_name,
        "family": env_family(args.env_name),
        "protocol": RAW_PROTOCOL,
        "raw_cache": raw_receipt,
        "split": {
            "trajectory_disjoint": True,
            "seed": SPLIT_SEED,
            "train_count": int(len(train_idx)),
            "validation_count": int(len(val_idx)),
            "test_count": int(len(test_idx)),
            "train_indices_sha256": hashlib.sha256(
                train_idx.tobytes()
            ).hexdigest(),
            "validation_indices_sha256": hashlib.sha256(
                val_idx.tobytes()
            ).hexdigest(),
            "test_indices_sha256": hashlib.sha256(
                test_idx.tobytes()
            ).hexdigest(),
        },
        "semantic_encoder": {
            "name": "DINOv2 ViT-S/14",
            "source": str(args.dinov2.relative_to(ROOT)),
            "weights": str(args.dino_weights.relative_to(ROOT)),
            "weights_sha256": sha256_file(args.dino_weights),
            "output": "frozen x_norm_patchtokens",
            "spatial_reduction": "1x1 plus 2x2 patch-token pyramid",
            "raw_feature_dim": raw_dim,
            "train_only_pca_dim": components,
            "pca_fit_uses_test": False,
        },
        "no_manual_cue_contract": contract,
        "feature_path": str(path.relative_to(ROOT)),
        "feature_sha256": sha256_file(path),
        "elapsed_seconds": float(time.time() - started),
    }
    receipt_path.write_text(stable_json(json_safe(receipt)))
    print(stable_json(json_safe(receipt)), flush=True)
    return receipt


class ActionConditionedHost(nn.Module):
    """Finite-window residual predictor over frozen DINO features."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        context: int,
        hidden: int,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.context = int(context)
        input_dim = context * (latent_dim + action_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        context_z: torch.Tensor,
        context_actions: torch.Tensor,
    ) -> torch.Tensor:
        flat = torch.cat(
            [context_z.flatten(1), context_actions.flatten(1)], dim=-1
        )
        return context_z[:, -1] + self.net(flat)


class RawMemoryConditioner(nn.Module):
    """DINO-event residual adapter using the shared CEM router/controller."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden: int,
        budget: int,
    ) -> None:
        super().__init__()
        self.controller = CEMController(
            latent_dim=latent_dim,
            budget=budget,
            quantile=0.78,
            beta=1e-3,
            verify_delta=0.0,
            ce_hidden=hidden,
        )
        self.query = nn.Sequential(
            nn.LayerNorm(latent_dim + action_dim),
            nn.Linear(latent_dim + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.value = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.modulation = nn.Linear(latent_dim, latent_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.need_gate = nn.Sequential(
            nn.LayerNorm(latent_dim + action_dim),
            nn.Linear(latent_dim + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.query_ce = nn.Sequential(
            nn.LayerNorm(2 * latent_dim + 3),
            nn.Linear(2 * latent_dim + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.log_scale = nn.Parameter(torch.tensor(-2.0))

    def query_vector(
        self, context_z: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.query(torch.cat([context_z[:, -1], action], dim=-1))

    def scores(
        self,
        context_z: torch.Tensor,
        action: torch.Tensor,
        events: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_vector(context_z, action)
        return self.controller.route(query, events, metadata[..., 0])

    def ce_hat(
        self,
        events: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        return self.controller.ce_hat(
            events, metadata[..., 0], metadata[..., 1]
        )

    def query_ce_hat(
        self,
        context_z: torch.Tensor,
        action: torch.Tensor,
        events: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_vector(context_z, action)
        expanded = query[:, None].expand(-1, events.shape[1], -1)
        return self.query_ce(
            torch.cat([events, expanded, metadata], dim=-1)
        ).squeeze(-1)

    def forward(
        self,
        context_z: torch.Tensor,
        action: torch.Tensor,
        events: torch.Tensor,
        metadata: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query_input = torch.cat([context_z[:, -1], action], dim=-1)
        query = self.query(query_input)
        scores = self.controller.route(
            query, events, metadata[..., 0]
        )
        masked_scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(masked_scores, dim=1)
        weights = torch.where(
            mask.any(1, keepdim=True), weights, torch.zeros_like(weights)
        )
        event = torch.einsum("bs,bsd->bd", weights, events)
        interaction = self.value(event) * torch.sigmoid(
            self.modulation(query)
        )
        need = torch.sigmoid(self.need_gate(query_input))
        present = mask.any(1, keepdim=True).float()
        scale = self.log_scale.exp().clamp(max=1.0)
        residual = present * need * scale * self.output(interaction)
        return residual, {
            "weights": weights,
            "router_score": scores,
            "need": need.squeeze(-1),
            "residual_norm": residual.square().mean(-1),
        }


@dataclass
class Event:
    event_id: int
    start: int
    end: int
    peak_t: int
    proposal_score: float
    surprise: float
    semantic_change: float
    vector: np.ndarray
    key_id: int = -1


@dataclass
class QuerySample:
    episode_id: int
    query_t: int
    context_z: np.ndarray
    action_history: np.ndarray
    future_actions: np.ndarray
    targets: np.ndarray
    events: list[Event]
    recent_event: np.ndarray


@dataclass(frozen=True)
class StoreConfig:
    promotion_threshold: float
    hysteresis: float
    budget: int
    topk: int
    verification_delay: int


@dataclass
class QueryTensors:
    context_z: torch.Tensor
    action_history: torch.Tensor
    future_actions: torch.Tensor
    targets: torch.Tensor
    events: torch.Tensor
    metadata: torch.Tensor
    valid: torch.Tensor
    recent_event: torch.Tensor

    def index(self, index: torch.Tensor) -> "QueryTensors":
        return QueryTensors(**{
            name: getattr(self, name)[index]
            for name in self.__dataclass_fields__
        })

    def __len__(self) -> int:
        return int(self.context_z.shape[0])


def transition_arrays(
    latents: np.ndarray,
    actions: np.ndarray,
    episodes: np.ndarray,
    context: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, action_xs, ys = [], [], []
    for episode in episodes:
        for time_index in range(context - 1, latents.shape[1] - 1):
            xs.append(
                latents[
                    episode,
                    time_index - context + 1:time_index + 1,
                ]
            )
            action_xs.append(
                actions[
                    episode,
                    time_index - context + 1:time_index + 1,
                ]
            )
            ys.append(latents[episode, time_index + 1])
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(action_xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
    )


def batches(
    count: int,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterable[np.ndarray]:
    order = rng.permutation(count)
    for start in range(0, count, batch_size):
        yield order[start:start + batch_size]


def evaluate_host_arrays(
    host: ActionConditionedHost,
    x: torch.Tensor,
    action_x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> float:
    losses = []
    host.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            pred = host(
                x[start:start + batch_size],
                action_x[start:start + batch_size],
            )
            losses.append(
                (pred - y[start:start + batch_size])
                .square().mean(-1).cpu()
            )
    return float(torch.cat(losses).mean())


def train_host(
    latents: np.ndarray,
    actions: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ActionConditionedHost, dict[str, Any]]:
    arrays = {}
    for name, indices in (("train", train_idx), ("validation", val_idx)):
        arrays[name] = transition_arrays(
            latents, actions, indices, args.context
        )
    tensors: dict[str, tuple[torch.Tensor, ...]] = {}
    for name, values in arrays.items():
        tensors[name] = tuple(
            torch.from_numpy(value).to(device) for value in values
        )
    host = ActionConditionedHost(
        latent_dim=latents.shape[-1],
        action_dim=actions.shape[-1],
        context=args.context,
        hidden=args.host_hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(
        host.parameters(),
        lr=args.host_lr,
        weight_decay=1e-4,
    )
    rng = np.random.default_rng(args.seed + 71)
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    train_x, train_a, train_y = tensors["train"]
    val_x, val_a, val_y = tensors["validation"]
    persistence_val = float(
        (val_x[:, -1] - val_y).square().mean().detach().cpu()
    )
    mean_target = train_y.mean(0, keepdim=True)
    mean_baseline_val = float(
        (mean_target - val_y).square().mean().detach().cpu()
    )
    for epoch in range(args.host_epochs):
        host.train()
        epoch_losses = []
        for indices in batches(
            len(train_x), args.batch_size, rng
        ):
            index = torch.as_tensor(indices, device=device)
            pred = host(train_x[index], train_a[index])
            loss = F.mse_loss(pred, train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(host.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        val_loss = evaluate_host_arrays(
            host, val_x, val_a, val_y, args.batch_size
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": val_loss,
        })
        if math.isfinite(val_loss) and val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in host.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.host_patience:
            break
    if best_state is None:
        raise RuntimeError("host training produced no finite checkpoint")
    host.load_state_dict(best_state)
    host.eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    summary = {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "validation_persistence_mse": persistence_val,
        "validation_mean_baseline_mse": mean_baseline_val,
        "validation_vs_persistence_ratio": best_loss / max(persistence_val, 1e-12),
        "validation_vs_mean_ratio": best_loss / max(mean_baseline_val, 1e-12),
        "reliable": bool(
            math.isfinite(best_loss)
            and best_loss < mean_baseline_val
            and best_loss < 1.5 * persistence_val
        ),
    }
    return host, summary


@torch.no_grad()
def one_step_surprise(
    host: ActionConditionedHost,
    latents: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    context: int,
    batch_size: int,
) -> np.ndarray:
    episodes, steps = latents.shape[:2]
    surprise = np.full((episodes, steps), np.nan, dtype=np.float32)
    rows, action_rows, locations = [], [], []
    for episode in range(episodes):
        for time_index in range(context - 1, steps - 1):
            rows.append(
                latents[
                    episode,
                    time_index - context + 1:time_index + 1,
                ]
            )
            action_rows.append(
                actions[
                    episode,
                    time_index - context + 1:time_index + 1,
                ]
            )
            locations.append((episode, time_index + 1))
    x = torch.from_numpy(np.asarray(rows, np.float32)).to(device)
    a = torch.from_numpy(np.asarray(action_rows, np.float32)).to(device)
    values = []
    for start in range(0, len(x), batch_size):
        pred = host(x[start:start + batch_size], a[start:start + batch_size])
        target = torch.from_numpy(np.asarray([
            latents[episode, time_index]
            for episode, time_index in locations[
                start:start + batch_size
            ]
        ], np.float32)).to(device)
        values.extend(
            (pred - target).square().mean(-1).cpu().numpy().tolist()
        )
    for (episode, time_index), value in zip(locations, values):
        surprise[episode, time_index] = float(value)
    return surprise


def discover_events(
    latents: np.ndarray,
    surprise: np.ndarray,
    train_idx: np.ndarray,
    quantile: float,
) -> tuple[list[list[Event]], dict[str, float]]:
    change = np.zeros(surprise.shape, dtype=np.float32)
    change[:, 1:] = np.mean(
        np.square(latents[:, 1:] - latents[:, :-1]), axis=-1
    )
    surprise_threshold = float(
        np.nanquantile(surprise[train_idx], quantile)
    )
    change_threshold = float(
        np.quantile(change[train_idx, 1:], quantile)
    )
    surprise_scale = max(surprise_threshold, 1e-8)
    change_scale = max(change_threshold, 1e-8)
    all_events: list[list[Event]] = []
    for episode in range(len(latents)):
        score = np.maximum(
            np.nan_to_num(surprise[episode], nan=0.0) / surprise_scale,
            change[episode] / change_scale,
        )
        candidates = [
            time_index for time_index in range(1, latents.shape[1] - 1)
            if (
                np.nan_to_num(surprise[episode, time_index], nan=0.0)
                > surprise_threshold
                or change[episode, time_index] > change_threshold
            )
        ]
        if not candidates:
            candidates = [int(np.argmax(score[1:-1])) + 1]
        groups: list[list[int]] = []
        for time_index in candidates:
            if not groups or time_index > groups[-1][-1] + 1:
                groups.append([time_index])
            else:
                groups[-1].append(time_index)
        events = []
        for event_id, group in enumerate(groups):
            peak = max(group, key=lambda time_index: score[time_index])
            start, end = min(group), max(group)
            events.append(Event(
                event_id=event_id,
                start=start,
                end=end,
                peak_t=int(peak),
                proposal_score=float(score[peak]),
                surprise=float(np.nan_to_num(
                    surprise[episode, peak], nan=0.0
                )),
                semantic_change=float(change[episode, peak]),
                vector=latents[episode, start:end + 1]
                .mean(0).astype(np.float32),
            ))
        all_events.append(events)
    return all_events, {
        "surprise_quantile": float(quantile),
        "surprise_threshold": surprise_threshold,
        "semantic_change_threshold": change_threshold,
        "grouping": "adjacent adaptive-threshold crossings",
    }


def assign_semantic_keys(
    events: list[list[Event]],
    train_idx: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    from sklearn.cluster import MiniBatchKMeans

    train_vectors = np.asarray([
        event.vector
        for episode in train_idx
        for event in events[int(episode)]
    ], dtype=np.float32)
    clusters = min(16, max(2, int(round(math.sqrt(len(train_vectors))))))
    clusters = min(clusters, len(train_vectors))
    model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        batch_size=256,
        n_init=3,
    ).fit(train_vectors)
    counts = np.zeros(clusters, dtype=np.int64)
    for episode_events in events:
        if not episode_events:
            continue
        vectors = np.asarray(
            [event.vector for event in episode_events], dtype=np.float32
        )
        keys = model.predict(vectors)
        for event, key in zip(episode_events, keys):
            event.key_id = int(key)
            counts[int(key)] += 1
    return {
        "method": "train-only MiniBatchKMeans on DINO event vectors",
        "cluster_count": int(clusters),
        "cluster_counts": counts.tolist(),
        "manual_semantic_labels": False,
    }


def build_queries(
    latents: np.ndarray,
    actions: np.ndarray,
    events: list[list[Event]],
    indices: np.ndarray,
    context: int,
    horizon: int,
) -> list[QuerySample]:
    queries = []
    for episode in indices:
        episode = int(episode)
        for query_t in range(context + 2, latents.shape[1] - horizon):
            legal_events = [
                event for event in events[episode]
                if event.end <= query_t - context
            ]
            if not legal_events:
                continue
            queries.append(QuerySample(
                episode_id=episode,
                query_t=query_t,
                context_z=latents[
                    episode, query_t - context + 1:query_t + 1
                ].astype(np.float32),
                action_history=actions[
                    episode, query_t - context + 1:query_t + 1
                ].astype(np.float32),
                future_actions=actions[
                    episode, query_t:query_t + horizon
                ].astype(np.float32),
                targets=latents[
                    episode, query_t + 1:query_t + horizon + 1
                ].astype(np.float32),
                events=legal_events,
                recent_event=latents[
                    episode, query_t - context
                ].astype(np.float32),
            ))
    return queries


def tensorize_queries(
    queries: list[QuerySample],
    device: torch.device,
    max_events: int,
) -> QueryTensors:
    if not queries:
        raise ValueError("query set is empty")
    latent_dim = queries[0].context_z.shape[-1]
    slots = min(
        max_events,
        max(len(query.events) for query in queries),
    )
    event_array = np.zeros(
        (len(queries), slots, latent_dim), dtype=np.float32
    )
    metadata = np.zeros((len(queries), slots, 3), dtype=np.float32)
    valid = np.zeros((len(queries), slots), dtype=bool)
    for row, query in enumerate(queries):
        selected = sorted(
            query.events,
            key=lambda event: event.proposal_score,
            reverse=True,
        )[:slots]
        selected = sorted(selected, key=lambda event: event.peak_t)
        for column, event in enumerate(selected):
            event_array[row, column] = event.vector
            metadata[row, column] = (
                (query.query_t - event.peak_t)
                / max(1, query.query_t),
                math.log1p(max(0.0, event.surprise)),
                math.log1p(max(0.0, event.semantic_change)),
            )
            valid[row, column] = True
    values = {
        "context_z": np.asarray(
            [query.context_z for query in queries], np.float32
        ),
        "action_history": np.asarray(
            [query.action_history for query in queries], np.float32
        ),
        "future_actions": np.asarray(
            [query.future_actions for query in queries], np.float32
        ),
        "targets": np.asarray(
            [query.targets for query in queries], np.float32
        ),
        "events": event_array,
        "metadata": metadata,
        "valid": valid,
        "recent_event": np.asarray(
            [query.recent_event for query in queries], np.float32
        )[:, None, :],
    }
    return QueryTensors(**{
        key: torch.from_numpy(value).to(device)
        for key, value in values.items()
    })


def rollout(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner | None,
    batch: QueryTensors,
    events: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    context_z = batch.context_z
    action_history = batch.action_history
    predictions, residual_norms, needs = [], [], []
    events = batch.events if events is None else events
    metadata = batch.metadata if metadata is None else metadata
    mask = batch.valid if mask is None else mask
    for step in range(batch.future_actions.shape[1]):
        action = batch.future_actions[:, step]
        context_actions = (
            action_history
            if step == 0
            else torch.cat(
                [action_history[:, 1:], action[:, None, :]], dim=1
            )
        )
        prediction = host(context_z, context_actions)
        if memory is not None:
            residual, telemetry = memory(
                context_z, action, events, metadata, mask
            )
            prediction = prediction + residual
            residual_norms.append(telemetry["residual_norm"])
            needs.append(telemetry["need"])
        predictions.append(prediction)
        context_z = torch.cat(
            [context_z[:, 1:], prediction[:, None, :]], dim=1
        )
        action_history = context_actions
    prediction_stack = torch.stack(predictions, dim=1)
    info = {}
    if residual_norms:
        info = {
            "residual_norm": torch.stack(residual_norms, 1),
            "need": torch.stack(needs, 1),
        }
    return prediction_stack, info


def horizon_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (prediction - target).square().mean(-1)


def train_memory(
    host: ActionConditionedHost,
    train: QueryTensors,
    validation: QueryTensors,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[RawMemoryConditioner, dict[str, Any]]:
    memory = RawMemoryConditioner(
        latent_dim=train.context_z.shape[-1],
        action_dim=train.future_actions.shape[-1],
        hidden=args.memory_hidden,
        budget=args.max_events,
    ).to(device)
    optimizer = torch.optim.AdamW(
        memory.parameters(),
        lr=args.memory_lr,
        weight_decay=1e-4,
    )
    rng = np.random.default_rng(args.seed + 811)
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    with torch.no_grad():
        no_memory_prediction, _ = rollout(host, None, validation)
        no_memory_val = float(
            horizon_loss(no_memory_prediction, validation.targets).mean()
        )
    for epoch in range(args.memory_epochs):
        memory.train()
        losses = []
        for indices in batches(len(train), args.batch_size, rng):
            index = torch.as_tensor(indices, device=device)
            batch = train.index(index)
            prediction, telemetry = rollout(host, memory, batch)
            prediction_loss = horizon_loss(
                prediction, batch.targets
            ).mean()
            residual_cost = telemetry["residual_norm"].mean()
            write_cost = memory.controller.write_cost(
                batch.metadata[..., 1][batch.valid]
            )
            loss = (
                prediction_loss
                + args.residual_cost * residual_cost
                + write_cost
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(memory.parameters(), 1.0)
            optimizer.step()
            losses.append(float(prediction_loss.detach()))
        memory.eval()
        with torch.no_grad():
            val_prediction, _ = rollout(host, memory, validation)
            val_loss = float(
                horizon_loss(
                    val_prediction, validation.targets
                ).mean()
            )
        history.append({
            "epoch": epoch + 1,
            "train_future_mse": float(np.mean(losses)),
            "validation_future_mse": val_loss,
        })
        if math.isfinite(val_loss) and val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in memory.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.memory_patience:
            break
    if best_state is None:
        raise RuntimeError("memory training produced no finite checkpoint")
    memory.load_state_dict(best_state)
    memory.eval()
    return memory, {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "no_memory_validation_mse": no_memory_val,
        "soft_store_relative_improvement": (
            no_memory_val - best_loss
        ) / max(no_memory_val, 1e-12),
    }


@torch.no_grad()
def compute_group_ce_targets(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner,
    batch: QueryTensors,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, slots = batch.valid.shape
    target = torch.zeros(
        (rows, slots), device=batch.valid.device, dtype=torch.float32
    )
    full_losses, base_losses = [], []
    for start in range(0, rows, batch_size):
        stop = min(rows, start + batch_size)
        index = torch.arange(start, stop, device=batch.valid.device)
        part = batch.index(index)
        base_prediction, _ = rollout(host, None, part)
        base_loss = horizon_loss(
            base_prediction, part.targets
        ).mean(1)
        full_losses.append(base_loss)
        base_losses.append(base_loss)
        for slot in range(slots):
            keep = torch.zeros_like(part.valid)
            keep[:, slot] = part.valid[:, slot]
            kept_prediction, _ = rollout(
                host, memory, part, mask=keep
            )
            kept_loss = horizon_loss(
                kept_prediction, part.targets
            ).mean(1)
            target[start:stop, slot] = (
                base_loss - kept_loss
            ) / base_loss.clamp_min(1e-8)
    target = target.masked_fill(~batch.valid, 0.0)
    return (
        target,
        torch.cat(full_losses),
        torch.cat(base_losses),
    )


def combined_ce_prediction(
    memory: RawMemoryConditioner,
    batch: QueryTensors,
) -> torch.Tensor:
    return memory.query_ce_hat(
        batch.context_z,
        batch.future_actions[:, 0],
        batch.events,
        batch.metadata,
    )


def train_ce_calibrator(
    memory: RawMemoryConditioner,
    train: QueryTensors,
    target: torch.Tensor,
    validation: QueryTensors,
    validation_target: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    parameters = list(memory.query_ce.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.ce_lr, weight_decay=1e-4
    )
    rng = np.random.default_rng(args.seed + 1901)
    history = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(args.ce_epochs):
        memory.train()
        epoch_losses = []
        for indices in batches(len(train), args.batch_size, rng):
            index = torch.as_tensor(
                indices, device=train.context_z.device
            )
            part = train.index(index)
            prediction = combined_ce_prediction(memory, part)
            truth = target[index].clamp(-1.0, 1.0)
            valid = part.valid
            regression = F.smooth_l1_loss(
                prediction[valid], truth[valid], beta=0.05
            )
            true_delta = truth.unsqueeze(2) - truth.unsqueeze(1)
            pred_delta = prediction.unsqueeze(2) - prediction.unsqueeze(1)
            pairs = (
                valid.unsqueeze(2)
                & valid.unsqueeze(1)
                & (true_delta.abs() > 1e-5)
            )
            ranking = (
                F.softplus(-true_delta.sign() * pred_delta)[pairs].mean()
                if bool(pairs.any())
                else regression.new_zeros(())
            )
            loss = regression + ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        memory.eval()
        with torch.no_grad():
            val_prediction = combined_ce_prediction(memory, validation)
            val_loss = float(F.smooth_l1_loss(
                val_prediction[validation.valid],
                validation_target[validation.valid].clamp(-1.0, 1.0),
                beta=0.05,
            ))
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_huber": val_loss,
        })
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in memory.state_dict().items()
            }
    if best_state is not None:
        memory.load_state_dict(best_state)
    memory.eval()
    return {
        "history": history,
        "best_validation_huber": best_loss,
        "target_quantiles": np.quantile(
            target[train.valid].detach().cpu().numpy(),
            [0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
        ).tolist(),
        "periodic_group_deletion_calibration": True,
        "ranking_loss": "all valid non-tied pairs",
    }


def ordered_query_events(
    query: QuerySample,
    max_events: int,
) -> list[Event]:
    selected = sorted(
        query.events,
        key=lambda event: event.proposal_score,
        reverse=True,
    )[:max_events]
    return sorted(selected, key=lambda event: event.peak_t)


@torch.no_grad()
def select_store_masks(
    memory: RawMemoryConditioner,
    batch: QueryTensors,
    queries: list[QuerySample],
    config: StoreConfig,
    *,
    with_log: bool = False,
) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    ce_hat = memory.query_ce_hat(
        batch.context_z,
        batch.future_actions[:, 0],
        batch.events,
        batch.metadata,
    )
    route = memory.scores(
        batch.context_z,
        batch.future_actions[:, 0],
        batch.events,
        batch.metadata,
    )
    centered_route = route - (
        route.masked_fill(~batch.valid, 0.0).sum(1, keepdim=True)
        / batch.valid.sum(1, keepdim=True).clamp_min(1)
    )
    combined = ce_hat + 0.01 * centered_route
    selected_mask = torch.zeros_like(batch.valid)
    telemetry = {
        "proposed": 0,
        "eligible_for_verification": 0,
        "promoted": 0,
        "rejected": 0,
        "superseded": 0,
        "evicted": 0,
        "fallback": 0,
        "retrieved": 0,
        "stale_selected": 0,
        "occupancy": [],
    }
    logs = []
    for row, query in enumerate(queries):
        row_events = ordered_query_events(query, batch.events.shape[1])
        store = VersionedEventStore(
            budget=int(config.budget),
            hysteresis=float(config.hysteresis),
        )
        transitions = []
        for column, event in enumerate(row_events):
            store.propose(
                event_id=event.event_id,
                key_id=event.key_id,
                event_timestamp=event.peak_t,
                proposed_at=event.start,
            )
            telemetry["proposed"] += 1
            verified_at = event.end + int(config.verification_delay)
            if verified_at > query.query_t:
                continue
            telemetry["eligible_for_verification"] += 1
            transition = store.verify(
                event_id=event.event_id,
                verified_at=verified_at,
                ce_hat=float(ce_hat[row, column]),
                threshold=float(config.promotion_threshold),
            )
            transitions.append(transition)
            if transition["transition"] == "promoted":
                telemetry["promoted"] += 1
                telemetry["superseded"] += int(
                    transition["superseded_event_id"] is not None
                )
                telemetry["evicted"] += len(
                    transition["evicted_event_ids"]
                )
            else:
                telemetry["rejected"] += 1
                telemetry["fallback"] += int(
                    transition["fallback_event_id"] is not None
                )
        active_ids = {record.event_id for record in store.active()}
        active_columns = [
            column for column, event in enumerate(row_events)
            if event.event_id in active_ids
        ]
        if active_columns:
            ranking = sorted(
                active_columns,
                key=lambda column: float(combined[row, column]),
                reverse=True,
            )
            retrieved = ranking[:min(int(config.topk), len(ranking))]
            selected_mask[row, retrieved] = True
            telemetry["retrieved"] += len(retrieved)
            latest = max(
                (row_events[column].peak_t for column in active_columns)
            )
            telemetry["stale_selected"] += sum(
                row_events[column].peak_t < latest for column in retrieved
            )
        else:
            retrieved = []
        telemetry["occupancy"].append(len(active_columns))
        if with_log:
            logs.append({
                "episode_id": int(query.episode_id),
                "query_t": int(query.query_t),
                "cue_window": None,
                "cue_window_used_by_model": False,
                "events": [
                    {
                        "event_id": int(event.event_id),
                        "event_timestamp": int(event.peak_t),
                        "group_start": int(event.start),
                        "group_end": int(event.end),
                        "key_id": int(event.key_id),
                        "proposal_score": float(event.proposal_score),
                        "host_surprise": float(event.surprise),
                        "semantic_change": float(event.semantic_change),
                        "ce_hat": float(ce_hat[row, column]),
                        "router_score": float(route[row, column]),
                        "retrieved": bool(column in retrieved),
                    }
                    for column, event in enumerate(row_events)
                ],
                "lifecycle": store.snapshot(),
                "transitions": transitions,
                "retrieved_event_ids": [
                    int(row_events[column].event_id)
                    for column in retrieved
                ],
            })
    occupancy = np.asarray(telemetry.pop("occupancy"), dtype=np.float64)
    telemetry["mean_occupancy"] = float(occupancy.mean())
    telemetry["maximum_occupancy"] = int(occupancy.max(initial=0))
    telemetry["write_rate"] = telemetry["proposed"] / max(
        1, len(queries)
    )
    telemetry["promoted_rate"] = telemetry["promoted"] / max(
        1, telemetry["eligible_for_verification"]
    )
    telemetry["rejected_rate"] = telemetry["rejected"] / max(
        1, telemetry["eligible_for_verification"]
    )
    telemetry["retrieval_rate"] = float(
        selected_mask.any(1).float().mean().cpu()
    )
    return selected_mask, telemetry, logs


@torch.no_grad()
def loss_with_mask(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner,
    batch: QueryTensors,
    mask: torch.Tensor,
    *,
    events: torch.Tensor | None = None,
    metadata: torch.Tensor | None = None,
) -> tuple[float, list[float], torch.Tensor]:
    prediction, _ = rollout(
        host,
        memory,
        batch,
        events=events,
        metadata=metadata,
        mask=mask,
    )
    loss = horizon_loss(prediction, batch.targets)
    return (
        float(loss.mean()),
        loss.mean(0).detach().cpu().numpy().astype(float).tolist(),
        loss.mean(1),
    )


def tune_store_config(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner,
    validation: QueryTensors,
    validation_queries: list[QuerySample],
    args: argparse.Namespace,
) -> tuple[StoreConfig, list[dict[str, Any]]]:
    memory.eval()
    with torch.no_grad():
        ce_values = memory.query_ce_hat(
            validation.context_z,
            validation.future_actions[:, 0],
            validation.events,
            validation.metadata,
        )[validation.valid].detach().cpu().numpy()
    thresholds = sorted(set(float(value) for value in np.quantile(
        ce_values, [0.30, 0.50, 0.70, 1.0]
    )))
    thresholds[-1] += max(1e-5, float(np.std(ce_values)) * 0.01)
    margins = [0.0, max(1e-5, float(np.std(ce_values)) * 0.10)]
    budgets = [value for value in (1, 2, 4) if value <= args.max_events]
    delays = [1, 3] if not args.smoke else [1]
    topks = [1, 2] if not args.smoke else [1]
    sweep = []
    best = None
    for threshold in thresholds:
        for margin in margins:
            for budget in budgets:
                for delay in delays:
                    for topk in topks:
                        if topk > budget:
                            continue
                        config = StoreConfig(
                            promotion_threshold=threshold,
                            hysteresis=margin,
                            budget=budget,
                            topk=topk,
                            verification_delay=delay,
                        )
                        mask, telemetry, _ = select_store_masks(
                            memory,
                            validation,
                            validation_queries,
                            config,
                        )
                        loss, horizons, _ = loss_with_mask(
                            host, memory, validation, mask
                        )
                        objective = loss + 1e-7 * telemetry["retrieval_rate"]
                        row = {
                            "config": asdict(config),
                            "validation_future_mse": loss,
                            "validation_horizon_mse": horizons,
                            "objective": objective,
                            "telemetry": telemetry,
                        }
                        sweep.append(row)
                        if best is None or objective < best[0]:
                            best = (objective, config)
    if best is None:
        raise RuntimeError("store tuning produced no configuration")
    return best[1], sweep


def rank_correlation(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float | None:
    if len(prediction) < 3:
        return None
    from scipy.stats import spearmanr

    value = float(spearmanr(prediction, target).statistic)
    return value if math.isfinite(value) else None


@torch.no_grad()
def evaluate_controls(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner,
    test: QueryTensors,
    test_queries: list[QuerySample],
    config: StoreConfig,
    seed: int,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any], list[dict[str, Any]]]:
    selected, telemetry, logs = select_store_masks(
        memory, test, test_queries, config, with_log=True
    )
    full, full_horizon, full_per_query = loss_with_mask(
        host, memory, test, selected
    )
    with torch.no_grad():
        no_prediction, _ = rollout(host, None, test)
        no_loss_tensor = horizon_loss(no_prediction, test.targets)
    no_memory = float(no_loss_tensor.mean())
    no_horizon = no_loss_tensor.mean(0).cpu().numpy().astype(float).tolist()
    reset = torch.zeros_like(selected)
    reset_loss, reset_horizon, _ = loss_with_mask(
        host, memory, test, reset
    )
    rng = np.random.default_rng(seed + 5101)
    permutation = torch.as_tensor(
        rng.permutation(len(test)), device=test.context_z.device
    )
    shuffled_events = test.events[permutation]
    shuffled_metadata = test.metadata[permutation]
    shuffled_mask = selected[permutation]
    shuffled_loss, shuffled_horizon, _ = loss_with_mask(
        host,
        memory,
        test,
        shuffled_mask,
        events=shuffled_events,
        metadata=shuffled_metadata,
    )
    generator = torch.Generator(device=test.context_z.device)
    generator.manual_seed(seed + 9107)
    random_events = torch.randn(
        test.events.shape,
        generator=generator,
        device=test.events.device,
    )
    source_norm = test.events.norm(dim=-1, keepdim=True)
    random_events = random_events / random_events.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    random_events = random_events * source_norm
    random_loss, random_horizon, _ = loss_with_mask(
        host,
        memory,
        test,
        selected,
        events=random_events,
    )
    recent_metadata = torch.zeros(
        (len(test), 1, 3), device=test.context_z.device
    )
    recent_metadata[..., 0] = 1.0 / max(2, test.context_z.shape[1] + 1)
    recent_mask = torch.ones(
        (len(test), 1), device=test.context_z.device, dtype=torch.bool
    )
    recent_loss, recent_horizon, _ = loss_with_mask(
        host,
        memory,
        test,
        recent_mask,
        events=test.recent_event,
        metadata=recent_metadata,
    )
    metrics = {
        "memory": {"mse": full, "horizon_mse": full_horizon},
        "no_memory": {"mse": no_memory, "horizon_mse": no_horizon},
        "reset_memory": {
            "mse": reset_loss,
            "horizon_mse": reset_horizon,
        },
        "shuffled_episode_memory": {
            "mse": shuffled_loss,
            "horizon_mse": shuffled_horizon,
        },
        "random_matched_norm_memory": {
            "mse": random_loss,
            "horizon_mse": random_horizon,
        },
        "recent_only": {
            "mse": recent_loss,
            "horizon_mse": recent_horizon,
        },
        "relative_improvement_vs_no_memory": (
            no_memory - full
        ) / max(no_memory, 1e-12),
        "relative_improvement_vs_recent_only": (
            recent_loss - full
        ) / max(recent_loss, 1e-12),
        "sample_count": len(test),
    }
    return metrics, selected, telemetry, logs


@torch.no_grad()
def causal_deletion_metrics(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner,
    test: QueryTensors,
    selected: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    scores = combined_ce_prediction(memory, test).masked_fill(
        ~selected, -1e9
    )
    eligible = selected.sum(1) >= 2
    if not bool(eligible.any()):
        return {
            "eligible_queries": 0,
            "high_ce_group_delta_mse": None,
            "matched_random_group_delta_mse": None,
            "high_minus_random": None,
        }
    index = torch.nonzero(eligible, as_tuple=False).flatten()
    part = test.index(index)
    part_selected = selected[index]
    part_scores = scores[index]
    high_mask = torch.zeros_like(part_selected)
    high_slot = part_scores.argmax(1)
    high_mask[
        torch.arange(len(index), device=index.device), high_slot
    ] = True
    rng = np.random.default_rng(seed + 7717)
    random_mask = torch.zeros_like(part_selected)
    random_slots = []
    for row in range(len(index)):
        candidates = torch.nonzero(
            part_selected[row], as_tuple=False
        ).flatten().detach().cpu().numpy()
        random_slots.append(int(rng.choice(candidates)))
    random_slot_t = torch.as_tensor(
        random_slots, device=index.device
    )
    random_mask[
        torch.arange(len(index), device=index.device), random_slot_t
    ] = True
    high_prediction, _ = rollout(
        host, memory, part, mask=high_mask
    )
    random_prediction, _ = rollout(
        host, memory, part, mask=random_mask
    )
    base_prediction, _ = rollout(host, None, part)
    high_loss = horizon_loss(
        high_prediction, part.targets
    ).mean(1)
    random_loss = horizon_loss(
        random_prediction, part.targets
    ).mean(1)
    base_loss = horizon_loss(
        base_prediction, part.targets
    ).mean(1)
    high_delta = base_loss - high_loss
    random_delta = base_loss - random_loss
    return {
        "eligible_queries": int(len(index)),
        "high_ce_group_delta_mse": float(high_delta.mean()),
        "matched_random_group_delta_mse": float(random_delta.mean()),
        "high_minus_random": float(
            high_delta.mean() - random_delta.mean()
        ),
        "high_positive_fraction": float((high_delta > 0).float().mean()),
        "random_positive_fraction": float(
            (random_delta > 0).float().mean()
        ),
    }


@torch.no_grad()
def budget_delay_curves(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner,
    test: QueryTensors,
    queries: list[QuerySample],
    config: StoreConfig,
    max_events: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    budgets = []
    for budget in [value for value in (1, 2, 4, 8) if value <= max_events]:
        variant = StoreConfig(
            promotion_threshold=config.promotion_threshold,
            hysteresis=config.hysteresis,
            budget=budget,
            topk=min(config.topk, budget),
            verification_delay=config.verification_delay,
        )
        mask, telemetry, _ = select_store_masks(
            memory, test, queries, variant
        )
        loss, _, _ = loss_with_mask(host, memory, test, mask)
        budgets.append({
            "budget": budget,
            "mse": loss,
            "retrieval_rate": telemetry["retrieval_rate"],
            "mean_occupancy": telemetry["mean_occupancy"],
        })
    delays = []
    for delay in (1, 2, 3, 4, 6):
        variant = StoreConfig(
            promotion_threshold=config.promotion_threshold,
            hysteresis=config.hysteresis,
            budget=config.budget,
            topk=config.topk,
            verification_delay=delay,
        )
        mask, telemetry, _ = select_store_masks(
            memory, test, queries, variant
        )
        loss, _, _ = loss_with_mask(host, memory, test, mask)
        delays.append({
            "verification_delay": delay,
            "mse": loss,
            "retrieval_rate": telemetry["retrieval_rate"],
            "promoted_rate": telemetry["promoted_rate"],
        })
    return budgets, delays


def repeat_query_tensors(
    batch: QueryTensors,
    index: torch.Tensor,
    repeats: int,
    future_actions: torch.Tensor,
) -> QueryTensors:
    def repeated(value: torch.Tensor) -> torch.Tensor:
        selected = value[index]
        return selected[:, None].expand(
            -1, repeats, *selected.shape[1:]
        ).reshape(-1, *selected.shape[1:])

    return QueryTensors(
        context_z=repeated(batch.context_z),
        action_history=repeated(batch.action_history),
        future_actions=future_actions.reshape(
            -1, *future_actions.shape[2:]
        ),
        targets=repeated(batch.targets),
        events=repeated(batch.events),
        metadata=repeated(batch.metadata),
        valid=repeated(batch.valid),
        recent_event=repeated(batch.recent_event),
    )


@torch.no_grad()
def action_sequence_ranking(
    host: ActionConditionedHost,
    memory: RawMemoryConditioner,
    test: QueryTensors,
    queries: list[QuerySample],
    config: StoreConfig,
    seed: int,
    limit: int = 256,
    candidate_count: int = 8,
) -> dict[str, Any]:
    count = min(limit, len(test))
    if count < candidate_count:
        return {
            "name": "post-hoc action-sequence identification",
            "sample_count": 0,
            "planning_claim": False,
        }
    rng = np.random.default_rng(seed + 11213)
    chosen = np.sort(rng.choice(len(test), size=count, replace=False))
    index = torch.as_tensor(chosen, device=test.context_z.device)
    neutral_actions = test.future_actions.clone()
    neutral_actions[:, 0] = 0.0
    neutral = QueryTensors(
        context_z=test.context_z,
        action_history=test.action_history,
        future_actions=neutral_actions,
        targets=test.targets,
        events=test.events,
        metadata=test.metadata,
        valid=test.valid,
        recent_event=test.recent_event,
    )
    neutral_mask, _, _ = select_store_masks(
        memory, neutral, queries, config
    )
    candidates = torch.empty(
        (
            count,
            candidate_count,
            test.future_actions.shape[1],
            test.future_actions.shape[2],
        ),
        device=test.context_z.device,
    )
    candidates[:, 0] = test.future_actions[index]
    all_indices = np.arange(len(test))
    for column in range(1, candidate_count):
        permutation = rng.choice(
            all_indices, size=count, replace=len(test) < count
        )
        candidates[:, column] = test.future_actions[
            torch.as_tensor(permutation, device=index.device)
        ]
    expanded = repeat_query_tensors(
        test, index, candidate_count, candidates
    )
    expanded_mask = neutral_mask[index][:, None].expand(
        -1, candidate_count, -1
    ).reshape(-1, neutral_mask.shape[1])
    memory_prediction, _ = rollout(
        host, memory, expanded, mask=expanded_mask
    )
    baseline_prediction, _ = rollout(host, None, expanded)
    memory_score = horizon_loss(
        memory_prediction, expanded.targets
    ).mean(1).reshape(count, candidate_count)
    baseline_score = horizon_loss(
        baseline_prediction, expanded.targets
    ).mean(1).reshape(count, candidate_count)

    def summarize(score: torch.Tensor) -> dict[str, float]:
        order = score.argsort(1)
        rank = (order == 0).nonzero()[:, 1].float() + 1.0
        return {
            "top1_accuracy": float((rank == 1).float().mean()),
            "mean_reciprocal_rank": float((1.0 / rank).mean()),
            "mean_true_action_rank": float(rank.mean()),
        }

    return {
        "name": "post-hoc action-sequence identification",
        "description": (
            "Ranks the observed future action sequence against seven "
            "episode-shuffled alternatives by latent rollout error."
        ),
        "sample_count": count,
        "candidate_count": candidate_count,
        "memory_retrieval_uses_candidate_actions": False,
        "planning_claim": False,
        "memory": summarize(memory_score),
        "no_memory": summarize(baseline_score),
    }


def host_test_quality(
    host: ActionConditionedHost,
    latents: np.ndarray,
    actions: np.ndarray,
    test_idx: np.ndarray,
    context: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    x, action_x, y = transition_arrays(
        latents, actions, test_idx, context
    )
    x_t = torch.from_numpy(x).to(device)
    action_t = torch.from_numpy(action_x).to(device)
    y_t = torch.from_numpy(y).to(device)
    mse = evaluate_host_arrays(
        host, x_t, action_t, y_t, batch_size
    )
    persistence = float((x_t[:, -1] - y_t).square().mean().cpu())
    mean_baseline = float(
        (y_t - y_t.mean(0, keepdim=True)).square().mean().cpu()
    )
    return {
        "test_one_step_mse": mse,
        "test_persistence_mse": persistence,
        "test_mean_baseline_mse": mean_baseline,
        "test_vs_persistence_ratio": mse / max(persistence, 1e-12),
        "test_vs_mean_ratio": mse / max(mean_baseline, 1e-12),
    }


def timeline_streams(
    logs: list[dict[str, Any]],
    surprise: np.ndarray,
    latents: np.ndarray,
    test_idx: np.ndarray,
    limit: int = 8,
) -> list[dict[str, Any]]:
    retrievals: dict[int, int] = {}
    for row in logs:
        retrievals[row["episode_id"]] = (
            retrievals.get(row["episode_id"], 0)
            + len(row["retrieved_event_ids"])
        )
    ranked = sorted(
        [int(value) for value in test_idx],
        key=lambda episode: retrievals.get(episode, 0),
        reverse=True,
    )[:limit]
    semantic_change = np.zeros(surprise.shape, dtype=np.float32)
    semantic_change[:, 1:] = np.mean(
        np.square(latents[:, 1:] - latents[:, :-1]), axis=-1
    )
    return [{
        "episode_id": episode,
        "host_surprise": [
            None if not math.isfinite(float(value)) else float(value)
            for value in surprise[episode]
        ],
        "semantic_change": semantic_change[episode].astype(float).tolist(),
        "retrieval_count": int(retrievals.get(episode, 0)),
    } for episode in ranked]


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = cell_dir(args.output, args.env_name, args.seed)
    result_path = output_dir / "result.json"
    if result_path.is_file() and not args.overwrite:
        return json.loads(result_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    path = feature_path(args.output, args.env_name)
    receipt_path = path.parent / "receipt.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing DINO features; run --prepare-features first: {path}"
        )
    device = resolve_device(args.gpu)
    set_seed(args.seed)
    with np.load(path, allow_pickle=False) as data:
        latents = np.asarray(data["latents"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        train_idx = np.asarray(data["train_indices"], dtype=np.int64)
        val_idx = np.asarray(data["val_indices"], dtype=np.int64)
        test_idx = np.asarray(data["test_indices"], dtype=np.int64)
    feature_receipt = json.loads(receipt_path.read_text())
    started = time.time()
    host, host_training = train_host(
        latents, actions, train_idx, val_idx, args, device
    )
    host_digest_before = tensor_digest(host)
    surprise = one_step_surprise(
        host,
        latents,
        actions,
        device,
        args.context,
        args.batch_size,
    )
    events, discovery = discover_events(
        latents,
        surprise,
        train_idx,
        args.event_quantile,
    )
    semantic_keys = assign_semantic_keys(
        events, train_idx, args.seed
    )
    query_sets = {
        "train": build_queries(
            latents,
            actions,
            events,
            train_idx,
            args.context,
            args.horizon,
        ),
        "validation": build_queries(
            latents,
            actions,
            events,
            val_idx,
            args.context,
            args.horizon,
        ),
        "test": build_queries(
            latents,
            actions,
            events,
            test_idx,
            args.context,
            args.horizon,
        ),
    }
    tensors = {
        name: tensorize_queries(values, device, args.max_events)
        for name, values in query_sets.items()
    }
    memory, memory_training = train_memory(
        host,
        tensors["train"],
        tensors["validation"],
        args,
        device,
    )
    train_target, _, _ = compute_group_ce_targets(
        host, memory, tensors["train"], args.batch_size
    )
    validation_target, _, _ = compute_group_ce_targets(
        host, memory, tensors["validation"], args.batch_size
    )
    ce_training = train_ce_calibrator(
        memory,
        tensors["train"],
        train_target,
        tensors["validation"],
        validation_target,
        args,
    )
    config, tuning_sweep = tune_store_config(
        host,
        memory,
        tensors["validation"],
        query_sets["validation"],
        args,
    )
    controls, selected, telemetry, logs = evaluate_controls(
        host,
        memory,
        tensors["test"],
        query_sets["test"],
        config,
        args.seed,
    )
    deletion = causal_deletion_metrics(
        host,
        memory,
        tensors["test"],
        tensors["test"].valid,
        args.seed,
    )
    deletion["reference_store"] = (
        "singleton automatic event group versus no-memory deletion; "
        "no event labels"
    )
    test_target, _, _ = compute_group_ce_targets(
        host, memory, tensors["test"], args.batch_size
    )
    with torch.no_grad():
        test_prediction = combined_ce_prediction(
            memory, tensors["test"]
        )
    valid = tensors["test"].valid
    prediction_np = test_prediction[valid].detach().cpu().numpy()
    target_np = test_target[valid].detach().cpu().numpy()
    calibration = {
        "spearman": rank_correlation(prediction_np, target_np),
        "pair_count": int(len(prediction_np)),
        "mae": float(np.mean(np.abs(prediction_np - target_np))),
        "predicted_quantiles": np.quantile(
            prediction_np, [0, .25, .5, .75, 1]
        ).tolist(),
        "true_quantiles": np.quantile(
            target_np, [0, .25, .5, .75, 1]
        ).tolist(),
    }
    verified_mask = selected & (test_target > 0)
    verified_loss, verified_horizon, _ = loss_with_mask(
        host, memory, tensors["test"], verified_mask
    )
    retrieved_count = int(selected.sum().item())
    accepted_count = int(verified_mask.sum().item())
    retrieve_then_verify = {
        "candidate_count": retrieved_count,
        "accepted_count": accepted_count,
        "rejected_count": retrieved_count - accepted_count,
        "acceptance_rate": (
            accepted_count / retrieved_count if retrieved_count else 0.0
        ),
        "posthoc_verified_memory_mse": verified_loss,
        "posthoc_verified_horizon_mse": verified_horizon,
        "verification_target": (
            "measured singleton group future-loss reduction > 0"
        ),
        "uses_observed_future_target": True,
        "used_for_primary_prediction": False,
        "interpretation": (
            "bounded retrieve-then-verify audit after the prediction horizon; "
            "reported separately to avoid test-target leakage"
        ),
    }
    for row, query_log in enumerate(logs):
        for column, event_log in enumerate(query_log["events"]):
            true_effect = float(test_target[row, column])
            event_log["true_group_effect_posthoc"] = true_effect
            event_log["retrieve_then_verify_accepted"] = bool(
                selected[row, column] and true_effect > 0
            )
    budget_curve, delay_curve = budget_delay_curves(
        host,
        memory,
        tensors["test"],
        query_sets["test"],
        config,
        args.max_events,
    )
    action_use = action_sequence_ranking(
        host,
        memory,
        tensors["test"],
        query_sets["test"],
        config,
        args.seed,
        limit=64 if args.smoke else 256,
    )
    host_quality = {
        **host_training,
        **host_test_quality(
            host,
            latents,
            actions,
            test_idx,
            args.context,
            args.batch_size,
            device,
        ),
    }
    host_digest_after = tensor_digest(host)
    if host_digest_after != host_digest_before:
        raise RuntimeError("frozen host changed during CEM training")
    contract = source_contract_audit()
    if not contract["passed"]:
        raise RuntimeError(f"no-manual-cue contract failed: {contract}")
    decision_log = {
        "schema": "cem_raw_ogbench_decision_log",
        "environment": args.env_name,
        "seed": args.seed,
        "protocol": RAW_PROTOCOL,
        "no_manual_cue_contract": contract,
        "cue_window": None,
        "cue_window_used_by_model": False,
        "selected_store_config": asdict(config),
        "discovery": discovery,
        "episode_streams": timeline_streams(
            logs, surprise, latents, test_idx
        ),
        "queries": logs,
    }
    (output_dir / "decision_log.json").write_text(
        stable_json(json_safe(decision_log))
    )
    torch.save({
        "schema": "cem_raw_ogbench_checkpoint",
        "host": host.state_dict(),
        "memory": memory.state_dict(),
        "host_config": {
            "latent_dim": int(latents.shape[-1]),
            "action_dim": int(actions.shape[-1]),
            "context": args.context,
            "hidden": args.host_hidden,
        },
        "memory_config": {
            "hidden": args.memory_hidden,
            "budget": args.max_events,
        },
        "store_config": asdict(config),
    }, output_dir / "model.pt")
    result = {
        "schema": "cem_raw_ogbench_cell",
        "status": "completed",
        "environment": args.env_name,
        "family": env_family(args.env_name),
        "seed": args.seed,
        "seed_role": "optimization seed; trajectory split is fixed and recorded",
        "protocol": RAW_PROTOCOL,
        "official_dinowm": False,
        "no_manual_cue_contract": contract,
        "feature_receipt": str(receipt_path.relative_to(ROOT)),
        "split": feature_receipt["split"],
        "config": {
            "context_window": args.context,
            "rollout_horizon": args.horizon,
            "max_provisional_events": args.max_events,
            "selected_store": asdict(config),
            "host_epochs_requested": args.host_epochs,
            "memory_epochs_requested": args.memory_epochs,
            "ce_epochs_requested": args.ce_epochs,
        },
        "host_predictor": host_quality,
        "host_frozen_digest_before_cem": host_digest_before,
        "host_frozen_digest_after_cem": host_digest_after,
        "host_unchanged": True,
        "discovery": {
            **discovery,
            "semantic_keys": semantic_keys,
            "event_counts": {
                "train": int(sum(
                    len(events[int(index)]) for index in train_idx
                )),
                "validation": int(sum(
                    len(events[int(index)]) for index in val_idx
                )),
                "test": int(sum(
                    len(events[int(index)]) for index in test_idx
                )),
            },
            "query_counts": {
                key: len(value) for key, value in query_sets.items()
            },
        },
        "memory_training": memory_training,
        "ce_training": ce_training,
        "validation_sweep": tuning_sweep,
        "test": {
            "controls": controls,
            "causal_deletion": deletion,
            "ce_calibration": calibration,
            "retrieve_then_verify": retrieve_then_verify,
            "telemetry": telemetry,
            "budget_curve": budget_curve,
            "verification_delay_curve": delay_curve,
            "downstream_use": action_use,
        },
        "artifacts": {
            "result": str(result_path.relative_to(ROOT)),
            "decision_log": str(
                (output_dir / "decision_log.json").relative_to(ROOT)
            ),
            "checkpoint": str(
                (output_dir / "model.pt").relative_to(ROOT)
            ),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    result_path.write_text(stable_json(json_safe(result)))
    print(stable_json(json_safe({
        "status": result["status"],
        "environment": args.env_name,
        "seed": args.seed,
        "host_test_mse": host_quality["test_one_step_mse"],
        "memory_relative_improvement": controls[
            "relative_improvement_vs_no_memory"
        ],
        "deletion_gap": deletion["high_minus_random"],
        "result": result["artifacts"]["result"],
    })), flush=True)
    return result


def summary_stat(values: list[float | None]) -> dict[str, Any]:
    array = np.asarray(
        [value for value in values if value is not None],
        dtype=np.float64,
    )
    if not len(array):
        return {"mean": None, "std": None, "ci95": [None, None], "count": 0}
    if len(array) == 1:
        interval = [float(array[0]), float(array[0])]
        std = 0.0
    else:
        std = float(array.std(ddof=1))
        half = 1.96 * std / math.sqrt(len(array))
        interval = [float(array.mean() - half), float(array.mean() + half)]
    return {
        "mean": float(array.mean()),
        "std": std,
        "ci95": interval,
        "count": int(len(array)),
        "values": array.astype(float).tolist(),
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    cells = []
    for path in sorted((args.output / "cells").glob("*/*/result.json")):
        try:
            result = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if result.get("schema") == "cem_raw_ogbench_cell":
            cells.append(result)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(cell["environment"], []).append(cell)
    environments = []
    for environment, rows in sorted(grouped.items()):
        environments.append({
            "environment": environment,
            "family": rows[0]["family"],
            "seeds": sorted(int(row["seed"]) for row in rows),
            "seed_count": len(rows),
            "host_test_mse": summary_stat([
                row["host_predictor"]["test_one_step_mse"] for row in rows
            ]),
            "host_vs_persistence_ratio": summary_stat([
                row["host_predictor"]["test_vs_persistence_ratio"]
                for row in rows
            ]),
            "memory_relative_improvement": summary_stat([
                row["test"]["controls"][
                    "relative_improvement_vs_no_memory"
                ] for row in rows
            ]),
            "memory_mse": summary_stat([
                row["test"]["controls"]["memory"]["mse"] for row in rows
            ]),
            "no_memory_mse": summary_stat([
                row["test"]["controls"]["no_memory"]["mse"] for row in rows
            ]),
            "recent_only_mse": summary_stat([
                row["test"]["controls"]["recent_only"]["mse"] for row in rows
            ]),
            "shuffled_memory_mse": summary_stat([
                row["test"]["controls"][
                    "shuffled_episode_memory"
                ]["mse"] for row in rows
            ]),
            "high_ce_deletion": summary_stat([
                row["test"]["causal_deletion"][
                    "high_ce_group_delta_mse"
                ] for row in rows
            ]),
            "random_deletion": summary_stat([
                row["test"]["causal_deletion"][
                    "matched_random_group_delta_mse"
                ] for row in rows
            ]),
            "deletion_gap": summary_stat([
                row["test"]["causal_deletion"]["high_minus_random"]
                for row in rows
            ]),
            "ce_spearman": summary_stat([
                row["test"]["ce_calibration"]["spearman"] for row in rows
            ]),
            "write_rate": summary_stat([
                row["test"]["telemetry"]["write_rate"] for row in rows
            ]),
            "promoted_rate": summary_stat([
                row["test"]["telemetry"]["promoted_rate"] for row in rows
            ]),
            "retrieval_rate": summary_stat([
                row["test"]["telemetry"]["retrieval_rate"] for row in rows
            ]),
            "action_ranking_top1": summary_stat([
                row["test"]["downstream_use"].get("memory", {}).get(
                    "top1_accuracy"
                ) for row in rows
            ]),
            "action_ranking_no_memory_top1": summary_stat([
                row["test"]["downstream_use"].get("no_memory", {}).get(
                    "top1_accuracy"
                ) for row in rows
            ]),
            "all_contracts_pass": all(
                row["no_manual_cue_contract"]["passed"] for row in rows
            ),
            "all_hosts_unchanged": all(row["host_unchanged"] for row in rows),
            "reliable_host_count": sum(
                bool(row["host_predictor"]["reliable"]) for row in rows
            ),
        })
    family_rows = []
    for family in sorted({row["family"] for row in cells}):
        rows = [row for row in cells if row["family"] == family]
        family_rows.append({
            "family": family,
            "environment_count": len({
                row["environment"] for row in rows
            }),
            "cell_count": len(rows),
            "memory_relative_improvement": summary_stat([
                row["test"]["controls"][
                    "relative_improvement_vs_no_memory"
                ] for row in rows
            ]),
            "deletion_gap": summary_stat([
                row["test"]["causal_deletion"]["high_minus_random"]
                for row in rows
            ]),
            "action_top1_improvement": summary_stat([
                (
                    row["test"]["downstream_use"].get("memory", {}).get(
                        "top1_accuracy", 0.0
                    )
                    - row["test"]["downstream_use"].get(
                        "no_memory", {}
                    ).get("top1_accuracy", 0.0)
                ) for row in rows
            ]),
        })
    positive_families = sum(
        (row["memory_relative_improvement"]["mean"] or 0.0) > 0
        for row in family_rows
    )
    official = None
    if OFFICIAL_REPORT.is_file():
        source = json.loads(OFFICIAL_REPORT.read_text())
        variant = source.get("variants", {}).get(
            "full_versioned_delayed_verification", {}
        )
        official = {
            "protocol": "official frozen DINO-WM Wall event-versioning",
            "separate_from_breadth": True,
            "source": str(OFFICIAL_REPORT.relative_to(ROOT)),
            "metrics": variant,
        }
    report = {
        "schema": "cem_raw_ogbench_report",
        "status": "completed" if cells else "empty",
        "protocol": RAW_PROTOCOL,
        "claim_boundary": (
            "Breadth results use a clean DINO-feature action-conditioned "
            "world model trained on raw OGBench trajectories; they are not "
            "official DINO-WM and do not execute planning."
        ),
        "no_manual_cue_contract": source_contract_audit(),
        "cell_count": len(cells),
        "environment_count": len(environments),
        "family_count": len(family_rows),
        "environments": environments,
        "families": family_rows,
        "aggregate": {
            "memory_relative_improvement": summary_stat([
                row["test"]["controls"][
                    "relative_improvement_vs_no_memory"
                ] for row in cells
            ]),
            "control_mse": {
                condition: summary_stat([
                    row["test"]["controls"][condition]["mse"]
                    for row in cells
                ])
                for condition in (
                    "memory",
                    "no_memory",
                    "reset_memory",
                    "shuffled_episode_memory",
                    "random_matched_norm_memory",
                    "recent_only",
                )
            },
            "horizon_relative_improvement": [
                summary_stat([
                    (
                        row["test"]["controls"]["no_memory"]["horizon_mse"][
                            horizon
                        ]
                        - row["test"]["controls"]["memory"]["horizon_mse"][
                            horizon
                        ]
                    ) / max(
                        row["test"]["controls"]["no_memory"]["horizon_mse"][
                            horizon
                        ],
                        1e-12,
                    )
                    for row in cells
                ])
                for horizon in range(min(
                    (
                        len(row["test"]["controls"]["memory"]["horizon_mse"])
                        for row in cells
                    ),
                    default=0,
                ))
            ],
            "deletion_gap": summary_stat([
                row["test"]["causal_deletion"]["high_minus_random"]
                for row in cells
            ]),
            "deletion_high_exceeds_random_count": sum(
                row["test"]["causal_deletion"]["high_minus_random"] > 0
                for row in cells
            ),
            "ce_spearman": summary_stat([
                row["test"]["ce_calibration"]["spearman"] for row in cells
            ]),
            "ce_positive_spearman_count": sum(
                (row["test"]["ce_calibration"]["spearman"] or 0.0) > 0
                for row in cells
            ),
            "recent_only_win_count": sum(
                row["test"]["controls"]["recent_only"]["mse"]
                < row["test"]["controls"]["memory"]["mse"]
                for row in cells
            ),
            "action_sequence_top1": {
                "memory": summary_stat([
                    row["test"]["downstream_use"]["memory"]["top1_accuracy"]
                    for row in cells
                ]),
                "no_memory": summary_stat([
                    row["test"]["downstream_use"]["no_memory"]["top1_accuracy"]
                    for row in cells
                ]),
                "memory_win_count": sum(
                    row["test"]["downstream_use"]["memory"]["top1_accuracy"]
                    > row["test"]["downstream_use"]["no_memory"][
                        "top1_accuracy"
                    ]
                    for row in cells
                ),
                "planning_claim": False,
            },
            "retrieve_then_verify": {
                "acceptance_rate": summary_stat([
                    row["test"]["retrieve_then_verify"]["acceptance_rate"]
                    for row in cells
                ]),
                "posthoc_verified_memory_mse": summary_stat([
                    row["test"]["retrieve_then_verify"][
                        "posthoc_verified_memory_mse"
                    ]
                    for row in cells
                ]),
                "uses_observed_future_target": True,
                "used_for_primary_prediction": False,
            },
            "positive_environment_count": sum(
                (
                    environment["memory_relative_improvement"]["mean"]
                    or 0.0
                ) > 0
                for environment in environments
            ),
            "positive_family_count": positive_families,
            "automatic_raw_event_memory_improves_across_families": bool(
                len(family_rows) >= 3
                and positive_families >= 3
                and (
                    summary_stat([
                        row["test"]["controls"][
                            "relative_improvement_vs_no_memory"
                        ] for row in cells
                    ])["mean"] or 0.0
                ) > 0
            ),
            "automatic_raw_event_memory_improves_endpoint": (
                "future-latent MSE relative to no memory; does not imply "
                "superiority to recent-only"
            ),
            "beats_recent_only_consistently": False,
        },
        "official_validation": official,
        "exclusions": [],
        "jobs_still_running": [],
        "artifacts": {
            "cells": "outputs/cem_raw_ogbench/cells/<env>/s<seed>",
            "report": "outputs/cem_raw_ogbench/report.json",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        stable_json(json_safe(report))
    )
    print(stable_json(json_safe(report)), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-root", type=Path, default=DEFAULT_CACHE_ROOT
    )
    parser.add_argument("--dinov2", type=Path, default=DEFAULT_DINOV2)
    parser.add_argument(
        "--torch-home", type=Path, default=DEFAULT_TORCH_HOME
    )
    parser.add_argument(
        "--dino-weights", type=Path, default=DEFAULT_DINO_WEIGHTS
    )
    parser.add_argument("--prepare-features", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--host-hidden", type=int, default=256)
    parser.add_argument("--host-epochs", type=int, default=50)
    parser.add_argument("--host-patience", type=int, default=8)
    parser.add_argument("--host-lr", type=float, default=3e-4)
    parser.add_argument("--memory-hidden", type=int, default=192)
    parser.add_argument("--memory-epochs", type=int, default=35)
    parser.add_argument("--memory-patience", type=int, default=7)
    parser.add_argument("--memory-lr", type=float, default=4e-4)
    parser.add_argument("--ce-epochs", type=int, default=25)
    parser.add_argument("--ce-lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--event-quantile", type=float, default=0.78)
    parser.add_argument("--residual-cost", type=float, default=1e-3)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "output",
        "cache_root",
        "dinov2",
        "torch_home",
        "dino_weights",
    ):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.smoke:
        args.max_episodes = args.max_episodes or 96
        args.latent_dim = min(args.latent_dim, 48)
        args.horizon = min(args.horizon, 2)
        args.host_hidden = min(args.host_hidden, 128)
        args.memory_hidden = min(args.memory_hidden, 96)
        args.host_epochs = min(args.host_epochs, 4)
        args.memory_epochs = min(args.memory_epochs, 4)
        args.ce_epochs = min(args.ce_epochs, 3)
        args.host_patience = min(args.host_patience, 3)
        args.memory_patience = min(args.memory_patience, 3)
        args.batch_size = min(args.batch_size, 128)
        args.max_events = min(args.max_events, 4)
    return args


def main() -> None:
    args = resolve_paths(parse_args())
    if args.aggregate:
        aggregate(args)
        return
    if not args.env_name:
        raise ValueError("--env-name is required")
    if args.prepare_features:
        prepare_features(args)
        return
    run_cell(args)


if __name__ == "__main__":
    main()
