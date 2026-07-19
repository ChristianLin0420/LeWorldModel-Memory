#!/usr/bin/env python3
"""Mem-JEPA v2 cue-card PointMaze diagnostic.

This runner tests the corrected first gate after the weak Stage-1 prototype:
can a non-recurrent, non-KV compact-slot JEPA model recover a transient cue
from real OGBench PointMaze frames when the cue disappears before the endpoint?

Scope boundary:
  * This is not a Paper-A claim.
  * This does not touch LeWM/DINO-WM checkpoints.
  * Passing this script only authorizes host-exposure experiments next.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from typing import Iterable

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "mem_jepa_v2"
DEFAULT_BASE_DATA = ROOT / "outputs" / "mem_jepa_stage1" / "data" / "pointmaze_stage1.npz"
ENV_NAME = "pointmaze-large-navigate-v0"
AGES = (4, 8, 15)
PALETTE = np.asarray([
    [230, 57, 70],
    [46, 204, 113],
    [52, 152, 219],
    [245, 203, 92],
], dtype=np.uint8)
CARD_POSITIONS = np.asarray([
    [5, 5],
    [47, 5],
    [5, 47],
    [47, 47],
], dtype=np.int64)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE_DATA)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--env-name", default=ENV_NAME)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--episodes", type=int, default=384)
    parser.add_argument("--length", type=int, default=24)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--age", type=int, choices=AGES, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--visible-count", type=int, default=10)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--target-weight", type=float, default=0.10)
    parser.add_argument("--contrast-weight", type=float, default=0.0)
    parser.add_argument("--proto-weight", type=float, default=2.0)
    parser.add_argument("--std-weight", type=float, default=0.15)
    parser.add_argument("--cov-weight", type=float, default=0.03)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def data_path(args: argparse.Namespace) -> Path:
    return args.data or (args.output / "data" / "cuecard_pointmaze_v2.npz")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((size, size), Image.Resampling.BILINEAR))


def collect_base_frames(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import ogbench  # noqa: WPS433

    rng = np.random.default_rng(args.seed)
    env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    frames, actions = [], []
    for ep in range(args.episodes):
        task_id = int(ep % 5) + 1
        env.reset(options={"task_id": task_id, "render_goal": True})
        episode_frames = [resize_frame(env.render(), args.img_size)]
        episode_actions = []
        for _ in range(args.length - 1):
            action = env.action_space.sample()
            if rng.random() < 0.05:
                action = np.zeros_like(action)
            episode_actions.append(np.asarray(action, dtype=np.float32))
            env.step(action)
            episode_frames.append(resize_frame(env.render(), args.img_size))
        frames.append(np.stack(episode_frames).astype(np.uint8))
        actions.append(np.stack(episode_actions).astype(np.float32))
    env.close()
    return np.stack(frames), np.stack(actions)


def build_cuecard_data(args: argparse.Namespace) -> Path:
    output = data_path(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"[mem-jepa-v2] data exists: {output}")
        return output

    if args.base_data.exists():
        with np.load(args.base_data, allow_pickle=False) as data:
            frames = data["frames"][:args.episodes]
            actions = data["actions"][:args.episodes]
        print(f"[mem-jepa-v2] reused base frames: {args.base_data}")
    else:
        frames, actions = collect_base_frames(args)
        print("[mem-jepa-v2] collected fresh OGBench frames")

    count = len(frames)
    rng = np.random.default_rng(args.seed)
    labels = np.tile(np.arange(len(PALETTE), dtype=np.int64), int(np.ceil(count / len(PALETTE))))[:count]
    positions = np.tile(np.arange(len(CARD_POSITIONS), dtype=np.int64), int(np.ceil(count / len(CARD_POSITIONS))))[:count]
    rng.shuffle(labels)
    rng.shuffle(positions)

    np.savez_compressed(
        output,
        frames=frames.astype(np.uint8),
        actions=actions.astype(np.float32),
        cue_labels=labels.astype(np.int64),
        cue_positions=positions.astype(np.int64),
        palette=PALETTE,
        card_positions=CARD_POSITIONS,
        env_name=args.env_name,
        seed=args.seed,
        length=frames.shape[1],
        img_size=frames.shape[-2],
    )
    print(f"[mem-jepa-v2] wrote data: {output}")
    return output


def draw_cue_card(frame: np.ndarray, label: int, position: int, *, size: int = 12) -> np.ndarray:
    image = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(image)
    x, y = CARD_POSITIONS[int(position)]
    color = tuple(int(c) for c in PALETTE[int(label)])
    draw.rounded_rectangle([x - 2, y - 2, x + size + 2, y + size + 2], radius=3,
                           fill=(255, 255, 255), outline=(17, 24, 39), width=1)
    draw.rounded_rectangle([x, y, x + size, y + size], radius=2,
                           fill=color, outline=color)
    return np.asarray(image, dtype=np.uint8)


def crop_card(frame: np.ndarray, position: int, *, size: int = 18) -> np.ndarray:
    x, y = CARD_POSITIONS[int(position)]
    pad = max(0, (size - 12) // 2)
    x0 = int(np.clip(x - pad, 0, frame.shape[1] - size))
    y0 = int(np.clip(y - pad, 0, frame.shape[0] - size))
    return frame[y0:y0 + size, x0:x0 + size]


def to_tensor(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)


class CueCardPointMazeDataset(Dataset):
    def __init__(self, archive: Path, *, age: int, split: str,
                 seed: int = 0, visible_count: int = 10) -> None:
        with np.load(archive, allow_pickle=False) as data:
            self.frames = data["frames"]
            self.actions = data["actions"]
            self.labels = data["cue_labels"]
            self.positions = data["cue_positions"]
        self.age = int(age)
        self.visible_count = int(visible_count)
        self.endpoint = self.frames.shape[1] - 3
        self.cue_time = self.endpoint - self.age
        if self.cue_time < 0:
            raise ValueError("episode length too short for requested age")
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.frames))
        cut = int(0.82 * len(order))
        self.index = order[:cut] if split == "train" else order[cut:]

    def __len__(self) -> int:
        return len(self.index)

    def _frame(self, episode: int, time: int, *, cue: bool) -> np.ndarray:
        frame = self.frames[episode, time]
        if cue and self.cue_time <= time <= min(self.cue_time + 1, self.endpoint):
            return draw_cue_card(frame, int(self.labels[episode]), int(self.positions[episode]))
        return frame

    def _visible_times(self, *, mode: str) -> list[int]:
        if mode == "short":
            start = max(0, self.endpoint - self.visible_count + 1)
            return list(range(start, self.endpoint + 1))[-self.visible_count:]
        dense = np.linspace(0, self.endpoint, num=max(self.visible_count - 1, 2), dtype=int).tolist()
        times = []
        for t in [*dense, self.cue_time + 1, self.endpoint]:
            t = int(np.clip(t, 0, self.endpoint))
            if t == self.cue_time:
                continue
            if t not in times:
                times.append(t)
        while len(times) < self.visible_count:
            times.append(self.endpoint)
        return times[:self.visible_count]

    def _visible_stack(self, episode: int, *, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        times = self._visible_times(mode=mode)
        cue_visible = mode == "full"
        frames = [to_tensor(self._frame(episode, t, cue=cue_visible)) for t in times]
        time_tensor = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)
        time_tensor /= float(max(self.frames.shape[1] - 1, 1))
        return torch.stack(frames), time_tensor

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode = int(self.index[item])
        label = int(self.labels[episode])
        position = int(self.positions[episode])
        target_frame = self._frame(episode, self.cue_time, cue=True)
        target_crop = crop_card(target_frame, position)
        full, full_times = self._visible_stack(episode, mode="full")
        no_cue, no_cue_times = self._visible_stack(episode, mode="no_cue")
        short, short_times = self._visible_stack(episode, mode="short")
        query = torch.tensor([
            self.cue_time / float(max(self.frames.shape[1] - 1, 1)),
            CARD_POSITIONS[position, 0] / float(self.frames.shape[-2]),
            CARD_POSITIONS[position, 1] / float(self.frames.shape[-3]),
        ], dtype=torch.float32)
        actions = torch.from_numpy(self.actions[episode].astype(np.float32)).flatten()
        return {
            "visible": full,
            "times": full_times,
            "visible_no_cue": no_cue,
            "times_no_cue": no_cue_times,
            "visible_short": short,
            "times_short": short_times,
            "target_crop": to_tensor(target_crop),
            "query": query,
            "actions": actions,
            "label": torch.tensor(label, dtype=torch.long),
            "position": torch.tensor(position, dtype=torch.long),
            "episode": torch.tensor(episode, dtype=torch.long),
        }


class PatchFrameEncoder(nn.Module):
    def __init__(self, dim: int, *, patch: int = 8, image_size: int = 64) -> None:
        super().__init__()
        self.grid = image_size // patch
        self.patch_size = patch
        self.patch = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.raw_proj = nn.Sequential(nn.Linear(7, dim), nn.LayerNorm(dim), nn.SiLU(),
                                      nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.pos = nn.Parameter(torch.randn(self.grid * self.grid, dim) * 0.02)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        learned = self.patch(x).flatten(2).transpose(1, 2)
        pooled = F.avg_pool2d(x, kernel_size=self.patch_size, stride=self.patch_size)
        pooled = pooled.flatten(2).transpose(1, 2)
        centered = pooled - pooled.mean(dim=1, keepdim=True)
        saturation = pooled.max(dim=2, keepdim=True).values - pooled.min(dim=2, keepdim=True).values
        raw = self.raw_proj(torch.cat([pooled - 0.5, centered, saturation], dim=2))
        tokens = learned + raw
        tokens = self.norm(tokens + self.pos.unsqueeze(0))
        return self.norm(tokens + self.ff(tokens))


def raw_patch_features(x: torch.Tensor, *, patch_size: int = 8) -> torch.Tensor:
    pooled = F.avg_pool2d(x, kernel_size=patch_size, stride=patch_size)
    pooled = pooled.flatten(2).transpose(1, 2)
    centered = pooled - pooled.mean(dim=1, keepdim=True)
    saturation = pooled.max(dim=2, keepdim=True).values - pooled.min(dim=2, keepdim=True).values
    return torch.cat([pooled - 0.5, centered, saturation], dim=2)


class DenseCropTargetEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        raw_dim = 14
        projection = torch.randn(raw_dim, dim) / raw_dim ** 0.5
        self.register_buffer("projection", projection)

    def forward(self, crop: torch.Tensor) -> torch.Tensor:
        # The first v2 attempt used a dense crop projection.  It still let
        # background/wall pixels dominate the cue-card color.  v2.1 focuses the
        # stop-gradient target on the central card region while remaining a
        # latent target, not a pixel decoder.
        _, _, height, width = crop.shape
        y0, y1 = height // 6, height - height // 6
        x0, x1 = width // 6, width - width // 6
        center = crop[:, :, y0:y1, x0:x1]
        mean = center.mean(dim=(2, 3))
        std = center.std(dim=(2, 3), unbiased=False)
        maximum = center.amax(dim=(2, 3))
        minimum = center.amin(dim=(2, 3))
        saturation = (maximum - minimum).mean(dim=1, keepdim=True)
        brightness = mean.mean(dim=1, keepdim=True)
        color_ratio = mean / mean.sum(dim=1, keepdim=True).clamp_min(1e-4)
        features = torch.cat([
            mean - 0.5,
            std,
            maximum - minimum,
            saturation,
            brightness - 0.5,
            color_ratio - (1.0 / 3.0),
        ], dim=1)
        return F.normalize(features @ self.projection, dim=-1)


class SlotCompiler(nn.Module):
    def __init__(self, dim: int, slots: int, heads: int) -> None:
        super().__init__()
        del heads
        self.slots = slots
        self.assign = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, slots))
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Direct learned assignment pooling.  This remains compact and
        # non-recurrent, but avoids the failure where random learned queries
        # ignore small saturated cue patches.
        logits = self.assign(tokens).transpose(1, 2)
        weights = F.softmax(logits, dim=-1)
        slots = weights @ tokens
        slots = self.norm(slots + self.ff(slots))
        return slots, weights


class QueryPredictor(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))

    def forward(self, slots: torch.Tensor, query_meta: torch.Tensor) -> torch.Tensor:
        query = self.query(query_meta).unsqueeze(1)
        context, _ = self.attn(query, slots, slots, need_weights=False)
        out = self.norm(query + context).squeeze(1)
        return F.normalize(self.head(out), dim=-1)


class MemJEPAV2(nn.Module):
    def __init__(self, *, dim: int, slots: int, heads: int,
                 action_dim: int, visible_count: int, image_size: int = 64) -> None:
        super().__init__()
        self.frame_encoder = PatchFrameEncoder(dim, image_size=image_size)
        self.target_encoder = DenseCropTargetEncoder(dim)
        self.time_proj = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action_proj = nn.Sequential(nn.Linear(action_dim, dim), nn.LayerNorm(dim),
                                         nn.SiLU(), nn.Linear(dim, dim))
        self.saliency_proj = nn.Sequential(nn.Linear(7, dim), nn.LayerNorm(dim), nn.SiLU(),
                                           nn.Linear(dim, dim))
        self.view_pos = nn.Parameter(torch.randn(visible_count, dim) * 0.02)
        self.action_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.compiler = SlotCompiler(dim, slots, heads)
        self.predictor = QueryPredictor(dim, heads)
        self.prototype_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, len(PALETTE)))

    def encode(self, visible: torch.Tensor, times: torch.Tensor,
               actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, views = visible.shape[:2]
        frames = visible.reshape(batch * views, *visible.shape[2:])
        patch_tokens = self.frame_encoder(frames).reshape(batch, views, -1, self.view_pos.shape[-1])
        patch_tokens = patch_tokens + self.time_proj(times).unsqueeze(2)
        patch_tokens = patch_tokens + self.view_pos[:views].view(1, views, 1, -1)
        raw = raw_patch_features(frames)
        saliency = raw[:, :, -1]
        pick = saliency.argmax(dim=1)
        saliency_features = raw[torch.arange(raw.shape[0], device=raw.device), pick]
        saliency_tokens = self.saliency_proj(saliency_features).reshape(batch, views, -1)
        saliency_tokens = saliency_tokens + self.time_proj(times) + self.view_pos[:views].unsqueeze(0)
        patch_tokens = patch_tokens.flatten(1, 2)
        patch_tokens = torch.cat([patch_tokens, saliency_tokens], dim=1)
        action = self.action_proj(actions).unsqueeze(1) + self.action_token.view(1, 1, -1)
        tokens = torch.cat([patch_tokens, action], dim=1)
        return self.compiler(tokens)

    def forward(self, batch: dict[str, torch.Tensor], *, mode: str = "full") -> dict[str, torch.Tensor]:
        if mode == "full":
            visible, times = batch["visible"], batch["times"]
        elif mode == "no_cue":
            visible, times = batch["visible_no_cue"], batch["times_no_cue"]
        elif mode == "short":
            visible, times = batch["visible_short"], batch["times_short"]
        else:
            raise ValueError(f"unknown mode: {mode}")
        slots, attn = self.encode(visible, times, batch["actions"])
        pred = self.predictor(slots, batch["query"])
        return {
            "pred": pred,
            "proto_logits": self.prototype_head(pred),
            "slots": slots,
            "attn": attn,
        }

    @torch.no_grad()
    def target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.target_encoder(batch["target_crop"])


@torch.no_grad()
def target_prototype(crop: torch.Tensor) -> torch.Tensor:
    """Extract a teacher prototype from the hidden target crop.

    This uses only the masked target crop, not the generated label.  It is the
    cue-card analogue of a self-supervised teacher assignment: the student must
    infer this target from the visible episode evidence.
    """
    _, _, height, width = crop.shape
    center = crop[:, :, height // 6:height - height // 6, width // 6:width - width // 6]
    mean_rgb = center.mean(dim=(2, 3))
    palette = torch.as_tensor(PALETTE, dtype=crop.dtype, device=crop.device) / 255.0
    distance = torch.cdist(mean_rgb, palette)
    return distance.argmin(dim=1)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target.detach(), dim=-1)).mean()


def contrastive_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    """Soft target-target relational contrast.

    Cue-card labels are deliberately balanced, so many samples share the same
    latent target.  A hard InfoNCE label would incorrectly treat same-cue samples
    as negatives.  This JEPA-style relational target keeps same-target examples
    compatible without using cue labels.
    """
    logits = pred @ target.detach().T / temperature
    with torch.no_grad():
        teacher = F.softmax((target @ target.T) / temperature, dim=1)
    return -(teacher * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def std_loss(features: torch.Tensor, floor: float = 0.08) -> torch.Tensor:
    std = torch.sqrt(features.float().var(dim=0, unbiased=False) + 1e-4)
    return F.relu(floor - std).mean()


def covariance_loss(features: torch.Tensor) -> torch.Tensor:
    x = features.float() - features.float().mean(dim=0, keepdim=True)
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).mean()


@torch.no_grad()
def canonical_embeddings(model: MemJEPAV2, device: torch.device) -> torch.Tensor:
    crops = []
    blank = np.full((64, 64, 3), 235, dtype=np.uint8)
    for label in range(len(PALETTE)):
        frame = draw_cue_card(blank, label, 0)
        crops.append(to_tensor(crop_card(frame, 0)))
    crop_tensor = torch.stack(crops).to(device)
    return model.target_encoder(crop_tensor)


@torch.no_grad()
def eval_loader(model: MemJEPAV2, loader: DataLoader,
                device: torch.device) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    all_pred, all_target = [], []
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        target = model.target(batch)
        labels = batch["label"]
        teacher = target_prototype(batch["target_crop"])
        batch_size = int(labels.shape[0])
        count += batch_size
        for mode in ("full", "no_cue", "short"):
            out = model(batch, mode=mode)
            pred = out["pred"]
            cue_pred = out["proto_logits"].argmax(dim=1)
            sums[f"{mode}_cue_acc"] = sums.get(f"{mode}_cue_acc", 0.0) + (
                cue_pred == labels).float().sum().item()
            sums[f"{mode}_teacher_acc"] = sums.get(f"{mode}_teacher_acc", 0.0) + (
                cue_pred == teacher).float().sum().item()
            sums[f"{mode}_cosine"] = sums.get(f"{mode}_cosine", 0.0) + (
                F.cosine_similarity(pred, target, dim=-1).sum().item())
            if mode == "full":
                all_pred.append(pred)
                all_target.append(target)
                sums["pred_std"] = sums.get("pred_std", 0.0) + float(pred.std(dim=0).mean()) * batch_size
                sums["target_std"] = sums.get("target_std", 0.0) + float(target.std(dim=0).mean()) * batch_size
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    if all_pred:
        pred = torch.cat(all_pred)
        target = torch.cat(all_target)
        sim = pred @ target.T
        rank = torch.argsort(sim, dim=1, descending=True)
        truth = torch.arange(len(pred), device=device).unsqueeze(1)
        eye = torch.eye(len(pred), dtype=torch.bool, device=device)
        metrics["full_retrieval_top1"] = (rank[:, :1] == truth).any(dim=1).float().mean().item()
        metrics["full_retrieval_top5"] = (rank[:, :5] == truth).any(dim=1).float().mean().item()
        metrics["full_retrieval_margin"] = (
            sim.diag() - sim.masked_fill(eye, -9).max(dim=1).values).mean().item()
    return metrics


def train_epoch(model: MemJEPAV2, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.no_grad():
            target = model.target(batch)
            teacher = target_prototype(batch["target_crop"])
        out = model(batch, mode="full")
        pred = out["pred"]
        slots = out["slots"].flatten(0, 1)
        losses = {
            "target": cosine_loss(pred, target),
            "contrast": contrastive_loss(pred, target, args.temperature),
            "prototype": F.cross_entropy(out["proto_logits"], teacher),
            "std": std_loss(pred) + 0.5 * std_loss(slots),
            "cov": covariance_loss(pred),
        }
        loss = (
            args.target_weight * losses["target"]
            + args.contrast_weight * losses["contrast"]
            + args.proto_weight * losses["prototype"]
            + args.std_weight * losses["std"]
            + args.cov_weight * losses["cov"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_size = int(batch["label"].shape[0])
        count += batch_size
        for key, value in {"loss": loss, **losses}.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach()) * batch_size
    return {key: value / max(count, 1) for key, value in sums.items()}


def make_curves(run_dir: Path, history: list[dict[str, float]],
                summary: dict[str, object]) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].set_title("Training objective")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("weighted loss")
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, [r["val_full_cue_acc"] for r in history], label="full")
    axes[1].plot(epochs, [r["val_no_cue_cue_acc"] for r in history], label="no cue")
    axes[1].plot(epochs, [r["val_short_cue_acc"] for r in history], label="short")
    axes[1].axhline(1.0 / len(PALETTE), color="#111827", linestyle="--", linewidth=1, label="chance")
    axes[1].set_title("Cue identity accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(epochs, [r["val_full_retrieval_top1"] for r in history], label="top-1")
    axes[2].plot(epochs, [r["val_full_retrieval_top5"] for r in history], label="top-5")
    axes[2].set_title("Hard retrieval")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylim(0, 1.02)
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Mem-JEPA v2 cue-card diagnostic · age {summary['age']}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=190)
    plt.close(fig)


@torch.no_grad()
def make_panel(model: MemJEPAV2, dataset: CueCardPointMazeDataset,
               run_dir: Path, device: torch.device) -> None:
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=False)
    batch = next(iter(loader))
    batch_dev = {key: value.to(device) for key, value in batch.items()}
    out = model(batch_dev, mode="full")
    pred_label = out["proto_logits"].argmax(dim=1).cpu().numpy()
    labels = batch["label"].numpy()
    fig, axes = plt.subplots(3, 6, figsize=(9.6, 5.0))
    names = ["full cue", "no cue", "short only"]
    keys = ["visible", "visible_no_cue", "visible_short"]
    for row, (name, key) in enumerate(zip(names, keys, strict=True)):
        frames = batch[key][0].permute(0, 2, 3, 1).numpy()
        for col in range(4):
            axes[row, col].imshow(frames[min(col, len(frames) - 1)])
            axes[row, col].set_title(f"{name} t{col}", fontsize=8)
        axes[row, 4].imshow(batch["target_crop"][0].permute(1, 2, 0).numpy())
        axes[row, 4].set_title("target crop", fontsize=8)
        axes[row, 5].axis("off")
        axes[row, 5].text(0.05, 0.62, f"true: {int(labels[0])}", fontsize=11)
        axes[row, 5].text(0.05, 0.42, f"pred: {int(pred_label[0])}", fontsize=11)
        axes[row, 5].text(0.05, 0.22, "metric only", fontsize=9, color="#5f6b7a")
        for col in range(5):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle("Cue-card flow: visible cue vs. controls", fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "cue_flow_panel.png", dpi=190)
    plt.close(fig)


def make_gif(dataset: CueCardPointMazeDataset, run_dir: Path) -> None:
    sample = dataset[0]
    full = (sample["visible"].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    frames = []
    for i, frame in enumerate(full):
        canvas = Image.new("RGB", (230, 185), "white")
        im = Image.fromarray(frame).resize((140, 140), Image.Resampling.NEAREST)
        canvas.paste(im, (45, 10))
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle([41, 6, 189, 154], radius=12,
                               outline=(118, 185, 0), width=4)
        draw.text((18, 164), f"visible frame {i}: cue may appear then vanish",
                  fill=(17, 24, 39))
        frames.append(np.asarray(canvas))
    target = (sample["target_crop"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    canvas = Image.new("RGB", (230, 185), "white")
    im = Image.fromarray(target).resize((120, 120), Image.Resampling.NEAREST)
    canvas.paste(im, (55, 20))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle([51, 16, 179, 144], radius=12,
                           outline=(245, 158, 11), width=4)
    draw.text((35, 164), "masked target crop", fill=(17, 24, 39))
    frames.append(np.asarray(canvas))
    imageio.mimsave(run_dir / "cue_mask_sequence.gif", frames, duration=0.58)


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    archive = data_path(args)
    if not archive.exists():
        build_cuecard_data(args)
    run_dir = args.output / "runs" / f"age{args.age}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_data = CueCardPointMazeDataset(
        archive, age=args.age, split="train", seed=args.seed,
        visible_count=args.visible_count)
    val_data = CueCardPointMazeDataset(
        archive, age=args.age, split="val", seed=args.seed,
        visible_count=args.visible_count)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    action_dim = int(train_data.actions.shape[1] * train_data.actions.shape[2])
    model = MemJEPAV2(
        dim=args.dim, slots=args.slots, heads=args.heads,
        action_dim=action_dim, visible_count=args.visible_count,
        image_size=args.img_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args)
        val_metrics = eval_loader(model, val_loader, device)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    final = history[-1]
    positive = (
        final["val_full_cue_acc"] >= 0.75
        and final["val_no_cue_cue_acc"] <= 0.40
        and final["val_short_cue_acc"] <= 0.40
    )
    summary = {
        "schema": "mem_jepa_v2_2_cuecard_run_v1",
        "age": args.age,
        "seed": args.seed,
        "epochs": args.epochs,
        "device": str(device),
        "data": str(archive),
        "final": final,
        "gate": {
            "target": "full cue accuracy >= 0.75 and no-cue/short controls <= 0.40",
            "passed": bool(positive),
            "teacher_note": "teacher prototype is extracted from the masked target crop; generated cue labels are metrics only",
        },
        "artifacts": {
            "curves": str(run_dir / "curves.png"),
            "cue_flow_panel": str(run_dir / "cue_flow_panel.png"),
            "cue_mask_sequence": str(run_dir / "cue_mask_sequence.gif"),
        },
        "claim_boundary": (
            "Cue-card v2 is a self-supervised first gate only. Passing permits "
            "host-exposure experiments; it is not a paper-level memory claim."
        ),
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "summary": summary}, run_dir / "checkpoint.pt")
    make_curves(run_dir, history, summary)
    make_panel(model, val_data, run_dir, device)
    make_gif(val_data, run_dir)
    return run_dir


def aggregate(args: argparse.Namespace) -> Path:
    summaries = []
    for path in sorted((args.output / "runs").glob("age*_seed*/summary.json")):
        summaries.append(json.loads(path.read_text()))
    if not summaries:
        raise SystemExit("no completed run summaries found")
    summary = {
        "schema": "mem_jepa_v2_2_cuecard_aggregate_v1",
        "runs": summaries,
        "by_age": {},
        "all_gates_passed": all(run["gate"]["passed"] for run in summaries),
    }
    for age in sorted({run["age"] for run in summaries}):
        candidates = [run for run in summaries if run["age"] == age]
        best = max(candidates, key=lambda run: run["final"]["val_full_cue_acc"])
        summary["by_age"][str(age)] = {
            "run": f"age{best['age']}_seed{best['seed']}",
            "full_cue_acc": best["final"]["val_full_cue_acc"],
            "no_cue_acc": best["final"]["val_no_cue_cue_acc"],
            "short_acc": best["final"]["val_short_cue_acc"],
            "retrieval_top1": best["final"]["val_full_retrieval_top1"],
            "retrieval_top5": best["final"]["val_full_retrieval_top5"],
            "gate_passed": best["gate"]["passed"],
        }
    out = args.output / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return out


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare_data:
        build_cuecard_data(args)
        return
    if args.aggregate:
        print(aggregate(args))
        return
    print(train(args))


if __name__ == "__main__":
    main()
