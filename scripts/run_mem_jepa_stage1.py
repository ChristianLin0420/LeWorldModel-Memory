#!/usr/bin/env python3
"""Mem-JEPA Stage-1 prototype on real OGBench PointMaze frames.

This is intentionally small and self-contained.  It tests the first training
mechanism only: JEPA-style masked latent target prediction from legal visible
episode fragments, with future and goal/exposure proxy heads.  It does not make
Paper-A claims and does not touch frozen LeWM/DINO-WM checkpoints.
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
DEFAULT_ROOT = ROOT / "outputs" / "mem_jepa_stage1"
ENV_NAME = "pointmaze-large-navigate-v0"
AGES = (4, 8, 15)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--env-name", default=ENV_NAME)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--episodes", type=int, default=384)
    parser.add_argument("--length", type=int, default=24)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--age", type=int, choices=AGES, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-decay", type=float, default=1.0,
                        help="1.0 keeps the target encoder fixed; <1.0 uses EMA")
    parser.add_argument("--std-weight", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def data_path(args: argparse.Namespace) -> Path:
    return args.data or (args.output / "data" / "pointmaze_stage1.npz")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((size, size),
                                                    Image.Resampling.BILINEAR))


def collect_pointmaze(args: argparse.Namespace) -> Path:
    """Collect a compact real-render cache from OGBench."""

    output = data_path(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"[mem-jepa] data exists: {output}")
        return output

    os.environ.setdefault("MUJOCO_GL", "egl")
    import ogbench  # noqa: WPS433

    rng = np.random.default_rng(args.seed)
    env = ogbench.make_env_and_datasets(args.env_name, env_only=True)
    frames, goals, actions = [], [], []
    for ep in range(args.episodes):
        task_id = int(ep % 5) + 1
        ob, info = env.reset(options={"task_id": task_id, "render_goal": True})
        del ob
        goal = info.get("goal_rendered")
        if goal is None:
            goal = env.render()
        episode_frames = [resize_frame(env.render(), args.img_size)]
        episode_actions = []
        for _ in range(args.length - 1):
            action = env.action_space.sample()
            # Small randomization avoids repeatedly following the exact same
            # action sampler stream after fixed resets.
            if rng.random() < 0.05:
                action = np.zeros_like(action)
            episode_actions.append(np.asarray(action, dtype=np.float32))
            env.step(action)
            episode_frames.append(resize_frame(env.render(), args.img_size))
        frames.append(np.stack(episode_frames).astype(np.uint8))
        goals.append(resize_frame(goal, args.img_size).astype(np.uint8))
        actions.append(np.stack(episode_actions).astype(np.float32))
    env.close()

    np.savez_compressed(
        output,
        frames=np.stack(frames),
        goals=np.stack(goals),
        actions=np.stack(actions),
        env_name=args.env_name,
        seed=args.seed,
        length=args.length,
        img_size=args.img_size,
    )
    print(f"[mem-jepa] wrote data: {output}")
    return output


class PointMazeMaskedDataset(Dataset):
    def __init__(self, archive: Path, *, age: int, split: str,
                 seed: int = 0) -> None:
        with np.load(archive, allow_pickle=False) as data:
            self.frames = data["frames"]
            self.goals = data["goals"]
            self.actions = data["actions"]
        if age < 4:
            raise ValueError("age must be >= 4")
        self.age = int(age)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.frames))
        cut = int(0.82 * len(order))
        index = order[:cut] if split == "train" else order[cut:]
        self.index = np.asarray(index, dtype=np.int64)
        self.endpoint = self.frames.shape[1] - 3
        if self.endpoint - self.age < 0:
            raise ValueError("episode length too short for requested age")
        self.visible_count = 8

    def __len__(self) -> int:
        return len(self.index)

    def _frame(self, episode: int, time: int) -> torch.Tensor:
        arr = self.frames[episode, time].astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode = int(self.index[item])
        endpoint = self.endpoint
        target_time = endpoint - self.age
        future_time = min(endpoint + 2, self.frames.shape[1] - 1)
        candidates = [
            0, target_time - 2, target_time - 1, target_time + 1,
            target_time + 2, endpoint - 3, endpoint - 2, endpoint - 1,
            endpoint,
        ]
        visible_times: list[int] = []
        for raw_time in candidates:
            time = int(np.clip(raw_time, 0, self.frames.shape[1] - 1))
            if time != target_time and time not in visible_times:
                visible_times.append(time)
        while len(visible_times) < self.visible_count:
            visible_times.append(endpoint)
        visible_times = visible_times[:self.visible_count]
        visible = torch.stack([self._frame(episode, t) for t in visible_times])
        times = torch.tensor(visible_times, dtype=torch.float32) / float(
            self.frames.shape[1] - 1)
        goal = torch.from_numpy(
            self.goals[episode].astype(np.float32) / 255.0).permute(2, 0, 1)
        actions = torch.from_numpy(self.actions[episode].astype(np.float32))
        return {
            "visible": visible,
            "times": times.unsqueeze(-1),
            "goal": goal,
            "actions": actions.flatten(),
            "target": self._frame(episode, target_time),
            "future": self._frame(episode, future_time),
            "episode": torch.tensor(episode, dtype=torch.long),
        }


class ImageEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(96 * 8 * 8, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class FixedPatchTargetEncoder(nn.Module):
    """Stop-gradient spatial target encoder.

    A tiny PointMaze dot can disappear under global feature pooling.  This fixed
    patch projection preserves spatial color layout while still training the
    student in latent space rather than reconstructing pixels.
    """

    def __init__(self, dim: int, grid: int = 16) -> None:
        super().__init__()
        self.grid = grid
        raw_dim = 7 * grid * grid
        projection = torch.randn(raw_dim, dim) / raw_dim ** 0.5
        self.register_buffer("projection", projection)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, (self.grid, self.grid))
        centered = pooled - pooled.mean(dim=(2, 3), keepdim=True)
        saturation = pooled.max(dim=1, keepdim=True).values - pooled.min(
            dim=1, keepdim=True).values
        features = torch.cat([pooled - 0.5, centered, saturation], dim=1)
        flat = features.flatten(1)
        return F.normalize(flat @ self.projection, dim=-1)


class EvidenceCompiler(nn.Module):
    def __init__(self, dim: int, slots: int, heads: int) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.randn(slots, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(),
                                nn.Linear(4 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.slots.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        slots, weights = self.attn(query, tokens, tokens,
                                   need_weights=True,
                                   average_attn_weights=False)
        slots = self.norm1(slots + query)
        slots = self.norm2(slots + self.ff(slots))
        return slots.mean(dim=1), weights.mean(dim=1)


class MemJEPA(nn.Module):
    def __init__(self, *, dim: int, slots: int, heads: int,
                 action_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.encoder = ImageEncoder(dim)
        self.target_encoder = FixedPatchTargetEncoder(dim)
        self.token_type = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.time_proj = nn.Sequential(nn.Linear(1, dim), nn.SiLU(),
                                       nn.Linear(dim, dim))
        self.action_proj = nn.Sequential(nn.Linear(action_dim, dim),
                                         nn.LayerNorm(dim), nn.SiLU(),
                                         nn.Linear(dim, dim))
        self.compiler = EvidenceCompiler(dim, slots, heads)
        self.mask_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.future_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.exposure_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    @torch.no_grad()
    def update_target(self, decay: float = 1.0) -> None:
        if decay >= 1.0:
            return
        for target, source in zip(self.target_encoder.parameters(),
                                  self.encoder.parameters(), strict=True):
            target.data.mul_(decay).add_(source.data, alpha=1.0 - decay)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visible = batch["visible"]
        batch_size, view_count = visible.shape[:2]
        images = visible.reshape(batch_size * view_count, *visible.shape[2:])
        view_tokens = self.encoder(images).reshape(batch_size, view_count, -1)
        view_tokens = view_tokens + self.time_proj(batch["times"])
        goal_token = self.encoder(batch["goal"]).unsqueeze(1)
        action_token = self.action_proj(batch["actions"]).unsqueeze(1)
        tokens = torch.cat([view_tokens, goal_token, action_token], dim=1)
        tokens = tokens + self.token_type[:tokens.shape[1]].unsqueeze(0)
        belief, attn = self.compiler(tokens)
        return {
            "target_pred": F.normalize(self.mask_head(belief), dim=-1),
            "future_pred": F.normalize(self.future_head(belief), dim=-1),
            "exposure_pred": F.normalize(self.exposure_head(belief), dim=-1),
            "belief": F.normalize(belief, dim=-1),
            "attn": attn,
        }

    @torch.no_grad()
    def targets(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "target": self.target_encoder(batch["target"]),
            "future": self.target_encoder(batch["future"]),
            "goal": self.target_encoder(batch["goal"]),
        }


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target.detach(), dim=-1)).mean()


def std_loss(features: torch.Tensor, floor: float = 0.04) -> torch.Tensor:
    std = torch.sqrt(features.float().var(dim=0, unbiased=False) + 1e-4)
    return F.relu(floor - std).mean()


@torch.no_grad()
def retrieval_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    sim = pred @ target.T
    rank = torch.argsort(sim, dim=1, descending=True)
    truth = torch.arange(len(pred), device=pred.device).unsqueeze(1)
    top1 = (rank[:, :1] == truth).any(dim=1).float().mean().item()
    top5 = (rank[:, :5] == truth).any(dim=1).float().mean().item()
    margin = (sim.diag() - sim.masked_fill(torch.eye(
        len(pred), dtype=torch.bool, device=pred.device), -9).max(dim=1).values
              ).mean().item()
    return {"top1": top1, "top5": top5, "margin": margin}


def run_epoch(model: MemJEPA, loader: DataLoader,
              optimizer: torch.optim.Optimizer | None, device: torch.device,
              *, target_decay: float, std_weight: float) -> dict[str, float]:
    model.train(optimizer is not None)
    sums: dict[str, float] = {}
    count = 0
    pred_target, gold_target = [], []
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True)
                 for key, value in batch.items()}
        with torch.no_grad():
            target = model.targets(batch)
        out = model(batch)
        losses = {
            "target": cosine_loss(out["target_pred"], target["target"]),
            "future": cosine_loss(out["future_pred"], target["future"]),
            "exposure": cosine_loss(out["exposure_pred"], target["goal"]),
        }
        anti_collapse = (
            std_loss(out["target_pred"])
            + std_loss(out["future_pred"])
            + std_loss(out["exposure_pred"])
            + std_loss(out["belief"])
        )
        losses["std"] = anti_collapse
        loss = (
            losses["target"]
            + 0.35 * losses["future"]
            + 0.25 * losses["exposure"]
            + std_weight * anti_collapse
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target(target_decay)
        batch_size = int(batch["visible"].shape[0])
        count += batch_size
        for key, value in {"loss": loss, **losses}.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach()) * batch_size
        sums["pred_std"] = sums.get("pred_std", 0.0) + float(
            out["target_pred"].detach().std(dim=0).mean()) * batch_size
        sums["target_std"] = sums.get("target_std", 0.0) + float(
            target["target"].detach().std(dim=0).mean()) * batch_size
        pred_target.append(out["target_pred"].detach())
        gold_target.append(target["target"].detach())
    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    if pred_target:
        metrics.update({f"target_retrieval_{key}": value for key, value in
                        retrieval_metrics(torch.cat(pred_target),
                                          torch.cat(gold_target)).items()})
    return metrics


def make_plots(run_dir: Path, history: list[dict[str, float]],
               summary: dict[str, object]) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="validation")
    axes[0].set_title("JEPA latent loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("weighted cosine loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, [r["val_target_retrieval_top1"] for r in history],
                 label="top-1")
    axes[1].plot(epochs, [r["val_target_retrieval_top5"] for r in history],
                 label="top-5")
    axes[1].set_title("Masked-target retrieval")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation retrieval")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle(f"Mem-JEPA Stage-1 · age {summary['age']}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=180)
    plt.close(fig)


@torch.no_grad()
def make_retrieval_panel(model: MemJEPA, dataset: PointMazeMaskedDataset,
                         run_dir: Path, device: torch.device) -> None:
    loader = DataLoader(dataset, batch_size=min(96, len(dataset)), shuffle=False)
    batch = next(iter(loader))
    batch_dev = {key: value.to(device) for key, value in batch.items()}
    target = model.targets(batch_dev)["target"]
    pred = model(batch_dev)["target_pred"]
    nearest = (pred @ target.T).argmax(dim=1).cpu().numpy()
    chosen = [0, min(1, len(nearest) - 1), min(2, len(nearest) - 1)]
    fig, axes = plt.subplots(len(chosen), 6, figsize=(9.0, 1.8 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None]
    for row, idx in enumerate(chosen):
        visible = batch["visible"][idx].permute(0, 2, 3, 1).numpy()
        true_target = batch["target"][idx].permute(1, 2, 0).numpy()
        retrieved = batch["target"][int(nearest[idx])].permute(1, 2, 0).numpy()
        for col in range(4):
            axes[row, col].imshow(visible[col])
            axes[row, col].set_title(f"visible {col}", fontsize=8)
        axes[row, 4].imshow(true_target)
        axes[row, 4].set_title("true masked", fontsize=8)
        axes[row, 5].imshow(retrieved)
        axes[row, 5].set_title("nearest pred", fontsize=8)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Embedding retrieval diagnostic (analysis only)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(run_dir / "retrieval_panel.png", dpi=180)
    plt.close(fig)


def make_mask_gif(dataset: PointMazeMaskedDataset, run_dir: Path) -> None:
    sample = dataset[0]
    frames = []
    visible = (sample["visible"].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    target = (sample["target"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    sequence = list(visible) + [target]
    labels = ["visible context"] * len(visible) + ["masked target"]
    colors = [(118, 185, 0)] * len(visible) + [(245, 158, 11)]
    for image, label, color in zip(sequence, labels, colors, strict=True):
        canvas = Image.new("RGB", (220, 180), "white")
        im = Image.fromarray(image).resize((140, 140), Image.Resampling.NEAREST)
        canvas.paste(im, (40, 12))
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle([36, 8, 184, 156], radius=12,
                               outline=color, width=4)
        draw.text((20, 162), label, fill=(17, 24, 39))
        frames.append(np.asarray(canvas))
    imageio.mimsave(run_dir / "mask_sequence.gif", frames, duration=0.65)


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    archive = data_path(args)
    if not archive.exists():
        collect_pointmaze(args)
    run_dir = args.output / "runs" / f"age{args.age}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_data = PointMazeMaskedDataset(archive, age=args.age, split="train",
                                        seed=args.seed)
    val_data = PointMazeMaskedDataset(archive, age=args.age, split="val",
                                      seed=args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    action_dim = int(train_data.actions.shape[1] * train_data.actions.shape[2])
    model = MemJEPA(dim=args.dim, slots=args.slots, heads=args.heads,
                    action_dim=action_dim,
                    num_tokens=train_data.visible_count + 2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, opt, device,
            target_decay=args.target_decay, std_weight=args.std_weight)
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, None, device,
                target_decay=args.target_decay, std_weight=args.std_weight)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    summary = {
        "schema": "mem_jepa_stage1_run_v1",
        "env_name": args.env_name,
        "age": args.age,
        "seed": args.seed,
        "epochs": args.epochs,
        "device": str(device),
        "target_decay": args.target_decay,
        "std_weight": args.std_weight,
        "data": str(archive),
        "final": history[-1],
        "artifacts": {
            "curves": str(run_dir / "curves.png"),
            "retrieval_panel": str(run_dir / "retrieval_panel.png"),
            "mask_sequence": str(run_dir / "mask_sequence.gif"),
        },
        "claim_boundary": (
            "Stage-1 diagnostic only: masked latent prediction and retrieval "
            "metrics are not Paper-A memory claims."
        ),
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "summary": summary},
               run_dir / "checkpoint.pt")
    make_plots(run_dir, history, summary)
    make_retrieval_panel(model, val_data, run_dir, device)
    make_mask_gif(val_data, run_dir)
    return run_dir


def aggregate(args: argparse.Namespace) -> Path:
    summaries = []
    for path in sorted((args.output / "runs").glob("age*_seed*/summary.json")):
        summaries.append(json.loads(path.read_text()))
    if not summaries:
        raise SystemExit("no completed run summaries found")
    summary = {
        "schema": "mem_jepa_stage1_aggregate_v1",
        "runs": summaries,
        "best_by_age": {},
    }
    for age in sorted({run["age"] for run in summaries}):
        candidates = [run for run in summaries if run["age"] == age]
        best = max(candidates,
                   key=lambda r: r["final"]["val_target_retrieval_top5"])
        summary["best_by_age"][str(age)] = {
            "run": f"age{best['age']}_seed{best['seed']}",
            "val_loss": best["final"]["val_loss"],
            "target_top1": best["final"]["val_target_retrieval_top1"],
            "target_top5": best["final"]["val_target_retrieval_top5"],
            "target_margin": best["final"]["val_target_retrieval_margin"],
        }
    out = args.output / "summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return out


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.prepare_data:
        collect_pointmaze(args)
        return
    if args.aggregate:
        print(aggregate(args))
        return
    print(train(args))


if __name__ == "__main__":
    main()
