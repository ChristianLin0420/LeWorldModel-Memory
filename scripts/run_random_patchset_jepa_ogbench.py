#!/usr/bin/env python3
"""Random patch-set JEPA memory stage on OGBench renders.

This is the stricter non-manual successor to ``run_random_target_jepa_ogbench``.
It does not use cue labels, cue crops, cue positions, or a manually selected
key frame in the training objective.  For each trajectory it mines target
patches from the causal stream using generic saliency/novelty scores, masks
those patches in the visible history, and predicts their stop-gradient latent
set from a compact slot bottleneck.

Cue labels are used only after training for the post-hoc audit readout.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_masked_evidence_jepa_ogbench as base  # noqa: E402


PATCH_SIZE = 16
TARGET_PATCHES = 8
MIN_SEPARATION = 6


def _safe_crop(frame: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
    return frame[int(y): int(y) + size, int(x): int(x) + size].copy()


def _mask_patch(frame: np.ndarray, y: int, x: int, size: int) -> None:
    frame[int(y): int(y) + size, int(x): int(x) + size] = np.asarray([18, 18, 18], dtype=np.uint8)


def _saliency_map(frames: np.ndarray, time: int) -> np.ndarray:
    frame = frames[int(time)].astype(np.float32)
    gray = frame.mean(axis=-1)
    local = np.abs(gray - gray.mean())
    if time > 0:
        prev = frames[int(time) - 1].astype(np.float32).mean(axis=-1)
        delta = np.abs(gray - prev)
    else:
        delta = np.zeros_like(gray)
    saturation = frame.max(axis=-1) - frame.min(axis=-1)
    return 0.45 * local + 0.35 * delta + 0.20 * saturation


def _patch_score(score: np.ndarray, y: int, x: int, size: int) -> float:
    return float(score[int(y): int(y) + size, int(x): int(x) + size].mean())


def _too_close(candidate: tuple[int, int, int], selected: list[tuple[int, int, int]]) -> bool:
    t, y, x = candidate
    for old_t, old_y, old_x in selected:
        if t == old_t and abs(y - old_y) < MIN_SEPARATION and abs(x - old_x) < MIN_SEPARATION:
            return True
    return False


def mine_patch_targets(
    frames: np.ndarray,
    *,
    endpoint: int,
    rng: np.random.Generator,
    variant: str,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Select target patches without labels or cue-layout knowledge."""

    if variant == "no_state":
        candidate_times = list(range(max(0, endpoint - 3), endpoint + 1))
    else:
        candidate_times = list(range(0, endpoint + 1))
    size = int(PATCH_SIZE)
    h, w = frames.shape[1:3]
    stride = max(4, size // 2)
    candidates: list[tuple[float, int, int, int]] = []
    for time in candidate_times:
        score = _saliency_map(frames, int(time))
        for y in range(0, max(1, h - size + 1), stride):
            for x in range(0, max(1, w - size + 1), stride):
                value = _patch_score(score, y, x, size)
                value += float(rng.normal(0.0, 1e-3))
                candidates.append((value, int(time), int(y), int(x)))
    candidates.sort(reverse=True, key=lambda row: row[0])
    selected: list[tuple[int, int, int]] = []
    for _, time, y, x in candidates:
        candidate = (time, y, x)
        if _too_close(candidate, selected):
            continue
        selected.append(candidate)
        if len(selected) >= TARGET_PATCHES:
            break
    while len(selected) < TARGET_PATCHES:
        selected.append(
            (
                int(rng.choice(candidate_times)),
                int(rng.integers(0, max(1, h - size + 1))),
                int(rng.integers(0, max(1, w - size + 1))),
            )
        )
    patches = np.stack([_safe_crop(frames[t], y, x, size) for t, y, x in selected]).astype(np.uint8)
    return patches, selected


class RandomPatchSetDataset(Dataset):
    """Dataset with mined patch-set targets and masked visible source patches."""

    def __init__(
        self,
        archive: Path,
        *,
        age: int,
        split: str,
        seed: int,
        validation_fraction: float,
        variant: str = "full",
        augment: bool = False,
        temporal_drop: float = 0.0,
        patch_drop: float = 0.0,
    ) -> None:
        with np.load(archive, allow_pickle=False) as data:
            self.frames = data["frames"]
            self.actions = data["actions"]
            self.labels = data["cue_labels"]
            self.positions = data["cue_positions"]
        self.age = int(age)
        self.endpoint = base.LAST_CUE_FRAME + self.age
        self.variant = str(variant)
        self.augment = bool(augment)
        self.temporal_drop = float(temporal_drop)
        self.patch_drop = float(patch_drop)
        rng = np.random.default_rng(97_101 + int(seed))
        order = rng.permutation(len(self.frames))
        val_count = max(base.CLASSES, int(round(len(order) * float(validation_fraction))))
        self.indices = order[:-val_count] if split == "train" else order[-val_count:]
        if self.endpoint >= self.frames.shape[1]:
            raise ValueError(f"age {age} exceeds cached sequence length")

    def __len__(self) -> int:
        return int(len(self.indices))

    def _valid_times(self, rng: np.random.Generator) -> list[int]:
        if self.variant == "no_state":
            return list(range(max(0, self.endpoint - 3), self.endpoint + 1))
        times = list(range(0, self.endpoint + 1))
        if self.augment and self.temporal_drop > 0:
            kept = [time for time in times if time == self.endpoint or rng.random() > self.temporal_drop]
            if not kept:
                kept = [self.endpoint]
            times = sorted(set(kept))
        return times

    def _maybe_mask_random_patch(self, frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.augment or rng.random() >= self.patch_drop:
            return frame
        out = frame.copy()
        size = max(6, frame.shape[0] // 5)
        x = int(rng.integers(0, max(1, frame.shape[1] - size)))
        y = int(rng.integers(0, max(1, frame.shape[0] - size)))
        _mask_patch(out, y, x, size)
        return out

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode = int(self.indices[item])
        label = int(self.labels[episode])
        position = int(self.positions[episode])
        rng = np.random.default_rng(30_000_011 + episode + 337 * self.age)
        clean = self.frames[episode].copy()
        full = base.inject_cue_sequence(clean, label, position)
        source = clean if self.variant in {"reset", "no_state"} else full
        patches, selected = mine_patch_targets(source, endpoint=self.endpoint, rng=rng, variant=self.variant)

        visible = source.copy()
        if self.variant != "no_state":
            for time, y, x in selected:
                _mask_patch(visible[int(time)], int(y), int(x), PATCH_SIZE)
        times = self._valid_times(rng)
        frame_tokens = np.zeros(
            (base.MAX_CONTEXT, clean.shape[-3], clean.shape[-2], clean.shape[-1]),
            dtype=np.uint8,
        )
        action_tokens = np.zeros((base.MAX_CONTEXT, self.actions.shape[-1]), dtype=np.float32)
        time_tokens = np.zeros((base.MAX_CONTEXT, 1), dtype=np.float32)
        valid = np.zeros((base.MAX_CONTEXT,), dtype=np.float32)
        for slot, time in enumerate(times[:base.MAX_CONTEXT]):
            frame_tokens[slot] = self._maybe_mask_random_patch(visible[time], rng)
            if time > 0:
                action_tokens[slot] = self.actions[episode, time - 1]
            time_tokens[slot, 0] = float(time) / float(base.LENGTH - 1)
            valid[slot] = 1.0
        return {
            "frames": torch.from_numpy(frame_tokens.astype(np.float32) / 255.0).permute(0, 3, 1, 2),
            "actions": torch.from_numpy(action_tokens),
            "times": torch.from_numpy(time_tokens),
            "valid": torch.from_numpy(valid),
            "target_patches": torch.from_numpy(patches.astype(np.float32) / 255.0).permute(0, 3, 1, 2),
            "label": torch.tensor(label, dtype=torch.long),
        }


class RandomPatchSetJEPA(nn.Module):
    def __init__(self, *, img_size: int, action_dim: int, dim: int, slots: int, heads: int, chunk: int = 0) -> None:
        super().__init__()
        self.frame = base.FrameEncoder(dim, img_size)
        self.patch = base.FrameEncoder(dim, PATCH_SIZE)
        self.action = nn.Sequential(nn.Linear(action_dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.chunk = int(chunk)
        if self.chunk > 0:
            self.memory = base.StreamingSlotMemory(dim, slots, heads, chunk=self.chunk)
        else:
            self.memory = base.SlotMemory(dim, slots, heads)
        self.slot_pred = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))

    def encode_context(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        frames = batch["frames"]
        bsz, steps = frames.shape[:2]
        flat = frames.reshape(bsz * steps, *frames.shape[2:])
        tokens = self.frame(flat).reshape(bsz, steps, -1)
        tokens = tokens + self.action(batch["actions"]) + self.time(batch["times"])
        slots, _ = self.memory(tokens, batch["valid"])
        return F.normalize(slots, dim=-1), F.normalize(slots.mean(dim=1), dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        slots, memory = self.encode_context(batch)
        pred_slots = F.normalize(self.slot_pred(slots), dim=-1)
        patches = batch["target_patches"]
        bsz, patch_count = patches.shape[:2]
        with torch.no_grad():
            target = self.patch(patches.reshape(bsz * patch_count, *patches.shape[2:])).reshape(bsz, patch_count, -1)
            target = F.normalize(target, dim=-1)
        return {"memory": memory, "pred_slots": pred_slots, "target_set": target}


def set_logits(pred_slots: torch.Tensor, target_set: torch.Tensor, temperature: float) -> torch.Tensor:
    sim = torch.einsum("bsd,tpd->bstp", pred_slots, target_set) / float(temperature)
    assigned = torch.logsumexp(sim, dim=1) - np.log(float(pred_slots.shape[1]))
    return assigned.mean(dim=-1)


def set_nce(pred_slots: torch.Tensor, target_set: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = set_logits(pred_slots, target_set, temperature)
    labels = torch.arange(len(pred_slots), device=pred_slots.device)
    return F.cross_entropy(logits, labels)


def set_cosine(pred_slots: torch.Tensor, target_set: torch.Tensor) -> torch.Tensor:
    sim = torch.einsum("bsd,bpd->bsp", pred_slots, target_set.detach())
    best = sim.max(dim=1).values
    return (1.0 - best.mean(dim=-1)).mean()


@torch.no_grad()
def set_retrieval(pred_slots: torch.Tensor, target_set: torch.Tensor, temperature: float) -> dict[str, float]:
    logits = set_logits(pred_slots, target_set, temperature)
    rank = torch.argsort(logits, dim=1, descending=True)
    truth = torch.arange(len(pred_slots), device=pred_slots.device).unsqueeze(1)
    top1 = (rank[:, :1] == truth).any(dim=1).float().mean().item()
    top5 = (rank[:, :5] == truth).any(dim=1).float().mean().item()
    eye = torch.eye(len(pred_slots), dtype=torch.bool, device=pred_slots.device)
    margin = (logits.diag() - logits.masked_fill(eye, -9).max(dim=1).values).mean().item()
    return {"top1": float(top1), "top5": float(top5), "margin": float(margin)}


def build_datasets(args):
    archive = base.cache_path(args)
    common = dict(age=args.age, seed=args.seed, validation_fraction=args.validation_fraction)
    return {
        "train_aug": RandomPatchSetDataset(
            archive,
            split="train",
            variant="full",
            augment=True,
            temporal_drop=args.temporal_drop,
            patch_drop=args.patch_drop,
            **common,
        ),
        "train_eval": RandomPatchSetDataset(archive, split="train", variant="full", augment=False, **common),
        "val_full": RandomPatchSetDataset(archive, split="val", variant="full", augment=False, **common),
        "val_reset": RandomPatchSetDataset(archive, split="val", variant="reset", augment=False, **common),
        "val_no_state": RandomPatchSetDataset(archive, split="val", variant="no_state", augment=False, **common),
    }


def run_epoch(model, loader: DataLoader, optimizer, device: torch.device, args) -> dict[str, float]:
    model.train(optimizer is not None)
    sums: dict[str, float] = {}
    count = 0
    pred_sets, target_sets = [], []
    for batch in loader:
        batch = base.move_batch(batch, device)
        out = model(batch)
        losses = {
            "nce": set_nce(out["pred_slots"], out["target_set"], args.temperature),
            "cos": set_cosine(out["pred_slots"], out["target_set"]),
            "std": base.std_loss(out["pred_slots"].flatten(0, 1)) + base.std_loss(out["memory"]),
        }
        loss = losses["nce"] + float(args.cos_weight) * losses["cos"] + float(args.std_weight) * losses["std"]
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        bsz = int(batch["label"].shape[0])
        count += bsz
        for name, value in {"loss": loss, **losses}.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach()) * bsz
        sums["pred_std"] = sums.get("pred_std", 0.0) + float(out["pred_slots"].detach().std(dim=(0, 1)).mean()) * bsz
        pred_sets.append(out["pred_slots"].detach())
        target_sets.append(out["target_set"].detach())
    metrics = {name: value / max(1, count) for name, value in sums.items()}
    if pred_sets:
        metrics.update({
            f"retrieval_{k}": v
            for k, v in set_retrieval(torch.cat(pred_sets), torch.cat(target_sets), args.temperature).items()
        })
    return metrics


@torch.no_grad()
def extract(model, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    memory, labels = [], []
    pred_sets, target_sets = [], []
    for batch in loader:
        batch = base.move_batch(batch, device)
        out = model(batch)
        memory.append(out["memory"].cpu().numpy())
        pred_sets.append(out["pred_slots"].cpu().numpy())
        target_sets.append(out["target_set"].cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
    return {
        "memory": np.concatenate(memory, axis=0),
        "pred_slots": np.concatenate(pred_sets, axis=0),
        "target_set": np.concatenate(target_sets, axis=0),
        "labels": np.concatenate(labels, axis=0).astype(np.int64),
    }


def train_cell(args) -> dict[str, Any]:
    if not base.cache_path(args).is_file():
        raise FileNotFoundError(base.cache_path(args))
    out_dir = base.result_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    base.set_seed(281_911 + int(args.seed) + 23 * int(args.age))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    datasets = build_datasets(args)
    train_loader = base.make_loader(datasets["train_aug"], args, shuffle=True)
    train_eval_loader = base.make_loader(datasets["train_eval"], args, shuffle=False)
    val_full_loader = base.make_loader(datasets["val_full"], args, shuffle=False)
    val_reset_loader = base.make_loader(datasets["val_reset"], args, shuffle=False)
    val_no_state_loader = base.make_loader(datasets["val_no_state"], args, shuffle=False)

    with np.load(base.cache_path(args), allow_pickle=False) as data:
        img_size = int(data["img_size"])
        action_dim = int(data["actions"].shape[-1])
    model = RandomPatchSetJEPA(
        img_size=img_size,
        action_dim=action_dim,
        dim=int(args.dim),
        slots=int(args.slots),
        heads=int(args.heads),
        chunk=int(getattr(args, "chunk", 0)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args)
        val_metrics = run_epoch(model, val_full_loader, None, device, args)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if epoch == 1 or epoch % max(1, int(args.epochs) // 4) == 0:
            print(
                base.stable_json(
                    {
                        "env": args.env_name,
                        "age": args.age,
                        "seed": args.seed,
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "val_top1": val_metrics["retrieval_top1"],
                    }
                ).strip(),
                flush=True,
            )

    train_features = extract(model, train_eval_loader, device)
    evals = {
        "full": extract(model, val_full_loader, device),
        "reset": extract(model, val_reset_loader, device),
        "no_state": extract(model, val_no_state_loader, device),
    }
    readout = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    readout.fit(train_features["memory"], train_features["labels"])
    readout_metrics = {
        name: base.readout_metric(readout, payload["memory"], payload["labels"])
        for name, payload in evals.items()
    }
    retrieval_metrics = {
        name: set_retrieval(
            torch.from_numpy(payload["pred_slots"]).to(device),
            torch.from_numpy(payload["target_set"]).to(device),
            args.temperature,
        )
        for name, payload in evals.items()
    }
    full = readout_metrics["full"]["balanced_accuracy"]
    reset = readout_metrics["reset"]["balanced_accuracy"]
    no_state = readout_metrics["no_state"]["balanced_accuracy"]
    result = {
        "schema": "random_patchset_jepa_ogbench_cell_v1",
        "status": "completed",
        "env_name": args.env_name,
        "age": int(args.age),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "dim": int(args.dim),
        "slots": int(args.slots),
        "heads": int(args.heads),
        "chunk": int(getattr(args, "chunk", 0)),
        "streaming": bool(int(getattr(args, "chunk", 0)) > 0),
        "cue_mode": str(getattr(args, "cue_mode", "color")),
        "target_patches": int(TARGET_PATCHES),
        "patch_size": int(PATCH_SIZE),
        "training_loss_uses_cue_labels": False,
        "posthoc_readout_uses_cue_labels": True,
        "manual_cue_feature_supplied": False,
        "manual_cue_crop_or_key_frame": False,
        "history": history,
        "retrieval": retrieval_metrics,
        "readout": readout_metrics,
        "gate": {
            "full_minimum": 0.75,
            "control_maximum": 0.35,
            "pass": bool(full >= 0.75 and reset <= 0.35 and no_state <= 0.35),
        },
        "claim_boundary": (
            "Random patch-set JEPA render-stage memory probe; saliency target mining is label-free, "
            "and cue labels are post-hoc evaluation only."
        ),
    }
    (out_dir / "result.json").write_text(base.stable_json(result))
    np.savez_compressed(
        out_dir / "features.npz",
        train_memory=train_features["memory"],
        train_labels=train_features["labels"],
        val_full_memory=evals["full"]["memory"],
        val_reset_memory=evals["reset"]["memory"],
        val_no_state_memory=evals["no_state"]["memory"],
        val_labels=evals["full"]["labels"],
    )
    base.plot_diagnostic(result, out_dir)
    torch.save({"model": model.state_dict(), "args": vars(args), "result": result}, out_dir / "model.pt")
    print(base.stable_json({"env": args.env_name, "age": args.age, "seed": args.seed, "full": full, "reset": reset, "no_state": no_state, "pass": result["gate"]["pass"]}), flush=True)
    return result


def main(argv=None) -> None:
    args = base.parse_args(argv)
    args.output = base.resolve_path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare_cache:
        print(base.stable_json(base.prepare_cache(args)), flush=True)
        return
    if args.aggregate:
        base.aggregate(args)
        return
    train_cell(args)


if __name__ == "__main__":
    main()
