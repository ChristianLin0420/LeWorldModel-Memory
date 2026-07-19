#!/usr/bin/env python3
"""Drop-in memory-architecture baselines for temporal patch-set JEPA.

This runner keeps the same OGBench cache, temporal-coverage patch mining,
fixed color/texture target encoder, loss, and post-hoc readout used by the
current slot-memory method.  Only the memory block changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_masked_evidence_jepa_ogbench as base  # noqa: E402
from scripts import run_multiview_patchset_color_jepa_ogbench as multiview  # noqa: E402
from scripts import run_random_patchset_jepa_ogbench as patchset  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs" / "memory_arch_baselines_v1"
DEFAULT_CACHE_ROOT = ROOT / "outputs" / "multiview_patchset_color_jepa_native_v1"
BASELINES = ("slot", "gru", "lstm", "mamba_lite")


_ORIGINAL_CACHE_PATH = base.cache_path


def patched_cache_path(args: argparse.Namespace) -> Path:
    cache_root = getattr(args, "cache_root", None)
    if cache_root is None:
        return _ORIGINAL_CACHE_PATH(args)
    root = Path(cache_root)
    if not root.is_absolute():
        root = ROOT / root
    return root / "cache" / base.env_key(args.env_name) / "render_cache.npz"


base.cache_path = patched_cache_path


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


class LastStateSlots(nn.Module):
    def __init__(self, dim: int, slots: int) -> None:
        super().__init__()
        self.slots = int(slots)
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, self.slots * dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        bsz, dim = state.shape
        return self.proj(state).reshape(bsz, self.slots, dim)


def last_valid(outputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    lengths = valid.float().sum(dim=1).clamp_min(1).long() - 1
    batch = torch.arange(outputs.shape[0], device=outputs.device)
    return outputs[batch, lengths]


class GRUSlotMemory(nn.Module):
    def __init__(self, dim: int, slots: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.to_slots = LastStateSlots(dim, slots)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(tokens * valid.unsqueeze(-1))
        return self.to_slots(last_valid(outputs, valid))


class LSTMSlotMemory(nn.Module):
    def __init__(self, dim: int, slots: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(dim, dim, batch_first=True)
        self.to_slots = LastStateSlots(dim, slots)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(tokens * valid.unsqueeze(-1))
        return self.to_slots(last_valid(outputs, valid))


class MambaLiteSlotMemory(nn.Module):
    """Small selective-SSM baseline implemented in plain PyTorch."""

    def __init__(self, dim: int, slots: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, 3 * dim)
        self.norm = nn.LayerNorm(dim)
        self.to_slots = LastStateSlots(dim, slots)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        bsz, steps, dim = tokens.shape
        state = tokens.new_zeros(bsz, dim)
        for step in range(steps):
            gate_raw, decay_raw, value_raw = self.in_proj(tokens[:, step]).chunk(3, dim=-1)
            update_gate = torch.sigmoid(gate_raw)
            keep = torch.sigmoid(decay_raw)
            value = torch.tanh(value_raw)
            candidate = keep * state + (1.0 - keep) * update_gate * value
            mask = valid[:, step].unsqueeze(-1)
            state = mask * candidate + (1.0 - mask) * state
        return self.to_slots(self.norm(state))


class BaselinePatchSetJEPA(nn.Module):
    def __init__(
        self,
        *,
        img_size: int,
        action_dim: int,
        dim: int,
        slots: int,
        heads: int,
        baseline: str,
    ) -> None:
        super().__init__()
        self.baseline = str(baseline)
        self.frame = base.FrameEncoder(dim, img_size)
        self.patch = multiview.FixedPatchTargetEncoder(dim)
        self.action = nn.Sequential(nn.Linear(action_dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        if baseline == "slot":
            self.memory = base.SlotMemory(dim, slots, heads)
        elif baseline == "gru":
            self.memory = GRUSlotMemory(dim, slots)
        elif baseline == "lstm":
            self.memory = LSTMSlotMemory(dim, slots)
        elif baseline == "mamba_lite":
            self.memory = MambaLiteSlotMemory(dim, slots)
        else:
            raise ValueError(f"unknown baseline {baseline!r}; choices={BASELINES}")
        self.slot_pred = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim))

    def encode_context(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        frames = batch["frames"]
        bsz, steps = frames.shape[:2]
        flat = frames.reshape(bsz * steps, *frames.shape[2:])
        tokens = self.frame(flat).reshape(bsz, steps, -1)
        tokens = tokens + self.action(batch["actions"]) + self.time(batch["times"])
        if self.baseline == "slot":
            slots, _ = self.memory(tokens, batch["valid"])
        else:
            slots = self.memory(tokens, batch["valid"])
        slots = F.normalize(slots, dim=-1)
        return slots, F.normalize(slots.mean(dim=1), dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        slots, memory = self.encode_context(batch)
        pred_slots = F.normalize(self.slot_pred(slots), dim=-1)
        patches = batch["target_patches"]
        bsz, patch_count = patches.shape[:2]
        with torch.no_grad():
            target = self.patch(patches.reshape(bsz * patch_count, *patches.shape[2:])).reshape(bsz, patch_count, -1)
            target = F.normalize(target, dim=-1)
        return {"memory": memory, "pred_slots": pred_slots, "target_set": target}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--env-name", default="pointmaze-large-navigate-v0")
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--baseline", choices=BASELINES, default="gru")
    parser.add_argument("--episodes", type=int, default=384)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--age", type=int, choices=base.AGES, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--dim", type=int, default=160)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--cos-weight", type=float, default=0.35)
    parser.add_argument("--std-weight", type=float, default=0.05)
    parser.add_argument("--temporal-drop", type=float, default=0.12)
    parser.add_argument("--patch-drop", type=float, default=0.20)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def make_loader(dataset: Any, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def run_epoch(
    model: BaselinePatchSetJEPA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train(optimizer is not None)
    sums: dict[str, float] = {}
    count = 0
    pred_sets, target_sets = [], []
    for batch in loader:
        batch = base.move_batch(batch, device)
        out = model(batch)
        losses = {
            "nce": patchset.set_nce(out["pred_slots"], out["target_set"], args.temperature),
            "cos": patchset.set_cosine(out["pred_slots"], out["target_set"]),
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
        pred_sets.append(out["pred_slots"].detach())
        target_sets.append(out["target_set"].detach())
    metrics = {name: value / max(1, count) for name, value in sums.items()}
    if pred_sets:
        metrics.update({
            f"retrieval_{k}": v
            for k, v in patchset.set_retrieval(
                torch.cat(pred_sets), torch.cat(target_sets), args.temperature
            ).items()
        })
    return metrics


@torch.no_grad()
def extract(model: BaselinePatchSetJEPA, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    memory, labels, pred_sets, target_sets = [], [], [], []
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


def result_dir(args: argparse.Namespace) -> Path:
    return Path(args.output) / base.env_key(args.env_name) / f"age_{int(args.age)}" / f"s{int(args.seed)}"


def train_cell(args: argparse.Namespace) -> dict[str, Any]:
    if not base.cache_path(args).is_file():
        raise FileNotFoundError(base.cache_path(args))
    out_dir = result_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    base.set_seed(511_031 + int(args.seed) + 29 * int(args.age) + 997 * BASELINES.index(args.baseline))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    datasets = multiview.build_datasets(args)
    train_loader = make_loader(datasets["train_aug"], args, shuffle=True)
    train_eval_loader = make_loader(datasets["train_eval"], args, shuffle=False)
    val_full_loader = make_loader(datasets["val_full"], args, shuffle=False)
    val_reset_loader = make_loader(datasets["val_reset"], args, shuffle=False)
    val_no_state_loader = make_loader(datasets["val_no_state"], args, shuffle=False)

    with np.load(base.cache_path(args), allow_pickle=False) as data:
        img_size = int(data["img_size"])
        action_dim = int(data["actions"].shape[-1])
    model = BaselinePatchSetJEPA(
        img_size=img_size,
        action_dim=action_dim,
        dim=int(args.dim),
        slots=int(args.slots),
        heads=int(args.heads),
        baseline=str(args.baseline),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args)
        val_metrics = run_epoch(model, val_full_loader, None, device, args)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if epoch == 1 or epoch % max(1, int(args.epochs) // 4) == 0:
            print(stable_json({
                "baseline": args.baseline,
                "env": args.env_name,
                "age": args.age,
                "seed": args.seed,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_top1": val_metrics["retrieval_top1"],
            }).strip(), flush=True)

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
        name: patchset.set_retrieval(
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
        "schema": "multiview_patchset_memory_baseline_cell_v1",
        "status": "completed",
        "baseline": str(args.baseline),
        "env_name": args.env_name,
        "age": int(args.age),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "dim": int(args.dim),
        "slots": int(args.slots),
        "heads": int(args.heads),
        "target_views": int(multiview.TARGET_VIEWS),
        "target_patches": int(multiview.TARGET_VIEWS * patchset.TARGET_PATCHES),
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
            "Architecture baseline for temporal-coverage patch-set JEPA; same data, targets, "
            "loss, and post-hoc readout as the slot-memory method."
        ),
    }
    (out_dir / "result.json").write_text(stable_json(result))
    np.savez_compressed(
        out_dir / "features.npz",
        train_memory=train_features["memory"],
        train_labels=train_features["labels"],
        val_full_memory=evals["full"]["memory"],
        val_reset_memory=evals["reset"]["memory"],
        val_no_state_memory=evals["no_state"]["memory"],
        val_labels=evals["full"]["labels"],
    )
    torch.save({"model": model.state_dict(), "args": vars(args), "result": result}, out_dir / "model.pt")
    print(stable_json({
        "baseline": args.baseline,
        "env": args.env_name,
        "age": args.age,
        "seed": args.seed,
        "full": full,
        "reset": reset,
        "no_state": no_state,
        "pass": result["gate"]["pass"],
    }), flush=True)
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    rows = [json.loads(path.read_text()) for path in sorted(Path(args.output).glob("*/*/s*/result.json"))]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["env_name"], int(row["age"])), []).append(row)
    summary_rows = []
    for (env_name, age), values in sorted(grouped.items()):
        full = [v["readout"]["full"]["balanced_accuracy"] for v in values]
        reset = [v["readout"]["reset"]["balanced_accuracy"] for v in values]
        no_state = [v["readout"]["no_state"]["balanced_accuracy"] for v in values]
        top1 = [v["retrieval"]["full"]["top1"] for v in values]
        summary_rows.append({
            "baseline": str(args.baseline),
            "env_name": env_name,
            "age": int(age),
            "seeds": [int(v["seed"]) for v in values],
            "seed_count": int(len(values)),
            "pass_count": int(sum(bool(v["gate"]["pass"]) for v in values)),
            "all_pass": bool(all(bool(v["gate"]["pass"]) for v in values)),
            "full_bacc_mean": float(np.mean(full)),
            "reset_bacc_mean": float(np.mean(reset)),
            "no_state_bacc_mean": float(np.mean(no_state)),
            "retrieval_top1_mean": float(np.mean(top1)),
        })
    summary = {
        "schema": "multiview_patchset_memory_baseline_summary_v1",
        "status": "completed" if rows else "empty",
        "baseline": str(args.baseline),
        "cell_count": int(len(rows)),
        "rows": summary_rows,
        "claim_boundary": (
            "Architecture baseline summary; labels are used only for post-hoc evaluation."
        ),
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "summary.json").write_text(stable_json(summary))
    print(stable_json(summary), flush=True)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.cache_root = args.cache_root if args.cache_root.is_absolute() else ROOT / args.cache_root
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare_cache:
        print(stable_json(base.prepare_cache(args)), flush=True)
        return
    if args.aggregate:
        aggregate(args)
        return
    train_cell(args)


if __name__ == "__main__":
    main()
