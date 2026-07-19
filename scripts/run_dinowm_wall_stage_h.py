#!/usr/bin/env python3
"""Stage-H carrier grid for official DINO-WM Wall.

Prerequisite: ``scripts/run_dinowm_wall_stage_g.py`` must have admitted the
Wall counterfactual cue deck.  This runner then builds a compact DINO feature
cache and trains the parameter-matched spatial carriers against the frozen
one-step Wall predictor.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import make_frozen_carrier  # noqa: E402
from lewm.official_tasks.dinowm_spatial_carrier import (  # noqa: E402
    spatial_carrier_forward,
)
from scripts.run_dinowm_wall_stage_g import (  # noqa: E402
    AGES,
    CLASSES,
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA,
    DEFAULT_DINOV2,
    DEFAULT_HYDRA,
    DEFAULT_TORCH_HOME,
    DEFAULT_VENDOR,
    LAST_CUE_FRAME,
    LENGTH,
    action_blocks,
    classify_record,
    encode_frames,
    episode_frames,
    fit_predict,
    labels_for,
    load_dinov2,
    load_wall_arrays,
    normalized_proprio,
    pooled,
    render_variant,
    sha256_file,
    stable_json,
)


DEFAULT_OUTPUT = ROOT / "outputs/dinowm_wall_audit_v1/stage_h_carriers"
DEFAULT_ADMISSION = ROOT / "outputs/dinowm_wall_audit_v1/stage_g_admission"
ARMS = ("none", "gru", "lstm", "ssm", "fixed_trust")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hydra", type=Path, default=DEFAULT_HYDRA)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--dinov2", type=Path, default=DEFAULT_DINOV2)
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--run-cell", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--windows-per-batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--readout", choices=["ridge", "lr"], default="ridge")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in ("output", "admission", "data", "checkpoint", "hydra",
                 "vendor", "dinov2", "torch_home"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, ROOT / value)
    return args


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def read_admission(path: Path) -> dict[str, Any]:
    summary = json.loads((path / "admission_summary.json").read_text())
    if not summary.get("admitted"):
        raise RuntimeError("Wall admission has not passed")
    return summary


def read_selection(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path / "selection.npz", allow_pickle=False)
    return (np.asarray(payload["train"], dtype=np.int64),
            np.asarray(payload["validation"], dtype=np.int64))


class FrozenWallHost:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        sys.path.insert(0, str(args.vendor.resolve()))
        payload = torch.load(args.checkpoint, map_location="cpu",
                             weights_only=False)
        self.predictor = payload["predictor"].eval().to(device)
        self.action_encoder = payload["action_encoder"].eval().to(device)
        self.proprio_encoder = payload["proprio_encoder"].eval().to(device)
        self.epoch = int(payload.get("epoch", -1))
        self.device = device
        del payload
        moved = 0
        for module in self.predictor.modules():
            bias = getattr(module, "bias", None)
            if torch.is_tensor(bias) and bias.ndim == 4:
                module.bias = bias.to(device)
                moved += 1
        if tuple(self.predictor.pos_embedding.shape) != (1, 196, 404):
            raise RuntimeError("Wall predictor token shape changed")
        if tuple(self.action_encoder.patch_embed.weight.shape) != (10, 10, 1):
            raise RuntimeError("Wall action encoder shape changed")
        if tuple(self.proprio_encoder.patch_embed.weight.shape) != (10, 2, 1):
            raise RuntimeError("Wall proprio encoder shape changed")
        if moved != 6:
            raise RuntimeError(f"expected six attention masks, moved {moved}")
        for module in (self.predictor, self.action_encoder,
                       self.proprio_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def compose(self, visual: torch.Tensor, proprio: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        if visual.shape[1:] != (1, 196, 384):
            raise RuntimeError(f"bad visual shape {tuple(visual.shape)}")
        if proprio.shape != (visual.shape[0], 1, 2):
            raise RuntimeError(f"bad proprio shape {tuple(proprio.shape)}")
        if actions.shape != (visual.shape[0], 1, 10):
            raise RuntimeError(f"bad action shape {tuple(actions.shape)}")
        prop = self.proprio_encoder(proprio).unsqueeze(2).expand(
            -1, -1, 196, -1)
        action = self.action_encoder(actions).unsqueeze(2).expand(
            -1, -1, 196, -1)
        return torch.cat((visual, prop, action), dim=-1)

    def predict(self, visual: torch.Tensor, proprio: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        context = self.compose(visual, proprio, actions)
        return self.predictor(context.reshape(
            context.shape[0], 196, 404)).reshape(context.shape[0], 1, 196, 404)

    @torch.no_grad()
    def target_nonaction(self, visual: torch.Tensor,
                         proprio: torch.Tensor) -> torch.Tensor:
        prop = self.proprio_encoder(proprio).unsqueeze(2).expand(
            -1, -1, 196, -1)
        return torch.cat((visual, prop), dim=-1)


def prepare_cache(args: argparse.Namespace) -> None:
    read_admission(args.admission)
    train_ids, val_ids = read_selection(args.admission)
    cache = args.output / "cache"
    if cache.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {cache}")
    cache.mkdir(parents=True, exist_ok=True)
    device = device_from(args)
    arrays = load_wall_arrays(args.data)
    model = load_dinov2(args, device)
    indices = np.concatenate([train_ids, val_ids])
    split = np.concatenate([
        np.zeros(len(train_ids), dtype=np.uint8),
        np.ones(len(val_ids), dtype=np.uint8),
    ])
    base_path = cache / "base_visual.npy"
    cue_path = cache / "cue_visual.npy"
    base = np.lib.format.open_memmap(
        base_path, mode="w+", dtype=np.float16,
        shape=(len(indices), LENGTH, 196, 384))
    cue = np.lib.format.open_memmap(
        cue_path, mode="w+", dtype=np.float16,
        shape=(len(indices), CLASSES, LAST_CUE_FRAME, 196, 384))
    batch = 16
    for offset in range(0, len(indices), batch):
        stop = min(len(indices), offset + batch)
        frames = [episode_frames(args.data, int(ep))
                  for ep in indices[offset:stop]]
        base_frames = np.stack(frames).reshape(-1, 224, 224, 3)
        encoded = encode_frames(
            model, base_frames, device,
            args.frame_batch_size).reshape(stop - offset, LENGTH, 196, 384)
        base[offset:stop] = encoded.astype(np.float16)
        cue_frames = []
        for item in frames:
            variants = [render_variant(item, label)
                        for label in range(CLASSES)]
            cue_frames.append(np.stack([
                variant[1:LAST_CUE_FRAME + 1] for variant in variants]))
        cue_encoded = encode_frames(
            model, np.stack(cue_frames).reshape(-1, 224, 224, 3),
            device, args.frame_batch_size).reshape(
                stop - offset, CLASSES, LAST_CUE_FRAME, 196, 384)
        cue[offset:stop] = cue_encoded.astype(np.float16)
        print(f"[wall-cache] {stop}/{len(indices)}", flush=True)
    base.flush()
    cue.flush()
    actions = action_blocks(arrays["actions"], indices)
    proprio = normalized_proprio(arrays["states"], indices)
    np.savez_compressed(
        cache / "metadata.npz",
        split=split,
        episode_index=indices.astype(np.int64),
        actions=actions.astype(np.float32),
        proprio=proprio.astype(np.float32),
    )
    manifest = {
        "schema": "dinowm_wall_stage_h_cache_v1",
        "status": "complete",
        "base_count": int(len(indices)),
        "train_bases": int(len(train_ids)),
        "validation_bases": int(len(val_ids)),
        "expanded_train": int(len(train_ids) * CLASSES),
        "expanded_validation": int(len(val_ids) * CLASSES),
        "base_visual": {
            "path": str(base_path.relative_to(ROOT)),
            "sha256": sha256_file(base_path),
            "dtype": "float16",
            "shape": [int(x) for x in base.shape],
        },
        "cue_visual": {
            "path": str(cue_path.relative_to(ROOT)),
            "sha256": sha256_file(cue_path),
            "dtype": "float16",
            "shape": [int(x) for x in cue.shape],
        },
        "metadata": {
            "path": str((cache / "metadata.npz").relative_to(ROOT)),
            "sha256": sha256_file(cache / "metadata.npz"),
        },
    }
    (cache / "manifest.json").write_text(stable_json(manifest))
    print(stable_json({"status": "cache_complete",
                       "manifest": str((cache / "manifest.json").relative_to(ROOT))}))


class WallFeatureBank:
    def __init__(self, root: Path) -> None:
        cache = root / "cache"
        self.base = np.load(cache / "base_visual.npy", mmap_mode="r")
        self.cue = np.load(cache / "cue_visual.npy", mmap_mode="r")
        meta = np.load(cache / "metadata.npz", allow_pickle=False)
        self.split = np.asarray(meta["split"], dtype=np.uint8)
        self.actions = np.asarray(meta["actions"], dtype=np.float32)
        self.proprio = np.asarray(meta["proprio"], dtype=np.float32)
        self.train_bases = np.flatnonzero(self.split == 0)
        self.val_bases = np.flatnonzero(self.split == 1)

    def expanded_indices(self, split: str) -> np.ndarray:
        bases = self.train_bases if split == "train" else self.val_bases
        return (bases[:, None] * CLASSES
                + np.arange(CLASSES, dtype=np.int64)[None]).reshape(-1)

    @staticmethod
    def decode(expanded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        expanded = np.asarray(expanded, dtype=np.int64)
        return expanded // CLASSES, expanded % CLASSES

    def labels(self, split: str) -> np.ndarray:
        return labels_for(len(self.train_bases if split == "train"
                              else self.val_bases))

    def visual(self, expanded: np.ndarray) -> np.ndarray:
        bases, labels = self.decode(expanded)
        visual = np.asarray(self.base[bases], dtype=np.float32).copy()
        visual[:, 1:LAST_CUE_FRAME + 1] = np.asarray(
            self.cue[bases, labels], dtype=np.float32)
        return visual


def shifted_objective(host: FrozenWallHost, carrier: torch.nn.Module,
                      visual: torch.Tensor, actions: torch.Tensor,
                      proprio: torch.Tensor, starts: Iterable[int]) \
        -> tuple[torch.Tensor, torch.Tensor]:
    output = spatial_carrier_forward(carrier, visual, actions)
    predictions, targets = [], []
    for start in starts:
        start = int(start)
        pred = host.predict(
            output.fused_visual[:, start:start + 1],
            proprio[:, start:start + 1],
            actions[:, start:start + 1])[:, :, :, :394]
        target = host.target_nonaction(
            visual[:, start + 1:start + 2],
            proprio[:, start + 1:start + 2])
        predictions.append(pred)
        targets.append(target)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    visual_loss = F.mse_loss(prediction[..., :384].float(),
                             target[..., :384].float())
    proprio_loss = F.mse_loss(prediction[..., 384:].float(),
                              target[..., 384:].float())
    return visual_loss + 0.25 * proprio_loss, visual_loss


def fit_readout(args: argparse.Namespace, train_x: np.ndarray,
                train_y: np.ndarray):
    if args.readout == "lr":
        model = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1500, random_state=0)
    else:
        model = RidgeClassifier(alpha=1.0)
    clf = make_pipeline(StandardScaler(), model)
    clf.fit(train_x, train_y)
    return clf


def train_cell(args: argparse.Namespace, host: FrozenWallHost,
               bank: WallFeatureBank, carrier: torch.nn.Module) \
        -> list[dict[str, Any]]:
    train = bank.expanded_indices("train")
    rng = np.random.default_rng(910000 + int(args.seed))
    optimizer = torch.optim.AdamW(
        carrier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    history = []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train))
        losses, visual_losses = [], []
        started = time.time()
        for offset in range(0, len(order), args.batch_size):
            expanded = train[order[offset:offset + args.batch_size]]
            bases, _ = bank.decode(expanded)
            visual = torch.from_numpy(bank.visual(expanded)).to(host.device)
            actions = torch.from_numpy(bank.actions[bases]).to(host.device)
            proprio = torch.from_numpy(bank.proprio[bases]).to(host.device)
            starts = rng.choice(
                np.arange(0, LENGTH - 1), size=args.windows_per_batch,
                replace=False)
            optimizer.zero_grad(set_to_none=True)
            loss, visual_loss = shifted_objective(
                host, carrier, visual, actions, proprio, starts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(carrier.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            visual_losses.append(float(visual_loss.detach().cpu()))
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "visual_loss": float(np.mean(visual_losses)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.time() - started),
        }
        history.append(row)
        print(f"[wall-cell] {args.arm}/s{args.seed} epoch "
              f"{epoch}/{args.epochs} loss={row['loss']:.6f}", flush=True)
    return history


@torch.no_grad()
def collect_features(host: FrozenWallHost, carrier: torch.nn.Module,
                     bank: WallFeatureBank, split: str,
                     batch_size: int) -> dict[int, dict[str, np.ndarray]]:
    carrier.eval()
    indices = bank.expanded_indices(split)
    rows: dict[int, dict[str, list[np.ndarray]]] = {
        age: {"full": [], "reset": [], "no_state": [], "prior": []}
        for age in AGES}
    for offset in range(0, len(indices), batch_size):
        expanded = indices[offset:offset + batch_size]
        bases, _ = bank.decode(expanded)
        visual = torch.from_numpy(bank.visual(expanded)).to(host.device)
        base_visual = torch.from_numpy(
            np.asarray(bank.base[bases], dtype=np.float32)).to(host.device)
        actions = torch.from_numpy(bank.actions[bases]).to(host.device)
        proprio = torch.from_numpy(bank.proprio[bases]).to(host.device)
        full = spatial_carrier_forward(carrier, visual, actions)
        for age in AGES:
            endpoint = LAST_CUE_FRAME + int(age)
            context = endpoint - 1
            full_prediction = host.predict(
                full.fused_visual[:, context:context + 1],
                proprio[:, context:context + 1],
                actions[:, context:context + 1])[:, -1, :, :384]
            reset = spatial_carrier_forward(
                carrier,
                base_visual[:, context:context + 1],
                actions[:, context:context])
            reset_prediction = host.predict(
                reset.fused_visual,
                proprio[:, context:context + 1],
                actions[:, context:context + 1])[:, -1, :, :384]
            no_prediction = host.predict(
                base_visual[:, context:context + 1],
                proprio[:, context:context + 1],
                actions[:, context:context + 1])[:, -1, :, :384]
            rows[age]["full"].append(pooled(
                full_prediction.float().cpu().numpy()))
            rows[age]["reset"].append(pooled(
                reset_prediction.float().cpu().numpy()))
            rows[age]["no_state"].append(pooled(
                no_prediction.float().cpu().numpy()))
            rows[age]["prior"].append(pooled(
                full.prior_visual[:, endpoint].float().cpu().numpy()))
    return {
        age: {name: np.concatenate(values)
              for name, values in record.items()}
        for age, record in rows.items()
    }


def evaluate_cell(host: FrozenWallHost, carrier: torch.nn.Module,
                  bank: WallFeatureBank, args: argparse.Namespace) \
        -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train = collect_features(host, carrier, bank, "train", args.batch_size)
    validation = collect_features(
        host, carrier, bank, "validation", args.batch_size)
    y_train = bank.labels("train")
    y_val = bank.labels("validation")
    metrics = {}
    arrays: dict[str, np.ndarray] = {"truth": y_val}
    for age in AGES:
        print(f"[wall-readout] {args.arm}/s{args.seed} "
              f"age{age} full readout={args.readout}", flush=True)
        full_clf = fit_readout(args, train[age]["full"], y_train)
        full_pred = full_clf.predict(validation[age]["full"]).astype(np.int64)
        reset_pred = full_clf.predict(
            validation[age]["reset"]).astype(np.int64)
        no_pred = full_clf.predict(
            validation[age]["no_state"]).astype(np.int64)
        print(f"[wall-readout] {args.arm}/s{args.seed} "
              f"age{age} prior readout={args.readout}", flush=True)
        prior_clf = fit_readout(args, train[age]["prior"], y_train)
        prior_pred = prior_clf.predict(
            validation[age]["prior"]).astype(np.int64)
        arrays[f"age_{age}_full"] = full_pred
        arrays[f"age_{age}_reset"] = reset_pred
        arrays[f"age_{age}_no_state"] = no_pred
        arrays[f"age_{age}_prior"] = prior_pred
        metrics[str(age)] = {
            "endpoint_frame": LAST_CUE_FRAME + age,
            "predictor_context": [LAST_CUE_FRAME + age - 1],
            "full": classify_record(full_pred, y_val),
            "reset_with_full_readout": classify_record(reset_pred, y_val),
            "no_state_with_full_readout": classify_record(no_pred, y_val),
            "prior": classify_record(prior_pred, y_val),
        }
    return metrics, arrays


def run_cell(args: argparse.Namespace) -> None:
    if args.arm is None:
        raise ValueError("--arm is required with --run-cell")
    set_seed(args.seed)
    output = args.output / "cells" / args.arm / f"s{args.seed}"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = device_from(args)
    host = FrozenWallHost(args, device)
    bank = WallFeatureBank(args.output)
    carrier = make_frozen_carrier(args.arm, 384, 10).to(device)
    if carrier.parameter_count():
        history = train_cell(args, host, bank, carrier)
    else:
        history = []
    metrics, arrays = evaluate_cell(host, carrier, bank, args)
    with (output / "history.csv").open("w", newline="") as stream:
        if history:
            writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    np.savez_compressed(output / "predictions.npz", **arrays)
    torch.save({"carrier_state_dict": carrier.state_dict(),
                "metrics": metrics}, output / "carrier.pt")
    manifest = {
        "schema": "dinowm_wall_stage_h_cell_v1",
        "status": "complete",
        "arm": args.arm,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "readout": args.readout,
        "wall_checkpoint_epoch": host.epoch,
        "parameters": int(carrier.parameter_count()),
        "metrics": metrics,
        "artifacts": {
            "history": "history.csv",
            "predictions": "predictions.npz",
            "carrier": {
                "path": "carrier.pt",
                "sha256": sha256_file(output / "carrier.pt"),
            },
        },
    }
    (output / "metrics.json").write_text(stable_json(manifest))
    print(stable_json({
        "status": "cell_complete",
        "arm": args.arm,
        "seed": int(args.seed),
        "metrics": str((output / "metrics.json").relative_to(ROOT)),
    }))


def aggregate(args: argparse.Namespace) -> None:
    summary: dict[str, Any] = {
        "schema": "dinowm_wall_stage_h_summary_v1",
        "status": "complete",
        "ages": list(AGES),
        "arms": {},
    }
    for arm in ARMS:
        arm_dir = args.output / "cells" / arm
        seeds = []
        for path in sorted(arm_dir.glob("s*/metrics.json")):
            payload = json.loads(path.read_text())
            seeds.append(payload)
        if not seeds:
            continue
        age_rows = {}
        for age in AGES:
            key = str(age)
            full = [s["metrics"][key]["full"]["balanced_accuracy"]
                    for s in seeds]
            reset = [s["metrics"][key]["reset_with_full_readout"]["balanced_accuracy"]
                     for s in seeds]
            none = [s["metrics"][key]["no_state_with_full_readout"]["balanced_accuracy"]
                    for s in seeds]
            prior = [s["metrics"][key]["prior"]["balanced_accuracy"]
                     for s in seeds]
            age_rows[key] = {
                "full_mean": float(np.mean(full)),
                "full_values": [float(x) for x in full],
                "reset_mean": float(np.mean(reset)),
                "reset_values": [float(x) for x in reset],
                "no_state_mean": float(np.mean(none)),
                "no_state_values": [float(x) for x in none],
                "prior_mean": float(np.mean(prior)),
                "prior_values": [float(x) for x in prior],
                "passes": bool(
                    np.mean(full) >= 0.75
                    and np.mean(reset) <= 0.30
                    and np.mean(none) <= 0.30),
            }
        summary["arms"][arm] = {
            "seeds": [int(s["seed"]) for s in seeds],
            "parameters": int(seeds[0]["parameters"]),
            "ages": age_rows,
        }
    (args.output / "summary.json").write_text(stable_json(summary))
    print(stable_json({
        "status": "aggregate_complete",
        "summary": str((args.output / "summary.json").relative_to(ROOT)),
    }))


def main() -> None:
    args = resolve_args(parse_args())
    if args.prepare_cache:
        prepare_cache(args)
    if args.run_cell:
        run_cell(args)
    if args.aggregate:
        aggregate(args)
    if not (args.prepare_cache or args.run_cell or args.aggregate):
        raise SystemExit("select --prepare-cache, --run-cell, or --aggregate")


if __name__ == "__main__":
    main()
