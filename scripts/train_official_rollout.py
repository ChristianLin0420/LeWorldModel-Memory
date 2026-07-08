#!/usr/bin/env python3
"""Fine-tune the released LeWM predictor for multi-step latent rollouts.

The official SIGReg Reacher encoder stays frozen and its cached latents are
the targets.  Only the released action encoder, predictor, and prediction
projection are optimized.  ``one_step`` uses the original final-token loss;
``overshoot_8`` autoregressively trains horizons 1..8.  Evaluation is label
free with respect to the memory cue and reports latent error, action
sensitivity, decoded endogenous pose error, and predicted-latent rank.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.official_lewm import (OFFICIAL_HISTORY,
                                        load_official_reacher_checkpoint)

TASKS = ("t1", "t3")
OBJECTIVES = {"one_step": 1, "overshoot_8": 8}
EVAL_HORIZONS = (1, 2, 4, 8, 16)
ANCHOR = 24
RIDGE_ALPHAS = np.logspace(-3, 3, 7)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path: Path) -> dict[str, np.ndarray | dict]:
    with np.load(path) as source:
        data = {key: source[key] for key in source.files}
    meta_raw = data.pop("meta_json", np.array("{}"))
    data["meta"] = json.loads(str(meta_raw))
    required = {"z", "actions", "xi", "endo_state"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{path} missing cache keys {sorted(missing)}")
    return data


def trainable_modules(model) -> list[torch.nn.Module]:
    modules = [model.action_encoder, model.predictor, model.pred_proj]
    for module in modules:
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.encoder.eval()
    model.projector.eval()
    return modules


def predict_last(model, latent: torch.Tensor,
                 actions: torch.Tensor) -> torch.Tensor:
    return model.predict(latent, actions)[:, -1]


def overshoot_loss(model, z: torch.Tensor, actions: torch.Tensor,
                   start: int, horizon: int) -> torch.Tensor:
    history = z[:, start:start + OFFICIAL_HISTORY]
    losses = []
    for step in range(horizon):
        action_window = actions[:, start + step:
                                start + step + OFFICIAL_HISTORY]
        prediction = predict_last(model, history, action_window)
        target = z[:, start + OFFICIAL_HISTORY + step]
        losses.append(F.mse_loss(prediction.float(), target.float()))
        history = torch.cat([history[:, 1:], prediction[:, None]], dim=1)
    return torch.stack(losses).mean()


def effective_rank(values: np.ndarray) -> float:
    centered = values.astype(np.float64) - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0)
    total = eigenvalues.sum()
    if total <= 1e-12:
        return 0.0
    probability = eigenvalues / total
    probability = probability[probability > 0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


@torch.no_grad()
def rollout(model, z: torch.Tensor, actions: torch.Tensor,
            max_horizon: int) -> torch.Tensor:
    start = ANCHOR - OFFICIAL_HISTORY + 1
    history = z[:, start:ANCHOR + 1]
    predictions = []
    for step in range(max_horizon):
        action_window = actions[:, start + step:ANCHOR + 1 + step]
        prediction = predict_last(model, history, action_window)
        predictions.append(prediction)
        history = torch.cat([history[:, 1:], prediction[:, None]], dim=1)
    return torch.stack(predictions, dim=1)


def fit_pose_probe(train: dict) -> object:
    z = np.asarray(train["z"], dtype=np.float32).reshape(-1, 192)
    pose = np.asarray(train["endo_state"], dtype=np.float32)[..., :2].reshape(-1, 2)
    if len(z) > 24_000:
        index = np.random.default_rng(91_107).choice(
            len(z), size=24_000, replace=False)
        z, pose = z[index], pose[index]
    probe = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    probe.fit(z, pose)
    return probe


def angular_error(prediction: np.ndarray, target: np.ndarray) -> float:
    difference = prediction - target
    wrapped = np.arctan2(np.sin(difference), np.cos(difference))
    return float(np.abs(wrapped).mean())


@torch.no_grad()
def evaluate(model, train: dict, val: dict, device: torch.device,
             seed: int) -> dict:
    model.eval()
    z = torch.from_numpy(np.asarray(val["z"], dtype=np.float32)).to(device)
    actions = torch.from_numpy(
        np.asarray(val["actions"], dtype=np.float32)).to(device)
    max_horizon = max(EVAL_HORIZONS)
    prediction = rollout(model, z, actions, max_horizon).float().cpu().numpy()

    permutation = np.random.default_rng(74_000 + seed).permutation(len(actions))
    shuffled = actions.clone()
    shuffled[:, ANCHOR:] = actions[permutation, ANCHOR:]
    shuffled_prediction = rollout(model, z, shuffled, max_horizon
                                  ).float().cpu().numpy()

    target = np.asarray(val["z"], dtype=np.float32)[:,
                        ANCHOR + 1:ANCHOR + 1 + max_horizon]
    copy_last = np.repeat(np.asarray(val["z"], dtype=np.float32)[:,
                                     ANCHOR:ANCHOR + 1], max_horizon, axis=1)
    pose_target = np.asarray(val["endo_state"], dtype=np.float32)[:,
                             ANCHOR + 1:ANCHOR + 1 + max_horizon, :2]
    pose_probe = fit_pose_probe(train)

    rows = {}
    for horizon in EVAL_HORIZONS:
        index = horizon - 1
        variance = float(np.var(target[:, index]))
        denominator = max(variance, 1e-8)
        mse = float(np.mean((prediction[:, index] - target[:, index]) ** 2))
        shuffled_mse = float(np.mean(
            (shuffled_prediction[:, index] - target[:, index]) ** 2))
        copy_mse = float(np.mean(
            (copy_last[:, index] - target[:, index]) ** 2))
        decoded_pose = pose_probe.predict(prediction[:, index])
        rows[str(horizon)] = {
            "normalized_latent_mse": mse / denominator,
            "copy_last_normalized_mse": copy_mse / denominator,
            "shuffled_action_normalized_mse": shuffled_mse / denominator,
            "true_action_advantage": (shuffled_mse - mse) / denominator,
            "pose_angular_mae": angular_error(
                decoded_pose, pose_target[:, index]),
            "predicted_effective_rank": effective_rank(prediction[:, index]),
            "target_effective_rank": effective_rank(target[:, index]),
        }
    gate_horizons = (1, 2, 4, 8)
    competent = all(
        rows[str(h)]["normalized_latent_mse"]
        < rows[str(h)]["copy_last_normalized_mse"]
        and rows[str(h)]["true_action_advantage"] > 0
        for h in gate_horizons
    )
    return {
        "anchor": ANCHOR,
        "horizons": rows,
        "rollout_competent_through_8": bool(competent),
        "gate": "better than copy-last and positive true-action advantage at 1,2,4,8",
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--objective", required=True,
                        choices=tuple(OBJECTIVES))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--cache-root",
                        default="outputs/paper_a_expansion/cache")
    parser.add_argument("--weights", default=(
        "outputs/paper_a_expansion/pretrained/lewm-reacher/weights.pt"))
    parser.add_argument("--output",
                        default="outputs/paper_a_expansion/rollout")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cache_root = Path(args.cache_root) / args.task
    train = load_cache(cache_root / "train.npz")
    val = load_cache(cache_root / "val.npz")
    train_z = np.asarray(train["z"], dtype=np.float32)
    train_actions = np.asarray(train["actions"], dtype=np.float32)
    if train_z.shape[:2] != (len(train_actions), train_actions.shape[1] + 1):
        raise ValueError("latent/action cache alignment failure")

    weights = Path(args.weights)
    model = load_official_reacher_checkpoint(weights, device)
    modules = trainable_modules(model)
    parameters = [parameter for module in modules
                  for parameter in module.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1))
    trainable = sum(parameter.numel() for parameter in parameters)
    horizon = OBJECTIVES[args.objective]
    output_dir = (Path(args.output) / args.task / args.objective
                  / f"s{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("history.csv", "metrics.json", "checkpoint.pt"):
        if (output_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite {output_dir / name}")

    rng = np.random.default_rng(88_000 + args.seed)
    history_rows = []
    with (output_dir / "history.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=("epoch", "loss", "lr"))
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(len(train_z))
            losses = []
            for offset in range(0, len(order), args.batch_size):
                index = order[offset:offset + args.batch_size]
                if len(index) < 4:
                    continue
                max_start = train_z.shape[1] - OFFICIAL_HISTORY - horizon
                start = int(rng.integers(0, max_start + 1))
                z = torch.from_numpy(train_z[index]).to(device)
                actions = torch.from_numpy(train_actions[index]).to(device)
                optimizer.zero_grad(set_to_none=True)
                amp = (torch.autocast("cuda", dtype=torch.bfloat16)
                       if device.type == "cuda"
                       else torch.autocast("cpu", enabled=False))
                with amp:
                    loss = overshoot_loss(model, z, actions, start, horizon)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            row = {"epoch": epoch, "loss": float(np.mean(losses)),
                   "lr": optimizer.param_groups[0]["lr"]}
            history_rows.append(row)
            writer.writerow(row)
            stream.flush()
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print(f"[rollout] {args.task}/{args.objective}/s{args.seed} "
                      f"e{epoch}/{args.epochs} loss={row['loss']:.5f}",
                      flush=True)

    evaluation = evaluate(model, train, val, device, args.seed)
    metrics = {
        "schema_version": 1,
        "study": "official-lewm-learned-rollout",
        "task": args.task,
        "objective": args.objective,
        "overshoot_horizon": horizon,
        "seed": args.seed,
        "official_weights": str(weights),
        "official_weights_sha256": _sha256(weights),
        "frozen_encoder": True,
        "trainable_components": ["action_encoder", "predictor", "pred_proj"],
        "trainable_parameters": trainable,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "final_train_loss": history_rows[-1]["loss"],
        **evaluation,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    torch.save({
        "predictor": model.predictor.state_dict(),
        "action_encoder": model.action_encoder.state_dict(),
        "pred_proj": model.pred_proj.state_dict(),
        "config": metrics,
    }, output_dir / "checkpoint.pt")
    print(f"[rollout] wrote {output_dir}; competent="
          f"{metrics['rollout_competent_through_8']}", flush=True)


if __name__ == "__main__":
    main()
