#!/usr/bin/env python3
"""Train and evaluate the joint-gradient ORBIT V10-J prototype.

Every design shares an affine-free per-frame encoder and an equal-weight
prediction/variance/off-diagonal-covariance objective. The deterministic clean online
embedding is both an active prediction target and the diversity view. SIGReg is logged
only as a diagnostic. Cross-model comparison uses a train-only physics-state ridge
probe in the same online coordinate, never private latent-coordinate MSE across models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.memory_model import MemoryLeWorldModel
from scripts.hacssm_v10_data import (
    DEFAULT_CORRUPTION_SEED,
    V10TrajectoryDataset,
    load_cache,
    sha256_file,
)


DESIGNS = (
    "none",
    "gru",
    "ssm",
    "hacssmv8",
    "orbitv10",
    "orbitv10_noaction",
    "orbitv10_additive",
    "orbitv10_scaled",
    "orbitv10_static",
)
HELDOUT_CONDITIONS = (
    "freeze",
    "gaussian_noise",
    "checkerboard",
    "long_freeze",
)
ROLLOUT_SCHEMA_VERSION = 1


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _finite_memory_log(model: MemoryLeWorldModel) -> dict[str, float]:
    result = {}
    for key, raw in model.horizons().items():
        if isinstance(raw, bool):
            result[key] = float(raw)
        elif isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            result[key] = float(raw)
    return result


def _model_impl(design: str) -> tuple[str, str]:
    if design == "none":
        return "ema", "none"
    return design, "both"


def _design_metadata(design: str) -> dict[str, Any]:
    if design not in DESIGNS:
        raise ValueError(f"unknown V10 design {design!r}")
    is_orbit = design.startswith("orbitv10")
    variant = design.removeprefix("orbitv10_") if design != "orbitv10" else "orthogonal"
    return {
        "memory_arch_schema_version": 10 if is_orbit else None,
        "memory_architecture": "orthogonal_recurrent_belief" if is_orbit else design,
        "memory_v10_variant": variant if is_orbit else None,
        "memory_internal_auxiliary": "none",
        "memory_teacher_present": False,
        "memory_fixed_horizon": False if is_orbit else None,
        "encoder_trained_end_to_end": True,
        "encoder_ema_teacher_present": False,
        "target_stop_gradient": False,
        "training_objective": "v10j_joint_pred_variance_covariance_equal_weight",
    }


def _matched_gru_hidden(embed_dim: int) -> int:
    """Closest GRU hidden width to the ~2D^2 V8/ORBIT memory budget."""
    target = 2 * embed_dim * embed_dim + 16 * embed_dim
    candidates = range(1, embed_dim + 1)
    return min(candidates, key=lambda hidden: abs(
        (4 * embed_dim * hidden + 3 * hidden * hidden + 6 * hidden) - target))


def build_model(args: argparse.Namespace, action_dim: int) -> MemoryLeWorldModel:
    memory_impl, memory_mode = _model_impl(args.memory_mode)
    model = MemoryLeWorldModel(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        action_dim=action_dim,
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
        predictor_layers=args.predictor_layers,
        predictor_heads=args.predictor_heads,
        predictor_norm="none",
        encoder_norm="causal",
        history_len=args.history_len,
        dropout=args.dropout,
        sigreg_lambda=args.sigreg_lambda,
        sigreg_projections=args.sigreg_projections,
        memory_impl=memory_impl,
        memory_mode=memory_mode,
        gru_hidden=_matched_gru_hidden(args.embed_dim),
        hier_loss_weight=0.0,
        encoder_type="vit",
    )
    if getattr(model, "encoder_norm", None) != "causal":
        raise RuntimeError("V10-J requires encoder_norm='causal'")
    if model.predictor_norm != "none":
        raise RuntimeError("V10-J requires predictor_norm='none'")
    return model


def encode_frames(encoder: torch.nn.Module, observations: torch.Tensor) -> torch.Tensor:
    if observations.dim() == 5:
        batch, length, channels, height, width = observations.shape
        encoded = encoder(observations.reshape(batch * length, channels, height, width))
        return encoded.reshape(batch, length, -1)
    return encoder(observations)


def encode_joint_clean_deterministic(
        model: MemoryLeWorldModel, clean: torch.Tensor) -> torch.Tensor:
    """Encode the joint target/diversity view without dropout while retaining gradients."""
    was_training = model.encoder.training
    model.encoder.eval()
    try:
        return model.encode(clean)
    finally:
        model.encoder.train(was_training)


def _loader(dataset: V10TrajectoryDataset, args: argparse.Namespace, *, train: bool,
            batch_size: int | None = None) -> DataLoader:
    generator = torch.Generator().manual_seed(100_000 + args.seed) if train else None
    return DataLoader(
        dataset,
        batch_size=batch_size or args.batch_size,
        shuffle=train,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        persistent_workers=args.num_workers > 0,
    )


def run_epoch(model: MemoryLeWorldModel, loader: DataLoader,
              optimizer: torch.optim.Optimizer | None, device: torch.device,
              use_amp: bool) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {
        "loss": 0.0,
        "pred_loss": 0.0,
        "variance_loss": 0.0,
        "covariance_loss": 0.0,
        "sigreg_loss": 0.0,
    }
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            observed = batch["observed"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp else torch.autocast("cpu", enabled=False)
            )
            with amp_context:
                joint_clean = encode_joint_clean_deterministic(model, clean)
                losses = model.compute_loss(
                    observed,
                    actions,
                    target_embeddings=joint_clean,
                    diversity_embeddings=joint_clean,
                    objective="v10j",
                    detach_target_embeddings=False,
                )
            if train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = observed.shape[0]
            for key in ("loss", "pred_loss", "variance_loss", "covariance_loss",
                        "sigreg_loss"):
                totals[key] += float(losses[key].detach()) * batch_size
            count += batch_size
    if count == 0:
        raise RuntimeError("empty V10 epoch")
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def _encode_clean(model: MemoryLeWorldModel, dataset: V10TrajectoryDataset,
                  args: argparse.Namespace, device: torch.device,
                  use_amp: bool, encoder: torch.nn.Module | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    selected_encoder = model.encoder if encoder is None else encoder
    selected_encoder.eval()
    latent_chunks, state_chunks = [], []
    for batch in _loader(dataset, args, train=False):
        clean = batch["clean"].to(device, non_blocking=True)
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp else torch.autocast("cpu", enabled=False)
        )
        with amp_context:
            latent = encode_frames(selected_encoder, clean)
        latent_chunks.append(latent[:, args.history_len:].float().cpu().numpy())
        state_chunks.append(
            batch["physics_state"][:, args.history_len:].float().numpy())
    return (
        np.concatenate(latent_chunks).reshape(-1, args.embed_dim),
        np.concatenate(state_chunks).reshape(-1, dataset.metadata.state_dim),
    )


def fit_state_probe(model: MemoryLeWorldModel, train_clean: V10TrajectoryDataset,
                    args: argparse.Namespace, device: torch.device,
                    use_amp: bool, encoder: torch.nn.Module | None = None
                    ) -> dict[str, np.ndarray]:
    features, states = _encode_clean(
        model, train_clean, args, device, use_amp, encoder=encoder)
    x_mean = features.mean(axis=0, dtype=np.float64)
    x_std = features.std(axis=0, dtype=np.float64).clip(min=1e-6)
    y_mean = states.mean(axis=0, dtype=np.float64)
    y_std = states.std(axis=0, dtype=np.float64).clip(min=1e-6)
    x = (features.astype(np.float64) - x_mean) / x_std
    y = (states.astype(np.float64) - y_mean) / y_std
    design = np.concatenate((x, np.ones((len(x), 1), dtype=np.float64)), axis=1)
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * args.probe_ridge
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(gram + penalty, design.T @ y)
    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
        "weights": weights.astype(np.float32),
    }


def _probe_predict(latent: torch.Tensor, probe: dict[str, np.ndarray]) -> torch.Tensor:
    device = latent.device
    dtype = latent.dtype
    x_mean = torch.as_tensor(probe["x_mean"], device=device, dtype=dtype)
    x_std = torch.as_tensor(probe["x_std"], device=device, dtype=dtype)
    weights = torch.as_tensor(probe["weights"], device=device, dtype=dtype)
    standardized = (latent - x_mean) / x_std
    design = torch.cat((standardized, torch.ones_like(standardized[..., :1])), dim=-1)
    prediction_norm = design @ weights
    y_mean = torch.as_tensor(probe["y_mean"], device=device, dtype=dtype)
    y_std = torch.as_tensor(probe["y_std"], device=device, dtype=dtype)
    return prediction_norm * y_std + y_mean


def _prediction_latents(model: MemoryLeWorldModel, observed: torch.Tensor,
                        actions: torch.Tensor, history_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    z = model.encode(observed)
    z_full = model._inject(z, actions=actions)
    batch, length, dimension = z.shape
    windows = length - history_len
    full_windows = z_full.unfold(1, history_len, 1)[:, :windows]
    full_windows = full_windows.permute(0, 1, 3, 2).reshape(
        batch * windows, history_len, dimension)
    action_windows = actions.unfold(1, history_len, 1)[:, :windows]
    action_windows = action_windows.permute(0, 1, 3, 2).reshape(
        batch * windows, history_len, actions.shape[-1])
    prediction = model.predictor(full_windows, action_windows)[:, -1]
    return prediction.reshape(batch, windows, dimension), z


def _phase_masks(batch: dict[str, Any], length: int, history_len: int,
                 device: torch.device) -> dict[str, torch.Tensor]:
    times = torch.arange(history_len, length, device=device).view(1, -1)
    start = batch["gap_start"].to(device).view(-1, 1)
    end = batch["gap_end"].to(device).view(-1, 1)
    return {
        "gap": (times >= start) & (times < end),
        "deep": (times >= start + history_len) & (times < end),
        "first_post": times == end,
        "post": (times > end) & (times <= end + history_len),
    }


def _r2(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = float(np.square(prediction - target).sum(dtype=np.float64))
    centered = target - target.mean(axis=0, keepdims=True)
    total = float(np.square(centered).sum(dtype=np.float64))
    return 1.0 - residual / max(total, 1e-12)


@torch.no_grad()
def evaluate_condition(model: MemoryLeWorldModel, dataset: V10TrajectoryDataset,
                       probe: dict[str, np.ndarray], args: argparse.Namespace,
                       device: torch.device, use_amp: bool,
                       rollout_episode: int | None = None
                       ) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    model.eval()
    phase_errors: dict[str, list[np.ndarray]] = {
        "gap": [], "deep": [], "first_post": [], "post": [], "primary": []}
    primary_predictions, primary_targets = [], []
    rollout = None
    loader = _loader(dataset, args, train=False)
    for batch in loader:
        observed = batch["observed"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        state = batch["physics_state"].to(device, non_blocking=True)
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp else torch.autocast("cpu", enabled=False)
        )
        with amp_context:
            latent_prediction, _z = _prediction_latents(
                model, observed, actions, args.history_len)
            state_prediction = _probe_predict(latent_prediction.float(), probe)
        target = state[:, args.history_len:].float()
        y_std = torch.as_tensor(probe["y_std"], device=device, dtype=torch.float32)
        per_step = ((state_prediction.float() - target) / y_std).square().mean(dim=-1)
        masks = _phase_masks(batch, dataset.metadata.length, args.history_len, device)
        masks["primary"] = masks["deep"] | masks["first_post"]
        for name, mask in masks.items():
            if bool(mask.any()):
                phase_errors[name].append(per_step[mask].cpu().numpy())
        primary = masks["primary"]
        primary_predictions.append(state_prediction[primary].float().cpu().numpy())
        primary_targets.append(target[primary].float().cpu().numpy())

        if rollout_episode is not None:
            episode_indices = batch["episode_index"].numpy()
            matches = np.nonzero(episode_indices == rollout_episode)[0]
            if len(matches):
                row = int(matches[0])
                times = np.arange(args.history_len, dataset.metadata.length, dtype=np.int64)
                start = int(batch["gap_start"][row])
                end = int(batch["gap_end"][row])
                phases = np.asarray([
                    "deep" if start + args.history_len <= t < end else
                    "gap" if start <= t < end else
                    "first_post" if t == end else
                    "post" if end < t <= end + args.history_len else "context"
                    for t in times
                ])
                rollout = {
                    "target_times": times,
                    "phase": phases,
                    "gap_start": np.asarray(start, dtype=np.int64),
                    "gap_end": np.asarray(end, dtype=np.int64),
                    "observed_rgb": np.rint(
                        batch["observed"][row].numpy().transpose(0, 2, 3, 1) * 255
                    ).clip(0, 255).astype(np.uint8),
                    "clean_rgb": np.rint(
                        batch["clean"][row].numpy().transpose(0, 2, 3, 1) * 255
                    ).clip(0, 255).astype(np.uint8),
                    "actions": batch["actions"][row].numpy().astype(np.float32),
                    "physics_state_target": target[row].cpu().numpy().astype(np.float32),
                    "physics_state_prediction": state_prediction[row].cpu().numpy().astype(np.float32),
                    "state_nmse_by_target_t": per_step[row].cpu().numpy().astype(np.float32),
                }
    metrics = {}
    for name, chunks in phase_errors.items():
        if not chunks:
            raise RuntimeError(f"no {name} samples for {dataset.view}")
        metrics[name] = float(np.concatenate(chunks).mean(dtype=np.float64))
    prediction = np.concatenate(primary_predictions)
    target = np.concatenate(primary_targets)
    metrics["r2"] = _r2(prediction, target)
    return metrics, rollout


@torch.no_grad()
def encoder_diagnostics(model: MemoryLeWorldModel, clean_dataset: V10TrajectoryDataset,
                        args: argparse.Namespace, device: torch.device,
                        encoder: torch.nn.Module | None = None) -> dict[str, float]:
    model.eval()
    selected_encoder = model.encoder if encoder is None else encoder
    selected_encoder.eval()
    sample0 = clean_dataset[0]["clean"]
    sample1 = clean_dataset[1]["clean"]
    frame = sample0[0].unsqueeze(0).to(device)
    singleton = selected_encoder(frame).float()
    paired = selected_encoder(
        torch.stack((sample0[0], sample1[-1])).to(device))[0:1].float()
    prefix = encode_frames(
        selected_encoder, sample0[:8].unsqueeze(0).to(device)).float()
    whole_prefix = encode_frames(
        selected_encoder, sample0.unsqueeze(0).to(device))[:, :8].float()

    loader = _loader(clean_dataset, args, train=False, batch_size=min(args.batch_size, 32))
    latents = []
    for batch_index, batch in enumerate(loader):
        latents.append(encode_frames(
            selected_encoder, batch["clean"].to(device)).float().cpu())
        if batch_index >= 1:
            break
    matrix = torch.cat(latents).reshape(-1, args.embed_dim).double()
    centered = matrix - matrix.mean(dim=0)
    variance = centered.square().mean(dim=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    effective_rank = torch.exp(-(
        probabilities * probabilities.clamp_min(1e-30).log()).sum())
    return {
        "encoder_mean_channel_variance": float(variance.mean()),
        "encoder_covariance_effective_rank": float(effective_rank),
        "encoder_singleton_max_abs": float((singleton - paired).abs().max()),
        "encoder_prefix_max_abs": float((prefix - whole_prefix).abs().max()),
    }


@torch.no_grad()
def orbit_diagnostics(model: MemoryLeWorldModel, dataset: V10TrajectoryDataset,
                      device: torch.device) -> dict[str, float]:
    if not model.memory_impl.startswith("orbitv10"):
        return {"orbit_orthogonality_error_max": 0.0, "orbit_streaming_max_abs": 0.0}
    model.eval()
    batch = dataset[0]
    observed = batch["observed"].unsqueeze(0).to(device)
    actions = batch["actions"].unsqueeze(0).to(device)
    z = model.encode(observed).float()
    memory = model.mem_orbitv10
    mixed, details = memory(z, actions, return_details=True)
    applicable = details["orthogonality_applicable"].bool()
    errors = details["orthogonality_error"]
    orthogonality_error = float(errors[applicable].max()) if bool(applicable.any()) else 0.0
    state = details["states"][:, 0]
    streamed = [mixed[:, 0]]
    for step in range(1, z.shape[1]):
        mixed_step, state = memory.step(state, z[:, step], actions[:, step - 1])
        streamed.append(mixed_step)
    streaming = torch.stack(streamed, dim=1)
    return {
        "orbit_orthogonality_error_max": orthogonality_error,
        "orbit_streaming_max_abs": float((streaming - mixed).abs().max()),
    }


@torch.no_grad()
def probe_ceiling(model: MemoryLeWorldModel, dataset: V10TrajectoryDataset,
                  probe: dict[str, np.ndarray], args: argparse.Namespace,
                  device: torch.device, use_amp: bool,
                  encoder: torch.nn.Module | None = None) -> dict[str, float]:
    features, states = _encode_clean(
        model, dataset, args, device, use_amp, encoder=encoder)
    prediction = _probe_predict(
        torch.from_numpy(features).to(device), probe).cpu().numpy()
    y_std = probe["y_std"]
    nmse = float(np.square((prediction - states) / y_std).mean(dtype=np.float64))
    return {"probe_ceiling_state_nmse": nmse, "probe_ceiling_r2": _r2(prediction, states)}


def _rollout_package(condition_rollouts: dict[str, dict[str, np.ndarray]],
                     output_path: Path, episode_index: int
                     ) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(ROLLOUT_SCHEMA_VERSION, dtype=np.int64),
        "episode_index": np.asarray(episode_index, dtype=np.int64),
        "conditions": np.asarray(HELDOUT_CONDITIONS),
    }
    video_rows = []
    flat_conditions, flat_times, flat_phases = [], [], []
    flat_targets, flat_predictions, flat_errors = [], [], []
    for condition in HELDOUT_CONDITIONS:
        rollout = condition_rollouts[condition]
        for key, value in rollout.items():
            arrays[f"{condition}_{key}"] = value
        observed = rollout["observed_rgb"]
        clean = rollout["clean_rgb"]
        separator = np.full((len(observed), observed.shape[1], 4, 3), 255, dtype=np.uint8)
        video_rows.append(np.concatenate((observed, separator, clean), axis=2))
        count = len(rollout["target_times"])
        flat_conditions.append(np.full(count, condition))
        flat_times.append(rollout["target_times"])
        flat_phases.append(rollout["phase"])
        flat_targets.append(rollout["physics_state_target"])
        flat_predictions.append(rollout["physics_state_prediction"])
        flat_errors.append(rollout["state_nmse_by_target_t"])
    arrays.update({
        "condition": np.concatenate(flat_conditions),
        "target_times": np.concatenate(flat_times),
        "phase": np.concatenate(flat_phases),
        "state_target": np.concatenate(flat_targets),
        "state_prediction": np.concatenate(flat_predictions),
        "state_nmse": np.concatenate(flat_errors),
    })
    video = np.concatenate(video_rows, axis=1).transpose(0, 3, 1, 2)
    np.savez_compressed(output_path, **arrays)
    return arrays, video


def _make_rollout_table(wandb: Any, arrays: dict[str, np.ndarray]) -> Any:
    rows = []
    for condition in HELDOUT_CONDITIONS:
        times = arrays[f"{condition}_target_times"]
        phases = arrays[f"{condition}_phase"]
        errors = arrays[f"{condition}_state_nmse_by_target_t"]
        for target_time, phase, error in zip(times, phases, errors):
            rows.append([condition, int(target_time), str(phase), float(error)])
    return wandb.Table(
        columns=["condition", "target_time", "phase", "normalized_state_mse"],
        data=rows,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--memory-mode", choices=DESIGNS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/hacssm_v10_r1_shared")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--predictor-layers", type=int, default=4)
    parser.add_argument("--predictor-heads", type=int, default=8)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sigreg-lambda", type=float, default=0.1)
    parser.add_argument("--sigreg-projections", type=int, default=512)
    parser.add_argument("--probe-ridge", type=float, default=1e-3)
    parser.add_argument("--corruption-seed", type=int, default=DEFAULT_CORRUPTION_SEED)
    parser.add_argument("--eval-rollout-episode", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=False)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="lewm-memory-popgym")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--wandb-study", default="hacssm-v10-r1")
    parser.add_argument("--extra-tag", default="")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("invalid training budget")
    if args.sigreg_projections < 1:
        raise ValueError("V10-J requires a SIGReg diagnostic projection")
    if args.probe_ridge < 0 or not math.isfinite(args.probe_ridge):
        raise ValueError("probe ridge must be finite and non-negative")
    if args.wandb and args.wandb_mode != "online":
        raise ValueError("official V10 W&B logging must be online")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    train_metadata = load_cache(args.train_data)
    val_metadata = load_cache(args.val_data)
    if train_metadata.split != "train" or val_metadata.split != "val":
        raise ValueError("V10 train/validation cache split mismatch")
    if train_metadata.env_id != val_metadata.env_id:
        raise ValueError("V10 train/validation environments differ")
    if (
        train_metadata.length != val_metadata.length
        or train_metadata.img_size != val_metadata.img_size
        or train_metadata.action_dim != val_metadata.action_dim
        or train_metadata.state_dim != val_metadata.state_dim
    ):
        raise ValueError("V10 train/validation cache schema mismatch")
    if args.img_size != train_metadata.img_size:
        raise ValueError("--img-size does not match V10 cache")
    if args.eval_rollout_episode < 0 or args.eval_rollout_episode >= val_metadata.episodes:
        raise ValueError("evaluation rollout episode is out of range")

    train_view = V10TrajectoryDataset(
        args.train_data, "train", args.corruption_seed, args.history_len)
    val_train_view = V10TrajectoryDataset(
        args.val_data, "train", args.corruption_seed, args.history_len)
    train_clean = V10TrajectoryDataset(
        args.train_data, "clean", args.corruption_seed, args.history_len)
    val_clean = V10TrajectoryDataset(
        args.val_data, "clean", args.corruption_seed, args.history_len)
    heldout = {
        condition: V10TrajectoryDataset(
            args.val_data, condition, args.corruption_seed, args.history_len)
        for condition in HELDOUT_CONDITIONS
    }

    model = build_model(args, train_metadata.action_dim).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _loader(train_view, args, train=True)
    val_loader = _loader(val_train_view, args, train=False)

    env_name = f"dmc:{train_metadata.env_id}"
    run_name = f"lewm-{env_name}-{args.memory_mode}-s{args.seed}"
    output_dir = Path(args.output_dir).resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json"):
        if (output_dir / filename).exists():
            raise FileExistsError(f"refusing to overwrite {output_dir / filename}")

    wb = None
    if args.wandb:
        import wandb
        tags = ["lewm-memory", "end-to-end-rgb", f"env:{env_name}",
                f"design:{args.memory_mode}", f"study:{args.wandb_study}"]
        if args.extra_tag:
            tags.extend(tag.strip() for tag in args.extra_tag.split(",") if tag.strip())
        wb = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            mode=args.wandb_mode,
            name=f"{args.wandb_study}-{run_name}",
            group=f"{args.wandb_study}:{env_name}",
            job_type=args.memory_mode,
            tags=tags,
            dir=str(output_dir),
            config=(vars(args) | {
                "env": env_name,
                "action_dim": train_metadata.action_dim,
                "state_dim": train_metadata.state_dim,
                "encoder_norm": "causal",
                "predictor_norm": "none",
                "training_objective": "v10j_joint_pred_variance_covariance_equal_weight",
                "clean_target_gradient_active": True,
                "ema_target_active": False,
                "target_stop_gradient": False,
                "prediction_loss_weight": 1.0,
                "variance_loss_weight": 1.0,
                "covariance_loss_weight": 1.0,
                "sigreg_optimization_weight": 0.0,
                "train_data_sha256": train_metadata.file_sha256,
                "val_data_sha256": val_metadata.file_sha256,
            }),
            settings=wandb.Settings(init_timeout=120),
        )
        if wb.offline or (args.wandb_entity and wb.entity != args.wandb_entity):
            raise RuntimeError("V10 W&B online/entity preflight failed")
        wb.define_metric("epoch")
        wb.define_metric("train/*", step_metric="epoch")
        wb.define_metric("val/*", step_metric="epoch")
        wb.define_metric("mem/*", step_metric="epoch")

    print(
        f"=== {run_name} | params={model.num_parameters():,} | "
        f"RGB end-to-end V10-J | amp={use_amp} ===", flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, use_amp)
        val_metrics = run_epoch(model, val_loader, None, device, use_amp)
        memory_log = _finite_memory_log(model)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"e{epoch:3d}/{args.epochs} ({time.time() - started:.1f}s) "
            f"train={train_metrics['loss']:.5f} pred={train_metrics['pred_loss']:.5f} "
            f"var={train_metrics['variance_loss']:.5f} "
            f"cov={train_metrics['covariance_loss']:.5f} | "
            f"val={val_metrics['loss']:.5f}", flush=True)
        if wb is not None:
            wb.log({
                "epoch": epoch,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
                **{f"mem/{key}": value for key, value in memory_log.items()},
            }, step=epoch)

    probe = fit_state_probe(model, train_clean, args, device, use_amp)
    online_diagnostics = encoder_diagnostics(model, val_clean, args, device)
    online_ceiling = probe_ceiling(
        model, val_clean, probe, args, device, use_amp)
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "env": env_name,
        "design": args.memory_mode,
        "seed": args.seed,
        "epochs": args.epochs,
        "encoder_type": "vit",
        "encoder_frozen": False,
        "encoder_norm": "causal",
        "predictor_norm": "none",
        "end_to_end_rgb": True,
        "clean_target_gradient_active": True,
        "ema_target_active": False,
        "target_stop_gradient": False,
        "training_objective": "v10j_joint_pred_variance_covariance_equal_weight",
        "prediction_loss_weight": 1.0,
        "variance_loss_weight": 1.0,
        "covariance_loss_weight": 1.0,
        "vicreg_gradient_active": True,
        "sigreg_gradient_active": False,
        "sigreg_optimization_weight": 0.0,
        "sigreg_lambda": args.sigreg_lambda,
        "probe_ridge": args.probe_ridge,
        "probe_fit_split": "clean_train_online_joint_only",
        "headline_metric": "heldout_state_nmse",
        "train_data": str(train_metadata.path),
        "val_data": str(val_metadata.path),
        "train_data_sha256": train_metadata.file_sha256,
        "val_data_sha256": val_metadata.file_sha256,
        "train_data_content_sha256": train_metadata.content_sha256,
        "val_data_content_sha256": val_metadata.content_sha256,
        "train_episodes": train_metadata.episodes,
        "val_episodes": val_metadata.episodes,
        "length": train_metadata.length,
        "action_dim": train_metadata.action_dim,
        "state_dim": train_metadata.state_dim,
        "trainable_parameters": model.num_parameters(),
        "final_train_loss": history[-1]["train"]["loss"],
        "final_val_loss": history[-1]["val"]["loss"],
        "val_pred_loss": history[-1]["val"]["pred_loss"],
        **online_diagnostics,
        **orbit_diagnostics(model, val_clean, device),
        **online_ceiling,
    }
    window = min(10, max(1, args.epochs // 2))
    recent = np.mean([row["val"]["loss"] for row in history[-window:]])
    previous_rows = history[-2 * window:-window]
    previous = np.mean([row["val"]["pred_loss"] for row in previous_rows]) if previous_rows else recent
    recent = np.mean([row["val"]["pred_loss"] for row in history[-window:]])
    metrics["convergence_relative_change"] = float(
        (previous - recent) / max(previous, 1e-12))

    clean_result, _ = evaluate_condition(
        model, val_clean, probe, args, device, use_amp)
    metrics["clean_state_nmse"] = clean_result["primary"]
    metrics["clean_state_r2"] = clean_result["r2"]

    condition_rollouts = {}
    heldout_primary = []
    for condition, dataset in heldout.items():
        result, rollout = evaluate_condition(
            model, dataset, probe, args, device, use_amp,
            rollout_episode=args.eval_rollout_episode)
        metrics[f"{condition}_state_nmse"] = result["primary"]
        metrics[f"{condition}_predicted_state_r2"] = result["r2"]
        for phase in ("gap", "deep", "first_post", "post"):
            metrics[f"{condition}_state_nmse_{phase}"] = result[phase]
        heldout_primary.append(result["primary"])
        if rollout is None:
            raise RuntimeError(f"missing rollout episode for {condition}")
        condition_rollouts[condition] = rollout
    metrics["heldout_state_nmse"] = float(np.mean(heldout_primary))
    metrics.update({
        f"memory_{key}": float(value)
        for key, value in model.horizons().items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    })

    rollout_path = output_dir / "eval_rollout.npz"
    rollout_arrays, rollout_video = _rollout_package(
        condition_rollouts, rollout_path, args.eval_rollout_episode)
    rollout_hash = sha256_file(rollout_path)
    metrics["eval_rollout_episode"] = args.eval_rollout_episode
    metrics["eval_rollout_sha256"] = rollout_hash

    # Save the scientific payload before declaring the remote run successful.
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "final_metrics": metrics,
        "history": history,
        "state_probe": probe,
    }, output_dir / "model.pt")
    with (output_dir / "metrics.json").open("x") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    wandb_record = None
    if wb is not None:
        import wandb
        table = _make_rollout_table(wandb, rollout_arrays)
        final_log = {
            **{f"eval/{key}": value for key, value in metrics.items()
               if isinstance(value, (int, float)) and math.isfinite(float(value))},
            "eval/rollout_trace": table,
            "eval/paired_rollout": wandb.Video(
                rollout_video, fps=6, format="mp4",
                caption=("rows: freeze, gaussian noise, checkerboard, long freeze; "
                         "left: corrupted RGB, right: clean target")),
        }
        artifact_name = f"eval-rollout-{wb.id}"
        artifact = wandb.Artifact(
            artifact_name,
            type="evaluation-rollout",
            metadata={
                "schema_version": 1,
                "study": args.wandb_study,
                "env": env_name,
                "design": args.memory_mode,
                "seed": args.seed,
                "episode": args.eval_rollout_episode,
                "sha256": rollout_hash,
                "semantics": "heldout-corruption normalized physics-state evaluation trace",
                **_design_metadata(args.memory_mode),
            },
        )
        artifact.add_file(str(rollout_path), name="eval_rollout.npz")
        wb.log_artifact(artifact)
        wb.log(final_log)
        wb.summary.update(metrics)
        wandb_record = {
            "schema_version": 1,
            "run_id": str(wb.id),
            "run_name": str(wb.name),
            "url": str(wb.url),
            "entity": str(wb.entity),
            "project": str(wb.project),
            "mode": "offline" if wb.offline else "online",
            "study": args.wandb_study,
            "state": "finished",
            "eval_rollout_artifact_name": artifact_name,
            "eval_rollout_sha256": rollout_hash,
            "eval_rollout_episode": args.eval_rollout_episode,
        }
        wb.finish(exit_code=0)
        if wandb_record["mode"] != "online":
            raise RuntimeError("V10 W&B run did not finish online")
        with (output_dir / "wandb_run.json").open("x") as stream:
            json.dump(wandb_record, stream, indent=2, sort_keys=True)
            stream.write("\n")

    print(
        f"=== done {run_name}: heldout_state_nmse={metrics['heldout_state_nmse']:.6f} "
        f"clean={metrics['clean_state_nmse']:.6f} ===", flush=True)


if __name__ == "__main__":
    main()
