#!/usr/bin/env python3
"""Train one V19 P0 host-preflight cell (docs/V19_PROPOSAL.md section 4.1).

Claim-1 preflight: is a host healthy on the certified memory tasks in the
corruption-on regime, before any memory carrier is attached?  Two hosts, no
memory carriers, health gates absolute per arm (not a ranking):

- ``sigreg`` (the bet, exact-LeWM recipe): ViT-tiny width encoder (D=192, 12
  layers, 3 heads, patch 8 on 64x64), [CLS] embedding through a one-hidden-
  layer MLP projection head with BatchNorm; causal AdaLN predictor (6 layers,
  16 heads, dropout 0.1, zero-init action conditioning at every layer);
  teacher-forcing next-latent prediction on the SINGLE observed (corrupted)
  stream over sliding H=3 windows.  ``L = L_pred + 0.1 * SIGReg(Z)`` with the
  V16 exact Epps-Pulley statistic (K=1 full space, M=1024 fresh sketch
  directions).  No paired clean/corrupted views and no per-frame causal
  normalization -- that is the registered exactness delta.
- ``vicreg`` (the reference): the V18 ``vicreg_none`` host verbatim at D=128
  (corrupted input stream, active clean targets, per-frame causal encoder
  normalization, unit-weight VICReg variance+covariance on the clean
  targets), reusing scripts/train_lewm_v8_v18.py code paths directly.

Per-epoch validation telemetry (the point of P0): V18 encoder health (mean
channel variance, covariance effective rank = exp of spectral entropy), val
predictive loss, and the V16 collapse signature -- raw SIGReg against the
analytic projected-zero plateau for the batch size, per-direction Epps-Pulley
min/median/max, and the encoder regularizer/prediction gradient ratio on one
batch per epoch.  Final gates land in gates.json, per-epoch rows in
history.csv, the covariance eigenspectrum trajectory in eigenspectra.npy, and
the final encoder state_dict in encoder.pt (P1b input).

Env overrides for smoke: V19_P0_EPOCHS, V19_P0_EPISODES.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.leworldmodel import LeWorldModel
from lewm.models.sigreg import MultiSubspaceSIGReg
from lewm.tasks_v19.base import EpisodeBatch, load_bank
import scripts.make_v19_p0_data as p0_data
import scripts.train_hacssm_v11 as v11
import scripts.train_lewm_v8_v18 as v18

HOSTS = ("sigreg", "vicreg")
DEFAULT_EPOCHS = 100
IMG_SIZE = 64
HISTORY_LEN = 3
SIGREG_LAMBDA = 0.1
SIGREG_PROJECTIONS = 1024

HOST_CONFIGS: dict[str, dict[str, Any]] = {
    # Exact-LeWM width (the bet); capacity difference vs the reference is
    # intentional and registered.
    "sigreg": dict(embed_dim=192, encoder_layers=12, encoder_heads=3,
                   predictor_layers=6, predictor_heads=16, patch_size=8,
                   dropout=0.1, history_len=HISTORY_LEN,
                   encoder_norm="batch", predictor_norm="batch"),
    # Exact-V18 host (the reference); see scripts/train_lewm_v8_v18.py.
    "vicreg": dict(embed_dim=128, encoder_layers=6, encoder_heads=4,
                   predictor_layers=4, predictor_heads=8, patch_size=8,
                   dropout=0.1, history_len=HISTORY_LEN,
                   encoder_norm="causal", predictor_norm="none"),
}

GATES = {
    "min_effective_rank": 16.0,
    "min_channel_variance": 1e-4,
    "max_convergence_relative_change": 0.05,
    "plateau_ratio_tolerance": 0.02,
    "plateau_min_consecutive_epochs": 10,
    "grad_ratio_threshold": 100.0,
    "grad_ratio_min_consecutive_epochs": 10,
}

HEALTH_EPISODES = 64   # V18 diagnostic subsample (2 batches x 32 episodes)
HEALTH_CHUNK = 8       # episodes per encode chunk during diagnostics

LOSS_KEYS = ("loss", "predictive_loss", "regularizer_loss", "sigreg_loss",
             "variance_loss", "covariance_loss")
HISTORY_FIELDS = (
    "epoch", "epoch_seconds",
    *(f"train_{key}" for key in LOSS_KEYS),
    *(f"val_{key}" for key in LOSS_KEYS),
    "ep_plateau_batch", "ep_ratio", "ep_dir_min", "ep_dir_median", "ep_dir_max",
    "grad_ratio",
    "encoder_mean_channel_variance", "encoder_covariance_effective_rank",
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

class P0EpisodeDataset(Dataset):
    """Episode-level view of the P0 cache banks.

    The observed stream is corrupted at cache time; the clean frames are the
    V18 paired targets and are attached only for the vicreg arm (the sigreg
    arm is single-stream by registration).
    """

    def __init__(self, observed: EpisodeBatch, clean: EpisodeBatch | None = None):
        if clean is not None and clean.frames.shape != observed.frames.shape:
            raise ValueError("observed/clean bank shape mismatch")
        self.observed = observed.frames
        self.clean = None if clean is None else clean.frames
        self.actions = observed.actions.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return self.observed.shape[0]

    @staticmethod
    def _frames_tensor(frames: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "observed": self._frames_tensor(self.observed[index]),
            "actions": torch.from_numpy(self.actions[index]),
            "episode_index": torch.tensor(index, dtype=torch.long),
        }
        if self.clean is not None:
            item["clean"] = self._frames_tensor(self.clean[index])
        return item


def _loader(dataset: Dataset, args: argparse.Namespace, *, train: bool) -> DataLoader:
    """V10/V16 loader conventions (seeded shuffle, drop_last on train)."""
    generator = torch.Generator().manual_seed(100_000 + args.seed) if train else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        persistent_workers=args.num_workers > 0,
    )


# --------------------------------------------------------------------------
# Hosts
# --------------------------------------------------------------------------

def build_sigreg_host(action_dim: int) -> LeWorldModel:
    """Exact-LeWM host: ViT-tiny width + MLP-BN projection head + K=1 SIGReg."""
    config = HOST_CONFIGS["sigreg"]
    world = LeWorldModel(
        img_size=IMG_SIZE,
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        action_dim=action_dim,
        encoder_layers=config["encoder_layers"],
        encoder_heads=config["encoder_heads"],
        predictor_layers=config["predictor_layers"],
        predictor_heads=config["predictor_heads"],
        history_len=config["history_len"],
        dropout=config["dropout"],
        predictor_norm=config["predictor_norm"],
        encoder_norm=config["encoder_norm"],
        sigreg_lambda=SIGREG_LAMBDA,
        sigreg_projections=SIGREG_PROJECTIONS,
    )
    # LeWM projection head: [CLS] -> Linear -> BatchNorm -> GELU -> Linear.
    # ViTTinyEncoder ships a single Linear+BN projector; the exact-LeWM recipe
    # registered for P0 uses the one-hidden-layer MLP head, batch statistics
    # in train and eval (track_running_stats=False, the LeWM convention).
    dim = config["embed_dim"]
    head = nn.Sequential(
        nn.Linear(dim, dim),
        nn.BatchNorm1d(dim, track_running_stats=False),
        nn.GELU(),
        nn.Linear(dim, dim),
    )
    head.apply(world.encoder._init_weights)
    world.encoder.projector = head
    # V16 exact Epps-Pulley statistic, full-space K=1 path, M=1024 fresh
    # sketch directions per forward (lewm/models/sigreg.py).
    world.sigreg = MultiSubspaceSIGReg(
        embed_dim=dim, num_subspaces=1, num_projections=SIGREG_PROJECTIONS)
    return world


def build_vicreg_host(action_dim: int):
    """V18 ``vicreg_none`` host verbatim via scripts/train_lewm_v8_v18.py."""
    config = HOST_CONFIGS["vicreg"]
    namespace = argparse.Namespace(
        design="vicreg_none", img_size=IMG_SIZE, patch_size=config["patch_size"],
        embed_dim=config["embed_dim"], encoder_layers=config["encoder_layers"],
        encoder_heads=config["encoder_heads"],
        predictor_layers=config["predictor_layers"],
        predictor_heads=config["predictor_heads"],
        history_len=config["history_len"], dropout=config["dropout"],
        sigreg_projections=512)
    return v18.build_model(namespace, action_dim)


def host_encoder(host: str, model: nn.Module) -> nn.Module:
    return model.encoder if host == "sigreg" else model.world.encoder


def host_encode(host: str, model: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    return model.encode(frames) if host == "sigreg" else model.world.encode(frames)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def sigreg_losses(world: LeWorldModel, observed: torch.Tensor,
                  actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Exact-LeWM objective on the single observed stream.

    ``L = ||z_hat_{t+1} - z_{t+1}||^2 + lambda * SIGReg(Z)`` with teacher-
    forcing over all aligned H=3 sliding windows (V18 window policy) and no
    stop-gradient anywhere -- prediction inputs, targets, and the SIGReg
    argument are all the same one-pass embeddings of the corrupted stream.
    """
    z = world.encode(observed)                                    # (B, L, D)
    latent_windows, action_windows, targets = v18.sliding_predictor_windows(
        z, actions, z, history=world.history_len)
    prediction = world.predictor(latent_windows, action_windows)[:, -1]
    predictive_loss = F.mse_loss(prediction.float(), targets.float())
    sigreg_raw = world.sigreg(z)          # FP32 internally, batch-size scaled
    regularizer_loss = world.sigreg_lambda * sigreg_raw
    zero = predictive_loss.new_zeros(())
    return {
        "loss": predictive_loss + regularizer_loss,
        "predictive_loss": predictive_loss,
        "regularizer_loss": regularizer_loss,
        "sigreg_loss": sigreg_raw,
        "variance_loss": zero,
        "covariance_loss": zero,
    }


def compute_host_losses(host: str, model: nn.Module,
                        batch: Mapping[str, torch.Tensor],
                        device: torch.device) -> dict[str, torch.Tensor]:
    observed = batch["observed"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    if host == "sigreg":
        return sigreg_losses(model, observed, actions)
    clean = batch["clean"].to(device, non_blocking=True)
    # V18 loss verbatim: sliding H=3 prediction of active clean targets plus
    # unit-weight VICReg variance+covariance on the clean-target embeddings.
    return v18.compute_losses(model, observed, clean, actions, sigreg_lambda=0.0)


# --------------------------------------------------------------------------
# Collapse instruments (the V16 signature)
# --------------------------------------------------------------------------

def analytic_delta_plateau(sigreg: MultiSubspaceSIGReg, batch_size: int) -> float:
    """Epps-Pulley value of an empirical delta distribution projected to zero.

    A collapsed batch embeds every sample at one point; every sketch
    projection is then a delta at zero whose ECF is identically (1, 0), so
    the statistic equals ``B * sum_j w_j (1 - phi_j)^2`` for every direction
    (``weights`` already carries the trapezoid quadrature and the phi
    window).  This is V16's projected-zero plateau: 25.731 at batch 64.
    """
    with torch.no_grad():
        return float(batch_size
                     * (((1.0 - sigreg.phi) ** 2) * sigreg.weights).sum())


@torch.no_grad()
def per_direction_ep_statistics(sigreg: MultiSubspaceSIGReg,
                                embeddings: torch.Tensor) -> torch.Tensor:
    """Per-sketch-direction Epps-Pulley statistics, shape (K*M,).

    Mirrors ``MultiSubspaceSIGReg.forward`` exactly (fresh unit directions,
    FP32 outside autocast, batch-size scaling) but keeps the per-direction
    resolution the projected-zero plateau detector needs; statistics are
    averaged over sequence positions.
    """
    embeddings = sigreg._canonicalize(embeddings)
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        projected = sigreg.project(embeddings.float())            # (K, T, B, d_s)
        directions = torch.randn(
            sigreg.num_subspaces, sigreg.subspace_dim, sigreg.num_projections,
            device=projected.device, dtype=torch.float32)
        directions.div_(directions.norm(p=2, dim=1, keepdim=True))
        samples = torch.einsum(
            'ktbd,kdn->ktbn', projected, directions).unsqueeze(-1) * sigreg.t
        ecf_real = samples.cos().mean(dim=2)
        ecf_imag = samples.sin().mean(dim=2)
        error = (ecf_real - sigreg.phi).square() + ecf_imag.square()
        statistic = (error @ sigreg.weights) * projected.size(2)  # (K, T, M)
        return statistic.mean(dim=1).reshape(-1)


def gradient_ratio(host: str, model: nn.Module, batch: Mapping[str, torch.Tensor],
                   device: torch.device, use_amp: bool) -> float:
    """||grad(weighted regularizer)|| / ||grad(L_pred)|| on encoder parameters.

    Measured on one fixed validation batch per epoch with a deterministic
    (eval-mode, dropout-off) forward -- the V16 collapse-signature instrument
    that read >12,000 in the trapped Sub-JEPA cells.
    """
    was_training = model.training
    model.eval()
    parameters = [parameter for parameter in host_encoder(host, model).parameters()
                  if parameter.requires_grad]
    try:
        with torch.enable_grad():
            amp_context = (torch.autocast("cuda", dtype=torch.bfloat16)
                           if use_amp else torch.autocast("cpu", enabled=False))
            with amp_context:
                losses = compute_host_losses(host, model, batch, device)
            regularizer_grads = torch.autograd.grad(
                losses["regularizer_loss"], parameters,
                retain_graph=True, allow_unused=True)
            prediction_grads = torch.autograd.grad(
                losses["predictive_loss"], parameters, allow_unused=True)
    finally:
        model.train(was_training)

    def _norm(grads: tuple[torch.Tensor | None, ...]) -> float:
        squares = [grad.float().square().sum() for grad in grads if grad is not None]
        if not squares:
            return 0.0
        return float(torch.stack(squares).sum().sqrt())

    return _norm(regularizer_grads) / max(_norm(prediction_grads), 1e-12)


# --------------------------------------------------------------------------
# Encoder health (V18 definition)
# --------------------------------------------------------------------------

def covariance_spectrum(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Channel variances and ascending covariance eigenvalues, float64.

    The math mirrors ``scripts.train_hacssm_v10.encoder_diagnostics``
    verbatim: center, per-channel mean square, (n-1)-normalized covariance,
    clamped eigvalsh.
    """
    matrix = matrix.double()
    centered = matrix - matrix.mean(dim=0)
    variance = centered.square().mean(dim=0)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    return variance, eigenvalues


def effective_rank(eigenvalues: torch.Tensor) -> float:
    """exp(spectral entropy) -- the V18 covariance effective rank."""
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    return float(torch.exp(
        -(probabilities * probabilities.clamp_min(1e-30).log()).sum()))


@torch.no_grad()
def encoder_health(host: str, model: nn.Module, health_frames: torch.Tensor,
                   device: torch.device, use_amp: bool
                   ) -> tuple[float, float, np.ndarray]:
    """V18 encoder diagnostics on the clean validation stream.

    Returns (mean channel variance, covariance effective rank, eigenvalues).
    ``health_frames`` is the fixed (E, L, C, H, W) clean-frame subsample.
    """
    was_training = model.training
    model.eval()
    try:
        latents = []
        for start in range(0, health_frames.shape[0], HEALTH_CHUNK):
            segment = health_frames[start:start + HEALTH_CHUNK].to(device)
            amp_context = (torch.autocast("cuda", dtype=torch.bfloat16)
                           if use_amp else torch.autocast("cpu", enabled=False))
            with amp_context:
                latents.append(host_encode(host, model, segment).float().cpu())
    finally:
        model.train(was_training)
    matrix = torch.cat(latents).reshape(-1, latents[0].shape[-1])
    variance, eigenvalues = covariance_spectrum(matrix)
    return (float(variance.mean()), effective_rank(eigenvalues),
            eigenvalues.numpy().astype(np.float64))


# --------------------------------------------------------------------------
# Epochs
# --------------------------------------------------------------------------

def run_epoch(host: str, model: nn.Module, loader: DataLoader,
              optimizer: torch.optim.Optimizer | None, device: torch.device,
              use_amp: bool) -> dict[str, float]:
    """One pass over the loader; mirrors the V16/V18 epoch conventions.

    Every episode contributes all its aligned H=3 windows (the V18 sampling).
    Validation passes additionally return the mean SIGReg/plateau ratio with
    the plateau evaluated at each batch's actual size.
    """
    train = optimizer is not None
    model.train(train)
    totals = {key: 0.0 for key in LOSS_KEYS}
    ratios: list[float] = []
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            if train:
                optimizer.zero_grad(set_to_none=True)
            amp_context = (torch.autocast("cuda", dtype=torch.bfloat16)
                           if use_amp else torch.autocast("cpu", enabled=False))
            with amp_context:
                losses = compute_host_losses(host, model, batch, device)
            if train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = batch["observed"].shape[0]
            for key in LOSS_KEYS:
                totals[key] += float(losses[key].detach()) * batch_size
            count += batch_size
            if host == "sigreg":
                plateau = analytic_delta_plateau(model.sigreg, batch_size)
                ratios.append(float(losses["sigreg_loss"].detach()) / plateau)
    if not count:
        raise RuntimeError("empty V19 P0 epoch")
    metrics = {key: value / count for key, value in totals.items()}
    metrics["ep_ratio"] = float(np.mean(ratios)) if ratios else float("nan")
    return metrics


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def _longest_streak(flags: Iterable[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def compute_gates(rows: list[dict[str, float]], host: str) -> dict[str, Any]:
    """Registered P0 health gates over the per-epoch telemetry rows.

    - rank >= 16 and channel variance >= 1e-4 at the final epoch;
    - late-window convergence of val predictive loss <= 5% (mean over epochs
      81-90 vs 91-100 at 100 epochs; V16 window rule for shorter runs);
    - plateau_flag: SIGReg within 2% of the analytic projected-zero plateau
      for >= 10 consecutive epochs (sigreg host only);
    - grad_ratio_flag: encoder regularizer/prediction gradient ratio > 100
      sustained for >= 10 consecutive epochs (reported, not gated).
    overall_pass = rank AND variance AND convergence AND (sigreg: NOT plateau).
    """
    if host not in HOSTS:
        raise ValueError(f"unknown host {host!r}")
    if not rows:
        raise ValueError("no telemetry rows")
    final = rows[-1]
    epochs = len(rows)
    window = min(10, max(1, epochs // 2))
    recent_rows = rows[-window:]
    previous_rows = rows[-2 * window:-window] or recent_rows
    previous = float(np.mean(
        [row["val_predictive_loss"] for row in previous_rows]))
    recent = float(np.mean([row["val_predictive_loss"] for row in recent_rows]))
    convergence = abs(previous - recent) / max(abs(previous), 1e-12)

    plateau_streak = _longest_streak(
        math.isfinite(row["ep_ratio"])
        and abs(row["ep_ratio"] - 1.0) <= GATES["plateau_ratio_tolerance"]
        for row in rows)
    grad_streak = _longest_streak(
        math.isfinite(row["grad_ratio"])
        and row["grad_ratio"] > GATES["grad_ratio_threshold"]
        for row in rows)

    final_rank = float(final["encoder_covariance_effective_rank"])
    final_variance = float(final["encoder_mean_channel_variance"])
    rank_pass = final_rank >= GATES["min_effective_rank"]
    variance_pass = final_variance >= GATES["min_channel_variance"]
    convergence_pass = convergence <= GATES["max_convergence_relative_change"]
    plateau_flag = plateau_streak >= GATES["plateau_min_consecutive_epochs"]
    grad_ratio_flag = grad_streak >= GATES["grad_ratio_min_consecutive_epochs"]
    overall_pass = bool(rank_pass and variance_pass and convergence_pass
                        and (not plateau_flag if host == "sigreg" else True))
    return {
        "host": host,
        "epochs": epochs,
        "convergence_window_epochs": window,
        "final_effective_rank": final_rank,
        "final_channel_variance": final_variance,
        "final_val_predictive_loss": float(final["val_predictive_loss"]),
        "convergence_relative_change": float(convergence),
        "plateau_max_streak_epochs": int(plateau_streak),
        "grad_ratio_max_streak_epochs": int(grad_streak),
        "rank_pass": bool(rank_pass),
        "variance_pass": bool(variance_pass),
        "convergence_pass": bool(convergence_pass),
        "plateau_flag": bool(plateau_flag),
        "grad_ratio_flag": bool(grad_ratio_flag),
        "overall_pass": overall_pass,
        "thresholds": dict(GATES),
    }


# --------------------------------------------------------------------------
# W&B (guarded: an outage must never invalidate a preflight run)
# --------------------------------------------------------------------------

def _guarded(stage: str, function, /, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - reporting is best-effort
        warnings.warn(f"wandb stage {stage} failed: {error!r}", stacklevel=2)
        return None


def _wandb_init(args: argparse.Namespace, run_config: dict, output_dir: Path):
    def _init():
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"p0-{args.host}-{args.task}-s{args.seed}",
            group=f"p0-{args.task}", tags=["p0", "v19"],
            dir=str(output_dir), config=run_config,
            settings=wandb.Settings(init_timeout=180))
        run.define_metric("epoch")
        for namespace in ("train/*", "val/*", "sig/*", "grad/*", "health/*",
                          "perf/*"):
            run.define_metric(namespace, step_metric="epoch")
        return run
    return _guarded("init", _init)


def _line_figure(values: list[float], title: str, ylabel: str,
                 threshold: float | None = None, log_scale: bool = False):
    from matplotlib.figure import Figure
    figure = Figure(figsize=(6.5, 3.5))
    axis = figure.subplots()
    epochs = np.arange(1, len(values) + 1)
    axis.plot(epochs, values, lw=1.5)
    if threshold is not None:
        axis.axhline(threshold, color="crimson", ls="--", lw=1.2,
                     label=f"threshold {threshold:g}")
        axis.legend(fontsize=8)
    if log_scale:
        axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    return figure


def _heatmap_figure(eigenspectra: np.ndarray, title: str):
    from matplotlib.figure import Figure
    figure = Figure(figsize=(7.0, 4.0))
    axis = figure.subplots()
    data = np.log10(np.maximum(eigenspectra, 1e-12))
    image = axis.imshow(data.T, aspect="auto", origin="lower",
                        extent=(1, eigenspectra.shape[0], 0,
                                eigenspectra.shape[1]), cmap="viridis")
    figure.colorbar(image, ax=axis, label="log10 eigenvalue")
    axis.set_xlabel("epoch")
    axis.set_ylabel("eigenvalue index (ascending)")
    axis.set_title(title)
    return figure


def _log_final_wandb(run, rows: list[dict[str, float]],
                     eigenspectra: np.ndarray, gates: dict, args) -> None:
    import wandb
    from lewm.tasks_v19.wandb_utils import _figure_to_image

    label = f"{args.host}/{args.task}/s{args.seed}"
    figures = {
        "figures/effective_rank": _line_figure(
            [row["encoder_covariance_effective_rank"] for row in rows],
            f"covariance effective rank — {label}", "effective rank",
            threshold=GATES["min_effective_rank"]),
        "figures/channel_variance": _line_figure(
            [row["encoder_mean_channel_variance"] for row in rows],
            f"mean channel variance — {label}", "variance",
            threshold=GATES["min_channel_variance"], log_scale=True),
        "figures/grad_ratio": _line_figure(
            [row["grad_ratio"] for row in rows],
            f"encoder grad ratio ||∇reg||/||∇pred|| — {label}", "ratio",
            threshold=GATES["grad_ratio_threshold"], log_scale=True),
        "figures/eigenspectrum": _heatmap_figure(
            eigenspectra, f"embedding covariance eigenspectrum — {label}"),
    }
    if args.host == "sigreg":
        figures["figures/ep_ratio"] = _line_figure(
            [row["ep_ratio"] for row in rows],
            f"SIGReg / projected-zero plateau — {label}", "EP ratio",
            threshold=1.0)
    run.log({key: wandb.Image(_figure_to_image(figure))
             for key, figure in figures.items()})

    gate_rows = [[name, gates[name]] for name in (
        "final_effective_rank", "final_channel_variance",
        "convergence_relative_change", "plateau_max_streak_epochs",
        "grad_ratio_max_streak_epochs", "rank_pass", "variance_pass",
        "convergence_pass", "plateau_flag", "grad_ratio_flag", "overall_pass")]
    run.log({"gates/table": wandb.Table(columns=["gate", "value"],
                                        data=gate_rows)})
    run.summary.update({f"gates/{key}": value for key, value in gates.items()
                        if not isinstance(value, dict)})


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=p0_data.P0_TASKS)
    parser.add_argument("--host", required=True, choices=HOSTS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", default="outputs/v19_p0")
    parser.add_argument("--data-root", default=p0_data.DEFAULT_ROOT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="lewm-v19")
    return parser.parse_args(argv)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _sanitize(value):
    """JSON-safe payload (gates.json is written with allow_nan=False)."""
    if isinstance(value, dict):
        return {key: _sanitize(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(entry) for entry in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    epochs = int(os.environ.get("V19_P0_EPOCHS", args.epochs))
    if epochs < 1 or args.batch_size < 4 or args.num_workers < 0:
        raise ValueError("invalid V19 P0 training configuration")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    cache = p0_data.ensure_task_cache(args.task, args.data_root)
    paths = p0_data.task_bank_paths(
        args.data_root, args.task,
        cache["train_episodes"], cache["val_episodes"])
    train_observed = load_bank(paths["train"]["observed"])
    train_clean = (load_bank(paths["train"]["clean"])
                   if args.host == "vicreg" else None)
    val_observed = load_bank(paths["val"]["observed"])
    val_clean = load_bank(paths["val"]["clean"])
    action_dim = int(train_observed.actions.shape[-1])

    train_dataset = P0EpisodeDataset(train_observed, train_clean)
    val_dataset = P0EpisodeDataset(
        val_observed, val_clean if args.host == "vicreg" else None)
    train_loader = _loader(train_dataset, args, train=True)
    val_loader = _loader(val_dataset, args, train=False)
    if len(train_dataset) < args.batch_size:
        raise ValueError("train bank smaller than one batch")

    # Fixed instruments: the first validation batch (deterministic order) for
    # the gradient-ratio and per-direction EP probes, and the first
    # HEALTH_EPISODES clean validation episodes for the V18 encoder health.
    fixed_batch = default_collate(
        [val_dataset[index]
         for index in range(min(args.batch_size, len(val_dataset)))])
    health_count = min(HEALTH_EPISODES, val_clean.num_episodes)
    health_frames = P0EpisodeDataset._frames_tensor(
        val_clean.frames[:health_count].reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    ).reshape(health_count, val_clean.length, 3, IMG_SIZE, IMG_SIZE)

    model = (build_sigreg_host(action_dim) if args.host == "sigreg"
             else build_vicreg_host(action_dim)).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output) / args.task / args.host / f"s{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("gates.json", "history.csv", "encoder.pt",
                     "eigenspectra.npy"):
        if (output_dir / filename).exists():
            raise FileExistsError(f"refusing to overwrite {output_dir / filename}")

    run_config = {
        **vars(args), "epochs_effective": epochs, "action_dim": action_dim,
        "trainable_parameters": trainable, "sigreg_lambda": SIGREG_LAMBDA,
        "sigreg_projections": SIGREG_PROJECTIONS,
        "host_config": HOST_CONFIGS[args.host], "gates": dict(GATES),
        "objective": ("lewm_exact_single_stream_sliding_h3_plus_k1_sigreg"
                      if args.host == "sigreg"
                      else v18.OBJECTIVE),
        "train_episodes": cache["train_episodes"],
        "val_episodes": cache["val_episodes"],
        "corruption_seed": p0_data.CORRUPTION_SEED,
        "data_sha256": {split: cache["splits"][split]["sha256"]
                        for split in cache["splits"]},
    }
    run = _wandb_init(args, run_config, output_dir) if args.wandb else None

    print(f"=== v19-p0 {args.host}/{args.task}/s{args.seed} | "
          f"params={trainable:,} | epochs={epochs} | "
          f"train={len(train_dataset)} val={len(val_dataset)} | amp={use_amp} ===",
          flush=True)

    rows: list[dict[str, float]] = []
    eigenspectra: list[np.ndarray] = []
    history_path = output_dir / "history.csv"
    with history_path.open("x", newline="") as history_stream:
        writer = csv.DictWriter(history_stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            started = time.time()
            train_metrics = run_epoch(
                args.host, model, train_loader, optimizer, device, use_amp)
            val_metrics = run_epoch(
                args.host, model, val_loader, None, device, use_amp)
            if args.host == "sigreg":
                with torch.no_grad():
                    model.eval()
                    amp_context = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if use_amp else torch.autocast("cpu", enabled=False))
                    with amp_context:
                        fixed_z = model.encode(
                            fixed_batch["observed"].to(device))
                    direction_stats = per_direction_ep_statistics(
                        model.sigreg, fixed_z.float())
                ep_dir = (float(direction_stats.min()),
                          float(direction_stats.median()),
                          float(direction_stats.max()))
            else:
                ep_dir = (float("nan"),) * 3
            grad_ratio = gradient_ratio(
                args.host, model, fixed_batch, device, use_amp)
            variance, rank, eigenvalues = encoder_health(
                args.host, model, health_frames, device, use_amp)
            eigenspectra.append(eigenvalues)
            row = {
                "epoch": epoch,
                "epoch_seconds": time.time() - started,
                **{f"train_{key}": train_metrics[key] for key in LOSS_KEYS},
                **{f"val_{key}": val_metrics[key] for key in LOSS_KEYS},
                "ep_plateau_batch": (
                    analytic_delta_plateau(model.sigreg, args.batch_size)
                    if args.host == "sigreg" else float("nan")),
                "ep_ratio": val_metrics["ep_ratio"],
                "ep_dir_min": ep_dir[0],
                "ep_dir_median": ep_dir[1],
                "ep_dir_max": ep_dir[2],
                "grad_ratio": grad_ratio,
                "encoder_mean_channel_variance": variance,
                "encoder_covariance_effective_rank": rank,
            }
            rows.append(row)
            writer.writerow({key: f"{value:.8g}" if isinstance(value, float)
                             else value for key, value in row.items()})
            history_stream.flush()
            print(
                f"e{epoch:3d}/{epochs} ({row['epoch_seconds']:.1f}s) "
                f"train={train_metrics['loss']:.5f} "
                f"pred={train_metrics['predictive_loss']:.5f} "
                f"reg={train_metrics['regularizer_loss']:.5f} | "
                f"val pred={val_metrics['predictive_loss']:.5f} "
                f"rank={rank:.1f} var={variance:.2e} "
                f"ep_ratio={row['ep_ratio']:.3f} grad_ratio={grad_ratio:.2f}",
                flush=True)
            if run is not None:
                payload = {
                    "epoch": epoch,
                    **{f"train/{key}": train_metrics[key] for key in LOSS_KEYS},
                    **{f"val/{key}": val_metrics[key] for key in LOSS_KEYS},
                    "grad/encoder_ratio": grad_ratio,
                    "health/mean_channel_variance": variance,
                    "health/covariance_effective_rank": rank,
                    "perf/epoch_seconds": row["epoch_seconds"],
                }
                if args.host == "sigreg":
                    payload.update({
                        "sig/raw": val_metrics["sigreg_loss"],
                        "sig/plateau_batch": row["ep_plateau_batch"],
                        "sig/ep_ratio": row["ep_ratio"],
                        "sig/dir_min": ep_dir[0],
                        "sig/dir_median": ep_dir[1],
                        "sig/dir_max": ep_dir[2],
                    })
                _guarded("epoch_log", run.log, payload, step=epoch)

    gates = compute_gates(rows, args.host)
    gates.update({
        "schema_version": 1,
        "study": "v19-p0-host-preflight",
        "task": args.task,
        "seed": args.seed,
        "train_episodes": cache["train_episodes"],
        "val_episodes": cache["val_episodes"],
        "trainable_parameters": trainable,
        "health_stream": "val_clean_first_64_episodes",
        "mean_epoch_seconds": float(np.mean(
            [row["epoch_seconds"] for row in rows])),
        "final_grad_ratio": _sanitize(rows[-1]["grad_ratio"]),
        "final_ep_ratio": _sanitize(rows[-1]["ep_ratio"]),
        "data_sha256": run_config["data_sha256"],
        "config": {key: value for key, value in run_config.items()
                   if key not in ("data_sha256",)},
    })

    spectra = np.stack(eigenspectra)
    np.save(output_dir / "eigenspectra.npy", spectra)
    torch.save({
        "schema_version": 1,
        "encoder_state_dict": host_encoder(args.host, model).state_dict(),
        "host": args.host,
        "task": args.task,
        "seed": args.seed,
        "epochs": epochs,
        "action_dim": action_dim,
        "img_size": IMG_SIZE,
        "host_config": HOST_CONFIGS[args.host],
        "sigreg_head": ("linear_bn_gelu_linear" if args.host == "sigreg"
                        else None),
        "gates": {key: value for key, value in gates.items()
                  if isinstance(value, (bool, int, float, str))},
    }, output_dir / "encoder.pt")
    _write_json_exclusive(output_dir / "gates.json", _sanitize(gates))

    if run is not None:
        _guarded("final_log", _log_final_wandb, run, rows, spectra, gates, args)
        _guarded("finish", run.finish)

    verdict = "PASS" if gates["overall_pass"] else "FAIL"
    print(f"=== done v19-p0 {args.host}/{args.task}/s{args.seed}: {verdict} "
          f"rank={gates['final_effective_rank']:.1f} "
          f"var={gates['final_channel_variance']:.2e} "
          f"conv={gates['convergence_relative_change']:.4f} "
          f"plateau={gates['plateau_flag']} grad_flag={gates['grad_ratio_flag']} "
          f"-> {output_dir} ===", flush=True)


if __name__ == "__main__":
    main()
