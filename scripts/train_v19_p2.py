#!/usr/bin/env python3
"""Train one V19 P2 development-grid cell (docs/V19_PROPOSAL.md 4.2-4.3).

The carrier sits between the frozen P0 host recipes' encoder and predictor:
the predictor consumes ``z_tilde`` windows from ``lewm.models.v19_carriers``.

- ``--host sigreg`` (exact-LeWM, single stream): the carrier runs on the one
  observed (corrupted) stream; prediction targets and the SIGReg argument
  stay the raw encoder embeddings, exactly as in P0 — the carrier only feeds
  the predictor input.
- ``--host vicreg`` (V18 reference): the carrier runs on the corrupted
  stream; targets stay the active clean-target embeddings per V18.

Because every carrier read is zero-initialized, each arm is exactly its P0
host at step 0.  Carrier parameters join the AdamW optimizer; the ``lkc_nll``
arm adds ``carrier.aux_loss()`` (the cell's own innovation likelihood) at
unit weight — the single admitted auxiliary term (proposal 4.2).

Everything else mirrors scripts/train_v19_p0.py by import (host builders,
dataset, health/collapse instruments, gates, checkpointing discipline).
After training, the run exports per-episode evaluation arrays (prior_read at
every t, carrier telemetry, encoder embedding of o_0, actions, xi, events) to
``<output>/<task>/<arm>/s<seed>/eval_export.npz`` for scripts/eval_v19_p2.py.

Data: t1-t4 reuse the P0 caches read-only when present; the development
tasks (t1dev/t2dev) and any missing size get their own caches generated
under ``<output>/data`` with the identical P0 corruption recipe — the P0
data root is never written.  Env overrides for smoke: V19_P0_EPOCHS,
V19_P0_EPISODES (P0 conventions, shared deliberately).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.v19_carriers import (Carrier, CarrierOutput, make_carrier,
                                      matched_gru_hidden, matched_ssm_width)
from lewm.tasks_v19 import make_task
from lewm.tasks_v19.base import EpisodeBatch, load_bank, save_bank
import scripts.make_v19_p0_data as p0_data
import scripts.train_hacssm_v11 as v11
import scripts.train_lewm_v8_v18 as v18
import scripts.train_v19_p0 as p0

ARMS = ("none", "acgru", "acssm", "lkc", "lkc_nll", "lkc_k0", "lkc_b0",
        "lkc_kfix", "lkc_rfix", "lkc_alearn", "lkc_a2")
P2_TASKS = (*p0_data.P0_TASKS, "t1dev", "t2dev", "t3dev")
DEFAULT_EPOCHS = p0.DEFAULT_EPOCHS
EXPORT_SCHEMA_VERSION = 1
EXPORT_CHUNK = 16
LOSS_KEYS = (*p0.LOSS_KEYS, "carrier_nll")
TELEMETRY_KEYS = ("k_mean", "k_std", "sigma_minus_mean", "r_mean",
                  "innovation_norm", "state_norm")
HISTORY_FIELDS = (
    "epoch", "epoch_seconds",
    *(f"train_{key}" for key in LOSS_KEYS),
    *(f"val_{key}" for key in LOSS_KEYS),
    "ep_plateau_batch", "ep_ratio", "ep_dir_min", "ep_dir_median", "ep_dir_max",
    "grad_ratio",
    "encoder_mean_channel_variance", "encoder_covariance_effective_rank",
    *(f"val_carrier_{key}" for key in TELEMETRY_KEYS),
)
# Event window pairs shaded in the annotated telemetry figures.
_EVENT_WINDOWS = (("cue_on", "cue_off", "tab:orange", "cue"),
                  ("gap_on", "gap_off", "tab:red", "gap"),
                  ("corrupt_on", "corrupt_off", "tab:gray", "corruption"))


# --------------------------------------------------------------------------
# Data (P0 caches read-only; dev tasks cached under the P2 root)
# --------------------------------------------------------------------------

def resolve_banks(task: str, p0_root: str | Path, p2_root: str | Path
                  ) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    """Locate (or build) the four banks of ``task`` at the current sizes.

    P0 caches are reused read-only when valid; otherwise the banks are
    generated under ``p2_root`` with the exact P0 recipe (stream, data seeds,
    corruption).  The P0 data root is never written by this function.
    """
    if task not in P2_TASKS:
        raise ValueError(f"task must be one of {P2_TASKS}, got {task!r}")
    train_episodes, val_episodes = p0_data.episode_sizes()
    roots = ((Path(p0_root), False), (Path(p2_root), True))
    for root, writable in roots:
        paths = p0_data.task_bank_paths(root, task, train_episodes, val_episodes)
        if all(p0_data._cache_valid(path)
               for split in paths.values() for path in split.values()):
            return paths, {"root": str(root), "generated": False,
                           "writable": writable}
    p2_root = Path(p2_root)
    if p2_root.resolve() == Path(p0_root).resolve():
        raise ValueError("refusing to generate banks inside the P0 data root")
    paths = p0_data.task_bank_paths(p2_root, task, train_episodes, val_episodes)
    for split, (episodes, seed) in (("train", (train_episodes, p0_data.TRAIN_SEED)),
                                    ("val", (val_episodes, p0_data.VAL_SEED))):
        started = time.time()
        print(f"[v19-p2-data] {task}/{split}: generating {episodes} episodes "
              f"(stream={p0_data.STREAM}, seed={seed})", flush=True)
        clean = make_task(task).generate(p0_data.STREAM, episodes, seed)
        if clean.length != p0_data.EPISODE_LENGTH:
            raise RuntimeError(f"expected L={p0_data.EPISODE_LENGTH}, "
                               f"got {clean.length}")
        observed = p0_data.corrupt_bank(clean, p0_data.CORRUPTION_SEED)
        save_bank(clean, paths[split]["clean"])
        save_bank(observed, paths[split]["observed"])
        print(f"[v19-p2-data] {task}/{split}: wrote banks "
              f"({time.time() - started:.1f}s)", flush=True)
    return paths, {"root": str(p2_root), "generated": True, "writable": True}


# --------------------------------------------------------------------------
# Host plumbing (thin adapters over the imported P0 builders)
# --------------------------------------------------------------------------

def host_predictor(host: str, model: nn.Module) -> nn.Module:
    return model.predictor if host == "sigreg" else model.world.predictor


def host_history(host: str, model: nn.Module) -> int:
    return model.history_len if host == "sigreg" else model.world.history_len


def _loader(dataset, args: argparse.Namespace, *, train: bool,
            device: torch.device) -> DataLoader:
    """P0 loader conventions, but CUDA is only touched for CUDA devices.

    (p0._loader pins host memory whenever CUDA exists; on a machine whose
    GPUs are saturated by the running P0 grid, a --device cpu run must not
    initialize any CUDA context.)
    """
    generator = torch.Generator().manual_seed(100_000 + args.seed) if train else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=train,
        persistent_workers=args.num_workers > 0,
    )


def compute_p2_losses(host: str, model: nn.Module, carrier: Carrier,
                      batch: Mapping[str, torch.Tensor], device: torch.device
                      ) -> tuple[dict[str, torch.Tensor], CarrierOutput]:
    """Host loss with the carrier between encoder and predictor.

    sigreg: P0 single-stream objective with z_tilde windows and raw-z targets
    and SIGReg(Z); vicreg: V18 objective verbatim with fused := z_tilde.
    The lkc_nll arm's innovation likelihood is added at unit weight.
    """
    observed = batch["observed"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    if host == "sigreg":
        z = model.encode(observed)
        output = carrier(z, actions)
        latent_windows, action_windows, targets = v18.sliding_predictor_windows(
            output.z_tilde, actions, z, history=model.history_len)
        prediction = model.predictor(latent_windows, action_windows)[:, -1]
        predictive_loss = F.mse_loss(prediction.float(), targets.float())
        sigreg_raw = model.sigreg(z)
        regularizer_loss = model.sigreg_lambda * sigreg_raw
        zero = predictive_loss.new_zeros(())
        losses = {
            "predictive_loss": predictive_loss,
            "regularizer_loss": regularizer_loss,
            "sigreg_loss": sigreg_raw,
            "variance_loss": zero,
            "covariance_loss": zero,
        }
    else:
        clean = batch["clean"].to(device, non_blocking=True)
        clean_z = v11.encode_clean_active(model, clean)
        observed_z = model.world.encode(observed)
        output = carrier(observed_z, actions)
        latent_windows, action_windows, targets = v18.sliding_predictor_windows(
            output.z_tilde, actions, clean_z, history=model.world.history_len)
        prediction = model.world.predictor(latent_windows, action_windows)[:, -1]
        predictive_loss = F.mse_loss(prediction.float(), targets.float())
        variance_loss, covariance_loss = v11._vicreg_terms(clean_z)
        regularizer_loss = variance_loss + covariance_loss
        zero = predictive_loss.new_zeros(())
        losses = {
            "predictive_loss": predictive_loss,
            "regularizer_loss": regularizer_loss,
            "sigreg_loss": zero,
            "variance_loss": variance_loss,
            "covariance_loss": covariance_loss,
        }
    aux = carrier.aux_loss()
    carrier_nll = aux if aux is not None else losses["predictive_loss"].new_zeros(())
    losses["carrier_nll"] = carrier_nll
    losses["loss"] = (losses["predictive_loss"] + losses["regularizer_loss"]
                      + carrier_nll)
    return losses, output


def run_epoch(host: str, model: nn.Module, carrier: Carrier, loader: DataLoader,
              optimizer: torch.optim.Optimizer | None, device: torch.device,
              use_amp: bool) -> dict[str, float]:
    """One pass over the loader; P0 epoch conventions plus carrier telemetry.

    Validation passes additionally return batch-weighted means of the carrier
    telemetry channels (NaN for channels the arm does not emit) and, on the
    sigreg host, the SIGReg/plateau ratio.
    """
    train = optimizer is not None
    model.train(train)
    carrier.train(train)
    totals = {key: 0.0 for key in LOSS_KEYS}
    telemetry_totals = {key: 0.0 for key in TELEMETRY_KEYS}
    telemetry_counts = {key: 0 for key in TELEMETRY_KEYS}
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
                losses, output = compute_p2_losses(
                    host, model, carrier, batch, device)
            if train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(model.parameters(), carrier.parameters()), 1.0)
                optimizer.step()
            batch_size = batch["observed"].shape[0]
            for key in LOSS_KEYS:
                totals[key] += float(losses[key].detach()) * batch_size
            count += batch_size
            if not train:
                for key in TELEMETRY_KEYS:
                    value = output.telemetry.get(key)
                    if value is not None:
                        telemetry_totals[key] += float(
                            value.detach().float().mean()) * batch_size
                        telemetry_counts[key] += batch_size
            if host == "sigreg":
                plateau = p0.analytic_delta_plateau(model.sigreg, batch_size)
                ratios.append(float(losses["sigreg_loss"].detach()) / plateau)
    if not count:
        raise RuntimeError("empty V19 P2 epoch")
    metrics = {key: value / count for key, value in totals.items()}
    metrics["ep_ratio"] = float(np.mean(ratios)) if ratios else float("nan")
    for key in TELEMETRY_KEYS:
        metrics[f"carrier_{key}"] = (
            telemetry_totals[key] / telemetry_counts[key]
            if telemetry_counts[key] else float("nan"))
    return metrics


def gradient_ratio(host: str, model: nn.Module, carrier: Carrier,
                   batch: Mapping[str, torch.Tensor], device: torch.device,
                   use_amp: bool) -> float:
    """P0's ||grad(reg)||/||grad(L_pred)|| encoder instrument, carrier in path."""
    was_training = model.training
    model.eval()
    carrier.eval()
    parameters = [parameter
                  for parameter in p0.host_encoder(host, model).parameters()
                  if parameter.requires_grad]
    try:
        with torch.enable_grad():
            amp_context = (torch.autocast("cuda", dtype=torch.bfloat16)
                           if use_amp else torch.autocast("cpu", enabled=False))
            with amp_context:
                losses, _ = compute_p2_losses(host, model, carrier, batch, device)
            regularizer_grads = torch.autograd.grad(
                losses["regularizer_loss"], parameters,
                retain_graph=True, allow_unused=True)
            prediction_grads = torch.autograd.grad(
                losses["predictive_loss"], parameters, allow_unused=True)
    finally:
        model.train(was_training)
        carrier.train(was_training)

    def _norm(grads: tuple[torch.Tensor | None, ...]) -> float:
        squares = [grad.float().square().sum()
                   for grad in grads if grad is not None]
        return float(torch.stack(squares).sum().sqrt()) if squares else 0.0

    return _norm(regularizer_grads) / max(_norm(prediction_grads), 1e-12)


# --------------------------------------------------------------------------
# Eval export (the P2/P3 evaluation coordinate: prior_read + telemetry)
# --------------------------------------------------------------------------

def write_eval_export(path: str | Path, arrays: Mapping[str, np.ndarray],
                      meta: Mapping[str, Any]) -> None:
    """Write an eval export NPZ: flat arrays plus a JSON metadata field."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    payload = {name: np.asarray(value) for name, value in arrays.items()}
    payload["meta_json"] = np.array(json.dumps(dict(meta), sort_keys=True))
    np.savez_compressed(path, **payload)


@torch.no_grad()
def export_eval(host: str, model: nn.Module, carrier: Carrier,
                val_observed: EpisodeBatch, task_name: str, arm: str, seed: int,
                out_path: Path, device: torch.device,
                chunk: int = EXPORT_CHUNK) -> dict[str, np.ndarray]:
    """Run the trained model over the val episodes and export probe inputs.

    Per episode: prior_read at every t, full carrier telemetry, the encoder
    embedding of o_0 (observed stream; frame 0 precedes every corruption
    window), executed actions, xi, and all event annotations.  Runs in fp32
    eval mode (no autocast) for deterministic probe inputs.
    """
    model.eval()
    carrier.eval()
    episodes, length = val_observed.num_episodes, val_observed.length
    prior_chunks: list[np.ndarray] = []
    z0_chunks: list[np.ndarray] = []
    telemetry_chunks: dict[str, list[np.ndarray]] = {}
    for start in range(0, episodes, chunk):
        stop = min(start + chunk, episodes)
        frames = p0.P0EpisodeDataset._frames_tensor(
            val_observed.frames[start:stop].reshape(-1, p0.IMG_SIZE,
                                                    p0.IMG_SIZE, 3)
        ).reshape(stop - start, length, 3, p0.IMG_SIZE, p0.IMG_SIZE).to(device)
        actions = torch.from_numpy(
            val_observed.actions[start:stop]).to(device)
        z = p0.host_encode(host, model, frames).float()
        output = carrier(z, actions)
        prior_chunks.append(output.prior_read.float().cpu().numpy())
        z0_chunks.append(z[:, 0].cpu().numpy())
        for key, value in output.telemetry.items():
            telemetry_chunks.setdefault(key, []).append(
                value.float().cpu().numpy())

    arrays: dict[str, np.ndarray] = {
        "prior_read": np.concatenate(prior_chunks).astype(np.float32),
        "enc_o0": np.concatenate(z0_chunks).astype(np.float32),
        "actions": val_observed.actions.astype(np.float32),
        "xi": val_observed.xi,
    }
    for name, value in val_observed.events.items():
        arrays[f"event_{name}"] = value
    for key, chunks in telemetry_chunks.items():
        arrays[f"tel_{key}"] = np.concatenate(chunks).astype(np.float32)

    task = make_task(task_name)
    if val_observed.xi_kind == "cont":
        index = np.arange(episodes)
        gap_on = val_observed.events["gap_on"]
        arrays["posterior_mean"] = task.posterior_mean_prediction(
            val_observed).astype(np.float32)
        arrays["frozen_pos"] = task._normalize(
            val_observed.exo_state[index, gap_on - 1, 0:2]).astype(np.float32)

    meta = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "task": task_name,
        "arm": arm,
        "seed": seed,
        "host": host,
        "xi_kind": val_observed.xi_kind,
        "n_classes": val_observed.n_classes,
        "episodes": episodes,
        "length": length,
        "embed_dim": int(arrays["enc_o0"].shape[-1]),
        "export_chunk_episodes": chunk,
        "carrier": carrier.describe(),
        "stream": val_observed.stream,
    }
    write_eval_export(out_path, arrays, meta)
    return arrays


# --------------------------------------------------------------------------
# Annotated telemetry figures
# --------------------------------------------------------------------------

def _shade_events(axis, events: Mapping[str, np.ndarray], episode: int) -> None:
    seen = set()
    for on_key, off_key, color, label in _EVENT_WINDOWS:
        if on_key in events and off_key in events:
            on = int(events[on_key][episode])
            off = int(events[off_key][episode])
            axis.axvspan(on, off, color=color, alpha=0.20,
                         label=None if label in seen else label)
            seen.add(label)


def telemetry_figures(arrays: Mapping[str, np.ndarray],
                      events: Mapping[str, np.ndarray],
                      label: str, episode: int = 0) -> dict[str, Any]:
    """k_t and sigma_t traces of one episode with event windows shaded."""
    from matplotlib.figure import Figure

    figures: dict[str, Any] = {}
    specs = (("k_trace", "tel_k_mean", "tel_k_std", "elementwise gain k_t"),
             ("sigma_trace", "tel_sigma_minus_mean", None,
              "predicted uncertainty sigma-_t"))
    for name, mean_key, std_key, ylabel in specs:
        if mean_key not in arrays:
            continue
        mean = np.asarray(arrays[mean_key][episode], dtype=np.float64)
        steps = np.arange(mean.shape[0])
        figure = Figure(figsize=(7.0, 3.4))
        axis = figure.subplots()
        axis.plot(steps[1:], mean[1:], lw=1.6, color="tab:blue", label=ylabel)
        if std_key is not None and std_key in arrays:
            std = np.asarray(arrays[std_key][episode], dtype=np.float64)
            axis.fill_between(steps[1:], mean[1:] - std[1:], mean[1:] + std[1:],
                              alpha=0.25, color="tab:blue", lw=0)
        _shade_events(axis, events, episode)
        if name == "sigma_trace":
            axis.set_yscale("log")
        axis.set_xlabel("t")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{ylabel} — {label} (episode {episode})")
        axis.legend(fontsize=8, loc="upper right")
        figures[name] = figure
    return figures


# --------------------------------------------------------------------------
# W&B
# --------------------------------------------------------------------------

def _wandb_init(args: argparse.Namespace, run_config: dict, output_dir: Path):
    def _init():
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"p2-{args.arm}-{args.task}-s{args.seed}",
            group=f"p2-{args.task}", tags=["p2", "v19", args.arm, args.host],
            dir=str(output_dir), config=run_config,
            settings=wandb.Settings(init_timeout=180))
        run.define_metric("epoch")
        for namespace in ("train/*", "val/*", "sig/*", "grad/*", "health/*",
                          "carrier/*", "perf/*"):
            run.define_metric(namespace, step_metric="epoch")
        return run
    return p0._guarded("init", _init)


def _log_export_figures(run, figures: dict[str, Any]) -> None:
    import wandb
    from lewm.tasks_v19.wandb_utils import _figure_to_image
    run.log({f"figures/{name}": wandb.Image(_figure_to_image(figure))
             for name, figure in figures.items()})


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=P2_TASKS)
    parser.add_argument("--host", required=True, choices=p0.HOSTS)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", default="outputs/v19_p2")
    parser.add_argument("--p0-data-root", default=p0_data.DEFAULT_ROOT,
                        help="read-only P0 cache root (never written)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--export-chunk", type=int, default=EXPORT_CHUNK)
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="lewm-v19")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    epochs = int(os.environ.get("V19_P0_EPOCHS", args.epochs))
    if epochs < 1 or args.batch_size < 4 or args.num_workers < 0:
        raise ValueError("invalid V19 P2 training configuration")
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    use_amp = not args.no_amp and device.type == "cuda"

    paths, data_origin = resolve_banks(
        args.task, args.p0_data_root, Path(args.output) / "data")
    train_observed = load_bank(paths["train"]["observed"])
    train_clean = (load_bank(paths["train"]["clean"])
                   if args.host == "vicreg" else None)
    val_observed = load_bank(paths["val"]["observed"])
    val_clean = load_bank(paths["val"]["clean"])
    action_dim = int(train_observed.actions.shape[-1])

    train_dataset = p0.P0EpisodeDataset(train_observed, train_clean)
    val_dataset = p0.P0EpisodeDataset(
        val_observed, val_clean if args.host == "vicreg" else None)
    train_loader = _loader(train_dataset, args, train=True, device=device)
    val_loader = _loader(val_dataset, args, train=False, device=device)
    if len(train_dataset) < args.batch_size:
        raise ValueError("train bank smaller than one batch")

    fixed_batch = default_collate(
        [val_dataset[index]
         for index in range(min(args.batch_size, len(val_dataset)))])
    health_count = min(p0.HEALTH_EPISODES, val_clean.num_episodes)
    health_frames = p0.P0EpisodeDataset._frames_tensor(
        val_clean.frames[:health_count].reshape(-1, p0.IMG_SIZE, p0.IMG_SIZE, 3)
    ).reshape(health_count, val_clean.length, 3, p0.IMG_SIZE, p0.IMG_SIZE)

    model = (p0.build_sigreg_host(action_dim) if args.host == "sigreg"
             else p0.build_vicreg_host(action_dim)).to(device)
    embed_dim = int(p0.HOST_CONFIGS[args.host]["embed_dim"])
    carrier = make_carrier(args.arm, embed_dim, action_dim).to(device)
    host_parameters = sum(parameter.numel() for parameter in model.parameters()
                          if parameter.requires_grad)
    carrier_parameters = sum(parameter.numel()
                             for parameter in carrier.parameters()
                             if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in itertools.chain(model.parameters(),
                                                    carrier.parameters())
         if parameter.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output) / args.task / args.arm / f"s{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("gates.json", "history.csv", "checkpoint.pt",
                     "eigenspectra.npy", "eval_export.npz"):
        if (output_dir / filename).exists():
            raise FileExistsError(f"refusing to overwrite {output_dir / filename}")

    run_config = {
        **vars(args), "epochs_effective": epochs, "action_dim": action_dim,
        "host_parameters": host_parameters,
        "carrier_parameters": carrier_parameters,
        "trainable_parameters": host_parameters + carrier_parameters,
        "host_config": p0.HOST_CONFIGS[args.host],
        "carrier_config": carrier.describe(),
        "matched_gru_hidden": matched_gru_hidden(embed_dim, action_dim),
        "matched_ssm_width": matched_ssm_width(embed_dim, action_dim),
        "gates": dict(p0.GATES),
        "objective": (
            "p2_sigreg_single_stream_ztilde_windows_raw_z_targets_k1_sigreg"
            if args.host == "sigreg"
            else "p2_v18_paired_ztilde_windows_clean_targets_vicreg"),
        "aux_loss": "carrier_innovation_nll_unit_weight"
                    if args.arm == "lkc_nll" else None,
        "train_episodes": train_observed.num_episodes,
        "val_episodes": val_observed.num_episodes,
        "corruption_seed": p0_data.CORRUPTION_SEED,
        "data_origin": data_origin,
    }
    run = _wandb_init(args, run_config, output_dir) if args.wandb else None

    print(f"=== v19-p2 {args.host}/{args.arm}/{args.task}/s{args.seed} | "
          f"host={host_parameters:,} carrier={carrier_parameters:,} | "
          f"epochs={epochs} | train={len(train_dataset)} "
          f"val={len(val_dataset)} | amp={use_amp} ===", flush=True)

    rows: list[dict[str, float]] = []
    eigenspectra: list[np.ndarray] = []
    with (output_dir / "history.csv").open("x", newline="") as history_stream:
        writer = csv.DictWriter(history_stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            started = time.time()
            train_metrics = run_epoch(args.host, model, carrier, train_loader,
                                      optimizer, device, use_amp)
            val_metrics = run_epoch(args.host, model, carrier, val_loader,
                                    None, device, use_amp)
            if args.host == "sigreg":
                with torch.no_grad():
                    model.eval()
                    amp_context = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if use_amp else torch.autocast("cpu", enabled=False))
                    with amp_context:
                        fixed_z = model.encode(
                            fixed_batch["observed"].to(device))
                    direction_stats = p0.per_direction_ep_statistics(
                        model.sigreg, fixed_z.float())
                ep_dir = (float(direction_stats.min()),
                          float(direction_stats.median()),
                          float(direction_stats.max()))
            else:
                ep_dir = (float("nan"),) * 3
            grad_ratio_value = gradient_ratio(
                args.host, model, carrier, fixed_batch, device, use_amp)
            variance, rank, eigenvalues = p0.encoder_health(
                args.host, model, health_frames, device, use_amp)
            eigenspectra.append(eigenvalues)
            row = {
                "epoch": epoch,
                "epoch_seconds": time.time() - started,
                **{f"train_{key}": train_metrics[key] for key in LOSS_KEYS},
                **{f"val_{key}": val_metrics[key] for key in LOSS_KEYS},
                "ep_plateau_batch": (
                    p0.analytic_delta_plateau(model.sigreg, args.batch_size)
                    if args.host == "sigreg" else float("nan")),
                "ep_ratio": val_metrics["ep_ratio"],
                "ep_dir_min": ep_dir[0],
                "ep_dir_median": ep_dir[1],
                "ep_dir_max": ep_dir[2],
                "grad_ratio": grad_ratio_value,
                "encoder_mean_channel_variance": variance,
                "encoder_covariance_effective_rank": rank,
                **{f"val_carrier_{key}": val_metrics[f"carrier_{key}"]
                   for key in TELEMETRY_KEYS},
            }
            rows.append(row)
            writer.writerow({key: f"{value:.8g}" if isinstance(value, float)
                             else value for key, value in row.items()})
            history_stream.flush()
            print(
                f"e{epoch:3d}/{epochs} ({row['epoch_seconds']:.1f}s) "
                f"train={train_metrics['loss']:.5f} "
                f"pred={train_metrics['predictive_loss']:.5f} "
                f"nll={train_metrics['carrier_nll']:.5f} | "
                f"val pred={val_metrics['predictive_loss']:.5f} "
                f"rank={rank:.1f} var={variance:.2e} "
                f"k={val_metrics['carrier_k_mean']:.3f} "
                f"grad_ratio={grad_ratio_value:.2f}", flush=True)
            if run is not None:
                payload = {
                    "epoch": epoch,
                    **{f"train/{key}": train_metrics[key] for key in LOSS_KEYS},
                    **{f"val/{key}": val_metrics[key] for key in LOSS_KEYS},
                    "grad/encoder_ratio": grad_ratio_value,
                    "health/mean_channel_variance": variance,
                    "health/covariance_effective_rank": rank,
                    "perf/epoch_seconds": row["epoch_seconds"],
                    **{f"carrier/{key}": val_metrics[f"carrier_{key}"]
                       for key in TELEMETRY_KEYS
                       if math.isfinite(val_metrics[f"carrier_{key}"])},
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
                p0._guarded("epoch_log", run.log, payload, step=epoch)

    gates = p0.compute_gates(rows, args.host)
    gates.update({
        "schema_version": 1,
        "study": "v19-p2-development-grid",
        "task": args.task,
        "arm": args.arm,
        "seed": args.seed,
        "train_episodes": train_observed.num_episodes,
        "val_episodes": val_observed.num_episodes,
        "host_parameters": host_parameters,
        "carrier_parameters": carrier_parameters,
        "mean_epoch_seconds": float(np.mean(
            [row["epoch_seconds"] for row in rows])),
        "final_carrier_k_mean": p0._sanitize(
            rows[-1]["val_carrier_k_mean"]),
        "final_carrier_nll": p0._sanitize(rows[-1]["val_carrier_nll"]),
        "config": {key: value for key, value in run_config.items()},
    })

    spectra = np.stack(eigenspectra)
    np.save(output_dir / "eigenspectra.npy", spectra)
    torch.save({
        "schema_version": 1,
        "model_state_dict": model.state_dict(),
        "carrier_state_dict": carrier.state_dict(),
        "encoder_state_dict": p0.host_encoder(args.host, model).state_dict(),
        "host": args.host,
        "arm": args.arm,
        "task": args.task,
        "seed": args.seed,
        "epochs": epochs,
        "action_dim": action_dim,
        "img_size": p0.IMG_SIZE,
        "host_config": p0.HOST_CONFIGS[args.host],
        "carrier_config": carrier.describe(),
    }, output_dir / "checkpoint.pt")
    p0._write_json_exclusive(output_dir / "gates.json", p0._sanitize(gates))

    export_arrays = export_eval(
        args.host, model, carrier, val_observed, args.task, args.arm,
        args.seed, output_dir / "eval_export.npz", device,
        chunk=args.export_chunk)
    figures = telemetry_figures(
        export_arrays, val_observed.events,
        label=f"{args.arm}/{args.task}/s{args.seed}")
    for name, figure in figures.items():
        figure.savefig(output_dir / f"{name}.png", dpi=130,
                       bbox_inches="tight")

    if run is not None:
        p0._guarded("final_log", p0._log_final_wandb,
                    run, rows, spectra, gates, args)
        if figures:
            p0._guarded("telemetry_figures", _log_export_figures, run, figures)
        p0._guarded("finish", run.finish)

    verdict = "PASS" if gates["overall_pass"] else "FAIL"
    print(f"=== done v19-p2 {args.host}/{args.arm}/{args.task}/s{args.seed}: "
          f"{verdict} rank={gates['final_effective_rank']:.1f} "
          f"conv={gates['convergence_relative_change']:.4f} "
          f"export={output_dir / 'eval_export.npz'} ===", flush=True)


if __name__ == "__main__":
    main()
