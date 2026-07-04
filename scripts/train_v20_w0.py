#!/usr/bin/env python3
"""Train one V20 W0 host-preflight cell (docs/V20_PROPOSAL.md 4.1, claims 1-2).

Arms (host objectives, no memory carriers — the P0 question re-asked):

- ``visreg60|visreg75|visreg90``: the exact-LeWM architecture from V19 P0's
  ``sigreg`` bet (D=192, 12 layers, MLP-BN projection head, single observed
  stream, sliding H=3 teacher forcing) with the SIGReg term replaced by the
  exact published VisReg objective (lewm/models/visreg.py) at outer lambda
  0.60 / 0.75 / 0.90 — the one registered host knob, swept once on t1 and
  frozen.  The V16 Epps-Pulley instruments stay on as *descriptive* telemetry
  (does VisReg move the EP statistic off the projected-zero plateau?); they do
  not gate.
- ``vicreg``: the V18 reference verbatim via scripts/train_v19_p0.py — run
  here only on the salience-ladder tasks (t1/t3/t4 vicreg cells are frozen
  P0-a2 evidence and are cited, not re-run).

Tasks: t1/t3/t4 (frozen amendment-2 banks, read-only from the P0-a2 root)
plus the W0 salience ladder t1s1/t1s2/t1s3 (generated under this run's
output root with the identical P0 corruption recipe).

Gates: the registered P0 health set verbatim (rank >= 16, channel variance
>= 1e-4, late-window convergence <= 5%) via scripts.train_v19_p0.compute_gates;
the sigreg-only plateau clause does not apply to either arm here.

Outputs per cell: gates.json, history.csv, encoder.pt (certify input),
eigenspectra.npy — the P0 layout, under <output>/<task>/<arm>/s<seed>.
Env overrides for smoke: V19_P0_EPOCHS, V19_P0_EPISODES (P0 conventions).
"""

from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.visreg import VisRegObjective
from lewm.tasks_v19 import make_task
from lewm.tasks_v19.base import load_bank, save_bank
import scripts.make_v19_p0_data as p0_data
import scripts.train_lewm_v8_v18 as v18
import scripts.train_v19_p0 as p0

ARMS = ("visreg60", "visreg75", "visreg90", "vicreg")
W0_TASKS = ("t1", "t3", "t4", "t1s1", "t1s2", "t1s3")
LADDER = ("t1s1", "t1s2", "t1s3", "t1")     # ascending registered salience
VISREG_SLICES = 4096
DEFAULT_EPOCHS = p0.DEFAULT_EPOCHS

LOSS_KEYS = ("loss", "predictive_loss", "regularizer_loss",
             "visreg_scale", "visreg_shape", "visreg_center",
             "variance_loss", "covariance_loss")
HISTORY_FIELDS = (
    "epoch", "epoch_seconds",
    *(f"train_{key}" for key in LOSS_KEYS),
    *(f"val_{key}" for key in LOSS_KEYS),
    "ep_ratio", "ep_dir_min", "ep_dir_median", "ep_dir_max",
    "grad_ratio",
    "encoder_mean_channel_variance", "encoder_covariance_effective_rank",
)


def arm_lambda(arm: str) -> float | None:
    """The single registered host knob; None for the vicreg reference."""
    if arm == "vicreg":
        return None
    if arm not in ARMS:
        raise ValueError(f"unknown W0 arm {arm!r}")
    return int(arm.removeprefix("visreg")) / 100.0


def host_kind(arm: str) -> str:
    return "vicreg" if arm == "vicreg" else "visreg"


# --------------------------------------------------------------------------
# Data: P0-a2 banks read-only; ladder banks generated under the W0 root
# --------------------------------------------------------------------------

def resolve_banks(task: str, p0_root: str | Path, w0_root: str | Path
                  ) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    """The P2 resolve_banks contract for the W0 task set."""
    if task not in W0_TASKS:
        raise ValueError(f"task must be one of {W0_TASKS}, got {task!r}")
    train_episodes, val_episodes = p0_data.episode_sizes()
    for root, writable in ((Path(p0_root), False), (Path(w0_root), True)):
        paths = p0_data.task_bank_paths(root, task, train_episodes, val_episodes)
        if all(p0_data._cache_valid(path)
               for split in paths.values() for path in split.values()):
            return paths, {"root": str(root), "generated": False,
                           "writable": writable}
    w0_root = Path(w0_root)
    if w0_root.resolve() == Path(p0_root).resolve():
        raise ValueError("refusing to generate banks inside the P0 data root")
    paths = p0_data.task_bank_paths(w0_root, task, train_episodes, val_episodes)
    for split, (episodes, seed) in (("train", (train_episodes, p0_data.TRAIN_SEED)),
                                    ("val", (val_episodes, p0_data.VAL_SEED))):
        started = time.time()
        print(f"[v20-w0-data] {task}/{split}: generating {episodes} episodes "
              f"(stream={p0_data.STREAM}, seed={seed})", flush=True)
        clean = make_task(task).generate(p0_data.STREAM, episodes, seed)
        observed = p0_data.corrupt_bank(clean, p0_data.CORRUPTION_SEED)
        save_bank(clean, paths[split]["clean"])
        save_bank(observed, paths[split]["observed"])
        print(f"[v20-w0-data] {task}/{split}: wrote banks "
              f"({time.time() - started:.1f}s)", flush=True)
    return paths, {"root": str(w0_root), "generated": True, "writable": True}


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def visreg_losses(world, visreg: VisRegObjective, lam: float,
                  observed: torch.Tensor, actions: torch.Tensor
                  ) -> dict[str, torch.Tensor]:
    """Exact-LeWM single-stream objective with the VisReg regularizer.

    ``L = L_pred + lambda * (scale + shape + center)`` over the same one-pass
    embeddings — the P0 sigreg objective with the anti-collapse term swapped
    and nothing else changed.
    """
    z = world.encode(observed)                                    # (B, L, D)
    latent_windows, action_windows, targets = v18.sliding_predictor_windows(
        z, actions, z, history=world.history_len)
    prediction = world.predictor(latent_windows, action_windows)[:, -1]
    predictive_loss = F.mse_loss(prediction.float(), targets.float())
    components = visreg(z)
    regularizer_loss = lam * components["total"]
    zero = predictive_loss.new_zeros(())
    return {
        "loss": predictive_loss + regularizer_loss,
        "predictive_loss": predictive_loss,
        "regularizer_loss": regularizer_loss,
        "visreg_scale": components["scale"],
        "visreg_shape": components["shape"],
        "visreg_center": components["center"],
        "variance_loss": zero,
        "covariance_loss": zero,
    }


def compute_w0_losses(arm: str, model: nn.Module,
                      visreg: VisRegObjective | None,
                      batch: Mapping[str, torch.Tensor],
                      device: torch.device) -> dict[str, torch.Tensor]:
    observed = batch["observed"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    if arm == "vicreg":
        clean = batch["clean"].to(device, non_blocking=True)
        losses = v18.compute_losses(model, observed, clean, actions,
                                    sigreg_lambda=0.0)
        zero = losses["predictive_loss"].new_zeros(())
        losses.update({"visreg_scale": zero, "visreg_shape": zero,
                       "visreg_center": zero})
        return losses
    return visreg_losses(model, visreg, arm_lambda(arm), observed, actions)


def gradient_ratio(arm: str, model: nn.Module, visreg: VisRegObjective | None,
                   batch: Mapping[str, torch.Tensor], device: torch.device,
                   use_amp: bool) -> float:
    """P0's ||grad(weighted reg)|| / ||grad(L_pred)|| encoder instrument."""
    kind = host_kind(arm)
    was_training = model.training
    model.eval()
    parameters = [parameter
                  for parameter in p0.host_encoder(
                      "sigreg" if kind == "visreg" else "vicreg",
                      model).parameters()
                  if parameter.requires_grad]
    try:
        with torch.enable_grad():
            amp_context = (torch.autocast("cuda", dtype=torch.bfloat16)
                           if use_amp else torch.autocast("cpu", enabled=False))
            with amp_context:
                losses = compute_w0_losses(arm, model, visreg, batch, device)
            regularizer_grads = torch.autograd.grad(
                losses["regularizer_loss"], parameters,
                retain_graph=True, allow_unused=True)
            prediction_grads = torch.autograd.grad(
                losses["predictive_loss"], parameters, allow_unused=True)
    finally:
        model.train(was_training)

    def _norm(grads: tuple[torch.Tensor | None, ...]) -> float:
        squares = [grad.float().square().sum()
                   for grad in grads if grad is not None]
        return float(torch.stack(squares).sum().sqrt()) if squares else 0.0

    return _norm(regularizer_grads) / max(_norm(prediction_grads), 1e-12)


def run_epoch(arm: str, model: nn.Module, visreg: VisRegObjective | None,
              loader, optimizer, device: torch.device, use_amp: bool
              ) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {key: 0.0 for key in LOSS_KEYS}
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            if train:
                optimizer.zero_grad(set_to_none=True)
            amp_context = (torch.autocast("cuda", dtype=torch.bfloat16)
                           if use_amp else torch.autocast("cpu", enabled=False))
            with amp_context:
                losses = compute_w0_losses(arm, model, visreg, batch, device)
            if train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = batch["observed"].shape[0]
            for key in LOSS_KEYS:
                totals[key] += float(losses[key].detach()) * batch_size
            count += batch_size
    if not count:
        raise RuntimeError("empty V20 W0 epoch")
    return {key: value / count for key, value in totals.items()}


# --------------------------------------------------------------------------
# W&B
# --------------------------------------------------------------------------

def _log_final_wandb(run, rows: list[dict[str, float]],
                     eigenspectra: np.ndarray, gates: dict,
                     args: argparse.Namespace) -> None:
    """P0's final-figure panel with W0 labels (args has .arm, not .host)."""
    import wandb
    from lewm.tasks_v19.wandb_utils import _figure_to_image

    label = f"{args.arm}/{args.task}/s{args.seed}"
    figures = {
        "figures/effective_rank": p0._line_figure(
            [row["encoder_covariance_effective_rank"] for row in rows],
            f"covariance effective rank — {label}", "effective rank",
            threshold=p0.GATES["min_effective_rank"]),
        "figures/channel_variance": p0._line_figure(
            [row["encoder_mean_channel_variance"] for row in rows],
            f"mean channel variance — {label}", "variance",
            threshold=p0.GATES["min_channel_variance"], log_scale=True),
        "figures/grad_ratio": p0._line_figure(
            [row["grad_ratio"] for row in rows],
            f"encoder grad ratio ||∇reg||/||∇pred|| — {label}", "ratio",
            threshold=p0.GATES["grad_ratio_threshold"], log_scale=True),
        "figures/eigenspectrum": p0._heatmap_figure(
            eigenspectra, f"embedding covariance eigenspectrum — {label}"),
    }
    if host_kind(args.arm) == "visreg":
        figures["figures/ep_ratio"] = p0._line_figure(
            [row["ep_ratio"] for row in rows],
            f"descriptive EP / projected-zero plateau — {label}", "EP ratio",
            threshold=1.0)
        for component in ("scale", "shape", "center"):
            figures[f"figures/visreg_{component}"] = p0._line_figure(
                [row[f"val_visreg_{component}"] for row in rows],
                f"VisReg {component} — {label}", component, log_scale=True)
    run.log({key: wandb.Image(_figure_to_image(figure))
             for key, figure in figures.items()})
    run.summary.update({f"gates/{key}": value for key, value in gates.items()
                        if isinstance(value, (bool, int, float, str))})


def _wandb_init(args: argparse.Namespace, run_config: dict, output_dir: Path):
    def _init():
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"w0-{args.arm}-{args.task}-s{args.seed}",
            group=f"w0-{args.task}", tags=["w0", "v20", args.arm],
            dir=str(output_dir), config=run_config,
            settings=wandb.Settings(init_timeout=180))
        run.define_metric("epoch")
        for namespace in ("train/*", "val/*", "vis/*", "sig/*", "grad/*",
                          "health/*", "perf/*"):
            run.define_metric(namespace, step_metric="epoch")
        return run
    return p0._guarded("init", _init)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=W0_TASKS)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", default="outputs/v20_w0")
    parser.add_argument("--p0-data-root", default="outputs/v19_p0_a2/data",
                        help="read-only P0-a2 cache root (never written)")
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
    parser.add_argument("--wandb-project", default="lewm-v20")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    epochs = int(os.environ.get("V19_P0_EPOCHS", args.epochs))
    if epochs < 1 or args.batch_size < 4 or args.num_workers < 0:
        raise ValueError("invalid V20 W0 training configuration")
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    use_amp = not args.no_amp and device.type == "cuda"
    kind = host_kind(args.arm)

    paths, data_origin = resolve_banks(
        args.task, args.p0_data_root, Path(args.output) / "data")
    train_observed = load_bank(paths["train"]["observed"])
    train_clean = (load_bank(paths["train"]["clean"])
                   if kind == "vicreg" else None)
    val_observed = load_bank(paths["val"]["observed"])
    val_clean = load_bank(paths["val"]["clean"])
    action_dim = int(train_observed.actions.shape[-1])

    train_dataset = p0.P0EpisodeDataset(train_observed, train_clean)
    val_dataset = p0.P0EpisodeDataset(
        val_observed, val_clean if kind == "vicreg" else None)
    train_loader = p0._loader(train_dataset, args, train=True)
    val_loader = p0._loader(val_dataset, args, train=False)
    if len(train_dataset) < args.batch_size:
        raise ValueError("train bank smaller than one batch")

    fixed_batch = default_collate(
        [val_dataset[index]
         for index in range(min(args.batch_size, len(val_dataset)))])
    health_count = min(p0.HEALTH_EPISODES, val_clean.num_episodes)
    health_frames = p0.P0EpisodeDataset._frames_tensor(
        val_clean.frames[:health_count].reshape(-1, p0.IMG_SIZE, p0.IMG_SIZE, 3)
    ).reshape(health_count, val_clean.length, 3, p0.IMG_SIZE, p0.IMG_SIZE)

    # visreg arms: the exact-LeWM architecture (P0's sigreg builder verbatim);
    # model.sigreg stays attached purely for the descriptive EP telemetry.
    model = (p0.build_vicreg_host(action_dim) if kind == "vicreg"
             else p0.build_sigreg_host(action_dim)).to(device)
    visreg = (VisRegObjective(VISREG_SLICES).to(device)
              if kind == "visreg" else None)
    encode_host = "vicreg" if kind == "vicreg" else "sigreg"
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output) / args.task / args.arm / f"s{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("gates.json", "history.csv", "encoder.pt",
                     "eigenspectra.npy"):
        if (output_dir / filename).exists():
            raise FileExistsError(f"refusing to overwrite {output_dir / filename}")

    run_config = {
        **vars(args), "epochs_effective": epochs, "action_dim": action_dim,
        "trainable_parameters": trainable,
        "host_kind": kind,
        "visreg_lambda": arm_lambda(args.arm),
        "visreg_config": visreg.describe() if visreg is not None else None,
        "host_config": p0.HOST_CONFIGS[
            "sigreg" if kind == "visreg" else "vicreg"],
        "gates": dict(p0.GATES),
        "objective": (
            "w0_lewm_exact_single_stream_sliding_h3_plus_visreg"
            if kind == "visreg" else v18.OBJECTIVE),
        "train_episodes": train_observed.num_episodes,
        "val_episodes": val_observed.num_episodes,
        "corruption_seed": p0_data.CORRUPTION_SEED,
        "data_origin": data_origin,
    }
    run = _wandb_init(args, run_config, output_dir) if args.wandb else None

    print(f"=== v20-w0 {args.arm}/{args.task}/s{args.seed} | "
          f"params={trainable:,} | epochs={epochs} | "
          f"train={len(train_dataset)} val={len(val_dataset)} | amp={use_amp} ===",
          flush=True)

    rows: list[dict[str, float]] = []
    eigenspectra: list[np.ndarray] = []
    with (output_dir / "history.csv").open("x", newline="") as history_stream:
        writer = csv.DictWriter(history_stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            started = time.time()
            train_metrics = run_epoch(args.arm, model, visreg, train_loader,
                                      optimizer, device, use_amp)
            val_metrics = run_epoch(args.arm, model, visreg, val_loader,
                                    None, device, use_amp)
            if kind == "visreg":
                # Descriptive EP instrument: fresh-sketch per-direction
                # statistics of the unused SIGReg head on the fixed batch —
                # does VisReg move the code off the projected-zero plateau?
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
                    plateau = p0.analytic_delta_plateau(
                        model.sigreg, fixed_z.shape[0])
                    raw_ep = float(model.sigreg(fixed_z.float()))
                ep_dir = (float(direction_stats.min()),
                          float(direction_stats.median()),
                          float(direction_stats.max()))
                ep_ratio = raw_ep / plateau
            else:
                ep_dir = (float("nan"),) * 3
                ep_ratio = float("nan")
            grad_ratio_value = gradient_ratio(
                args.arm, model, visreg, fixed_batch, device, use_amp)
            variance, rank, eigenvalues = p0.encoder_health(
                encode_host, model, health_frames, device, use_amp)
            eigenspectra.append(eigenvalues)
            row = {
                "epoch": epoch,
                "epoch_seconds": time.time() - started,
                **{f"train_{key}": train_metrics[key] for key in LOSS_KEYS},
                **{f"val_{key}": val_metrics[key] for key in LOSS_KEYS},
                "ep_ratio": ep_ratio,
                "ep_dir_min": ep_dir[0],
                "ep_dir_median": ep_dir[1],
                "ep_dir_max": ep_dir[2],
                "grad_ratio": grad_ratio_value,
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
                f"ep_ratio={ep_ratio:.3f} grad_ratio={grad_ratio_value:.2f}",
                flush=True)
            if run is not None:
                payload = {
                    "epoch": epoch,
                    **{f"train/{key}": train_metrics[key] for key in LOSS_KEYS},
                    **{f"val/{key}": val_metrics[key] for key in LOSS_KEYS},
                    "grad/encoder_ratio": grad_ratio_value,
                    "health/mean_channel_variance": variance,
                    "health/covariance_effective_rank": rank,
                    "perf/epoch_seconds": row["epoch_seconds"],
                }
                if kind == "visreg":
                    payload.update({
                        "vis/scale": val_metrics["visreg_scale"],
                        "vis/shape": val_metrics["visreg_shape"],
                        "vis/center": val_metrics["visreg_center"],
                        "sig/ep_ratio": ep_ratio,
                        "sig/dir_min": ep_dir[0],
                        "sig/dir_median": ep_dir[1],
                        "sig/dir_max": ep_dir[2],
                    })
                p0._guarded("epoch_log", run.log, payload, step=epoch)

    # The registered P0 gate math verbatim; the plateau clause is
    # sigreg-specific, so both W0 arms gate as the reference host does.
    gates = p0.compute_gates(rows, "vicreg")
    gates.update({
        "schema_version": 1,
        "study": "v20-w0-host-preflight",
        "host": args.arm,
        "task": args.task,
        "seed": args.seed,
        "visreg_lambda": arm_lambda(args.arm),
        "train_episodes": train_observed.num_episodes,
        "val_episodes": val_observed.num_episodes,
        "trainable_parameters": trainable,
        "mean_epoch_seconds": float(np.mean(
            [row["epoch_seconds"] for row in rows])),
        "final_ep_ratio": p0._sanitize(rows[-1]["ep_ratio"]),
        "final_grad_ratio": p0._sanitize(rows[-1]["grad_ratio"]),
        "config": {key: value for key, value in run_config.items()},
    })

    spectra = np.stack(eigenspectra)
    np.save(output_dir / "eigenspectra.npy", spectra)
    torch.save({
        "schema_version": 1,
        "encoder_state_dict": p0.host_encoder(encode_host, model).state_dict(),
        "host": args.arm,
        "host_kind": kind,
        "task": args.task,
        "seed": args.seed,
        "epochs": epochs,
        "action_dim": action_dim,
        "img_size": p0.IMG_SIZE,
        "visreg_lambda": arm_lambda(args.arm),
        "host_config": p0.HOST_CONFIGS[
            "sigreg" if kind == "visreg" else "vicreg"],
        "sigreg_head": "linear_bn_gelu_linear" if kind == "visreg" else None,
        "gates": {key: value for key, value in gates.items()
                  if isinstance(value, (bool, int, float, str))},
    }, output_dir / "encoder.pt")
    p0._write_json_exclusive(output_dir / "gates.json", p0._sanitize(gates))

    if run is not None:
        p0._guarded("final_log", _log_final_wandb,
                    run, rows, spectra, gates, args)
        p0._guarded("finish", run.finish)

    verdict = "PASS" if gates["overall_pass"] else "FAIL"
    print(f"=== done v20-w0 {args.arm}/{args.task}/s{args.seed}: {verdict} "
          f"rank={gates['final_effective_rank']:.1f} "
          f"var={gates['final_channel_variance']:.2e} "
          f"conv={gates['convergence_relative_change']:.4f} "
          f"-> {output_dir} ===", flush=True)


if __name__ == "__main__":
    main()
