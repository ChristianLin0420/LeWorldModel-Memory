#!/usr/bin/env python3
"""Train one V20 W1 development-grid cell (docs/V20_PROPOSAL.md 4.3/6).

W1 trains the three arms that need training — the DFC family shares one
checkpoint (training is exactly the V19 fixed-trust recipe; the slow filter
is deployment-only, applied by scripts/eval_v20_w1.py):

    lkc_rfix   fixed-trust LKC (the V19 P3 winner's recipe) — the shared
               checkpoint for the dfc / dfc_etafix / lkc_rfix eval variants
    acgru      the action-conditioned recurrent envelope reference
    none       the no-carrier floor

on the W0-selected host:

    --host vicreg      V18 reference verbatim (scripts/train_v19_p2.py paths)
    --host visregXX    exact-LeWM architecture + published VisReg at
                       lambda = XX/100 (scripts/train_v20_w0.py objective),
                       carrier between encoder and predictor, single stream,
                       raw-z prediction targets and raw-z regularizer — the
                       P2-sigreg wiring with the anti-collapse term swapped.

Everything else mirrors the V19 P2 trainer: P0 health gates, carrier
telemetry, checkpoint + eval_export.npz in the P2 layout.  Data: t1dev/t3dev
banks are reused read-only from the V19 P2 cache; missing banks are generated
under <output>/data with the identical recipe.  Env overrides for smoke:
V19_P0_EPOCHS, V19_P0_EPISODES.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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

from lewm.models.v19_carriers import Carrier, CarrierOutput, make_carrier
from lewm.models.visreg import VisRegObjective
from lewm.tasks_v19.base import load_bank
import scripts.make_v19_p0_data as p0_data
import scripts.train_lewm_v8_v18 as v18
import scripts.train_v19_p0 as p0
import scripts.train_v19_p2 as p2
import scripts.train_v20_w0 as w0

ARMS = ("none", "acgru", "lkc_rfix")
W1_TASKS = ("t1dev", "t3dev", "t1", "t3", "t4")   # dev now; t1/t3/t4 for W3
HOSTS = ("vicreg", "visreg60", "visreg75", "visreg90")
DEFAULT_EPOCHS = p0.DEFAULT_EPOCHS

LOSS_KEYS = ("loss", "predictive_loss", "regularizer_loss",
             "visreg_scale", "visreg_shape", "visreg_center",
             "variance_loss", "covariance_loss")
TELEMETRY_KEYS = p2.TELEMETRY_KEYS
HISTORY_FIELDS = (
    "epoch", "epoch_seconds",
    *(f"train_{key}" for key in LOSS_KEYS),
    *(f"val_{key}" for key in LOSS_KEYS),
    "grad_ratio",
    "encoder_mean_channel_variance", "encoder_covariance_effective_rank",
    *(f"val_carrier_{key}" for key in TELEMETRY_KEYS),
)


def encode_host(host: str) -> str:
    """The p0 encode/encoder path a W1 host uses."""
    return "vicreg" if host == "vicreg" else "sigreg"


def build_host(host: str, action_dim: int) -> nn.Module:
    if host == "vicreg":
        return p0.build_vicreg_host(action_dim)
    return p0.build_sigreg_host(action_dim)


def resolve_banks(task: str, p2_data_root: str | Path, w1_root: str | Path
                  ) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    """Dev banks read-only from the V19 P2 cache; W3 tasks from the P0-a2
    cache; anything missing is generated under the W1 root."""
    if task not in W1_TASKS:
        raise ValueError(f"task must be one of {W1_TASKS}, got {task!r}")
    train_episodes, val_episodes = p0_data.episode_sizes()
    candidates = ((Path(p2_data_root), False),
                  (Path("outputs/v19_p0_a2/data"), False),
                  (Path(w1_root), True))
    for root, writable in candidates:
        paths = p0_data.task_bank_paths(root, task, train_episodes, val_episodes)
        if all(p0_data._cache_valid(path)
               for split in paths.values() for path in split.values()):
            return paths, {"root": str(root), "generated": False,
                           "writable": writable}
    return _generate(task, w1_root)


def _generate(task: str, w1_root: str | Path
              ) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    from lewm.tasks_v19 import make_task
    from lewm.tasks_v19.base import save_bank
    w1_root = Path(w1_root)
    train_episodes, val_episodes = p0_data.episode_sizes()
    paths = p0_data.task_bank_paths(w1_root, task, train_episodes, val_episodes)
    for split, (episodes, seed) in (("train", (train_episodes, p0_data.TRAIN_SEED)),
                                    ("val", (val_episodes, p0_data.VAL_SEED))):
        started = time.time()
        print(f"[v20-w1-data] {task}/{split}: generating {episodes} episodes",
              flush=True)
        clean = make_task(task).generate(p0_data.STREAM, episodes, seed)
        observed = p0_data.corrupt_bank(clean, p0_data.CORRUPTION_SEED)
        save_bank(clean, paths[split]["clean"])
        save_bank(observed, paths[split]["observed"])
        print(f"[v20-w1-data] {task}/{split}: wrote banks "
              f"({time.time() - started:.1f}s)", flush=True)
    return paths, {"root": str(w1_root), "generated": True, "writable": True}


def compute_w1_losses(host: str, model: nn.Module,
                      visreg: VisRegObjective | None, lam: float | None,
                      carrier: Carrier, batch: Mapping[str, torch.Tensor],
                      device: torch.device
                      ) -> tuple[dict[str, torch.Tensor], CarrierOutput]:
    """Host loss with the carrier between encoder and predictor.

    vicreg: the P2 vicreg objective verbatim.  visreg: the P2-sigreg wiring
    (z_tilde windows, raw-z targets) with lambda * VisReg(Z) replacing SIGReg.
    """
    if host == "vicreg":
        losses, output = p2.compute_p2_losses(
            "vicreg", model, carrier, batch, device)
        zero = losses["predictive_loss"].new_zeros(())
        losses = {**losses, "visreg_scale": zero, "visreg_shape": zero,
                  "visreg_center": zero}
        losses.pop("sigreg_loss", None)
        losses.pop("carrier_nll", None)
        losses["loss"] = losses["predictive_loss"] + losses["regularizer_loss"]
        return losses, output
    observed = batch["observed"].to(device, non_blocking=True)
    actions = batch["actions"].to(device, non_blocking=True)
    z = model.encode(observed)
    output = carrier(z, actions)
    latent_windows, action_windows, targets = v18.sliding_predictor_windows(
        output.z_tilde, actions, z, history=model.history_len)
    prediction = model.predictor(latent_windows, action_windows)[:, -1]
    predictive_loss = F.mse_loss(prediction.float(), targets.float())
    components = visreg(z)
    regularizer_loss = lam * components["total"]
    zero = predictive_loss.new_zeros(())
    losses = {
        "loss": predictive_loss + regularizer_loss,
        "predictive_loss": predictive_loss,
        "regularizer_loss": regularizer_loss,
        "visreg_scale": components["scale"],
        "visreg_shape": components["shape"],
        "visreg_center": components["center"],
        "variance_loss": zero,
        "covariance_loss": zero,
    }
    return losses, output


def run_epoch(host: str, model: nn.Module, visreg: VisRegObjective | None,
              lam: float | None, carrier: Carrier, loader, optimizer,
              device: torch.device, use_amp: bool) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    carrier.train(train)
    totals = {key: 0.0 for key in LOSS_KEYS}
    telemetry_totals = {key: 0.0 for key in TELEMETRY_KEYS}
    telemetry_counts = {key: 0 for key in TELEMETRY_KEYS}
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            if train:
                optimizer.zero_grad(set_to_none=True)
            amp_context = (torch.autocast("cuda", dtype=torch.bfloat16)
                           if use_amp else torch.autocast("cpu", enabled=False))
            with amp_context:
                losses, output = compute_w1_losses(
                    host, model, visreg, lam, carrier, batch, device)
            if train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(model.parameters(), carrier.parameters()),
                    1.0)
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
    if not count:
        raise RuntimeError("empty V20 W1 epoch")
    metrics = {key: value / count for key, value in totals.items()}
    for key in TELEMETRY_KEYS:
        metrics[f"carrier_{key}"] = (
            telemetry_totals[key] / telemetry_counts[key]
            if telemetry_counts[key] else float("nan"))
    return metrics


def gradient_ratio(host: str, model: nn.Module,
                   visreg: VisRegObjective | None, lam: float | None,
                   carrier: Carrier, batch: Mapping[str, torch.Tensor],
                   device: torch.device, use_amp: bool) -> float:
    was_training = model.training
    model.eval()
    carrier.eval()
    parameters = [parameter
                  for parameter in p0.host_encoder(
                      encode_host(host), model).parameters()
                  if parameter.requires_grad]
    try:
        with torch.enable_grad():
            amp_context = (torch.autocast("cuda", dtype=torch.bfloat16)
                           if use_amp else torch.autocast("cpu", enabled=False))
            with amp_context:
                losses, _ = compute_w1_losses(
                    host, model, visreg, lam, carrier, batch, device)
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


def _wandb_init(args: argparse.Namespace, run_config: dict, output_dir: Path):
    def _init():
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"w1-{args.arm}-{args.task}-s{args.seed}",
            group=f"w1-{args.task}", tags=["w1", "v20", args.arm, args.host],
            dir=str(output_dir), config=run_config,
            settings=wandb.Settings(init_timeout=180))
        run.define_metric("epoch")
        for namespace in ("train/*", "val/*", "vis/*", "grad/*", "health/*",
                          "carrier/*", "perf/*"):
            run.define_metric(namespace, step_metric="epoch")
        return run
    return p0._guarded("init", _init)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=W1_TASKS)
    parser.add_argument("--host", required=True, choices=HOSTS)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", default="outputs/v20_w1")
    parser.add_argument("--p2-data-root", default="outputs/v19_p2/data",
                        help="read-only V19 P2 dev-bank root (never written)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--export-chunk", type=int, default=p2.EXPORT_CHUNK)
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="lewm-v20")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    epochs = int(os.environ.get("V19_P0_EPOCHS", args.epochs))
    if epochs < 1 or args.batch_size < 4 or args.num_workers < 0:
        raise ValueError("invalid V20 W1 training configuration")
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available()
        else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    use_amp = not args.no_amp and device.type == "cuda"
    lam = w0.arm_lambda(args.host) if args.host != "vicreg" else None

    paths, data_origin = resolve_banks(
        args.task, args.p2_data_root, Path(args.output) / "data")
    train_observed = load_bank(paths["train"]["observed"])
    train_clean = (load_bank(paths["train"]["clean"])
                   if args.host == "vicreg" else None)
    val_observed = load_bank(paths["val"]["observed"])
    val_clean = load_bank(paths["val"]["clean"])
    action_dim = int(train_observed.actions.shape[-1])

    train_dataset = p0.P0EpisodeDataset(train_observed, train_clean)
    val_dataset = p0.P0EpisodeDataset(
        val_observed, val_clean if args.host == "vicreg" else None)
    train_loader = p2._loader(train_dataset, args, train=True, device=device)
    val_loader = p2._loader(val_dataset, args, train=False, device=device)
    if len(train_dataset) < args.batch_size:
        raise ValueError("train bank smaller than one batch")

    fixed_batch = default_collate(
        [val_dataset[index]
         for index in range(min(args.batch_size, len(val_dataset)))])
    health_count = min(p0.HEALTH_EPISODES, val_clean.num_episodes)
    health_frames = p0.P0EpisodeDataset._frames_tensor(
        val_clean.frames[:health_count].reshape(-1, p0.IMG_SIZE, p0.IMG_SIZE, 3)
    ).reshape(health_count, val_clean.length, 3, p0.IMG_SIZE, p0.IMG_SIZE)

    model = build_host(args.host, action_dim).to(device)
    visreg = (VisRegObjective(w0.VISREG_SLICES).to(device)
              if args.host != "vicreg" else None)
    embed_dim = int(p0.HOST_CONFIGS[
        "vicreg" if args.host == "vicreg" else "sigreg"]["embed_dim"])
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
        "visreg_lambda": lam,
        "host_config": p0.HOST_CONFIGS[
            "vicreg" if args.host == "vicreg" else "sigreg"],
        "carrier_config": carrier.describe(),
        "gates": dict(p0.GATES),
        "objective": (
            "w1_v18_paired_ztilde_windows_clean_targets_vicreg"
            if args.host == "vicreg"
            else "w1_lewm_single_stream_ztilde_windows_raw_z_targets_visreg"),
        "train_episodes": train_observed.num_episodes,
        "val_episodes": val_observed.num_episodes,
        "corruption_seed": p0_data.CORRUPTION_SEED,
        "data_origin": data_origin,
    }
    run = _wandb_init(args, run_config, output_dir) if args.wandb else None

    print(f"=== v20-w1 {args.host}/{args.arm}/{args.task}/s{args.seed} | "
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
            train_metrics = run_epoch(args.host, model, visreg, lam, carrier,
                                      train_loader, optimizer, device, use_amp)
            val_metrics = run_epoch(args.host, model, visreg, lam, carrier,
                                    val_loader, None, device, use_amp)
            grad_ratio_value = gradient_ratio(
                args.host, model, visreg, lam, carrier, fixed_batch, device,
                use_amp)
            variance, rank, eigenvalues = p0.encoder_health(
                encode_host(args.host), model, health_frames, device, use_amp)
            eigenspectra.append(eigenvalues)
            row = {
                "epoch": epoch,
                "epoch_seconds": time.time() - started,
                **{f"train_{key}": train_metrics[key] for key in LOSS_KEYS},
                **{f"val_{key}": val_metrics[key] for key in LOSS_KEYS},
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
                f"pred={train_metrics['predictive_loss']:.5f} | "
                f"val pred={val_metrics['predictive_loss']:.5f} "
                f"rank={rank:.1f} k={val_metrics['carrier_k_mean']:.3f} "
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
                p0._guarded("epoch_log", run.log, payload, step=epoch)

    # ep_ratio is not tracked in W1 (no SIGReg head in the loss path);
    # compute_gates only reads it via plateau logic, which the vicreg gate
    # variant skips.
    for row in rows:
        row.setdefault("ep_ratio", float("nan"))
    gates = p0.compute_gates(rows, "vicreg")
    gates.update({
        "schema_version": 1,
        "study": "v20-w1-development-grid",
        "host": args.host,
        "task": args.task,
        "arm": args.arm,
        "seed": args.seed,
        "visreg_lambda": lam,
        "train_episodes": train_observed.num_episodes,
        "val_episodes": val_observed.num_episodes,
        "host_parameters": host_parameters,
        "carrier_parameters": carrier_parameters,
        "mean_epoch_seconds": float(np.mean(
            [row["epoch_seconds"] for row in rows])),
        "final_carrier_k_mean": p0._sanitize(rows[-1]["val_carrier_k_mean"]),
        "config": {key: value for key, value in run_config.items()},
    })

    spectra = np.stack(eigenspectra)
    np.save(output_dir / "eigenspectra.npy", spectra)
    torch.save({
        "schema_version": 1,
        "model_state_dict": model.state_dict(),
        "carrier_state_dict": carrier.state_dict(),
        "encoder_state_dict": p0.host_encoder(
            encode_host(args.host), model).state_dict(),
        "host": args.host,
        "arm": args.arm,
        "task": args.task,
        "seed": args.seed,
        "epochs": epochs,
        "action_dim": action_dim,
        "img_size": p0.IMG_SIZE,
        "visreg_lambda": lam,
        "host_config": p0.HOST_CONFIGS[
            "vicreg" if args.host == "vicreg" else "sigreg"],
        "carrier_config": carrier.describe(),
    }, output_dir / "checkpoint.pt")
    p0._write_json_exclusive(output_dir / "gates.json", p0._sanitize(gates))

    export_arrays = p2.export_eval(
        encode_host(args.host), model, carrier, val_observed, args.task,
        args.arm, args.seed, output_dir / "eval_export.npz", device,
        chunk=args.export_chunk)
    figures = p2.telemetry_figures(
        export_arrays, val_observed.events,
        label=f"{args.arm}/{args.task}/s{args.seed}")
    for name, figure in figures.items():
        figure.savefig(output_dir / f"{name}.png", dpi=130,
                       bbox_inches="tight")

    if run is not None:
        if figures:
            p0._guarded("telemetry_figures", p2._log_export_figures,
                        run, figures)
        p0._guarded("finish", run.finish)

    verdict = "PASS" if gates["overall_pass"] else "FAIL"
    print(f"=== done v20-w1 {args.host}/{args.arm}/{args.task}/s{args.seed}: "
          f"{verdict} rank={gates['final_effective_rank']:.1f} "
          f"conv={gates['convergence_relative_change']:.4f} "
          f"export={output_dir / 'eval_export.npz'} ===", flush=True)


if __name__ == "__main__":
    main()
