#!/usr/bin/env python3
"""Train the minimal Sub-JEPA-v16 host/memory factorial.

V16 deliberately introduces no new recurrent architecture.  It keeps the repaired
causal LeWM encoder, one-token action-conditioned predictor, synchronized clean target
view, and the existing none/SSM/compact-V8 memory paths.  The only candidate change is
Sub-JEPA's frozen row-orthonormal multi-subspace Gaussian regularizer.  A K=1
implementation-matched full-space control and the V10/V11 VICReg host make the effect
of the regularizer identifiable.

These runs use already-opened V11 trajectories and are adaptive development evidence,
not a confirmation experiment or an executed-return result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.memory_model import MemoryLeWorldModel
from lewm.models.sigreg import MultiSubspaceSIGReg
from scripts.hacssm_v11_data import (
    DEFAULT_CORRUPTION_SEED,
    DEFAULT_TRAIN_SEED,
    DEFAULT_VAL_SEED,
    V11TrajectoryDataset,
    load_cache,
    sha256_file,
)
from scripts.train_hacssm_v10 import (
    _finite_memory_log,
    _loader,
    _matched_gru_hidden,
    encoder_diagnostics,
)
import scripts.train_hacssm_v11 as v11


REGULARIZERS = ("fullsig", "subjepa16", "subjepa32", "vicreg")
MEMORIES = ("none", "ssm", "hacssmv8")
DESIGNS = tuple(
    f"{regularizer}_{memory}"
    for regularizer in REGULARIZERS
    for memory in MEMORIES
)
HELDOUT_CONDITIONS = v11.HELDOUT_CONDITIONS
ROLLOUT_SCHEMA_VERSION = v11.ROLLOUT_SCHEMA_VERSION
OBJECTIVE = "v16_paired_next_clean_plus_collapse_regularizer"
HISTORY_KEYS = (
    "loss",
    "predictive_loss",
    "regularizer_loss",
    "sigreg_loss",
    "variance_loss",
    "covariance_loss",
)


def parse_design(design: str) -> tuple[str, str, int | None]:
    if design not in DESIGNS:
        raise ValueError(f"unknown V16 design {design!r}")
    for regularizer in REGULARIZERS:
        prefix = f"{regularizer}_"
        if design.startswith(prefix):
            memory = design.removeprefix(prefix)
            subspaces = {
                "fullsig": 1,
                "subjepa16": 16,
                "subjepa32": 32,
                "vicreg": None,
            }[regularizer]
            return regularizer, memory, subspaces
    raise AssertionError("unreachable V16 design parser")


def design_metadata(design: str, embed_dim: int = 128) -> dict[str, Any]:
    regularizer, memory, subspaces = parse_design(design)
    return {
        "method": "Sub-JEPA-v16-development",
        "evidence_scope": "excluded_adaptive_opened_cache_development",
        "confirmation_evidence": False,
        "executed_return_evaluation": False,
        "regularizer": regularizer,
        "regularizer_family": (
            "vicreg_variance_covariance"
            if regularizer == "vicreg"
            else "epps_pulley_frozen_orthogonal_subspaces"
        ),
        "num_subspaces": subspaces,
        "subspace_dim": None if subspaces is None else embed_dim // subspaces,
        "projection_policy": (
            "not_applicable"
            if subspaces is None
            else "independent_frozen_row_orthonormal_qr"
        ),
        "projection_requires_grad": False,
        "sketch_direction_policy": (
            "not_applicable" if subspaces is None else "fresh_unit_directions_per_forward"),
        "regularizer_source": "active_clean_target",
        "memory_architecture": memory,
        "memory_specific_loss_weight": 0.0,
        "new_memory_architecture": False,
        "observation_correction_branch": False,
        "one_token_predictor": True,
        "paired_clean_target": True,
        "clean_target_gradient_active": True,
        "target_stop_gradient": False,
        "reward_used_for_training": False,
        "state_labels_used_for_training": False,
        "training_objective": OBJECTIVE,
    }


class V16ExperimentModel(nn.Module):
    def __init__(self, world: MemoryLeWorldModel, design: str):
        super().__init__()
        self.world = world
        self.design = design

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)


def build_model(args: argparse.Namespace, action_dim: int) -> V16ExperimentModel:
    regularizer, memory, subspaces = parse_design(args.design)
    memory_impl, memory_mode = (
        ("ema", "none") if memory == "none" else (memory, "both"))
    world = MemoryLeWorldModel(
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
    if world.encoder_norm != "causal" or world.predictor_norm != "none":
        raise RuntimeError("V16 requires the repaired causal/no-output-norm host")
    if memory == "none":
        # The EMA modules exist only because MemoryLeWorldModel uses this historical
        # spelling for the exact no-memory path.  Freeze them so the parameter count
        # and optimizer contain no unused memory parameters.
        world.memory.requires_grad_(False)
        world.fusion.requires_grad_(False)
    if regularizer != "vicreg":
        if subspaces is None:
            raise AssertionError("Gaussian V16 design omitted its subspace count")
        world.sigreg = MultiSubspaceSIGReg(
            embed_dim=args.embed_dim,
            num_subspaces=subspaces,
            num_projections=args.sigreg_projections,
        )
    return V16ExperimentModel(world, args.design)


def memory_representations(
        model: V16ExperimentModel, z: torch.Tensor, actions: torch.Tensor
        ) -> dict[str, Any]:
    """Common causal prior/posterior contract for the three frozen memory choices."""
    world = model.world
    _, memory, _ = parse_design(model.design)
    if memory == "none":
        prior = torch.zeros_like(z)
        prior[:, 0] = z[:, 0]
        prior[:, 1:] = v11.one_token_prediction(world, z[:, :-1], actions)
        return {
            "fused": z,
            "prior": prior,
            "posterior": z,
            "details": {},
        }
    if memory == "ssm":
        states = world.mem_ssm(z)
        decay = torch.sigmoid(world.mem_ssm.raw_decay).to(dtype=states.dtype)
        prior = torch.zeros_like(states)
        bias = world.mem_ssm.in_proj.bias.to(dtype=states.dtype)
        prior[:, 0] = states[:, 0]
        prior[:, 1:] = (1.0 - decay) * states[:, :-1] + decay * bias
        return {
            "fused": world.mem_ssm.fuse(z, states),
            "prior": v11._rms_read(prior),
            "posterior": v11._rms_read(states),
            "details": {"states": states, "priors": prior},
        }
    if memory == "hacssmv8":
        mixed, details = world.mem_hacssmv8(z, actions, return_details=True)
        route = details["route"].to(
            device=z.device, dtype=z.dtype).view(1, 1, -1, 1)
        prior = (details["priors"] * route).sum(dim=2)
        posterior = (details["states"] * route).sum(dim=2)
        return {
            "fused": world.mem_hacssmv8.fuse(z, mixed),
            "prior": v11._rms_read(prior),
            "posterior": v11._rms_read(posterior),
            "details": details,
        }
    raise AssertionError(f"unhandled V16 memory {memory!r}")


def compute_losses(
        model: V16ExperimentModel, observed: torch.Tensor,
        clean: torch.Tensor, actions: torch.Tensor,
        sigreg_lambda: float) -> dict[str, torch.Tensor]:
    regularizer, _, _ = parse_design(model.design)
    clean_z = v11.encode_clean_active(model, clean)
    observed_z = model.world.encode(observed)
    memory = memory_representations(model, observed_z, actions)
    prediction = v11.one_token_prediction(
        model.world, memory["fused"][:, :-1], actions)
    predictive_loss = F.mse_loss(prediction.float(), clean_z[:, 1:].float())
    zero = predictive_loss.new_zeros(())
    sigreg_loss = zero
    variance_loss = zero
    covariance_loss = zero
    if regularizer == "vicreg":
        variance_loss, covariance_loss = v11._vicreg_terms(clean_z)
        regularizer_loss = variance_loss + covariance_loss
    else:
        # Epps--Pulley characteristic functions are numerically sensitive under
        # BF16.  The regularizer retains gradients while computing in FP32.
        with torch.autocast(device_type=clean_z.device.type, enabled=False):
            sigreg_loss = model.world.sigreg(clean_z.float())
        regularizer_loss = float(sigreg_lambda) * sigreg_loss
    return {
        "loss": predictive_loss + regularizer_loss,
        "predictive_loss": predictive_loss,
        "regularizer_loss": regularizer_loss,
        "sigreg_loss": sigreg_loss,
        "variance_loss": variance_loss,
        "covariance_loss": covariance_loss,
    }


def run_epoch(
        model: V16ExperimentModel, loader,
        optimizer: torch.optim.Optimizer | None, device: torch.device,
        use_amp: bool, sigreg_lambda: float) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {key: 0.0 for key in HISTORY_KEYS}
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
                if use_amp else torch.autocast("cpu", enabled=False))
            with amp_context:
                losses = compute_losses(
                    model, observed, clean, actions, sigreg_lambda)
            if train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = observed.shape[0]
            for key in HISTORY_KEYS:
                totals[key] += float(losses[key].detach()) * batch_size
            count += batch_size
    if not count:
        raise RuntimeError("empty V16 epoch")
    return {key: value / count for key, value in totals.items()}


def _validate_data_contract(args: argparse.Namespace):
    train = load_cache(args.train_data)
    val = load_cache(args.val_data)
    if train.split != "train" or val.split != "val":
        raise ValueError("V16 requires explicit train/validation cache roles")
    if (train.seed, val.seed) != (DEFAULT_TRAIN_SEED, DEFAULT_VAL_SEED):
        raise ValueError("V16 cache seeds differ from the frozen V11 data contract")
    if train.smooth_rho != 0.0 or val.smooth_rho != 0.0:
        raise ValueError("V16 requires the IID-action V11 cache contract")
    fields = (
        "env_id", "length", "img_size", "action_dim", "state_dim",
        "task_observation_dim", "task_observation_keys", "task_observation_shapes")
    if tuple(getattr(train, key) for key in fields) != tuple(
            getattr(val, key) for key in fields):
        raise ValueError("V16 train/validation cache schema mismatch")
    if args.eval_rollout_episode not in range(val.episodes):
        raise ValueError("V16 evaluation rollout episode is out of range")
    return train, val


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--design", choices=DESIGNS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/subjepa_v16_development")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
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
    parser.add_argument("--sigreg-quad-nodes", type=int, default=17)
    parser.add_argument("--probe-ridge", type=float, default=1e-3)
    parser.add_argument("--eval-target-key", default="task_observation",
                        choices=("task_observation",))
    parser.add_argument("--corruption-seed", type=int, default=DEFAULT_CORRUPTION_SEED)
    parser.add_argument("--eval-rollout-episode", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=False)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="lewm-memory-popgym")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--wandb-study", default="subjepa-v16-development")
    parser.add_argument("--extra-tag", default="excluded-adaptive-development,v16")
    return parser.parse_args(argv)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 4 or args.num_workers < 0
            or args.sigreg_projections < 1 or args.sigreg_quad_nodes != 17
            or not math.isfinite(args.sigreg_lambda) or args.sigreg_lambda < 0):
        raise ValueError("invalid V16 training configuration")
    if args.wandb and args.wandb_mode != "online":
        raise ValueError("V16 online logging cannot use an offline W&B run")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    train_metadata, val_metadata = _validate_data_contract(args)

    train_view = V11TrajectoryDataset(
        args.train_data, "train", args.corruption_seed, args.history_len)
    val_train_view = V11TrajectoryDataset(
        args.val_data, "train", args.corruption_seed, args.history_len)
    train_clean = V11TrajectoryDataset(
        args.train_data, "clean", args.corruption_seed, args.history_len)
    val_clean = V11TrajectoryDataset(
        args.val_data, "clean", args.corruption_seed, args.history_len)
    heldout = {
        condition: V11TrajectoryDataset(
            args.val_data, condition, args.corruption_seed, args.history_len)
        for condition in HELDOUT_CONDITIONS
    }
    sample = train_clean[0]
    if args.eval_target_key not in sample:
        raise RuntimeError(f"V16 cache is missing {args.eval_target_key!r}")
    eval_target_dim = int(sample[args.eval_target_key].shape[-1])
    if eval_target_dim != train_metadata.task_observation_dim:
        raise RuntimeError("V16 evaluation target width differs from cache metadata")

    model = build_model(args, train_metadata.action_dim).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _loader(train_view, args, train=True)
    val_loader = _loader(val_train_view, args, train=False)

    # Reuse the thoroughly tested V11 probe/rollout machinery with V16's explicit
    # three-memory representation contract.  Each training process is isolated.
    v11.memory_representations = memory_representations

    env_name = f"dmc:{train_metadata.env_id}"
    run_name = f"lewm-{env_name}-{args.design}-s{args.seed}"
    output_dir = Path(args.output_dir).resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    core_files = ("model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json")
    for filename in core_files:
        if (output_dir / filename).exists():
            raise FileExistsError(f"refusing to overwrite {output_dir / filename}")

    metadata = design_metadata(args.design, args.embed_dim)
    wb = None
    if args.wandb:
        import wandb
        tags = [
            "lewm-memory", "end-to-end-rgb", "subjepa-v16",
            "excluded-adaptive-development", f"env:{env_name}",
            f"design:{args.design}", f"study:{args.wandb_study}",
        ]
        if args.extra_tag:
            tags.extend(tag.strip() for tag in args.extra_tag.split(",") if tag.strip())
        wb = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            mode=args.wandb_mode,
            name=f"{args.wandb_study}-{run_name}",
            group=f"{args.wandb_study}:{env_name}",
            job_type=args.design,
            tags=tags,
            dir=str(output_dir),
            config=(vars(args) | {
                "env": env_name,
                "action_dim": train_metadata.action_dim,
                "state_dim": train_metadata.state_dim,
                "eval_target_dim": eval_target_dim,
                "prediction_loss_weight": 1.0,
                "sigreg_loss_weight": (
                    0.0 if metadata["regularizer"] == "vicreg"
                    else args.sigreg_lambda),
                "variance_loss_weight": (
                    1.0 if metadata["regularizer"] == "vicreg" else 0.0),
                "covariance_loss_weight": (
                    1.0 if metadata["regularizer"] == "vicreg" else 0.0),
                **metadata,
            }),
            settings=wandb.Settings(init_timeout=180),
        )
        if wb.offline or (args.wandb_entity and wb.entity != args.wandb_entity):
            raise RuntimeError("V16 W&B online/entity preflight failed")
        wb.define_metric("epoch")
        for namespace in ("train/*", "val/*", "mem/*", "perf/*"):
            wb.define_metric(namespace, step_metric="epoch")

    print(
        f"=== {run_name} | params={model.num_parameters():,} | "
        f"regularizer={metadata['regularizer']} memory={metadata['memory_architecture']} "
        f"amp={use_amp} ===",
        flush=True,
    )
    history = []
    epoch_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, use_amp, args.sigreg_lambda)
        val_metrics = run_epoch(
            model, val_loader, None, device, use_amp, args.sigreg_lambda)
        memory_log = _finite_memory_log(model.world)
        epoch_seconds = time.time() - started
        epoch_times.append(epoch_seconds)
        history.append({
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "train": train_metrics,
            "val": val_metrics,
        })
        print(
            f"e{epoch:3d}/{args.epochs} ({epoch_seconds:.1f}s) "
            f"train={train_metrics['loss']:.5f} "
            f"pred={train_metrics['predictive_loss']:.5f} "
            f"reg={train_metrics['regularizer_loss']:.5f} | "
            f"val={val_metrics['loss']:.5f}",
            flush=True,
        )
        if wb is not None:
            wb.log({
                "epoch": epoch,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
                **{f"mem/{key}": value for key, value in memory_log.items()},
                "perf/epoch_seconds": epoch_seconds,
            }, step=epoch)

    probes = v11.fit_state_probes(model, train_clean, args, device, use_amp)
    inverse_probe = v11.fit_inverse_action_probe(
        model, train_clean, args, device, use_amp)
    inverse_metrics = v11.evaluate_inverse_action_probe(
        model, val_clean, inverse_probe, args, device, use_amp)
    ceilings = v11.probe_ceilings(model, val_clean, probes, args, device, use_amp)
    action_mean = train_view.actions.mean(
        axis=(0, 1), dtype=np.float64).astype(np.float32)
    action_std = train_view.actions.std(
        axis=(0, 1), dtype=np.float64).clip(min=1e-6).astype(np.float32)
    regularizer, _, subspaces = parse_design(args.design)
    reg_diagnostics: dict[str, Any] = {
        "subspace_projection_orthogonality_max_abs": 0.0,
        "subspace_projection_frozen": regularizer != "vicreg",
    }
    if regularizer != "vicreg":
        matrices = model.world.sigreg.projection_matrices.float()
        identity = torch.eye(
            matrices.shape[1], device=matrices.device, dtype=matrices.dtype)
        error = matrices @ matrices.transpose(-1, -2) - identity
        reg_diagnostics["subspace_projection_orthogonality_max_abs"] = float(
            error.abs().max())
        reg_diagnostics["subspace_projection_count"] = int(subspaces or 0)
        reg_diagnostics["subspace_projection_dimension"] = int(matrices.shape[1])
        projection_bytes = matrices.detach().cpu().contiguous().numpy().astype(
            "<f4", copy=False).tobytes()
        reg_diagnostics["subspace_projection_sha256"] = hashlib.sha256(
            projection_bytes).hexdigest()

    metrics: dict[str, Any] = {
        "schema_version": 1,
        "env": env_name,
        "design": args.design,
        "seed": args.seed,
        "epochs": args.epochs,
        "encoder_type": "vit",
        "encoder_frozen": False,
        "encoder_norm": "causal",
        "predictor_norm": "none",
        "end_to_end_rgb": True,
        "prediction_loss_weight": 1.0,
        "sigreg_lambda": args.sigreg_lambda,
        "sigreg_projections_per_subspace": args.sigreg_projections,
        "sigreg_quad_nodes": args.sigreg_quad_nodes,
        "probe_ridge": args.probe_ridge,
        "probe_fit_split": "clean_train_coordinate_specific",
        "headline_metric": "heldout_prior_state_nmse",
        "eval_target_key": args.eval_target_key,
        "eval_target_dim": eval_target_dim,
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
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "final_train_loss": history[-1]["train"]["loss"],
        "final_val_loss": history[-1]["val"]["loss"],
        "val_predictive_loss": history[-1]["val"]["predictive_loss"],
        "val_regularizer_loss": history[-1]["val"]["regularizer_loss"],
        "mean_epoch_seconds": float(np.mean(epoch_times)),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0),
        **metadata,
        **reg_diagnostics,
        **ceilings,
        **encoder_diagnostics(model.world, val_clean, args, device),
        **inverse_metrics,
        **v11._ridge_action_history(
            train_view.actions, val_clean.actions,
            action_mean, action_std, args.probe_ridge),
        **v11.action_only_integrator_probe(train_clean, val_clean, heldout, args),
        **v11.initial_encoder_integrator_probe(
            model, train_clean, val_clean, heldout, args, device, use_amp),
    }
    metrics.update({
        f"memory_{key}": float(value)
        for key, value in model.world.horizons().items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    })
    window = min(10, max(1, args.epochs // 2))
    previous_rows = history[-2 * window:-window]
    recent_rows = history[-window:]
    for loss_name in ("predictive_loss", "regularizer_loss", "loss"):
        previous = (
            np.mean([row["val"][loss_name] for row in previous_rows])
            if previous_rows
            else np.mean([row["val"][loss_name] for row in recent_rows]))
        recent = np.mean([row["val"][loss_name] for row in recent_rows])
        metrics[f"{loss_name}_convergence_relative_change"] = float(
            (previous - recent) / max(abs(previous), 1e-12))

    clean_result, _ = v11.evaluate_condition(
        model, val_clean, probes, args, device, use_amp)
    for coordinate in ("prior", "posterior", "encoder", "predictor"):
        metrics[f"clean_{coordinate}_state_nmse"] = clean_result[
            f"{coordinate}_primary"]
        metrics[f"clean_{coordinate}_state_r2"] = clean_result[f"{coordinate}_r2"]
    condition_rollouts = {}
    heldout_primary = {
        coordinate: []
        for coordinate in ("prior", "posterior", "encoder", "predictor")
    }
    for condition, dataset in heldout.items():
        result, rollout = v11.evaluate_condition(
            model, dataset, probes, args, device, use_amp,
            rollout_episode=args.eval_rollout_episode)
        for coordinate in heldout_primary:
            primary = result[f"{coordinate}_primary"]
            metrics[f"{condition}_{coordinate}_state_nmse"] = primary
            metrics[f"{condition}_{coordinate}_state_r2"] = result[
                f"{coordinate}_r2"]
            heldout_primary[coordinate].append(primary)
            for phase in ("gap", "deep", "first_post", "post"):
                metrics[f"{condition}_{coordinate}_state_nmse_{phase}"] = result[
                    f"{coordinate}_{phase}"]
        if rollout is None:
            raise RuntimeError(f"missing V16 rollout episode for {condition}")
        condition_rollouts[condition] = rollout
    for coordinate, values in heldout_primary.items():
        metrics[f"heldout_{coordinate}_state_nmse"] = float(np.mean(values))

    rollout_path = output_dir / "eval_rollout.npz"
    rollout_arrays, rollout_video = v11._rollout_package(
        condition_rollouts, rollout_path, args.eval_rollout_episode)
    rollout_hash = sha256_file(rollout_path)
    metrics["eval_rollout_episode"] = args.eval_rollout_episode
    metrics["eval_rollout_sha256"] = rollout_hash
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "final_metrics": metrics,
        "history": history,
        "state_probes": probes,
        "inverse_action_probe": inverse_probe,
    }, output_dir / "model.pt")
    _write_json_exclusive(output_dir / "metrics.json", metrics)

    wandb_record: dict[str, Any] = {
        "schema_version": 1,
        "mode": "disabled",
        "study": args.wandb_study,
        "state": "not_requested",
        "eval_rollout_sha256": rollout_hash,
    }
    if wb is not None:
        try:
            import wandb
            table = v11._make_rollout_table(wandb, rollout_arrays)
            wb.log({
                **{f"eval/{key}": value for key, value in metrics.items()
                   if isinstance(value, (int, float)) and math.isfinite(float(value))},
                "eval/rollout_trace": table,
                "eval/paired_rollout": wandb.Video(
                    rollout_video, fps=6, format="mp4",
                    caption="rows: four held-out corruptions; left observed, right clean"),
            })
            artifact_name = f"eval-rollout-{wb.id}"
            artifact = wandb.Artifact(
                artifact_name,
                type="evaluation-rollout",
                metadata={
                    "schema_version": ROLLOUT_SCHEMA_VERSION,
                    "study": args.wandb_study,
                    "env": env_name,
                    "design": args.design,
                    "seed": args.seed,
                    "episode": args.eval_rollout_episode,
                    "sha256": rollout_hash,
                    **metadata,
                },
            )
            artifact.add_file(str(rollout_path), name="eval_rollout.npz")
            wb.log_artifact(artifact)
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
            }
            wb.finish(exit_code=0)
        except Exception as error:  # preserve complete local science on sync-only failures
            wandb_record = {
                "schema_version": 1,
                "run_id": str(getattr(wb, "id", "unknown")),
                "run_name": str(getattr(wb, "name", "unknown")),
                "url": str(getattr(wb, "url", "")),
                "mode": "online",
                "study": args.wandb_study,
                "state": "sync_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "eval_rollout_sha256": rollout_hash,
            }
            try:
                wb.finish(exit_code=1, quiet=True)
            except Exception:
                pass
    _write_json_exclusive(output_dir / "wandb_run.json", wandb_record)

    print(
        f"=== done {run_name}: heldout_prior_state_nmse="
        f"{metrics['heldout_prior_state_nmse']:.6f} ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
