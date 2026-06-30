#!/usr/bin/env python3
"""Train the CF-HIRO-v13 split-agreement normal predictive observer.

The candidate fits every memory operator from detached, dropout-off, FP64 train-split
embeddings before epoch one and after each optimizer epoch.  Gradient training has only
one-token next-clean prediction and unit-weight VICReg variance/covariance terms.  No
reward, simulator state, validation embedding, visibility label, corruption identity,
memory loss, or gradient-trained memory scalar enters the fit.

The established V12 driver is reused only as an evaluation/W&B/rollout harness.  This
module replaces its model, fit, representation, diagnostic, metadata, and serialization
hooks at runtime; baselines are delegated unchanged to the frozen V11 trainer.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import inspect
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.cf_hiro import CFHIROFit, fit_cf_hiro
from lewm.models.memory_model import MemoryLeWorldModel
from scripts.hacssm_v11_data import (
    DEFAULT_CORRUPTION_SEED,
    DEFAULT_TRAIN_SEED,
    DEFAULT_VAL_SEED,
    V11TrajectoryDataset,
    load_cache,
)
from scripts.train_hacssm_v10 import _loader, _matched_gru_hidden
import scripts.train_hacssm_v11 as v11
import scripts.train_siro_v12 as v12


CF_HIRO_DESIGNS = (
    "cfhirov13",
    "cfhirov13_fullanchor",
    "cfhirov13_triangular",
    "cfhirov13_noshrink",
    "cfhirov13_noaction",
    "cfhirov13_nocorrect",
)
BASELINES = ("ssm", "hacssmv8", "kdiov11")
DESIGNS = (*CF_HIRO_DESIGNS, *BASELINES)
CORE_MODES = {
    "cfhirov13": "full",
    "cfhirov13_fullanchor": "fullanchor",
    "cfhirov13_triangular": "triangular",
    "cfhirov13_noshrink": "noshrink",
    "cfhirov13_noaction": "noaction",
    "cfhirov13_nocorrect": "nocorrect",
}
OBJECTIVE = "cfhirov13_one_token_next_clean_plus_unit_vicreg"
V11_COMPARATOR_RANKING = "rawdiff_displacement_detached"


class CFHIROExperimentModel(nn.Module):
    def __init__(self, world: MemoryLeWorldModel):
        super().__init__()
        self.world = world

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)


def full_hankel_state_dim(length: int, output_dim: int, action_dim: int) -> int:
    """Fixed full rectangular Ho--Kalman schema; no data-selected model order."""
    for name, value in (("length", length), ("output_dim", output_dim),
                        ("action_dim", action_dim)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    lag_count = length - 1
    if lag_count < 4:
        raise ValueError("CF-HIRO requires at least five frames")
    block_rows = lag_count // 2
    block_columns = lag_count - block_rows
    return min(block_rows * output_dim, block_columns * action_dim)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--memory-mode", choices=DESIGNS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/hacssm_v13_screen_cfhiro30")
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
    parser.add_argument("--wandb-study", default="hacssm-v13-screen-cfhiro30")
    parser.add_argument("--extra-tag", default="excluded-adaptive-screen")
    return parser.parse_args(argv)


def _delegate_argv(args: argparse.Namespace) -> list[str]:
    if args.memory_mode not in BASELINES:
        raise ValueError("only V13 baselines may delegate to the V11 trainer")
    values: dict[str, Any] = {
        "train-data": args.train_data,
        "val-data": args.val_data,
        "memory-mode": args.memory_mode,
        "seed": args.seed,
        "output-dir": args.output_dir,
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "lr": args.lr,
        "weight-decay": args.weight_decay,
        "num-workers": args.num_workers,
        "img-size": args.img_size,
        "patch-size": args.patch_size,
        "embed-dim": args.embed_dim,
        "encoder-layers": args.encoder_layers,
        "encoder-heads": args.encoder_heads,
        "predictor-layers": args.predictor_layers,
        "predictor-heads": args.predictor_heads,
        "history-len": args.history_len,
        "dropout": args.dropout,
        "sigreg-lambda": args.sigreg_lambda,
        "sigreg-projections": args.sigreg_projections,
        "probe-ridge": args.probe_ridge,
        "eval-target-key": args.eval_target_key,
        "corruption-seed": args.corruption_seed,
        "eval-rollout-episode": args.eval_rollout_episode,
        "device": args.device,
        "wandb-project": args.wandb_project,
        "wandb-mode": args.wandb_mode,
        "wandb-study": args.wandb_study,
        "extra-tag": ",".join(filter(None, (
            args.extra_tag, "v13-matched-v11-trainer"))),
    }
    if args.memory_mode == "kdiov11":
        values["development-action-ranking"] = V11_COMPARATOR_RANKING
    if args.wandb_entity:
        values["wandb-entity"] = args.wandb_entity
    result: list[str] = []
    for key, value in values.items():
        result.extend((f"--{key}", str(value)))
    if args.no_amp:
        result.append("--no-amp")
    result.append("--wandb" if args.wandb else "--no-wandb")
    return result


def validate_data_contract(args: argparse.Namespace):
    train = load_cache(args.train_data)
    val = load_cache(args.val_data)
    if train.split != "train" or val.split != "val":
        raise ValueError("CF-HIRO requires explicit train/validation cache roles")
    if (train.seed, val.seed) != (DEFAULT_TRAIN_SEED, DEFAULT_VAL_SEED):
        raise ValueError("CF-HIRO cache seeds differ from the frozen IID protocol")
    if train.smooth_rho != 0.0 or val.smooth_rho != 0.0:
        raise ValueError(
            "CF-HIRO action moments require IID actions with smooth_rho=0; "
            "full-rank correlated actions are not sufficient")
    fields = (
        "env_id", "length", "img_size", "action_dim", "state_dim",
        "task_observation_dim", "task_observation_keys", "task_observation_shapes")
    if tuple(getattr(train, key) for key in fields) != tuple(
            getattr(val, key) for key in fields):
        raise ValueError("CF-HIRO train/validation cache schema mismatch")
    if args.eval_rollout_episode not in range(val.episodes):
        raise ValueError("evaluation rollout episode is out of range")
    args.cf_hiro_state_dim = full_hankel_state_dim(
        train.length, args.embed_dim, train.action_dim)
    args.cf_hiro_fit_path = str(train.path.resolve())
    return train, val


def build_model(args: argparse.Namespace, action_dim: int) -> CFHIROExperimentModel:
    state_dim = getattr(args, "cf_hiro_state_dim", None)
    if not isinstance(state_dim, int) or state_dim < 1:
        metadata = load_cache(args.train_data)
        state_dim = full_hankel_state_dim(
            metadata.length, args.embed_dim, action_dim)
        args.cf_hiro_state_dim = state_dim
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
        memory_impl=args.memory_mode,
        memory_mode="both",
        gru_hidden=_matched_gru_hidden(args.embed_dim),
        hier_loss_weight=0.0,
        encoder_type="vit",
        cf_hiro_state_dim=state_dim,
    )
    if world.encoder_norm != "causal" or world.predictor_norm != "none":
        raise RuntimeError("CF-HIRO requires causal affine-free encoding and no predictor norm")
    memory = getattr(world, "mem_cfhirov13", None)
    if memory is None or memory.parameter_count() != 0:
        raise RuntimeError("CF-HIRO memory must be installed with zero optimizer parameters")
    return CFHIROExperimentModel(world)


def cf_hiro_memory(model: CFHIROExperimentModel):
    memory = getattr(model.world, "mem_cfhirov13", None)
    if memory is None:
        raise RuntimeError("CF-HIRO experiment model is missing mem_cfhirov13")
    return memory


@torch.no_grad()
def collect_detached_fit_views(
        model: CFHIROExperimentModel, clean_dataset: V11TrajectoryDataset,
        observed_dataset: V11TrajectoryDataset, args: argparse.Namespace,
        device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect aligned train-only views; validation datasets are rejected here."""
    expected_path = Path(args.cf_hiro_fit_path).resolve()
    for label, dataset, view in (
            ("clean", clean_dataset, "clean"),
            ("observed", observed_dataset, "train")):
        if dataset.metadata.split != "train" or dataset.metadata.path.resolve() != expected_path:
            raise RuntimeError(f"CF-HIRO {label} fit view is not the registered train cache")
        if dataset.view != view:
            raise RuntimeError(f"CF-HIRO {label} fit view must be {view!r}")
    model.eval()
    clean_chunks, observed_chunks, action_chunks, index_chunks = [], [], [], []
    clean_loader = _loader(clean_dataset, args, train=False)
    observed_loader = _loader(observed_dataset, args, train=False)
    for clean_batch, observed_batch in zip(clean_loader, observed_loader, strict=True):
        if not torch.equal(clean_batch["episode_index"], observed_batch["episode_index"]):
            raise RuntimeError("CF-HIRO paired fit loaders lost episode alignment")
        if not torch.equal(clean_batch["actions"], observed_batch["actions"]):
            raise RuntimeError("CF-HIRO paired fit loaders disagree on executed actions")
        if int(observed_batch["gap_start"].min()) < 5:
            raise RuntimeError("CF-HIRO requires the registered exact initial observation")
        clean = clean_batch["clean"].to(device, non_blocking=True)
        observed = observed_batch["observed"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=False):
            clean_z = model.world.encode(clean.float())
            observed_z = model.world.encode(observed.float())
        clean_chunks.append(clean_z.float().cpu())
        observed_chunks.append(observed_z.float().cpu())
        action_chunks.append(clean_batch["actions"].float().cpu())
        index_chunks.append(clean_batch["episode_index"].cpu())
    indices = torch.cat(index_chunks)
    order = torch.argsort(indices)
    if not torch.equal(indices[order], torch.arange(len(clean_dataset), dtype=indices.dtype)):
        raise RuntimeError("CF-HIRO fit did not consume every train episode exactly once")
    clean_z = torch.cat(clean_chunks)[order].double()
    observed_z = torch.cat(observed_chunks)[order].double()
    actions = torch.cat(action_chunks)[order].double()
    if float((clean_z[:, 0] - observed_z[:, 0]).abs().max()) != 0.0:
        raise RuntimeError("CF-HIRO paired fit does not have an exact initial observation")
    return clean_z, observed_z, actions


def _fit_candidate(clean: torch.Tensor, observed: torch.Tensor,
                   actions: torch.Tensor, design: str) -> CFHIROFit:
    mode = CORE_MODES[design]
    parameters = inspect.signature(fit_cf_hiro).parameters
    kwargs: dict[str, Any] = {}
    if "mode" in parameters:
        kwargs["mode"] = mode
    elif "variant" in parameters:
        kwargs["variant"] = mode
    elif mode not in {"full", "noaction", "nocorrect", "fullanchor"}:
        raise RuntimeError(
            "the finalized CF-HIRO fitter must accept a mode/variant for fit-changing controls")
    fit = fit_cf_hiro(clean, observed, actions, **kwargs)
    if not isinstance(fit, CFHIROFit):
        raise TypeError("fit_cf_hiro must return CFHIROFit")
    return fit


def _relative_delta(current: torch.Tensor, previous: torch.Tensor) -> float:
    denominator = previous.double().norm().clamp_min(torch.finfo(torch.float64).tiny)
    return float((current.double() - previous.double()).norm() / denominator)


def _with_receipts(fit: CFHIROFit, receipts: Mapping[str, Any]) -> CFHIROFit:
    if dataclasses.is_dataclass(fit):
        return dataclasses.replace(fit, receipts=copy.deepcopy(dict(receipts)))
    raise TypeError("CFHIROFit must remain a dataclass for immutable receipt updates")


@torch.no_grad()
def refit_cf_hiro_operators(
        model: CFHIROExperimentModel, clean_dataset: V11TrajectoryDataset,
        observed_dataset: V11TrajectoryDataset, args: argparse.Namespace,
        device: torch.device, *, fit_index: int,
        previous_fit: CFHIROFit | None = None) -> tuple[CFHIROFit, float]:
    started = time.time()
    clean, observed, actions = collect_detached_fit_views(
        model, clean_dataset, observed_dataset, args, device)
    fit = _fit_candidate(clean, observed, actions, args.memory_mode)
    state_dim = int(fit.state_matrix.shape[0])
    if state_dim != args.cf_hiro_state_dim:
        raise RuntimeError(
            f"CF-HIRO fit changed fixed state schema {args.cf_hiro_state_dim}->{state_dim}")
    receipts = dict(fit.receipts)
    receipts.update({
        "fit_index": int(fit_index),
        "fit_split": "train_only",
        "fit_uses_validation": False,
        "fit_uses_reward": False,
        "fit_uses_task_state": False,
        "fit_uses_corruption_identity": False,
        "fit_gradient_active": False,
        "fit_episode_count": int(clean.shape[0]),
        "fit_length": int(clean.shape[1]),
        "fit_design": args.memory_mode,
    })
    if previous_fit is None:
        receipts["refit_has_previous"] = False
    else:
        receipts["refit_has_previous"] = True
        for field in (
                "state_matrix", "action_matrix", "read_matrix",
                "process_covariance", "measurement_covariance", "steady_gain"):
            if hasattr(fit, field) and hasattr(previous_fit, field):
                receipts[f"refit_{field}_relative_delta"] = _relative_delta(
                    getattr(fit, field), getattr(previous_fit, field))
    fit = _with_receipts(fit, receipts)
    cf_hiro_memory(model).install_fit(fit)
    return fit, time.time() - started


def memory_representations(
        model: CFHIROExperimentModel, z: torch.Tensor,
        actions: torch.Tensor) -> dict[str, Any]:
    read, details = cf_hiro_memory(model)(z, actions, return_details=True)
    if not isinstance(details, Mapping):
        raise RuntimeError("CF-HIRO forward details must be a mapping")
    prior = details.get("prior_reads")
    posterior = details.get("posterior_reads", details.get("reads", read))
    if not isinstance(prior, torch.Tensor) or tuple(prior.shape) != tuple(z.shape):
        raise RuntimeError("CF-HIRO details must expose strict pre-observation prior_reads")
    if not isinstance(posterior, torch.Tensor) or tuple(posterior.shape) != tuple(z.shape):
        raise RuntimeError("CF-HIRO details must expose posterior_reads")
    if not torch.equal(read, posterior):
        raise RuntimeError("CF-HIRO direct fusion must equal its posterior read")
    return {"fused": read, "prior": prior, "posterior": posterior, "details": details}


def _payload(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.float() if tensor.is_floating_point() else tensor
    if dataclasses.is_dataclass(value):
        return {field.name: _payload(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_payload(item) for item in value)
    if isinstance(value, list):
        return [_payload(item) for item in value]
    return copy.deepcopy(value)


def operator_fit_payload(fit: CFHIROFit) -> dict[str, Any]:
    payload = _payload(fit)
    if not isinstance(payload, dict):
        raise TypeError("CF-HIRO fit payload must serialize to a dictionary")
    return payload


def _flatten_scalars(value: Any, prefix: str, result: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten_scalars(item, f"{prefix}_{key}" if prefix else str(key), result)
    elif isinstance(value, (bool, int, float, str)):
        if not isinstance(value, float) or math.isfinite(value):
            result[prefix] = value


def scalar_fit_receipts(receipts: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    _flatten_scalars(receipts, "cf_hiro_fit", flattened)
    return flattened


def design_metadata(design: str) -> dict[str, Any]:
    if design not in CF_HIRO_DESIGNS:
        raise ValueError(f"unknown CF-HIRO design {design!r}")
    mode = CORE_MODES[design]
    return {
        "method": "CF-HIRO-v13",
        "variant": mode,
        "self_supervision": "paired_train_rgb_plus_executed_iid_actions",
        "reward_or_state_labels_used_for_fit": False,
        "validation_used_for_fit": False,
        "memory_gradient_parameter_count": 0,
        "fit_schedule": "before_epoch1_and_after_every_epoch",
        "fit_precision": "detached_fp64_cpu",
        "predictor_fusion": "direct_posterior_read_no_observation_bypass",
        "anchor_policy": "full_z0" if mode == "fullanchor" else "output_complement",
        "transition_policy": "real_schur_triangular" if mode == "triangular" else "normal_blocks",
        "fold_shrink_policy": "disabled" if mode == "noshrink" else "positive_part_agreement",
        "action_policy": "zero" if mode == "noaction" else "identified",
        "correction_policy": "zero" if mode == "nocorrect" else "steady_riccati",
        "cross_fitted_claim": False,
    }


def _open_loop_transition(
        state: torch.Tensor, action: torch.Tensor,
        transition: torch.Tensor, action_map: torch.Tensor,
        action_mean: torch.Tensor, *, noaction: bool) -> torch.Tensor:
    """Propagate a diagnostic state without leaking the caller's AMP dtype policy."""
    action = action.to(device=state.device, dtype=state.dtype)
    transition = transition.to(state)
    if noaction:
        effect = torch.zeros_like(state)
    else:
        effect = (action - action_mean.to(state)) @ action_map.to(state).T
    return state @ transition.T + effect


@torch.no_grad()
def cf_hiro_diagnostics(
        model: CFHIROExperimentModel, dataset: V11TrajectoryDataset,
        fit: CFHIROFit, args: argparse.Namespace, device: torch.device,
        use_amp: bool) -> dict[str, Any]:
    """Streaming, direct-sum, and global action-derangement receipts."""
    del fit
    model.eval()
    memory = cf_hiro_memory(model)
    first = next(iter(_loader(dataset, args, train=False)))
    frames = first["clean"][:2].to(device)
    actions = first["actions"][:2].to(device)
    amp_context = lambda: (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_amp else torch.autocast("cpu", enabled=False))
    with amp_context():
        z = model.world.encode(frames)
        batch_read, details = memory(z, actions, return_details=True)
        state = memory.initial_state(z[:, 0])
        streamed = [memory.read_state(state)]
        for step in range(1, z.shape[1]):
            read, state = memory.step(state, z[:, step], actions[:, step - 1])
            streamed.append(read)
    streaming_error = float((torch.stack(streamed, dim=1).float()
                             - batch_read.float()).abs().max())

    all_z, all_actions, indices = [], [], []
    for batch in _loader(dataset, args, train=False):
        with amp_context():
            encoded = model.world.encode(batch["clean"].to(device, non_blocking=True))
        all_z.append(encoded.float().cpu())
        all_actions.append(batch["actions"].float().cpu())
        indices.append(batch["episode_index"].cpu())
    order = torch.argsort(torch.cat(indices))
    clean_z = torch.cat(all_z)[order].to(device)
    true_actions = torch.cat(all_actions)[order].to(device)
    negative_actions = torch.roll(true_actions, shifts=1, dims=0)
    with amp_context():
        true_read, true_details = memory(clean_z, true_actions, return_details=True)
        negative_read, negative_details = memory(
            clean_z, negative_actions, return_details=True)
    posterior_means = true_details.get("state_means")
    complement_anchors = true_details.get("complement_anchors")
    if (not isinstance(posterior_means, torch.Tensor)
            or not isinstance(complement_anchors, torch.Tensor)):
        raise RuntimeError("CF-HIRO diagnostics require posterior state/complement sequences")

    # Start at every clean posterior and propagate the entire remaining suffix without
    # another observation. Negative actions are one deterministic global episode roll,
    # fixed before any diagnostic batching.
    state_true = posterior_means[:, :-1]
    state_negative = posterior_means[:, :-1]
    complement = complement_anchors[:, :-1]
    transition = memory.state_matrix.to(state_true)
    action_map = memory.action_matrix.to(state_true)
    action_mean = memory.action_mean.to(state_true)
    read_matrix = memory.read_matrix.to(state_true)
    output_mean = memory.output_mean.to(state_true)
    true_mse, negative_mse, pair_accuracy, divergences = [], [], [], []
    horizon = true_actions.shape[1]
    for offset in range(horizon):
        valid = horizon - offset
        state_true = state_true[:, :valid]
        state_negative = state_negative[:, :valid]
        complement = complement[:, :valid]
        true_action = true_actions[:, offset:offset + valid]
        negative_action = negative_actions[:, offset:offset + valid]

        state_true = _open_loop_transition(
            state_true, true_action, transition, action_map, action_mean,
            noaction=memory.mode == "noaction")
        state_negative = (
            state_true if memory.mode == "noaction"
            else _open_loop_transition(
                state_negative, negative_action, transition, action_map,
                action_mean, noaction=False))
        true_suffix_read = output_mean + complement + state_true @ read_matrix.T
        negative_suffix_read = output_mean + complement + state_negative @ read_matrix.T
        target = clean_z[:, offset + 1:offset + 1 + valid]
        positive = (true_suffix_read - target).square().mean(dim=-1)
        negative = (negative_suffix_read - target).square().mean(dim=-1)
        true_mse.append(positive.mean())
        negative_mse.append(negative.mean())
        pair_accuracy.append((
            (positive < negative).float()
            + 0.5 * (positive == negative).float()).mean())
        divergences.append((true_suffix_read - negative_suffix_read).square().mean())
    true_by_horizon = torch.stack(true_mse)
    negative_by_horizon = torch.stack(negative_mse)
    advantage = negative_by_horizon - true_by_horizon
    accuracy = torch.stack(pair_accuracy)
    divergence = torch.stack(divergences)

    # Algebra receipts are evaluated in FP64 rather than inheriting AMP/FP32
    # pseudoinverse roundoff from the prediction path.
    clean_initial = clean_z[:, 0].double()
    output_mean64 = memory.output_mean.to(device=device, dtype=torch.float64)
    centered = clean_initial - output_mean64
    read_matrix = memory.read_matrix.to(device=device, dtype=torch.float64)
    projector = read_matrix @ torch.linalg.pinv(read_matrix)
    complement = centered @ (torch.eye(
        memory.output_dim, device=device, dtype=torch.float64) - projector).T
    dynamic = centered @ projector.T
    projector_error = max(
        float((projector @ projector - projector).abs().max()),
        float((projector - projector.T).abs().max()),
    )
    complement_orthogonality = float((complement @ read_matrix).abs().max())
    initial_reconstruction = float((
        complement + dynamic + output_mean64 - clean_initial
    ).abs().max())
    core = memory.diagnostics()
    result: dict[str, Any] = {
        "cf_hiro_streaming_max_abs": streaming_error,
        "cf_hiro_action_derangement": "global_cyclic_episode_roll_plus_one",
        "cf_hiro_action_diagnostic_episodes": int(len(clean_z)),
        "cf_hiro_true_action_one_step_mse": float(true_by_horizon[0]),
        "cf_hiro_deranged_action_one_step_mse": float(negative_by_horizon[0]),
        "cf_hiro_true_action_one_step_advantage": float(advantage[0]),
        "cf_hiro_true_action_suffix_mse": float(true_by_horizon.mean()),
        "cf_hiro_deranged_action_suffix_mse": float(negative_by_horizon.mean()),
        "cf_hiro_true_action_suffix_advantage": float(advantage.mean()),
        "cf_hiro_action_pair_accuracy": float(accuracy.mean()),
        "cf_hiro_prior_rollout_divergence": float(divergence.mean()),
        "cf_hiro_posterior_derangement_divergence": float(
            (true_read - negative_read).square().mean()),
        "cf_hiro_action_effect_rms": float(
            ((true_actions.to(action_map) - action_mean) @ action_map.T)
            .square().mean().sqrt()
            if memory.mode != "noaction" else 0.0),
        "cf_hiro_complement_anchor_rms": float(complement.square().mean().sqrt()),
        "cf_hiro_dynamic_initial_rms": float(dynamic.square().mean().sqrt()),
        "cf_hiro_projector_algebra_max_abs": projector_error,
        "cf_hiro_complement_dynamic_orthogonality_max_abs": complement_orthogonality,
        "cf_hiro_initial_reconstruction_max_abs": initial_reconstruction,
        "cf_hiro_exact_noaction": bool(
            args.memory_mode != "cfhirov13_noaction"
            or torch.equal(state_true, state_negative)),
        "cf_hiro_exact_nocorrect": bool(
            args.memory_mode != "cfhirov13_nocorrect"
            or float((true_details.get("corrections", torch.zeros((), device=device))).abs().max())
            == 0.0),
    }
    for key, value in core.items():
        if isinstance(value, (bool, int, float, str)):
            result[f"cf_hiro_core_{key}"] = value
    for horizon in (1, 4, 8, 16, 47):
        index = min(horizon, len(advantage)) - 1
        result[f"cf_hiro_true_action_advantage_h{horizon}"] = float(
            advantage[index])
        result[f"cf_hiro_action_pair_accuracy_h{horizon}"] = float(
            accuracy[index])
        result[f"cf_hiro_action_rollout_divergence_h{horizon}"] = float(
            divergence[index])
    return result


def _run_candidate(args: argparse.Namespace) -> None:
    """Bind the candidate into the frozen V12 evaluation harness for this process."""
    replacements = {
        "OBJECTIVE": OBJECTIVE,
        "parse_args": lambda argv=None: args,
        "_validate_data_contract": validate_data_contract,
        "build_model": build_model,
        "_siro_memory": cf_hiro_memory,
        "refit_siro_operators": refit_cf_hiro_operators,
        "memory_representations": memory_representations,
        "operator_fit_payload": operator_fit_payload,
        "_scalar_fit_receipts": scalar_fit_receipts,
        "_design_metadata": design_metadata,
        "siro_diagnostics": cf_hiro_diagnostics,
    }
    original = {name: getattr(v12, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(v12, name, value)
        v12.main([])
    finally:
        for name, value in original.items():
            setattr(v12, name, value)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("invalid CF-HIRO training budget")
    if args.wandb and args.wandb_mode != "online":
        raise ValueError("CF-HIRO W&B logging must be online")
    if args.memory_mode in BASELINES:
        v11.main(_delegate_argv(args))
        return
    _run_candidate(args)


if __name__ == "__main__":
    main()
