#!/usr/bin/env python3
"""Train CVPF-v15 with a differentiable cross-fold filtration envelope.

Deployment operators are refreshed from aligned, detached, dropout-off FP64
train embeddings before epoch one and after every epoch.  In contrast to V14,
the optimized graph also contains a unit-weight, symmetric within-minibatch
future-prediction envelope.  Its OAS regressions are differentiated through in
the nominated design; ``detachid`` removes only those identification gradients
and ``noenvelope`` removes the envelope exactly.

The frozen V12 program remains the common W&B, probe, rollout, and artifact
harness.  Every comparator delegates through the frozen V14/V13 drivers.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.cvpf import CVPFFit, CVPFState, fit_cvpf
from lewm.models.memory_model import MemoryLeWorldModel
from scripts.hacssm_v11_data import (
    DEFAULT_CORRUPTION_SEED,
    DEFAULT_TRAIN_SEED,
    DEFAULT_VAL_SEED,
    V11TrajectoryDataset,
    load_cache,
)
from scripts.train_hacssm_v10 import _loader, _matched_gru_hidden
import scripts.train_cf_ebo_v14 as v14
import scripts.train_hacssm_v11 as v11
import scripts.train_siro_v12 as v12


CVPF_DESIGNS = (
    "cvpfv15",
    "cvpfv15_nocorrect",
    "cvpfv15_noaction",
    "cvpfv15_norisk",
    "cvpfv15_norho",
    "cvpfv15_anchoronly",
    "cvpfv15_detachid",
    "cvpfv15_noenvelope",
)
BASELINES = (
    "cfebov14_norisk",
    "cfhirov13_nocorrect",
    "ssm",
    "hacssmv8",
    "kdiov11",
)
DESIGNS = (*CVPF_DESIGNS, *BASELINES)
CORE_MODES = {
    "cvpfv15": "full",
    "cvpfv15_nocorrect": "nocorrect",
    "cvpfv15_noaction": "noaction",
    "cvpfv15_norisk": "norisk",
    "cvpfv15_norho": "norho",
    "cvpfv15_anchoronly": "anchoronly",
    "cvpfv15_detachid": "full",
    "cvpfv15_noenvelope": "full",
}
OBJECTIVE = "cvpfv15_one_token_unit_vicreg_plus_unit_cross_fold_filtration"
HISTORY_KEYS = (
    "loss", "predictive_loss", "context_loss", "variance_loss",
    "covariance_loss", "filtration_loss")


class CVPFExperimentModel(nn.Module):
    def __init__(self, world: MemoryLeWorldModel):
        super().__init__()
        self.world = world

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--memory-mode", choices=DESIGNS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/hacssm_v15_screen_cvpf30")
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
    parser.add_argument("--wandb-study", default="hacssm-v15-screen-cvpf30")
    parser.add_argument("--extra-tag", default="excluded-adaptive-screen")
    return parser.parse_args(argv)


def _delegate_argv(args: argparse.Namespace) -> list[str]:
    """Pass a V15 comparator unchanged into the frozen V14 driver."""
    if args.memory_mode not in BASELINES:
        raise ValueError("only registered V15 comparators may delegate")
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
            args.extra_tag, "v15-matched-frozen-v14-driver"))),
    }
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
        raise ValueError("CVPF requires explicit train/validation cache roles")
    if (train.seed, val.seed) != (DEFAULT_TRAIN_SEED, DEFAULT_VAL_SEED):
        raise ValueError("CVPF cache seeds differ from the frozen IID protocol")
    if train.smooth_rho != 0.0 or val.smooth_rho != 0.0:
        raise ValueError("CVPF action certificates require IID actions with smooth_rho=0")
    fields = (
        "env_id", "length", "img_size", "action_dim", "state_dim",
        "task_observation_dim", "task_observation_keys", "task_observation_shapes")
    if tuple(getattr(train, key) for key in fields) != tuple(
            getattr(val, key) for key in fields):
        raise ValueError("CVPF train/validation cache schema mismatch")
    if train.length < 2:
        raise ValueError("CVPF requires at least one future step")
    if args.eval_rollout_episode not in range(val.episodes):
        raise ValueError("evaluation rollout episode is out of range")
    args.cvpf_horizon = int(train.length - 1)
    args.cvpf_fit_path = str(train.path.resolve())
    return train, val


def build_model(args: argparse.Namespace, action_dim: int) -> CVPFExperimentModel:
    horizon = getattr(args, "cvpf_horizon", None)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        horizon = int(load_cache(args.train_data).length - 1)
        args.cvpf_horizon = horizon
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
        cvpf_horizon=horizon,
    )
    if world.encoder_norm != "causal" or world.predictor_norm != "none":
        raise RuntimeError("CVPF requires causal affine-free encoding and no predictor norm")
    memory = getattr(world, "mem_cvpfv15", None)
    if memory is None or memory.parameter_count() != 0:
        raise RuntimeError("CVPF memory must be installed with zero optimizer parameters")
    return CVPFExperimentModel(world)


def cvpf_memory(model: CVPFExperimentModel):
    memory = getattr(model.world, "mem_cvpfv15", None)
    if memory is None:
        raise RuntimeError("CVPF experiment model is missing mem_cvpfv15")
    return memory


@torch.no_grad()
def collect_detached_fit_views(
        model: CVPFExperimentModel, clean_dataset: V11TrajectoryDataset,
        observed_dataset: V11TrajectoryDataset, args: argparse.Namespace,
        device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode every aligned train-only clean/train-view episode in dropout-off FP32."""
    expected_path = Path(args.cvpf_fit_path).resolve()
    for label, dataset, view in (
            ("clean", clean_dataset, "clean"),
            ("observed", observed_dataset, "train")):
        if dataset.metadata.split != "train" or dataset.metadata.path.resolve() != expected_path:
            raise RuntimeError(f"CVPF {label} fit view is not the registered train cache")
        if dataset.view != view:
            raise RuntimeError(f"CVPF {label} fit view must be {view!r}")
    model.eval()
    clean_chunks, observed_chunks, action_chunks, index_chunks = [], [], [], []
    for clean_batch, observed_batch in zip(
            _loader(clean_dataset, args, train=False),
            _loader(observed_dataset, args, train=False), strict=True):
        if not torch.equal(clean_batch["episode_index"], observed_batch["episode_index"]):
            raise RuntimeError("CVPF paired fit loaders lost episode alignment")
        if not torch.equal(clean_batch["actions"], observed_batch["actions"]):
            raise RuntimeError("CVPF paired fit loaders disagree on executed actions")
        if int(observed_batch["gap_start"].min()) < 5:
            raise RuntimeError("CVPF requires the registered exact initial observation")
        with torch.autocast(device_type=device.type, enabled=False):
            clean_z = model.world.encode(
                clean_batch["clean"].to(device, non_blocking=True).float())
            observed_z = model.world.encode(
                observed_batch["observed"].to(device, non_blocking=True).float())
        clean_chunks.append(clean_z.float().cpu())
        observed_chunks.append(observed_z.float().cpu())
        action_chunks.append(clean_batch["actions"].float().cpu())
        index_chunks.append(clean_batch["episode_index"].cpu())
    indices = torch.cat(index_chunks)
    order = torch.argsort(indices)
    expected = torch.arange(len(clean_dataset), dtype=indices.dtype)
    if not torch.equal(indices[order], expected):
        raise RuntimeError("CVPF fit did not consume every train episode exactly once")
    clean = torch.cat(clean_chunks)[order].double()
    observed = torch.cat(observed_chunks)[order].double()
    actions = torch.cat(action_chunks)[order].double()
    if float((clean[:, 0] - observed[:, 0]).abs().max()) != 0.0:
        raise RuntimeError("CVPF paired fit does not have an exact initial observation")
    return clean, observed, actions


def _with_receipts(fit: CVPFFit, receipts: Mapping[str, Any]) -> CVPFFit:
    if not dataclasses.is_dataclass(fit):
        raise TypeError("CVPFFit must remain a dataclass for immutable receipt updates")
    return dataclasses.replace(fit, receipts=copy.deepcopy(dict(receipts)))


def _relative_delta(current: torch.Tensor, previous: torch.Tensor) -> float:
    denominator = previous.double().norm().clamp_min(torch.finfo(torch.float64).tiny)
    return float((current.double() - previous.double()).norm() / denominator)


@torch.no_grad()
def refit_cvpf_operators(
        model: CVPFExperimentModel, clean_dataset: V11TrajectoryDataset,
        observed_dataset: V11TrajectoryDataset, args: argparse.Namespace,
        device: torch.device, *, fit_index: int,
        previous_fit: CVPFFit | None = None) -> tuple[CVPFFit, float]:
    started = time.time()
    clean, observed, actions = collect_detached_fit_views(
        model, clean_dataset, observed_dataset, args, device)
    fit = fit_cvpf(clean, observed, actions, mode=CORE_MODES[args.memory_mode])
    if not isinstance(fit, CVPFFit):
        raise TypeError("fit_cvpf must return CVPFFit")
    receipts = dict(fit.receipts)
    receipts.update({
        "fit_index": int(fit_index),
        "fit_split": "train_only",
        "fit_uses_validation": False,
        "fit_uses_reward": False,
        "fit_uses_task_state": False,
        "fit_uses_corruption_identity": False,
        "fit_precision": "detached_fp64_cpu",
        "fit_episode_count": int(clean.shape[0]),
        "fit_length": int(clean.shape[1]),
        "fit_design": args.memory_mode,
        "global_deployment_fit_gradient_active": False,
        "minibatch_envelope_fit_gradient_active": (
            args.memory_mode not in ("cvpfv15_detachid", "cvpfv15_noenvelope")),
        "fit_gradient_active": (
            args.memory_mode not in ("cvpfv15_detachid", "cvpfv15_noenvelope")),
    })
    receipts["refit_has_previous"] = previous_fit is not None
    if previous_fit is not None:
        for field in dataclasses.fields(fit):
            current = getattr(fit, field.name)
            previous = getattr(previous_fit, field.name, None)
            if (isinstance(current, torch.Tensor) and isinstance(previous, torch.Tensor)
                    and current.shape == previous.shape and current.is_floating_point()):
                receipts[f"refit_{field.name}_relative_delta"] = _relative_delta(
                    current, previous)
    fit = _with_receipts(fit, receipts)
    cvpf_memory(model).install_fit(fit)
    return fit, time.time() - started


def memory_representations(
        model: CVPFExperimentModel, z: torch.Tensor,
        actions: torch.Tensor) -> dict[str, Any]:
    read, details = cvpf_memory(model)(z, actions, return_details=True)
    if not isinstance(details, Mapping):
        raise RuntimeError("CVPF forward details must be a mapping")
    prior = details.get("prior_reads")
    posterior = details.get("posterior_reads", details.get("reads", read))
    if not isinstance(prior, torch.Tensor) or tuple(prior.shape) != tuple(z.shape):
        raise RuntimeError("CVPF details must expose strict pre-observation prior_reads")
    if not isinstance(posterior, torch.Tensor) or tuple(posterior.shape) != tuple(z.shape):
        raise RuntimeError("CVPF details must expose posterior_reads")
    if not torch.equal(read, posterior):
        raise RuntimeError("CVPF direct fusion must equal its posterior current-output read")
    return {"fused": read, "prior": prior, "posterior": posterior, "details": details}


class _LinearMap(NamedTuple):
    x_mean: torch.Tensor
    y_mean: torch.Tensor
    weight: torch.Tensor


class _EnvelopeFit(NamedTuple):
    anchor: _LinearMap
    action_residualizer: _LinearMap
    action: _LinearMap
    correction: _LinearMap


def _oas_linear_map(x: torch.Tensor, y: torch.Tensor) -> _LinearMap:
    """Differentiable OAS linear conditional mean with only a machine floor."""
    if x.dim() != 2 or y.dim() != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("OAS map requires aligned rank-two tensors with at least two rows")
    x = x.float()
    y = y.float()
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    xc = x - x_mean
    yc = y - y_mean
    count, dimension = x.shape
    covariance = (xc.T @ xc) / count
    covariance = 0.5 * (covariance + covariance.T)
    alpha = covariance.square().mean()
    target_variance = torch.trace(covariance) / dimension
    target_squared = target_variance.square()
    numerator = alpha + target_squared
    denominator = (count + 1) * (alpha - target_squared / dimension)
    safe_denominator = torch.where(
        denominator <= 0, torch.ones_like(denominator), denominator)
    shrinkage = torch.where(
        denominator <= 0,
        torch.ones_like(denominator),
        (numerator / safe_denominator).clamp(0.0, 1.0))
    identity = torch.eye(dimension, device=x.device, dtype=x.dtype)
    covariance = ((1.0 - shrinkage) * covariance
                  + shrinkage * target_variance * identity)
    scale = covariance.diagonal().abs().mean().clamp_min(torch.finfo(x.dtype).eps)
    covariance = covariance + (
        torch.finfo(x.dtype).eps * dimension * scale) * identity
    cross = (xc.T @ yc) / count
    weight = torch.linalg.solve(covariance, cross)
    return _LinearMap(x_mean, y_mean, weight)


def _apply_linear_map(fit: _LinearMap, value: torch.Tensor) -> torch.Tensor:
    return fit.y_mean + (value - fit.x_mean) @ fit.weight


def _detach_envelope_fit(fit: _EnvelopeFit) -> _EnvelopeFit:
    def detached(value: _LinearMap) -> _LinearMap:
        return _LinearMap(*(tensor.detach() for tensor in value))
    return _EnvelopeFit(*(detached(value) for value in fit))


def _fit_envelope_half(
        clean: torch.Tensor, observed: torch.Tensor, actions: torch.Tensor,
        *, use_action: bool, use_correction: bool) -> _EnvelopeFit:
    current = clean[:, :-1].reshape(-1, clean.shape[-1])
    target = clean[:, 1:].reshape(-1, clean.shape[-1])
    flat_actions = actions.reshape(-1, actions.shape[-1])
    anchor = _oas_linear_map(current, target)
    anchor_prediction = _apply_linear_map(anchor, current)
    action_residualizer = _oas_linear_map(current, flat_actions)
    action_residual = flat_actions - _apply_linear_map(action_residualizer, current)
    action_target = target - anchor_prediction
    action = _oas_linear_map(action_residual, action_target)
    if not use_action:
        action = _LinearMap(
            action.x_mean, torch.zeros_like(action.y_mean),
            torch.zeros_like(action.weight))
    prior = anchor_prediction + _apply_linear_map(action, action_residual)
    # The paired observed view enters only after the anchor/action information set.
    innovation = observed[:, 1:].reshape(-1, clean.shape[-1]) - prior
    correction_target = target - prior
    correction = _oas_linear_map(innovation, correction_target)
    if not use_correction:
        correction = _LinearMap(
            correction.x_mean, torch.zeros_like(correction.y_mean),
            torch.zeros_like(correction.weight))
    return _EnvelopeFit(anchor, action_residualizer, action, correction)


def _fit_envelope(
        clean: torch.Tensor, observed: torch.Tensor, actions: torch.Tensor,
        *, use_action: bool, use_correction: bool) -> _EnvelopeFit:
    return _fit_envelope_half(
        clean, observed, actions,
        use_action=use_action, use_correction=use_correction)


def _score_envelope_half(
        fit: _EnvelopeFit, clean: torch.Tensor, observed: torch.Tensor,
        actions: torch.Tensor) -> torch.Tensor:
    """Mean recursive error over the complete valid source/future triangle."""
    length = clean.shape[1]
    states = clean[:, :-1]
    squared_error = clean.new_zeros(())
    scalar_count = 0
    for offset in range(length - 1):
        valid = length - 1 - offset
        current = states[:, :valid]
        action = actions[:, offset:offset + valid]
        anchor = _apply_linear_map(fit.anchor, current)
        action_residual = action - _apply_linear_map(
            fit.action_residualizer, current)
        prior = anchor + _apply_linear_map(fit.action, action_residual)
        observation = observed[:, offset + 1:offset + 1 + valid]
        innovation = observation - prior
        posterior = prior + _apply_linear_map(fit.correction, innovation)
        target = clean[:, offset + 1:offset + 1 + valid]
        squared_error = squared_error + (posterior - target).square().sum()
        scalar_count += posterior.numel()
        states = posterior
    if scalar_count == 0:
        raise RuntimeError("filtration envelope has no valid future suffix")
    return squared_error / scalar_count


def filtration_envelope_loss(
        clean: torch.Tensor, observed: torch.Tensor, actions: torch.Tensor,
        design: str) -> torch.Tensor:
    """Symmetric differentiable fit-on-one-half/score-on-the-other objective."""
    if design not in CVPF_DESIGNS:
        raise ValueError(f"unknown CVPF envelope design {design!r}")
    if design == "cvpfv15_noenvelope":
        return clean.sum() * 0.0
    if (clean.dim() != 3 or observed.shape != clean.shape
            or actions.shape[:2] != (clean.shape[0], clean.shape[1] - 1)
            or clean.shape[0] < 4):
        raise ValueError("CVPF envelope requires B>=4 aligned (B,L,D)/(B,L-1,A) tensors")
    use_action = design not in ("cvpfv15_noaction", "cvpfv15_anchoronly")
    use_correction = design not in ("cvpfv15_nocorrect", "cvpfv15_anchoronly")
    detach_identification = design == "cvpfv15_detachid"
    even = torch.arange(clean.shape[0], device=clean.device) % 2 == 0
    losses = []
    for fit_mask, score_mask in ((even, ~even), (~even, even)):
        fit = _fit_envelope(
            clean[fit_mask], observed[fit_mask], actions[fit_mask],
            use_action=use_action, use_correction=use_correction)
        if detach_identification:
            fit = _detach_envelope_fit(fit)
        losses.append(_score_envelope_half(
            fit, clean[score_mask], observed[score_mask], actions[score_mask]))
    return torch.stack(losses).mean()


def compute_cvpf_losses(
        model: CVPFExperimentModel, observed: torch.Tensor,
        clean: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
    clean_z = v11.encode_clean_active(model, clean)
    observed_z = model.world.encode(observed)
    memory = memory_representations(model, observed_z, actions)
    prediction = v11.one_token_prediction(
        model.world, memory["fused"][:, :-1], actions)
    predictive_loss = F.mse_loss(prediction.float(), clean_z[:, 1:].float())
    variance_loss, covariance_loss = v11._vicreg_terms(clean_z)
    with torch.autocast(device_type=clean_z.device.type, enabled=False):
        filtration_loss = filtration_envelope_loss(
            clean_z.float(), observed_z.float(), actions.float(),
            model.world.memory_impl)
    return {
        "loss": predictive_loss + variance_loss + covariance_loss + filtration_loss,
        "predictive_loss": predictive_loss,
        "context_loss": predictive_loss,
        "variance_loss": variance_loss,
        "covariance_loss": covariance_loss,
        "filtration_loss": filtration_loss,
    }


def run_epoch(
        model: CVPFExperimentModel, loader,
        optimizer: torch.optim.Optimizer | None, device: torch.device,
        use_amp: bool) -> dict[str, float]:
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
                losses = compute_cvpf_losses(model, observed, clean, actions)
            if train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = observed.shape[0]
            for key in HISTORY_KEYS:
                totals[key] += float(losses[key].detach()) * batch_size
            count += batch_size
    if not count:
        raise RuntimeError("empty CVPF epoch")
    return {key: value / count for key, value in totals.items()}


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


def operator_fit_payload(fit: CVPFFit) -> dict[str, Any]:
    payload = _payload(fit)
    if not isinstance(payload, dict):
        raise TypeError("CVPFFit payload must serialize to a dictionary")
    expected = {field.name for field in dataclasses.fields(fit)}
    if set(payload) != expected:
        raise RuntimeError("CVPFFit payload omitted fitted fields")
    return payload


def _flatten_scalars(value: Any, prefix: str, result: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten_scalars(item, f"{prefix}_{key}" if prefix else str(key), result)
    elif isinstance(value, (bool, int, float, str)):
        if not isinstance(value, float) or math.isfinite(value):
            result[prefix] = value


def scalar_fit_receipts(receipts: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    _flatten_scalars(receipts, "cvpf_fit", result)
    return result


def design_metadata(design: str) -> dict[str, Any]:
    if design not in CVPF_DESIGNS:
        raise ValueError(f"unknown CVPF design {design!r}")
    mode = CORE_MODES[design]
    envelope = design != "cvpfv15_noenvelope"
    fit_gradient = envelope and design != "cvpfv15_detachid"
    return {
        "method": "CVPF-v15",
        "variant": design.removeprefix("cvpfv15").lstrip("_") or "full",
        "core_mode": mode,
        "self_supervision": "paired_train_rgb_plus_executed_iid_actions",
        "reward_or_state_labels_used_for_fit": False,
        "validation_used_for_fit": False,
        "memory_gradient_parameter_count": 0,
        "fit_schedule": "before_epoch1_and_after_every_epoch",
        "fit_precision": "detached_fp64_cpu",
        "predictor_fusion": "direct_posterior_current_clean_coordinate",
        "future_horizon_policy": "all_available_valid_suffixes",
        "statistical_regularization": "oas_no_tuned_ridge",
        "cross_fold_policy": "symmetric_even_odd_episode_halves",
        "envelope_weight": 1.0 if envelope else 0.0,
        "memory_specific_loss_weight": 1.0 if envelope else 0.0,
        "fit_gradient_active": fit_gradient,
        "identified_fit_gradient_active": fit_gradient,
        "deployment_global_fit_gradient_active": False,
        "operator_gradient_control": (
            "inactive" if not envelope else
            "detached" if design == "cvpfv15_detachid" else "live"),
        "action_policy": "zero" if mode in ("noaction", "anchoronly") else "identified",
        "correction_policy": "zero" if mode in ("nocorrect", "anchoronly") else "identified",
        "risk_policy": "unit" if mode == "norisk" else "opposite_fold_per_mode",
        "rho_policy": "unit" if mode == "norho" else "normalized_pls_strength",
        "cross_fitted_claim": False,
        "cross_fold_calibration": True,
        "cvpf_identification_detached": design == "cvpfv15_detachid",
        "cvpf_envelope_active": envelope,
        "cvpf_envelope_weight": 1.0 if envelope else 0.0,
        "cvpf_exact_noaction": mode in ("noaction", "anchoronly"),
        "cvpf_exact_nocorrect": mode in ("nocorrect", "anchoronly"),
        "cvpf_exact_norisk": mode == "norisk",
        "cvpf_exact_norho": mode == "norho",
        "cvpf_exact_anchoronly": mode == "anchoronly",
    }


def scalar_core_diagnostics(memory) -> dict[str, Any]:
    """Export the canonical analyzer key plus namespaced finite core scalars."""
    diagnostics = memory.diagnostics()
    shift_closure = diagnostics.get("shift_closure_max_abs")
    if not isinstance(shift_closure, (int, float)) or not math.isfinite(
            float(shift_closure)):
        raise RuntimeError("CVPF core omitted finite projected-shift closure")
    result: dict[str, Any] = {
        "cvpf_shift_closure_relative": float(shift_closure)}
    for key, value in diagnostics.items():
        if isinstance(value, (bool, int, float, str)):
            if not isinstance(value, float) or math.isfinite(value):
                result[f"cvpf_core_{key}"] = value
    return result


def recursive_action_suffix_diagnostics(
        memory, clean_z: torch.Tensor, true_actions: torch.Tensor,
        negative_actions: torch.Tensor, details: Mapping[str, Any]) -> dict[str, Any]:
    """Observation-free all-source suffix comparison under true/deranged actions."""
    required = [details.get(key) for key in (
        "anchor_states", "action_states", "observation_states")]
    if any(not isinstance(value, torch.Tensor) for value in required):
        raise RuntimeError("CVPF suffix audit requires all three posterior role states")
    anchor, action_state, observation_state = (
        value[:, :-1].float().clone() for value in required)
    # At t=0 the stored anchor already forecasts t=1 because z0 is emitted by
    # exact override.  States recorded at t>=1 still decode their current block
    # and must be shifted once before the first future action.
    if anchor.shape[1] > 1:
        anchor[:, 1:] = F.linear(anchor[:, 1:], memory.anchor_shift.float())
        action_state[:, 1:] = F.linear(
            action_state[:, 1:], memory.action_shift.float())
        observation_state[:, 1:] = F.linear(
            observation_state[:, 1:], memory.observation_shift.float())
    true_state = CVPFState(anchor, action_state, observation_state)
    negative_state = CVPFState(
        anchor.clone(), action_state.clone(), observation_state.clone())
    true_mse, negative_mse, accuracies, divergences = [], [], [], []
    horizon = true_actions.shape[1]

    def advance(state: CVPFState, action: torch.Tensor):
        score = F.linear(
            action.float() - memory.action_source_mean.float(),
            memory.action_encoder.float())
        current_action = state.action + score
        read = (
            memory.output_mean.float()
            + F.linear(state.anchor, memory.anchor_decoder[0].float())
            + F.linear(current_action, memory.action_decoder[0].float())
            + F.linear(
                state.observation, memory.observation_decoder[0].float()))
        shifted = CVPFState(
            F.linear(state.anchor, memory.anchor_shift.float()),
            F.linear(current_action, memory.action_shift.float()),
            F.linear(
                state.observation, memory.observation_shift.float()))
        return read, shifted

    for offset in range(horizon):
        valid = horizon - offset
        true_state = CVPFState(*(value[:, :valid] for value in true_state))
        negative_state = CVPFState(*(value[:, :valid] for value in negative_state))
        true_read, true_state = advance(
            true_state, true_actions[:, offset:offset + valid])
        negative_read, negative_state = advance(
            negative_state, negative_actions[:, offset:offset + valid])
        target = clean_z[:, offset + 1:offset + 1 + valid].float()
        positive = (true_read - target).square().mean(dim=-1)
        negative = (negative_read - target).square().mean(dim=-1)
        true_mse.append(positive.mean())
        negative_mse.append(negative.mean())
        accuracies.append(((positive < negative).float()
                           + .5 * (positive == negative).float()).mean())
        divergences.append((true_read - negative_read).square().mean())
    true_by_horizon = torch.stack(true_mse)
    negative_by_horizon = torch.stack(negative_mse)
    advantage = negative_by_horizon - true_by_horizon
    accuracy = torch.stack(accuracies)
    divergence = torch.stack(divergences)
    result = {
        "cvpf_true_action_suffix_mse": float(true_by_horizon.mean()),
        "cvpf_deranged_action_suffix_mse": float(negative_by_horizon.mean()),
        "cvpf_true_action_suffix_advantage": float(advantage.mean()),
        "cvpf_action_pair_accuracy": float(accuracy.mean()),
        "cvpf_prior_rollout_divergence": float(divergence.mean()),
        "cvpf_action_suffix_horizons": int(horizon),
        "cvpf_action_suffix_semantics": (
            "all_sources_observation_free_equal_horizon_mean"),
    }
    for selected in (1, 4, 8, 16, 47):
        index = min(selected, horizon) - 1
        result[f"cvpf_true_action_advantage_h{selected}"] = float(advantage[index])
        result[f"cvpf_action_pair_accuracy_h{selected}"] = float(accuracy[index])
        result[f"cvpf_action_rollout_divergence_h{selected}"] = float(divergence[index])
    return result


@torch.no_grad()
def cvpf_diagnostics(
        model: CVPFExperimentModel, dataset: V11TrajectoryDataset,
        fit: CVPFFit, args: argparse.Namespace, device: torch.device,
        use_amp: bool) -> dict[str, Any]:
    """Audit streaming, path activity, initial exactness, and action sensitivity."""
    del fit
    model.eval()
    memory = cvpf_memory(model)
    amp_context = lambda: (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_amp else torch.autocast("cpu", enabled=False))
    first = next(iter(_loader(dataset, args, train=False)))
    frames = first["clean"][:2].to(device)
    actions = first["actions"][:2].to(device)
    with amp_context():
        z = model.world.encode(frames)
        batch_read, batch_details = memory(z, actions, return_details=True)
        state = memory.initial_state(z[:, 0])
        streamed = [z[:, 0]]
        for step in range(1, z.shape[1]):
            output = memory.step(
                state, z[:, step], actions[:, step - 1], return_details=True)
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError("CVPF step must expose at least (read,state)")
            read, state = output[:2]
            streamed.append(read)
    streaming_error = float((torch.stack(streamed, dim=1).float()
                             - batch_read.float()).abs().max())
    prefix_error = 0.0
    with amp_context():
        for end in range(1, z.shape[1] + 1):
            prefix_read = memory(z[:, :end], actions[:, :max(end - 1, 0)])
            prefix_error = max(prefix_error, float((
                prefix_read.float() - batch_read[:, :end].float()).abs().max()))

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
    true_prior = true_details["prior_reads"][:, 1:].float()
    negative_prior = negative_details["prior_reads"][:, 1:].float()
    target = clean_z[:, 1:].float()
    positive = (true_prior - target).square().mean(dim=-1)
    negative = (negative_prior - target).square().mean(dim=-1)
    pair_accuracy = ((positive < negative).float()
                     + .5 * (positive == negative).float()).mean()

    result: dict[str, Any] = {
        "cvpf_streaming_max_abs": streaming_error,
        "cvpf_prefix_closure_max_abs": prefix_error,
        "cvpf_initial_reconstruction_max_abs": float(
            (true_read[:, 0].float() - clean_z[:, 0].float()).abs().max()),
        "cvpf_true_action_prior_mse": float(positive.mean()),
        "cvpf_deranged_action_prior_mse": float(negative.mean()),
        "cvpf_true_action_prior_advantage": float(negative.mean() - positive.mean()),
        "cvpf_one_step_action_pair_accuracy": float(pair_accuracy),
        "cvpf_one_step_prior_divergence": float(
            (true_prior - negative_prior).square().mean()),
        "cvpf_action_derangement": "global_cyclic_episode_roll_plus_one",
        "cvpf_action_diagnostic_episodes": int(len(clean_z)),
    }
    for key in (
            "anchor_states", "action_states", "observation_states", "innovations",
            "action_effects", "corrections"):
        value = true_details.get(key)
        if isinstance(value, torch.Tensor) and value.numel():
            value = value.float()
            if not torch.isfinite(value).all():
                raise RuntimeError(f"CVPF non-finite {key} telemetry")
            result[f"cvpf_{key}_rms"] = float(value.square().mean().sqrt())
            result[f"cvpf_{key}_max_abs"] = float(value.abs().max())
    action_effects = true_details.get("action_effects")
    corrections = true_details.get("corrections")
    action_is_zero = bool(
        isinstance(action_effects, torch.Tensor)
        and float(action_effects.abs().max()) == 0.0)
    correction_is_zero = bool(
        isinstance(corrections, torch.Tensor)
        and float(corrections.abs().max()) == 0.0)
    result["cvpf_exact_noaction"] = bool(
        args.memory_mode in ("cvpfv15_noaction", "cvpfv15_anchoronly")
        and action_is_zero)
    result["cvpf_exact_nocorrect"] = bool(
        args.memory_mode in ("cvpfv15_nocorrect", "cvpfv15_anchoronly")
        and correction_is_zero)
    result["cvpf_exact_anchoronly"] = bool(
        args.memory_mode == "cvpfv15_anchoronly"
        and action_is_zero and correction_is_zero)
    result["cvpf_exact_norisk"] = args.memory_mode == "cvpfv15_norisk"
    result["cvpf_exact_norho"] = args.memory_mode == "cvpfv15_norho"
    result["cvpf_identification_detached"] = (
        args.memory_mode == "cvpfv15_detachid")
    result["cvpf_envelope_active"] = args.memory_mode != "cvpfv15_noenvelope"
    result["cvpf_envelope_weight"] = (
        0.0 if args.memory_mode == "cvpfv15_noenvelope" else 1.0)
    result.update(recursive_action_suffix_diagnostics(
        memory, clean_z, true_actions, negative_actions, true_details))
    # The top-level alias is the analyzer/auditor contract.  The core keeps the
    # historical max-over-role name, while this metric states explicitly that
    # it is a relative decoded-suffix residual.
    result.update(scalar_core_diagnostics(memory))
    return result


def _run_candidate(args: argparse.Namespace) -> None:
    """Bind CVPF into the common V12 evaluation harness for this process."""
    replacements = {
        "OBJECTIVE": OBJECTIVE,
        "parse_args": lambda argv=None: args,
        "_validate_data_contract": validate_data_contract,
        "build_model": build_model,
        "_siro_memory": cvpf_memory,
        "refit_siro_operators": refit_cvpf_operators,
        "memory_representations": memory_representations,
        "compute_siro_losses": compute_cvpf_losses,
        "HISTORY_KEYS": HISTORY_KEYS,
        "run_epoch": run_epoch,
        "operator_fit_payload": operator_fit_payload,
        "_scalar_fit_receipts": scalar_fit_receipts,
        "_design_metadata": design_metadata,
        "siro_diagnostics": cvpf_diagnostics,
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
    if args.epochs < 1 or args.batch_size < 4 or args.num_workers < 0:
        raise ValueError("CVPF requires epochs>=1, batch_size>=4, and num_workers>=0")
    if args.wandb and args.wandb_mode != "online":
        raise ValueError("CVPF W&B logging must be online")
    if args.memory_mode in BASELINES:
        v14.main(_delegate_argv(args))
        return
    _run_candidate(args)


if __name__ == "__main__":
    main()
