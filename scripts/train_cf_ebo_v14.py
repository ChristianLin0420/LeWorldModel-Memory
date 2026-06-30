#!/usr/bin/env python3
"""Train the CF-EBO-v14 cross-fold-calibrated energy-bounded observer.

Every fitted memory operator is refreshed from detached, dropout-off, FP64
train-split embeddings before epoch one and after each optimizer epoch.  The
optimized graph contains only one-token next-clean prediction and unit-weight
VICReg variance/covariance terms.  Reward, simulator state, validation data,
corruption identity, and memory-specific losses never enter the fit.

The established V12 driver is reused only as the common evaluation, W&B, and
rollout harness.  Fresh V13/SSM/V8/KDIO comparators delegate to their frozen
trainers instead of being reimplemented here.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.cf_ebo import CEBOFit, fit_cf_ebo
from lewm.models.cf_hiro import HIROState
from lewm.models.memory_model import MemoryLeWorldModel
from scripts.hacssm_v11_data import (
    DEFAULT_CORRUPTION_SEED,
    DEFAULT_TRAIN_SEED,
    DEFAULT_VAL_SEED,
    V11TrajectoryDataset,
    load_cache,
)
from scripts.train_hacssm_v10 import _loader, _matched_gru_hidden
from scripts.train_cf_hiro_v13 import full_hankel_state_dim
import scripts.train_cf_hiro_v13 as v13
import scripts.train_siro_v12 as v12


CF_EBO_DESIGNS = (
    "cfebov14",
    "cfebov14_nocorrect",
    "cfebov14_noaction",
    "cfebov14_norisk",
    "cfebov14_noenergycap",
    "cfebov14_noradial",
)
BASELINES = ("cfhirov13_nocorrect", "ssm", "hacssmv8", "kdiov11")
DESIGNS = (*CF_EBO_DESIGNS, *BASELINES)
CORE_MODES = {
    "cfebov14": "full",
    "cfebov14_nocorrect": "nocorrect",
    "cfebov14_noaction": "noaction",
    "cfebov14_norisk": "norisk",
    "cfebov14_noenergycap": "noenergycap",
    "cfebov14_noradial": "noradial",
}
OBJECTIVE = "cfebov14_one_token_next_clean_plus_unit_vicreg"


class CFEBOExperimentModel(nn.Module):
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
    parser.add_argument("--output-dir", default="outputs/hacssm_v14_screen_cfebo30")
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
    parser.add_argument("--wandb-study", default="hacssm-v14-screen-cfebo30")
    parser.add_argument("--extra-tag", default="excluded-adaptive-screen")
    return parser.parse_args(argv)


def _delegate_argv(args: argparse.Namespace) -> list[str]:
    """Build an argument vector accepted unchanged by the frozen V13 driver."""
    if args.memory_mode not in BASELINES:
        raise ValueError("only registered V14 comparators may delegate")
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
            args.extra_tag, "v14-matched-v13-driver"))),
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
        raise ValueError("CF-EBO requires explicit train/validation cache roles")
    if (train.seed, val.seed) != (DEFAULT_TRAIN_SEED, DEFAULT_VAL_SEED):
        raise ValueError("CF-EBO cache seeds differ from the frozen IID protocol")
    if train.smooth_rho != 0.0 or val.smooth_rho != 0.0:
        raise ValueError(
            "CF-EBO cross-fold action-risk calibration requires IID actions with smooth_rho=0")
    fields = (
        "env_id", "length", "img_size", "action_dim", "state_dim",
        "task_observation_dim", "task_observation_keys", "task_observation_shapes")
    if tuple(getattr(train, key) for key in fields) != tuple(
            getattr(val, key) for key in fields):
        raise ValueError("CF-EBO train/validation cache schema mismatch")
    if args.eval_rollout_episode not in range(val.episodes):
        raise ValueError("evaluation rollout episode is out of range")
    args.cf_hiro_state_dim = full_hankel_state_dim(
        train.length, args.embed_dim, train.action_dim)
    args.cf_ebo_fit_path = str(train.path.resolve())
    return train, val


def build_model(args: argparse.Namespace, action_dim: int) -> CFEBOExperimentModel:
    state_dim = getattr(args, "cf_hiro_state_dim", None)
    if not isinstance(state_dim, int) or isinstance(state_dim, bool) or state_dim < 1:
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
        raise RuntimeError("CF-EBO requires causal affine-free encoding and no predictor norm")
    memory = getattr(world, "mem_cfebov14", None)
    if memory is None or memory.parameter_count() != 0:
        raise RuntimeError("CF-EBO memory must be installed with zero optimizer parameters")
    return CFEBOExperimentModel(world)


def cf_ebo_memory(model: CFEBOExperimentModel):
    memory = getattr(model.world, "mem_cfebov14", None)
    if memory is None:
        raise RuntimeError("CF-EBO experiment model is missing mem_cfebov14")
    return memory


@torch.no_grad()
def collect_detached_fit_views(
        model: CFEBOExperimentModel, clean_dataset: V11TrajectoryDataset,
        observed_dataset: V11TrajectoryDataset, args: argparse.Namespace,
        device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode aligned train-only paired views in dropout-off FP32."""
    expected_path = Path(args.cf_ebo_fit_path).resolve()
    for label, dataset, view in (
            ("clean", clean_dataset, "clean"),
            ("observed", observed_dataset, "train")):
        if dataset.metadata.split != "train" or dataset.metadata.path.resolve() != expected_path:
            raise RuntimeError(f"CF-EBO {label} fit view is not the registered train cache")
        if dataset.view != view:
            raise RuntimeError(f"CF-EBO {label} fit view must be {view!r}")
    model.eval()
    clean_chunks, observed_chunks, action_chunks, index_chunks = [], [], [], []
    clean_loader = _loader(clean_dataset, args, train=False)
    observed_loader = _loader(observed_dataset, args, train=False)
    for clean_batch, observed_batch in zip(clean_loader, observed_loader, strict=True):
        if not torch.equal(clean_batch["episode_index"], observed_batch["episode_index"]):
            raise RuntimeError("CF-EBO paired fit loaders lost episode alignment")
        if not torch.equal(clean_batch["actions"], observed_batch["actions"]):
            raise RuntimeError("CF-EBO paired fit loaders disagree on executed actions")
        if int(observed_batch["gap_start"].min()) < 5:
            raise RuntimeError("CF-EBO requires the registered exact initial observation")
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
    if not torch.equal(indices[order], torch.arange(len(clean_dataset), dtype=indices.dtype)):
        raise RuntimeError("CF-EBO fit did not consume every train episode exactly once")
    clean_z = torch.cat(clean_chunks)[order].double()
    observed_z = torch.cat(observed_chunks)[order].double()
    actions = torch.cat(action_chunks)[order].double()
    if float((clean_z[:, 0] - observed_z[:, 0]).abs().max()) != 0.0:
        raise RuntimeError("CF-EBO paired fit does not have an exact initial observation")
    return clean_z, observed_z, actions


def _fit_candidate(clean: torch.Tensor, observed: torch.Tensor,
                   actions: torch.Tensor, design: str) -> CEBOFit:
    if design not in CORE_MODES:
        raise ValueError(f"unknown CF-EBO design {design!r}")
    fit = fit_cf_ebo(clean, observed, actions, mode=CORE_MODES[design])
    if not isinstance(fit, CEBOFit):
        raise TypeError("fit_cf_ebo must return CEBOFit")
    return fit


def _fit_state_dim(fit: CEBOFit) -> int:
    for name in ("F", "state_matrix"):
        value = getattr(fit, name, None)
        if isinstance(value, torch.Tensor) and value.dim() == 2:
            return int(value.shape[0])
    raise RuntimeError("CEBOFit does not expose its state transition")


def _relative_delta(current: torch.Tensor, previous: torch.Tensor) -> float:
    denominator = previous.double().norm().clamp_min(torch.finfo(torch.float64).tiny)
    return float((current.double() - previous.double()).norm() / denominator)


def _with_receipts(fit: CEBOFit, receipts: Mapping[str, Any]) -> CEBOFit:
    if not dataclasses.is_dataclass(fit):
        raise TypeError("CEBOFit must remain a dataclass for immutable receipt updates")
    return dataclasses.replace(fit, receipts=copy.deepcopy(dict(receipts)))


@torch.no_grad()
def refit_cf_ebo_operators(
        model: CFEBOExperimentModel, clean_dataset: V11TrajectoryDataset,
        observed_dataset: V11TrajectoryDataset, args: argparse.Namespace,
        device: torch.device, *, fit_index: int,
        previous_fit: CEBOFit | None = None) -> tuple[CEBOFit, float]:
    started = time.time()
    clean, observed, actions = collect_detached_fit_views(
        model, clean_dataset, observed_dataset, args, device)
    fit = _fit_candidate(clean, observed, actions, args.memory_mode)
    state_dim = _fit_state_dim(fit)
    if state_dim != args.cf_hiro_state_dim:
        raise RuntimeError(
            f"CF-EBO fit changed fixed state schema {args.cf_hiro_state_dim}->{state_dim}")
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
    cf_ebo_memory(model).install_fit(fit)
    return fit, time.time() - started


def memory_representations(
        model: CFEBOExperimentModel, z: torch.Tensor,
        actions: torch.Tensor) -> dict[str, Any]:
    read, details = cf_ebo_memory(model)(z, actions, return_details=True)
    if not isinstance(details, Mapping):
        raise RuntimeError("CF-EBO forward details must be a mapping")
    prior = details.get("prior_reads")
    posterior = details.get("posterior_reads", details.get("reads", read))
    if not isinstance(prior, torch.Tensor) or tuple(prior.shape) != tuple(z.shape):
        raise RuntimeError("CF-EBO details must expose strict pre-observation prior_reads")
    if not isinstance(posterior, torch.Tensor) or tuple(posterior.shape) != tuple(z.shape):
        raise RuntimeError("CF-EBO details must expose posterior_reads")
    if not torch.equal(read, posterior):
        raise RuntimeError("CF-EBO direct fusion must equal its posterior read")
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


def operator_fit_payload(fit: CEBOFit) -> dict[str, Any]:
    payload = _payload(fit)
    if not isinstance(payload, dict):
        raise TypeError("CEBOFit payload must serialize to a dictionary")
    expected = {field.name for field in dataclasses.fields(fit)}
    if set(payload) != expected:
        raise RuntimeError("CEBOFit payload omitted fitted fields")
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
    _flatten_scalars(receipts, "cf_ebo_fit", flattened)
    return flattened


def scalar_core_diagnostics(memory: CF_EBOv14Memory) -> dict[str, Any]:
    """Export scalar live-state diagnostics without changing their semantics."""
    return {
        f"cf_ebo_core_{key}": value
        for key, value in memory.diagnostics().items()
        if isinstance(value, (bool, int, float, str))
    }


def design_metadata(design: str) -> dict[str, Any]:
    if design not in CF_EBO_DESIGNS:
        raise ValueError(f"unknown CF-EBO design {design!r}")
    mode = CORE_MODES[design]
    return {
        "method": "CF-EBO-v14",
        "variant": mode,
        "self_supervision": "paired_train_rgb_plus_executed_iid_actions",
        "reward_or_state_labels_used_for_fit": False,
        "validation_used_for_fit": False,
        "memory_gradient_parameter_count": 0,
        "fit_schedule": "before_epoch1_and_after_every_epoch",
        "fit_precision": "detached_fp64_cpu",
        "predictor_fusion": "direct_posterior_read_no_observation_bypass",
        "anchor_policy": "rank_aware_output_complement",
        "transition_policy": "v13_normal_blocks_padded_observable_energy_support",
        "action_policy": "zero" if mode == "noaction" else (
            "unshrunk" if mode == "norisk" else "cross_fold_calibrated_risk_shrunk"),
        "correction_policy": "zero" if mode == "nocorrect" else (
            "unit_risk_energy_bounded" if mode == "norisk"
            else "cross_fold_calibrated_energy_bounded"),
        "risk_policy": "disabled_both_paths" if mode == "norisk" else "cross_fold_eb",
        "energy_cap_policy": "disabled" if mode == "noenergycap" else "observability_energy",
        "radial_policy": "disabled" if mode == "noradial" else "self_supervised_radial",
        "cross_fitted_claim": False,
        "cross_fold_calibration": True,
    }


def _flatten_hiro_state(state: HIROState) -> tuple[HIROState, tuple[int, ...]]:
    prefix = tuple(state.mean.shape[:-1])
    return HIROState(
        state.mean.reshape(-1, state.mean.shape[-1]),
        state.complement.reshape(-1, state.complement.shape[-1])), prefix


def _reshape_hiro_state(state: HIROState, prefix: tuple[int, ...]) -> HIROState:
    return HIROState(
        state.mean.reshape(*prefix, state.mean.shape[-1]),
        state.complement.reshape(*prefix, state.complement.shape[-1]))


@torch.no_grad()
def condition_correction_evidence(
        model: CFEBOExperimentModel, dataset: V11TrajectoryDataset,
        args: argparse.Namespace, device: torch.device, use_amp: bool,
        *, label: str) -> dict[str, float]:
    """Summarize observable correction evidence for one evaluation condition."""
    memory = cf_ebo_memory(model)
    scores, gates, corrections, normalized = [], [], [], []
    amp_context = lambda: (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_amp else torch.autocast("cpu", enabled=False))
    for batch in _loader(dataset, args, train=False):
        with amp_context():
            z = model.world.encode(batch["observed"].to(device, non_blocking=True))
            _, details = memory(
                z, batch["actions"].to(device, non_blocking=True),
                return_details=True)
        required = {
            "innovation_scores": scores,
            "radial_gates": gates,
            "corrections": corrections,
            "normalized_innovations": normalized,
        }
        for key, destination in required.items():
            value = details.get(key)
            if not isinstance(value, torch.Tensor) or value.shape[:2] != z.shape[:2]:
                raise RuntimeError(
                    f"CF-EBO {label} diagnostics require {key} with a (B,T,...) prefix")
            # t=0 has no transition/correction and is excluded from evidence summaries.
            destination.append(value[:, 1:].detach().float().cpu())
    score = torch.cat(scores).reshape(-1)
    gate = torch.cat(gates).reshape(-1)
    correction = torch.cat(corrections).reshape(-1, memory.state_dim)
    normalized_innovation = torch.cat(normalized).reshape(-1, memory.output_dim)
    for name, value in (("score", score), ("gate", gate),
                        ("correction", correction),
                        ("normalized_innovation", normalized_innovation)):
        if not value.numel() or not torch.isfinite(value).all():
            raise RuntimeError(f"CF-EBO {label} {name} telemetry is empty or non-finite")
    prefix = f"cf_ebo_{label}"
    return {
        f"{prefix}_innovation_score_mean": float(score.mean()),
        f"{prefix}_innovation_score_max": float(score.max()),
        f"{prefix}_radial_gate_mean": float(gate.mean()),
        f"{prefix}_radial_gate_min": float(gate.min()),
        f"{prefix}_radial_gate_max": float(gate.max()),
        f"{prefix}_correction_rms": float(correction.square().mean().sqrt()),
        f"{prefix}_correction_norm_max": float(
            torch.linalg.vector_norm(correction, dim=-1).max()),
        f"{prefix}_correction_energy_max": float(
            correction.square().sum(dim=-1).max()),
        f"{prefix}_normalized_innovation_rms": float(
            normalized_innovation.square().mean().sqrt()),
        f"{prefix}_evidence_samples": int(score.numel()),
    }


@torch.no_grad()
def cf_ebo_diagnostics(
        model: CFEBOExperimentModel, dataset: V11TrajectoryDataset,
        fit: CEBOFit, args: argparse.Namespace, device: torch.device,
        use_amp: bool) -> dict[str, Any]:
    """Streaming, direct-sum, correction, and global action receipts."""
    del fit
    model.eval()
    memory = cf_ebo_memory(model)
    amp_context = lambda: (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_amp else torch.autocast("cpu", enabled=False))
    first = next(iter(_loader(dataset, args, train=False)))
    frames = first["clean"][:2].to(device)
    actions = first["actions"][:2].to(device)
    with amp_context():
        z = model.world.encode(frames)
        batch_read, _ = memory(z, actions, return_details=True)
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
    posterior_means = true_details.get("state_means")
    complement_anchors = true_details.get("complement_anchors")
    if (not isinstance(posterior_means, torch.Tensor)
            or not isinstance(complement_anchors, torch.Tensor)):
        raise RuntimeError("CF-EBO diagnostics require state/complement sequences")

    state_true = HIROState(posterior_means[:, :-1], complement_anchors[:, :-1])
    state_negative = state_true
    true_mse, negative_mse, pair_accuracy, divergences = [], [], [], []
    horizon = true_actions.shape[1]
    for offset in range(horizon):
        valid = horizon - offset
        state_true = HIROState(
            state_true.mean[:, :valid], state_true.complement[:, :valid])
        state_negative = HIROState(
            state_negative.mean[:, :valid], state_negative.complement[:, :valid])
        true_action = true_actions[:, offset:offset + valid]
        negative_action = negative_actions[:, offset:offset + valid]

        flat_true, prefix = _flatten_hiro_state(state_true)
        flat_true = memory.transition(flat_true, true_action.reshape(-1, true_action.shape[-1]))
        state_true = _reshape_hiro_state(flat_true, prefix)
        if memory.mode == "noaction":
            state_negative = state_true
        else:
            flat_negative, prefix = _flatten_hiro_state(state_negative)
            flat_negative = memory.transition(
                flat_negative, negative_action.reshape(-1, negative_action.shape[-1]))
            state_negative = _reshape_hiro_state(flat_negative, prefix)
        flat_true, prefix = _flatten_hiro_state(state_true)
        flat_negative, _ = _flatten_hiro_state(state_negative)
        true_suffix_read = memory.read_state(flat_true).reshape(
            *prefix, memory.output_dim)
        negative_suffix_read = memory.read_state(flat_negative).reshape(
            *prefix, memory.output_dim)
        target = clean_z[:, offset + 1:offset + 1 + valid]
        positive = (true_suffix_read - target).square().mean(dim=-1)
        negative = (negative_suffix_read - target).square().mean(dim=-1)
        true_mse.append(positive.mean())
        negative_mse.append(negative.mean())
        pair_accuracy.append(((positive < negative).float()
                              + .5 * (positive == negative).float()).mean())
        divergences.append((true_suffix_read - negative_suffix_read).square().mean())
    true_by_horizon = torch.stack(true_mse)
    negative_by_horizon = torch.stack(negative_mse)
    advantage = negative_by_horizon - true_by_horizon
    accuracy = torch.stack(pair_accuracy)
    divergence = torch.stack(divergences)

    initial = memory.initial_state(clean_z[:, 0])
    initial_read = memory.read_state(initial)
    output_mean = getattr(memory, "output_mean", torch.zeros(
        memory.output_dim, device=device)).to(initial_read)
    dynamic_initial = initial_read - output_mean - initial.complement.to(initial_read)
    corrections = true_details.get("corrections", torch.zeros((), device=device))
    action_effects = true_details.get("action_effects", torch.zeros((), device=device))
    result: dict[str, Any] = {
        "cf_ebo_streaming_max_abs": streaming_error,
        "cf_ebo_action_derangement": "global_cyclic_episode_roll_plus_one",
        "cf_ebo_action_diagnostic_episodes": int(len(clean_z)),
        "cf_ebo_true_action_one_step_mse": float(true_by_horizon[0]),
        "cf_ebo_deranged_action_one_step_mse": float(negative_by_horizon[0]),
        "cf_ebo_true_action_one_step_advantage": float(advantage[0]),
        "cf_ebo_true_action_suffix_mse": float(true_by_horizon.mean()),
        "cf_ebo_deranged_action_suffix_mse": float(negative_by_horizon.mean()),
        "cf_ebo_true_action_suffix_advantage": float(advantage.mean()),
        "cf_ebo_action_pair_accuracy": float(accuracy.mean()),
        "cf_ebo_prior_rollout_divergence": float(divergence.mean()),
        "cf_ebo_action_effect_rms": float(action_effects.float().square().mean().sqrt()),
        "cf_ebo_correction_rms": float(corrections.float().square().mean().sqrt()),
        "cf_ebo_complement_anchor_rms": float(
            initial.complement.float().square().mean().sqrt()),
        "cf_ebo_dynamic_initial_rms": float(
            dynamic_initial.float().square().mean().sqrt()),
        "cf_ebo_initial_reconstruction_max_abs": float(
            (initial_read.float() - clean_z[:, 0].float()).abs().max()),
        "cf_ebo_exact_noaction": bool(
            args.memory_mode != "cfebov14_noaction"
            or float(action_effects.abs().max()) == 0.0),
        "cf_ebo_exact_nocorrect": bool(
            args.memory_mode != "cfebov14_nocorrect"
            or float(corrections.abs().max()) == 0.0),
    }
    # Core details may expose energy/radial scales. Aggregate every finite tensor
    # without baking private key names into the trainer.
    for key, value in true_details.items():
        if (isinstance(value, torch.Tensor) and value.is_floating_point()
                and value.numel() and torch.isfinite(value).all()
                and any(token in key for token in (
                    "energy", "scale", "alpha", "radial", "score", "normalized"))):
            result[f"cf_ebo_{key}_mean"] = float(value.float().mean())
            result[f"cf_ebo_{key}_min"] = float(value.float().min())
            result[f"cf_ebo_{key}_max"] = float(value.float().max())
    result.update(scalar_core_diagnostics(memory))
    for selected_horizon in (1, 4, 8, 16, 47):
        index = min(selected_horizon, len(advantage)) - 1
        result[f"cf_ebo_true_action_advantage_h{selected_horizon}"] = float(
            advantage[index])
        result[f"cf_ebo_action_pair_accuracy_h{selected_horizon}"] = float(
            accuracy[index])
        result[f"cf_ebo_action_rollout_divergence_h{selected_horizon}"] = float(
            divergence[index])
    evidence_datasets = {
        "clean": dataset,
        "val_train_view": V11TrajectoryDataset(
            args.val_data, "train", args.corruption_seed, args.history_len),
        **{
            condition: V11TrajectoryDataset(
                args.val_data, condition, args.corruption_seed, args.history_len)
            for condition in v12.HELDOUT_CONDITIONS
        },
    }
    for label, evidence_dataset in evidence_datasets.items():
        result.update(condition_correction_evidence(
            model, evidence_dataset, args, device, use_amp, label=label))
    return result


def _run_candidate(args: argparse.Namespace) -> None:
    """Bind CF-EBO into the common V12 evaluation harness for this process."""
    replacements = {
        "OBJECTIVE": OBJECTIVE,
        "parse_args": lambda argv=None: args,
        "_validate_data_contract": validate_data_contract,
        "build_model": build_model,
        "_siro_memory": cf_ebo_memory,
        "refit_siro_operators": refit_cf_ebo_operators,
        "memory_representations": memory_representations,
        "operator_fit_payload": operator_fit_payload,
        "_scalar_fit_receipts": scalar_fit_receipts,
        "_design_metadata": design_metadata,
        "siro_diagnostics": cf_ebo_diagnostics,
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
        raise ValueError("invalid CF-EBO training budget")
    if args.wandb and args.wandb_mode != "online":
        raise ValueError("CF-EBO W&B logging must be online")
    if args.memory_mode in BASELINES:
        v13.main(_delegate_argv(args))
        return
    _run_candidate(args)


if __name__ == "__main__":
    main()
