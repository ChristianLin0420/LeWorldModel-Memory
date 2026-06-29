#!/usr/bin/env python3
"""Train and evaluate the KDIO-v11 predictive-state study.

V11 makes the fused recurrent belief the sole temporal input to LeWM's predictor:
every ``(belief_t, action_t)`` pair is decoded independently to the next clean online
embedding.  KDIO additionally transports each *deployed observed posterior* through
every available future action suffix and matches the reached state to the synchronized
clean embedding.  Detached symmetric positive/negative trajectories rank relative latent
displacements by a parameter-free energy ratio under a cyclic episode derangement.  Inverse
action decodability is measured only after training by a clean-train ridge probe.  A detached
clean-view branch self-calibrates KDIO's full-rank
innovation precision and mean in the affine-free ``D-1`` contrast subspace by an
epoch-end closed-form OAS Gaussian fit; it supplies no corruption label, calibration
learning rate, loss weight, tuned ridge, or
visibility mask and cannot change the encoder through that likelihood. No simulator state or reward enters optimization. The flattened native DMC
task observation is read only after training to fit train-split ridge probes; raw
simulator physics state remains archived but unused.
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
    _phase_masks,
    _probe_predict,
    _r2,
    encoder_diagnostics,
    orbit_diagnostics,
)


DESIGNS = (
    "ssm",
    "hacssmv8",
    "orbitv10",
    "kdiov11",
    "kdiov11_unconstrained",
    "kdiov11_fixedscale",
    "kdiov11_nocalibration",
    "kdiov11_diagonal",
    "kdiov11_h1",
    "kdiov11_firstorder",
    "kdiov11_nodrift",
    "kdiov11_noautonomy",
    "kdiov11_noaction",
    "kdiov11_noactionswap",
    "kdiov11_nosuffix",
    "kdiov11_noreliability",
)
KDIO_DESIGNS = frozenset(design for design in DESIGNS if design.startswith("kdiov11"))
SUFFIX_DESIGNS = KDIO_DESIGNS - {"kdiov11_nosuffix"}
ACTION_SWAP_DESIGNS = SUFFIX_DESIGNS
CALIBRATED_DESIGNS = KDIO_DESIGNS - {"kdiov11_nocalibration"}
HELDOUT_CONDITIONS = ("freeze", "gaussian_noise", "checkerboard", "long_freeze")
ROLLOUT_SCHEMA_VERSION = 2
ACTION_RANKING_MODES = (
    "relative_displacement_detached",
    "relative_endpoint_detached",
    "rawdiff_displacement_detached",
    "relative_displacement_livegamma",
)
DEFAULT_ACTION_RANKING = ACTION_RANKING_MODES[0]
OBJECTIVE = (
    "v11b_scaled_stiefel_detached_relative_displacement_suffix_ranking_oas")


def _model_impl(design: str) -> tuple[str, str]:
    if design not in DESIGNS:
        raise ValueError(f"unknown V11 design {design!r}")
    # ``nosuffix`` is an objective-only control: its deployed inference path is exactly
    # full KDIO, while the trainer below removes the suffix term.
    objective_only = {
        "kdiov11_nosuffix", "kdiov11_nocalibration", "kdiov11_diagonal"}
    return ("kdiov11" if design in objective_only else design), "both"


def _design_metadata(
        design: str, action_ranking_mode: str = DEFAULT_ACTION_RANKING
        ) -> dict[str, Any]:
    is_kdio = design in KDIO_DESIGNS
    if action_ranking_mode not in ACTION_RANKING_MODES:
        raise ValueError(f"unknown V11 action-ranking mode {action_ranking_mode!r}")
    has_suffix = design in SUFFIX_DESIGNS
    rank_optimized = has_suffix and design != "kdiov11_noactionswap"
    rank_action_active = rank_optimized and design != "kdiov11_noaction"
    variant = design.removeprefix("kdiov11_") if design != "kdiov11" else "full"
    return {
        "memory_arch_schema_version": 11 if is_kdio else None,
        "memory_architecture": "kick_drift_innovation_observer" if is_kdio else design,
        "memory_v11_variant": variant if is_kdio else None,
        "action_frame": (
            "learned_scale_frobenius_normalized_free_geometry"
            if design == "kdiov11_unconstrained" else
            "fixed_unit_scale_canonical_thin_qr"
            if design == "kdiov11_fixedscale" else
            "learned_positive_scale_canonical_thin_qr" if is_kdio else
            "not_applicable"),
        "action_scale_initialization": 1.0 if is_kdio else None,
        "action_scale_parameterization": (
            "exp_unclipped_fp32" if is_kdio and design != "kdiov11_fixedscale" else
            "exact_one_log_scale_tensor_retained" if is_kdio else
            "not_applicable"),
        "suffix_rollout_applicable": has_suffix,
        "suffix_horizon_policy": (
            "h1_only" if design == "kdiov11_h1" else
            "all_available_equal_horizon_weight" if has_suffix else
            "not_applicable_suffix_equals_context"
        ),
        "suffix_anchor": (
            "observed_corrupted_online_posterior" if has_suffix else "not_applicable"),
        "action_swap_applicable": design in ACTION_SWAP_DESIGNS,
        "action_swap_gradient_active": (
            rank_action_active),
        "action_swap_negative": (
            "cyclic_batch_deranged_suffix" if design in ACTION_SWAP_DESIGNS else
            "not_applicable"),
        "development_action_ranking": action_ranking_mode,
        "action_rank_diagnostic_computed": has_suffix,
        "action_rank_optimized": rank_optimized,
        "action_rank_geometry": (
            "displacement" if "displacement" in action_ranking_mode else "endpoint"),
        "action_rank_energy": (
            "log_ratio" if action_ranking_mode.startswith("relative_") else
            "raw_difference"),
        "action_rank_legacy_metric_alias_semantics": (
            "rank_energy_not_live_endpoint_suffix_mse" if has_suffix else
            "not_applicable"),
        "action_rank_source_gradient_active": False,
        "action_rank_target_gradient_active": False,
        "action_rank_scale_parameter_retained": is_kdio,
        "action_rank_direction_parameter_retained": is_kdio,
        "action_rank_transition_parameters_retained": is_kdio,
        "action_rank_scale_gradient_active": (
            rank_action_active and design != "kdiov11_fixedscale"
            and action_ranking_mode == "relative_displacement_livegamma"),
        "action_rank_direction_gradient_active": rank_action_active,
        "action_rank_transition_gradient_active": (
            rank_action_active and design != "kdiov11_noautonomy"),
        "inverse_gradient_active": False,
        "inverse_evaluation": "clean_train_ridge_three_frame_to_action",
        "innovation_calibration": (
            "identity_mu0_CI" if design == "kdiov11_nocalibration" else
            "epoch_end_reliability_open_clean_diagonal_oas"
            if design == "kdiov11_diagonal" else
            "epoch_end_reliability_open_clean_full_oas" if is_kdio else
            "not_applicable"),
        "encoder_trained_end_to_end": True,
        "target_stop_gradient": False,
        "training_objective": OBJECTIVE,
    }


class V11ExperimentModel(nn.Module):
    """World model; all optimized parameters participate in deployed prediction."""

    def __init__(self, world: MemoryLeWorldModel):
        super().__init__()
        self.world = world

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)


def build_model(args: argparse.Namespace, action_dim: int, action_mean: np.ndarray,
                action_std: np.ndarray) -> V11ExperimentModel:
    memory_impl, memory_mode = _model_impl(args.memory_mode)
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
        raise RuntimeError("V11 requires causal affine-free encoder norm and no predictor norm")
    return V11ExperimentModel(world)


def encode_clean_active(model: V11ExperimentModel, clean: torch.Tensor) -> torch.Tensor:
    """Dropout-off same-encoder target pass with gradients retained."""
    encoder = model.world.encoder
    was_training = encoder.training
    encoder.eval()
    try:
        return model.world.encode(clean)
    finally:
        encoder.train(was_training)


def one_token_prediction(world: MemoryLeWorldModel, beliefs: torch.Tensor,
                         actions: torch.Tensor) -> torch.Tensor:
    """Decode each belief independently; no Transformer temporal bypass is possible."""
    batch, steps, dimension = beliefs.shape
    if tuple(actions.shape[:2]) != (batch, steps):
        raise ValueError("belief/action alignment mismatch")
    prediction = world.predictor(
        beliefs.reshape(batch * steps, 1, dimension),
        actions.reshape(batch * steps, 1, actions.shape[-1]),
    )[:, -1]
    return prediction.reshape(batch, steps, dimension)


def _rms_read(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)


def memory_representations(model: V11ExperimentModel, z: torch.Tensor,
                           actions: torch.Tensor) -> dict[str, Any]:
    """Return deployed fusion plus strictly pre/post-observation D-dimensional reads.

    The primary ``prior`` never includes ``z_t``. All recurrent families use their best
    native transition prior in a common RMS-normalized D coordinate. The warm-start at
    t=0 is excluded by the probe burn-in and is therefore not an estimand.
    """
    world = model.world
    if world.memory_impl == "ssm":
        states = world.mem_ssm(z)
        decay = torch.sigmoid(world.mem_ssm.raw_decay).to(dtype=states.dtype)
        prior = torch.zeros_like(states)
        bias = world.mem_ssm.in_proj.bias.to(dtype=states.dtype)
        prior[:, 1:] = (1.0 - decay) * states[:, :-1] + decay * bias
        return {
            "fused": world.mem_ssm.fuse(z, states),
            "prior": _rms_read(prior),
            "posterior": _rms_read(states),
            "details": {"states": states, "priors": prior},
        }
    if world.memory_impl == "hacssmv8":
        mixed, details = world.mem_hacssmv8(z, actions, return_details=True)
        route = details["route"].to(device=z.device, dtype=z.dtype).view(1, 1, -1, 1)
        prior = (details["priors"] * route).sum(dim=2)
        posterior = (details["states"] * route).sum(dim=2)
        return {
            "fused": world.mem_hacssmv8.fuse(z, mixed),
            "prior": _rms_read(prior), "posterior": _rms_read(posterior),
            "details": details,
        }
    if world.memory_impl == "orbitv10":
        mixed, details = world.mem_orbitv10(z, actions, return_details=True)
        return {
            "fused": world.mem_orbitv10.fuse(z, mixed),
            "prior": _rms_read(details["priors"]),
            "posterior": _rms_read(details["states"]),
            "details": details,
        }
    if world.memory_impl in KDIO_DESIGNS:
        mixed, details = world.mem_kdiov11(z, actions, return_details=True)
        return {
            "fused": world.mem_kdiov11.fuse(z, mixed),
            "prior": world.mem_kdiov11.read_state(details["priors"]),
            "posterior": world.mem_kdiov11.read_state(details["states"]),
            "details": details,
        }
    raise ValueError(f"V11 has no prior-read contract for {world.memory_impl!r}")


def _vicreg_terms(clean_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = clean_z.reshape(-1, clean_z.shape[-1]).float()
    centered = features - features.mean(dim=0, keepdim=True)
    variance = centered.square().sum(dim=0) / max(len(centered) - 1, 1)
    variance_loss = torch.relu(
        1.0 - torch.sqrt(variance + torch.finfo(variance.dtype).eps)).mean()
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = off_diagonal.square().sum() / clean_z.shape[-1]
    return variance_loss, covariance_loss


def _second_order_inverse_inputs(
        clean_z: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align ``(z[t-1],z[t],z[t+1])`` with executed transition action ``a[t]``."""
    if clean_z.shape[1] < 3 or actions.shape[1] != clean_z.shape[1] - 1:
        raise ValueError("second-order inverse dynamics requires T>=3 and T-1 actions")
    inputs = torch.cat(
        (clean_z[:, :-2], clean_z[:, 1:-1], clean_z[:, 2:]), dim=-1)
    return inputs, actions[:, 1:]


@torch.no_grad()
def _clean_calibration_priors(model: V11ExperimentModel, clean_z: torch.Tensor,
                              actions: torch.Tensor) -> torch.Tensor:
    """C-independent clean priors from the deployed recurrence with reliability fixed open.

    The observer retains its learned prior-conditioned ordered base gains, dynamics, and
    streaming state.  Only ``r`` is forced to one, so C/mu cannot alter the distribution they
    fit while the calibration path remains the intended clean deployed operating path.
    """
    memory = model.world.mem_kdiov11
    _, details = memory(
        clean_z, actions, reliability_override=1.0, return_details=True)
    return details["q_priors"][:, 1:]


def _action_rank_pair_loss(positive_energy: torch.Tensor,
                           negative_energy: torch.Tensor,
                           action_ranking_mode: str = DEFAULT_ACTION_RANKING
                           ) -> torch.Tensor:
    """Return the parameter-free pair loss before horizon/example reduction."""
    if action_ranking_mode not in ACTION_RANKING_MODES:
        raise ValueError(f"unknown V11 action-ranking mode {action_ranking_mode!r}")
    if positive_energy.shape != negative_energy.shape:
        raise ValueError("positive and negative action-rank energies must align")
    if action_ranking_mode == "rawdiff_displacement_detached":
        return F.softplus(positive_energy - negative_energy)
    tiny = torch.finfo(positive_energy.dtype).tiny
    positive_log_energy = torch.log(positive_energy.clamp_min(tiny))
    negative_log_energy = (
        positive_log_energy if negative_energy is positive_energy else
        torch.log(negative_energy.clamp_min(tiny)))
    return F.softplus(
        positive_log_energy - negative_log_energy)


def _kdio_suffix_objectives(
        model: V11ExperimentModel, states: torch.Tensor,
        actions: torch.Tensor, clean_z: torch.Tensor, *, h1_only: bool,
        action_ranking_mode: str = DEFAULT_ACTION_RANKING,
        negative_actions: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor | int]:
    """Equal-horizon live prediction and detached symmetric action ranking.

    One live positive trajectory supplies the suffix MSE and therefore trains the observed
    source, clean target, and action scale.  A separate positive/negative pair begins at the
    same detached observed posterior.  Its target and (by default) scale are detached, while
    gradients still reach action geometry and the shared autonomous transition/read mechanisms.
    The development modes alter only rank geometry, energy, or scale detachment; they do not
    register new designs.
    """
    memory = model.world.mem_kdiov11
    batch, length, two, dimension = states.shape
    if two != 2:
        raise ValueError(f"KDIO state must be (B,T,2,D), got {tuple(states.shape)}")
    horizon = length - 1
    if horizon < 1:
        raise ValueError("KDIO suffix loss requires at least two frames")
    expected_actions = (batch, horizon, memory.action_dim)
    if tuple(actions.shape) != expected_actions:
        raise ValueError(
            f"KDIO actions must have shape {expected_actions}, got {tuple(actions.shape)}")
    if action_ranking_mode not in ACTION_RANKING_MODES:
        raise ValueError(f"unknown V11 action-ranking mode {action_ranking_mode!r}")
    if negative_actions is None:
        if batch < 2:
            raise ValueError("batch derangement requires at least two rows")
        negative_actions = torch.roll(actions, shifts=1, dims=0)
    elif tuple(negative_actions.shape) != expected_actions:
        raise ValueError(
            "explicit negative actions must match true actions, got "
            f"{tuple(negative_actions.shape)} != {expected_actions}")

    # U is one property of this rollout graph. Scaling actions, rather than U, permits the
    # positive reconstruction and rank paths to share geometry while choosing gamma gradients
    # independently. Validation happens once, outside the triangular scan.
    action_direction = memory._validate_cached_action_frame(memory.action_direction())
    live_scale = memory.action_scale()
    rank_scale = (
        live_scale if action_ranking_mode == "relative_displacement_livegamma"
        else live_scale.detach())
    live_current = states[:, :-1]
    rank_source = states[:, :-1].detach()
    rank_positive_current = rank_source
    rank_negative_current = rank_source
    rank_source_read = memory.read_state(rank_source).float()
    clean_source = clean_z[:, :-1].detach().float()
    live_energies = []
    rank_positive_energies = []
    rank_negative_energies = []
    ranking_losses = []
    pair_accuracies = []
    divergence_squared = []
    active_horizons = 1 if h1_only else horizon
    for offset in range(active_horizons):
        valid_sources = horizon - offset
        live_current = live_current[:, :valid_sources]
        rank_positive_current = rank_positive_current[:, :valid_sources]
        rank_negative_current = rank_negative_current[:, :valid_sources]
        combined_state = torch.cat(
            (live_current, rank_positive_current, rank_negative_current), dim=0)
        positive_actions = actions[:, offset:offset + valid_sources]
        deranged_actions = negative_actions[:, offset:offset + valid_sources]
        combined_actions = torch.cat((
            positive_actions * live_scale,
            positive_actions * rank_scale,
            deranged_actions * rank_scale,
        ), dim=0)
        combined_next = memory._transition_prevalidated(
            combined_state.reshape(3 * batch * valid_sources, 2, dimension),
            combined_actions.reshape(
                3 * batch * valid_sources, actions.shape[-1]),
            action_direction,
        ).reshape(3 * batch, valid_sources, 2, dimension)
        live_current, rank_positive_current, rank_negative_current = (
            combined_next.split(batch, dim=0))

        live_target = clean_z[:, offset + 1:offset + 1 + valid_sources].float()
        live_read = memory.read_state(live_current).float()
        live_energy = (live_read - live_target).square().mean(dim=-1)
        positive_read = memory.read_state(rank_positive_current).float()
        negative_read = memory.read_state(rank_negative_current).float()
        rank_target = live_target.detach()
        if "displacement" in action_ranking_mode:
            shared_source = rank_source_read[:, :valid_sources]
            positive_read = positive_read - shared_source
            negative_read = negative_read - shared_source
            rank_target = (
                rank_target - clean_source[:, :valid_sources]).detach()
        positive = (positive_read - rank_target).square().mean(dim=-1)
        negative = (negative_read - rank_target).square().mean(dim=-1)
        if memory.mode == "noaction":
            # The intervention removes the only difference between the pair. Sharing the
            # energy node makes ln(2), 0.5 accuracy, and cancellation gradients exact rather
            # than relying on two floating-point-identical graphs to cancel after reduction.
            negative = positive
        live_energies.append(live_energy.mean())
        rank_positive_energies.append(positive.mean())
        rank_negative_energies.append(negative.mean())
        ranking_losses.append(_action_rank_pair_loss(
            positive, negative, action_ranking_mode).mean())
        pair_accuracies.append((
            (positive < negative).float()
            + 0.5 * (positive == negative).float()).mean())
        divergence_squared.append(
            (positive_read - negative_read).square().mean())
    live_by_horizon = torch.stack(live_energies)
    positive_by_horizon = torch.stack(rank_positive_energies)
    negative_by_horizon = torch.stack(rank_negative_energies)
    positive_energy = positive_by_horizon.mean()
    negative_energy = negative_by_horizon.mean()
    return {
        "suffix_loss": live_by_horizon.mean(),
        "action_swap_loss": torch.stack(ranking_losses).mean(),
        "action_swap_positive_energy": positive_energy,
        "action_swap_negative_energy": negative_energy,
        "action_swap_advantage": negative_energy - positive_energy,
        "action_swap_pair_accuracy": torch.stack(pair_accuracies).mean(),
        "live_energy_by_horizon": live_by_horizon,
        "positive_energy_by_horizon": positive_by_horizon,
        "negative_energy_by_horizon": negative_by_horizon,
        "ranking_loss_by_horizon": torch.stack(ranking_losses),
        "pair_accuracy_by_horizon": torch.stack(pair_accuracies),
        "divergence_squared_by_horizon": torch.stack(divergence_squared),
        "horizons": active_horizons,
    }


def compute_v11_losses(model: V11ExperimentModel, observed: torch.Tensor,
                       clean: torch.Tensor, actions: torch.Tensor,
                       design: str,
                       action_ranking_mode: str = DEFAULT_ACTION_RANKING
                       ) -> dict[str, torch.Tensor]:
    clean_z = encode_clean_active(model, clean)
    observed_z = model.world.encode(observed)
    observed_memory = memory_representations(model, observed_z, actions)
    belief = observed_memory["fused"]
    # The suffix source is the same observed/corrupted posterior deployed online.  Using a
    # clean source leaves the innovation gates outside their only long-horizon objective and
    # creates a train/deployment mismatch precisely inside the memory gaps.
    details = observed_memory["details"] if design in KDIO_DESIGNS else None
    context_prediction = one_token_prediction(model.world, belief[:, :-1], actions)
    context_loss = F.mse_loss(context_prediction.float(), clean_z[:, 1:].float())

    if design in SUFFIX_DESIGNS:
        suffix = _kdio_suffix_objectives(
            model, details["states"], actions, clean_z,
            h1_only=design == "kdiov11_h1",
            action_ranking_mode=action_ranking_mode,
        )
        suffix_loss = suffix["suffix_loss"]
        raw_action_swap_loss = suffix["action_swap_loss"]
        action_swap_positive_energy = suffix["action_swap_positive_energy"]
        action_swap_negative_energy = suffix["action_swap_negative_energy"]
        action_swap_advantage = suffix["action_swap_advantage"]
        action_swap_pair_accuracy = suffix["action_swap_pair_accuracy"]
        suffix_horizons = int(suffix["horizons"])
        suffix_applicable = context_loss.new_ones(())
        action_swap_applicable = context_loss.new_ones(())
    else:
        # Literal identity is part of the V11 contract, not a second copy of the loss.
        # This applies both to non-KDIO references and to the full-inference ``nosuffix``
        # objective control.
        suffix_loss = context_loss
        suffix_horizons = 0
        suffix_applicable = context_loss.new_zeros(())
        raw_action_swap_loss = context_loss.detach() * 0.0
        action_swap_positive_energy = context_loss.detach() * 0.0
        action_swap_negative_energy = context_loss.detach() * 0.0
        action_swap_advantage = context_loss.detach() * 0.0
        action_swap_pair_accuracy = context_loss.detach() * 0.0
        action_swap_applicable = context_loss.new_zeros(())
    predictive_loss = 0.5 * (context_loss + suffix_loss)
    action_swap_loss = (
        raw_action_swap_loss.detach() * 0.0
        if design == "kdiov11_noactionswap" else raw_action_swap_loss)
    if design in KDIO_DESIGNS:
        # Fit uncertainty to clean innovations without letting the encoder shrink or rotate
        # them to game the likelihood.  C and mu affect deployed gating as detached statistics;
        # predictive gradients instead tune the prior-conditioned process tolerance.
        memory = model.world.mem_kdiov11
        with torch.no_grad():
            clean_q_priors = _clean_calibration_priors(
                model, clean_z.detach(), actions.detach())
        raw_calibration_nll = memory.clean_innovation_nll(
            clean_z[:, 1:].detach(), clean_q_priors.detach())
        if design in CALIBRATED_DESIGNS:
            calibration_nll = raw_calibration_nll
            calibration_applicable = context_loss.new_ones(())
        else:
            calibration_nll = raw_calibration_nll * 0.0
            calibration_applicable = context_loss.new_zeros(())
    else:
        calibration_nll = context_loss.detach() * 0.0
        calibration_applicable = context_loss.new_zeros(())
    variance_loss, covariance_loss = _vicreg_terms(clean_z)
    total = predictive_loss + action_swap_loss + variance_loss + covariance_loss
    return {
        "loss": total,
        "predictive_loss": predictive_loss,
        "context_loss": context_loss,
        "suffix_loss": suffix_loss,
        "action_swap_loss": action_swap_loss,
        "action_swap_diagnostic_loss": raw_action_swap_loss.detach(),
        "action_swap_positive_energy": action_swap_positive_energy,
        "action_swap_negative_energy": action_swap_negative_energy,
        "action_swap_advantage": action_swap_advantage,
        "action_swap_pair_accuracy": action_swap_pair_accuracy,
        "action_swap_applicable": action_swap_applicable,
        "action_swap_horizons": context_loss.new_tensor(float(suffix_horizons)),
        "calibration_nll": calibration_nll,
        "calibration_applicable": calibration_applicable,
        "variance_loss": variance_loss,
        "covariance_loss": covariance_loss,
        "suffix_applicable": suffix_applicable,
        "suffix_horizons": context_loss.new_tensor(float(suffix_horizons)),
    }


HISTORY_KEYS = (
    "loss", "predictive_loss", "context_loss", "suffix_loss", "action_swap_loss",
    "action_swap_diagnostic_loss", "action_swap_positive_energy",
    "action_swap_negative_energy", "action_swap_advantage",
    "action_swap_pair_accuracy", "action_swap_applicable", "action_swap_horizons",
    "calibration_nll", "calibration_applicable",
    "variance_loss", "covariance_loss", "suffix_applicable", "suffix_horizons",
)


def run_epoch(model: V11ExperimentModel, loader, optimizer: torch.optim.Optimizer | None,
              device: torch.device, use_amp: bool, design: str,
              action_ranking_mode: str = DEFAULT_ACTION_RANKING
              ) -> dict[str, float]:
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
                losses = compute_v11_losses(
                    model, observed, clean, actions, design, action_ranking_mode)
            if train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = observed.shape[0]
            for key in HISTORY_KEYS:
                totals[key] += float(losses[key].detach()) * batch_size
            count += batch_size
    if not count:
        raise RuntimeError("empty V11 epoch")
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def calibrate_clean_statistics(
        model: V11ExperimentModel, dataset: V11TrajectoryDataset,
        args: argparse.Namespace, device: torch.device, use_amp: bool,
        *, diagonal_only: bool = False
        ) -> dict[str, float]:
    """Fit the final-model clean innovation distribution once after a train epoch."""
    model.eval()
    memory = model.world.mem_kdiov11
    memory.reset_clean_calibration()
    for batch in _loader(dataset, args, train=False):
        clean = batch["clean"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp else torch.autocast("cpu", enabled=False))
        with amp_context:
            clean_z = model.world.encode(clean)
            q_priors = _clean_calibration_priors(model, clean_z, actions)
        memory.accumulate_clean_calibration(clean_z[:, 1:], q_priors)
    return memory.finalize_clean_calibration(diagonal_only=diagonal_only)


def _fit_ridge(features: np.ndarray, states: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    x_mean = features.mean(axis=0, dtype=np.float64)
    x_std = features.std(axis=0, dtype=np.float64).clip(min=1e-6)
    y_mean = states.mean(axis=0, dtype=np.float64)
    y_std = states.std(axis=0, dtype=np.float64).clip(min=1e-6)
    x = (features.astype(np.float64) - x_mean) / x_std
    y = (states.astype(np.float64) - y_mean) / y_std
    design = np.concatenate((x, np.ones((len(x), 1), dtype=np.float64)), axis=1)
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(gram + penalty, design.T @ y)
    return {key: value.astype(np.float32) for key, value in {
        "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
        "weights": weights,
    }.items()}


@torch.no_grad()
def collect_representations(model: V11ExperimentModel, dataset: V11TrajectoryDataset,
                            args: argparse.Namespace, device: torch.device,
                            use_amp: bool) -> dict[str, np.ndarray]:
    model.eval()
    chunks = {
        "prior": [], "posterior": [], "encoder": [], "predictor": [], "state": []}
    h = args.history_len
    for batch in _loader(dataset, args, train=False):
        frames = batch["clean"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp else torch.autocast("cpu", enabled=False))
        with amp_context:
            z = model.world.encode(frames)
            memory = memory_representations(model, z, actions)
            prediction = one_token_prediction(model.world, memory["fused"][:, :-1], actions)
        chunks["encoder"].append(z[:, h:].float().cpu().numpy())
        chunks["prior"].append(memory["prior"][:, h:].float().cpu().numpy())
        chunks["posterior"].append(memory["posterior"][:, h:].float().cpu().numpy())
        chunks["predictor"].append(prediction[:, h - 1:].float().cpu().numpy())
        if args.eval_target_key not in batch:
            raise RuntimeError(
                f"V11 evaluation target {args.eval_target_key!r} is absent from the dataset")
        chunks["state"].append(batch[args.eval_target_key][:, h:].float().numpy())
    result = {}
    for key, values in chunks.items():
        width = values[0].shape[-1] if key == "state" else args.embed_dim
        result[key] = np.concatenate(values).reshape(-1, width)
    return result


def fit_state_probes(model: V11ExperimentModel, dataset: V11TrajectoryDataset,
                     args: argparse.Namespace, device: torch.device,
                     use_amp: bool) -> dict[str, dict[str, np.ndarray]]:
    representations = collect_representations(model, dataset, args, device, use_amp)
    return {
        name: _fit_ridge(representations[name], representations["state"], args.probe_ridge)
        for name in ("prior", "posterior", "encoder", "predictor")
    }


def _state_metrics(prediction: torch.Tensor, target: torch.Tensor,
                   probe: dict[str, np.ndarray], masks: Mapping[str, torch.Tensor]
                   ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    y_std = torch.as_tensor(probe["y_std"], device=target.device, dtype=torch.float32)
    per_step = ((prediction.float() - target.float()) / y_std).square().mean(dim=-1)
    errors = {name: per_step[mask].cpu().numpy() for name, mask in masks.items()
              if bool(mask.any())}
    primary = masks["primary"]
    return errors, prediction[primary].float().cpu().numpy(), target[primary].float().cpu().numpy()


@torch.no_grad()
def evaluate_condition(model: V11ExperimentModel, dataset: V11TrajectoryDataset,
                       probes: Mapping[str, dict[str, np.ndarray]], args: argparse.Namespace,
                       device: torch.device, use_amp: bool,
                       rollout_episode: int | None = None):
    model.eval()
    phase_chunks = {
        coordinate: {phase: [] for phase in ("gap", "deep", "first_post", "post", "primary")}
        for coordinate in ("prior", "posterior", "encoder", "predictor")
    }
    primary_prediction = {coordinate: [] for coordinate in phase_chunks}
    primary_target = {coordinate: [] for coordinate in phase_chunks}
    observer_keys = (
        "innovation_ratio", "innovation_energy", "process_tolerance", "reliability",
        "position_base_gain", "velocity_base_gain", "velocity_base_ratio",
        "q_gates", "v_gates", "action_effect_norm",
        "action_tanh_derivative_mean", "action_tanh_saturation_proxy",
    )
    observer_phases = ("all", "gap", "deep", "first_post", "post", "primary")
    observer_chunks = (
        {key: {phase: [] for phase in observer_phases} for key in observer_keys}
        if model.world.memory_impl in KDIO_DESIGNS else None)
    ordered_violation = 0.0
    rollout = None
    h = args.history_len
    for batch in _loader(dataset, args, train=False):
        observed = batch["observed"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        if args.eval_target_key not in batch:
            raise RuntimeError(
                f"V11 evaluation target {args.eval_target_key!r} is absent from the dataset")
        state = batch[args.eval_target_key].to(device, non_blocking=True)
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp else torch.autocast("cpu", enabled=False))
        with amp_context:
            z = model.world.encode(observed)
            memory = memory_representations(model, z, actions)
            predicted = one_token_prediction(model.world, memory["fused"][:, :-1], actions)
        features = {
            "encoder": z[:, h:].float(),
            "prior": memory["prior"][:, h:].float(),
            "posterior": memory["posterior"][:, h:].float(),
            "predictor": predicted[:, h - 1:].float(),
        }
        target = state[:, h:].float()
        masks = _phase_masks(batch, dataset.metadata.length, h, device)
        masks["primary"] = masks["deep"] | masks["first_post"]
        if observer_chunks is not None:
            observer_masks = dict(masks)
            observer_masks["all"] = torch.ones_like(masks["primary"], dtype=torch.bool)
            details = memory["details"]
            for key in observer_keys:
                values = details[key][:, h:].float()
                if values.shape[-1:] == (1,):
                    values = values.squeeze(-1)
                if values.shape != observer_masks["all"].shape:
                    raise RuntimeError(
                        f"observer diagnostic {key} has shape {tuple(values.shape)}")
                for phase, mask in observer_masks.items():
                    if bool(mask.any()):
                        observer_chunks[key][phase].append(values[mask].cpu().numpy())
            violation = torch.relu(
                details["v_gates"][:, h:].float()
                - details["q_gates"][:, h:].float()).max()
            ordered_violation = max(ordered_violation, float(violation))
        state_predictions = {}
        errors_by_coordinate = {}
        for coordinate in features:
            state_prediction = _probe_predict(features[coordinate], probes[coordinate])
            state_predictions[coordinate] = state_prediction
            errors, selected_prediction, selected_target = _state_metrics(
                state_prediction, target, probes[coordinate], masks)
            errors_by_coordinate[coordinate] = (
                (state_prediction - target).float().square().mean(dim=-1))
            for phase, value in errors.items():
                phase_chunks[coordinate][phase].append(value)
            primary_prediction[coordinate].append(selected_prediction)
            primary_target[coordinate].append(selected_target)

        if rollout_episode is not None:
            episode_indices = batch["episode_index"].numpy()
            matches = np.nonzero(episode_indices == rollout_episode)[0]
            if len(matches):
                row = int(matches[0])
                times = np.arange(h, dataset.metadata.length, dtype=np.int64)
                start, end = int(batch["gap_start"][row]), int(batch["gap_end"][row])
                phases = np.asarray([
                    "deep" if start + h <= t < end else
                    "gap" if start <= t < end else
                    "first_post" if t == end else
                    "post" if end < t <= end + h else "context"
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
                    "evaluation_target": target[row].cpu().numpy().astype(np.float32),
                }
                for coordinate in features:
                    prediction = state_predictions[coordinate][row].cpu().numpy().astype(np.float32)
                    y_std = np.asarray(probes[coordinate]["y_std"], dtype=np.float32)
                    error = np.square(
                        (prediction - rollout["evaluation_target"]) / y_std).mean(axis=-1)
                    rollout[f"{coordinate}_state_prediction"] = prediction
                    rollout[f"{coordinate}_state_nmse_by_target_t"] = error.astype(np.float32)
    metrics = {}
    for coordinate in phase_chunks:
        for phase, values in phase_chunks[coordinate].items():
            if not values:
                raise RuntimeError(f"no {coordinate}/{phase} samples for {dataset.view}")
            metrics[f"{coordinate}_{phase}"] = float(
                np.concatenate(values).mean(dtype=np.float64))
        prediction = np.concatenate(primary_prediction[coordinate])
        target = np.concatenate(primary_target[coordinate])
        metrics[f"{coordinate}_r2"] = _r2(prediction, target)
    if observer_chunks is not None:
        for key, phases in observer_chunks.items():
            for phase, chunks in phases.items():
                if not chunks:
                    raise RuntimeError(f"no observer {key}/{phase} samples for {dataset.view}")
                values = np.concatenate(chunks).astype(np.float64, copy=False)
                metrics[f"observer_{key}_{phase}_mean"] = float(values.mean())
                metrics[f"observer_{key}_{phase}_std"] = float(values.std())
        metrics["observer_ordered_gain_violation_max"] = ordered_violation
    return metrics, rollout


@torch.no_grad()
def probe_ceilings(model: V11ExperimentModel, dataset: V11TrajectoryDataset,
                   probes, args, device, use_amp) -> dict[str, float]:
    values = collect_representations(model, dataset, args, device, use_amp)
    result = {}
    for coordinate in ("prior", "posterior", "encoder", "predictor"):
        prediction = _probe_predict(
            torch.from_numpy(values[coordinate]).to(device), probes[coordinate]).cpu().numpy()
        y_std = probes[coordinate]["y_std"]
        result[f"{coordinate}_probe_ceiling_state_nmse"] = float(
            np.square((prediction - values["state"]) / y_std).mean(dtype=np.float64))
        result[f"{coordinate}_probe_ceiling_r2"] = _r2(prediction, values["state"])
    return result


def _ridge_action_history(train_actions: np.ndarray, val_actions: np.ndarray,
                          mean: np.ndarray, std: np.ndarray, ridge: float) -> dict[str, float]:
    x_train = ((train_actions[:, :-1] - mean) / std).reshape(-1, train_actions.shape[-1])
    y_train = ((train_actions[:, 1:] - mean) / std).reshape(-1, train_actions.shape[-1])
    design = np.concatenate((x_train, np.ones((len(x_train), 1))), axis=1).astype(np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    x_val = ((val_actions[:, :-1] - mean) / std).reshape(-1, val_actions.shape[-1])
    y_val = ((val_actions[:, 1:] - mean) / std).reshape(-1, val_actions.shape[-1])
    prediction = np.concatenate((x_val, np.ones((len(x_val), 1))), axis=1) @ weights
    return {
        "action_history_probe_nmse": float(np.square(prediction - y_val).mean()),
        "action_history_probe_r2": _r2(prediction, y_val),
    }


def _integrated_action_features(actions: torch.Tensor, history: int = 3) -> torch.Tensor:
    """Action-only audit: local history, cumulative action, and normalized time."""
    batch, transitions, action_dim = actions.shape
    padded = F.pad(actions, (0, 0, history, 0))
    cumulative = actions.cumsum(dim=1)
    cumulative = torch.cat((actions.new_zeros(batch, 1, action_dim), cumulative), dim=1)
    features = []
    for target_t in range(transitions + 1):
        local = padded[:, target_t:target_t + history].reshape(batch, history * action_dim)
        time = actions.new_full((batch, 1), target_t / max(transitions, 1))
        features.append(torch.cat((local, cumulative[:, target_t], time), dim=-1))
    return torch.stack(features, dim=1)


@torch.no_grad()
def action_only_integrator_probe(train_dataset: V11TrajectoryDataset,
                                 val_dataset: V11TrajectoryDataset,
                                 heldout: Mapping[str, V11TrajectoryDataset],
                                 args: argparse.Namespace) -> dict[str, float]:
    def collect(dataset, *, primary_only: bool):
        features, targets = [], []
        for batch in _loader(dataset, args, train=False):
            if args.eval_target_key not in batch:
                raise RuntimeError(f"missing {args.eval_target_key!r} for integrator probe")
            x = _integrated_action_features(
                batch["actions"], args.history_len)[:, args.history_len:]
            y = batch[args.eval_target_key][:, args.history_len:]
            if primary_only:
                masks = _phase_masks(
                    batch, dataset.metadata.length, args.history_len, torch.device("cpu"))
                mask = masks["deep"] | masks["first_post"]
                x, y = x[mask], y[mask]
            features.append(x.reshape(-1, x.shape[-1]))
            targets.append(y.reshape(-1, y.shape[-1]))
        x = torch.cat(features).numpy()
        y = torch.cat(targets).numpy()
        return x, y

    train_x, train_y = collect(train_dataset, primary_only=False)
    val_x, val_y = collect(val_dataset, primary_only=False)
    probe = _fit_ridge(train_x, train_y, args.probe_ridge)
    clean_prediction = _probe_predict(torch.from_numpy(val_x), probe).numpy()
    result = {
        "action_only_integrator_clean_nmse": float(
            np.square((clean_prediction - val_y) / probe["y_std"]).mean(dtype=np.float64)),
        "action_only_integrator_clean_r2": _r2(clean_prediction, val_y),
    }
    condition_nmse, selected_predictions, selected_targets = [], [], []
    for condition, dataset in heldout.items():
        x, y = collect(dataset, primary_only=True)
        prediction = _probe_predict(torch.from_numpy(x), probe).numpy()
        nmse = float(np.square(
            (prediction - y) / probe["y_std"]).mean(dtype=np.float64))
        result[f"action_only_integrator_{condition}_nmse"] = nmse
        condition_nmse.append(nmse)
        selected_predictions.append(prediction)
        selected_targets.append(y)
    result["action_only_integrator_probe_nmse"] = float(np.mean(condition_nmse))
    result["action_only_integrator_probe_r2"] = _r2(
        np.concatenate(selected_predictions), np.concatenate(selected_targets))
    return result


@torch.no_grad()
def initial_encoder_integrator_probe(
        model: V11ExperimentModel, train_dataset: V11TrajectoryDataset,
        val_dataset: V11TrajectoryDataset, heldout: Mapping[str, V11TrajectoryDataset],
        args: argparse.Namespace, device: torch.device, use_amp: bool) -> dict[str, float]:
    """Strong legal baseline: visible initial encoding plus action/time integration."""
    model.eval()

    def collect(dataset, *, primary_only: bool):
        features, targets = [], []
        for batch in _loader(dataset, args, train=False):
            if args.eval_target_key not in batch:
                raise RuntimeError(f"missing {args.eval_target_key!r} for initial-state probe")
            if bool(batch["corruption_mask"][:, 0].any()):
                raise RuntimeError("initial-state control requires a visible deployment frame t=0")
            initial = batch["observed"][:, 0].to(device, non_blocking=True)
            amp_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp else torch.autocast("cpu", enabled=False))
            with amp_context:
                z0 = model.world.encode(initial).float().cpu()
            action_features = _integrated_action_features(
                batch["actions"], args.history_len)[:, args.history_len:]
            z0 = z0.unsqueeze(1).expand(-1, action_features.shape[1], -1)
            x = torch.cat((z0, action_features), dim=-1)
            y = batch[args.eval_target_key][:, args.history_len:]
            if primary_only:
                masks = _phase_masks(
                    batch, dataset.metadata.length, args.history_len, torch.device("cpu"))
                mask = masks["deep"] | masks["first_post"]
                x, y = x[mask], y[mask]
            features.append(x.reshape(-1, x.shape[-1]))
            targets.append(y.reshape(-1, y.shape[-1]))
        return torch.cat(features).numpy(), torch.cat(targets).numpy()

    train_x, train_y = collect(train_dataset, primary_only=False)
    clean_x, clean_y = collect(val_dataset, primary_only=False)
    probe = _fit_ridge(train_x, train_y, args.probe_ridge)
    clean_prediction = _probe_predict(torch.from_numpy(clean_x), probe).numpy()
    result = {
        "initial_encoder_integrator_clean_nmse": float(np.square(
            (clean_prediction - clean_y) / probe["y_std"]).mean(dtype=np.float64)),
        "initial_encoder_integrator_clean_r2": _r2(clean_prediction, clean_y),
    }
    condition_nmse, selected_predictions, selected_targets = [], [], []
    for condition, dataset in heldout.items():
        x, y = collect(dataset, primary_only=True)
        prediction = _probe_predict(torch.from_numpy(x), probe).numpy()
        nmse = float(np.square(
            (prediction - y) / probe["y_std"]).mean(dtype=np.float64))
        result[f"initial_encoder_integrator_{condition}_nmse"] = nmse
        condition_nmse.append(nmse)
        selected_predictions.append(prediction)
        selected_targets.append(y)
    result["initial_encoder_integrator_probe_nmse"] = float(np.mean(condition_nmse))
    result["initial_encoder_integrator_probe_r2"] = _r2(
        np.concatenate(selected_predictions), np.concatenate(selected_targets))
    return result


@torch.no_grad()
def collect_inverse_action_data(
        model: V11ExperimentModel, dataset: V11TrajectoryDataset,
        args, device, use_amp) -> tuple[np.ndarray, np.ndarray]:
    """Collect clean three-frame coordinates and aligned executed actions."""
    model.eval()
    features, targets = [], []
    for batch in _loader(dataset, args, train=False):
        clean = batch["clean"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp else torch.autocast("cpu", enabled=False))
        with amp_context:
            z = model.world.encode(clean)
        inverse_input, inverse_actions = _second_order_inverse_inputs(z, actions)
        features.append(inverse_input.float().cpu().numpy())
        targets.append(inverse_actions.float().cpu().numpy())
    return (
        np.concatenate(features).reshape(-1, 3 * model.world.embed_dim),
        np.concatenate(targets).reshape(-1, model.world.action_dim),
    )


def fit_inverse_action_probe(
        model: V11ExperimentModel, dataset: V11TrajectoryDataset,
        args, device, use_amp) -> dict[str, np.ndarray]:
    features, actions = collect_inverse_action_data(
        model, dataset, args, device, use_amp)
    return _fit_ridge(features, actions, args.probe_ridge)


def evaluate_inverse_action_probe(
        model: V11ExperimentModel, dataset: V11TrajectoryDataset,
        probe: dict[str, np.ndarray], args, device, use_amp) -> dict[str, float]:
    features, target = collect_inverse_action_data(
        model, dataset, args, device, use_amp)
    prediction = _probe_predict(torch.from_numpy(features), probe).numpy()
    return {
        "inverse_action_nmse": float(np.square(
            (prediction - target) / probe["y_std"]).mean(dtype=np.float64)),
        "inverse_action_r2": _r2(prediction, target),
        "inverse_action_probe_samples": float(len(target)),
        "inverse_action_probe_input_dim": float(features.shape[-1]),
        "inverse_action_probe_output_dim": float(target.shape[-1]),
    }


@torch.no_grad()
def aggregate_action_rank_receipts(
        model: V11ExperimentModel, states: torch.Tensor, actions: torch.Tensor,
        clean_z: torch.Tensor, negative_actions: torch.Tensor, *,
        action_ranking_mode: str = DEFAULT_ACTION_RANKING,
        episode_batch_size: int, device: torch.device,
        ) -> dict[str, Any]:
    """Aggregate V11b rank receipts without defining negatives inside eval batches."""
    if action_ranking_mode not in ACTION_RANKING_MODES:
        raise ValueError(f"unknown V11 action-ranking mode {action_ranking_mode!r}")
    if not isinstance(episode_batch_size, int) or episode_batch_size < 1:
        raise ValueError("episode_batch_size must be a positive integer")
    if states.dim() != 4 or states.shape[2] != 2:
        raise ValueError("receipt states must have shape (N,T,2,D)")
    episodes, length, _, dimension = states.shape
    horizon = length - 1
    expected_actions = (episodes, horizon, model.world.action_dim)
    if (tuple(actions.shape) != expected_actions
            or tuple(negative_actions.shape) != expected_actions):
        raise ValueError("receipt true/negative actions do not align with states")
    if tuple(clean_z.shape) != (episodes, length, dimension):
        raise ValueError("receipt clean embeddings do not align with states")
    if episodes < 1 or horizon < 1:
        raise ValueError("receipt requires at least one episode and one transition")

    keys = (
        "live_energy_by_horizon", "positive_energy_by_horizon",
        "negative_energy_by_horizon", "ranking_loss_by_horizon",
        "pair_accuracy_by_horizon", "divergence_squared_by_horizon",
    )
    sums = {key: np.zeros(horizon, dtype=np.float64) for key in keys}
    action_effect_squared_sum = 0.0
    action_effect_scalar_count = 0
    memory = model.world.mem_kdiov11
    effective_frame = memory._validate_cached_action_frame(memory.action_frame())
    for start in range(0, episodes, episode_batch_size):
        stop = min(start + episode_batch_size, episodes)
        chunk_size = stop - start
        chunk_states = states[start:stop].to(device)
        chunk_actions = actions[start:stop].to(device)
        chunk_clean_z = clean_z[start:stop].to(device)
        chunk_negative = negative_actions[start:stop].to(device)
        suffix = _kdio_suffix_objectives(
            model, chunk_states, chunk_actions, chunk_clean_z,
            negative_actions=chunk_negative, h1_only=False,
            action_ranking_mode=action_ranking_mode)
        for key in keys:
            values = suffix[key].detach().double().cpu().numpy()
            if values.shape != (horizon,):
                raise RuntimeError(f"invalid {key} receipt shape {values.shape}")
            # Every episode has the same number of valid sources at a fixed horizon, so
            # episode weighting exactly reconstructs the global pair mean.
            sums[key] += values * chunk_size

        flat_sources = chunk_states[:, :-1].reshape(
            chunk_size * horizon, 2, dimension)
        flat_actions = chunk_actions.reshape(
            chunk_size * horizon, model.world.action_dim)
        true_next = memory._transition_prevalidated(
            flat_sources, flat_actions, effective_frame)
        zero_next = memory._transition_prevalidated(
            flat_sources, torch.zeros_like(flat_actions), effective_frame)
        action_difference = (
            memory.read_state(true_next).float()
            - memory.read_state(zero_next).float())
        action_effect_squared_sum += float(
            action_difference.double().square().sum().cpu())
        action_effect_scalar_count += action_difference.numel()

    means = {key: value / episodes for key, value in sums.items()}
    return {
        **means,
        "divergence_by_horizon": np.sqrt(
            np.maximum(means["divergence_squared_by_horizon"], 0.0)),
        "action_effect_rms": math.sqrt(
            action_effect_squared_sum / action_effect_scalar_count),
        "episode_count": episodes,
        "horizon_count": horizon,
        "pair_count": episodes * horizon * (horizon + 1) // 2,
        "pair_count_by_horizon": (
            episodes * np.arange(horizon, 0, -1, dtype=np.int64)),
    }


@torch.no_grad()
def _collect_clean_kdio_receipt_inputs(
        model: V11ExperimentModel, dataset: V11TrajectoryDataset,
        args: argparse.Namespace, device: torch.device, use_amp: bool,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode every clean validation episode, then restore canonical episode order."""
    states, actions, embeddings, episode_indices = [], [], [], []
    memory = model.world.mem_kdiov11
    for batch in _loader(dataset, args, train=False):
        clean = batch["clean"].to(device, non_blocking=True)
        batch_actions = batch["actions"].to(device, non_blocking=True)
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp else torch.autocast("cpu", enabled=False))
        with amp_context:
            z = model.world.encode(clean)
            _, details = memory(z, batch_actions, return_details=True)
        states.append(details["states"].float().cpu())
        actions.append(batch_actions.float().cpu())
        embeddings.append(z.float().cpu())
        episode_indices.append(batch["episode_index"].long())
    indices = torch.cat(episode_indices)
    order = torch.argsort(indices)
    expected = torch.arange(len(dataset), dtype=torch.long)
    if not torch.equal(indices[order], expected):
        raise RuntimeError("clean-validation receipt did not cover every episode exactly once")
    return (
        torch.cat(states, dim=0)[order],
        torch.cat(actions, dim=0)[order],
        torch.cat(embeddings, dim=0)[order],
    )


@torch.no_grad()
def kdio_diagnostics(
        model: V11ExperimentModel, dataset: V11TrajectoryDataset,
        args: argparse.Namespace, device: torch.device, use_amp: bool,
        action_ranking_mode: str = DEFAULT_ACTION_RANKING,
        ) -> dict[str, Any]:
    defaults = {
        "kdio_inverse_error_max": 0.0,
        "kdio_volume_error_max": 0.0,
        "kdio_streaming_max_abs": 0.0,
        "kdio_action_effect_rms": 0.0,
        "kdio_true_action_one_step_mse": 0.0,
        "kdio_shuffled_action_one_step_mse": 0.0,
        "kdio_true_action_one_step_advantage": 0.0,
        "kdio_true_action_suffix_mse": 0.0,
        "kdio_shuffled_action_suffix_mse": 0.0,
        "kdio_true_action_suffix_advantage": 0.0,
        "kdio_action_swap_pair_accuracy": 0.0,
        "kdio_true_action_advantage_h1": 0.0,
        "kdio_true_action_advantage_h4": 0.0,
        "kdio_true_action_advantage_h8": 0.0,
        "kdio_true_action_advantage_h16": 0.0,
        "kdio_true_action_advantage_h47": 0.0,
        "kdio_action_rollout_divergence_h1": 0.0,
        "kdio_action_rollout_divergence_h4": 0.0,
        "kdio_action_rollout_divergence_h8": 0.0,
        "kdio_action_rollout_divergence_h16": 0.0,
        "kdio_action_rollout_divergence_h47": 0.0,
        "kdio_action_rank_episode_count": 0,
        "kdio_action_rank_pair_count": 0,
        "kdio_action_rank_horizon_count": 0,
        "kdio_live_suffix_mse": 0.0,
        "kdio_action_rank_positive_energy": 0.0,
        "kdio_action_rank_negative_energy": 0.0,
        "kdio_action_rank_relative_advantage": 0.0,
        "kdio_action_rank_loss": 0.0,
        "kdio_action_rank_permutation_scheme": "not_applicable",
        "kdio_action_rank_permutation_sha256": "not_applicable",
        "kdio_action_rank_data_receipt_sha256": "not_applicable",
    }
    if model.world.memory_impl not in KDIO_DESIGNS:
        return defaults
    model.eval()
    all_states, all_actions, all_z = _collect_clean_kdio_receipt_inputs(
        model, dataset, args, device, use_amp)
    episodes = all_states.shape[0]
    permutation = np.roll(np.arange(episodes, dtype=np.int64), 1)
    negative_actions = all_actions.index_select(
        0, torch.from_numpy(permutation.copy()))
    receipt = aggregate_action_rank_receipts(
        model, all_states, all_actions, all_z, negative_actions,
        action_ranking_mode=action_ranking_mode,
        episode_batch_size=args.batch_size, device=device)

    # Inverse/volume/streaming checks are structural identities, so two clean episodes suffice.
    structural_count = min(2, episodes)
    z = all_z[:structural_count].to(device)
    actions = all_actions[:structural_count].to(device)
    memory = model.world.mem_kdiov11
    mixed, details = memory(z, actions, return_details=True)
    action_frame = memory.action_frame()
    state = details["states"][:, 0]
    streamed = [mixed[:, 0]]
    for step in range(1, z.shape[1]):
        mixed_step, state = memory.step(state, z[:, step], actions[:, step - 1])
        streamed.append(mixed_step)
    streaming = torch.stack(streamed, dim=1)
    applicable = details["inverse_applicable"].bool()
    inverse_error = details["inverse_error"].abs()
    volume_applicable = details["volume_preserving_applicable"].bool()
    volume_error = details["volume_error"].abs()
    if memory.mode != "firstorder":
        prior_states = details["priors"][:, 1:].reshape(-1, 2, memory.embed_dim)
        source_states = details["states"][:, :-1].reshape(-1, 2, memory.embed_dim)
        recovered = memory.inverse_transition(
            prior_states, actions.reshape(-1, memory.action_dim),
            cached_action_frame=action_frame)
        measured_inverse_error = float((recovered - source_states).abs().max())
    else:
        measured_inverse_error = 0.0

    positive_by_horizon = receipt["positive_energy_by_horizon"]
    negative_by_horizon = receipt["negative_energy_by_horizon"]
    advantage_by_horizon = (
        (negative_by_horizon - positive_by_horizon)
        / np.maximum(np.abs(negative_by_horizon), np.finfo(np.float64).tiny))
    aggregate_relative_advantage = float(
        (negative_by_horizon.mean() - positive_by_horizon.mean())
        / max(abs(float(negative_by_horizon.mean())), np.finfo(np.float64).tiny))
    divergence = receipt["divergence_by_horizon"]
    permutation_hash = hashlib.sha256(
        np.asarray(permutation, dtype="<i8").tobytes()).hexdigest()
    receipt_payload = {
        "schema": "v11b-full-clean-validation-action-rank-v1",
        "data_file_sha256": dataset.metadata.file_sha256,
        "data_content_sha256": dataset.metadata.content_sha256,
        "dataset_view": dataset.view,
        "episodes": int(receipt["episode_count"]),
        "horizons": int(receipt["horizon_count"]),
        "pairs": int(receipt["pair_count"]),
        "permutation": "global_cyclic_episode_roll_plus_one",
        "permutation_sha256": permutation_hash,
        "action_ranking_mode": action_ranking_mode,
    }
    data_receipt_hash = hashlib.sha256(json.dumps(
        receipt_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()

    def selected(values: np.ndarray, horizon_number: int) -> float:
        return float(values[horizon_number - 1]) if len(values) >= horizon_number else 0.0

    result: dict[str, Any] = {
        "kdio_inverse_error_max": measured_inverse_error,
        "kdio_volume_error_max": (
            float(volume_error[volume_applicable].max())
            if bool(volume_applicable.any()) else 0.0),
        "kdio_streaming_max_abs": float((streaming - mixed).abs().max()),
        "kdio_action_effect_rms": float(receipt["action_effect_rms"]),
        "kdio_live_suffix_mse": float(
            receipt["live_energy_by_horizon"].mean()),
        "kdio_action_rank_positive_energy": float(positive_by_horizon.mean()),
        "kdio_action_rank_negative_energy": float(negative_by_horizon.mean()),
        "kdio_action_rank_relative_advantage": aggregate_relative_advantage,
        "kdio_action_rank_loss": float(
            receipt["ranking_loss_by_horizon"].mean()),
        "kdio_true_action_one_step_mse": float(positive_by_horizon[0]),
        "kdio_shuffled_action_one_step_mse": float(negative_by_horizon[0]),
        "kdio_true_action_one_step_advantage": float(advantage_by_horizon[0]),
        "kdio_true_action_suffix_mse": float(positive_by_horizon.mean()),
        "kdio_shuffled_action_suffix_mse": float(negative_by_horizon.mean()),
        "kdio_true_action_suffix_advantage": aggregate_relative_advantage,
        "kdio_action_swap_pair_accuracy": float(
            receipt["pair_accuracy_by_horizon"].mean()),
        "kdio_true_action_advantage_h1": selected(advantage_by_horizon, 1),
        "kdio_true_action_advantage_h4": selected(advantage_by_horizon, 4),
        "kdio_true_action_advantage_h8": selected(advantage_by_horizon, 8),
        "kdio_true_action_advantage_h16": selected(advantage_by_horizon, 16),
        "kdio_true_action_advantage_h47": selected(advantage_by_horizon, 47),
        "kdio_action_rollout_divergence_h1": selected(divergence, 1),
        "kdio_action_rollout_divergence_h4": selected(divergence, 4),
        "kdio_action_rollout_divergence_h8": selected(divergence, 8),
        "kdio_action_rollout_divergence_h16": selected(divergence, 16),
        "kdio_action_rollout_divergence_h47": selected(divergence, 47),
        "kdio_action_rank_episode_count": int(receipt["episode_count"]),
        "kdio_action_rank_pair_count": int(receipt["pair_count"]),
        "kdio_action_rank_horizon_count": int(receipt["horizon_count"]),
        "kdio_action_rank_permutation_scheme": (
            "global_cyclic_episode_roll_plus_one"),
        "kdio_action_rank_permutation_sha256": permutation_hash,
        "kdio_action_rank_dataset_view": dataset.view,
        "kdio_action_rank_data_file_sha256": dataset.metadata.file_sha256,
        "kdio_action_rank_data_content_sha256": dataset.metadata.content_sha256,
        "kdio_action_rank_data_receipt_sha256": data_receipt_hash,
        "kdio_action_rank_development_mode": action_ranking_mode,
        "kdio_action_rank_eval_episode_batch_size": args.batch_size,
    }
    for index in range(int(receipt["horizon_count"])):
        horizon_number = index + 1
        result.update({
            f"kdio_action_rank_pair_count_h{horizon_number}": int(
                receipt["pair_count_by_horizon"][index]),
            f"kdio_action_rank_live_energy_h{horizon_number}": float(
                receipt["live_energy_by_horizon"][index]),
            f"kdio_live_suffix_mse_h{horizon_number}": float(
                receipt["live_energy_by_horizon"][index]),
            f"kdio_action_rank_positive_energy_h{horizon_number}": float(
                positive_by_horizon[index]),
            f"kdio_action_rank_negative_energy_h{horizon_number}": float(
                negative_by_horizon[index]),
            f"kdio_action_rank_relative_advantage_h{horizon_number}": float(
                advantage_by_horizon[index]),
            f"kdio_action_rank_pair_accuracy_h{horizon_number}": float(
                receipt["pair_accuracy_by_horizon"][index]),
            f"kdio_action_rank_loss_h{horizon_number}": float(
                receipt["ranking_loss_by_horizon"][index]),
        })
    return result


def _rollout_package(condition_rollouts: Mapping[str, Mapping[str, np.ndarray]],
                     output_path: Path, episode_index: int):
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(ROLLOUT_SCHEMA_VERSION, dtype=np.int64),
        "episode_index": np.asarray(episode_index, dtype=np.int64),
        "conditions": np.asarray(HELDOUT_CONDITIONS),
    }
    videos = []
    flat = {key: [] for key in (
        "condition", "target_times", "phase", "state_target",
        "prior_state_prediction", "prior_state_nmse",
        "posterior_state_prediction", "posterior_state_nmse",
        "encoder_state_prediction", "encoder_state_nmse",
        "predictor_state_prediction", "predictor_state_nmse",
    )}
    for condition in HELDOUT_CONDITIONS:
        rollout = condition_rollouts[condition]
        for key, value in rollout.items():
            arrays[f"{condition}_{key}"] = value
        observed, clean = rollout["observed_rgb"], rollout["clean_rgb"]
        separator = np.full((len(observed), observed.shape[1], 4, 3), 255, dtype=np.uint8)
        videos.append(np.concatenate((observed, separator, clean), axis=2))
        count = len(rollout["target_times"])
        flat["condition"].append(np.full(count, condition))
        flat["target_times"].append(rollout["target_times"])
        flat["phase"].append(rollout["phase"])
        flat["state_target"].append(rollout["evaluation_target"])
        for coordinate in ("prior", "posterior", "encoder", "predictor"):
            flat[f"{coordinate}_state_prediction"].append(
                rollout[f"{coordinate}_state_prediction"])
            flat[f"{coordinate}_state_nmse"].append(
                rollout[f"{coordinate}_state_nmse_by_target_t"])
    arrays.update({key: np.concatenate(values) for key, values in flat.items()})
    video = np.concatenate(videos, axis=1).transpose(0, 3, 1, 2)
    np.savez_compressed(output_path, **arrays)
    return arrays, video


def _make_rollout_table(wandb, arrays):
    rows = []
    for condition in HELDOUT_CONDITIONS:
        for values in zip(
            arrays[f"{condition}_target_times"], arrays[f"{condition}_phase"],
            arrays[f"{condition}_prior_state_nmse_by_target_t"],
            arrays[f"{condition}_posterior_state_nmse_by_target_t"],
            arrays[f"{condition}_encoder_state_nmse_by_target_t"],
            arrays[f"{condition}_predictor_state_nmse_by_target_t"],
        ):
            rows.append([condition, int(values[0]), str(values[1]),
                         float(values[2]), float(values[3]), float(values[4]),
                         float(values[5])])
    return wandb.Table(
        columns=["condition", "target_time", "phase", "prior_state_nmse",
                 "posterior_state_nmse", "encoder_state_nmse", "predictor_state_nmse"],
        data=rows)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--memory-mode", choices=DESIGNS, required=True)
    parser.add_argument(
        "--development-action-ranking", choices=ACTION_RANKING_MODES,
        default=DEFAULT_ACTION_RANKING,
        help="development-only V11b action-ranking ablation; does not add a design")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/hacssm_v11_shared")
    parser.add_argument("--epochs", type=int, default=100)
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
    parser.add_argument("--wandb-study", default="hacssm-v11")
    parser.add_argument("--extra-tag", default="")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("invalid V11 training budget")
    if args.wandb and args.wandb_mode != "online":
        raise ValueError("official V11 W&B logging must be online")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    train_metadata = load_cache(args.train_data)
    val_metadata = load_cache(args.val_data)
    if train_metadata.split != "train" or val_metadata.split != "val":
        raise ValueError("V11 cache split mismatch")
    if (train_metadata.seed, val_metadata.seed) != (DEFAULT_TRAIN_SEED, DEFAULT_VAL_SEED):
        raise ValueError("V11 cache rollout seeds differ from the frozen IID-action protocol")
    if train_metadata.smooth_rho != 0.0 or val_metadata.smooth_rho != 0.0:
        raise ValueError("V11 requires IID actions with smooth_rho=0")
    if (train_metadata.env_id, train_metadata.length, train_metadata.img_size,
            train_metadata.action_dim, train_metadata.state_dim,
            train_metadata.task_observation_dim, train_metadata.task_observation_keys,
            train_metadata.task_observation_shapes) != (
            val_metadata.env_id, val_metadata.length, val_metadata.img_size,
            val_metadata.action_dim, val_metadata.state_dim,
            val_metadata.task_observation_dim, val_metadata.task_observation_keys,
            val_metadata.task_observation_shapes):
        raise ValueError("V11 train/validation cache schema mismatch")
    if args.eval_rollout_episode not in range(val_metadata.episodes):
        raise ValueError("evaluation rollout episode is out of range")

    train_view = V11TrajectoryDataset(
        args.train_data, "train", args.corruption_seed, args.history_len)
    val_train_view = V11TrajectoryDataset(
        args.val_data, "train", args.corruption_seed, args.history_len)
    train_clean = V11TrajectoryDataset(
        args.train_data, "clean", args.corruption_seed, args.history_len)
    val_clean = V11TrajectoryDataset(
        args.val_data, "clean", args.corruption_seed, args.history_len)
    heldout = {condition: V11TrajectoryDataset(
        args.val_data, condition, args.corruption_seed, args.history_len)
        for condition in HELDOUT_CONDITIONS}
    train_sample, val_sample = train_clean[0], val_clean[0]
    if args.eval_target_key not in train_sample or args.eval_target_key not in val_sample:
        raise RuntimeError(
            f"official V11 requires dataset field {args.eval_target_key!r}; rebuild V11 caches")
    eval_target_dim = int(train_sample[args.eval_target_key].shape[-1])
    if eval_target_dim < 1 or int(val_sample[args.eval_target_key].shape[-1]) != eval_target_dim:
        raise ValueError("V11 evaluation-target dimensions differ across splits")
    if eval_target_dim != train_metadata.task_observation_dim:
        raise ValueError("V11 dataset target width differs from cache metadata")
    action_mean = train_view.actions.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    action_std = train_view.actions.std(axis=(0, 1), dtype=np.float64).clip(min=1e-6).astype(np.float32)

    model = build_model(
        args, train_metadata.action_dim, action_mean, action_std).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _loader(train_view, args, train=True)
    val_loader = _loader(val_train_view, args, train=False)

    env_name = f"dmc:{train_metadata.env_id}"
    ranking_suffix = (
        "" if args.development_action_ranking == DEFAULT_ACTION_RANKING
        else f"-rank-{args.development_action_ranking}")
    run_name = (
        f"lewm-{env_name}-{args.memory_mode}-s{args.seed}{ranking_suffix}")
    output_dir = Path(args.output_dir).resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json"):
        if (output_dir / filename).exists():
            raise FileExistsError(f"refusing to overwrite {output_dir / filename}")

    wb = None
    if args.wandb:
        import wandb
        tags = ["lewm-memory", "end-to-end-rgb", "predictive-state",
                f"env:{env_name}", f"design:{args.memory_mode}",
                f"study:{args.wandb_study}",
                f"action-ranking:{args.development_action_ranking}"]
        if args.extra_tag:
            tags.extend(tag.strip() for tag in args.extra_tag.split(",") if tag.strip())
        wb = wandb.init(
            entity=args.wandb_entity, project=args.wandb_project, mode=args.wandb_mode,
            name=f"{args.wandb_study}-{run_name}",
            group=f"{args.wandb_study}:{env_name}", job_type=args.memory_mode,
            tags=tags, dir=str(output_dir),
            config=(vars(args) | {
                "env": env_name, "action_dim": train_metadata.action_dim,
                "state_dim": train_metadata.state_dim,
                "eval_target_key": args.eval_target_key,
                "eval_target_dim": eval_target_dim, "encoder_norm": "causal",
                "predictor_norm": "none", "training_objective": OBJECTIVE,
                "one_token_predictor": True, "clean_target_gradient_active": True,
                "prediction_loss_weight": 1.0,
                "action_swap_loss_weight": float(
                    args.memory_mode in ACTION_SWAP_DESIGNS
                    and args.memory_mode != "kdiov11_noactionswap"),
                "inverse_loss_weight": 0.0,
                "innovation_calibration_method": (
                    _design_metadata(
                        args.memory_mode, args.development_action_ranking
                    )["innovation_calibration"]),
                "innovation_calibration_gradient_active": False,
                "variance_loss_weight": 1.0, "covariance_loss_weight": 1.0,
                "inverse_gradient_active": False,
                **_design_metadata(
                    args.memory_mode, args.development_action_ranking),
            }), settings=wandb.Settings(init_timeout=120))
        if wb.offline or (args.wandb_entity and wb.entity != args.wandb_entity):
            raise RuntimeError("V11 W&B online/entity preflight failed")
        wb.define_metric("epoch")
        for namespace in ("train/*", "val/*", "mem/*", "cal/*"):
            wb.define_metric(namespace, step_metric="epoch")
        wb.define_metric("perf/*", step_metric="epoch")

    print(f"=== {run_name} | params={model.num_parameters():,} | V11 | amp={use_amp} ===",
          flush=True)
    history = []
    epoch_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, use_amp, args.memory_mode,
            args.development_action_ranking)
        calibration_log = {}
        if args.memory_mode in CALIBRATED_DESIGNS:
            calibration_log = calibrate_clean_statistics(
                model, train_clean, args, device, use_amp,
                diagonal_only=args.memory_mode == "kdiov11_diagonal")
        val_metrics = run_epoch(
            model, val_loader, None, device, use_amp, args.memory_mode,
            args.development_action_ranking)
        memory_log = _finite_memory_log(model.world)
        epoch_seconds = time.time() - started
        epoch_times.append(epoch_seconds)
        history.append({"epoch": epoch, "epoch_seconds": epoch_seconds,
                        "train": train_metrics, "val": val_metrics})
        print(
            f"e{epoch:3d}/{args.epochs} ({epoch_seconds:.1f}s) "
            f"train={train_metrics['loss']:.5f} pred={train_metrics['predictive_loss']:.5f} "
            f"asr={train_metrics['action_swap_loss']:.5f} "
            f"var={train_metrics['variance_loss']:.5f} "
            f"cov={train_metrics['covariance_loss']:.5f} | val={val_metrics['loss']:.5f}",
            flush=True)
        if wb is not None:
            wb.log({"epoch": epoch,
                    **{f"train/{key}": value for key, value in train_metrics.items()},
                    **{f"val/{key}": value for key, value in val_metrics.items()},
                    **{f"mem/{key}": value for key, value in memory_log.items()},
                    **{f"cal/{key}": value for key, value in calibration_log.items()},
                    "perf/epoch_seconds": epoch_seconds}, step=epoch)

    probes = fit_state_probes(model, train_clean, args, device, use_amp)
    inverse_action_probe = fit_inverse_action_probe(
        model, train_clean, args, device, use_amp)
    inverse_action_metrics = evaluate_inverse_action_probe(
        model, val_clean, inverse_action_probe, args, device, use_amp)
    ceilings = probe_ceilings(model, val_clean, probes, args, device, use_amp)
    metrics: dict[str, Any] = {
        "schema_version": 2, "env": env_name, "design": args.memory_mode,
        "seed": args.seed, "epochs": args.epochs, "encoder_type": "vit",
        "encoder_frozen": False, "encoder_norm": "causal", "predictor_norm": "none",
        "end_to_end_rgb": True, "one_token_predictor": True,
        "clean_target_gradient_active": True, "target_stop_gradient": False,
        "ema_target_active": False, "training_objective": OBJECTIVE,
        "prediction_loss_weight": 1.0,
        "action_swap_loss_weight": float(
            args.memory_mode in ACTION_SWAP_DESIGNS
            and args.memory_mode != "kdiov11_noactionswap"),
        "inverse_loss_weight": 0.0,
        "innovation_calibration_method": (
            _design_metadata(
                args.memory_mode, args.development_action_ranking
            )["innovation_calibration"]),
        "innovation_calibration_gradient_active": False,
        "variance_loss_weight": 1.0, "covariance_loss_weight": 1.0,
        "inverse_gradient_active": False,
        "probe_ridge": args.probe_ridge,
        "probe_fit_split": "clean_train_coordinate_specific_prior_posterior_encoder_predictor",
        "headline_metric": "heldout_prior_state_nmse",
        "eval_target_key": args.eval_target_key,
        "eval_target_dim": eval_target_dim,
        "train_data": str(train_metadata.path), "val_data": str(val_metadata.path),
        "train_data_sha256": train_metadata.file_sha256,
        "val_data_sha256": val_metadata.file_sha256,
        "train_data_content_sha256": train_metadata.content_sha256,
        "val_data_content_sha256": val_metadata.content_sha256,
        "train_episodes": train_metadata.episodes, "val_episodes": val_metadata.episodes,
        "length": train_metadata.length, "action_dim": train_metadata.action_dim,
        "state_dim": train_metadata.state_dim, "trainable_parameters": model.num_parameters(),
        "action_mean": action_mean.tolist(), "action_std": action_std.tolist(),
        "final_train_loss": history[-1]["train"]["loss"],
        "final_val_loss": history[-1]["val"]["loss"],
        "val_predictive_loss": history[-1]["val"]["predictive_loss"],
        "mean_epoch_seconds": float(np.mean(epoch_times)),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0),
        **_design_metadata(
            args.memory_mode, args.development_action_ranking), **ceilings,
        **encoder_diagnostics(model.world, val_clean, args, device),
        **orbit_diagnostics(model.world, val_clean, device),
        **kdio_diagnostics(
            model, val_clean, args, device, use_amp,
            args.development_action_ranking),
        **inverse_action_metrics,
        **_ridge_action_history(
            train_view.actions, val_clean.actions, action_mean, action_std, args.probe_ridge),
        **action_only_integrator_probe(train_clean, val_clean, heldout, args),
        **initial_encoder_integrator_probe(
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
    for loss_name in ("predictive_loss", "action_swap_loss", "calibration_nll", "loss"):
        previous = np.mean([row["val"][loss_name] for row in previous_rows]) \
            if previous_rows else np.mean([row["val"][loss_name] for row in recent_rows])
        recent = np.mean([row["val"][loss_name] for row in recent_rows])
        metrics[f"{loss_name}_convergence_relative_change"] = float(
            (previous - recent) / max(abs(previous), 1e-12))

    clean_result, _ = evaluate_condition(
        model, val_clean, probes, args, device, use_amp)
    for coordinate in ("prior", "posterior", "encoder", "predictor"):
        metrics[f"clean_{coordinate}_state_nmse"] = clean_result[f"{coordinate}_primary"]
        metrics[f"clean_{coordinate}_state_r2"] = clean_result[f"{coordinate}_r2"]
    metrics.update({
        f"clean_{key}": value for key, value in clean_result.items()
        if key.startswith("observer_")
    })
    condition_rollouts = {}
    heldout_primary = {
        coordinate: [] for coordinate in ("prior", "posterior", "encoder", "predictor")}
    for condition, dataset in heldout.items():
        result, rollout = evaluate_condition(
            model, dataset, probes, args, device, use_amp,
            rollout_episode=args.eval_rollout_episode)
        for coordinate in heldout_primary:
            primary = result[f"{coordinate}_primary"]
            metrics[f"{condition}_{coordinate}_state_nmse"] = primary
            metrics[f"{condition}_{coordinate}_state_r2"] = result[f"{coordinate}_r2"]
            heldout_primary[coordinate].append(primary)
            for phase in ("gap", "deep", "first_post", "post"):
                metrics[f"{condition}_{coordinate}_state_nmse_{phase}"] = result[
                    f"{coordinate}_{phase}"]
        metrics.update({
            f"{condition}_{key}": value for key, value in result.items()
            if key.startswith("observer_")
        })
        if rollout is None:
            raise RuntimeError(f"missing rollout episode for {condition}")
        condition_rollouts[condition] = rollout
    for coordinate, values in heldout_primary.items():
        metrics[f"heldout_{coordinate}_state_nmse"] = float(np.mean(values))

    rollout_path = output_dir / "eval_rollout.npz"
    rollout_arrays, rollout_video = _rollout_package(
        condition_rollouts, rollout_path, args.eval_rollout_episode)
    rollout_hash = sha256_file(rollout_path)
    metrics["eval_rollout_episode"] = args.eval_rollout_episode
    metrics["eval_rollout_sha256"] = rollout_hash

    torch.save({
        "model_state_dict": model.state_dict(), "args": vars(args),
        "final_metrics": metrics, "history": history, "state_probes": probes,
        "inverse_action_probe": inverse_action_probe,
        "action_history_probe": {
            key: metrics[key] for key in ("action_history_probe_nmse", "action_history_probe_r2")},
    }, output_dir / "model.pt")
    with (output_dir / "metrics.json").open("x") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    if wb is not None:
        import wandb
        table = _make_rollout_table(wandb, rollout_arrays)
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
            artifact_name, type="evaluation-rollout",
            metadata={
                "schema_version": ROLLOUT_SCHEMA_VERSION, "study": args.wandb_study,
                "env": env_name, "design": args.memory_mode, "seed": args.seed,
                "episode": args.eval_rollout_episode, "sha256": rollout_hash,
                "semantics": "heldout pre-observation-prior normalized task-observation trace",
                **_design_metadata(
                    args.memory_mode, args.development_action_ranking),
            })
        artifact.add_file(str(rollout_path), name="eval_rollout.npz")
        wb.log_artifact(artifact)
        wb.summary.update(metrics)
        record = {
            "schema_version": 2, "run_id": str(wb.id), "run_name": str(wb.name),
            "url": str(wb.url), "entity": str(wb.entity), "project": str(wb.project),
            "mode": "offline" if wb.offline else "online", "study": args.wandb_study,
            "state": "finished", "eval_rollout_artifact_name": artifact_name,
            "eval_rollout_sha256": rollout_hash,
            "eval_rollout_episode": args.eval_rollout_episode,
        }
        wb.finish(exit_code=0)
        if record["mode"] != "online":
            raise RuntimeError("V11 W&B run did not finish online")
        with (output_dir / "wandb_run.json").open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")

    print(
        f"=== done {run_name}: heldout_prior_state_nmse="
        f"{metrics['heldout_prior_state_nmse']:.6f} ===", flush=True)


if __name__ == "__main__":
    main()
