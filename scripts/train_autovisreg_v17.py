#!/usr/bin/env python3
"""Train the V17 coefficient-free AutoVISReg LeWorldModel development study.

V17 changes only the clean-target anti-collapse update.  The causal encoder,
one-token predictor, corruption pairing, memory implementations, optimizer,
data, and evaluation machinery are inherited unchanged from V16/V11.

The candidate combines independently implemented, published Gaussian-Wasserstein
uniformity and VISReg geometry with two host repairs established before the grid
was frozen:

* Gaussian W2 uniformity acts on the full covariance spectrum, jointly detecting
  missing scale and dimensional collapse without a variance/covariance weight;
* a target-scale gate introduces sliced shape pressure only as scale recovers,
  avoiding the enormous tie-sort gradient observed at LeWorldModel's collapsed
  code; and
* prediction and regularizer gradients on the shared encoder are combined by a
  scale-invariant angular bisector with the original prediction-gradient norm.

There is no selectable SSL loss coefficient, schedule, projection count, or
task/memory-specific setting.  The number of fresh slices is the structural
dimension rule K=2D.  V17 is excluded adaptive-development evidence, not an
official VISReg reproduction or confirmation experiment.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.memory_model import MemoryLeWorldModel
from scripts.train_hacssm_v10 import (
    _matched_gru_hidden,
    encoder_diagnostics as _host_encoder_diagnostics,
)
import scripts.train_hacssm_v11 as v11
import scripts.train_subjepa_v16 as v16


FAMILIES = ("autovisreg", "vicreg")
MEMORIES = ("none", "ssm", "hacssmv8")
# Host arms are deliberately first in every task queue.
DESIGNS = tuple(
    f"{family}_{memory}"
    for memory in MEMORIES
    for family in FAMILIES
)
FULLSIG_DESIGNS = DESIGNS  # compatibility name consumed by the closed runner
OBJECTIVE = "v17_paired_next_clean_plus_coefficient_free_wasserstein_visreg"
VISREG_SOURCE = "https://arxiv.org/abs/2606.02572"
WASSERSTEIN_SSL_SOURCE = (
    "https://proceedings.iclr.cc/paper_files/paper/2024/file/"
    "21bcef9a879b85714387f94d7ecc2c91-Paper-Conference.pdf"
)

HISTORY_KEYS = (
    "loss",
    "predictive_loss",
    "regularizer_loss",
    "center_loss",
    "scale_loss",
    "wasserstein_loss",
    "shape_loss",
    "covariance_loss",
    "variance_loss",
    "shape_gate",
    "batch_channel_variance",
    "batch_covariance_trace",
    "batch_effective_rank",
    "batch_rank_valid",
    "gradient_prediction_norm",
    "gradient_regularizer_norm",
    "gradient_adaptive_scale",
    "gradient_cosine",
    "gradient_conflict",
    "gradient_preclip_norm",
    "gradient_clip_fraction",
    "gradient_finite",
)


def parse_design(design: str) -> tuple[str, str]:
    if design not in DESIGNS:
        raise ValueError(f"unknown AutoVISReg-v17 design {design!r}")
    family, memory = design.rsplit("_", 1)
    return family, memory


def design_metadata(design: str, embed_dim: int = 128) -> dict[str, Any]:
    family, memory = parse_design(design)
    candidate = family == "autovisreg"
    return {
        "method": "AutoWasserstein-VISReg-v17-development",
        "evidence_scope": "excluded_adaptive_opened_cache_development",
        "confirmation_evidence": False,
        "executed_return_evaluation": False,
        "regularizer": family,
        "regularizer_family": (
            "gaussian_w2_uniformity_plus_self_paced_visreg_shape"
            if candidate else "vicreg_variance_covariance_control"
        ),
        "regularizer_source": "active_clean_target",
        "visreg_source": VISREG_SOURCE if candidate else "not_applicable",
        "wasserstein_uniformity_source": (
            WASSERSTEIN_SSL_SOURCE if candidate else "not_applicable"
        ),
        "projection_policy": (
            "fresh_full_space_gaussian_unit_slices"
            if candidate else "not_applicable"
        ),
        "projection_count_rule": "2_times_embedding_dimension" if candidate else "not_applicable",
        "projection_count": 2 * int(embed_dim) if candidate else 0,
        "projection_hyperparameter_exposed": False,
        "sigreg_lambda": 0.0,
        "sigreg_projections_per_subspace": 0,
        "sigreg_quad_nodes": 0,
        "temporal_statistics_policy": (
            "pooled_batch_time" if candidate else "pooled_batch_time"
        ),
        "shape_gate_rule": (
            "clamp(mean_channel_std,zero,one)" if candidate else "not_applicable"
        ),
        "correlation_gate_rule": "not_applicable",
        "loss_balance_rule": (
            "scale_invariant_shared_encoder_angular_bisector"
            if candidate else "fixed_unit_control"
        ),
        "gradient_conflict_rule": (
            "common_descent_bisector_with_prediction_gradient_norm"
            if candidate else "not_applicable"
        ),
        "new_ssl_tunable_hyperparameters": 0,
        "fixed_ssl_loss_coefficient": False if candidate else True,
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


class V17ExperimentModel(nn.Module):
    def __init__(self, world: MemoryLeWorldModel, design: str):
        super().__init__()
        self.world = world
        self.design = design

    def num_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def build_model(args: argparse.Namespace, action_dim: int) -> V17ExperimentModel:
    _, memory = parse_design(args.design)
    memory_impl, memory_mode = (
        ("ema", "none") if memory == "none" else (memory, "both")
    )
    # Keep the otherwise-unused legacy SIGReg allocation at the V16 value.  Its
    # construction consumes RNG before later modules are initialized, so this
    # preserves an exact same-seed VICReg host control.
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
        sigreg_lambda=0.0,
        sigreg_projections=512,
        memory_impl=memory_impl,
        memory_mode=memory_mode,
        gru_hidden=_matched_gru_hidden(args.embed_dim),
        hier_loss_weight=0.0,
        encoder_type="vit",
    )
    if world.encoder_norm != "causal" or world.predictor_norm != "none":
        raise RuntimeError("V17 requires the repaired causal/no-output-norm host")
    if memory == "none":
        world.memory.requires_grad_(False)
        world.fusion.requires_grad_(False)
    return V17ExperimentModel(world, args.design)


def memory_representations(
    model: V17ExperimentModel,
    z: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, Any]:
    """Exact V16 causal prior/posterior contract for the frozen memories."""
    _, memory = parse_design(model.design)
    world = model.world
    if memory == "none":
        prior = torch.zeros_like(z)
        prior[:, 0] = z[:, 0]
        prior[:, 1:] = v11.one_token_prediction(world, z[:, :-1], actions)
        return {"fused": z, "prior": prior, "posterior": z, "details": {}}
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
            device=z.device, dtype=z.dtype
        ).view(1, 1, -1, 1)
        prior = (details["priors"] * route).sum(dim=2)
        posterior = (details["states"] * route).sum(dim=2)
        return {
            "fused": world.mem_hacssmv8.fuse(z, mixed),
            "prior": v11._rms_read(prior),
            "posterior": v11._rms_read(posterior),
            "details": details,
        }
    raise AssertionError(f"unhandled V17 memory {memory!r}")


def _normal_quantiles(samples: int, *, device: torch.device) -> torch.Tensor:
    if samples < 2:
        raise ValueError("VISReg requires at least two independent batch samples")
    probabilities = torch.arange(
        1, samples + 1, device=device, dtype=torch.float32
    ) / float(samples + 1)
    return math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)


def visreg_terms(
    clean_z: torch.Tensor,
    *,
    include_rank_guard: bool = True,
) -> dict[str, torch.Tensor]:
    """FP32 Gaussian-W2 uniformity with a self-paced VISReg shape term.

    The pooled empirical Gaussian ``N(mu, covariance)`` is matched to
    ``N(0,I)``.  Dividing squared W2 by D yields exactly ``mean(mu**2)`` plus
    ``mean((sqrt(eigenvalue)-1)**2)``.  This single full-spectrum distance
    detects both constant and dimensional collapse without separately weighted
    variance/covariance terms.  The affine-free LayerNorm host fixes each
    token's raw norm at sqrt(D), making ``I`` the corresponding isotropic target.

    VISReg's fresh K=2D sliced shape term supplies a nonzero diversity gradient
    at tied embeddings.  Its gate is the current mean coordinate standard
    deviation in units of the W2 target—not a schedule or selected threshold.
    """
    if clean_z.ndim != 3 or clean_z.shape[0] < 2:
        raise ValueError(
            f"expected clean embeddings (B,T,D), B>=2; got {tuple(clean_z.shape)}"
        )
    features = clean_z.float().reshape(-1, clean_z.shape[-1])
    samples, dimension = features.shape
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    covariance = centered.T @ centered / float(samples)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    # The source VISReg numerical floor is retained for the square-root and
    # standardization operations; it is not exposed to the experiment CLI.
    root_eigenvalues = torch.sqrt(eigenvalues + 1.0e-6)
    center_loss = mean.square().mean()
    wasserstein_loss = (root_eigenvalues - 1.0).square().mean()
    std = torch.sqrt(torch.diagonal(covariance).clamp_min(0.0) + 1.0e-6)

    standardized = centered / std.detach()
    directions = torch.randn(
        dimension, 2 * dimension, device=features.device, dtype=torch.float32
    )
    directions = F.normalize(directions, dim=0)
    projected = standardized @ directions
    projected = projected.sort(dim=0).values
    quantiles = _normal_quantiles(samples, device=features.device).view(samples, 1)
    shape_loss = (projected - quantiles).square().mean()
    gate = std.mean().detach().clamp(min=0.0, max=1.0)
    # ``include_rank_guard=False`` is retained only for a synthetic ablation in
    # the trainer test; every registered experiment uses the full W2 spectrum.
    active_wasserstein = (
        wasserstein_loss if include_rank_guard else wasserstein_loss.new_zeros(())
    )
    regularizer_loss = center_loss + active_wasserstein + gate * shape_loss
    zero = wasserstein_loss.new_zeros(())
    return {
        "regularizer_loss": regularizer_loss,
        "center_loss": center_loss,
        "scale_loss": active_wasserstein,
        "wasserstein_loss": active_wasserstein,
        "shape_loss": shape_loss,
        "covariance_loss": zero,
        "variance_loss": zero,
        "shape_gate": gate,
    }


def _representation_diagnostics(clean_z: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.autocast(device_type=clean_z.device.type, enabled=False):
        features = clean_z.detach().reshape(-1, clean_z.shape[-1]).float()
        centered = features - features.mean(dim=0, keepdim=True)
        denominator = max(len(features) - 1, 1)
        covariance = centered.T @ centered / denominator
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        trace = eigenvalues.sum()
        probabilities = eigenvalues / trace.clamp_min(torch.finfo(eigenvalues.dtype).eps)
        entropy = -(probabilities * probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()).sum()
        effective_rank = entropy.exp()
        valid = (trace > 1.0e-4).to(dtype=features.dtype)
        return {
            "batch_channel_variance": torch.diagonal(covariance).mean(),
            "batch_covariance_trace": trace,
            "batch_effective_rank": effective_rank,
            "batch_rank_valid": valid,
        }


def compute_losses(
    model: V17ExperimentModel,
    observed: torch.Tensor,
    clean: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    family, _ = parse_design(model.design)
    clean_z = v11.encode_clean_active(model, clean)
    observed_z = model.world.encode(observed)
    memory = memory_representations(model, observed_z, actions)
    prediction = v11.one_token_prediction(
        model.world, memory["fused"][:, :-1], actions
    )
    predictive_loss = F.mse_loss(prediction.float(), clean_z[:, 1:].float())
    if family == "autovisreg":
        with torch.autocast(device_type=clean_z.device.type, enabled=False):
            terms = visreg_terms(clean_z, include_rank_guard=True)
    else:
        variance_loss, covariance_loss = v11._vicreg_terms(clean_z)
        zero = predictive_loss.new_zeros(())
        terms = {
            "regularizer_loss": variance_loss + covariance_loss,
            "center_loss": zero,
            "scale_loss": zero,
            "wasserstein_loss": zero,
            "shape_loss": zero,
            "covariance_loss": covariance_loss,
            "variance_loss": variance_loss,
            "shape_gate": zero,
        }
    diagnostics = _representation_diagnostics(clean_z)
    return {
        "loss": predictive_loss + terms["regularizer_loss"],
        "predictive_loss": predictive_loss,
        **terms,
        **diagnostics,
    }


def _sum_squares(gradients: Iterable[torch.Tensor | None]) -> torch.Tensor:
    values = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
    if not values:
        return torch.zeros((), dtype=torch.float32)
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


def _dot(
    left: Iterable[torch.Tensor | None],
    right: Iterable[torch.Tensor | None],
) -> torch.Tensor:
    products = [
        a.float().mul(b.float()).sum()
        for a, b in zip(left, right, strict=True)
        if a is not None and b is not None
    ]
    if not products:
        return torch.zeros((), dtype=torch.float32)
    total = products[0]
    for value in products[1:]:
        total = total + value
    return total


def compose_adaptive_gradients(
    model: V17ExperimentModel,
    predictive_loss: torch.Tensor,
    regularizer_loss: torch.Tensor,
) -> dict[str, float]:
    """Assign coefficient-free, conflict-safe gradients to model parameters.

    Prediction gradients are computed for every trainable parameter.  Only the
    shared encoder receives the normalized diversity correction.  The raw
    regularizer magnitude cancels from ``scale = ||g_pred||/||g_reg||``;
    predictor and memory updates remain exactly prediction-only.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    encoder_parameters = [
        parameter for parameter in model.world.encoder.parameters()
        if parameter.requires_grad
    ]
    prediction_gradients = torch.autograd.grad(
        predictive_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    if regularizer_loss.requires_grad:
        regularizer_gradients = torch.autograd.grad(
            regularizer_loss,
            encoder_parameters,
            retain_graph=False,
            allow_unused=True,
        )
    else:
        regularizer_gradients = tuple(None for _ in encoder_parameters)
    prediction_by_id = {
        id(parameter): gradient
        for parameter, gradient in zip(parameters, prediction_gradients, strict=True)
    }
    encoder_prediction = [prediction_by_id[id(parameter)] for parameter in encoder_parameters]

    prediction_square = _sum_squares(encoder_prediction)
    regularizer_square = _sum_squares(regularizer_gradients).to(prediction_square.device)
    dot = _dot(encoder_prediction, regularizer_gradients).to(prediction_square.device)
    prediction_norm = prediction_square.sqrt()
    regularizer_norm = regularizer_square.sqrt()
    eps = torch.finfo(torch.float32).eps

    if not bool(torch.isfinite(torch.stack((prediction_norm, regularizer_norm, dot))).all()):
        raise FloatingPointError("non-finite V17 encoder gradient statistics")
    conflict = dot < 0
    if float(regularizer_norm) <= eps:
        adaptive_scale = regularizer_norm.new_zeros(())
        composition_scale = regularizer_norm.new_ones(())
    elif float(prediction_norm) <= eps:
        # Preserve a recovery update at an exact predictive stationary point.
        # Unit norm is the inherited host clip scale, not an SSL coefficient.
        adaptive_scale = regularizer_norm.reciprocal()
        composition_scale = regularizer_norm.new_ones(())
    else:
        # The scale-invariant angular bisector is a common descent direction
        # for both objectives whenever they are not exactly antiparallel.
        # Normalize its encoder norm back to the prediction-gradient norm so
        # gradient balancing cannot silently change the host learning rate.
        adaptive_scale = prediction_norm / regularizer_norm
        bisector_square = (
            prediction_square
            + adaptive_scale.square() * regularizer_square
            + 2.0 * adaptive_scale * dot
        ).clamp_min(0.0)
        if float(bisector_square.sqrt()) <= eps:
            # No common descent direction exists at exact antiparallelity;
            # retain the anti-collapse direction at the host gradient norm.
            composition_scale = regularizer_norm.new_ones(())
        else:
            composition_scale = prediction_norm / bisector_square.sqrt()

    regularizer_by_id = {
        id(parameter): gradient
        for parameter, gradient in zip(
            encoder_parameters, regularizer_gradients, strict=True
        )
    }
    encoder_ids = set(regularizer_by_id)
    for parameter, prediction_gradient in zip(
        parameters, prediction_gradients, strict=True
    ):
        if prediction_gradient is None:
            combined = None
        elif id(parameter) not in encoder_ids:
            combined = prediction_gradient
        else:
            regularizer_gradient = regularizer_by_id[id(parameter)]
            if regularizer_gradient is None or float(regularizer_norm) <= eps:
                combined = prediction_gradient
            else:
                if float(prediction_norm) <= eps:
                    combined = adaptive_scale.to(regularizer_gradient.dtype) * regularizer_gradient
                else:
                    bisector = (
                        prediction_gradient
                        + adaptive_scale.to(regularizer_gradient.dtype)
                        * regularizer_gradient
                    )
                    if float(composition_scale) == 1.0 and float(
                            (prediction_square
                             + adaptive_scale.square() * regularizer_square
                             + 2.0 * adaptive_scale * dot).clamp_min(0.0).sqrt()
                            ) <= eps:
                        combined = adaptive_scale.to(regularizer_gradient.dtype) * regularizer_gradient
                    else:
                        combined = composition_scale.to(bisector.dtype) * bisector
        parameter.grad = None if combined is None else combined.detach()

    cosine = (
        dot / (prediction_norm * regularizer_norm).clamp_min(eps)
        if float(prediction_norm) > eps and float(regularizer_norm) > eps
        else dot.new_zeros(())
    )
    diagnostics = {
        "gradient_prediction_norm": float(prediction_norm.detach()),
        "gradient_regularizer_norm": float(regularizer_norm.detach()),
        "gradient_adaptive_scale": float(adaptive_scale.detach()),
        "gradient_cosine": float(cosine.detach()),
        "gradient_conflict": float(bool(conflict)),
    }
    if not all(math.isfinite(value) for value in diagnostics.values()):
        raise FloatingPointError("non-finite V17 adaptive gradient composition")
    return diagnostics


def _zero_gradient_diagnostics() -> dict[str, float]:
    return {
        "gradient_prediction_norm": 0.0,
        "gradient_regularizer_norm": 0.0,
        "gradient_adaptive_scale": 0.0,
        "gradient_cosine": 0.0,
        "gradient_conflict": 0.0,
        "gradient_preclip_norm": 0.0,
        "gradient_clip_fraction": 0.0,
        "gradient_finite": 1.0,
    }


def run_epoch(
    model: V17ExperimentModel,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    use_amp: bool,
    _unused_sigreg_lambda: float,
) -> dict[str, float]:
    train = optimizer is not None
    family, _ = parse_design(model.design)
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
                if use_amp
                else torch.autocast("cpu", enabled=False)
            )
            with amp_context:
                losses = compute_losses(model, observed, clean, actions)
            if not all(bool(value.detach().isfinite()) for value in losses.values()):
                raise FloatingPointError("non-finite V17 loss or representation diagnostic")

            gradient_metrics = _zero_gradient_diagnostics()
            if train:
                if family == "autovisreg":
                    gradient_metrics.update(compose_adaptive_gradients(
                        model, losses["predictive_loss"], losses["regularizer_loss"]
                    ))
                else:
                    losses["loss"].backward()
                preclip = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not bool(torch.isfinite(preclip)):
                    raise FloatingPointError("non-finite V17 pre-clip gradient norm")
                preclip_value = float(preclip)
                gradient_metrics["gradient_preclip_norm"] = preclip_value
                gradient_metrics["gradient_clip_fraction"] = float(preclip_value > 1.0)
                gradient_metrics["gradient_finite"] = 1.0
                optimizer.step()

            batch_size = int(observed.shape[0])
            merged = {**losses, **gradient_metrics}
            for key in HISTORY_KEYS:
                value = merged[key]
                scalar = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
                if not math.isfinite(scalar):
                    raise FloatingPointError(f"non-finite V17 history metric {key}")
                totals[key] += scalar * batch_size
            count += batch_size
    if not count:
        raise RuntimeError("empty V17 epoch")
    result = {key: value / count for key, value in totals.items()}
    if train:
        model.world._v17_last_gradient_metrics = {
            "train_gradient_prediction_norm": result["gradient_prediction_norm"],
            "train_gradient_regularizer_norm": result["gradient_regularizer_norm"],
            "train_gradient_cosine": result["gradient_cosine"],
            "train_gradient_adaptive_scale": result["gradient_adaptive_scale"],
            "train_gradient_preclip_norm": result["gradient_preclip_norm"],
            "train_gradient_clip_fraction": result["gradient_clip_fraction"],
            "train_gradient_conflict_fraction": result["gradient_conflict"],
            "train_gradient_nonfinite_events": 0.0,
        }
    return result


def encoder_diagnostics(
    world: MemoryLeWorldModel,
    clean_dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    result = _host_encoder_diagnostics(world, clean_dataset, args, device)
    result.update(getattr(world, "_v17_last_gradient_metrics", {
        "train_gradient_prediction_norm": 0.0,
        "train_gradient_regularizer_norm": 0.0,
        "train_gradient_cosine": 0.0,
        "train_gradient_adaptive_scale": 0.0,
        "train_gradient_preclip_norm": 0.0,
        "train_gradient_clip_fraction": 0.0,
        "train_gradient_conflict_fraction": 0.0,
        "train_gradient_nonfinite_events": 0.0,
    }))
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--design", choices=DESIGNS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/autovisreg_v17_development")
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
    parser.add_argument("--probe-ridge", type=float, default=1e-3)
    parser.add_argument("--eval-target-key", default="task_observation", choices=("task_observation",))
    parser.add_argument("--corruption-seed", type=int, default=11012)
    parser.add_argument("--eval-rollout-episode", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=False)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="lewm-memory-popgym")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--wandb-study", default="autovisreg-v17-development")
    parser.add_argument("--extra-tag", default="excluded-adaptive-development,autovisreg-v17")
    args = parser.parse_args(argv)
    # Compatibility-only fields required by the inherited, already-tested V16
    # evaluation harness.  None participates in the V17 objective.
    args.sigreg_lambda = 0.0
    args.sigreg_projections = 512
    args.sigreg_quad_nodes = 17
    return args


def _compat_parse_design(design: str) -> tuple[str, str, None]:
    """Tell V16's post-train harness that no subspace receipt is applicable."""
    _, memory = parse_design(design)
    return "vicreg", memory, None


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 4 or args.num_workers < 0:
        raise ValueError("invalid V17 training configuration")
    if args.wandb and args.wandb_mode != "online":
        raise ValueError("V17 W&B logging must be online")

    # Reuse only the frozen orchestration/evaluation shell; all design-dependent
    # functions and metadata are replaced explicitly here.
    v16.DESIGNS = DESIGNS
    v16.FULLSIG_DESIGNS = DESIGNS
    v16.HISTORY_KEYS = HISTORY_KEYS
    v16.OBJECTIVE = OBJECTIVE
    v16.build_model = build_model
    v16.compute_losses = compute_losses
    v16.run_epoch = run_epoch
    v16.memory_representations = memory_representations
    v16.design_metadata = design_metadata
    v16.parse_design = _compat_parse_design
    v16.encoder_diagnostics = encoder_diagnostics
    v16.parse_args = lambda _argv=None: args
    if not args.wandb:
        v16.main(None)
        return

    # The inherited harness has one literal legacy tag.  Intercept only that
    # metadata field; the run, artifact, and logging implementations stay the
    # tested V16 versions.
    import wandb
    original_init = wandb.init

    def _v17_wandb_init(*init_args, **init_kwargs):
        tags = [
            "autovisreg-v17" if tag == "subjepa-v16" else tag
            for tag in init_kwargs.get("tags", [])
        ]
        init_kwargs["tags"] = tags
        return original_init(*init_args, **init_kwargs)

    wandb.init = _v17_wandb_init
    try:
        v16.main(None)
    finally:
        wandb.init = original_init


if __name__ == "__main__":
    main()
