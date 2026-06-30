#!/usr/bin/env python3
"""Train and evaluate the SIRO-v12 self-identified residual observer.

SIRO fits its memory operators from detached train-split RGB embeddings before epoch one
and after every optimizer epoch.  The fit uses FP64 sufficient statistics, machine-scale
Moore--Penrose inverses, OAS covariance estimates, and no simulator state, reward,
visibility mask, corruption identity, tuned ridge, or memory-specific loss.  Gradient
training uses only one-token next-clean prediction plus unit-weight VICReg variance and
covariance terms.

The six SIRO modes share one three-stream ``[anchor,residual,action]`` tensor/API schema.
Full SIRO and all controls except ``noanchor`` identify dynamics in episode-anchor-centered
coordinates.  ``spectralshrink`` alone reads the action stream through fitted
cross-reachability shrinkage; ``identityA`` is the recurrent identity-transition analogue;
``identityK`` applies raw observed innovations; ``noaction`` zeros only the deployed action
map; and ``noanchor`` is the absolute-coordinate control with a zero anchor stream.
The KDIO-v11 row is delegated to the frozen V11 trainer with the preregistered
``rawdiff_displacement_detached`` development objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
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
)
import scripts.train_hacssm_v11 as v11


SIRO_DESIGNS = (
    "sirov12",
    "sirov12_spectralshrink",
    "sirov12_identityA",
    "sirov12_identityK",
    "sirov12_noaction",
    "sirov12_noanchor",
)
DESIGNS = (*SIRO_DESIGNS, "kdiov11")
HELDOUT_CONDITIONS = v11.HELDOUT_CONDITIONS
ROLLOUT_SCHEMA_VERSION = v11.ROLLOUT_SCHEMA_VERSION
OBJECTIVE = "sirov12_identified_residual_one_token_vicreg"
V11_COMPARATOR_RANKING = "rawdiff_displacement_detached"
FLOAT32_STABILITY_CAP = float(np.nextafter(np.float32(1.0), np.float32(0.0)))


def _machine_rtol(shape: tuple[int, ...], dtype: torch.dtype = torch.float64) -> float:
    return float(max(shape) * torch.finfo(dtype).eps)


def _sym(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (value + value.T)


def _machine_pinv(value: torch.Tensor, *, hermitian: bool = False) -> torch.Tensor:
    if value.dtype != torch.float64 or value.dim() != 2:
        raise ValueError("machine pseudoinverse requires one FP64 matrix")
    return torch.linalg.pinv(
        value, atol=0.0, rtol=_machine_rtol(tuple(value.shape), value.dtype),
        hermitian=hermitian)


def _numerical_rank(value: torch.Tensor) -> int:
    singular = torch.linalg.svdvals(value)
    if not len(singular) or float(singular[0]) == 0.0:
        return 0
    tolerance = _machine_rtol(tuple(value.shape), value.dtype) * singular[0]
    return int((singular > tolerance).sum())


def _machine_floor(scale: torch.Tensor, dimension: int) -> torch.Tensor:
    tiny = torch.finfo(torch.float64).tiny
    return (torch.finfo(torch.float64).eps * dimension
            * scale.abs().clamp_min(tiny))


def _roundoff_psd(value: torch.Tensor, *, label: str) -> torch.Tensor:
    """Clamp only roundoff-scale negative eigenvalues; reject material indefiniteness."""
    value = _sym(value.double())
    eigenvalues, eigenvectors = torch.linalg.eigh(value)
    scale = eigenvalues.abs().amax().clamp_min(torch.finfo(torch.float64).tiny)
    tolerance = _machine_floor(scale, value.shape[0])
    if float(eigenvalues.min()) < -float(tolerance):
        raise RuntimeError(
            f"{label} has material negative eigenvalue {float(eigenvalues.min()):.3e} "
            f"below {-float(tolerance):.3e}")
    return _sym((eigenvectors * eigenvalues.clamp_min(0.0).unsqueeze(0)) @ eigenvectors.T)


def _positive_part(value: torch.Tensor) -> torch.Tensor:
    value = _sym(value.double())
    eigenvalues, eigenvectors = torch.linalg.eigh(value)
    return _sym((eigenvectors * eigenvalues.clamp_min(0.0).unsqueeze(0)) @ eigenvectors.T)


@dataclass(frozen=True)
class OASResult:
    mean: torch.Tensor
    covariance: torch.Tensor
    shrinkage: float
    condition: float
    count: int


def oas_from_sufficient_statistics(
        count: int, value_sum: torch.Tensor, value_cross: torch.Tensor,
        *, label: str) -> OASResult:
    """Fit the same closed-form OAS estimator used by V11 from FP64 statistics."""
    if count < 2 or value_sum.dtype != torch.float64 or value_cross.dtype != torch.float64:
        raise ValueError(f"{label} OAS requires count>=2 and FP64 statistics")
    dimension = int(value_sum.numel())
    if tuple(value_cross.shape) != (dimension, dimension):
        raise ValueError(f"{label} OAS cross-statistic shape mismatch")
    mean = value_sum / count
    covariance = _sym(value_cross / count - torch.outer(mean, mean))
    alpha = covariance.square().mean()
    target_variance = torch.trace(covariance) / dimension
    target_squared = target_variance.square()
    numerator = alpha + target_squared
    denominator = (count + 1) * (alpha - target_squared / dimension)
    shrinkage = (torch.ones_like(numerator) if denominator <= 0 else
                 torch.clamp(numerator / denominator, min=0.0, max=1.0))
    covariance = (1.0 - shrinkage) * covariance
    covariance.diagonal().add_(shrinkage * target_variance)
    covariance.diagonal().add_(_machine_floor(target_variance, dimension))
    covariance = _roundoff_psd(covariance, label=label)
    return OASResult(
        mean=mean,
        covariance=covariance,
        shrinkage=float(shrinkage),
        condition=float(torch.linalg.cond(covariance)),
        count=int(count),
    )


def oas_from_values(values: torch.Tensor, *, label: str) -> OASResult:
    if values.dim() != 2 or values.dtype != torch.float64 or len(values) < 2:
        raise ValueError(f"{label} values must be an FP64 (N,D) tensor with N>=2")
    return oas_from_sufficient_statistics(
        len(values), values.sum(dim=0), values.T @ values, label=label)


@dataclass(frozen=True)
class LinearFit:
    identified_A: torch.Tensor
    raw_A: torch.Tensor
    action_B: torch.Tensor
    drift_b: torch.Tensor
    parity_B0: torch.Tensor
    parity_B1: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    action_oas: OASResult
    residual_oas: OASResult
    residuals: torch.Tensor
    anchor_centered: bool
    receipts: dict[str, Any]


def _standardize_actions(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if actions.dim() != 2 or actions.dtype != torch.float64:
        raise ValueError("actions must be FP64 (N,A)")
    mean = actions.mean(dim=0)
    variance = (actions - mean).square().mean(dim=0)
    scale = actions.square().mean().sqrt().clamp_min(torch.finfo(torch.float64).tiny)
    floor = _machine_floor(scale.square(), actions.shape[1])
    if bool((variance <= floor).any()):
        raise RuntimeError("action standardization found a machine-rank-deficient coordinate")
    std = variance.sqrt()
    return (actions - mean) / std, mean, std


def _fwl_raw_A(
        source: torch.Tensor, target: torch.Tensor, standardized_action: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """FWL fit ``target = source A^T + action B^T + b`` from normal statistics."""
    count, dimension = source.shape
    action_dim = standardized_action.shape[1]
    ones = source.new_ones(count, 1)
    q = torch.cat((source, ones), dim=1)
    q_cross = q.T @ q
    q_action = q.T @ standardized_action
    q_target = q.T @ target
    action_cross = standardized_action.T @ standardized_action
    action_target = standardized_action.T @ target
    q_inverse = _machine_pinv(_sym(q_cross), hermitian=True)
    residual_action_cross = _sym(
        action_cross - q_action.T @ q_inverse @ q_action)
    residual_action_target = action_target - q_action.T @ q_inverse @ q_target
    action_coefficient = _machine_pinv(
        residual_action_cross, hermitian=True) @ residual_action_target
    q_coefficient = q_inverse @ (q_target - q_action @ action_coefficient)
    raw_A = q_coefficient[:dimension].T
    raw_b = q_coefficient[dimension]
    raw_B_standard = action_coefficient.T
    receipts = {
        "fwl_q_rank": _numerical_rank(q_cross),
        "fwl_residual_action_rank": _numerical_rank(residual_action_cross),
        "fwl_q_dimension": dimension + 1,
        "fwl_action_dimension": action_dim,
    }
    return raw_A, raw_B_standard, raw_b, receipts


def _refit_action_and_bias(
        source: torch.Tensor, target: torch.Tensor, standardized_action: torch.Tensor,
        fixed_A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refit standardized-action coefficient and intercept while holding ``A`` fixed."""
    residual_target = target - source @ fixed_A.T
    regressors = torch.cat(
        (standardized_action, standardized_action.new_ones(len(standardized_action), 1)),
        dim=1)
    coefficient = _machine_pinv(
        _sym(regressors.T @ regressors), hermitian=True) @ (
            regressors.T @ residual_target)
    B_standard = coefficient[:-1].T
    bias_standard = coefficient[-1]
    residual = residual_target - regressors @ coefficient
    return B_standard, bias_standard, residual


def _native_action_parameters(
        B_standard: torch.Tensor, bias_standard: torch.Tensor,
        action_mean: torch.Tensor, action_std: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    B_native = B_standard / action_std.unsqueeze(0)
    b_native = bias_standard - B_native @ action_mean
    return B_native, b_native


def fit_linear_dynamics(
        clean_z: torch.Tensor, native_actions: torch.Tensor, *, identity_A: bool,
        anchor_centered: bool = True,
        ) -> LinearFit:
    """Fit SIRO's common A/B/b and parity action maps from detached clean embeddings."""
    if (clean_z.dim() != 3 or native_actions.dim() != 3
            or clean_z.dtype != torch.float64 or native_actions.dtype != torch.float64
            or clean_z.shape[:2] != (native_actions.shape[0], native_actions.shape[1] + 1)):
        raise ValueError("clean_z/actions must be aligned FP64 (E,L,D)/(E,L-1,A)")
    episodes, length, dimension = clean_z.shape
    anchors = clean_z[:, 0]
    fit_coordinates = (
        clean_z - anchors.unsqueeze(1) if anchor_centered else clean_z)
    source = fit_coordinates[:, :-1].reshape(-1, dimension)
    target = fit_coordinates[:, 1:].reshape(-1, dimension)
    actions = native_actions.reshape(-1, native_actions.shape[-1])
    standardized, action_mean, action_std = _standardize_actions(actions)
    raw_A, _, _, fwl_receipts = _fwl_raw_A(source, target, standardized)
    raw_singular = torch.linalg.svdvals(raw_A)
    if identity_A:
        identified_A = torch.eye(dimension, dtype=torch.float64)
    else:
        left, singular, right = torch.linalg.svd(raw_A, full_matrices=False)
        cap = raw_A.new_tensor(FLOAT32_STABILITY_CAP)
        identified_A = (left * singular.clamp_max(cap).unsqueeze(0)) @ right
    B_standard, b_standard, residuals = _refit_action_and_bias(
        source, target, standardized, identified_A)
    action_B, drift_b = _native_action_parameters(
        B_standard, b_standard, action_mean, action_std)

    parity_maps = []
    for parity in (0, 1):
        indices = torch.arange(episodes) % 2 == parity
        if int(indices.sum()) < 1:
            raise RuntimeError("even/odd action fit requires both episode parities")
        parity_source = fit_coordinates[indices, :-1].reshape(-1, dimension)
        parity_target = fit_coordinates[indices, 1:].reshape(-1, dimension)
        parity_actions_native = native_actions[indices].reshape(-1, native_actions.shape[-1])
        parity_actions = (parity_actions_native - action_mean) / action_std
        parity_B_std, parity_b_std, _ = _refit_action_and_bias(
            parity_source, parity_target, parity_actions, identified_A)
        parity_B, _ = _native_action_parameters(
            parity_B_std, parity_b_std, action_mean, action_std)
        parity_maps.append(parity_B)

    action_oas = oas_from_values(actions, label="native-action")
    residual_oas = oas_from_values(residuals, label="identified-residual")
    effective_singular = torch.linalg.svdvals(identified_A)
    B_singular = torch.linalg.svdvals(action_B)
    prediction = source @ identified_A.T + actions @ action_B.T + drift_b
    final_residual = target - prediction
    action_cross = standardized.T @ final_residual
    intercept_cross = final_residual.sum(dim=0)
    target_norm = target.norm().clamp_min(torch.finfo(torch.float64).tiny)
    action_denominator = standardized.norm().clamp_min(
        torch.finfo(torch.float64).tiny) * target_norm
    intercept_denominator = math.sqrt(len(target)) * target_norm
    anchor_values = anchors - anchors.mean(dim=0, keepdim=True)
    anchor_covariance = _sym(anchor_values.T @ anchor_values / episodes)
    anchor_eigenvalues = torch.linalg.eigvalsh(anchor_covariance).clamp_min(0.0)
    anchor_probabilities = anchor_eigenvalues / anchor_eigenvalues.sum().clamp_min(
        torch.finfo(torch.float64).tiny)
    anchor_effective_rank = torch.exp(-(
        anchor_probabilities
        * anchor_probabilities.clamp_min(torch.finfo(torch.float64).tiny).log()).sum())
    residual_values = final_residual - final_residual.mean(dim=0, keepdim=True)
    repeated_anchor = anchor_values.unsqueeze(1).expand(
        -1, length - 1, -1).reshape(-1, dimension)
    anchor_residual_cross = repeated_anchor.T @ residual_values / len(final_residual)
    residual_covariance = _sym(residual_values.T @ residual_values / len(final_residual))
    cross_denominator = torch.sqrt(
        torch.trace(anchor_covariance).clamp_min(torch.finfo(torch.float64).tiny)
        * torch.trace(residual_covariance).clamp_min(torch.finfo(torch.float64).tiny))
    receipts: dict[str, Any] = {
        **fwl_receipts,
        "fit_transition_samples": int(len(source)),
        "fit_episodes": int(episodes),
        "fit_length": int(length),
        "stability_cap": FLOAT32_STABILITY_CAP,
        "identity_A": bool(identity_A),
        "anchor_centered_fit": bool(anchor_centered),
        "centered_x0_max_abs": float(fit_coordinates[:, 0].abs().max()),
        "anchor_mean_channel_variance": float(
            torch.diagonal(anchor_covariance).mean()),
        "anchor_covariance_effective_rank": float(anchor_effective_rank),
        "normalized_residual_anchor_cross_covariance": float(
            anchor_residual_cross.norm() / cross_denominator),
        "raw_A_singular_max": float(raw_singular.max()),
        "raw_A_singular_min": float(raw_singular.min()),
        "identified_A_singular_max": float(effective_singular.max()),
        "identified_A_singular_min": float(effective_singular.min()),
        "identified_A_rank": _numerical_rank(identified_A),
        "action_B_singular_max": float(B_singular.max()),
        "action_B_singular_min": float(B_singular.min()),
        "action_B_rank": _numerical_rank(action_B),
        "action_B0_rank": _numerical_rank(parity_maps[0]),
        "action_B1_rank": _numerical_rank(parity_maps[1]),
        "final_action_residual_cross_relative": float(
            action_cross.norm() / action_denominator),
        "final_intercept_residual_cross_relative": float(
            intercept_cross.norm() / intercept_denominator),
        "action_oas_shrinkage": action_oas.shrinkage,
        "action_covariance_condition": action_oas.condition,
        "residual_oas_shrinkage": residual_oas.shrinkage,
        "residual_covariance_condition": residual_oas.condition,
        "residual_rms": float(final_residual.square().mean().sqrt()),
    }
    return LinearFit(
        identified_A=identified_A,
        raw_A=raw_A,
        action_B=action_B,
        drift_b=drift_b,
        parity_B0=parity_maps[0],
        parity_B1=parity_maps[1],
        action_mean=action_mean,
        action_std=action_std,
        action_oas=action_oas,
        residual_oas=residual_oas,
        residuals=final_residual,
        anchor_centered=anchor_centered,
        receipts=receipts,
    )


@dataclass(frozen=True)
class ReachabilityFit:
    action_read_R: torch.Tensor
    signal_S: torch.Tensor
    noise_N: torch.Tensor
    reachability_W: torch.Tensor
    age_J: torch.Tensor
    receipts: dict[str, Any]


def _canonical_eigenvectors(eigenvectors: torch.Tensor) -> torch.Tensor:
    result = eigenvectors.clone()
    for column in range(result.shape[1]):
        vector = result[:, column]
        pivot = int(vector.abs().argmax())
        if float(vector[pivot]) < 0.0:
            result[:, column].neg_()
    return result


def fit_reachability(linear: LinearFit, length: int) -> ReachabilityFit:
    """Fit all-lag cross-reachability, noise, reachability, and age receipts."""
    if length < 2:
        raise ValueError("reachability requires length>=2")
    A = linear.identified_A
    dimension = A.shape[0]
    Qa = linear.action_oas.covariance
    Qeps = linear.residual_oas.covariance
    B = linear.action_B
    cross = _sym(linear.parity_B0 @ Qa @ linear.parity_B1.T)
    action_covariance = _sym(B @ Qa @ B.T)
    S_raw = A.new_zeros(dimension, dimension)
    N_raw = torch.zeros_like(S_raw)
    W_raw = torch.zeros_like(S_raw)
    J_raw = torch.zeros_like(S_raw)
    power = torch.eye(dimension, dtype=torch.float64)
    horizon = length - 1
    weights = []
    for lag in range(horizon):
        weight = (horizon - lag) / horizon
        weights.append(weight)
        S_raw.add_(weight * power @ cross @ power.T)
        N_raw.add_(weight * power @ Qeps @ power.T)
        propagated_action = power @ action_covariance @ power.T
        W_raw.add_(weight * propagated_action)
        J_raw.add_((lag + 1) * weight * propagated_action)
        power = A @ power
    S = _positive_part(_sym(S_raw))
    N = _roundoff_psd(N_raw, label="reachability-noise")
    W = _roundoff_psd(W_raw, label="reachability-energy")
    J = _roundoff_psd(J_raw, label="reachability-age")
    combined = _roundoff_psd(S + N, label="reachability-signal-plus-noise")
    R = S @ _machine_pinv(combined, hermitian=True)

    kappa, basis = torch.linalg.eigh(S)
    order = torch.argsort(kappa, descending=True)
    kappa = kappa[order]
    basis = _canonical_eigenvectors(basis[:, order])
    numerator = torch.diagonal(basis.T @ J @ basis)
    denominator = torch.diagonal(basis.T @ W @ basis)
    energy_floor = _machine_floor(
        torch.linalg.matrix_norm(W, ord=2), dimension)
    tau = torch.where(
        denominator > energy_floor,
        numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny),
        torch.zeros_like(denominator))
    eigenvector_hash = hashlib.sha256(
        basis.contiguous().numpy().tobytes()).hexdigest()
    generalized = _machine_pinv(W, hermitian=True) @ J
    generalized_eigen = torch.linalg.eigvals(generalized).real
    parity_denominator = (
        linear.parity_B0.norm() * linear.parity_B1.norm()).clamp_min(
            torch.finfo(torch.float64).tiny)
    parity_scale = (0.5 * (
        linear.parity_B0.norm() + linear.parity_B1.norm())).clamp_min(
            torch.finfo(torch.float64).tiny)
    reachability_trace = torch.trace(W).abs().clamp_min(
        torch.finfo(torch.float64).tiny)
    receipts = {
        "reachability_lags": horizon,
        "survival_weight_first": float(weights[0]),
        "survival_weight_last": float(weights[-1]),
        "signal_trace": float(torch.trace(S)),
        "noise_trace": float(torch.trace(N)),
        "reachability_trace": float(torch.trace(W)),
        "age_trace": float(torch.trace(J)),
        "signal_rank": _numerical_rank(S),
        "noise_rank": _numerical_rank(N),
        "reachability_rank": _numerical_rank(W),
        "read_R_norm": float(R.norm()),
        "read_R_singular_max": float(torch.linalg.svdvals(R).max()),
        "read_R_rank": _numerical_rank(R),
        "parity_B_relative_disagreement": float(
            (linear.parity_B0 - linear.parity_B1).norm() / parity_scale),
        "parity_B_cosine_alignment": float(
            (linear.parity_B0 * linear.parity_B1).sum() / parity_denominator),
        "cross_signal_to_full_reachability_trace_ratio": float(
            torch.trace(S) / reachability_trace),
        "age_tau_min": float(tau.min()),
        "age_tau_max": float(tau.max()),
        "age_tau_mean": float(tau.mean()),
        "age_zero_energy_modes": int((denominator <= energy_floor).sum()),
        "age_basis_sha256": eigenvector_hash,
        "age_kappa": [float(value) for value in kappa],
        "age_tau": [float(value) for value in tau],
        "generalized_age_rho_min": float(generalized_eigen.min()),
        "generalized_age_rho_max": float(generalized_eigen.max()),
    }
    return ReachabilityFit(R, S, N, W, J, receipts)


@dataclass(frozen=True)
class InnovationFit:
    lmmse_K: torch.Tensor
    clean_mean: torch.Tensor
    observed_mean: torch.Tensor
    receipts: dict[str, Any]


def fit_actual_prior_innovation(
        clean_z: torch.Tensor, observed_z: torch.Tensor, actions: torch.Tensor,
        A: torch.Tensor, B: torch.Tensor, b: torch.Tensor, R: torch.Tensor,
        *, anchor_centered: bool,
        ) -> InnovationFit:
    """Fit paired OAS-LMMSE K from the actual clean-source causal SIRO prior."""
    if clean_z.shape != observed_z.shape or clean_z.dtype != torch.float64:
        raise ValueError("clean/observed fit embeddings must share FP64 shape")
    episodes, length, dimension = clean_z.shape
    if actions.shape[:2] != (episodes, length - 1):
        raise ValueError("innovation-fit action alignment mismatch")
    count = 0
    joint_sum = torch.zeros(2 * dimension, dtype=torch.float64)
    joint_cross = torch.zeros(2 * dimension, 2 * dimension, dtype=torch.float64)
    anchor_mismatch = (clean_z[:, 0] - observed_z[:, 0]).abs().max()
    input_scale = torch.maximum(
        clean_z[:, 0].abs().max(), observed_z[:, 0].abs().max()).clamp_min(1.0)
    anchor_tolerance = (
        max(dimension, 1) * torch.finfo(torch.float32).eps * input_scale)
    if anchor_mismatch > anchor_tolerance:
        raise RuntimeError(
            "registered SIRO fit requires the initial observed/clean embedding to match; "
            f"error={float(anchor_mismatch):.3e}, tolerance={float(anchor_tolerance):.3e}")
    if anchor_centered:
        c = clean_z[:, 0].clone()
        r = torch.zeros_like(c)
    else:
        c = torch.zeros_like(clean_z[:, 0])
        r = clean_z[:, 0].clone()
    u = torch.zeros_like(r)
    for step in range(length - 1):
        r_prior = r @ A.T + b
        u_prior = u @ A.T + actions[:, step] @ B.T
        prior = c + r_prior + u_prior @ R.T
        clean_innovation = clean_z[:, step + 1] - prior
        observed_innovation = observed_z[:, step + 1] - prior
        values = torch.cat((clean_innovation, observed_innovation), dim=-1)
        count += len(values)
        joint_sum.add_(values.sum(dim=0))
        joint_cross.add_(values.T @ values)
        r = r_prior + clean_innovation
        u = u_prior
        if not torch.equal(c + r + u @ R.T, clean_z[:, step + 1]):
            error = (c + r + u @ R.T - clean_z[:, step + 1]).abs().max()
            scale = clean_z[:, step + 1].abs().max().clamp_min(1.0)
            tolerance = 32 * torch.finfo(torch.float64).eps * scale
            if error > tolerance:
                raise RuntimeError(f"clean-source posterior is not exact: {float(error):.3e}")
    joint = oas_from_sufficient_statistics(
        count, joint_sum, joint_cross, label="paired-innovation")
    covariance = joint.covariance
    covariance_co = covariance[:dimension, dimension:]
    covariance_oo = covariance[dimension:, dimension:]
    K = covariance_co @ _machine_pinv(covariance_oo, hermitian=True)
    singular = torch.linalg.svdvals(K)
    return InnovationFit(
        lmmse_K=K,
        clean_mean=joint.mean[:dimension],
        observed_mean=joint.mean[dimension:],
        receipts={
            "innovation_samples": count,
            "initial_anchor_max_abs_mismatch": float(anchor_mismatch),
            "innovation_oas_shrinkage": joint.shrinkage,
            "innovation_covariance_condition": joint.condition,
            "lmmse_K_norm": float(K.norm()),
            "lmmse_K_singular_min": float(singular.min()),
            "lmmse_K_singular_max": float(singular.max()),
            "lmmse_K_rank": _numerical_rank(K),
        },
    )


@dataclass(frozen=True)
class SIROOperatorFit:
    linear: LinearFit
    reachability: ReachabilityFit
    innovation: InnovationFit
    receipts: dict[str, Any]


def fit_siro_from_embeddings(
        clean_z: torch.Tensor, observed_z: torch.Tensor, actions: torch.Tensor,
        design: str, *, fit_index: int) -> SIROOperatorFit:
    if design not in SIRO_DESIGNS:
        raise ValueError(f"unknown SIRO design {design!r}")
    anchor_centered = design != "sirov12_noanchor"
    linear = fit_linear_dynamics(
        clean_z, actions,
        identity_A=design == "sirov12_identityA",
        anchor_centered=anchor_centered)
    reachability = fit_reachability(linear, clean_z.shape[1])
    effective_B = (
        torch.zeros_like(linear.action_B)
        if design == "sirov12_noaction" else linear.action_B)
    effective_R = (
        reachability.action_read_R
        if design == "sirov12_spectralshrink"
        else torch.eye(clean_z.shape[-1], dtype=torch.float64))
    innovation = fit_actual_prior_innovation(
        clean_z, observed_z, actions,
        linear.identified_A, effective_B, linear.drift_b, effective_R,
        anchor_centered=anchor_centered)
    receipts = {
        "fit_index": int(fit_index),
        "fit_finite": True,
        "effective_action_zero": design == "sirov12_noaction",
        "effective_A_identity": design == "sirov12_identityA",
        "effective_K_identity": design == "sirov12_identityK",
        "effective_R_identity": design != "sirov12_spectralshrink",
        "anchor_centered_fit": anchor_centered,
        **linear.receipts,
        **reachability.receipts,
        **innovation.receipts,
    }
    for tensor in (
            linear.identified_A, linear.action_B, linear.drift_b,
            reachability.action_read_R, innovation.lmmse_K,
            innovation.clean_mean, innovation.observed_mean):
        if not torch.isfinite(tensor).all():
            raise RuntimeError("SIRO fit produced a non-finite tensor")
    return SIROOperatorFit(linear, reachability, innovation, receipts)


def _relative_frobenius_delta(current: torch.Tensor, previous: torch.Tensor) -> float:
    denominator = torch.maximum(current.norm(), previous.norm()).clamp_min(
        torch.finfo(torch.float64).tiny)
    return float((current - previous).norm() / denominator)


def _effective_fit_operators_fp64(
        fit: SIROOperatorFit, design: str
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dimension = fit.linear.identified_A.shape[0]
    A = fit.linear.identified_A
    B = (torch.zeros_like(fit.linear.action_B)
         if design == "sirov12_noaction" else fit.linear.action_B)
    R = (fit.reachability.action_read_R
         if design == "sirov12_spectralshrink"
         else torch.eye(dimension, dtype=torch.float64))
    return A, B, fit.linear.drift_b, R


def _clean_prior_drift_receipts(
        clean_z: torch.Tensor, actions: torch.Tensor,
        previous: SIROOperatorFit, current: SIROOperatorFit, design: str,
        ) -> dict[str, float]:
    """Compare old/new actual clean-source priors on one current embedding snapshot."""
    old_A, old_B, old_b, old_R = _effective_fit_operators_fp64(previous, design)
    new_A, new_B, new_b, new_R = _effective_fit_operators_fp64(current, design)
    anchor_centered = design != "sirov12_noanchor"
    if anchor_centered:
        old_c = clean_z[:, 0].clone()
        new_c = clean_z[:, 0].clone()
        old_r = torch.zeros_like(old_c)
        new_r = torch.zeros_like(new_c)
    else:
        old_c = torch.zeros_like(clean_z[:, 0])
        new_c = torch.zeros_like(clean_z[:, 0])
        old_r = clean_z[:, 0].clone()
        new_r = clean_z[:, 0].clone()
    old_u = torch.zeros_like(old_r)
    new_u = torch.zeros_like(new_r)
    old_error_sum = clean_z.new_zeros(())
    new_error_sum = clean_z.new_zeros(())
    shift_sum = clean_z.new_zeros(())
    old_prediction_sum = clean_z.new_zeros(())
    elements = 0
    for step in range(actions.shape[1]):
        old_rp = old_r @ old_A.T + old_b
        old_up = old_u @ old_A.T + actions[:, step] @ old_B.T
        old_prior = old_c + old_rp + old_up @ old_R.T
        new_rp = new_r @ new_A.T + new_b
        new_up = new_u @ new_A.T + actions[:, step] @ new_B.T
        new_prior = new_c + new_rp + new_up @ new_R.T
        target = clean_z[:, step + 1]
        old_error_sum.add_((old_prior - target).square().sum())
        new_error_sum.add_((new_prior - target).square().sum())
        shift_sum.add_((new_prior - old_prior).square().sum())
        old_prediction_sum.add_(old_prior.square().sum())
        elements += target.numel()
        old_r = old_rp + (target - old_prior)
        new_r = new_rp + (target - new_prior)
        old_u, new_u = old_up, new_up
    pre_mse = old_error_sum / elements
    post_mse = new_error_sum / elements
    shift_mse = shift_sum / elements
    relative_shift = torch.sqrt(
        shift_sum / old_prediction_sum.clamp_min(torch.finfo(torch.float64).tiny))
    return {
        "pre_refit_clean_prior_mse": float(pre_mse),
        "post_refit_clean_prior_mse": float(post_mse),
        "pre_post_refit_clean_prior_shift_mse": float(shift_mse),
        "pre_post_refit_clean_prior_relative_shift": float(relative_shift),
        "pre_post_refit_clean_prior_mse_relative_improvement": float(
            (pre_mse - post_mse)
            / pre_mse.abs().clamp_min(torch.finfo(torch.float64).tiny)),
    }


def add_refit_drift_receipts(
        fit: SIROOperatorFit, previous: SIROOperatorFit | None,
        clean_z: torch.Tensor, actions: torch.Tensor, design: str,
        ) -> SIROOperatorFit:
    """Attach scalar operator/prediction drift; this is not coordinate Procrustes drift."""
    baseline = fit if previous is None else previous
    drift = {
        "operator_A_relative_frobenius_delta": _relative_frobenius_delta(
            fit.linear.identified_A, baseline.linear.identified_A),
        "operator_B_relative_frobenius_delta": _relative_frobenius_delta(
            fit.linear.action_B, baseline.linear.action_B),
        "operator_K_relative_frobenius_delta": _relative_frobenius_delta(
            fit.innovation.lmmse_K, baseline.innovation.lmmse_K),
        "operator_R_relative_frobenius_delta": _relative_frobenius_delta(
            fit.reachability.action_read_R, baseline.reachability.action_read_R),
        "operator_delta_denominator": "max_current_previous_frobenius",
        "drift_receipt_semantics": "operator_and_same_snapshot_clean_prior_not_procrustes",
        **_clean_prior_drift_receipts(
            clean_z, actions, baseline, fit, design),
    }
    return SIROOperatorFit(
        fit.linear, fit.reachability, fit.innovation,
        {**fit.receipts, **drift})


def _design_metadata(design: str) -> dict[str, Any]:
    if design not in DESIGNS:
        raise ValueError(f"unknown V12 design {design!r}")
    if design == "kdiov11":
        return {
            "memory_arch_schema_version": 11,
            "memory_architecture": "kick_drift_innovation_observer",
            "memory_v12_variant": None,
            "identified_operator_fit": False,
            "v11_comparator_action_ranking": V11_COMPARATOR_RANKING,
        }
    variant = design.removeprefix("sirov12_") if design != "sirov12" else "full"
    return {
        "memory_arch_schema_version": 12,
        "memory_architecture": "stable_identified_residual_observer",
        "memory_v12_variant": variant,
        "identified_operator_fit": True,
        "identified_fit_schedule": "before_epoch1_after_every_epoch_and_after_final_epoch",
        "identified_fit_gradient_active": False,
        "identified_fit_arithmetic": "fp64_sufficient_statistics",
        "identified_fit_regularization": "oas_and_machine_tolerance_only",
        "memory_specific_loss_weight": 0.0,
        "anchor_policy": (
            "disabled_absolute_coordinate_fit"
            if design == "sirov12_noanchor"
            else "conserved_visible_initial_embedding"),
        "identified_coordinate": (
            "absolute_embedding"
            if design == "sirov12_noanchor"
            else "episode_initial_anchor_centered_residual"),
        "action_read_policy": (
            "cross_reachability_wiener"
            if design == "sirov12_spectralshrink" else "identity"),
        "transition_policy": "identity" if design == "sirov12_identityA" else "identified",
        "innovation_policy": "raw_identity" if design == "sirov12_identityK" else "paired_oas_lmmse",
        "action_policy": "zero" if design == "sirov12_noaction" else "identified",
        "identityA_semantics": (
            "anchor_plus_cumulative_identified_action_and_linear_drift_recurrent_analogue"
            if design == "sirov12_identityA" else None),
    }


class SIROExperimentModel(nn.Module):
    def __init__(self, world: MemoryLeWorldModel):
        super().__init__()
        self.world = world

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)


def build_model(args: argparse.Namespace, action_dim: int) -> SIROExperimentModel:
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
    )
    if world.encoder_norm != "causal" or world.predictor_norm != "none":
        raise RuntimeError("SIRO requires causal affine-free encoder norm and no predictor norm")
    if not hasattr(world, "mem_sirov12"):
        raise RuntimeError(
            "MemoryLeWorldModel did not install mem_sirov12; install the common "
            "StableIdentifiedResidualObserverMemory integration first")
    return SIROExperimentModel(world)


def _siro_memory(model: SIROExperimentModel):
    memory = getattr(model.world, "mem_sirov12", None)
    if memory is None:
        raise RuntimeError("SIRO model is missing mem_sirov12")
    return memory


def install_operator_fit(model: SIROExperimentModel, fit: SIROOperatorFit) -> None:
    memory = _siro_memory(model)
    method = getattr(memory, "install_fitted_operators", None)
    if method is None:
        raise RuntimeError("mem_sirov12 must expose install_fitted_operators")
    device = next(model.parameters()).device

    def converted(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(device=device, dtype=torch.float32)

    method(
        identified_A=converted(fit.linear.identified_A),
        action_B=converted(fit.linear.action_B),
        drift_b=converted(fit.linear.drift_b),
        action_read_R=converted(fit.reachability.action_read_R),
        lmmse_K=converted(fit.innovation.lmmse_K),
        clean_innovation_mean=converted(fit.innovation.clean_mean),
        observed_innovation_mean=converted(fit.innovation.observed_mean),
        receipts=fit.receipts,
    )


def operator_fit_payload(fit: SIROOperatorFit) -> dict[str, Any]:
    """Serialize every fitted tensor, including diagnostics not consumed at deployment."""
    return {
        "anchor_centered_fit": fit.linear.anchor_centered,
        "identified_A": fit.linear.identified_A.float().cpu(),
        "raw_A": fit.linear.raw_A.float().cpu(),
        "action_B": fit.linear.action_B.float().cpu(),
        "drift_b": fit.linear.drift_b.float().cpu(),
        "action_B0": fit.linear.parity_B0.float().cpu(),
        "action_B1": fit.linear.parity_B1.float().cpu(),
        "action_mean": fit.linear.action_mean.float().cpu(),
        "action_std": fit.linear.action_std.float().cpu(),
        "Qa": fit.linear.action_oas.covariance.float().cpu(),
        "Qeps": fit.linear.residual_oas.covariance.float().cpu(),
        "signal_S": fit.reachability.signal_S.float().cpu(),
        "noise_N": fit.reachability.noise_N.float().cpu(),
        "reachability_W": fit.reachability.reachability_W.float().cpu(),
        "age_J": fit.reachability.age_J.float().cpu(),
        "action_read_R": fit.reachability.action_read_R.float().cpu(),
        "lmmse_K": fit.innovation.lmmse_K.float().cpu(),
        "clean_innovation_mean": fit.innovation.clean_mean.float().cpu(),
        "observed_innovation_mean": fit.innovation.observed_mean.float().cpu(),
        "receipts": fit.receipts,
    }


@torch.no_grad()
def collect_detached_fit_views(
        model: SIROExperimentModel, clean_dataset: V11TrajectoryDataset,
        observed_dataset: V11TrajectoryDataset, args: argparse.Namespace,
        device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode aligned train-only clean/observed RGB once in dropout-off FP32."""
    model.eval()
    clean_chunks, observed_chunks, action_chunks, index_chunks = [], [], [], []
    clean_loader = _loader(clean_dataset, args, train=False)
    observed_loader = _loader(observed_dataset, args, train=False)
    for clean_batch, observed_batch in zip(clean_loader, observed_loader, strict=True):
        clean_indices = clean_batch["episode_index"]
        observed_indices = observed_batch["episode_index"]
        if not torch.equal(clean_indices, observed_indices):
            raise RuntimeError("paired SIRO fit loaders lost episode alignment")
        if not torch.equal(clean_batch["actions"], observed_batch["actions"]):
            raise RuntimeError("paired SIRO fit loaders disagree on actions")
        if int(observed_batch["gap_start"].min()) < 5:
            raise RuntimeError(
                "anchor-centered SIRO requires every registered corruption to start at t>=5")
        clean = clean_batch["clean"].to(device, non_blocking=True)
        observed = observed_batch["observed"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=False):
            clean_z = model.world.encode(clean.float())
            observed_z = model.world.encode(observed.float())
        clean_chunks.append(clean_z.float().cpu())
        observed_chunks.append(observed_z.float().cpu())
        action_chunks.append(clean_batch["actions"].float().cpu())
        index_chunks.append(clean_indices.cpu())
    indices = torch.cat(index_chunks)
    order = torch.argsort(indices)
    expected = torch.arange(len(clean_dataset), dtype=indices.dtype)
    if not torch.equal(indices[order], expected):
        raise RuntimeError("SIRO fit did not observe every train episode exactly once")
    return (
        torch.cat(clean_chunks)[order].double(),
        torch.cat(observed_chunks)[order].double(),
        torch.cat(action_chunks)[order].double(),
    )


@torch.no_grad()
def refit_siro_operators(
        model: SIROExperimentModel, clean_dataset: V11TrajectoryDataset,
        observed_dataset: V11TrajectoryDataset, args: argparse.Namespace,
        device: torch.device, *, fit_index: int,
        previous_fit: SIROOperatorFit | None = None,
        ) -> tuple[SIROOperatorFit, float]:
    started = time.time()
    clean_z, observed_z, actions = collect_detached_fit_views(
        model, clean_dataset, observed_dataset, args, device)
    fit = fit_siro_from_embeddings(
        clean_z, observed_z, actions, args.memory_mode, fit_index=fit_index)
    fit = add_refit_drift_receipts(
        fit, previous_fit, clean_z, actions, args.memory_mode)
    install_operator_fit(model, fit)
    return fit, time.time() - started


def memory_representations(
        model: SIROExperimentModel, z: torch.Tensor, actions: torch.Tensor
        ) -> dict[str, Any]:
    memory = _siro_memory(model)
    mixed, details = memory(z, actions, return_details=True)
    if "states" not in details or "priors" not in details:
        raise RuntimeError("mem_sirov12 details must contain states and priors")
    return {
        "fused": memory.fuse(z, mixed),
        "prior": memory.read_state(details["priors"]),
        "posterior": memory.read_state(details["states"]),
        "details": details,
    }


def compute_siro_losses(
        model: SIROExperimentModel, observed: torch.Tensor,
        clean: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
    clean_z = v11.encode_clean_active(model, clean)
    observed_z = model.world.encode(observed)
    memory = memory_representations(model, observed_z, actions)
    prediction = v11.one_token_prediction(
        model.world, memory["fused"][:, :-1], actions)
    predictive_loss = F.mse_loss(prediction.float(), clean_z[:, 1:].float())
    variance_loss, covariance_loss = v11._vicreg_terms(clean_z)
    return {
        "loss": predictive_loss + variance_loss + covariance_loss,
        "predictive_loss": predictive_loss,
        "context_loss": predictive_loss,
        "variance_loss": variance_loss,
        "covariance_loss": covariance_loss,
    }


HISTORY_KEYS = (
    "loss", "predictive_loss", "context_loss", "variance_loss", "covariance_loss")


def run_epoch(
        model: SIROExperimentModel, loader,
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
                losses = compute_siro_losses(model, observed, clean, actions)
            if train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_size = observed.shape[0]
            for key in HISTORY_KEYS:
                totals[key] += float(losses[key].detach()) * batch_size
            count += batch_size
    if not count:
        raise RuntimeError("empty SIRO epoch")
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def collect_representations(
        model: SIROExperimentModel, dataset: V11TrajectoryDataset,
        args: argparse.Namespace, device: torch.device,
        use_amp: bool) -> dict[str, np.ndarray]:
    model.eval()
    chunks = {
        "prior": [], "posterior": [], "encoder": [], "predictor": [], "state": []}
    history = args.history_len
    for batch in _loader(dataset, args, train=False):
        frames = batch["clean"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp else torch.autocast("cpu", enabled=False))
        with amp_context:
            z = model.world.encode(frames)
            memory = memory_representations(model, z, actions)
            prediction = v11.one_token_prediction(
                model.world, memory["fused"][:, :-1], actions)
        chunks["encoder"].append(z[:, history:].float().cpu().numpy())
        chunks["prior"].append(memory["prior"][:, history:].float().cpu().numpy())
        chunks["posterior"].append(memory["posterior"][:, history:].float().cpu().numpy())
        chunks["predictor"].append(prediction[:, history - 1:].float().cpu().numpy())
        if args.eval_target_key not in batch:
            raise RuntimeError(f"missing evaluation target {args.eval_target_key!r}")
        chunks["state"].append(batch[args.eval_target_key][:, history:].float().numpy())
    result = {}
    for key, values in chunks.items():
        width = values[0].shape[-1] if key == "state" else args.embed_dim
        result[key] = np.concatenate(values).reshape(-1, width)
    return result


def fit_state_probes(
        model: SIROExperimentModel, dataset: V11TrajectoryDataset,
        args: argparse.Namespace, device: torch.device,
        use_amp: bool) -> dict[str, dict[str, np.ndarray]]:
    values = collect_representations(model, dataset, args, device, use_amp)
    return {
        coordinate: v11._fit_ridge(
            values[coordinate], values["state"], args.probe_ridge)
        for coordinate in ("prior", "posterior", "encoder", "predictor")
    }


@torch.no_grad()
def evaluate_condition(
        model: SIROExperimentModel, dataset: V11TrajectoryDataset,
        probes: Mapping[str, dict[str, np.ndarray]], args: argparse.Namespace,
        device: torch.device, use_amp: bool,
        rollout_episode: int | None = None):
    model.eval()
    phases = ("gap", "deep", "first_post", "post", "primary")
    coordinates = ("prior", "posterior", "encoder", "predictor")
    phase_chunks = {
        coordinate: {phase: [] for phase in phases} for coordinate in coordinates}
    primary_prediction = {coordinate: [] for coordinate in coordinates}
    primary_target = {coordinate: [] for coordinate in coordinates}
    rollout = None
    history = args.history_len
    for batch in _loader(dataset, args, train=False):
        observed = batch["observed"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        target_state = batch[args.eval_target_key].to(device, non_blocking=True)
        amp_context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp else torch.autocast("cpu", enabled=False))
        with amp_context:
            z = model.world.encode(observed)
            memory = memory_representations(model, z, actions)
            prediction = v11.one_token_prediction(
                model.world, memory["fused"][:, :-1], actions)
        features = {
            "encoder": z[:, history:].float(),
            "prior": memory["prior"][:, history:].float(),
            "posterior": memory["posterior"][:, history:].float(),
            "predictor": prediction[:, history - 1:].float(),
        }
        target = target_state[:, history:].float()
        masks = _phase_masks(batch, dataset.metadata.length, history, device)
        masks["primary"] = masks["deep"] | masks["first_post"]
        state_predictions = {}
        for coordinate, feature in features.items():
            state_prediction = _probe_predict(feature, probes[coordinate])
            state_predictions[coordinate] = state_prediction
            y_std = torch.as_tensor(
                probes[coordinate]["y_std"], device=device, dtype=torch.float32)
            per_step = ((state_prediction.float() - target) / y_std).square().mean(dim=-1)
            for phase, mask in masks.items():
                if bool(mask.any()):
                    phase_chunks[coordinate][phase].append(per_step[mask].cpu().numpy())
            selected = masks["primary"]
            primary_prediction[coordinate].append(
                state_prediction[selected].float().cpu().numpy())
            primary_target[coordinate].append(target[selected].cpu().numpy())

        if rollout_episode is not None:
            matches = np.nonzero(
                batch["episode_index"].numpy() == rollout_episode)[0]
            if len(matches):
                row = int(matches[0])
                times = np.arange(history, dataset.metadata.length, dtype=np.int64)
                start, end = int(batch["gap_start"][row]), int(batch["gap_end"][row])
                labels = np.asarray([
                    "deep" if start + history <= target_t < end else
                    "gap" if start <= target_t < end else
                    "first_post" if target_t == end else
                    "post" if end < target_t <= end + history else "context"
                    for target_t in times])
                rollout = {
                    "target_times": times,
                    "phase": labels,
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
                for coordinate in coordinates:
                    state_prediction = state_predictions[coordinate][row].cpu().numpy()
                    y_std = np.asarray(probes[coordinate]["y_std"], dtype=np.float32)
                    error = np.square(
                        (state_prediction - rollout["evaluation_target"]) / y_std
                    ).mean(axis=-1)
                    rollout[f"{coordinate}_state_prediction"] = state_prediction.astype(np.float32)
                    rollout[f"{coordinate}_state_nmse_by_target_t"] = error.astype(np.float32)
    metrics = {}
    for coordinate in coordinates:
        for phase, chunks in phase_chunks[coordinate].items():
            if not chunks:
                raise RuntimeError(f"no {coordinate}/{phase} samples for {dataset.view}")
            metrics[f"{coordinate}_{phase}"] = float(
                np.concatenate(chunks).mean(dtype=np.float64))
        prediction = np.concatenate(primary_prediction[coordinate])
        target = np.concatenate(primary_target[coordinate])
        metrics[f"{coordinate}_r2"] = _r2(prediction, target)
    return metrics, rollout


@torch.no_grad()
def probe_ceilings(
        model: SIROExperimentModel, dataset: V11TrajectoryDataset,
        probes, args, device, use_amp) -> dict[str, float]:
    values = collect_representations(model, dataset, args, device, use_amp)
    result = {}
    for coordinate in ("prior", "posterior", "encoder", "predictor"):
        prediction = _probe_predict(
            torch.from_numpy(values[coordinate]).to(device), probes[coordinate]
        ).cpu().numpy()
        result[f"{coordinate}_probe_ceiling_state_nmse"] = float(np.square(
            (prediction - values["state"]) / probes[coordinate]["y_std"]
        ).mean(dtype=np.float64))
        result[f"{coordinate}_probe_ceiling_r2"] = _r2(
            prediction, values["state"])
    return result


def _effective_operators(
        fit: SIROOperatorFit, design: str, *, device: torch.device,
        dtype: torch.dtype = torch.float32):
    dimension = fit.linear.identified_A.shape[0]
    A = fit.linear.identified_A.to(device=device, dtype=dtype)
    B = fit.linear.action_B.to(device=device, dtype=dtype)
    R = torch.eye(dimension, device=device, dtype=dtype)
    if design == "sirov12_noaction":
        B = torch.zeros_like(B)
    if design == "sirov12_spectralshrink":
        R = fit.reachability.action_read_R.to(device=device, dtype=dtype)
    return A, B, fit.linear.drift_b.to(device=device, dtype=dtype), R


@torch.no_grad()
def siro_diagnostics(
        model: SIROExperimentModel, dataset: V11TrajectoryDataset,
        fit: SIROOperatorFit, args: argparse.Namespace,
        device: torch.device, use_amp: bool) -> dict[str, Any]:
    """Verify true API streaming and globally deranged action-prior semantics."""
    model.eval()
    memory = _siro_memory(model)
    first_batch = next(iter(_loader(dataset, args, train=False)))
    frames = first_batch["clean"][:2].to(device)
    actions = first_batch["actions"][:2].to(device)
    def amp_context():
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_amp else torch.autocast("cpu", enabled=False))

    with amp_context():
        z = model.world.encode(frames)
        batch_mixed, batch_details = memory(z, actions, return_details=True)
        streaming_state = memory.initial_state(z[:, 0])
        streaming_reads = [memory.read_state(streaming_state)]
        streaming_states = [streaming_state]
        for step in range(1, z.shape[1]):
            output = memory.step(
                streaming_state, z[:, step], actions[:, step - 1],
                return_details=False)
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("mem_sirov12.step must return (mixed,new_state)")
            mixed_step, streaming_state = output
            streaming_reads.append(mixed_step)
            streaming_states.append(streaming_state)
    streaming_read = torch.stack(streaming_reads, dim=1)
    streaming_state_sequence = torch.stack(streaming_states, dim=1)
    streaming_error = max(
        float((streaming_read.float() - batch_mixed.float()).abs().max()),
        float((streaming_state_sequence.float()
               - batch_details["states"].float()).abs().max()))
    initial_anchor = batch_details["states"][:, :1, 0]
    anchor_invariance_error = float((
        batch_details["states"][:, :, 0] - initial_anchor).abs().max())

    states, all_actions, all_clean_z, episode_indices = [], [], [], []
    for batch in _loader(dataset, args, train=False):
        clean = batch["clean"].to(device, non_blocking=True)
        batch_actions = batch["actions"].to(device, non_blocking=True)
        with amp_context():
            clean_z = model.world.encode(clean)
            _, details = memory(clean_z, batch_actions, return_details=True)
        states.append(details["states"].float().cpu())
        all_actions.append(batch_actions.float().cpu())
        all_clean_z.append(clean_z.float().cpu())
        episode_indices.append(batch["episode_index"].cpu())
    order = torch.argsort(torch.cat(episode_indices))
    state = torch.cat(states)[order].to(device)
    true_actions = torch.cat(all_actions)[order].to(device)
    clean_z = torch.cat(all_clean_z)[order].to(device)
    negative_actions = torch.roll(true_actions, shifts=1, dims=0)
    A, B, b, R = _effective_operators(fit, args.memory_mode, device=device)
    horizon = true_actions.shape[1]
    true_current = state[:, :-1]
    negative_current = state[:, :-1]
    true_mse, negative_mse, pair_accuracy, divergence = [], [], [], []
    for offset in range(horizon):
        valid = horizon - offset
        true_current = true_current[:, :valid]
        negative_current = negative_current[:, :valid]
        true_action = true_actions[:, offset:offset + valid]
        negative_action = negative_actions[:, offset:offset + valid]

        def transition(current: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            c, r, u = (
                current[..., 0, :], current[..., 1, :], current[..., 2, :])
            r_next = r @ A.T + b
            u_next = u @ A.T + action @ B.T
            return torch.stack((c, r_next, u_next), dim=-2)

        true_current = transition(true_current, true_action)
        if args.memory_mode == "sirov12_noaction":
            negative_current = true_current
        else:
            negative_current = transition(negative_current, negative_action)
        target = clean_z[:, offset + 1:offset + 1 + valid]

        def read(current: torch.Tensor) -> torch.Tensor:
            return (current[..., 0, :] + current[..., 1, :]
                    + current[..., 2, :] @ R.T)

        true_read = read(true_current)
        negative_read = read(negative_current)
        positive_energy = (true_read - target).square().mean(dim=-1)
        negative_energy = (negative_read - target).square().mean(dim=-1)
        true_mse.append(positive_energy.mean())
        negative_mse.append(negative_energy.mean())
        pair_accuracy.append((
            (positive_energy < negative_energy).float()
            + 0.5 * (positive_energy == negative_energy).float()).mean())
        divergence.append((true_read - negative_read).square().mean())
    true_by_horizon = torch.stack(true_mse)
    negative_by_horizon = torch.stack(negative_mse)
    advantage = negative_by_horizon - true_by_horizon
    accuracy = torch.stack(pair_accuracy)
    divergence_by_horizon = torch.stack(divergence)
    action_effect = (true_actions @ B.T) @ R.T

    def selected(values: torch.Tensor, horizon_number: int) -> float:
        return float(values[min(horizon_number, len(values)) - 1])

    result: dict[str, Any] = {
        "siro_streaming_max_abs": streaming_error,
        "siro_anchor_invariance_max_abs": anchor_invariance_error,
        "siro_action_effect_rms": float(action_effect.square().mean().sqrt()),
        "siro_true_action_one_step_mse": float(true_by_horizon[0]),
        "siro_deranged_action_one_step_mse": float(negative_by_horizon[0]),
        "siro_true_action_one_step_advantage": float(advantage[0]),
        "siro_true_action_suffix_mse": float(true_by_horizon.mean()),
        "siro_deranged_action_suffix_mse": float(negative_by_horizon.mean()),
        "siro_true_action_suffix_advantage": float(advantage.mean()),
        "siro_action_pair_accuracy": float(accuracy.mean()),
        "siro_action_derangement": "global_cyclic_episode_roll_plus_one",
        "siro_action_diagnostic_episodes": int(len(state)),
    }
    for horizon_number in (1, 4, 8, 16, 47):
        result[f"siro_true_action_advantage_h{horizon_number}"] = selected(
            advantage, horizon_number)
        result[f"siro_action_pair_accuracy_h{horizon_number}"] = selected(
            accuracy, horizon_number)
        result[f"siro_action_rollout_divergence_h{horizon_number}"] = selected(
            divergence_by_horizon, horizon_number)
    return result


def _scalar_fit_receipts(receipts: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in receipts.items():
        if isinstance(value, (bool, int, float, str)):
            result[f"siro_fit_{key}"] = value
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--memory-mode", choices=DESIGNS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default="outputs/hacssm_v12_screen_siro30")
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
    parser.add_argument("--wandb-study", default="hacssm-v12-screen-siro30")
    parser.add_argument("--extra-tag", default="excluded-adaptive-screen")
    return parser.parse_args(argv)


def _v11_delegate_argv(args: argparse.Namespace) -> list[str]:
    values: dict[str, Any] = {
        "train-data": args.train_data,
        "val-data": args.val_data,
        "memory-mode": "kdiov11",
        "development-action-ranking": V11_COMPARATOR_RANKING,
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
            args.extra_tag, "siro-v12-rawdiff-v11-comparator"))),
    }
    if args.wandb_entity:
        values["wandb-entity"] = args.wandb_entity
    result = []
    for key, value in values.items():
        result.extend((f"--{key}", str(value)))
    if args.no_amp:
        result.append("--no-amp")
    result.append("--wandb" if args.wandb else "--no-wandb")
    return result


def _validate_data_contract(args: argparse.Namespace):
    train_metadata = load_cache(args.train_data)
    val_metadata = load_cache(args.val_data)
    if train_metadata.split != "train" or val_metadata.split != "val":
        raise ValueError("SIRO requires train/validation cache roles")
    if (train_metadata.seed, val_metadata.seed) != (DEFAULT_TRAIN_SEED, DEFAULT_VAL_SEED):
        raise ValueError("SIRO cache rollout seeds differ from the frozen IID protocol")
    if train_metadata.smooth_rho != 0.0 or val_metadata.smooth_rho != 0.0:
        raise ValueError("SIRO requires IID actions with smooth_rho=0")
    fields = (
        "env_id", "length", "img_size", "action_dim", "state_dim",
        "task_observation_dim", "task_observation_keys", "task_observation_shapes")
    if tuple(getattr(train_metadata, key) for key in fields) != tuple(
            getattr(val_metadata, key) for key in fields):
        raise ValueError("SIRO train/validation cache schema mismatch")
    if args.eval_rollout_episode not in range(val_metadata.episodes):
        raise ValueError("evaluation rollout episode is out of range")
    return train_metadata, val_metadata


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("invalid SIRO training budget")
    if args.wandb and args.wandb_mode != "online":
        raise ValueError("SIRO W&B logging must be online")
    if args.memory_mode == "kdiov11":
        v11.main(_v11_delegate_argv(args))
        return

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
        for condition in HELDOUT_CONDITIONS}
    sample = train_clean[0]
    if args.eval_target_key not in sample:
        raise RuntimeError(f"SIRO cache is missing {args.eval_target_key!r}")
    eval_target_dim = int(sample[args.eval_target_key].shape[-1])
    if eval_target_dim != train_metadata.task_observation_dim:
        raise RuntimeError("SIRO evaluation target width differs from cache metadata")

    model = build_model(args, train_metadata.action_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _loader(train_view, args, train=True)
    val_loader = _loader(val_train_view, args, train=False)
    env_name = f"dmc:{train_metadata.env_id}"
    run_name = f"lewm-{env_name}-{args.memory_mode}-s{args.seed}"
    output_dir = Path(args.output_dir).resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("model.pt", "metrics.json", "eval_rollout.npz", "wandb_run.json"):
        if (output_dir / filename).exists():
            raise FileExistsError(f"refusing to overwrite {output_dir / filename}")

    wb = None
    if args.wandb:
        import wandb
        tags = [
            "lewm-memory", "end-to-end-rgb", "identified-observer",
            "excluded-adaptive-screen", f"env:{env_name}",
            f"design:{args.memory_mode}", f"study:{args.wandb_study}"]
        if args.extra_tag:
            tags.extend(tag.strip() for tag in args.extra_tag.split(",") if tag.strip())
        wb = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            mode=args.wandb_mode,
            name=f"{args.wandb_study}-{run_name}",
            group=f"{args.wandb_study}:{env_name}",
            job_type=args.memory_mode,
            tags=tags,
            dir=str(output_dir),
            config=(vars(args) | {
                "env": env_name,
                "action_dim": train_metadata.action_dim,
                "state_dim": train_metadata.state_dim,
                "eval_target_dim": eval_target_dim,
                "training_objective": OBJECTIVE,
                "one_token_predictor": True,
                "prediction_loss_weight": 1.0,
                "variance_loss_weight": 1.0,
                "covariance_loss_weight": 1.0,
                "memory_specific_loss_weight": 0.0,
                **_design_metadata(args.memory_mode),
            }),
            settings=wandb.Settings(init_timeout=120),
        )
        if wb.offline or (args.wandb_entity and wb.entity != args.wandb_entity):
            raise RuntimeError("SIRO W&B online/entity preflight failed")
        wb.define_metric("epoch")
        for namespace in ("train/*", "val/*", "fit/*", "mem/*", "perf/*"):
            wb.define_metric(namespace, step_metric="epoch")

    architecture_label = _design_metadata(args.memory_mode).get("method", "SIRO")
    print(
        f"=== {run_name} | params={model.num_parameters():,} | "
        f"{architecture_label} | amp={use_amp} ===",
        flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    fit, fit0_seconds = refit_siro_operators(
        model, train_clean, train_view, args, device,
        fit_index=0, previous_fit=None)
    initial_fit_receipts = dict(fit.receipts)
    fit_history = [{"fit_index": 0, "fit_seconds": fit0_seconds,
                    "receipts": _scalar_fit_receipts(fit.receipts)}]
    history = []
    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, use_amp)
        previous_fit = fit
        fit, fit_seconds = refit_siro_operators(
            model, train_clean, train_view, args, device,
            fit_index=epoch, previous_fit=previous_fit)
        val_metrics = run_epoch(model, val_loader, None, device, use_amp)
        memory_log = _finite_memory_log(model.world)
        epoch_seconds = time.time() - started
        epoch_times.append(epoch_seconds)
        scalar_fit = _scalar_fit_receipts(fit.receipts)
        history.append({
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "fit_seconds": fit_seconds,
            "train": train_metrics,
            "val": val_metrics,
            "fit": scalar_fit,
        })
        fit_history.append({
            "fit_index": epoch, "fit_seconds": fit_seconds,
            "receipts": scalar_fit})
        print(
            f"e{epoch:3d}/{args.epochs} ({epoch_seconds:.1f}s; fit {fit_seconds:.1f}s) "
            f"train={train_metrics['loss']:.5f} pred={train_metrics['predictive_loss']:.5f} "
            f"var={train_metrics['variance_loss']:.5f} "
            f"cov={train_metrics['covariance_loss']:.5f} | "
            f"val={val_metrics['loss']:.5f}",
            flush=True)
        if wb is not None:
            wb.log({
                "epoch": epoch,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
                **{f"fit/{key.removeprefix('siro_fit_')}": value
                   for key, value in scalar_fit.items()
                   if isinstance(value, (int, float, bool))},
                **{f"mem/{key}": value for key, value in memory_log.items()},
                "perf/epoch_seconds": epoch_seconds,
                "perf/fit_seconds": fit_seconds,
            }, step=epoch)

    probes = fit_state_probes(model, train_clean, args, device, use_amp)
    inverse_probe = v11.fit_inverse_action_probe(
        model, train_clean, args, device, use_amp)
    inverse_metrics = v11.evaluate_inverse_action_probe(
        model, val_clean, inverse_probe, args, device, use_amp)
    ceilings = probe_ceilings(model, val_clean, probes, args, device, use_amp)
    diagnostics = siro_diagnostics(model, val_clean, fit, args, device, use_amp)
    action_mean = train_view.actions.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    action_std = train_view.actions.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "env": env_name,
        "design": args.memory_mode,
        "seed": args.seed,
        "epochs": args.epochs,
        "encoder_type": "vit",
        "encoder_frozen": False,
        "encoder_norm": "causal",
        "predictor_norm": "none",
        "end_to_end_rgb": True,
        "one_token_predictor": True,
        "training_objective": OBJECTIVE,
        "prediction_loss_weight": 1.0,
        "variance_loss_weight": 1.0,
        "covariance_loss_weight": 1.0,
        "memory_specific_loss_weight": 0.0,
        "fit_gradient_active": False,
        "fit_updates": args.epochs + 1,
        "fit0_seconds": fit0_seconds,
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
        "mean_epoch_seconds": float(np.mean(epoch_times)),
        "mean_fit_seconds": float(np.mean([
            row["fit_seconds"] for row in fit_history])),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0),
        **_design_metadata(args.memory_mode),
        **_scalar_fit_receipts(fit.receipts),
        **ceilings,
        **encoder_diagnostics(model.world, val_clean, args, device),
        **diagnostics,
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
        if isinstance(value, (int, float)) and math.isfinite(float(value))})
    window = min(10, max(1, args.epochs // 2))
    previous_rows = history[-2 * window:-window]
    recent_rows = history[-window:]
    for loss_name in ("predictive_loss", "loss"):
        previous = np.mean([row["val"][loss_name] for row in previous_rows]) \
            if previous_rows else np.mean([row["val"][loss_name] for row in recent_rows])
        recent = np.mean([row["val"][loss_name] for row in recent_rows])
        metrics[f"{loss_name}_convergence_relative_change"] = float(
            (previous - recent) / max(abs(previous), 1e-12))

    clean_result, _ = evaluate_condition(
        model, val_clean, probes, args, device, use_amp)
    for coordinate in ("prior", "posterior", "encoder", "predictor"):
        metrics[f"clean_{coordinate}_state_nmse"] = clean_result[
            f"{coordinate}_primary"]
        metrics[f"clean_{coordinate}_state_r2"] = clean_result[f"{coordinate}_r2"]
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
        if rollout is None:
            raise RuntimeError(f"missing rollout episode for {condition}")
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
        "fit_history": fit_history,
        "initial_fit_receipts": initial_fit_receipts,
        "final_operator_fit": operator_fit_payload(fit),
        "state_probes": probes,
        "inverse_action_probe": inverse_probe,
    }, output_dir / "model.pt")
    with (output_dir / "metrics.json").open("x") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    if wb is not None:
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
                "design": args.memory_mode,
                "seed": args.seed,
                "episode": args.eval_rollout_episode,
                "sha256": rollout_hash,
                "semantics": "heldout pre-observation-prior task-observation trace",
                **_design_metadata(args.memory_mode),
            })
        artifact.add_file(str(rollout_path), name="eval_rollout.npz")
        wb.log_artifact(artifact)
        wb.summary.update(metrics)
        record = {
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
            "eval_rollout_episode": args.eval_rollout_episode,
        }
        wb.finish(exit_code=0)
        if record["mode"] != "online":
            raise RuntimeError("SIRO W&B run did not finish online")
        with (output_dir / "wandb_run.json").open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")

    print(
        f"=== done {run_name}: heldout_prior_state_nmse="
        f"{metrics['heldout_prior_state_nmse']:.6f} ===",
        flush=True)


if __name__ == "__main__":
    main()
