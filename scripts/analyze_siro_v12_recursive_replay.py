#!/usr/bin/env python3
"""Train-only recursive replay audit for completed full SIRO-v12 checkpoints.

This is an excluded adaptive diagnostic, not another training program.  It loads the
four completed *full* SIRO checkpoints, re-encodes only their registered train split,
and compares five observation-update mechanisms under immutable episode parity folds:

* the V12 clean-history LMMSE gain with the current contractive dynamics fit;
* identity innovation correction with the same dynamics;
* horizon-wise LMMSE gains fitted on the opposite fold's recursively deployed history;
* a covariance/Riccati observer with the current dynamics; and
* the same Riccati observer with OAS whitening and a normal-stable Schur projection.

No validation trajectory, task state, reward, corruption label, W&B API, optimizer, or
gradient is used.  By default the command fails closed while the source screen is
active or incomplete.  Results are written only below an explicit, new report root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.linalg import schur


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.hacssm_v11_data import V11TrajectoryDataset, sha256_file
from scripts.run_siro_v12_screen import SEED, TASKS, run_directory
import scripts.audit_siro_v12_screen as independent_audit
from scripts.train_siro_v12 import (
    FLOAT32_STABILITY_CAP,
    OASResult,
    _machine_floor,
    _machine_pinv,
    _native_action_parameters,
    _refit_action_and_bias,
    _roundoff_psd,
    _standardize_actions,
    build_model,
    collect_detached_fit_views,
    fit_linear_dynamics,
)


VARIANTS = (
    "old_clean_history_k_current_a",
    "identity_k_current_a",
    "deployed_history_lmmse_current_a",
    "riccati_current_a",
    "riccati_normal_stable_a",
)
SCHEMA_VERSION = 1


class ReplayNumericalError(RuntimeError):
    """A fail-closed FP64 replay error with no implication of artifact corruption."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sym(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (value + value.T)


def _finite_float(value: torch.Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"recursive replay produced non-finite scalar {result!r}")
    return result


def centered_oas_from_values(
        values: torch.Tensor, *, label: str,
        ) -> tuple[OASResult, dict[str, Any]]:
    """Numerically stable FP64 OAS from an explicitly centered Gram matrix.

    The V12 training helper consumes uncentered ``sum``/``X^T X`` statistics and then
    subtracts ``mean mean^T``.  Recursive deployed histories can have a very large
    common offset and a small innovation spread, making that subtraction catastrophically
    indefinite even in FP64.  This replay-only estimator uses a scaled reference shift
    for the mean and forms the covariance from centered values divided by ``sqrt(N)``.
    Every rescaling is checked; representational overflow remains a hard failure.
    """
    if (not isinstance(values, torch.Tensor) or values.dtype != torch.float64
            or values.dim() != 2 or len(values) < 2 or values.shape[1] < 1):
        raise ValueError(f"{label}: centered OAS requires FP64 (N,D), N>=2")
    if not torch.isfinite(values).all():
        raise ReplayNumericalError(f"{label}: input contains NaN or infinity")
    count, dimension = values.shape
    input_abs_max = values.abs().max()
    reference = values[0]
    shifted = values - reference
    if not torch.isfinite(shifted).all():
        raise ReplayNumericalError(
            f"{label}: reference subtraction overflowed; dynamic range is not FP64-safe")
    shift_scale = shifted.abs().max()
    if float(shift_scale) == 0.0:
        mean_shift = torch.zeros_like(reference)
    else:
        scaled_shift = shifted / shift_scale
        if not torch.isfinite(scaled_shift).all():
            raise ReplayNumericalError(f"{label}: shift scaling produced non-finite values")
        mean_shift = scaled_shift.mean(dim=0) * shift_scale
    mean = reference + mean_shift
    centered = values - mean
    if not torch.isfinite(mean).all() or not torch.isfinite(centered).all():
        raise ReplayNumericalError(f"{label}: explicit centering overflowed")
    centered_abs_max = centered.abs().max()
    scaled_centered = centered / math.sqrt(count)
    covariance_raw = _sym(scaled_centered.T @ scaled_centered)
    if not torch.isfinite(covariance_raw).all():
        raise ReplayNumericalError(
            f"{label}: centered Gram matrix overflowed; covariance is not FP64-safe")
    raw_eigenvalues = torch.linalg.eigvalsh(covariance_raw)
    if not torch.isfinite(raw_eigenvalues).all():
        raise ReplayNumericalError(f"{label}: centered covariance eigensolve is non-finite")
    try:
        covariance_raw = _roundoff_psd(covariance_raw, label=f"{label}-centered")
    except RuntimeError as exc:
        raise ReplayNumericalError(f"{label}: centered covariance is materially indefinite") from exc

    covariance_scale = covariance_raw.abs().max()
    if float(covariance_scale) == 0.0:
        shrinkage_tensor = covariance_scale.new_tensor(1.0)
    else:
        normalized = covariance_raw / covariance_scale
        alpha = normalized.square().mean()
        target_normalized = torch.trace(normalized) / dimension
        numerator = alpha + target_normalized.square()
        denominator = (count + 1) * (
            alpha - target_normalized.square() / dimension)
        shrinkage_tensor = (
            torch.ones_like(numerator) if denominator <= 0
            else torch.clamp(numerator / denominator, min=0.0, max=1.0))
    if not torch.isfinite(shrinkage_tensor):
        raise ReplayNumericalError(f"{label}: scaled OAS shrinkage is non-finite")
    target_variance = torch.trace(covariance_raw) / dimension
    covariance = (1.0 - shrinkage_tensor) * covariance_raw
    covariance.diagonal().add_(shrinkage_tensor * target_variance)
    covariance.diagonal().add_(_machine_floor(target_variance, dimension))
    if not torch.isfinite(covariance).all():
        raise ReplayNumericalError(f"{label}: OAS covariance rescaling overflowed")
    try:
        covariance = _roundoff_psd(covariance, label=f"{label}-oas")
    except RuntimeError as exc:
        raise ReplayNumericalError(f"{label}: OAS covariance is materially indefinite") from exc
    condition = torch.linalg.cond(covariance)
    if not torch.isfinite(condition):
        raise ReplayNumericalError(
            f"{label}: OAS covariance is singular/non-finite after machine floor")
    result = OASResult(
        mean=mean,
        covariance=covariance,
        shrinkage=float(shrinkage_tensor),
        condition=float(condition),
        count=int(count),
    )
    diagnostics = {
        "algorithm": "scaled_reference_mean_explicit_centered_gram_oas_fp64",
        "count": int(count),
        "dimension": int(dimension),
        "finite_input": True,
        "input_abs_max": _finite_float(input_abs_max),
        "reference_abs_max": _finite_float(reference.abs().max()),
        "reference_shift_abs_max": _finite_float(shift_scale),
        "centered_abs_max": _finite_float(centered_abs_max),
        "centered_to_input_abs_ratio": _finite_float(
            centered_abs_max / input_abs_max.clamp_min(torch.finfo(torch.float64).tiny)),
        "raw_covariance_abs_max": _finite_float(covariance_scale),
        "raw_covariance_min_eigenvalue": _finite_float(raw_eigenvalues.min()),
        "raw_covariance_max_eigenvalue": _finite_float(raw_eigenvalues.max()),
        "oas_shrinkage": result.shrinkage,
        "oas_condition": result.condition,
        "oas_covariance_min_eigenvalue": _finite_float(
            torch.linalg.eigvalsh(result.covariance).min()),
    }
    return result, diagnostics


def immutable_parity_folds(episodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exhaustive even/odd episode-index folds with no RNG dependency."""
    if episodes < 4:
        raise ValueError("recursive parity replay requires at least four episodes")
    indices = torch.arange(episodes, dtype=torch.long)
    folds = indices[indices.remainder(2) == 0], indices[indices.remainder(2) == 1]
    if not len(folds[0]) or not len(folds[1]):
        raise RuntimeError("parity replay produced an empty fold")
    if len(torch.unique(torch.cat(folds))) != episodes:
        raise RuntimeError("parity folds are not disjoint and exhaustive")
    return folds


def _validate_embeddings(
        clean_z: torch.Tensor, observed_z: torch.Tensor,
        actions: torch.Tensor) -> tuple[int, int, int, int]:
    if (clean_z.dtype != torch.float64 or observed_z.dtype != torch.float64
            or actions.dtype != torch.float64):
        raise ValueError("recursive replay requires detached FP64 embeddings/actions")
    if clean_z.dim() != 3 or observed_z.shape != clean_z.shape or actions.dim() != 3:
        raise ValueError("expected aligned (E,L,D)/(E,L,D)/(E,L-1,A) tensors")
    episodes, length, dimension = clean_z.shape
    if actions.shape[:2] != (episodes, length - 1) or episodes < 4 or length < 2:
        raise ValueError("recursive replay embedding/action dimensions are inconsistent")
    if not all(torch.isfinite(value).all() for value in (clean_z, observed_z, actions)):
        raise ValueError("recursive replay inputs contain non-finite values")
    return episodes, length, dimension, actions.shape[-1]


def _matrix_sqrt_factors(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(_sym(covariance.double()))
    if float(eigenvalues.min()) <= 0.0:
        raise RuntimeError("OAS state covariance is not positive definite")
    whitening = (
        eigenvectors * eigenvalues.rsqrt().unsqueeze(0)) @ eigenvectors.T
    coloring = (
        eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T
    identity_error = (whitening @ coloring - torch.eye(
        len(eigenvalues), dtype=torch.float64)).abs().max()
    if float(identity_error) > 128 * len(eigenvalues) * torch.finfo(torch.float64).eps:
        raise RuntimeError("state whitening/coloring factors failed roundtrip")
    return _sym(whitening), _sym(coloring)


def project_real_normal_stable(
        raw: torch.Tensor, *, cap: float = FLOAT32_STABILITY_CAP,
        ) -> tuple[torch.Tensor, dict[str, Any]]:
    """Project a real matrix to normal, strictly stable real-Schur blocks.

    LAPACK's real Schur form contains exact zero subdiagonals between its 1x1/2x2
    blocks, so the block partition does not require a user tolerance.  A 2x2 block is
    replaced by its nearest rotation/scale block before its radius is machine-capped.
    """
    if raw.dtype != torch.float64 or raw.dim() != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("normal-stable projection requires one square FP64 matrix")
    if not 0.0 < cap < 1.0:
        raise ValueError("normal-stable cap must lie strictly between zero and one")
    triangular, orthogonal = schur(raw.detach().cpu().numpy(), output="real")
    n = triangular.shape[0]
    blocks = np.zeros_like(triangular)
    radii: list[float] = []
    block_sizes: list[int] = []
    i = 0
    while i < n:
        if i + 1 < n and triangular[i + 1, i] != 0.0:
            block = triangular[i:i + 2, i:i + 2]
            alpha = 0.5 * (block[0, 0] + block[1, 1])
            beta = 0.5 * (block[1, 0] - block[0, 1])
            radius = float(math.hypot(alpha, beta))
            scale = min(1.0, cap / radius) if radius else 1.0
            blocks[i:i + 2, i:i + 2] = scale * np.array(
                [[alpha, -beta], [beta, alpha]], dtype=np.float64)
            radii.append(min(radius, cap))
            block_sizes.append(2)
            i += 2
        else:
            value = float(np.clip(triangular[i, i], -cap, cap))
            blocks[i, i] = value
            radii.append(abs(value))
            block_sizes.append(1)
            i += 1
    projected_np = orthogonal @ blocks @ orthogonal.T
    projected = torch.from_numpy(projected_np).to(dtype=torch.float64)
    commutator = projected.T @ projected - projected @ projected.T
    denominator = projected.norm().square().clamp_min(torch.finfo(torch.float64).tiny)
    singular_max = torch.linalg.svdvals(projected).max()
    spectral_radius = torch.linalg.eigvals(projected).abs().max()
    receipts = {
        "schur_block_sizes": block_sizes,
        "schur_blocks_1x1": block_sizes.count(1),
        "schur_blocks_2x2": block_sizes.count(2),
        "block_radius_max": max(radii, default=0.0),
        "normality_relative_commutator": _finite_float(commutator.norm() / denominator),
        "singular_max": _finite_float(singular_max),
        "spectral_radius": _finite_float(spectral_radius),
        "relative_projection_delta": _finite_float(
            (projected - raw).norm()
            / raw.norm().clamp_min(torch.finfo(torch.float64).tiny)),
        "stability_cap": float(cap),
    }
    tolerance = 1024 * n * torch.finfo(torch.float64).eps
    if receipts["normality_relative_commutator"] > tolerance:
        raise RuntimeError("normal-stable projection is not numerically normal")
    if receipts["singular_max"] > cap + tolerance:
        raise RuntimeError("normal-stable projection exceeds its stability cap")
    return projected, receipts


def operator_algebra_receipts(A: torch.Tensor, horizon: int) -> dict[str, float]:
    if A.dtype != torch.float64 or A.shape != (A.shape[0], A.shape[0]):
        raise ValueError("operator receipts require one square FP64 matrix")
    denominator = A.norm().square().clamp_min(torch.finfo(torch.float64).tiny)
    commutator = A.T @ A - A @ A.T
    power = torch.eye(A.shape[0], dtype=torch.float64)
    transient = 0.0
    for _ in range(horizon):
        power = A @ power
        transient = max(transient, float(torch.linalg.svdvals(power).max()))
    return {
        "spectral_radius": _finite_float(torch.linalg.eigvals(A).abs().max()),
        "singular_max": _finite_float(torch.linalg.svdvals(A).max()),
        "normality_relative_commutator": _finite_float(
            commutator.norm() / denominator),
        "transient_singular_max_through_horizon": _finite_float(transient),
    }


@dataclass(frozen=True)
class Dynamics:
    A: torch.Tensor
    B: torch.Tensor
    b: torch.Tensor
    Q: torch.Tensor
    R: torch.Tensor
    observation_noise_mean: torch.Tensor
    whitening: torch.Tensor
    coloring: torch.Tensor
    noaction_b: torch.Tensor
    receipts: dict[str, Any]

    def to_coordinate(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.whitening.T

    def from_coordinate(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.coloring.T


def _fit_current_dynamics(
        clean_z: torch.Tensor, observed_z: torch.Tensor,
        actions: torch.Tensor) -> Dynamics:
    linear = fit_linear_dynamics(
        clean_z, actions, identity_A=False, anchor_centered=True)
    dimension = clean_z.shape[-1]
    identity = torch.eye(dimension, dtype=torch.float64)
    clean_x = clean_z - clean_z[:, :1]
    observed_x = observed_z - clean_z[:, :1]
    source = clean_x[:, :-1].reshape(-1, dimension)
    target = clean_x[:, 1:].reshape(-1, dimension)
    noaction_b = (target - source @ linear.identified_A.T).mean(dim=0)
    process_oas, process_oas_diagnostics = centered_oas_from_values(
        linear.residuals, label="replay-current-process")
    observation_noise = (observed_x - clean_x)[:, 1:].reshape(-1, dimension)
    observation_oas, observation_oas_diagnostics = centered_oas_from_values(
        observation_noise, label="replay-observation")
    receipts = {
        "fit_coordinate": "anchor_relative_original",
        "state_whitening_condition": 1.0,
        "process_oas_shrinkage": process_oas.shrinkage,
        "process_covariance_condition": process_oas.condition,
        "observation_oas_shrinkage": observation_oas.shrinkage,
        "observation_covariance_condition": observation_oas.condition,
        "process_centered_oas": process_oas_diagnostics,
        "observation_centered_oas": observation_oas_diagnostics,
        "action_B_rank": int(torch.linalg.matrix_rank(linear.action_B)),
        **operator_algebra_receipts(linear.identified_A, clean_z.shape[1] - 1),
    }
    return Dynamics(
        A=linear.identified_A,
        B=linear.action_B,
        b=linear.drift_b,
        Q=process_oas.covariance,
        R=observation_oas.covariance,
        observation_noise_mean=observation_oas.mean,
        whitening=identity,
        coloring=identity,
        noaction_b=noaction_b,
        receipts=receipts,
    )


def _fit_normal_dynamics(
        clean_z: torch.Tensor, observed_z: torch.Tensor,
        actions: torch.Tensor) -> Dynamics:
    clean_x = clean_z - clean_z[:, :1]
    observed_x = observed_z - clean_z[:, :1]
    episodes, length, dimension = clean_x.shape
    state_values = clean_x[:, 1:].reshape(-1, dimension)
    state_oas, state_oas_diagnostics = centered_oas_from_values(
        state_values, label="replay-state-whitening")
    whitening, coloring = _matrix_sqrt_factors(state_oas.covariance)
    clean_q = clean_x @ whitening.T
    observed_q = observed_x @ whitening.T
    preliminary = fit_linear_dynamics(
        clean_q, actions, identity_A=False, anchor_centered=True)
    normal_A, projection = project_real_normal_stable(preliminary.raw_A)

    source = clean_q[:, :-1].reshape(-1, dimension)
    target = clean_q[:, 1:].reshape(-1, dimension)
    native_actions = actions.reshape(-1, actions.shape[-1])
    standardized, action_mean, action_std = _standardize_actions(native_actions)
    B_standard, b_standard, residual = _refit_action_and_bias(
        source, target, standardized, normal_A)
    B, b = _native_action_parameters(
        B_standard, b_standard, action_mean, action_std)
    process_oas, process_oas_diagnostics = centered_oas_from_values(
        residual, label="replay-normal-process")
    observation_noise = (observed_q - clean_q)[:, 1:].reshape(-1, dimension)
    observation_oas, observation_oas_diagnostics = centered_oas_from_values(
        observation_noise, label="replay-normal-observation")
    noaction_b = (target - source @ normal_A.T).mean(dim=0)
    receipts = {
        "fit_coordinate": "anchor_relative_oas_whitened",
        "state_oas_shrinkage": state_oas.shrinkage,
        "state_whitening_condition": state_oas.condition,
        "state_centered_oas": state_oas_diagnostics,
        "whitening_roundtrip_max_abs": _finite_float(
            (whitening @ coloring - torch.eye(dimension, dtype=torch.float64)).abs().max()),
        "process_oas_shrinkage": process_oas.shrinkage,
        "process_covariance_condition": process_oas.condition,
        "observation_oas_shrinkage": observation_oas.shrinkage,
        "observation_covariance_condition": observation_oas.condition,
        "process_centered_oas": process_oas_diagnostics,
        "observation_centered_oas": observation_oas_diagnostics,
        "action_B_rank": int(torch.linalg.matrix_rank(B)),
        "normal_projection": projection,
        **operator_algebra_receipts(normal_A, length - 1),
    }
    return Dynamics(
        A=normal_A,
        B=B,
        b=b,
        Q=process_oas.covariance,
        R=observation_oas.covariance,
        observation_noise_mean=observation_oas.mean,
        whitening=whitening,
        coloring=coloring,
        noaction_b=noaction_b,
        receipts=receipts,
    )


@dataclass
class _Accumulator:
    prior_squared_error: float = 0.0
    posterior_squared_error: float = 0.0
    target_squared: float = 0.0
    elements: int = 0

    def add(
            self, prior_error: torch.Tensor, posterior_error: torch.Tensor,
            target: torch.Tensor) -> None:
        self.prior_squared_error += float(prior_error.square().sum())
        self.posterior_squared_error += float(posterior_error.square().sum())
        self.target_squared += float(target.square().sum())
        self.elements += target.numel()

    def result(self) -> dict[str, float]:
        if not self.elements or self.target_squared <= 0.0:
            raise RuntimeError("recursive replay has no nonzero clean target energy")
        prior_mse = self.prior_squared_error / self.elements
        posterior_mse = self.posterior_squared_error / self.elements
        target_mse = self.target_squared / self.elements
        return {
            "recursive_clean_prior_mse": prior_mse,
            "recursive_clean_posterior_mse": posterior_mse,
            "recursive_clean_target_mean_square": target_mse,
            "recursive_clean_prior_nmse": prior_mse / target_mse,
            "recursive_clean_posterior_nmse": posterior_mse / target_mse,
            "posterior_to_prior_mse_ratio": posterior_mse / max(
                prior_mse, torch.finfo(torch.float64).tiny),
            "evaluated_scalar_elements": self.elements,
        }


def _psd_min(value: torch.Tensor) -> float:
    return _finite_float(torch.linalg.eigvalsh(_sym(value)).min())


def _replay_fixed_gain(
        clean_z: torch.Tensor, observed_z: torch.Tensor, actions: torch.Tensor,
        dynamics: Dynamics, mode: str, *, innovation_fit: Any | None = None,
        ) -> dict[str, Any]:
    if mode not in {"old", "identity", "riccati"}:
        raise ValueError(f"unknown fixed replay mode {mode!r}")
    clean_x = clean_z - clean_z[:, :1]
    observed_x = observed_z - clean_z[:, :1]
    clean_q = dynamics.to_coordinate(clean_x)
    observed_q = dynamics.to_coordinate(observed_x)
    state = clean_q[:, 0].clone()
    accumulator = _Accumulator()
    dimension = clean_q.shape[-1]
    identity = torch.eye(dimension, dtype=torch.float64)
    covariance = torch.zeros(dimension, dimension, dtype=torch.float64)
    covariance_min = 0.0
    gain_singular_max = 0.0
    gain_singular_min = math.inf
    gains = []
    for step in range(actions.shape[1]):
        prior = state @ dynamics.A.T + actions[:, step] @ dynamics.B.T + dynamics.b
        innovation = observed_q[:, step + 1] - prior
        if mode == "old":
            if innovation_fit is None:
                raise ValueError("old replay requires its opposite-fold innovation fit")
            gain = innovation_fit.lmmse_K
            correction = (
                innovation_fit.clean_mean
                + (innovation - innovation_fit.observed_mean) @ gain.T)
        elif mode == "identity":
            gain = identity
            correction = innovation
        else:
            covariance_prior = (
                dynamics.A @ covariance @ dynamics.A.T + dynamics.Q)
            innovation_covariance = _sym(covariance_prior + dynamics.R)
            gain = covariance_prior @ _machine_pinv(
                innovation_covariance, hermitian=True)
            correction = (
                innovation - dynamics.observation_noise_mean) @ gain.T
            residual = identity - gain
            covariance = _sym(
                residual @ covariance_prior @ residual.T
                + gain @ dynamics.R @ gain.T)
            covariance_min = min(covariance_min, _psd_min(covariance))
        # Preserve the identity control algebraically and bitwise; ``prior +
        # (observed-prior)`` can otherwise retain a roundoff ulp.
        state = observed_q[:, step + 1].clone() if mode == "identity" else prior + correction
        prior_error = dynamics.from_coordinate(prior - clean_q[:, step + 1])
        posterior_error = dynamics.from_coordinate(state - clean_q[:, step + 1])
        target = clean_x[:, step + 1]
        accumulator.add(prior_error, posterior_error, target)
        singular = torch.linalg.svdvals(gain)
        gain_singular_max = max(gain_singular_max, float(singular.max()))
        gain_singular_min = min(gain_singular_min, float(singular.min()))
        gains.append(gain)
    result = accumulator.result()
    result["algebra"] = {
        "gain_steps": len(gains),
        "gain_singular_max": gain_singular_max,
        "gain_singular_min": gain_singular_min,
        "gain_sequence_sha256": _tensor_sha256(torch.stack(gains)),
        "posterior_covariance_min_eigenvalue": covariance_min,
        "process_covariance_min_eigenvalue": _psd_min(dynamics.Q),
        "observation_covariance_min_eigenvalue": _psd_min(dynamics.R),
        "identity_posterior_observation_max_abs": (
            _finite_float((state - observed_q[:, -1]).abs().max())
            if mode == "identity" else None),
        "old_clean_history_centered_oas": (
            innovation_fit.receipts if mode == "old" else None),
    }
    if mode == "identity" and result["algebra"]["identity_posterior_observation_max_abs"] != 0.0:
        raise RuntimeError("identity correction did not reproduce the observed embedding")
    return result


def _joint_lmmse(
        desired: torch.Tensor, innovation: torch.Tensor,
        *, label: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    joint, centered_diagnostics = centered_oas_from_values(
        torch.cat((desired, innovation), dim=-1), label=label)
    dimension = desired.shape[-1]
    covariance = joint.covariance
    gain = covariance[:dimension, dimension:] @ _machine_pinv(
        covariance[dimension:, dimension:], hermitian=True)
    return (
        gain,
        joint.mean[:dimension],
        joint.mean[dimension:],
        {
            "samples": len(desired),
            "oas_shrinkage": joint.shrinkage,
            "joint_covariance_condition": joint.condition,
            "gain_singular_max": _finite_float(torch.linalg.svdvals(gain).max()),
            "centered_oas": centered_diagnostics,
        },
    )


def _fit_clean_history_innovation(
        clean_z: torch.Tensor, observed_z: torch.Tensor, actions: torch.Tensor,
        dynamics: Dynamics) -> SimpleNamespace:
    """Replay V12's clean-reset fit with stable explicitly centered joint OAS."""
    clean_x = clean_z - clean_z[:, :1]
    observed_x = observed_z - clean_z[:, :1]
    clean_q = dynamics.to_coordinate(clean_x)
    observed_q = dynamics.to_coordinate(observed_x)
    state = clean_q[:, 0].clone()
    clean_innovations = []
    observed_innovations = []
    for step in range(actions.shape[1]):
        prior = (
            state @ dynamics.A.T
            + actions[:, step] @ dynamics.B.T + dynamics.b)
        clean_innovation = clean_q[:, step + 1] - prior
        observed_innovation = observed_q[:, step + 1] - prior
        clean_innovations.append(clean_innovation)
        observed_innovations.append(observed_innovation)
        # The old fit's defining exposure mismatch: every next prior starts from an
        # exactly clean-reset posterior, never from its recursively corrupted history.
        state = clean_q[:, step + 1].clone()
    clean_values = torch.cat(clean_innovations)
    observed_values = torch.cat(observed_innovations)
    gain, clean_mean, observed_mean, receipts = _joint_lmmse(
        clean_values, observed_values, label="old-clean-history-innovation")
    receipts = {
        **receipts,
        "history_policy": "exact_clean_reset_after_every_step",
        "horizons": actions.shape[1],
        "episodes": clean_z.shape[0],
    }
    return SimpleNamespace(
        lmmse_K=gain,
        clean_mean=clean_mean,
        observed_mean=observed_mean,
        receipts=receipts,
    )


def _replay_crossfit_deployed(
        clean_z: torch.Tensor, observed_z: torch.Tensor, actions: torch.Tensor,
        folds: tuple[torch.Tensor, torch.Tensor],
        dynamics: tuple[Dynamics, Dynamics]) -> dict[str, Any]:
    """Fit each horizon's gain on the other fold's deployed-history innovations."""
    clean_x = clean_z - clean_z[:, :1]
    observed_x = observed_z - clean_z[:, :1]
    states: dict[int, torch.Tensor] = {}
    clean_q: dict[int, torch.Tensor] = {}
    observed_q: dict[int, torch.Tensor] = {}
    accumulators = {0: _Accumulator(), 1: _Accumulator()}
    for fold_id, indices in enumerate(folds):
        dyn = dynamics[fold_id]
        clean_q[fold_id] = dyn.to_coordinate(clean_x[indices])
        observed_q[fold_id] = dyn.to_coordinate(observed_x[indices])
        states[fold_id] = clean_q[fold_id][:, 0].clone()
    gain_receipts = []
    gain_tensors = []
    for step in range(actions.shape[1]):
        priors: dict[int, torch.Tensor] = {}
        innovations: dict[int, torch.Tensor] = {}
        desired: dict[int, torch.Tensor] = {}
        fits = {}
        for fold_id, indices in enumerate(folds):
            dyn = dynamics[fold_id]
            priors[fold_id] = (
                states[fold_id] @ dyn.A.T
                + actions[indices, step] @ dyn.B.T + dyn.b)
            innovations[fold_id] = (
                observed_q[fold_id][:, step + 1] - priors[fold_id])
            desired[fold_id] = clean_q[fold_id][:, step + 1] - priors[fold_id]
        # A gain applied to fold f is fitted exclusively from fold 1-f.  Since all
        # earlier fold states obeyed the same rule, the source histories are deployed
        # recursively and never corrected by a gain fitted from their own episodes.
        for target_fold in (0, 1):
            source_fold = 1 - target_fold
            fits[target_fold] = _joint_lmmse(
                desired[source_fold], innovations[source_fold],
                label=f"deployed-h{step + 1}-source-fold{source_fold}")
        for fold_id, indices in enumerate(folds):
            gain, clean_mean, observed_mean, receipt = fits[fold_id]
            correction = clean_mean + (
                innovations[fold_id] - observed_mean) @ gain.T
            states[fold_id] = priors[fold_id] + correction
            dyn = dynamics[fold_id]
            accumulators[fold_id].add(
                dyn.from_coordinate(priors[fold_id] - clean_q[fold_id][:, step + 1]),
                dyn.from_coordinate(states[fold_id] - clean_q[fold_id][:, step + 1]),
                clean_x[indices, step + 1],
            )
            gain_receipts.append({
                "horizon": step + 1,
                "target_fold": fold_id,
                "source_fold": 1 - fold_id,
                **receipt,
            })
            gain_tensors.append(gain)
    fold_results = [accumulators[index].result() for index in (0, 1)]
    aggregate = _combine_fold_results(fold_results)
    aggregate["folds"] = fold_results
    aggregate["algebra"] = {
        "gain_fits": len(gain_receipts),
        "every_gain_cross_fitted": all(
            row["target_fold"] != row["source_fold"] for row in gain_receipts),
        "gain_sequence_sha256": _tensor_sha256(torch.stack(gain_tensors)),
        "gain_singular_max": max(row["gain_singular_max"] for row in gain_receipts),
        "gain_receipts": gain_receipts,
    }
    return aggregate


def _combine_fold_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    elements = sum(int(row["evaluated_scalar_elements"]) for row in rows)
    if elements < 1:
        raise RuntimeError("cannot combine empty fold results")
    prior_mse = sum(
        float(row["recursive_clean_prior_mse"]) * int(row["evaluated_scalar_elements"])
        for row in rows) / elements
    posterior_mse = sum(
        float(row["recursive_clean_posterior_mse"]) * int(row["evaluated_scalar_elements"])
        for row in rows) / elements
    target_mse = sum(
        float(row["recursive_clean_target_mean_square"])
        * int(row["evaluated_scalar_elements"]) for row in rows) / elements
    return {
        "recursive_clean_prior_mse": prior_mse,
        "recursive_clean_posterior_mse": posterior_mse,
        "recursive_clean_target_mean_square": target_mse,
        "recursive_clean_prior_nmse": prior_mse / target_mse,
        "recursive_clean_posterior_nmse": posterior_mse / target_mse,
        "posterior_to_prior_mse_ratio": posterior_mse / max(
            prior_mse, torch.finfo(torch.float64).tiny),
        "evaluated_scalar_elements": elements,
    }


def _action_receipts(
        clean_z: torch.Tensor, actions: torch.Tensor,
        folds: tuple[torch.Tensor, torch.Tensor], dynamics: tuple[Dynamics, Dynamics],
        ) -> dict[str, Any]:
    partial = []
    for fold_id, indices in enumerate(folds):
        dyn = dynamics[fold_id]
        x = clean_z[indices] - clean_z[indices, :1]
        q = dyn.to_coordinate(x)
        source = q[:, :-1].reshape(-1, q.shape[-1])
        target = q[:, 1:].reshape(-1, q.shape[-1])
        action = actions[indices].reshape(-1, actions.shape[-1])
        full = source @ dyn.A.T + action @ dyn.B.T + dyn.b
        base = source @ dyn.A.T + dyn.noaction_b
        full_sse = (target - full).square().sum()
        base_sse = (target - base).square().sum().clamp_min(
            torch.finfo(torch.float64).tiny)
        partial.append(_finite_float(1.0 - full_sse / base_sse))
    # dyn[0] was trained on odd episodes and dyn[1] on even episodes.  Compare in
    # the original anchor-relative coordinate so whitening cannot inflate agreement.
    original_B = [dyn.coloring @ dyn.B for dyn in dynamics]
    denominator = (
        original_B[0].norm() * original_B[1].norm()).clamp_min(
            torch.finfo(torch.float64).tiny)
    scale = torch.maximum(original_B[0].norm(), original_B[1].norm()).clamp_min(
        torch.finfo(torch.float64).tiny)
    return {
        "held_fold_partial_r2": partial,
        "both_fold_partial_r2_positive": all(value > 0.0 for value in partial),
        "mean_partial_r2": sum(partial) / len(partial),
        "B_fold_cosine": _finite_float(
            (original_B[0] * original_B[1]).sum() / denominator),
        "B_fold_relative_disagreement": _finite_float(
            (original_B[0] - original_B[1]).norm() / scale),
        "B_source_fold_for_target_even": "odd_episode_indices",
        "B_source_fold_for_target_odd": "even_episode_indices",
    }


def analyze_task_embeddings(
        clean_z: torch.Tensor, observed_z: torch.Tensor,
        actions: torch.Tensor) -> dict[str, Any]:
    """Run all five train-only parity-fold replays on one encoder snapshot."""
    episodes, length, dimension, action_dim = _validate_embeddings(
        clean_z, observed_z, actions)
    folds = immutable_parity_folds(episodes)
    if float((clean_z[:, 0] - observed_z[:, 0]).abs().max()) != 0.0:
        raise RuntimeError("recursive replay requires the registered exact clean anchor")

    # dynamics[target_fold] is always fitted from the opposite fold.
    current = tuple(
        _fit_current_dynamics(
            clean_z[folds[1 - target]], observed_z[folds[1 - target]],
            actions[folds[1 - target]])
        for target in (0, 1))
    normal = tuple(
        _fit_normal_dynamics(
            clean_z[folds[1 - target]], observed_z[folds[1 - target]],
            actions[folds[1 - target]])
        for target in (0, 1))

    results: dict[str, Any] = {}
    old_rows, identity_rows, riccati_rows, normal_rows = [], [], [], []
    for target_fold, indices in enumerate(folds):
        source = folds[1 - target_fold]
        dyn = current[target_fold]
        old_fit = _fit_clean_history_innovation(
            clean_z[source], observed_z[source], actions[source], dyn)
        old_rows.append(_replay_fixed_gain(
            clean_z[indices], observed_z[indices], actions[indices], dyn,
            "old", innovation_fit=old_fit))
        identity_rows.append(_replay_fixed_gain(
            clean_z[indices], observed_z[indices], actions[indices], dyn, "identity"))
        riccati_rows.append(_replay_fixed_gain(
            clean_z[indices], observed_z[indices], actions[indices], dyn, "riccati"))
        normal_rows.append(_replay_fixed_gain(
            clean_z[indices], observed_z[indices], actions[indices],
            normal[target_fold], "riccati"))

    for name, rows in (
            (VARIANTS[0], old_rows),
            (VARIANTS[1], identity_rows),
            (VARIANTS[3], riccati_rows),
            (VARIANTS[4], normal_rows)):
        combined = _combine_fold_results(rows)
        combined["folds"] = rows
        results[name] = combined
    results[VARIANTS[2]] = _replay_crossfit_deployed(
        clean_z, observed_z, actions, folds, current)
    results = {name: results[name] for name in VARIANTS}

    action = _action_receipts(clean_z, actions, folds, current)
    return {
        "episodes": episodes,
        "length": length,
        "dimension": dimension,
        "action_dim": action_dim,
        "fold_contract": {
            "assignment": "episode_index_mod_2",
            "even_count": len(folds[0]),
            "odd_count": len(folds[1]),
            "even_index_sha256": _tensor_sha256(folds[0]),
            "odd_index_sha256": _tensor_sha256(folds[1]),
            "disjoint_exhaustive": True,
        },
        "action_identification": action,
        "operators": {
            "current_a_by_target_fold": [dyn.receipts for dyn in current],
            "normal_stable_a_by_target_fold": [dyn.receipts for dyn in normal],
        },
        "variants": results,
    }


def _relative_reduction(candidate: float, reference: float) -> float:
    return (reference - candidate) / max(abs(reference), 1e-30)


def _within_relative(candidate: float, reference: float, tolerance: float) -> bool:
    """Two-sided relative agreement used by diagnostic equivalence gates."""
    if tolerance < 0.0 or not all(map(math.isfinite, (candidate, reference, tolerance))):
        raise ValueError("relative agreement requires finite values and nonnegative tolerance")
    return abs(candidate / max(abs(reference), 1e-30) - 1.0) <= tolerance


def aggregate_and_gate(tasks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(tasks) != set(TASKS):
        raise ValueError(f"gate requires exactly the frozen tasks {TASKS}")
    means = {
        variant: float(np.mean([
            tasks[task]["variants"][variant]["recursive_clean_prior_nmse"]
            for task in TASKS]))
        for variant in VARIANTS
    }
    old = VARIANTS[0]
    deployed = VARIANTS[2]
    riccati = VARIANTS[3]
    normal = VARIANTS[4]
    recursive_task_wins = sum(
        _relative_reduction(
            tasks[task]["variants"][normal]["recursive_clean_prior_nmse"],
            tasks[task]["variants"][old]["recursive_clean_prior_nmse"]) >= 0.10
        for task in TASKS)
    action_tasks = sum(
        bool(tasks[task]["action_identification"]["both_fold_partial_r2_positive"])
        for task in TASKS)
    checks = {
        "normal_riccati_at_least_10pct_better_than_old_on_3of4": (
            recursive_task_wins >= 3),
        "riccati_within_5pct_of_deployed_history_lmmse_equal_task_mean": (
            _within_relative(means[riccati], means[deployed], 0.05)),
        "normal_a_noninferior_to_current_a_riccati_within_1pct": (
            means[normal] <= 1.01 * means[riccati]),
        "positive_action_partial_r2_both_folds_on_3of4": action_tasks >= 3,
        "all_deployed_gains_cross_fitted": all(
            tasks[task]["variants"][deployed]["algebra"]["every_gain_cross_fitted"]
            for task in TASKS),
        "all_normal_operators_stable_and_numerically_normal": all(
            fold["singular_max"] <= FLOAT32_STABILITY_CAP + 1e-10
            and fold["normality_relative_commutator"] <= 1e-10
            and fold["transient_singular_max_through_horizon"] <= 1.0 + 1e-10
            for task in TASKS
            for fold in tasks[task]["operators"]["normal_stable_a_by_target_fold"]),
        "all_riccati_covariances_psd_to_roundoff": all(
            fold["algebra"]["posterior_covariance_min_eigenvalue"] >= -1e-10
            for task in TASKS
            for variant in (riccati, normal)
            for fold in tasks[task]["variants"][variant]["folds"]),
    }
    return {
        "equal_task_mean_recursive_clean_prior_nmse": means,
        "normal_vs_old_relative_reduction": _relative_reduction(
            means[normal], means[old]),
        "riccati_vs_deployed_relative_difference": (
            means[riccati] / means[deployed] - 1.0),
        "normal_vs_current_riccati_relative_difference": (
            means[normal] / means[riccati] - 1.0),
        "normal_10pct_task_wins": recursive_task_wins,
        "positive_action_partial_r2_tasks": action_tasks,
        "checks": checks,
        "stage_a_pass": all(checks.values()),
        "decision": "PROCEED_TO_EXCLUDED_V13_DIAGNOSTIC" if all(checks.values()) else "STOP",
    }


def _load_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def validate_screen_ready(screen_root: Path, epochs: int) -> dict[str, Any]:
    """Validate 28 bundles, then interpret the frozen analyzer's rank-only failure.

    The frozen analyzer labels ``anchor effective rank < 16`` as an integrity error and
    consequently reports only 21 completed cells.  Effective rank is a scientific
    representation-quality result, not evidence that a local artifact is missing or
    corrupt.  This admission check therefore validates every bundle independently and
    accepts that negative analyzer receipt *only* when its complete error list exactly
    equals the rank failures recomputed from the independently validated metrics.
    """
    screen_root = screen_root.resolve()
    if (screen_root / ".siro_v12_screen.lock").exists():
        raise RuntimeError(f"refusing recursive replay while SIRO screen is active: {screen_root}")
    protocol_path = screen_root / "screen_protocol.json"
    runs_path = screen_root / "screen_runs.json"
    analysis_path = screen_root / "screen_analysis.json"
    decision_path = screen_root / "screen_decision.json"
    for path in (protocol_path, runs_path, analysis_path, decision_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"recursive replay requires a completed, analyzed screen; missing {path}")
    protocol = _load_json(protocol_path)
    if protocol.get("seed") != SEED or protocol.get("epochs") != epochs:
        raise RuntimeError("screen protocol seed/epoch contract mismatch")
    if protocol.get("runs") != len(protocol.get("tasks", ())) * len(
            protocol.get("designs", ())):
        raise RuntimeError("screen protocol run ledger is inconsistent")
    runs = _load_json(runs_path)
    if len(runs) != protocol["runs"]:
        raise RuntimeError("screen run ledger is incomplete")
    for relative, expected in protocol.get("source_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"hashed source drifted since the screen: {relative}")

    # Reuse only the independent audit's artifact/protocol validators—not its analyzer
    # verdict—to perform a direct, read-only validation of all models, metrics, rollout
    # hashes, W&B receipts, checkpoint histories/fits, and the runner completion ledger.
    independently_validated_protocol = independent_audit.validate_protocol(screen_root)
    if independently_validated_protocol.get("epochs") != epochs:
        raise RuntimeError("independent bundle validator epoch contract mismatch")
    independent_audit.validate_root_directory_set(screen_root)
    independent_rows = []
    for (task, design), directory in independent_audit.expected_run_directories(
            screen_root).items():
        independent_rows.append(independent_audit.validate_cell(
            directory, task=task, design=design,
            protocol=independently_validated_protocol))
    if len(independent_rows) != protocol["runs"]:
        raise RuntimeError("independent bundle validation did not cover all 28 cells")
    run_ids = [row["wandb_run_id"] for row in independent_rows]
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("independently validated bundles contain duplicate W&B run IDs")
    independent_audit.validate_runner_receipt(
        screen_root, independent_rows, independently_validated_protocol)

    rows_by_pair = {
        (row["task"], row["design"]): row for row in independent_rows}
    rank_failures = []
    expected_rank_errors = []
    for task in TASKS:
        for design in independent_audit.SIRO_DESIGNS:
            row = rows_by_pair[(task, design)]
            rank = _finite_float(row["anchor_covariance_effective_rank"])
            if rank < 16.0:
                error = f"{task}/{design}: fit anchor effective rank below 16"
                rank_failures.append({
                    "task": task,
                    "design": design,
                    "anchor_covariance_effective_rank": rank,
                    "threshold": 16.0,
                    "frozen_analyzer_error": error,
                })
                expected_rank_errors.append(error)

    analysis = _load_json(analysis_path)
    decision = _load_json(decision_path)
    reported_errors = analysis.get("integrity_errors")
    analyzer_common = (
        analysis.get("schema_version") == 1
        and analysis.get("scope") == protocol.get("scope")
        and analysis.get("study") == protocol.get("study")
        and analysis.get("seed") == SEED
        and analysis.get("epochs") == epochs
        and analysis.get("expected_cells") == protocol["runs"])
    if not analyzer_common:
        raise RuntimeError("frozen analyzer receipt metadata does not match the protocol")
    if reported_errors != expected_rank_errors:
        raise RuntimeError(
            "frozen analyzer errors are not exactly the recomputed rank<16 failures: "
            f"reported={reported_errors!r}, expected={expected_rank_errors!r}")
    if expected_rank_errors:
        expected_completed = protocol["runs"] - len(expected_rank_errors)
        if (analysis.get("integrity_passed") is not False
                or analysis.get("completed_cells") != expected_completed
                or analysis.get("status") != "INCOMPLETE_OR_INVALID"
                or analysis.get("scientific_gate_passed") is not False
                or analysis.get("continue_to_100_epochs") is not False):
            raise RuntimeError(
                "frozen analyzer negative receipt is inconsistent with its rank-only errors")
        analyzer_semantic_classification = (
            "REPRESENTATION_RANK_GATE_MISCLASSIFIED_AS_ARTIFACT_INTEGRITY")
    else:
        if (analysis.get("integrity_passed") is not True
                or analysis.get("completed_cells") != protocol["runs"]):
            raise RuntimeError(
                "frozen analyzer reported an integrity failure without recomputed rank errors")
        analyzer_semantic_classification = "NO_SEMANTIC_MISMATCH"
    decision_pairs = {
        "status": "status",
        "integrity_passed": "integrity_passed",
        "continue_to_100_epochs": "continue_to_100_epochs",
        "scientific_gate_passed": "scientific_gate_passed",
    }
    if any(decision.get(key) != analysis.get(source)
           for key, source in decision_pairs.items()):
        raise RuntimeError("screen decision receipt disagrees with the frozen analysis")
    if decision.get("automatic_launch_performed") is not False:
        raise RuntimeError("screen decision unexpectedly records an automatic launch")

    # Stage A is admitted only after the separately generated, read-only audit agrees
    # that all artifacts are valid and that the same seven strings are representation
    # failures.  The receipt lives outside the immutable source-screen namespace.
    audit_receipt_path = (
        screen_root.parent / f"{screen_root.name}_audit" / "integrity_report.json")
    if not audit_receipt_path.is_file():
        raise FileNotFoundError(
            f"recursive replay requires the corrected independent audit receipt: "
            f"{audit_receipt_path}")
    audit_receipt = _load_json(audit_receipt_path)
    expected_audit = {
        "scope": "independent_read_only_siro_v12_screen_audit",
        "root": str(screen_root),
        "status": "PASS_COMPLETE_NEGATIVE",
        "passed": True,
        "expected_cells": protocol["runs"],
        "validated_cells": protocol["runs"],
        "protocol_validated": True,
        "artifact_integrity_passed": True,
        "analyzer_receipt_consistent": True,
        "representation_gate_passed": False,
        "representation_failures": expected_rank_errors,
        "errors": [],
    }
    mismatched_audit = {
        key: {"actual": audit_receipt.get(key), "expected": value}
        for key, value in expected_audit.items()
        if audit_receipt.get(key) != value}
    if mismatched_audit:
        raise RuntimeError(
            f"corrected independent audit receipt mismatch: {mismatched_audit!r}")

    checkpoints = {}
    for task in TASKS:
        run_dir = run_directory(screen_root, task, "sirov12")
        paths = {
            name: run_dir / filename for name, filename in (
                ("checkpoint", "model.pt"),
                ("metrics", "metrics.json"),
                ("rollout", "eval_rollout.npz"),
                ("wandb_receipt", "wandb_run.json"),
            )}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"incomplete full-SIRO checkpoint bundle: {path}")
        metrics = _load_json(paths["metrics"])
        receipt = _load_json(paths["wandb_receipt"])
        expected_env = f"dmc:{task}"
        if (metrics.get("env"), metrics.get("design"), metrics.get("seed"),
                metrics.get("epochs"), metrics.get("fit_updates")) != (
                expected_env, "sirov12", SEED, epochs, epochs + 1):
            raise RuntimeError(f"{task}: full-SIRO metrics contract is incomplete")
        if receipt.get("state") != "finished" or receipt.get("mode") != "online":
            raise RuntimeError(f"{task}: W&B receipt is not finished online")
        if sha256_file(paths["rollout"]) != metrics.get("eval_rollout_sha256"):
            raise RuntimeError(f"{task}: rollout digest mismatch")
        checkpoints[task] = {
            "run_dir": str(run_dir),
            **{f"{name}_path": str(path) for name, path in paths.items()},
            **{f"{name}_sha256": sha256_file(path) for name, path in paths.items()},
        }
    return {
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "screen_runs_path": str(runs_path),
        "screen_runs_sha256": sha256_file(runs_path),
        "screen_analysis_path": str(analysis_path),
        "screen_analysis_sha256": sha256_file(analysis_path),
        "screen_decision_path": str(decision_path),
        "screen_decision_sha256": sha256_file(decision_path),
        "corrected_independent_audit_receipt": {
            "path": str(audit_receipt_path),
            "sha256": sha256_file(audit_receipt_path),
            "status": audit_receipt["status"],
            "passed": audit_receipt["passed"],
            "validated_cells": audit_receipt["validated_cells"],
            "artifact_integrity_passed": audit_receipt[
                "artifact_integrity_passed"],
            "representation_gate_passed": audit_receipt[
                "representation_gate_passed"],
            "representation_failures": audit_receipt[
                "representation_failures"],
        },
        "independent_bundle_validation": {
            "validator_source": str(Path(independent_audit.__file__).resolve()),
            "validator_source_sha256": sha256_file(
                Path(independent_audit.__file__).resolve()),
            "validated_cells": len(independent_rows),
            "expected_cells": protocol["runs"],
            "unique_wandb_run_ids": len(set(run_ids)),
            "runner_receipt_validated": True,
            "artifact_bundle_integrity_passed": True,
            "cells": [{
                "task": row["task"],
                "design": row["design"],
                "directory": row["directory"],
                "rollout_sha256": row["rollout_sha256"],
                "wandb_run_id": row["wandb_run_id"],
            } for row in independent_rows],
        },
        "frozen_analyzer_semantic_receipt": {
            "classification": analyzer_semantic_classification,
            "reported_integrity_passed": analysis.get("integrity_passed"),
            "reported_completed_cells": analysis.get("completed_cells"),
            "independently_validated_artifact_cells": len(independent_rows),
            "reported_errors": reported_errors,
            "recomputed_rank_failures": rank_failures,
            "reported_errors_exactly_match_recomputed_rank_failures": True,
            "negative_receipt_accepted": bool(expected_rank_errors),
            "interpretation": (
                "anchor effective rank below 16 is retained as a negative scientific "
                "representation result; it is not treated as missing/corrupt artifact "
                "integrity after direct validation of all 28 bundles"),
        },
        "checkpoints": checkpoints,
    }


def _checkpoint_args(saved: Mapping[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    required = (
        "train_data", "memory_mode", "seed", "epochs", "img_size", "patch_size",
        "embed_dim", "encoder_layers", "encoder_heads", "predictor_layers",
        "predictor_heads", "history_len", "dropout", "sigreg_lambda",
        "sigreg_projections")
    missing = [key for key in required if key not in saved]
    if missing:
        raise RuntimeError(f"checkpoint args omit required fields: {missing}")
    values = dict(saved)
    values["batch_size"] = args.batch_size
    values["num_workers"] = args.num_workers
    values["memory_mode"] = "sirov12"
    return SimpleNamespace(**values)


def _encode_checkpoint(
        checkpoint_path: Path, args: argparse.Namespace,
        device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not {
            "args", "model_state_dict", "final_metrics", "final_operator_fit"} <= set(checkpoint):
        raise RuntimeError(f"invalid full-SIRO checkpoint schema: {checkpoint_path}")
    saved_args = _checkpoint_args(checkpoint["args"], args)
    if saved_args.seed != SEED or saved_args.epochs != args.epochs:
        raise RuntimeError("checkpoint seed/epoch contract mismatch")
    train_path = Path(saved_args.train_data)
    if not train_path.is_absolute():
        train_path = (ROOT / train_path).resolve()
    final_metrics = checkpoint["final_metrics"]
    if sha256_file(train_path) != final_metrics.get("train_data_sha256"):
        raise RuntimeError("checkpoint train-cache digest mismatch")
    action_dim = int(final_metrics["action_dim"])
    model = build_model(saved_args, action_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    clean = V11TrajectoryDataset(
        train_path, "clean", int(checkpoint["args"]["corruption_seed"]),
        int(saved_args.history_len))
    observed = V11TrajectoryDataset(
        train_path, "train", int(checkpoint["args"]["corruption_seed"]),
        int(saved_args.history_len))
    clean_z, observed_z, actions = collect_detached_fit_views(
        model, clean, observed, saved_args, device)
    return clean_z, observed_z, actions, {
        "train_data": str(train_path),
        "train_data_sha256": sha256_file(train_path),
        "checkpoint_final_fit_index": int(
            checkpoint["final_operator_fit"]["receipts"]["fit_index"]),
        "checkpoint_history_rows": len(checkpoint.get("history", ())),
        "clean_embedding_sha256": _tensor_sha256(clean_z),
        "observed_embedding_sha256": _tensor_sha256(observed_z),
        "actions_sha256": _tensor_sha256(actions),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen-root", type=Path,
        default=Path("outputs/hacssm_v12_screen_siro30"))
    parser.add_argument(
        "--report-root", type=Path, required=True,
        help="new output directory; the analyzer refuses to reuse it")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("invalid recursive replay execution budget")
    args.screen_root = (
        args.screen_root if args.screen_root.is_absolute()
        else (ROOT / args.screen_root).resolve())
    args.report_root = (
        args.report_root if args.report_root.is_absolute()
        else (ROOT / args.report_root).resolve())
    if args.report_root == args.screen_root or args.screen_root in args.report_root.parents:
        raise ValueError(
            "recursive replay report root must be outside the immutable source screen")
    if args.report_root.exists():
        raise FileExistsError(f"refusing to reuse report root {args.report_root}")
    readiness = validate_screen_ready(args.screen_root, args.epochs)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA replay requested but CUDA is unavailable")
    device = torch.device(args.device)
    # Reserve the fresh namespace only after all immutable-input checks pass.  A task
    # failure then has a guaranteed exclusive place for its fail-closed receipt.
    args.report_root.mkdir(parents=True, exist_ok=False)
    task_results = {}
    inputs = {}
    for task in TASKS:
        stage = "checkpoint_encoding"
        try:
            checkpoint_path = Path(
                readiness["checkpoints"][task]["checkpoint_path"])
            print(f"encoding completed full-SIRO checkpoint for {task}", flush=True)
            clean_z, observed_z, actions, receipt = _encode_checkpoint(
                checkpoint_path, args, device)
            inputs[task] = receipt
            stage = "recursive_operator_replay"
            print(f"running train-only recursive parity replay for {task}", flush=True)
            task_results[task] = analyze_task_embeddings(clean_z, observed_z, actions)
        except Exception as exc:  # fail closed, but retain evidence instead of a traceback
            failure = {
                "schema_version": SCHEMA_VERSION,
                "scope": "excluded_adaptive_v12b_train_only_recursive_replay",
                "status": "FAIL_CLOSED_TASK_REPLAY",
                "decision": "STOP",
                "failed_task": task,
                "failed_stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "explicit_replay_numerical_error": isinstance(
                    exc, ReplayNumericalError),
                "completed_tasks": list(task_results),
                "encoded_inputs": inputs,
                "partial_task_results": task_results,
                "screen_root": str(args.screen_root),
                "report_root": str(args.report_root),
                "readiness": readiness,
                "uses_validation": False,
                "uses_wandb_api": False,
                "optimizer_steps": 0,
                "source": str(Path(__file__).resolve()),
                "source_sha256": _sha256(Path(__file__).resolve()),
            }
            failure_path = args.report_root / "recursive_replay_failure.json"
            with failure_path.open("x") as stream:
                json.dump(
                    failure, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            print(json.dumps({
                "status": failure["status"],
                "decision": failure["decision"],
                "failed_task": task,
                "failed_stage": stage,
                "error_type": failure["error_type"],
                "error": failure["error"],
                "receipt": str(failure_path),
            }, indent=2, sort_keys=True), file=sys.stderr, flush=True)
            raise SystemExit(2) from None
    gate = aggregate_and_gate(task_results)
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "excluded_adaptive_v12b_train_only_recursive_replay",
        "status": "COMPLETE",
        "uses_validation": False,
        "uses_task_state": False,
        "uses_reward": False,
        "uses_wandb_api": False,
        "optimizer_steps": 0,
        "fold_assignment": "immutable_episode_index_mod_2",
        "screen_root": str(args.screen_root),
        "report_root": str(args.report_root),
        "epochs": args.epochs,
        "seed": SEED,
        "variants": list(VARIANTS),
        "readiness": readiness,
        "encoded_inputs": inputs,
        "tasks": task_results,
        "gate": gate,
        "source": str(Path(__file__).resolve()),
        "source_sha256": _sha256(Path(__file__).resolve()),
    }
    path = args.report_root / "recursive_replay_analysis.json"
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(gate, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
