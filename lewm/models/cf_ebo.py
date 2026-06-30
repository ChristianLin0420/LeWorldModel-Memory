"""Cross-fold-calibrated energy-bounded observer (CF-EBO-v14).

V13 showed that a normal state transition is not enough to make an observer
safe: its Riccati gain had a very large *future output* response.  This module
keeps V13's pooled, normal all-lag realization only as a coordinate discovery
step and replaces its deployed action/correction rules by three train-only,
self-supervised certificates:

* an observability-Gramian coordinate system in which
  ``F.T @ F + H.T @ H == P_obs`` for a machine-rank support projector;
* symmetric held-fold predictive-risk shrinkage for the action and correction
  paths; and
* a spectrally capped correction from whitened innovations, followed by a
  redescending self-normalized radial gate.

The deployed recurrence has no learned parameters, consumes only observations
and executed actions, and never uses reward, task state, validation data, or a
corruption label.  The V13 realization remains a compositional dependency and
is deliberately not relabelled as a new identification primitive.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, NamedTuple, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from lewm.models.cf_hiro import (
    FLOAT32_STABILITY_BOUNDARY,
    HIROState,
    _as_cpu_double,
    _machine_pinv,
    _matrix_rank,
    _oas_covariance,
    _refit_action_from_all_lags,
    _symmetrize,
    estimate_all_lag_markov_moments,
    realize_fold_agreement_hankel,
)


Tensor = torch.Tensor


CF_EBO_MODES = frozenset({
    "full", "nocorrect", "noaction", "norisk", "noenergycap", "noradial",
})


def _canonical_mode(mode: str) -> str:
    if mode not in CF_EBO_MODES:
        raise ValueError(f"unknown CF-EBO mode {mode!r}")
    return mode


@dataclass(frozen=True)
class EnergyRealization:
    """Observable energy coordinates derived from one stable V13 realization."""

    state_matrix: Tensor
    read_matrix: Tensor
    transform: Tensor
    inverse_transform: Tensor
    gramian: Tensor
    energy_support_projector: Tensor
    receipts: Mapping[str, Any]


@dataclass(frozen=True)
class RiskEstimate:
    """One directional positive-part empirical-Bayes risk certificate."""

    reliability: float
    mean_improvement: float
    variance_of_mean: float
    base_mean: float
    candidate_mean: float
    receipts: Mapping[str, Any]


@dataclass(frozen=True)
class CorrectionFit:
    """One paired-view correction map in observability-energy coordinates."""

    raw_matrix: Tensor
    deployed_matrix: Tensor
    innovation_covariance: Tensor
    innovation_whitener: Tensor
    innovation_rank: int
    receipts: Mapping[str, Any]


@dataclass(frozen=True)
class CEBOFit:
    """Complete detached fit installed by :class:`CrossFitEnergyBoundedObserverMemory`."""

    state_matrix: Tensor
    action_matrix: Tensor
    raw_action_matrix: Tensor
    read_matrix: Tensor
    correction_matrix: Tensor
    raw_correction_matrix: Tensor
    innovation_covariance: Tensor
    innovation_whitener: Tensor
    initial_map: Tensor
    output_projector: Tensor
    complement_projector: Tensor
    energy_support_projector: Tensor
    output_mean: Tensor
    action_mean: Tensor
    action_reliability: Tensor
    correction_reliability: Tensor
    innovation_rank: Tensor
    markov_even: Tensor
    markov_odd: Tensor
    receipts: Mapping[str, Any]


def _observable_energy_coordinates(state_matrix: Tensor, read_matrix: Tensor) -> EnergyRealization:
    """Return coordinates whose state energy equals total future read energy.

    For stable ``A`` and read ``C``, the infinite observability Gramian obeys
    ``W=A.T W A+C.T C``.  With ``h=W^(1/2)x``, the transformed operators obey
    ``F.T F+H.T H=P_obs``. Exact unobservable directions are removed only at the
    dtype/shape-derived machine cutoff, then represented by exact zero padding so
    V14 preserves the fixed V13 host schema. ``P_obs`` selects the active prefix.
    """
    state_matrix = _as_cpu_double(state_matrix, "state_matrix", 2)
    read_matrix = _as_cpu_double(read_matrix, "read_matrix", 2)
    state_dim = state_matrix.shape[0]
    if state_matrix.shape != (state_dim, state_dim):
        raise ValueError("state_matrix must be square")
    if read_matrix.shape[1] != state_dim:
        raise ValueError("read_matrix width must equal state dimension")
    radius = float(torch.linalg.eigvals(state_matrix).abs().max())
    if radius >= 1.0:
        raise ValueError("observability energy requires a strictly stable transition")
    try:
        from scipy.linalg import solve_discrete_lyapunov
    except ImportError as error:  # pragma: no cover - project declares scipy
        raise RuntimeError("CF-EBO fitting requires the declared scipy dependency") from error

    a = state_matrix.numpy()
    c = read_matrix.numpy()
    gramian_numpy = solve_discrete_lyapunov(a.T, c.T @ c)
    gramian = _symmetrize(torch.from_numpy(np.array(gramian_numpy, copy=True)).double())
    eigenvalues, eigenvectors = torch.linalg.eigh(gramian)
    scale = max(1.0, float(eigenvalues.abs().max()))
    cutoff = state_dim * torch.finfo(torch.float64).eps * scale
    keep = eigenvalues > cutoff
    rank = int(torch.count_nonzero(keep))
    positive = eigenvalues[keep]
    basis = eigenvectors[:, keep]
    transform = torch.zeros(state_dim, state_dim, dtype=torch.float64)
    inverse_transform = torch.zeros(state_dim, state_dim, dtype=torch.float64)
    state_energy = torch.zeros(state_dim, state_dim, dtype=torch.float64)
    read_energy = torch.zeros(read_matrix.shape[0], state_dim, dtype=torch.float64)
    support = torch.zeros(state_dim, state_dim, dtype=torch.float64)
    if rank:
        root = positive.sqrt()
        active_transform = root.unsqueeze(1) * basis.T
        active_inverse = basis * positive.rsqrt().unsqueeze(0)
        transform[:rank] = active_transform
        inverse_transform[:, :rank] = active_inverse
        state_energy[:rank, :rank] = active_transform @ state_matrix @ active_inverse
        read_energy[:, :rank] = read_matrix @ active_inverse
        support[:rank, :rank] = torch.eye(rank, dtype=torch.float64)
    dissipativity = state_energy.T @ state_energy + read_energy.T @ read_energy
    dissipativity_max = float((dissipativity - support).abs().max())
    dissipativity_operator = float(torch.linalg.matrix_norm(
        dissipativity - support, ord=2))
    lyapunov_residual = (
        gramian - state_matrix.T @ gramian @ state_matrix - read_matrix.T @ read_matrix)
    receipts = {
        "energy_coordinate_rule": "infinite_observability_gramian",
        "energy_lyapunov_solver": "scipy_solve_discrete_lyapunov",
        "energy_state_rank": rank,
        "energy_state_source_order": state_dim,
        "energy_inactive_padding": state_dim - rank,
        "energy_support_projector_rank": rank,
        "energy_support_projector_rule": "active_prefix_after_machine_eigentruncation",
        "energy_support_projector_symmetry_max_abs": float((support - support.T).abs().max()),
        "energy_support_projector_idempotence_max_abs": float(
            (support @ support - support).abs().max()),
        "energy_rank_cutoff": cutoff,
        "energy_gramian_minimum_active_eigenvalue": (
            float(positive.min()) if rank else 0.0),
        "energy_gramian_maximum_active_eigenvalue": (
            float(positive.max()) if rank else 0.0),
        "energy_gramian_active_condition": (
            float(positive.max() / positive.min()) if rank else 1.0),
        "energy_gramian_full_minimum_eigenvalue": float(eigenvalues.min()),
        "energy_lyapunov_relative_residual": float(
            lyapunov_residual.norm() / gramian.norm().clamp_min(torch.finfo(torch.float64).tiny)),
        "energy_dissipativity_max_abs": dissipativity_max,
        "energy_dissipativity_operator_norm": dissipativity_operator,
        "energy_identity": "F^T_F_plus_H^T_H_equals_Pobs",
        "future_read_energy_identity": (
            "sum_k_norm_H_Fk_delta_sq_equals_norm_Pobs_delta_sq"),
        "source_spectral_radius": radius,
        "energy_state_spectral_radius": float(torch.linalg.eigvals(state_energy).abs().max()),
        "energy_state_operator_norm": float(torch.linalg.matrix_norm(state_energy, ord=2)),
    }
    return EnergyRealization(
        state_matrix=state_energy,
        read_matrix=read_energy,
        transform=transform,
        inverse_transform=inverse_transform,
        gramian=gramian,
        energy_support_projector=support,
        receipts=receipts,
    )


def _directional_risk(base_losses: Tensor, candidate_losses: Tensor, label: str) -> RiskEstimate:
    """Positive-part EB reliability from paired held-fold episode risks."""
    base_losses = _as_cpu_double(base_losses, f"{label}_base_losses", 1)
    candidate_losses = _as_cpu_double(candidate_losses, f"{label}_candidate_losses", 1)
    if base_losses.shape != candidate_losses.shape or base_losses.numel() < 2:
        raise ValueError("paired risk needs at least two shape-matched episode losses")
    improvement = base_losses - candidate_losses
    mean = improvement.mean()
    variance_of_mean = improvement.var(unbiased=True) / improvement.numel()
    base_mean = base_losses.mean()
    candidate_mean = candidate_losses.mean()
    scale_square = max(
        float(base_mean.square()), float(mean.square()), float(variance_of_mean),
        torch.finfo(torch.float64).tiny)
    machine_floor = torch.finfo(torch.float64).eps * scale_square
    if float(mean) <= 0.0:
        reliability = 0.0
    else:
        reliability = float(
            (mean.square() - variance_of_mean).clamp_min(0.0)
            / (mean.square() + machine_floor))
    reliability = min(1.0, max(0.0, reliability))
    receipts = {
        f"{label}_episodes": int(improvement.numel()),
        f"{label}_base_mean": float(base_mean),
        f"{label}_candidate_mean": float(candidate_mean),
        f"{label}_mean_improvement": float(mean),
        f"{label}_variance_of_mean": float(variance_of_mean),
        f"{label}_machine_floor": machine_floor,
        f"{label}_positive_mean": bool(float(mean) > 0.0),
        f"{label}_eb_reliability": reliability,
    }
    return RiskEstimate(
        reliability=reliability,
        mean_improvement=float(mean),
        variance_of_mean=float(variance_of_mean),
        base_mean=float(base_mean),
        candidate_mean=float(candidate_mean),
        receipts=receipts,
    )


def _symmetric_reliability(
        first: RiskEstimate, second: RiskEstimate, label: str) -> Tuple[float, Dict[str, Any]]:
    # The weaker held-fold direction is the certificate.  A geometric mean can
    # partially hide one poor direction behind one excellent direction, whereas
    # the minimum has the intended conservative, no-worse-direction semantics.
    reliability = min(first.reliability, second.reliability)
    return reliability, {
        f"{label}_combination": "minimum_directional_positive_part_EB",
        f"{label}_first_direction_reliability": first.reliability,
        f"{label}_second_direction_reliability": second.reliability,
        f"{label}_combined_risk_reliability": reliability,
        **dict(first.receipts),
        **dict(second.receipts),
    }


def _direct_sum_maps(read_matrix: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
    read_matrix = _as_cpu_double(read_matrix, "read_matrix", 2)
    output_dim = read_matrix.shape[0]
    initial_map = _machine_pinv(read_matrix)
    output_projector = _symmetrize(read_matrix @ initial_map)
    identity = torch.eye(output_dim, dtype=torch.float64)
    complement_projector = _symmetrize(identity - output_projector)
    rank = _matrix_rank(read_matrix)
    codimension = output_dim - rank
    receipts = {
        "read_machine_rank": rank,
        "output_dimension": output_dim,
        "complement_codimension": codimension,
        "complement_present": codimension > 0,
        "complement_policy": "machine_rank_output_orthogonal_complement",
        "output_projector_idempotence_max_abs": float(
            (output_projector @ output_projector - output_projector).abs().max()),
        "complement_projector_idempotence_max_abs": float(
            (complement_projector @ complement_projector - complement_projector).abs().max()),
        "direct_sum_projector_sum_max_abs": float(
            (output_projector + complement_projector - identity).abs().max()),
        "complement_read_orthogonality_max_abs": float(
            (complement_projector @ read_matrix).abs().max()),
    }
    return initial_map, output_projector, complement_projector, receipts


def _initial_components(
        initial_output: Tensor, output_mean: Tensor, initial_map: Tensor,
        complement_projector: Tensor) -> Tuple[Tensor, Tensor]:
    centered = initial_output - output_mean
    state = centered @ initial_map.T
    complement = centered @ complement_projector.T
    return state, complement


def _read(
        state: Tensor, complement: Tensor, output_mean: Tensor, read_matrix: Tensor) -> Tensor:
    return output_mean + complement + state @ read_matrix.T


def _open_loop_episode_losses(
        clean: Tensor, actions: Tensor, state_matrix: Tensor, action_matrix: Tensor,
        read_matrix: Tensor, initial_map: Tensor, complement_projector: Tensor,
        output_mean: Tensor, action_mean: Tensor) -> Tensor:
    state, complement = _initial_components(
        clean[:, 0], output_mean, initial_map, complement_projector)
    losses = torch.zeros(clean.shape[0], dtype=torch.float64)
    for index in range(actions.shape[1]):
        state = (
            state @ state_matrix.T
            + (actions[:, index] - action_mean) @ action_matrix.T)
        prediction = _read(state, complement, output_mean, read_matrix)
        losses.add_((prediction - clean[:, index + 1]).square().mean(dim=-1))
    return losses / actions.shape[1]


def _reconstruct_states_from_all_futures(
        clean: Tensor, actions: Tensor, state_matrix: Tensor, action_matrix: Tensor,
        read_matrix: Tensor, output_mean: Tensor, action_mean: Tensor,
        complement_projector: Tensor) -> Tensor:
    """All-future clean state reconstruction in the deployed energy coordinate."""
    episodes, length, _ = clean.shape
    state_dim = state_matrix.shape[0]
    _, complements = _initial_components(
        clean[:, 0], output_mean, _machine_pinv(read_matrix), complement_projector)
    centered = clean - output_mean.view(1, 1, -1) - complements.unsqueeze(1)
    centered_actions = actions - action_mean.view(1, 1, -1)
    reconstructed = []
    for start in range(length):
        remaining = length - start
        power = torch.eye(state_dim, dtype=torch.float64)
        rows = []
        input_state = torch.zeros(episodes, state_dim, dtype=torch.float64)
        input_reads = []
        for offset in range(remaining):
            rows.append(read_matrix @ power)
            input_reads.append(input_state @ read_matrix.T)
            if offset < remaining - 1:
                input_state = (
                    input_state @ state_matrix.T
                    + centered_actions[:, start + offset] @ action_matrix.T)
                power = power @ state_matrix
        observability = torch.cat(rows, dim=0)
        target = centered[:, start:] - torch.stack(input_reads, dim=1)
        state = target.reshape(episodes, -1) @ _machine_pinv(observability).T
        reconstructed.append(state)
    return torch.stack(reconstructed, dim=1)


def _open_loop_sequences(
        clean: Tensor, observed: Tensor, actions: Tensor, state_matrix: Tensor,
        action_matrix: Tensor, read_matrix: Tensor, initial_map: Tensor,
        complement_projector: Tensor, output_mean: Tensor, action_mean: Tensor,
        reconstructed_states: Tensor) -> Tuple[Tensor, Tensor]:
    state, complement = _initial_components(
        clean[:, 0], output_mean, initial_map, complement_projector)
    errors, innovations = [], []
    for index in range(1, clean.shape[1]):
        state = (
            state @ state_matrix.T
            + (actions[:, index - 1] - action_mean) @ action_matrix.T)
        prior_read = _read(state, complement, output_mean, read_matrix)
        errors.append(reconstructed_states[:, index] - state)
        innovations.append(observed[:, index] - prior_read)
    return torch.stack(errors, dim=1), torch.stack(innovations, dim=1)


def _fit_correction(
        state_errors: Tensor, innovations: Tensor, *, energy_cap: bool,
        label: str, support_projector: Tensor | None = None) -> CorrectionFit:
    state_errors = _as_cpu_double(state_errors, f"{label}_state_errors", 3)
    innovations = _as_cpu_double(innovations, f"{label}_innovations", 3)
    if state_errors.shape[:2] != innovations.shape[:2]:
        raise ValueError("correction state errors and innovations must align")
    errors = state_errors.reshape(-1, state_errors.shape[-1])
    innovation = innovations.reshape(-1, innovations.shape[-1])
    covariance, covariance_receipts = _oas_covariance(innovation, f"{label}_innovation")
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    scale = max(1.0, float(eigenvalues.abs().max()))
    cutoff = covariance.shape[0] * torch.finfo(torch.float64).eps * scale
    keep = eigenvalues > cutoff
    rank = int(torch.count_nonzero(keep))
    if rank < 1:
        raise ValueError("innovation covariance has zero machine rank")
    whitener = (
        eigenvectors[:, keep] * eigenvalues[keep].rsqrt().unsqueeze(0)) @ eigenvectors[:, keep].T
    normalized = innovation @ whitener.T
    gram = normalized.T @ normalized
    raw = (errors.T @ normalized) @ _machine_pinv(gram, hermitian=True)
    if support_projector is not None:
        support_projector = _as_cpu_double(
            support_projector, f"{label}_support_projector", 2)
        if support_projector.shape != (raw.shape[0], raw.shape[0]):
            raise ValueError("correction support projector has the wrong state shape")
        raw = support_projector @ raw
    left, singular, right_h = torch.linalg.svd(raw, full_matrices=False)
    capped = singular.clamp_max(1.0)
    deployed = (left * (capped if energy_cap else singular).unsqueeze(0)) @ right_h
    whitening_identity = whitener @ covariance @ whitener.T
    support_projector = eigenvectors[:, keep] @ eigenvectors[:, keep].T
    fit_scores = normalized.square().sum(dim=-1)
    fit_gates = torch.minimum(
        torch.ones_like(fit_scores),
        fit_scores.new_full(fit_scores.shape, float(rank))
        / fit_scores.clamp_min(torch.finfo(torch.float64).tiny))
    receipts = {
        **covariance_receipts,
        f"{label}_correction_samples": int(errors.shape[0]),
        f"{label}_innovation_machine_rank": rank,
        f"{label}_innovation_rank_cutoff": cutoff,
        f"{label}_whitening_max_abs": float(
            (whitening_identity - support_projector).abs().max()),
        f"{label}_whitener_symmetry_max_abs": float((whitener - whitener.T).abs().max()),
        f"{label}_fit_innovation_score_mean": float(fit_scores.mean()),
        f"{label}_fit_innovation_score_max": float(fit_scores.max()),
        f"{label}_fit_radial_gate_mean": float(fit_gates.mean()),
        f"{label}_fit_radial_gate_min": float(fit_gates.min()),
        f"{label}_fit_radial_gate_max": float(fit_gates.max()),
        f"{label}_raw_correction_operator_norm": float(singular.max()),
        f"{label}_deployed_correction_operator_norm": float(
            capped.max() if energy_cap else singular.max()),
        f"{label}_energy_cap_active": energy_cap,
        f"{label}_energy_cap_projection_frobenius": float((deployed - raw).norm()),
        f"{label}_future_energy_bound": (
            "norm_delta_sq_le_innovation_rank" if energy_cap else "none_control"),
    }
    return CorrectionFit(
        raw_matrix=raw,
        deployed_matrix=deployed,
        innovation_covariance=covariance,
        innovation_whitener=whitener,
        innovation_rank=rank,
        receipts=receipts,
    )


def _recursive_correction_losses(
        clean: Tensor, observed: Tensor, actions: Tensor, state_matrix: Tensor,
        action_matrix: Tensor, read_matrix: Tensor, correction: CorrectionFit | None,
        initial_map: Tensor, complement_projector: Tensor, output_mean: Tensor,
        action_mean: Tensor, *, radial: bool) -> Tensor:
    state, complement = _initial_components(
        observed[:, 0], output_mean, initial_map, complement_projector)
    losses = torch.zeros(clean.shape[0], dtype=torch.float64)
    for index in range(1, clean.shape[1]):
        state = (
            state @ state_matrix.T
            + (actions[:, index - 1] - action_mean) @ action_matrix.T)
        prior_read = _read(state, complement, output_mean, read_matrix)
        losses.add_((prior_read - clean[:, index]).square().mean(dim=-1))
        if correction is not None:
            innovation = observed[:, index] - prior_read
            normalized = innovation @ correction.innovation_whitener.T
            score = normalized.square().sum(dim=-1)
            if radial:
                gate = torch.minimum(
                    torch.ones_like(score),
                    score.new_full(score.shape, float(correction.innovation_rank))
                    / score.clamp_min(torch.finfo(torch.float64).tiny))
            else:
                gate = torch.ones_like(score)
            state = state + (normalized * gate.unsqueeze(-1)) @ correction.deployed_matrix.T
    return losses / (clean.shape[1] - 1)


def fit_cf_ebo(clean: Tensor, observed: Tensor, actions: Tensor, *, mode: str = "full") -> CEBOFit:
    """Fit CF-EBO entirely from paired train embeddings and executed actions."""
    mode = _canonical_mode(mode)
    clean = _as_cpu_double(clean, "clean", 3)
    observed = _as_cpu_double(observed, "observed", 3)
    actions = _as_cpu_double(actions, "actions", 3)
    if clean.shape != observed.shape:
        raise ValueError("clean and observed paired views must have identical shapes")
    if tuple(actions.shape[:2]) != (clean.shape[0], clean.shape[1] - 1):
        raise ValueError("actions must align with every clean/observed transition")
    if clean.shape[0] < 4:
        raise ValueError("CF-EBO cross-fold risks require at least four episodes")

    markov = estimate_all_lag_markov_moments(clean, actions)
    coordinate = realize_fold_agreement_hankel(
        markov.even, markov.odd,
        agreement_mode="empirical_bayes", transition_mode="normal")
    energy = _observable_energy_coordinates(coordinate.state_matrix, coordinate.read_matrix)
    state_matrix = energy.state_matrix
    read_matrix = energy.read_matrix
    output_mean = clean.mean(dim=(0, 1))
    action_mean = actions.mean(dim=(0, 1))
    initial_map, output_projector, complement_projector, direct_receipts = \
        _direct_sum_maps(read_matrix)
    initial_map = energy.energy_support_projector @ initial_map

    # V14 deliberately preserves pooled V13 A,C and changes only the fixed-coordinate
    # B fit.  Fold-specific B maps therefore live in one common energy coordinate.
    action_even, even_b_receipts = _refit_action_from_all_lags(
        state_matrix, read_matrix, markov.even)
    action_odd, odd_b_receipts = _refit_action_from_all_lags(
        state_matrix, read_matrix, markov.odd)
    action_pooled, pooled_b_receipts = _refit_action_from_all_lags(
        state_matrix, read_matrix, markov.average)
    action_even = energy.energy_support_projector @ action_even
    action_odd = energy.energy_support_projector @ action_odd
    action_pooled = energy.energy_support_projector @ action_pooled
    folds = {
        "even": torch.arange(0, clean.shape[0], 2),
        "odd": torch.arange(1, clean.shape[0], 2),
    }
    zero_action = torch.zeros_like(action_pooled)
    action_risks = []
    for fit_name, score_name, fitted_action in (
            ("even", "odd", action_even), ("odd", "even", action_odd)):
        indices = folds[score_name]
        base = _open_loop_episode_losses(
            clean[indices], actions[indices], state_matrix, zero_action,
            read_matrix, initial_map, complement_projector, output_mean, action_mean)
        candidate = _open_loop_episode_losses(
            clean[indices], actions[indices], state_matrix, fitted_action,
            read_matrix, initial_map, complement_projector, output_mean, action_mean)
        action_risks.append(_directional_risk(
            base, candidate, f"action_{fit_name}_to_{score_name}"))
    computed_action_reliability, action_risk_receipts = _symmetric_reliability(
        action_risks[0], action_risks[1], "action")
    action_reliability = (
        0.0 if mode == "noaction" else
        1.0 if mode == "norisk" else computed_action_reliability)
    action_matrix = action_pooled * action_reliability

    reconstructed = _reconstruct_states_from_all_futures(
        clean, actions, state_matrix, action_matrix, read_matrix,
        output_mean, action_mean, complement_projector)
    state_errors, innovations = _open_loop_sequences(
        clean, observed, actions, state_matrix, action_matrix, read_matrix,
        initial_map, complement_projector, output_mean, action_mean, reconstructed)
    energy_cap = mode != "noenergycap"
    radial = mode != "noradial"
    fold_corrections = {
        name: _fit_correction(
            state_errors[indices], innovations[indices], energy_cap=energy_cap,
            label=f"correction_{name}",
            support_projector=energy.energy_support_projector)
        for name, indices in folds.items()
    }
    correction_risks = []
    for fit_name, score_name in (("even", "odd"), ("odd", "even")):
        indices = folds[score_name]
        base = _recursive_correction_losses(
            clean[indices], observed[indices], actions[indices], state_matrix,
            action_matrix, read_matrix, None, initial_map, complement_projector,
            output_mean, action_mean, radial=radial)
        candidate = _recursive_correction_losses(
            clean[indices], observed[indices], actions[indices], state_matrix,
            action_matrix, read_matrix, fold_corrections[fit_name], initial_map,
            complement_projector, output_mean, action_mean, radial=radial)
        correction_risks.append(_directional_risk(
            base, candidate, f"correction_{fit_name}_to_{score_name}"))
    computed_correction_reliability, correction_risk_receipts = _symmetric_reliability(
        correction_risks[0], correction_risks[1], "correction")
    correction_reliability = (
        0.0 if mode == "nocorrect" else
        1.0 if mode == "norisk" else computed_correction_reliability)
    correction = _fit_correction(
        state_errors, innovations, energy_cap=energy_cap, label="correction_pooled",
        support_projector=energy.energy_support_projector)

    centered_initial = clean[:, 0] - output_mean
    initial_state = centered_initial @ initial_map.T
    initial_complement = centered_initial @ complement_projector.T
    reconstruction_error = float((
        output_mean + initial_complement + initial_state @ read_matrix.T - clean[:, 0]
    ).abs().max())
    receipts: Dict[str, Any] = {
        "method": "CF-EBO-v14",
        "fit_mode": mode,
        "self_supervision": "paired_views_plus_executed_randomized_actions",
        "reward_or_state_labels_used": False,
        "validation_or_corruption_identity_used": False,
        "learned_memory_parameters": 0,
        "source_coordinate_realization": "V13_pooled_normal_A_C_only",
        "source_action_map_deployed": False,
        "source_riccati_gain_deployed": False,
        "action_refit": "fixed_energy_A_C_fold_specific_and_pooled_all_lags",
        "action_risk": "symmetric_recursive_held_fold_positive_part_EB",
        "correction_fit": "paired_state_error_on_OAS_whitened_innovation",
        "correction_risk": "symmetric_recursive_held_fold_positive_part_EB",
        "correction_energy_cap": energy_cap,
        "radial_gate": radial,
        "risk_shrinkage": mode != "norisk",
        "computed_action_reliability": computed_action_reliability,
        "deployed_action_reliability": action_reliability,
        "computed_correction_reliability": computed_correction_reliability,
        "deployed_correction_reliability": correction_reliability,
        "noaction_exact": mode == "noaction",
        "nocorrect_exact": mode == "nocorrect",
        "norisk_exact": mode == "norisk",
        "noenergycap_exact": mode == "noenergycap",
        "noradial_exact": mode == "noradial",
        "initial_direct_sum_reconstruction_max_abs": reconstruction_error,
        "unavoidable_structural_choices": [
            "V13 pooled normal all-lag coordinate realization",
            "infinite observability-Gramian energy coordinate",
            "two deterministic episode folds",
            "positive-part empirical-Bayes held-fold risk shrinkage",
            "OAS innovation whitening",
            "unit spectral correction cap",
            "innovation-rank radial boundary",
        ],
        **dict(markov.receipts),
        **{f"v13_coordinate_{key}": value
           for key, value in coordinate.receipts.items()},
        **dict(energy.receipts),
        **direct_receipts,
        **{f"action_even_{key}": value for key, value in even_b_receipts.items()},
        **{f"action_odd_{key}": value for key, value in odd_b_receipts.items()},
        **{f"action_pooled_{key}": value for key, value in pooled_b_receipts.items()},
        **action_risk_receipts,
        **dict(fold_corrections["even"].receipts),
        **dict(fold_corrections["odd"].receipts),
        **dict(correction.receipts),
        **correction_risk_receipts,
    }
    return CEBOFit(
        state_matrix=state_matrix,
        action_matrix=action_matrix,
        raw_action_matrix=action_pooled,
        read_matrix=read_matrix,
        correction_matrix=correction.deployed_matrix,
        raw_correction_matrix=correction.raw_matrix,
        innovation_covariance=correction.innovation_covariance,
        innovation_whitener=correction.innovation_whitener,
        initial_map=initial_map,
        output_projector=output_projector,
        complement_projector=complement_projector,
        energy_support_projector=energy.energy_support_projector,
        output_mean=output_mean,
        action_mean=action_mean,
        action_reliability=torch.tensor(action_reliability, dtype=torch.float64),
        correction_reliability=torch.tensor(correction_reliability, dtype=torch.float64),
        innovation_rank=torch.tensor(correction.innovation_rank, dtype=torch.long),
        markov_even=markov.even,
        markov_odd=markov.odd,
        receipts=receipts,
    )


class CrossFitEnergyBoundedObserverMemory(nn.Module):
    """Zero-parameter cross-fold-calibrated memory with a rank-aware direct sum."""

    MODES = CF_EBO_MODES

    def __init__(
            self, output_dim: int, action_dim: int, state_dim: int,
            mode: str = "full", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        for name, value in (
                ("output_dim", output_dim), ("action_dim", action_dim),
                ("state_dim", state_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not dtype.is_floating_point:
            raise ValueError("CF-EBO buffers require a floating-point dtype")
        self.output_dim = output_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.mode = _canonical_mode(mode)
        self.register_buffer("state_matrix", torch.eye(state_dim, dtype=dtype))
        self.register_buffer("action_matrix", torch.zeros(state_dim, action_dim, dtype=dtype))
        self.register_buffer("raw_action_matrix", torch.zeros(state_dim, action_dim, dtype=dtype))
        self.register_buffer("read_matrix", torch.zeros(output_dim, state_dim, dtype=dtype))
        self.register_buffer("correction_matrix", torch.zeros(state_dim, output_dim, dtype=dtype))
        self.register_buffer("raw_correction_matrix", torch.zeros(state_dim, output_dim, dtype=dtype))
        self.register_buffer("innovation_covariance", torch.eye(output_dim, dtype=dtype))
        self.register_buffer("innovation_whitener", torch.eye(output_dim, dtype=dtype))
        self.register_buffer("initial_map", torch.zeros(state_dim, output_dim, dtype=dtype))
        self.register_buffer("output_projector", torch.zeros(output_dim, output_dim, dtype=dtype))
        self.register_buffer("complement_projector", torch.eye(output_dim, dtype=dtype))
        self.register_buffer("energy_support_projector", torch.eye(state_dim, dtype=dtype))
        self.register_buffer("output_mean", torch.zeros(output_dim, dtype=dtype))
        self.register_buffer("action_mean", torch.zeros(action_dim, dtype=dtype))
        self.register_buffer("action_reliability", torch.zeros((), dtype=dtype))
        self.register_buffer("correction_reliability", torch.zeros((), dtype=dtype))
        self.register_buffer("innovation_rank", torch.zeros((), dtype=torch.long))
        self.register_buffer("fit_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("operators_installed", torch.zeros((), dtype=torch.bool))
        self._fit_receipts: Dict[str, Any] = {}

    @classmethod
    def from_fit(
            cls, fit: CEBOFit, mode: str = "full",
            dtype: torch.dtype | None = None) -> "CrossFitEnergyBoundedObserverMemory":
        if not isinstance(fit, CEBOFit):
            raise TypeError("fit must be a CEBOFit")
        memory = cls(
            fit.read_matrix.shape[0], fit.action_matrix.shape[1],
            fit.state_matrix.shape[0], mode=mode,
            dtype=dtype or fit.state_matrix.dtype)
        memory.install_fit(fit)
        return memory

    def get_extra_state(self) -> Dict[str, Any]:
        return {"fit_receipts": copy.deepcopy(self._fit_receipts)}

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or not isinstance(state.get("fit_receipts", {}), Mapping):
            raise RuntimeError("invalid CF-EBO extra state")
        self._fit_receipts = copy.deepcopy(dict(state.get("fit_receipts", {})))

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def _validate_fit_mode(self, fit: CEBOFit) -> None:
        actual = fit.receipts.get("fit_mode")
        if actual != self.mode:
            raise ValueError(f"{self.mode} memory requires fit_mode={self.mode!r}, got {actual!r}")

    @torch.no_grad()
    def install_fit(self, fit: CEBOFit) -> None:
        if not isinstance(fit, CEBOFit):
            raise TypeError("fit must be a CEBOFit")
        self._validate_fit_mode(fit)
        expected = {
            "state_matrix": (self.state_dim, self.state_dim),
            "action_matrix": (self.state_dim, self.action_dim),
            "raw_action_matrix": (self.state_dim, self.action_dim),
            "read_matrix": (self.output_dim, self.state_dim),
            "correction_matrix": (self.state_dim, self.output_dim),
            "raw_correction_matrix": (self.state_dim, self.output_dim),
            "innovation_covariance": (self.output_dim, self.output_dim),
            "innovation_whitener": (self.output_dim, self.output_dim),
            "initial_map": (self.state_dim, self.output_dim),
            "output_projector": (self.output_dim, self.output_dim),
            "complement_projector": (self.output_dim, self.output_dim),
            "energy_support_projector": (self.state_dim, self.state_dim),
            "output_mean": (self.output_dim,),
            "action_mean": (self.action_dim,),
            "action_reliability": (),
            "correction_reliability": (),
            "innovation_rank": (),
        }
        candidates: Dict[str, Tensor] = {}
        for name, shape in expected.items():
            value = getattr(fit, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if name == "innovation_rank":
                if value.dtype == torch.bool or value.is_floating_point():
                    raise ValueError("innovation_rank must have an integer dtype")
            elif not value.is_floating_point():
                raise ValueError(f"{name} must be floating point")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            candidates[name] = value.detach()

        covariance = _symmetrize(candidates["innovation_covariance"].double())
        minimum = float(torch.linalg.eigvalsh(covariance).min())
        tolerance = self.output_dim * torch.finfo(torch.float64).eps * max(
            1.0, float(covariance.abs().max()))
        if minimum <= tolerance:
            raise ValueError("innovation_covariance must be symmetric positive definite")
        if float((candidates["innovation_covariance"].double() - covariance).abs().max()) \
                > tolerance:
            raise ValueError("innovation_covariance must be symmetric positive definite")
        whitener = candidates["innovation_whitener"].double()
        if float((whitener - whitener.T).abs().max()) > 1e-9:
            raise ValueError("innovation_whitener must be symmetric")
        rank = int(candidates["innovation_rank"])
        if not 1 <= rank <= self.output_dim:
            raise ValueError("innovation_rank is outside the output dimension")
        alpha_b = float(candidates["action_reliability"])
        alpha_k = float(candidates["correction_reliability"])
        if not (0.0 <= alpha_b <= 1.0 and 0.0 <= alpha_k <= 1.0):
            raise ValueError("risk reliabilities must lie in [0,1]")
        if self.mode == "noaction" and torch.count_nonzero(candidates["action_matrix"]):
            raise ValueError("noaction fit must deploy an exact zero action matrix")
        if self.mode == "nocorrect" and alpha_k != 0.0:
            raise ValueError("nocorrect fit must deploy zero correction reliability")
        if self.mode == "norisk" and (alpha_b != 1.0 or alpha_k != 1.0):
            raise ValueError("norisk fit must deploy unit reliabilities")

        state = candidates["state_matrix"].double()
        read = candidates["read_matrix"].double()
        support = candidates["energy_support_projector"].double()
        energy_error = state.T @ state + read.T @ read - support
        if float(energy_error.abs().max()) > 2e-8:
            raise ValueError("state/read operators violate the CF-EBO energy identity")
        support_errors = (
            (support - support.T).abs().max(),
            (support @ support - support).abs().max(),
            ((torch.eye(self.state_dim, dtype=torch.float64) - support) @ state).abs().max(),
            (state @ (torch.eye(self.state_dim, dtype=torch.float64) - support)).abs().max(),
            (read @ (torch.eye(self.state_dim, dtype=torch.float64) - support)).abs().max(),
            ((torch.eye(self.state_dim, dtype=torch.float64) - support)
             @ candidates["action_matrix"].double()).abs().max(),
            ((torch.eye(self.state_dim, dtype=torch.float64) - support)
             @ candidates["raw_action_matrix"].double()).abs().max(),
            ((torch.eye(self.state_dim, dtype=torch.float64) - support)
             @ candidates["correction_matrix"].double()).abs().max(),
            ((torch.eye(self.state_dim, dtype=torch.float64) - support)
             @ candidates["raw_correction_matrix"].double()).abs().max(),
            ((torch.eye(self.state_dim, dtype=torch.float64) - support)
             @ candidates["initial_map"].double()).abs().max(),
        )
        if max(float(value) for value in support_errors) > 2e-8:
            raise ValueError("energy support projector or inactive padding is inconsistent")
        radius = float(torch.linalg.eigvals(state).abs().max())
        if radius >= 1.0:
            raise ValueError("state_matrix must be strictly stable")
        if self.mode != "noenergycap" and float(torch.linalg.matrix_norm(
                candidates["correction_matrix"].double(), ord=2)) > 1.0 + 1e-9:
            raise ValueError("correction_matrix violates the unit energy cap")
        output_projector = candidates["output_projector"].double()
        complement = candidates["complement_projector"].double()
        output_identity = torch.eye(self.output_dim, dtype=torch.float64)
        projector_errors = (
            (output_projector - output_projector.T).abs().max(),
            (output_projector @ output_projector - output_projector).abs().max(),
            (complement - complement.T).abs().max(),
            (complement @ complement - complement).abs().max(),
            (output_projector + complement - output_identity).abs().max(),
            (complement @ read).abs().max(),
        )
        if max(float(value) for value in projector_errors) > 2e-8:
            raise ValueError("direct-sum projectors are inconsistent")
        expected_output = read @ candidates["initial_map"].double()
        if float((expected_output - output_projector).abs().max()) > 2e-8:
            raise ValueError("initial_map does not induce output_projector")

        # Validation is complete before the first mutation.
        for name, value in candidates.items():
            destination = getattr(self, name)
            destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
        self.fit_updates.add_(1)
        self.operators_installed.fill_(True)
        self._fit_receipts = copy.deepcopy(dict(fit.receipts))

    def _validate_state(self, state: HIROState) -> int:
        if not isinstance(state, HIROState):
            raise ValueError("state must be an HIROState(mean,complement)")
        if state.mean.dim() != 2 or tuple(state.mean.shape[1:]) != (self.state_dim,):
            raise ValueError(f"state mean must have shape (B,{self.state_dim})")
        if tuple(state.complement.shape) != (state.mean.shape[0], self.output_dim):
            raise ValueError(f"state complement must have shape (B,{self.output_dim})")
        if (not state.mean.is_floating_point() or not state.complement.is_floating_point()
                or not torch.isfinite(state.mean).all()
                or not torch.isfinite(state.complement).all()):
            raise ValueError("state must contain finite floating-point values")
        return state.mean.shape[0]

    def initial_state(self, observation: Tensor) -> HIROState:
        if (not isinstance(observation, torch.Tensor) or observation.dim() != 2
                or observation.shape[0] < 1 or observation.shape[1] != self.output_dim
                or not observation.is_floating_point() or not torch.isfinite(observation).all()):
            raise ValueError(f"observation must be finite with shape (B,{self.output_dim})")
        centered = observation - self.output_mean.to(observation)
        mean = functional.linear(centered, self.initial_map.to(centered))
        complement = functional.linear(centered, self.complement_projector.to(centered))
        return HIROState(mean, complement)

    def read_state(self, state: HIROState) -> Tensor:
        self._validate_state(state)
        return (self.output_mean.to(state.mean) + state.complement.to(state.mean)
                + functional.linear(state.mean, self.read_matrix.to(state.mean)))

    def transition(self, state: HIROState, action: Tensor, return_details: bool = False):
        batch = self._validate_state(state)
        if (not isinstance(action, torch.Tensor) or tuple(action.shape) != (batch, self.action_dim)
                or not action.is_floating_point() or not torch.isfinite(action).all()):
            raise ValueError(f"action must be finite with shape ({batch},{self.action_dim})")
        action = action.to(state.mean)
        effective_action = action - self.action_mean.to(action)
        action_effect = functional.linear(effective_action, self.action_matrix.to(action))
        prior = HIROState(
            functional.linear(state.mean, self.state_matrix.to(state.mean)) + action_effect,
            state.complement)
        if not return_details:
            return prior
        return prior, {
            "effective_action": effective_action,
            "action_effect": action_effect,
            "prior_read": self.read_state(prior),
            "complement_anchor": prior.complement,
        }

    def correct(self, prior: HIROState, observation: Tensor, return_details: bool = False):
        batch = self._validate_state(prior)
        if (not isinstance(observation, torch.Tensor)
                or tuple(observation.shape) != (batch, self.output_dim)
                or not observation.is_floating_point() or not torch.isfinite(observation).all()):
            raise ValueError(f"observation must be finite with shape ({batch},{self.output_dim})")
        observation = observation.to(prior.mean)
        # Do whitening and radial arithmetic outside AMP.  Scaling the innovation
        # *before* the matrix multiply prevents a finite BF16/FP32 outlier from
        # overflowing whitening; the final gated normalized vector is bounded by
        # sqrt(innovation_rank) in every energy-capped radial mode.
        prior_read = self.read_state(prior)
        with torch.autocast(device_type=prior.mean.device.type, enabled=False):
            arithmetic_dtype = (
                torch.float64 if prior.mean.dtype == torch.float64 else torch.float32)
            innovation = (
                observation.to(dtype=arithmetic_dtype)
                - prior_read.to(dtype=arithmetic_dtype))
            # Opposite-sign finite BF16 endpoints can exceed the FP32 range after
            # subtraction.  The fallback is a vector subtraction only; dense
            # whitening remains FP32 after source-dtype normalization.
            if not torch.isfinite(innovation).all():
                innovation = observation.double() - prior_read.double()
            # Compute the scale in the source dtype before converting the unit
            # direction to FP32.  This also handles finite FP64 values above the
            # FP32 range without an inf/inf intermediate.
            input_scale_source = innovation.abs().amax(
                dim=-1, keepdim=True).clamp_min(1.0)
            normalized_unit = functional.linear(
                (innovation / input_scale_source).float(),
                self.innovation_whitener.float())
            input_scale = input_scale_source.double()
            score = (input_scale.squeeze(-1).square()
                     * normalized_unit.double().square().sum(dim=-1))
        if self.mode == "noradial":
            radial_gate = torch.ones_like(score)
        else:
            radial_gate = torch.minimum(
                torch.ones_like(score), self.innovation_rank.to(score)
                / score.clamp_min(torch.finfo(torch.float64).tiny))
        normalized = normalized_unit.double() * input_scale.double()
        gated_normalized = (
            normalized_unit.double()
            * (input_scale.double() * radial_gate.unsqueeze(-1)))
        if self.mode == "nocorrect":
            correction = torch.zeros_like(prior.mean)
            effective_gain = prior.mean.new_zeros(batch, self.state_dim, self.output_dim)
        else:
            correction = functional.linear(
                gated_normalized.float(), self.correction_matrix.float())
            correction = (
                correction * self.correction_reliability.float()).to(prior.mean)
            base_gain = self.correction_matrix.float() @ self.innovation_whitener.float()
            effective_gain = (
                self.correction_reliability.float()
                * radial_gate.float().view(batch, 1, 1) * base_gain.unsqueeze(0)).to(prior.mean)
        posterior = HIROState(prior.mean + correction, prior.complement)
        if not return_details:
            return posterior
        return posterior, {
            "innovation": innovation,
            "normalized_innovation": normalized,
            "innovation_score": score,
            "radial_gate": radial_gate,
            "gain": effective_gain,
            "correction": correction,
            "posterior_read": self.read_state(posterior),
            "complement_anchor": posterior.complement,
        }

    def step(
            self, state: HIROState, observation: Tensor, action: Tensor,
            return_details: bool = False):
        prior, transition_details = self.transition(state, action, return_details=True)
        posterior, correction_details = self.correct(prior, observation, return_details=True)
        read = correction_details["posterior_read"]
        if not return_details:
            return read, posterior
        return read, posterior, {
            **transition_details, **correction_details,
            "prior_state": prior.mean,
            "posterior_state": posterior.mean,
            "prior_read": transition_details["prior_read"],
        }

    def rollout_transition(self, state: HIROState, actions: Tensor) -> HIROState:
        batch = self._validate_state(state)
        if (not isinstance(actions, torch.Tensor) or actions.dim() != 3
                or tuple(actions.shape[::2]) != (batch, self.action_dim)
                or not actions.is_floating_point() or not torch.isfinite(actions).all()):
            raise ValueError(f"actions must have shape (B,H,{self.action_dim})")
        means, complements = [], []
        for index in range(actions.shape[1]):
            state = self.transition(state, actions[:, index])
            means.append(state.mean)
            complements.append(state.complement)
        if not means:
            return HIROState(
                state.mean.new_empty(batch, 0, self.state_dim),
                state.complement.new_empty(batch, 0, self.output_dim))
        return HIROState(torch.stack(means, dim=1), torch.stack(complements, dim=1))

    def forward(self, observations: Tensor, actions: Tensor, return_details: bool = False):
        if (not isinstance(observations, torch.Tensor) or observations.dim() != 3
                or observations.shape[0] < 1 or observations.shape[1] < 1
                or observations.shape[2] != self.output_dim
                or not observations.is_floating_point() or not torch.isfinite(observations).all()):
            raise ValueError(f"observations must have shape (B,T,{self.output_dim})")
        expected_actions = (observations.shape[0], observations.shape[1] - 1, self.action_dim)
        if (not isinstance(actions, torch.Tensor) or tuple(actions.shape) != expected_actions
                or not actions.is_floating_point() or not torch.isfinite(actions).all()):
            raise ValueError(f"actions must have shape {expected_actions}")
        actions = actions.to(observations)
        state = self.initial_state(observations[:, 0])
        initial_read = self.read_state(state)
        posterior_reads = [initial_read]
        prior_reads = [initial_read]
        means = [state.mean]
        complements = [state.complement]
        innovations = [torch.zeros_like(observations[:, 0])]
        normalized = [torch.zeros_like(observations[:, 0])]
        scores = [observations.new_zeros(observations.shape[0])]
        radial_gates = [observations.new_ones(observations.shape[0])]
        corrections = [observations.new_zeros(observations.shape[0], self.state_dim)]
        action_effects = [observations.new_zeros(observations.shape[0], self.state_dim)]
        gains = [observations.new_zeros(
            observations.shape[0], self.state_dim, self.output_dim)]
        for index in range(1, observations.shape[1]):
            read, state, details = self.step(
                state, observations[:, index], actions[:, index - 1], return_details=True)
            posterior_reads.append(read)
            prior_reads.append(details["prior_read"])
            means.append(state.mean)
            complements.append(state.complement)
            innovations.append(details["innovation"])
            normalized.append(details["normalized_innovation"])
            scores.append(details["innovation_score"])
            radial_gates.append(details["radial_gate"])
            corrections.append(details["correction"])
            action_effects.append(details["action_effect"])
            gains.append(details["gain"])
        reads = torch.stack(posterior_reads, dim=1)
        if not return_details:
            return reads
        return reads, {
            "reads": reads,
            "posterior_reads": reads,
            "prior_reads": torch.stack(prior_reads, dim=1),
            "state_means": torch.stack(means, dim=1),
            "complement_anchors": torch.stack(complements, dim=1),
            "innovations": torch.stack(innovations, dim=1),
            "normalized_innovations": torch.stack(normalized, dim=1),
            "innovation_scores": torch.stack(scores, dim=1),
            "radial_gates": torch.stack(radial_gates, dim=1),
            "corrections": torch.stack(corrections, dim=1),
            "action_effects": torch.stack(action_effects, dim=1),
            "gains": torch.stack(gains, dim=1),
        }

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, Any]:
        state = self.state_matrix.double()
        read = self.read_matrix.double()
        support = self.energy_support_projector.double()
        energy_error = state.T @ state + read.T @ read - support
        support_rank = _matrix_rank(support.cpu())
        inactive = torch.eye(
            self.state_dim, dtype=torch.float64, device=state.device) - support
        complement_rank = _matrix_rank(self.complement_projector.double().cpu())
        # Fit receipts are useful telemetry, but deployed operator state is the
        # authority for core diagnostics. Starting from receipts and then
        # installing live values prevents an old or malformed receipt from
        # shadowing an ablation's exact deployed state.
        diagnostics = copy.deepcopy(self._fit_receipts)
        diagnostics.update({
            "mode": self.mode,
            "operators_installed": bool(self.operators_installed),
            "fit_updates": int(self.fit_updates),
            "gradient_parameter_count": self.parameter_count(),
            "state_dim": self.state_dim,
            "streaming_mean_floats": self.state_dim,
            "streaming_complement_floats": self.output_dim,
            "streaming_covariance_floats": 0,
            "state_spectral_radius": float(torch.linalg.eigvals(state).abs().max()),
            "state_operator_norm": float(torch.linalg.matrix_norm(state, ord=2)),
            "energy_identity_max_abs": float(energy_error.abs().max()),
            "energy_identity_operator_norm": float(torch.linalg.matrix_norm(
                energy_error, ord=2)),
            "energy_support_rank": support_rank,
            "energy_inactive_padding": self.state_dim - support_rank,
            "energy_support_projector_idempotence_max_abs": float(
                (support @ support - support).abs().max()),
            "energy_support_projector_symmetry_max_abs": float(
                (support - support.T).abs().max()),
            "energy_support_state_left_max_abs": float((inactive @ state).abs().max()),
            "energy_support_state_right_max_abs": float((state @ inactive).abs().max()),
            "energy_support_read_max_abs": float((read @ inactive).abs().max()),
            "energy_support_action_max_abs": float(
                (inactive @ self.action_matrix.double()).abs().max()),
            "energy_support_raw_action_max_abs": float(
                (inactive @ self.raw_action_matrix.double()).abs().max()),
            "energy_support_correction_max_abs": float(
                (inactive @ self.correction_matrix.double()).abs().max()),
            "energy_support_raw_correction_max_abs": float(
                (inactive @ self.raw_correction_matrix.double()).abs().max()),
            "energy_support_initial_map_max_abs": float(
                (inactive @ self.initial_map.double()).abs().max()),
            "raw_action_matrix_norm": float(self.raw_action_matrix.double().norm()),
            "effective_action_matrix_norm": float(self.action_matrix.double().norm()),
            "raw_correction_operator_norm": float(torch.linalg.matrix_norm(
                self.raw_correction_matrix.double(), ord=2)),
            "deployed_correction_operator_norm": float(torch.linalg.matrix_norm(
                self.correction_matrix.double(), ord=2)),
            "action_reliability": float(self.action_reliability),
            "correction_reliability": float(self.correction_reliability),
            "innovation_rank": int(self.innovation_rank),
            "radial_gate_active": self.mode != "noradial",
            "energy_cap_active": self.mode != "noenergycap",
            "complement_codimension": complement_rank,
            "complement_present": complement_rank > 0,
            "noaction_exact": self.mode == "noaction",
            "nocorrect_exact": self.mode == "nocorrect",
            "norisk_exact": self.mode == "norisk",
            "noenergycap_exact": self.mode == "noenergycap",
            "noradial_exact": self.mode == "noradial",
        })
        return diagnostics


CF_EBOv14Memory = CrossFitEnergyBoundedObserverMemory
