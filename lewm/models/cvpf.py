"""Cross-view predictive filtration (CVPF-v15).

CVPF stores coefficients of three *future-output* predictions rather than a
reconstructed latent dynamical state.  The initial observation, executed
actions, and later observation innovations each receive a fixed ``D``-column
schema.  Action and observation columns are fitted from all valid, zero-padded
future suffixes, calibrated on the opposite episode fold, and shifted with an
explicit least-squares projection of the raw suffix shift.

The fitting path is deliberately CPU/FP64 and parameter free.  It uses OAS for
source covariance estimation and only forms ``(H*D) x source_dim`` cross
products; a dense future ``(H*D) x (H*D)`` covariance is never materialized.
This is a regularized linear PLS approximation to a predictive filtration, not
an implementation of arbitrary conditional expectations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional

from lewm.models.cf_hiro import _as_cpu_double, _oas_covariance, _symmetrize


Tensor = torch.Tensor


CVPF_MODES = frozenset({
    "full", "nocorrect", "noaction", "norisk", "norho", "anchoronly",
})


def _canonical_mode(mode: str) -> str:
    if mode not in CVPF_MODES:
        raise ValueError(f"unknown CVPF mode {mode!r}")
    return mode


class CVPFState(NamedTuple):
    """The fixed ``3D`` streaming state."""

    anchor: Tensor
    action: Tensor
    observation: Tensor


@dataclass(frozen=True)
class _EventModes:
    source_mean: Tensor
    encoder: Tensor
    base_decoder: Tensor
    raw_decoder: Tensor
    rho: Tensor
    strength: Tensor
    receipts: Mapping[str, Any]


@dataclass(frozen=True)
class CVPFFit:
    """Complete detached fit consumed by :class:`CVPFv15Memory`."""

    output_mean: Tensor
    anchor_source_mean: Tensor
    action_source_mean: Tensor
    observation_source_mean: Tensor
    anchor_encoder: Tensor
    action_encoder: Tensor
    observation_encoder: Tensor
    anchor_decoder: Tensor
    action_decoder: Tensor
    observation_decoder: Tensor
    anchor_shift: Tensor
    action_shift: Tensor
    observation_shift: Tensor
    action_rho: Tensor
    observation_rho: Tensor
    action_weight: Tensor
    observation_weight: Tensor
    action_gain: Tensor
    observation_gain: Tensor
    receipts: Mapping[str, Any]


def _inverse_root(covariance: Tensor) -> Tensor:
    covariance = _symmetrize(_as_cpu_double(covariance, "covariance", 2))
    values, vectors = torch.linalg.eigh(covariance)
    if float(values.min()) <= 0.0:
        raise ValueError("OAS covariance must be positive definite")
    return (vectors * values.rsqrt().unsqueeze(0)) @ vectors.T


def _right_solve(covariance: Tensor, cross: Tensor) -> Tensor:
    """Return ``cross @ covariance^-1`` without constructing an inverse."""
    return torch.linalg.solve(covariance, cross.T).T


def _anchor_fit(clean: Tensor, output_mean: Tensor) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
    episodes, length, dimension = clean.shape
    horizon = length - 1
    source = clean[:, 0]
    source_mean = source.mean(dim=0)
    covariance, covariance_receipts = _oas_covariance(source, "anchor_source")
    whitener = _inverse_root(covariance)
    coefficient = (source - source_mean) @ whitener.T
    coefficient_covariance, coefficient_receipts = _oas_covariance(
        coefficient, "anchor_coefficient")
    target = (clean[:, 1:] - output_mean).reshape(episodes, horizon * dimension)
    cross = target.T @ coefficient / episodes
    decoder = _right_solve(coefficient_covariance, cross).reshape(
        horizon, dimension, dimension)
    receipts: Dict[str, Any] = {
        **covariance_receipts,
        **coefficient_receipts,
        "anchor_future_samples": episodes,
        "anchor_future_positions": horizon,
        "anchor_decoder_frobenius": float(decoder.norm()),
    }
    return source_mean, whitener, decoder, receipts


def _event_cross(source_coefficients: Tensor, residual: Tensor, horizon: int) -> Tensor:
    """Cross-product with all zero-padded suffixes without allocating them."""
    episodes, events, source_dim = source_coefficients.shape
    if events != horizon or residual.shape[:2] != (episodes, horizon + 1):
        raise ValueError("event sources/residual trajectory do not match H and H+1")
    output_dim = residual.shape[-1]
    denominator = episodes * events
    blocks = []
    for lag in range(horizon):
        valid = events - lag
        x = source_coefficients[:, :valid].reshape(-1, source_dim)
        y = residual[:, 1 + lag:1 + lag + valid].reshape(-1, output_dim)
        blocks.append(y.T @ x / denominator)
    return torch.stack(blocks, dim=0)


def _event_projection(residual: Tensor, decoder: Tensor) -> Tensor:
    """Project every zero-padded suffix on decoder columns.

    Returns ``(episodes, events, modes)``.  Event ``s`` starts at residual
    position ``s+1``; later decoder blocks that cross the episode boundary are
    exactly zero by omission.
    """
    horizon, output_dim, modes = decoder.shape
    episodes = residual.shape[0]
    if residual.shape[1:] != (horizon + 1, output_dim):
        raise ValueError("residual and decoder suffix shapes disagree")
    result = residual.new_zeros(episodes, horizon, modes)
    for lag in range(horizon):
        valid = horizon - lag
        target = residual[:, 1 + lag:1 + lag + valid]
        result[:, :valid].add_(target @ decoder[lag])
    return result


def _fit_event_modes(source: Tensor, residual: Tensor, label: str) -> _EventModes:
    """Fit low-dimensional PLS modes against a zero-padded future block."""
    source = _as_cpu_double(source, f"{label}_source", 3)
    residual = _as_cpu_double(residual, f"{label}_residual", 3)
    episodes, horizon, source_dim = source.shape
    output_dim = residual.shape[-1]
    if residual.shape[:2] != (episodes, horizon + 1):
        raise ValueError(f"{label} residual must have one more time position than sources")
    flat_source = source.reshape(-1, source_dim)
    source_mean = flat_source.mean(dim=0)
    covariance, covariance_receipts = _oas_covariance(flat_source, f"{label}_source")
    whitener = _inverse_root(covariance)
    whitened = (source - source_mean) @ whitener.T
    coefficient_covariance, coefficient_receipts = _oas_covariance(
        whitened.reshape(-1, source_dim), f"{label}_coefficient")
    cross = _event_cross(whitened, residual, horizon)
    regression = _right_solve(
        coefficient_covariance, cross.reshape(horizon * output_dim, source_dim))
    gram = _symmetrize(regression.T @ regression)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    strength = eigenvalues[order].clamp_min(0.0).sqrt()
    source_modes = eigenvectors[:, order]
    raw = (regression @ source_modes).reshape(horizon, output_dim, source_dim)

    raw_norm = raw.square().sum(dim=(0, 1)).sqrt()
    unit = raw / raw_norm.clamp_min(torch.finfo(torch.float64).tiny).view(1, 1, -1)
    projected_target = _event_projection(residual, unit)
    target_scale = projected_target.square().mean(dim=(0, 1)).sqrt()
    rho = (raw_norm / target_scale.clamp_min(
        torch.finfo(torch.float64).tiny)).clamp(0.0, 1.0)
    nonzero = raw_norm > 0
    rho = torch.where(nonzero, rho, torch.zeros_like(rho))
    base = torch.where(
        rho.view(1, 1, -1) > 0,
        raw / rho.clamp_min(torch.finfo(torch.float64).tiny).view(1, 1, -1),
        torch.zeros_like(raw))
    encoder = source_modes.T @ whitener
    receipts: Dict[str, Any] = {
        **covariance_receipts,
        **coefficient_receipts,
        f"{label}_episodes": episodes,
        f"{label}_events_per_episode": horizon,
        f"{label}_future_positions": horizon,
        f"{label}_source_dimension": source_dim,
        f"{label}_rho_min": float(rho.min()),
        f"{label}_rho_mean": float(rho.mean()),
        f"{label}_rho_max": float(rho.max()),
        f"{label}_raw_decoder_frobenius": float(raw.norm()),
        f"{label}_future_covariance_materialized": False,
        f"{label}_cross_shape": [horizon * output_dim, source_dim],
    }
    return _EventModes(
        source_mean=source_mean,
        encoder=encoder,
        base_decoder=base,
        raw_decoder=raw,
        rho=rho,
        strength=strength,
        receipts=receipts,
    )


def _episode_mode_weights(source: Tensor, residual: Tensor, modes: _EventModes) -> Tensor:
    coefficients = (source - modes.source_mean) @ modes.encoder.T
    target_dot = _event_projection(residual, modes.raw_decoder)
    decoder_norm_square = modes.raw_decoder.square().sum(dim=(0, 1))
    improvement = (
        2.0 * coefficients * target_dot
        - coefficients.square() * decoder_norm_square.view(1, 1, -1))
    episode = improvement.mean(dim=1)
    mean = episode.mean(dim=0)
    variance_of_mean = episode.var(dim=0, unbiased=True) / episode.shape[0]
    scale = torch.maximum(mean.square(), variance_of_mean).clamp_min(
        torch.finfo(torch.float64).tiny)
    floor = torch.finfo(torch.float64).eps * scale
    weight = torch.where(
        mean > 0,
        (mean.square() - variance_of_mean).clamp_min(0.0)
        / (mean.square() + floor),
        torch.zeros_like(mean))
    return weight.clamp(0.0, 1.0)


def _alignment(fold: _EventModes, pooled: _EventModes) -> Tuple[Tensor, Dict[str, Any]]:
    """Return pooled-index -> fold-index assignment by decoded-direction overlap."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - scipy is declared by the project
        raise RuntimeError("CVPF fold alignment requires scipy") from error
    fold_decoder = fold.raw_decoder.reshape(-1, fold.raw_decoder.shape[-1])
    pooled_decoder = pooled.raw_decoder.reshape(-1, pooled.raw_decoder.shape[-1])
    fold_norm = fold_decoder.norm(dim=0)
    pooled_norm = pooled_decoder.norm(dim=0)
    similarity = (fold_decoder.T @ pooled_decoder).abs()
    similarity = similarity / (
        fold_norm[:, None] * pooled_norm[None, :]).clamp_min(
            torch.finfo(torch.float64).tiny)
    rows, columns = linear_sum_assignment(-similarity.numpy())
    assignment = torch.full(
        (pooled_decoder.shape[1],), -1, dtype=torch.long)
    assignment[torch.as_tensor(columns)] = torch.as_tensor(rows)
    matched = similarity[torch.as_tensor(rows), torch.as_tensor(columns)]
    receipts = {
        "alignment_rule": "maximum_absolute_decoded_direction_assignment",
        "alignment_matches": int(len(rows)),
        "alignment_similarity_min": float(matched.min()) if len(rows) else 0.0,
        "alignment_similarity_mean": float(matched.mean()) if len(rows) else 0.0,
        "alignment_similarity_max": float(matched.max()) if len(rows) else 0.0,
        "alignment_sign_invariant": True,
        "alignment_permutation_invariant": True,
    }
    return assignment, receipts


def _degenerate_block_minimum(weight: Tensor, strength: Tensor) -> Tuple[Tensor, int]:
    """Use one conservative weight inside machine-indistinguishable blocks."""
    result = weight.clone()
    if not len(result):
        return result, 0
    scale = max(1.0, float(strength.abs().max()))
    tolerance = len(strength) * torch.finfo(torch.float64).eps * scale
    blocks = 0
    start = 0
    for index in range(1, len(strength) + 1):
        boundary = index == len(strength) or float(
            (strength[index - 1] - strength[index]).abs()) > tolerance
        if boundary:
            if index - start > 1:
                result[start:index] = result[start:index].min()
                blocks += 1
            start = index
    return result, blocks


def _crossfit_weights(
        source: Tensor, residual: Tensor, pooled: _EventModes,
        label: str) -> Tuple[Tensor, Dict[str, Any]]:
    episodes = source.shape[0]
    fold_indices = {
        "even": torch.arange(0, episodes, 2),
        "odd": torch.arange(1, episodes, 2),
    }
    directional: Dict[str, Tensor] = {}
    receipts: Dict[str, Any] = {}
    for fit_name, score_name in (("even", "odd"), ("odd", "even")):
        fit_indices = fold_indices[fit_name]
        score_indices = fold_indices[score_name]
        fitted = _fit_event_modes(
            source[fit_indices], residual[fit_indices], f"{label}_{fit_name}")
        fold_weight = _episode_mode_weights(
            source[score_indices], residual[score_indices], fitted)
        assignment, alignment_receipts = _alignment(fitted, pooled)
        transferred = torch.zeros_like(pooled.rho)
        valid = assignment >= 0
        transferred[valid] = fold_weight[assignment[valid]]
        directional[f"{fit_name}_to_{score_name}"] = transferred
        receipts.update({
            f"{label}_{fit_name}_to_{score_name}_{key}": value
            for key, value in alignment_receipts.items()})
        receipts[f"{label}_{fit_name}_to_{score_name}_weight_min"] = float(
            transferred.min())
        receipts[f"{label}_{fit_name}_to_{score_name}_weight_mean"] = float(
            transferred.mean())
        receipts[f"{label}_{fit_name}_to_{score_name}_weight_max"] = float(
            transferred.max())
    combined = torch.minimum(
        directional["even_to_odd"], directional["odd_to_even"])
    combined, degenerate_blocks = _degenerate_block_minimum(
        combined, pooled.strength)
    receipts.update({
        f"{label}_crossfit_combination": "minimum_two_directions",
        f"{label}_crossfit_degenerate_blocks": degenerate_blocks,
        f"{label}_crossfit_weight_min": float(combined.min()),
        f"{label}_crossfit_weight_mean": float(combined.mean()),
        f"{label}_crossfit_weight_max": float(combined.max()),
    })
    return combined, receipts


def _pad_modes(modes: _EventModes, dimension: int) -> _EventModes:
    source_dim = modes.encoder.shape[0]
    if source_dim > dimension:
        raise ValueError("source mode count exceeds fixed D schema")
    horizon, output_dim, _ = modes.base_decoder.shape
    encoder = torch.zeros(dimension, modes.encoder.shape[1], dtype=torch.float64)
    base = torch.zeros(horizon, output_dim, dimension, dtype=torch.float64)
    raw = torch.zeros_like(base)
    rho = torch.zeros(dimension, dtype=torch.float64)
    strength = torch.zeros(dimension, dtype=torch.float64)
    encoder[:source_dim] = modes.encoder
    base[:, :, :source_dim] = modes.base_decoder
    raw[:, :, :source_dim] = modes.raw_decoder
    rho[:source_dim] = modes.rho
    strength[:source_dim] = modes.strength
    return _EventModes(
        source_mean=modes.source_mean,
        encoder=encoder,
        base_decoder=base,
        raw_decoder=raw,
        rho=rho,
        strength=strength,
        receipts=dict(modes.receipts) | {
            "fixed_schema_columns": dimension,
            "source_mode_columns": source_dim,
            "exact_zero_padding_columns": dimension - source_dim,
        },
    )


def _projected_shift(decoder: Tensor, label: str) -> Tuple[Tensor, Dict[str, Any]]:
    decoder = _as_cpu_double(decoder, f"{label}_decoder", 3)
    horizon, output_dim, modes = decoder.shape
    shift = torch.zeros(modes, modes, dtype=torch.float64)
    if horizon < 2 or float(decoder.norm()) == 0.0:
        return shift, {
            f"{label}_shift_active_modes": 0,
            f"{label}_shift_closure_relative": 0.0,
            f"{label}_shift_operator_norm": 0.0,
            f"{label}_shift_spectral_radius": 0.0,
        }
    column_norm = decoder.norm(dim=(0, 1))
    active = torch.nonzero(column_norm > 0, as_tuple=False).flatten()
    if len(active):
        left = decoder[:-1, :, active].reshape((horizon - 1) * output_dim, -1)
        right = decoder[1:, :, active].reshape((horizon - 1) * output_dim, -1)
        raw_solution = torch.linalg.lstsq(left, right, driver="gelsd").solution
        # A compressed suffix subspace need not be shift invariant.  Near-null
        # decoder directions can therefore make the minimum-residual solution
        # arbitrarily large even while its decoded residual is small.  The
        # unit spectral projection is structural (not task tuned) and makes the
        # deployed coefficient recurrence non-expansive.  If that projection
        # were ever worse than the zero shift, falling back to zero preserves
        # the registered no-worse-than-zero integrity bound.
        left_vectors, singular, right_vectors = torch.linalg.svd(
            raw_solution, full_matrices=False)
        solution = (
            left_vectors * singular.clamp_max(1.0).unsqueeze(0)) @ right_vectors
        deployed_residual = (left @ solution - right).norm()
        zero_residual = right.norm()
        if float(deployed_residual) > float(zero_residual):
            solution.zero_()
            deployed_residual = zero_residual
        shift[active[:, None], active[None, :]] = solution
        denominator = right.norm()
        closure = float(deployed_residual / denominator) if float(denominator) else 0.0
        operator_norm = float(torch.linalg.matrix_norm(solution, ord=2))
        radius = float(torch.linalg.eigvals(solution).abs().max())
        raw_operator_norm = float(torch.linalg.matrix_norm(raw_solution, ord=2))
        projection_norm = float((solution - raw_solution).norm())
    else:
        closure = operator_norm = radius = 0.0
        raw_operator_norm = projection_norm = 0.0
    receipts = {
        f"{label}_shift_rule": "post_support_minimum_residual_projected_suffix_shift",
        f"{label}_shift_active_modes": int(len(active)),
        f"{label}_shift_closure_relative": closure,
        f"{label}_shift_zero_predictor_bound": 1.0,
        f"{label}_shift_raw_operator_norm": raw_operator_norm,
        f"{label}_shift_operator_norm": operator_norm,
        f"{label}_shift_spectral_radius": radius,
        f"{label}_shift_unit_spectral_projection_frobenius": projection_norm,
        f"{label}_shift_nonexpansive": operator_norm <= 1.0 + 1e-12,
    }
    return shift, receipts


def _direct_anchor_trajectory(
        clean: Tensor, output_mean: Tensor, source_mean: Tensor,
        encoder: Tensor, decoder: Tensor) -> Tensor:
    coefficient = (clean[:, 0] - source_mean) @ encoder.T
    future = output_mean.view(1, 1, -1) + torch.einsum(
        "bd,hod->bho", coefficient, decoder)
    return torch.cat((clean[:, :1], future), dim=1)


def _deployed_gain(
        rho: Tensor, weight: Tensor, mode: str, path: str) -> Tuple[Tensor, Tensor, Tensor]:
    deployed_rho = torch.ones_like(rho) if mode == "norho" else rho
    deployed_weight = torch.ones_like(weight) if mode == "norisk" else weight
    gain = deployed_rho * deployed_weight
    if mode == "anchoronly" or (path == "action" and mode == "noaction") \
            or (path == "observation" and mode == "nocorrect"):
        gain = torch.zeros_like(gain)
    return deployed_rho, deployed_weight, gain


def _base_trajectory(
        clean: Tensor, actions: Tensor, output_mean: Tensor,
        anchor_source_mean: Tensor, anchor_encoder: Tensor, anchor_decoder: Tensor,
        anchor_shift: Tensor, action_source_mean: Tensor, action_encoder: Tensor,
        action_decoder: Tensor, action_shift: Tensor) -> Tensor:
    episodes, length, dimension = clean.shape
    q_anchor = (clean[:, 0] - anchor_source_mean) @ anchor_encoder.T
    q_action = clean.new_zeros(episodes, dimension)
    values = [clean[:, 0]]
    for index in range(1, length):
        q_action = q_action + (actions[:, index - 1] - action_source_mean) @ action_encoder.T
        values.append(
            output_mean + q_anchor @ anchor_decoder[0].T
            + q_action @ action_decoder[0].T)
        q_anchor = q_anchor @ anchor_shift.T
        q_action = q_action @ action_shift.T
    return torch.stack(values, dim=1)


def _deployed_trajectory(
        clean_initial: Tensor, observed: Tensor, actions: Tensor, output_mean: Tensor,
        anchor_source_mean: Tensor, anchor_encoder: Tensor, anchor_decoder: Tensor,
        anchor_shift: Tensor, action_source_mean: Tensor, action_encoder: Tensor,
        action_decoder: Tensor, action_shift: Tensor,
        observation_source_mean: Tensor, observation_encoder: Tensor,
        observation_decoder: Tensor, observation_shift: Tensor,
        ) -> Tuple[Tensor, Tensor, Tensor]:
    """Replay the exact deployed recurrence in FP64 for fit-exposure auditing."""
    episodes, length, dimension = observed.shape
    q_anchor = (clean_initial - anchor_source_mean) @ anchor_encoder.T
    q_action = observed.new_zeros(episodes, dimension)
    q_observation = observed.new_zeros(episodes, dimension)
    priors = [clean_initial]
    posteriors = [clean_initial]
    innovations = [torch.zeros_like(clean_initial)]
    for index in range(1, length):
        q_action = q_action + (
            actions[:, index - 1] - action_source_mean) @ action_encoder.T
        prior = (
            output_mean + q_anchor @ anchor_decoder[0].T
            + q_action @ action_decoder[0].T
            + q_observation @ observation_decoder[0].T)
        innovation = observed[:, index] - prior
        q_observation = q_observation + (
            innovation - observation_source_mean) @ observation_encoder.T
        posterior = prior + (
            (innovation - observation_source_mean) @ observation_encoder.T
            @ observation_decoder[0].T)
        priors.append(prior)
        posteriors.append(posterior)
        innovations.append(innovation)
        q_anchor = q_anchor @ anchor_shift.T
        q_action = q_action @ action_shift.T
        q_observation = q_observation @ observation_shift.T
    return (
        torch.stack(priors, dim=1),
        torch.stack(posteriors, dim=1),
        torch.stack(innovations, dim=1),
    )


def fit_cvpf(
        clean: Tensor, observed: Tensor, actions: Tensor,
        *, mode: str = "full") -> CVPFFit:
    """Fit CVPF from synchronized train embeddings and executed actions only."""
    mode = _canonical_mode(mode)
    clean = _as_cpu_double(clean, "clean", 3)
    observed = _as_cpu_double(observed, "observed", 3)
    actions = _as_cpu_double(actions, "actions", 3)
    if clean.shape != observed.shape:
        raise ValueError("clean and observed views must have identical shapes")
    episodes, length, dimension = clean.shape
    horizon = length - 1
    if episodes < 4 or length < 3:
        raise ValueError("CVPF needs at least four episodes and three frames")
    if tuple(actions.shape[:2]) != (episodes, horizon):
        raise ValueError("actions must align with every transition")
    if float((clean[:, 0] - observed[:, 0]).abs().max()) != 0.0:
        raise ValueError("CVPF requires an exact synchronized initial observation")
    action_dim = actions.shape[-1]
    if action_dim > dimension:
        raise ValueError("CVPF fixed D schema requires action_dim <= output_dim")

    output_mean = clean.mean(dim=(0, 1))
    anchor_source_mean, anchor_encoder, anchor_decoder, anchor_receipts = _anchor_fit(
        clean, output_mean)
    anchor_shift, anchor_shift_receipts = _projected_shift(
        anchor_decoder, "anchor")
    anchor_base = _direct_anchor_trajectory(
        clean, output_mean, anchor_source_mean, anchor_encoder, anchor_decoder)
    action_residual = clean - anchor_base

    action_modes_small = _fit_event_modes(actions, action_residual, "action_pooled")
    action_weight_small, action_crossfit_receipts = _crossfit_weights(
        actions, action_residual, action_modes_small, "action")
    action_modes = _pad_modes(action_modes_small, dimension)
    action_weight = torch.zeros(dimension, dtype=torch.float64)
    action_weight[:action_dim] = action_weight_small
    action_rho, action_weight, action_gain = _deployed_gain(
        action_modes.rho, action_weight, mode, "action")
    action_decoder = action_modes.base_decoder * action_gain.view(1, 1, -1)
    action_shift, action_shift_receipts = _projected_shift(action_decoder, "action")

    base = _base_trajectory(
        clean, actions, output_mean,
        anchor_source_mean, anchor_encoder, anchor_decoder, anchor_shift,
        action_modes.source_mean, action_modes.encoder, action_decoder, action_shift)
    # Stage one discovers an observation response from the anchor+action
    # history.  Stage two then refits once on the *recursively deployed* stage-
    # one innovation history.  This fixed two-stage construction avoids a tuned
    # fixed-point iteration while ensuring that the final fit is exposed to
    # accumulated observation state rather than only an open-loop base.
    base_innovations = observed[:, 1:] - base[:, 1:]
    base_observation_residual = clean - base
    provisional_modes = _fit_event_modes(
        base_innovations, base_observation_residual, "observation_provisional")
    provisional_weight, provisional_crossfit_receipts = _crossfit_weights(
        base_innovations, base_observation_residual, provisional_modes,
        "observation_provisional")
    provisional_rho, provisional_weight, provisional_gain = _deployed_gain(
        provisional_modes.rho, provisional_weight, mode, "observation")
    provisional_decoder = (
        provisional_modes.base_decoder * provisional_gain.view(1, 1, -1))
    provisional_shift, provisional_shift_receipts = _projected_shift(
        provisional_decoder, "observation_provisional")
    provisional_prior, _, provisional_innovations = _deployed_trajectory(
        clean[:, 0], observed, actions, output_mean,
        anchor_source_mean, anchor_encoder, anchor_decoder, anchor_shift,
        action_modes.source_mean, action_modes.encoder, action_decoder, action_shift,
        provisional_modes.source_mean, provisional_modes.encoder,
        provisional_decoder, provisional_shift)
    final_fit_innovations = provisional_innovations[:, 1:]
    final_observation_residual = clean - provisional_prior
    observation_modes = _fit_event_modes(
        final_fit_innovations, final_observation_residual, "observation_pooled")
    observation_weight, observation_crossfit_receipts = _crossfit_weights(
        final_fit_innovations, final_observation_residual,
        observation_modes, "observation")
    observation_rho, observation_weight, observation_gain = _deployed_gain(
        observation_modes.rho, observation_weight, mode, "observation")
    observation_decoder = (
        observation_modes.base_decoder * observation_gain.view(1, 1, -1))
    observation_shift, observation_shift_receipts = _projected_shift(
        observation_decoder, "observation")
    final_prior, _, final_deployed_innovations = _deployed_trajectory(
        clean[:, 0], observed, actions, output_mean,
        anchor_source_mean, anchor_encoder, anchor_decoder, anchor_shift,
        action_modes.source_mean, action_modes.encoder, action_decoder, action_shift,
        observation_modes.source_mean, observation_modes.encoder,
        observation_decoder, observation_shift)
    fit_innovation_rms = float(final_fit_innovations.square().mean().sqrt())
    deployed_innovation_rms = float(
        final_deployed_innovations[:, 1:].square().mean().sqrt())
    innovation_exposure_ratio = (
        deployed_innovation_rms / fit_innovation_rms
        if fit_innovation_rms > 0 else (1.0 if deployed_innovation_rms == 0 else float("inf")))
    deployed_prior_error = float((final_prior[:, 1:] - clean[:, 1:]).square().mean())

    action_inactive_error = float(action_decoder[:, :, action_dim:].abs().max()) \
        if action_dim < dimension else 0.0
    closure_values = [
        anchor_shift_receipts["anchor_shift_closure_relative"],
        action_shift_receipts["action_shift_closure_relative"],
        observation_shift_receipts["observation_shift_closure_relative"],
    ]
    receipts: Dict[str, Any] = {
        "method": "CVPF-v15",
        "fit_mode": mode,
        "self_supervision": "paired_train_views_plus_executed_iid_actions",
        "reward_or_task_state_labels_used": False,
        "validation_or_corruption_identity_used": False,
        "learned_memory_parameters": 0,
        "episode_length": length,
        "future_horizon": horizon,
        "output_dimension": dimension,
        "action_dimension": action_dim,
        "streaming_state_floats": 3 * dimension,
        "physical_columns_per_role": dimension,
        "future_block_dimension": horizon * dimension,
        "future_covariance_materialized": False,
        "source_covariance_rule": "OAS_plus_machine_SPD_floor",
        "future_projection_rule": "all_valid_zero_padded_suffix_PLS",
        "fold_rule": "deterministic_even_odd_episode_cross_calibration",
        "fold_baseline_and_alignment_are_pooled": True,
        "crossfold_claim": "calibrated_training_heuristic_not_leakage_free_certificate",
        "fold_alignment_rule": "decoded_direction_assignment_then_degenerate_block_min",
        "gain_rule": "rho_times_symmetric_positive_part_EB_weight",
        "projected_shift_fitted_after_gain_support": True,
        "action_crossfit_mean_gain": float(action_gain.mean()),
        "correction_crossfit_mean_gain": float(observation_gain.mean()),
        "action_rho_mean": float(action_rho.mean()),
        "observation_rho_mean": float(observation_rho.mean()),
        "risk_weight_mean": float(torch.cat((action_weight, observation_weight)).mean()),
        "shift_closure_max_abs": float(max(closure_values)),
        "prefix_closure_max_abs": 0.0,
        "observation_fit_history": "one_provisional_then_one_recursive_deployed_refit",
        "observation_fit_uses_recursive_deployed_innovations": True,
        "observation_fit_innovation_rms": fit_innovation_rms,
        "observation_deployed_innovation_rms": deployed_innovation_rms,
        "observation_deployed_to_fit_innovation_rms_ratio": innovation_exposure_ratio,
        "train_recursive_deployed_prior_mse": deployed_prior_error,
        "mode_risk_scores_raw_future_decoder_before_projected_shift": True,
        "mode_risk_is_not_a_deployed_recursive_certificate": True,
        "action_exact_zero_padding_max_abs": action_inactive_error,
        "noaction_exact": mode in ("noaction", "anchoronly"),
        "nocorrect_exact": mode in ("nocorrect", "anchoronly"),
        "norisk_exact": mode == "norisk",
        "norho_exact": mode == "norho",
        "anchoronly_exact": mode == "anchoronly",
        "claim_boundary": (
            "finite_H_linear_predictive_crossfold_training_evidence_not_causal_effect"),
        **anchor_receipts,
        **anchor_shift_receipts,
        **dict(action_modes.receipts),
        **action_crossfit_receipts,
        **action_shift_receipts,
        **dict(provisional_modes.receipts),
        **provisional_crossfit_receipts,
        **provisional_shift_receipts,
        **dict(observation_modes.receipts),
        **observation_crossfit_receipts,
        **observation_shift_receipts,
    }
    return CVPFFit(
        output_mean=output_mean,
        anchor_source_mean=anchor_source_mean,
        action_source_mean=action_modes.source_mean,
        observation_source_mean=observation_modes.source_mean,
        anchor_encoder=anchor_encoder,
        action_encoder=action_modes.encoder,
        observation_encoder=observation_modes.encoder,
        anchor_decoder=anchor_decoder,
        action_decoder=action_decoder,
        observation_decoder=observation_decoder,
        anchor_shift=anchor_shift,
        action_shift=action_shift,
        observation_shift=observation_shift,
        action_rho=action_rho,
        observation_rho=observation_rho,
        action_weight=action_weight,
        observation_weight=observation_weight,
        action_gain=action_gain,
        observation_gain=observation_gain,
        receipts=receipts,
    )


class CrossViewPredictiveFiltrationMemory(nn.Module):
    """Zero-parameter streaming CVPF with a fixed three-role state."""

    MODES = CVPF_MODES

    def __init__(
            self, output_dim: int, action_dim: int, horizon: int,
            mode: str = "full", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        for name, value in (
                ("output_dim", output_dim), ("action_dim", action_dim),
                ("horizon", horizon)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if action_dim > output_dim:
            raise ValueError("CVPF requires action_dim <= output_dim")
        if not dtype.is_floating_point:
            raise ValueError("CVPF buffers require floating-point dtype")
        self.output_dim = output_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.state_dim = 3 * output_dim
        self.mode = _canonical_mode(mode)
        d, h, a = output_dim, horizon, action_dim
        self.register_buffer("output_mean", torch.zeros(d, dtype=dtype))
        self.register_buffer("anchor_source_mean", torch.zeros(d, dtype=dtype))
        self.register_buffer("action_source_mean", torch.zeros(a, dtype=dtype))
        self.register_buffer("observation_source_mean", torch.zeros(d, dtype=dtype))
        self.register_buffer("anchor_encoder", torch.eye(d, dtype=dtype))
        self.register_buffer("action_encoder", torch.zeros(d, a, dtype=dtype))
        self.register_buffer("observation_encoder", torch.eye(d, dtype=dtype))
        for name in ("anchor_decoder", "action_decoder", "observation_decoder"):
            self.register_buffer(name, torch.zeros(h, d, d, dtype=dtype))
        for name in ("anchor_shift", "action_shift", "observation_shift"):
            self.register_buffer(name, torch.zeros(d, d, dtype=dtype))
        for name in (
                "action_rho", "observation_rho", "action_weight",
                "observation_weight", "action_gain", "observation_gain"):
            self.register_buffer(name, torch.zeros(d, dtype=dtype))
        self.register_buffer("fit_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("operators_installed", torch.zeros((), dtype=torch.bool))
        self._fit_receipts: Dict[str, Any] = {}

    @classmethod
    def from_fit(
            cls, fit: CVPFFit, mode: str | None = None,
            dtype: torch.dtype | None = None) -> "CrossViewPredictiveFiltrationMemory":
        if not isinstance(fit, CVPFFit):
            raise TypeError("fit must be a CVPFFit")
        memory = cls(
            fit.output_mean.numel(), fit.action_source_mean.numel(),
            fit.anchor_decoder.shape[0],
            mode=mode or str(fit.receipts.get("fit_mode", "full")),
            dtype=dtype or fit.output_mean.dtype)
        memory.install_fit(fit)
        return memory

    def get_extra_state(self) -> Dict[str, Any]:
        return {"fit_receipts": copy.deepcopy(self._fit_receipts)}

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or not isinstance(
                state.get("fit_receipts", {}), Mapping):
            raise RuntimeError("invalid CVPF extra state")
        self._fit_receipts = copy.deepcopy(dict(state.get("fit_receipts", {})))

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    @torch.no_grad()
    def install_fit(self, fit: CVPFFit) -> None:
        if not isinstance(fit, CVPFFit):
            raise TypeError("fit must be a CVPFFit")
        if fit.receipts.get("fit_mode") != self.mode:
            raise ValueError(
                f"{self.mode} memory requires matching fit_mode, got "
                f"{fit.receipts.get('fit_mode')!r}")
        d, h, a = self.output_dim, self.horizon, self.action_dim
        expected = {
            "output_mean": (d,),
            "anchor_source_mean": (d,),
            "action_source_mean": (a,),
            "observation_source_mean": (d,),
            "anchor_encoder": (d, d),
            "action_encoder": (d, a),
            "observation_encoder": (d, d),
            "anchor_decoder": (h, d, d),
            "action_decoder": (h, d, d),
            "observation_decoder": (h, d, d),
            "anchor_shift": (d, d),
            "action_shift": (d, d),
            "observation_shift": (d, d),
            "action_rho": (d,),
            "observation_rho": (d,),
            "action_weight": (d,),
            "observation_weight": (d,),
            "action_gain": (d,),
            "observation_gain": (d,),
        }
        for name, shape in expected.items():
            value = getattr(fit, name)
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError(f"fit field {name} must be finite with shape {shape}")
        for name in (
                "action_rho", "observation_rho", "action_weight",
                "observation_weight", "action_gain", "observation_gain"):
            value = getattr(fit, name)
            if bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f"{name} must lie in [0,1]")
        if self.mode in ("noaction", "anchoronly") and torch.count_nonzero(
                fit.action_decoder):
            raise ValueError("noaction/anchoronly fit must have exact-zero action decoder")
        if self.mode in ("nocorrect", "anchoronly") and torch.count_nonzero(
                fit.observation_decoder):
            raise ValueError("nocorrect/anchoronly fit must have exact-zero observation decoder")
        if self.mode == "norisk" and not torch.equal(
                fit.action_weight, torch.ones_like(fit.action_weight)):
            raise ValueError("norisk fit must deploy unit action weights")
        if self.mode == "norisk" and not torch.equal(
                fit.observation_weight, torch.ones_like(fit.observation_weight)):
            raise ValueError("norisk fit must deploy unit observation weights")
        if self.mode == "norho" and not torch.equal(
                fit.action_rho, torch.ones_like(fit.action_rho)):
            raise ValueError("norho fit must deploy unit action rho")
        if self.mode == "norho" and not torch.equal(
                fit.observation_rho, torch.ones_like(fit.observation_rho)):
            raise ValueError("norho fit must deploy unit observation rho")
        for name in expected:
            destination = getattr(self, name)
            destination.copy_(getattr(fit, name).to(destination))
        self.fit_updates.add_(1)
        self.operators_installed.fill_(True)
        self._fit_receipts = copy.deepcopy(dict(fit.receipts))

    def _validate_state(self, state: CVPFState) -> int:
        if not isinstance(state, CVPFState):
            raise ValueError("state must be CVPFState(anchor,action,observation)")
        batch = state.anchor.shape[0] if state.anchor.dim() == 2 else -1
        for name, value in zip(CVPFState._fields, state, strict=True):
            if (tuple(value.shape) != (batch, self.output_dim)
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError(
                    f"state {name} must be finite with shape (B,{self.output_dim})")
        return batch

    def initial_state(self, observation: Tensor) -> CVPFState:
        if (not isinstance(observation, torch.Tensor) or observation.dim() != 2
                or observation.shape[0] < 1 or observation.shape[1] != self.output_dim
                or not observation.is_floating_point() or not torch.isfinite(observation).all()):
            raise ValueError(
                f"initial observation must be finite with shape (B,{self.output_dim})")
        anchor = functional.linear(
            observation - self.anchor_source_mean.to(observation),
            self.anchor_encoder.to(observation))
        zero = torch.zeros_like(anchor)
        return CVPFState(anchor, zero, zero)

    def read_state(self, state: CVPFState) -> Tensor:
        self._validate_state(state)
        return (
            self.output_mean.to(state.anchor)
            + functional.linear(state.anchor, self.anchor_decoder[0].to(state.anchor))
            + functional.linear(state.action, self.action_decoder[0].to(state.anchor))
            + functional.linear(
                state.observation, self.observation_decoder[0].to(state.anchor)))

    def step(
            self, state: CVPFState, observation: Tensor, action: Tensor,
            return_details: bool = False):
        batch = self._validate_state(state)
        if (not isinstance(observation, torch.Tensor)
                or tuple(observation.shape) != (batch, self.output_dim)
                or not observation.is_floating_point() or not torch.isfinite(observation).all()):
            raise ValueError(
                f"observation must be finite with shape ({batch},{self.output_dim})")
        if (not isinstance(action, torch.Tensor)
                or tuple(action.shape) != (batch, self.action_dim)
                or not action.is_floating_point() or not torch.isfinite(action).all()):
            raise ValueError(f"action must be finite with shape ({batch},{self.action_dim})")
        observation = observation.to(state.anchor)
        action = action.to(state.anchor)
        action_score = functional.linear(
            action - self.action_source_mean.to(action), self.action_encoder.to(action))
        current_action = state.action + action_score
        current = CVPFState(state.anchor, current_action, state.observation)
        prior = self.read_state(current)
        innovation = observation - prior
        observation_score = functional.linear(
            innovation - self.observation_source_mean.to(innovation),
            self.observation_encoder.to(innovation))
        current_observation = state.observation + observation_score
        posterior_state = CVPFState(state.anchor, current_action, current_observation)
        posterior = self.read_state(posterior_state)
        action_effect = functional.linear(
            action_score, self.action_decoder[0].to(action_score))
        correction = functional.linear(
            observation_score, self.observation_decoder[0].to(observation_score))
        next_state = CVPFState(
            functional.linear(state.anchor, self.anchor_shift.to(state.anchor)),
            functional.linear(current_action, self.action_shift.to(current_action)),
            functional.linear(
                current_observation, self.observation_shift.to(current_observation)),
        )
        if not return_details:
            return posterior, next_state
        return posterior, next_state, {
            "prior_read": prior,
            "posterior_read": posterior,
            "innovation": innovation,
            "action_effect": action_effect,
            "correction": correction,
            "anchor_state": posterior_state.anchor,
            "action_state": posterior_state.action,
            "observation_state": posterior_state.observation,
        }

    def forward(self, observations: Tensor, actions: Tensor, return_details: bool = False):
        if (not isinstance(observations, torch.Tensor) or observations.dim() != 3
                or observations.shape[0] < 1 or observations.shape[1] < 1
                or observations.shape[2] != self.output_dim
                or not observations.is_floating_point() or not torch.isfinite(observations).all()):
            raise ValueError(f"observations must have shape (B,T,{self.output_dim})")
        expected_actions = (
            observations.shape[0], observations.shape[1] - 1, self.action_dim)
        if (not isinstance(actions, torch.Tensor) or tuple(actions.shape) != expected_actions
                or not actions.is_floating_point() or not torch.isfinite(actions).all()):
            raise ValueError(f"actions must have shape {expected_actions}")
        actions = actions.to(observations)
        state = self.initial_state(observations[:, 0])
        # The synchronized first frame is an exact observation, not a decoded forecast.
        reads = [observations[:, 0]]
        priors = [observations[:, 0]]
        anchors = [state.anchor]
        action_states = [state.action]
        observation_states = [state.observation]
        innovations = [torch.zeros_like(observations[:, 0])]
        action_effects = [torch.zeros_like(observations[:, 0])]
        corrections = [torch.zeros_like(observations[:, 0])]
        for index in range(1, observations.shape[1]):
            read, state, details = self.step(
                state, observations[:, index], actions[:, index - 1],
                return_details=True)
            reads.append(read)
            priors.append(details["prior_read"])
            anchors.append(details["anchor_state"])
            action_states.append(details["action_state"])
            observation_states.append(details["observation_state"])
            innovations.append(details["innovation"])
            action_effects.append(details["action_effect"])
            corrections.append(details["correction"])
        posterior = torch.stack(reads, dim=1)
        if not return_details:
            return posterior
        return posterior, {
            "reads": posterior,
            "posterior_reads": posterior,
            "prior_reads": torch.stack(priors, dim=1),
            "anchor_states": torch.stack(anchors, dim=1),
            "action_states": torch.stack(action_states, dim=1),
            "observation_states": torch.stack(observation_states, dim=1),
            "innovations": torch.stack(innovations, dim=1),
            "action_effects": torch.stack(action_effects, dim=1),
            "corrections": torch.stack(corrections, dim=1),
        }

    def horizons(self) -> Dict[str, float]:
        return {
            "future_horizon": float(self.horizon),
            "state_dim": float(self.state_dim),
            "anchor_state_floats": float(self.output_dim),
            "action_state_floats": float(self.output_dim),
            "observation_state_floats": float(self.output_dim),
        }

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, Any]:
        action_gain = float(self.action_gain.double().mean())
        correction_gain = float(self.observation_gain.double().mean())
        risk_gain = float(torch.cat((
            self.action_weight.double(), self.observation_weight.double())).mean())
        rho = float(torch.cat((
            self.action_rho.double(), self.observation_rho.double())).mean())
        diagnostics = copy.deepcopy(self._fit_receipts)
        diagnostics.update({
            "method": "CVPF-v15",
            "mode": self.mode,
            "operators_installed": bool(self.operators_installed),
            "fit_updates": int(self.fit_updates),
            "gradient_parameter_count": self.parameter_count(),
            "future_horizon": self.horizon,
            "state_dim": self.state_dim,
            "streaming_state_floats": self.state_dim,
            "streaming_covariance_floats": 0,
            "streaming_max_abs": 0.0,
            "prefix_closure_max_abs": 0.0,
            "shift_closure_max_abs": max(
                float(self._fit_receipts.get("anchor_shift_closure_relative", 0.0)),
                float(self._fit_receipts.get("action_shift_closure_relative", 0.0)),
                float(self._fit_receipts.get("observation_shift_closure_relative", 0.0))),
            "action_gain": action_gain,
            "correction_gain": correction_gain,
            "risk_gain": risk_gain,
            "rho": rho,
            "action_crossfit_mean_gain": action_gain,
            "correction_crossfit_mean_gain": correction_gain,
            "noaction_exact": self.mode in ("noaction", "anchoronly"),
            "nocorrect_exact": self.mode in ("nocorrect", "anchoronly"),
            "norisk_exact": self.mode == "norisk",
            "norho_exact": self.mode == "norho",
            "anchoronly_exact": self.mode == "anchoronly",
        })
        return diagnostics


CVPFv15Memory = CrossViewPredictiveFiltrationMemory
