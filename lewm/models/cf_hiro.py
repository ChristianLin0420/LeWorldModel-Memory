"""Cross-fold-agreement Hankel--Riccati observer (CF-HIRO) prototype.

This is an isolated numerical prototype for a self-supervised predictive-state
memory.  It deliberately has no dependency on the V12 implementation or trainer.
Given paired clean/observed embedding trajectories and randomized executed actions,
the fitter performs four detached operations:

1. estimate every available action-to-future cross moment separately on even and
   odd episode folds;
2. realize the resulting all-lag block Hankel operator, continuously shrinking each
   shared singular direction by an empirical-Bayes positive-part fold agreement;
3. project the recurrence onto orthogonal coordinates containing independent real-normal
   contraction blocks; and
4. estimate process/measurement covariance from the paired views, solve an offline
   Riccati fixed point, and deploy only its fixed gain on the online state mean.

There are no selected temporal horizons, loss weights, learned memory parameters,
or manually selected state rank.  This does *not* make realization choice-free.
The following choices are structural and are reported in every fit receipt:

* the all-lag Hankel is split into the unique left-balanced integer partition;
* every numerical singular direction is retained (machine-precision pseudoinverse
  totalizes exact null directions rather than pretending they are identifiable);
* fold agreement uses the equal-fold empirical-Bayes positive-part decomposition;
* inter-block Schur couplings are removed, complex blocks are projected to canonical
  real-normal rotations, and every block is contracted to the float32 boundary;
* covariance uses analytic OAS shrinkage toward an isotropic target; and
* the clean member of a paired view is treated as the process target while their
  difference is treated as measurement corruption.

The random-action moment identity is causal only when actions are randomized (or at
least sequentially exogenous).  Full-rank action covariance is audited, but the code
cannot prove exogeneity from observational data.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, NamedTuple, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
FLOAT32_STABILITY_BOUNDARY = 1.0 - math.sqrt(torch.finfo(torch.float32).eps)


def _as_cpu_double(value: Tensor, name: str, dimensions: int) -> Tensor:
    if not isinstance(value, torch.Tensor) or value.dim() != dimensions:
        actual = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        raise ValueError(f"{name} must be a {dimensions}-D tensor, got {actual}")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    return value.detach().to(device="cpu", dtype=torch.float64).contiguous()


def _machine_pinv(value: Tensor, *, hermitian: bool = False) -> Tensor:
    """Moore--Penrose inverse with only a dtype/shape-derived numerical cutoff."""
    if value.numel() == 0:
        return value.transpose(-2, -1)
    rtol = max(value.shape[-2:]) * torch.finfo(value.dtype).eps
    return torch.linalg.pinv(value, atol=0.0, rtol=rtol, hermitian=hermitian)


def _matrix_rank(value: Tensor) -> int:
    if value.numel() == 0:
        return 0
    singular = torch.linalg.svdvals(value)
    tolerance = max(value.shape[-2:]) * torch.finfo(value.dtype).eps * singular.max()
    return int(torch.count_nonzero(singular > tolerance))


def _symmetrize(value: Tensor) -> Tensor:
    return 0.5 * (value + value.transpose(-2, -1))


@dataclass(frozen=True)
class FoldAgreementMarkovMoments:
    """All action-to-future Markov moments from two episode folds."""

    even: Tensor
    odd: Tensor
    average: Tensor
    sample_counts: Tuple[int, ...]
    receipts: Mapping[str, Any]


@dataclass(frozen=True)
class HankelRealization:
    """Continuously fold-agreement-shrunk realization in orthogonal coordinates."""

    state_matrix: Tensor
    action_matrix: Tensor
    read_matrix: Tensor
    fold_agreement: Tensor
    hankel_singular_values: Tensor
    shrunk_singular_values: Tensor
    receipts: Mapping[str, Any]


@dataclass(frozen=True)
class CFHIROFit:
    """Complete detached fit consumed by :class:`FoldAgreementHankelRiccatiMemory`."""

    state_matrix: Tensor
    action_matrix: Tensor
    read_matrix: Tensor
    process_covariance: Tensor
    measurement_covariance: Tensor
    initial_covariance: Tensor
    initial_map: Tensor
    output_mean: Tensor
    action_mean: Tensor
    steady_prior_covariance: Tensor
    steady_gain: Tensor
    markov_even: Tensor
    markov_odd: Tensor
    fold_agreement: Tensor
    receipts: Mapping[str, Any]


class HIROState(NamedTuple):
    """Online dynamic mean plus an immutable output-space complement anchor."""

    mean: Tensor
    complement: Tensor


CF_HIRO_MODES = frozenset({
    "full", "noaction", "noshrink", "fullanchor", "nocorrect", "triangular",
})
_MODE_ALIASES = {
    "noagreement": "noshrink",
    "nocorrection": "nocorrect",
}


def _canonical_mode(mode: str) -> str:
    mode = _MODE_ALIASES.get(mode, mode)
    if mode not in CF_HIRO_MODES:
        raise ValueError(f"unknown CF-HIRO mode {mode!r}")
    return mode


def estimate_all_lag_markov_moments(
        clean: Tensor, actions: Tensor) -> FoldAgreementMarkovMoments:
    """Estimate ``E[y[t+j+1] a[t]^T] E[a[t]a[t]^T]^-1`` at every lag.

    Centering is performed separately at every source time and on each episode fold.
    This removes time-dependent means without mixing future samples.  Under randomized
    sequentially exogenous actions, the population moments are ``C A^j B``.  Even and
    odd episodes are never pooled until after both estimates and their action-rank
    audits have been produced.
    """
    clean = _as_cpu_double(clean, "clean", 3)
    actions = _as_cpu_double(actions, "actions", 3)
    episodes, length, output_dim = clean.shape
    if episodes < 4 or length < 5:
        raise ValueError("all-lag fold agreement requires at least 4 episodes and 5 frames")
    if tuple(actions.shape[:2]) != (episodes, length - 1):
        raise ValueError(
            f"actions must have shape ({episodes},{length - 1},A), got {tuple(actions.shape)}")
    action_dim = actions.shape[-1]
    fold_indices = {
        "even": torch.arange(0, episodes, 2),
        "odd": torch.arange(1, episodes, 2),
    }
    estimates: Dict[str, Tensor] = {}
    fold_ranks: Dict[str, list[int]] = {}
    fold_conditions: Dict[str, list[float]] = {}
    sample_counts: list[int] = []

    for fold_name, indices in fold_indices.items():
        moments = []
        ranks = []
        conditions = []
        for lag in range(length - 1):
            action_gram = torch.zeros(action_dim, action_dim, dtype=torch.float64)
            output_action = torch.zeros(output_dim, action_dim, dtype=torch.float64)
            count = 0
            for source in range(length - 1 - lag):
                source_action = actions[indices, source]
                future_output = clean[indices, source + lag + 1]
                source_action = source_action - source_action.mean(dim=0, keepdim=True)
                future_output = future_output - future_output.mean(dim=0, keepdim=True)
                action_gram.add_(source_action.T @ source_action)
                output_action.add_(future_output.T @ source_action)
                count += int(indices.numel())
            rank = _matrix_rank(action_gram)
            if rank != action_dim:
                raise ValueError(
                    "randomized-action audit failed: action covariance is rank "
                    f"{rank}/{action_dim} on {fold_name} fold at lag {lag}")
            singular = torch.linalg.svdvals(action_gram)
            moments.append(output_action @ _machine_pinv(action_gram, hermitian=True))
            ranks.append(rank)
            conditions.append(float(singular.max() / singular.min()))
            if fold_name == "even":
                sample_counts.append(count)
        estimates[fold_name] = torch.stack(moments)
        fold_ranks[fold_name] = ranks
        fold_conditions[fold_name] = conditions

    held_fold_action_r2: Dict[str, float] = {}
    held_fold_action_r2_by_lag: Dict[str, list[float]] = {}
    for fit_fold, score_fold in (("even", "odd"), ("odd", "even")):
        indices = fold_indices[score_fold]
        total_squared_error = torch.zeros((), dtype=torch.float64)
        total_target_energy = torch.zeros((), dtype=torch.float64)
        per_lag = []
        for lag in range(length - 1):
            lag_squared_error = torch.zeros((), dtype=torch.float64)
            lag_target_energy = torch.zeros((), dtype=torch.float64)
            action_to_future = estimates[fit_fold][lag]
            for source in range(length - 1 - lag):
                source_action = actions[indices, source]
                future_output = clean[indices, source + lag + 1]
                source_action = source_action - source_action.mean(dim=0, keepdim=True)
                future_output = future_output - future_output.mean(dim=0, keepdim=True)
                error = future_output - source_action @ action_to_future.T
                lag_squared_error.add_(error.square().sum())
                lag_target_energy.add_(future_output.square().sum())
            denominator = lag_target_energy.clamp_min(torch.finfo(torch.float64).tiny)
            per_lag.append(float(1.0 - lag_squared_error / denominator))
            total_squared_error.add_(lag_squared_error)
            total_target_energy.add_(lag_target_energy)
        direction = f"{fit_fold}_to_{score_fold}"
        held_fold_action_r2[direction] = float(
            1.0 - total_squared_error
            / total_target_energy.clamp_min(torch.finfo(torch.float64).tiny))
        held_fold_action_r2_by_lag[direction] = per_lag

    average = 0.5 * (estimates["even"] + estimates["odd"])
    receipts: Dict[str, Any] = {
        "markov_identity": "E[y_{t+j+1}a_t^T]E[a_ta_t^T]^-1",
        "randomization_assumption": "sequentially_exogenous_executed_actions",
        "randomization_proved_from_data": False,
        "fold_assignment": "episode_index_even_odd",
        "time_specific_centering": True,
        "all_available_lags": True,
        "markov_lag_count": length - 1,
        "first_lag": 0,
        "last_lag": length - 2,
        "output_dim": output_dim,
        "action_dim": action_dim,
        "even_episodes": int(fold_indices["even"].numel()),
        "odd_episodes": int(fold_indices["odd"].numel()),
        "even_action_ranks": fold_ranks["even"],
        "odd_action_ranks": fold_ranks["odd"],
        "maximum_action_gram_condition": max(
            fold_conditions["even"] + fold_conditions["odd"]),
        "held_fold_action_r2_even_to_odd": held_fold_action_r2["even_to_odd"],
        "held_fold_action_r2_odd_to_even": held_fold_action_r2["odd_to_even"],
        "held_fold_action_r2_min": min(held_fold_action_r2.values()),
        "held_fold_action_r2_mean": sum(held_fold_action_r2.values()) / 2.0,
        "held_fold_action_r2_h1_even_to_odd": (
            held_fold_action_r2_by_lag["even_to_odd"][0]),
        "held_fold_action_r2_h1_odd_to_even": (
            held_fold_action_r2_by_lag["odd_to_even"][0]),
        "held_fold_action_r2_by_lag_even_to_odd": (
            held_fold_action_r2_by_lag["even_to_odd"]),
        "held_fold_action_r2_by_lag_odd_to_even": (
            held_fold_action_r2_by_lag["odd_to_even"]),
    }
    return FoldAgreementMarkovMoments(
        even=estimates["even"], odd=estimates["odd"], average=average,
        sample_counts=tuple(sample_counts), receipts=receipts)


def _block_hankel(markov: Tensor, rows: int, columns: int, shift: int) -> Tensor:
    blocks = []
    for row in range(rows):
        blocks.append(torch.cat(
            [markov[row + column + shift] for column in range(columns)], dim=1))
    return torch.cat(blocks, dim=0)


def _operator_receipts(value: np.ndarray, label: str) -> Dict[str, Any]:
    """Return finite-horizon operator/transient diagnostics for one square matrix."""
    if value.shape[0] == 0:
        raise ValueError("operator diagnostics require a non-empty square matrix")
    tensor = torch.from_numpy(np.array(value, copy=True)).double()
    spectral_radius = float(torch.linalg.eigvals(tensor).abs().max())
    operator_norm = float(torch.linalg.matrix_norm(tensor, ord=2))
    commutator = tensor.T @ tensor - tensor @ tensor.T
    scale = max(float(tensor.norm().square()), torch.finfo(torch.float64).tiny)
    horizons = (1, 2, 4, 8, 16, 32)
    power = torch.eye(tensor.shape[0], dtype=torch.float64)
    norms = []
    cursor = 0
    for horizon in horizons:
        for _ in range(cursor, horizon):
            power = power @ tensor
        cursor = horizon
        norms.append(float(torch.linalg.matrix_norm(power, ord=2)))
    return {
        f"{label}_spectral_radius": spectral_radius,
        f"{label}_operator_norm": operator_norm,
        f"{label}_normality_residual": float(commutator.norm()),
        f"{label}_relative_normality_residual": float(commutator.norm()) / scale,
        f"{label}_transient_horizons": list(horizons),
        f"{label}_power_operator_norms": norms,
        f"{label}_maximum_power_operator_norm": max(norms),
    }


def _schur_blocks(matrix: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    index = 0
    while index < matrix.shape[0]:
        width = (2 if index + 1 < matrix.shape[0]
                 and abs(matrix[index + 1, index]) > tolerance else 1)
        blocks.append((index, width))
        index += width
    return blocks


def _orthogonal_contraction(
        value: Tensor, *, deployment: str
        ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    """Project a recurrence into stable orthogonal coordinates.

    ``normal`` removes every coupling between real-Schur diagonal blocks and replaces
    each complex 2x2 block by ``[[a,b],[-b,a]]`` with the same (contracted)
    eigenvalues. ``triangular`` is the explicit control that retains all quasi-upper
    inter-block couplings while applying the same radial eigenvalue contraction.
    """
    if deployment not in {"normal", "triangular"}:
        raise ValueError(f"unknown transition deployment {deployment!r}")
    try:
        from scipy.linalg import schur
    except ImportError as error:  # pragma: no cover - project declares scipy
        raise RuntimeError("CF-HIRO fitting requires the declared scipy dependency") from error

    source = value.detach().cpu().double().numpy()
    schur_matrix, basis = schur(source, output="real")
    raw_schur = np.array(schur_matrix, copy=True)
    before = float(np.max(np.abs(np.linalg.eigvals(raw_schur))))
    scale = max(1.0, float(np.max(np.abs(raw_schur))))
    tolerance = value.shape[0] * np.finfo(np.float64).eps * scale
    blocks = _schur_blocks(raw_schur, tolerance)
    triangular = np.array(raw_schur, copy=True)
    block_scales = []
    for index, width in blocks:
        block = triangular[index:index + width, index:index + width]
        radius = float(np.max(np.abs(np.linalg.eigvals(block))))
        factor = (min(1.0, FLOAT32_STABILITY_BOUNDARY / radius)
                  if radius > 0.0 else 1.0)
        triangular[index:index + width, index:index + width] *= factor
        block_scales.append(factor)

    normal = np.zeros_like(triangular)
    for index, width in blocks:
        block = triangular[index:index + width, index:index + width]
        if width == 1:
            normal[index, index] = block[0, 0]
            continue
        # A real-Schur complex block has conjugate eigenvalues a +/- ib.  Orthogonal
        # coordinates alone cannot remove its nonnormal scale imbalance, so this is an
        # explicit nearest-structure projection preserving those contracted eigenvalues.
        a = 0.5 * float(np.trace(block))
        imaginary_square = max(0.0, float(np.linalg.det(block)) - a * a)
        sign_source = float(block[0, 1])
        b = math.copysign(math.sqrt(imaginary_square), sign_source or 1.0)
        normal[index:index + 2, index:index + 2] = np.asarray(
            ((a, b), (-b, a)), dtype=np.float64)

    selected = normal if deployment == "normal" else triangular
    after = float(np.max(np.abs(np.linalg.eigvals(selected))))
    diagonal_blocks = np.zeros_like(triangular)
    for index, width in blocks:
        diagonal_blocks[index:index + width, index:index + width] = \
            triangular[index:index + width, index:index + width]
    interblock_norm = float(np.linalg.norm(triangular - diagonal_blocks))
    projection_distance = float(np.linalg.norm(selected - raw_schur))
    projection_denominator = max(float(np.linalg.norm(raw_schur)), np.finfo(np.float64).tiny)
    basis_error = float(np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1])))
    receipts = {
        "coordinate_system": "orthogonal_real_block_coordinates",
        "transition_deployment": deployment,
        "schur_block_detection_tolerance": tolerance,
        "float32_stability_boundary": FLOAT32_STABILITY_BOUNDARY,
        "spectral_radius_before_projection": before,
        "spectral_radius_after_projection": after,
        "schur_block_radial_scales": block_scales,
        "schur_block_widths": [width for _, width in blocks],
        "schur_interblock_coupling_norm": interblock_norm,
        "interblock_couplings_zeroed": deployment == "normal",
        "complex_blocks_canonical_real_normal": deployment == "normal",
        "orthogonal_basis_error": basis_error,
        "transition_projection_frobenius_distance": projection_distance,
        "transition_projection_relative_distance": projection_distance / projection_denominator,
        "stability_projection_active": any(factor < 1.0 for factor in block_scales),
        **_operator_receipts(raw_schur, "raw_schur"),
        **_operator_receipts(triangular, "triangular_contraction"),
        **_operator_receipts(normal, "normal_contraction"),
        **_operator_receipts(selected, "deployed_transition"),
    }
    return (
        torch.from_numpy(np.array(selected, copy=True)).double(),
        torch.from_numpy(np.array(basis, copy=True)).double(),
        receipts,
    )


def _markov_from_filtered_hankels(
        h0: Tensor, h1: Tensor, *, rows: int, columns: int,
        lag_count: int, output_dim: int, action_dim: int) -> Tensor:
    """Average repeated anti-diagonal blocks into one filtered all-lag sequence."""
    values = h0.new_zeros(lag_count, output_dim, action_dim)
    counts = h0.new_zeros(lag_count)
    for source, shift in ((h0, 0), (h1, 1)):
        for row in range(rows):
            for column in range(columns):
                lag = row + column + shift
                values[lag].add_(source[
                    row * output_dim:(row + 1) * output_dim,
                    column * action_dim:(column + 1) * action_dim])
                counts[lag].add_(1.0)
    if bool((counts == 0).any()):
        raise RuntimeError("filtered Hankel anti-diagonals did not cover every lag")
    return values / counts.view(-1, 1, 1)


def _refit_action_from_all_lags(
        state_matrix: Tensor, read_matrix: Tensor, markov: Tensor
        ) -> Tuple[Tensor, Dict[str, Any]]:
    """Least-squares refit of B with fixed A,C using every filtered Markov lag."""
    state_dim = state_matrix.shape[0]
    power = torch.eye(state_dim, dtype=torch.float64)
    regressors = []
    for _ in range(markov.shape[0]):
        regressors.append(read_matrix @ power)
        power = power @ state_matrix
    regressor = torch.cat(regressors, dim=0)
    target = markov.reshape(-1, markov.shape[-1])
    action_matrix = _machine_pinv(regressor) @ target
    predicted = regressor @ action_matrix
    denominator = target.norm().clamp_min(torch.finfo(torch.float64).tiny)
    return action_matrix, {
        "action_refit": "all_filtered_markov_lags_fixed_transition_and_read",
        "action_refit_lags": int(markov.shape[0]),
        "action_refit_regressor_rank": _matrix_rank(regressor),
        "action_refit_relative_residual": float((predicted - target).norm() / denominator),
    }


def realize_fold_agreement_hankel(
        even_markov: Tensor, odd_markov: Tensor, *,
        agreement_mode: str = "empirical_bayes", transition_mode: str = "normal",
        ) -> HankelRealization:
    """Realize all lags with continuous shared-basis fold-agreement shrinkage.

    For shared singular direction ``i``, let ``m_i`` be the average fold projection
    and ``d_i`` half their difference. Its empirical-Bayes reliability is the
    continuous positive-part estimate ``max(m_i^2-d_i^2,0)/(m_i^2+machine_floor)``.
    ``unit`` is the exact no-shrink control. No direction is physically removed: the
    state schema retains the full rectangular Hankel order even for zero weights.
    """
    even_markov = _as_cpu_double(even_markov, "even_markov", 3)
    odd_markov = _as_cpu_double(odd_markov, "odd_markov", 3)
    if even_markov.shape != odd_markov.shape:
        raise ValueError("even and odd Markov tensors must have identical shapes")
    if agreement_mode not in {"empirical_bayes", "unit"}:
        raise ValueError(f"unknown fold-agreement mode {agreement_mode!r}")
    if transition_mode not in {"normal", "triangular"}:
        raise ValueError(f"unknown transition mode {transition_mode!r}")
    lag_count, output_dim, action_dim = even_markov.shape
    if lag_count < 4:
        raise ValueError("Hankel realization needs at least four Markov lags")

    # Unique deterministic left-balanced split. H1 consumes the final available lag.
    block_rows = lag_count // 2
    block_columns = lag_count - block_rows
    h0_even = _block_hankel(even_markov, block_rows, block_columns, shift=0)
    h0_odd = _block_hankel(odd_markov, block_rows, block_columns, shift=0)
    h1_even = _block_hankel(even_markov, block_rows, block_columns, shift=1)
    h1_odd = _block_hankel(odd_markov, block_rows, block_columns, shift=1)
    h0 = 0.5 * (h0_even + h0_odd)
    h1 = 0.5 * (h1_even + h1_odd)

    left, singular, right_h = torch.linalg.svd(h0, full_matrices=False)
    right = right_h.T
    fold_even_projection = torch.diagonal(left.T @ h0_even @ right)
    fold_odd_projection = torch.diagonal(left.T @ h0_odd @ right)
    signal = 0.5 * (fold_even_projection + fold_odd_projection)
    disagreement = 0.5 * (fold_even_projection - fold_odd_projection)
    scale_square = max(float(singular.max().square()), torch.finfo(torch.float64).tiny)
    machine_floor = torch.finfo(torch.float64).eps * scale_square
    if agreement_mode == "unit":
        agreement = torch.ones_like(signal)
    else:
        agreement = (signal.square() - disagreement.square()).clamp_min(0.0) / (
            signal.square() + machine_floor)
    shrunk = singular * agreement

    # The identical reliability filter is applied on the left and right state axes of
    # both H0 and H1. This prevents the old unshrunk-H1 / shrunk-H0 mismatch from
    # amplifying A by inverse agreement factors.
    root_agreement = agreement.sqrt()
    projected_h1 = left.T @ h1 @ right
    filtered_h0 = (left * shrunk.unsqueeze(0)) @ right_h
    filtered_h1 = left @ (
        root_agreement.unsqueeze(1) * projected_h1
        * root_agreement.unsqueeze(0)) @ right_h

    square_root = torch.sqrt(shrunk.clamp_min(0.0))
    observability = left * square_root.unsqueeze(0)
    reachability = square_root.unsqueeze(1) * right_h
    state_raw = (
        _machine_pinv(observability) @ filtered_h1 @ _machine_pinv(reachability))
    read_raw = observability[:output_dim]

    state_deployed, schur_basis, schur_receipts = _orthogonal_contraction(
        state_raw, deployment=transition_mode)
    read_schur = read_raw @ schur_basis
    filtered_markov = _markov_from_filtered_hankels(
        filtered_h0, filtered_h1, rows=block_rows, columns=block_columns,
        lag_count=lag_count, output_dim=output_dim, action_dim=action_dim)
    action_schur, action_receipts = _refit_action_from_all_lags(
        state_deployed, read_schur, filtered_markov)
    state_order = state_deployed.shape[0]
    receipts: Dict[str, Any] = {
        "realization": "two_fold_agreement_all_lag_block_hankel",
        "hankel_split_rule": "left_balanced_integer_partition",
        "block_rows": block_rows,
        "block_columns": block_columns,
        "h0_last_lag": lag_count - 2,
        "h1_last_lag": lag_count - 1,
        "all_lags_consumed": True,
        "state_order_rule": "full_rectangular_hankel_numerical_order",
        "state_order": state_order,
        "model_order_threshold": None,
        "machine_pseudoinverse_rtol": max(h0.shape) * torch.finfo(torch.float64).eps,
        "fold_agreement_mode": agreement_mode,
        "fold_shrinkage_rule": (
            "unit_no_shrink_control" if agreement_mode == "unit" else
            "positive_part_max(m^2-d^2,0)/(m^2+machine_floor)"),
        "agreement_applied_to_h0_and_h1": True,
        "fold_agreement_min": float(agreement.min()),
        "fold_agreement_mean": float(agreement.mean()),
        "fold_agreement_max": float(agreement.max()),
        "hankel_rank_machine": _matrix_rank(h0),
        **schur_receipts,
        **action_receipts,
    }
    return HankelRealization(
        state_matrix=state_deployed,
        action_matrix=action_schur,
        read_matrix=read_schur,
        fold_agreement=agreement,
        hankel_singular_values=singular,
        shrunk_singular_values=shrunk,
        receipts=receipts,
    )


def _oas_covariance(values: Tensor, label: str) -> Tuple[Tensor, Dict[str, Any]]:
    """Analytic Oracle Approximating Shrinkage with a machine-only SPD floor."""
    if values.dim() != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError(f"{label} covariance needs a non-empty 2-D sample matrix")
    centered = values - values.mean(dim=0, keepdim=True)
    samples, dimension = centered.shape
    empirical = centered.T @ centered / samples
    mu = torch.trace(empirical) / dimension
    alpha = empirical.square().mean()
    numerator = alpha + mu.square()
    denominator = (samples + 1) * (alpha - mu.square() / dimension)
    if float(denominator) <= 0.0:
        shrinkage = 1.0
    else:
        shrinkage = min(1.0, float(numerator / denominator))
    covariance = (1.0 - shrinkage) * empirical + shrinkage * mu * torch.eye(
        dimension, dtype=torch.float64)
    covariance = _symmetrize(covariance)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    scale = max(1.0, float(eigenvalues.abs().max()))
    floor = dimension * torch.finfo(torch.float64).eps * scale
    added_floor = max(0.0, floor - float(eigenvalues.min()))
    if added_floor:
        covariance = covariance + added_floor * torch.eye(dimension, dtype=torch.float64)
    receipts = {
        f"{label}_samples": samples,
        f"{label}_dimension": dimension,
        f"{label}_oas_shrinkage": shrinkage,
        f"{label}_machine_spd_floor": floor,
        f"{label}_added_diagonal": added_floor,
        f"{label}_minimum_eigenvalue": float(torch.linalg.eigvalsh(covariance).min()),
    }
    return covariance, receipts


def _reconstruct_states_from_all_futures(
        outputs: Tensor, actions: Tensor, realization: HankelRealization,
        output_mean: Tensor, action_mean: Tensor) -> Tensor:
    """Infer each state from all remaining clean outputs; no future horizon is chosen."""
    episodes, length, _ = outputs.shape
    state_matrix = realization.state_matrix
    action_matrix = realization.action_matrix
    read_matrix = realization.read_matrix
    state_order = state_matrix.shape[0]
    centered_outputs = outputs - output_mean.view(1, 1, -1)
    centered_actions = actions - action_mean.view(1, 1, -1)
    reconstructed = []

    for start in range(length):
        remaining = length - start
        power = torch.eye(state_order, dtype=torch.float64)
        observability_rows = []
        input_state = torch.zeros(episodes, state_order, dtype=torch.float64)
        input_outputs = []
        for offset in range(remaining):
            observability_rows.append(read_matrix @ power)
            input_outputs.append(input_state @ read_matrix.T)
            if offset < remaining - 1:
                input_state = (
                    input_state @ state_matrix.T
                    + centered_actions[:, start + offset] @ action_matrix.T)
                power = power @ state_matrix
        observability = torch.cat(observability_rows, dim=0)
        target = centered_outputs[:, start:] - torch.stack(input_outputs, dim=1)
        state = target.reshape(episodes, -1) @ _machine_pinv(observability).T
        reconstructed.append(state)
    return torch.stack(reconstructed, dim=1)


def _fit_riccati_statistics(
        clean: Tensor, observed: Tensor, actions: Tensor,
        realization: HankelRealization, output_mean: Tensor,
        action_mean: Tensor, initial_map: Tensor, *, full_anchor: bool,
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
    """Fit paired-view Q/R and solve the corresponding steady Riccati equation."""
    try:
        from scipy.linalg import solve_discrete_are
    except ImportError as error:  # pragma: no cover - project declares scipy
        raise RuntimeError("CF-HIRO fitting requires the declared scipy dependency") from error

    centered_initial = clean[:, 0] - output_mean
    if full_anchor:
        fitted_initial_state = centered_initial.new_zeros(
            centered_initial.shape[0], realization.state_matrix.shape[0])
        complement_anchor = centered_initial
        anchor_policy = "full_initial_output_with_zero_dynamic_state"
    else:
        fitted_initial_state = centered_initial @ initial_map.T
        complement_anchor = centered_initial - fitted_initial_state @ realization.read_matrix.T
        anchor_policy = "orthogonal_output_complement_CCplus"
    # The offline smoother reconstructs only the dynamic direct-sum member. The known
    # episode anchor is subtracted exactly as it will be in every online read.
    dynamic_clean = clean - complement_anchor.unsqueeze(1)
    states = _reconstruct_states_from_all_futures(
        dynamic_clean, actions, realization, output_mean, action_mean)
    centered_actions = actions - action_mean.view(1, 1, -1)
    transition_prediction = (
        states[:, :-1] @ realization.state_matrix.T
        + centered_actions @ realization.action_matrix.T)
    process_residual = (states[:, 1:] - transition_prediction).reshape(
        -1, states.shape[-1])
    measurement_residual = (observed - clean).reshape(-1, clean.shape[-1])
    process_covariance, q_receipts = _oas_covariance(
        process_residual, "process_residual")
    measurement_covariance, r_receipts = _oas_covariance(
        measurement_residual, "paired_measurement_residual")

    state_matrix = realization.state_matrix
    read_matrix = realization.read_matrix
    prior_numpy = solve_discrete_are(
        state_matrix.numpy().T, read_matrix.numpy().T,
        process_covariance.numpy(), measurement_covariance.numpy(), balanced=True)
    prior_covariance = _symmetrize(torch.from_numpy(np.array(prior_numpy, copy=True)).double())
    innovation_covariance = (
        read_matrix @ prior_covariance @ read_matrix.T + measurement_covariance)
    gain = torch.linalg.solve(
        innovation_covariance,
        read_matrix @ prior_covariance).T
    identity = torch.eye(state_matrix.shape[0], dtype=torch.float64)
    residual_map = identity - gain @ read_matrix
    # Joseph form is used here and in every online correction.
    posterior_covariance = _symmetrize(
        residual_map @ prior_covariance @ residual_map.T
        + gain @ measurement_covariance @ gain.T)
    next_prior = (
        state_matrix @ posterior_covariance @ state_matrix.T + process_covariance)
    dare_scale = max(1.0, float(prior_covariance.norm()))
    receipts = {
        **q_receipts,
        **r_receipts,
        "clean_view_role": "process_target",
        "paired_difference_role": "measurement_corruption",
        "paired_view_independence_assumption": True,
        "state_reconstruction": "all_remaining_clean_futures",
        "state_reconstruction_selected_horizon": None,
        "state_reconstruction_anchor_policy": anchor_policy,
        "state_reconstruction_complement_rms": float(
            complement_anchor.square().mean().sqrt()),
        "state_reconstruction_initial_direct_sum_error": float(
            (output_mean + complement_anchor
             + fitted_initial_state @ realization.read_matrix.T - clean[:, 0]).abs().max()),
        "riccati_solver": "scipy_solve_discrete_are_dual",
        "offline_covariance_update": "joseph",
        "online_covariance_update": "none_fixed_steady_gain_mean_only",
        "steady_riccati_relative_residual": float(
            (next_prior - prior_covariance).norm() / dare_scale),
        "steady_prior_minimum_eigenvalue": float(
            torch.linalg.eigvalsh(prior_covariance).min()),
        "steady_posterior_minimum_eigenvalue": float(
            torch.linalg.eigvalsh(posterior_covariance).min()),
        "steady_gain_operator_norm": float(torch.linalg.matrix_norm(gain, ord=2)),
    }
    return (
        process_covariance, measurement_covariance, posterior_covariance, gain,
        {**receipts, "steady_prior_covariance": prior_covariance},
    )


def fit_cf_hiro(
        clean: Tensor, observed: Tensor, actions: Tensor, *, mode: str = "full",
        ) -> CFHIROFit:
    """Fit the complete split-fold-agreement observer in float64 on CPU.

    Even/odd moments are estimated separately, but their realization basis is learned
    from the pooled mean and the covariance fit uses all episodes. The folds estimate
    agreement; they do not constitute out-of-fold estimation.
    """
    mode = _canonical_mode(mode)
    clean = _as_cpu_double(clean, "clean", 3)
    observed = _as_cpu_double(observed, "observed", 3)
    actions = _as_cpu_double(actions, "actions", 3)
    if clean.shape != observed.shape:
        raise ValueError("clean and observed paired views must have identical shapes")
    if tuple(actions.shape[:2]) != (clean.shape[0], clean.shape[1] - 1):
        raise ValueError("actions must align with every clean/observed transition")

    markov = estimate_all_lag_markov_moments(clean, actions)
    realization = realize_fold_agreement_hankel(
        markov.even, markov.odd,
        agreement_mode="unit" if mode == "noshrink" else "empirical_bayes",
        transition_mode="triangular" if mode == "triangular" else "normal")
    output_mean = clean.mean(dim=(0, 1))
    action_mean = actions.mean(dim=(0, 1))
    initial_map = _machine_pinv(realization.read_matrix)
    process, measurement, posterior, gain, riccati_receipts = _fit_riccati_statistics(
        clean, observed, actions, realization, output_mean, action_mean, initial_map,
        full_anchor=mode == "fullanchor")
    prior = riccati_receipts.pop("steady_prior_covariance")
    receipts: Dict[str, Any] = {
        "method": "CF-HIRO-v13",
        "fit_mode": mode,
        "cf_expansion": "cross_fold_agreement",
        "fold_role": "agreement_estimation_not_out_of_fold_estimation",
        "self_supervision": "paired_views_plus_executed_randomized_actions",
        "reward_or_state_labels_used": False,
        "selected_temporal_horizons": None,
        "loss_weights": None,
        "learned_memory_parameters": 0,
        "affine_state_drift": "zero_by_global_centering",
        "unavoidable_realization_choices": [
            "left-balanced all-lag Hankel split",
            "full numerical Hankel order with machine pseudoinverse",
            "empirical-Bayes positive-part fold-agreement shrinkage",
            ("stabilized quasi-upper triangular control" if mode == "triangular" else
             "orthogonal block-diagonal real-normal contraction"),
            "OAS isotropic covariance target",
            "clean-view process target and paired-difference measurement noise",
        ],
        **dict(markov.receipts),
        **dict(realization.receipts),
        **riccati_receipts,
    }
    return CFHIROFit(
        state_matrix=realization.state_matrix,
        action_matrix=realization.action_matrix,
        read_matrix=realization.read_matrix,
        process_covariance=process,
        measurement_covariance=measurement,
        initial_covariance=posterior,
        initial_map=initial_map,
        output_mean=output_mean,
        action_mean=action_mean,
        steady_prior_covariance=prior,
        steady_gain=gain,
        markov_even=markov.even,
        markov_odd=markov.odd,
        fold_agreement=realization.fold_agreement,
        receipts=receipts,
    )


class FoldAgreementHankelRiccatiMemory(nn.Module):
    """Parameter-free direct-sum observer with a fixed steady Riccati gain."""

    MODES = frozenset(CF_HIRO_MODES | set(_MODE_ALIASES))

    def __init__(
            self, output_dim: int, action_dim: int, state_dim: int,
            mode: str = "full", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        for name, value in (
                ("output_dim", output_dim), ("action_dim", action_dim),
                ("state_dim", state_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        mode = _canonical_mode(mode)
        if not dtype.is_floating_point:
            raise ValueError("CF-HIRO buffers require a floating-point dtype")
        self.output_dim = output_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.mode = mode
        self.register_buffer("state_matrix", torch.eye(state_dim, dtype=dtype))
        self.register_buffer("action_matrix", torch.zeros(state_dim, action_dim, dtype=dtype))
        self.register_buffer("read_matrix", torch.zeros(output_dim, state_dim, dtype=dtype))
        self.register_buffer("process_covariance", torch.eye(state_dim, dtype=dtype))
        self.register_buffer("measurement_covariance", torch.eye(output_dim, dtype=dtype))
        self.register_buffer("initial_covariance", torch.eye(state_dim, dtype=dtype))
        self.register_buffer("steady_prior_covariance", torch.eye(state_dim, dtype=dtype))
        self.register_buffer("steady_gain", torch.zeros(state_dim, output_dim, dtype=dtype))
        self.register_buffer("initial_map", torch.zeros(state_dim, output_dim, dtype=dtype))
        self.register_buffer("output_projector", torch.zeros(output_dim, output_dim, dtype=dtype))
        self.register_buffer("output_mean", torch.zeros(output_dim, dtype=dtype))
        self.register_buffer("action_mean", torch.zeros(action_dim, dtype=dtype))
        self.register_buffer("fit_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("operators_installed", torch.zeros((), dtype=torch.bool))
        self._fit_receipts: Dict[str, Any] = {}

    @classmethod
    def from_fit(
            cls, fit: CFHIROFit, mode: str = "full",
            dtype: torch.dtype | None = None) -> "FoldAgreementHankelRiccatiMemory":
        if not isinstance(fit, CFHIROFit):
            raise TypeError("fit must be a CFHIROFit")
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
            raise RuntimeError("invalid CF-HIRO extra state")
        self._fit_receipts = copy.deepcopy(dict(state.get("fit_receipts", {})))

    @staticmethod
    def _validate_psd(value: Tensor, name: str, *, positive: bool = False) -> None:
        work = _symmetrize(value.detach().double())
        tolerance = max(value.shape) * torch.finfo(value.dtype).eps * max(
            1.0, float(work.abs().max()))
        symmetry = float((value.detach().double() - value.detach().double().T).abs().max())
        minimum = float(torch.linalg.eigvalsh(work).min())
        if symmetry > tolerance or minimum < (-tolerance if not positive else tolerance):
            qualifier = "positive definite" if positive else "positive semidefinite"
            raise ValueError(f"{name} must be symmetric {qualifier}")

    def _validate_fit_mode(self, fit: CFHIROFit) -> None:
        agreement = fit.receipts.get("fold_agreement_mode")
        deployment = fit.receipts.get("transition_deployment")
        expected_agreement = "unit" if self.mode == "noshrink" else "empirical_bayes"
        expected_deployment = "triangular" if self.mode == "triangular" else "normal"
        if agreement != expected_agreement:
            raise ValueError(
                f"{self.mode} requires fold_agreement_mode={expected_agreement!r}, "
                f"got {agreement!r}")
        if deployment != expected_deployment:
            raise ValueError(
                f"{self.mode} requires transition_deployment={expected_deployment!r}, "
                f"got {deployment!r}")

    @torch.no_grad()
    def install_fit(self, fit: CFHIROFit) -> None:
        if not isinstance(fit, CFHIROFit):
            raise TypeError("fit must be a CFHIROFit")
        self._validate_fit_mode(fit)
        expected = {
            "state_matrix": (self.state_dim, self.state_dim),
            "action_matrix": (self.state_dim, self.action_dim),
            "read_matrix": (self.output_dim, self.state_dim),
            "process_covariance": (self.state_dim, self.state_dim),
            "measurement_covariance": (self.output_dim, self.output_dim),
            "initial_covariance": (self.state_dim, self.state_dim),
            "steady_prior_covariance": (self.state_dim, self.state_dim),
            "steady_gain": (self.state_dim, self.output_dim),
            "initial_map": (self.state_dim, self.output_dim),
            "output_mean": (self.output_dim,),
            "action_mean": (self.action_dim,),
        }
        candidates: Dict[str, Tensor] = {}
        for name, shape in expected.items():
            value = getattr(fit, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite floating-point values")
            candidates[name] = value.detach()
        self._validate_psd(candidates["process_covariance"], "process_covariance")
        self._validate_psd(
            candidates["measurement_covariance"], "measurement_covariance", positive=True)
        self._validate_psd(candidates["initial_covariance"], "initial_covariance")
        self._validate_psd(
            candidates["steady_prior_covariance"], "steady_prior_covariance")

        state = candidates["state_matrix"].double()
        radius = float(torch.linalg.eigvals(state).abs().max())
        if radius > FLOAT32_STABILITY_BOUNDARY + 2e-7:
            raise ValueError("state_matrix exceeds the declared deployment stability boundary")
        state_scale = max(float(state.norm().square()), torch.finfo(torch.float64).tiny)
        relative_normality = float((state.T @ state - state @ state.T).norm()) / state_scale
        normality_tolerance = 2e-5 if candidates["state_matrix"].dtype == torch.float32 else 1e-10
        if self.mode != "triangular" and relative_normality > normality_tolerance:
            raise ValueError("state_matrix must be a real-normal contraction")
        if self.mode == "triangular":
            lower_error = float(torch.tril(state, diagonal=-2).abs().max()) \
                if self.state_dim > 2 else 0.0
            tolerance = self.state_dim * torch.finfo(torch.float64).eps * max(
                1.0, float(state.abs().max())) * 8.0
            if lower_error > tolerance:
                raise ValueError("triangular state_matrix must be quasi-upper triangular")

        projector = candidates["read_matrix"] @ candidates["initial_map"]
        if not torch.isfinite(projector).all():
            raise ValueError("output projector is non-finite")
        converted = {
            **candidates,
            "output_projector": projector,
        }
        # All validation precedes mutation, so a rejected fit is atomic.
        for name, value in converted.items():
            destination = getattr(self, name)
            destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
        self.fit_updates.add_(1)
        self.operators_installed.fill_(True)
        self._fit_receipts = copy.deepcopy(dict(fit.receipts))

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def _validate_state(self, state: HIROState) -> int:
        if not isinstance(state, HIROState):
            raise ValueError("state must be an HIROState(mean,complement)")
        if (state.mean.dim() != 2 or state.mean.shape[0] < 1
                or state.mean.shape[1] != self.state_dim):
            raise ValueError(f"state mean must have shape (B,{self.state_dim})")
        if tuple(state.complement.shape) != (state.mean.shape[0], self.output_dim):
            raise ValueError(f"state complement must have shape (B,{self.output_dim})")
        if (not state.mean.is_floating_point() or not state.complement.is_floating_point()
                or not torch.isfinite(state.mean).all()
                or not torch.isfinite(state.complement).all()):
            raise ValueError("state must contain finite floating-point values")
        return state.mean.shape[0]

    def _validate_action(self, action: Tensor, batch: int) -> Tensor:
        if (not isinstance(action, torch.Tensor)
                or tuple(action.shape) != (batch, self.action_dim)
                or not action.is_floating_point() or not torch.isfinite(action).all()):
            raise ValueError(f"action must be finite with shape ({batch},{self.action_dim})")
        return action

    def initial_state(self, observation: Tensor) -> HIROState:
        if (not isinstance(observation, torch.Tensor) or observation.dim() != 2
                or tuple(observation.shape[1:]) != (self.output_dim,)
                or observation.shape[0] < 1 or not observation.is_floating_point()
                or not torch.isfinite(observation).all()):
            raise ValueError(f"observation must be finite with shape (B,{self.output_dim})")
        centered = observation - self.output_mean.to(observation)
        if self.mode == "fullanchor":
            mean = centered.new_zeros(centered.shape[0], self.state_dim)
            complement = centered
        else:
            mean = F.linear(centered, self.initial_map.to(centered))
            # In column notation this is c_perp=(I-C C+)(z0-mu). Reusing the
            # deployed Cx association also minimizes direct-sum roundoff at t=0.
            complement = centered - F.linear(mean, self.read_matrix.to(mean))
        return HIROState(mean, complement)

    def read_state(self, state: HIROState) -> Tensor:
        if (not isinstance(state, HIROState)
                or state.mean.shape[-1] != self.state_dim
                or state.complement.shape[:-1] != state.mean.shape[:-1]
                or state.complement.shape[-1] != self.output_dim):
            raise ValueError("state has an invalid direct-sum shape")
        if not torch.isfinite(state.mean).all() or not torch.isfinite(state.complement).all():
            raise ValueError("state must be finite")
        return (self.output_mean.to(state.mean) + state.complement.to(state.mean)
                + F.linear(state.mean, self.read_matrix.to(state.mean)))

    def transition(
            self, state: HIROState, action: Tensor,
            return_details: bool = False):
        batch = self._validate_state(state)
        action = self._validate_action(action, batch).to(state.mean)
        if self.mode == "noaction":
            effective_action = torch.zeros_like(action)
            action_effect = torch.zeros_like(state.mean)
        else:
            effective_action = action - self.action_mean.to(action)
            action_effect = F.linear(effective_action, self.action_matrix.to(action))
        prior_mean = F.linear(state.mean, self.state_matrix.to(state.mean)) + action_effect
        prior = HIROState(prior_mean, state.complement)
        if not return_details:
            return prior
        return prior, {
            "effective_action": effective_action,
            "action_effect": action_effect,
            "prior_read": self.read_state(prior),
            "complement_anchor": prior.complement,
        }

    def correct(
            self, prior: HIROState, observation: Tensor,
            return_details: bool = False):
        batch = self._validate_state(prior)
        if (not isinstance(observation, torch.Tensor)
                or tuple(observation.shape) != (batch, self.output_dim)
                or not observation.is_floating_point() or not torch.isfinite(observation).all()):
            raise ValueError(f"observation must be finite with shape ({batch},{self.output_dim})")
        observation = observation.to(prior.mean)
        innovation = observation - self.read_state(prior)
        if self.mode == "nocorrect":
            effective_gain = torch.zeros_like(self.steady_gain).to(prior.mean)
            correction = torch.zeros_like(prior.mean)
        else:
            effective_gain = self.steady_gain.to(prior.mean)
            correction = F.linear(innovation, effective_gain)
        posterior = HIROState(prior.mean + correction, prior.complement)
        expanded_gain = effective_gain.unsqueeze(0).expand(batch, -1, -1)
        if not return_details:
            return posterior
        return posterior, {
            "innovation": innovation,
            "gain": expanded_gain,
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
                or actions.shape[0] != batch or actions.shape[2] != self.action_dim
                or not actions.is_floating_point() or not torch.isfinite(actions).all()):
            raise ValueError(f"actions must have shape (B,H,{self.action_dim})")
        actions = actions.to(state.mean)
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

    def forward(
            self, observations: Tensor, actions: Tensor,
            return_details: bool = False):
        if (not isinstance(observations, torch.Tensor) or observations.dim() != 3
                or observations.shape[0] < 1 or observations.shape[1] < 1
                or observations.shape[2] != self.output_dim
                or not observations.is_floating_point()
                or not torch.isfinite(observations).all()):
            raise ValueError(f"observations must have shape (B,T,{self.output_dim})")
        expected_actions = (
            observations.shape[0], observations.shape[1] - 1, self.action_dim)
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
        corrections = [torch.zeros(
            observations.shape[0], self.state_dim,
            device=observations.device, dtype=observations.dtype)]
        action_effects = [torch.zeros(
            observations.shape[0], self.state_dim,
            device=observations.device, dtype=observations.dtype)]
        gains = [torch.zeros(
            observations.shape[0], self.state_dim, self.output_dim,
            device=observations.device, dtype=observations.dtype)]
        for index in range(1, observations.shape[1]):
            read, state, details = self.step(
                state, observations[:, index], actions[:, index - 1],
                return_details=True)
            posterior_reads.append(read)
            prior_reads.append(details["prior_read"])
            means.append(state.mean)
            complements.append(state.complement)
            innovations.append(details["innovation"])
            corrections.append(details["correction"])
            action_effects.append(details["action_effect"])
            gains.append(details["gain"])
        posterior_sequence = torch.stack(posterior_reads, dim=1)
        if not return_details:
            return posterior_sequence
        return posterior_sequence, {
            "reads": posterior_sequence,
            "posterior_reads": posterior_sequence,
            "prior_reads": torch.stack(prior_reads, dim=1),
            "state_means": torch.stack(means, dim=1),
            "complement_anchors": torch.stack(complements, dim=1),
            "innovations": torch.stack(innovations, dim=1),
            "corrections": torch.stack(corrections, dim=1),
            "action_effects": torch.stack(action_effects, dim=1),
            "gains": torch.stack(gains, dim=1),
        }

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, Any]:
        state = self.state_matrix.double()
        state_scale = max(float(state.norm().square()), torch.finfo(torch.float64).tiny)
        relative_normality = float((state.T @ state - state @ state.T).norm()) / state_scale
        diagnostics = {
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
            "state_relative_normality_residual": relative_normality,
            "state_is_real_normal_contraction": bool(
                self.mode != "triangular" and relative_normality <= 2e-5
                and float(torch.linalg.matrix_norm(state, ord=2))
                <= FLOAT32_STABILITY_BOUNDARY + 2e-6),
            "action_matrix_norm": float(self.action_matrix.double().norm()),
            "effective_action_matrix_norm": (
                0.0 if self.mode == "noaction" else float(self.action_matrix.double().norm())),
            "steady_gain_norm": float(self.steady_gain.double().norm()),
            "effective_steady_gain_norm": (
                0.0 if self.mode == "nocorrect" else float(self.steady_gain.double().norm())),
            "process_minimum_eigenvalue": float(
                torch.linalg.eigvalsh(self.process_covariance.double()).min()),
            "measurement_minimum_eigenvalue": float(
                torch.linalg.eigvalsh(self.measurement_covariance.double()).min()),
            "online_covariance_update": "none_fixed_steady_gain_mean_only",
            "offline_covariance_update": "joseph_fixed_point_receipt",
            "direct_sum_initialization": "Cplus_and_immutable_output_complement",
            "noaction_exact": self.mode == "noaction",
            "nocorrect_exact": self.mode == "nocorrect",
            "fullanchor_exact": self.mode == "fullanchor",
        }
        diagnostics.update(copy.deepcopy(self._fit_receipts))
        return diagnostics


CFHIROv13Memory = FoldAgreementHankelRiccatiMemory
