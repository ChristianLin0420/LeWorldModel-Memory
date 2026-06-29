"""Cross-fold-agreement Hankel--Riccati observer (CF-HIRO) prototype.

This is an isolated numerical prototype for a self-supervised predictive-state
memory.  It deliberately has no dependency on the V12 implementation or trainer.
Given paired clean/observed embedding trajectories and randomized executed actions,
the fitter performs four detached operations:

1. estimate every available action-to-future cross moment separately on even and
   odd episode folds;
2. realize the resulting all-lag block Hankel operator, continuously shrinking each
   shared singular direction by its fold agreement;
3. deploy the recurrence in real-Schur coordinates; and
4. estimate process/measurement covariance from the paired views and initialize a
   Riccati observer whose online covariance correction is in Joseph form.

There are no selected temporal horizons, loss weights, learned memory parameters,
or manually selected state rank.  This does *not* make realization choice-free.
The following choices are structural and are reported in every fit receipt:

* the all-lag Hankel is split into the unique left-balanced integer partition;
* every numerical singular direction is retained (machine-precision pseudoinverse
  totalizes exact null directions rather than pretending they are identifiable);
* fold agreement uses the equal-fold signal/disagreement variance decomposition;
* unstable Schur diagonal blocks are radially projected to the float32 roundoff
  stability boundary;
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
    """Continuously cross-fold-shrunk realization in real-Schur coordinates."""

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
    """Online posterior mean and covariance in the fitted Schur coordinates."""

    mean: Tensor
    covariance: Tensor


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


def _real_schur_stabilize(value: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    """Return a stable real-Schur matrix and its orthogonal coordinate transform."""
    try:
        from scipy.linalg import schur
    except ImportError as error:  # pragma: no cover - project declares scipy
        raise RuntimeError("CF-HIRO fitting requires the declared scipy dependency") from error

    source = value.detach().cpu().double().numpy()
    schur_matrix, basis = schur(source, output="real")
    before = float(np.max(np.abs(np.linalg.eigvals(schur_matrix))))
    scale = max(1.0, float(np.max(np.abs(schur_matrix))))
    tolerance = value.shape[0] * np.finfo(np.float64).eps * scale
    block_scales = []
    index = 0
    while index < schur_matrix.shape[0]:
        width = (2 if index + 1 < schur_matrix.shape[0]
                 and abs(schur_matrix[index + 1, index]) > tolerance else 1)
        block = schur_matrix[index:index + width, index:index + width]
        radius = float(np.max(np.abs(np.linalg.eigvals(block))))
        factor = (min(1.0, FLOAT32_STABILITY_BOUNDARY / radius)
                  if radius > 0.0 else 1.0)
        schur_matrix[index:index + width, index:index + width] *= factor
        block_scales.append(factor)
        index += width
    after = float(np.max(np.abs(np.linalg.eigvals(schur_matrix))))
    receipts = {
        "coordinate_system": "real_schur",
        "schur_block_detection_tolerance": tolerance,
        "float32_stability_boundary": FLOAT32_STABILITY_BOUNDARY,
        "spectral_radius_before_projection": before,
        "spectral_radius_after_projection": after,
        "schur_block_radial_scales": block_scales,
        "stability_projection_active": any(factor < 1.0 for factor in block_scales),
    }
    return (
        torch.from_numpy(np.array(schur_matrix, copy=True)).double(),
        torch.from_numpy(np.array(basis, copy=True)).double(),
        receipts,
    )


def realize_fold_agreement_hankel(
        even_markov: Tensor, odd_markov: Tensor) -> HankelRealization:
    """Realize all lags with continuous shared-basis fold-agreement shrinkage.

    For shared singular direction ``i``, let ``m_i`` be the average fold projection
    and ``d_i`` half their difference.  Its reliability is
    ``m_i^2 / (m_i^2 + d_i^2 + machine_floor)``.  No direction is selected by a
    model-order threshold; the state order is the full rectangular Hankel order.
    """
    even_markov = _as_cpu_double(even_markov, "even_markov", 3)
    odd_markov = _as_cpu_double(odd_markov, "odd_markov", 3)
    if even_markov.shape != odd_markov.shape:
        raise ValueError("even and odd Markov tensors must have identical shapes")
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
    agreement = signal.square() / (
        signal.square() + disagreement.square() + machine_floor)
    shrunk = singular * agreement

    square_root = torch.sqrt(shrunk.clamp_min(0.0))
    observability = left * square_root.unsqueeze(0)
    reachability = square_root.unsqueeze(1) * right_h
    state_raw = (
        _machine_pinv(observability) @ h1 @ _machine_pinv(reachability))
    action_raw = reachability[:, :action_dim]
    read_raw = observability[:output_dim]

    state_schur, schur_basis, schur_receipts = _real_schur_stabilize(state_raw)
    action_schur = schur_basis.T @ action_raw
    read_schur = read_raw @ schur_basis
    state_order = state_schur.shape[0]
    receipts: Dict[str, Any] = {
        "realization": "cross_fitted_all_lag_block_hankel",
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
        "fold_shrinkage_rule": "m^2/(m^2+d^2+machine_floor)",
        "fold_agreement_min": float(agreement.min()),
        "fold_agreement_mean": float(agreement.mean()),
        "fold_agreement_max": float(agreement.max()),
        "hankel_rank_machine": _matrix_rank(h0),
        **schur_receipts,
    }
    return HankelRealization(
        state_matrix=state_schur,
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
        action_mean: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
    """Fit paired-view Q/R and solve the corresponding steady Riccati equation."""
    try:
        from scipy.linalg import solve_discrete_are
    except ImportError as error:  # pragma: no cover - project declares scipy
        raise RuntimeError("CF-HIRO fitting requires the declared scipy dependency") from error

    states = _reconstruct_states_from_all_futures(
        clean, actions, realization, output_mean, action_mean)
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
        "riccati_solver": "scipy_solve_discrete_are_dual",
        "covariance_update": "joseph",
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


def fit_cf_hiro(clean: Tensor, observed: Tensor, actions: Tensor) -> CFHIROFit:
    """Fit the complete split-sample fold-agreement observer in float64 on CPU.

    Even/odd moments are estimated separately, but their realization basis is learned
    from the pooled mean and the covariance fit uses all episodes. This is therefore
    a stability-shrinkage diagnostic, not statistically cross-fitted estimation.
    """
    clean = _as_cpu_double(clean, "clean", 3)
    observed = _as_cpu_double(observed, "observed", 3)
    actions = _as_cpu_double(actions, "actions", 3)
    if clean.shape != observed.shape:
        raise ValueError("clean and observed paired views must have identical shapes")
    if tuple(actions.shape[:2]) != (clean.shape[0], clean.shape[1] - 1):
        raise ValueError("actions must align with every clean/observed transition")

    markov = estimate_all_lag_markov_moments(clean, actions)
    realization = realize_fold_agreement_hankel(markov.even, markov.odd)
    output_mean = clean.mean(dim=(0, 1))
    action_mean = actions.mean(dim=(0, 1))
    process, measurement, posterior, gain, riccati_receipts = _fit_riccati_statistics(
        clean, observed, actions, realization, output_mean, action_mean)
    prior = riccati_receipts.pop("steady_prior_covariance")
    initial_map = _machine_pinv(realization.read_matrix)
    receipts: Dict[str, Any] = {
        "method": "CF-HIRO-v13-prototype",
        "cf_expansion": "cross_fold_agreement_not_cross_fitted",
        "self_supervision": "paired_views_plus_executed_randomized_actions",
        "reward_or_state_labels_used": False,
        "selected_temporal_horizons": None,
        "loss_weights": None,
        "learned_memory_parameters": 0,
        "affine_state_drift": "zero_by_global_centering",
        "unavoidable_realization_choices": [
            "left-balanced all-lag Hankel split",
            "full numerical Hankel order with machine pseudoinverse",
            "equal-fold signal/disagreement shrinkage",
            "float32-roundoff Schur stability projection",
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
    """Parameter-free streaming action recurrence with Joseph Riccati correction."""

    MODES = frozenset({"full", "noaction"})

    def __init__(
            self, output_dim: int, action_dim: int, state_dim: int,
            mode: str = "full", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        for name, value in (
                ("output_dim", output_dim), ("action_dim", action_dim),
                ("state_dim", state_dim)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if mode not in self.MODES:
            raise ValueError(f"unknown CF-HIRO mode {mode!r}")
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
        self.register_buffer("initial_map", torch.zeros(state_dim, output_dim, dtype=dtype))
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

    @torch.no_grad()
    def install_fit(self, fit: CFHIROFit) -> None:
        if not isinstance(fit, CFHIROFit):
            raise TypeError("fit must be a CFHIROFit")
        expected = {
            "state_matrix": (self.state_dim, self.state_dim),
            "action_matrix": (self.state_dim, self.action_dim),
            "read_matrix": (self.output_dim, self.state_dim),
            "process_covariance": (self.state_dim, self.state_dim),
            "measurement_covariance": (self.output_dim, self.output_dim),
            "initial_covariance": (self.state_dim, self.state_dim),
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
        state = candidates["state_matrix"].double()
        lower_error = float(torch.tril(state, diagonal=-2).abs().max()) \
            if self.state_dim > 2 else 0.0
        schur_tolerance = self.state_dim * torch.finfo(torch.float64).eps * max(
            1.0, float(state.abs().max())) * 8.0
        if lower_error > schur_tolerance:
            raise ValueError("state_matrix must be in real-Schur quasi-upper-triangular form")
        radius = float(torch.linalg.eigvals(state).abs().max())
        if radius > FLOAT32_STABILITY_BOUNDARY + 2e-7:
            raise ValueError("state_matrix exceeds the declared deployment stability boundary")
        for name, value in candidates.items():
            destination = getattr(self, name)
            destination.copy_(value.to(device=destination.device, dtype=destination.dtype))
        self.fit_updates.add_(1)
        self.operators_installed.fill_(True)
        self._fit_receipts = copy.deepcopy(dict(fit.receipts))

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def _validate_state(self, state: HIROState) -> int:
        if not isinstance(state, HIROState):
            raise ValueError("state must be an HIROState(mean,covariance)")
        if (state.mean.dim() != 2 or state.mean.shape[0] < 1
                or state.mean.shape[1] != self.state_dim):
            raise ValueError(f"state mean must have shape (B,{self.state_dim})")
        expected_covariance = (state.mean.shape[0], self.state_dim, self.state_dim)
        if tuple(state.covariance.shape) != expected_covariance:
            raise ValueError(f"state covariance must have shape {expected_covariance}")
        if (not state.mean.is_floating_point() or not state.covariance.is_floating_point()
                or not torch.isfinite(state.mean).all()
                or not torch.isfinite(state.covariance).all()):
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
        mean = F.linear(
            observation - self.output_mean.to(observation),
            self.initial_map.to(observation))
        covariance = self.initial_covariance.to(observation).expand(
            observation.shape[0], -1, -1).clone()
        return HIROState(mean, covariance)

    def read_state(self, state: HIROState) -> Tensor:
        if not isinstance(state, HIROState) or state.mean.shape[-1] != self.state_dim:
            raise ValueError("state has an invalid mean shape")
        if not torch.isfinite(state.mean).all():
            raise ValueError("state mean must be finite")
        return F.linear(state.mean, self.read_matrix.to(state.mean)) + self.output_mean.to(state.mean)

    def transition(
            self, state: HIROState, action: Tensor,
            return_details: bool = False):
        batch = self._validate_state(state)
        action = self._validate_action(action, batch).to(state.mean)
        state_matrix = self.state_matrix.to(state.mean)
        if self.mode == "noaction":
            effective_action = torch.zeros_like(action)
            action_effect = torch.zeros_like(state.mean)
        else:
            effective_action = action - self.action_mean.to(action)
            action_effect = F.linear(effective_action, self.action_matrix.to(action))
        prior_mean = F.linear(state.mean, state_matrix) + action_effect
        process = self.process_covariance.to(state.covariance)
        prior_covariance = _symmetrize(
            state_matrix @ state.covariance @ state_matrix.T + process)
        prior = HIROState(prior_mean, prior_covariance)
        if not return_details:
            return prior
        return prior, {
            "effective_action": effective_action,
            "action_effect": action_effect,
            "prior_read": self.read_state(prior),
            "prior_covariance": prior_covariance,
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
        read_matrix = self.read_matrix.to(prior.mean)
        measurement = self.measurement_covariance.to(prior.covariance)
        innovation = observation - self.read_state(prior)
        read_times_prior = read_matrix @ prior.covariance
        innovation_covariance = (
            read_times_prior @ read_matrix.T + measurement)
        gain = torch.linalg.solve(
            innovation_covariance, read_times_prior).transpose(-2, -1)
        correction = torch.einsum("bij,bj->bi", gain, innovation)
        posterior_mean = prior.mean + correction
        identity = torch.eye(
            self.state_dim, device=prior.mean.device, dtype=prior.mean.dtype)
        residual_map = identity - gain @ read_matrix
        # Joseph form, not the cancellation-prone (I-KC)P shortcut.
        posterior_covariance = _symmetrize(
            residual_map @ prior.covariance @ residual_map.transpose(-2, -1)
            + gain @ measurement @ gain.transpose(-2, -1))
        posterior = HIROState(posterior_mean, posterior_covariance)
        if not return_details:
            return posterior
        return posterior, {
            "innovation": innovation,
            "innovation_covariance": innovation_covariance,
            "gain": gain,
            "correction": correction,
            "posterior_covariance": posterior_covariance,
            "posterior_read": self.read_state(posterior),
        }

    def step(
            self, state: HIROState, observation: Tensor, action: Tensor,
            return_details: bool = False):
        prior, transition_details = self.transition(state, action, return_details=True)
        posterior, correction_details = self.correct(
            prior, observation, return_details=True)
        read = correction_details["posterior_read"]
        if not return_details:
            return read, posterior
        return read, posterior, {
            **transition_details, **correction_details,
            "prior_state": prior.mean,
            "posterior_state": posterior.mean,
        }

    def rollout_transition(self, state: HIROState, actions: Tensor) -> HIROState:
        batch = self._validate_state(state)
        if (not isinstance(actions, torch.Tensor) or actions.dim() != 3
                or actions.shape[0] != batch or actions.shape[2] != self.action_dim
                or not actions.is_floating_point() or not torch.isfinite(actions).all()):
            raise ValueError(f"actions must have shape (B,H,{self.action_dim})")
        actions = actions.to(state.mean)
        means, covariances = [], []
        for index in range(actions.shape[1]):
            state = self.transition(state, actions[:, index])
            means.append(state.mean)
            covariances.append(state.covariance)
        if not means:
            return HIROState(
                state.mean.new_empty(batch, 0, self.state_dim),
                state.covariance.new_empty(batch, 0, self.state_dim, self.state_dim))
        return HIROState(torch.stack(means, dim=1), torch.stack(covariances, dim=1))

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
        reads = [self.read_state(state)]
        means = [state.mean]
        covariances = [state.covariance]
        innovations = [torch.zeros_like(observations[:, 0])]
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
            reads.append(read)
            means.append(state.mean)
            covariances.append(state.covariance)
            innovations.append(details["innovation"])
            action_effects.append(details["action_effect"])
            gains.append(details["gain"])
        read_sequence = torch.stack(reads, dim=1)
        if not return_details:
            return read_sequence
        return read_sequence, {
            "reads": read_sequence,
            "state_means": torch.stack(means, dim=1),
            "state_covariances": torch.stack(covariances, dim=1),
            "innovations": torch.stack(innovations, dim=1),
            "action_effects": torch.stack(action_effects, dim=1),
            "gains": torch.stack(gains, dim=1),
        }

    @torch.no_grad()
    def diagnostics(self) -> Dict[str, Any]:
        state = self.state_matrix.double()
        diagnostics = {
            "mode": self.mode,
            "operators_installed": bool(self.operators_installed),
            "fit_updates": int(self.fit_updates),
            "gradient_parameter_count": self.parameter_count(),
            "state_dim": self.state_dim,
            "streaming_mean_floats": self.state_dim,
            "streaming_covariance_floats": self.state_dim * self.state_dim,
            "state_spectral_radius": float(torch.linalg.eigvals(state).abs().max()),
            "state_is_real_schur": bool(float(torch.tril(state, diagonal=-2).abs().max())
                                        <= self.state_dim * torch.finfo(torch.float64).eps
                                        * max(1.0, float(state.abs().max())) * 8.0),
            "action_matrix_norm": float(self.action_matrix.double().norm()),
            "effective_action_matrix_norm": (
                0.0 if self.mode == "noaction" else float(self.action_matrix.double().norm())),
            "process_minimum_eigenvalue": float(
                torch.linalg.eigvalsh(self.process_covariance.double()).min()),
            "measurement_minimum_eigenvalue": float(
                torch.linalg.eigvalsh(self.measurement_covariance.double()).min()),
            "online_covariance_update": "joseph",
            "noaction_exact": self.mode == "noaction",
        }
        diagnostics.update(copy.deepcopy(self._fit_receipts))
        return diagnostics


CFHIROv13Memory = FoldAgreementHankelRiccatiMemory
