#!/usr/bin/env python3
"""Focused numerical tests for SIRO-v12's detached operator fit."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_siro_v12 import (
    FLOAT32_STABILITY_CAP,
    _machine_pinv,
    add_refit_drift_receipts,
    fit_actual_prior_innovation,
    fit_linear_dynamics,
    fit_reachability,
    fit_siro_from_embeddings,
    oas_from_values,
)


def synthetic_system(*, episodes=48, length=13, dimension=6, action_dim=2,
                     spectral_max=.82, noise=0.0, anchor_centered=True):
    generator = torch.Generator().manual_seed(12_012)
    matrix = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    left, _, right = torch.linalg.svd(matrix)
    singular = torch.linspace(.25, spectral_max, dimension, dtype=torch.float64)
    A = (left * singular.unsqueeze(0)) @ right
    B = .2 * torch.randn(
        dimension, action_dim, generator=generator, dtype=torch.float64)
    b = .01 * torch.randn(dimension, generator=generator, dtype=torch.float64)
    actions = torch.randn(
        episodes, length - 1, action_dim, generator=generator, dtype=torch.float64)
    anchors = torch.randn(
        episodes, dimension, generator=generator, dtype=torch.float64)
    coordinate = torch.zeros(episodes, length, dimension, dtype=torch.float64)
    if not anchor_centered:
        coordinate[:, 0] = anchors
    for step in range(length - 1):
        coordinate[:, step + 1] = (
            coordinate[:, step] @ A.T + actions[:, step] @ B.T + b)
        if noise:
            coordinate[:, step + 1].add_(
                noise * torch.randn(
                    episodes, dimension, generator=generator, dtype=torch.float64))
    state = (coordinate + anchors.unsqueeze(1)
             if anchor_centered else coordinate)
    return state, actions, A, B, b


def test_machine_pinv_totalizes_rank_deficiency_without_ridge() -> None:
    value = torch.diag(torch.tensor([3.0, 0.0, 1.0], dtype=torch.float64))
    inverse = _machine_pinv(value, hermitian=True)
    assert torch.equal(inverse, torch.diag(torch.tensor(
        [1 / 3, 0.0, 1.0], dtype=torch.float64)))
    assert torch.allclose(value @ inverse @ value, value, atol=1e-15, rtol=1e-15)


def test_fwl_recovers_stable_dynamics_and_native_action_coordinates() -> None:
    state, actions, A, B, b = synthetic_system()
    fit = fit_linear_dynamics(state, actions, identity_A=False)
    assert torch.allclose(fit.identified_A, A, atol=2e-12, rtol=2e-12)
    assert torch.allclose(fit.action_B, B, atol=2e-12, rtol=2e-12)
    assert torch.allclose(fit.drift_b, b, atol=2e-12, rtol=2e-12)
    assert fit.receipts["identified_A_singular_max"] <= FLOAT32_STABILITY_CAP + 1e-12
    assert fit.receipts["action_B_rank"] == actions.shape[-1]
    assert fit.receipts["action_B0_rank"] == actions.shape[-1]
    assert fit.receipts["action_B1_rank"] == actions.shape[-1]
    assert fit.receipts["final_action_residual_cross_relative"] < 1e-12
    assert fit.receipts["final_intercept_residual_cross_relative"] < 1e-12
    assert fit.receipts["anchor_centered_fit"]
    assert fit.receipts["centered_x0_max_abs"] == 0.0


def test_singular_cap_and_identity_control_refit() -> None:
    state, actions, _, _, _ = synthetic_system(spectral_max=1.25)
    capped = fit_linear_dynamics(state, actions, identity_A=False)
    identity = fit_linear_dynamics(state, actions, identity_A=True)
    assert abs(capped.receipts["identified_A_singular_max"]
               - FLOAT32_STABILITY_CAP) <= 2e-12
    assert torch.equal(
        identity.identified_A,
        torch.eye(state.shape[-1], dtype=torch.float64))
    coordinate = state - state[:, :1]
    source, target = coordinate[:, :-1], coordinate[:, 1:]
    prediction = (
        source @ identity.identified_A.T
        + actions @ identity.action_B.T + identity.drift_b)
    residual = (target - prediction).reshape(-1, state.shape[-1])
    standardized = (
        actions.reshape(-1, actions.shape[-1]) - identity.action_mean
    ) / identity.action_std
    scale = standardized.norm() * target.norm()
    assert float((standardized.T @ residual).norm() / scale) < 1e-12


def test_oas_is_data_derived_positive_and_finite() -> None:
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(400, 5, generator=generator, dtype=torch.float64)
    values[:, 4] = values[:, 0] + 1e-8 * values[:, 4]
    fit = oas_from_values(values, label="test")
    assert 0.0 <= fit.shrinkage <= 1.0
    assert torch.linalg.eigvalsh(fit.covariance).min() > 0.0
    assert math.isfinite(fit.condition)


def test_reachability_uses_every_transition_lag_and_frozen_survival() -> None:
    state, actions, _, _, _ = synthetic_system(length=9, noise=1e-4)
    linear = fit_linear_dynamics(state, actions, identity_A=False)
    reachability = fit_reachability(linear, state.shape[1])
    assert reachability.receipts["reachability_lags"] == 8
    assert reachability.receipts["survival_weight_first"] == 1.0
    assert reachability.receipts["survival_weight_last"] == 1 / 8
    assert len(reachability.receipts["age_kappa"]) == state.shape[-1]
    assert len(reachability.receipts["age_tau"]) == state.shape[-1]
    assert all(value >= 0.0 for value in reachability.receipts["age_tau"])
    assert torch.isfinite(reachability.action_read_R).all()
    expected = reachability.signal_S @ _machine_pinv(
        reachability.signal_S + reachability.noise_N, hermitian=True)
    assert torch.allclose(
        reachability.action_read_R, expected, atol=2e-10, rtol=2e-10)


def test_actual_prior_alignment_and_action_index() -> None:
    clean, actions, _, _, _ = synthetic_system(noise=2e-4)
    observed = clean.clone()
    observed[:, 3:7].add_(.04)
    linear = fit_linear_dynamics(clean, actions, identity_A=False)
    reachability = fit_reachability(linear, clean.shape[1])
    innovation = fit_actual_prior_innovation(
        clean, observed, actions,
        linear.identified_A, linear.action_B, linear.drift_b,
        torch.eye(clean.shape[-1], dtype=torch.float64),
        anchor_centered=True)
    assert innovation.receipts["innovation_samples"] == (
        clean.shape[0] * (clean.shape[1] - 1))
    assert torch.isfinite(innovation.lmmse_K).all()
    assert innovation.receipts["lmmse_K_rank"] == clean.shape[-1]
    fit = fit_siro_from_embeddings(
        clean, observed, actions, "sirov12_noaction", fit_index=0)
    assert fit.receipts["effective_action_zero"]
    assert fit.receipts["fit_finite"]
    # Shifting the action index must measurably worsen the fitted clean transition.
    coordinate = clean - clean[:, :1]
    source, target = coordinate[:, :-1], coordinate[:, 1:]
    factual = (
        source @ linear.identified_A.T
        + actions @ linear.action_B.T + linear.drift_b)
    shifted = (
        source @ linear.identified_A.T
        + torch.roll(actions, 1, dims=1) @ linear.action_B.T + linear.drift_b)
    assert (factual - target).square().mean() < (shifted - target).square().mean()


def test_refit_drift_and_parity_receipts_are_explicit() -> None:
    clean, actions, _, _, _ = synthetic_system(noise=1e-4)
    observed = clean.clone()
    observed[:, 1:] += .01 * torch.sin(clean[:, 1:])
    first = fit_siro_from_embeddings(
        clean, observed, actions, "sirov12", fit_index=0)
    baseline = add_refit_drift_receipts(
        first, None, clean, actions, "sirov12")
    assert baseline.receipts["operator_A_relative_frobenius_delta"] == 0.0
    assert baseline.receipts["pre_post_refit_clean_prior_shift_mse"] == 0.0
    changed_clean = clean + 1e-3 * torch.cos(clean)
    changed_observed = observed.clone()
    changed_observed[:, 0] = changed_clean[:, 0]
    second = fit_siro_from_embeddings(
        changed_clean, changed_observed, actions, "sirov12", fit_index=1)
    changed = add_refit_drift_receipts(
        second, baseline, changed_clean, actions, "sirov12")
    for key in (
            "operator_A_relative_frobenius_delta",
            "operator_B_relative_frobenius_delta",
            "operator_K_relative_frobenius_delta",
            "operator_R_relative_frobenius_delta",
            "pre_refit_clean_prior_mse",
            "post_refit_clean_prior_mse",
            "pre_post_refit_clean_prior_shift_mse",
            "pre_post_refit_clean_prior_relative_shift",
            "parity_B_relative_disagreement",
            "parity_B_cosine_alignment",
            "cross_signal_to_full_reachability_trace_ratio"):
        assert math.isfinite(float(changed.receipts[key]))
    assert changed.receipts["pre_post_refit_clean_prior_shift_mse"] > 0.0
    assert changed.receipts["drift_receipt_semantics"].endswith("not_procrustes")


def test_anchor_translation_invariance_and_absolute_noanchor_control() -> None:
    clean, actions, _, _, _ = synthetic_system(noise=1e-5)
    centered = fit_linear_dynamics(
        clean, actions, identity_A=False, anchor_centered=True)
    generator = torch.Generator().manual_seed(12_013)
    offsets = 3.0 * torch.randn(
        clean.shape[0], 1, clean.shape[2], generator=generator,
        dtype=torch.float64)
    translated = fit_linear_dynamics(
        clean + offsets, actions, identity_A=False, anchor_centered=True)
    for first, second in (
            (centered.identified_A, translated.identified_A),
            (centered.action_B, translated.action_B),
            (centered.drift_b, translated.drift_b)):
        assert torch.allclose(first, second, atol=2e-12, rtol=2e-12)
    assert centered.receipts["centered_x0_max_abs"] == 0.0
    assert translated.receipts["centered_x0_max_abs"] == 0.0

    absolute, absolute_actions, A, B, b = synthetic_system(
        anchor_centered=False)
    noanchor = fit_linear_dynamics(
        absolute, absolute_actions, identity_A=False,
        anchor_centered=False)
    assert not noanchor.anchor_centered
    assert not noanchor.receipts["anchor_centered_fit"]
    assert noanchor.receipts["centered_x0_max_abs"] > 0.0
    assert torch.allclose(noanchor.identified_A, A, atol=2e-12, rtol=2e-12)
    assert torch.allclose(noanchor.action_B, B, atol=2e-12, rtol=2e-12)
    assert torch.allclose(noanchor.drift_b, b, atol=2e-12, rtol=2e-12)

    fit = fit_siro_from_embeddings(
        absolute, absolute.clone(), absolute_actions,
        "sirov12_noanchor", fit_index=0)
    assert not fit.receipts["anchor_centered_fit"]
    assert fit.receipts["initial_anchor_max_abs_mismatch"] == 0.0


def main() -> None:
    tests = (
        test_machine_pinv_totalizes_rank_deficiency_without_ridge,
        test_fwl_recovers_stable_dynamics_and_native_action_coordinates,
        test_singular_cap_and_identity_control_refit,
        test_oas_is_data_derived_positive_and_finite,
        test_reachability_uses_every_transition_lag_and_frozen_survival,
        test_actual_prior_alignment_and_action_index,
        test_refit_drift_and_parity_receipts_are_explicit,
        test_anchor_translation_invariance_and_absolute_noanchor_control,
    )
    for test in tests:
        test()
    print(f"All {len(tests)} SIRO-v12 fit tests passed.")


if __name__ == "__main__":
    main()
