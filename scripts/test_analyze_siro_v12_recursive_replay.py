#!/usr/bin/env python3
"""Focused CPU tests for the isolated SIRO-v12b recursive replay audit."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_siro_v12_recursive_replay import (
    FLOAT32_STABILITY_CAP,
    VARIANTS,
    analyze_task_embeddings,
    centered_oas_from_values,
    immutable_parity_folds,
    operator_algebra_receipts,
    project_real_normal_stable,
    _within_relative,
    validate_screen_ready,
)


def _synthetic() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(12_201)
    episodes, length, dimension, action_dim = 16, 8, 4, 2
    actions = torch.randn(
        episodes, length - 1, action_dim, generator=generator,
        dtype=torch.float64)
    A = torch.tensor([
        [0.72, -0.10, 0.00, 0.00],
        [0.10, 0.72, 0.00, 0.00],
        [0.00, 0.00, 0.56, 0.08],
        [0.00, 0.00, -0.08, 0.56],
    ], dtype=torch.float64)
    B = torch.tensor([
        [0.30, 0.00], [0.00, 0.25], [0.20, -0.15], [-0.10, 0.20],
    ], dtype=torch.float64)
    anchor = torch.randn(
        episodes, dimension, generator=generator, dtype=torch.float64)
    x = torch.zeros(episodes, length, dimension, dtype=torch.float64)
    for step in range(length - 1):
        process = 0.015 * torch.randn(
            episodes, dimension, generator=generator, dtype=torch.float64)
        x[:, step + 1] = x[:, step] @ A.T + actions[:, step] @ B.T + process
    clean = anchor[:, None] + x
    observed = clean.clone()
    # Correlated observation errors create a real deployed-history mismatch while
    # preserving the exact registered t=0 anchor.
    noise = torch.zeros_like(x)
    for step in range(1, length):
        fresh = 0.07 * torch.randn(
            episodes, dimension, generator=generator, dtype=torch.float64)
        noise[:, step] = 0.75 * noise[:, step - 1] + fresh
    observed[:, 1:] += noise[:, 1:]
    return clean, observed, actions


def test_parity_folds_are_immutable_disjoint_and_exhaustive() -> None:
    even, odd = immutable_parity_folds(11)
    assert torch.equal(even, torch.tensor([0, 2, 4, 6, 8, 10]))
    assert torch.equal(odd, torch.tensor([1, 3, 5, 7, 9]))
    assert not set(even.tolist()) & set(odd.tolist())
    assert sorted(torch.cat((even, odd)).tolist()) == list(range(11))


def test_normal_projection_removes_nonnormal_transient_growth() -> None:
    raw = torch.tensor([
        [1.08, 7.0, 0.0, 0.0],
        [0.0, 0.91, 3.0, 0.0],
        [0.0, 0.0, 0.82, -0.3],
        [0.0, 0.0, 0.3, 0.82],
    ], dtype=torch.float64)
    projected, receipts = project_real_normal_stable(raw)
    algebra = operator_algebra_receipts(projected, 20)
    assert receipts["normality_relative_commutator"] < 1e-12
    assert algebra["normality_relative_commutator"] < 1e-12
    assert algebra["spectral_radius"] <= FLOAT32_STABILITY_CAP + 1e-12
    assert algebra["singular_max"] <= FLOAT32_STABILITY_CAP + 1e-12
    assert algebra["transient_singular_max_through_horizon"] <= 1.0 + 1e-12
    assert receipts["relative_projection_delta"] > 0.0


def test_centered_oas_survives_huge_offset_cancellation() -> None:
    generator = torch.Generator().manual_seed(12_202)
    values = (
        torch.full((600, 6), 1.0e10, dtype=torch.float64)
        + torch.randn(600, 6, generator=generator, dtype=torch.float64))
    mean = values.mean(dim=0)
    naive = values.T @ values / len(values) - torch.outer(mean, mean)
    naive = 0.5 * (naive + naive.T)
    assert float(torch.linalg.eigvalsh(naive).min()) < -1.0

    fit, diagnostics = centered_oas_from_values(
        values, label="synthetic-huge-offset")
    assert torch.isfinite(fit.mean).all()
    assert torch.isfinite(fit.covariance).all()
    assert float(torch.linalg.eigvalsh(fit.covariance).min()) >= 0.0
    assert diagnostics["algorithm"] == (
        "scaled_reference_mean_explicit_centered_gram_oas_fp64")
    assert diagnostics["input_abs_max"] > 1.0e9
    assert diagnostics["centered_abs_max"] < 10.0
    assert diagnostics["centered_to_input_abs_ratio"] < 1.0e-8
    assert diagnostics["oas_condition"] < 10.0


def test_relative_agreement_gate_is_two_sided() -> None:
    assert _within_relative(1.04, 1.0, 0.05)
    assert _within_relative(0.96, 1.0, 0.05)
    assert not _within_relative(0.5, 100.0, 0.05)
    assert not _within_relative(100.0, 0.5, 0.05)


def test_all_replays_are_finite_crossfit_and_algebra_checked() -> None:
    clean, observed, actions = _synthetic()
    result = analyze_task_embeddings(clean, observed, actions)
    assert tuple(result["variants"]) == VARIANTS
    assert result["fold_contract"]["disjoint_exhaustive"] is True
    assert result["fold_contract"]["even_count"] == 8
    assert result["fold_contract"]["odd_count"] == 8
    for variant in VARIANTS:
        metrics = result["variants"][variant]
        assert metrics["recursive_clean_prior_mse"] >= 0.0
        assert metrics["recursive_clean_posterior_mse"] >= 0.0
        assert metrics["recursive_clean_prior_nmse"] >= 0.0
        assert metrics["evaluated_scalar_elements"] == 16 * 7 * 4
    identity = result["variants"]["identity_k_current_a"]
    assert all(
        fold["algebra"]["identity_posterior_observation_max_abs"] == 0.0
        for fold in identity["folds"])
    deployed = result["variants"]["deployed_history_lmmse_current_a"]
    assert deployed["algebra"]["every_gain_cross_fitted"] is True
    assert deployed["algebra"]["gain_fits"] == 2 * 7
    assert all(
        row["source_fold"] == 1 - row["target_fold"]
        for row in deployed["algebra"]["gain_receipts"])
    for variant in ("riccati_current_a", "riccati_normal_stable_a"):
        assert all(
            fold["algebra"]["posterior_covariance_min_eigenvalue"] > -1e-10
            for fold in result["variants"][variant]["folds"])
    assert result["action_identification"]["mean_partial_r2"] > 0.0
    assert result["action_identification"]["B_fold_cosine"] > 0.8


def test_active_or_incomplete_screen_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / ".siro_v12_screen.lock").touch()
        try:
            validate_screen_ready(root, 30)
        except RuntimeError as exc:
            assert "active" in str(exc)
        else:
            raise AssertionError("active screen was accepted")
        (root / ".siro_v12_screen.lock").unlink()
        try:
            validate_screen_ready(root, 30)
        except FileNotFoundError as exc:
            assert "completed" in str(exc)
        else:
            raise AssertionError("incomplete screen was accepted")


def main() -> None:
    tests = (
        test_parity_folds_are_immutable_disjoint_and_exhaustive,
        test_normal_projection_removes_nonnormal_transient_growth,
        test_centered_oas_survives_huge_offset_cancellation,
        test_relative_agreement_gate_is_two_sided,
        test_all_replays_are_finite_crossfit_and_algebra_checked,
        test_active_or_incomplete_screen_fails_closed,
    )
    for test in tests:
        test()
    print(f"All {len(tests)} SIRO-v12b recursive replay tests passed.")


if __name__ == "__main__":
    main()
