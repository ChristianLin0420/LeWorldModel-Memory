#!/usr/bin/env python3
"""Synthetic correctness tests for the isolated CF-HIRO-v13 prototype."""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.cf_hiro import (
    CFHIROFit,
    CFHIROv13Memory,
    FLOAT32_STABILITY_BOUNDARY,
    FoldAgreementHankelRiccatiMemory,
    estimate_all_lag_markov_moments,
    fit_cf_hiro,
    realize_fold_agreement_hankel,
)


def synthetic_system(
        *, episodes: int = 800, length: int = 10, state_dim: int = 3,
        output_dim: int = 4, action_dim: int = 2, observation_noise: float = .025,
        process_noise: float = .002, seed: int = 13_001):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(state_dim, state_dim, generator=generator, dtype=torch.float64)
    left, _, right = torch.linalg.svd(raw)
    transition = (left * torch.linspace(
        .32, .79, state_dim, dtype=torch.float64).unsqueeze(0)) @ right
    action = .28 * torch.randn(
        state_dim, action_dim, generator=generator, dtype=torch.float64)
    read = torch.randn(
        output_dim, state_dim, generator=generator, dtype=torch.float64)
    read = read / read.norm(dim=1, keepdim=True)
    executed = torch.randn(
        episodes, length - 1, action_dim,
        generator=generator, dtype=torch.float64)
    states = torch.zeros(episodes, length, state_dim, dtype=torch.float64)
    states[:, 0] = torch.randn(
        episodes, state_dim, generator=generator, dtype=torch.float64)
    for index in range(length - 1):
        states[:, index + 1] = (
            states[:, index] @ transition.T + executed[:, index] @ action.T)
        if process_noise:
            states[:, index + 1].add_(process_noise * torch.randn(
                episodes, state_dim, generator=generator, dtype=torch.float64))
    clean = states @ read.T
    observed = clean + observation_noise * torch.randn(
        clean.shape, generator=generator, dtype=torch.float64)
    return clean, observed, executed, transition, action, read


def true_markov(transition, action, read, count):
    moments = []
    power = torch.eye(transition.shape[0], dtype=torch.float64)
    for _ in range(count):
        moments.append(read @ power @ action)
        power = power @ transition
    return torch.stack(moments)


def test_all_lag_randomized_action_moments_and_fold_separation() -> None:
    clean, _, actions, transition, action, read = synthetic_system(
        episodes=1600, process_noise=0.0, observation_noise=0.0)
    fitted = estimate_all_lag_markov_moments(clean, actions)
    expected = true_markov(transition, action, read, clean.shape[1] - 1)
    relative = (fitted.average - expected).norm() / expected.norm()
    assert float(relative) < .075
    assert fitted.even.shape == expected.shape
    assert fitted.odd.shape == expected.shape
    assert fitted.receipts["all_available_lags"]
    assert fitted.receipts["markov_lag_count"] == clean.shape[1] - 1
    assert fitted.receipts["last_lag"] == clean.shape[1] - 2
    assert len(fitted.sample_counts) == clean.shape[1] - 1
    assert fitted.receipts["even_episodes"] == fitted.receipts["odd_episodes"] == 800
    assert not fitted.receipts["randomization_proved_from_data"]


def test_randomization_rank_audit_rejects_noninterventional_actions() -> None:
    clean, _, actions, _, _, _ = synthetic_system(episodes=20)
    actions.zero_()
    try:
        estimate_all_lag_markov_moments(clean, actions)
    except ValueError as error:
        assert "randomized-action audit failed" in str(error)
    else:
        raise AssertionError("rank-deficient actions were accepted as randomized")


def test_exact_all_lag_hankel_realization_and_schur_deployment() -> None:
    _, _, _, transition, action, read = synthetic_system(
        episodes=8, process_noise=0.0, observation_noise=0.0)
    moments = true_markov(transition, action, read, 9)
    realization = realize_fold_agreement_hankel(moments, moments.clone())
    predicted = true_markov(
        realization.state_matrix, realization.action_matrix,
        realization.read_matrix, moments.shape[0])
    assert torch.allclose(predicted, moments, atol=2e-8, rtol=2e-8)
    assert float(torch.tril(realization.state_matrix, diagonal=-2).abs().max()) < 1e-12
    assert float(torch.linalg.eigvals(realization.state_matrix).abs().max()) \
        <= FLOAT32_STABILITY_BOUNDARY + 1e-12
    assert realization.receipts["all_lags_consumed"]
    assert realization.receipts["h1_last_lag"] == moments.shape[0] - 1
    expected_order = min(
        realization.receipts["block_rows"] * read.shape[0],
        realization.receipts["block_columns"] * action.shape[1])
    assert realization.state_matrix.shape == (expected_order, expected_order)
    assert realization.receipts["model_order_threshold"] is None
    assert torch.all(realization.fold_agreement <= 1.0)
    assert torch.all(realization.fold_agreement >= 0.0)


def test_fold_agreement_is_continuous_not_binary_model_selection() -> None:
    _, _, _, transition, action, read = synthetic_system(episodes=8)
    moments = true_markov(transition, action, read, 9)
    generator = torch.Generator().manual_seed(13_009)
    perturbation = .08 * torch.randn(
        moments.shape, generator=generator, dtype=torch.float64)
    noisy = realize_fold_agreement_hankel(
        moments + perturbation, moments - perturbation)
    smaller = realize_fold_agreement_hankel(
        moments + .99 * perturbation, moments - .99 * perturbation)
    assert torch.any((noisy.fold_agreement > 0.0) & (noisy.fold_agreement < .999))
    assert noisy.state_matrix.shape == smaller.state_matrix.shape
    assert not torch.equal(noisy.fold_agreement, smaller.fold_agreement)
    assert noisy.receipts["state_order_rule"].startswith("full_rectangular")
    assert noisy.receipts["fold_shrinkage_rule"].startswith("m^2/")


def _small_fit() -> CFHIROFit:
    clean, observed, actions, _, _, _ = synthetic_system(
        episodes=420, length=9, observation_noise=.03, process_noise=.003)
    return fit_cf_hiro(clean, observed, actions)


def test_complete_paired_view_fit_has_auditable_q_r_and_riccati() -> None:
    fit = _small_fit()
    state_dim = fit.state_matrix.shape[0]
    assert fit.receipts["selected_temporal_horizons"] is None
    assert fit.receipts["loss_weights"] is None
    assert fit.receipts["learned_memory_parameters"] == 0
    assert fit.receipts["state_reconstruction_selected_horizon"] is None
    assert fit.receipts["covariance_update"] == "joseph"
    assert len(fit.receipts["unavoidable_realization_choices"]) == 6
    assert fit.process_covariance.shape == (state_dim, state_dim)
    assert fit.measurement_covariance.shape[0] == fit.read_matrix.shape[0]
    assert float(torch.linalg.eigvalsh(fit.process_covariance).min()) > 0.0
    assert float(torch.linalg.eigvalsh(fit.measurement_covariance).min()) > 0.0
    assert float(torch.linalg.eigvalsh(fit.initial_covariance).min()) >= -1e-12
    assert fit.receipts["steady_riccati_relative_residual"] < 1e-8
    # Paired differences have iid variance 0.03^2 in this generator.
    measured_variance = float(torch.trace(fit.measurement_covariance)
                              / fit.measurement_covariance.shape[0])
    assert abs(measured_variance - .03 ** 2) < 1.5e-4
    assert fit.markov_even.shape[0] == 8
    assert fit.markov_odd.shape == fit.markov_even.shape
    assert fit.fold_agreement.numel() == state_dim


def test_joseph_online_update_is_psd_and_matches_formula() -> None:
    fit = _small_fit()
    memory = FoldAgreementHankelRiccatiMemory.from_fit(fit)
    batch = 5
    observation0 = torch.randn(batch, memory.output_dim, dtype=torch.float64)
    observation1 = torch.randn(batch, memory.output_dim, dtype=torch.float64)
    action = torch.randn(batch, memory.action_dim, dtype=torch.float64)
    state = memory.initial_state(observation0)
    prior = memory.transition(state, action)
    posterior, details = memory.correct(prior, observation1, return_details=True)
    gain = details["gain"]
    identity = torch.eye(memory.state_dim, dtype=torch.float64)
    residual_map = identity - gain @ memory.read_matrix
    expected = (
        residual_map @ prior.covariance @ residual_map.transpose(-2, -1)
        + gain @ memory.measurement_covariance @ gain.transpose(-2, -1))
    expected = .5 * (expected + expected.transpose(-2, -1))
    assert torch.allclose(posterior.covariance, expected, atol=2e-12, rtol=2e-12)
    assert float(torch.linalg.eigvalsh(posterior.covariance).min()) >= -2e-12
    assert torch.allclose(
        details["posterior_read"], memory.read_state(posterior),
        atol=1e-12, rtol=1e-12)


def test_noaction_is_bit_exact_and_keeps_fitted_action_operator() -> None:
    fit = _small_fit()
    memory = FoldAgreementHankelRiccatiMemory.from_fit(fit, mode="noaction")
    observation = torch.randn(4, memory.output_dim, dtype=torch.float64)
    state = memory.initial_state(observation)
    first_action = torch.randn(4, memory.action_dim, dtype=torch.float64)
    second_action = torch.randn(4, memory.action_dim, dtype=torch.float64) * 100
    first, details = memory.transition(state, first_action, return_details=True)
    second = memory.transition(state, second_action)
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.covariance, second.covariance)
    assert torch.count_nonzero(details["effective_action"]) == 0
    assert torch.count_nonzero(details["action_effect"]) == 0
    assert torch.equal(memory.action_matrix, fit.action_matrix)
    assert memory.diagnostics()["effective_action_matrix_norm"] == 0.0
    full = FoldAgreementHankelRiccatiMemory.from_fit(fit, mode="full")
    full_first = full.transition(full.initial_state(observation), first_action)
    full_second = full.transition(full.initial_state(observation), second_action)
    assert not torch.equal(full_first.mean, full_second.mean)


def test_batched_forward_matches_explicit_streaming_and_rollout() -> None:
    fit = _small_fit()
    for mode in ("full", "noaction"):
        memory = FoldAgreementHankelRiccatiMemory.from_fit(fit, mode=mode)
        generator = torch.Generator().manual_seed(13_100)
        observations = torch.randn(
            3, 7, memory.output_dim, generator=generator, dtype=torch.float64)
        actions = torch.randn(
            3, 6, memory.action_dim, generator=generator, dtype=torch.float64)
        reads, details = memory(observations, actions, return_details=True)
        state = memory.initial_state(observations[:, 0])
        streamed_reads = [memory.read_state(state)]
        streamed_means = [state.mean]
        streamed_covariances = [state.covariance]
        for index in range(1, observations.shape[1]):
            read, state = memory.step(
                state, observations[:, index], actions[:, index - 1])
            streamed_reads.append(read)
            streamed_means.append(state.mean)
            streamed_covariances.append(state.covariance)
        assert torch.equal(reads, torch.stack(streamed_reads, dim=1))
        assert torch.equal(details["state_means"], torch.stack(streamed_means, dim=1))
        assert torch.equal(
            details["state_covariances"], torch.stack(streamed_covariances, dim=1))

        initial = memory.initial_state(observations[:, 0])
        rolled = memory.rollout_transition(initial, actions)
        explicit_means, explicit_covariances = [], []
        state = initial
        for index in range(actions.shape[1]):
            state = memory.transition(state, actions[:, index])
            explicit_means.append(state.mean)
            explicit_covariances.append(state.covariance)
        assert torch.equal(rolled.mean, torch.stack(explicit_means, dim=1))
        assert torch.equal(rolled.covariance, torch.stack(explicit_covariances, dim=1))


def test_schema_roundtrip_no_parameters_and_atomic_validation() -> None:
    fit = _small_fit()
    memory = CFHIROv13Memory.from_fit(fit)
    assert CFHIROv13Memory is FoldAgreementHankelRiccatiMemory
    assert list(memory.parameters()) == []
    assert memory.parameter_count() == 0
    assert memory.diagnostics()["state_is_real_schur"]
    state_dict = memory.state_dict()
    restored = FoldAgreementHankelRiccatiMemory(
        memory.output_dim, memory.action_dim, memory.state_dim,
        dtype=torch.float64)
    restored.load_state_dict(state_dict)
    for name, value in memory.named_buffers():
        assert torch.equal(value, dict(restored.named_buffers())[name])
    assert restored.diagnostics()["method"] == "CF-HIRO-v13-prototype"

    before = {name: value.clone() for name, value in memory.named_buffers()}
    bad = copy.copy(fit)
    object.__setattr__(bad, "measurement_covariance", -fit.measurement_covariance)
    try:
        memory.install_fit(bad)
    except ValueError as error:
        assert "positive definite" in str(error)
    else:
        raise AssertionError("negative measurement covariance was installed")
    for name, value in memory.named_buffers():
        assert torch.equal(value, before[name])


def main() -> None:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} CF-HIRO-v13 synthetic tests passed.")


if __name__ == "__main__":
    main()
