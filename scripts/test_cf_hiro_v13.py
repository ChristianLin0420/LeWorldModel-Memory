#!/usr/bin/env python3
"""Synthetic correctness tests for the isolated CF-HIRO-v13 numerical core."""

from __future__ import annotations

import copy
import functools
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
    _orthogonal_contraction,
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
    assert fitted.receipts["held_fold_action_r2_even_to_odd"] > 0.0
    assert fitted.receipts["held_fold_action_r2_odd_to_even"] > 0.0
    assert len(fitted.receipts["held_fold_action_r2_by_lag_even_to_odd"]) \
        == clean.shape[1] - 1


def test_randomization_rank_audit_rejects_noninterventional_actions() -> None:
    clean, _, actions, _, _, _ = synthetic_system(episodes=20)
    actions.zero_()
    try:
        estimate_all_lag_markov_moments(clean, actions)
    except ValueError as error:
        assert "randomized-action audit failed" in str(error)
    else:
        raise AssertionError("rank-deficient actions were accepted as randomized")


def test_realization_is_full_schema_real_normal_and_all_lag_refit() -> None:
    _, _, _, transition, action, read = synthetic_system(
        episodes=8, process_noise=0.0, observation_noise=0.0)
    moments = true_markov(transition, action, read, 9)
    realization = realize_fold_agreement_hankel(moments, moments.clone())
    matrix = realization.state_matrix
    scale = matrix.norm().square().clamp_min(torch.finfo(torch.float64).tiny)
    relative_normality = (matrix.T @ matrix - matrix @ matrix.T).norm() / scale
    assert float(relative_normality) < 1e-12
    assert float(torch.linalg.matrix_norm(matrix, ord=2)) \
        <= FLOAT32_STABILITY_BOUNDARY + 1e-12
    assert realization.receipts["transition_deployment"] == "normal"
    assert realization.receipts["interblock_couplings_zeroed"]
    assert realization.receipts["complex_blocks_canonical_real_normal"]
    assert realization.receipts["agreement_applied_to_h0_and_h1"]
    assert realization.receipts["action_refit_lags"] == moments.shape[0]
    assert realization.receipts["all_lags_consumed"]
    expected_order = min(
        realization.receipts["block_rows"] * read.shape[0],
        realization.receipts["block_columns"] * action.shape[1])
    assert matrix.shape == (expected_order, expected_order)
    assert realization.receipts["state_order"] == expected_order
    assert realization.receipts["model_order_threshold"] is None
    assert torch.all(realization.fold_agreement <= 1.0)
    assert torch.all(realization.fold_agreement >= 0.0)


def test_empirical_bayes_positive_part_and_exact_noshrink_control() -> None:
    _, _, _, transition, action, read = synthetic_system(episodes=8)
    moments = true_markov(transition, action, read, 9)
    generator = torch.Generator().manual_seed(13_009)
    perturbation = .08 * torch.randn(
        moments.shape, generator=generator, dtype=torch.float64)
    shrunk = realize_fold_agreement_hankel(
        moments + perturbation, moments - perturbation)
    less_noise = realize_fold_agreement_hankel(
        moments + .5 * perturbation, moments - .5 * perturbation)
    noshrink = realize_fold_agreement_hankel(
        moments + perturbation, moments - perturbation, agreement_mode="unit")
    assert torch.any((shrunk.fold_agreement > 0.0) & (shrunk.fold_agreement < 1.0))
    assert float(less_noise.fold_agreement.mean()) > float(shrunk.fold_agreement.mean())
    assert torch.equal(noshrink.fold_agreement, torch.ones_like(noshrink.fold_agreement))
    assert shrunk.state_matrix.shape == noshrink.state_matrix.shape
    assert shrunk.receipts["fold_shrinkage_rule"].startswith("positive_part")
    assert noshrink.receipts["fold_shrinkage_rule"] == "unit_no_shrink_control"


def test_normal_projection_removes_transient_couplings_and_triangular_keeps_them() -> None:
    raw = torch.tensor([
        [.82, 1.70, -.25, .40],
        [0.0, .61, 1.20, -.30],
        [0.0, 0.0, .35, .90],
        [0.0, 0.0, 0.0, -.20],
    ], dtype=torch.float64)
    normal, normal_basis, normal_receipts = _orthogonal_contraction(
        raw, deployment="normal")
    triangular, triangular_basis, triangular_receipts = _orthogonal_contraction(
        raw, deployment="triangular")
    identity = torch.eye(4, dtype=torch.float64)
    assert torch.allclose(normal_basis.T @ normal_basis, identity, atol=1e-12)
    assert torch.allclose(triangular_basis.T @ triangular_basis, identity, atol=1e-12)
    assert torch.count_nonzero(normal - torch.diag(torch.diagonal(normal))) == 0
    assert torch.count_nonzero(torch.triu(triangular, diagonal=1)) > 0
    assert normal_receipts["normal_contraction_relative_normality_residual"] < 1e-14
    assert triangular_receipts["deployed_transition_relative_normality_residual"] > 1e-3
    assert normal_receipts["deployed_transition_operator_norm"] \
        <= FLOAT32_STABILITY_BOUNDARY + 1e-12
    assert triangular_receipts["deployed_transition_operator_norm"] \
        > triangular_receipts["deployed_transition_spectral_radius"]
    assert normal_receipts["transition_projection_frobenius_distance"] > 0.0


@functools.lru_cache(maxsize=None)
def _small_fit(mode: str = "full") -> CFHIROFit:
    clean, observed, actions, _, _, _ = synthetic_system(
        episodes=320, length=9, observation_noise=.03, process_noise=.003)
    return fit_cf_hiro(clean, observed, actions, mode=mode)


def test_complete_fit_has_offline_joseph_and_no_out_of_fold_claim() -> None:
    fit = _small_fit()
    state_dim = fit.state_matrix.shape[0]
    assert fit.receipts["selected_temporal_horizons"] is None
    assert fit.receipts["loss_weights"] is None
    assert fit.receipts["learned_memory_parameters"] == 0
    assert fit.receipts["state_reconstruction_selected_horizon"] is None
    assert fit.receipts["offline_covariance_update"] == "joseph"
    assert fit.receipts["online_covariance_update"] == "none_fixed_steady_gain_mean_only"
    assert fit.receipts["fold_role"] == "agreement_estimation_not_out_of_fold_estimation"
    assert len(fit.receipts["unavoidable_realization_choices"]) == 6
    assert fit.process_covariance.shape == (state_dim, state_dim)
    assert fit.measurement_covariance.shape[0] == fit.read_matrix.shape[0]
    assert fit.steady_gain.shape == (state_dim, fit.read_matrix.shape[0])
    assert float(torch.linalg.eigvalsh(fit.process_covariance).min()) > 0.0
    assert float(torch.linalg.eigvalsh(fit.measurement_covariance).min()) > 0.0
    assert float(torch.linalg.eigvalsh(fit.initial_covariance).min()) >= -1e-12
    assert fit.receipts["steady_riccati_relative_residual"] < 1e-8
    identity = torch.eye(state_dim, dtype=torch.float64)
    residual = identity - fit.steady_gain @ fit.read_matrix
    joseph = (residual @ fit.steady_prior_covariance @ residual.T
              + fit.steady_gain @ fit.measurement_covariance @ fit.steady_gain.T)
    assert torch.allclose(fit.initial_covariance, .5 * (joseph + joseph.T), atol=2e-10)
    measured_variance = float(torch.trace(fit.measurement_covariance)
                              / fit.measurement_covariance.shape[0])
    assert abs(measured_variance - .03 ** 2) < 1.5e-4


def test_direct_sum_initialization_and_immutable_complement_anchor() -> None:
    fit = _small_fit()
    generator = torch.Generator().manual_seed(13_101)
    observation0 = torch.randn(5, fit.read_matrix.shape[0],
                               generator=generator, dtype=torch.float64)
    observation1 = torch.randn(5, fit.read_matrix.shape[0],
                               generator=generator, dtype=torch.float64)
    action = torch.randn(5, fit.action_matrix.shape[1],
                         generator=generator, dtype=torch.float64)
    memory = FoldAgreementHankelRiccatiMemory.from_fit(fit)
    state = memory.initial_state(observation0)
    centered = observation0 - memory.output_mean
    assert torch.allclose(memory.output_projector,
                          memory.read_matrix @ memory.initial_map, atol=1e-12)
    assert torch.allclose(state.mean, centered @ memory.initial_map.T, atol=1e-12)
    assert torch.allclose(
        state.complement, centered - state.mean @ memory.read_matrix.T, atol=1e-12)
    assert torch.allclose(memory.read_state(state), observation0, atol=2e-12, rtol=2e-12)
    prior = memory.transition(state, action)
    posterior = memory.correct(prior, observation1)
    assert torch.equal(prior.complement, state.complement)
    assert torch.equal(posterior.complement, state.complement)

    fullanchor_fit = _small_fit("fullanchor")
    assert fullanchor_fit.receipts["state_reconstruction_anchor_policy"] \
        == "full_initial_output_with_zero_dynamic_state"
    assert fullanchor_fit.receipts["state_reconstruction_initial_direct_sum_error"] < 1e-12
    fullanchor = FoldAgreementHankelRiccatiMemory.from_fit(
        fullanchor_fit, mode="fullanchor")
    anchored = fullanchor.initial_state(observation0)
    assert torch.count_nonzero(anchored.mean) == 0
    assert torch.equal(anchored.complement, observation0 - fullanchor.output_mean)
    assert torch.allclose(fullanchor.read_state(anchored), observation0, atol=1e-12)


def test_fixed_gain_mean_correction_and_nocorrect_control_are_exact() -> None:
    fit = _small_fit()
    memory = FoldAgreementHankelRiccatiMemory.from_fit(fit)
    batch = 5
    observation0 = torch.randn(batch, memory.output_dim, dtype=torch.float64)
    observation1 = torch.randn(batch, memory.output_dim, dtype=torch.float64)
    action = torch.randn(batch, memory.action_dim, dtype=torch.float64)
    state = memory.initial_state(observation0)
    prior = memory.transition(state, action)
    posterior, details = memory.correct(prior, observation1, return_details=True)
    expected_innovation = observation1 - memory.read_state(prior)
    expected_correction = expected_innovation @ memory.steady_gain.T
    assert torch.equal(details["innovation"], expected_innovation)
    assert torch.allclose(details["correction"], expected_correction, atol=1e-12)
    assert torch.allclose(posterior.mean, prior.mean + expected_correction, atol=1e-12)
    assert torch.equal(
        details["gain"], memory.steady_gain.unsqueeze(0).expand(batch, -1, -1))
    assert torch.equal(posterior.complement, prior.complement)
    assert not hasattr(posterior, "covariance")

    nocorrect = FoldAgreementHankelRiccatiMemory.from_fit(fit, mode="nocorrect")
    no_prior = nocorrect.transition(nocorrect.initial_state(observation0), action)
    no_posterior, no_details = nocorrect.correct(
        no_prior, observation1, return_details=True)
    assert torch.equal(no_posterior.mean, no_prior.mean)
    assert torch.equal(no_posterior.complement, no_prior.complement)
    assert torch.count_nonzero(no_details["gain"]) == 0
    assert torch.count_nonzero(no_details["correction"]) == 0


def test_noaction_noshrink_and_triangular_controls_are_schema_matched() -> None:
    full_fit = _small_fit()
    noaction = FoldAgreementHankelRiccatiMemory.from_fit(full_fit, mode="noaction")
    observation = torch.randn(4, noaction.output_dim, dtype=torch.float64)
    state = noaction.initial_state(observation)
    first_action = torch.randn(4, noaction.action_dim, dtype=torch.float64)
    second_action = torch.randn(4, noaction.action_dim, dtype=torch.float64) * 100
    first, details = noaction.transition(state, first_action, return_details=True)
    second = noaction.transition(state, second_action)
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.complement, second.complement)
    assert torch.count_nonzero(details["effective_action"]) == 0
    assert torch.count_nonzero(details["action_effect"]) == 0
    assert torch.equal(noaction.action_matrix, full_fit.action_matrix)

    noshrink_fit = _small_fit("noshrink")
    noshrink = FoldAgreementHankelRiccatiMemory.from_fit(
        noshrink_fit, mode="noagreement")
    assert noshrink.mode == "noshrink"
    assert torch.equal(noshrink_fit.fold_agreement,
                       torch.ones_like(noshrink_fit.fold_agreement))
    triangular_fit = _small_fit("triangular")
    triangular = FoldAgreementHankelRiccatiMemory.from_fit(
        triangular_fit, mode="triangular")
    for control in (noaction, noshrink, triangular):
        assert control.state_dim == full_fit.state_matrix.shape[0]
        assert control.parameter_count() == 0
        assert control.diagnostics()["streaming_covariance_floats"] == 0

    try:
        FoldAgreementHankelRiccatiMemory.from_fit(full_fit, mode="noshrink")
    except ValueError as error:
        assert "fold_agreement_mode" in str(error)
    else:
        raise AssertionError("noshrink accepted an agreement-shrunk fit")


def test_batched_forward_matches_streaming_strict_priors_and_rollout() -> None:
    fit_by_mode = {
        "full": _small_fit(),
        "noaction": _small_fit(),
        "fullanchor": _small_fit(),
        "nocorrect": _small_fit(),
        "noshrink": _small_fit("noshrink"),
        "triangular": _small_fit("triangular"),
    }
    for mode, fit in fit_by_mode.items():
        memory = FoldAgreementHankelRiccatiMemory.from_fit(fit, mode=mode)
        generator = torch.Generator().manual_seed(13_100)
        observations = torch.randn(
            3, 7, memory.output_dim, generator=generator, dtype=torch.float64)
        actions = torch.randn(
            3, 6, memory.action_dim, generator=generator, dtype=torch.float64)
        reads, details = memory(observations, actions, return_details=True)
        state = memory.initial_state(observations[:, 0])
        streamed_reads = [memory.read_state(state)]
        streamed_priors = [memory.read_state(state)]
        streamed_means = [state.mean]
        anchor = state.complement.clone()
        for index in range(1, observations.shape[1]):
            prior = memory.transition(state, actions[:, index - 1])
            streamed_priors.append(memory.read_state(prior))
            read, state = memory.step(
                state, observations[:, index], actions[:, index - 1])
            streamed_reads.append(read)
            streamed_means.append(state.mean)
            assert torch.equal(state.complement, anchor)
        assert torch.equal(reads, torch.stack(streamed_reads, dim=1))
        assert torch.equal(details["posterior_reads"], reads)
        assert torch.equal(details["prior_reads"], torch.stack(streamed_priors, dim=1))
        assert torch.equal(details["state_means"], torch.stack(streamed_means, dim=1))
        assert torch.equal(
            details["complement_anchors"], anchor.unsqueeze(1).expand(-1, 7, -1))

        initial = memory.initial_state(observations[:, 0])
        rolled = memory.rollout_transition(initial, actions)
        explicit_means, explicit_complements = [], []
        state = initial
        for index in range(actions.shape[1]):
            state = memory.transition(state, actions[:, index])
            explicit_means.append(state.mean)
            explicit_complements.append(state.complement)
        assert torch.equal(rolled.mean, torch.stack(explicit_means, dim=1))
        assert torch.equal(rolled.complement, torch.stack(explicit_complements, dim=1))


def test_schema_roundtrip_no_parameters_and_atomic_validation() -> None:
    fit = _small_fit()
    memory = CFHIROv13Memory.from_fit(fit)
    assert CFHIROv13Memory is FoldAgreementHankelRiccatiMemory
    assert list(memory.parameters()) == []
    assert memory.parameter_count() == 0
    diagnostics = memory.diagnostics()
    assert diagnostics["state_is_real_normal_contraction"]
    assert diagnostics["online_covariance_update"] == "none_fixed_steady_gain_mean_only"
    state_dict = memory.state_dict()
    restored = FoldAgreementHankelRiccatiMemory(
        memory.output_dim, memory.action_dim, memory.state_dim,
        dtype=torch.float64)
    restored.load_state_dict(state_dict)
    for name, value in memory.named_buffers():
        assert torch.equal(value, dict(restored.named_buffers())[name])
    assert restored.diagnostics()["method"] == "CF-HIRO-v13"

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
