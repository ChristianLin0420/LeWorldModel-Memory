#!/usr/bin/env python3
"""Synthetic correctness tests for the isolated CF-EBO-v14 numerical core."""

from __future__ import annotations

import copy
import functools
import sys
from dataclasses import replace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.cf_ebo import (
    CEBOFit,
    CF_EBO_MODES,
    CF_EBOv14Memory,
    CrossFitEnergyBoundedObserverMemory,
    _directional_risk,
    _direct_sum_maps,
    _fit_correction,
    _observable_energy_coordinates,
    _symmetric_reliability,
    fit_cf_ebo,
)


def synthetic_system(
        *, episodes: int = 200, length: int = 9, state_dim: int = 5,
        output_dim: int = 6, action_dim: int = 2, observation_noise: float = .03,
        process_noise: float = .003, seed: int = 14_001):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(state_dim, state_dim, generator=generator, dtype=torch.float64)
    left, _, right = torch.linalg.svd(raw)
    transition = (left * torch.linspace(
        .25, .78, state_dim, dtype=torch.float64).unsqueeze(0)) @ right
    action = .32 * torch.randn(
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
        states[:, index + 1].add_(process_noise * torch.randn(
            episodes, state_dim, generator=generator, dtype=torch.float64))
    clean = states @ read.T
    observed = clean + observation_noise * torch.randn(
        clean.shape, generator=generator, dtype=torch.float64)
    return clean, observed, executed


@functools.lru_cache(maxsize=None)
def _fit(mode: str = "full") -> CEBOFit:
    clean, observed, actions = synthetic_system()
    return fit_cf_ebo(clean, observed, actions, mode=mode)


def test_observability_energy_identity_and_future_telescope() -> None:
    generator = torch.Generator().manual_seed(14_101)
    raw = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    left, _, right = torch.linalg.svd(raw)
    state = (left * torch.tensor([.31, .48, .66, .81], dtype=torch.float64)) @ right
    read = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    energy = _observable_energy_coordinates(state, read)
    identity = torch.eye(4, dtype=torch.float64)
    assert torch.allclose(
        energy.state_matrix.T @ energy.state_matrix
        + energy.read_matrix.T @ energy.read_matrix,
        identity, atol=3e-12, rtol=3e-12)
    assert energy.receipts["energy_dissipativity_operator_norm"] < 1e-11
    assert energy.receipts["energy_lyapunov_relative_residual"] < 1e-12
    assert float(torch.linalg.matrix_norm(energy.state_matrix, ord=2)) <= 1.0 + 1e-12

    delta = torch.randn(4, generator=generator, dtype=torch.float64)
    cursor = delta.clone()
    total = torch.zeros((), dtype=torch.float64)
    for _ in range(500):
        total.add_((energy.read_matrix @ cursor).square().sum())
        cursor = energy.state_matrix @ cursor
    assert torch.allclose(total, delta.square().sum(), atol=2e-10, rtol=2e-10)


def test_directional_risk_and_minimum_two_way_certificate() -> None:
    base = torch.ones(8, dtype=torch.float64)
    good = _directional_risk(base, .5 * base, "good")
    weaker = _directional_risk(base, .75 * base, "weaker")
    bad = _directional_risk(base, 1.1 * base, "bad")
    assert abs(good.reliability - 1.0) < 1e-14
    assert abs(weaker.reliability - 1.0) < 1e-14
    assert bad.reliability == 0.0
    reliability, receipts = _symmetric_reliability(good, bad, "mechanism")
    assert reliability == 0.0
    assert receipts["mechanism_combination"] == "minimum_directional_positive_part_EB"

    # A noisy positive mean below its variance-of-mean estimate receives exact zero.
    alternating = torch.tensor([0., 2., 0., 2.], dtype=torch.float64)
    uncertain = _directional_risk(
        torch.ones(4, dtype=torch.float64),
        torch.ones(4, dtype=torch.float64) - .01 * alternating,
        "uncertain")
    assert 0.0 <= uncertain.reliability < 1.0


def test_whitened_correction_cap_and_noenergy_control() -> None:
    generator = torch.Generator().manual_seed(14_102)
    innovation = torch.randn(80, 5, 3, generator=generator, dtype=torch.float64)
    matrix = 3.0 * torch.randn(4, 3, generator=generator, dtype=torch.float64)
    errors = innovation @ matrix.T + .01 * torch.randn(
        80, 5, 4, generator=generator, dtype=torch.float64)
    capped = _fit_correction(errors, innovation, energy_cap=True, label="cap")
    raw = _fit_correction(errors, innovation, energy_cap=False, label="raw")
    assert float(torch.linalg.matrix_norm(capped.raw_matrix, ord=2)) > 1.0
    assert float(torch.linalg.matrix_norm(capped.deployed_matrix, ord=2)) <= 1.0 + 1e-12
    assert torch.allclose(raw.deployed_matrix, raw.raw_matrix, atol=2e-12, rtol=2e-12)
    whitened_covariance = (
        capped.innovation_whitener @ capped.innovation_covariance
        @ capped.innovation_whitener.T)
    assert torch.allclose(
        whitened_covariance, torch.eye(3, dtype=torch.float64),
        atol=2e-10, rtol=2e-10)
    assert capped.receipts["cap_future_energy_bound"] \
        == "norm_delta_sq_le_innovation_rank"
    assert raw.receipts["raw_future_energy_bound"] == "none_control"


def test_complete_fit_uses_fixed_coordinate_B_and_no_riccati_deployment() -> None:
    fit = _fit()
    assert isinstance(fit, CEBOFit)
    assert fit.receipts["method"] == "CF-EBO-v14"
    assert fit.receipts["source_coordinate_realization"] == "V13_pooled_normal_A_C_only"
    assert fit.receipts["source_action_map_deployed"] is False
    assert fit.receipts["source_riccati_gain_deployed"] is False
    assert fit.receipts["action_refit"] \
        == "fixed_energy_A_C_fold_specific_and_pooled_all_lags"
    assert fit.receipts["action_combination"] == "minimum_directional_positive_part_EB"
    assert fit.receipts["correction_combination"] \
        == "minimum_directional_positive_part_EB"
    assert float(fit.action_reliability) == min(
        fit.receipts["action_first_direction_reliability"],
        fit.receipts["action_second_direction_reliability"])
    assert fit.receipts["action_combined_risk_reliability"] \
        == float(fit.action_reliability)
    assert float(fit.correction_reliability) == min(
        fit.receipts["correction_first_direction_reliability"],
        fit.receipts["correction_second_direction_reliability"])
    assert fit.receipts["correction_combined_risk_reliability"] \
        == float(fit.correction_reliability)
    assert "action_reliability" not in fit.receipts
    assert "correction_reliability" not in fit.receipts
    assert 0.0 <= float(fit.action_reliability) <= 1.0
    assert 0.0 <= float(fit.correction_reliability) <= 1.0
    assert fit.receipts["energy_dissipativity_operator_norm"] < 1e-10
    assert fit.receipts["correction_pooled_deployed_correction_operator_norm"] <= 1.0
    # V13 action-refit telemetry is namespaced and cannot overwrite the V14 rule.
    assert "v13_coordinate_action_refit" in fit.receipts
    assert fit.receipts["action_refit"] != fit.receipts["v13_coordinate_action_refit"]


def test_rank_aware_direct_sum_and_zero_codimension_are_explicit() -> None:
    fit = _fit()
    memory = CrossFitEnergyBoundedObserverMemory.from_fit(fit)
    assert fit.receipts["complement_codimension"] == 1
    assert fit.receipts["complement_present"] is True
    observation = torch.randn(7, memory.output_dim, dtype=torch.float64)
    state = memory.initial_state(observation)
    assert torch.allclose(memory.read_state(state), observation, atol=3e-11, rtol=3e-11)
    assert float((state.complement @ memory.read_matrix).abs().max()) < 1e-10

    full_row_read = torch.randn(4, 7, dtype=torch.float64)
    initial, output, complement, receipts = _direct_sum_maps(full_row_read)
    assert receipts["complement_codimension"] == 0
    assert receipts["complement_present"] is False
    assert torch.allclose(complement, torch.zeros_like(complement), atol=2e-12)
    assert torch.allclose(full_row_read @ initial, output, atol=2e-12)


def test_modes_are_exact_and_schema_matched() -> None:
    fits = {mode: _fit(mode) for mode in sorted(CF_EBO_MODES)}
    memories = {
        mode: CrossFitEnergyBoundedObserverMemory.from_fit(fit, mode=mode)
        for mode, fit in fits.items()
    }
    full_shape = fits["full"].state_matrix.shape
    for mode, memory in memories.items():
        assert memory.state_matrix.shape == full_shape
        assert memory.parameter_count() == 0
        assert list(memory.parameters()) == []
        assert memory.diagnostics()[f"{mode}_exact"] if mode != "full" else True
    assert torch.count_nonzero(fits["noaction"].action_matrix) == 0
    assert float(fits["noaction"].action_reliability) == 0.0
    assert float(fits["nocorrect"].correction_reliability) == 0.0
    assert float(fits["norisk"].action_reliability) == 1.0
    assert float(fits["norisk"].correction_reliability) == 1.0
    # Even a legacy/malformed receipt cannot shadow live deployed ablation state.
    memories["noaction"]._fit_receipts["action_reliability"] = .75
    memories["nocorrect"]._fit_receipts["correction_reliability"] = .75
    memories["norisk"]._fit_receipts.update({
        "action_reliability": .25, "correction_reliability": .25})
    assert memories["noaction"].diagnostics()["action_reliability"] == 0.0
    assert memories["nocorrect"].diagnostics()["correction_reliability"] == 0.0
    assert memories["norisk"].diagnostics()["action_reliability"] == 1.0
    assert memories["norisk"].diagnostics()["correction_reliability"] == 1.0
    assert torch.allclose(
        fits["noenergycap"].correction_matrix,
        fits["noenergycap"].raw_correction_matrix,
        atol=2e-12, rtol=2e-12)

    observation = torch.randn(4, memories["noaction"].output_dim, dtype=torch.float64)
    state = memories["noaction"].initial_state(observation)
    first = memories["noaction"].transition(
        state, torch.randn(4, memories["noaction"].action_dim, dtype=torch.float64))
    second = memories["noaction"].transition(
        state, 100 * torch.randn(4, memories["noaction"].action_dim, dtype=torch.float64))
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.complement, second.complement)

    nocorrect = memories["nocorrect"]
    prior = nocorrect.transition(
        nocorrect.initial_state(observation),
        torch.randn(4, nocorrect.action_dim, dtype=torch.float64))
    posterior, details = nocorrect.correct(
        prior, torch.randn_like(observation), return_details=True)
    assert torch.equal(prior.mean, posterior.mean)
    assert torch.equal(prior.complement, posterior.complement)
    assert torch.count_nonzero(details["correction"]) == 0
    assert torch.count_nonzero(details["gain"]) == 0


def test_radial_gate_is_redescending_and_correction_energy_is_bounded() -> None:
    memory = CrossFitEnergyBoundedObserverMemory.from_fit(_fit())
    observation = torch.zeros(3, memory.output_dim, dtype=torch.float64)
    prior = memory.initial_state(observation)
    ordinary = torch.ones_like(observation)
    extreme = ordinary * 1e100
    _, ordinary_details = memory.correct(prior, ordinary, return_details=True)
    _, extreme_details = memory.correct(prior, extreme, return_details=True)
    assert torch.isfinite(extreme_details["innovation_score"]).all()
    assert torch.isfinite(extreme_details["radial_gate"]).all()
    assert torch.isfinite(extreme_details["correction"]).all()
    assert torch.all(extreme_details["radial_gate"] < ordinary_details["radial_gate"])
    bound = float(memory.correction_reliability.square() * memory.innovation_rank)
    correction_energy = extreme_details["correction"].double().square().sum(dim=-1)
    assert torch.all(correction_energy <= bound * (1.0 + 1e-8) + 1e-10)

    noradial = CrossFitEnergyBoundedObserverMemory.from_fit(
        _fit("noradial"), mode="noradial")
    _, no_details = noradial.correct(
        noradial.initial_state(observation), ordinary, return_details=True)
    assert torch.equal(no_details["radial_gate"], torch.ones_like(no_details["radial_gate"]))


def test_extreme_finite_bfloat16_innovation_stays_finite() -> None:
    memory = CrossFitEnergyBoundedObserverMemory.from_fit(
        _fit(), dtype=torch.bfloat16)
    initial = torch.zeros(2, memory.output_dim, dtype=torch.bfloat16)
    state = memory.initial_state(initial)
    maximum = torch.finfo(torch.bfloat16).max
    observation = torch.full_like(initial, maximum)
    posterior, details = memory.correct(state, observation, return_details=True)
    for tensor in (
            posterior.mean, posterior.complement, details["innovation"],
            details["normalized_innovation"], details["innovation_score"],
            details["radial_gate"], details["correction"], details["posterior_read"]):
        assert torch.isfinite(tensor).all()
    assert torch.all(details["radial_gate"] >= 0.0)
    assert torch.all(details["radial_gate"] <= 1.0)
    bound = float(memory.correction_reliability.float().square() * memory.innovation_rank)
    energy = details["correction"].float().square().sum(dim=-1)
    assert torch.all(energy <= bound * (1.0 + 2e-2) + 1e-4)


def test_batched_forward_matches_streaming_and_strict_priors_for_every_mode() -> None:
    generator = torch.Generator().manual_seed(14_103)
    for mode in sorted(CF_EBO_MODES):
        memory = CrossFitEnergyBoundedObserverMemory.from_fit(_fit(mode), mode=mode)
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
        assert torch.isfinite(details["innovation_scores"]).all()
        assert torch.isfinite(details["radial_gates"]).all()

        rolled = memory.rollout_transition(
            memory.initial_state(observations[:, 0]), actions)
        explicit_means, explicit_complements = [], []
        state = memory.initial_state(observations[:, 0])
        for index in range(actions.shape[1]):
            state = memory.transition(state, actions[:, index])
            explicit_means.append(state.mean)
            explicit_complements.append(state.complement)
        assert torch.equal(rolled.mean, torch.stack(explicit_means, dim=1))
        assert torch.equal(rolled.complement, torch.stack(explicit_complements, dim=1))


def test_schema_roundtrip_no_parameters_and_atomic_validation() -> None:
    fit = _fit()
    memory = CF_EBOv14Memory.from_fit(fit)
    assert CF_EBOv14Memory is CrossFitEnergyBoundedObserverMemory
    assert memory.parameter_count() == 0
    assert memory.diagnostics()["method"] == "CF-EBO-v14"
    state_dict = memory.state_dict()
    restored = CrossFitEnergyBoundedObserverMemory(
        memory.output_dim, memory.action_dim, memory.state_dim,
        dtype=torch.float64)
    restored.load_state_dict(state_dict)
    for name, value in memory.named_buffers():
        assert torch.equal(value, dict(restored.named_buffers())[name])
    assert restored.diagnostics()["method"] == "CF-EBO-v14"

    before = {name: value.clone() for name, value in memory.named_buffers()}
    bad = replace(fit, innovation_covariance=-fit.innovation_covariance)
    try:
        memory.install_fit(bad)
    except ValueError as error:
        assert "positive definite" in str(error)
    else:
        raise AssertionError("negative innovation covariance was installed")
    for name, value in memory.named_buffers():
        assert torch.equal(value, before[name])

    wrong_mode = CrossFitEnergyBoundedObserverMemory(
        memory.output_dim, memory.action_dim, memory.state_dim,
        mode="noaction", dtype=torch.float64)
    try:
        wrong_mode.install_fit(fit)
    except ValueError as error:
        assert "fit_mode" in str(error)
    else:
        raise AssertionError("a full fit was installed into a noaction memory")


def test_fit_rejects_rank_deficient_actions_and_pads_unobservable_schema() -> None:
    clean, observed, actions = synthetic_system(episodes=20)
    actions.zero_()
    try:
        fit_cf_ebo(clean, observed, actions)
    except ValueError as error:
        assert "randomized-action audit failed" in str(error)
    else:
        raise AssertionError("rank-deficient actions were accepted")

    state = torch.diag(torch.tensor([.5, .7], dtype=torch.float64))
    read = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    energy = _observable_energy_coordinates(state, read)
    support = energy.energy_support_projector
    assert energy.state_matrix.shape == (2, 2)
    assert energy.read_matrix.shape == (1, 2)
    assert energy.receipts["energy_state_rank"] == 1
    assert energy.receipts["energy_inactive_padding"] == 1
    assert torch.equal(support, torch.diag(torch.tensor(
        [1.0, 0.0], dtype=torch.float64)))
    assert torch.allclose(
        energy.state_matrix.T @ energy.state_matrix
        + energy.read_matrix.T @ energy.read_matrix,
        support, atol=2e-12, rtol=2e-12)
    assert torch.count_nonzero(energy.state_matrix[1]) == 0
    assert torch.count_nonzero(energy.state_matrix[:, 1]) == 0
    assert torch.count_nonzero(energy.read_matrix[:, 1]) == 0
    delta = torch.tensor([2.0, 7.0], dtype=torch.float64)
    cursor = delta.clone()
    future = torch.zeros((), dtype=torch.float64)
    for _ in range(200):
        future.add_((energy.read_matrix @ cursor).square().sum())
        cursor = energy.state_matrix @ cursor
    assert torch.allclose(
        future, (support @ delta).square().sum(), atol=2e-12, rtol=2e-12)


def main() -> None:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} CF-EBO-v14 synthetic tests passed.")


if __name__ == "__main__":
    main()
