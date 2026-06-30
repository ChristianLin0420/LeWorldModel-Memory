#!/usr/bin/env python3
"""Synthetic correctness tests for the isolated CVPF-v15 numerical core."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.cvpf import (
    CVPFFit,
    CVPFState,
    CVPFv15Memory,
    CrossViewPredictiveFiltrationMemory,
    fit_cvpf,
)


def _synthetic(episodes: int = 48, length: int = 9, dimension: int = 6,
               action_dim: int = 2):
    generator = torch.Generator().manual_seed(15_515)
    transition = torch.diag(torch.linspace(.55, .86, dimension, dtype=torch.float64))
    transition[0, 1] = -.12
    transition[1, 0] = .09
    action_map = .22 * torch.randn(
        dimension, action_dim, generator=generator, dtype=torch.float64)
    actions = torch.randn(
        episodes, length - 1, action_dim, generator=generator,
        dtype=torch.float64)
    clean = torch.zeros(episodes, length, dimension, dtype=torch.float64)
    clean[:, 0] = torch.randn(
        episodes, dimension, generator=generator, dtype=torch.float64)
    for index in range(length - 1):
        clean[:, index + 1] = (
            clean[:, index] @ transition.T + actions[:, index] @ action_map.T)
    observed = clean + .025 * torch.randn(
        clean.shape, generator=generator, dtype=torch.float64)
    observed[::2, 3:6] = 0.0
    observed[:, 0] = clean[:, 0]
    return clean, observed, actions


def test_fit_schema_receipts_and_bounds() -> None:
    clean, observed, actions = _synthetic()
    fit = fit_cvpf(clean, observed, actions)
    assert isinstance(fit, CVPFFit)
    h, d, a = clean.shape[1] - 1, clean.shape[2], actions.shape[2]
    assert fit.anchor_decoder.shape == (h, d, d)
    assert fit.action_decoder.shape == (h, d, d)
    assert fit.observation_decoder.shape == (h, d, d)
    assert fit.action_encoder.shape == (d, a)
    assert fit.observation_encoder.shape == (d, d)
    for value in (
            fit.action_rho, fit.observation_rho, fit.action_weight,
            fit.observation_weight, fit.action_gain, fit.observation_gain):
        assert torch.isfinite(value).all()
        assert bool(((0 <= value) & (value <= 1)).all())
    assert fit.receipts["future_horizon"] == h
    assert fit.receipts["streaming_state_floats"] == 3 * d
    assert fit.receipts["future_covariance_materialized"] is False
    assert fit.receipts["learned_memory_parameters"] == 0
    assert fit.receipts["observation_fit_uses_recursive_deployed_innovations"] is True
    assert 0.0 < fit.receipts[
        "observation_deployed_to_fit_innovation_rms_ratio"] < float("inf")
    assert fit.receipts["fold_baseline_and_alignment_are_pooled"] is True
    assert fit.receipts["mode_risk_is_not_a_deployed_recursive_certificate"] is True


def test_streaming_matches_batch_and_initial_is_exact() -> None:
    clean, observed, actions = _synthetic()
    memory = CVPFv15Memory.from_fit(fit_cvpf(clean, observed, actions))
    batch, details = memory(observed[:5], actions[:5], return_details=True)
    state = memory.initial_state(observed[:5, 0])
    streamed = [observed[:5, 0]]
    for index in range(1, observed.shape[1]):
        output, state = memory.step(
            state, observed[:5, index], actions[:5, index - 1])
        streamed.append(output)
    assert torch.equal(torch.stack(streamed, dim=1), batch)
    assert torch.equal(batch[:, 0], observed[:5, 0])
    assert torch.equal(details["posterior_reads"], batch)
    assert details["prior_reads"].shape == batch.shape


def test_prefix_causality() -> None:
    clean, observed, actions = _synthetic()
    memory = CVPFv15Memory.from_fit(fit_cvpf(clean, observed, actions))
    altered_observed = observed[:3].clone()
    altered_actions = actions[:3].clone()
    altered_observed[:, 5:] = 999.0
    altered_actions[:, 5:] = -999.0
    reference = memory(observed[:3], actions[:3])
    altered = memory(altered_observed, altered_actions)
    assert torch.equal(reference[:, :5], altered[:, :5])


def test_exact_inference_ablations() -> None:
    clean, observed, actions = _synthetic()
    for mode in ("nocorrect", "noaction", "anchoronly"):
        fit = fit_cvpf(clean, observed, actions, mode=mode)
        memory = CVPFv15Memory.from_fit(fit)
        _, details = memory(observed[:4], actions[:4], return_details=True)
        if mode in ("noaction", "anchoronly"):
            assert torch.count_nonzero(fit.action_decoder) == 0
            assert torch.count_nonzero(details["action_effects"]) == 0
        if mode in ("nocorrect", "anchoronly"):
            assert torch.count_nonzero(fit.observation_decoder) == 0
            assert torch.count_nonzero(details["corrections"]) == 0


def test_norisk_and_norho_are_exact_unit_controls() -> None:
    clean, observed, actions = _synthetic()
    risk = fit_cvpf(clean, observed, actions, mode="norisk")
    assert torch.equal(risk.action_weight, torch.ones_like(risk.action_weight))
    assert torch.equal(
        risk.observation_weight, torch.ones_like(risk.observation_weight))
    rho = fit_cvpf(clean, observed, actions, mode="norho")
    assert torch.equal(rho.action_rho, torch.ones_like(rho.action_rho))
    assert torch.equal(rho.observation_rho, torch.ones_like(rho.observation_rho))


def test_action_schema_padding_is_exact_zero() -> None:
    clean, observed, actions = _synthetic(dimension=7, action_dim=2)
    fit = fit_cvpf(clean, observed, actions, mode="norisk")
    assert torch.count_nonzero(fit.action_encoder[2:]) == 0
    assert torch.count_nonzero(fit.action_decoder[:, :, 2:]) == 0
    assert torch.count_nonzero(fit.action_shift[2:]) == 0
    assert torch.count_nonzero(fit.action_shift[:, 2:]) == 0
    assert fit.receipts["action_exact_zero_padding_max_abs"] == 0.0


def test_projected_shifts_are_bounded_and_closure_is_audited() -> None:
    clean, observed, actions = _synthetic()
    fit = fit_cvpf(clean, observed, actions, mode="norisk")
    for role in ("anchor", "action", "observation"):
        matrix = getattr(fit, f"{role}_shift")
        assert float(torch.linalg.matrix_norm(matrix, ord=2)) <= 1.0 + 1e-10
        closure = fit.receipts[f"{role}_shift_closure_relative"]
        assert 0.0 <= closure <= 1.0 + 1e-10
        assert fit.receipts[f"{role}_shift_nonexpansive"] is True


def test_fixed_stream_state_has_exactly_three_d_float_blocks() -> None:
    clean, observed, actions = _synthetic()
    memory = CVPFv15Memory.from_fit(fit_cvpf(clean, observed, actions))
    state = memory.initial_state(observed[:4, 0])
    assert isinstance(state, CVPFState)
    assert sum(value.shape[-1] for value in state) == 3 * clean.shape[-1]
    assert memory.state_dim == 3 * clean.shape[-1]
    assert memory.parameter_count() == 0
    assert list(memory.parameters()) == []


def test_state_dict_roundtrip_preserves_outputs_and_receipts() -> None:
    clean, observed, actions = _synthetic()
    fit = fit_cvpf(clean, observed, actions)
    first = CVPFv15Memory.from_fit(fit)
    second = CVPFv15Memory(
        clean.shape[-1], actions.shape[-1], clean.shape[1] - 1, mode="full",
        dtype=first.output_mean.dtype)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    assert torch.equal(first(observed[:3], actions[:3]), second(observed[:3], actions[:3]))
    assert first.diagnostics() == second.diagnostics()
    assert CVPFv15Memory is CrossViewPredictiveFiltrationMemory


def test_fit_rejects_nonidentical_initial_views() -> None:
    clean, observed, actions = _synthetic()
    observed = observed.clone()
    observed[0, 0, 0] += 1e-4
    try:
        fit_cvpf(clean, observed, actions)
    except ValueError as error:
        assert "initial observation" in str(error)
    else:
        raise AssertionError("misaligned initial paired view was accepted")


def test_fit_rejects_unknown_mode_and_bad_action_shape() -> None:
    clean, observed, actions = _synthetic()
    try:
        fit_cvpf(clean, observed, actions, mode="unknown")
    except ValueError as error:
        assert "unknown CVPF mode" in str(error)
    else:
        raise AssertionError("unknown mode accepted")
    try:
        fit_cvpf(clean, observed, actions[:, :-1])
    except ValueError as error:
        assert "actions must align" in str(error)
    else:
        raise AssertionError("bad action alignment accepted")


def main() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} CVPF-v15 synthetic tests passed.")


if __name__ == "__main__":
    main()
