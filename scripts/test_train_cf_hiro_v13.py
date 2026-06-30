#!/usr/bin/env python3
"""Objective, fit-payload, and experiment metadata tests for the V13 trainer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.test_cf_hiro_v13_integration import _synthetic, _world
import scripts.train_cf_hiro_v13 as train
import scripts.train_siro_v12 as v12


def _fit_and_model(mode: str = "cfhirov13"):
    clean, observed, actions = _synthetic()
    fit = train._fit_candidate(clean, observed, actions, mode)
    model = train.CFHIROExperimentModel(_world(mode, fit.state_matrix.shape[0]))
    model.world.mem_cfhirov13.install_fit(fit)
    return fit, model


def test_candidate_mode_reaches_fit_and_deployment_receipts() -> None:
    for design, expected in train.CORE_MODES.items():
        fit, model = _fit_and_model(design)
        assert fit.receipts["fit_mode"] == expected
        assert model.world.mem_cfhirov13.mode == expected
        assert fit.receipts["fold_agreement_mode"] == (
            "unit" if expected == "noshrink" else "empirical_bayes")
        assert fit.receipts["transition_deployment"] == (
            "triangular" if expected == "triangular" else "normal")


def test_one_token_objective_has_no_memory_term_and_backpropagates_end_to_end() -> None:
    _, model = _fit_and_model()
    generator = torch.Generator().manual_seed(13_888)
    observed = torch.rand(2, 7, 3, 8, 8, generator=generator)
    clean = torch.rand(2, 7, 3, 8, 8, generator=generator)
    actions = torch.randn(2, 6, 2, generator=generator)
    original = v12.memory_representations
    try:
        v12.memory_representations = train.memory_representations
        losses = v12.compute_siro_losses(model, observed, clean, actions)
    finally:
        v12.memory_representations = original
    assert set(losses) == {
        "loss", "predictive_loss", "context_loss", "variance_loss", "covariance_loss"}
    assert torch.equal(losses["context_loss"], losses["predictive_loss"])
    assert torch.allclose(
        losses["loss"], losses["predictive_loss"]
        + losses["variance_loss"] + losses["covariance_loss"])
    losses["loss"].backward()
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for parameter in model.world.encoder.parameters())
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for parameter in model.world.predictor.parameters())
    memory = model.world.mem_cfhirov13
    assert list(memory.parameters()) == []
    assert all(not buffer.requires_grad for buffer in memory.buffers())


def test_complete_operator_payload_and_nested_scalar_receipts() -> None:
    fit, _ = _fit_and_model()
    payload = train.operator_fit_payload(fit)
    expected = {
        "state_matrix", "action_matrix", "read_matrix", "process_covariance",
        "measurement_covariance", "initial_covariance", "initial_map",
        "output_mean", "action_mean", "steady_prior_covariance", "steady_gain",
        "markov_even", "markov_odd", "fold_agreement", "receipts",
    }
    assert set(payload) == expected
    for key in expected - {"receipts"}:
        assert isinstance(payload[key], torch.Tensor)
        assert payload[key].device.type == "cpu"
        assert payload[key].dtype == torch.float32
    scalars = train.scalar_fit_receipts({
        "outer": {"value": 3.0, "flag": True}, "label": "train_only"})
    assert scalars == {
        "cf_hiro_fit_outer_value": 3.0,
        "cf_hiro_fit_outer_flag": True,
        "cf_hiro_fit_label": "train_only",
    }


def test_metadata_discloses_every_intervention_and_no_crossfit_claim() -> None:
    for design, mode in train.CORE_MODES.items():
        metadata = train.design_metadata(design)
        assert metadata["variant"] == mode
        assert metadata["memory_gradient_parameter_count"] == 0
        assert metadata["cross_fitted_claim"] is False
        assert metadata["validation_used_for_fit"] is False
        assert metadata["predictor_fusion"].startswith("direct_posterior")
    assert train.design_metadata("cfhirov13_noaction")["action_policy"] == "zero"
    assert train.design_metadata("cfhirov13_nocorrect")["correction_policy"] == "zero"
    assert train.design_metadata("cfhirov13_fullanchor")["anchor_policy"] == "full_z0"


def test_open_loop_diagnostic_casts_fp32_actions_to_amp_state_dtype() -> None:
    generator = torch.Generator().manual_seed(13_889)
    state = torch.randn(2, 3, 5, generator=generator).to(torch.bfloat16)
    action = torch.randn(2, 3, 2, generator=generator, dtype=torch.float32)
    transition = torch.randn(5, 5, generator=generator).to(torch.bfloat16)
    action_map = torch.randn(5, 2, generator=generator).to(torch.bfloat16)
    action_mean = torch.randn(2, generator=generator).to(torch.bfloat16)
    result = train._open_loop_transition(
        state, action, transition, action_map, action_mean, noaction=False)
    expected = (
        state @ transition.T
        + (action.to(torch.bfloat16) - action_mean) @ action_map.T)
    assert result.dtype == torch.bfloat16
    assert torch.equal(result, expected)
    noaction = train._open_loop_transition(
        state, action, transition, action_map, action_mean, noaction=True)
    assert torch.equal(noaction, state @ transition.T)


def main() -> None:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} CF-HIRO-v13 trainer tests passed.")


if __name__ == "__main__":
    main()
