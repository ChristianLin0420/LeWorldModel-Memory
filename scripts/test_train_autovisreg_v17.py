#!/usr/bin/env python3
"""Focused objective and gradient-composer tests for AutoVISReg-W2 V17."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.train_autovisreg_v17 as v17
import scripts.train_subjepa_v16 as v16


def _args(design: str) -> argparse.Namespace:
    return argparse.Namespace(
        design=design,
        img_size=16,
        patch_size=8,
        embed_dim=32,
        encoder_layers=1,
        encoder_heads=4,
        predictor_layers=1,
        predictor_heads=4,
        history_len=3,
        dropout=0.0,
        sigreg_lambda=0.1,
        sigreg_projections=512,
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(17020)
    clean = torch.rand(4, 4, 3, 16, 16)
    observed = clean.clone()
    observed[:, 2] = 0.0
    actions = torch.randn(4, 3, 2)
    return observed, clean, actions


class _ToyWorld(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.vector = nn.Parameter(torch.tensor([0.3, -0.2]))
        self.head = nn.Parameter(torch.tensor(0.4))


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.world = _ToyWorld()


def _toy_losses(model: _ToyModel) -> tuple[torch.Tensor, torch.Tensor]:
    vector = model.world.encoder.vector
    prediction = vector[0] + 2.0 * model.world.head
    # Negative cosine with prediction, but not antiparallel.
    regularizer = -0.5 * vector[0] + vector[1]
    return prediction, regularizer


def test_registry_and_metadata_freeze_the_narrow_factorial() -> None:
    assert v17.DESIGNS == (
        "autovisreg_none", "vicreg_none",
        "autovisreg_ssm", "vicreg_ssm",
        "autovisreg_hacssmv8", "vicreg_hacssmv8",
    )
    assert v17.parse_design("autovisreg_hacssmv8") == (
        "autovisreg", "hacssmv8")
    metadata = v17.design_metadata("autovisreg_none", 32)
    assert metadata["new_ssl_tunable_hyperparameters"] == 0
    assert metadata["projection_count"] == 64
    assert metadata["projection_hyperparameter_exposed"] is False
    assert metadata["regularizer_family"] == (
        "gaussian_w2_uniformity_plus_self_paced_visreg_shape")
    assert metadata["new_memory_architecture"] is False


def test_vicreg_control_host_matches_v16_same_seed() -> None:
    torch.manual_seed(17021)
    candidate = v17.build_model(_args("vicreg_none"), action_dim=2)
    torch.manual_seed(17021)
    reference = v16.build_model(_args("vicreg_none"), action_dim=2)
    candidate_state = candidate.world.state_dict()
    reference_state = reference.world.state_dict()
    assert candidate_state.keys() == reference_state.keys()
    for key in candidate_state:
        assert torch.equal(candidate_state[key], reference_state[key]), key


def test_every_design_is_finite_and_differentiable() -> None:
    observed, clean, actions = _batch()
    for design in v17.DESIGNS:
        torch.manual_seed(17022)
        model = v17.build_model(_args(design), action_dim=2)
        losses = v17.compute_losses(model, observed, clean, actions)
        assert set(v17.HISTORY_KEYS[:9]) <= set(losses)
        assert all(bool(value.isfinite()) for value in losses.values())
        if design.startswith("autovisreg_"):
            diagnostics = v17.compose_adaptive_gradients(
                model, losses["predictive_loss"], losses["regularizer_loss"])
            assert all(torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()
                       if parameter.grad is not None)
            assert diagnostics["gradient_regularizer_norm"] > 0.0
        else:
            losses["loss"].backward()
        assert any(parameter.grad is not None
                   for parameter in model.world.encoder.parameters())


def test_wasserstein_term_detects_dimensional_collapse() -> None:
    samples, dimension = 64, 8
    torch.manual_seed(17023)
    scalar = torch.randn(samples, 1)
    rank_one = scalar.expand(samples, dimension).reshape(8, 8, dimension)
    isotropic = torch.randn(samples, dimension)
    isotropic = (isotropic - isotropic.mean(0)) / isotropic.std(0, unbiased=False)
    isotropic = isotropic.reshape(8, 8, dimension)
    torch.manual_seed(17024)
    collapsed = v17.visreg_terms(rank_one)
    torch.manual_seed(17024)
    full = v17.visreg_terms(isotropic)
    assert float(collapsed["wasserstein_loss"]) > float(full["wasserstein_loss"]) + 0.5


def test_near_collapse_gradient_is_finite_and_expands_variance() -> None:
    torch.manual_seed(17025)
    embeddings = (1.0e-4 * torch.randn(16, 4, 8)).requires_grad_()
    torch.manual_seed(17026)
    loss = v17.visreg_terms(embeddings)["regularizer_loss"]
    gradient = torch.autograd.grad(loss, embeddings)[0]
    assert bool(gradient.isfinite().all())
    before = embeddings.detach().var(dim=(0, 1), unbiased=False).mean()
    after_embeddings = embeddings.detach() - 0.1 * gradient
    after = after_embeddings.var(dim=(0, 1), unbiased=False).mean()
    assert float(after) > float(before)


def test_gradient_bisector_is_invariant_to_regularizer_rescaling() -> None:
    first = _ToyModel()
    second = copy.deepcopy(first)
    pred_a, reg_a = _toy_losses(first)
    pred_b, reg_b = _toy_losses(second)
    v17.compose_adaptive_gradients(first, pred_a, reg_a)
    v17.compose_adaptive_gradients(second, pred_b, 37.0 * reg_b)
    for parameter_a, parameter_b in zip(
            first.parameters(), second.parameters(), strict=True):
        assert parameter_a.grad is not None and parameter_b.grad is not None
        assert torch.allclose(parameter_a.grad, parameter_b.grad, atol=1e-6, rtol=1e-6)


def test_gradient_bisector_is_common_descent_and_preserves_nonencoder() -> None:
    model = _ToyModel()
    prediction, regularizer = _toy_losses(model)
    prediction_gradient = torch.autograd.grad(
        prediction, model.world.encoder.vector, retain_graph=True)[0]
    regularizer_gradient = torch.autograd.grad(
        regularizer, model.world.encoder.vector, retain_graph=True)[0]
    diagnostics = v17.compose_adaptive_gradients(model, prediction, regularizer)
    combined = model.world.encoder.vector.grad
    assert combined is not None
    assert float(combined @ prediction_gradient) > 0.0
    assert float(combined @ regularizer_gradient) > 0.0
    assert model.world.head.grad is not None
    assert torch.equal(model.world.head.grad, torch.tensor(2.0))
    assert diagnostics["gradient_conflict"] == 1.0


def test_zero_regularizer_gradient_falls_back_to_prediction() -> None:
    model = _ToyModel()
    prediction, _ = _toy_losses(model)
    expected = torch.autograd.grad(
        prediction,
        tuple(model.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    diagnostics = v17.compose_adaptive_gradients(
        model, prediction, prediction.new_zeros(()))
    for parameter, gradient in zip(model.parameters(), expected, strict=True):
        if gradient is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, gradient)
    assert diagnostics["gradient_regularizer_norm"] == 0.0


def main() -> None:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} AutoVISReg-W2 V17 trainer tests")


if __name__ == "__main__":
    main()
