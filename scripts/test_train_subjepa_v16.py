#!/usr/bin/env python3
"""Focused integration tests for the minimal Sub-JEPA-v16 trainer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_subjepa_v16 import (
    DESIGNS,
    build_model,
    compute_losses,
    design_metadata,
    memory_representations,
    parse_design,
)


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
        sigreg_projections=5,
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(16020)
    clean = torch.rand(4, 4, 3, 16, 16)
    observed = clean.clone()
    observed[:, 2] = 0.0
    actions = torch.randn(4, 3, 2)
    return observed, clean, actions


def test_exact_factorial_and_metadata() -> None:
    assert len(DESIGNS) == 12
    assert len(set(DESIGNS)) == 12
    assert parse_design("fullsig_none") == ("fullsig", "none", 1)
    assert parse_design("subjepa16_ssm") == ("subjepa16", "ssm", 16)
    assert parse_design("subjepa32_hacssmv8") == (
        "subjepa32", "hacssmv8", 32)
    assert parse_design("vicreg_none") == ("vicreg", "none", None)
    metadata = design_metadata("subjepa16_ssm", embed_dim=32)
    assert metadata["subspace_dim"] == 2
    assert metadata["new_memory_architecture"] is False
    assert metadata["observation_correction_branch"] is False
    assert metadata["confirmation_evidence"] is False


def test_builds_every_regularizer_memory_combination() -> None:
    for design in DESIGNS:
        torch.manual_seed(16021)
        model = build_model(_args(design), action_dim=2)
        regularizer, memory, subspaces = parse_design(design)
        if regularizer == "vicreg":
            assert model.world.sigreg.__class__.__name__ == "SIGReg"
        else:
            assert model.world.sigreg.__class__.__name__ == "MultiSubspaceSIGReg"
            assert model.world.sigreg.num_subspaces == subspaces
        expected_impl = "ema" if memory == "none" else memory
        assert model.world.memory_impl == expected_impl
        if memory == "none":
            assert not any(parameter.requires_grad for parameter in model.world.memory.parameters())
            assert not any(parameter.requires_grad for parameter in model.world.fusion.parameters())


def test_no_memory_contract_is_exact() -> None:
    model = build_model(_args("fullsig_none"), action_dim=2)
    z = torch.randn(3, 4, 32)
    actions = torch.randn(3, 3, 2)
    result = memory_representations(model, z, actions)
    assert torch.equal(result["fused"], z)
    assert torch.equal(result["posterior"], z)
    assert torch.equal(result["prior"][:, 0], z[:, 0])
    assert tuple(result["prior"].shape) == tuple(z.shape)


def test_representative_objectives_are_finite_and_differentiable() -> None:
    observed, clean, actions = _batch()
    designs = (
        "fullsig_none",
        "subjepa16_ssm",
        "subjepa32_hacssmv8",
        "vicreg_none",
    )
    for design in designs:
        torch.manual_seed(16022)
        model = build_model(_args(design), action_dim=2)
        losses = compute_losses(model, observed, clean, actions, sigreg_lambda=0.1)
        assert set(losses) == {
            "loss", "predictive_loss", "regularizer_loss", "sigreg_loss",
            "variance_loss", "covariance_loss",
        }
        assert all(bool(value.isfinite()) for value in losses.values())
        losses["loss"].backward()
        encoder_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.world.encoder.parameters()
            if parameter.grad is not None)
        assert encoder_gradient > 0.0
        if design.startswith("vicreg_"):
            assert float(losses["sigreg_loss"].detach()) == 0.0
            assert float((losses["variance_loss"] + losses["covariance_loss"]).detach()) > 0.0
        else:
            assert float(losses["sigreg_loss"].detach()) > 0.0
            assert float(losses["variance_loss"].detach()) == 0.0
            assert float(losses["covariance_loss"].detach()) == 0.0


def main() -> None:
    tests = (
        test_exact_factorial_and_metadata,
        test_builds_every_regularizer_memory_combination,
        test_no_memory_contract_is_exact,
        test_representative_objectives_are_finite_and_differentiable,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} V16 trainer tests")


if __name__ == "__main__":
    main()
