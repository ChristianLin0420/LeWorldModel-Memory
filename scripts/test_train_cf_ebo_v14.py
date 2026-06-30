#!/usr/bin/env python3
"""Objective, serialization, metadata, and AMP tests for the V14 trainer."""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.cf_ebo import CF_EBOv14Memory, CEBOFit
from scripts.test_cf_ebo_v14_integration import _synthetic, _world
import scripts.train_cf_ebo_v14 as train
import scripts.train_siro_v12 as v12


def _fit_and_model(design: str = "cfebov14"):
    clean, observed, actions = _synthetic()
    fit = train._fit_candidate(clean, observed, actions, design)
    model = train.CFEBOExperimentModel(_world(design, train._fit_state_dim(fit)))
    model.world.mem_cfebov14.install_fit(fit)
    return fit, model


def test_candidate_mode_reaches_fit_and_deployment() -> None:
    for design, expected in train.CORE_MODES.items():
        fit, model = _fit_and_model(design)
        assert isinstance(fit, CEBOFit)
        assert fit.receipts["fit_mode"] == expected
        assert model.world.mem_cfebov14.mode == expected


def test_exported_exact_control_reliabilities_use_deployed_state() -> None:
    expected = {
        "cfebov14_nocorrect": {"correction": 0.0},
        "cfebov14_noaction": {"action": 0.0},
        "cfebov14_norisk": {"action": 1.0, "correction": 1.0},
    }
    for design, reliabilities in expected.items():
        fit, model = _fit_and_model(design)
        exported = train.scalar_core_diagnostics(model.world.mem_cfebov14)
        for mechanism, value in reliabilities.items():
            assert exported[f"cf_ebo_core_{mechanism}_reliability"] == value
        assert exported["cf_ebo_core_action_combined_risk_reliability"] \
            == fit.receipts["action_combined_risk_reliability"]
        assert exported["cf_ebo_core_correction_combined_risk_reliability"] \
            == fit.receipts["correction_combined_risk_reliability"]


def test_one_token_objective_has_no_memory_term_and_backpropagates() -> None:
    _, model = _fit_and_model()
    generator = torch.Generator().manual_seed(14_888)
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
    memory = model.world.mem_cfebov14
    assert list(memory.parameters()) == []
    assert all(not buffer.requires_grad for buffer in memory.buffers())


def test_complete_operator_payload_serializes_every_dataclass_field() -> None:
    fit, _ = _fit_and_model()
    payload = train.operator_fit_payload(fit)
    expected = {field.name for field in dataclasses.fields(CEBOFit)}
    assert set(payload) == expected
    assert "receipts" in payload
    for field in dataclasses.fields(CEBOFit):
        original = getattr(fit, field.name)
        serialized = payload[field.name]
        if isinstance(original, torch.Tensor):
            assert isinstance(serialized, torch.Tensor)
            assert serialized.device.type == "cpu"
            if original.is_floating_point():
                assert serialized.dtype == torch.float32
    scalars = train.scalar_fit_receipts({
        "outer": {"value": 3.0, "flag": True}, "label": "train_only"})
    assert scalars == {
        "cf_ebo_fit_outer_value": 3.0,
        "cf_ebo_fit_outer_flag": True,
        "cf_ebo_fit_label": "train_only",
    }


def test_metadata_discloses_all_interventions_and_claim_boundary() -> None:
    for design, mode in train.CORE_MODES.items():
        metadata = train.design_metadata(design)
        assert metadata["variant"] == mode
        assert metadata["memory_gradient_parameter_count"] == 0
        assert metadata["validation_used_for_fit"] is False
        assert metadata["cross_fitted_claim"] is False
        assert metadata["cross_fold_calibration"] is True
        assert metadata["predictor_fusion"].startswith("direct_posterior")
    assert train.design_metadata("cfebov14_noaction")["action_policy"] == "zero"
    assert train.design_metadata("cfebov14_nocorrect")["correction_policy"] == "zero"
    assert train.design_metadata("cfebov14_norisk")["action_policy"] == "unshrunk"
    assert train.design_metadata("cfebov14_norisk")["risk_policy"] == "disabled_both_paths"
    assert train.design_metadata("cfebov14_noenergycap")["energy_cap_policy"] == "disabled"
    assert train.design_metadata("cfebov14_noradial")["radial_policy"] == "disabled"


def test_bfloat16_memory_diagnostics_accept_fp32_actions() -> None:
    fit, model = _fit_and_model()
    memory = model.world.mem_cfebov14.to(torch.bfloat16)
    generator = torch.Generator().manual_seed(14_889)
    observations = torch.randn(
        3, 7, memory.output_dim, generator=generator).to(torch.bfloat16)
    actions = torch.randn(
        3, 6, memory.action_dim, generator=generator, dtype=torch.float32)
    reads, details = memory(observations, actions, return_details=True)
    assert reads.dtype == torch.bfloat16
    assert torch.isfinite(reads).all()
    assert torch.isfinite(details["corrections"]).all()
    state = memory.initial_state(observations[:, 0])
    streamed = [memory.read_state(state)]
    for index in range(1, observations.shape[1]):
        read, state = memory.step(
            state, observations[:, index], actions[:, index - 1])
        streamed.append(read)
    assert torch.equal(reads, torch.stack(streamed, dim=1))


def test_condition_evidence_uses_core_details_and_is_finite() -> None:
    _, model = _fit_and_model()

    class TinyCondition(Dataset):
        def __init__(self) -> None:
            generator = torch.Generator().manual_seed(14_890)
            self.frames = torch.rand(5, 7, 3, 8, 8, generator=generator)
            self.actions = torch.randn(5, 6, 2, generator=generator)

        def __len__(self) -> int:
            return len(self.frames)

        def __getitem__(self, index: int):
            return {"observed": self.frames[index], "actions": self.actions[index]}

    args = type("Args", (), {"batch_size": 2, "num_workers": 0, "seed": 14})()
    metrics = train.condition_correction_evidence(
        model, TinyCondition(), args, torch.device("cpu"), False, label="synthetic")
    expected = {
        "cf_ebo_synthetic_innovation_score_mean",
        "cf_ebo_synthetic_innovation_score_max",
        "cf_ebo_synthetic_radial_gate_mean",
        "cf_ebo_synthetic_radial_gate_min",
        "cf_ebo_synthetic_radial_gate_max",
        "cf_ebo_synthetic_correction_rms",
        "cf_ebo_synthetic_correction_norm_max",
        "cf_ebo_synthetic_correction_energy_max",
        "cf_ebo_synthetic_normalized_innovation_rms",
        "cf_ebo_synthetic_evidence_samples",
    }
    assert set(metrics) == expected
    assert metrics["cf_ebo_synthetic_evidence_samples"] == 5 * 6
    assert all(math.isfinite(float(value)) for value in metrics.values())
    assert 0.0 <= metrics["cf_ebo_synthetic_radial_gate_min"]
    assert metrics["cf_ebo_synthetic_radial_gate_max"] <= 1.0


def test_state_dict_roundtrip_keeps_complete_fitted_memory() -> None:
    fit, model = _fit_and_model()
    memory = model.world.mem_cfebov14
    restored = CF_EBOv14Memory(
        output_dim=memory.output_dim, action_dim=memory.action_dim,
        state_dim=memory.state_dim, mode=memory.mode)
    restored.load_state_dict(memory.state_dict())
    assert restored.mode == memory.mode
    for name, value in memory.named_buffers():
        assert torch.equal(value, dict(restored.named_buffers())[name])
    assert restored.diagnostics()["method"] == memory.diagnostics()["method"]


def main() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} CF-EBO-v14 trainer tests passed.")


if __name__ == "__main__":
    main()
