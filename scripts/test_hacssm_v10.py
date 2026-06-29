#!/usr/bin/env python3
"""Focused architectural and integration invariants for ORBIT-v10."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lewm.models.encoder import ViTTinyEncoder
from lewm.models.memory import OrthogonalRecurrentBeliefMemory as V10Memory
from lewm.models.memory_model import MemoryLeWorldModel


MODE_MAP = {
    "orbitv10": "orthogonal",
    "orbitv10_noaction": "noaction",
    "orbitv10_additive": "additive",
    "orbitv10_scaled": "scaled",
    "orbitv10_static": "static",
}


def _randomize(memory: V10Memory, scale: float = 0.2) -> None:
    generator = torch.Generator().manual_seed(1002)
    with torch.no_grad():
        for parameter in memory.parameters():
            parameter.copy_(scale * torch.randn(
                parameter.shape, generator=generator, dtype=parameter.dtype,
                device=parameter.device))


def make_precomputed_model(mode: str = "orbitv10") -> MemoryLeWorldModel:
    return MemoryLeWorldModel(
        img_size=8,
        patch_size=4,
        embed_dim=8,
        action_dim=3,
        encoder_layers=1,
        encoder_heads=2,
        predictor_layers=1,
        predictor_heads=2,
        history_len=2,
        dropout=0.0,
        encoder_norm="none",
        predictor_norm="none",
        sigreg_projections=8,
        encoder_type="precomputed",
        memory_impl=mode,
        memory_mode="both",
        # V10 must remain a two-term model even if legacy caller plumbing is nonzero.
        hier_loss_weight=0.2,
    )


def test_parameter_contract_and_common_schema() -> None:
    memories = [V10Memory(128, 6, mode=mode) for mode in MODE_MAP.values()]
    expected = 2 * 128**2 + 2 * 6 * 128 + 2 * 128 + 2
    assert expected == 34_562
    assert V10Memory.expected_parameter_count(128, 6) == expected
    assert all(memory.parameter_count() == expected for memory in memories)
    assert all(tuple(memory.W_a.weight.shape) == (256, 6) for memory in memories)
    assert all(memory.n_pairs == 64 for memory in memories)
    assert all(
        torch.allclose(
            torch.sigmoid(memory.gate_bias), torch.tensor(0.11920292),
            atol=1e-7, rtol=0.0)
        for memory in memories)

    signatures = [
        [(name, tuple(value.shape)) for name, value in memory.state_dict().items()]
        for memory in memories
    ]
    assert all(signature == signatures[0] for signature in signatures[1:])
    assert {name for name, _ in memories[0].named_parameters()} == {
        "w_z", "w_e", "gate_bias", "shrink_logit",
        "W_x.weight", "W_a.weight", "W_o.weight",
    }


def test_identity_initialization_and_residual_fusion() -> None:
    torch.manual_seed(1003)
    state = torch.randn(4, 8)
    actions = torch.randn(4, 3)
    z = torch.randn(4, 7, 8)
    sequence_actions = torch.randn(4, 6, 3)
    for mode in MODE_MAP.values():
        memory = V10Memory(8, 3, mode=mode)
        prior = memory.transition(state, actions)
        assert torch.equal(prior, state), mode
        mixed = memory(z, sequence_actions)
        assert torch.equal(memory.fuse(z, mixed), z), mode


def test_exact_orthogonal_transport_and_noncommuting_pairings() -> None:
    torch.manual_seed(1004)
    memory = V10Memory(8, 3, mode="orthogonal")
    with torch.no_grad():
        memory.W_a.weight.normal_(std=0.35)
    state = torch.randn(32, 8)
    action_a = torch.randn(32, 3)
    action_b = torch.randn(32, 3)
    prior, details = memory.transition(state, action_a, return_details=True)

    assert torch.allclose(prior.norm(dim=-1), state.norm(dim=-1), atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        details["block_norm_sq"], torch.ones_like(details["block_norm_sq"]),
        atol=2e-6, rtol=0.0)
    assert float(details["orthogonality_error"].detach().max()) <= 2e-6
    assert torch.allclose(
        details["transport_norm_ratio"],
        torch.ones_like(details["transport_norm_ratio"]), atol=2e-6, rtol=2e-6)

    ab = memory.transition(memory.transition(state, action_a), action_b)
    ba = memory.transition(memory.transition(state, action_b), action_a)
    assert float((ab - ba).detach().abs().max()) > 1e-4


def test_transport_falsification_controls_are_functional() -> None:
    torch.manual_seed(1005)
    state = torch.randn(5, 8)
    actions = torch.randn(5, 3)

    full = V10Memory(8, 3, mode="orthogonal")
    _randomize(full)
    noaction = V10Memory(8, 3, mode="noaction")
    additive = V10Memory(8, 3, mode="additive")
    scaled = V10Memory(8, 3, mode="scaled")
    for control in (noaction, additive, scaled):
        control.load_state_dict(full.state_dict(), strict=True)

    assert torch.equal(noaction.transition(state, actions), state)
    assert torch.equal(
        noaction.transition(state, actions),
        noaction.transition(state, torch.zeros_like(actions)))
    additive_prior, additive_details = additive.transition(
        state, actions, return_details=True)
    scaled_prior, scaled_details = scaled.transition(
        state, actions, return_details=True)
    full_prior = full.transition(state, actions)
    assert not bool(additive_details["orthogonality_applicable"].any())
    assert bool(scaled_details["orthogonality_applicable"].all())
    assert not torch.allclose(additive_prior, full_prior)
    assert not torch.allclose(scaled_prior, full_prior)
    assert float((scaled_prior.norm(dim=-1) - state.norm(dim=-1)).detach().abs().max()) > 1e-4

    # Every action-dependent mode is identity under the explicit zero-action intervention.
    for memory in (full, additive, scaled):
        assert torch.equal(
            memory.transition(state, actions, action_override=0.0), state)


def test_forward_details_and_streaming_step_agree() -> None:
    torch.manual_seed(1006)
    memory = V10Memory(8, 3)
    _randomize(memory, scale=0.1)
    z = torch.randn(3, 11, 8)
    actions = torch.randn(3, 10, 3)
    mixed, details = memory(z, actions, return_details=True)

    assert mixed.shape == (3, 11, 8)
    assert details["x"].shape == (3, 11, 8)
    assert details["priors"].shape == (3, 11, 8)
    assert details["states"].shape == (3, 11, 8)
    assert details["gates"].shape == (3, 11, 1)
    for key in ("rotation_cos", "rotation_sin", "block_norm_sq"):
        assert details[key].shape == (3, 11, 2, 4)
    assert details["orthogonality_error"].shape == (3, 11)
    assert details["orthogonality_applicable"].shape == (3, 11)
    assert not bool(details["orthogonality_applicable"][:, 0].any())
    assert bool(details["orthogonality_applicable"][:, 1:].all())
    assert details["transport_norm_ratio"].shape == (3, 11)
    assert all(torch.isfinite(value).all() for value in details.values())
    assert bool((details["gates"] >= 0).all()) and bool((details["gates"] <= 1).all())

    state = memory.W_x(z[:, 0])
    streamed = [memory._rms_norm(state)]
    streamed_states = [state]
    for t in range(1, z.shape[1]):
        read, state, _ = memory.step(
            state, z[:, t], actions[:, t - 1], return_details=True)
        streamed.append(read)
        streamed_states.append(state)
    assert torch.allclose(mixed, torch.stack(streamed, dim=1), atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        details["states"], torch.stack(streamed_states, dim=1), atol=2e-6, rtol=2e-6)


def test_gate_and_action_overrides_are_exact() -> None:
    torch.manual_seed(1007)
    memory = V10Memory(8, 3)
    _randomize(memory, scale=0.1)
    z = torch.randn(2, 9, 8)
    actions = torch.randn(2, 8, 3)

    _, closed = memory(z, actions, gate_override=0.0, return_details=True)
    _, opened = memory(z, actions, gate_override=1.0, return_details=True)
    assert torch.equal(closed["states"][:, 1:], closed["priors"][:, 1:])
    assert torch.allclose(opened["states"], opened["x"], atol=2e-7, rtol=0.0)

    _, zero_action = memory(
        z, actions, action_override=0.0, gate_override=0.0, return_details=True)
    expected = zero_action["states"][:, :1].expand_as(zero_action["states"])
    assert torch.equal(zero_action["states"], expected)

    try:
        memory(z, actions, gate_override=1.1)
    except ValueError as exc:
        assert "[0,1]" in str(exc)
    else:
        raise AssertionError("invalid gate override was accepted")


def test_static_gate_is_input_independent() -> None:
    torch.manual_seed(1008)
    dynamic = V10Memory(8, 3, mode="orthogonal")
    _randomize(dynamic, scale=0.25)
    static = V10Memory(8, 3, mode="static")
    static.load_state_dict(dynamic.state_dict(), strict=True)
    z = torch.randn(4, 13, 8)
    actions = torch.randn(4, 12, 3)
    _, dynamic_details = dynamic(z, actions, return_details=True)
    _, static_details = static(z, actions, return_details=True)
    expected = torch.sigmoid(static.gate_bias).view(1, 1, 1).expand(4, 13, 1)
    assert torch.equal(static_details["gates"], expected)
    assert not torch.allclose(dynamic_details["gates"], expected)


def test_causal_encoder_norm_has_no_batch_peer_dependency() -> None:
    torch.manual_seed(1009)
    encoder = ViTTinyEncoder(
        img_size=8, patch_size=4, embed_dim=8, num_layers=1, num_heads=2,
        dropout=0.0, encoder_norm="none")
    encoder.eval()
    frame = torch.randn(1, 3, 8, 8)
    peers = torch.randn(5, 3, 8, 8)
    alone = encoder(frame)
    batched = encoder(torch.cat((frame, peers), dim=0))[:1]
    assert torch.allclose(alone, batched, atol=1e-7, rtol=1e-6)
    assert isinstance(encoder.projector[1], nn.Identity)

    legacy = ViTTinyEncoder(
        img_size=8, patch_size=4, embed_dim=8, num_layers=1, num_heads=2,
        dropout=0.0)
    assert isinstance(legacy.projector[1], nn.BatchNorm1d)
    try:
        ViTTinyEncoder(
            img_size=8, patch_size=4, embed_dim=8, num_layers=1, num_heads=2,
            encoder_norm="future")
    except ValueError as exc:
        assert "encoder_norm" in str(exc)
    else:
        raise AssertionError("invalid encoder normalization was accepted")


def test_end_to_end_model_uses_only_lewm_two_term_loss() -> None:
    torch.manual_seed(1010)
    model = MemoryLeWorldModel(
        img_size=8, patch_size=4, embed_dim=8, action_dim=3,
        encoder_layers=1, encoder_heads=2, predictor_layers=1, predictor_heads=2,
        history_len=2, dropout=0.0, encoder_norm="none", predictor_norm="none",
        sigreg_projections=8, encoder_type="vit", memory_impl="orbitv10",
        memory_mode="both", hier_loss_weight=0.2)
    observations = torch.randn(3, 6, 3, 8, 8)
    clean_targets = torch.randn(3, 6, 3, 8, 8)
    actions = torch.randn(3, 5, 3)
    losses = model.compute_loss(
        observations, actions, target_observations=clean_targets)
    assert set(losses) == {
        "loss", "pred_loss", "sigreg_loss", "pred_loss_all_valid",
    }
    assert torch.equal(
        losses["loss"], losses["pred_loss"] + model.sigreg_lambda * losses["sigreg_loss"])
    assert not any("teacher" in name or "hier_" in name for name in model.state_dict())
    losses["loss"].backward()
    assert model.encoder.patch_embed.weight.grad is not None
    assert torch.isfinite(model.encoder.patch_embed.weight.grad).all()
    assert model.mem_orbitv10.W_o.weight.grad is not None
    assert torch.isfinite(model.mem_orbitv10.W_o.weight.grad).all()

    with torch.no_grad():
        model.mem_orbitv10.W_o.weight.copy_(torch.eye(8))
        influence = model.memory_influence(observations, actions)
    assert set(influence) == {"pred_full", "infl_all"}
    assert all(torch.isfinite(value).all() for value in influence.values())


if __name__ == "__main__":
    tests = (
        test_parameter_contract_and_common_schema,
        test_identity_initialization_and_residual_fusion,
        test_exact_orthogonal_transport_and_noncommuting_pairings,
        test_transport_falsification_controls_are_functional,
        test_forward_details_and_streaming_step_agree,
        test_gate_and_action_overrides_are_exact,
        test_static_gate_is_input_independent,
        test_causal_encoder_norm_has_no_batch_peer_dependency,
        test_end_to_end_model_uses_only_lewm_two_term_loss,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
    print(f"All {len(tests)} ORBIT-v10 tests passed.")
