#!/usr/bin/env python3
"""Focused unit tests for the isolated SIRO-v12 memory primitive."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.siro import SIROv12Memory, StableIdentifiedResidualObserverMemory
from lewm.models.memory_model import MemoryLeWorldModel


def _operators(dimension: int, action_dim: int):
    diagonal = torch.linspace(0.35, 0.85, dimension, dtype=torch.float64)
    A = torch.diag(diagonal)
    generator = torch.Generator().manual_seed(12_012)
    B = torch.randn(
        dimension, action_dim, generator=generator, dtype=torch.float64) * 0.12
    b = torch.linspace(-0.04, 0.05, dimension, dtype=torch.float64)
    K = torch.diag(torch.linspace(0.2, 0.8, dimension, dtype=torch.float64))
    R = torch.diag(torch.linspace(0.25, 0.9, dimension, dtype=torch.float64))
    mu_c = torch.linspace(-0.03, 0.02, dimension, dtype=torch.float64)
    mu_o = torch.linspace(0.01, -0.02, dimension, dtype=torch.float64)
    return {
        "identified_A": A,
        "action_B": B,
        "drift_b": b,
        "action_read_R": R,
        "lmmse_K": K,
        "clean_innovation_mean": mu_c,
        "observed_innovation_mean": mu_o,
        "receipts": {"fit_samples": 1234, "oas_shrinkage": 0.17},
    }


def _memory(mode: str = "full", dimension: int = 6,
            action_dim: int = 3) -> StableIdentifiedResidualObserverMemory:
    memory = StableIdentifiedResidualObserverMemory(dimension, action_dim, mode)
    memory.install_fitted_operators(**_operators(dimension, action_dim))
    return memory


def _world(memory_impl: str) -> MemoryLeWorldModel:
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
        encoder_norm="causal",
        predictor_norm="none",
        sigreg_projections=8,
        encoder_type="precomputed",
        memory_impl=memory_impl,
        memory_mode="both",
        hier_loss_weight=0.0,
    )


def test_common_schema_counts_alias_and_no_parameters() -> None:
    dimension, action_dim = 6, 3
    expected_names = {
        "identified_A", "action_B", "drift_b", "action_read_R", "lmmse_K",
        "clean_innovation_mean", "observed_innovation_mean", "fit_updates",
        "operators_installed",
    }
    schemas = []
    for mode in sorted(StableIdentifiedResidualObserverMemory.MODES):
        memory = _memory(mode, dimension, action_dim)
        assert expected_names <= dict(memory.named_buffers()).keys()
        assert list(memory.parameters()) == []
        assert memory.parameter_count() == 0
        schemas.append(tuple(memory.state_dict().keys()))
        assert memory.expected_streaming_scalar_count(dimension) == 3 * dimension
        assert memory.expected_fitted_scalar_count(dimension, action_dim) == (
            3 * dimension * dimension + dimension * action_dim + 3 * dimension)
    assert all(schema == schemas[0] for schema in schemas[1:])
    assert SIROv12Memory is StableIdentifiedResidualObserverMemory


def test_memory_lewm_registry_aliases_and_direct_fusion() -> None:
    aliases = {
        "sirov12": "full",
        "sirov12_spectralshrink": "spectralshrink",
        "sirov12_identityA": "identityA",
        "sirov12_identityK": "identityK",
        "sirov12_noaction": "noaction",
        "sirov12_noanchor": "noanchor",
    }
    worlds = {name: _world(name) for name in aliases}
    assert all(world.mem_sirov12.mode == aliases[name]
               for name, world in worlds.items())
    schemas = [tuple(world.mem_sirov12.state_dict()) for world in worlds.values()]
    assert all(schema == schemas[0] for schema in schemas[1:])

    z = torch.randn(3, 7, 8)
    actions = torch.randn(3, 6, 3)
    injected, details = worlds["sirov12"]._inject(
        z, actions, return_memory_details=True)
    assert torch.equal(injected, details["reads"])
    assert details["states"].shape == (3, 7, 3, 8)
    assert worlds["sirov12"].horizons()["recurrent_floats"] == 24
    try:
        worlds["sirov12"]._inject(z, actions, gate_override=1.0)
    except ValueError as error:
        assert "no visibility mask or learned gate" in str(error)
    else:
        raise AssertionError("SIRO accepted an unsupported learned-gate override")


def test_transition_algebra_and_mode_interventions() -> None:
    torch.manual_seed(3)
    batch, dimension, action_dim = 4, 6, 3
    state = torch.randn(batch, 3, dimension)
    action = torch.randn(batch, action_dim)
    fitted = _operators(dimension, action_dim)
    eye = torch.eye(dimension)

    for mode in (
            "full", "spectralshrink", "identityA", "identityK", "noaction",
            "noanchor"):
        memory = _memory(mode, dimension, action_dim)
        prior, details = memory.transition(state, action, return_details=True)
        A = eye if mode == "identityA" else fitted["identified_A"].float()
        B = torch.zeros_like(fitted["action_B"].float()) \
            if mode == "noaction" else fitted["action_B"].float()
        expected_c = state[:, 0]
        expected_r = state[:, 1] @ A.T + fitted["drift_b"].float()
        expected_u = state[:, 2] @ A.T + action @ B.T
        assert torch.equal(prior[:, 0], expected_c)
        assert torch.allclose(prior[:, 1], expected_r, atol=1e-6, rtol=1e-6)
        assert torch.allclose(prior[:, 2], expected_u, atol=1e-6, rtol=1e-6)
        assert torch.equal(details["c_prior"], expected_c)
        assert torch.allclose(details["r_prior"], expected_r)
        assert torch.allclose(details["u_prior"], expected_u)

        R = fitted["action_read_R"].float() if mode == "spectralshrink" else eye
        expected_read = expected_c + expected_r + expected_u @ R.T
        assert torch.allclose(memory.read_state(prior), expected_read, atol=1e-6)

    noaction = _memory("noaction", dimension, action_dim)
    zero_u_state = state.clone()
    zero_u_state[:, 2].zero_()
    prior, details = noaction.transition(zero_u_state, action, return_details=True)
    assert torch.count_nonzero(prior[:, 2]) == 0
    assert torch.count_nonzero(details["action_effect"]) == 0
    assert torch.count_nonzero(details["effective_action"]) == 0
    assert torch.equal(noaction.action_B, fitted["action_B"].float())


def test_anchor_is_bit_exact_through_transition_correction_and_rollout() -> None:
    torch.manual_seed(4)
    memory = _memory("full")
    batch, length = 3, 7
    z = torch.randn(batch, length, memory.embed_dim)
    actions = torch.randn(batch, length - 1, memory.action_dim)
    _, details = memory(z, actions, return_details=True)
    expected = z[:, :1].expand(-1, length, -1)
    assert torch.equal(details["c_states"], expected)
    assert torch.equal(details["c_priors"], expected)

    initial = memory.initial_state(z[:, 0])
    rolled = memory.rollout_transition(initial, actions)
    rolled_anchor = z[:, 0].unsqueeze(1).expand(-1, length - 1, -1)
    assert torch.equal(rolled[:, :, 0], rolled_anchor)


def test_noanchor_initialization_recovers_absolute_coordinate_control() -> None:
    torch.manual_seed(6)
    memory = _memory("noanchor")
    z0 = torch.randn(4, memory.embed_dim)
    action = torch.randn(4, memory.action_dim)
    state = memory.initial_state(z0)
    assert torch.count_nonzero(state[:, 0]) == 0
    assert torch.equal(state[:, 1], z0)
    assert torch.count_nonzero(state[:, 2]) == 0
    assert torch.equal(memory.read_state(state), z0)

    prior = memory.transition(state, action)
    expected_r = z0 @ memory.identified_A.T + memory.drift_b
    expected_u = action @ memory.action_B.T
    assert torch.count_nonzero(prior[:, 0]) == 0
    assert torch.allclose(prior[:, 1], expected_r, atol=1e-6)
    assert torch.allclose(prior[:, 2], expected_u, atol=1e-6)
    assert torch.allclose(memory.read_state(prior), expected_r + expected_u, atol=1e-6)


def test_identityA_observation_free_rollout_is_anchor_action_drift_integrator() -> None:
    torch.manual_seed(8)
    memory = _memory("identityA")
    batch, horizon = 2, 6
    z0 = torch.randn(batch, memory.embed_dim)
    actions = torch.randn(batch, horizon, memory.action_dim)
    initial = memory.initial_state(z0)
    rolled = memory.rollout_transition(initial, actions)
    cumulative_action = torch.cumsum(actions @ memory.action_B.T, dim=1)
    step = torch.arange(
        1, horizon + 1, dtype=z0.dtype).view(1, horizon, 1)
    expected_r = step * memory.drift_b.view(1, 1, -1)
    expected_read = z0.unsqueeze(1) + expected_r + cumulative_action
    assert torch.equal(
        rolled[:, :, 0], z0.unsqueeze(1).expand(-1, horizon, -1))
    assert torch.allclose(rolled[:, :, 1], expected_r, atol=1e-6)
    assert torch.allclose(rolled[:, :, 2], cumulative_action, atol=1e-6)
    assert torch.allclose(memory.read_state(rolled), expected_read, atol=1e-6)


def test_identityK_is_exact_raw_innovation_and_writes_only_r() -> None:
    torch.manual_seed(5)
    memory = _memory("identityK")
    state = torch.randn(5, 3, memory.embed_dim)
    action = torch.randn(5, memory.action_dim)
    target = torch.randn(5, memory.embed_dim)
    prior = memory.transition(state, action)
    prior_read = memory.read_state(prior)
    mixed, posterior, details = memory.step(
        state, target, action, return_details=True)
    assert torch.equal(details["correction"], target - prior_read)
    assert torch.equal(posterior[:, 0], prior[:, 0])
    assert torch.equal(posterior[:, 2], prior[:, 2])
    assert torch.allclose(mixed, target, atol=2e-6, rtol=2e-6)
    assert torch.allclose(memory.read_state(posterior), target, atol=2e-6, rtol=2e-6)


def test_full_correction_formula_and_spectral_read() -> None:
    torch.manual_seed(7)
    memory = _memory("spectralshrink")
    state = torch.randn(3, 3, memory.embed_dim)
    action = torch.randn(3, memory.action_dim)
    target = torch.randn(3, memory.embed_dim)
    prior = memory.transition(state, action)
    innovation = target - memory.read_state(prior)
    expected = (memory.clean_innovation_mean
                + (innovation - memory.observed_innovation_mean) @ memory.lmmse_K.T)
    mixed, posterior, details = memory.step(
        state, target, action, return_details=True)
    assert torch.allclose(details["correction"], expected, atol=1e-6)
    assert torch.equal(posterior[:, 0], prior[:, 0])
    assert torch.equal(posterior[:, 2], prior[:, 2])
    assert torch.allclose(posterior[:, 1], prior[:, 1] + expected, atol=1e-6)
    assert torch.allclose(mixed, memory.read_state(posterior), atol=1e-6)


def test_forward_matches_explicit_streaming_and_rollout() -> None:
    torch.manual_seed(11)
    batch, length = 4, 8
    for mode in StableIdentifiedResidualObserverMemory.MODES:
        memory = _memory(mode)
        z = torch.randn(batch, length, memory.embed_dim)
        actions = torch.randn(batch, length - 1, memory.action_dim)
        mixed, details = memory(z, actions, return_details=True)
        state = memory.initial_state(z[:, 0])
        streamed = [memory.read_state(state)]
        states = [state]
        for index in range(1, length):
            value, state = memory.step(state, z[:, index], actions[:, index - 1])
            streamed.append(value)
            states.append(state)
        assert torch.allclose(mixed, torch.stack(streamed, dim=1), atol=2e-6)
        assert torch.allclose(details["states"], torch.stack(states, dim=1), atol=2e-6)
        assert torch.allclose(
            details["reads"], memory.read_state(details["states"]), atol=2e-6)
        assert torch.allclose(
            details["prior_reads"], memory.read_state(details["priors"]), atol=2e-6)

        rolled = memory.rollout_transition(
            details["states"][:, 0], actions, return_details=False)
        rolled_state = details["states"][:, 0]
        explicit = []
        for index in range(length - 1):
            rolled_state = memory.transition(rolled_state, actions[:, index])
            explicit.append(rolled_state)
        assert torch.allclose(rolled, torch.stack(explicit, dim=1), atol=2e-6)


def test_full_three_stream_state_collapses_exactly_to_anchor_plus_sum() -> None:
    """Full SIRO's r/u split records lineage but adds no predictive state capacity."""
    torch.manual_seed(12_021)
    memory = _memory("full")
    z = torch.randn(3, 9, memory.embed_dim)
    actions = torch.randn(3, 8, memory.action_dim)
    mixed, details = memory(z, actions, return_details=True)
    anchor = z[:, 0]
    collapsed = torch.zeros_like(anchor)
    collapsed_reads = [anchor]
    A = memory.identified_A
    B = memory.action_B
    for step in range(actions.shape[1]):
        collapsed_prior = (
            collapsed @ A.T + memory.drift_b
            + actions[:, step] @ B.T)
        innovation = z[:, step + 1] - (anchor + collapsed_prior)
        correction = (
            memory.clean_innovation_mean
            + (innovation - memory.observed_innovation_mean) @ memory.lmmse_K.T)
        collapsed = collapsed_prior + correction
        collapsed_reads.append(anchor + collapsed)
        assert torch.allclose(
            collapsed, details["r_states"][:, step + 1]
            + details["u_states"][:, step + 1], atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        mixed, torch.stack(collapsed_reads, dim=1), atol=2e-6, rtol=2e-6)


def test_action_override_and_empty_rollout_contract() -> None:
    memory = _memory("full")
    state = torch.randn(2, 3, memory.embed_dim)
    action = torch.randn(2, memory.action_dim)
    zero = memory.transition(state, action, action_override=0.0)
    explicit = memory.transition(state, torch.zeros_like(action))
    assert torch.equal(zero, explicit)
    empty = memory.rollout_transition(
        state, torch.empty(2, 0, memory.action_dim))
    assert empty.shape == (2, 0, 3, memory.embed_dim)


def test_state_dict_roundtrip_includes_fitted_receipts() -> None:
    memory = _memory("full")
    state = memory.state_dict()
    assert "_extra_state" in state
    restored = StableIdentifiedResidualObserverMemory(
        memory.embed_dim, memory.action_dim, "full")
    restored.load_state_dict(state)
    for name in (
        "identified_A", "action_B", "drift_b", "action_read_R", "lmmse_K",
        "clean_innovation_mean", "observed_innovation_mean",
    ):
        assert torch.equal(getattr(memory, name), getattr(restored, name))
    assert restored.operator_diagnostics()["fit_samples"] == 1234
    assert restored.operator_diagnostics()["oas_shrinkage"] == 0.17
    assert int(restored.fit_updates) == 1
    assert bool(restored.operators_installed)


def test_rejected_install_is_atomic_and_stability_is_enforced() -> None:
    memory = _memory("full")
    before = {name: value.clone() for name, value in memory.named_buffers()}
    invalid = _operators(memory.embed_dim, memory.action_dim)
    invalid["identified_A"] = torch.eye(memory.embed_dim, dtype=torch.float64) * 1.1
    try:
        memory.install_fitted_operators(**invalid)
    except ValueError as error:
        assert "spectrally stable" in str(error)
    else:
        raise AssertionError("unstable A was accepted")
    for name, value in memory.named_buffers():
        assert torch.equal(value, before[name])


def test_gradients_reach_inputs_but_not_fitted_operators() -> None:
    torch.manual_seed(17)
    memory = _memory("full")
    z = torch.randn(3, 5, memory.embed_dim, requires_grad=True)
    actions = torch.randn(3, 4, memory.action_dim, requires_grad=True)
    mixed = memory(z, actions)
    loss = mixed.square().mean()
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert float(z.grad.abs().sum()) > 0.0
    assert actions.grad is not None and torch.isfinite(actions.grad).all()
    assert float(actions.grad.abs().sum()) > 0.0
    assert list(memory.parameters()) == []
    for _, value in memory.named_buffers():
        assert not value.requires_grad


def test_strict_shape_and_finite_validation() -> None:
    memory = _memory("full")
    good_z = torch.randn(2, 4, memory.embed_dim)
    good_a = torch.randn(2, 3, memory.action_dim)
    bad_cases = (
        lambda: memory(good_z[:, :, :-1], good_a),
        lambda: memory(good_z, good_a[:, :-1]),
        lambda: memory.initial_state(torch.randn(2, memory.embed_dim + 1)),
        lambda: memory.transition(
            torch.randn(2, 3, memory.embed_dim), torch.randn(2, memory.action_dim + 1)),
    )
    for call in bad_cases:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid SIRO input was accepted")
    nonfinite = good_z.clone()
    nonfinite[0, 0, 0] = math.nan
    try:
        memory(nonfinite, good_a)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite SIRO input was accepted")


def main() -> None:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} SIRO-v12 memory tests passed.")


if __name__ == "__main__":
    main()
