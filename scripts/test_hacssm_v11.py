#!/usr/bin/env python3
"""Focused architectural and integration invariants for KDIO-v11."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.utils.data import Dataset


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lewm.models.memory import KickDriftInnovationObserverMemory as V11Memory
from lewm.models.memory_model import MemoryLeWorldModel
from scripts.train_hacssm_v11 import (
    ACTION_RANKING_MODES,
    DEFAULT_ACTION_RANKING,
    V11ExperimentModel,
    _action_rank_pair_loss,
    _design_metadata,
    _kdio_suffix_objectives,
    _second_order_inverse_inputs,
    aggregate_action_rank_receipts,
    compute_v11_losses,
    evaluate_inverse_action_probe,
    fit_inverse_action_probe,
    parse_args,
)


MODE_MAP = {
    "kdiov11": "full",
    "kdiov11_unconstrained": "unconstrained",
    "kdiov11_fixedscale": "fixedscale",
    "kdiov11_h1": "full",
    "kdiov11_noactionswap": "full",
    "kdiov11_firstorder": "firstorder",
    "kdiov11_nodrift": "nodrift",
    "kdiov11_noaction": "noaction",
    "kdiov11_noautonomy": "noautonomy",
    "kdiov11_noreliability": "noreliability",
    "kdiov11_static": "static",
}


def _randomize(memory: V11Memory, scale: float = 0.15) -> None:
    generator = torch.Generator().manual_seed(11001)
    with torch.no_grad():
        for parameter in memory.parameters():
            parameter.copy_(scale * torch.randn(
                parameter.shape, generator=generator, dtype=parameter.dtype,
                device=parameter.device))


def make_precomputed_model(mode: str = "kdiov11") -> MemoryLeWorldModel:
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
        memory_impl=mode,
        memory_mode="both",
        hier_loss_weight=0.0,
    )


def test_parameter_contract_and_common_schema() -> None:
    memories = [V11Memory(128, 6, mode=mode) for mode in V11Memory.MODES]
    expected = 128**2 + 6 * 128 + 5 * 128 + 4
    assert expected == 17_796
    assert V11Memory.expected_parameter_count(128, 6) == expected
    assert V11Memory.expected_fitted_scalar_count(128, 6) == 8_255
    assert V11Memory.expected_total_scalar_count(128, 6) == 26_051
    assert all(memory.parameter_count() == expected for memory in memories)
    count_receipt = memories[0].horizons()
    assert count_receipt["nominal_optimizer_scalars"] == 17_796
    assert count_receipt["fitted_memory_scalars"] == 8_255
    assert count_receipt["total_memory_scalars"] == 26_051
    assert all(tuple(memory.W_a.weight.shape) == (128, 6) for memory in memories)
    assert all(tuple(memory.W_o.weight.shape) == (128, 128) for memory in memories)
    assert all(tuple(memory.w_q.shape) == (128,) for memory in memories)
    assert all(torch.equal(
        memory.W_a.weight, memories[0].W_a.weight) for memory in memories[1:])
    identity = torch.eye(6)
    assert all(torch.allclose(
        memory.action_frame().T @ memory.action_frame(), identity,
        atol=2e-6, rtol=2e-6)
        for memory in memories)
    assert all(torch.equal(memory.log_action_scale, torch.zeros(()))
               for memory in memories)
    assert all(
        torch.allclose(
            torch.sigmoid(memory.position_gain_bias), torch.tensor(0.11920292),
            atol=1e-7, rtol=0.0)
        for memory in memories)
    assert all(
        torch.allclose(
            memory.observation_gain_biases()[0]
            * memory.observation_gain_biases()[1], torch.tensor(0.01420934),
            atol=1e-7, rtol=0.0)
        for memory in memories)
    signatures = [
        [(name, tuple(value.shape)) for name, value in memory.state_dict().items()]
        for memory in memories
    ]
    assert all(signature == signatures[0] for signature in signatures[1:])
    assert {name for name, _ in memories[0].named_parameters()} == {
        "w_q", "b_f", "position_gain_vector", "velocity_ratio_vector",
        "process_tolerance_vector", "clean_innovation_mean",
        "position_gain_bias", "velocity_ratio_bias", "process_tolerance_bias",
        "innovation_precision_packed",
        "log_action_scale", "W_a.weight", "W_o.weight",
    }
    assert all(torch.equal(
        memory.innovation_precision_factor(), torch.eye(127)) for memory in memories)
    assert all(torch.allclose(
        memory._innovation_contrast @ memory._innovation_contrast.T,
        torch.eye(127), atol=1e-6, rtol=1e-6) for memory in memories)
    assert all(torch.allclose(
        memory._innovation_contrast @ torch.ones(128), torch.zeros(127),
        atol=1e-6, rtol=0.0) for memory in memories)
    try:
        V11Memory(4, 5)
    except ValueError as exc:
        assert "action_dim <= embed_dim" in str(exc)
    else:
        raise AssertionError("KDIO accepted action_dim > embed_dim")


def test_stiefel_kick_initialization_and_residual_identity() -> None:
    torch.manual_seed(11002)
    q = torch.randn(4, 8)
    v = torch.randn(4, 8)
    state = torch.stack((q, v), dim=1)
    action = torch.randn(4, 3)
    z = torch.randn(4, 7, 8)
    actions = torch.randn(4, 6, 3)
    for mode in V11Memory.MODES:
        memory = V11Memory(8, 3, mode=mode)
        prior, details = memory.transition(state, action, return_details=True)
        effective_action = (
            torch.zeros_like(action) if mode == "noaction" else action)
        expected_kick = torch.tanh(torch.nn.functional.linear(
            effective_action, memory.action_frame()))
        assert torch.allclose(
            details["kick"], expected_kick, atol=1e-7, rtol=1e-6), mode
        if mode == "firstorder":
            assert torch.allclose(
                prior[:, 0], q + expected_kick, atol=1e-7, rtol=1e-6)
            assert torch.equal(prior[:, 1], expected_kick)
        elif mode == "nodrift":
            assert torch.equal(prior[:, 0], q)
            assert torch.allclose(
                prior[:, 1], v + expected_kick, atol=1e-7, rtol=1e-6)
        else:
            assert torch.allclose(
                prior[:, 0], q + v + expected_kick, atol=1e-7, rtol=1e-6)
            assert torch.allclose(
                prior[:, 1], v + expected_kick, atol=1e-7, rtol=1e-6)
        mixed = memory(z, actions)
        assert torch.equal(memory.fuse(z, mixed), z), mode


def test_kick_drift_equations_state_dependence_and_inverse() -> None:
    torch.manual_seed(11003)
    memory = V11Memory(8, 3, mode="full")
    _randomize(memory)
    state = torch.randn(16, 2, 8)
    action = torch.randn(16, 3)
    prior, details = memory.transition(state, action, return_details=True)

    q, v = state[:, 0], state[:, 1]
    expected_kick = torch.tanh(
        memory.w_q * memory._rms_norm(q)
        + torch.nn.functional.linear(action, memory.action_frame())
        + memory.b_f)
    expected_v = v + expected_kick
    expected_q = q + expected_v
    assert torch.allclose(details["kick"], expected_kick, atol=1e-7, rtol=1e-6)
    assert torch.allclose(prior[:, 0], expected_q, atol=3e-7, rtol=1e-6)
    assert torch.allclose(prior[:, 1], expected_v, atol=1e-7, rtol=1e-6)
    recovered, inverse_details = memory.inverse_transition(
        prior, action, return_details=True)
    assert torch.allclose(recovered, state, atol=2e-6, rtol=2e-6)
    assert float(inverse_details["inverse_roundtrip_error"].detach().max()) <= 2e-6
    assert bool(details["inverse_applicable"].all())
    assert float(details["inverse_error"].detach().max()) <= 2e-6

    altered = state.clone()
    altered[:, 0] = altered[:, 0].roll(1, dims=-1)
    _, altered_details = memory.transition(altered, action, return_details=True)
    assert not torch.allclose(details["kick"], altered_details["kick"])


def test_scaled_stiefel_free_geometry_fixedscale_gradients_and_cache() -> None:
    torch.manual_seed(11012)
    full = V11Memory(8, 3, mode="full")
    unconstrained = V11Memory(8, 3, mode="unconstrained")
    fixedscale = V11Memory(8, 3, mode="fixedscale")
    with torch.no_grad():
        raw = torch.randn_like(full.W_a.weight)
        raw[:, 0] *= 0.2
        raw[:, 1] *= 1.7
        raw[:, 2] *= 0.6
        full.W_a.weight.copy_(raw)
        full.log_action_scale.copy_(torch.log(torch.tensor(0.4)))
    unconstrained.load_state_dict(full.state_dict(), strict=True)
    fixedscale.load_state_dict(full.state_dict(), strict=True)

    frame = full.action_frame()
    direction = full.action_direction()
    free_direction = unconstrained.action_direction()
    free_frame = unconstrained.action_frame()
    identity = torch.eye(3)
    assert torch.allclose(direction.T @ direction, identity, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        frame.T @ frame, 0.4**2 * identity, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        free_direction.norm(), torch.tensor(3.0).sqrt(), atol=2e-6, rtol=2e-6)
    assert float((
        free_direction.T @ free_direction - identity).abs().max().detach()) > 0.1
    with torch.no_grad():
        unconstrained.W_a.weight.mul_(7.0)
    assert torch.allclose(
        unconstrained.action_direction(), free_direction, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        unconstrained.action_frame(), free_frame, atol=2e-6, rtol=2e-6)
    free_singular = torch.linalg.svdvals(unconstrained.action_frame())
    assert torch.allclose(
        free_singular.square().mean().sqrt(), unconstrained.action_scale(),
        atol=2e-6, rtol=2e-6)
    expected_free = (0.4 * np.sqrt(3.0) * unconstrained.W_a.weight
                     / unconstrained.W_a.weight.norm())
    assert torch.allclose(free_frame, expected_free, atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        fixedscale.action_frame(), fixedscale.action_direction(),
        atol=0.0, rtol=0.0)
    assert torch.equal(fixedscale.action_scale(), torch.ones(()))
    action = torch.randn(10, 3)
    lifted = torch.nn.functional.linear(action, frame)
    assert torch.allclose(
        lifted.norm(dim=-1), 0.4 * action.norm(dim=-1), atol=2e-6, rtol=2e-6)

    state = torch.randn(10, 2, 8)
    full_prior = full.transition(state, action)
    free_prior, free_details = unconstrained.transition(
        state, action, return_details=True)
    free_expected = torch.tanh(
        unconstrained.w_q * unconstrained._rms_norm(state[:, 0])
        + torch.nn.functional.linear(action, free_frame) + unconstrained.b_f)
    assert torch.allclose(
        free_details["kick"], free_expected, atol=1e-7, rtol=1e-6)
    assert not torch.allclose(full_prior, free_prior)

    target = torch.randn_like(full_prior)
    (full_prior - target).square().mean().backward()
    assert full.W_a.weight.grad is not None
    assert torch.isfinite(full.W_a.weight.grad).all()
    assert float(full.W_a.weight.grad.abs().sum()) > 0.0
    assert full.log_action_scale.grad is not None
    assert torch.isfinite(full.log_action_scale.grad)
    assert float(full.log_action_scale.grad.abs()) > 0.0
    fixed_target = torch.randn_like(full_prior)
    (fixedscale.transition(state, action) - fixed_target).square().mean().backward()
    assert fixedscale.W_a.weight.grad is not None
    assert fixedscale.log_action_scale.grad is None

    cached = full.action_frame()
    automatic = full.transition(state, action)
    reused = full.transition(state, action, cached_action_frame=cached)
    assert torch.equal(automatic, reused)
    actions = torch.randn(10, 5, 3)
    with mock.patch.object(
            full, "action_frame", wraps=full.action_frame) as action_frame_call:
        full.rollout_transition(state, actions)
        assert action_frame_call.call_count == 1
    z = torch.randn(10, 6, 8)
    with mock.patch.object(
            full, "action_frame", wraps=full.action_frame) as action_frame_call:
        full(z, actions)
        assert action_frame_call.call_count == 1
    diagnostics = full.horizons()
    assert diagnostics["action_frame_gram_error"] <= 2e-6
    assert abs(diagnostics["action_scale"] - 0.4) <= 2e-6
    assert abs(diagnostics["action_frame_singular_min"] - 0.4) <= 2e-6
    assert abs(diagnostics["action_frame_singular_max"] - 0.4) <= 2e-6
    assert abs(diagnostics["action_frame_condition"] - 1.0) <= 2e-6
    assert diagnostics["action_frame_constrained"]
    assert not unconstrained.horizons()["action_frame_constrained"]


def test_mode_interventions_are_exact_and_same_tensor() -> None:
    torch.manual_seed(11004)
    full = V11Memory(8, 3, mode="full")
    _randomize(full)
    controls = {
        mode: V11Memory(8, 3, mode=mode)
        for mode in V11Memory.MODES - {"full"}
    }
    for memory in controls.values():
        memory.load_state_dict(full.state_dict(), strict=True)
    state = torch.randn(6, 2, 8)
    action = torch.randn(6, 3)

    noaction = controls["noaction"]
    assert torch.equal(
        noaction.transition(state, action),
        noaction.transition(state, torch.zeros_like(action)))
    assert torch.allclose(
        full.transition(state, action, action_override=0.0),
        noaction.transition(state, action), atol=1e-7, rtol=1e-6)

    noautonomy = controls["noautonomy"]
    zero_prior, zero_details = noautonomy.transition(
        state, torch.zeros_like(action), return_details=True)
    assert torch.equal(zero_details["kick"], torch.zeros_like(state[:, 0]))
    assert torch.equal(zero_prior[:, 1], state[:, 1])
    assert torch.equal(zero_prior[:, 0], state[:, 0] + state[:, 1])

    firstorder = controls["firstorder"]
    changed_velocity = state.clone()
    changed_velocity[:, 1] += 100.0
    assert torch.allclose(
        firstorder.transition(state, action),
        firstorder.transition(changed_velocity, action), atol=1e-7, rtol=1e-6)
    try:
        firstorder.inverse_transition(firstorder.transition(state, action), action)
    except ValueError as exc:
        assert "not invertible" in str(exc)
    else:
        raise AssertionError("firstorder unexpectedly exposed an inverse")

    nodrift = controls["nodrift"]
    nodrift_prior = nodrift.transition(state, action)
    assert torch.equal(nodrift_prior[:, 0], state[:, 0])
    assert not torch.allclose(nodrift_prior[:, 1], state[:, 1])
    assert torch.allclose(
        nodrift.inverse_transition(nodrift_prior, action),
        state, atol=2e-6, rtol=2e-6)


def test_rollout_transition_and_reverse_recovery() -> None:
    torch.manual_seed(11005)
    memory = V11Memory(8, 3)
    _randomize(memory, scale=0.08)
    initial = torch.randn(5, 2, 8)
    actions = torch.randn(5, 9, 3)
    rollout, details = memory.rollout_transition(
        initial, actions, return_details=True)
    assert rollout.shape == (5, 9, 2, 8)
    assert details["kick"].shape == (5, 9, 8)
    assert details["inverse_error"].shape == (5, 9)
    for key in (
            "action_effect_norm", "action_tanh_derivative_mean",
            "action_tanh_saturation_proxy"):
        assert details[key].shape == (5, 9), key
        assert torch.isfinite(details[key]).all(), key
    assert bool(details["inverse_applicable"].all())
    assert float(details["inverse_error"].detach().max()) <= 3e-6
    state = initial
    manual = []
    for t in range(actions.shape[1]):
        state = memory.transition(state, actions[:, t])
        manual.append(state)
    assert torch.allclose(rollout, torch.stack(manual, dim=1), atol=2e-6, rtol=2e-6)
    for t in range(actions.shape[1] - 1, -1, -1):
        state = memory.inverse_transition(state, actions[:, t])
    assert torch.allclose(state, initial, atol=4e-6, rtol=4e-6)
    assert memory.read_state(rollout[:, -1]).shape == (5, 8)
    assert memory.rollout_transition(initial, actions[:, :0]).shape == (5, 0, 2, 8)


def test_forward_details_and_streaming_step_agree() -> None:
    torch.manual_seed(11006)
    memory = V11Memory(8, 3)
    _randomize(memory, scale=0.08)
    z = torch.randn(3, 11, 8)
    actions = torch.randn(3, 10, 3)
    mixed, details = memory(z, actions, return_details=True)
    assert mixed.shape == (3, 11, 8)
    assert details["x"].shape == (3, 11, 8)
    assert details["priors"].shape == (3, 11, 2, 8)
    assert details["states"].shape == (3, 11, 2, 8)
    for key in ("q_priors", "v_priors", "q_states", "v_states", "kick"):
        assert details[key].shape == (3, 11, 8), key
    assert details["gates"].shape == (3, 11, 2, 1)
    assert details["q_gates"].shape == (3, 11, 1)
    assert details["v_gates"].shape == (3, 11, 1)
    assert details["inverse_error"].shape == (3, 11)
    assert details["inverse_applicable"].shape == (3, 11)
    for key in (
            "action_effect_norm", "action_tanh_derivative_mean",
            "action_tanh_saturation_proxy"):
        assert details[key].shape == (3, 11), key
    assert not bool(details["inverse_applicable"][:, 0].any())
    assert bool(details["inverse_applicable"][:, 1:].all())
    assert memory.read_state(details["states"]).shape == (3, 11, 8)
    assert all(
        torch.isfinite(value).all()
        for value in details.values()
        if value.is_floating_point())

    state = torch.stack((z[:, 0], torch.zeros_like(z[:, 0])), dim=1)
    streamed = [memory.read_state(state)]
    streamed_states = [state]
    for t in range(1, z.shape[1]):
        read, state, _ = memory.step(
            state, z[:, t], actions[:, t - 1], return_details=True)
        streamed.append(read)
        streamed_states.append(state)
    assert torch.allclose(mixed, torch.stack(streamed, dim=1), atol=2e-6, rtol=2e-6)
    assert torch.allclose(
        details["states"], torch.stack(streamed_states, dim=1),
        atol=2e-6, rtol=2e-6)


def test_gate_and_static_controls() -> None:
    torch.manual_seed(11007)
    dynamic = V11Memory(8, 3, mode="full")
    _randomize(dynamic)
    static = V11Memory(8, 3, mode="static")
    static.load_state_dict(dynamic.state_dict(), strict=True)
    z = torch.randn(4, 9, 8)
    actions = torch.randn(4, 8, 3)
    _, dynamic_details = dynamic(z, actions, return_details=True)
    _, static_details = static(z, actions, return_details=True)
    q_static = torch.sigmoid(static.position_gain_bias)
    v_static = q_static * torch.sigmoid(static.velocity_ratio_bias)
    expected = torch.stack((q_static, v_static)).view(
        1, 1, 2, 1).expand(4, 9, 2, 1)
    assert torch.equal(static_details["gates"], expected)
    assert not torch.allclose(dynamic_details["gates"], expected)
    assert bool((dynamic_details["v_gates"] <= dynamic_details["q_gates"]).all())

    no_reliability = V11Memory(8, 3, mode="noreliability")
    no_reliability.load_state_dict(dynamic.state_dict(), strict=True)
    _, no_reliability_details = no_reliability(z, actions, return_details=True)
    assert torch.allclose(
        no_reliability_details["q_gates"].squeeze(-1),
        no_reliability_details["position_base_gain"], atol=1e-7, rtol=1e-6)
    assert torch.allclose(
        no_reliability_details["v_gates"].squeeze(-1),
        no_reliability_details["velocity_base_gain"], atol=1e-7, rtol=1e-6)
    assert not torch.allclose(no_reliability_details["gates"], expected)
    assert bool((dynamic_details["reliability"] <= 1).all())
    assert bool((dynamic_details["reliability"] > 0).all())

    _, closed = dynamic(
        z, actions, gate_override=0.0, return_details=True)
    _, opened = dynamic(
        z, actions, gate_override=torch.tensor([1.0, 0.0]),
        return_details=True)
    assert torch.equal(closed["states"][:, 1:], closed["priors"][:, 1:])
    assert torch.allclose(opened["q_states"], z, atol=3e-7, rtol=0.0)
    assert torch.equal(opened["v_states"], opened["v_priors"])
    try:
        dynamic(z, actions, gate_override=1.1)
    except ValueError as exc:
        assert "[0,1]" in str(exc)
    else:
        raise AssertionError("invalid gate override was accepted")


def test_monotone_scale_free_reliability_and_ordered_velocity_gain() -> None:
    memory = V11Memory(8, 3, mode="full")
    with torch.no_grad():
        memory.position_gain_vector.copy_(torch.linspace(-0.5, 0.5, 8))
        memory.velocity_ratio_vector.copy_(torch.linspace(0.4, -0.4, 8))
    angles = torch.linspace(0.0, torch.pi / 2, 7)
    q_prior = torch.zeros(7, 8)
    q_prior[:, 0] = 1.0
    z = torch.zeros_like(q_prior)
    z[:, 0] = torch.cos(angles)
    z[:, 1] = torch.sin(angles)
    gates, details = memory._innovation_gates(
        z, q_prior, return_details=True)
    assert torch.all(details["innovation_ratio"][1:] >=
                     details["innovation_ratio"][:-1])
    assert torch.all(details["reliability"][1:] <= details["reliability"][:-1])
    assert torch.all(gates[1:, :, 0] <= gates[:-1, :, 0])
    assert torch.all(gates[:, 1] <= gates[:, 0])
    normalized = memory._normalized_innovation(z, q_prior)
    expected_energy = torch.nn.functional.linear(
        normalized, memory._innovation_contrast).square().mean(dim=-1)
    assert torch.allclose(
        details["innovation_energy"], expected_energy, atol=2e-6, rtol=2e-6)

    # Prior conditioning is causal and nontrivial: it may select a different base gain for a
    # different predicted state, without weakening monotonicity for any fixed predicted state.
    varied_prior = torch.eye(8)
    varied_z = varied_prior.clone()
    _, varied = memory._innovation_gates(
        varied_z, varied_prior, return_details=True)
    assert float(varied["position_base_gain"].std().detach()) > 0.0
    assert float(varied["velocity_base_ratio"].std().detach()) > 0.0


def test_clean_precision_statistics_are_nongradient_and_fp32() -> None:
    torch.manual_seed(11010)
    memory = V11Memory(8, 3)
    z = torch.randn(5, 8)
    q_prior = torch.randn(5, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        gates, _ = memory._innovation_gates(z, q_prior, return_details=True)
        nll = memory.clean_innovation_nll(z.detach(), q_prior.detach())
    assert gates.dtype == z.dtype
    assert nll.dtype == torch.float32
    assert not nll.requires_grad

    gates.square().mean().backward()
    assert memory.innovation_precision_packed.grad is None
    assert memory.clean_innovation_mean.grad is None
    assert memory.process_tolerance_vector.grad is not None
    assert memory.process_tolerance_bias.grad is not None



def test_epoch_oas_calibration_is_closed_form_and_hyperparameter_free() -> None:
    torch.manual_seed(11011)
    memory = V11Memory(8, 3)
    z = torch.randn(40, 13, 8)
    q_prior = z + 0.15 * torch.randn_like(z)
    memory.reset_clean_calibration()
    memory.accumulate_clean_calibration(z[:20], q_prior[:20])
    memory.accumulate_clean_calibration(z[20:], q_prior[20:])
    receipt = memory.finalize_clean_calibration()
    assert receipt["samples"] == 520
    assert 0.0 <= receipt["oas_shrinkage"] <= 1.0
    assert receipt["covariance_condition"] >= 1.0
    assert memory.calibration_updates.item() == 1
    assert memory.calibration_samples.item() == 520

    normalized = memory._normalized_innovation(z, q_prior)
    values = torch.nn.functional.linear(
        normalized, memory._innovation_contrast).reshape(-1, 7).double()
    mean = values.mean(dim=0)
    empirical = (values - mean).T @ (values - mean) / len(values)
    alpha = empirical.square().mean()
    target = torch.trace(empirical) / 7
    numerator = alpha + target.square()
    denominator = (len(values) + 1) * (alpha - target.square() / 7)
    shrinkage = 1.0 if denominator <= 0 else min(float(numerator / denominator), 1.0)
    shrunk = (1.0 - shrinkage) * empirical
    shrunk.diagonal().add_(shrinkage * target)
    floor = torch.finfo(torch.float64).eps * 7 * target.abs().clamp_min(
        torch.finfo(torch.float64).tiny)
    shrunk.diagonal().add_(floor)
    factor = memory.innovation_precision_factor().double()
    whitened_covariance = factor @ shrunk @ factor.T
    assert torch.allclose(
        memory.clean_innovation_mean.double(), mean, atol=2e-7, rtol=2e-6)
    assert torch.allclose(
        whitened_covariance, torch.eye(7, dtype=torch.float64),
        atol=3e-5, rtol=3e-5)

    diagonal = V11Memory(8, 3)
    diagonal.reset_clean_calibration()
    diagonal.accumulate_clean_calibration(z, q_prior)
    diagonal.finalize_clean_calibration(diagonal_only=True)
    diagonal_factor = diagonal.innovation_precision_factor()
    assert torch.count_nonzero(torch.tril(diagonal_factor, diagonal=-1)) == 0
    assert bool(diagonal.calibration_diagonal_only)


def test_model_registry_and_objective_aliases() -> None:
    torch.manual_seed(11008)
    models = {name: make_precomputed_model(name) for name in MODE_MAP}
    assert all(model.mem_kdiov11.mode == MODE_MAP[name]
               for name, model in models.items())
    signatures = {
        name: [(key, tuple(value.shape))
               for key, value in model.mem_kdiov11.state_dict().items()]
        for name, model in models.items()
    }
    first = signatures["kdiov11"]
    assert all(signature == first for signature in signatures.values())

    full = models["kdiov11"]
    h1 = models["kdiov11_h1"]
    noactionswap = models["kdiov11_noactionswap"]
    h1.mem_kdiov11.load_state_dict(full.mem_kdiov11.state_dict())
    noactionswap.mem_kdiov11.load_state_dict(full.mem_kdiov11.state_dict())
    z = torch.randn(3, 7, 8)
    actions = torch.randn(3, 6, 3)
    full_injected, details = full._inject(
        z, actions, return_memory_details=True)
    assert torch.equal(full_injected, h1._inject(z, actions))
    assert torch.equal(full_injected, noactionswap._inject(z, actions))
    assert details["states"].shape == (3, 7, 2, 8)
    assert full.horizons()["recurrent_floats"] == 16

    targets = torch.randn_like(z, requires_grad=True)
    diversity = torch.randn_like(z, requires_grad=True)
    losses = full.compute_loss(
        z, actions, target_embeddings=targets,
        diversity_embeddings=diversity, objective="v10j",
        detach_target_embeddings=False)
    assert torch.equal(
        losses["loss"], losses["pred_loss"] + losses["variance_loss"]
        + losses["covariance_loss"])
    losses["loss"].backward()
    assert targets.grad is not None and torch.isfinite(targets.grad).all()


def test_suffix_and_fusion_gradients_reach_all_mechanisms() -> None:
    torch.manual_seed(11009)
    memory = V11Memory(8, 3)
    z = torch.randn(4, 8, 8, requires_grad=True)
    actions = torch.randn(4, 7, 3)
    mixed, details = memory(z, actions, return_details=True)
    source = details["states"][:, 2]
    rollout = memory.rollout_transition(source, actions[:, 2:6])
    suffix_prediction = memory.read_state(rollout[:, -1])
    suffix_target = torch.randn_like(suffix_prediction)
    fused = memory.fuse(z, mixed)
    loss = (suffix_prediction - suffix_target).square().mean() + fused.square().mean()
    loss.backward()
    for parameter in (
            memory.W_a.weight, memory.w_q, memory.b_f,
            memory.log_action_scale,
            memory.position_gain_vector, memory.velocity_ratio_vector,
            memory.process_tolerance_vector,
            memory.position_gain_bias, memory.velocity_ratio_bias,
            memory.process_tolerance_bias,
            memory.W_o.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert float(memory.W_a.weight.grad.abs().sum()) > 0.0
    assert float(memory.w_q.grad.abs().sum()) > 0.0
    assert float(memory.b_f.grad.abs().sum()) > 0.0
    assert float(memory.W_o.weight.grad.abs().sum()) > 0.0
    assert memory.innovation_precision_packed.grad is None
    assert memory.clean_innovation_mean.grad is None
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_all_suffix_action_ranking_and_objective_controls() -> None:
    torch.manual_seed(11013)
    world = make_precomputed_model("kdiov11")
    model = V11ExperimentModel(world)
    memory = world.mem_kdiov11
    batch, length, dimension, action_dim = 4, 6, 8, 3
    initial = torch.randn(batch, 2, dimension)
    actions = torch.randn(batch, length - 1, action_dim)
    rollout = memory.rollout_transition(initial, actions)
    states = torch.cat((initial.unsqueeze(1), rollout), dim=1)
    clean_z = memory.read_state(states)
    objectives = _kdio_suffix_objectives(
        model, states, actions, clean_z, h1_only=False)
    assert objectives["horizons"] == length - 1
    assert tuple(objectives["positive_energy_by_horizon"].shape) == (length - 1,)
    assert float(objectives["suffix_loss"].detach()) < 1e-10
    assert float(objectives["action_swap_negative_energy"].detach()) > 1e-5
    assert float(objectives["action_swap_loss"].detach()) < np.log(2.0)
    assert float(objectives["action_swap_pair_accuracy"].detach()) > 0.5

    # Independent, deliberately unvectorized recurrence verifies displacement alignment,
    # scale placement, relative energy, and equal-horizon reduction.
    shuffled = torch.roll(actions, shifts=1, dims=0)
    live_state = states[:, :-1]
    positive_state = states[:, :-1].detach()
    negative_state = states[:, :-1].detach()
    source_read = memory.read_state(states[:, :-1].detach())
    clean_source = clean_z[:, :-1].detach()
    manual_live, manual_positive, manual_negative = [], [], []
    manual_rank, manual_accuracy = [], []
    cached = memory.action_frame()
    for offset in range(length - 1):
        valid = length - 1 - offset
        live_state = memory.transition(
            live_state[:, :valid].reshape(batch * valid, 2, dimension),
            actions[:, offset:offset + valid].reshape(batch * valid, action_dim),
            cached_action_frame=cached).reshape(batch, valid, 2, dimension)
        positive_state = memory.transition(
            positive_state[:, :valid].reshape(batch * valid, 2, dimension),
            actions[:, offset:offset + valid].reshape(batch * valid, action_dim),
            cached_action_frame=cached).reshape(batch, valid, 2, dimension)
        negative_state = memory.transition(
            negative_state[:, :valid].reshape(batch * valid, 2, dimension),
            shuffled[:, offset:offset + valid].reshape(batch * valid, action_dim),
            cached_action_frame=cached).reshape(batch, valid, 2, dimension)
        target = clean_z[:, offset + 1:offset + 1 + valid]
        target_delta = target.detach() - clean_source[:, :valid]
        live = (memory.read_state(live_state) - target).square().mean(dim=-1)
        positive = (
            memory.read_state(positive_state) - source_read[:, :valid]
            - target_delta).square().mean(dim=-1)
        negative = (
            memory.read_state(negative_state) - source_read[:, :valid]
            - target_delta).square().mean(dim=-1)
        manual_live.append(live.mean())
        manual_positive.append(positive.mean())
        manual_negative.append(negative.mean())
        manual_rank.append(_action_rank_pair_loss(
            positive, negative, DEFAULT_ACTION_RANKING).mean())
        manual_accuracy.append(((positive < negative).float()
                                + 0.5 * (positive == negative).float()).mean())
    assert torch.allclose(
        objectives["live_energy_by_horizon"], torch.stack(manual_live),
        atol=1e-7, rtol=1e-6)
    assert torch.allclose(
        objectives["positive_energy_by_horizon"], torch.stack(manual_positive),
        atol=1e-7, rtol=1e-6)
    assert torch.allclose(
        objectives["negative_energy_by_horizon"], torch.stack(manual_negative),
        atol=1e-7, rtol=1e-6)
    assert torch.allclose(
        objectives["action_swap_loss"], torch.stack(manual_rank).mean(),
        atol=1e-7, rtol=1e-6)
    assert torch.equal(
        objectives["action_swap_pair_accuracy"], torch.stack(manual_accuracy).mean())
    with mock.patch.object(
            memory, "action_direction", wraps=memory.action_direction) as direction_call:
        h1 = _kdio_suffix_objectives(
            model, states, actions, clean_z, h1_only=True)
        assert direction_call.call_count == 1
    assert h1["horizons"] == 1
    assert tuple(h1["positive_energy_by_horizon"].shape) == (1,)
    assert torch.equal(
        h1["positive_energy_by_horizon"],
        objectives["positive_energy_by_horizon"][:1])
    assert torch.equal(
        h1["negative_energy_by_horizon"],
        objectives["negative_energy_by_horizon"][:1])

    noaction_world = make_precomputed_model("kdiov11_noaction")
    noaction_world.mem_kdiov11.load_state_dict(memory.state_dict(), strict=True)
    noaction = _kdio_suffix_objectives(
        V11ExperimentModel(noaction_world), states.detach(), actions,
        clean_z.detach(), h1_only=False)
    assert torch.equal(
        noaction["action_swap_positive_energy"],
        noaction["action_swap_negative_energy"])
    assert torch.allclose(
        noaction["action_swap_loss"],
        torch.tensor(np.log(2.0), dtype=noaction["action_swap_loss"].dtype),
        atol=1e-7, rtol=0.0)
    assert torch.equal(noaction["action_swap_pair_accuracy"], torch.tensor(0.5))
    noaction["action_swap_loss"].backward()
    for name, parameter in noaction_world.mem_kdiov11.named_parameters():
        assert _is_exact_zero_gradient(parameter.grad), name

    observed = torch.randn(batch, length, dimension)
    clean = torch.randn_like(observed)
    full = compute_v11_losses(model, observed, clean, actions, "kdiov11")
    removed = compute_v11_losses(
        model, observed, clean, actions, "kdiov11_noactionswap")
    nosuffix = compute_v11_losses(
        model, observed, clean, actions, "kdiov11_nosuffix")
    h1_losses = compute_v11_losses(
        model, observed, clean, actions, "kdiov11_h1")
    assert torch.equal(
        full["loss"], full["predictive_loss"] + full["action_swap_loss"]
        + full["variance_loss"] + full["covariance_loss"])
    assert float(full["action_swap_applicable"]) == 1.0
    assert float(full["action_swap_horizons"]) == length - 1
    assert torch.equal(removed["action_swap_loss"], torch.zeros(()))
    assert torch.equal(
        removed["action_swap_diagnostic_loss"],
        full["action_swap_diagnostic_loss"])
    assert torch.equal(nosuffix["suffix_loss"], nosuffix["context_loss"])
    assert torch.equal(nosuffix["action_swap_loss"], torch.zeros(()))
    assert float(nosuffix["action_swap_applicable"]) == 0.0
    assert float(h1_losses["action_swap_horizons"]) == 1.0


def _is_exact_zero_gradient(gradient: torch.Tensor | None) -> bool:
    return gradient is None or torch.count_nonzero(gradient).item() == 0


def test_action_ranking_default_detach_gradients_and_development_modes() -> None:
    torch.manual_seed(11017)
    batch, length, dimension, action_dim = 4, 6, 8, 3
    actions = torch.randn(batch, length - 1, action_dim)
    fixed_states = torch.randn(batch, length, 2, dimension)
    fixed_targets = torch.randn(batch, length, dimension)
    reference = make_precomputed_model("kdiov11")
    reference_state = reference.state_dict()
    mode_losses = {}
    for action_ranking_mode in ACTION_RANKING_MODES:
        world = make_precomputed_model("kdiov11")
        world.load_state_dict(reference_state, strict=True)
        memory = world.mem_kdiov11
        states = fixed_states.clone().requires_grad_()
        targets = fixed_targets.clone().requires_grad_()
        objective = _kdio_suffix_objectives(
            V11ExperimentModel(world), states, actions, targets, h1_only=False,
            action_ranking_mode=action_ranking_mode)
        mode_losses[action_ranking_mode] = float(
            objective["action_swap_loss"].detach())
        gradients = torch.autograd.grad(
            objective["action_swap_loss"],
            (memory.log_action_scale, states, targets, memory.W_a.weight,
             memory.w_q, memory.b_f), allow_unused=True)
        gamma_gradient, state_gradient, target_gradient = gradients[:3]
        direction_gradient, state_kick_gradient, bias_gradient = gradients[3:]
        assert direction_gradient is not None
        assert torch.isfinite(direction_gradient).all()
        assert float(direction_gradient.abs().sum()) > 0.0
        assert state_kick_gradient is not None
        assert bias_gradient is not None
        assert float(state_kick_gradient.abs().sum()) > 0.0
        assert float(bias_gradient.abs().sum()) > 0.0
        assert _is_exact_zero_gradient(state_gradient)
        assert _is_exact_zero_gradient(target_gradient)
        if action_ranking_mode == "relative_displacement_livegamma":
            assert gamma_gradient is not None
            assert torch.isfinite(gamma_gradient)
            assert float(gamma_gradient.abs()) > 0.0
        else:
            assert _is_exact_zero_gradient(gamma_gradient)

    # Detaching gamma changes only backward semantics, while geometry/energy ablations alter
    # the scalar objective on the same forward tensors.
    assert np.isclose(
        mode_losses["relative_displacement_detached"],
        mode_losses["relative_displacement_livegamma"], rtol=0.0, atol=0.0)
    assert not np.isclose(
        mode_losses["relative_displacement_detached"],
        mode_losses["relative_endpoint_detached"])
    assert not np.isclose(
        mode_losses["relative_displacement_detached"],
        mode_losses["rawdiff_displacement_detached"])

    live_world = make_precomputed_model("kdiov11")
    live_world.load_state_dict(reference_state, strict=True)
    live_memory = live_world.mem_kdiov11
    live_states = fixed_states.clone().requires_grad_()
    live_targets = fixed_targets.clone().requires_grad_()
    live_objective = _kdio_suffix_objectives(
        V11ExperimentModel(live_world), live_states, actions, live_targets,
        h1_only=False)
    live_gradients = torch.autograd.grad(
        live_objective["suffix_loss"],
        (live_memory.log_action_scale, live_states, live_targets),
        allow_unused=True)
    assert all(gradient is not None for gradient in live_gradients)
    assert all(float(gradient.abs().sum()) > 0.0 for gradient in live_gradients)

    for implementation in ("kdiov11_unconstrained", "kdiov11_fixedscale"):
        world = make_precomputed_model(implementation)
        memory = world.mem_kdiov11
        objective = _kdio_suffix_objectives(
            V11ExperimentModel(world),
            torch.randn(batch, length, 2, dimension), actions,
            torch.randn(batch, length, dimension), h1_only=False)
        objective["action_swap_loss"].backward()
        assert memory.W_a.weight.grad is not None, implementation
        assert torch.isfinite(memory.W_a.weight.grad).all(), implementation
        assert float(memory.W_a.weight.grad.abs().sum()) > 0.0, implementation
        assert _is_exact_zero_gradient(memory.log_action_scale.grad)


def test_relative_action_rank_is_energy_rescale_invariant() -> None:
    positive = torch.tensor([0.03, 0.4, 1.7], dtype=torch.float64)
    negative = torch.tensor([0.2, 0.8, 1.1], dtype=torch.float64)
    relative = _action_rank_pair_loss(
        positive, negative, "relative_displacement_detached")
    scaled = _action_rank_pair_loss(
        137.0 * positive, 137.0 * negative,
        "relative_displacement_detached")
    assert torch.allclose(relative, scaled, atol=1e-14, rtol=1e-14)
    raw = _action_rank_pair_loss(
        positive, negative, "rawdiff_displacement_detached")
    raw_scaled = _action_rank_pair_loss(
        137.0 * positive, 137.0 * negative,
        "rawdiff_displacement_detached")
    assert not torch.allclose(raw, raw_scaled)


def test_action_rank_metadata_is_variant_exact() -> None:
    cli = parse_args([
        "--train-data", "train.npz", "--val-data", "val.npz",
        "--memory-mode", "kdiov11", "--seed", "1"])
    assert cli.development_action_ranking == DEFAULT_ACTION_RANKING
    cli = parse_args([
        "--train-data", "train.npz", "--val-data", "val.npz",
        "--memory-mode", "kdiov11", "--seed", "1",
        "--development-action-ranking", "relative_endpoint_detached"])
    assert cli.development_action_ranking == "relative_endpoint_detached"
    default = _design_metadata("kdiov11")
    assert default["action_rank_optimized"]
    assert default["action_rank_direction_gradient_active"]
    assert default["action_rank_transition_gradient_active"]
    assert not default["action_rank_scale_gradient_active"]
    live = _design_metadata(
        "kdiov11", "relative_displacement_livegamma")
    assert live["action_rank_scale_gradient_active"]
    fixed = _design_metadata(
        "kdiov11_fixedscale", "relative_displacement_livegamma")
    assert not fixed["action_rank_scale_gradient_active"]
    noaction = _design_metadata(
        "kdiov11_noaction", "relative_displacement_livegamma")
    assert not noaction["action_swap_gradient_active"]
    assert not noaction["action_rank_direction_gradient_active"]
    assert not noaction["action_rank_transition_gradient_active"]
    removed = _design_metadata(
        "kdiov11_noactionswap", "relative_displacement_livegamma")
    assert removed["action_rank_diagnostic_computed"]
    assert not removed["action_rank_optimized"]
    assert not removed["action_rank_scale_gradient_active"]
    assert not _design_metadata("kdiov11_nosuffix")["action_rank_optimized"]


def test_global_action_rank_receipt_is_eval_batch_invariant() -> None:
    torch.manual_seed(11018)
    episodes, length, dimension, action_dim = 240, 6, 8, 3
    world = make_precomputed_model("kdiov11")
    model = V11ExperimentModel(world)
    memory = world.mem_kdiov11
    initial = torch.randn(episodes, 2, dimension)
    actions = torch.randn(episodes, length - 1, action_dim)
    rollout = memory.rollout_transition(initial, actions)
    states = torch.cat((initial.unsqueeze(1), rollout), dim=1)
    clean_z = memory.read_state(states)
    permutation = torch.roll(torch.arange(episodes), shifts=1)
    negative_actions = actions.index_select(0, permutation)
    receipts = [aggregate_action_rank_receipts(
        model, states, actions, clean_z, negative_actions,
        episode_batch_size=batch_size, device=torch.device("cpu"))
        for batch_size in (17, 64, 240)]
    expected_pairs = episodes * (length - 1) * length // 2
    assert all(receipt["episode_count"] == episodes for receipt in receipts)
    assert all(receipt["pair_count"] == expected_pairs for receipt in receipts)
    for key in (
            "live_energy_by_horizon", "positive_energy_by_horizon",
            "negative_energy_by_horizon", "ranking_loss_by_horizon",
            "pair_accuracy_by_horizon", "divergence_by_horizon",
            "pair_count_by_horizon"):
        for receipt in receipts[1:]:
            assert np.allclose(
                receipts[0][key], receipt[key], atol=2e-7, rtol=2e-7), key
    assert np.allclose(
        [receipt["action_effect_rms"] for receipt in receipts],
        receipts[0]["action_effect_rms"], atol=2e-7, rtol=2e-7)


class _SyntheticInverseDataset(Dataset):
    def __init__(self, seed: int, episodes: int = 24,
                 length: int = 7, dimension: int = 8, action_dim: int = 3):
        generator = torch.Generator().manual_seed(seed)
        self.clean = torch.randn(
            episodes, length, dimension, generator=generator)
        self.actions = torch.zeros(episodes, length - 1, action_dim)
        self.actions[:, 1:] = (
            self.clean[:, 2:, :action_dim]
            - self.clean[:, 1:-1, :action_dim])

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"clean": self.clean[index], "actions": self.actions[index]}


def test_inverse_action_ridge_is_evaluation_only_and_serializable() -> None:
    torch.manual_seed(11014)
    model = V11ExperimentModel(make_precomputed_model("kdiov11"))
    assert not any("inverse_head" in name for name, _ in model.named_parameters())
    aligned_z = torch.arange(2 * 6 * 8, dtype=torch.float32).reshape(2, 6, 8)
    aligned_actions = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    aligned_input, aligned_target = _second_order_inverse_inputs(
        aligned_z, aligned_actions)
    assert torch.equal(
        aligned_input,
        torch.cat((aligned_z[:, :-2], aligned_z[:, 1:-1], aligned_z[:, 2:]), dim=-1))
    assert torch.equal(aligned_target, aligned_actions[:, 1:])
    args = SimpleNamespace(batch_size=6, num_workers=0, probe_ridge=1e-3)
    train = _SyntheticInverseDataset(11015)
    val = _SyntheticInverseDataset(11016)
    probe = fit_inverse_action_probe(
        model, train, args, torch.device("cpu"), False)
    metrics = evaluate_inverse_action_probe(
        model, val, probe, args, torch.device("cpu"), False)
    assert probe["x_mean"].shape == (24,)
    assert probe["weights"].shape == (25, 3)
    assert metrics["inverse_action_probe_input_dim"] == 24
    assert metrics["inverse_action_probe_output_dim"] == 3
    assert metrics["inverse_action_nmse"] < 1e-4
    assert metrics["inverse_action_r2"] > 0.999
    assert all(np.isfinite(value).all() for value in probe.values())


if __name__ == "__main__":
    tests = (
        test_parameter_contract_and_common_schema,
        test_stiefel_kick_initialization_and_residual_identity,
        test_kick_drift_equations_state_dependence_and_inverse,
        test_scaled_stiefel_free_geometry_fixedscale_gradients_and_cache,
        test_mode_interventions_are_exact_and_same_tensor,
        test_rollout_transition_and_reverse_recovery,
        test_forward_details_and_streaming_step_agree,
        test_gate_and_static_controls,
        test_monotone_scale_free_reliability_and_ordered_velocity_gain,
        test_clean_precision_statistics_are_nongradient_and_fp32,
        test_epoch_oas_calibration_is_closed_form_and_hyperparameter_free,
        test_model_registry_and_objective_aliases,
        test_suffix_and_fusion_gradients_reach_all_mechanisms,
        test_all_suffix_action_ranking_and_objective_controls,
        test_action_ranking_default_detach_gradients_and_development_modes,
        test_relative_action_rank_is_energy_rescale_invariant,
        test_action_rank_metadata_is_variant_exact,
        test_global_action_rank_receipt_is_eval_batch_invariant,
        test_inverse_action_ridge_is_evaluation_only_and_serializable,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
    print(f"All {len(tests)} KDIO-v11 tests passed.")
