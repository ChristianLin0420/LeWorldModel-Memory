"""CPU tests for the publication frozen-host carrier swap."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.frozen_swap_carriers import (
    FROZEN_SWAP_CARRIER_NAMES,
    OFFICIAL_LEWM_ACTION_DIM,
    OFFICIAL_LEWM_EMBED_DIM,
    FrozenActionConditionedGRU,
    FrozenActionConditionedLSTM,
    FrozenDiagonalSSM,
    FrozenFixedTrustLKC,
    clone_stream_state,
    fixed_trust_lkc_parameter_count,
    gru_parameter_count,
    lstm_parameter_count,
    make_frozen_swap_carrier,
    make_frozen_carrier,
    matched_gru_hidden,
    matched_lstm_hidden,
    matched_ssm_width,
    parameter_match_report,
    parameter_report,
    repeat_stream_state,
    ssm_parameter_count,
)

D, A, B, L = 16, 3, 3, 9


def _inputs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(B, L, D, generator=generator)
    actions = torch.randn(B, L - 1, A, generator=generator)
    return z, actions


def _set_nonzero_read(carrier, seed: int = 11) -> None:
    if not hasattr(carrier, "w_o"):
        return
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        carrier.w_o.weight.copy_(
            0.05 * torch.randn(carrier.w_o.weight.shape, generator=generator))


def _state_tensors(state):
    return tuple(vars(state).values())


@pytest.mark.parametrize("name", FROZEN_SWAP_CARRIER_NAMES)
def test_registry_shapes_and_zero_initialized_host_equality(name):
    torch.manual_seed(0)
    carrier = make_frozen_swap_carrier(name, D, A)
    z, actions = _inputs()
    output = carrier(z, actions)
    assert output.z_tilde.shape == z.shape
    assert output.prior_read.shape == z.shape
    assert torch.equal(output.z_tilde, z)
    assert torch.equal(output.prior_read, torch.zeros_like(z))
    assert carrier.describe()["parameters"] == carrier.parameter_count()


@pytest.mark.parametrize("name", FROZEN_SWAP_CARRIER_NAMES)
def test_future_observation_causality_and_preobservation_prior(name):
    torch.manual_seed(1)
    carrier = make_frozen_swap_carrier(name, D, A).eval()
    _set_nonzero_read(carrier)
    z, actions = _inputs(2)
    cut = 4
    with torch.no_grad():
        reference = carrier(z, actions)

        future_z = z.clone()
        future_z[:, cut + 1:] += 17.0
        future = carrier(future_z, actions)
        assert torch.allclose(reference.z_tilde[:, :cut + 1],
                              future.z_tilde[:, :cut + 1], atol=1e-6)
        assert torch.allclose(reference.prior_read[:, :cut + 1],
                              future.prior_read[:, :cut + 1], atol=1e-6)

        current_z = z.clone()
        current_z[:, cut] -= 13.0
        current = carrier(current_z, actions)
        # The evaluation coordinate at t is read before z_t is consumed.
        assert torch.allclose(reference.prior_read[:, cut],
                              current.prior_read[:, cut], atol=1e-6)


@pytest.mark.parametrize("name", FROZEN_SWAP_CARRIER_NAMES)
def test_sequence_forward_equals_streamed_observation_updates(name):
    torch.manual_seed(3)
    carrier = make_frozen_swap_carrier(name, D, A).eval()
    _set_nonzero_read(carrier, seed=12)
    z, actions = _inputs(4)
    with torch.no_grad():
        sequence = carrier(z, actions)
        step = carrier.initialize(z[:, 0])
        fused = [step.fused_z]
        priors = [step.prior_read]
        state = step.state
        for t in range(1, L):
            step = carrier.observe(state, z[:, t], actions[:, t - 1])
            state = step.state
            fused.append(step.fused_z)
            priors.append(step.prior_read)
    assert torch.allclose(sequence.z_tilde, torch.stack(fused, dim=1),
                          atol=2e-6, rtol=1e-6)
    assert torch.allclose(sequence.prior_read, torch.stack(priors, dim=1),
                          atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize("name", FROZEN_SWAP_CARRIER_NAMES)
def test_clone_repeat_and_imagine_candidate_states(name):
    torch.manual_seed(5)
    carrier = make_frozen_swap_carrier(name, D, A).eval()
    _set_nonzero_read(carrier, seed=13)
    z, actions = _inputs(6)
    state = carrier.initialize(z[:, 0]).state

    cloned = clone_stream_state(state)
    for source, copy in zip(_state_tensors(state), _state_tensors(cloned)):
        assert torch.equal(source, copy)
        if source.numel():
            assert source.data_ptr() != copy.data_ptr()

    candidates = 4
    repeated = repeat_stream_state(state, candidates)
    for source, expanded in zip(_state_tensors(state),
                                _state_tensors(repeated)):
        assert expanded.shape[0] == B * candidates
        expected = source.repeat_interleave(candidates, dim=0)
        assert torch.equal(expanded, expected)
        if source.numel():
            assert source.data_ptr() != expanded.data_ptr()

    candidate_actions = actions[:, 0].repeat_interleave(candidates, dim=0)
    imagined = carrier.imagine(repeated, candidate_actions)
    assert imagined.fused_z is None
    assert imagined.prior_read.shape == (B * candidates, D)
    for tensor in _state_tensors(imagined.state):
        assert tensor.shape[0] == B * candidates
        assert torch.isfinite(tensor).all()


def test_repeat_rejects_invalid_candidate_count():
    z, _ = _inputs()
    state = make_frozen_swap_carrier("acgru", D, A).initialize(z[:, 0]).state
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            repeat_stream_state(state, invalid)


@pytest.mark.parametrize("name", FROZEN_SWAP_CARRIER_NAMES[1:])
def test_nonzero_finite_gradients_through_learned_carriers(name):
    torch.manual_seed(7)
    carrier = make_frozen_swap_carrier(name, D, A)
    _set_nonzero_read(carrier, seed=14)
    z, actions = _inputs(8)
    output = carrier(z, actions)
    loss = (output.z_tilde.square().mean()
            + output.prior_read.square().mean())
    loss.backward()
    gradients = [parameter.grad for parameter in carrier.parameters()
                 if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0
    assert float(carrier.w_o.weight.grad.abs().sum()) > 0.0


def test_official_parameter_report_and_real_module_counts():
    report = parameter_match_report()
    assert report["embed_dim"] == OFFICIAL_LEWM_EMBED_DIM == 192
    assert report["action_dim"] == OFFICIAL_LEWM_ACTION_DIM == 10
    assert report["target_parameters"] == 76_032
    expected = {
        "none": (None, 0, -76_032),
        "acgru": (74, 75_924, -108),
        "aclstm": (61, 76_372, 340),
        "diag_ssm": (192, 76_032, 0),
        "lkc_fixed_trust": (192, 76_032, 0),
    }
    for name, (width, count, delta) in expected.items():
        entry = report["arms"][name]
        assert (entry["width"], entry["parameters"], entry["delta"]) == (
            width, count, delta)
        carrier = make_frozen_swap_carrier(
            name, OFFICIAL_LEWM_EMBED_DIM, OFFICIAL_LEWM_ACTION_DIM)
        assert carrier.parameter_count() == count


def test_trainer_facing_factory_and_parameter_report_alias():
    expected_types = {
        "gru": FrozenActionConditionedGRU,
        "lstm": FrozenActionConditionedLSTM,
        "ssm": FrozenDiagonalSSM,
        "fixed_trust": FrozenFixedTrustLKC,
    }
    assert make_frozen_carrier("none", D, A).name == "none"
    for name, expected_type in expected_types.items():
        assert isinstance(make_frozen_carrier(name, D, A), expected_type)
    assert parameter_report(D, A) == parameter_match_report(D, A)
    with pytest.raises(KeyError, match="expected one of"):
        make_frozen_carrier("acgru", D, A)


@pytest.mark.parametrize(
    "width_fn,count_fn,carrier_type,width_attr",
    [
        (matched_gru_hidden, gru_parameter_count,
         FrozenActionConditionedGRU, "hidden_dim"),
        (matched_lstm_hidden, lstm_parameter_count,
         FrozenActionConditionedLSTM, "hidden_dim"),
        (matched_ssm_width, ssm_parameter_count,
         FrozenDiagonalSSM, "width"),
    ],
)
@pytest.mark.parametrize("embed_dim,action_dim", [(12, 2), (16, 3), (192, 10)])
def test_integer_width_is_globally_closest_and_formula_is_exact(
        width_fn, count_fn, carrier_type, width_attr, embed_dim, action_dim):
    target = fixed_trust_lkc_parameter_count(embed_dim, action_dim)
    width = width_fn(embed_dim, action_dim)
    carrier = carrier_type(embed_dim, action_dim)
    assert getattr(carrier, width_attr) == width
    achieved = count_fn(width, embed_dim, action_dim)
    assert carrier.parameter_count() == achieved
    for neighbor in (width - 1, width + 1):
        if neighbor > 0:
            assert abs(achieved - target) <= abs(
                count_fn(neighbor, embed_dim, action_dim) - target)


def test_fixed_trust_lkc_is_constant_trust_and_exact_target():
    carrier = FrozenFixedTrustLKC(D, A)
    assert carrier.r_fixed and carrier.r_head is None
    assert carrier.r_const is not None
    assert carrier.parameter_count() == fixed_trust_lkc_parameter_count(D, A)
    assert carrier.describe()["imagine_rule"] == (
        "Kalman_predict_without_correction")


def test_stream_shape_errors_are_actionable():
    carrier = make_frozen_swap_carrier("aclstm", D, A)
    z, actions = _inputs()
    with pytest.raises(ValueError, match="z0"):
        carrier.initialize(z[:, 0, :-1])
    state = carrier.initialize(z[:, 0]).state
    with pytest.raises(ValueError, match="a_prev"):
        carrier.observe(state, z[:, 1], actions[:, 0, :-1])
    with pytest.raises(ValueError, match="state batch"):
        carrier.observe(repeat_stream_state(state, 2), z[:, 1], actions[:, 0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
