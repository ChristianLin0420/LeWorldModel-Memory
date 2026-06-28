#!/usr/bin/env python3
"""Dependency-free invariants for HACSSM-v5.

Run from the repository root with ``python scripts/test_hacssm_v5.py``.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lewm.models.memory import (
    HACSSMv5Memory,
    HierarchicalActionConditionedMemory,
    HierarchicalActionConditionedSSMMemory,
)


def _assert_close(actual, expected, *, atol=1e-6, rtol=1e-6, message=''):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        error = float((actual - expected).abs().max())
        raise AssertionError(message or f'tensors differ (max absolute error {error:.3e})')


def _expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def test_initialization():
    D, A = 12, 5
    memory = HACSSMv5Memory(D, A)
    assert HACSSMv5Memory is HierarchicalActionConditionedSSMMemory
    _assert_close(memory.W_x.weight, torch.eye(D), message='W_x is not identity initialized')
    _assert_close(memory.W_a.weight, torch.zeros_like(memory.W_a.weight))
    _assert_close(memory.W_o.weight, torch.zeros_like(memory.W_o.weight))
    _assert_close(memory.w_z, torch.zeros(D))
    _assert_close(memory.w_e, torch.zeros(D))
    _assert_close(memory.gate_bias, torch.full((2,), 2.0))
    _assert_close(memory.route_weights(), torch.full((2,), 0.5))

    expected_fast_taus = torch.logspace(math.log10(1.5), math.log10(8.0), D)
    expected_medium_taus = torch.logspace(math.log10(8.0), math.log10(64.0), D)
    expected = torch.stack((
        1.0 - torch.exp(-1.0 / expected_fast_taus),
        1.0 - torch.exp(-1.0 / expected_medium_taus),
    ))
    _assert_close(memory.initial_fast_taus, expected_fast_taus)
    _assert_close(memory.initial_medium_taus, expected_medium_taus)
    _assert_close(memory.betas, expected, atol=2e-7, rtol=2e-6)
    assert bool((memory.betas[0] >= memory.betas[1]).all())


def test_parameter_matching():
    D, A = 128, 6
    expected = 2 * D * D + 2 * A * D + 4 * D + 4
    assert expected == 34820
    signatures = []
    for mode in sorted(HACSSMv5Memory.MODES):
        memory = HACSSMv5Memory(D, A, mode=mode)
        assert memory.parameter_count() == expected
        assert memory.expected_parameter_count(D, A) == expected
        signatures.append([(name, tuple(value.shape)) for name, value in memory.named_parameters()])
    assert all(signature == signatures[0] for signature in signatures[1:])


def test_monotone_and_fixed_betas():
    torch.manual_seed(3)
    memory = HACSSMv5Memory(17, 4)
    with torch.no_grad():
        memory.theta_medium.normal_(mean=-1.0, std=4.0)
        memory.theta_gap.normal_(mean=0.0, std=5.0)
    beta = memory.betas
    assert bool((beta > 0).all()) and bool((beta < 1).all())
    assert bool((beta[0] >= beta[1]).all()), 'hard monotonicity was violated'
    beta.sum().backward()
    assert memory.theta_medium.grad is not None and torch.isfinite(memory.theta_medium.grad).all()
    assert memory.theta_gap.grad is not None and torch.isfinite(memory.theta_gap.grad).all()

    fixed = HACSSMv5Memory(17, 4, mode='fixedbeta')
    before = fixed.betas.clone()
    with torch.no_grad():
        fixed.theta_medium.fill_(100.0)
        fixed.theta_gap.fill_(-100.0)
    _assert_close(fixed.betas, before, atol=0.0, rtol=0.0,
                  message='fixedbeta changed when its disconnected logits changed')


def test_action_causality_gradient_and_noaction():
    torch.manual_seed(4)
    B, T, D, A = 2, 5, 10, 3
    z = torch.randn(B, T, D)
    actions = torch.randn(B, T - 1, A, requires_grad=True)
    memory = HACSSMv5Memory(D, A)
    with torch.no_grad():
        memory.W_a.weight.normal_(std=0.25)

    _, details = memory(z, actions, gate_override=0.0, return_details=True)
    changed = actions.detach().clone()
    changed[:, 2] += 4.0
    _, changed_details = memory(z, changed, gate_override=0.0, return_details=True)
    _assert_close(details['states'][:, :3], changed_details['states'][:, :3],
                  message='a_t affected a state at or before t')
    assert not torch.allclose(details['states'][:, 3], changed_details['states'][:, 3])

    loss = details['states'][:, -1].square().mean()
    action_grad = torch.autograd.grad(loss, actions)[0]
    assert torch.isfinite(action_grad).all() and float(action_grad.abs().sum()) > 0.0

    noaction = HACSSMv5Memory(D, A, mode='noaction')
    with torch.no_grad():
        noaction.W_a.weight.normal_(std=5.0)
    out_a = noaction(z, actions.detach(), gate_override=0.0)
    out_b = noaction(z, changed, gate_override=0.0)
    _assert_close(out_a, out_b, atol=0.0, rtol=0.0)


def test_static_overrides_masks_and_ssm_control():
    torch.manual_seed(5)
    B, T, D, A = 3, 6, 9, 4
    z = torch.randn(B, T, D)
    actions = torch.randn(B, T - 1, A)
    static = HACSSMv5Memory(D, A, mode='static')
    _, details = static(z, actions, return_details=True)
    expected_gate = torch.sigmoid(static.gate_bias).view(1, 1, 2, 1).expand(B, T, -1, -1)
    _assert_close(details['gates'], expected_gate)

    _, zero_details = static(z, actions, gate_override=0.0, beta_override=0.0,
                             return_details=True)
    _assert_close(zero_details['gates'], torch.zeros_like(zero_details['gates']))
    _assert_close(zero_details['states'],
                  zero_details['states'][:, :1].expand_as(zero_details['states']))
    _assert_close(zero_details['betas'], torch.zeros_like(zero_details['betas']))

    masked_zero = static(z, actions, memory_update_mask=torch.zeros(B, T))
    masked_one = static(z, actions, memory_update_mask=torch.ones(B, T))
    _assert_close(masked_zero, masked_one, atol=0.0, rtol=0.0,
                  message='memory_update_mask changed V5 output')
    _expect_value_error(lambda: static(z, actions, memory_update_mask=torch.full((B, T), 2.0)))

    control = HACSSMv5Memory(D, A, mode='ssmcontrol')
    with torch.no_grad():
        control.W_a.weight.normal_(std=1.0)
    out_a, control_details = control(z, actions, return_details=True)
    out_b = control(z, actions * -7.0)
    _assert_close(out_a, out_b, atol=0.0, rtol=0.0)
    _assert_close(control_details['gates'], torch.ones_like(control_details['gates']),
                  atol=0.0, rtol=0.0)


def test_shapes_finiteness_single_and_validation():
    torch.manual_seed(6)
    B, T, D, A = 2, 7, 11, 3
    z = torch.randn(B, T, D)
    actions = torch.randn(B, T - 1, A)
    memory = HACSSMv5Memory(D, A, mode='single')
    mixed, details = memory(z, actions, return_details=True)
    assert mixed.shape == (B, T, D) and torch.isfinite(mixed).all()
    assert details['x'].shape == (B, T, D)
    assert details['priors'].shape == (B, T, 2, D)
    assert details['states'].shape == (B, T, 2, D)
    assert details['gates'].shape == (B, T, 2, 1)
    assert details['betas'].shape == (B, T, 2, D)
    _assert_close(details['route'], torch.tensor([0.0, 1.0]))
    expected_mixed = details['states'][:, :, 1]
    expected_mixed = expected_mixed * torch.rsqrt(
        expected_mixed.square().mean(dim=-1, keepdim=True) + memory.rms_eps)
    _assert_close(mixed, expected_mixed)

    _expect_value_error(lambda: memory(z[:, :, :-1], actions))
    _expect_value_error(lambda: memory(z, actions[:, :-1]))
    _expect_value_error(lambda: memory(z, actions, gate_override=1.1))
    _expect_value_error(lambda: memory(z, actions, beta_override=-0.1))
    _expect_value_error(lambda: HACSSMv5Memory(D, A, mode='unknown'))


def test_identity_rollout_and_beta_roles():
    torch.manual_seed(7)
    B, T, D, A = 2, 5, 8, 3
    z = torch.randn(B, T, D)
    actions = torch.randn(B, T - 1, A)
    memory = HACSSMv5Memory(D, A)
    mixed = memory(z, actions)
    _assert_close(memory.fuse(z, mixed), z, atol=0.0, rtol=0.0,
                  message='zero-init residual is not identity')

    with torch.no_grad():
        memory.W_a.weight.normal_(std=0.2)
    source = torch.randn(B, 2, D)
    rollout_actions = torch.randn(B, 4, A)
    rollout = memory.action_rollout(source, rollout_actions)
    assert rollout.shape == (B, 4, 2, D) and torch.isfinite(rollout).all()
    state = source
    manual = []
    for step in range(rollout_actions.shape[1]):
        state = memory.transition(state, rollout_actions[:, step])
        manual.append(state)
    _assert_close(rollout, torch.stack(manual, dim=1))
    frozen = memory.action_rollout(source, rollout_actions, beta_override=0.0)
    _assert_close(frozen, source.unsqueeze(1).expand_as(frozen))

    # With no action and a fully open gate, the observation correction is exactly beta-scaled.
    probe = HACSSMv5Memory(D, A)
    _, details = probe(z[:, :2], actions[:, :1], gate_override=1.0, return_details=True)
    beta = details['betas'][:, 1]
    expected = details['priors'][:, 1] + beta * (
        details['x'][:, 1].unsqueeze(1) - details['priors'][:, 1])
    _assert_close(details['states'][:, 1], expected)


def test_v4_custom_taus_preserve_default():
    default = HierarchicalActionConditionedMemory(16, 4)
    assert default.K == 3 and default.taus == [2.0, 8.0, 32.0]
    custom = HierarchicalActionConditionedMemory(16, 4, taus=(2, 8))
    assert custom.K == 2 and custom.taus == [2.0, 8.0]
    assert custom.parameter_count() == custom.expected_parameter_count(16, 4, taus=(2, 8))
    z = torch.randn(2, 4, 16)
    actions = torch.randn(2, 3, 4)
    mixed, details = custom(z, actions, return_details=True)
    assert mixed.shape == z.shape and details['states'].shape == (2, 4, 2, 16)


def main():
    tests = [
        test_initialization,
        test_parameter_matching,
        test_monotone_and_fixed_betas,
        test_action_causality_gradient_and_noaction,
        test_static_overrides_masks_and_ssm_control,
        test_shapes_finiteness_single_and_validation,
        test_identity_rollout_and_beta_roles,
        test_v4_custom_taus_preserve_default,
    ]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'PASS all {len(tests)} HACSSM-v5 tests')


if __name__ == '__main__':
    main()
