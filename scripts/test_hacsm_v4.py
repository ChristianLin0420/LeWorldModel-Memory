"""Focused invariants for the HACSM-v4 recurrent memory."""

import math
import os
import sys
from contextlib import contextmanager

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lewm.models.memory import (HACSMv4Memory, HierarchicalActionConditionedMemory,
                                tau_to_alpha)


@contextmanager
def _raises(exception_type, match: str):
    """Small dependency-free equivalent of ``pytest.raises(..., match=...)``."""
    try:
        yield
    except exception_type as exc:
        if match not in str(exc) and match.replace('\\', '') not in str(exc):
            raise AssertionError(f'{exc!r} does not contain {match!r}') from exc
    else:
        raise AssertionError(f'expected {exception_type.__name__} containing {match!r}')


def _randomize_action_map(memory: HierarchicalActionConditionedMemory) -> None:
    with torch.no_grad():
        memory.W_a.weight.normal_(mean=0.0, std=0.25)


def test_exact_initialization_shapes_and_residual_identity():
    torch.manual_seed(10)
    B, T, D, A = 2, 6, 8, 4
    z = torch.randn(B, T, D)
    actions = torch.randn(B, T - 1, A)
    memory = HierarchicalActionConditionedMemory(D, A)

    assert HACSMv4Memory is HierarchicalActionConditionedMemory
    assert memory.K == 3
    assert memory.taus == [2.0, 8.0, 32.0]
    assert torch.equal(memory.W_x.weight, torch.eye(D))
    assert torch.count_nonzero(memory.W_a.weight) == 0
    assert torch.count_nonzero(memory.W_o.weight) == 0
    assert torch.count_nonzero(memory.w_z) == 0
    assert torch.count_nonzero(memory.w_e) == 0
    assert torch.equal(memory.gate_bias, torch.full((3,), 2.0))
    expected_betas = torch.tensor([tau_to_alpha(tau) for tau in (2, 8, 32)])
    assert torch.allclose(memory.betas, expected_betas)
    assert torch.allclose(memory.route_weights(), torch.full((3,), 1 / 3))

    mixed, details = memory(z, actions, return_details=True)
    assert mixed.shape == (B, T, D)
    assert set(details) == {'x', 'priors', 'states', 'gates', 'route'}
    assert details['x'].shape == (B, T, D)
    assert details['priors'].shape == (B, T, 3, D)
    assert details['states'].shape == (B, T, 3, D)
    assert details['gates'].shape == (B, T, 3, 1)
    assert details['route'].shape == (3,)
    assert torch.allclose(details['x'], z)
    assert torch.allclose(details['states'][:, 0], z[:, 0].unsqueeze(1).expand(-1, 3, -1))
    assert torch.allclose(details['priors'][:, 0], details['states'][:, 0])
    assert torch.allclose(details['gates'], torch.full_like(details['gates'], torch.sigmoid(torch.tensor(2.0))))
    assert torch.allclose(memory.fuse(z, mixed), z)
    assert torch.allclose(
        mixed.square().mean(-1), torch.ones(B, T), atol=5e-5, rtol=5e-5)


def test_parameter_formula_and_matched_control_capacity():
    D, A = 11, 5
    memories = [HierarchicalActionConditionedMemory(D, A, mode=mode)
                for mode in ('dynamic', 'static', 'noaction', 'single', 'oracle')]
    expected = 2 * D * D + 2 * A * D + 2 * D + 2 * 3
    assert expected == HierarchicalActionConditionedMemory.expected_parameter_count(D, A)
    assert {memory.parameter_count() for memory in memories} == {expected}
    assert {sum(parameter.numel() for parameter in memory.parameters())
            for memory in memories} == {expected}

    single = memories[-2]
    assert torch.equal(single.route_weights(), torch.tensor([0.0, 1.0, 0.0]))
    assert single.route_logits.requires_grad


def test_single_mode_changes_only_the_read_route():
    torch.manual_seed(16)
    B, T, D, A = 2, 6, 7, 3
    dynamic = HierarchicalActionConditionedMemory(D, A, mode='dynamic')
    single = HierarchicalActionConditionedMemory(D, A, mode='single')
    _randomize_action_map(dynamic)
    with torch.no_grad():
        dynamic.w_z.normal_()
        dynamic.w_e.normal_()
    single.load_state_dict(dynamic.state_dict())
    z = torch.randn(B, T, D)
    actions = torch.randn(B, T - 1, A)

    _, dynamic_details = dynamic(z, actions, return_details=True)
    single_mixed, single_details = single(z, actions, return_details=True)
    assert torch.allclose(single_details['priors'], dynamic_details['priors'])
    assert torch.allclose(single_details['states'], dynamic_details['states'])
    assert torch.allclose(single_details['gates'], dynamic_details['gates'])
    expected = single_details['states'][:, :, 1]
    expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + single.rms_eps)
    assert torch.allclose(single_mixed, expected)


def test_action_indexing_and_no_future_action_leakage():
    torch.manual_seed(11)
    B, T, D, A = 2, 7, 6, 3
    memory = HierarchicalActionConditionedMemory(D, A)
    _randomize_action_map(memory)
    z = torch.randn(B, T, D)
    actions = torch.randn(B, T - 1, A)

    _, base = memory(z, actions, gate_override=0.0, return_details=True)
    changed = actions.clone()
    changed[:, 3] += 9.0                 # a_3 maps state 3 -> state 4
    _, intervention = memory(z, changed, gate_override=0.0, return_details=True)
    assert torch.allclose(base['states'][:, :4], intervention['states'][:, :4])
    assert not torch.allclose(base['states'][:, 4], intervention['states'][:, 4])

    # Every stored prior uses exactly the preceding action.
    for t in range(1, T):
        expected = memory.transition(base['states'][:, t - 1], actions[:, t - 1])
        assert torch.allclose(base['priors'][:, t], expected, atol=1e-6)

    # The rollout excludes its source; output j is source advanced by actions 0..j.
    source = base['states'][:, 2]
    rollout_actions = actions[:, 2:6]
    rollout = memory.action_rollout(source, rollout_actions)
    assert rollout.shape == (B, 4, 3, D)
    state = source
    for step in range(4):
        state = memory.transition(state, rollout_actions[:, step])
        assert torch.allclose(rollout[:, step], state, atol=1e-6)


def test_closed_correction_gate_and_action_override():
    torch.manual_seed(12)
    B, T, D, A = 2, 5, 7, 2
    memory = HierarchicalActionConditionedMemory(D, A)
    z = torch.randn(B, T, D)
    actions = torch.randn(B, T - 1, A)

    # At initialization the action prior is identity, so a closed correction exactly freezes state.
    _, details = memory(z, actions, gate_override=0.0, return_details=True)
    warm = z[:, :1].unsqueeze(2).expand(-1, T, 3, -1)
    assert torch.allclose(details['states'], warm)
    assert torch.allclose(details['states'][:, 1:], details['priors'][:, 1:])

    # An explicit zero-action intervention reproduces the learned-map initialization even after
    # the action map is nonzero, while a real action changes the prior.
    _randomize_action_map(memory)
    _, real = memory(z, actions, gate_override=0.0, return_details=True)
    _, zeroed = memory(
        z, actions, gate_override=0.0, action_override=0.0, return_details=True)
    assert not torch.allclose(real['states'][:, 1:], zeroed['states'][:, 1:])
    assert torch.allclose(zeroed['states'], warm)


def test_static_and_nonoracle_masks_are_invariant():
    torch.manual_seed(13)
    B, T, D, A = 3, 6, 8, 3
    z = torch.randn(B, T, D)
    actions = torch.randn(B, T - 1, A)
    mask_a = torch.zeros(B, T)
    mask_b = torch.ones(B, T)

    static = HierarchicalActionConditionedMemory(D, A, mode='static')
    _, details_a = static(z, actions, memory_update_mask=mask_a, return_details=True)
    output_b, details_b = static(z, actions, memory_update_mask=mask_b, return_details=True)
    assert torch.allclose(details_a['gates'], details_b['gates'])
    assert torch.allclose(details_a['gates'], torch.sigmoid(static.gate_bias).view(1, 1, 3, 1))
    assert torch.allclose(static(z, actions, memory_update_mask=mask_a), output_b)

    dynamic = HierarchicalActionConditionedMemory(D, A, mode='dynamic')
    assert torch.allclose(
        dynamic(z, actions, memory_update_mask=mask_a),
        dynamic(z, actions, memory_update_mask=mask_b),
    )


def test_oracle_fails_closed_and_uses_only_explicit_mask():
    torch.manual_seed(14)
    B, T, D, A = 2, 5, 6, 2
    oracle = HierarchicalActionConditionedMemory(D, A, mode='oracle')
    z = torch.randn(B, T, D)
    actions = torch.randn(B, T - 1, A)

    with _raises(ValueError, 'memory_update_mask'):
        oracle(z, actions)
    with _raises(ValueError, 'memory_update_mask'):
        oracle(z, actions, gate_override=0.5)

    visibility = torch.tensor([[1, 1, 0, 0, 1], [1, 0, 0, 1, 1]], dtype=torch.bool)
    _, details = oracle(z, actions, memory_update_mask=visibility, return_details=True)
    expected = visibility[:, :, None, None].expand(B, T, 3, 1)
    assert torch.equal(details['gates'].bool(), expected)

    with _raises(ValueError, '[0,1]'):
        oracle(z, actions, memory_update_mask=visibility, gate_override=1.01)


def test_action_path_has_gradient_and_noaction_is_action_invariant():
    torch.manual_seed(15)
    B, D, A = 4, 9, 3
    source = torch.randn(B, 3, D)
    actions = torch.randn(B, A)
    dynamic = HierarchicalActionConditionedMemory(D, A)

    # W_a starts at zero but is immediately trainable through the action-only prior.
    transitioned = dynamic.transition(source, actions)
    transitioned.square().mean().backward()
    assert dynamic.W_a.weight.grad is not None
    assert torch.isfinite(dynamic.W_a.weight.grad).all()
    assert dynamic.W_a.weight.grad.norm() > 0

    _randomize_action_map(dynamic)
    assert not torch.allclose(
        dynamic.transition(source, actions), dynamic.transition(source, actions + 1.0))

    noaction = HierarchicalActionConditionedMemory(D, A, mode='noaction')
    noaction.load_state_dict(dynamic.state_dict())
    assert torch.allclose(
        noaction.transition(source, actions), noaction.transition(source, actions + 1.0))


def test_strict_shape_finite_and_range_validation():
    B, T, D, A = 2, 4, 5, 2
    memory = HierarchicalActionConditionedMemory(D, A)
    z = torch.randn(B, T, D)
    actions = torch.randn(B, T - 1, A)

    with _raises(ValueError, 'shape'):
        memory(z, actions[:, :-1])
    with _raises(ValueError, 'latent dim'):
        memory(torch.randn(B, T, D + 1), actions)
    bad_actions = actions.clone()
    bad_actions[0, 0, 0] = math.nan
    with _raises(ValueError, 'non-finite'):
        memory(z, bad_actions)
    bad_step = actions[:, 0].clone()
    bad_step[0, 0] = math.nan
    with _raises(ValueError, 'finite'):
        memory.transition(torch.randn(B, 3, D), bad_step, action_override=0.0)
    with _raises(ValueError, '[0,1]'):
        memory(z, actions, gate_override=-0.1)
    with _raises(ValueError, 'broadcastable'):
        memory(z, actions, action_override=torch.zeros(B, T, A))
    with _raises(ValueError, 'broadcastable'):
        memory(z, actions, memory_update_mask=torch.ones(B, T + 1))
    with _raises(ValueError, 'shape (B,T,D)'):
        memory.fuse(z[:, 0], z[:, 0])

    # Natural per-level and time-by-level intervention shapes are explicitly supported.
    _, per_level = memory(
        z, actions, gate_override=torch.tensor([0.0, 0.5, 1.0]), return_details=True)
    assert torch.allclose(
        per_level['gates'],
        torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 3, 1).expand(B, T, -1, -1),
    )
    time_level = torch.rand(T, 3)
    _, per_time_level = memory(z, actions, gate_override=time_level, return_details=True)
    assert torch.allclose(
        per_time_level['gates'], time_level.view(1, T, 3, 1).expand(B, -1, -1, -1))


if __name__ == '__main__':
    tests = [
        test_exact_initialization_shapes_and_residual_identity,
        test_parameter_formula_and_matched_control_capacity,
        test_single_mode_changes_only_the_read_route,
        test_action_indexing_and_no_future_action_leakage,
        test_closed_correction_gate_and_action_override,
        test_static_and_nonoracle_masks_are_invariant,
        test_oracle_fails_closed_and_uses_only_explicit_mask,
        test_action_path_has_gradient_and_noaction_is_action_invariant,
        test_strict_shape_finite_and_range_validation,
    ]
    for test in tests:
        test()
        print(f'{test.__name__}: OK')
    print(f'All {len(tests)} HACSM-v4 tests passed.')
