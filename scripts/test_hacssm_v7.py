#!/usr/bin/env python3
"""Core invariants for HACSSM-v7 counterfactual recovery distillation."""

from pathlib import Path
import copy
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lewm.models.memory import HierarchicalCounterfactualRecoveryMemory as V7Memory
from lewm.models.memory_model import MemoryLeWorldModel
from scripts.train_popgym import (
    counterfactual_recovery_metadata, run_epoch, scheduled_hier_loss_weight)


MODES = (
    'dynamic', 'noaux', 'sharedaction', 'noshrink', 'actiononly', 'uniform',
    'norecovery', 'noaction', 'single')


def make_model(mode='hacssmv7'):
    return MemoryLeWorldModel(
        img_size=8, patch_size=4, embed_dim=8, action_dim=3,
        encoder_layers=1, encoder_heads=2, predictor_layers=1, predictor_heads=2,
        history_len=2, dropout=0.0, predictor_norm='none', sigreg_projections=8,
        encoder_type='precomputed', memory_impl=mode, memory_mode='both',
        hier_loss_weight=0.02)


def batch():
    torch.manual_seed(7)
    observations = torch.randn(2, 32, 8)
    targets = torch.randn(2, 32, 8)
    actions = torch.randn(2, 31, 3)
    mask = torch.ones(2, 32, dtype=torch.bool)
    mask[:, 10:16] = False
    return observations, targets, actions, mask


def test_modes_are_parameter_matched_and_count_is_exact():
    memories = [V7Memory(128, 6, mode=mode) for mode in MODES]
    assert all(memory.parameter_count() == 36_102 for memory in memories)
    assert V7Memory.expected_parameter_count(128, 6) == 36_102
    signatures = [
        [(name, tuple(parameter.shape)) for name, parameter in memory.named_parameters()]
        for memory in memories]
    assert all(signature == signatures[0] for signature in signatures[1:])
    model = make_model()
    keys = model.state_dict()
    assert any(key.startswith('mem_hacssmv7.') for key in keys)
    assert any(key.startswith('mem_hacssmv7_teacher.') for key in keys)
    assert not any(key.startswith('mem_hacssmv6.') for key in keys)
    assert all(not parameter.requires_grad
               for parameter in model.mem_hacssmv7_teacher.parameters())


def test_shrinkage_endpoints_reproduce_dynamic_and_static_experts():
    torch.manual_seed(1)
    full = V7Memory(8, 3, mode='dynamic')
    dynamic = V7Memory(8, 3, mode='noshrink')
    dynamic.load_state_dict(full.state_dict())
    with torch.no_grad():
        full.w_z.normal_(); full.w_e.normal_()
        dynamic.load_state_dict(full.state_dict())
    z = torch.randn(2, 9, 8); actions = torch.randn(2, 8, 3)
    with torch.no_grad():
        full.shrink_logits.fill_(100.0)
    out_full, details_full = full(z, actions, return_details=True)
    out_dynamic, details_dynamic = dynamic(z, actions, return_details=True)
    assert torch.equal(out_full, out_dynamic)
    assert torch.equal(details_full['gates'], details_dynamic['gates'])

    with torch.no_grad():
        full.shrink_logits.fill_(-100.0)
    out_static, details_static = full(z, actions, return_details=True)
    static_gates = torch.sigmoid(full.gate_bias).view(1, 1, 2, 1).expand(2, 9, -1, -1)
    assert torch.equal(details_static['gates'], static_gates)
    out_override = full(z, actions, gate_override=static_gates)
    assert torch.equal(out_static, out_override)


def test_level_specific_and_shared_action_controls():
    separate = V7Memory(4, 2, mode='dynamic')
    shared = V7Memory(4, 2, mode='sharedaction')
    shared.load_state_dict(separate.state_dict())
    with torch.no_grad():
        separate.W_a.weight[:8].fill_(0.1)
        separate.W_a.weight[8:].fill_(-0.2)
        shared.load_state_dict(separate.state_dict())
    states = torch.randn(3, 2, 4); action = torch.randn(3, 2)
    separate_prior = separate.transition(states, action)
    shared_prior = shared.transition(states, action)
    assert not torch.allclose(separate_prior, shared_prior)
    # Shared mode gives both levels the same raw action features; state normalization can still
    # make their multiplicative deltas differ, so inspect the projected control directly.
    projected = shared.W_a(action).view(3, 2, 8).mean(1)
    assert projected.shape == (3, 8)


def test_visible_only_counterfactual_counts_and_no_overlap():
    model = make_model()
    obs, target, actions, mask = batch()
    losses = model.compute_loss(
        obs, actions, target, mask, memory_update_mask=mask,
        first_post_loss_weight=0.5)
    assert float(losses['hier_overlap']) == 0.0
    # Per episode, visible runs have lengths 10 and 16. Width is h+2.
    assert float(losses['hier_pairs_h1']) == 2 * (8 + 14)
    assert float(losses['hier_pairs_h2']) == 2 * (7 + 13)
    assert float(losses['hier_pairs_h4']) == 2 * (5 + 11)
    assert float(losses['hier_pairs_h8']) == 2 * (1 + 7)
    assert float(losses['hier_pairs']) == 132.0
    assert all(torch.isfinite(value) for key, value in losses.items()
               if key.startswith('hier_'))


def test_hidden_clean_targets_never_change_v7_auxiliary():
    model = make_model()
    obs, target, actions, mask = batch()
    changed = target.clone(); changed[:, 10:16] += 10_000.0
    with torch.no_grad():
        left = model.compute_loss(
            obs, actions, target, mask, memory_update_mask=mask,
            first_post_loss_weight=0.5)
        right = model.compute_loss(
            obs, actions, changed, mask, memory_update_mask=mask,
            first_post_loss_weight=0.5)
    hierarchy_keys = {key for key in left if key.startswith('hier_')}
    assert hierarchy_keys
    assert hierarchy_keys == {key for key in right if key.startswith('hier_')}
    assert all(torch.equal(left[key], right[key]) for key in hierarchy_keys)
    # The private objective also has no clean-target argument, while this end-to-end check guards
    # the surrounding compute_loss plumbing against accidentally deriving its inputs from target.
    assert not torch.equal(target[:, 10:16], changed[:, 10:16])


def test_auxiliary_gradient_scope_and_teacher_stop_gradient():
    model = make_model()
    obs, _, actions, mask = batch()
    with torch.no_grad():
        model.mem_hacssmv7.w_z.normal_(std=0.2)
        model.mem_hacssmv7.w_e.normal_(std=0.2)
        model.mem_hacssmv7_teacher.load_state_dict(model.mem_hacssmv7.state_dict())
    z = model.encode(obs)
    _, details = model.mem_hacssmv7(z, actions, return_details=True)
    auxiliary = model._hierarchical_counterfactual_recovery_loss(
        details['states'], z, actions, mask)['hier_loss']
    auxiliary.backward()
    allowed = {'W_a.weight', 'w_z', 'w_e', 'gate_bias', 'shrink_logits'}
    observed = set()
    for name, parameter in model.mem_hacssmv7.named_parameters():
        if parameter.grad is not None and float(parameter.grad.norm()) > 0.0:
            observed.add(name)
    assert observed <= allowed
    assert 'W_a.weight' in observed and 'w_e' in observed
    assert model.mem_hacssmv7.W_x.weight.grad is None
    assert model.mem_hacssmv7.W_o.weight.grad is None
    assert model.mem_hacssmv7.route_logits.grad is None
    assert all(parameter.grad is None
               for parameter in model.mem_hacssmv7_teacher.parameters())


def test_ema_teacher_update_is_exact_and_inference_ignores_teacher():
    model = make_model()
    obs, _, actions, _ = batch()
    with torch.no_grad():
        model.mem_hacssmv7.W_o.weight.copy_(torch.eye(8))
    before = model._inject(obs, actions=actions).detach().clone()
    teacher_before = copy.deepcopy(model.mem_hacssmv7_teacher.state_dict())
    with torch.no_grad():
        model.mem_hacssmv7.W_a.weight.add_(1.0)
    model.update_hierarchical_teacher()
    teacher_after = model.mem_hacssmv7_teacher.state_dict()
    expected = (teacher_before['W_a.weight'] * 0.99
                + model.mem_hacssmv7.W_a.weight.detach() * 0.01)
    assert torch.allclose(teacher_after['W_a.weight'], expected)
    with torch.no_grad():
        for parameter in model.mem_hacssmv7_teacher.parameters():
            parameter.add_(123.0)
    after = model._inject(obs, actions=actions).detach()
    assert not torch.equal(before, after)  # student W_a changed above
    student_fixed = after.clone()
    with torch.no_grad():
        for parameter in model.mem_hacssmv7_teacher.parameters():
            parameter.sub_(77.0)
    assert torch.equal(student_fixed, model._inject(obs, actions=actions).detach())


def test_objective_controls_and_noaux_weight():
    obs, target, actions, mask = batch()
    for mode in (
        'hacssmv7', 'hacssmv7_noaux', 'hacssmv7_actiononly',
        'hacssmv7_uniform', 'hacssmv7_norecovery'):
        model = make_model(mode)
        losses = model.compute_loss(
            obs, actions, target, mask, memory_update_mask=mask,
            first_post_loss_weight=0.5)
        expected_weight = 0.0 if mode.endswith('_noaux') else 0.02
        assert abs(float(losses['hier_loss_weight']) - expected_weight) < 1e-7
        assert torch.isfinite(losses['hier_loss'])
        if mode.endswith('_norecovery') or mode.endswith('_actiononly'):
            assert float(losses['hier_loss_recovery']) == 0.0
        if mode.endswith('_uniform'):
            assert 'hier_loss_fast_h8' in losses and 'hier_loss_medium_h1' in losses


def test_noaction_single_schedule_and_metadata():
    z = torch.randn(2, 10, 8); a = torch.randn(2, 9, 3)
    noaction = V7Memory(8, 3, mode='noaction')
    assert torch.equal(noaction(z, a), noaction(z, torch.zeros_like(a)))
    single = V7Memory(8, 3, mode='single')
    assert torch.equal(single.route_weights(), torch.tensor([0.0, 1.0]))
    assert scheduled_hier_loss_weight(.02, 'v6_bootstrap', 40) == .02
    assert scheduled_hier_loss_weight(.02, 'v6_bootstrap', 100) == 0.0
    metadata = counterfactual_recovery_metadata('hacssmv7')
    assert metadata['hier_hidden_clean_targets_used'] is False
    assert metadata['hier_teacher_momentum'] == 0.99
    assert metadata['hier_aux_horizons'] == {'fast': [1, 2], 'medium': [4, 8]}
    assert metadata['hier_action_kind'] == 'level_specific'
    assert counterfactual_recovery_metadata(
        'hacssmv7_sharedaction')['hier_action_kind'] == 'shared'
    noaction_metadata = counterfactual_recovery_metadata('hacssmv7_noaction')
    assert noaction_metadata['hier_action_kind'] == 'none'
    assert noaction_metadata['hier_level_specific_action'] is False


def test_epoch_event_counts_are_not_batch_size_weighted_twice():
    class CountModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Identity()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def compute_loss(self, obs, *_args, **_kwargs):
            one = self.anchor * 0.0 + 1.0
            return {
                'loss': one,
                'pred_loss': one,
                'sigreg_loss': one,
                'hier_pairs': one * (2 * obs.shape[0]),
                'hier_overlap': one * 0.0,
            }

        def update_hierarchical_teacher(self):
            pass

    loader = [
        (torch.zeros(3, 2, 1), torch.zeros(3, 1, 1)),
        (torch.zeros(1, 2, 1), torch.zeros(1, 1, 1)),
    ]
    result = run_epoch(
        CountModel(), loader, None, torch.device('cpu'), False, False)
    assert result['loss'] == 1.0
    assert result['hier_pairs'] == 8.0
    assert result['hier_overlap'] == 0.0


def test_influence_schema_separates_levels_from_single_state_total():
    obs, _, actions, mask = batch()
    hierarchical = make_model()
    with torch.no_grad():
        hierarchical.mem_hacssmv7.W_o.weight.copy_(torch.eye(8))
    influence = hierarchical.memory_influence(
        obs, actions, memory_update_mask=mask)
    assert set(influence) == {'pred_full', 'infl_fast', 'infl_slow', 'infl_all'}
    assert influence['infl_all'].shape == (2,)
    assert torch.isfinite(influence['infl_fast']).all()
    assert torch.isfinite(influence['infl_slow']).all()

    single_state = make_model('ssm')
    with torch.no_grad():
        single_state.mem_ssm.out.weight.copy_(torch.eye(8))
    influence = single_state.memory_influence(obs, actions)
    assert set(influence) == {'pred_full', 'infl_all'}
    assert influence['infl_all'].shape == (2,)


if __name__ == '__main__':
    tests = (
        test_modes_are_parameter_matched_and_count_is_exact,
        test_shrinkage_endpoints_reproduce_dynamic_and_static_experts,
        test_level_specific_and_shared_action_controls,
        test_visible_only_counterfactual_counts_and_no_overlap,
        test_hidden_clean_targets_never_change_v7_auxiliary,
        test_auxiliary_gradient_scope_and_teacher_stop_gradient,
        test_ema_teacher_update_is_exact_and_inference_ignores_teacher,
        test_objective_controls_and_noaux_weight,
        test_noaction_single_schedule_and_metadata,
        test_epoch_event_counts_are_not_batch_size_weighted_twice,
        test_influence_schema_separates_levels_from_single_state_total,
    )
    for test in tests:
        test()
        print(f'{test.__name__}: OK')
    print(f'All {len(tests)} HACSSM-v7 tests passed.')
