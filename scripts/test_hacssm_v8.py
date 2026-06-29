#!/usr/bin/env python3
"""Core invariants for HACSSM-v8 shared-action shrinkage predict/correct memory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lewm.models.memory import SharedActionShrinkageMemory as V8Memory
from lewm.models.memory_model import MemoryLeWorldModel
from scripts.train_popgym import shared_action_shrinkage_metadata


SHARED_MODES = ('learned', 'rho1', 'rho0', 'noaction', 'single')
EXTERNAL_MODES = (
    'hacssmv8', 'hacssmv8_dynamic', 'hacssmv8_static',
    'hacssmv8_levelaction', 'hacssmv8_redundant',
    'hacssmv8_noaction', 'hacssmv8_single',
)


def make_model(mode='hacssmv8'):
    return MemoryLeWorldModel(
        img_size=8, patch_size=4, embed_dim=8, action_dim=3,
        encoder_layers=1, encoder_heads=2, predictor_layers=1, predictor_heads=2,
        history_len=2, dropout=0.0, predictor_norm='none', sigreg_projections=8,
        encoder_type='precomputed', memory_impl=mode, memory_mode='both',
        hier_loss_weight=0.2)


def batch():
    torch.manual_seed(81)
    observations = torch.randn(2, 32, 8)
    targets = torch.randn(2, 32, 8)
    actions = torch.randn(2, 31, 3)
    mask = torch.ones(2, 32, dtype=torch.bool)
    mask[:, 10:16] = False
    return observations, targets, actions, mask


def copy_compact_to_redundant(compact: V8Memory, redundant: V8Memory) -> None:
    compact_state = compact.state_dict()
    redundant_state = redundant.state_dict()
    for name in redundant_state:
        if name == 'W_a.weight':
            shared = compact_state[name]
            redundant_state[name] = shared.repeat(redundant.K, 1)
        else:
            redundant_state[name] = compact_state[name].clone()
    redundant.load_state_dict(redundant_state, strict=True)


def test_parameter_counts_and_physical_shared_head():
    shared = [V8Memory(128, 6, mode=mode) for mode in SHARED_MODES]
    assert all(memory.parameter_count() == 34_566 for memory in shared)
    assert V8Memory.expected_parameter_count(128, 6) == 34_566
    assert all(tuple(memory.W_a.weight.shape) == (256, 6) for memory in shared)
    signatures = [[(name, tuple(value.shape)) for name, value in memory.named_parameters()]
                  for memory in shared]
    assert all(signature == signatures[0] for signature in signatures[1:])

    for mode in ('levelaction', 'redundant'):
        memory = V8Memory(128, 6, mode=mode)
        assert memory.parameter_count() == 36_102
        assert tuple(memory.W_a.weight.shape) == (512, 6)
    assert V8Memory.expected_parameter_count(128, 6, wide_action=True) == 36_102


def test_shrinkage_endpoints_are_exact():
    torch.manual_seed(82)
    learned = V8Memory(8, 3, mode='learned')
    rho1 = V8Memory(8, 3, mode='rho1')
    rho0 = V8Memory(8, 3, mode='rho0')
    with torch.no_grad():
        learned.w_z.normal_(); learned.w_e.normal_(); learned.W_a.weight.normal_()
    rho1.load_state_dict(learned.state_dict(), strict=True)
    rho0.load_state_dict(learned.state_dict(), strict=True)
    z = torch.randn(2, 9, 8)
    actions = torch.randn(2, 8, 3)

    with torch.no_grad():
        learned.shrink_logits.fill_(100.0)
    learned_dynamic, learned_details = learned(z, actions, return_details=True)
    exact_dynamic, dynamic_details = rho1(z, actions, return_details=True)
    assert torch.equal(learned_dynamic, exact_dynamic)
    assert torch.equal(learned_details['gates'], dynamic_details['gates'])

    with torch.no_grad():
        learned.shrink_logits.fill_(-100.0)
    learned_static, static_details = learned(z, actions, return_details=True)
    exact_static, exact_static_details = rho0(z, actions, return_details=True)
    assert torch.equal(learned_static, exact_static)
    assert torch.equal(static_details['gates'], exact_static_details['gates'])
    expected = torch.sigmoid(rho0.gate_bias).view(1, 1, 2, 1).expand(2, 9, -1, -1)
    assert torch.equal(exact_static_details['gates'], expected)


def test_action_sharing_and_redundant_receipt():
    torch.manual_seed(83)
    compact = V8Memory(8, 3, mode='learned')
    redundant = V8Memory(8, 3, mode='redundant')
    with torch.no_grad():
        compact.W_a.weight.normal_()
        compact.W_x.weight.normal_()
        compact.W_o.weight.normal_()
        compact.w_z.normal_()
        compact.w_e.normal_()
    copy_compact_to_redundant(compact, redundant)
    states = torch.randn(4, 2, 8)
    actions = torch.randn(4, 3)
    features = compact._action_features(actions)
    assert torch.equal(features[:, 0], features[:, 1])
    assert torch.equal(compact.transition(states, actions), redundant.transition(states, actions))
    z = torch.randn(4, 10, 8)
    sequence_actions = torch.randn(4, 9, 3)
    assert torch.equal(compact(z, sequence_actions), redundant(z, sequence_actions))


def test_structural_controls():
    z = torch.randn(2, 10, 8)
    actions = torch.randn(2, 9, 3)
    noaction = V8Memory(8, 3, mode='noaction')
    assert torch.equal(noaction(z, actions), noaction(z, torch.zeros_like(actions)))
    single = V8Memory(8, 3, mode='single')
    assert torch.equal(single.route_weights(), torch.tensor([0.0, 1.0]))


def test_model_has_no_teacher_or_auxiliary():
    model = make_model()
    keys = model.state_dict()
    assert any(key.startswith('mem_hacssmv8.') for key in keys)
    assert not any('teacher' in key for key in keys)
    assert not hasattr(model, 'mem_hacssmv8_teacher')
    observations, targets, actions, mask = batch()
    losses = model.compute_loss(
        observations, actions, targets, mask, memory_update_mask=mask,
        first_post_loss_weight=0.5)
    assert set(losses) == {
        'loss', 'pred_loss', 'sigreg_loss',
        'pred_loss_all_valid', 'pred_loss_first_post',
    }
    expected = losses['pred_loss'] + model.sigreg_lambda * losses['sigreg_loss']
    assert torch.equal(losses['loss'], expected)
    model.update_hierarchical_teacher()  # deliberate no-op for V8


def test_all_external_modes_construct_and_run():
    observations, _, actions, mask = batch()
    for mode in EXTERNAL_MODES:
        model = make_model(mode)
        with torch.no_grad():
            injected = model._inject(
                observations, actions=actions, memory_update_mask=mask)
        assert injected.shape == observations.shape
        assert torch.isfinite(injected).all()
        assert not any('teacher' in key for key in model.state_dict())


def test_hidden_clean_targets_do_not_change_v8_loss():
    model = make_model()
    model.eval()
    observations, targets, actions, mask = batch()
    changed = targets.clone()
    changed[:, 10:16] += 100_000.0
    with torch.no_grad():
        left = model.compute_loss(
            observations, actions, targets, mask, memory_update_mask=mask,
            first_post_loss_weight=0.5)
        right = model.compute_loss(
            observations, actions, changed, mask, memory_update_mask=mask,
            first_post_loss_weight=0.5)
    assert left.keys() == right.keys()
    assert all(torch.equal(left[key], right[key]) for key in left)


def test_influence_schema_remains_hierarchical():
    model = make_model()
    observations, _, actions, mask = batch()
    with torch.no_grad():
        model.mem_hacssmv8.W_o.weight.copy_(torch.eye(8))
        influence = model.memory_influence(
            observations, actions, memory_update_mask=mask)
    assert set(influence) == {'pred_full', 'infl_fast', 'infl_slow', 'infl_all'}
    assert all(torch.isfinite(value).all() for value in influence.values())


def test_metadata_is_complete_and_has_no_auxiliary():
    for mode in EXTERNAL_MODES:
        metadata = shared_action_shrinkage_metadata(mode)
        assert metadata['memory_arch_schema_version'] == 8
        assert metadata['memory_inference_taus'] == [2.0, 8.0]
        assert metadata['memory_internal_auxiliary'] == 'none'
        assert metadata['memory_teacher_present'] is False
    assert shared_action_shrinkage_metadata('hacssmv8')['memory_action_kind'] == 'physically_shared'
    assert shared_action_shrinkage_metadata('hacssmv8_dynamic')[
        'memory_shrinkage_kind'] == 'dynamic_only'
    assert shared_action_shrinkage_metadata('hacssmv8_static')[
        'memory_shrinkage_kind'] == 'static_only'
    assert shared_action_shrinkage_metadata('hacssmv8_levelaction')[
        'memory_action_kind'] == 'level_specific'
    assert shared_action_shrinkage_metadata('hacssmv8_redundant')[
        'memory_action_kind'] == 'redundant_shared_average'


def test_cli_exposes_all_modes():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / 'scripts' / 'train_popgym.py'), '--help'],
        cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    help_text = result.stdout + result.stderr
    assert all(mode in help_text for mode in EXTERNAL_MODES)


if __name__ == '__main__':
    tests = (
        test_parameter_counts_and_physical_shared_head,
        test_shrinkage_endpoints_are_exact,
        test_action_sharing_and_redundant_receipt,
        test_structural_controls,
        test_model_has_no_teacher_or_auxiliary,
        test_all_external_modes_construct_and_run,
        test_hidden_clean_targets_do_not_change_v8_loss,
        test_influence_schema_remains_hierarchical,
        test_metadata_is_complete_and_has_no_auxiliary,
        test_cli_exposes_all_modes,
    )
    for test in tests:
        test()
        print(f'{test.__name__}: OK')
    print(f'All {len(tests)} HACSSM-v8 tests passed.')
