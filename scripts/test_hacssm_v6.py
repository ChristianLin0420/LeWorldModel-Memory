#!/usr/bin/env python3
"""Core, integration, and schedule tests for dense self-supervised HACSSM-v6."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from lewm.models.memory_model import MemoryLeWorldModel
from scripts.train_popgym import (
    hierarchical_objective_metadata,
    scheduled_hier_loss_weight,
)


V6_MODES = (
    'hacssmv6', 'hacssmv6_noaux', 'hacssmv6_aux_noaction',
    'hacssmv6_uniform', 'hacssmv6_sourcegrad', 'hacssmv6_fastonly',
    'hacssmv6_mediumonly', 'hacssmv6_noaction', 'hacssmv6_static',
    'hacssmv6_single',
)

DYNAMIC_INFERENCE_MODES = V6_MODES[:7]


def _model(mode: str, *, dim: int = 16, actions: int = 3,
           weight: float = 0.02) -> MemoryLeWorldModel:
    return MemoryLeWorldModel(
        img_size=16,
        patch_size=8,
        embed_dim=dim,
        action_dim=actions,
        encoder_layers=1,
        encoder_heads=2,
        predictor_layers=1,
        predictor_heads=4,
        predictor_norm='none',
        history_len=3,
        dropout=0.0,
        sigreg_projections=8,
        encoder_type='precomputed',
        memory_mode='both',
        memory_impl=mode,
        hier_loss_weight=weight,
    )


def _batch(batch: int = 2, length: int = 32, dim: int = 16, action_dim: int = 3):
    torch.manual_seed(61)
    observed = torch.randn(batch, length, dim)
    target = torch.randn(batch, length, dim)
    actions = torch.randn(batch, length - 1, action_dim)
    valid = torch.ones(batch, length, dtype=torch.bool)
    valid[:, 10:16] = False
    return observed, actions, target, valid


def _auxiliary(model: MemoryLeWorldModel, observed, actions, valid):
    _, details = model._inject(observed, actions=actions, return_memory_details=True)
    return model._hierarchical_consistency_loss(details['states'], actions, valid)


def test_modes_are_matched_and_use_dedicated_state_namespace() -> None:
    models = [_model(mode) for mode in V6_MODES]
    assert len({model.mem_hacssmv6.parameter_count() for model in models}) == 1
    assert len({model.num_parameters() for model in models}) == 1
    for model in models:
        assert model.mem_hacssmv6.K == 2
        assert model.mem_hacssmv6.taus == [2.0, 8.0]
        state_keys = tuple(model.state_dict())
        assert any(key.startswith('mem_hacssmv6.') for key in state_keys)
        assert not any(key.startswith('mem_hacsmv4.') for key in state_keys)
    try:
        models[0]._inject(torch.randn(2, 8, 16))
        raise AssertionError('HACSSM-v6 accepted a missing action stream')
    except ValueError as exc:
        assert 'requires actions' in str(exc)


def test_inference_matches_v4_two_level_and_aux_variants_are_identical() -> None:
    torch.manual_seed(62)
    z = torch.randn(2, 12, 16)
    actions = torch.randn(2, 11, 3)
    v4 = _model('hacsmv4_two_noaux')
    full = _model('hacssmv6')
    full.mem_hacssmv6.load_state_dict(v4.mem_hacsmv4.state_dict(), strict=True)
    torch.testing.assert_close(
        full._inject(z, actions=actions), v4._inject(z, actions=actions),
        rtol=0.0, atol=0.0)

    reference = full._inject(z, actions=actions)
    for mode in DYNAMIC_INFERENCE_MODES:
        candidate = _model(mode)
        candidate.load_state_dict(full.state_dict(), strict=True)
        torch.testing.assert_close(
            candidate._inject(z, actions=actions), reference, rtol=0.0, atol=0.0)


def test_dense_visible_pairs_and_same_level_posterior_targets() -> None:
    observed, actions, target, valid = _batch()
    model = _model('hacssmv6')
    losses = model.compute_loss(
        observed, actions, target, valid, memory_update_mask=valid,
        first_post_loss_weight=0.5)
    expected_pairs = {1: 25, 2: 24, 4: 22, 8: 18}
    for horizon, per_episode in expected_pairs.items():
        assert losses[f'hier_pairs_h{horizon}'].item() == observed.shape[0] * per_episode
        assert torch.isfinite(losses[f'hier_loss_h{horizon}'])
    torch.testing.assert_close(
        losses['hier_loss'],
        torch.stack((losses['hier_loss_fast'], losses['hier_loss_medium'])).mean())

    # V6's auxiliary target is the online posterior state, not the clean target latent.
    changed_target = torch.randn_like(target) * 1e5
    changed = model.compute_loss(
        observed, actions, changed_target, valid, memory_update_mask=valid,
        first_post_loss_weight=0.5)
    for key in ('hier_loss', 'hier_loss_fast', 'hier_loss_medium',
                'hier_loss_h1', 'hier_loss_h2', 'hier_loss_h4', 'hier_loss_h8'):
        torch.testing.assert_close(losses[key], changed[key], rtol=0.0, atol=0.0)


def test_hidden_endpoint_is_never_a_consistency_target() -> None:
    torch.manual_seed(63)
    model = _model('hacssmv6')
    states = torch.randn(2, 12, 2, 16)
    actions = torch.randn(2, 11, 3)
    valid = torch.ones(2, 12, dtype=torch.bool)
    valid[:, -1] = False
    baseline = model._hierarchical_consistency_loss(states, actions, valid)
    changed_states = states.clone()
    # The final state is never a source.  If masked endpoint filtering is correct, changing it
    # cannot alter any auxiliary term.
    changed_states[:, -1] = torch.randn_like(changed_states[:, -1]) * 1e6
    changed = model._hierarchical_consistency_loss(changed_states, actions, valid)
    for key, value in baseline.items():
        torch.testing.assert_close(value, changed[key], rtol=0.0, atol=0.0)


def test_detached_objective_trains_only_action_map() -> None:
    observed, actions, _, valid = _batch()
    model = _model('hacssmv6')
    auxiliary = _auxiliary(model, observed, actions, valid)
    auxiliary['hier_loss'].backward()
    nonzero = []
    for name, parameter in model.mem_hacssmv6.named_parameters():
        if parameter.grad is not None and bool((parameter.grad != 0).any()):
            nonzero.append(name)
    assert nonzero == ['W_a.weight']
    assert torch.isfinite(model.mem_hacssmv6.W_a.weight.grad).all()


def test_aux_noaction_has_zero_action_map_gradient() -> None:
    observed, actions, _, valid = _batch()
    model = _model('hacssmv6_aux_noaction')
    auxiliary = _auxiliary(model, observed, actions, valid)
    auxiliary['hier_loss'].backward()
    grad = model.mem_hacssmv6.W_a.weight.grad
    assert grad is not None
    torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0.0, atol=0.0)


def test_sourcegrad_ablation_restores_source_gradient() -> None:
    torch.manual_seed(64)
    model = _model('hacssmv6_sourcegrad')
    states = torch.randn(2, 12, 2, 16, requires_grad=True)
    actions = torch.randn(2, 11, 3)
    valid = torch.ones(2, 12, dtype=torch.bool)
    result = model._hierarchical_consistency_loss(states, actions, valid)
    result['hier_loss'].backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    assert states.grad.norm() > 0


def test_hierarchy_controls_have_locked_horizons_and_finite_level_schema() -> None:
    observed, actions, _, valid = _batch()
    full = _auxiliary(_model('hacssmv6'), observed, actions, valid)
    uniform = _auxiliary(_model('hacssmv6_uniform'), observed, actions, valid)
    fast = _auxiliary(_model('hacssmv6_fastonly'), observed, actions, valid)
    medium = _auxiliary(_model('hacssmv6_mediumonly'), observed, actions, valid)

    for horizon in (1, 2, 4, 8):
        assert uniform[f'hier_pairs_h{horizon}'].item() == 2 * full[f'hier_pairs_h{horizon}'].item()
    assert {key for key in fast if key.startswith('hier_loss_h')} == {
        'hier_loss_h1', 'hier_loss_h2'}
    assert {key for key in medium if key.startswith('hier_loss_h')} == {
        'hier_loss_h4', 'hier_loss_h8'}
    assert fast['hier_loss_medium'].item() == 0.0
    assert medium['hier_loss_fast'].item() == 0.0
    assert torch.isfinite(fast['hier_loss_fast']) and torch.isfinite(fast['hier_loss_medium'])
    assert torch.isfinite(medium['hier_loss_fast']) and torch.isfinite(medium['hier_loss_medium'])
    torch.testing.assert_close(fast['hier_loss'], fast['hier_loss_fast'])
    torch.testing.assert_close(medium['hier_loss'], medium['hier_loss_medium'])


def test_noaux_changes_only_effective_training_weight() -> None:
    observed, actions, target, valid = _batch()
    full = _model('hacssmv6')
    noaux = _model('hacssmv6_noaux')
    noaux.load_state_dict(full.state_dict(), strict=True)
    full_losses = full.compute_loss(
        observed, actions, target, valid, memory_update_mask=valid,
        first_post_loss_weight=0.5)
    noaux_losses = noaux.compute_loss(
        observed, actions, target, valid, memory_update_mask=valid,
        first_post_loss_weight=0.5)
    for key in ('pred_loss', 'sigreg_loss', 'hier_loss'):
        torch.testing.assert_close(full_losses[key], noaux_losses[key])
    torch.testing.assert_close(
        full_losses['loss'] - noaux_losses['loss'], 0.02 * full_losses['hier_loss'])
    assert noaux_losses['hier_loss_weight'].item() == 0.0


def test_v6_bootstrap_schedule_and_metadata_are_exact() -> None:
    base = 0.02
    assert scheduled_hier_loss_weight(base, 'v6_bootstrap', 1) == base
    assert scheduled_hier_loss_weight(base, 'v6_bootstrap', 40) == base
    assert abs(scheduled_hier_loss_weight(base, 'v6_bootstrap', 70) - 0.01) < 1e-12
    assert scheduled_hier_loss_weight(base, 'v6_bootstrap', 100) == 0.0
    assert scheduled_hier_loss_weight(base, 'v6_bootstrap', 200) == 0.0
    metadata = hierarchical_objective_metadata('hacssmv6')
    assert metadata['hier_target_kind'] == 'same_level_posterior_stop_gradient'
    assert metadata['hier_aux_horizons'] == {'fast': [1, 2], 'medium': [4, 8]}
    assert not metadata['hier_source_gradient']
    assert hierarchical_objective_metadata('ssm') == {}


if __name__ == '__main__':
    tests = (
        test_modes_are_matched_and_use_dedicated_state_namespace,
        test_inference_matches_v4_two_level_and_aux_variants_are_identical,
        test_dense_visible_pairs_and_same_level_posterior_targets,
        test_hidden_endpoint_is_never_a_consistency_target,
        test_detached_objective_trains_only_action_map,
        test_aux_noaction_has_zero_action_map_gradient,
        test_sourcegrad_ablation_restores_source_gradient,
        test_hierarchy_controls_have_locked_horizons_and_finite_level_schema,
        test_noaux_changes_only_effective_training_weight,
        test_v6_bootstrap_schedule_and_metadata_are_exact,
    )
    for test in tests:
        test()
        print(f'{test.__name__}: OK')
    print(f'All {len(tests)} HACSSM-v6 tests passed.')
