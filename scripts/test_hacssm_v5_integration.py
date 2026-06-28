#!/usr/bin/env python3
"""Integration, boundary-loss, and schedule tests for HACSSM-v5."""

from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from lewm.models.memory_model import MemoryLeWorldModel
from scripts.train_popgym import scheduled_hier_loss_weight


V5_MODES = (
    'hacssmv5', 'hacssmv5_static', 'hacssmv5_noaction',
    'hacssmv5_fixedbeta', 'hacssmv5_fixedbeta_noaux',
    'hacssmv5_noaux', 'hacssmv5_single', 'hacssmv5_ssmcontrol',
)


def _model(mode: str, *, dim: int = 16, actions: int = 3,
           weight: float = 0.05) -> MemoryLeWorldModel:
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
    torch.manual_seed(51)
    observed = torch.randn(batch, length, dim)
    target = torch.randn(batch, length, dim)
    actions = torch.randn(batch, length - 1, action_dim)
    valid = torch.ones(batch, length, dtype=torch.bool)
    valid[:, 10:16] = False
    return observed, actions, target, valid


def test_modes_parameter_match_and_require_actions() -> None:
    models = [_model(mode) for mode in V5_MODES]
    memory_counts = {model.mem_hacssmv5.parameter_count() for model in models}
    model_counts = {model.num_parameters() for model in models}
    assert len(memory_counts) == 1
    assert len(model_counts) == 1
    assert next(iter(memory_counts)) == 2 * 16 * 16 + 2 * 3 * 16 + 4 * 16 + 4
    try:
        models[0]._inject(torch.randn(2, 8, 16))
        raise AssertionError('HACSSM-v5 accepted a missing action stream')
    except ValueError as exc:
        assert 'requires actions' in str(exc)


def test_boundary_pairs_and_hidden_targets_are_excluded() -> None:
    observed, actions, target, valid = _batch()
    model = _model('hacssmv5')
    losses = model.compute_loss(
        observed, actions, target, valid, memory_update_mask=valid,
        first_post_loss_weight=0.5)
    for horizon in (1, 2, 4, 8):
        assert losses[f'hier_pairs_h{horizon}'].item() == observed.shape[0]
    torch.testing.assert_close(
        losses['hier_loss'],
        torch.stack((losses['hier_loss_fast'], losses['hier_loss_medium'])).mean())
    torch.testing.assert_close(
        losses['loss'],
        losses['pred_loss'] + model.sigreg_lambda * losses['sigreg_loss']
        + 0.05 * losses['hier_loss'])

    hidden_changed = target.clone()
    hidden_changed[:, 10:16] = torch.randn_like(hidden_changed[:, 10:16]) * 1e6
    changed = model.compute_loss(
        observed, actions, hidden_changed, valid, memory_update_mask=valid,
        first_post_loss_weight=0.5)
    for key in ('loss', 'pred_loss', 'sigreg_loss', 'hier_loss',
                'hier_loss_h1', 'hier_loss_h2', 'hier_loss_h4', 'hier_loss_h8'):
        torch.testing.assert_close(losses[key], changed[key], rtol=0.0, atol=0.0)

    boundary_changed = target.clone()
    boundary_changed[:, 16] += 100.0
    boundary = model.compute_loss(
        observed, actions, boundary_changed, valid, memory_update_mask=valid,
        first_post_loss_weight=0.5)
    assert not torch.allclose(losses['hier_loss'], boundary['hier_loss'])


def test_boundary_targets_stop_gradient_and_train_action_map() -> None:
    observed, actions, target, valid = _batch()
    model = _model('hacssmv5')
    _, details = model._inject(observed, actions=actions, return_memory_details=True)
    differentiable_target = target.clone().requires_grad_(True)
    auxiliary = model._hierarchical_boundary_loss(
        details['states'], actions, differentiable_target, valid)
    auxiliary['hier_loss'].backward()
    assert differentiable_target.grad is None
    grad = model.mem_hacssmv5.W_a.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0


def test_noaux_changes_only_the_training_objective() -> None:
    observed, actions, target, valid = _batch()
    full = _model('hacssmv5')
    noaux = _model('hacssmv5_noaux')
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
        full_losses['loss'] - noaux_losses['loss'], 0.05 * full_losses['hier_loss'])
    assert noaux_losses['hier_loss_weight'].item() == 0.0


def test_frontloaded_schedule_is_exact() -> None:
    base = 0.05
    assert scheduled_hier_loss_weight(base, 'v5_frontload', 1) == base
    assert scheduled_hier_loss_weight(base, 'v5_frontload', 20) == base
    assert abs(scheduled_hier_loss_weight(base, 'v5_frontload', 70) - 0.025) < 1e-12
    assert abs(scheduled_hier_loss_weight(base, 'v5_frontload', 119)) < 2e-5
    assert abs(scheduled_hier_loss_weight(base, 'v5_frontload', 120)) < 1e-15
    assert scheduled_hier_loss_weight(base, 'v5_frontload', 121) == 0.0
    assert scheduled_hier_loss_weight(base, 'fixed', 200) == base


def test_two_level_v4_control_and_predictor_causality() -> None:
    observed, actions, target, valid = _batch()
    model = _model('hacsmv4_two_noaux')
    assert model.mem_hacsmv4.K == 2
    assert model.mem_hacsmv4.taus == [2.0, 8.0]
    losses = model.compute_loss(
        observed, actions, target, valid, memory_update_mask=valid,
        first_post_loss_weight=0.5)
    assert 'hier_loss_slow' not in losses
    assert losses['hier_loss_weight'].item() == 0.0
    assert not any(isinstance(module, nn.BatchNorm1d) for module in model.predictor.modules())


if __name__ == '__main__':
    tests = (
        test_modes_parameter_match_and_require_actions,
        test_boundary_pairs_and_hidden_targets_are_excluded,
        test_boundary_targets_stop_gradient_and_train_action_map,
        test_noaux_changes_only_the_training_objective,
        test_frontloaded_schedule_is_exact,
        test_two_level_v4_control_and_predictor_causality,
    )
    for test in tests:
        test()
        print(f'{test.__name__}: OK')
    print(f'All {len(tests)} HACSSM-v5 integration tests passed.')
