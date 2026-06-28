"""Integration and loss-indexing tests for HACSM-v4 inside MemoryLeWorldModel."""

from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from lewm.models.memory_model import MemoryLeWorldModel


MODES = (
    'hacsmv4', 'hacsmv4_static', 'hacsmv4_noaction',
    'hacsmv4_noaux', 'hacsmv4_single', 'hacsmv4_oracle',
)


def _model(mode: str, *, dim: int = 16, actions: int = 3) -> MemoryLeWorldModel:
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
        hier_loss_weight=0.1,
    )


def _batch(batch: int = 2, length: int = 32, dim: int = 16, action_dim: int = 3):
    torch.manual_seed(31)
    observed = torch.randn(batch, length, dim)
    target = torch.randn(batch, length, dim)
    action = torch.randn(batch, length - 1, action_dim)
    valid = torch.ones(batch, length, dtype=torch.bool)
    valid[:, 10:16] = False
    return observed, action, target, valid


def test_modes_are_parameter_matched_and_require_actions() -> None:
    models = [_model(mode) for mode in MODES]
    memory_counts = [model.mem_hacsmv4.parameter_count() for model in models]
    assert len(set(memory_counts)) == 1
    assert len({model.num_parameters() for model in models}) == 1
    assert torch.equal(
        models[4].mem_hacsmv4.route_weights(), torch.tensor([0.0, 1.0, 0.0]))
    try:
        models[0]._inject(torch.randn(2, 8, 16))
        raise AssertionError('HACSM-v4 accepted a missing action stream')
    except ValueError as exc:
        assert 'requires actions' in str(exc)


def test_loss_pair_counts_balance_and_no_hidden_target_use() -> None:
    observed, action, target, valid = _batch()
    model = _model('hacsmv4')
    losses = model.compute_loss(
        observed, action, target, valid, memory_update_mask=valid,
        first_post_loss_weight=0.5)
    expected_pairs = {1: 25, 2: 24, 4: 22, 8: 18, 16: 16}
    for horizon, per_episode in expected_pairs.items():
        assert losses[f'hier_pairs_h{horizon}'].item() == observed.shape[0] * per_episode
        assert torch.isfinite(losses[f'hier_loss_h{horizon}'])
    torch.testing.assert_close(
        losses['hier_loss'],
        torch.stack((losses['hier_loss_fast'], losses['hier_loss_medium'],
                     losses['hier_loss_slow'])).mean())
    torch.testing.assert_close(
        losses['loss'],
        losses['pred_loss'] + model.sigreg_lambda * losses['sigreg_loss']
        + 0.1 * losses['hier_loss'])

    changed_target = target.clone()
    changed_target[:, 10:16] = 1e6 * torch.randn_like(changed_target[:, 10:16])
    changed = model.compute_loss(
        observed, action, changed_target, valid, memory_update_mask=valid,
        first_post_loss_weight=0.5)
    for key in ('loss', 'pred_loss', 'sigreg_loss', 'hier_loss',
                'hier_loss_h1', 'hier_loss_h2', 'hier_loss_h4',
                'hier_loss_h8', 'hier_loss_h16'):
        torch.testing.assert_close(losses[key], changed[key], rtol=0.0, atol=0.0)


def test_auxiliary_targets_stop_gradient_and_action_map_learns() -> None:
    observed, action, target, valid = _batch()
    model = _model('hacsmv4')
    _, details = model._inject(observed, actions=action, return_memory_details=True)
    detached_test_target = target.clone().requires_grad_(True)
    auxiliary = model._hierarchical_auxiliary_loss(
        details['states'], action, detached_test_target, valid)
    auxiliary['hier_loss'].backward()
    assert detached_test_target.grad is None
    grad = model.mem_hacsmv4.W_a.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0


def test_noaux_removes_only_auxiliary_objective() -> None:
    observed, action, target, valid = _batch()
    full = _model('hacsmv4')
    noaux = _model('hacsmv4_noaux')
    noaux.load_state_dict(full.state_dict(), strict=True)
    full_losses = full.compute_loss(
        observed, action, target, valid, memory_update_mask=valid,
        first_post_loss_weight=0.5)
    noaux_losses = noaux.compute_loss(
        observed, action, target, valid, memory_update_mask=valid,
        first_post_loss_weight=0.5)
    for key in ('pred_loss', 'sigreg_loss', 'hier_loss'):
        torch.testing.assert_close(full_losses[key], noaux_losses[key])
    torch.testing.assert_close(
        full_losses['loss'] - noaux_losses['loss'], 0.1 * full_losses['hier_loss'])
    assert noaux_losses['hier_loss_weight'].item() == 0.0


def test_predictor_has_no_cross_window_batch_norm() -> None:
    model = _model('hacsmv4')
    assert model.predictor_norm == 'none'
    assert not any(isinstance(module, nn.BatchNorm1d) for module in model.predictor.modules())


if __name__ == '__main__':
    tests = (
        test_modes_are_parameter_matched_and_require_actions,
        test_loss_pair_counts_balance_and_no_hidden_target_use,
        test_auxiliary_targets_stop_gradient_and_action_map_learns,
        test_noaux_removes_only_auxiliary_objective,
        test_predictor_has_no_cross_window_batch_norm,
    )
    for test in tests:
        test()
        print(f'{test.__name__}: OK')
    print(f'All {len(tests)} HACSM-v4 integration tests passed.')
