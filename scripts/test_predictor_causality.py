"""Focused regression tests for predictor normalization and sliding-window causality."""

from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from lewm.models.encoder import Predictor
from lewm.models.memory_model import MemoryLeWorldModel


def _predictor(output_norm: str) -> Predictor:
    torch.manual_seed(17)
    model = Predictor(
        embed_dim=16,
        action_dim=3,
        num_layers=1,
        num_heads=4,
        history_len=3,
        dropout=0.0,
        output_norm=output_norm,
    )
    model.eval()
    return model


@torch.no_grad()
def test_layer_norm_is_batch_independent() -> None:
    """Adding unrelated windows must not change an existing window's prediction."""
    predictor = _predictor('layer')
    anchor_z = torch.randn(1, 3, 16)
    anchor_a = torch.randn(1, 3, 3)
    unrelated_z = 40.0 + 7.0 * torch.randn(9, 3, 16)
    unrelated_a = torch.randn(9, 3, 3)

    alone = predictor(anchor_z, anchor_a)
    together = predictor(
        torch.cat((anchor_z, unrelated_z), dim=0),
        torch.cat((anchor_a, unrelated_a), dim=0),
    )[:1]
    torch.testing.assert_close(alone, together, rtol=1e-6, atol=1e-6)


@torch.no_grad()
def test_layer_norm_is_future_window_independent() -> None:
    """Later flattened windows cannot alter the first window's next-latent output."""
    predictor = _predictor('layer')
    windows = torch.randn(6, 3, 16)
    actions = torch.randn(6, 3, 3)

    first_alone = predictor(windows[:1], actions[:1])[:, -1]
    # Mimic a different future suffix in MemoryLeWorldModel's flattened B*W batch.
    changed = windows.clone()
    changed[1:] = -100.0 + 20.0 * torch.randn_like(changed[1:])
    first_with_changed_future = predictor(changed, actions)[:1, -1]
    torch.testing.assert_close(first_alone, first_with_changed_future, rtol=1e-6, atol=1e-6)


@torch.no_grad()
def test_no_output_norm_is_batch_and_future_window_independent() -> None:
    """The selected HACSM-v4 cohort mode is causal without constraining target scale."""
    predictor = _predictor('none')
    windows = torch.randn(7, 3, 16)
    actions = torch.randn(7, 3, 3)
    first_alone = predictor(windows[:1], actions[:1])
    changed = windows.clone()
    changed[1:] = 80.0 * torch.randn_like(changed[1:])
    first_together = predictor(changed, actions)[:1]
    torch.testing.assert_close(first_alone, first_together, rtol=1e-6, atol=1e-6)


@torch.no_grad()
def test_legacy_batch_norm_exhibits_the_regression() -> None:
    """Guard that the test perturbation detects the legacy batch-stat coupling."""
    predictor = _predictor('batch')
    anchor_z = torch.randn(1, 3, 16)
    anchor_a = torch.randn(1, 3, 3)
    unrelated_z = 100.0 + torch.randn(8, 3, 16)
    unrelated_a = torch.randn(8, 3, 3)

    alone = predictor(anchor_z, anchor_a)
    together = predictor(
        torch.cat((anchor_z, unrelated_z), dim=0),
        torch.cat((anchor_a, unrelated_a), dim=0),
    )[:1]
    assert not torch.allclose(alone, together, rtol=1e-4, atol=1e-4)


def test_model_plumbing_and_backward_compatibility() -> None:
    legacy = MemoryLeWorldModel(
        img_size=16,
        patch_size=8,
        embed_dim=16,
        action_dim=3,
        encoder_layers=1,
        encoder_heads=2,
        predictor_layers=1,
        predictor_heads=4,
        history_len=3,
        sigreg_projections=8,
    )
    causal = MemoryLeWorldModel(
        img_size=16,
        patch_size=8,
        embed_dim=16,
        action_dim=3,
        encoder_layers=1,
        encoder_heads=2,
        predictor_layers=1,
        predictor_heads=4,
        history_len=3,
        predictor_norm='layer',
        sigreg_projections=8,
    )
    causal_unconstrained = MemoryLeWorldModel(
        img_size=16,
        patch_size=8,
        embed_dim=16,
        action_dim=3,
        encoder_layers=1,
        encoder_heads=2,
        predictor_layers=1,
        predictor_heads=4,
        history_len=3,
        predictor_norm='none',
        sigreg_projections=8,
    )
    assert legacy.predictor_norm == 'batch'
    assert isinstance(legacy.predictor.projector[1], nn.BatchNorm1d)
    assert causal.predictor_norm == 'layer'
    assert isinstance(causal.predictor.projector[1], nn.LayerNorm)
    assert causal_unconstrained.predictor_norm == 'none'
    assert isinstance(causal_unconstrained.predictor.projector[1], nn.Identity)
    # BN(track_running_stats=False) and LayerNorm have the same two affine state entries,
    # allowing otherwise-identical experiment checkpoints to retain the same parameter count.
    assert legacy.num_parameters() == causal.num_parameters()


def main() -> None:
    test_layer_norm_is_batch_independent()
    test_layer_norm_is_future_window_independent()
    test_no_output_norm_is_batch_and_future_window_independent()
    test_legacy_batch_norm_exhibits_the_regression()
    test_model_plumbing_and_backward_compatibility()
    print('Predictor causal-normalization tests passed.')


if __name__ == '__main__':
    main()
