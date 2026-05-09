"""
Quick test script to verify LeWorldModel implementation.
Tests: forward pass, loss computation, SIGReg, planning, end-to-end training step.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lewm.models.leworldmodel import LeWorldModel
from lewm.models.sigreg import SIGReg


def test_encoder():
    """Test encoder forward pass."""
    print("Testing encoder...")
    from lewm.models.encoder import ViTTinyEncoder

    encoder = ViTTinyEncoder(img_size=64, patch_size=8, embed_dim=64, num_heads=4)
    x = torch.randn(4, 3, 64, 64)
    z = encoder(x)
    assert z.shape == (4, 64), f"Expected (4, 64), got {z.shape}"
    print(f"  Encoder output shape: {z.shape} OK")


def test_predictor():
    """Test predictor forward pass."""
    print("Testing predictor...")
    from lewm.models.encoder import Predictor

    predictor = Predictor(embed_dim=64, action_dim=2, num_layers=2, num_heads=4, history_len=3)
    z = torch.randn(4, 3, 64)  # (B, N, D)
    a = torch.randn(4, 3, 2)  # (B, N, A)
    z_pred = predictor(z, a)
    assert z_pred.shape == (4, 3, 64), f"Expected (4, 3, 64), got {z_pred.shape}"
    print(f"  Predictor output shape: {z_pred.shape} OK")


def test_sigreg():
    """Test SIGReg regularizer."""
    print("Testing SIGReg...")
    reg = SIGReg(num_projections=256, embed_dim=64)
    z = torch.randn(32, 64)
    loss = reg(z)
    assert loss.dim() == 0, f"Expected scalar, got {loss.shape}"
    print(f"  SIGReg loss: {loss.item():.4f} OK")

    # Test with 3D input
    z3d = torch.randn(32, 3, 64)
    loss3d = reg(z3d)
    print(f"  SIGReg loss (3D input): {loss3d.item():.4f} OK")


def test_full_model():
    """Test full LeWorldModel forward pass and loss."""
    print("Testing full LeWorldModel...")
    model = LeWorldModel(
        img_size=64,
        patch_size=8,
        embed_dim=64,
        action_dim=2,
        encoder_layers=2,
        encoder_heads=2,
        predictor_layers=2,
        predictor_heads=4,
        history_len=3,
        sigreg_lambda=0.1,
        sigreg_projections=256,
    )

    # Create dummy data
    B, N = 4, 3
    obs = torch.randn(B, N + 1, 3, 64, 64)
    act = torch.randn(B, N, 2)

    losses = model.compute_loss(obs, act)
    assert 'loss' in losses
    assert 'pred_loss' in losses
    assert 'sigreg_loss' in losses
    print(f"  Total loss: {losses['loss'].item():.4f}")
    print(f"  Pred loss: {losses['pred_loss'].item():.4f}")
    print(f"  SIGReg loss: {losses['sigreg_loss'].item():.4f}")
    print("  Loss computation OK")

    # Test backward pass
    losses['loss'].backward()
    print("  Backward pass OK")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")


def test_planning():
    """Test CEM planning."""
    print("Testing CEM planning...")
    model = LeWorldModel(
        img_size=64,
        patch_size=8,
        embed_dim=64,
        action_dim=2,
        encoder_layers=2,
        encoder_heads=2,
        predictor_layers=2,
        predictor_heads=4,
        history_len=3,
        sigreg_projections=256,
    )
    model.eval()

    obs_init = torch.randn(1, 3, 64, 64)
    obs_goal = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        action = model.plan(
            obs_init=obs_init,
            obs_goal=obs_goal,
            horizon=3,
            num_samples=50,
            num_elites=10,
            num_iterations=5,
        )

    assert action.shape == (2,), f"Expected (2,), got {action.shape}"
    print(f"  Planned action: {action.numpy()} OK")


def test_training_step():
    """Test a single training step."""
    print("Testing training step...")
    model = LeWorldModel(
        img_size=64,
        patch_size=8,
        embed_dim=64,
        action_dim=2,
        encoder_layers=2,
        encoder_heads=2,
        predictor_layers=2,
        predictor_heads=4,
        history_len=3,
        sigreg_projections=256,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    obs = torch.randn(8, 4, 3, 64, 64)
    act = torch.randn(8, 3, 2)

    # Training step
    model.train()
    optimizer.zero_grad()
    losses = model.compute_loss(obs, act)
    loss = losses['loss']
    loss.backward()
    optimizer.step()

    print(f"  Loss after step: {loss.item():.4f}")
    print("  Training step OK")


if __name__ == '__main__':
    print("=" * 60)
    print("LeWorldModel Implementation Tests")
    print("=" * 60)

    test_encoder()
    test_predictor()
    test_sigreg()
    test_full_model()
    test_planning()
    test_training_step()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
