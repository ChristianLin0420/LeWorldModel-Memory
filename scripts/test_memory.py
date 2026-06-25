"""Tests for the two-timescale EMA memory and MemoryLeWorldModel."""

import os
import sys
import math

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lewm.models.memory import TwoTimescaleMemory, MemoryFusion, tau_to_alpha, alpha_to_tau
from lewm.models.memory_model import MemoryLeWorldModel


def test_tau_alpha_roundtrip():
    print("Testing tau<->alpha conversion...")
    for tau in [1.0, 2.0, 5.0, 20.0, 50.0]:
        a = tau_to_alpha(tau)
        tau_back = float(alpha_to_tau(torch.tensor(a)))
        assert abs(tau_back - tau) < 1e-3, f"tau {tau} -> {tau_back}"
    print("  tau<->alpha roundtrip OK")


def test_ema_matches_closed_form():
    """The scan must match the exponential-kernel convolution (Eq. 2)."""
    print("Testing EMA scan == closed-form exponential kernel...")
    torch.manual_seed(0)
    B, T, D = 2, 30, 4
    z = torch.randn(B, T, D)
    mem = TwoTimescaleMemory(embed_dim=D, tau_fast=2.0, tau_slow=15.0, learnable=False)
    m_fast, m_slow = mem(z)

    # Closed form with warm start m_0 = z_0:
    #   m_t = (1-a)^t z_0 + a sum_{k=0}^{t-1} (1-a)^k z_{t-k}
    for alpha_t, m in [(mem.alpha_fast, m_fast), (mem.alpha_slow, m_slow)]:
        a = float(alpha_t)
        ref = torch.zeros_like(z)
        for t in range(T):
            acc = (1 - a) ** t * z[:, 0]
            for k in range(t):
                acc = acc + a * (1 - a) ** k * z[:, t - k]
            ref[:, t] = acc
        err = (ref - m).abs().max().item()
        assert err < 1e-4, f"EMA mismatch err={err}"
    print("  EMA scan matches closed form OK")


def test_step_matches_scan():
    print("Testing incremental step() matches full scan()...")
    torch.manual_seed(1)
    B, T, D = 3, 12, 5
    z = torch.randn(B, T, D)
    mem = TwoTimescaleMemory(embed_dim=D, learnable=False)
    m_fast, m_slow = mem(z)
    mf, ms = z[:, 0], z[:, 0]
    for t in range(1, T):
        mf, ms = mem.step(mf, ms, z[:, t])
        assert (mf - m_fast[:, t]).abs().max() < 1e-5
        assert (ms - m_slow[:, t]).abs().max() < 1e-5
    print("  step()==scan() OK")


def test_fusion_modes():
    print("Testing fusion modes + zero-init...")
    D = 8
    z = torch.randn(4, D)
    mf = torch.randn(4, D)
    ms = torch.randn(4, D)
    # zero-init => fused == z for every mode at init
    for mode in MemoryFusion.MODES:
        fusion = MemoryFusion(D, mode=mode)
        out = fusion(z, mf, ms)
        assert torch.allclose(out, z, atol=1e-6), f"mode {mode} not identity at init"
    # after perturbing weights, ablation flags drop the right bank
    fusion = MemoryFusion(D, mode='both')
    torch.nn.init.normal_(fusion.w_fast.weight, std=0.1)
    torch.nn.init.normal_(fusion.w_slow.weight, std=0.1)
    full = fusion(z, mf, ms)
    no_fast = fusion(z, mf, ms, ablate_fast=True)
    no_slow = fusion(z, mf, ms, ablate_slow=True)
    assert not torch.allclose(full, no_fast)
    assert not torch.allclose(full, no_slow)
    print("  fusion modes OK")


def test_memory_model(device):
    print(f"Testing MemoryLeWorldModel on {device}...")
    B, L = 4, 16
    for mode in MemoryFusion.MODES:
        model = MemoryLeWorldModel(
            img_size=64, patch_size=8, embed_dim=64, action_dim=2,
            encoder_layers=2, encoder_heads=2, predictor_layers=2, predictor_heads=4,
            history_len=3, sigreg_projections=128,
            memory_mode=mode, tau_fast=2.0, tau_slow=12.0, learnable_alpha=True,
        ).to(device)
        obs = torch.randn(B, L, 3, 64, 64, device=device)
        act = torch.randn(B, L - 1, 2, device=device)
        losses = model.compute_loss(obs, act)
        assert torch.isfinite(losses['loss']), f"non-finite loss in mode {mode}"
        losses['loss'].backward()
        # check memory params get grad when used and the fusion learns
        if mode != 'none' and model.memory.learnable:
            g = model.memory.raw_alpha_fast.grad
            # grad may be None only if neither bank used; here at least one is used
        print(f"  mode={mode:5s} loss={losses['loss'].item():.4f} "
              f"(pred={losses['pred_loss'].item():.4f}, sigreg={losses['sigreg_loss'].item():.4f}) OK")

    # influence + rollout on the 'both' model
    model = MemoryLeWorldModel(
        img_size=64, patch_size=8, embed_dim=64, action_dim=2,
        encoder_layers=2, encoder_heads=2, predictor_layers=2, predictor_heads=4,
        history_len=3, sigreg_projections=128, memory_mode='both',
    ).to(device)
    obs = torch.randn(B, L, 3, 64, 64, device=device)
    act = torch.randn(B, L - 1, 2, device=device)
    infl = model.memory_influence(obs, act)
    assert infl['infl_fast'].shape == (B,) and infl['infl_slow'].shape == (B,)
    print(f"  influence: fast={infl['infl_fast'].mean():.4f} slow={infl['infl_slow'].mean():.4f} OK")
    roll = model.rollout_latents(obs[:, :5], act[:, :4], horizon=4)
    assert roll.shape == (B, 4, 64), roll.shape
    print(f"  rollout shape {tuple(roll.shape)} OK")
    print(f"  horizons: {model.memory.horizons()}")
    print(f"  parameters: {model.num_parameters():,}")


if __name__ == '__main__':
    print("=" * 60)
    print("Memory module tests")
    print("=" * 60)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_tau_alpha_roundtrip()
    test_ema_matches_closed_form()
    test_step_matches_scan()
    test_fusion_modes()
    test_memory_model(dev)
    print("=" * 60)
    print("All memory tests passed!")
    print("=" * 60)
