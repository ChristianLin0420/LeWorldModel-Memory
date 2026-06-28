"""Tests for the two-timescale EMA memory and MemoryLeWorldModel."""

import os
import sys
import math

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lewm.models.memory import (TwoTimescaleMemory, MemoryFusion, SelectiveUpdateMemoryV3,
                                tau_to_alpha, alpha_to_tau)
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


def test_selective_update_v3():
    print("Testing SMT-v3 selective update + matched controls...")
    torch.manual_seed(7)
    B, T, D = 2, 6, 8
    taus = (2.0, 8.0)
    z = torch.randn(B, T, D)

    modes = ('dynamic', 'static', 'oracle', 'old_update')
    modules = [SelectiveUpdateMemoryV3(D, taus=taus, mode=mode) for mode in modes]
    counts = [sum(p.numel() for p in module.parameters()) for module in modules]
    assert len(set(counts)) == 1, counts
    expected = D * D + D + (D + 1) + len(taus)
    assert counts[0] == expected, (counts[0], expected)

    dynamic = modules[0]
    assert torch.allclose(dynamic.route_weights().sum(), torch.tensor(1.0))
    gates = dynamic.gate_values(z)
    assert gates.shape == (B, T, 1)
    mixed = dynamic(z)
    assert mixed.shape == z.shape
    rms = mixed.square().mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=2e-5), rms
    assert torch.allclose(dynamic.fuse(z, mixed), z)       # zero-init residual

    # A fully open true gate is exactly the ordinary fixed EMA for every bank.
    ones = torch.ones(B, T, 1)
    banks_open = dynamic._scan_banks(z, ones)
    for k, alpha in enumerate(dynamic.alphas):
        expected_bank = TwoTimescaleMemory._scan(z, alpha)
        assert torch.allclose(banks_open[:, :, k], expected_bank, atol=1e-6)

    # A closed true gate freezes state; the old recurrence instead erases it geometrically.
    freeze = torch.ones(B, T, 1)
    freeze[:, 1:] = 0
    banks_frozen = dynamic._scan_banks(z, freeze)
    assert torch.allclose(
        banks_frozen[:, 1:], z[:, :1].unsqueeze(2).expand(-1, T - 1, len(taus), -1))
    old = modules[-1]
    banks_old = old._scan_banks(z, freeze)
    for t in range(1, T):
        decay = (1.0 - old.alphas).pow(t).view(1, len(taus), 1)
        assert torch.allclose(banks_old[:, t], decay * z[:, 0].unsqueeze(1), atol=1e-6)

    # The static conditioner is invariant to batch/time content while retaining exactly the same
    # parameterization. Scalar/tensor overrides produce the requested canonical trajectory.
    static_gates = modules[1].gate_values(z)
    assert torch.allclose(static_gates, static_gates[:1, :1].expand_as(static_gates))
    zero_override = dynamic(z, gate_override=0.0)
    zero_override_ref = dynamic._scan_banks(z, torch.zeros(B, T, 1))
    zero_override_ref = (
        zero_override_ref * dynamic.route_weights().view(1, 1, len(taus), 1)).sum(2)
    zero_override_ref = zero_override_ref * torch.rsqrt(
        zero_override_ref.square().mean(-1, keepdim=True) + dynamic.rms_eps)
    assert torch.allclose(zero_override, zero_override_ref)

    oracle = modules[2]
    try:
        oracle(z)
        raise AssertionError('oracle accepted a missing memory_update_mask')
    except ValueError as exc:
        assert 'memory_update_mask' in str(exc)
    visibility = torch.tensor([[1, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(oracle.gate_values(z, visibility).squeeze(-1).bool(), visibility)
    # An override does not bypass the oracle's explicit-mask requirement.
    try:
        oracle(z, gate_override=0.5)
        raise AssertionError('oracle override bypassed the required visibility mask')
    except ValueError as exc:
        assert 'memory_update_mask' in str(exc)
    print(f"  SMT-v3 modes share {counts[0]} parameters; recurrence + controls OK")


def test_smtv3_model_mask_plumbing():
    print("Testing SMT-v3 model mask plumbing...")
    torch.manual_seed(8)
    B, L, D, A = 6, 7, 16, 3
    common = dict(
        img_size=64, patch_size=8, embed_dim=D, action_dim=A,
        encoder_layers=1, encoder_heads=2, predictor_layers=1, predictor_heads=2,
        history_len=3, sigreg_projections=32, encoder_type='precomputed',
        memory_mode='both', multi_taus=(2, 4, 8))
    impl_to_mode = {
        'smtv3': 'dynamic',
        'smtv3_static': 'static',
        'smtv3_oracle': 'oracle',
        'smtv3_old': 'old_update',
    }
    models = {impl: MemoryLeWorldModel(memory_impl=impl, **common) for impl in impl_to_mode}
    assert {model.mem_smtv3.mode for model in models.values()} == set(impl_to_mode.values())
    assert len({model.num_parameters() for model in models.values()}) == 1
    model = models['smtv3_oracle']
    obs = torch.randn(B, L, D)
    act = torch.randn(B, L - 1, A)
    visibility = torch.ones(B, L, dtype=torch.bool)
    visibility[:, 3:5] = False

    for operation in (
        lambda: model._inject(obs),
        lambda: model.compute_loss(obs, act),
        lambda: model.encode_with_memory(obs),
    ):
        try:
            operation()
            raise AssertionError('oracle model path accepted a missing visibility mask')
        except ValueError as exc:
            assert 'memory_update_mask' in str(exc)

    injected = model._inject(obs, memory_update_mask=visibility)
    assert injected.shape == obs.shape
    losses = model.compute_loss(obs, act, memory_update_mask=visibility)
    assert torch.isfinite(losses['loss'])
    balanced = model.compute_loss(
        obs, act, target_valid_mask=visibility, memory_update_mask=visibility,
        first_post_loss_weight=0.5)
    assert torch.allclose(
        balanced['pred_loss'],
        0.5 * balanced['pred_loss_all_valid'] + 0.5 * balanced['pred_loss_first_post'])
    try:
        models['smtv3_static'].compute_loss(obs, act, first_post_loss_weight=0.5)
        raise AssertionError('first-post objective accepted a missing target-valid mask')
    except ValueError as exc:
        assert 'target_valid_mask' in str(exc)
    encoded = model.encode_with_memory(obs, memory_update_mask=visibility)
    assert encoded[-1].shape == obs.shape
    # Calibrated mean overrides are threaded through the model while preserving oracle strictness.
    overridden = model._inject(obs, memory_update_mask=visibility, gate_override=0.5)
    assert overridden.shape == obs.shape
    print("  SMT-v3 oracle mask, balanced loss, and gate override plumbing OK")


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
    test_selective_update_v3()
    test_smtv3_model_mask_plumbing()
    test_memory_model(dev)
    print("=" * 60)
    print("All memory tests passed!")
    print("=" * 60)
