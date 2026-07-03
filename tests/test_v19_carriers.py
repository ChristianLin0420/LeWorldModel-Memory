"""Fast CPU tests for the V19 carriers (lewm/models/v19_carriers.py).

Covers the invariants the P2/P3 evaluation relies on: strict causality of
every carrier, exact zero-init inertness (each arm is its host at step 0),
the k=0 arm as pure transport, integrator representability through the
eigenvalue-1 hold channel, single-intervention flag exclusivity, NLL
finiteness and gradient flow into the observation-variance head, the V18
parameter-matching rule, and telemetry shape/typing contracts.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.v19_carriers import (CARRIER_NAMES, TWO_SCALAR_DECAYS,
                                      LatentKalmanCell, acgru_parameter_count,
                                      acssm_parameter_count, fixed_spectrum,
                                      lkc_parameter_count, make_carrier,
                                      matched_gru_hidden, matched_ssm_width)

D, A, B, L = 12, 2, 3, 16


def _data(seed: int = 0, batch: int = B, length: int = L, dim: int = D,
          action_dim: int = A) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(batch, length, dim, generator=generator)
    actions = torch.randn(batch, length - 1, action_dim, generator=generator)
    return z, actions


def _randomize(carrier: torch.nn.Module, seed: int = 1) -> None:
    """Make the read/transport non-trivial so causality tests have teeth."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in carrier.parameters():
            parameter.copy_(0.3 * torch.randn(parameter.shape,
                                              generator=generator))


# --------------------------------------------------------------------------
# Registry, shapes, telemetry contracts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", CARRIER_NAMES)
def test_registry_shapes_and_telemetry(name):
    torch.manual_seed(0)
    carrier = make_carrier(name, D, A)
    z, actions = _data()
    output = carrier(z, actions)
    assert output.z_tilde.shape == (B, L, D)
    assert output.prior_read.shape == (B, L, D)
    for key, value in output.telemetry.items():
        assert value.shape[:2] == (B, L), key
        assert value.dim() in (2, 3), key
        if value.dim() == 3:
            assert value.shape[2] == carrier.state_dim, key
    description = carrier.describe()
    assert description["carrier"] in {"none", "acgru", "acssm", "lkc"}
    if name.startswith("lkc"):
        assert set(output.telemetry) == {
            "k", "m_minus", "k_mean", "k_std", "sigma_minus_mean", "r_mean",
            "innovation_norm"}
        assert len(description["spectrum_a"]) == carrier.state_dim


def test_unknown_carrier_and_bad_shapes():
    with pytest.raises(KeyError):
        make_carrier("lkc_bogus", D, A)
    carrier = make_carrier("lkc", D, A)
    z, actions = _data()
    with pytest.raises(ValueError):
        carrier(z, actions[:, :-1])          # wrong action length
    with pytest.raises(ValueError):
        carrier(z[..., :-1], actions)        # wrong embed dim


# --------------------------------------------------------------------------
# Zero-init inertness (each arm == host at step 0)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", CARRIER_NAMES)
def test_zero_init_inertness(name):
    torch.manual_seed(0)
    carrier = make_carrier(name, D, A)
    z, actions = _data()
    output = carrier(z, actions)
    assert torch.equal(output.z_tilde, z)
    assert torch.equal(output.prior_read, torch.zeros_like(z))


# --------------------------------------------------------------------------
# Causality: z_tilde_t and prior_read_t invariant to future z/a
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["lkc", "lkc_nll", "lkc_k0", "lkc_kfix",
                                  "acgru", "acssm"])
def test_causality(name):
    torch.manual_seed(0)
    carrier = make_carrier(name, D, A)
    _randomize(carrier)
    z, actions = _data(seed=2)
    cut = 7
    reference = carrier(z, actions)
    z_future = z.clone()
    a_future = actions.clone()
    z_future[:, cut + 1:] += torch.randn_like(z_future[:, cut + 1:])
    a_future[:, cut:] += torch.randn_like(a_future[:, cut:])
    perturbed = carrier(z_future, a_future)
    # z_tilde_t depends only on z_{<=t}, a_{<t}; prior_read_t on z_{<t}, a_{<t}.
    assert torch.allclose(reference.z_tilde[:, :cut + 1],
                          perturbed.z_tilde[:, :cut + 1], atol=1e-6)
    assert torch.allclose(reference.prior_read[:, :cut + 1],
                          perturbed.prior_read[:, :cut + 1], atol=1e-6)
    # ... and the future genuinely changed (the test has teeth).
    assert not torch.allclose(reference.z_tilde[:, cut + 1:],
                              perturbed.z_tilde[:, cut + 1:], atol=1e-6)


# --------------------------------------------------------------------------
# Variant flags: exactly one registered intervention
# --------------------------------------------------------------------------

def test_variant_flag_mutual_exclusion():
    with pytest.raises(ValueError):
        LatentKalmanCell(D, A, nll=True, k_zero=True)
    with pytest.raises(ValueError):
        LatentKalmanCell(D, A, a_learned=True, a_twoscalar=True)
    with pytest.raises(ValueError):
        LatentKalmanCell(D, A, k_fixed=True, r_fixed=True)
    assert LatentKalmanCell(D, A).variant == "pure"
    assert LatentKalmanCell(D, A, b_zero=True).variant == "b_zero"


# --------------------------------------------------------------------------
# k = 0: pure transport
# --------------------------------------------------------------------------

def test_k_zero_is_pure_transport():
    torch.manual_seed(0)
    carrier = make_carrier("lkc_k0", D, A)
    _randomize(carrier)
    z, actions = _data(seed=3)
    output = carrier(z, actions)
    # Correction skipped entirely: gain telemetry identically zero and
    # m_t == m_minus_t, hence z_tilde_t - z_t == prior_read_t for t >= 1.
    assert torch.equal(output.telemetry["k"],
                       torch.zeros_like(output.telemetry["k"]))
    assert torch.allclose(output.z_tilde[:, 1:] - z[:, 1:],
                          output.prior_read[:, 1:], atol=1e-6)


def test_integrator_representability_via_hold_channel():
    """Open-loop cell with the hold channel represents cumulative action sums."""
    carrier = make_carrier("lkc_k0", D, A)
    hold = carrier.hold_index
    assert hold == carrier.state_dim - 1
    assert float(carrier.decay()[hold]) == 1.0
    with torch.no_grad():
        carrier.w_x.weight.zero_()                    # no observation lift
        carrier.b.weight.zero_()
        carrier.b.weight[hold, 0] = 1.0               # accumulate a[..., 0]
        carrier.w_o.weight.zero_()
        carrier.w_o.weight[0, hold] = 1.0             # read it on output dim 0
    z, actions = _data(seed=4)
    output = carrier(z, actions)
    for t in range(1, L):
        expected = actions[:, :t, 0].sum(dim=1)
        assert torch.allclose(output.telemetry["m_minus"][:, t, hold],
                              expected, atol=1e-5)
        assert torch.allclose(output.prior_read[:, t, 0], expected, atol=1e-5)
        assert torch.allclose(output.z_tilde[:, t, 0], z[:, t, 0] + expected,
                              atol=1e-5)


# --------------------------------------------------------------------------
# Intervention semantics
# --------------------------------------------------------------------------

def test_b_zero_freezes_action_transport():
    carrier = make_carrier("lkc_b0", D, A)
    assert not carrier.b.weight.requires_grad
    assert float(carrier.b.weight.abs().max()) == 0.0
    z, actions = _data(seed=5)
    reference = carrier(z, actions)
    permuted = carrier(z, actions[:, torch.randperm(L - 1)])
    # With B == 0 the carrier ignores actions entirely.
    assert torch.allclose(reference.z_tilde, permuted.z_tilde, atol=1e-6)


def test_k_fixed_gain_ignores_sigma_and_r():
    torch.manual_seed(0)
    carrier = make_carrier("lkc_kfix", D, A)
    z, actions = _data(seed=6)
    output = carrier(z, actions)
    k = output.telemetry["k"][:, 1:]
    expected = torch.sigmoid(carrier.k_raw.detach()).expand_as(k)
    assert torch.allclose(k, expected, atol=1e-6)
    assert float(expected.mean()) == pytest.approx(0.5)  # init sigmoid(0)


def test_r_fixed_has_constant_trust_and_no_head():
    carrier = make_carrier("lkc_rfix", D, A)
    assert carrier.r_head is None
    z, actions = _data(seed=7)
    output = carrier(z, actions)
    r_mean = output.telemetry["r_mean"]
    assert torch.allclose(r_mean, torch.full_like(r_mean, 1.0), atol=1e-5)


# --------------------------------------------------------------------------
# Spectrum registration
# --------------------------------------------------------------------------

def test_fixed_spectrum_and_hold_channel():
    taus, decays = fixed_spectrum(D)
    assert decays.shape == (D,)
    assert decays[-1] == 1.0 and math.isinf(taus[-1])
    assert np.allclose(decays[:-1], np.exp(-1.0 / taus[:-1]))
    assert np.all(np.diff(taus[:-1]) > 0)             # log-spaced, increasing
    assert taus[0] == pytest.approx(2.0) and taus[-2] == pytest.approx(96.0)
    carrier = make_carrier("lkc", D, A)
    assert np.allclose(carrier.decay().detach().numpy(), decays)


def test_a_learned_initializes_at_fixed_spectrum():
    carrier = make_carrier("lkc_alearn", D, A)
    _, decays = fixed_spectrum(D)
    assert carrier.a_raw.shape == (D - 1,)
    assert np.allclose(carrier.decay().detach().numpy(), decays, atol=1e-6)
    assert float(carrier.decay().detach()[carrier.hold_index]) == 1.0
    z, actions = _data(seed=8)
    with torch.no_grad():
        carrier.w_o.weight.normal_(std=0.1)
    carrier(z, actions).z_tilde.sum().backward()
    assert carrier.a_raw.grad is not None             # A really is learned
    assert not carrier.a_hold.requires_grad           # hold stays fixed


def test_a_twoscalar_tiles_coarse_decays_without_hold():
    carrier = make_carrier("lkc_a2", D, A)
    decays = carrier.decay().detach().numpy()
    assert carrier.hold_index is None
    expected = np.tile(TWO_SCALAR_DECAYS, (D + 1) // 2)[:D]
    assert np.allclose(decays, expected, atol=1e-7)
    assert not np.any(decays == 1.0)                  # no hold channel
    assert not any(parameter.requires_grad is True and name == "a_raw"
                   for name, parameter in carrier.named_parameters())


# --------------------------------------------------------------------------
# NLL variant
# --------------------------------------------------------------------------

def test_nll_finiteness_and_gradient_flow_to_r_head():
    torch.manual_seed(0)
    carrier = make_carrier("lkc_nll", D, A)
    with pytest.raises(RuntimeError):
        carrier.aux_loss()                             # requires a forward
    z, actions = _data(seed=9)
    carrier(z, actions)
    aux = carrier.aux_loss()
    assert aux is not None and aux.dim() == 0
    assert math.isfinite(float(aux.detach()))
    aux.backward()
    assert carrier.r_head.weight.grad is not None
    assert float(carrier.r_head.weight.grad.abs().sum()) > 0.0
    assert float(carrier.r_head.bias.grad.abs().sum()) > 0.0
    assert carrier.q_raw.grad is not None              # sigma path is live


def test_aux_loss_none_for_non_nll_variants():
    for name in ("lkc", "lkc_k0", "acgru", "acssm", "none"):
        carrier = make_carrier(name, D, A)
        z, actions = _data(seed=10)
        carrier(z, actions)
        assert carrier.aux_loss() is None


# --------------------------------------------------------------------------
# Parameter matching (the V18 rule)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("embed_dim,action_dim", [(128, 2), (192, 2), (12, 3)])
def test_parameter_matching_within_5_percent(embed_dim, action_dim):
    lkc = make_carrier("lkc", embed_dim, action_dim)
    target = lkc.parameter_count()
    assert target == lkc_parameter_count(embed_dim, action_dim)
    gru = make_carrier("acgru", embed_dim, action_dim)
    ssm = make_carrier("acssm", embed_dim, action_dim)
    assert gru.parameter_count() == acgru_parameter_count(
        gru.hidden_dim, embed_dim, action_dim)
    assert ssm.parameter_count() == acssm_parameter_count(
        ssm.width, embed_dim, action_dim)
    assert gru.hidden_dim == matched_gru_hidden(embed_dim, action_dim)
    assert ssm.width == matched_ssm_width(embed_dim, action_dim)
    for reference in (gru, ssm):
        mismatch = abs(reference.parameter_count() - target) / target
        assert mismatch <= 0.05, (reference.name, mismatch)


# --------------------------------------------------------------------------
# Telemetry t=0 convention
# --------------------------------------------------------------------------

def test_telemetry_t0_rows():
    torch.manual_seed(0)
    carrier = make_carrier("lkc", D, A)
    z, actions = _data(seed=11)
    telemetry = carrier(z, actions).telemetry
    assert torch.equal(telemetry["m_minus"][:, 0],
                       torch.zeros(B, carrier.state_dim))
    assert torch.equal(telemetry["k"][:, 0], torch.ones(B, carrier.state_dim))
    assert torch.equal(telemetry["innovation_norm"][:, 0], torch.zeros(B))
    assert torch.allclose(telemetry["sigma_minus_mean"][:, 0],
                          torch.full((B,), carrier.sigma0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
