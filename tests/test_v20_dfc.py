"""V20 DFC slow-filter tests (docs/V20_PROPOSAL.md 4.2).

The load-bearing property is subsumption: at rho = 0 with P_0 = 0 the dual
filter IS the V19 fixed-trust arm, bit-for-bit up to float associativity —
the W1 gate 'dfc must not lose to its own limit' presumes this identity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.v19_carriers import LatentKalmanCell
from lewm.models.v20_dfc import (RHO_GRID, DFCResult, SlowFilterConfig,
                                 dfc_stream_eval)

EMBED, ACTIONS, EPISODES, LENGTH = 8, 2, 3, 16


def _carrier(seed: int = 0) -> LatentKalmanCell:
    torch.manual_seed(seed)
    carrier = LatentKalmanCell(EMBED, ACTIONS, r_fixed=True)
    with torch.no_grad():
        # Move the calibration off its symmetric init so the identity test
        # exercises non-degenerate values, and give the read/action maps
        # real content (they are zero-initialized).
        carrier.r_const.add_(0.3 * torch.randn(EMBED))
        carrier.q_raw.add_(0.3 * torch.randn(EMBED))
        carrier.b.weight.copy_(0.2 * torch.randn(EMBED, ACTIONS))
        carrier.w_o.weight.copy_(0.2 * torch.randn(EMBED, EMBED))
    return carrier.eval()


def _bank(seed: int = 1, scale: float = 1.0
          ) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    z = scale * torch.randn(EPISODES, LENGTH, EMBED, generator=generator)
    actions = torch.randn(EPISODES, LENGTH - 1, ACTIONS, generator=generator)
    return z, actions


class TestSubsumption:
    def test_rho_zero_is_fixed_trust_exactly(self):
        carrier = _carrier()
        z, actions = _bank()
        result = dfc_stream_eval(carrier, z, actions,
                                 SlowFilterConfig(rho=0.0, p_init=0.0))
        with torch.no_grad():
            reference = carrier(z, actions).prior_read.numpy()
        np.testing.assert_allclose(result.prior_read, reference,
                                   rtol=1e-5, atol=1e-6)
        assert float(np.abs(result.telemetry["eta_mean"]).max()) == 0.0
        assert float(result.phi_trace["episode_end_drift"].max()) == 0.0

    def test_rho_zero_k_matches(self):
        carrier = _carrier()
        z, actions = _bank()
        result = dfc_stream_eval(carrier, z, actions, SlowFilterConfig())
        with torch.no_grad():
            reference = carrier(z, actions).telemetry["k_mean"].numpy()
        np.testing.assert_allclose(result.telemetry["k_mean"][:, 1:],
                                   reference[:, 1:], rtol=1e-5, atol=1e-6)


class TestAdaptation:
    def test_inflated_noise_raises_r(self):
        """Innovations systematically larger than the trained S must push the
        trust parameter r upward (the miscalibration -> slow-correction
        route; docs/V20_PROPOSAL.md 3, routing)."""
        carrier = _carrier()
        z, actions = _bank(seed=2, scale=6.0)      # inflated observation noise
        result = dfc_stream_eval(
            carrier, z, actions, SlowFilterConfig(rho=1e-2))
        assert float(result.telemetry["eta_mean"][:, 1:].mean()) > 0
        assert float(result.phi_trace["episode_end_drift"][-1]) > 0
        assert (result.phi_trace["r_final"].mean()
                > result.phi_trace["r_init"].mean())

    def test_drift_monotone_in_rho(self):
        """The walk rate rho is the adaptation-speed knob: on the same
        miscalibrated stream, larger rho => larger phi drift.  (Note the
        derived gain self-normalizes by s ~ E[g^2], so drift is NOT monotone
        in raw gradient magnitude — that is the natural-gradient property,
        registered in the ledger, and why this test sweeps rho instead.)"""
        carrier = _carrier()
        z, actions = _bank(seed=3, scale=6.0)
        drifts = [float(dfc_stream_eval(
            carrier, z, actions, SlowFilterConfig(rho=rho)
        ).phi_trace["episode_end_drift"][-1]) for rho in (1e-6, 1e-4, 1e-2)]
        assert drifts[0] < drifts[1] < drifts[2]

    def test_phi_persists_across_episodes(self):
        """Streaming semantics: drift is non-decreasing in magnitude at
        episode boundaries only if updates keep accumulating — check the
        final episode starts from the drifted phi, not from init."""
        carrier = _carrier()
        z, actions = _bank(seed=4, scale=6.0)
        result = dfc_stream_eval(carrier, z, actions,
                                 SlowFilterConfig(rho=1e-2))
        first_end = float(result.telemetry["phi_drift"][0, LENGTH - 1])
        last_start = float(result.telemetry["phi_drift"][-1, 1])
        assert last_start >= first_end * 0.5
        assert last_start > 0

    def test_eta_fixed_control(self):
        carrier = _carrier()
        z, actions = _bank(seed=5, scale=6.0)
        result = dfc_stream_eval(
            carrier, z, actions,
            SlowFilterConfig(rho=0.0, eta_fixed=1e-2))
        eta = result.telemetry["eta_mean"][:, 1:]
        np.testing.assert_allclose(eta, np.full_like(eta, 1e-2), rtol=1e-6)
        assert float(result.phi_trace["episode_end_drift"][-1]) > 0

    def test_derived_gain_anneals_without_walk(self):
        """rho = 0 with P_0 > 0: the gain must decay toward zero (the
        classical no-walk parameter filter forgets its prior uncertainty)."""
        carrier = _carrier()
        z, actions = _bank(seed=6, scale=2.0)
        result = dfc_stream_eval(
            carrier, z, actions, SlowFilterConfig(rho=0.0, p_init=1.0))
        eta = result.telemetry["eta_mean"]
        assert float(eta[0, 1]) > float(eta[-1, LENGTH - 1])


class TestValidation:
    def test_rejects_non_rfix_carrier(self):
        torch.manual_seed(0)
        carrier = LatentKalmanCell(EMBED, ACTIONS)   # learned trust head
        z, actions = _bank()
        with pytest.raises(ValueError):
            dfc_stream_eval(carrier, z, actions, SlowFilterConfig())

    def test_rejects_bad_shapes(self):
        carrier = _carrier()
        z, actions = _bank()
        with pytest.raises(ValueError):
            dfc_stream_eval(carrier, z[0], actions, SlowFilterConfig())

    def test_config_described_in_result(self):
        carrier = _carrier()
        z, actions = _bank()
        result = dfc_stream_eval(carrier, z, actions,
                                 SlowFilterConfig(rho=RHO_GRID[0]))
        assert isinstance(result, DFCResult)
        assert result.config["rho"] == RHO_GRID[0]
        assert result.config["gradient"] == "truncated_one_step_rpe"
