"""V21 X0b carrier tests (docs/V21_PROPOSAL.md 4/X0.3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models.v19_carriers import (ActionConditionedGRU,
                                      lkc_parameter_count)
from lewm.models.v21_carriers import (GatedDeltaCell, SlowGateGRU,
                                      gdelta_parameter_count,
                                      make_carrier_v21, matched_gdelta_dim)

EMBED, ACTIONS, BATCH, LENGTH = 16, 2, 3, 12


def _inputs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(BATCH, LENGTH, EMBED, generator=generator)
    actions = torch.randn(BATCH, LENGTH - 1, ACTIONS, generator=generator)
    return z, actions


class TestSlowGateGRU:
    def test_chrono_bias_ladder(self):
        cell = SlowGateGRU(EMBED, ACTIONS)
        hidden = cell.hidden_dim
        bias = cell.cell.bias_hh[hidden:2 * hidden].detach().numpy()
        taus = np.exp(-bias)
        assert taus[0] == pytest.approx(2.0, rel=1e-4)
        assert taus[-1] == pytest.approx(96.0, rel=1e-4)
        assert np.all(np.diff(taus) > 0)
        assert float(cell.cell.bias_ih[hidden:2 * hidden].abs().max()) == 0.0

    def test_zero_init_identity_and_causality(self):
        cell = SlowGateGRU(EMBED, ACTIONS)
        z, actions = _inputs()
        output = cell(z, actions)
        assert torch.allclose(output.z_tilde, z, atol=1e-6)   # W_o zero-init
        assert float(output.prior_read.abs().max()) == 0.0
        assert output.telemetry["state_norm"].shape == (BATCH, LENGTH)

    def test_parameter_count_matches_plain_gru(self):
        plain = ActionConditionedGRU(EMBED, ACTIONS)
        slow = SlowGateGRU(EMBED, ACTIONS)
        assert plain.parameter_count() == slow.parameter_count()

    def test_update_gates_slow_at_init(self):
        """The property chrono-init guarantees: at init the update gate z
        (state-overwrite rate) is far below the plain GRU's ~0.5 and spans
        the tau ladder — z_k ~= 1/(1+tau_k) in [1/97, 1/3]."""
        torch.manual_seed(0)
        plain = ActionConditionedGRU(EMBED, ACTIONS).eval()
        torch.manual_seed(0)
        slow = SlowGateGRU(EMBED, ACTIONS).eval()
        z, actions = _inputs(7)
        hidden = slow.hidden_dim

        def mean_update_gate(cell_module, module):
            gates = []
            h = torch.zeros(BATCH, module.hidden_dim)
            a_zero = torch.zeros(BATCH, ACTIONS)
            for t in range(LENGTH):
                a_prev = actions[:, t - 1] if t > 0 else a_zero
                u = torch.cat([z[:, t], a_prev], dim=-1)
                pre = (u @ cell_module.weight_ih.T + cell_module.bias_ih
                       + h @ cell_module.weight_hh.T + cell_module.bias_hh)
                gates.append(torch.sigmoid(
                    pre[:, hidden:2 * hidden]).mean().item())
                h = cell_module(u, h)
            return float(np.mean(gates))

        with torch.no_grad():
            z_slow = mean_update_gate(slow.cell, slow)
            z_plain = mean_update_gate(plain.cell, plain)
        assert z_slow < 0.25
        assert z_plain > 0.35
        assert z_slow < z_plain / 2


class TestGatedDeltaCell:
    def test_zero_init_identity_and_shapes(self):
        cell = GatedDeltaCell(EMBED, ACTIONS)
        z, actions = _inputs(1)
        output = cell(z, actions)
        assert torch.allclose(output.z_tilde, z, atol=1e-6)
        assert float(output.prior_read.abs().max()) == 0.0
        assert output.telemetry["alpha_mean"].shape == (BATCH, LENGTH)
        assert float(output.telemetry["alpha_mean"].mean()) > 0.9  # slow init

    def test_prior_read_is_pre_observation(self):
        """prior_read at t must not depend on z_t: perturb frame t and check
        prior_read[:, t] is unchanged (it reads S_{t-1} with a static query)."""
        cell = GatedDeltaCell(EMBED, ACTIONS)
        with torch.no_grad():
            cell.w_o.weight.copy_(0.3 * torch.randn(EMBED, cell.state_dim))
        z, actions = _inputs(2)
        t_probe = 6
        with torch.no_grad():
            base = cell(z, actions).prior_read[:, t_probe].clone()
            z2 = z.clone()
            z2[:, t_probe] += 5.0
            perturbed = cell(z2, actions).prior_read[:, t_probe]
        assert torch.allclose(base, perturbed, atol=1e-6)

    def test_delta_rule_writes_and_decays(self):
        cell = GatedDeltaCell(EMBED, ACTIONS)
        z, actions = _inputs(3)
        with torch.no_grad():
            norms = cell(z, actions).telemetry["state_norm"]
        assert float(norms[:, 1:].min()) > 0          # state gets written
        assert torch.isfinite(norms).all()

    def test_parameter_matching(self):
        d = matched_gdelta_dim(EMBED, ACTIONS)
        target = lkc_parameter_count(EMBED, ACTIONS)
        achieved = gdelta_parameter_count(d, EMBED, ACTIONS)
        assert abs(achieved - target) <= target * 0.1
        cell = GatedDeltaCell(EMBED, ACTIONS)
        assert cell.parameter_count() == gdelta_parameter_count(
            cell.state_dim, EMBED, ACTIONS)

    def test_gradients_flow(self):
        cell = GatedDeltaCell(EMBED, ACTIONS)
        with torch.no_grad():
            cell.w_o.weight.copy_(0.1 * torch.randn(EMBED, cell.state_dim))
        z, actions = _inputs(4)
        output = cell(z, actions)
        (output.z_tilde.square().mean()
         + output.prior_read.square().mean()).backward()
        for name, parameter in cell.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


class TestRegistry:
    def test_v21_names(self):
        assert isinstance(make_carrier_v21("acgru_chrono", EMBED, ACTIONS),
                          SlowGateGRU)
        assert isinstance(make_carrier_v21("gdelta", EMBED, ACTIONS),
                          GatedDeltaCell)
        swept = make_carrier_v21("acgru_h24", EMBED, ACTIONS)
        assert isinstance(swept, ActionConditionedGRU)
        assert swept.hidden_dim == 24

    def test_v19_names_pass_through(self):
        carrier = make_carrier_v21("lkc_rfix", EMBED, ACTIONS)
        assert carrier.name == "lkc"
        assert carrier.r_fixed
        with pytest.raises(KeyError):
            make_carrier_v21("unknown_arm", EMBED, ACTIONS)
