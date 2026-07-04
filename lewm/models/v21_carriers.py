"""V21 X0b baseline-parity carriers (docs/V21_PROPOSAL.md 4/X0.3).

Two additions to the V19 carrier registry, both answering panel objection I4
(baseline asymmetry / missing rivals):

- ``SlowGateGRU`` — the symmetric-repair control: the "trust must be slow"
  diagnosis applied to the BASELINE.  A standard action-conditioned GRUCell
  whose update-gate bias is chrono-initialized (Tallec & Ollivier 2018,
  GRU form: b_z = -log(tau)) over the SAME log-spaced timescale ladder the
  LKC uses (tau in [2, 96]), so z_k ~= 1/(1+tau_k) at init — if "slow" is a
  generic regularizer any recurrence benefits from, this arm will show it.

- ``GatedDeltaCell`` — a parameter-matched, action-conditioned member of the
  input-dependent-gain family the V19/V20 related work names as the LKC's
  modern rival (Gated DeltaNet, arXiv:2412.06464).  Matrix state
  S_t in R^{d x d}; per step with u_t = [z_t; a_{t-1}] (a_{-1} = 0):

      k_t = l2norm(W_k u_t)   q_t = l2norm(W_q u_t)   v_t = W_v u_t
      alpha_t = sigmoid(w_a u_t + b_a)      (scalar decay gate)
      beta_t  = sigmoid(w_b u_t + b_b)      (scalar write gate)
      S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
      z_tilde_t   = z_t + W_o (S_t q_t)          (W_o zero-init)
      prior_read_t = W_o (S_{t-1} q_bar)         (q_bar a learned static
                                                  query: pre-observation
                                                  readout uses NO frame-t
                                                  content, the registry
                                                  convention)

Both cells keep every V19 carrier convention: causal, float32 with autocast
off, zero-init read (each arm is exactly the no-carrier host at step 0),
``prior_read`` as the evaluation coordinate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.v19_carriers import (ActionConditionedGRU, Carrier,
                                      CarrierOutput, TAU_RANGE,
                                      lkc_parameter_count, make_carrier)

CHRONO_TAU_RANGE = TAU_RANGE          # the LKC ladder, reused by registration
ALPHA_BIAS_INIT = 3.0                 # sigmoid(3) ~= 0.95: slow decay at init
EPS = 1e-8

V21_CARRIER_NAMES = ("acgru_chrono", "gdelta")


class SlowGateGRU(ActionConditionedGRU):
    """Chrono-initialized ac-GRU: the symmetric-repair control."""

    name = "acgru_chrono"

    def __init__(self, embed_dim: int, action_dim: int,
                 hidden_dim: int | None = None) -> None:
        super().__init__(embed_dim, action_dim, hidden_dim)
        hidden = self.hidden_dim
        taus = np.geomspace(CHRONO_TAU_RANGE[0], CHRONO_TAU_RANGE[1], hidden)
        with torch.no_grad():
            # PyTorch GRUCell gate order: (reset, update, new).  The update
            # gate z has preactivation W_ih x + b_ih + W_hh h + b_hh over the
            # [hidden:2*hidden] chunk; chrono sets b_z = -log(tau) so
            # z ~= sigmoid(-log tau) = 1/(1+tau) at init.
            self.cell.bias_hh[hidden:2 * hidden] = torch.tensor(
                -np.log(taus), dtype=torch.float32)
            self.cell.bias_ih[hidden:2 * hidden] = 0.0
        self._chrono_taus = taus.tolist()

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update({
            "carrier": self.name,
            "chrono_init": "b_z = -log(tau), tau log-spaced",
            "tau_range": list(CHRONO_TAU_RANGE),
        })
        return base


class GatedDeltaCell(Carrier):
    """Minimal gated delta-rule linear-attention carrier (matrix state)."""

    name = "gdelta"

    def __init__(self, embed_dim: int, action_dim: int,
                 state_dim: int | None = None) -> None:
        super().__init__(embed_dim, action_dim)
        self.state_dim = (int(state_dim) if state_dim is not None
                          else matched_gdelta_dim(embed_dim, action_dim))
        d, inp = self.state_dim, embed_dim + action_dim
        self.w_k = nn.Linear(inp, d, bias=False)
        self.w_q = nn.Linear(inp, d, bias=False)
        self.w_v = nn.Linear(inp, d, bias=False)
        self.alpha_head = nn.Linear(inp, 1)
        self.beta_head = nn.Linear(inp, 1)
        self.q_bar = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.w_o = nn.Linear(d, embed_dim, bias=False)
        with torch.no_grad():
            self.alpha_head.weight.zero_()
            self.alpha_head.bias.fill_(ALPHA_BIAS_INIT)
            self.beta_head.weight.zero_()
            self.beta_head.bias.zero_()          # beta ~= 0.5 at init
            self.w_o.weight.zero_()

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> CarrierOutput:
        self._validate(z, actions)
        with torch.autocast(device_type=z.device.type, enabled=False):
            z32, a32 = z.float(), actions.float()
            batch, length, _ = z32.shape
            d = self.state_dim
            a_zero = z32.new_zeros(batch, self.action_dim)
            state = z32.new_zeros(batch, d, d)                # S_{t-1}
            q_bar = self.q_bar / self.q_bar.norm().clamp_min(EPS)
            reads, priors = [], []
            alpha_seq, beta_seq, norm_seq = [], [], []
            for t in range(length):
                a_prev = a32[:, t - 1] if t > 0 else a_zero
                u = torch.cat([z32[:, t], a_prev], dim=-1)
                k = F.normalize(self.w_k(u), dim=-1, eps=EPS)  # (B, d)
                q = F.normalize(self.w_q(u), dim=-1, eps=EPS)
                v = self.w_v(u)
                alpha = torch.sigmoid(self.alpha_head(u))      # (B, 1)
                beta = torch.sigmoid(self.beta_head(u))
                priors.append(state @ q_bar)                   # (B, d)
                # S <- alpha S (I - beta k k^T) + beta v k^T
                sk = torch.einsum("bij,bj->bi", state, k)      # S k, (B, d)
                state = (alpha.unsqueeze(-1) * state
                         - (alpha * beta).unsqueeze(-1)
                         * torch.einsum("bi,bj->bij", sk, k)
                         + beta.unsqueeze(-1)
                         * torch.einsum("bi,bj->bij", v, k))
                reads.append(torch.einsum("bij,bj->bi", state, q))
                alpha_seq.append(alpha.squeeze(-1))
                beta_seq.append(beta.squeeze(-1))
                norm_seq.append(state.flatten(1).norm(dim=-1))
            reads_all = torch.stack(reads, dim=1)              # (B, L, d)
            priors_all = torch.stack(priors, dim=1)
            z_tilde = z32 + self.w_o(reads_all)
            prior_read = self.w_o(priors_all)
            telemetry = {
                "state_norm": torch.stack(norm_seq, dim=1),
                "prior_state_norm": priors_all.norm(dim=-1),
                "alpha_mean": torch.stack(alpha_seq, dim=1),
                "beta_mean": torch.stack(beta_seq, dim=1),
            }
        return CarrierOutput(z_tilde=z_tilde.to(z.dtype),
                             prior_read=prior_read.to(z.dtype),
                             telemetry=telemetry)

    def describe(self) -> dict[str, Any]:
        return {
            "carrier": self.name,
            "embed_dim": self.embed_dim,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "rule": "gated_delta: S<-alpha S(I-beta kk^T)+beta vk^T",
            "alpha_bias_init": ALPHA_BIAS_INIT,
            "prior_convention": "W_o_S_t_minus_1_static_query",
            "parameters": self.parameter_count(),
        }


def gdelta_parameter_count(d: int, embed_dim: int, action_dim: int) -> int:
    """W_k/W_q/W_v (3*d*inp) + gates (2*(inp+1)) + q_bar (d) + W_o (D*d)."""
    inp = embed_dim + action_dim
    return 3 * d * inp + 2 * (inp + 1) + d + embed_dim * d


def matched_gdelta_dim(embed_dim: int, action_dim: int) -> int:
    """State dim closest in parameters to LKC-pure (the V18 matching rule)."""
    target = lkc_parameter_count(embed_dim, action_dim)
    candidates = range(2, 4 * embed_dim + 1)
    return min(candidates, key=lambda d: (
        abs(gdelta_parameter_count(d, embed_dim, action_dim) - target), d))


def make_carrier_v21(name: str, embed_dim: int, action_dim: int,
                     **kwargs) -> Carrier:
    """V19 registry plus the X0b arms.

    Sweep labels: an optional learning-rate suffix ``_l1|_l3|_l10`` (mapped
    to the lr by the launcher, stripped here) and ``acgru_hH`` for a width
    override — so every sweep cell gets its own run directory while the
    carrier construction stays canonical."""
    import re
    base = re.sub(r"_l\d+$", "", name)
    if base == "acgru_chrono":
        return SlowGateGRU(embed_dim, action_dim, kwargs.get("hidden_dim"))
    if base == "gdelta":
        return GatedDeltaCell(embed_dim, action_dim, kwargs.get("state_dim"))
    if base.startswith("acgru_h"):
        return ActionConditionedGRU(embed_dim, action_dim,
                                    hidden_dim=int(base.removeprefix("acgru_h")))
    return make_carrier(base, embed_dim, action_dim)
