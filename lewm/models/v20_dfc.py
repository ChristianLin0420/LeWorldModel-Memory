"""V20 DFC — the deployment-time slow trust filter (docs/V20_PROPOSAL.md 4.2).

The Dual-Filter Carrier is NOT a new training-time module: training is exactly
the V19 fixed-trust recipe (``lkc_rfix``, the only envelope-beating P3 arm).
At deployment, a second diagonal random-walk Kalman filter runs over the fast
filter's calibration parameters

    phi = (r_raw, q_raw, B)        theta (encoder, W_x, W_o, A) never moves

using the innovation NLL the fast filter computes anyway as its observation
likelihood:

    l_t(phi) = 1/2 sum_i [ eps_i^2 / S_i + log S_i ],
    eps_t = x_t - m-_t,  S_t = sigma-_t + softplus(r_raw)

Per step (streaming; episodes in bank order; phi, P, s persist across the
stream; the fast state resets per episode):

    P-_t  = P_{t-1} + rho                      (parameter random walk)
    eta_t = P-_t / (P-_t + s_t)                (derived per-parameter gain)
    phi_t = phi_{t-1} - eta_t (.) g_t          (g_t = truncated grad of l_t)
    P_t   = (1 - eta_t) P-_t

with s_t an EMA of g_t^2 (the standard diagonal curvature/observation-noise
proxy).  Gradients are the classical recursive-prediction-error truncation:
the one-step dependence only (the recursion history's dependence on phi is
ignored), which makes every step a handful of N-vector operations:

    dl/dS        = 1/2 (1/S - eps^2/S^2)
    g_r_raw      = dl/dS * sigmoid(r_raw)             (softplus' = sigmoid)
    g_q_raw      = dl/dS * sigmoid(q_raw)             (dS/dq = 1, truncated)
    g_B[i, j]    = -(eps_i / S_i) * a_prev_j          (eps = x - a.m - B a)

Registered limits (each an explicit configuration, gated in W1/W2):

    rho = 0, P_0 = 0     => eta_t == 0: phi frozen — EXACTLY the V19
                            fixed-trust arm (tests assert bitwise-close
                            prior_read equality against LatentKalmanCell).
    eta_fixed = c        => AdaJEPA-style constant step phi <- phi - c*g
                            (no P/s machinery) — the source-paper control.

The correction at step t uses the *pre-update* phi; the phi update applies
from the next step (standard filter ordering, registered).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from lewm.models.v19_carriers import (LatentKalmanCell, R_FLOOR, SIGMA_CEIL,
                                      SIGMA_FLOOR)

EMA_DECAY = 0.99          # s_t estimator decay — stated, not swept (ledger)
S_FLOOR = 1e-12
RHO_GRID = (1e-6, 1e-4, 1e-2)          # the ONE new knob, swept on dev only
ETA_FIXED_GRID = (1e-3, 1e-2, 1e-1)    # dfc_etafix control sweep (dev only)


@dataclass
class SlowFilterConfig:
    """Deployment-time slow-filter configuration (honest ledger, 4.2)."""

    rho: float = 0.0
    eta_fixed: float | None = None      # None => derived gain (the candidate)
    ema_decay: float = EMA_DECAY
    p_init: float = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "rho": self.rho,
            "eta_fixed": self.eta_fixed,
            "ema_decay": self.ema_decay,
            "p_init": self.p_init,
            "gradient": "truncated_one_step_rpe",
            "ordering": "correct_with_pre_update_phi_then_update",
            "streaming": "episodes_in_bank_order_phi_persists_state_resets",
        }


@dataclass
class DFCResult:
    """Streamed deployment outputs (numpy, fp32)."""

    prior_read: np.ndarray            # (E, L, D)
    telemetry: dict[str, np.ndarray]  # per-step (E, L) traces
    phi_trace: dict[str, np.ndarray]  # per-episode-end parameter summaries
    config: dict[str, Any] = field(default_factory=dict)


def _validate_carrier(carrier: LatentKalmanCell) -> None:
    if not isinstance(carrier, LatentKalmanCell):
        raise TypeError("DFC requires a LatentKalmanCell")
    if not carrier.r_fixed:
        raise ValueError("DFC adapts the fixed-trust variant only "
                         "(train arm lkc_rfix; Insight 9)")


@torch.no_grad()
def dfc_stream_eval(carrier: LatentKalmanCell, z: torch.Tensor,
                    actions: torch.Tensor, config: SlowFilterConfig
                    ) -> DFCResult:
    """Run the dual filter over a full evaluation bank, streaming.

    ``z`` is (E, L, D) frozen-encoder embeddings, ``actions`` (E, L-1, A).
    Everything runs in float32 on ``z``'s device with the LKC's registered
    floors/ceils mirrored exactly (the rho=0 subsumption test depends on it).
    """
    _validate_carrier(carrier)
    if z.dim() != 3 or actions.dim() != 3 or z.shape[0] != actions.shape[0]:
        raise ValueError("z must be (E, L, D) and actions (E, L-1, A)")
    device = z.device
    episodes, length, _ = z.shape
    n = carrier.state_dim

    a_diag = carrier.decay().to(device).float()                  # (N,)
    w_x = carrier.w_x.weight.detach().to(device).float()         # (N, D)
    w_o = carrier.w_o.weight.detach().to(device).float()         # (D, N)

    # phi: deployment copies of the trained calibration parameters.
    r_raw = carrier.r_const.detach().clone().to(device).float()  # (N,)
    q_raw = carrier.q_raw.detach().clone().to(device).float()    # (N,)
    b_mat = carrier.b.weight.detach().clone().to(device).float() # (N, A)
    phi0 = {"r_raw": r_raw.clone(), "q_raw": q_raw.clone(),
            "b": b_mat.clone()}

    derived = config.eta_fixed is None
    state_p = {name: torch.full_like(tensor, float(config.p_init))
               for name, tensor in (("r_raw", r_raw), ("q_raw", q_raw),
                                    ("b", b_mat))}
    state_s = {name: torch.zeros_like(tensor)
               for name, tensor in (("r_raw", r_raw), ("q_raw", q_raw),
                                    ("b", b_mat))}

    prior_read = torch.zeros(episodes, length, z.shape[-1], device=device)
    trace_keys = ("eta_mean", "p_mean", "nll", "k_mean", "innovation_norm",
                  "phi_drift", "calib_ratio")
    traces = {key: torch.zeros(episodes, length, device=device)
              for key in trace_keys}
    phi_end_drift = torch.zeros(episodes, device=device)

    z32 = z.float()
    a32 = actions.float()
    for episode in range(episodes):
        x = z32[episode] @ w_x.T                                 # (L, N)
        m = x[0]
        sigma = torch.full_like(m, carrier.sigma0)
        for t in range(1, length):
            r = F.softplus(r_raw).clamp_min(R_FLOOR)
            q = F.softplus(q_raw)
            a_prev = a32[episode, t - 1]                         # (A,)
            m_minus = a_diag * m + b_mat @ a_prev
            sigma_minus = (a_diag * a_diag * sigma + q).clamp(
                SIGMA_FLOOR, SIGMA_CEIL)
            total = sigma_minus + r
            innovation = x[t] - m_minus
            nll = 0.5 * (innovation.square() / total + total.log()).sum()

            # Correction with the PRE-update phi (registered ordering).
            k = sigma_minus / total
            m = m_minus + k * innovation
            sigma = ((1.0 - k) * sigma_minus).clamp(SIGMA_FLOOR, SIGMA_CEIL)
            prior_read[episode, t] = w_o @ m_minus

            # Truncated one-step gradients of l_t wrt phi.
            dl_ds = 0.5 * (1.0 / total - innovation.square() / total.square())
            grads = {
                "r_raw": dl_ds * torch.sigmoid(r_raw),
                "q_raw": dl_ds * torch.sigmoid(q_raw),
                "b": torch.outer(-(innovation / total), a_prev),
            }
            eta_values = []
            for name, tensor in (("r_raw", r_raw), ("q_raw", q_raw),
                                 ("b", b_mat)):
                grad = grads[name]
                state_s[name].mul_(config.ema_decay).add_(
                    grad.square(), alpha=1.0 - config.ema_decay)
                if derived:
                    p_minus = state_p[name] + config.rho
                    eta = p_minus / (p_minus + state_s[name] + S_FLOOR)
                    state_p[name] = (1.0 - eta) * p_minus
                else:
                    eta = torch.full_like(tensor, float(config.eta_fixed))
                tensor.sub_(eta * grad)
                eta_values.append(float(eta.mean()))

            traces["eta_mean"][episode, t] = float(np.mean(eta_values))
            traces["p_mean"][episode, t] = float(
                torch.cat([value.reshape(-1)
                           for value in state_p.values()]).mean())
            traces["nll"][episode, t] = float(nll)
            # Calibration-certificate coordinate (claim 3): mean squared
            # innovation z-score; 1.0 = perfectly calibrated trust.
            traces["calib_ratio"][episode, t] = float(
                (innovation.square() / total).mean())
            traces["k_mean"][episode, t] = float(k.mean())
            traces["innovation_norm"][episode, t] = float(
                innovation.norm())
            traces["phi_drift"][episode, t] = float(torch.sqrt(
                (r_raw - phi0["r_raw"]).square().sum()
                + (q_raw - phi0["q_raw"]).square().sum()
                + (b_mat - phi0["b"]).square().sum()))
        phi_end_drift[episode] = traces["phi_drift"][episode, length - 1]

    return DFCResult(
        prior_read=prior_read.cpu().numpy().astype(np.float32),
        telemetry={key: value.cpu().numpy().astype(np.float32)
                   for key, value in traces.items()},
        phi_trace={
            "episode_end_drift": phi_end_drift.cpu().numpy().astype(np.float32),
            "r_final": F.softplus(r_raw).cpu().numpy().astype(np.float32),
            "r_init": F.softplus(phi0["r_raw"]).cpu().numpy().astype(np.float32),
            "q_final": F.softplus(q_raw).cpu().numpy().astype(np.float32),
            "q_init": F.softplus(phi0["q_raw"]).cpu().numpy().astype(np.float32),
        },
        config=config.describe(),
    )
