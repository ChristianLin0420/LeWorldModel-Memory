"""V19 memory carriers: the Latent Kalman Cell and its references.

Implements docs/V19_PROPOSAL.md section 4.2 (carrier equations + honest
ledger) and the carrier arms of section 4.3.  A carrier transforms a
latent/action sequence *causally*:

    forward(z: (B, L, D), a: (B, L-1, A)) -> CarrierOutput

with ``z_tilde[:, t]`` a function of ``z[:, :t+1]`` and ``a[:, :t]`` only, and
``prior_read[:, t]`` the carrier's belief readout *before* seeing frame t
(``W_o m_minus_t`` for the LKC; ``W_o h_{t-1}`` for the recurrent references).
``prior_read`` carries no current-frame content by construction: it is the
evaluation coordinate for "what the carrier transported".

Registered conventions (resolved ambiguities, frozen here):

- t = 0 rows: there is no predict/correct step at t=0 (``m_0 = W_x z_0``,
  ``sigma_0 = sigma0``).  We register ``m_minus_0 := 0`` (hence
  ``prior_read_0 = 0``), gain telemetry ``k_0 := 1`` (the state fully adopts
  x_0; ``k_0 := 0`` for the k_zero arm), ``sigma_minus_0 := sigma0`` and
  ``innovation_0 := 0``.  Evaluation never reads t = 0.
- All carrier recursions run in float32 with autocast disabled: the
  uncertainty recursion and the innovation NLL are not bf16-safe.
- The LKC-NLL auxiliary loss reduces with a *mean* over steps, batch and
  channels (length-invariant unit weight); the trainer adds it at weight 1.
- The r-head weight is zero-initialized so r_t == softplus(bias) ~= 1.0 at
  step 0 and input dependence is learned, mirroring the zero-init discipline
  of B and W_o (the whole cell is exactly the no-carrier host at init).
- ``k_fixed``: the applied gain is ``sigmoid(c)`` (init 0.5); the sigma
  recursion is tracked with the *applied* gain for telemetry but never feeds
  back into the gain (it ignores sigma/r by definition of the intervention).
- ``a_twoscalar`` tiles [exp(-1/2), exp(-1/8)] alternately over all N
  channels (the V5/V9 coarse-decay control; no hold channel by design).
- AcSSM decay init follows the S4D-real spread: a_k = exp(-DELTA * (k + 1/2))
  with DELTA = 1/64 (the episode length), spanning sub-step to episode-scale
  timescales.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Registered constants (docs/V19_PROPOSAL.md 4.2 hand-choice ledger).
TAU_RANGE = (2.0, 96.0)      # half-life range of the fixed spectrum (log-spaced)
SIGMA_FLOOR = 1e-6
SIGMA_CEIL = 1e6
R_FLOOR = 1e-4
Q_INIT = 0.01
SIGMA0_INIT = 1.0
R_INIT = 1.0
K_FIXED_INIT = 0.5
TWO_SCALAR_DECAYS = (math.exp(-1.0 / 2.0), math.exp(-1.0 / 8.0))
SSM_DELTA = 1.0 / 64.0       # S4D-real discretization step (episode length 64)

CARRIER_NAMES = (
    "none", "acgru", "acssm",
    "lkc", "lkc_nll", "lkc_k0", "lkc_b0", "lkc_kfix", "lkc_rfix",
    "lkc_alearn", "lkc_a2",
)

_LKC_VARIANTS: dict[str, dict[str, bool]] = {
    "lkc": {},
    "lkc_nll": {"nll": True},
    "lkc_k0": {"k_zero": True},
    "lkc_b0": {"b_zero": True},
    "lkc_kfix": {"k_fixed": True},
    "lkc_rfix": {"r_fixed": True},
    "lkc_alearn": {"a_learned": True},
    "lkc_a2": {"a_twoscalar": True},
}


def _softplus_inverse(value: float) -> float:
    """x such that softplus(x) == value (value > 0)."""
    return math.log(math.expm1(value))


def _logit(value: np.ndarray) -> np.ndarray:
    return np.log(value) - np.log1p(-value)


def fixed_spectrum(state_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """(tau, a) of the registered fixed diagonal: N-1 log-spaced half-lives in
    TAU_RANGE (a_i = exp(-1/tau_i)) plus one exact eigenvalue-1 hold channel
    (tau = inf) at the last index."""
    if state_dim < 2:
        raise ValueError(f"state_dim must be >= 2, got {state_dim}")
    taus = np.geomspace(TAU_RANGE[0], TAU_RANGE[1], state_dim - 1)
    decays = np.exp(-1.0 / taus)
    return (np.concatenate([taus, [np.inf]]),
            np.concatenate([decays, [1.0]]).astype(np.float64))


@dataclass
class CarrierOutput:
    """Causal carrier outputs.

    Attributes:
        z_tilde: (B, L, D) residual read the predictor consumes.
        prior_read: (B, L, D) pre-observation belief readout at every t.
        telemetry: per-step diagnostics, each (B, L) or (B, L, N).
    """

    z_tilde: torch.Tensor
    prior_read: torch.Tensor
    telemetry: dict[str, torch.Tensor] = field(default_factory=dict)


class Carrier(nn.Module, abc.ABC):
    """Base class: causal (z, a) -> CarrierOutput transform."""

    name: str = "carrier"

    def __init__(self, embed_dim: int, action_dim: int) -> None:
        super().__init__()
        if embed_dim < 2 or action_dim < 1:
            raise ValueError(f"invalid carrier dims D={embed_dim}, A={action_dim}")
        self.embed_dim = int(embed_dim)
        self.action_dim = int(action_dim)

    def _validate(self, z: torch.Tensor, actions: torch.Tensor) -> None:
        if z.dim() != 3 or z.shape[-1] != self.embed_dim:
            raise ValueError(f"z must be (B, L, {self.embed_dim}), got {tuple(z.shape)}")
        expected = (z.shape[0], z.shape[1] - 1, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(
                f"actions must be {expected}, got {tuple(actions.shape)}")

    @abc.abstractmethod
    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> CarrierOutput:
        """Apply the carrier causally over the sequence."""

    def aux_loss(self) -> torch.Tensor | None:
        """Variant-specific auxiliary loss from the last forward (None if none)."""
        return None

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @abc.abstractmethod
    def describe(self) -> dict[str, Any]:
        """All frozen carrier hyperparameters (run-config registration)."""


class NoCarrier(Carrier):
    """Identity passthrough: uniform plumbing for the no-carrier reference."""

    name = "none"

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> CarrierOutput:
        self._validate(z, actions)
        return CarrierOutput(z_tilde=z, prior_read=torch.zeros_like(z), telemetry={})

    def describe(self) -> dict[str, Any]:
        return {"carrier": self.name, "embed_dim": self.embed_dim,
                "action_dim": self.action_dim, "parameters": 0}


class LatentKalmanCell(Carrier):
    """The V19 candidate: one latent Kalman cell (proposal 4.2, verbatim).

    State m_t, sigma_t in R^N with N = D; x_t = W_x z_t (H == I in the lifted
    basis).  Per step t = 1..L-1:

        predict: m_minus = A (.) m_{t-1} + B a_{t-1}
                 sigma_minus = A^2 (.) sigma_{t-1} + q
        correct: k = sigma_minus / (sigma_minus + r_t)
                 m_t = m_minus + k (.) (x_t - m_minus)
                 sigma_t = (1 - k) (.) sigma_minus
        read:    z_tilde_t = z_t + W_o m_t          (residual, W_o zero-init)
        prior:   prior_read_t = W_o m_minus_t       (pure carrier prior)

    Each constructor flag is exactly ONE registered intervention (4.3); at
    most one may be set.
    """

    name = "lkc"

    def __init__(self, embed_dim: int, action_dim: int, *,
                 nll: bool = False, k_zero: bool = False, b_zero: bool = False,
                 k_fixed: bool = False, r_fixed: bool = False,
                 a_learned: bool = False, a_twoscalar: bool = False) -> None:
        super().__init__(embed_dim, action_dim)
        flags = {"nll": nll, "k_zero": k_zero, "b_zero": b_zero,
                 "k_fixed": k_fixed, "r_fixed": r_fixed,
                 "a_learned": a_learned, "a_twoscalar": a_twoscalar}
        active = [name for name, value in flags.items() if value]
        if len(active) > 1:
            raise ValueError(
                f"LKC variants are single registered interventions; got {active}")
        self.variant = active[0] if active else "pure"
        self.nll, self.k_zero, self.b_zero = nll, k_zero, b_zero
        self.k_fixed, self.r_fixed = k_fixed, r_fixed
        self.a_learned, self.a_twoscalar = a_learned, a_twoscalar

        self.state_dim = self.embed_dim                       # N = D
        n = self.state_dim

        # A: transition spectrum (fixed buffer unless a_learned).
        if a_twoscalar:
            decays = np.tile(TWO_SCALAR_DECAYS,
                             (n + 1) // 2)[:n].astype(np.float64)
            self._spectrum_tau: list[float | None] = (
                -1.0 / np.log(decays)).tolist()
            self.hold_index: int | None = None
            self.register_buffer(
                "a_fixed", torch.tensor(decays, dtype=torch.float32))
            self.a_raw = None
        else:
            taus, decays = fixed_spectrum(n)
            # JSON-safe registration: the eigenvalue-1 hold channel has
            # tau = inf, recorded as None.
            self._spectrum_tau = [tau if math.isfinite(tau) else None
                                  for tau in taus.tolist()]
            self.hold_index = n - 1
            if a_learned:
                # sigmoid-reparam around the fixed spectrum; the eigenvalue-1
                # hold channel remains a fixed buffer (sigmoid cannot reach 1).
                self.a_raw = nn.Parameter(torch.tensor(
                    _logit(decays[:-1]), dtype=torch.float32))
                self.register_buffer(
                    "a_hold", torch.ones(1, dtype=torch.float32))
                self.a_fixed = None
            else:
                self.a_raw = None
                self.register_buffer(
                    "a_fixed", torch.tensor(decays, dtype=torch.float32))
        self._spectrum_a = decays.tolist()

        # W_x identity-init (square by N = D); B and W_o zero-init so the
        # model is exactly the no-carrier host at step 0.
        self.w_x = nn.Linear(self.embed_dim, n, bias=False)
        self.b = nn.Linear(self.action_dim, n, bias=False)
        self.w_o = nn.Linear(n, self.embed_dim, bias=False)
        with torch.no_grad():
            self.w_x.weight.copy_(torch.eye(n, self.embed_dim))
            self.b.weight.zero_()
            self.w_o.weight.zero_()
        if b_zero:
            self.b.weight.requires_grad_(False)

        self.q_raw = nn.Parameter(
            torch.full((n,), _softplus_inverse(Q_INIT)))
        self.sigma0 = SIGMA0_INIT

        if r_fixed:
            self.r_head = None
            self.r_const = nn.Parameter(
                torch.full((n,), _softplus_inverse(R_INIT)))
        else:
            self.r_head = nn.Linear(self.embed_dim, n)
            with torch.no_grad():
                self.r_head.weight.zero_()
                self.r_head.bias.fill_(_softplus_inverse(R_INIT))
            self.r_const = None

        if k_fixed:
            self.k_raw = nn.Parameter(
                torch.full((n,), _logit(np.array(K_FIXED_INIT)).item()))
        else:
            self.k_raw = None

        self._nll_value: torch.Tensor | None = None

    def decay(self) -> torch.Tensor:
        """Effective diagonal A, shape (N,)."""
        if self.a_learned:
            return torch.cat([torch.sigmoid(self.a_raw), self.a_hold])
        return self.a_fixed

    def observation_variance(self, z: torch.Tensor) -> torch.Tensor:
        """r_t per channel, (B, L, N), softplus with floor R_FLOOR."""
        if self.r_fixed:
            r = F.softplus(self.r_const).clamp_min(R_FLOOR)
            return r.expand(z.shape[0], z.shape[1], self.state_dim)
        return F.softplus(self.r_head(z)).clamp_min(R_FLOOR)

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> CarrierOutput:
        self._validate(z, actions)
        with torch.autocast(device_type=z.device.type, enabled=False):
            z32 = z.float()
            a32 = actions.float()
            batch, length, _ = z32.shape
            a_diag = self.decay()
            x = self.w_x(z32)                                   # (B, L, N)
            drive = self.b(a32)                                 # (B, L-1, N)
            r_all = self.observation_variance(z32)              # (B, L, N)
            q = F.softplus(self.q_raw)

            m = x[:, 0]
            sigma = torch.full_like(m, self.sigma0)
            zeros_n = torch.zeros_like(m)
            m_seq = [m]
            m_minus_seq = [zeros_n]
            k_seq = [zeros_n if self.k_zero else torch.ones_like(m)]
            sigma_minus_seq = [sigma]
            innovation_seq = [z32.new_zeros(batch)]
            nll_terms: list[torch.Tensor] = []

            for t in range(1, length):
                m_minus = a_diag * m + drive[:, t - 1]
                sigma_minus = (a_diag * a_diag * sigma + q).clamp(
                    SIGMA_FLOOR, SIGMA_CEIL)
                r_t = r_all[:, t]
                innovation = x[:, t] - m_minus
                if self.nll:
                    total = sigma_minus + r_t
                    nll_terms.append(
                        0.5 * (innovation.square() / total + total.log()))
                if self.k_zero:
                    k = zeros_n.expand_as(m_minus)
                    m = m_minus
                    sigma = sigma_minus
                else:
                    if self.k_fixed:
                        k = torch.sigmoid(self.k_raw).expand_as(m_minus)
                    else:
                        k = sigma_minus / (sigma_minus + r_t)
                    m = m_minus + k * innovation
                    sigma = ((1.0 - k) * sigma_minus).clamp(
                        SIGMA_FLOOR, SIGMA_CEIL)
                m_seq.append(m)
                m_minus_seq.append(m_minus)
                k_seq.append(k)
                sigma_minus_seq.append(sigma_minus)
                innovation_seq.append(innovation.norm(dim=-1))

            m_all = torch.stack(m_seq, dim=1)                   # (B, L, N)
            m_minus_all = torch.stack(m_minus_seq, dim=1)
            k_all = torch.stack(k_seq, dim=1)
            sigma_minus_all = torch.stack(sigma_minus_seq, dim=1)
            innovation_all = torch.stack(innovation_seq, dim=1)  # (B, L)

            z_tilde = z32 + self.w_o(m_all)
            prior_read = self.w_o(m_minus_all)
            self._nll_value = (torch.stack(nll_terms).mean()
                               if nll_terms else None)
            telemetry = {
                "k": k_all,
                "m_minus": m_minus_all,
                "k_mean": k_all.mean(dim=-1),
                "k_std": k_all.std(dim=-1),
                "sigma_minus_mean": sigma_minus_all.mean(dim=-1),
                "r_mean": r_all.mean(dim=-1),
                "innovation_norm": innovation_all,
            }
        return CarrierOutput(z_tilde=z_tilde.to(z.dtype),
                             prior_read=prior_read.to(z.dtype),
                             telemetry=telemetry)

    def aux_loss(self) -> torch.Tensor | None:
        if not self.nll:
            return None
        if self._nll_value is None:
            raise RuntimeError("LKC-NLL aux_loss() requires a forward pass first")
        return self._nll_value

    def describe(self) -> dict[str, Any]:
        return {
            "carrier": self.name,
            "variant": self.variant,
            "embed_dim": self.embed_dim,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "spectrum_tau": self._spectrum_tau,
            "spectrum_a": self._spectrum_a,
            "tau_range": list(TAU_RANGE),
            "hold_index": self.hold_index,
            "q_init": Q_INIT,
            "sigma0": self.sigma0,
            "r_init": R_INIT,
            "r_floor": R_FLOOR,
            "sigma_floors": [SIGMA_FLOOR, SIGMA_CEIL],
            "k_fixed_init": K_FIXED_INIT if self.k_fixed else None,
            "nll_reduction": ("mean_over_steps_batch_channels"
                              if self.nll else None),
            "parameters": self.parameter_count(),
        }


class ActionConditionedGRU(Carrier):
    """Ac-GRU reference: GRUCell over [z_t; a_{t-1}] (a_{-1} = 0).

    Convention (documented): for a GRU the pre-frame-t belief is h_{t-1}, so
    ``prior_read_t = W_o h_{t-1}`` (zero at t = 0).  Hidden width follows the
    V18 parameter-matching rule against LKC-pure.
    """

    name = "acgru"

    def __init__(self, embed_dim: int, action_dim: int,
                 hidden_dim: int | None = None) -> None:
        super().__init__(embed_dim, action_dim)
        self.hidden_dim = (int(hidden_dim) if hidden_dim is not None
                           else matched_gru_hidden(embed_dim, action_dim))
        self.cell = nn.GRUCell(embed_dim + action_dim, self.hidden_dim)
        self.w_o = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        with torch.no_grad():
            self.w_o.weight.zero_()

    def _scan(self, z32: torch.Tensor, a32: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = z32.shape
        h = z32.new_zeros(batch, self.hidden_dim)
        a_zero = z32.new_zeros(batch, self.action_dim)
        priors, states = [], []
        for t in range(length):
            a_prev = a32[:, t - 1] if t > 0 else a_zero
            priors.append(h)
            h = self.cell(torch.cat([z32[:, t], a_prev], dim=-1), h)
            states.append(h)
        return torch.stack(states, dim=1), torch.stack(priors, dim=1)

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> CarrierOutput:
        self._validate(z, actions)
        with torch.autocast(device_type=z.device.type, enabled=False):
            z32, a32 = z.float(), actions.float()
            states, priors = self._scan(z32, a32)
            z_tilde = z32 + self.w_o(states)
            prior_read = self.w_o(priors)
            telemetry = {
                "state_norm": states.norm(dim=-1),
                "prior_state_norm": priors.norm(dim=-1),
            }
        return CarrierOutput(z_tilde=z_tilde.to(z.dtype),
                             prior_read=prior_read.to(z.dtype),
                             telemetry=telemetry)

    def describe(self) -> dict[str, Any]:
        return {"carrier": self.name, "embed_dim": self.embed_dim,
                "action_dim": self.action_dim, "hidden_dim": self.hidden_dim,
                "prior_convention": "W_o_h_t_minus_1_before_frame_t",
                "parameters": self.parameter_count()}


class ActionConditionedSSM(Carrier):
    """Ac-SSM reference: diagonal linear recurrence over [z_t; a_{t-1}].

    h_t = a (.) h_{t-1} + W_in [z_t; a_{t-1}], a = sigmoid(a_raw) learned,
    initialized with the S4D-real spread a_k = exp(-SSM_DELTA * (k + 1/2)).
    Read and prior_read conventions match Ac-GRU; width parameter-matched.
    """

    name = "acssm"

    def __init__(self, embed_dim: int, action_dim: int,
                 width: int | None = None) -> None:
        super().__init__(embed_dim, action_dim)
        self.width = (int(width) if width is not None
                      else matched_ssm_width(embed_dim, action_dim))
        decays = np.exp(-SSM_DELTA * (np.arange(self.width) + 0.5))
        decays = np.clip(decays, 1e-4, 1.0 - 1e-4)
        self.a_raw = nn.Parameter(torch.tensor(_logit(decays),
                                               dtype=torch.float32))
        self.w_in = nn.Linear(embed_dim + action_dim, self.width)
        self.w_o = nn.Linear(self.width, embed_dim, bias=False)
        with torch.no_grad():
            self.w_o.weight.zero_()

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> CarrierOutput:
        self._validate(z, actions)
        with torch.autocast(device_type=z.device.type, enabled=False):
            z32, a32 = z.float(), actions.float()
            batch, length, _ = z32.shape
            decay = torch.sigmoid(self.a_raw)
            h = z32.new_zeros(batch, self.width)
            a_zero = z32.new_zeros(batch, self.action_dim)
            priors, states = [], []
            for t in range(length):
                a_prev = a32[:, t - 1] if t > 0 else a_zero
                priors.append(h)
                h = decay * h + self.w_in(
                    torch.cat([z32[:, t], a_prev], dim=-1))
                states.append(h)
            states_all = torch.stack(states, dim=1)
            priors_all = torch.stack(priors, dim=1)
            z_tilde = z32 + self.w_o(states_all)
            prior_read = self.w_o(priors_all)
            telemetry = {
                "state_norm": states_all.norm(dim=-1),
                "prior_state_norm": priors_all.norm(dim=-1),
            }
        return CarrierOutput(z_tilde=z_tilde.to(z.dtype),
                             prior_read=prior_read.to(z.dtype),
                             telemetry=telemetry)

    def describe(self) -> dict[str, Any]:
        decays = torch.sigmoid(self.a_raw.detach()).tolist()
        return {"carrier": self.name, "embed_dim": self.embed_dim,
                "action_dim": self.action_dim, "width": self.width,
                "ssm_delta": SSM_DELTA, "decay_init": decays,
                "prior_convention": "W_o_h_t_minus_1_before_frame_t",
                "parameters": self.parameter_count()}


# --------------------------------------------------------------------------
# Parameter matching (the V18 rule, mirroring train_hacssm_v10._matched_gru_hidden)
# --------------------------------------------------------------------------

def lkc_parameter_count(embed_dim: int, action_dim: int) -> int:
    """Analytic parameter count of LKC-pure with N = D.

    W_x (N*D) + B (N*A) + W_o (D*N) + q_raw (N) + r-head (D*N + N).
    A is a fixed buffer and sigma0 a constant: neither is a parameter.
    """
    n = embed_dim
    return (n * embed_dim + n * action_dim + embed_dim * n + n
            + embed_dim * n + n)


def acgru_parameter_count(hidden: int, embed_dim: int, action_dim: int) -> int:
    """GRUCell (3h(D+A) + 3h^2 + 6h) + W_o (D*h)."""
    inp = embed_dim + action_dim
    return 3 * hidden * inp + 3 * hidden * hidden + 6 * hidden + embed_dim * hidden


def acssm_parameter_count(width: int, embed_dim: int, action_dim: int) -> int:
    """W_in (n(D+A) + n) + a_raw (n) + W_o (D*n)."""
    inp = embed_dim + action_dim
    return width * inp + width + width + embed_dim * width


def _matched_width(count_fn, embed_dim: int, action_dim: int) -> int:
    target = lkc_parameter_count(embed_dim, action_dim)
    candidates = range(1, 4 * embed_dim + 1)
    return min(candidates, key=lambda width: (
        abs(count_fn(width, embed_dim, action_dim) - target), width))


def matched_gru_hidden(embed_dim: int, action_dim: int) -> int:
    """Ac-GRU hidden width closest in parameters to LKC-pure (V18 rule)."""
    return _matched_width(acgru_parameter_count, embed_dim, action_dim)


def matched_ssm_width(embed_dim: int, action_dim: int) -> int:
    """Ac-SSM width closest in parameters to LKC-pure (V18 rule)."""
    return _matched_width(acssm_parameter_count, embed_dim, action_dim)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

def make_carrier(name: str, embed_dim: int, action_dim: int) -> Carrier:
    """Instantiate a registered V19 carrier arm by name."""
    if name == "none":
        return NoCarrier(embed_dim, action_dim)
    if name == "acgru":
        return ActionConditionedGRU(embed_dim, action_dim)
    if name == "acssm":
        return ActionConditionedSSM(embed_dim, action_dim)
    if name in _LKC_VARIANTS:
        return LatentKalmanCell(embed_dim, action_dim, **_LKC_VARIANTS[name])
    raise KeyError(f"unknown carrier {name!r}; expected one of {CARRIER_NAMES}")
