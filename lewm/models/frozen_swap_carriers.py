"""Publication carriers for the frozen-host swap experiment.

This module is deliberately separate from :mod:`lewm.models.v19_carriers`.
It reuses that module's causal ``CarrierOutput`` contract without changing the
registered V19 implementations.  Every observed sequence follows

    forward(z: (B, L, D), actions: (B, L - 1, A)) -> CarrierOutput,

where ``prior_read[:, t]`` is computed before consuming ``z[:, t]`` and the
read projection is zero-initialized.  Consequently every arm is exactly the
frozen no-carrier host at initialization.

The publication comparison targets the parameter count of the fixed-trust
LKC (``r_t`` is a learned channel-wise constant rather than an input-dependent
head).  GRU, LSTM and diagonal-SSM widths are selected by an integer search
for the globally closest count.  For the official LeWM dimensions D=192 and
A=10 the registered counts are::

    fixed-trust LKC  76,032 parameters
    Ac-GRU (h=74)    75,924 parameters  (-108)
    Ac-LSTM (h=61)   76,372 parameters  (+340)
    diagonal SSM     76,032 parameters  (width=192, exact)

Streaming/rollout rule
----------------------
``initialize(z0)`` consumes the first observation.  ``observe(state, z_t,
a_prev)`` first produces the pre-observation prior, then consumes ``z_t``.
``imagine(state, a_t)`` performs one transition without an observation:

* fixed-trust LKC: the registered Kalman predict step, with no correction;
* GRU/LSTM/SSM: one recurrent update with an all-zero latent token and
  ``a_t``.  This is an explicit action-only extrapolation rule, not an
  assertion that a recurrent baseline has a separately identified dynamics
  model;
* no-carrier: unchanged empty state and a zero read.

An imagined step returns no fused observation (``fused_z is None``).  Its
``prior_read`` is the residual carrier read available to a learned world-model
rollout.  States can be cloned or repeat-interleaved across action candidates
without aliasing the source tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.v19_carriers import (
    Carrier,
    CarrierOutput,
    LatentKalmanCell,
    NoCarrier,
    R_FLOOR,
    SIGMA_CEIL,
    SIGMA_FLOOR,
    SSM_DELTA,
)

OFFICIAL_LEWM_EMBED_DIM = 192
OFFICIAL_LEWM_ACTION_DIM = 10

FROZEN_SWAP_CARRIER_NAMES = (
    "none",
    "acgru",
    "aclstm",
    "diag_ssm",
    "lkc_fixed_trust",
)

# Short experiment labels consumed by the frozen-host trainer.
FROZEN_CARRIER_NAMES = ("none", "gru", "lstm", "ssm", "fixed_trust")


@dataclass(frozen=True)
class EmptyCarrierState:
    """Batch/device/dtype anchor for the no-carrier streaming arm."""

    anchor: torch.Tensor  # (B, 0)


@dataclass(frozen=True)
class RecurrentCarrierState:
    """Vector state used by the GRU and diagonal SSM."""

    hidden: torch.Tensor


@dataclass(frozen=True)
class LSTMCarrierState:
    """Hidden and cell state of the action-conditioned LSTM."""

    hidden: torch.Tensor
    cell: torch.Tensor


@dataclass(frozen=True)
class KalmanCarrierState:
    """Posterior (or predict-only imagined) LKC mean and variance."""

    mean: torch.Tensor
    variance: torch.Tensor


StreamState = (EmptyCarrierState | RecurrentCarrierState | LSTMCarrierState
               | KalmanCarrierState)
StateT = TypeVar("StateT", bound=StreamState)


@dataclass(frozen=True)
class StreamingCarrierOutput:
    """One streaming carrier step.

    ``fused_z`` is present for ``initialize`` and ``observe`` and absent for
    ``imagine``.  ``prior_read`` always has shape ``(B, D)``.
    """

    state: StreamState
    fused_z: torch.Tensor | None
    prior_read: torch.Tensor


def _state_tensors(state: StreamState) -> tuple[torch.Tensor, ...]:
    return tuple(getattr(state, item.name) for item in fields(state))


def clone_stream_state(state: StateT) -> StateT:
    """Clone every tensor in ``state`` without retaining storage aliases."""

    return type(state)(*(value.clone() for value in _state_tensors(state)))


def repeat_stream_state(state: StateT, repeats: int) -> StateT:
    """Repeat-interleave a batched state for ``repeats`` action candidates.

    For a source batch ``[s0, s1]`` and ``repeats=3``, the result ordering is
    ``[s0, s0, s0, s1, s1, s1]``.  Returned tensors own new storage.
    """

    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError(f"repeats must be a positive integer, got {repeats!r}")
    return type(state)(*(value.repeat_interleave(repeats, dim=0)
                         for value in _state_tensors(state)))


class StreamingCarrierMixin:
    """Shared validation and state-copy helpers for publication carriers."""

    embed_dim: int
    action_dim: int

    def _validate_z_step(self, z: torch.Tensor, name: str = "z") -> None:
        if z.dim() != 2 or z.shape[-1] != self.embed_dim:
            raise ValueError(
                f"{name} must be (B, {self.embed_dim}), got {tuple(z.shape)}")

    def _validate_action_step(
            self, action: torch.Tensor, batch: int,
            name: str = "action") -> None:
        expected = (batch, self.action_dim)
        if tuple(action.shape) != expected:
            raise ValueError(f"{name} must be {expected}, got {tuple(action.shape)}")

    @staticmethod
    def clone_state(state: StateT) -> StateT:
        return clone_stream_state(state)

    @staticmethod
    def repeat_state(state: StateT, repeats: int) -> StateT:
        return repeat_stream_state(state, repeats)


class FrozenNoCarrier(StreamingCarrierMixin, NoCarrier):
    """No-carrier reference with uniform streaming plumbing."""

    def initialize(self, z0: torch.Tensor) -> StreamingCarrierOutput:
        self._validate_z_step(z0, "z0")
        state = EmptyCarrierState(z0.new_empty(z0.shape[0], 0))
        return StreamingCarrierOutput(state, z0, torch.zeros_like(z0))

    def observe(self, state: EmptyCarrierState, z_t: torch.Tensor,
                a_prev: torch.Tensor) -> StreamingCarrierOutput:
        self._validate_z_step(z_t, "z_t")
        self._validate_action_step(a_prev, z_t.shape[0], "a_prev")
        _validate_state_batch(state, z_t.shape[0], EmptyCarrierState)
        return StreamingCarrierOutput(state, z_t, torch.zeros_like(z_t))

    def imagine(self, state: EmptyCarrierState,
                a_t: torch.Tensor) -> StreamingCarrierOutput:
        batch = state.anchor.shape[0]
        self._validate_action_step(a_t, batch, "a_t")
        prior = a_t.new_zeros(batch, self.embed_dim)
        return StreamingCarrierOutput(state, None, prior)


class FrozenActionConditionedGRU(StreamingCarrierMixin, Carrier):
    """Parameter-matched GRU with the V19 pre-observation read convention."""

    name = "acgru"

    def __init__(self, embed_dim: int, action_dim: int,
                 hidden_dim: int | None = None) -> None:
        super().__init__(embed_dim, action_dim)
        self.hidden_dim = (matched_gru_hidden(embed_dim, action_dim)
                           if hidden_dim is None else int(hidden_dim))
        _validate_width(self.hidden_dim, "hidden_dim")
        self.cell = nn.GRUCell(embed_dim + action_dim, self.hidden_dim)
        self.w_o = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        nn.init.zeros_(self.w_o.weight)

    @property
    def state_dim(self) -> int:
        return self.hidden_dim

    def initialize(self, z0: torch.Tensor) -> StreamingCarrierOutput:
        self._validate_z_step(z0, "z0")
        with torch.autocast(device_type=z0.device.type, enabled=False):
            z32 = z0.float()
            hidden0 = z32.new_zeros(z32.shape[0], self.hidden_dim)
            action0 = z32.new_zeros(z32.shape[0], self.action_dim)
            prior = self.w_o(hidden0)
            hidden = self.cell(torch.cat([z32, action0], dim=-1), hidden0)
            fused = z32 + self.w_o(hidden)
        return _cast_stream_output(
            RecurrentCarrierState(hidden), fused, prior, z0.dtype)

    def observe(self, state: RecurrentCarrierState, z_t: torch.Tensor,
                a_prev: torch.Tensor) -> StreamingCarrierOutput:
        self._validate_z_step(z_t, "z_t")
        self._validate_action_step(a_prev, z_t.shape[0], "a_prev")
        _validate_state_batch(state, z_t.shape[0], RecurrentCarrierState)
        with torch.autocast(device_type=z_t.device.type, enabled=False):
            z32, a32 = z_t.float(), a_prev.float()
            prior = self.w_o(state.hidden)
            hidden = self.cell(torch.cat([z32, a32], dim=-1), state.hidden)
            fused = z32 + self.w_o(hidden)
        return _cast_stream_output(
            RecurrentCarrierState(hidden), fused, prior, z_t.dtype)

    def imagine(self, state: RecurrentCarrierState,
                a_t: torch.Tensor) -> StreamingCarrierOutput:
        batch = state.hidden.shape[0]
        self._validate_action_step(a_t, batch, "a_t")
        with torch.autocast(device_type=a_t.device.type, enabled=False):
            zero_z = state.hidden.new_zeros(batch, self.embed_dim)
            hidden = self.cell(torch.cat([zero_z, a_t.float()], dim=-1),
                               state.hidden)
            prior = self.w_o(hidden)
        return StreamingCarrierOutput(RecurrentCarrierState(hidden), None,
                                      prior.to(a_t.dtype))

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> CarrierOutput:
        return _stream_forward(self, z, actions, "hidden")

    def describe(self) -> dict[str, Any]:
        return _matched_description(self, self.hidden_dim, "hidden_dim",
                                    "zero_latent_action_only_GRU_step")


class FrozenActionConditionedLSTM(StreamingCarrierMixin, Carrier):
    """Parameter-matched action-conditioned LSTM baseline."""

    name = "aclstm"

    def __init__(self, embed_dim: int, action_dim: int,
                 hidden_dim: int | None = None) -> None:
        super().__init__(embed_dim, action_dim)
        self.hidden_dim = (matched_lstm_hidden(embed_dim, action_dim)
                           if hidden_dim is None else int(hidden_dim))
        _validate_width(self.hidden_dim, "hidden_dim")
        self.cell = nn.LSTMCell(embed_dim + action_dim, self.hidden_dim)
        self.w_o = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        nn.init.zeros_(self.w_o.weight)

    @property
    def state_dim(self) -> int:
        return self.hidden_dim

    def initialize(self, z0: torch.Tensor) -> StreamingCarrierOutput:
        self._validate_z_step(z0, "z0")
        with torch.autocast(device_type=z0.device.type, enabled=False):
            z32 = z0.float()
            hidden0 = z32.new_zeros(z32.shape[0], self.hidden_dim)
            cell0 = torch.zeros_like(hidden0)
            action0 = z32.new_zeros(z32.shape[0], self.action_dim)
            prior = self.w_o(hidden0)
            hidden, cell = self.cell(torch.cat([z32, action0], dim=-1),
                                     (hidden0, cell0))
            fused = z32 + self.w_o(hidden)
        return _cast_stream_output(
            LSTMCarrierState(hidden, cell), fused, prior, z0.dtype)

    def observe(self, state: LSTMCarrierState, z_t: torch.Tensor,
                a_prev: torch.Tensor) -> StreamingCarrierOutput:
        self._validate_z_step(z_t, "z_t")
        self._validate_action_step(a_prev, z_t.shape[0], "a_prev")
        _validate_state_batch(state, z_t.shape[0], LSTMCarrierState)
        with torch.autocast(device_type=z_t.device.type, enabled=False):
            z32, a32 = z_t.float(), a_prev.float()
            prior = self.w_o(state.hidden)
            hidden, cell = self.cell(torch.cat([z32, a32], dim=-1),
                                     (state.hidden, state.cell))
            fused = z32 + self.w_o(hidden)
        return _cast_stream_output(
            LSTMCarrierState(hidden, cell), fused, prior, z_t.dtype)

    def imagine(self, state: LSTMCarrierState,
                a_t: torch.Tensor) -> StreamingCarrierOutput:
        batch = state.hidden.shape[0]
        self._validate_action_step(a_t, batch, "a_t")
        with torch.autocast(device_type=a_t.device.type, enabled=False):
            zero_z = state.hidden.new_zeros(batch, self.embed_dim)
            hidden, cell = self.cell(torch.cat([zero_z, a_t.float()], dim=-1),
                                     (state.hidden, state.cell))
            prior = self.w_o(hidden)
        return StreamingCarrierOutput(LSTMCarrierState(hidden, cell), None,
                                      prior.to(a_t.dtype))

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> CarrierOutput:
        return _stream_forward(self, z, actions, "hidden", "cell")

    def describe(self) -> dict[str, Any]:
        return _matched_description(self, self.hidden_dim, "hidden_dim",
                                    "zero_latent_action_only_LSTM_step")


class FrozenDiagonalSSM(StreamingCarrierMixin, Carrier):
    """Parameter-matched learned diagonal linear state-space baseline."""

    name = "diag_ssm"

    def __init__(self, embed_dim: int, action_dim: int,
                 width: int | None = None) -> None:
        super().__init__(embed_dim, action_dim)
        self.width = (matched_ssm_width(embed_dim, action_dim)
                      if width is None else int(width))
        _validate_width(self.width, "width")
        decays = torch.exp(-SSM_DELTA * (
            torch.arange(self.width, dtype=torch.float32) + 0.5))
        decays = decays.clamp(1e-4, 1.0 - 1e-4)
        self.a_raw = nn.Parameter(torch.logit(decays))
        self.w_in = nn.Linear(embed_dim + action_dim, self.width)
        self.w_o = nn.Linear(self.width, embed_dim, bias=False)
        nn.init.zeros_(self.w_o.weight)

    @property
    def state_dim(self) -> int:
        return self.width

    def _step(self, hidden: torch.Tensor, z: torch.Tensor,
              action: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(self.a_raw) * hidden
                + self.w_in(torch.cat([z, action], dim=-1)))

    def initialize(self, z0: torch.Tensor) -> StreamingCarrierOutput:
        self._validate_z_step(z0, "z0")
        with torch.autocast(device_type=z0.device.type, enabled=False):
            z32 = z0.float()
            hidden0 = z32.new_zeros(z32.shape[0], self.width)
            action0 = z32.new_zeros(z32.shape[0], self.action_dim)
            prior = self.w_o(hidden0)
            hidden = self._step(hidden0, z32, action0)
            fused = z32 + self.w_o(hidden)
        return _cast_stream_output(
            RecurrentCarrierState(hidden), fused, prior, z0.dtype)

    def observe(self, state: RecurrentCarrierState, z_t: torch.Tensor,
                a_prev: torch.Tensor) -> StreamingCarrierOutput:
        self._validate_z_step(z_t, "z_t")
        self._validate_action_step(a_prev, z_t.shape[0], "a_prev")
        _validate_state_batch(state, z_t.shape[0], RecurrentCarrierState)
        with torch.autocast(device_type=z_t.device.type, enabled=False):
            z32, a32 = z_t.float(), a_prev.float()
            prior = self.w_o(state.hidden)
            hidden = self._step(state.hidden, z32, a32)
            fused = z32 + self.w_o(hidden)
        return _cast_stream_output(
            RecurrentCarrierState(hidden), fused, prior, z_t.dtype)

    def imagine(self, state: RecurrentCarrierState,
                a_t: torch.Tensor) -> StreamingCarrierOutput:
        batch = state.hidden.shape[0]
        self._validate_action_step(a_t, batch, "a_t")
        with torch.autocast(device_type=a_t.device.type, enabled=False):
            zero_z = state.hidden.new_zeros(batch, self.embed_dim)
            hidden = self._step(state.hidden, zero_z, a_t.float())
            prior = self.w_o(hidden)
        return StreamingCarrierOutput(RecurrentCarrierState(hidden), None,
                                      prior.to(a_t.dtype))

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> CarrierOutput:
        return _stream_forward(self, z, actions, "hidden")

    def describe(self) -> dict[str, Any]:
        result = _matched_description(self, self.width, "width",
                                      "zero_latent_action_only_SSM_step")
        result.update({"ssm_delta": SSM_DELTA,
                       "decay_init": torch.sigmoid(
                           self.a_raw.detach()).tolist()})
        return result


class FrozenFixedTrustLKC(StreamingCarrierMixin, LatentKalmanCell):
    """The publication fixed-trust LKC target (constant learned ``r``)."""

    name = "lkc_fixed_trust"

    def __init__(self, embed_dim: int, action_dim: int) -> None:
        super().__init__(embed_dim, action_dim, r_fixed=True)

    def initialize(self, z0: torch.Tensor) -> StreamingCarrierOutput:
        self._validate_z_step(z0, "z0")
        with torch.autocast(device_type=z0.device.type, enabled=False):
            z32 = z0.float()
            mean = self.w_x(z32)
            variance = torch.full_like(mean, self.sigma0)
            prior = torch.zeros_like(z32)
            fused = z32 + self.w_o(mean)
        return _cast_stream_output(
            KalmanCarrierState(mean, variance), fused, prior, z0.dtype)

    def _predict(self, state: KalmanCarrierState,
                 action: torch.Tensor) -> KalmanCarrierState:
        decay = self.decay()
        mean = decay * state.mean + self.b(action)
        variance = (decay.square() * state.variance
                    + F.softplus(self.q_raw)).clamp(SIGMA_FLOOR, SIGMA_CEIL)
        return KalmanCarrierState(mean, variance)

    def observe(self, state: KalmanCarrierState, z_t: torch.Tensor,
                a_prev: torch.Tensor) -> StreamingCarrierOutput:
        self._validate_z_step(z_t, "z_t")
        self._validate_action_step(a_prev, z_t.shape[0], "a_prev")
        _validate_state_batch(state, z_t.shape[0], KalmanCarrierState)
        with torch.autocast(device_type=z_t.device.type, enabled=False):
            z32, a32 = z_t.float(), a_prev.float()
            predicted = self._predict(state, a32)
            prior = self.w_o(predicted.mean)
            innovation = self.w_x(z32) - predicted.mean
            r = F.softplus(self.r_const).clamp_min(R_FLOOR)
            gain = predicted.variance / (predicted.variance + r)
            mean = predicted.mean + gain * innovation
            variance = ((1.0 - gain) * predicted.variance).clamp(
                SIGMA_FLOOR, SIGMA_CEIL)
            fused = z32 + self.w_o(mean)
        return _cast_stream_output(
            KalmanCarrierState(mean, variance), fused, prior, z_t.dtype)

    def imagine(self, state: KalmanCarrierState,
                a_t: torch.Tensor) -> StreamingCarrierOutput:
        batch = state.mean.shape[0]
        self._validate_action_step(a_t, batch, "a_t")
        with torch.autocast(device_type=a_t.device.type, enabled=False):
            predicted = self._predict(state, a_t.float())
            prior = self.w_o(predicted.mean)
        return StreamingCarrierOutput(predicted, None, prior.to(a_t.dtype))

    def describe(self) -> dict[str, Any]:
        result = super().describe()
        result.update({
            "carrier": self.name,
            "publication_arm": "fixed_trust_lkc",
            "parameter_target": self.parameter_count(),
            "prior_convention": "W_o_m_minus_before_frame_t",
            "imagine_rule": "Kalman_predict_without_correction",
        })
        return result


def _validate_width(width: int, name: str) -> None:
    if width < 1:
        raise ValueError(f"{name} must be positive, got {width}")


def _validate_state_batch(state: StreamState, batch: int,
                          expected_type: type[StreamState]) -> None:
    if not isinstance(state, expected_type):
        raise TypeError(f"state must be {expected_type.__name__}, "
                        f"got {type(state).__name__}")
    for tensor in _state_tensors(state):
        if tensor.shape[0] != batch:
            raise ValueError(
                f"state batch must be {batch}, got {tensor.shape[0]}")


def _cast_stream_output(state: StreamState, fused: torch.Tensor,
                        prior: torch.Tensor,
                        dtype: torch.dtype) -> StreamingCarrierOutput:
    return StreamingCarrierOutput(state, fused.to(dtype), prior.to(dtype))


def _stream_forward(carrier: Carrier, z: torch.Tensor, actions: torch.Tensor,
                    *telemetry_fields: str) -> CarrierOutput:
    carrier._validate(z, actions)
    with torch.autocast(device_type=z.device.type, enabled=False):
        first = carrier.initialize(z[:, 0])
        state = first.state
        fused = [first.fused_z]
        priors = [first.prior_read]
        telemetry: dict[str, list[torch.Tensor]] = {
            f"{name}_norm": [getattr(state, name).norm(dim=-1)]
            for name in telemetry_fields
        }
        for t in range(1, z.shape[1]):
            step = carrier.observe(state, z[:, t], actions[:, t - 1])
            state = step.state
            fused.append(step.fused_z)
            priors.append(step.prior_read)
            for name in telemetry_fields:
                telemetry[f"{name}_norm"].append(
                    getattr(state, name).norm(dim=-1))
        stacked_telemetry = {
            name: torch.stack(values, dim=1)
            for name, values in telemetry.items()
        }
    return CarrierOutput(torch.stack(fused, dim=1),
                         torch.stack(priors, dim=1), stacked_telemetry)


# -------------------------------------------------------------------------
# Exact count formulae and integer matching
# -------------------------------------------------------------------------

def fixed_trust_lkc_parameter_count(embed_dim: int, action_dim: int) -> int:
    """``W_x + B + W_o + q + r_const`` for N=D."""

    _validate_dims(embed_dim, action_dim)
    return 2 * embed_dim * embed_dim + embed_dim * action_dim + 2 * embed_dim


def gru_parameter_count(width: int, embed_dim: int, action_dim: int) -> int:
    """PyTorch GRUCell (two bias vectors) plus bias-free residual read."""

    _validate_dims(embed_dim, action_dim)
    _validate_width(width, "width")
    inp = embed_dim + action_dim
    return 3 * width * inp + 3 * width * width + 6 * width + embed_dim * width


def lstm_parameter_count(width: int, embed_dim: int, action_dim: int) -> int:
    """PyTorch LSTMCell (two bias vectors) plus bias-free residual read."""

    _validate_dims(embed_dim, action_dim)
    _validate_width(width, "width")
    inp = embed_dim + action_dim
    return 4 * width * inp + 4 * width * width + 8 * width + embed_dim * width


def ssm_parameter_count(width: int, embed_dim: int, action_dim: int) -> int:
    """Diagonal decay, biased input lift, and bias-free residual read."""

    _validate_dims(embed_dim, action_dim)
    _validate_width(width, "width")
    return width * (2 * embed_dim + action_dim + 2)


def _validate_dims(embed_dim: int, action_dim: int) -> None:
    if embed_dim < 2 or action_dim < 1:
        raise ValueError(f"invalid carrier dims D={embed_dim}, A={action_dim}")


def _closest_integer_width(count_fn, embed_dim: int, action_dim: int) -> int:
    """Global closest positive width for a strictly increasing count formula."""

    target = fixed_trust_lkc_parameter_count(embed_dim, action_dim)
    low, high = 1, 1
    while count_fn(high, embed_dim, action_dim) < target:
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if count_fn(middle, embed_dim, action_dim) < target:
            low = middle
        else:
            high = middle
    candidates = {low, high}
    return min(candidates, key=lambda width: (
        abs(count_fn(width, embed_dim, action_dim) - target), width))


def matched_gru_hidden(embed_dim: int, action_dim: int) -> int:
    return _closest_integer_width(gru_parameter_count, embed_dim, action_dim)


def matched_lstm_hidden(embed_dim: int, action_dim: int) -> int:
    return _closest_integer_width(lstm_parameter_count, embed_dim, action_dim)


def matched_ssm_width(embed_dim: int, action_dim: int) -> int:
    return _closest_integer_width(ssm_parameter_count, embed_dim, action_dim)


def parameter_match_report(
        embed_dim: int = OFFICIAL_LEWM_EMBED_DIM,
        action_dim: int = OFFICIAL_LEWM_ACTION_DIM) -> dict[str, Any]:
    """Return exact, JSON-serializable publication parameter counts."""

    target = fixed_trust_lkc_parameter_count(embed_dim, action_dim)
    specs = {
        "acgru": (matched_gru_hidden(embed_dim, action_dim),
                  gru_parameter_count, "hidden_dim"),
        "aclstm": (matched_lstm_hidden(embed_dim, action_dim),
                   lstm_parameter_count, "hidden_dim"),
        "diag_ssm": (matched_ssm_width(embed_dim, action_dim),
                     ssm_parameter_count, "width"),
    }
    arms: dict[str, dict[str, int | float | str | None]] = {
        "none": {"width_name": None, "width": None, "parameters": 0,
                 "delta": -target, "relative_mismatch": 1.0},
        "lkc_fixed_trust": {
            "width_name": "state_dim", "width": embed_dim,
            "parameters": target, "delta": 0, "relative_mismatch": 0.0,
        },
    }
    for name, (width, count_fn, width_name) in specs.items():
        count = count_fn(width, embed_dim, action_dim)
        arms[name] = {
            "width_name": width_name,
            "width": width,
            "parameters": count,
            "delta": count - target,
            "relative_mismatch": abs(count - target) / target,
        }
    return {"embed_dim": embed_dim, "action_dim": action_dim,
            "target_parameters": target, "arms": arms}


def _matched_description(carrier: Carrier, width: int, width_name: str,
                         imagine_rule: str) -> dict[str, Any]:
    report = parameter_match_report(carrier.embed_dim, carrier.action_dim)
    target = report["target_parameters"]
    count = carrier.parameter_count()
    return {
        "carrier": carrier.name,
        "embed_dim": carrier.embed_dim,
        "action_dim": carrier.action_dim,
        width_name: width,
        "parameters": count,
        "parameter_target": target,
        "parameter_delta": count - target,
        "relative_mismatch": abs(count - target) / target,
        "prior_convention": "W_o_state_t_minus_1_before_frame_t",
        "imagine_rule": imagine_rule,
    }


def make_frozen_swap_carrier(name: str, embed_dim: int, action_dim: int,
                             **kwargs: Any) -> Carrier:
    """Build one registered frozen-host publication carrier."""

    aliases = {
        "ssm": "diag_ssm",
        "acssm": "diag_ssm",
        "lstm": "aclstm",
        "lkc": "lkc_fixed_trust",
        "lkc_rfix": "lkc_fixed_trust",
        "fixed_trust_lkc": "lkc_fixed_trust",
    }
    canonical = aliases.get(name, name)
    if canonical == "none":
        return FrozenNoCarrier(embed_dim, action_dim)
    if canonical == "acgru":
        return FrozenActionConditionedGRU(
            embed_dim, action_dim, kwargs.get("hidden_dim"))
    if canonical == "aclstm":
        return FrozenActionConditionedLSTM(
            embed_dim, action_dim, kwargs.get("hidden_dim"))
    if canonical == "diag_ssm":
        return FrozenDiagonalSSM(embed_dim, action_dim, kwargs.get("width"))
    if canonical == "lkc_fixed_trust":
        return FrozenFixedTrustLKC(embed_dim, action_dim)
    raise KeyError(
        f"unknown frozen-swap carrier {name!r}; expected one of "
        f"{FROZEN_SWAP_CARRIER_NAMES}")


def make_frozen_carrier(name: str, embed_dim: int, action_dim: int) -> Carrier:
    """Trainer-facing factory with concise publication arm names.

    Accepted names are exactly ``none``, ``gru``, ``lstm``, ``ssm`` and
    ``fixed_trust``.  The longer registry labels remain available through
    :func:`make_frozen_swap_carrier` for analysis code and figure ledgers.
    """

    mapping = {
        "none": "none",
        "gru": "acgru",
        "lstm": "aclstm",
        "ssm": "diag_ssm",
        "fixed_trust": "lkc_fixed_trust",
    }
    try:
        canonical = mapping[name]
    except KeyError as error:
        raise KeyError(
            f"unknown frozen carrier {name!r}; expected one of "
            f"{FROZEN_CARRIER_NAMES}") from error
    return make_frozen_swap_carrier(canonical, embed_dim, action_dim)


def parameter_report(embed_dim: int, action_dim: int) -> dict[str, Any]:
    """Trainer-facing alias for the exact parameter-match ledger."""

    return parameter_match_report(embed_dim, action_dim)


__all__ = [
    "OFFICIAL_LEWM_EMBED_DIM", "OFFICIAL_LEWM_ACTION_DIM",
    "FROZEN_SWAP_CARRIER_NAMES", "FROZEN_CARRIER_NAMES",
    "EmptyCarrierState",
    "RecurrentCarrierState", "LSTMCarrierState", "KalmanCarrierState",
    "StreamingCarrierOutput", "clone_stream_state", "repeat_stream_state",
    "FrozenNoCarrier", "FrozenActionConditionedGRU",
    "FrozenActionConditionedLSTM", "FrozenDiagonalSSM",
    "FrozenFixedTrustLKC", "fixed_trust_lkc_parameter_count",
    "gru_parameter_count", "lstm_parameter_count", "ssm_parameter_count",
    "matched_gru_hidden", "matched_lstm_hidden", "matched_ssm_width",
    "parameter_match_report", "parameter_report", "make_frozen_swap_carrier",
    "make_frozen_carrier",
]
