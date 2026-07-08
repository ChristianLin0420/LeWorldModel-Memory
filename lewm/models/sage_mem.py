"""SAGE-Mem: an audit-guided causal memory carrier.

SAGE-Mem (Surprise-gated, Age-balanced, Global, Exposure-calibrated
Memory) is kept separate from the registered publication carriers while it is
being evaluated.  Its inference rule targets the failures localized by the
memory audit:

* fixed multi-scale decay protects old events from uniform overwrite;
* an innovation-dependent gate writes surprising observations preferentially;
* deterministic local/regional/global aggregation exposes spatial memory to a
  frozen DINO-WM predictor; and
* a reset-maturity trace suppresses reads from newly reset state.

The module supports both carrier sequence layouts directly::

    LeWM:   z (B, L, D),       actions (B, L - 1, A)
    DINO:   z (B, L, P, D),    actions (B, L - 1, A)

Streaming ``initialize``/``observe`` calls accept ``(B,D)`` or ``(B,P,D)``
observations.  ``prior_read`` at time t is always formed from the action-
predicted state before observation t is consumed.  All recurrence arithmetic
is performed in fp32 with autocast disabled; reads are cast back to the input
dtype at the boundary.

The only trainable tensors are W_x, B, W_o, and two channel-wise gate vectors.
Consequently the count is exactly

    D^2 + DA + D^2 + D + D = D(2D + A + 2),

matching the fixed-trust and diagonal-SSM publication arms.  W_o is
zero-initialized, so both fused and prior outputs are exactly the no-state host
at initialization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.v19_carriers import Carrier, CarrierOutput
from lewm.models.frozen_swap_carriers import StreamingCarrierOutput


DEFAULT_HALF_LIVES = (4.0, 8.0, 16.0, 64.0)
DEFAULT_AGGREGATION_WEIGHTS = (0.5, 0.25, 0.25)
DEFAULT_REGION_SIZE = 2
DEFAULT_MATURITY_SCALE = 4.0
DEFAULT_GATE_SLOPE = 4.0
DEFAULT_GATE_REFERENCE = 1.0
SAGE_MEM_API_VERSION = "sage_mem_v1_api_v1"
SAGE_MEM_VARIANTS = (
    "full", "next_only", "no_exposure", "exposure_only",
)


@dataclass(frozen=True)
class SAGEMemState:
    """Streaming state, kept in fp32 even for lower-precision host inputs.

    ``memory`` has shape ``(B,D)`` for LeWM or ``(B,P,D)`` for a spatial
    host.  ``write_mass`` has the matching leading dimensions and a singleton
    feature dimension.  It is converted to read confidence by
    ``1 - exp(-write_mass / maturity_scale)``.
    """

    memory: torch.Tensor
    write_mass: torch.Tensor


def sage_mem_parameter_count(embed_dim: int, action_dim: int) -> int:
    """Exact SAGE-Mem inference parameter count."""

    if embed_dim < 2 or action_dim < 1:
        raise ValueError(
            f"invalid carrier dims D={embed_dim}, A={action_dim}")
    return embed_dim * (2 * embed_dim + action_dim + 2)


def _canonical_variant(variant: str) -> str:
    if not isinstance(variant, str):
        raise ValueError("variant must be a string")
    canonical = variant.removeprefix("sage_mem_")
    if canonical not in SAGE_MEM_VARIANTS:
        raise ValueError(
            f"unknown SAGE-Mem variant {variant!r}; expected one of "
            f"{SAGE_MEM_VARIANTS}")
    return canonical


class SAGEMem(Carrier):
    """Surprise-gated, multi-timescale carrier with spatial exposure.

    The fixed decay spectrum is assigned round-robin over channels.  Spatial
    aggregation uses a 2-D patch grid when ``P`` is a compatible square.  For
    non-grid token sets it falls back to the global mean for the regional
    component; ``P=1`` is an exact identity aggregation.
    """

    name = "sage_mem"

    def __init__(
            self, embed_dim: int, action_dim: int, *,
            variant: str = "full",
            half_lives: Sequence[float] = DEFAULT_HALF_LIVES,
            aggregation_weights: Sequence[float] =
            DEFAULT_AGGREGATION_WEIGHTS,
            region_size: int = DEFAULT_REGION_SIZE,
            maturity_scale: float = DEFAULT_MATURITY_SCALE) -> None:
        super().__init__(embed_dim, action_dim)
        self.variant = _canonical_variant(variant)
        half_life_values = tuple(float(value) for value in half_lives)
        if not half_life_values or len(half_life_values) > embed_dim:
            raise ValueError(
                "half_lives must contain between 1 and embed_dim values")
        if any(not math.isfinite(value) or value <= 0.0
               for value in half_life_values):
            raise ValueError("half_lives must be positive and finite")

        weights = tuple(float(value) for value in aggregation_weights)
        if len(weights) != 3 or any(
                not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError(
                "aggregation_weights must be three finite non-negative values")
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            raise ValueError("aggregation_weights must have positive sum")
        self.aggregation_weights = tuple(value / weight_sum
                                         for value in weights)

        if (not isinstance(region_size, int) or isinstance(region_size, bool)
                or region_size < 1):
            raise ValueError("region_size must be a positive integer")
        if not math.isfinite(maturity_scale) or maturity_scale <= 0.0:
            raise ValueError("maturity_scale must be positive and finite")
        self.region_size = region_size
        self.maturity_scale = float(maturity_scale)

        # Exact matched budget: D^2 + DA + D^2 + 2D.
        self.w_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.action_projection = nn.Linear(
            action_dim, embed_dim, bias=False)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gate_threshold = nn.Parameter(torch.empty(embed_dim))
        self.gate_log_slope = nn.Parameter(torch.empty(embed_dim))

        # Identity lift makes innovation magnitudes interpretable at step 0;
        # action dynamics start neutral and are learned from the host loss.
        with torch.no_grad():
            self.w_x.weight.copy_(torch.eye(embed_dim))
            self.action_projection.weight.zero_()
            self.w_o.weight.zero_()
            self.gate_threshold.fill_(math.log1p(DEFAULT_GATE_REFERENCE))
            self.gate_log_slope.fill_(
                math.log(math.expm1(DEFAULT_GATE_SLOPE)))

        assignments = torch.arange(embed_dim) % len(half_life_values)
        half_life_tensor = torch.tensor(
            half_life_values, dtype=torch.float32)[assignments]
        decay = torch.exp2(-1.0 / half_life_tensor)
        self.register_buffer("half_lives", half_life_tensor)
        self.register_buffer("decay", decay)
        self.register_buffer(
            "maturity_decay",
            torch.tensor(
                2.0 ** (-1.0 / max(half_life_values)),
                dtype=torch.float32),
        )

    @property
    def state_dim(self) -> int:
        return self.embed_dim

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------

    def initialize(self, z0: torch.Tensor) -> StreamingCarrierOutput:
        """Reset state and consume the first observation."""

        self._validate_z_step(z0, "z0")
        with torch.autocast(device_type=z0.device.type, enabled=False):
            output, _, _, _ = self._initialize_with_telemetry(z0)
        return output

    def observe(self, state: SAGEMemState, z_t: torch.Tensor,
                a_prev: torch.Tensor) -> StreamingCarrierOutput:
        """Predict with ``a_prev``, emit the causal prior, then consume z_t."""

        self._validate_z_step(z_t, "z_t")
        self._validate_action_step(a_prev, z_t.shape[0], "a_prev")
        self._validate_state(state, z_t)
        with torch.autocast(device_type=z_t.device.type, enabled=False):
            output, _, _, _ = self._observe_with_telemetry(
                state, z_t, a_prev)
        return output

    def imagine(self, state: SAGEMemState,
                a_t: torch.Tensor) -> StreamingCarrierOutput:
        """Apply one action-only prediction without an observation write."""

        self._validate_state(state)
        self._validate_action_step(a_t, state.memory.shape[0], "a_t")
        if a_t.device != state.memory.device:
            raise ValueError("a_t and state must share a device")
        with torch.autocast(device_type=a_t.device.type, enabled=False):
            if self.variant == "exposure_only":
                state = SAGEMemState(
                    torch.zeros_like(state.memory),
                    torch.zeros_like(state.write_mass),
                )
            predicted = self._predict(state, a_t.float())
            prior = self._read(predicted)
        # The state remains fp32; the host-facing read follows action dtype,
        # matching the other streaming publication carriers.
        return StreamingCarrierOutput(  # type: ignore[arg-type]
            predicted, None, prior.to(a_t.dtype))

    @staticmethod
    def clone_state(state: SAGEMemState) -> SAGEMemState:
        if not isinstance(state, SAGEMemState):
            raise TypeError(
                f"state must be SAGEMemState, got {type(state).__name__}")
        return SAGEMemState(state.memory.clone(), state.write_mass.clone())

    @staticmethod
    def repeat_state(state: SAGEMemState, repeats: int) -> SAGEMemState:
        if not isinstance(state, SAGEMemState):
            raise TypeError(
                f"state must be SAGEMemState, got {type(state).__name__}")
        if (not isinstance(repeats, int) or isinstance(repeats, bool)
                or repeats < 1):
            raise ValueError(
                f"repeats must be a positive integer, got {repeats!r}")
        return SAGEMemState(
            state.memory.repeat_interleave(repeats, dim=0),
            state.write_mass.repeat_interleave(repeats, dim=0),
        )

    # ------------------------------------------------------------------
    # Batched LeWM and spatial DINO-WM API
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor,
                actions: torch.Tensor) -> CarrierOutput:
        """Apply SAGE-Mem causally to LeWM or spatial DINO-WM latents."""

        parts = self._forward_parts(z, actions, reset_mask=None)
        return CarrierOutput(
            z_tilde=parts["fused"],
            prior_read=parts["prior"],
            telemetry=parts["diagnostics"],
        )

    def forward_sequence(
            self, features: torch.Tensor, actions: torch.Tensor, *,
            reset_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Versioned protocol API with optional per-episode legal resets.

        ``reset_mask`` is boolean ``(B,L)``.  A true entry clears persistent
        state immediately before that observation.  ``posterior`` is the raw
        fp32 state after each observation; ``exposure`` is only the residual
        actually injected into the frozen host.  The ``no_exposure`` control
        therefore returns an all-zero exposure while retaining its posterior
        and causal prior for storage diagnostics.
        """

        return self._forward_parts(features, actions, reset_mask=reset_mask)

    def _forward_parts(
            self, z: torch.Tensor, actions: torch.Tensor, *,
            reset_mask: torch.Tensor | None
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        self._validate_sequence(z, actions)
        reset_mask = self._validate_reset_mask(reset_mask, z)
        with torch.autocast(device_type=z.device.type, enabled=False):
            first, diagnostics, posterior, exposure = (
                self._initialize_with_telemetry(z[:, 0]))
            state = first.state
            fused = [first.fused_z]
            priors = [first.prior_read]
            posteriors = [posterior]
            exposures = [exposure]
            traces = {name: [value]
                      for name, value in diagnostics.items()}

            for time in range(1, z.shape[1]):
                state = self._apply_reset_mask(
                    state, reset_mask[:, time])
                step, diagnostics, posterior, exposure = (
                    self._observe_with_telemetry(
                        state, z[:, time], actions[:, time - 1]))
                state = step.state
                fused.append(step.fused_z)
                priors.append(step.prior_read)
                posteriors.append(posterior)
                exposures.append(exposure)
                for name, value in diagnostics.items():
                    traces[name].append(value)

            telemetry = {
                name: torch.stack(values, dim=1)
                for name, values in traces.items()
            }
        return {
            "fused": torch.stack(fused, dim=1),
            "prior": torch.stack(priors, dim=1),
            "posterior": torch.stack(posteriors, dim=1),
            "exposure": torch.stack(exposures, dim=1),
            "diagnostics": telemetry,
        }

    # ------------------------------------------------------------------
    # Parameter-free spatial exposure
    # ------------------------------------------------------------------

    def aggregation_components(
            self, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return aligned local, regional, and global memory tensors.

        The returned tensors have the same shape as ``memory``.  ``(B,D)``
        and ``(B,1,D)`` inputs are exact identity cases.  A 196-token DINO
        stream is interpreted as a 14x14 grid and pooled in non-overlapping
        2x2 regions before broadcasting back to patch positions.
        """

        if memory.ndim not in (2, 3) or memory.shape[-1] != self.embed_dim:
            raise ValueError(
                "memory must be (B,D) or (B,P,D), got "
                + str(tuple(memory.shape)))
        if memory.ndim == 2:
            return memory, memory, memory

        patches = memory.shape[1]
        if patches < 1:
            raise ValueError("memory must contain at least one patch")
        if patches == 1:
            return memory, memory, memory

        local = memory
        global_value = memory.mean(dim=1, keepdim=True)
        global_memory = global_value.expand_as(memory)

        side = math.isqrt(patches)
        region = self.region_size
        if side * side == patches and side % region == 0:
            batch, _, feature = memory.shape
            grid = memory.reshape(batch, side, side, feature)
            coarse = grid.reshape(
                batch, side // region, region,
                side // region, region, feature).mean(dim=(2, 4))
            regional = coarse[:, :, None, :, None, :].expand(
                batch, side // region, region,
                side // region, region, feature).reshape_as(memory)
        else:
            # No spatial geometry is invented for an arbitrary token set.
            regional = global_memory
        return local, regional, global_memory

    def aggregate(self, memory: torch.Tensor) -> torch.Tensor:
        """Fixed local/regional/global mixture used by the residual read."""

        local, regional, global_memory = self.aggregation_components(memory)
        if memory.ndim == 2 or memory.shape[1] == 1:
            return memory
        local_weight, regional_weight, global_weight = (
            self.aggregation_weights)
        return (local_weight * local
                + regional_weight * regional
                + global_weight * global_memory)

    # ------------------------------------------------------------------
    # Recurrence internals
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_fp32(value: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        return F.linear(value, layer.weight.float(), bias=None)

    def _predict(self, state: SAGEMemState,
                 action: torch.Tensor) -> SAGEMemState:
        action_effect = self._linear_fp32(action, self.action_projection)
        if state.memory.ndim == 3:
            action_effect = action_effect[:, None, :]
        memory = (self.decay.float() * state.memory.float()
                  + action_effect)
        write_mass = self.maturity_decay.float() * state.write_mass.float()
        return SAGEMemState(memory, write_mass)

    def _correct(
            self, predicted: SAGEMemState, observation: torch.Tensor
    ) -> tuple[SAGEMemState, torch.Tensor, torch.Tensor]:
        lifted = self._linear_fp32(observation.float(), self.w_x)
        innovation = lifted - predicted.memory
        surprise = torch.log1p(innovation.abs())
        slope = F.softplus(self.gate_log_slope.float())
        gate = torch.sigmoid(
            slope * (surprise - self.gate_threshold.float()))
        memory = predicted.memory + gate * innovation
        write_mass = predicted.write_mass + gate.mean(
            dim=-1, keepdim=True)
        return SAGEMemState(memory, write_mass), gate, innovation

    def _confidence(self, state: SAGEMemState) -> torch.Tensor:
        scaled = state.write_mass.clamp_min(0.0) / self.maturity_scale
        return -torch.expm1(-scaled)

    def _read(self, state: SAGEMemState) -> torch.Tensor:
        exposed = self.aggregate(state.memory)
        residual = self._linear_fp32(exposed, self.w_o)
        return self._confidence(state) * residual

    def _applied_exposure(self, state: SAGEMemState) -> torch.Tensor:
        candidate = self._read(state)
        if self.variant == "no_exposure":
            return torch.zeros_like(candidate)
        return candidate

    def _empty_state(self, observation: torch.Tensor) -> SAGEMemState:
        memory = observation.float().new_zeros(observation.shape)
        write_mass = memory.new_zeros(memory.shape[:-1] + (1,))
        return SAGEMemState(memory, write_mass)

    @staticmethod
    def _apply_reset_mask(
            state: SAGEMemState, reset: torch.Tensor) -> SAGEMemState:
        memory_mask = reset.reshape(
            (reset.shape[0],) + (1,) * (state.memory.ndim - 1))
        mass_mask = reset.reshape(
            (reset.shape[0],) + (1,) * (state.write_mass.ndim - 1))
        return SAGEMemState(
            torch.where(memory_mask, torch.zeros_like(state.memory),
                        state.memory),
            torch.where(mass_mask, torch.zeros_like(state.write_mass),
                        state.write_mass),
        )

    def _initialize_with_telemetry(
            self, z0: torch.Tensor
    ) -> tuple[
        StreamingCarrierOutput, dict[str, torch.Tensor],
        torch.Tensor, torch.Tensor,
    ]:
        z32 = z0.float()
        predicted = self._empty_state(z32)
        prior = self._read(predicted)
        state, gate, innovation = self._correct(predicted, z32)
        exposure = self._applied_exposure(state)
        fused = z32 + exposure
        diagnostics = self._diagnostics(
            state, predicted, gate, innovation, prior, exposure)
        output = StreamingCarrierOutput(  # type: ignore[arg-type]
            state, fused.to(z0.dtype), prior.to(z0.dtype))
        return output, diagnostics, state.memory, exposure.to(z0.dtype)

    def _observe_with_telemetry(
            self, state: SAGEMemState, z_t: torch.Tensor,
            a_prev: torch.Tensor
    ) -> tuple[
        StreamingCarrierOutput, dict[str, torch.Tensor],
        torch.Tensor, torch.Tensor,
    ]:
        z32, action32 = z_t.float(), a_prev.float()
        if self.variant == "exposure_only":
            state = self._empty_state(z32)
        predicted = self._predict(state, action32)
        prior = self._read(predicted)
        corrected, gate, innovation = self._correct(predicted, z32)
        exposure = self._applied_exposure(corrected)
        fused = z32 + exposure
        diagnostics = self._diagnostics(
            corrected, predicted, gate, innovation, prior, exposure)
        output = StreamingCarrierOutput(  # type: ignore[arg-type]
            corrected, fused.to(z_t.dtype), prior.to(z_t.dtype))
        return (output, diagnostics, corrected.memory,
                exposure.to(z_t.dtype))

    def _diagnostics(
            self, state: SAGEMemState, predicted: SAGEMemState,
            gate: torch.Tensor, innovation: torch.Tensor,
            prior: torch.Tensor, exposure: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local, regional, global_memory = self.aggregation_components(
            state.memory)
        return {
            "state_norm": self._mean_patch_norm(state.memory),
            "prior_state_norm": self._mean_patch_norm(predicted.memory),
            "innovation_norm": self._mean_patch_norm(innovation),
            "write_gate_mean": self._mean_nonbatch(gate),
            "maturity_mean": self._mean_nonbatch(self._confidence(state)),
            "prior_read_norm": self._mean_patch_norm(prior),
            "exposure_norm": self._mean_patch_norm(exposure),
            "local_norm": self._mean_patch_norm(local),
            "regional_norm": self._mean_patch_norm(regional),
            "global_norm": self._mean_patch_norm(global_memory),
        }

    @staticmethod
    def _mean_patch_norm(value: torch.Tensor) -> torch.Tensor:
        norm = value.norm(dim=-1)
        return norm if norm.ndim == 1 else norm.mean(dim=1)

    @staticmethod
    def _mean_nonbatch(value: torch.Tensor) -> torch.Tensor:
        dimensions = tuple(range(1, value.ndim))
        return value.mean(dim=dimensions)

    # ------------------------------------------------------------------
    # Validation and registration metadata
    # ------------------------------------------------------------------

    def _validate_z_step(self, z: torch.Tensor, name: str) -> None:
        if (z.ndim not in (2, 3) or z.shape[-1] != self.embed_dim
                or (z.ndim == 3 and z.shape[1] < 1)):
            raise ValueError(
                f"{name} must be (B,{self.embed_dim}) or "
                f"(B,P,{self.embed_dim}), got {tuple(z.shape)}")
        if not z.is_floating_point():
            raise ValueError(f"{name} must be floating point")

    def _validate_action_step(
            self, action: torch.Tensor, batch: int, name: str) -> None:
        expected = (batch, self.action_dim)
        if tuple(action.shape) != expected:
            raise ValueError(
                f"{name} must be {expected}, got {tuple(action.shape)}")
        if not action.is_floating_point():
            raise ValueError(f"{name} must be floating point")

    def _validate_state(
            self, state: SAGEMemState,
            observation: torch.Tensor | None = None) -> None:
        if not isinstance(state, SAGEMemState):
            raise TypeError(
                f"state must be SAGEMemState, got {type(state).__name__}")
        if (state.memory.ndim not in (2, 3)
                or state.memory.shape[-1] != self.embed_dim):
            raise ValueError("state memory has an invalid shape")
        expected_mass = state.memory.shape[:-1] + (1,)
        if tuple(state.write_mass.shape) != expected_mass:
            raise ValueError(
                f"state write_mass must be {expected_mass}, got "
                f"{tuple(state.write_mass.shape)}")
        if state.memory.device != state.write_mass.device:
            raise ValueError("state tensors must share a device")
        if (state.memory.dtype != torch.float32
                or state.write_mass.dtype != torch.float32):
            raise ValueError("SAGE-Mem recurrence state must remain fp32")
        if observation is not None:
            if tuple(observation.shape) != tuple(state.memory.shape):
                raise ValueError(
                    "observation and state memory shapes must match, got "
                    f"{tuple(observation.shape)} and "
                    f"{tuple(state.memory.shape)}")
            if observation.device != state.memory.device:
                raise ValueError("observation and state must share a device")

    def _validate_sequence(
            self, z: torch.Tensor, actions: torch.Tensor) -> None:
        if (z.ndim not in (3, 4) or z.shape[-1] != self.embed_dim
                or z.shape[1] < 1
                or (z.ndim == 4 and z.shape[2] < 1)):
            raise ValueError(
                f"z must be (B,L,{self.embed_dim}) or "
                f"(B,L,P,{self.embed_dim}), got {tuple(z.shape)}")
        if not z.is_floating_point():
            raise ValueError("z must be floating point")
        expected = (z.shape[0], z.shape[1] - 1, self.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(
                f"actions must be {expected}, got {tuple(actions.shape)}")
        if not actions.is_floating_point():
            raise ValueError("actions must be floating point")
        if z.device != actions.device:
            raise ValueError("z and actions must share a device")

    @staticmethod
    def _validate_reset_mask(
            reset_mask: torch.Tensor | None,
            z: torch.Tensor) -> torch.Tensor:
        expected = (z.shape[0], z.shape[1])
        if reset_mask is None:
            return torch.zeros(expected, dtype=torch.bool, device=z.device)
        if not isinstance(reset_mask, torch.Tensor):
            raise ValueError("reset_mask must be a boolean tensor or None")
        if tuple(reset_mask.shape) != expected:
            raise ValueError(
                f"reset_mask must be {expected}, got "
                f"{tuple(reset_mask.shape)}")
        if reset_mask.dtype != torch.bool:
            raise ValueError("reset_mask must have boolean dtype")
        if reset_mask.device != z.device:
            raise ValueError("reset_mask and features must share a device")
        return reset_mask

    def persistent_state_floats(self) -> int:
        """Persistent fp32 scalars per vector or patch stream."""

        return self.embed_dim + 1

    def estimate_flops(
            self, *, batch_size: int, timesteps: int, tokens: int) -> int:
        """Deterministic inference FLOP estimate used by fairness ledgers.

        A multiply-add counts as two FLOPs.  The estimate includes one latent
        lift, two projected reads, action prediction, gate arithmetic, and two
        local/regional/global aggregations per observed token.  It deliberately
        uses the same formula for all four controls because their allocated
        inference structure and parameter budget are identical.
        """

        values = {
            "batch_size": batch_size,
            "timesteps": timesteps,
            "tokens": tokens,
        }
        for name, value in values.items():
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        d, a = self.embed_dim, self.action_dim
        per_token_step = 6 * d * d + 32 * d
        per_batch_step = 2 * d * a
        return int(batch_size * timesteps
                   * (tokens * per_token_step + per_batch_step))

    def describe(self) -> dict[str, Any]:
        return {
            "api_version": SAGE_MEM_API_VERSION,
            "carrier": self.name,
            "variant": self.variant,
            "embed_dim": self.embed_dim,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "parameters": self.parameter_count(),
            "parameter_target": sage_mem_parameter_count(
                self.embed_dim, self.action_dim),
            "parameter_formula": "D(2D+A+2)",
            "half_lives": [float(value) for value in
                           torch.unique(self.half_lives).tolist()],
            "decay_assignment": "round_robin_over_channels",
            "write_rule": "sigmoid_surprise_gated_predict_correct",
            "surprise": "log1p(abs(W_x z - m_minus))",
            "aggregation": "fixed_local_regional_global",
            "aggregation_weights": list(self.aggregation_weights),
            "region_size": self.region_size,
            "maturity_scale": self.maturity_scale,
            "prior_convention": "predicted_state_before_observation_t",
            "imagine_rule": "fixed_decay_action_predict_without_write",
            "internal_dtype": "float32",
            "zero_initialized_read": True,
            "persistent_state": self.variant != "exposure_only",
            "host_exposure_enabled": self.variant != "no_exposure",
            "training_objective": (
                "next_feature_only" if self.variant == "next_only"
                else "registered_label_free_objective"),
        }


def build_sage_mem_v1(*, embed_dim: int, action_dim: int, variant: str,
                      config: Mapping[str, Any]) -> SAGEMem:
    """Build a versioned SAGE-Mem control from a sealed configuration.

    Model options may be supplied directly or under a ``sage_mem`` mapping.
    Unrelated top-level protocol fields are intentionally ignored, while an
    unknown key inside the dedicated mapping fails closed.
    """

    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    allowed = {
        "half_lives", "aggregation_weights", "region_size",
        "maturity_scale",
    }
    nested = config.get("sage_mem", {})
    if not isinstance(nested, Mapping):
        raise ValueError("config.sage_mem must be a mapping")
    unknown = set(nested).difference(allowed)
    if unknown:
        raise ValueError(
            f"unknown config.sage_mem keys: {sorted(unknown)}")

    options: dict[str, Any] = {}
    for key in allowed:
        direct_present = key in config
        nested_present = key in nested
        if direct_present and nested_present and config[key] != nested[key]:
            raise ValueError(f"conflicting direct and nested value for {key}")
        if nested_present:
            options[key] = nested[key]
        elif direct_present:
            options[key] = config[key]
    return SAGEMem(
        embed_dim, action_dim, variant=variant, **options)


__all__ = [
    "DEFAULT_HALF_LIVES", "DEFAULT_AGGREGATION_WEIGHTS",
    "DEFAULT_REGION_SIZE", "DEFAULT_MATURITY_SCALE",
    "SAGE_MEM_API_VERSION", "SAGE_MEM_VARIANTS",
    "SAGEMemState", "SAGEMem", "sage_mem_parameter_count",
    "build_sage_mem_v1",
]
