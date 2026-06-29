"""SIRO-v12 stable identified residual observer.

This module is intentionally self contained.  SIRO's transition and observation
operators are fitted outside the module from training trajectories and installed as
non-gradient buffers.  The online cell keeps three causal streams:

``c``
    the initial episode anchor, copied exactly for the lifetime of the stream;
``r``
    anchor-relative autonomous dynamics and observation innovations;
``u``
    the response driven only by executed actions.

For the nominated read, ``h = c + r + u``.  The ``spectralshrink`` ablation instead
reads ``h = c + r + R u`` using the fitted action-read operator.  No optimizer parameter,
fixed memory bank, selected horizon, or learned correction head is introduced here.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableIdentifiedResidualObserverMemory(nn.Module):
    """Anchor-centered observer with externally fitted stable linear operators.

    The persistent state has shape ``(B,3,D)`` and stores ``[c,r,u]``.  Given a
    native action ``a`` the observation-free transition is

    .. code-block:: text

        c^- = c
        r^- = A_eff r + b
        u^- = A_eff u + B_eff a
        h^- = c^- + r^- + R_eff u^-

    An observed embedding supplies ``e = z-h^-``.  Full SIRO writes the fitted
    paired-view LMMSE estimate ``mu_c + K(e-mu_o)`` to ``r`` only.  ``identityK``
    writes ``e`` exactly, so its posterior read is exactly ``z``.  ``identityA``
    replaces only ``A_eff`` by the identity, ``noaction`` replaces only ``B_eff``
    by zero, and ``spectralshrink`` is the only mode that deploys the fitted ``R``.
    ``noanchor`` retains the common three-stream schema but initializes ``c=0,r=z0``;
    paired with an absolute-coordinate fit, it is the exact pre-anchor control.

    All modes retain one common serialized tensor schema.  The class deliberately
    does not implement the fitting algorithm; :meth:`install_fitted_operators`
    is the audited boundary between detached system identification and deployment.
    """

    MODES = frozenset({
        "full", "spectralshrink", "identityA", "identityK", "noaction",
        "noanchor",
    })

    def __init__(self, embed_dim: int, action_dim: int, mode: str = "full") -> None:
        super().__init__()
        if (not isinstance(embed_dim, int) or isinstance(embed_dim, bool)
                or embed_dim < 1):
            raise ValueError(
                f"SIRO embed_dim must be a positive integer, got {embed_dim!r}")
        if (not isinstance(action_dim, int) or isinstance(action_dim, bool)
                or action_dim < 1):
            raise ValueError(
                f"SIRO action_dim must be a positive integer, got {action_dim!r}")
        if mode not in self.MODES:
            raise ValueError(f"unknown SIRO mode {mode!r}; expected one of {sorted(self.MODES)}")

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.mode = mode

        # Every fitted tensor is a persistent, non-gradient buffer.  Identity/zero
        # initialization makes the unfitted cell a well-defined raw-observation tracker.
        self.register_buffer("identified_A", torch.eye(embed_dim))
        self.register_buffer("action_B", torch.zeros(embed_dim, action_dim))
        self.register_buffer("drift_b", torch.zeros(embed_dim))
        self.register_buffer("action_read_R", torch.eye(embed_dim))
        self.register_buffer("lmmse_K", torch.eye(embed_dim))
        self.register_buffer("clean_innovation_mean", torch.zeros(embed_dim))
        self.register_buffer("observed_innovation_mean", torch.zeros(embed_dim))

        self.register_buffer("fit_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("operators_installed", torch.zeros((), dtype=torch.bool))

        # Standard numerical receipts are recomputed from the exact installed tensors.
        # Arbitrary fit-specific receipts supplied by the trainer are serialized through
        # get_extra_state, keeping the tensor schema identical across every intervention.
        self.register_buffer("receipt_A_spectral_radius", torch.ones(()))
        self.register_buffer("receipt_A_operator_norm", torch.ones(()))
        self.register_buffer("receipt_B_frobenius_norm", torch.zeros(()))
        self.register_buffer("receipt_K_operator_norm", torch.ones(()))
        self.register_buffer("receipt_R_operator_norm", torch.ones(()))
        self.register_buffer("receipt_R_identity_error", torch.zeros(()))
        self.register_buffer("receipt_install_max_abs", torch.ones(()))
        self._fit_receipts: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Counts and fitted-state installation
    # ------------------------------------------------------------------
    @classmethod
    def expected_parameter_count(cls, embed_dim: int, action_dim: int) -> int:
        """SIRO stores no gradient-trained memory scalar."""
        del embed_dim, action_dim
        return 0

    @classmethod
    def expected_fitted_scalar_count(cls, embed_dim: int, action_dim: int) -> int:
        """Common persistent fitted tensors: ``3D^2 + AD + 3D`` scalars."""
        return 3 * embed_dim * embed_dim + embed_dim * action_dim + 3 * embed_dim

    @classmethod
    def expected_total_scalar_count(cls, embed_dim: int, action_dim: int) -> int:
        """Gradient plus fitted operators; equal to the fitted count for SIRO."""
        return cls.expected_fitted_scalar_count(embed_dim, action_dim)

    @classmethod
    def expected_deployed_scalar_count(
            cls, embed_dim: int, action_dim: int, mode: str = "full") -> int:
        """Return the number of functionally active fitted scalars for ``mode``.

        This is distinct from checkpoint storage: every mode stores the common fitted
        schema returned by :meth:`expected_fitted_scalar_count`.
        """
        if mode not in cls.MODES:
            raise ValueError(f"unknown SIRO mode {mode!r}")
        d2 = embed_dim * embed_dim
        da = embed_dim * action_dim
        # A, B, b, K, and two innovation means are active in full SIRO.
        active = 2 * d2 + da + 3 * embed_dim
        if mode == "spectralshrink":
            active += d2
        elif mode == "identityA":
            active -= d2
        elif mode == "identityK":
            active -= d2 + 2 * embed_dim
        elif mode == "noaction":
            active -= da
        return active

    @classmethod
    def expected_streaming_scalar_count(cls, embed_dim: int) -> int:
        """The persistent online state is exactly ``3D`` floats."""
        return 3 * embed_dim

    def parameter_count(self) -> int:
        """SIRO has no gradient-trained memory parameter."""
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)

    def fitted_scalar_count(self) -> int:
        """Return the common fitted tensor count for this instance."""
        return self.expected_fitted_scalar_count(self.embed_dim, self.action_dim)

    @staticmethod
    def _coerce_fitted_tensor(
            value: torch.Tensor, shape: Tuple[int, ...], name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            actual = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise ValueError(f"{name} must have shape {shape}, got {actual}")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite floating-point values")
        return value.detach()

    @staticmethod
    def _symmetric_psd_tolerance(value: torch.Tensor) -> float:
        dtype = value.dtype if value.dtype in (torch.float32, torch.float64) else torch.float32
        scale = max(1.0, float(value.detach().double().abs().max()))
        return max(value.shape) * torch.finfo(dtype).eps * scale * 4.0

    @classmethod
    def _validate_symmetric_psd(cls, value: torch.Tensor, name: str) -> None:
        tolerance = cls._symmetric_psd_tolerance(value)
        work = value.detach().double()
        symmetry_error = float((work - work.T).abs().max())
        if symmetry_error > tolerance:
            raise ValueError(
                f"{name} must be symmetric up to numerical tolerance; "
                f"error={symmetry_error:.3e}, tolerance={tolerance:.3e}")
        minimum = float(torch.linalg.eigvalsh(0.5 * (work + work.T)).min())
        if minimum < -tolerance:
            raise ValueError(
                f"{name} must be positive semidefinite; min_eigenvalue={minimum:.3e}")

    @torch.no_grad()
    def install_fitted_operators(
            self, *, identified_A: torch.Tensor, action_B: torch.Tensor,
            drift_b: torch.Tensor, action_read_R: torch.Tensor,
            lmmse_K: torch.Tensor, clean_innovation_mean: torch.Tensor,
            observed_innovation_mean: torch.Tensor,
            receipts: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        """Atomically install one detached SIRO fit.

        Tensor names and shapes are the trainer-facing frozen API.  ``action_B`` is
        expressed in native environment-action coordinates.  ``action_read_R`` is
        fitted and serialized for every mode but is functionally used only by
        ``spectralshrink``.  Likewise all interventions retain the fitted A/B/K tensors.
        """
        D, A = self.embed_dim, self.action_dim
        candidates = {
            "identified_A": self._coerce_fitted_tensor(
                identified_A, (D, D), "identified_A"),
            "action_B": self._coerce_fitted_tensor(action_B, (D, A), "action_B"),
            "drift_b": self._coerce_fitted_tensor(drift_b, (D,), "drift_b"),
            "action_read_R": self._coerce_fitted_tensor(
                action_read_R, (D, D), "action_read_R"),
            "lmmse_K": self._coerce_fitted_tensor(lmmse_K, (D, D), "lmmse_K"),
            "clean_innovation_mean": self._coerce_fitted_tensor(
                clean_innovation_mean, (D,), "clean_innovation_mean"),
            "observed_innovation_mean": self._coerce_fitted_tensor(
                observed_innovation_mean, (D,), "observed_innovation_mean"),
        }

        # A fitted transition must be stable/nonexpansive.  The identity-A intervention is
        # applied only after installation and does not relax the fitted-operator receipt.
        A_double = candidates["identified_A"].double()
        spectral_radius = float(torch.linalg.eigvals(A_double).abs().max())
        stability_tolerance = (
            D * torch.finfo(candidates["identified_A"].dtype).eps
            * max(1.0, float(A_double.abs().max())) * 8.0)
        if spectral_radius > 1.0 + stability_tolerance:
            raise ValueError(
                "identified_A must be spectrally stable/nonexpansive; "
                f"radius={spectral_radius:.8f}, tolerance={stability_tolerance:.3e}")

        if receipts is not None and not isinstance(receipts, Mapping):
            raise ValueError("receipts must be a mapping or None")

        # All validation precedes the first copy, making a rejected installation atomic.
        for name, value in candidates.items():
            destination = getattr(self, name)
            destination.copy_(value.to(device=destination.device, dtype=destination.dtype))

        self.fit_updates.add_(1)
        self.operators_installed.fill_(True)
        self.receipt_A_spectral_radius.fill_(spectral_radius)
        self.receipt_A_operator_norm.fill_(
            float(torch.linalg.matrix_norm(A_double, ord=2)))
        self.receipt_B_frobenius_norm.fill_(
            float(torch.linalg.vector_norm(candidates["action_B"].double())))
        self.receipt_K_operator_norm.fill_(
            float(torch.linalg.matrix_norm(candidates["lmmse_K"].double(), ord=2)))
        self.receipt_R_operator_norm.fill_(
            float(torch.linalg.matrix_norm(candidates["action_read_R"].double(), ord=2)))
        identity = torch.eye(D, dtype=torch.float64, device=A_double.device)
        self.receipt_R_identity_error.fill_(
            float((candidates["action_read_R"].double() - identity).abs().max()))
        self.receipt_install_max_abs.fill_(max(
            float(value.double().abs().max()) for value in candidates.values()))
        self._fit_receipts = copy.deepcopy(dict(receipts or {}))
        return self.operator_diagnostics()

    def get_extra_state(self) -> Dict[str, Any]:
        """Serialize fit-specific receipts without changing the common tensor schema."""
        return {"fit_receipts": copy.deepcopy(self._fit_receipts)}

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise RuntimeError("invalid SIRO extra state")
        receipts = state.get("fit_receipts", {})
        if not isinstance(receipts, Mapping):
            raise RuntimeError("invalid SIRO fit receipts in extra state")
        self._fit_receipts = copy.deepcopy(dict(receipts))

    # ------------------------------------------------------------------
    # Effective interventions and validation
    # ------------------------------------------------------------------
    def effective_A(self, reference: torch.Tensor | None = None) -> torch.Tensor:
        base = self.identified_A
        if self.mode == "identityA":
            base = torch.eye(self.embed_dim, device=base.device, dtype=base.dtype)
        if reference is not None:
            base = base.to(device=reference.device, dtype=reference.dtype)
        return base

    def effective_B(self, reference: torch.Tensor | None = None) -> torch.Tensor:
        base = (torch.zeros_like(self.action_B)
                if self.mode == "noaction" else self.action_B)
        if reference is not None:
            base = base.to(device=reference.device, dtype=reference.dtype)
        return base

    def effective_R(self, reference: torch.Tensor | None = None) -> torch.Tensor:
        base = self.action_read_R
        if self.mode != "spectralshrink":
            base = torch.eye(self.embed_dim, device=base.device, dtype=base.dtype)
        if reference is not None:
            base = base.to(device=reference.device, dtype=reference.dtype)
        return base

    def _validate_state(self, state: torch.Tensor) -> int:
        expected_tail = (3, self.embed_dim)
        if (not isinstance(state, torch.Tensor) or state.dim() != 3
                or state.shape[0] < 1 or tuple(state.shape[1:]) != expected_tail):
            actual = tuple(state.shape) if isinstance(state, torch.Tensor) else type(state).__name__
            raise ValueError(f"state must have shape (B,3,{self.embed_dim}), got {actual}")
        if not state.is_floating_point() or not torch.isfinite(state).all():
            raise ValueError("state must contain finite floating-point values")
        return state.shape[0]

    def _validate_latents(self, z: torch.Tensor) -> Tuple[int, int, int]:
        if not isinstance(z, torch.Tensor) or z.dim() != 3:
            actual = tuple(z.shape) if isinstance(z, torch.Tensor) else type(z).__name__
            raise ValueError(f"SIRO expects z with shape (B,T,D), got {actual}")
        B, T, D = z.shape
        if B < 1 or T < 1 or D != self.embed_dim:
            raise ValueError(
                f"SIRO expected non-empty (B,T,{self.embed_dim}), got {tuple(z.shape)}")
        if not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError("z must contain finite floating-point values")
        return B, T, D

    def _validate_actions(
            self, actions: torch.Tensor, batch: int, steps: int,
            *, name: str = "actions") -> torch.Tensor:
        expected = (batch, steps, self.action_dim)
        if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != expected:
            actual = tuple(actions.shape) if isinstance(actions, torch.Tensor) else type(actions).__name__
            raise ValueError(f"{name} must have shape {expected}, got {actual}")
        if not actions.is_floating_point() or not torch.isfinite(actions).all():
            raise ValueError(f"{name} must contain finite floating-point values")
        return actions

    @staticmethod
    def _coerce_action_override(value: Any, actions: torch.Tensor) -> torch.Tensor:
        override = torch.as_tensor(value, device=actions.device, dtype=actions.dtype)
        try:
            override = torch.broadcast_to(override, actions.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"action_override with shape {tuple(override.shape)} is not "
                f"broadcastable to {tuple(actions.shape)}") from exc
        if not torch.isfinite(override).all():
            raise ValueError("action_override contains non-finite values")
        return override

    # ------------------------------------------------------------------
    # Online recurrence
    # ------------------------------------------------------------------
    def initial_state(self, z0: torch.Tensor) -> torch.Tensor:
        if (not isinstance(z0, torch.Tensor) or z0.dim() != 2
                or z0.shape[0] < 1 or z0.shape[1] != self.embed_dim):
            actual = tuple(z0.shape) if isinstance(z0, torch.Tensor) else type(z0).__name__
            raise ValueError(
                f"z0 must have shape (B,{self.embed_dim}), got {actual}")
        if not z0.is_floating_point() or not torch.isfinite(z0).all():
            raise ValueError("z0 must contain finite floating-point values")
        zero = torch.zeros_like(z0)
        if self.mode == "noanchor":
            return torch.stack((zero, z0, zero), dim=1)
        return torch.stack((z0, zero, zero), dim=1)

    def _read_unchecked(self, state: torch.Tensor) -> torch.Tensor:
        c, r, u = state[..., 0, :], state[..., 1, :], state[..., 2, :]
        R = self.effective_R(r)
        return c + r + F.linear(u, R)

    def read_state(self, state: torch.Tensor) -> torch.Tensor:
        if (not isinstance(state, torch.Tensor) or state.dim() < 2
                or tuple(state.shape[-2:]) != (3, self.embed_dim)):
            actual = tuple(state.shape) if isinstance(state, torch.Tensor) else type(state).__name__
            raise ValueError(
                f"state read requires (...,3,{self.embed_dim}), got {actual}")
        if not state.is_floating_point() or not torch.isfinite(state).all():
            raise ValueError("state read requires finite floating-point values")
        return self._read_unchecked(state)

    def _transition_unchecked(
            self, state: torch.Tensor, action: torch.Tensor
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        c, r, u = state[:, 0], state[:, 1], state[:, 2]
        A = self.effective_A(r)
        B = self.effective_B(r)
        b = self.drift_b.to(device=r.device, dtype=r.dtype)
        # No arithmetic is applied to c: cloning/stacking preserves the anchor bits exactly.
        c_prior = c
        r_prior = F.linear(r, A) + b
        effective_action = (
            torch.zeros_like(action) if self.mode == "noaction" else action)
        action_effect = F.linear(effective_action, B)
        u_prior = F.linear(u, A) + action_effect
        prior = torch.stack((c_prior, r_prior, u_prior), dim=1)
        return prior, {
            "c_prior": c_prior,
            "r_prior": r_prior,
            "u_prior": u_prior,
            "prior_read": self._read_unchecked(prior),
            "effective_action": effective_action,
            "action_effect": action_effect,
            "action_effect_norm": action_effect.norm(dim=-1),
        }

    def transition(
            self, state: torch.Tensor, action: torch.Tensor,
            action_override: Any = None, return_details: bool = False):
        batch = self._validate_state(state)
        if (not isinstance(action, torch.Tensor)
                or tuple(action.shape) != (batch, self.action_dim)
                or not action.is_floating_point() or not torch.isfinite(action).all()):
            actual = tuple(action.shape) if isinstance(action, torch.Tensor) else type(action).__name__
            raise ValueError(
                f"action must be finite floating point with shape "
                f"{(batch, self.action_dim)}, got {actual}")
        action = action.to(device=state.device, dtype=state.dtype)
        if action_override is not None:
            action = self._coerce_action_override(action_override, action)
        # The no-action intervention is applied by B_eff=0, not by changing the
        # serialized action or fitted B tensor.
        prior, details = self._transition_unchecked(state, action)
        return (prior, details) if return_details else prior

    def _correction_unchecked(
            self, prior: torch.Tensor, z_t: torch.Tensor
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        prior_read = self._read_unchecked(prior)
        innovation = z_t - prior_read
        if self.mode == "identityK":
            correction = innovation
        else:
            K = self.lmmse_K.to(device=innovation.device, dtype=innovation.dtype)
            mu_c = self.clean_innovation_mean.to(
                device=innovation.device, dtype=innovation.dtype)
            mu_o = self.observed_innovation_mean.to(
                device=innovation.device, dtype=innovation.dtype)
            correction = mu_c + F.linear(innovation - mu_o, K)
        state = prior.clone()
        state[:, 1] = state[:, 1] + correction
        posterior_read = self._read_unchecked(state)
        return state, {
            "innovation": innovation,
            "correction": correction,
            "correction_norm": correction.norm(dim=-1),
            "posterior_read": posterior_read,
        }

    def step(
            self, state: torch.Tensor, z_t: torch.Tensor, action: torch.Tensor,
            action_override: Any = None, return_details: bool = False):
        batch = self._validate_state(state)
        if (not isinstance(z_t, torch.Tensor)
                or tuple(z_t.shape) != (batch, self.embed_dim)
                or not z_t.is_floating_point() or not torch.isfinite(z_t).all()):
            actual = tuple(z_t.shape) if isinstance(z_t, torch.Tensor) else type(z_t).__name__
            raise ValueError(
                f"z_t must be finite floating point with shape "
                f"{(batch, self.embed_dim)}, got {actual}")
        z_t = z_t.to(device=state.device, dtype=state.dtype)
        prior, transition_details = self.transition(
            state, action, action_override=action_override, return_details=True)
        new_state, correction_details = self._correction_unchecked(prior, z_t)
        mixed = correction_details["posterior_read"]
        if not return_details:
            return mixed, new_state
        return mixed, new_state, {
            "x": z_t,
            "prior": prior,
            "state": new_state,
            "c_state": new_state[:, 0],
            "r_state": new_state[:, 1],
            "u_state": new_state[:, 2],
            **transition_details,
            **correction_details,
        }

    def rollout_transition(
            self, initial_state: torch.Tensor, actions: torch.Tensor,
            action_override: Any = None, return_details: bool = False):
        batch = self._validate_state(initial_state)
        if not isinstance(actions, torch.Tensor) or actions.dim() != 3:
            actual = tuple(actions.shape) if isinstance(actions, torch.Tensor) else type(actions).__name__
            raise ValueError(f"rollout actions must have shape (B,H,A), got {actual}")
        actions = self._validate_actions(actions, batch, actions.shape[1])
        actions = actions.to(device=initial_state.device, dtype=initial_state.dtype)
        if action_override is not None:
            actions = self._coerce_action_override(action_override, actions)
        state = initial_state
        states = []
        detail_lists: Dict[str, list[torch.Tensor]] = {}
        for index in range(actions.shape[1]):
            state, step_details = self._transition_unchecked(state, actions[:, index])
            states.append(state)
            if return_details:
                for key, value in step_details.items():
                    detail_lists.setdefault(key, []).append(value)
        sequence = (torch.stack(states, dim=1) if states else
                    initial_state.new_empty(batch, 0, 3, self.embed_dim))
        if not return_details:
            return sequence
        details = {
            key: torch.stack(values, dim=1)
            for key, values in detail_lists.items()
        }
        return sequence, details

    def forward(
            self, z: torch.Tensor, actions: torch.Tensor,
            action_override: Any = None, return_details: bool = False):
        batch, length, _ = self._validate_latents(z)
        actions = self._validate_actions(actions, batch, length - 1)
        actions = actions.to(device=z.device, dtype=z.dtype)
        if action_override is not None:
            actions = self._coerce_action_override(action_override, actions)

        state = self.initial_state(z[:, 0])
        states = [state]
        priors = [state]
        mixed = [self._read_unchecked(state)]
        zero = torch.zeros_like(z[:, 0])
        innovations = [zero]
        corrections = [zero]
        action_effects = [zero]
        effective_actions = []

        for index in range(1, length):
            prior, transition_details = self._transition_unchecked(
                state, actions[:, index - 1])
            state, correction_details = self._correction_unchecked(prior, z[:, index])
            priors.append(prior)
            states.append(state)
            mixed.append(correction_details["posterior_read"])
            innovations.append(correction_details["innovation"])
            corrections.append(correction_details["correction"])
            action_effects.append(transition_details["action_effect"])
            effective_actions.append(transition_details["effective_action"])

        mixed_sequence = torch.stack(mixed, dim=1)
        if not return_details:
            return mixed_sequence
        state_sequence = torch.stack(states, dim=1)
        prior_sequence = torch.stack(priors, dim=1)
        return mixed_sequence, {
            "x": z,
            "states": state_sequence,
            "priors": prior_sequence,
            "c_states": state_sequence[:, :, 0],
            "r_states": state_sequence[:, :, 1],
            "u_states": state_sequence[:, :, 2],
            "c_priors": prior_sequence[:, :, 0],
            "r_priors": prior_sequence[:, :, 1],
            "u_priors": prior_sequence[:, :, 2],
            "reads": mixed_sequence,
            "prior_reads": self._read_unchecked(prior_sequence),
            "innovations": torch.stack(innovations, dim=1),
            "corrections": torch.stack(corrections, dim=1),
            "correction_norm": torch.stack(corrections, dim=1).norm(dim=-1),
            "action_effect": torch.stack(action_effects, dim=1),
            "action_effect_norm": torch.stack(action_effects, dim=1).norm(dim=-1),
            "effective_actions": (
                torch.stack(effective_actions, dim=1) if effective_actions else
                actions.new_empty(batch, 0, self.action_dim)),
        }

    def fuse(self, z: torch.Tensor, mixed: torch.Tensor) -> torch.Tensor:
        """SIRO's fitted posterior is the predictor input; there is no residual bypass."""
        if (not isinstance(z, torch.Tensor) or not isinstance(mixed, torch.Tensor)
                or z.dim() != 3 or mixed.dim() != 3 or z.shape != mixed.shape
                or z.shape[-1] != self.embed_dim):
            z_shape = tuple(z.shape) if isinstance(z, torch.Tensor) else type(z).__name__
            mixed_shape = (tuple(mixed.shape) if isinstance(mixed, torch.Tensor)
                           else type(mixed).__name__)
            raise ValueError(
                f"z and mixed must share shape (B,T,{self.embed_dim}), got "
                f"{z_shape} and {mixed_shape}")
        if (not z.is_floating_point() or not mixed.is_floating_point()
                or not torch.isfinite(z).all() or not torch.isfinite(mixed).all()):
            raise ValueError("z and mixed must contain finite floating-point values")
        return mixed

    # ------------------------------------------------------------------
    # Auditable receipts
    # ------------------------------------------------------------------
    @torch.no_grad()
    def operator_diagnostics(self) -> Dict[str, Any]:
        A = self.identified_A.double()
        B = self.action_B.double()
        K = self.lmmse_K.double()
        R = self.action_read_R.double()
        effective_A = self.effective_A().double()
        effective_B = self.effective_B().double()
        effective_R = self.effective_R().double()
        diagnostics: Dict[str, Any] = {
            "mode": self.mode,
            "operators_installed": bool(self.operators_installed),
            "fit_updates": int(self.fit_updates),
            "gradient_parameter_count": self.parameter_count(),
            "fitted_scalar_count": self.expected_fitted_scalar_count(
                self.embed_dim, self.action_dim),
            "deployed_active_scalar_count": self.expected_deployed_scalar_count(
                self.embed_dim, self.action_dim, self.mode),
            "streaming_scalar_count": self.expected_streaming_scalar_count(self.embed_dim),
            "identified_A_spectral_radius": float(torch.linalg.eigvals(A).abs().max()),
            "identified_A_operator_norm": float(torch.linalg.matrix_norm(A, ord=2)),
            "effective_A_identity_error": float((
                effective_A - torch.eye(self.embed_dim, dtype=torch.float64,
                                        device=effective_A.device)).abs().max()),
            "action_B_frobenius_norm": float(torch.linalg.vector_norm(B)),
            "effective_B_frobenius_norm": float(torch.linalg.vector_norm(effective_B)),
            "lmmse_K_operator_norm": float(torch.linalg.matrix_norm(K, ord=2)),
            "action_read_R_operator_norm": float(torch.linalg.matrix_norm(R, ord=2)),
            "effective_R_operator_norm": float(torch.linalg.matrix_norm(effective_R, ord=2)),
            "action_read_R_identity_error": float((
                R - torch.eye(self.embed_dim, dtype=torch.float64,
                              device=R.device)).abs().max()),
            "identityA_active": self.mode == "identityA",
            "identityK_active": self.mode == "identityK",
            "noaction_active": self.mode == "noaction",
            "noanchor_active": self.mode == "noanchor",
            "spectral_read_active": self.mode == "spectralshrink",
        }
        diagnostics.update(copy.deepcopy(self._fit_receipts))
        return diagnostics

    def horizons(self) -> Dict[str, float]:
        """Uniform numeric summary; SIRO has no prescribed discrete horizons."""
        with torch.no_grad():
            return {
                "tau_fast": float("inf"),
                "tau_slow": float("inf"),
                "n_banks": 0,
                "recurrent_floats": self.expected_streaming_scalar_count(self.embed_dim),
                "identified_A_spectral_radius": float(
                    torch.linalg.eigvals(self.identified_A.double()).abs().max()),
                "action_B_norm": float(self.action_B.double().norm()),
                "lmmse_K_norm": float(self.lmmse_K.double().norm()),
                "spectral_read_active": float(self.mode == "spectralshrink"),
                "identity_A_active": float(self.mode == "identityA"),
                "identity_K_active": float(self.mode == "identityK"),
                "action_active": float(self.mode != "noaction"),
                "anchor_active": float(self.mode != "noanchor"),
            }


SIROv12Memory = StableIdentifiedResidualObserverMemory


__all__ = ["StableIdentifiedResidualObserverMemory", "SIROv12Memory"]
