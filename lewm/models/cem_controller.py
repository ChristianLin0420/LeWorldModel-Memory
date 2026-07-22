#!/usr/bin/env python3
"""Causal-Effect Memory (CEM) controller for a FROZEN world-model host.

This module implements the trainable *controller* half of the CEM paradigm
described in ``docs/EMERGENT_CUE_MEMORY_PARADIGM.md``.  Everything below the
host encoder/predictor stays frozen; only the pieces here train.

One scalar governs all three decisions:

    CE(m) = increase in the FROZEN host's own future-latent rollout loss when
            memory item ``m`` is deleted (a ``do``-operation).

The three decisions are three estimators of that scalar:

  * **WRITE** -- 1-step, zero-cost estimator.  A host-surprise gate admits a
    latent iff the host's own instantaneous prediction error exceeds a
    self-calibrating quantile of its running error stream.  Label-free,
    saliency-free (a colour the host already predicts has ~0 reducible
    surprise and is never written).
  * **KEEP** -- amortized multi-step estimator.  A small head ``ce_head``
    predicts CE and eviction drops the lowest predicted CE.  ``ce_head`` is
    distilled against periodic *true* hard-deletion CE on a sampled subset.
  * **RECALL** -- bounded verified estimator.  A router prefilters candidates
    by content + a learned age-kernel + predicted-need, then retrieve-then-
    verify accepts a key only if a real ``do`` lowers host rollout loss by
    ``delta``.

The controller is deliberately host-agnostic: it operates on pooled latent
*vectors* for its WRITE/KEEP/RECALL scoring, while the host-specific residual
injection (which may be spatial) is supplied by the caller.  The two host
runners (``run_cem_lewm_pusht.py``, ``run_cem_dinowm.py``) wire it up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# WRITE: self-calibrating host-surprise gate.
# --------------------------------------------------------------------------- #
class SurpriseWriteGate(nn.Module):
    """Label-free write gate driven by the frozen host's own surprise.

    ``surprise`` is the host's instantaneous next-latent prediction error
    ``s_t = ||host.predict(z_{<t}) - z_t||^2`` (a free byproduct of the host
    forward).  A latent is *admissible* iff ``s_t`` exceeds a running quantile
    ``Q_rho`` of the host's own error stream.  The threshold is maintained as
    an EMA of the empirical batch quantile, so it self-calibrates and needs no
    labels and no saliency heuristic.

    A soft (differentiable) gate ``sigmoid((s - tau) / temp)`` is also returned
    for the write-cost term ``beta * E[gate]``.
    """

    def __init__(self, quantile: float = 0.8, temperature: float = 0.25,
                 ema: float = 0.05) -> None:
        super().__init__()
        self.quantile = float(quantile)
        self.ema = float(ema)
        # ``temp`` is a positive, learnable softness for the differentiable gate.
        self.log_temp = nn.Parameter(torch.tensor(float(temperature)).log())
        self.register_buffer("threshold", torch.tensor(float("nan")))
        self.register_buffer("scale", torch.tensor(1.0))

    @torch.no_grad()
    def update_threshold(self, surprise: torch.Tensor) -> torch.Tensor:
        flat = surprise.detach().reshape(-1).float()
        if flat.numel() == 0:
            return self.threshold
        batch_q = torch.quantile(flat, self.quantile)
        batch_scale = flat.std().clamp_min(1e-6)
        if torch.isnan(self.threshold):
            self.threshold = batch_q
            self.scale = batch_scale
        else:
            self.threshold = (1.0 - self.ema) * self.threshold + self.ema * batch_q
            self.scale = (1.0 - self.ema) * self.scale + self.ema * batch_scale
        return self.threshold

    def soft_gate(self, surprise: torch.Tensor) -> torch.Tensor:
        thr = self.threshold
        if torch.isnan(thr):
            thr = torch.quantile(surprise.detach().reshape(-1).float(),
                                 self.quantile)
        temp = self.log_temp.exp().clamp_min(1e-3)
        return torch.sigmoid((surprise - thr) / (self.scale * temp))

    @torch.no_grad()
    def hard_gate(self, surprise: torch.Tensor) -> torch.Tensor:
        thr = self.threshold
        if torch.isnan(thr):
            thr = torch.quantile(surprise.detach().reshape(-1).float(),
                                 self.quantile)
        return (surprise > thr).float()


# --------------------------------------------------------------------------- #
# KEEP: amortized CE head + learned age kernel.
# --------------------------------------------------------------------------- #
class CEHead(nn.Module):
    """Amortized regressor of the true hard-deletion CE (necessity)."""

    def __init__(self, latent_dim: int, hidden: int = 128) -> None:
        super().__init__()
        # inputs: [pooled slot repr | age_norm | surprise_at_write]
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim + 2),
            nn.Linear(latent_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, slot_repr: torch.Tensor, age_norm: torch.Tensor,
                surprise: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([
            slot_repr,
            age_norm.reshape(*age_norm.shape, 1) if age_norm.dim() == slot_repr.dim() - 1 else age_norm,
            surprise.reshape(*surprise.shape, 1) if surprise.dim() == slot_repr.dim() - 1 else surprise,
        ], dim=-1)
        return self.net(feats).squeeze(-1)


class AgeKernel(nn.Module):
    """Learned age-kernel ``phi(tau_now - tau_i)`` over normalized ages.

    Fed a small featurization of the (normalized) age so the router can learn a
    non-trivial, non-monotone preference rather than collapsing to recency.
    """

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, age_norm: torch.Tensor) -> torch.Tensor:
        feats = torch.stack([
            age_norm,
            torch.sin(age_norm * 3.14159),
            torch.cos(age_norm * 3.14159),
        ], dim=-1)
        return self.net(feats).squeeze(-1)


class Router(nn.Module):
    """Correlation prefilter: content match + learned age-kernel + need."""

    def __init__(self, latent_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.key_proj = nn.Linear(latent_dim, hidden)
        self.query_proj = nn.Linear(latent_dim, hidden)
        self.age_kernel = AgeKernel()
        self.need = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def score(self, query: torch.Tensor, keys: torch.Tensor,
              age_norm: torch.Tensor) -> torch.Tensor:
        # query: (B, D)  keys: (B, S, D)  age_norm: (B, S)
        q = F.normalize(self.query_proj(query), dim=-1)
        k = F.normalize(self.key_proj(keys), dim=-1)
        content = torch.einsum("bd,bsd->bs", q, k)
        age = self.age_kernel(age_norm)
        need = self.need(keys).squeeze(-1)
        return content + age + need


@dataclass
class MemoryStore:
    """Bounded, timestamped slot store (append-only with capacity eviction).

    Holds a python-side ledger for logging (WRITE/KEEP/RECALL decisions); the
    actual differentiable slot tensors are re-derived from the host prefix each
    forward pass by the residual writer.  This ledger produces the shared
    ``decision_log.json`` events.
    """

    budget: int
    events: list = field(default_factory=list)

    def write(self, slot_id: int, written_at: int, cue_timestamp: float,
              surprise_at_write: float, ce_hat: float) -> None:
        self.events.append({
            "slot_id": int(slot_id),
            "written_at": int(written_at),
            "cue_timestamp": float(cue_timestamp),
            "surprise_at_write": float(surprise_at_write),
            "ce_hat": float(ce_hat),
            "ce_true": None,
            "status": "kept",
            "evicted_at": None,
            "retrieved_at": None,
            "verify_delta": None,
        })

    def apply_budget(self, evict_at: int) -> None:
        kept = [e for e in self.events if e["status"] == "kept"]
        if len(kept) <= self.budget:
            return
        order = sorted(kept, key=lambda e: e["ce_hat"])
        for e in order[: len(kept) - self.budget]:
            e["status"] = "evicted"
            e["evicted_at"] = int(evict_at)


@dataclass
class EventRecord:
    """One generic event version in a bounded CEM store."""

    event_id: int
    key_id: int
    event_timestamp: int
    proposed_at: int
    verified_at: int | None = None
    ce_hat: float = 0.0
    status: str = "provisional"
    supersedes: int | None = None
    superseded_by: int | None = None
    rejected_reason: str | None = None
    evicted_at: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": int(self.event_id),
            "key_id": int(self.key_id),
            "event_timestamp": int(self.event_timestamp),
            "proposed_at": int(self.proposed_at),
            "verified_at": (
                None if self.verified_at is None else int(self.verified_at)
            ),
            "ce_hat": float(self.ce_hat),
            "status": self.status,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "rejected_reason": self.rejected_reason,
            "evicted_at": (
                None if self.evicted_at is None else int(self.evicted_at)
            ),
        }


@dataclass
class VersionedEventStore:
    """Bounded event store that preserves an old version until verification.

    Semantic keys are supplied by the caller. A provisional replacement cannot
    remove an active version. Promotion requires a positive threshold and, for
    a same-key replacement, the configured hysteresis margin.
    """

    budget: int
    hysteresis: float = 0.0
    records: list[EventRecord] = field(default_factory=list)

    def propose(
        self,
        *,
        event_id: int,
        key_id: int,
        event_timestamp: int,
        proposed_at: int,
    ) -> EventRecord:
        record = EventRecord(
            event_id=int(event_id),
            key_id=int(key_id),
            event_timestamp=int(event_timestamp),
            proposed_at=int(proposed_at),
        )
        self.records.append(record)
        return record

    def _record(self, event_id: int) -> EventRecord:
        for record in self.records:
            if record.event_id == int(event_id):
                return record
        raise KeyError(f"unknown event_id={event_id}")

    def active(self) -> list[EventRecord]:
        return [record for record in self.records if record.status == "active"]

    def active_for_key(self, key_id: int) -> EventRecord | None:
        matches = [
            record for record in self.active() if record.key_id == int(key_id)
        ]
        if not matches:
            return None
        return max(matches, key=lambda record: record.event_timestamp)

    def verify(
        self,
        *,
        event_id: int,
        verified_at: int,
        ce_hat: float,
        threshold: float,
    ) -> dict[str, Any]:
        record = self._record(event_id)
        if record.status != "provisional":
            raise ValueError(
                f"event {event_id} is {record.status}, not provisional"
            )
        record.verified_at = int(verified_at)
        record.ce_hat = float(ce_hat)
        previous = self.active_for_key(record.key_id)
        required = float(threshold)
        if previous is not None:
            required = max(required, previous.ce_hat + float(self.hysteresis))
        if record.ce_hat <= required:
            record.status = "rejected"
            record.rejected_reason = (
                "below_hysteresis" if previous is not None
                else "below_promotion_threshold"
            )
            return {
                "transition": "rejected",
                "event_id": record.event_id,
                "fallback_event_id": (
                    None if previous is None else previous.event_id
                ),
                "required_ce_hat": required,
            }
        record.status = "active"
        if previous is not None:
            previous.status = "superseded"
            previous.superseded_by = record.event_id
            record.supersedes = previous.event_id
        evicted = self.apply_budget(evict_at=verified_at)
        return {
            "transition": "promoted",
            "event_id": record.event_id,
            "superseded_event_id": (
                None if previous is None else previous.event_id
            ),
            "evicted_event_ids": evicted,
            "required_ce_hat": required,
        }

    def apply_budget(self, *, evict_at: int) -> list[int]:
        active = self.active()
        overflow = max(0, len(active) - int(self.budget))
        if overflow == 0:
            return []
        evicted = sorted(
            active, key=lambda record: (record.ce_hat, record.event_timestamp)
        )[:overflow]
        for record in evicted:
            record.status = "evicted"
            record.evicted_at = int(evict_at)
        return [record.event_id for record in evicted]

    def snapshot(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self.records]


class CEMController(nn.Module):
    """The trainable CEM controller (host-agnostic scoring half).

    Bundles the shared trainable pieces -- surprise WRITE gate, amortized
    ``ce_head`` (KEEP), and router (RECALL) -- plus the free-energy write cost
    and the distillation loss that teaches ``ce_head`` to match true CE.

    The host-specific residual writer/injector is *not* owned here; the runner
    passes it in and drives ``inject``.  This keeps the controller reusable
    across LeWM (192-d vector latents) and DINO-WM (196x384 spatial latents).
    """

    def __init__(self, *, latent_dim: int, budget: int = 4,
                 quantile: float = 0.8, beta: float = 1.0e-2,
                 verify_delta: float = 0.0, ce_hidden: int = 128) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.budget = int(budget)
        self.beta = float(beta)
        self.verify_delta = float(verify_delta)
        self.write_gate = SurpriseWriteGate(quantile=quantile)
        self.ce_head = CEHead(latent_dim, hidden=ce_hidden)
        self.router = Router(latent_dim, hidden=ce_hidden)

    # -- WRITE ------------------------------------------------------------- #
    def write_cost(self, surprise: torch.Tensor) -> torch.Tensor:
        """Free-energy write cost ``beta * E[gate]`` (description length)."""
        return self.beta * self.write_gate.soft_gate(surprise).mean()

    # -- KEEP -------------------------------------------------------------- #
    def ce_hat(self, slot_repr: torch.Tensor, age_norm: torch.Tensor,
               surprise: torch.Tensor) -> torch.Tensor:
        return self.ce_head(slot_repr, age_norm, surprise)

    @staticmethod
    def distillation_loss(ce_hat: torch.Tensor,
                          ce_true: torch.Tensor) -> torch.Tensor:
        """Calibrate CE values and ordering against normalized deletion effects.

        Smooth-L1 supplies scale calibration while the all-pairs logistic term
        directly trains the KEEP ordering. Tied true effects are excluded.
        """
        target = ce_true.detach()
        regression = F.smooth_l1_loss(ce_hat, target)
        true_delta = target.unsqueeze(-1) - target.unsqueeze(-2)
        hat_delta = ce_hat.unsqueeze(-1) - ce_hat.unsqueeze(-2)
        mask = true_delta.abs() > 1e-6
        if not bool(mask.any()):
            return regression
        signs = true_delta.sign()
        ranking = F.softplus(-signs * hat_delta)[mask].mean()
        return regression + ranking

    # -- RECALL ------------------------------------------------------------ #
    def route(self, query: torch.Tensor, keys: torch.Tensor,
              age_norm: torch.Tensor) -> torch.Tensor:
        return self.router.score(query, keys, age_norm)

    def accept(self, verify_delta: torch.Tensor) -> torch.Tensor:
        """Retrieve-then-verify: accept iff a real ``do`` lowers host loss."""
        return (verify_delta > self.verify_delta).float()


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation used to report surrogate fidelity."""
    a = a.detach().reshape(-1).float()
    b = b.detach().reshape(-1).float()
    if a.numel() < 3:
        return float("nan")
    ar = a.argsort().argsort().float()
    br = b.argsort().argsort().float()
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = (ar.norm() * br.norm()).clamp_min(1e-8)
    return float((ar @ br) / denom)


__all__ = [
    "SurpriseWriteGate",
    "CEHead",
    "AgeKernel",
    "Router",
    "MemoryStore",
    "EventRecord",
    "VersionedEventStore",
    "CEMController",
    "spearman",
]
