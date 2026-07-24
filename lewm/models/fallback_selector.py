"""Flat frame+event fallback selector with calibrated ensemble uncertainty."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


FRAME = 0
EVENT = 1
RECENT = 2
CANDIDATE_TYPE_NAMES = {
    FRAME: "frame",
    EVENT: "event",
    RECENT: "recent",
}


@dataclass
class FallbackCandidateBatch:
    """Candidate features for one occupied store/query batch."""

    latent: torch.Tensor
    query: torch.Tensor
    metadata: torch.Tensor
    candidate_type: torch.Tensor
    discovery_uncertainty: torch.Tensor
    router_score: torch.Tensor
    occupied: torch.Tensor
    valid: torch.Tensor

    def index(self, index: torch.Tensor) -> "FallbackCandidateBatch":
        return FallbackCandidateBatch(
            **{
                field: getattr(self, field)[index]
                for field in self.__dataclass_fields__
            }
        )

    def __len__(self) -> int:
        return int(self.latent.shape[0])


class CandidateTypeSelector(nn.Module):
    """Heteroscedastic conditional-effect head with candidate type."""

    def __init__(
        self,
        latent_dim: int,
        metadata_dim: int,
        hidden: int = 192,
        type_dim: int = 16,
    ) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(
            len(CANDIDATE_TYPE_NAMES), type_dim
        )
        input_dim = (
            4 * latent_dim
            + metadata_dim
            + type_dim
            + 4
        )
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden, 2)

    def forward(
        self,
        batch: FallbackCandidateBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = batch.latent.shape[1]
        query = batch.query[:, None].expand(-1, slots, -1)
        count = batch.occupied.sum(1, keepdim=True).clamp_min(1)
        store_mean = (
            batch.latent * batch.occupied.unsqueeze(-1)
        ).sum(1) / count
        store_mean = store_mean[:, None].expand_as(batch.latent)
        occupancy = (
            batch.occupied.sum(1, keepdim=True).float() / slots
        )[:, None].expand(-1, slots, -1)
        features = torch.cat(
            [
                batch.latent,
                query,
                store_mean,
                batch.latent - store_mean,
                batch.metadata,
                self.type_embedding(batch.candidate_type),
                batch.discovery_uncertainty.unsqueeze(-1),
                batch.router_score.unsqueeze(-1),
                batch.occupied.unsqueeze(-1).float(),
                occupancy,
            ],
            dim=-1,
        )
        output = self.output(self.encoder(features))
        return output[..., 0], output[..., 1].clamp(-8.0, 3.0)


class FallbackSelectorEnsemble(nn.Module):
    """Bootstrap ensemble combining epistemic and aleatoric uncertainty."""

    def __init__(
        self,
        latent_dim: int,
        metadata_dim: int,
        hidden: int = 192,
        members: int = 5,
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList(
            [
                CandidateTypeSelector(
                    latent_dim=latent_dim,
                    metadata_dim=metadata_dim,
                    hidden=hidden,
                )
                for _ in range(members)
            ]
        )
        self.register_buffer("uncertainty_scale", torch.tensor(1.0))

    def forward(
        self,
        batch: FallbackCandidateBatch,
    ) -> dict[str, torch.Tensor]:
        outputs = [member(batch) for member in self.members]
        means = torch.stack([output[0] for output in outputs], dim=0)
        log_variances = torch.stack(
            [output[1] for output in outputs], dim=0
        )
        mean = means.mean(0)
        epistemic_variance = means.var(0, unbiased=False)
        aleatoric_variance = torch.exp(log_variances).mean(0)
        variance = (
            epistemic_variance + aleatoric_variance
        ) * self.uncertainty_scale.square()
        return {
            "mean": mean,
            "std": variance.clamp_min(1e-12).sqrt(),
            "member_mean": means,
            "member_log_variance": log_variances,
            "epistemic_std": epistemic_variance.clamp_min(0).sqrt(),
            "aleatoric_std": aleatoric_variance.clamp_min(0).sqrt(),
        }

    def lower_confidence_bound(
        self,
        batch: FallbackCandidateBatch,
        z_value: float = 1.2815515655446004,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self(batch)
        return output["mean"] - z_value * output["std"], output

    @torch.no_grad()
    def calibrate_scale(
        self,
        batch: FallbackCandidateBatch,
        target: torch.Tensor,
        valid: torch.Tensor,
        target_coverage: float = 0.90,
    ) -> float:
        output = self(batch)
        standardized = (
            (target - output["mean"]).abs()
            / output["std"].clamp_min(1e-8)
        )[valid]
        if not standardized.numel():
            self.uncertainty_scale.fill_(1.0)
            return 1.0
        normal_quantile = 1.6448536269514722 if target_coverage == 0.90 else 1.0
        scale = float(
            torch.quantile(standardized.float(), target_coverage)
            / normal_quantile
        )
        self.uncertainty_scale.fill_(max(scale, 1e-3))
        return float(self.uncertainty_scale)


def serialized_token_bytes(
    latent_dim: int,
    token_count: int,
    *,
    dtype_bytes: int = 4,
) -> int:
    return int(latent_dim) * int(token_count) * int(dtype_bytes)


__all__ = [
    "FRAME",
    "EVENT",
    "RECENT",
    "CANDIDATE_TYPE_NAMES",
    "FallbackCandidateBatch",
    "CandidateTypeSelector",
    "FallbackSelectorEnsemble",
    "serialized_token_bytes",
]
