"""Spatial/object-preserving memory conditioner for a frozen global host."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


HISTORY = 0
RECENT = 1
QUERY = 2


@dataclass
class SpatialTokenBatch:
    """Fixed-budget location-preserving token payload."""

    feature: torch.Tensor
    delta: torch.Tensor
    coordinates: torch.Tensor
    extent: torch.Tensor
    metadata: torch.Tensor
    valid: torch.Tensor
    kind: torch.Tensor

    def index(self, index: torch.Tensor) -> "SpatialTokenBatch":
        return SpatialTokenBatch(
            feature=self.feature[index],
            delta=self.delta[index],
            coordinates=self.coordinates[index],
            extent=self.extent[index],
            metadata=self.metadata[index],
            valid=self.valid[index],
            kind=self.kind[index],
        )

    def __len__(self) -> int:
        return int(self.feature.shape[0])


class SpatialMemoryConditioner(nn.Module):
    """Current patch slots query historical slots before a bounded host residual."""

    def __init__(
        self,
        host_dim: int,
        action_dim: int,
        feature_dim: int,
        metadata_dim: int,
        token_count: int,
        *,
        code_dim: int = 64,
        hidden: int = 160,
        heads: int = 4,
        max_residual: float = 0.75,
        gate_init: float = -2.0,
        use_delta: bool = False,
    ) -> None:
        super().__init__()
        if code_dim % heads:
            raise ValueError("code_dim must be divisible by heads")
        self.host_dim = int(host_dim)
        self.feature_dim = int(feature_dim)
        self.metadata_dim = int(metadata_dim)
        self.token_count = int(token_count)
        self.code_dim = int(code_dim)
        self.max_residual = float(max_residual)
        self.use_delta = bool(use_delta)
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, code_dim),
        )
        self.delta_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, code_dim),
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(4, code_dim),
            nn.GELU(),
            nn.Linear(code_dim, code_dim),
        )
        self.metadata_encoder = nn.Sequential(
            nn.LayerNorm(metadata_dim),
            nn.Linear(metadata_dim, code_dim),
        )
        self.kind_embedding = nn.Embedding(3, code_dim)
        self.global_query = nn.Sequential(
            nn.LayerNorm(2 * host_dim + action_dim),
            nn.Linear(2 * host_dim + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, code_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            code_dim,
            heads,
            batch_first=True,
            dropout=0.0,
        )
        self.slot_decoder = nn.Sequential(
            nn.LayerNorm(3 * code_dim),
            nn.Linear(3 * code_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, feature_dim),
        )
        self.slot_gate = nn.Linear(2 * code_dim, 1)
        nn.init.zeros_(self.slot_gate.weight)
        nn.init.constant_(self.slot_gate.bias, gate_init)
        flattened = token_count * feature_dim
        self.recent_head = nn.Sequential(
            nn.LayerNorm(flattened),
            nn.Linear(flattened, hidden),
            nn.GELU(),
            nn.Linear(hidden, host_dim),
        )
        self.history_head = nn.Sequential(
            nn.LayerNorm(flattened),
            nn.Linear(flattened, hidden),
            nn.GELU(),
            nn.Linear(hidden, host_dim),
        )
        for head in (self.recent_head, self.history_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.recent_scale_logit = nn.Parameter(torch.tensor(gate_init))
        self.history_scale_logit = nn.Parameter(torch.tensor(gate_init))

    def encode_tokens(self, value: SpatialTokenBatch) -> torch.Tensor:
        code = (
            self.feature_encoder(value.feature)
            + self.position_encoder(
                torch.cat([value.coordinates, value.extent], dim=-1)
            )
            + self.metadata_encoder(value.metadata)
            + self.kind_embedding(value.kind)[:, None]
        )
        if self.use_delta:
            code = code + self.delta_encoder(value.delta)
        return F.normalize(code, dim=-1)

    def forward(
        self,
        base_prediction: torch.Tensor,
        context_z: torch.Tensor,
        action: torch.Tensor,
        query: SpatialTokenBatch,
        memory: SpatialTokenBatch | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if memory is None:
            zero = base_prediction.new_zeros(base_prediction.shape[0])
            return base_prediction, {
                "residual_norm": zero,
                "attention_entropy": zero,
                "attention_overlap": zero,
                "patch_utilization": zero,
                "identity_preservation": zero,
                "slot_gate": zero,
                "locality_loss": zero.mean(),
                "alignment_loss": zero.mean(),
            }
        if query.feature.shape[1] != self.token_count:
            raise ValueError("query token count violates fixed budget")
        if memory.feature.shape[1] != self.token_count:
            raise ValueError("memory token count violates fixed budget")
        present = memory.valid.any(1)
        safe_valid = memory.valid.clone()
        safe_valid[~present, 0] = True
        memory_code = self.encode_tokens(memory)
        query_code = self.encode_tokens(query)
        global_code = self.global_query(
            torch.cat(
                [context_z[:, -1], base_prediction, action],
                dim=-1,
            )
        )
        query_code = F.normalize(query_code + global_code[:, None], dim=-1)
        attended, weights = self.cross_attention(
            query_code,
            memory_code,
            memory_code,
            key_padding_mask=~safe_valid,
            need_weights=True,
            average_attn_weights=False,
        )
        mean_weight = weights.mean(1).masked_fill(
            ~memory.valid[:, None],
            0.0,
        )
        slot_hidden = torch.cat(
            [query_code, attended, attended - query_code],
            dim=-1,
        )
        slot_delta = self.slot_decoder(slot_hidden)
        gate = torch.sigmoid(
            self.slot_gate(torch.cat([query_code, attended], dim=-1))
        )
        spatial_residual = slot_delta * gate * query.valid[..., None]
        flattened = spatial_residual.flatten(1)
        recent_raw = self.recent_head(flattened)
        history_raw = self.history_head(flattened)
        is_recent = memory.kind == RECENT
        recent_scale = self.max_residual * torch.sigmoid(
            self.recent_scale_logit
        )
        history_scale = self.max_residual * torch.sigmoid(
            self.history_scale_logit
        )
        residual = torch.where(
            is_recent[:, None],
            recent_scale * torch.tanh(recent_raw),
            history_scale * torch.tanh(history_raw),
        )
        residual = residual * present[:, None]

        query_feature = F.normalize(query.feature, dim=-1)
        memory_feature = F.normalize(memory.feature, dim=-1)
        similarity = torch.einsum(
            "bqd,bkd->bqk",
            query_feature,
            memory_feature,
        )
        coordinate_distance = torch.cdist(
            query.coordinates,
            memory.coordinates,
        )
        target_logit = 2.0 * similarity - coordinate_distance / 0.5
        target_logit = target_logit.masked_fill(
            ~memory.valid[:, None],
            -1e9,
        )
        target = torch.softmax(target_logit, dim=-1).detach()
        locality_loss = -(
            target
            * mean_weight.clamp_min(1e-8).log()
            * query.valid[..., None]
        ).sum(-1).sum() / query.valid.sum().clamp_min(1)
        overlap = (
            mean_weight
            * (coordinate_distance <= 0.5)
            * query.valid[..., None]
        ).sum((1, 2)) / query.valid.sum(1).clamp_min(1)
        memory_use = mean_weight.sum(1)
        utilization = (
            memory_use
            > (0.5 * query.valid.sum(1, keepdim=True) / self.token_count)
        ).float().sum(1)
        probability = mean_weight / mean_weight.sum(
            -1,
            keepdim=True,
        ).clamp_min(1e-8)
        entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(-1)
        entropy = (
            entropy * query.valid
        ).sum(1) / query.valid.sum(1).clamp_min(1)
        identity = (
            mean_weight * similarity * query.valid[..., None]
        ).sum((1, 2)) / query.valid.sum(1).clamp_min(1)
        adjacent = torch.cdist(query.coordinates, query.coordinates) <= 0.55
        pair_valid = (
            query.valid[:, :, None]
            & query.valid[:, None, :]
            & adjacent
        )
        residual_distance = (
            spatial_residual[:, :, None] - spatial_residual[:, None, :]
        ).square().mean(-1)
        alignment_loss = (
            residual_distance[pair_valid].mean()
            if bool(pair_valid.any())
            else residual.new_zeros(())
        )
        return base_prediction + residual, {
            "residual_norm": residual.square().mean(-1),
            "attention_entropy": entropy,
            "attention_overlap": overlap,
            "patch_utilization": utilization,
            "identity_preservation": identity,
            "slot_gate": gate.mean((1, 2)),
            "locality_loss": locality_loss,
            "alignment_loss": alignment_loss,
            "attention": mean_weight,
            "memory_code": memory_code,
            "query_code": query_code,
            "spatial_residual": spatial_residual,
        }


class PatchAlignmentAuxiliary(nn.Module):
    """Train-only reconstruction head for masked patch supervision."""

    def __init__(self, code_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.reconstruction = nn.Sequential(
            nn.LayerNorm(code_dim),
            nn.Linear(code_dim, 2 * code_dim),
            nn.GELU(),
            nn.Linear(2 * code_dim, feature_dim),
        )

    def forward(self, query_code: torch.Tensor) -> torch.Tensor:
        return self.reconstruction(query_code)


def masked_spatial_tokens(
    value: SpatialTokenBatch,
    mask: torch.Tensor,
) -> SpatialTokenBatch:
    """Zero selected patch content while preserving positions and byte budget."""

    if mask.shape != value.valid.shape:
        raise ValueError("patch mask shape differs from token validity")
    keep = (~mask).unsqueeze(-1)
    return SpatialTokenBatch(
        feature=value.feature * keep,
        delta=value.delta * keep,
        coordinates=value.coordinates,
        extent=value.extent,
        metadata=value.metadata,
        valid=value.valid,
        kind=value.kind,
    )


def serialized_spatial_bytes(
    token_count: int,
    feature_dim: int,
    metadata_dim: int,
    *,
    dtype_bytes: int = 4,
) -> int:
    fields = 2 * feature_dim + 2 + 2 + metadata_dim
    return int(token_count) * fields * int(dtype_bytes)


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


__all__ = [
    "HISTORY",
    "RECENT",
    "QUERY",
    "SpatialTokenBatch",
    "SpatialMemoryConditioner",
    "PatchAlignmentAuxiliary",
    "masked_spatial_tokens",
    "serialized_spatial_bytes",
    "parameter_count",
]
