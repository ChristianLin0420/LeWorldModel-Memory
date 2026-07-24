"""Compact host conditioner and abstaining utility gate for native trajectories."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


FRAME = 0
EVENT = 1
RECENT = 2
TOKEN_TYPE_NAMES = {
    FRAME: "frame",
    EVENT: "event",
    RECENT: "recent",
}


@dataclass
class MemoryTokenBatch:
    """Fixed-budget memory tokens supplied to the frozen prediction host."""

    values: torch.Tensor
    metadata: torch.Tensor
    token_type: torch.Tensor
    valid: torch.Tensor

    def index(self, index: torch.Tensor) -> "MemoryTokenBatch":
        return MemoryTokenBatch(
            values=self.values[index],
            metadata=self.metadata[index],
            token_type=self.token_type[index],
            valid=self.valid[index],
        )

    def __len__(self) -> int:
        return int(self.values.shape[0])


class NativeMemoryConditioner(nn.Module):
    """Normalized semantic tokens cross-attend into a frozen host prediction.

    The base encoder and predictor live outside this module.  Calling with
    ``memory=None`` is an exact identity path.  The residual projection is
    zero-initialized, so the initialized memory path is also an identity.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        metadata_dim: int,
        *,
        code_dim: int = 64,
        hidden: int = 160,
        heads: int = 4,
        max_residual: float = 0.5,
    ) -> None:
        super().__init__()
        if code_dim % heads:
            raise ValueError("code_dim must be divisible by heads")
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.metadata_dim = int(metadata_dim)
        self.code_dim = int(code_dim)
        self.max_residual = float(max_residual)
        self.semantic = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, code_dim),
            nn.LayerNorm(code_dim),
        )
        self.metadata = nn.Sequential(
            nn.LayerNorm(metadata_dim),
            nn.Linear(metadata_dim, code_dim),
        )
        self.token_type = nn.Embedding(len(TOKEN_TYPE_NAMES), code_dim)
        self.query = nn.Sequential(
            nn.LayerNorm(2 * latent_dim + action_dim),
            nn.Linear(2 * latent_dim + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, code_dim),
            nn.LayerNorm(code_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            code_dim,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.recent_attention = nn.MultiheadAttention(
            code_dim,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.recent_decoder = nn.Sequential(
            nn.LayerNorm(latent_dim + 2 * code_dim),
            nn.Linear(latent_dim + 2 * code_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.history_decoder = nn.Sequential(
            nn.LayerNorm(latent_dim + 3 * code_dim),
            nn.Linear(latent_dim + 3 * code_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        for decoder in (self.recent_decoder, self.history_decoder):
            nn.init.zeros_(decoder[-1].weight)
            nn.init.zeros_(decoder[-1].bias)
        self.recent_scale_logit = nn.Parameter(torch.tensor(-2.0))
        self.history_scale_logit = nn.Parameter(torch.tensor(-2.0))

    def encode_tokens(self, memory: MemoryTokenBatch) -> torch.Tensor:
        code = (
            self.semantic(memory.values)
            + self.metadata(memory.metadata)
            + self.token_type(memory.token_type)
        )
        return F.normalize(code, dim=-1)

    def forward(
        self,
        base_prediction: torch.Tensor,
        context_z: torch.Tensor,
        action: torch.Tensor,
        memory: MemoryTokenBatch | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if memory is None:
            zero = base_prediction.new_zeros(base_prediction.shape[0])
            return base_prediction, {
                "residual_norm": zero,
                "attention_entropy": zero,
                "present": zero,
            }
        if memory.values.shape[:2] != memory.valid.shape:
            raise ValueError("memory value and validity shapes disagree")
        present = memory.valid.any(1)
        token_code = self.encode_tokens(memory)
        query_code = F.normalize(
            self.query(
                torch.cat(
                    [context_z[:, -1], base_prediction, action],
                    dim=-1,
                )
            ),
            dim=-1,
        )
        recent_valid = memory.valid & (memory.token_type == RECENT)
        history_valid = memory.valid & (memory.token_type != RECENT)

        def attend(
            module: nn.MultiheadAttention,
            valid: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            available = valid.any(1)
            safe_valid = valid.clone()
            safe_valid[~available, 0] = True
            attended, weight = module(
                query_code[:, None],
                token_code,
                token_code,
                key_padding_mask=~safe_valid,
                need_weights=True,
                average_attn_weights=True,
            )
            attended = attended[:, 0] * available[:, None]
            probability = weight[:, 0].masked_fill(~valid, 0.0)
            return attended, probability, available

        recent_code, recent_probability, recent_present = attend(
            self.recent_attention,
            recent_valid,
        )
        history_code, history_probability, history_present = attend(
            self.cross_attention,
            history_valid,
        )
        recent_raw = self.recent_decoder(
            torch.cat(
                [base_prediction, query_code, recent_code],
                dim=-1,
            )
        )
        history_raw = self.history_decoder(
            torch.cat(
                [
                    base_prediction,
                    query_code,
                    history_code,
                    history_code - recent_code,
                ],
                dim=-1,
            )
        )
        recent_scale = self.max_residual * torch.sigmoid(
            self.recent_scale_logit
        )
        history_scale = self.max_residual * torch.sigmoid(
            self.history_scale_logit
        )
        recent_residual = (
            recent_scale
            * torch.tanh(recent_raw)
            * recent_present[:, None]
        )
        history_residual = (
            history_scale
            * torch.tanh(history_raw)
            * history_present[:, None]
        )
        residual = (recent_residual + history_residual) * present[:, None]
        probability = recent_probability + history_probability
        entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(1)
        return base_prediction + residual, {
            "residual_norm": residual.square().mean(-1),
            "attention_entropy": entropy,
            "present": present.float(),
            "history_present": history_present.float(),
            "recent_residual_norm": recent_residual.square().mean(-1),
            "history_residual_norm": history_residual.square().mean(-1),
            "token_code": token_code,
            "query_code": query_code,
            "attention": probability,
            "history_attention": history_probability,
            "recent_attention": recent_probability,
        }


def geometry_regularization(
    code: torch.Tensor,
    tokens: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Preserve within-read cosine geometry through the semantic bottleneck."""

    source = F.normalize(tokens, dim=-1)
    code_distance = 1.0 - torch.einsum("bid,bjd->bij", code, code)
    source_distance = 1.0 - torch.einsum("bid,bjd->bij", source, source)
    pair_valid = valid[:, :, None] & valid[:, None, :]
    upper = torch.triu(
        torch.ones(
            valid.shape[1],
            valid.shape[1],
            dtype=torch.bool,
            device=valid.device,
        ),
        diagonal=1,
    )
    pair_valid &= upper[None]
    if not bool(pair_valid.any()):
        return code.new_zeros(())
    return F.smooth_l1_loss(
        code_distance[pair_valid],
        source_distance[pair_valid].detach(),
        beta=0.1,
    )


def variance_regularization(
    code: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style variance and covariance penalties for normalized codes."""

    flat = code[valid]
    if flat.shape[0] < 2:
        zero = code.new_zeros(())
        return zero, zero
    flat = flat * (flat.shape[-1] ** 0.5)
    centered = flat - flat.mean(0, keepdim=True)
    std = torch.sqrt(centered.var(0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - std).mean()
    covariance = centered.T @ centered / max(1, flat.shape[0] - 1)
    covariance.fill_diagonal_(0.0)
    covariance_loss = covariance.square().sum() / flat.shape[-1]
    return variance, covariance_loss


class UtilityGateHead(nn.Module):
    """Predict utility, heteroscedastic uncertainty, and a help probability."""

    def __init__(self, input_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.output = nn.Linear(hidden, 3)

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.output(self.encoder(features))
        return output[:, 0], output[:, 1].clamp(-9.0, 2.0), output[:, 2]


class UtilityGateEnsemble(nn.Module):
    """Cross-fitted ensemble with conformal one-sided lower bounds."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden: int = 128,
        members: int = 3,
    ) -> None:
        super().__init__()
        self.members = nn.ModuleList(
            [UtilityGateHead(input_dim, hidden) for _ in range(members)]
        )
        self.register_buffer("conformal_quantile", torch.tensor(1.2815516))
        self.register_buffer("probability_temperature", torch.tensor(1.0))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = [member(features) for member in self.members]
        means = torch.stack([value[0] for value in outputs])
        log_variances = torch.stack([value[1] for value in outputs])
        logits = torch.stack([value[2] for value in outputs])
        mean = means.mean(0)
        epistemic = means.var(0, unbiased=False)
        aleatoric = torch.exp(log_variances).mean(0)
        std = (epistemic + aleatoric).clamp_min(1e-12).sqrt()
        probability = torch.sigmoid(
            logits / self.probability_temperature.clamp_min(1e-3)
        ).mean(0)
        return {
            "mean": mean,
            "std": std,
            "probability": probability,
            "member_mean": means,
            "member_log_variance": log_variances,
            "member_logit": logits,
        }

    def lower_confidence_bound(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self(features)
        lower = output["mean"] - self.conformal_quantile * output["std"]
        return lower, output


def parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


__all__ = [
    "FRAME",
    "EVENT",
    "RECENT",
    "TOKEN_TYPE_NAMES",
    "MemoryTokenBatch",
    "NativeMemoryConditioner",
    "UtilityGateEnsemble",
    "UtilityGateHead",
    "geometry_regularization",
    "variance_regularization",
    "parameter_count",
]
