"""Loader for the official LeWorldModel Reacher checkpoint.

The module layout follows the MIT-licensed reference implementation at
https://github.com/lucas-maes/le-wm so the released Hugging Face state dict
loads strictly.  The small wrapper only adds deterministic image preprocessing
and batched sequence encoding for this repository's evaluation pipeline.

Reference implementation copyright (c) 2026 Lucas Maes, MIT License.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import ViTConfig, ViTModel


OFFICIAL_IMAGE_SIZE = 224
OFFICIAL_EMBED_DIM = 192
OFFICIAL_HISTORY = 3
OFFICIAL_ACTION_DIM = 10


def _modulate(x: torch.Tensor, shift: torch.Tensor,
              scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        batch, steps, _ = x.shape
        q, k, v = (
            value.reshape(batch, steps, self.heads, -1).transpose(1, 2)
            for value in qkv
        )
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=causal)
        out = out.transpose(1, 2).reshape(batch, steps, -1)
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head,
                              dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
         gate_mlp) = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 depth: int, heads: int, dim_head: int, mlp_dim: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([
            ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.input_proj = (nn.Linear(input_dim, hidden_dim)
                           if input_dim != hidden_dim else nn.Identity())
        self.cond_proj = (nn.Linear(input_dim, hidden_dim)
                          if input_dim != hidden_dim else nn.Identity())
        self.output_proj = (nn.Linear(hidden_dim, output_dim)
                            if hidden_dim != output_dim else nn.Identity())

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        c = self.cond_proj(c)
        for block in self.layers:
            x = block(x, c)
        return self.output_proj(self.norm(x))


class Predictor(nn.Module):
    def __init__(self, *, num_frames: int = OFFICIAL_HISTORY,
                 input_dim: int = OFFICIAL_EMBED_DIM,
                 hidden_dim: int = OFFICIAL_EMBED_DIM,
                 output_dim: int = OFFICIAL_EMBED_DIM, depth: int = 6,
                 heads: int = 16, mlp_dim: int = 2048, dim_head: int = 64,
                 dropout: float = 0.1, emb_dropout: float = 0.0) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim, hidden_dim, output_dim, depth, heads, dim_head,
            mlp_dim, dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        steps = x.size(1)
        x = self.dropout(x + self.pos_embedding[:, :steps])
        return self.transformer(x, c)


class Embedder(nn.Module):
    def __init__(self, input_dim: int = OFFICIAL_ACTION_DIM,
                 smoothed_dim: int | None = None,
                 emb_dim: int = OFFICIAL_EMBED_DIM,
                 mlp_scale: int = 4) -> None:
        super().__init__()
        smoothed_dim = input_dim if smoothed_dim is None else smoothed_dim
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim,
                                     kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x.float().transpose(1, 2)).transpose(1, 2)
        return self.embed(x)


class MLP(nn.Module):
    def __init__(self, input_dim: int = OFFICIAL_EMBED_DIM,
                 hidden_dim: int = 2048,
                 output_dim: int = OFFICIAL_EMBED_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_official_reacher_model() -> "OfficialLeWM":
    config = ViTConfig(
        image_size=OFFICIAL_IMAGE_SIZE,
        patch_size=14,
        num_channels=3,
        hidden_size=OFFICIAL_EMBED_DIM,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    encoder = ViTModel(config, add_pooling_layer=False)
    return OfficialLeWM(
        encoder=encoder,
        predictor=Predictor(),
        action_encoder=Embedder(),
        projector=MLP(),
        pred_proj=MLP(),
    )


class OfficialLeWM(nn.Module):
    def __init__(self, encoder: nn.Module, predictor: Predictor,
                 action_encoder: Embedder, projector: MLP,
                 pred_proj: MLP) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector
        self.pred_proj = pred_proj

    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode normalized BCHW tensors at 224x224 into 192-D latents."""
        output = self.encoder(pixel_values=pixels,
                              interpolate_pos_encoding=True)
        return self.projector(output.last_hidden_state[:, 0])

    def encode_sequence(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dim() != 5:
            raise ValueError(f"pixels must be (B,T,C,H,W), got {pixels.shape}")
        batch, steps = pixels.shape[:2]
        latent = self.encode_pixels(pixels.flatten(0, 1))
        return latent.reshape(batch, steps, -1)

    def predict(self, latent: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        embedded_actions = self.action_encoder(actions)
        prediction = self.predictor(latent, embedded_actions)
        batch, steps = prediction.shape[:2]
        prediction = self.pred_proj(prediction.flatten(0, 1))
        return prediction.reshape(batch, steps, -1)


def load_official_reacher_checkpoint(
        weights: str | Path, device: torch.device | str = "cpu"
        ) -> OfficialLeWM:
    model = build_official_reacher_model()
    state = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def preprocess_frames(
        frames: torch.Tensor,
        *, image_size: int = OFFICIAL_IMAGE_SIZE,
        ) -> torch.Tensor:
    """Apply the released LeWM image transform to BCHW frames.

    The reference pipeline converts uint8 images with ``scale=True``, applies
    ImageNet normalization, and then resizes.  Keeping that order (including
    multiplication by ``1 / 255`` rather than division) makes this helper
    numerically identical to the torchvision-v2 transform used by LeWM.  The
    keyword-only size keeps the Reacher API unchanged while allowing official
    environment configs to select their published image size.
    """
    if frames.dim() != 4 or frames.shape[1] != 3:
        raise ValueError(f"frames must be BCHW, got {frames.shape}")
    if isinstance(image_size, bool) or not isinstance(image_size, int) \
            or image_size <= 0:
        raise ValueError(f"image_size must be a positive integer, got {image_size!r}")
    value = frames.float()
    if value.max() > 1.5:
        value = value * (1.0 / 255.0)
    mean = value.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = value.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    value = (value - mean) / std
    return F.interpolate(value, size=(image_size, image_size),
                         mode="bilinear", align_corners=False,
                         antialias=True)
