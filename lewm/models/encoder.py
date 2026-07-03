"""
LeWorldModel (LeWM) - Core Implementation
Based on: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
arXiv:2603.19312
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ViTTinyEncoder(nn.Module):
    """
    ViT-Tiny encoder from raw pixels.
    Paper config: patch_size=14, 12 layers, 3 attention heads, hidden_dim=192.
    Uses the last-layer [CLS] token, followed by a one-layer projector and a
    configurable output normalization.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 192,
        num_layers: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        encoder_norm: str = 'batch',
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder_norm = encoder_norm
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        if encoder_norm == 'batch':
            # Historical LeWM checkpoints use batch statistics in both train and eval.
            projector_norm = nn.BatchNorm1d(embed_dim, track_running_stats=False)
        elif encoder_norm == 'layer':
            projector_norm = nn.LayerNorm(embed_dim)
        elif encoder_norm == 'causal':
            # V10-J: per-frame, batch-independent, and affine-free. Explicit train-time
            # variance/covariance terms provide cross-frame anti-collapse pressure.
            projector_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        elif encoder_norm == 'none':
            # Batch-independent path for causal batch-size-one streaming.  The ViT itself
            # already uses only per-token LayerNorm; this leaves no cross-example operation.
            projector_norm = nn.Identity()
        else:
            raise ValueError(
                f"unknown encoder_norm {encoder_norm!r}; "
                "expected batch, layer, causal, or none")

        # Keep the historical one-layer projection and its parameter names.  Only the optional
        # normalization module changes, so ``encoder_norm='batch'`` remains checkpoint-compatible.
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            projector_norm,
        )

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) raw pixel images
        Returns:
            z: (B, embed_dim) latent embeddings
        """
        B = x.shape[0]

        # Patch embedding: (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract [CLS] token embedding
        cls_emb = x[:, 0]

        z = self.projector(cls_emb)

        return z


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        # need_weights=False: identical math, but skips materializing the
        # (B*heads, N, N) attention-weight tensor (V19 P0 OOM engineering note).
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization for action conditioning.
    Actions are mapped to scale and shift parameters for layer norm.
    """

    def __init__(self, dim: int, action_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_scale = nn.Linear(action_dim, dim)
        self.ada_shift = nn.Linear(action_dim, dim)
        # Initialize to zero so action conditioning impacts training progressively
        nn.init.zeros_(self.ada_scale.weight)
        nn.init.zeros_(self.ada_scale.bias)
        nn.init.zeros_(self.ada_shift.weight)
        nn.init.zeros_(self.ada_shift.bias)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) latent sequence
            action: (B, N, A) action vectors (one per timestep)
        """
        x = self.norm(x)
        # action: (B, N, A) -> scale/shift: (B, N, D)
        scale = self.ada_scale(action)
        shift = self.ada_shift(action)
        return x * (1 + scale) + shift


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with AdaLN for action conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        action_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ada_ln = AdaLN(dim, action_dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        # need_weights=False: identical math, but skips materializing the
        # (B*heads, N, N) attention-weight tensor (V19 P0 OOM engineering note).
        attn_out = self.attn(normed, normed, normed, attn_mask=attn_mask,
                             need_weights=False)[0]
        x = x + attn_out

        # AdaLN + MLP with residual
        x = x + self.mlp(self.ada_ln(x, action))

        return x


class Predictor(nn.Module):
    """
    Transformer predictor with AdaLN action conditioning.
    Paper config: 6 layers, 16 attention heads, 10% dropout, ~10M params.
    Takes history of N frame representations, predicts next frame embedding autoregressively.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        action_dim: int = 2,
        num_layers: int = 6,
        num_heads: int = 16,
        history_len: int = 3,
        dropout: float = 0.1,
        output_norm: str = 'batch',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.history_len = history_len
        self.output_norm = output_norm

        if output_norm == 'batch':
            projector_norm = nn.BatchNorm1d(embed_dim, track_running_stats=False)
        elif output_norm == 'layer':
            # Per-token normalization: unlike batch-stat BN, this cannot couple one
            # sliding prediction window to other (including later) windows in a batch.
            projector_norm = nn.LayerNorm(embed_dim)
        elif output_norm == 'none':
            projector_norm = nn.Identity()
        else:
            raise ValueError(
                f"unknown predictor output_norm {output_norm!r}; expected batch, layer, or none")

        # Learned positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, history_len + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # AdaLN transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                action_dim=action_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Legacy checkpoints use batch-stat BN.  Causal sliding-window experiments should use
        # ``output_norm='layer'`` so each token is normalized independently of other windows.
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            projector_norm,
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Causal attention mask to prevent looking at future embeddings."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        z_history: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_history: (B, N, D) sequence of latent embeddings
            actions: (B, N) action indices or (B, N, A) action vectors
        Returns:
            z_pred: (B, N, D) predicted next-step embeddings
        """
        B, N, D = z_history.shape

        # Add positional embeddings
        x = z_history + self.pos_embed[:, :N, :]

        # Causal mask
        causal_mask = self._causal_mask(N, z_history.device)

        # Process through AdaLN transformer blocks
        for block in self.blocks:
            x = block(x, actions, attn_mask=causal_mask)

        x = self.norm(x)

        # Project. Flattening also supports the legacy BatchNorm1d implementation; LayerNorm
        # and Identity operate independently on each row.
        x_flat = x.reshape(B * N, D)
        z_pred = self.projector(x_flat).reshape(B, N, D)

        return z_pred

    def autoregressive_rollout(
        self,
        z_init: torch.Tensor,
        action_sequence: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """
        Autoregressive rollout for planning.
        Args:
            z_init: (B, D) initial latent state
            action_sequence: (B, H, A) action sequence
            horizon: int, planning horizon H
        Returns:
            z_rollout: (B, H, D) predicted latent states
        """
        B = z_init.shape[0]
        z_rollout = []
        z_current = z_init

        for t in range(horizon):
            # Use last history_len frames (or pad with available)
            # For simplicity, use current frame repeated
            z_input = z_current.unsqueeze(1)  # (B, 1, D)
            a_t = action_sequence[:, t, :]  # (B, A)

            # Single step prediction
            z_pred = self.forward(z_input, a_t.unsqueeze(1))  # (B, 1, D)
            z_next = z_pred[:, 0, :]  # (B, D)
            z_rollout.append(z_next)
            z_current = z_next

        return torch.stack(z_rollout, dim=1)  # (B, H, D)


class FrozenDINOEncoder(nn.Module):
    """Frozen pretrained DINOv2 ViT-S backbone (the DINO-WM backbone) + a trainable
    projection — the 'frozen large pretrained backbone at scale' encoder. The 21.6M DINOv2
    params are frozen; only the projector (and the memory/predictor) are trained."""

    def __init__(self, embed_dim: int = 128,
                 model_name: str = 'vit_small_patch14_dinov2.lvd142m', img: int = 224):
        super().__init__()
        import timm
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0,
                                          img_size=img, dynamic_img_size=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.img = img
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.projector = nn.Sequential(
            nn.Linear(self.backbone.num_features, embed_dim),
            nn.BatchNorm1d(embed_dim, track_running_stats=False),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()                 # keep frozen backbone in eval always
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            xi = F.interpolate(x, size=(self.img, self.img), mode='bilinear', align_corners=False)
            xi = (xi - self.mean) / self.std
            f = self.backbone(xi)
        return self.projector(f.float())
