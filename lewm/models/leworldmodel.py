"""
LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.
Full model combining encoder + predictor with two-term training loss.

Reference: arXiv:2603.19312
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from lewm.models.encoder import ViTTinyEncoder, Predictor
from lewm.models.sigreg import SIGReg


class LeWorldModel(nn.Module):
    """
    LeWorldModel (LeWM): Stable End-to-End JEPA from Pixels.

    Training objective:
        L_LeWM = L_pred + lambda * SIGReg(Z)

    where:
        L_pred = ||z_hat_{t+1} - z_{t+1}||^2 (MSE prediction loss)
        SIGReg enforces Gaussian-distributed latent embeddings

    Key properties:
        - No EMA, stop-gradient, frozen encoders, or reconstruction loss
        - Only 1 tunable hyperparameter (lambda)
        - ~15M parameters in paper config
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 192,
        action_dim: int = 2,
        encoder_layers: int = 12,
        encoder_heads: int = 3,
        predictor_layers: int = 6,
        predictor_heads: int = 16,
        history_len: int = 3,
        dropout: float = 0.1,
        predictor_norm: str = 'batch',
        encoder_norm: str = 'batch',
        sigreg_lambda: float = 0.1,
        sigreg_projections: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.predictor_norm = predictor_norm
        self.encoder_norm = encoder_norm
        self.sigreg_lambda = sigreg_lambda

        # Encoder: ViT-Tiny
        self.encoder = ViTTinyEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            dropout=dropout,
            encoder_norm=encoder_norm,
        )

        # Predictor: Transformer with AdaLN
        self.predictor = Predictor(
            embed_dim=embed_dim,
            action_dim=action_dim,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            history_len=history_len,
            dropout=dropout,
            output_norm=predictor_norm,
        )

        # SIGReg regularizer
        self.sigreg = SIGReg(
            num_projections=sigreg_projections,
            embed_dim=embed_dim,
        )

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Encode observations to latent embeddings.

        Args:
            observations: (B, C, H, W) or (B, N, C, H, W)
        Returns:
            z: (B, D) or (B, N, D)
        """
        if observations.dim() == 5:
            B, N, C, H, W = observations.shape
            obs_flat = observations.reshape(B * N, C, H, W)
            z = self.encoder(obs_flat)
            z = z.reshape(B, N, -1)
        else:
            z = self.encoder(observations)
        return z

    def predict(
        self,
        z_history: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next latent embeddings.

        Args:
            z_history: (B, N, D) latent history
            actions: (B, N, A) action vectors
        Returns:
            z_pred: (B, N, D) predicted next-step embeddings
        """
        return self.predictor(z_history, actions)

    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the two-term LeWM training loss.

        Args:
            observations: (B, N+1, C, H, W) sequence of observations
            actions: (B, N, A) sequence of actions

        Returns:
            dict with keys: 'loss', 'pred_loss', 'sigreg_loss'
        """
        B, N_plus_1 = observations.shape[0], observations.shape[1]
        N = N_plus_1 - 1

        # Encode all observations
        z_all = self.encode(observations)  # (B, N+1, D)

        # Current and next embeddings
        z_current = z_all[:, :-1, :]  # (B, N, D)
        z_next = z_all[:, 1:, :]  # (B, N, D)

        # Predict next embeddings
        z_pred = self.predictor(z_current, actions)  # (B, N, D)

        # Prediction loss: MSE between predicted and actual next embeddings
        pred_loss = F.mse_loss(z_pred, z_next)

        # SIGReg: enforce Gaussian-distributed embeddings
        sigreg_loss = self.sigreg(z_current)

        # Total loss
        total_loss = pred_loss + self.sigreg_lambda * sigreg_loss

        return {
            'loss': total_loss,
            'pred_loss': pred_loss,
            'sigreg_loss': sigreg_loss,
        }

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Alias for compute_loss."""
        return self.compute_loss(observations, actions)

    @torch.no_grad()
    def plan(
        self,
        obs_init: torch.Tensor,
        obs_goal: torch.Tensor,
        horizon: int = 5,
        num_samples: int = 300,
        num_elites: int = 30,
        num_iterations: int = 30,
        action_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Plan action sequence using CEM in latent space (MPC).

        Args:
            obs_init: (1, C, H, W) initial observation
            obs_goal: (1, C, H, W) goal observation
            horizon: planning horizon
            num_samples: number of candidate sequences per CEM iteration
            num_elites: number of elite samples
            num_iterations: CEM optimization iterations
            action_bounds: (low, high) action bounds, each (action_dim,)

        Returns:
            best_action: (action_dim,) first action of optimized sequence
        """
        self.eval()

        # Encode initial and goal states together as a batch of 2 (the projector BN uses
        # batch statistics, so it needs >1 sample; encoding the pair jointly satisfies that).
        z_pair = self.encode(torch.cat([obs_init, obs_goal], dim=0))  # (2, D)
        z_init = z_pair[:obs_init.shape[0]]   # (1, D)
        z_goal = z_pair[obs_init.shape[0]:]   # (1, D)

        # Initialize CEM distribution
        mu = torch.zeros(horizon, self.action_dim, device=z_init.device)
        sigma = torch.ones(horizon, self.action_dim, device=z_init.device)

        if action_bounds is not None:
            low, high = action_bounds
            low = low.to(z_init.device)
            high = high.to(z_init.device)

        for iteration in range(num_iterations):
            # Sample candidate action sequences
            noise = torch.randn(
                num_samples, horizon, self.action_dim, device=z_init.device
            )
            action_seqs = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise

            # Clip to action bounds
            if action_bounds is not None:
                action_seqs = torch.clamp(
                    action_seqs, low.unsqueeze(0), high.unsqueeze(0)
                )

            # Roll out in latent space
            z_current = z_init.expand(num_samples, -1)  # (num_samples, D)

            for t in range(horizon):
                a_t = action_seqs[:, t, :]  # (num_samples, A)
                z_input = z_current.unsqueeze(1)  # (num_samples, 1, D)
                a_input = a_t.unsqueeze(1)  # (num_samples, 1, A)
                z_pred = self.predictor(z_input, a_input)  # (num_samples, 1, D)
                z_current = z_pred[:, 0, :]  # (num_samples, D)

            # Cost: distance from final state to goal
            costs = ((z_current - z_goal) ** 2).sum(dim=-1)  # (num_samples,)

            # Select elites
            elite_indices = costs.topk(num_elites, largest=False).indices
            elite_actions = action_seqs[elite_indices]

            # Update distribution
            mu = elite_actions.mean(dim=0)
            sigma = elite_actions.std(dim=0).clamp(min=1e-3)

        return mu[0]
