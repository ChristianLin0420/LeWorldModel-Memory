"""
SIGReg: Sketched-Isotropic-Gaussian Regularizer
Based on Appendix A of the paper.

Enforces Gaussian-distributed latent embeddings by:
1. Projecting embeddings onto M random unit-norm directions
2. Applying the Epps-Pulley normality test statistic to each 1D projection
3. Averaging across projections

By the Cramer-Wold theorem, matching all 1D marginals ≈ matching the full joint distribution.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SIGReg(nn.Module):
    """
    SIGReg regularizer for preventing representation collapse in JEPA training.

    The regularizer encourages latent embeddings to follow an isotropic Gaussian
    distribution N(0, I) by:
    1. Projecting embeddings onto M random directions: h^(m) = Z @ u^(m)
    2. Computing the Epps-Pulley test statistic for each projection
    3. Averaging the test statistics

    The Epps-Pulley test statistic is based on the empirical characteristic function (ECF):
        T = integral w(t) |phi_N(t; h) - phi_0(t)|^2 dt

    where phi_0(t) is the characteristic function of N(0,1):
        phi_0(t) = exp(-t^2/2)

    and the weighting function is w(t) = exp(-lambda * t^2) with lambda = 0.2.

    The integral is computed via trapezoid quadrature with T nodes in [t_min, t_max].
    """

    def __init__(
        self,
        num_projections: int = 1024,
        embed_dim: int = 192,
        num_quad_nodes: int = 100,
        t_min: float = 0.2,
        t_max: float = 4.0,
        weight_lambda: float = 0.2,
    ):
        super().__init__()
        self.num_projections = num_projections
        self.embed_dim = embed_dim

        # Pre-sample random projection directions (unit-norm)
        directions = torch.randn(num_projections, embed_dim)
        directions = directions / directions.norm(dim=1, keepdim=True)
        self.register_buffer('directions', directions)  # (M, D)

        # Quadrature nodes for Epps-Pulley integral
        t = torch.linspace(t_min, t_max, num_quad_nodes)
        self.register_buffer('t', t)

        # Weighting function: w(t) = exp(-lambda * t^2)
        weights = torch.exp(-weight_lambda * t ** 2)
        # Trapezoid quadrature weights (including 0.5 for endpoints)
        dt = (t_max - t_min) / (num_quad_nodes - 1)
        trap_weights = torch.full_like(weights, dt)
        trap_weights[0] *= 0.5
        trap_weights[-1] *= 0.5
        self.register_buffer('weights', weights * trap_weights)

    def empirical_characteristic_function(
        self,
        h: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the empirical characteristic function for projected data.

        Args:
            h: (B, M) projected data (B samples, M projections)
            t: (T,) quadrature nodes

        Returns:
            ecf: (T, M) empirical characteristic function values
        """
        # h: (B, M), t: (T,)
        # We want: phi(t, m) = (1/B) * sum_b exp(i * t * h[b, m])
        # Using Euler's formula: exp(i*x) = cos(x) + i*sin(x)
        # For real-valued data, we compute real and imaginary parts

        # h @ t^T -> (B, M, T) via outer product along t for each projection
        # Actually: we need (t_m * h_b) for each b, m
        # h: (B, M), t: (T,) -> th: (B, M, T)
        th = h.unsqueeze(2) * t.unsqueeze(0).unsqueeze(0)  # (B, M, T)

        # ECF: mean over batch dimension of exp(i * th)
        # Real part: cos(th), Imaginary part: sin(th)
        ecf_real = torch.cos(th).mean(dim=0)  # (M, T)
        ecf_imag = torch.sin(th).mean(dim=0)  # (M, T)

        return ecf_real, ecf_imag

    def target_ecf(self, t: torch.Tensor) -> torch.Tensor:
        """
        Characteristic function of N(0, 1): phi_0(t) = exp(-t^2 / 2)
        This is purely real.
        """
        return torch.exp(-t ** 2 / 2)

    def epps_pulley_statistic(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the Epps-Pulley test statistic for each projection.

        Args:
            h: (B, M) projected data

        Returns:
            T: (M,) test statistics for each projection
        """
        # Compute ECF
        ecf_real, ecf_imag = self.empirical_characteristic_function(h, self.t)

        # Target ECF (real-valued for N(0,1))
        target = self.target_ecf(self.t)  # (T,)

        # Squared difference: |phi_N(t) - phi_0(t)|^2
        diff_real = ecf_real - target.unsqueeze(0)  # (M, T)
        diff_imag = ecf_imag  # (M, T)

        squared_diff = diff_real ** 2 + diff_imag ** 2  # (M, T)

        # Weighted integral via trapezoid quadrature
        T_stat = (squared_diff * self.weights.unsqueeze(0)).sum(dim=1)  # (M,)

        return T_stat

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss.

        Args:
            z: (B, D) or (B, N, D) latent embeddings
               If 3D, we flatten across batch and sequence dimensions.

        Returns:
            loss: scalar SIGReg loss (mean Epps-Pulley statistic across projections)
        """
        if z.dim() == 3:
            B, N, D = z.shape
            z = z.reshape(B * N, D)

        # Project embeddings onto random directions
        # z: (B, D), directions: (M, D) -> h: (B, M)
        h = z @ self.directions.T

        # Compute Epps-Pulley statistic for each projection
        T_stats = self.epps_pulley_statistic(h)  # (M,)

        # Average across projections
        return T_stats.mean()


class MultiSubspaceSIGReg(nn.Module):
    """Frozen row-orthonormal multi-subspace SIGReg from Sub-JEPA.

    The ambient embedding is projected into ``num_subspaces`` independently
    initialized subspaces.  Each projection matrix has orthonormal rows and is
    stored as a buffer, so it cannot co-adapt with the encoder.  The subspace
    width is fixed to ``embed_dim // num_subspaces`` as in the paper.

    Within every subspace and at every sequence position, fresh random unit
    directions are sampled on each forward pass.  Their one-dimensional
    marginals are matched to ``N(0, 1)`` with the official 17-knot
    Epps--Pulley characteristic-function statistic.

    Args:
        embed_dim: Ambient embedding width ``D``.
        num_subspaces: Number of subspaces ``K``.  ``D`` must be divisible by
            ``K`` and each subspace has width ``D / K``.
        num_projections: Number of fresh one-dimensional directions sampled in
            each subspace on every forward pass.
    """

    _NUM_KNOTS = 17

    def __init__(
        self,
        embed_dim: int = 192,
        num_subspaces: int = 4,
        num_projections: int = 1024,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_subspaces = int(num_subspaces)
        self.num_projections = int(num_projections)

        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.num_subspaces <= 0:
            raise ValueError(
                f"num_subspaces must be positive, got {self.num_subspaces}")
        if self.embed_dim % self.num_subspaces != 0:
            raise ValueError(
                "embed_dim must be divisible by num_subspaces; "
                f"got embed_dim={self.embed_dim}, num_subspaces={self.num_subspaces}")
        if self.num_projections <= 0:
            raise ValueError(
                f"num_projections must be positive, got {self.num_projections}")

        self.subspace_dim = self.embed_dim // self.num_subspaces

        # Construct W_k in R^(d_s x D) with W_k W_k^T = I using a reduced QR
        # factorization of a D x d_s Gaussian matrix.  Different subspaces are
        # sampled independently; row orthogonality is within each W_k.
        projections = []
        for _ in range(self.num_subspaces):
            q, _ = torch.linalg.qr(
                torch.randn(self.embed_dim, self.subspace_dim, dtype=torch.float32),
                mode='reduced',
            )
            projections.append(q.transpose(0, 1))
        self.register_buffer(
            'projection_matrices', torch.stack(projections, dim=0).contiguous())

        # Official Sub-JEPA/LeJEPA Epps--Pulley discretization: 17 knots over
        # [0, 3], doubled trapezoid weights for the symmetric integral, and the
        # standard-normal characteristic function as both target and window.
        t = torch.linspace(0.0, 3.0, self._NUM_KNOTS, dtype=torch.float32)
        dt = 3.0 / (self._NUM_KNOTS - 1)
        quadrature = torch.full((self._NUM_KNOTS,), 2.0 * dt, dtype=torch.float32)
        quadrature[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', phi)
        self.register_buffer('weights', quadrature * phi)

    def _canonicalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
        elif embeddings.dim() != 3:
            raise ValueError(
                "MultiSubspaceSIGReg expects (B, D) or (B, T, D), "
                f"got shape {tuple(embeddings.shape)}")
        if embeddings.size(-1) != self.embed_dim:
            raise ValueError(
                f"expected embedding dimension {self.embed_dim}, "
                f"got {embeddings.size(-1)}")
        if embeddings.size(0) == 0 or embeddings.size(1) == 0:
            raise ValueError(
                f"batch and sequence dimensions must be non-empty, got {tuple(embeddings.shape)}")
        if not embeddings.is_floating_point():
            raise TypeError(
                f"embeddings must be floating point, got dtype={embeddings.dtype}")
        return embeddings

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project ``(B,T,D)`` or ``(B,D)`` embeddings to ``(K,T,B,d_s)``."""
        embeddings = self._canonicalize(embeddings)
        # The public helper preserves the caller's ordinary dtype behavior.  The
        # complete loss below explicitly calls it with FP32 inputs outside AMP.
        projected = torch.einsum(
            'btd,ked->btke', embeddings, self.projection_matrices.to(embeddings.dtype))
        return projected.permute(2, 1, 0, 3).contiguous()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return the scalar mean statistic across subspaces, time, and slices."""
        embeddings = self._canonicalize(embeddings)
        device_type = embeddings.device.type
        # Matmul/einsum and trigonometry must remain FP32 even when the caller is
        # under BF16/FP16 autocast.  The cast remains differentiable to the input.
        with torch.autocast(device_type=device_type, enabled=False):
            projected = self.project(embeddings.float())  # (K, T, B, d_s)
            directions = torch.randn(
                self.num_subspaces,
                self.subspace_dim,
                self.num_projections,
                device=projected.device,
                dtype=torch.float32,
            )
            directions.div_(directions.norm(p=2, dim=1, keepdim=True))

            # (K,T,B,d_s) x (K,d_s,M) -> (K,T,B,M,knots).
            samples = torch.einsum(
                'ktbd,kdn->ktbn', projected, directions).unsqueeze(-1) * self.t
            ecf_real = samples.cos().mean(dim=2)
            ecf_imag = samples.sin().mean(dim=2)
            error = (ecf_real - self.phi).square() + ecf_imag.square()
            statistic = (error @ self.weights) * projected.size(2)
            return statistic.mean()


def sigreg_loss(
    z: torch.Tensor,
    num_projections: int = 1024,
    embed_dim: int = 192,
) -> torch.Tensor:
    """
    Functional interface for SIGReg loss.
    Creates a SIGReg module on-the-fly (for simple use cases).
    Prefer using the SIGReg class directly for efficiency.
    """
    reg = SIGReg(
        num_projections=num_projections,
        embed_dim=embed_dim,
    ).to(z.device)
    return reg(z)
