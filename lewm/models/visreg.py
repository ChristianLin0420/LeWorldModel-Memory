"""Published VisReg objective (arXiv:2606.02572), exact recipe.

Three decoupled distributional constraints on a batch of embeddings, combined
by the host with a single outer lambda (published range [0.6, 0.9]):

    L_scale  = (1/D) sum_j (1 - sigma_j)^2          sigma_j = per-coord std
    L_shape  = SW2^2( standardized Z , N(0, I) )    K fresh Gaussian slices,
               sorted projections matched to Blom normal quantiles
    L_center = (1/D) ||mu||^2
    total    = L_scale + L_shape + L_center

Fidelity notes (docs/V20_PROPOSAL.md 2.2/4.1 — the registered V17 deltas this
module deliberately does NOT reproduce):

- no self-paced shape multiplier, no gradient bisector, no eigenvalue-W2
  substitution (the V17 "AutoVISReg" machinery);
- slice count K = 4096, not K = 2D;
- the shape term standardizes by *detached* mean/std, so shape cannot fight
  scale (the V17 preflight failure mode: "shape gradient dominated the scale
  direction");
- constant-restoring-gradient property at collapse: at sigma = 0 the scale
  gradient is exactly -2/D per coordinate std — never the SIGReg
  projected-zero plateau (tests/test_v20_w0.py checks this numerically).

The module is stateless (no trainable parameters); fresh slice directions are
drawn from the global torch RNG each forward, mirroring the SIGReg
fresh-sketch convention of lewm/models/sigreg.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_SLICES = 4096
EPS = 1e-6


class VisRegObjective(nn.Module):
    """Exact published VisReg loss on (B, L, D) or (N, D) embeddings."""

    def __init__(self, num_slices: int = DEFAULT_SLICES, eps: float = EPS
                 ) -> None:
        super().__init__()
        if num_slices < 1:
            raise ValueError(f"num_slices must be >= 1, got {num_slices}")
        self.num_slices = int(num_slices)
        self.eps = float(eps)
        self._quantile_cache: dict[tuple[int, str], torch.Tensor] = {}

    def _quantiles(self, count: int, device: torch.device) -> torch.Tensor:
        """Standard-normal quantiles at Blom plotting positions, shape (n,).

        q_i = ndtri((i - 0.375) / (n + 0.25)), i = 1..n — the standard
        expected-normal-order-statistic approximation the sorted projections
        are matched against.
        """
        key = (count, str(device))
        cached = self._quantile_cache.get(key)
        if cached is None:
            ranks = torch.arange(1, count + 1, dtype=torch.float64,
                                 device=device)
            positions = (ranks - 0.375) / (count + 0.25)
            cached = torch.special.ndtri(positions).to(torch.float32)
            self._quantile_cache[key] = cached
        return cached

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        if embeddings.dim() != 2:
            raise ValueError(
                f"expected (B, L, D) or (N, D), got {tuple(embeddings.shape)}")
        if embeddings.shape[0] < 2:
            raise ValueError("VisReg needs at least 2 samples")
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            z = embeddings.float()
            samples, dim = z.shape
            mu = z.mean(dim=0)
            sigma = z.std(dim=0, unbiased=False)
            scale = (1.0 - sigma).square().mean()
            center = mu.square().mean()

            # Shape: stop-grad standardization (shape cannot fight scale).
            z_hat = (z - mu.detach()) / (sigma.detach() + self.eps)
            directions = torch.randn(dim, self.num_slices, device=z.device,
                                     dtype=torch.float32)
            directions = directions / directions.norm(dim=0, keepdim=True)
            projected = z_hat @ directions                    # (N, K)
            projected_sorted = projected.sort(dim=0).values
            quantiles = self._quantiles(samples, z.device)    # (N,)
            shape = (projected_sorted - quantiles[:, None]).square().mean()

            total = scale + shape + center
        return {"total": total, "scale": scale, "shape": shape,
                "center": center}

    def describe(self) -> dict[str, object]:
        return {
            "objective": "visreg_published_arxiv_2606_02572",
            "num_slices": self.num_slices,
            "eps": self.eps,
            "scale": "mean_j (1 - std_j)^2 (biased std)",
            "shape": ("SW2^2 to N(0,I): sorted slice projections vs Blom "
                      "normal quantiles, stop-grad standardization"),
            "center": "mean_j mu_j^2",
            "combiner": "single outer lambda in the host loss",
            "v17_deltas_not_reproduced": [
                "self_paced_shape_multiplier", "gradient_bisector",
                "eigenvalue_w2_scale_term", "k_equals_2d_slices"],
        }
