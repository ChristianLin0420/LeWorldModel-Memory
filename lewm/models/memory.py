"""
Two-timescale exponential memory for JEPA-style world models.

This module is the core contribution of LeWM-Memory. The base LeWorldModel encodes
each frame *independently* (a memoryless ViT) and the predictor only attends over a
short window of `history_len` latents. It therefore has no mechanism to carry
information across long temporal gaps -- it cannot represent how short- vs long-term
context shapes the *dynamics* of the latent space.

We add an explicit, mathematically transparent memory: two exponential-moving-average
(EMA) banks over the latent stream z_t (the SIGReg-regularized encoder output),

    m_t = (1 - alpha) * m_{t-1} + alpha * z_t                                  (1)

Unrolling the recurrence shows the memory is a causal convolution of the latent
history with an *exponential kernel*:

    m_t = alpha * sum_{k>=0} (1 - alpha)^k z_{t-k},     K(k) = alpha (1 - alpha)^k.  (2)

The kernel decays geometrically, so the *effective memory horizon* (time constant) is

    tau = -1 / ln(1 - alpha)   ( ~ 1/alpha  for small alpha ).                 (3)

Two banks with different alphas give a clean fast/slow (short/long-term) split:
    - fast bank: large alpha_f  -> small tau_f  (short-term / working memory)
    - slow bank: small alpha_s  -> large tau_s  (long-term / episodic memory)

Eq. (1) is exactly a diagonal linear state-space model (a leaky integrator); it is the
simplest member of the SSM / linear-attention family (S4, Mamba, gated linear
attention all generalize a per-channel decay). The two-timescale split is also the
algorithmic core of Complementary Learning Systems theory (fast hippocampal vs. slow
neocortical memory). The whole thing adds only two scalars (alpha_f, alpha_s) plus two
zero-initialized projections, so the model stays "elegant": a 2-term loss with a handful
of interpretable knobs.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


def tau_to_alpha(tau: float) -> float:
    """Convert an effective memory horizon tau (in steps) to an EMA rate alpha.

    Inverts Eq. (3):  tau = -1 / ln(1 - alpha)  =>  alpha = 1 - exp(-1 / tau).
    """
    return 1.0 - math.exp(-1.0 / float(tau))


def alpha_to_tau(alpha: torch.Tensor) -> torch.Tensor:
    """Effective memory horizon tau = -1 / ln(1 - alpha) (Eq. 3)."""
    return -1.0 / torch.log(1.0 - alpha.clamp(max=1.0 - 1e-6) + 1e-8)


class TwoTimescaleMemory(nn.Module):
    """Two EMA memory banks (fast/short-term and slow/long-term) over a latent stream.

    The EMA rates are parameterized through their logits so that
    ``alpha = sigmoid(raw_alpha) in (0, 1)`` always holds; they may be fixed
    (for clean, known horizons) or learned (and the discovered tau is logged).

    Args:
        embed_dim: latent dimensionality (kept for interface symmetry / checks).
        tau_fast: initial effective horizon of the fast bank, in steps.
        tau_slow: initial effective horizon of the slow bank, in steps.
        learnable: if True, alpha_f / alpha_s are learned; else they are buffers.
    """

    def __init__(
        self,
        embed_dim: int,
        tau_fast: float = 2.0,
        tau_slow: float = 20.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.learnable = learnable

        a_f = tau_to_alpha(tau_fast)
        a_s = tau_to_alpha(tau_slow)
        raw_f = math.log(a_f / (1.0 - a_f))   # logit(alpha_fast)
        raw_s = math.log(a_s / (1.0 - a_s))   # logit(alpha_slow)

        if learnable:
            self.raw_alpha_fast = nn.Parameter(torch.tensor(raw_f, dtype=torch.float32))
            self.raw_alpha_slow = nn.Parameter(torch.tensor(raw_s, dtype=torch.float32))
        else:
            self.register_buffer('raw_alpha_fast', torch.tensor(raw_f, dtype=torch.float32))
            self.register_buffer('raw_alpha_slow', torch.tensor(raw_s, dtype=torch.float32))

    @property
    def alpha_fast(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha_fast)

    @property
    def alpha_slow(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha_slow)

    def horizons(self) -> Dict[str, float]:
        """Return current alphas and their effective horizons tau (for logging)."""
        with torch.no_grad():
            af, as_ = self.alpha_fast, self.alpha_slow
            return {
                'alpha_fast': float(af),
                'alpha_slow': float(as_),
                'tau_fast': float(alpha_to_tau(af)),
                'tau_slow': float(alpha_to_tau(as_)),
            }

    @staticmethod
    def _scan(z: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Causal EMA scan implementing Eq. (1) along the time axis.

        Warm-started with m_0 = z_0 so the memory is well defined at t=0 and the
        early estimate is unbiased. Differentiable through `alpha` (sequence lengths
        here are short, <= ~64, so a Python scan is fine and exact).

        Args:
            z: (B, T, D) latent sequence.
            alpha: scalar EMA rate in (0, 1).
        Returns:
            m: (B, T, D) memory states, m[:, t] = m_t.
        """
        B, T, D = z.shape
        one_minus = 1.0 - alpha
        out = [z[:, 0]]
        m_prev = z[:, 0]
        for t in range(1, T):
            m_prev = one_minus * m_prev + alpha * z[:, t]
            out.append(m_prev)
        return torch.stack(out, dim=1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute fast and slow memory states for a latent sequence.

        Args:
            z: (B, T, D) latent sequence.
        Returns:
            (m_fast, m_slow), each (B, T, D).
        """
        m_fast = self._scan(z, self.alpha_fast)
        m_slow = self._scan(z, self.alpha_slow)
        return m_fast, m_slow

    def step(
        self,
        m_fast_prev: torch.Tensor,
        m_slow_prev: torch.Tensor,
        z_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step EMA update for autoregressive rollout / planning.

        Args:
            m_fast_prev, m_slow_prev: (B, D) previous memory states.
            z_t: (B, D) current latent.
        Returns:
            (m_fast, m_slow) updated states, each (B, D).
        """
        af, as_ = self.alpha_fast, self.alpha_slow
        m_fast = (1.0 - af) * m_fast_prev + af * z_t
        m_slow = (1.0 - as_) * m_slow_prev + as_ * z_t
        return m_fast, m_slow

    def kernel(self, length: int, which: str = 'slow') -> torch.Tensor:
        """Return the (un-normalized truncated) exponential memory kernel K(k), Eq. (2),
        for k = 0..length-1. Useful for plotting the memory profile."""
        alpha = self.alpha_slow if which == 'slow' else self.alpha_fast
        k = torch.arange(length, device=alpha.device, dtype=alpha.dtype)
        return alpha * (1.0 - alpha) ** k


class MemoryFusion(nn.Module):
    """Inject the two memory banks into the latent stream before the predictor.

    Fused latent (the only thing the short-context predictor ever sees):

        z~_t = z_t + 1[short] * W_f m^f_t + 1[long] * W_s m^s_t                 (4)

    W_f and W_s are zero-initialized, so at the start of training the model is *exactly*
    the memoryless baseline JEPA and learns to recruit memory only as it helps -- the
    same zero-init philosophy as the predictor's AdaLN. The `mode` flag gives the four
    ablations for free without changing any other code:

        none  -> z~ = z                  (baseline, memoryless)
        short -> z~ = z + W_f m^f
        long  -> z~ = z + W_s m^s
        both  -> z~ = z + W_f m^f + W_s m^s
    """

    MODES = ('none', 'short', 'long', 'both')

    def __init__(self, embed_dim: int, mode: str = 'both'):
        super().__init__()
        assert mode in self.MODES, f"mode must be one of {self.MODES}, got {mode}"
        self.mode = mode
        self.embed_dim = embed_dim
        self.w_fast = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_slow = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.zeros_(self.w_fast.weight)
        nn.init.zeros_(self.w_slow.weight)

    @property
    def use_fast(self) -> bool:
        return self.mode in ('short', 'both')

    @property
    def use_slow(self) -> bool:
        return self.mode in ('long', 'both')

    def forward(
        self,
        z: torch.Tensor,
        m_fast: torch.Tensor,
        m_slow: torch.Tensor,
        ablate_fast: bool = False,
        ablate_slow: bool = False,
    ) -> torch.Tensor:
        """Apply Eq. (4).

        Args:
            z, m_fast, m_slow: (..., D) tensors (broadcasting over leading dims).
            ablate_fast / ablate_slow: at eval time, drop a bank to *measure its causal
                influence on the prediction/decision* (used by the probing/visualization).
        Returns:
            z~ with the same shape as z.
        """
        out = z
        if self.use_fast and not ablate_fast:
            out = out + self.w_fast(m_fast)
        if self.use_slow and not ablate_slow:
            out = out + self.w_slow(m_slow)
        return out
