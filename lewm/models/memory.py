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


class MultiTimescaleMemory(nn.Module):
    """E3 baseline: K leaky integrators at log-spaced FIXED horizons + K zero-init read-outs.

    Generalizes the two-bank design to a fixed multi-scale bank (e.g. tau in {2,4,8,16,32,64}),
    so the model can *read from* a spectrum of horizons without anyone choosing tau per task --
    the fix for "learned alpha does not self-tune". Decay rates are fixed (buffers)."""

    def __init__(self, embed_dim: int, taus=(2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.embed_dim = embed_dim
        self.taus = list(taus)
        alphas = [tau_to_alpha(t) for t in self.taus]
        self.register_buffer('alphas', torch.tensor(alphas, dtype=torch.float32))
        self.readouts = nn.ModuleList([nn.Linear(embed_dim, embed_dim, bias=False) for _ in self.taus])
        for ro in self.readouts:
            nn.init.zeros_(ro.weight)

    def banks(self, z: torch.Tensor):
        return [TwoTimescaleMemory._scan(z, self.alphas[k]) for k in range(len(self.taus))]

    def fuse(self, z: torch.Tensor, banks) -> torch.Tensor:
        out = z
        for k, m in enumerate(banks):
            out = out + self.readouts[k](m)
        return out

    def horizons(self):
        return {'tau_fast': float(self.taus[0]), 'tau_slow': float(self.taus[-1]),
                'alpha_fast': float(self.alphas[0]), 'alpha_slow': float(self.alphas[-1]),
                'n_banks': len(self.taus)}


class SelectiveMultiTimescaleMemory(nn.Module):
    """Selective Multi-Timescale (SMT) memory -- a *learnable, scalable* short/long memory.

    Motivation (from our controlled study): a FIXED log-spaced bank of EMA horizons beats every
    *learned* memory here, and a learnable scalar decay alpha does NOT self-tune (weak gradient on
    a decay rate). So instead of learning the decays, we keep the fixed log-spaced basis (the
    reliable prior) and move ALL learnability to *input-conditioned gating*:

        write/input gate   i_t   = sigmoid(W_i z_t)                         (what to store)
        bank-k recurrence  m^k_t = (1-a_k) m^k_{t-1} + a_k (i_t ⊙ z_t)      (a_k FIXED, log-spaced)
        read router        r_t   = softmax(W_r z_t / T)  ∈ Δ^{K-1}          (which horizon to read)
        memory read-out    o_t   = W_o ( Σ_k r_{t,k} · m^k_t )              (W_o zero-init)

    This is a *selective diagonal linear SSM over a fixed timescale basis*: O(L·K·D), linear in
    sequence length and parallelizable via an associative scan; the only learned, content-dependent
    parts are a small write gate and a K-way router (a "mixture of timescales" with learned, per-step
    routing). Decays stay fixed, so horizons are known and the router r_t is a per-step distribution
    over KNOWN horizons -> directly interpretable as short-vs-long selection. W_o is zero-initialized,
    so training begins exactly at the memoryless baseline (same philosophy as MemoryFusion).

    Differs from: Mamba/S6 (learns input-dependent timescale Delta), Mega (learns EMA coefficients),
    HGRN2 (learns data-dependent decay), RetNet (fixed multi-scale decay, no selectivity). Here the
    decays are fixed and the *selection over* them is what is learned.
    """

    def __init__(self, embed_dim: int, taus=(2, 4, 8, 16, 32, 64), router_temp: float = 1.0,
                 router_mode: str = 'softmax'):
        super().__init__()
        self.embed_dim = embed_dim
        self.taus = list(taus)
        self.K = len(self.taus)
        self.router_temp = router_temp
        # 'softmax' = convex mixture over horizons (v1); 'sigmoid' = independent additive gates so
        # every bank can contribute fully (v2) -- like the fixed K-bank but input-conditioned.
        self.router_mode = router_mode
        alphas = [tau_to_alpha(t) for t in self.taus]
        self.register_buffer('alphas', torch.tensor(alphas, dtype=torch.float32))  # FIXED
        self.in_gate = nn.Linear(embed_dim, embed_dim)        # learned write gate (what to store)
        nn.init.constant_(self.in_gate.bias, 1.0)             # start ~open (sigmoid~0.73): populate memory
        self.router = nn.Linear(embed_dim, self.K)            # learned read router (which horizon)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        # NOTE: a *small* (not zero) read-out init -- the router/write-gate sit upstream of this
        # multiplicative read-out, so a zero-init would give them exactly zero gradient at step 0.
        # Small init keeps the model near the memoryless baseline yet trains every part from step 1.
        nn.init.normal_(self.out.weight, std=1e-2)

    def route_weights(self, z: torch.Tensor) -> torch.Tensor:
        """Per-step router weights over the K fixed horizons (B,L,K) -- for read-out & interpretability.
        softmax -> convex mixture (sums to 1); sigmoid -> independent additive gates."""
        s = self.router(z) / self.router_temp
        return torch.softmax(s, dim=-1) if self.router_mode == 'softmax' else torch.sigmoid(s)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return the routed memory read-out (B,L,D) to be fused into the predictor input."""
        zi = torch.sigmoid(self.in_gate(z)) * z               # gated (written) input
        banks = torch.stack([TwoTimescaleMemory._scan(zi, self.alphas[k]) for k in range(self.K)],
                            dim=2)                            # (B,L,K,D)
        w = self.route_weights(z).unsqueeze(-1)               # (B,L,K,1)
        return (w * banks).sum(dim=2)                         # (B,L,D)

    def fuse(self, z: torch.Tensor, mixed: torch.Tensor) -> torch.Tensor:
        return z + self.out(mixed)

    def horizons(self):
        return {'tau_fast': float(self.taus[0]), 'tau_slow': float(self.taus[-1]),
                'alpha_fast': float(self.alphas[0]), 'alpha_slow': float(self.alphas[-1]),
                'n_banks': self.K}


class OCSMTMemory(nn.Module):
    """OC-SMT: Over-Complete fixed log-spaced EMA basis + L0 (hard-concrete) sparse READ gates.

    Makes the bank a *learnable, variable-size* active set with NO constant K (docs/LEARNABLE_MEMORY.md
    §8). We keep the decays FIXED (learning them fails, §5.4) but enlarge the basis to an over-complete
    grid of M horizons and learn a per-bank hard-concrete L0 gate, so 'how many banks are on' is a
    differentiable, penalized quantity rather than a hand-set constant:

        gate logit   l_{t,m} = (W_g z_t)_m
        hard-concrete g_{t,m} ∈ {0} ∪ (0,1]   (exact zeros via the stretch [gamma,zeta], Louizos 2018)
        write gate   i_t = sigmoid(W_i z_t) ;  m^m_t = (1-a_m) m^m_{t-1} + a_m (i_t ⊙ z_t)   (a_m FIXED)
        read-out     o_t = W_o( Σ_m g_{t,m} m^m_t ) ;  z~_t = z_t + o_t        (additive, as SMT-v2)
        L0 penalty   λ0 · Σ_m P(g_{t,m} > 0)                                   (cost for being dense)

    The L0 penalty is the anti-collapse pressure the SMT softmax router lacked: a uniform/dense gate
    now *costs* loss, so SGD turns banks off and the effective size emerges from data (≈0 on a Markov
    env, a few long horizons on T-Maze/Distractor). Decays never change; state M·D (constant in L)."""

    BETA, GAMMA, ZETA = 2.0 / 3.0, -0.1, 1.1            # hard-concrete temperature + stretch

    def __init__(self, embed_dim: int, M: int = 28, tau_min: float = 1.5, tau_max: float = 256.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.M = M
        taus = torch.logspace(math.log10(tau_min), math.log10(tau_max), M)
        self.register_buffer('taus', taus)
        self.register_buffer('alphas', 1.0 - torch.exp(-1.0 / taus))   # (M,) FIXED
        self.in_gate = nn.Linear(embed_dim, embed_dim)
        nn.init.constant_(self.in_gate.bias, 1.0)         # write gate starts ~open
        self.gate = nn.Linear(embed_dim, M)               # W_g: hard-concrete gate logits
        nn.init.constant_(self.gate.bias, 1.0)            # start banks OPEN so memory is used; an ANNEALED
        #   L0 (populate-then-sparsify, in train_memory) then prunes. NOTE (empirical): on clean short
        #   tasks the L0 is bistable (dense-or-collapse) -- the over-complete banks are cheap+useful, so
        #   aggressive pruning collapses memory; the practical operating point keeps most banks. See §9.
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.normal_(self.out.weight, std=5e-4)        # small: M~28 open banks sum -> keep near baseline
        self.last_l0 = None                               # stashed expected #open gates (for the loss)

    def _sample_gate(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + logits) / self.BETA)
        else:
            s = torch.sigmoid(logits)                     # deterministic at eval
        return (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0, 1)

    def route_weights(self, z: torch.Tensor) -> torch.Tensor:
        """Deterministic gate (B,L,M) for eval / interpretability; a bank is 'active' iff > 0."""
        s = torch.sigmoid(self.gate(z))
        return (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.gate(z)                             # (B,L,M)
        g = self._sample_gate(logits)
        # expected number of open gates per step (Louizos L0): P(g>0)=sigmoid(logit - beta*log(-gamma/zeta))
        p_open = torch.sigmoid(logits - self.BETA * math.log(-self.GAMMA / self.ZETA))
        self.last_l0 = p_open.mean(dim=(0, 1)).sum()      # sum over M of mean-over-(B,L) open prob
        zi = torch.sigmoid(self.in_gate(z)) * z
        banks = torch.stack([TwoTimescaleMemory._scan(zi, self.alphas[m]) for m in range(self.M)], dim=2)
        return (g.unsqueeze(-1) * banks).sum(dim=2)       # (B,L,D)

    def fuse(self, z: torch.Tensor, mixed: torch.Tensor) -> torch.Tensor:
        return z + self.out(mixed)

    @torch.no_grad()
    def active_count(self, z: torch.Tensor, thresh: float = 1e-3) -> torch.Tensor:
        """Mean number of active (gate>thresh) banks per step -- the learned effective bank size."""
        return (self.route_weights(z) > thresh).float().sum(-1).mean()

    def horizons(self):
        return {'tau_fast': float(self.taus[0]), 'tau_slow': float(self.taus[-1]),
                'alpha_fast': float(self.alphas[0]), 'alpha_slow': float(self.alphas[-1]),
                'n_banks': self.M}


class GRUMemory(nn.Module):
    """E2 baseline: a GRU over the latent stream; its (causal) hidden state is injected via a
    zero-init read-out -- the simplest *learned* recurrent memory, matched to the EMA interface."""

    def __init__(self, embed_dim: int, hidden: int = None):
        super().__init__()
        h = hidden or embed_dim
        self.gru = nn.GRU(embed_dim, h, batch_first=True)
        self.readout = nn.Linear(h, embed_dim, bias=False)
        nn.init.zeros_(self.readout.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(z)            # (B,T,h), causal
        return out

    def fuse(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return z + self.readout(h)

    def horizons(self):
        return {'tau_fast': float('nan'), 'tau_slow': float('nan'),
                'alpha_fast': float('nan'), 'alpha_slow': float('nan')}


class SSMMemory(nn.Module):
    """E2(a) baseline: a learned diagonal linear state-space / RetNet-lite memory — a
    *per-channel* leaky integrator with learned decay and an input projection. Generalizes
    the fixed scalar EMA to D learned decays spanning a range (multi-scale within one bank).

        u_t = W_in z_t ;  s_t = (1-a) ⊙ s_{t-1} + a ⊙ u_t ;  m_t = W_out s_t   (a ∈ (0,1)^D)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, embed_dim)
        # init per-channel decay spread across horizons tau ~ logspace(2..64)
        taus = torch.logspace(math.log10(2), math.log10(64), embed_dim)
        a = 1.0 - torch.exp(-1.0 / taus)
        self.raw_decay = nn.Parameter(torch.log(a / (1 - a)))         # logit(alpha) per channel
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        u = self.in_proj(z)
        a = torch.sigmoid(self.raw_decay)                            # (D,)
        s = torch.zeros(z.shape[0], z.shape[2], device=z.device, dtype=z.dtype)
        out = []
        for t in range(z.shape[1]):
            s = (1 - a) * s + a * u[:, t]
            out.append(s)
        return torch.stack(out, dim=1)

    def fuse(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return z + self.out(m)

    def horizons(self):
        return {'tau_fast': float('nan'), 'tau_slow': float('nan'),
                'alpha_fast': float('nan'), 'alpha_slow': float('nan')}


class RetrievalMemory(nn.Module):
    """E2(a) baseline: episodic retrieval — causal self-attention over the stored stream of
    past latents (a differentiable FIFO latent cache), injected via a zero-init read-out.
    Tests whether attending over *raw* past latents beats exponential compression."""

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        L = z.shape[1]
        mask = torch.triu(torch.ones(L, L, device=z.device), diagonal=1).bool()  # causal
        m, _ = self.attn(z, z, z, attn_mask=mask, need_weights=False)
        return m

    def fuse(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return z + self.out(m)

    def horizons(self):
        return {'tau_fast': float('nan'), 'tau_slow': float('nan'),
                'alpha_fast': float('nan'), 'alpha_slow': float('nan')}


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
