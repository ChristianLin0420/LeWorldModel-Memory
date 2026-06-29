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
import torch.nn.functional as F


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
    """Selective Multi-Timescale (SMT) memory over a fixed short/long basis.

    Motivation (from our controlled study): a FIXED log-spaced bank of EMA horizons beats every
    *learned* memory here, and a learnable scalar decay alpha does NOT self-tune (weak gradient on
    a decay rate). So instead of learning the decays, we keep the fixed log-spaced basis (the
    reliable prior) and move ALL learnability to *input-conditioned gating*:

        write/input gate   i_t   = sigmoid(W_i z_t)                         (what to store)
        bank-k recurrence  m^k_t = (1-a_k) m^k_{t-1} + a_k (i_t ⊙ z_t)      (a_k FIXED, log-spaced)
        read router        r_t   = g(W_r z_t / T), g=softmax or sigmoid      (which horizon to read)
        memory read-out    o_t   = W_o ( Σ_k r_{t,k} · m^k_t )              (W_o small-init)

    This is a diagonal linear SSM over a fixed timescale basis: O(L·K·D), linear in sequence
    length. The recurrence admits an associative scan, although this implementation is sequential.
    A write gate and K-way router are learned functions of the input; that mechanism permits
    content dependence but does not guarantee it. In the experiments recorded in
    docs/LEARNABLE_MEMORY.md, their outputs become almost static. Decays stay fixed, so horizons
    are known and the router remains directly measurable. W_o uses a small nonzero initialization
    so the upstream router and write gate receive gradients from step one.

    Differs from: Mamba/S6 (learns input-dependent timescale Delta), Mega (learns EMA coefficients),
    HGRN2 (learns data-dependent decay), RetNet (fixed multi-scale decay with content-dependent
    Q/K/V). Here the decays are fixed and explicit write/read gates over EMA banks are learned.
    """

    def __init__(self, embed_dim: int, taus=(2, 4, 8, 16, 32, 64), router_temp: float = 1.0,
                 router_mode: str = 'softmax'):
        super().__init__()
        self.embed_dim = embed_dim
        self.taus = list(taus)
        self.K = len(self.taus)
        self.router_temp = router_temp
        # 'softmax' = convex mixture over horizons (v1); 'scaled_softmax' keeps the same convex
        # relative weights but matches sigmoid's initial total read mass (K/2), separating routing
        # geometry from the v1-v2 amplitude change; 'sigmoid' = independent additive gates (v2).
        if router_mode not in {'softmax', 'scaled_softmax', 'sigmoid'}:
            raise ValueError(f"unknown SMT router mode '{router_mode}'")
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
        softmax -> convex mixture (sums to 1); scaled_softmax -> the same relative mixture with
        total mass K/2; sigmoid -> independent additive gates."""
        s = self.router(z) / self.router_temp
        if self.router_mode == 'softmax':
            return torch.softmax(s, dim=-1)
        if self.router_mode == 'scaled_softmax':
            return torch.softmax(s, dim=-1) * (self.K / 2.0)
        return torch.sigmoid(s)

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


class SelectiveUpdateMemoryV3(nn.Module):
    """Fixed EMA basis with a scalar *true selective-update* gate.

    SMT-v1/v2 multiplies the value being written but leaves the old state decay active.  A closed
    gate therefore writes zero while erasing the stored state.  V3 instead gates the complete EMA
    update:

        g_t = sigmoid(w_g^T (LayerNorm(z_t) + e) + b_g)
        m_t^k = (1 - a_k g_t) m_{t-1}^k + a_k g_t z_t

    Thus ``g_t=0`` exactly freezes every bank and ``g_t=1`` recovers the ordinary fixed EMA.  The
    ``old_update`` control uses the same dynamic conditioner but the old erasing recurrence,
    ``m_t^k=(1-a_k)m_{t-1}^k+a_k(g_t z_t)``.

    All variants instantiate exactly the same parameters.  ``static`` removes only the input
    conditioning by evaluating the same gate at the learned offset ``e``.  ``oracle`` replaces the
    learned gate with an explicit visibility/update mask and fails closed when that mask is absent.
    A global simplex mixture and parameter-free RMS normalization prevent per-step read mass from
    changing the residual amplitude.  One shared zero-initialized output projection is used for all
    banks and variants.
    """

    MODES = {'dynamic', 'static', 'oracle', 'old_update'}

    def __init__(self, embed_dim: int, taus=(2, 4, 8, 16, 32, 64),
                 mode: str = 'dynamic', rms_eps: float = 1e-6):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown SMT-v3 mode '{mode}'")
        if not taus:
            raise ValueError('SMT-v3 requires at least one fixed horizon')
        if any(float(tau) <= 0 for tau in taus):
            raise ValueError(f'SMT-v3 horizons must be positive, got {tuple(taus)}')
        if rms_eps <= 0:
            raise ValueError(f'rms_eps must be positive, got {rms_eps}')

        self.embed_dim = embed_dim
        self.taus = list(taus)
        self.K = len(self.taus)
        self.mode = mode
        self.rms_eps = float(rms_eps)
        self.register_buffer(
            'alphas', torch.tensor([tau_to_alpha(t) for t in self.taus], dtype=torch.float32))

        # Every mode owns these exact parameters.  The learned offset gives the static control an
        # active, trainable conditioner without adding parameters that the dynamic model lacks.
        self.conditioner_offset = nn.Parameter(torch.zeros(embed_dim))
        self.write_gate = nn.Linear(embed_dim, 1)
        nn.init.constant_(self.write_gate.bias, 2.0)       # start mostly open: sigmoid(2)=0.881
        self.route_logits = nn.Parameter(torch.zeros(self.K))
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def _validate_z(self, z: torch.Tensor) -> Tuple[int, int, int]:
        if z.dim() != 3:
            raise ValueError(f'SMT-v3 expects z with shape (B,T,D), got {tuple(z.shape)}')
        B, T, D = z.shape
        if T < 1:
            raise ValueError('SMT-v3 requires a non-empty time dimension')
        if D != self.embed_dim:
            raise ValueError(f'SMT-v3 expected latent dim {self.embed_dim}, got {D}')
        return B, T, D

    @staticmethod
    def _check_unit_interval(values: torch.Tensor, name: str) -> None:
        if not torch.isfinite(values).all():
            raise ValueError(f'{name} contains non-finite values')
        if bool((values < 0).any()) or bool((values > 1).any()):
            raise ValueError(f'{name} must lie in [0,1]')

    def _coerce_gate(self, value, z: torch.Tensor, name: str) -> torch.Tensor:
        """Convert a scalar/broadcastable gate to the canonical ``(B,T,1)`` shape."""
        B, T, _ = z.shape
        gate = torch.as_tensor(value, device=z.device, dtype=z.dtype)
        if gate.dim() == 2 and tuple(gate.shape) == (B, T):
            gate = gate.unsqueeze(-1)
        target = (B, T, 1)
        try:
            gate = torch.broadcast_to(gate, target)
        except RuntimeError as exc:
            raise ValueError(
                f'{name} with shape {tuple(gate.shape)} is not broadcastable to {target}') from exc
        self._check_unit_interval(gate, name)
        return gate

    def gate_values(self, z: torch.Tensor,
                    memory_update_mask: torch.Tensor = None) -> torch.Tensor:
        """Return scalar write/update gates with shape ``(B,T,1)``.

        ``memory_update_mask`` is used only by the oracle variant.  True/one means update from the
        current observation and false/zero means freeze.  Other variants deliberately ignore it so
        callers may pass a common batch structure without leaking visibility into learned controls.
        """
        self._validate_z(z)
        if self.mode == 'oracle':
            if memory_update_mask is None:
                raise ValueError(
                    'SMT-v3 oracle requires an explicit memory_update_mask; refusing to infer '
                    'visibility from targets or inputs')
            return self._coerce_gate(memory_update_mask, z, 'memory_update_mask')

        if self.mode == 'static':
            conditioner = self.conditioner_offset.view(1, 1, -1).expand(
                z.shape[0], z.shape[1], -1)
        else:                                               # dynamic and old_update
            conditioner = F.layer_norm(z, (self.embed_dim,))
            conditioner = conditioner + self.conditioner_offset.view(1, 1, -1)
        return torch.sigmoid(self.write_gate(conditioner))

    def route_weights(self) -> torch.Tensor:
        """Global simplex weights over the fixed horizons, shape ``(K,)``."""
        return torch.softmax(self.route_logits, dim=0)

    def _scan_banks(self, z: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        """Return all recurrent states as ``(B,T,K,D)`` for the supplied canonical gates."""
        B, T, D = self._validate_z(z)
        if gates.shape != (B, T, 1):
            raise ValueError(f'expected gates {(B, T, 1)}, got {tuple(gates.shape)}')
        alpha = self.alphas.to(dtype=z.dtype).view(1, self.K, 1)
        m_prev = z[:, 0].unsqueeze(1).expand(-1, self.K, -1)
        states = [m_prev]
        for t in range(1, T):
            z_t = z[:, t].unsqueeze(1)                    # (B,1,D)
            g_t = gates[:, t].unsqueeze(1)                # (B,1,1)
            if self.mode == 'old_update':
                m_prev = (1.0 - alpha) * m_prev + alpha * (g_t * z_t)
            else:
                rate = alpha * g_t
                m_prev = (1.0 - rate) * m_prev + rate * z_t
            states.append(m_prev)
        return torch.stack(states, dim=1)

    def forward(self, z: torch.Tensor, memory_update_mask: torch.Tensor = None,
                gate_override=None) -> torch.Tensor:
        """Return the globally mixed, RMS-normalized memory read ``(B,T,D)``.

        ``gate_override`` accepts a scalar or any tensor broadcastable to ``(B,T,1)`` and is used
        for calibrated-mean counterfactuals.  The oracle remains strict: it validates that an
        explicit visibility mask was supplied before applying an override.
        """
        gates = self.gate_values(z, memory_update_mask=memory_update_mask)
        if gate_override is not None:
            gates = self._coerce_gate(gate_override, z, 'gate_override')
        banks = self._scan_banks(z, gates)
        route = self.route_weights().to(dtype=z.dtype).view(1, 1, self.K, 1)
        mixed = (banks * route).sum(dim=2)
        return mixed * torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + self.rms_eps)

    def fuse(self, z: torch.Tensor, mixed: torch.Tensor) -> torch.Tensor:
        return z + self.out(mixed)

    def horizons(self):
        return {'tau_fast': float(self.taus[0]), 'tau_slow': float(self.taus[-1]),
                'alpha_fast': float(self.alphas[0]), 'alpha_slow': float(self.alphas[-1]),
                'n_banks': self.K}


class HierarchicalActionConditionedMemory(nn.Module):
    """HACSM-v4: hierarchical selective memory with a causal action prior.

    Three full-width states use fixed structural horizons ``(2, 8, 32)``.  The action that maps
    observation ``t-1`` to observation ``t`` first advances every state, after which a selective
    observation correction is applied::

        x_t       = W_x z_t
        (d, v)    = split(W_a a_{t-1})
        p_t^k     = m_{t-1}^k + beta_k tanh(v + d * LN(m_{t-1}^k))
        g_t^k     = sigmoid((w_z . LN(z_t) + w_e . LN(x_t - p_t^k)) / sqrt(D) + b_k)
        m_t^k     = p_t^k + beta_k g_t^k (x_t - p_t^k)

    Here ``beta_k = 1-exp(-1/tau_k)`` is fixed.  The first state is warm-started with ``m_0=x_0``.
    A global simplex route mixes the levels, a parameter-free RMS normalization fixes its scale,
    and a zero-initialized bias-free projection provides identity-at-initialization residual fusion.

    ``dynamic``, ``static``, ``noaction``, ``single``, and ``oracle`` deliberately instantiate
    exactly the same parameters.  Static gates use only ``sigmoid(b_k)``; noaction retains the
    dynamic correction but zeros the action contribution; single retains the full recurrence but
    fixes the read route to the middle (tau=8) level; oracle gates require an explicit mask and
    fails closed.  Gate and action overrides are analysis-only inputs and are validated before use.
    In particular, an override never bypasses the oracle's explicit-mask requirement.

    ``forward`` returns the normalized mixed memory, matching the other memory classes in this
    module; call :meth:`fuse` for ``z + W_o mixed``.  ``action_rollout`` advances source states using
    only future actions, providing a leakage-free primitive for hierarchical self-supervised losses.
    """

    MODES = {'dynamic', 'static', 'noaction', 'single', 'oracle'}
    TAUS = (2.0, 8.0, 32.0)

    def __init__(self, embed_dim: int, action_dim: int, mode: str = 'dynamic',
                 rms_eps: float = 1e-6, taus=None):
        super().__init__()
        if not isinstance(embed_dim, int) or isinstance(embed_dim, bool) or embed_dim < 1:
            raise ValueError(f'embed_dim must be a positive integer, got {embed_dim!r}')
        if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim < 1:
            raise ValueError(f'action_dim must be a positive integer, got {action_dim!r}')
        if mode not in self.MODES:
            raise ValueError(f"unknown HACSM-v4 mode '{mode}'")
        if not math.isfinite(float(rms_eps)) or rms_eps <= 0:
            raise ValueError(f'rms_eps must be positive and finite, got {rms_eps}')
        selected_taus = self.TAUS if taus is None else tuple(float(tau) for tau in taus)
        if not selected_taus or any(not math.isfinite(tau) or tau <= 0 for tau in selected_taus):
            raise ValueError(f'taus must be a non-empty sequence of positive finite values, got {taus}')

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.mode = mode
        self.rms_eps = float(rms_eps)
        self.taus = list(selected_taus)
        self.K = len(self.taus)
        self.register_buffer(
            'betas',
            torch.tensor([tau_to_alpha(tau) for tau in self.taus], dtype=torch.float32),
        )

        # All transforms are shared across levels.  Functional layer normalization below is
        # intentionally affine-free: it keeps the hierarchy's parameter count transparent.
        self.W_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_a = nn.Linear(action_dim, 2 * embed_dim, bias=False)
        self.w_z = nn.Parameter(torch.zeros(embed_dim))
        self.w_e = nn.Parameter(torch.zeros(embed_dim))
        self.gate_bias = nn.Parameter(torch.full((self.K,), 2.0))
        self.route_logits = nn.Parameter(torch.zeros(self.K))
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.eye_(self.W_x.weight)
        nn.init.zeros_(self.W_a.weight)
        nn.init.zeros_(self.W_o.weight)

    @classmethod
    def expected_parameter_count(cls, embed_dim: int, action_dim: int, taus=None) -> int:
        """Return ``2D^2 + 2AD + 2D + 2K``, the exact trainable parameter count."""
        n_levels = len(cls.TAUS if taus is None else tuple(taus))
        return (2 * embed_dim * embed_dim + 2 * action_dim * embed_dim
                + 2 * embed_dim + 2 * n_levels)

    def parameter_count(self) -> int:
        """Trainable parameter count, useful when constructing matched controls."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_latents(self, z: torch.Tensor) -> Tuple[int, int, int]:
        if not isinstance(z, torch.Tensor) or z.dim() != 3:
            shape = tuple(z.shape) if isinstance(z, torch.Tensor) else type(z).__name__
            raise ValueError(f'HACSM-v4 expects z with shape (B,T,D), got {shape}')
        B, T, D = z.shape
        if B < 1 or T < 1:
            raise ValueError(f'HACSM-v4 requires non-empty batch and time dimensions, got {B}, {T}')
        if D != self.embed_dim:
            raise ValueError(f'HACSM-v4 expected latent dim {self.embed_dim}, got {D}')
        if not z.is_floating_point():
            raise ValueError(f'HACSM-v4 requires floating-point latents, got {z.dtype}')
        if not torch.isfinite(z).all():
            raise ValueError('z contains non-finite values')
        return B, T, D

    def _validate_actions(self, actions: torch.Tensor, B: int, steps: int,
                          *, name: str = 'actions') -> torch.Tensor:
        expected = (B, steps, self.action_dim)
        if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != expected:
            shape = tuple(actions.shape) if isinstance(actions, torch.Tensor) else type(actions).__name__
            raise ValueError(f'{name} must have shape {expected}, got {shape}')
        if not actions.is_floating_point():
            raise ValueError(f'{name} must be floating point, got {actions.dtype}')
        if not torch.isfinite(actions).all():
            raise ValueError(f'{name} contains non-finite values')
        return actions

    def _coerce_action_override(self, value, actions: torch.Tensor, name: str) -> torch.Tensor:
        """Replace actions with a finite scalar/broadcastable tensor for interventions."""
        override = torch.as_tensor(value, device=actions.device, dtype=actions.dtype)
        try:
            override = torch.broadcast_to(override, actions.shape)
        except RuntimeError as exc:
            raise ValueError(
                f'{name} with shape {tuple(override.shape)} is not broadcastable to '
                f'{tuple(actions.shape)}') from exc
        if not torch.isfinite(override).all():
            raise ValueError(f'{name} contains non-finite values')
        return override

    @staticmethod
    def _check_unit_interval(values: torch.Tensor, name: str) -> None:
        if not torch.isfinite(values).all():
            raise ValueError(f'{name} contains non-finite values')
        if bool((values < 0).any()) or bool((values > 1).any()):
            raise ValueError(f'{name} must lie in [0,1]')

    def _coerce_gates(self, value, B: int, T: int, *, device, dtype,
                      name: str) -> torch.Tensor:
        """Canonicalize scalar/per-step/per-level gates to ``(B,T,K,1)``."""
        gates = torch.as_tensor(value, device=device, dtype=dtype)
        if gates.dim() == 1 and tuple(gates.shape) == (self.K,):
            gates = gates.view(1, 1, self.K, 1)
        elif gates.dim() == 1 and tuple(gates.shape) == (T,):
            gates = gates.view(1, T, 1, 1)
        elif gates.dim() == 2 and tuple(gates.shape) == (T, self.K):
            gates = gates.unsqueeze(0).unsqueeze(-1)
        elif gates.dim() == 2 and tuple(gates.shape) == (B, T):
            gates = gates.unsqueeze(-1).unsqueeze(-1)
        elif gates.dim() == 2 and tuple(gates.shape) == (B, self.K):
            gates = gates.unsqueeze(1).unsqueeze(-1)
        elif gates.dim() == 3 and tuple(gates.shape) == (B, T, 1):
            gates = gates.unsqueeze(-2)
        elif gates.dim() == 3 and tuple(gates.shape) == (B, T, self.K):
            gates = gates.unsqueeze(-1)
        target = (B, T, self.K, 1)
        try:
            gates = torch.broadcast_to(gates, target)
        except RuntimeError as exc:
            raise ValueError(
                f'{name} with shape {tuple(gates.shape)} is not broadcastable to {target}') from exc
        self._check_unit_interval(gates, name)
        return gates

    def route_weights(self) -> torch.Tensor:
        """Return the learned global simplex route over the three levels, shape ``(K,)``."""
        if self.mode == 'single':
            # Keep route_logits instantiated for an exactly parameter-matched hierarchy ablation,
            # but make its read path a fixed one-hot selection of the middle (tau=8) state.
            route = torch.zeros_like(self.route_logits)
            route[self.K // 2] = 1.0
            return route
        return torch.softmax(self.route_logits, dim=0)

    def _validate_states(self, states: torch.Tensor) -> Tuple[int, int, int]:
        expected_tail = (self.K, self.embed_dim)
        if not isinstance(states, torch.Tensor) or states.dim() != 3:
            shape = tuple(states.shape) if isinstance(states, torch.Tensor) else type(states).__name__
            raise ValueError(f'states must have shape (B,{self.K},{self.embed_dim}), got {shape}')
        B, K, D = states.shape
        if B < 1 or (K, D) != expected_tail:
            raise ValueError(
                f'states must have shape (B,{self.K},{self.embed_dim}), got {tuple(states.shape)}')
        if not states.is_floating_point() or not torch.isfinite(states).all():
            raise ValueError('states must be finite floating-point values')
        return B, K, D

    def _action_prior_unchecked(self, states: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Internal transition after public boundaries have validated states/actions once."""
        B = states.shape[0]
        if self.mode == 'noaction':
            action_features = states.new_zeros(B, 2 * self.embed_dim)
        else:
            action_features = self.W_a(action)
        d, v = action_features.chunk(2, dim=-1)
        normalized = F.layer_norm(states, (self.embed_dim,))
        delta = torch.tanh(
            v.unsqueeze(1) + d.unsqueeze(1) * normalized
        )
        beta = self.betas.to(dtype=states.dtype).view(1, self.K, 1)
        return states + beta * delta

    def _action_prior(self, states: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Validated one-step transition; ``action`` maps source to returned state."""
        B, _, _ = self._validate_states(states)
        if not isinstance(action, torch.Tensor) or tuple(action.shape) != (B, self.action_dim):
            shape = tuple(action.shape) if isinstance(action, torch.Tensor) else type(action).__name__
            raise ValueError(f'action must have shape {(B, self.action_dim)}, got {shape}')
        if not action.is_floating_point() or not torch.isfinite(action).all():
            raise ValueError('action must contain finite floating-point values')
        action = action.to(device=states.device, dtype=states.dtype)
        return self._action_prior_unchecked(states, action)

    def transition(self, states: torch.Tensor, action: torch.Tensor,
                   action_override=None) -> torch.Tensor:
        """Public one-step action-only transition used by causal auxiliary objectives."""
        B, _, _ = self._validate_states(states)
        if not isinstance(action, torch.Tensor) or tuple(action.shape) != (B, self.action_dim):
            shape = tuple(action.shape) if isinstance(action, torch.Tensor) else type(action).__name__
            raise ValueError(f'action must have shape {(B, self.action_dim)}, got {shape}')
        if not action.is_floating_point() or not torch.isfinite(action).all():
            raise ValueError('action must contain finite floating-point values')
        if action_override is not None:
            action = self._coerce_action_override(action_override, action, 'action_override')
        return self._action_prior(states, action)

    def action_rollout(self, source_states: torch.Tensor, actions: torch.Tensor,
                       action_override=None) -> torch.Tensor:
        """Roll source states forward using actions only.

        ``actions[:, j]`` maps rollout state ``j`` to ``j+1``.  The returned tensor has shape
        ``(B,H,K,D)`` and excludes the supplied source, so index ``h-1`` is the horizon-``h`` state.
        No observation at or after the source is consumed.
        """
        B, _, _ = self._validate_states(source_states)
        if not isinstance(actions, torch.Tensor) or actions.dim() != 3:
            shape = tuple(actions.shape) if isinstance(actions, torch.Tensor) else type(actions).__name__
            raise ValueError(
                f'rollout actions must have shape (B,H,{self.action_dim}), got {shape}')
        actions = self._validate_actions(actions, B, actions.shape[1], name='rollout actions')
        actions = actions.to(device=source_states.device, dtype=source_states.dtype)
        if action_override is not None:
            actions = self._coerce_action_override(action_override, actions, 'action_override')

        state = source_states
        trajectory = []
        for step in range(actions.shape[1]):
            state = self._action_prior_unchecked(state, actions[:, step])
            trajectory.append(state)
        if not trajectory:
            return source_states.new_empty(B, 0, self.K, self.embed_dim)
        return torch.stack(trajectory, dim=1)

    def _dynamic_gate(self, z_t: torch.Tensor, x_t: torch.Tensor,
                      prior: torch.Tensor) -> torch.Tensor:
        normalized_z = F.layer_norm(z_t, (self.embed_dim,))
        normalized_error = F.layer_norm(x_t.unsqueeze(1) - prior, (self.embed_dim,))
        z_score = torch.einsum('bd,d->b', normalized_z, self.w_z).unsqueeze(-1)
        error_score = torch.einsum('bkd,d->bk', normalized_error, self.w_e)
        logits = ((z_score + error_score) / math.sqrt(self.embed_dim)
                  + self.gate_bias.view(1, self.K))
        return torch.sigmoid(logits).unsqueeze(-1)

    def forward(self, z: torch.Tensor, actions: torch.Tensor,
                memory_update_mask: torch.Tensor = None, gate_override=None,
                action_override=None, return_details: bool = False):
        """Return mixed memory, and optionally its complete causal trajectory.

        Args:
            z: latent observations ``(B,T,D)``.
            actions: transitions ``(B,T-1,A)``; ``actions[:,t]`` maps ``z_t`` to ``z_{t+1}``.
            memory_update_mask: explicit oracle correction mask.  Non-oracle modes ignore it.
            gate_override: scalar/tensor broadcastable to ``(B,T,K,1)``.
            action_override: finite scalar/tensor broadcastable to the action tensor; replaces it.
            return_details: return ``(mixed, details)`` when true.  Because ``m_0=x_0`` is an
                unconditional warm start, ``details['gates'][:,0]`` records the gate that would
                apply at that step for analysis but does not alter the initial state.
        """
        B, T, _ = self._validate_latents(z)
        actions = self._validate_actions(actions, B, T - 1)
        actions = actions.to(device=z.device, dtype=z.dtype)
        if action_override is not None:
            actions = self._coerce_action_override(action_override, actions, 'action_override')

        # Validate the oracle mask before considering an override: interventions may modify an
        # explicitly specified oracle, but can never turn it into an implicit one.
        oracle_gates = None
        if self.mode == 'oracle':
            if memory_update_mask is None:
                raise ValueError(
                    'HACSM-v4 oracle requires an explicit memory_update_mask; refusing to infer '
                    'visibility from targets or inputs')
            oracle_gates = self._coerce_gates(
                memory_update_mask, B, T, device=z.device, dtype=z.dtype,
                name='memory_update_mask')
        elif memory_update_mask is not None:
            # Non-oracle controls remain invariant to mask values, but validating a supplied mask
            # catches malformed shared batch plumbing instead of silently hiding it.
            self._coerce_gates(
                memory_update_mask, B, T, device=z.device, dtype=z.dtype,
                name='memory_update_mask')
        override_gates = None
        if gate_override is not None:
            override_gates = self._coerce_gates(
                gate_override, B, T, device=z.device, dtype=z.dtype, name='gate_override')

        x = self.W_x(z)
        initial = x[:, 0].unsqueeze(1).expand(-1, self.K, -1)
        states = [initial]
        priors = [initial]

        if override_gates is not None:
            gate_0 = override_gates[:, 0]
        elif self.mode == 'oracle':
            gate_0 = oracle_gates[:, 0]
        elif self.mode == 'static':
            gate_0 = torch.sigmoid(self.gate_bias).view(1, self.K, 1).expand(B, -1, -1)
        else:
            gate_0 = self._dynamic_gate(z[:, 0], x[:, 0], initial)
        gates = [gate_0]

        state = initial
        beta = self.betas.to(dtype=z.dtype).view(1, self.K, 1)
        for t in range(1, T):
            prior = self._action_prior_unchecked(state, actions[:, t - 1])
            if override_gates is not None:
                gate = override_gates[:, t]
            elif self.mode == 'oracle':
                gate = oracle_gates[:, t]
            elif self.mode == 'static':
                gate = torch.sigmoid(self.gate_bias).view(1, self.K, 1).expand(B, -1, -1)
            else:
                gate = self._dynamic_gate(z[:, t], x[:, t], prior)
            state = prior + beta * gate * (x[:, t].unsqueeze(1) - prior)
            priors.append(prior)
            states.append(state)
            gates.append(gate)

        state_sequence = torch.stack(states, dim=1)
        prior_sequence = torch.stack(priors, dim=1)
        gate_sequence = torch.stack(gates, dim=1)
        route = self.route_weights().to(dtype=z.dtype)
        mixed = (state_sequence * route.view(1, 1, self.K, 1)).sum(dim=2)
        mixed = mixed * torch.rsqrt(
            mixed.square().mean(dim=-1, keepdim=True) + self.rms_eps)

        if not return_details:
            return mixed
        details = {
            'x': x,
            'priors': prior_sequence,
            'states': state_sequence,
            'gates': gate_sequence,
            'route': route,
        }
        return mixed, details

    def fuse(self, z: torch.Tensor, mixed: torch.Tensor) -> torch.Tensor:
        """Zero-initialized residual fusion ``z + W_o mixed``."""
        if not isinstance(z, torch.Tensor) or not isinstance(mixed, torch.Tensor):
            raise ValueError('z and mixed must be tensors with shape (B,T,D)')
        if z.dim() != 3 or mixed.dim() != 3:
            raise ValueError(
                f'z and mixed must have shape (B,T,D), got {tuple(z.shape)} and '
                f'{tuple(mixed.shape)}')
        if z.shape != mixed.shape:
            raise ValueError(
                f'z and mixed must have identical shapes, got {tuple(z.shape)} and '
                f'{tuple(mixed.shape)}')
        if z.shape[-1] != self.embed_dim:
            raise ValueError(f'HACSM-v4 expected final dim {self.embed_dim}, got {z.shape[-1]}')
        if not z.is_floating_point() or not mixed.is_floating_point():
            raise ValueError('z and mixed must be floating-point tensors')
        if not torch.isfinite(z).all() or not torch.isfinite(mixed).all():
            raise ValueError('z and mixed must contain only finite values')
        return z + self.W_o(mixed)

    def horizons(self) -> Dict[str, float]:
        result = {
            'tau_fast': self.taus[0],
            'tau_slow': self.taus[-1],
            'alpha_fast': float(self.betas[0]),
            'alpha_slow': float(self.betas[-1]),
            'n_banks': self.K,
        }
        for index, (tau, beta) in enumerate(zip(self.taus, self.betas)):
            result[f'tau_{index}'] = float(tau)
            result[f'beta_{index}'] = float(beta)
        return result


# Short name for experiment configuration and checkpoint metadata.
HACSMv4Memory = HierarchicalActionConditionedMemory


class HierarchicalCounterfactualRecoveryMemory(HierarchicalActionConditionedMemory):
    """HACSSM-v7 inference memory for hierarchical counterfactual recovery.

    V7 keeps the successful fixed ``tau=(2,8)`` predict/correct hierarchy, but gives each
    level its own action transition and learns a per-level shrinkage between the exact static
    and dynamic correction experts::

        static_k  = sigmoid(b_k)
        dynamic_k = sigmoid(b_k + innovation_t^k)
        gate_t^k  = (1-rho_k) static_k + rho_k dynamic_k

    ``rho_k=sigmoid(shrink_logits_k)`` is initialized to one half.  The convex gate is useful
    beyond the V7 objective: V6 found that static correction wins three environments while
    dynamic correction wins two.  All modes instantiate the same tensors.  ``sharedaction``
    averages the two level-specific action features before using them, ``noshrink`` fixes
    ``rho=1``, ``noaction`` zeros the recurrent action features, and ``single`` fixes the read
    to the medium state.  Other mode names are training-objective controls with identical
    online inference.
    """

    MODES = {
        'dynamic', 'noaux', 'sharedaction', 'noshrink', 'actiononly', 'uniform',
        'norecovery', 'noaction', 'single',
    }
    TAUS = (2.0, 8.0)

    def __init__(self, embed_dim: int, action_dim: int, mode: str = 'dynamic',
                 rms_eps: float = 1e-6):
        if mode not in self.MODES:
            raise ValueError(f"unknown HACSSM-v7 mode '{mode}'")
        parent_mode = mode if mode in {'noaction', 'single'} else 'dynamic'
        super().__init__(
            embed_dim=embed_dim, action_dim=action_dim, mode=parent_mode,
            rms_eps=rms_eps, taus=self.TAUS)
        self.v7_mode = mode

        # Replace V4's shared A->2D transition by one exactly matched A->(K*2D) tensor.
        self.W_a = nn.Linear(action_dim, self.K * 2 * embed_dim, bias=False)
        self.shrink_logits = nn.Parameter(torch.zeros(self.K))
        nn.init.zeros_(self.W_a.weight)

    @classmethod
    def expected_parameter_count(cls, embed_dim: int, action_dim: int) -> int:
        """Return ``2D^2 + 4AD + 2D + 3K`` for K=2."""
        K = len(cls.TAUS)
        return (2 * embed_dim * embed_dim + 2 * K * action_dim * embed_dim
                + 2 * embed_dim + 3 * K)

    def shrinkage(self) -> torch.Tensor:
        if self.v7_mode == 'noshrink':
            return torch.ones_like(self.shrink_logits)
        return torch.sigmoid(self.shrink_logits)

    def _action_prior_unchecked(self, states: torch.Tensor,
                                action: torch.Tensor) -> torch.Tensor:
        B = states.shape[0]
        if self.v7_mode == 'noaction':
            features = states.new_zeros(B, self.K, 2 * self.embed_dim)
        else:
            features = self.W_a(action).view(B, self.K, 2 * self.embed_dim)
            if self.v7_mode == 'sharedaction':
                features = features.mean(dim=1, keepdim=True).expand(-1, self.K, -1)
        d, v = features.chunk(2, dim=-1)
        normalized = F.layer_norm(states, (self.embed_dim,))
        delta = torch.tanh(v + d * normalized)
        beta = self.betas.to(device=states.device, dtype=states.dtype).view(1, self.K, 1)
        return states + beta * delta

    def _dynamic_gate(self, z_t: torch.Tensor, x_t: torch.Tensor,
                      prior: torch.Tensor) -> torch.Tensor:
        dynamic = super()._dynamic_gate(z_t, x_t, prior)
        static = torch.sigmoid(self.gate_bias).view(1, self.K, 1).expand_as(dynamic)
        rho = self.shrinkage().to(device=z_t.device, dtype=z_t.dtype).view(1, self.K, 1)
        return (1.0 - rho) * static + rho * dynamic

    def correction_step(self, states: torch.Tensor, z_t: torch.Tensor,
                        action: torch.Tensor, *, x_t: torch.Tensor = None,
                        action_override=None, gate_override=None) -> tuple[torch.Tensor, ...]:
        """Advance by one action and correct from one observation.

        ``x_t`` may be a detached precomputed ``W_x z_t`` during V7 self-supervision,
        preventing the auxiliary from updating ``W_x`` while retaining gradients to the
        action and correction parameters.
        """
        B, _, _ = self._validate_states(states)
        if not isinstance(z_t, torch.Tensor) or tuple(z_t.shape) != (B, self.embed_dim):
            shape = tuple(z_t.shape) if isinstance(z_t, torch.Tensor) else type(z_t).__name__
            raise ValueError(f'z_t must have shape {(B, self.embed_dim)}, got {shape}')
        if not z_t.is_floating_point() or not torch.isfinite(z_t).all():
            raise ValueError('z_t must contain finite floating-point values')
        if not isinstance(action, torch.Tensor) or tuple(action.shape) != (B, self.action_dim):
            shape = tuple(action.shape) if isinstance(action, torch.Tensor) else type(action).__name__
            raise ValueError(f'action must have shape {(B, self.action_dim)}, got {shape}')
        if not action.is_floating_point() or not torch.isfinite(action).all():
            raise ValueError('action must contain finite floating-point values')
        action = action.to(device=states.device, dtype=states.dtype)
        z_t = z_t.to(device=states.device, dtype=states.dtype)
        if action_override is not None:
            action = self._coerce_action_override(action_override, action, 'action_override')
        prior = self._action_prior_unchecked(states, action)
        if x_t is None:
            x_t = self.W_x(z_t)
        else:
            if not isinstance(x_t, torch.Tensor) or tuple(x_t.shape) != (B, self.embed_dim):
                shape = tuple(x_t.shape) if isinstance(x_t, torch.Tensor) else type(x_t).__name__
                raise ValueError(f'x_t must have shape {(B, self.embed_dim)}, got {shape}')
            x_t = x_t.to(device=states.device, dtype=states.dtype)
        if gate_override is None:
            gate = self._dynamic_gate(z_t, x_t, prior)
        else:
            gate = self._coerce_gates(
                gate_override, B, 1, device=states.device, dtype=states.dtype,
                name='gate_override')[:, 0]
        beta = self.betas.to(device=states.device, dtype=states.dtype).view(1, self.K, 1)
        posterior = prior + beta * gate * (x_t.unsqueeze(1) - prior)
        return posterior, prior, gate

    def horizons(self) -> Dict[str, float]:
        result = super().horizons()
        rho = self.shrinkage().detach()
        action_heads = self.W_a.weight.detach().view(
            self.K, 2 * self.embed_dim, self.action_dim)
        result.update({
            'rho_fast': float(rho[0]),
            'rho_medium': float(rho[1]),
            'action_head_fast_norm': float(action_heads[0].norm()),
            'action_head_medium_norm': float(action_heads[1].norm()),
            'action_head_cosine': float(F.cosine_similarity(
                action_heads[0].reshape(1, -1),
                action_heads[1].reshape(1, -1), dim=-1)[0]),
        })
        return result


HCRDv7Memory = HierarchicalCounterfactualRecoveryMemory


class SharedActionShrinkageMemory(HierarchicalActionConditionedMemory):
    """HACSSM-v8 shared-action shrinkage predict/correct memory.

    V8 retains only the V7 inference mechanisms supported by the completed ablations: two
    fixed rates ``tau=(2,8)``, causal action transport, per-level convex shrinkage between
    static and dynamic correction, and a joint read.  The nominated mode has one *physical*
    ``A -> 2D`` action projection shared by both levels.  It has no teacher and defines no
    internal-state auxiliary objective; those are model/trainer concerns and deliberately do
    not appear in this module.

    ``rho1`` and ``rho0`` are exact dynamic/static shrinkage endpoints. ``levelaction`` restores
    V7's separate heads, while ``redundant`` keeps the same wide tensor but averages its heads
    before either transition.  The latter is an optimization/parameterization receipt for the
    compact shared head, not the nominated architecture.
    """

    MODES = {
        # ``dynamic`` is required internally because the parent constructor validates through
        # the subclass mode set.  It is not exposed by the CLI or the seven-design V8 protocol.
        'dynamic', 'learned', 'rho1', 'rho0', 'levelaction', 'redundant',
        'noaction', 'single',
    }
    TAUS = (2.0, 8.0)

    def __init__(self, embed_dim: int, action_dim: int, mode: str = 'learned',
                 rms_eps: float = 1e-6):
        if mode not in self.MODES:
            raise ValueError(f"unknown HACSSM-v8 mode '{mode}'")
        parent_mode = mode if mode in {'noaction', 'single'} else 'dynamic'
        super().__init__(
            embed_dim=embed_dim, action_dim=action_dim, mode=parent_mode,
            rms_eps=rms_eps, taus=self.TAUS)
        self.v8_mode = mode
        self.shrink_logits = nn.Parameter(torch.zeros(self.K))

        # The nominated/shared endpoint modes retain the parent's physical A->2D map.  Only the
        # two explicit action-parameterization controls instantiate V7's A->(K*2D) tensor.
        if mode in {'levelaction', 'redundant'}:
            self.W_a = nn.Linear(action_dim, self.K * 2 * embed_dim, bias=False)
            nn.init.zeros_(self.W_a.weight)

    @classmethod
    def expected_parameter_count(cls, embed_dim: int, action_dim: int,
                                 *, wide_action: bool = False) -> int:
        """Return ``2D^2 + 2AD + 2D + 3K`` (or ``2D^2 + 2KAD + 2D + 3K``)."""
        K = len(cls.TAUS)
        action_parameters = 2 * (K if wide_action else 1) * action_dim * embed_dim
        return (2 * embed_dim * embed_dim + action_parameters
                + 2 * embed_dim + 3 * K)

    def shrinkage(self) -> torch.Tensor:
        if self.v8_mode == 'rho1':
            return torch.ones_like(self.shrink_logits)
        if self.v8_mode == 'rho0':
            return torch.zeros_like(self.shrink_logits)
        return torch.sigmoid(self.shrink_logits)

    def _action_features(self, action: torch.Tensor) -> torch.Tensor:
        """Return functional action features as ``(B,K,2D)`` for exact mode auditing."""
        B = action.shape[0]
        if self.v8_mode == 'noaction':
            return action.new_zeros(B, self.K, 2 * self.embed_dim)
        if self.v8_mode in {'levelaction', 'redundant'}:
            features = self.W_a(action).view(B, self.K, 2 * self.embed_dim)
            if self.v8_mode == 'redundant':
                features = features.mean(dim=1, keepdim=True).expand(-1, self.K, -1)
            return features
        shared = self.W_a(action).unsqueeze(1)
        return shared.expand(-1, self.K, -1)

    def _action_prior_unchecked(self, states: torch.Tensor,
                                action: torch.Tensor) -> torch.Tensor:
        features = self._action_features(action)
        d, v = features.chunk(2, dim=-1)
        normalized = F.layer_norm(states, (self.embed_dim,))
        delta = torch.tanh(v + d * normalized)
        beta = self.betas.to(device=states.device, dtype=states.dtype).view(1, self.K, 1)
        return states + beta * delta

    def _dynamic_gate(self, z_t: torch.Tensor, x_t: torch.Tensor,
                      prior: torch.Tensor) -> torch.Tensor:
        dynamic = super()._dynamic_gate(z_t, x_t, prior)
        static = torch.sigmoid(self.gate_bias).view(1, self.K, 1).expand_as(dynamic)
        rho = self.shrinkage().to(device=z_t.device, dtype=z_t.dtype).view(1, self.K, 1)
        return (1.0 - rho) * static + rho * dynamic

    def horizons(self) -> Dict[str, float]:
        result = super().horizons()
        rho = self.shrinkage().detach()
        result.update({
            'rho_fast': float(rho[0]),
            'rho_medium': float(rho[1]),
        })
        if self.v8_mode in {'levelaction', 'redundant'}:
            action_heads = self.W_a.weight.detach().view(
                self.K, 2 * self.embed_dim, self.action_dim)
            result.update({
                'action_head_fast_norm': float(action_heads[0].norm()),
                'action_head_medium_norm': float(action_heads[1].norm()),
                'action_head_cosine': float(F.cosine_similarity(
                    action_heads[0].reshape(1, -1),
                    action_heads[1].reshape(1, -1), dim=-1)[0]),
            })
        else:
            result['action_head_shared_norm'] = float(self.W_a.weight.detach().norm())
        return result


SASPCv8Memory = SharedActionShrinkageMemory


class OrthogonalRecurrentBeliefMemory(nn.Module):
    """ORBIT-v10: one persistent belief with exact action-conditioned transport.

    The recurrent state has no learned or fixed decay.  Instead, two action-conditioned
    Givens layers transport it through products of orthogonal maps.  The first layer pairs
    adjacent coordinates; the second applies a perfect shuffle, rotates cross-half pairs,
    and unshuffles.  The overlapping pairings make the product more expressive than a
    single block-diagonal torus while retaining an exact norm-preserving contract::

        p_t = T(a_{t-1}) m_{t-1},       T(a)^T T(a) = I
        m_t = p_t + g_t (W_x z_t - p_t)
        z~_t = z_t + W_o RMSNorm(m_t)

    Each Givens block receives ``(du,dv)`` from the shared action tensor and normalizes
    ``(1+du,dv)`` into its cosine/sine pair.  Therefore every action map is identity at
    initialization and zero action is always identity.  ``additive`` and ``scaled`` reuse
    exactly the same tensors: they are falsification controls for isometry rather than wider
    alternatives.  ``noaction`` fixes transport to identity, and ``static`` removes only the
    input-conditioned part of the V8-style shrinkage gate.

    No visibility mask, memory target, teacher, auxiliary loss, or task-specific horizon is
    consumed.  The correction bias initializes to ``-2`` (gate ``.119``): because V10 has no
    V8 beta multiplier, this matches the effective initial correction scale of V8's medium
    bank without constraining the value after initialization.  ``memory_update_mask`` is
    accepted only to validate shared data plumbing and is otherwise ignored.  :meth:`step`
    exposes the exact batch-size-one streaming update.
    """

    MODES = {'orthogonal', 'noaction', 'additive', 'scaled', 'static'}
    N_ROTATION_LAYERS = 2

    def __init__(self, embed_dim: int, action_dim: int, mode: str = 'orthogonal',
                 rms_eps: float = 1e-6):
        super().__init__()
        if (not isinstance(embed_dim, int) or isinstance(embed_dim, bool)
                or embed_dim < 2 or embed_dim % 2):
            raise ValueError(
                f'ORBIT-v10 embed_dim must be a positive even integer >=2, got {embed_dim!r}')
        if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim < 1:
            raise ValueError(f'action_dim must be a positive integer, got {action_dim!r}')
        if mode not in self.MODES:
            raise ValueError(f"unknown ORBIT-v10 mode '{mode}'")
        if not math.isfinite(float(rms_eps)) or rms_eps <= 0:
            raise ValueError(f'rms_eps must be positive and finite, got {rms_eps!r}')

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.mode = mode
        self.rms_eps = float(rms_eps)
        self.n_pairs = embed_dim // 2

        self.W_x = nn.Linear(embed_dim, embed_dim, bias=False)
        # Two layers x D outputs/layer; each D-vector is D/2 (du,dv) pairs.
        self.W_a = nn.Linear(action_dim, self.N_ROTATION_LAYERS * embed_dim, bias=False)
        self.w_z = nn.Parameter(torch.zeros(embed_dim))
        self.w_e = nn.Parameter(torch.zeros(embed_dim))
        # V8 initialized sigmoid(+2) but multiplied it by beta(tau=8)=.1175, for an effective
        # correction near .103.  V10 has no decay/beta term, so sigmoid(-2)=.119 is the closest
        # simple initialization.  This is initialization only, not a frozen update rate.
        self.gate_bias = nn.Parameter(torch.tensor(-2.0))
        self.shrink_logit = nn.Parameter(torch.zeros(()))
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.eye_(self.W_x.weight)
        nn.init.zeros_(self.W_a.weight)
        nn.init.zeros_(self.W_o.weight)

        # [0,H,1,H+1,...] is a perfect shuffle.  Adjacent rotations in this coordinate
        # system pair the two halves, overlapping the first layer's adjacent pairing.
        shuffle = torch.arange(embed_dim, dtype=torch.long).view(2, self.n_pairs)
        shuffle = shuffle.transpose(0, 1).reshape(-1)
        self.register_buffer('perfect_shuffle', shuffle)
        self.register_buffer('inverse_shuffle', torch.argsort(shuffle))

    @classmethod
    def expected_parameter_count(cls, embed_dim: int, action_dim: int) -> int:
        """Return ``2D^2 + 2AD + 2D + 2`` for every V10 mode."""
        return (2 * embed_dim * embed_dim
                + cls.N_ROTATION_LAYERS * action_dim * embed_dim
                + 2 * embed_dim + 2)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)

    def shrinkage(self) -> torch.Tensor:
        return torch.sigmoid(self.shrink_logit)

    def _rms_norm(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(
            value.square().mean(dim=-1, keepdim=True) + self.rms_eps)

    def _validate_latents(self, z: torch.Tensor) -> Tuple[int, int, int]:
        if not isinstance(z, torch.Tensor) or z.dim() != 3:
            shape = tuple(z.shape) if isinstance(z, torch.Tensor) else type(z).__name__
            raise ValueError(f'ORBIT-v10 expects z with shape (B,T,D), got {shape}')
        B, T, D = z.shape
        if B < 1 or T < 1 or D != self.embed_dim:
            raise ValueError(
                f'ORBIT-v10 expected non-empty (B,T,{self.embed_dim}), got {tuple(z.shape)}')
        if not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError('z must contain finite floating-point values')
        return B, T, D

    def _validate_actions(self, actions: torch.Tensor, B: int, steps: int,
                          *, name: str = 'actions') -> torch.Tensor:
        expected = (B, steps, self.action_dim)
        if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != expected:
            shape = tuple(actions.shape) if isinstance(actions, torch.Tensor) else type(actions).__name__
            raise ValueError(f'{name} must have shape {expected}, got {shape}')
        if not actions.is_floating_point() or not torch.isfinite(actions).all():
            raise ValueError(f'{name} must contain finite floating-point values')
        return actions

    def _validate_state(self, state: torch.Tensor) -> int:
        if (not isinstance(state, torch.Tensor) or state.dim() != 2
                or state.shape[0] < 1 or state.shape[1] != self.embed_dim):
            shape = tuple(state.shape) if isinstance(state, torch.Tensor) else type(state).__name__
            raise ValueError(f'state must have shape (B,{self.embed_dim}), got {shape}')
        if not state.is_floating_point() or not torch.isfinite(state).all():
            raise ValueError('state must contain finite floating-point values')
        return state.shape[0]

    def _coerce_action_override(self, value, actions: torch.Tensor) -> torch.Tensor:
        override = torch.as_tensor(value, device=actions.device, dtype=actions.dtype)
        try:
            override = torch.broadcast_to(override, actions.shape)
        except RuntimeError as exc:
            raise ValueError(
                f'action_override with shape {tuple(override.shape)} is not broadcastable to '
                f'{tuple(actions.shape)}') from exc
        if not torch.isfinite(override).all():
            raise ValueError('action_override contains non-finite values')
        return override

    def _coerce_gates(self, value, B: int, T: int, *, device,
                      dtype: torch.dtype, name: str) -> torch.Tensor:
        gates = torch.as_tensor(value, device=device, dtype=dtype)
        if gates.dim() == 1 and tuple(gates.shape) == (T,):
            gates = gates.view(1, T, 1)
        elif gates.dim() == 1 and tuple(gates.shape) == (B,):
            gates = gates.view(B, 1, 1)
        elif gates.dim() == 2 and tuple(gates.shape) == (B, T):
            gates = gates.unsqueeze(-1)
        elif gates.dim() == 2 and tuple(gates.shape) == (B, 1):
            gates = gates.unsqueeze(1)
        target = (B, T, 1)
        try:
            gates = torch.broadcast_to(gates, target)
        except RuntimeError as exc:
            raise ValueError(
                f'{name} with shape {tuple(gates.shape)} is not broadcastable to {target}') from exc
        if (not torch.isfinite(gates).all() or bool((gates < 0).any())
                or bool((gates > 1).any())):
            raise ValueError(f'{name} must contain finite values in [0,1]')
        return gates

    def _validate_update_mask(self, mask: torch.Tensor, B: int, T: int,
                              *, device, dtype: torch.dtype) -> None:
        # Reuse the gate validator, but intentionally discard the result: V10 never receives
        # privileged visibility.  This catches malformed shared-batch plumbing.
        self._coerce_gates(
            mask, B, T, device=device, dtype=dtype, name='memory_update_mask')

    def _action_features(self, action: torch.Tensor) -> torch.Tensor:
        """Return the common action tensor as ``(B,2,D)`` in every mode."""
        return self.W_a(action).view(
            action.shape[0], self.N_ROTATION_LAYERS, self.embed_dim)

    def _rotation_components(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pairs = features.view(
            features.shape[0], self.N_ROTATION_LAYERS, self.n_pairs, 2)
        u = 1.0 + pairs[..., 0]
        v = pairs[..., 1]
        if self.mode == 'noaction':
            return torch.ones_like(u), torch.zeros_like(v)
        if self.mode == 'scaled':
            # Same complex blocks without unit normalization: a direct falsification control
            # for exact isometry.  It remains identity at the shared zero initialization.
            return u, v
        radius = torch.sqrt(u.square() + v.square())
        valid = radius > self.rms_eps
        safe_radius = radius.clamp_min(self.rms_eps)
        # The exact zero pair is assigned identity rather than the zero matrix, so even an
        # adversarially learned ``(du,dv)=(-1,0)`` cannot violate the isometry contract.
        cosine = torch.where(valid, u / safe_radius, torch.ones_like(u))
        sine = torch.where(valid, v / safe_radius, torch.zeros_like(v))
        return cosine, sine

    def _rotate_layer(self, state: torch.Tensor, cosine: torch.Tensor,
                      sine: torch.Tensor, layer: int) -> torch.Tensor:
        if layer == 1:
            state = state.index_select(-1, self.perfect_shuffle)
        pairs = state.reshape(state.shape[0], self.n_pairs, 2)
        real, imag = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack((
            cosine * real - sine * imag,
            sine * real + cosine * imag,
        ), dim=-1).reshape(state.shape[0], self.embed_dim)
        if layer == 1:
            rotated = rotated.index_select(-1, self.inverse_shuffle)
        return rotated

    def _transition_unchecked(self, state: torch.Tensor, action: torch.Tensor):
        features = self._action_features(action)
        if self.mode == 'additive':
            d, v = features[:, 0], features[:, 1]
            prior = state + torch.tanh(
                v + d * F.layer_norm(state, (self.embed_dim,)))
            cosine = state.new_ones(
                state.shape[0], self.N_ROTATION_LAYERS, self.n_pairs)
            sine = torch.zeros_like(cosine)
        else:
            cosine, sine = self._rotation_components(features)
            prior = state
            for layer in range(self.N_ROTATION_LAYERS):
                prior = self._rotate_layer(
                    prior, cosine[:, layer], sine[:, layer], layer)

        block_norm_sq = cosine.square() + sine.square()
        orthogonality_error = (block_norm_sq - 1.0).abs().amax(dim=(1, 2))
        # Additive transport has no rotation blocks.  Identity cosine/sine placeholders keep
        # the five modes serialization-compatible, but are explicitly marked non-applicable
        # so they cannot become a false orthogonality receipt for the additive Jacobian.
        orthogonality_applicable = torch.full(
            (state.shape[0],), self.mode != 'additive', device=state.device,
            dtype=torch.bool)
        source_norm = state.norm(dim=-1)
        prior_norm = prior.norm(dim=-1)
        transport_norm_ratio = torch.where(
            source_norm > self.rms_eps,
            prior_norm / source_norm.clamp_min(self.rms_eps),
            torch.ones_like(source_norm),
        )
        return prior, {
            'rotation_cos': cosine,
            'rotation_sin': sine,
            'block_norm_sq': block_norm_sq,
            'orthogonality_error': orthogonality_error,
            'orthogonality_applicable': orthogonality_applicable,
            'transport_norm_ratio': transport_norm_ratio,
        }

    def transition(self, state: torch.Tensor, action: torch.Tensor,
                   action_override=None, return_details: bool = False):
        """Apply one action-only transport without consuming an observation."""
        B = self._validate_state(state)
        if (not isinstance(action, torch.Tensor)
                or tuple(action.shape) != (B, self.action_dim)
                or not action.is_floating_point() or not torch.isfinite(action).all()):
            shape = tuple(action.shape) if isinstance(action, torch.Tensor) else type(action).__name__
            raise ValueError(
                f'action must be finite floating point with shape {(B, self.action_dim)}, got {shape}')
        action = action.to(device=state.device, dtype=state.dtype)
        if action_override is not None:
            action = self._coerce_action_override(action_override, action)
        prior, details = self._transition_unchecked(state, action)
        return (prior, details) if return_details else prior

    def _dynamic_gate(self, z_t: torch.Tensor, x_t: torch.Tensor,
                      prior: torch.Tensor) -> torch.Tensor:
        normalized_z = F.layer_norm(z_t, (self.embed_dim,))
        normalized_error = F.layer_norm(x_t - prior, (self.embed_dim,))
        score = (torch.einsum('bd,d->b', normalized_z, self.w_z)
                 + torch.einsum('bd,d->b', normalized_error, self.w_e))
        dynamic = torch.sigmoid(self.gate_bias + score / math.sqrt(self.embed_dim))
        static = torch.sigmoid(self.gate_bias).expand_as(dynamic)
        rho = self.shrinkage().to(device=z_t.device, dtype=z_t.dtype)
        return ((1.0 - rho) * static + rho * dynamic).unsqueeze(-1)

    def step(self, state: torch.Tensor, z_t: torch.Tensor, action: torch.Tensor,
             gate_override=None, action_override=None, return_details: bool = False):
        """Advance one explicit ``D``-float streaming state.

        ``action`` maps the previous state/observation to ``z_t``.  Returns
        ``(mixed, new_state)`` or ``(mixed, new_state, details)``.
        """
        B = self._validate_state(state)
        if (not isinstance(z_t, torch.Tensor) or tuple(z_t.shape) != (B, self.embed_dim)
                or not z_t.is_floating_point() or not torch.isfinite(z_t).all()):
            shape = tuple(z_t.shape) if isinstance(z_t, torch.Tensor) else type(z_t).__name__
            raise ValueError(f'z_t must be finite with shape {(B, self.embed_dim)}, got {shape}')
        z_t = z_t.to(device=state.device, dtype=state.dtype)
        prior, transition_details = self.transition(
            state, action, action_override=action_override, return_details=True)
        x_t = self.W_x(z_t)
        gate = (torch.sigmoid(self.gate_bias).view(1, 1).expand(B, 1)
                if self.mode == 'static' else self._dynamic_gate(z_t, x_t, prior))
        if gate_override is not None:
            gate = self._coerce_gates(
                gate_override, B, 1, device=state.device, dtype=state.dtype,
                name='gate_override')[:, 0]
        new_state = prior + gate * (x_t - prior)
        mixed = self._rms_norm(new_state)
        details = {
            'x': x_t,
            'prior': prior,
            'state': new_state,
            'gate': gate,
            **transition_details,
        }
        if return_details:
            return mixed, new_state, details
        return mixed, new_state

    def forward(self, z: torch.Tensor, actions: torch.Tensor,
                memory_update_mask: torch.Tensor = None, gate_override=None,
                action_override=None, return_details: bool = False):
        """Run the causal ORBIT filter over ``(B,T,D)`` latent observations."""
        B, T, _ = self._validate_latents(z)
        actions = self._validate_actions(actions, B, T - 1)
        actions = actions.to(device=z.device, dtype=z.dtype)
        if action_override is not None:
            actions = self._coerce_action_override(action_override, actions)
        if memory_update_mask is not None:
            self._validate_update_mask(
                memory_update_mask, B, T, device=z.device, dtype=z.dtype)
        override_gates = None
        if gate_override is not None:
            override_gates = self._coerce_gates(
                gate_override, B, T, device=z.device, dtype=z.dtype,
                name='gate_override')

        x = self.W_x(z)
        state = x[:, 0]
        initial_gate = (torch.sigmoid(self.gate_bias).view(1, 1).expand(B, 1)
                        if self.mode == 'static'
                        else self._dynamic_gate(z[:, 0], x[:, 0], state))
        if override_gates is not None:
            initial_gate = override_gates[:, 0]
        identity_cos = z.new_ones(B, self.N_ROTATION_LAYERS, self.n_pairs)
        identity_sin = torch.zeros_like(identity_cos)

        states = [state]
        priors = [state]
        gates = [initial_gate]
        cosines = [identity_cos]
        sines = [identity_sin]
        block_norms = [torch.ones_like(identity_cos)]
        orthogonality_errors = [z.new_zeros(B)]
        orthogonality_applicable = [torch.zeros(B, device=z.device, dtype=torch.bool)]
        norm_ratios = [z.new_ones(B)]

        for t in range(1, T):
            prior, transition_details = self._transition_unchecked(
                state, actions[:, t - 1])
            gate = (torch.sigmoid(self.gate_bias).view(1, 1).expand(B, 1)
                    if self.mode == 'static'
                    else self._dynamic_gate(z[:, t], x[:, t], prior))
            if override_gates is not None:
                gate = override_gates[:, t]
            state = prior + gate * (x[:, t] - prior)
            priors.append(prior)
            states.append(state)
            gates.append(gate)
            cosines.append(transition_details['rotation_cos'])
            sines.append(transition_details['rotation_sin'])
            block_norms.append(transition_details['block_norm_sq'])
            orthogonality_errors.append(transition_details['orthogonality_error'])
            orthogonality_applicable.append(
                transition_details['orthogonality_applicable'])
            norm_ratios.append(transition_details['transport_norm_ratio'])

        state_sequence = torch.stack(states, dim=1)
        mixed = self._rms_norm(state_sequence)
        if not return_details:
            return mixed
        details = {
            'x': x,
            'priors': torch.stack(priors, dim=1),
            'states': state_sequence,
            'gates': torch.stack(gates, dim=1),
            'rotation_cos': torch.stack(cosines, dim=1),
            'rotation_sin': torch.stack(sines, dim=1),
            'block_norm_sq': torch.stack(block_norms, dim=1),
            'orthogonality_error': torch.stack(orthogonality_errors, dim=1),
            'orthogonality_applicable': torch.stack(
                orthogonality_applicable, dim=1),
            'transport_norm_ratio': torch.stack(norm_ratios, dim=1),
        }
        return mixed, details

    def fuse(self, z: torch.Tensor, mixed: torch.Tensor) -> torch.Tensor:
        if (not isinstance(z, torch.Tensor) or not isinstance(mixed, torch.Tensor)
                or z.dim() != 3 or mixed.dim() != 3 or z.shape != mixed.shape
                or z.shape[-1] != self.embed_dim):
            z_shape = tuple(z.shape) if isinstance(z, torch.Tensor) else type(z).__name__
            mixed_shape = tuple(mixed.shape) if isinstance(mixed, torch.Tensor) else type(mixed).__name__
            raise ValueError(
                f'z and mixed must share shape (B,T,{self.embed_dim}), got '
                f'{z_shape} and {mixed_shape}')
        if (not z.is_floating_point() or not mixed.is_floating_point()
                or not torch.isfinite(z).all() or not torch.isfinite(mixed).all()):
            raise ValueError('z and mixed must contain finite floating-point values')
        return z + self.W_o(mixed)

    def horizons(self) -> Dict[str, float]:
        """Expose the absence of decay together with auditable learned mechanisms."""
        with torch.no_grad():
            return {
                'tau_fast': float('inf'),
                'tau_slow': float('inf'),
                'n_banks': 1,
                'rho_state': float(self.shrinkage()),
                'gate_static': float(torch.sigmoid(self.gate_bias)),
                'action_head_shared_norm': float(self.W_a.weight.norm()),
                'orthogonal_transport': self.mode in {'orthogonal', 'static', 'noaction'},
                'orthogonality_diagnostic_applicable': self.mode != 'additive',
                'transport_layers': self.N_ROTATION_LAYERS,
                'recurrent_floats': self.embed_dim,
            }


ORBITv10Memory = OrthogonalRecurrentBeliefMemory


class LearnedOrderedInnovationFilterMemory(nn.Module):
    """HACSSM-v9 learned ordered innovation filter (LOIF).

    LOIF keeps two full-width recurrent states but removes fixed memory horizons, free route
    logits, and memory-specific auxiliary targets.  Two learned scalar retention poles are
    ordered by construction.  Their complementary process scales, one causal observation
    resistance, and scalar filtering updates determine both correction gains and state fusion::

        alpha_f = (1-eps) sigmoid(u_fast)
        alpha_s = alpha_f + (1-eps-alpha_f) sigmoid(u_delta)
        q_k     = (1-alpha_k) (1+alpha_k)
        p_t^k   = alpha_k m_{t-1}^k + sqrt(q_k) h_t^k
        K_t^k   = P_t^{k,-} / (P_t^{k,-} + R_t)
        m_t^k   = p_t^k + K_t^k (x_t - p_t^k)

    ``P`` and ``R`` are operational positive scales under the ordinary prediction MSE, not
    calibrated posterior variances.  Scale recurrences are performed in log space.  Every
    control owns the exact same trainable tensors; a control disconnects only the path named by
    its intervention.  In particular, ``singlebank`` still needs both pole logits because the
    ordered parameterization of ``alpha_s`` depends on both.

    The streaming state is exactly two D-vectors plus two scalar log scales.  ``forward`` exposes
    the full causal trajectory for diagnostics, while :meth:`filter_step` is the matching
    batch-size-one/general-batch streaming primitive.
    """

    MODES = {
        'learned', 'fixedalpha', 'globalR', 'innovationonly', 'latentonly',
        'uniformfusion', 'noaction', 'singlebank',
    }
    K = 2
    FIXED_ALPHAS = (math.exp(-1.0 / 2.0), math.exp(-1.0 / 8.0))

    def __init__(self, embed_dim: int, action_dim: int, mode: str = 'learned',
                 eps: float = 1e-6):
        super().__init__()
        if not isinstance(embed_dim, int) or isinstance(embed_dim, bool) or embed_dim < 1:
            raise ValueError(f'embed_dim must be a positive integer, got {embed_dim!r}')
        if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim < 1:
            raise ValueError(f'action_dim must be a positive integer, got {action_dim!r}')
        if mode not in self.MODES:
            raise ValueError(f"unknown LOIF-v9 mode '{mode}'")
        if not math.isfinite(float(eps)) or not 0.0 < float(eps) < 0.5:
            raise ValueError(f'eps must be finite and lie in (0,.5), got {eps!r}')

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.mode = mode
        self.eps = float(eps)

        # Exact architecture contract: 2D^2 + 2AD + 2D + 3 parameters.  The affine-free
        # normalizations below deliberately introduce no hidden scale or offset tensors.
        self.W_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_a = nn.Linear(action_dim, 2 * embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_z = nn.Parameter(torch.zeros(embed_dim))
        self.w_e = nn.Parameter(torch.zeros(embed_dim))
        self.u_fast = nn.Parameter(torch.zeros(()))
        self.u_delta = nn.Parameter(torch.zeros(()))
        self.b_R = nn.Parameter(torch.zeros(()))

        nn.init.eye_(self.W_x.weight)
        nn.init.zeros_(self.W_a.weight)
        nn.init.zeros_(self.W_o.weight)
        # The learned and fixed-alpha branches must begin at the same point so their contrast
        # isolates whether the retentions learn.  This is the frozen tau-mapped initialization,
        # not a post-initialization constraint on the nominated model.
        upper = 1.0 - self.eps
        fixed_fast, fixed_slow = self.FIXED_ALPHAS
        fast_fraction = fixed_fast / upper
        delta_fraction = (fixed_slow - fixed_fast) / (upper - fixed_fast)
        with torch.no_grad():
            self.u_fast.fill_(math.log(fast_fraction / (1.0 - fast_fraction)))
            self.u_delta.fill_(math.log(delta_fraction / (1.0 - delta_fraction)))

    @classmethod
    def expected_parameter_count(cls, embed_dim: int, action_dim: int) -> int:
        """Return the exact ``2D^2 + 2AD + 2D + 3`` trainable-parameter contract."""
        return (2 * embed_dim * embed_dim + 2 * action_dim * embed_dim
                + 2 * embed_dim + 3)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)

    def ordered_alphas(self) -> torch.Tensor:
        """Return ``(alpha_fast, alpha_slow)`` as direct old-state retentions."""
        if self.mode == 'fixedalpha':
            # Keep u_fast/u_delta physically present but exactly disconnected in this control.
            return self.b_R.new_tensor(self.FIXED_ALPHAS)
        upper = 1.0 - self.eps
        alpha_fast = upper * torch.sigmoid(self.u_fast)
        alpha_slow = alpha_fast + (upper - alpha_fast) * torch.sigmoid(self.u_delta)
        return torch.stack((alpha_fast, alpha_slow))

    def process_scales(self, alphas: torch.Tensor = None) -> torch.Tensor:
        """Return coupled process scales ``q=(1-alpha)(1+alpha)`` without cancellation."""
        if alphas is None:
            alphas = self.ordered_alphas()
        if not isinstance(alphas, torch.Tensor) or tuple(alphas.shape) != (self.K,):
            shape = tuple(alphas.shape) if isinstance(alphas, torch.Tensor) else type(alphas).__name__
            raise ValueError(f'alphas must have shape ({self.K},), got {shape}')
        return (1.0 - alphas) * (1.0 + alphas)

    def _rms_norm(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(
            value.square().mean(dim=-1, keepdim=True) + self.eps)

    def _validate_latents(self, z: torch.Tensor) -> Tuple[int, int, int]:
        if not isinstance(z, torch.Tensor) or z.dim() != 3:
            shape = tuple(z.shape) if isinstance(z, torch.Tensor) else type(z).__name__
            raise ValueError(f'LOIF-v9 expects z with shape (B,T,D), got {shape}')
        B, T, D = z.shape
        if B < 1 or T < 1 or D != self.embed_dim:
            raise ValueError(
                f'LOIF-v9 expected non-empty (B,T,{self.embed_dim}), got {tuple(z.shape)}')
        if not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError('z must contain finite floating-point values')
        return B, T, D

    def _validate_actions(self, actions: torch.Tensor, B: int, steps: int,
                          *, name: str = 'actions') -> torch.Tensor:
        expected = (B, steps, self.action_dim)
        if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != expected:
            shape = tuple(actions.shape) if isinstance(actions, torch.Tensor) else type(actions).__name__
            raise ValueError(f'{name} must have shape {expected}, got {shape}')
        if not actions.is_floating_point() or not torch.isfinite(actions).all():
            raise ValueError(f'{name} must contain finite floating-point values')
        return actions

    def _validate_update_mask(self, mask: torch.Tensor, B: int, T: int) -> None:
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (B, T):
            shape = tuple(mask.shape) if isinstance(mask, torch.Tensor) else type(mask).__name__
            raise ValueError(f'memory_update_mask must have shape {(B, T)}, got {shape}')
        if mask.dtype != torch.bool:
            if not mask.is_floating_point() or not torch.isfinite(mask).all():
                raise ValueError('memory_update_mask must be boolean or finite in [0,1]')
            if bool((mask < 0).any()) or bool((mask > 1).any()):
                raise ValueError('memory_update_mask must lie in [0,1]')

    def _coerce_action_override(self, value, actions: torch.Tensor) -> torch.Tensor:
        override = torch.as_tensor(value, device=actions.device, dtype=actions.dtype)
        try:
            override = torch.broadcast_to(override, actions.shape)
        except RuntimeError as exc:
            raise ValueError(
                f'action_override with shape {tuple(override.shape)} is not broadcastable to '
                f'{tuple(actions.shape)}') from exc
        if not torch.isfinite(override).all():
            raise ValueError('action_override contains non-finite values')
        return override

    def _coerce_resistance_override(self, value, B: int, T: int, *, device,
                                    dtype: torch.dtype) -> torch.Tensor:
        """Fail closed unless supplied R values are finite, positive, and time aligned."""
        resistance = torch.as_tensor(value, device=device, dtype=dtype)
        if resistance.dim() == 0:
            resistance = resistance.expand(B, T)
        elif resistance.dim() == 3 and tuple(resistance.shape) == (B, T, 1):
            resistance = resistance.squeeze(-1)
        elif tuple(resistance.shape) != (B, T):
            raise ValueError(
                f'resistance_override with shape {tuple(resistance.shape)} is not '
                f'a scalar, {(B, T)}, or {(B, T, 1)}')
        if not torch.isfinite(resistance).all() or bool((resistance <= 0).any()):
            raise ValueError('resistance_override must contain finite values strictly above zero')
        return resistance

    def _fusion_weights(self, log_scales: torch.Tensor) -> torch.Tensor:
        if self.mode == 'singlebank':
            weights = torch.zeros_like(log_scales)
            weights[..., 1] = 1.0
            return weights
        if self.mode == 'uniformfusion':
            return torch.full_like(log_scales, 0.5)
        return torch.softmax(-log_scales, dim=-1)

    def _resistance(self, z_t: torch.Tensor, x_t: torch.Tensor,
                    prior_mixture: torch.Tensor) -> torch.Tensor:
        """Infer the one shared causal positive observation-resistance scale."""
        score = z_t.new_zeros(z_t.shape[0])
        if self.mode not in {'globalR', 'innovationonly'}:
            normalized_z = F.layer_norm(
                z_t, (self.embed_dim,), eps=self.eps)
            score = score + torch.einsum('bd,d->b', normalized_z, self.w_z)
        if self.mode not in {'globalR', 'latentonly'}:
            innovation = x_t - prior_mixture
            score = score + torch.einsum('bd,d->b', innovation, self.w_e)
        score = score / math.sqrt(self.embed_dim)
        return F.softplus(self.b_R + score) + self.eps

    def _action_prior(self, states: torch.Tensor, action: torch.Tensor,
                      alphas: torch.Tensor, qs: torch.Tensor) -> torch.Tensor:
        B = states.shape[0]
        if self.mode == 'noaction':
            action_features = states.new_zeros(B, 2 * self.embed_dim)
        else:
            action_features = self.W_a(action)
        d, v = action_features.chunk(2, dim=-1)
        normalized = F.layer_norm(
            states, (self.embed_dim,), eps=self.eps)
        innovation = torch.tanh(v.unsqueeze(1) + d.unsqueeze(1) * normalized)
        alpha_state = alphas.to(device=states.device, dtype=states.dtype).view(1, self.K, 1)
        q_state = qs.to(device=states.device, dtype=states.dtype).view(1, self.K, 1)
        return alpha_state * states + torch.sqrt(q_state) * innovation

    def _filter_step_unchecked(self, states: torch.Tensor, log_scales: torch.Tensor,
                               z_t: torch.Tensor, action: torch.Tensor,
                               alphas: torch.Tensor, qs: torch.Tensor,
                               resistance_override: torch.Tensor = None,
                               x_t: torch.Tensor = None):
        """One already-validated causal transition/correction step."""
        prior = self._action_prior(states, action, alphas, qs)

        scale_dtype = log_scales.dtype
        alpha_scale = alphas.to(device=log_scales.device, dtype=scale_dtype)
        q_scale = qs.to(device=log_scales.device, dtype=scale_dtype)
        log_prior_scales = torch.logaddexp(
            log_scales + 2.0 * torch.log(alpha_scale).view(1, self.K),
            torch.log(q_scale).view(1, self.K),
        )
        prior_weights = self._fusion_weights(log_prior_scales)
        prior_mixture = (
            prior * prior_weights.to(dtype=prior.dtype).unsqueeze(-1)
        ).sum(dim=1)
        if x_t is None:
            x_t = self._rms_norm(self.W_x(z_t))
        resistance = (self._resistance(z_t, x_t, prior_mixture)
                      if resistance_override is None else resistance_override)
        resistance = resistance.to(device=log_scales.device, dtype=scale_dtype)
        log_R = torch.log(resistance)
        gains = torch.sigmoid(log_prior_scales - log_R.unsqueeze(-1))
        posterior = prior + gains.to(dtype=prior.dtype).unsqueeze(-1) * (
            x_t.unsqueeze(1) - prior)
        log_post_scales = (
            log_prior_scales + log_R.unsqueeze(-1)
            - torch.logaddexp(log_prior_scales, log_R.unsqueeze(-1))
        )
        read_weights = self._fusion_weights(log_post_scales)
        mixed = self._rms_norm((
            posterior * read_weights.to(dtype=posterior.dtype).unsqueeze(-1)
        ).sum(dim=1))
        nominal = alpha_scale.view(1, self.K) * (1.0 - gains)
        details = {
            'x': x_t,
            'priors': prior,
            'states': posterior,
            'log_prior_scales': log_prior_scales,
            'log_P': log_post_scales,
            'log_scales': log_post_scales,
            'resistance': resistance,
            'log_R': log_R,
            'log_resistance': log_R,
            'gains': gains,
            'prior_weights': prior_weights,
            'read_weights': read_weights,
            'nominal_direct_coefficients': nominal,
        }
        return mixed, posterior, log_post_scales, details

    def filter_step(self, states: torch.Tensor, log_scales: torch.Tensor,
                    z_t: torch.Tensor, action: torch.Tensor,
                    resistance_override=None):
        """Advance the explicit ``(2D+2)`` streaming state by one observed step.

        ``action`` maps the previous observation/state to ``z_t``.  Returns
        ``(mixed, states, log_scales, details)``.
        """
        if (not isinstance(states, torch.Tensor) or states.dim() != 3
                or tuple(states.shape[1:]) != (self.K, self.embed_dim)):
            shape = tuple(states.shape) if isinstance(states, torch.Tensor) else type(states).__name__
            raise ValueError(
                f'states must have shape (B,{self.K},{self.embed_dim}), got {shape}')
        B = states.shape[0]
        if (not states.is_floating_point() or not torch.isfinite(states).all()
                or not isinstance(log_scales, torch.Tensor)
                or tuple(log_scales.shape) != (B, self.K)
                or not log_scales.is_floating_point() or not torch.isfinite(log_scales).all()):
            raise ValueError('states/log_scales must be finite floating-point streaming state')
        if (not isinstance(z_t, torch.Tensor)
                or tuple(z_t.shape) != (B, self.embed_dim)
                or not z_t.is_floating_point() or not torch.isfinite(z_t).all()):
            shape = tuple(z_t.shape) if isinstance(z_t, torch.Tensor) else type(z_t).__name__
            raise ValueError(f'z_t must be finite with shape {(B, self.embed_dim)}, got {shape}')
        if (not isinstance(action, torch.Tensor)
                or tuple(action.shape) != (B, self.action_dim)
                or not action.is_floating_point() or not torch.isfinite(action).all()):
            shape = tuple(action.shape) if isinstance(action, torch.Tensor) else type(action).__name__
            raise ValueError(f'action must be finite with shape {(B, self.action_dim)}, got {shape}')
        action = action.to(device=states.device, dtype=states.dtype)
        z_t = z_t.to(device=states.device, dtype=states.dtype)
        override = None
        if resistance_override is not None:
            override = self._coerce_resistance_override(
                resistance_override, B, 1, device=log_scales.device,
                dtype=log_scales.dtype)[:, 0]
        alphas = self.ordered_alphas()
        qs = self.process_scales(alphas)
        return self._filter_step_unchecked(
            states, log_scales, z_t, action, alphas, qs, override)

    def forward(self, z: torch.Tensor, actions: torch.Tensor,
                memory_update_mask: torch.Tensor = None, action_override=None,
                resistance_override=None, return_details: bool = False):
        """Run the causal filter and optionally return every V9 diagnostic trajectory.

        ``resistance_override`` accepts a finite positive scalar, ``(B,T)``, or ``(B,T,1)``
        tensor.  It is an analysis-only intervention; unlike a visibility mask it directly
        replaces the operational ``R_t`` values and fails closed on malformed input.
        """
        B, T, _ = self._validate_latents(z)
        actions = self._validate_actions(actions, B, T - 1)
        actions = actions.to(device=z.device, dtype=z.dtype)
        if action_override is not None:
            actions = self._coerce_action_override(action_override, actions)
        if memory_update_mask is not None:
            # V9 never consumes visibility metadata; validate shared plumbing, then ignore it.
            self._validate_update_mask(memory_update_mask, B, T)

        scale_dtype = (torch.float32 if z.dtype in {torch.float16, torch.bfloat16}
                       else z.dtype)
        override = None
        if resistance_override is not None:
            override = self._coerce_resistance_override(
                resistance_override, B, T, device=z.device, dtype=scale_dtype)

        alphas = self.ordered_alphas()
        qs = self.process_scales(alphas)
        x = self._rms_norm(self.W_x(z))
        state = x[:, 0].unsqueeze(1).expand(-1, self.K, -1)
        log_scale = torch.zeros(B, self.K, device=z.device, dtype=scale_dtype)
        initial_weights = self._fusion_weights(log_scale)
        initial_mixed = self._rms_norm((
            state * initial_weights.to(dtype=state.dtype).unsqueeze(-1)
        ).sum(dim=1))
        prior_mixture_0 = (
            state * initial_weights.to(dtype=state.dtype).unsqueeze(-1)
        ).sum(dim=1)
        resistance_0 = (self._resistance(z[:, 0], x[:, 0], prior_mixture_0)
                        if override is None else override[:, 0])
        resistance_0 = resistance_0.to(dtype=scale_dtype)
        log_R_0 = torch.log(resistance_0)
        gains_0 = torch.zeros_like(log_scale)
        nominal_0 = alphas.to(device=z.device, dtype=scale_dtype).view(1, self.K).expand(B, -1)

        mixed_sequence = [initial_mixed]
        states = [state]
        priors = [state]
        log_prior_scales = [log_scale]
        log_scales = [log_scale]
        resistances = [resistance_0]
        log_resistances = [log_R_0]
        gains = [gains_0]
        prior_weights = [initial_weights]
        read_weights = [initial_weights]
        nominal_coefficients = [nominal_0]

        for t in range(1, T):
            mixed, state, log_scale, step_details = self._filter_step_unchecked(
                state, log_scale, z[:, t], actions[:, t - 1], alphas, qs,
                None if override is None else override[:, t], x[:, t])
            mixed_sequence.append(mixed)
            states.append(state)
            priors.append(step_details['priors'])
            log_prior_scales.append(step_details['log_prior_scales'])
            log_scales.append(log_scale)
            resistances.append(step_details['resistance'])
            log_resistances.append(step_details['log_R'])
            gains.append(step_details['gains'])
            prior_weights.append(step_details['prior_weights'])
            read_weights.append(step_details['read_weights'])
            nominal_coefficients.append(step_details['nominal_direct_coefficients'])

        mixed = torch.stack(mixed_sequence, dim=1)
        if not return_details:
            return mixed
        log_P = torch.stack(log_scales, dim=1)
        log_R = torch.stack(log_resistances, dim=1)
        details = {
            'x': x,
            'priors': torch.stack(priors, dim=1),
            'states': torch.stack(states, dim=1),
            'log_prior_scales': torch.stack(log_prior_scales, dim=1),
            'scales': torch.exp(log_P),
            'log_P': log_P,
            'log_scales': log_P,
            'resistance': torch.stack(resistances, dim=1),
            'log_R': log_R,
            'log_resistance': log_R,
            'gains': torch.stack(gains, dim=1),
            'prior_weights': torch.stack(prior_weights, dim=1),
            'read_weights': torch.stack(read_weights, dim=1),
            'nominal_direct_coefficients': torch.stack(nominal_coefficients, dim=1),
            'alphas': alphas,
            'q': qs,
            'qs': qs,
        }
        return mixed, details

    def fuse(self, z: torch.Tensor, mixed: torch.Tensor) -> torch.Tensor:
        if (not isinstance(z, torch.Tensor) or not isinstance(mixed, torch.Tensor)
                or z.dim() != 3 or mixed.dim() != 3 or z.shape != mixed.shape
                or z.shape[-1] != self.embed_dim):
            z_shape = tuple(z.shape) if isinstance(z, torch.Tensor) else type(z).__name__
            mixed_shape = tuple(mixed.shape) if isinstance(mixed, torch.Tensor) else type(mixed).__name__
            raise ValueError(
                f'z and mixed must share shape (B,T,{self.embed_dim}), got '
                f'{z_shape} and {mixed_shape}')
        if (not z.is_floating_point() or not mixed.is_floating_point()
                or not torch.isfinite(z).all() or not torch.isfinite(mixed).all()):
            raise ValueError('z and mixed must contain finite floating-point values')
        return z + self.W_o(mixed)

    def horizons(self) -> Dict[str, float]:
        """Expose learned pole/process-scale metadata without inventing fixed horizons."""
        with torch.no_grad():
            alphas = self.ordered_alphas()
            qs = self.process_scales(alphas)
            taus = -1.0 / torch.log(alphas.clamp(min=torch.finfo(alphas.dtype).tiny,
                                                 max=1.0 - self.eps))
            return {
                'tau_fast': float(taus[0]),
                'tau_slow': float(taus[1]),
                'alpha_fast': float(alphas[0]),
                'alpha_slow': float(alphas[1]),
                'q_fast': float(qs[0]),
                'q_slow': float(qs[1]),
                'pole_separation': float(alphas[1] - alphas[0]),
                'evidence_offset': float(self.b_R),
                'action_head_shared_norm': float(self.W_a.weight.norm()),
                'n_banks': self.K,
            }


LOIFv9Memory = LearnedOrderedInnovationFilterMemory


class HierarchicalActionConditionedSSMMemory(nn.Module):
    """HACSSM-v5: a hard-monotone, action-conditioned two-level SSM memory.

    V5 retains V4's causal action prior and selective observation correction, but replaces its
    three fixed scalar rates with two learned *per-channel* rates.  The parameterization enforces
    the fast/medium ordering for every channel throughout optimization::

        beta_medium = sigmoid(theta_medium)
        beta_fast   = beta_medium + (1-beta_medium) sigmoid(theta_gap)

    Both rates are therefore in ``(0,1)`` and ``beta_fast >= beta_medium`` by construction.  The
    initial rate spectra correspond to log-spaced fast horizons 1.5--8 and medium horizons 8--64;
    these bands are initialization priors, not constraints on the learned horizons.  ``fixedbeta``
    instantiates the same logits as every other mode but uses exact buffered initial rates.

    For level ``k`` and transition action ``a_t: z_t -> z_{t+1}``, the recurrence is::

        p_{t+1}^k = m_t^k + beta_k tanh(v_t + d_t * LN(m_t^k))
        m_{t+1}^k = p_{t+1}^k + beta_k g_{t+1}^k (W_x z_{t+1} - p_{t+1}^k)

    where ``(d_t,v_t)=split(W_a a_t)``.  Thus the same rate controls both action advance and
    observation correction.  Dynamic gates use V4's shared ``w_z``/``w_e`` vectors and one scalar
    bias per level.  A global simplex read, parameter-free RMS normalization, and zero-initialized
    residual output projection complete the memory.

    ``dynamic``, ``static``, ``noaction``, ``fixedbeta``, ``single``, and ``ssmcontrol`` instantiate
    exactly the same tensors and trainable parameter count.  ``static`` uses only the learned gate
    biases; ``noaction`` zeros the action features; ``single`` reads only the medium state;
    ``ssmcontrol`` zeros action features and fixes both correction gates to one.  A supplied
    ``memory_update_mask`` is validated but deliberately ignored in every mode.  Gate, action, and
    beta overrides are analysis-only interventions; beta overrides may intentionally bypass the
    learned monotonic constraint, but must remain finite rates in ``[0,1]``.
    """

    MODES = {'dynamic', 'static', 'noaction', 'fixedbeta', 'single', 'ssmcontrol'}
    K = 2

    def __init__(self, embed_dim: int, action_dim: int, mode: str = 'dynamic',
                 rms_eps: float = 1e-6):
        super().__init__()
        if not isinstance(embed_dim, int) or isinstance(embed_dim, bool) or embed_dim < 1:
            raise ValueError(f'embed_dim must be a positive integer, got {embed_dim!r}')
        if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim < 1:
            raise ValueError(f'action_dim must be a positive integer, got {action_dim!r}')
        if mode not in self.MODES:
            raise ValueError(f"unknown HACSSM-v5 mode '{mode}'")
        if not math.isfinite(float(rms_eps)) or rms_eps <= 0:
            raise ValueError(f'rms_eps must be positive and finite, got {rms_eps}')

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.mode = mode
        self.rms_eps = float(rms_eps)

        fast_taus = torch.logspace(math.log10(1.5), math.log10(8.0), embed_dim)
        medium_taus = torch.logspace(math.log10(8.0), math.log10(64.0), embed_dim)
        beta_fast = 1.0 - torch.exp(-1.0 / fast_taus)
        beta_medium = 1.0 - torch.exp(-1.0 / medium_taus)
        gap_fraction = ((beta_fast - beta_medium) / (1.0 - beta_medium)).clamp(
            torch.finfo(beta_fast.dtype).eps, 1.0 - torch.finfo(beta_fast.dtype).eps)

        # These logits exist in every mode.  The fixedbeta control intentionally disconnects them
        # from the recurrence and uses the exact pre-logit rates stored below.
        self.theta_medium = nn.Parameter(torch.logit(beta_medium))
        self.theta_gap = nn.Parameter(torch.logit(gap_fraction))
        self.register_buffer('initial_fast_taus', fast_taus)
        self.register_buffer('initial_medium_taus', medium_taus)
        self.register_buffer('fixed_betas', torch.stack((beta_fast, beta_medium), dim=0))

        self.W_x = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_a = nn.Linear(action_dim, 2 * embed_dim, bias=False)
        self.w_z = nn.Parameter(torch.zeros(embed_dim))
        self.w_e = nn.Parameter(torch.zeros(embed_dim))
        self.gate_bias = nn.Parameter(torch.full((self.K,), 2.0))
        self.route_logits = nn.Parameter(torch.zeros(self.K))
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        nn.init.eye_(self.W_x.weight)
        nn.init.zeros_(self.W_a.weight)
        nn.init.zeros_(self.W_o.weight)

    @classmethod
    def expected_parameter_count(cls, embed_dim: int, action_dim: int) -> int:
        """Return the exact count ``2D^2 + 2AD + 4D + 4``."""
        return (2 * embed_dim * embed_dim + 2 * action_dim * embed_dim
                + 4 * embed_dim + 4)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def betas(self) -> torch.Tensor:
        """Current rates as ``(2,D)`` ordered fast, medium."""
        if self.mode == 'fixedbeta':
            return self.fixed_betas
        beta_medium = torch.sigmoid(self.theta_medium)
        beta_fast = beta_medium + (1.0 - beta_medium) * torch.sigmoid(self.theta_gap)
        return torch.stack((beta_fast, beta_medium), dim=0)

    def _validate_latents(self, z: torch.Tensor) -> Tuple[int, int, int]:
        if not isinstance(z, torch.Tensor) or z.dim() != 3:
            shape = tuple(z.shape) if isinstance(z, torch.Tensor) else type(z).__name__
            raise ValueError(f'HACSSM-v5 expects z with shape (B,T,D), got {shape}')
        B, T, D = z.shape
        if B < 1 or T < 1:
            raise ValueError(f'HACSSM-v5 requires non-empty batch and time dimensions, got {B}, {T}')
        if D != self.embed_dim:
            raise ValueError(f'HACSSM-v5 expected latent dim {self.embed_dim}, got {D}')
        if not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError('z must contain finite floating-point values')
        return B, T, D

    def _validate_actions(self, actions: torch.Tensor, B: int, steps: int,
                          *, name: str = 'actions') -> torch.Tensor:
        expected = (B, steps, self.action_dim)
        if not isinstance(actions, torch.Tensor) or tuple(actions.shape) != expected:
            shape = tuple(actions.shape) if isinstance(actions, torch.Tensor) else type(actions).__name__
            raise ValueError(f'{name} must have shape {expected}, got {shape}')
        if not actions.is_floating_point() or not torch.isfinite(actions).all():
            raise ValueError(f'{name} must contain finite floating-point values')
        return actions

    @staticmethod
    def _check_unit_interval(values: torch.Tensor, name: str) -> None:
        if not torch.isfinite(values).all():
            raise ValueError(f'{name} contains non-finite values')
        if bool((values < 0).any()) or bool((values > 1).any()):
            raise ValueError(f'{name} must lie in [0,1]')

    def _coerce_action_override(self, value, actions: torch.Tensor, name: str) -> torch.Tensor:
        override = torch.as_tensor(value, device=actions.device, dtype=actions.dtype)
        try:
            override = torch.broadcast_to(override, actions.shape)
        except RuntimeError as exc:
            raise ValueError(
                f'{name} with shape {tuple(override.shape)} is not broadcastable to '
                f'{tuple(actions.shape)}') from exc
        if not torch.isfinite(override).all():
            raise ValueError(f'{name} contains non-finite values')
        return override

    def _coerce_gates(self, value, B: int, T: int, *, device, dtype,
                      name: str) -> torch.Tensor:
        gates = torch.as_tensor(value, device=device, dtype=dtype)
        if gates.dim() == 1 and tuple(gates.shape) == (self.K,):
            gates = gates.view(1, 1, self.K, 1)
        elif gates.dim() == 1 and tuple(gates.shape) == (T,):
            gates = gates.view(1, T, 1, 1)
        elif gates.dim() == 2 and tuple(gates.shape) == (T, self.K):
            gates = gates.unsqueeze(0).unsqueeze(-1)
        elif gates.dim() == 2 and tuple(gates.shape) == (B, T):
            gates = gates.unsqueeze(-1).unsqueeze(-1)
        elif gates.dim() == 2 and tuple(gates.shape) == (B, self.K):
            gates = gates.unsqueeze(1).unsqueeze(-1)
        elif gates.dim() == 3 and tuple(gates.shape) == (B, T, 1):
            gates = gates.unsqueeze(-2)
        elif gates.dim() == 3 and tuple(gates.shape) == (B, T, self.K):
            gates = gates.unsqueeze(-1)
        target = (B, T, self.K, 1)
        try:
            gates = torch.broadcast_to(gates, target)
        except RuntimeError as exc:
            raise ValueError(
                f'{name} with shape {tuple(gates.shape)} is not broadcastable to {target}') from exc
        self._check_unit_interval(gates, name)
        return gates

    def _coerce_betas(self, value, B: int, T: int, *, device, dtype,
                      name: str = 'beta_override') -> torch.Tensor:
        """Canonicalize rate interventions to ``(B,T,2,D)``."""
        rates = torch.as_tensor(value, device=device, dtype=dtype)
        if rates.dim() == 1 and tuple(rates.shape) == (self.K,):
            rates = rates.view(1, 1, self.K, 1)
        elif rates.dim() == 1 and tuple(rates.shape) == (self.embed_dim,):
            rates = rates.view(1, 1, 1, self.embed_dim)
        elif rates.dim() == 2 and tuple(rates.shape) == (self.K, self.embed_dim):
            rates = rates.view(1, 1, self.K, self.embed_dim)
        elif rates.dim() == 2 and tuple(rates.shape) == (T, self.K):
            rates = rates.view(1, T, self.K, 1)
        elif rates.dim() == 3 and tuple(rates.shape) == (B, self.K, self.embed_dim):
            rates = rates.unsqueeze(1)
        elif rates.dim() == 3 and tuple(rates.shape) == (T, self.K, self.embed_dim):
            rates = rates.unsqueeze(0)
        target = (B, T, self.K, self.embed_dim)
        try:
            rates = torch.broadcast_to(rates, target)
        except RuntimeError as exc:
            raise ValueError(
                f'{name} with shape {tuple(rates.shape)} is not broadcastable to {target}') from exc
        self._check_unit_interval(rates, name)
        return rates

    def _beta_sequence(self, B: int, T: int, reference: torch.Tensor,
                       beta_override=None) -> torch.Tensor:
        if beta_override is not None:
            return self._coerce_betas(
                beta_override, B, T, device=reference.device, dtype=reference.dtype)
        return self.betas.to(device=reference.device, dtype=reference.dtype).view(
            1, 1, self.K, self.embed_dim).expand(B, T, -1, -1)

    def route_weights(self) -> torch.Tensor:
        """Global simplex route; ``single`` deterministically reads the medium state."""
        if self.mode == 'single':
            route = torch.zeros_like(self.route_logits)
            route[1] = 1.0
            return route
        return torch.softmax(self.route_logits, dim=0)

    def _validate_states(self, states: torch.Tensor) -> int:
        expected_tail = (self.K, self.embed_dim)
        if not isinstance(states, torch.Tensor) or states.dim() != 3:
            shape = tuple(states.shape) if isinstance(states, torch.Tensor) else type(states).__name__
            raise ValueError(f'states must have shape (B,{self.K},{self.embed_dim}), got {shape}')
        B, K, D = states.shape
        if B < 1 or (K, D) != expected_tail:
            raise ValueError(
                f'states must have shape (B,{self.K},{self.embed_dim}), got {tuple(states.shape)}')
        if not states.is_floating_point() or not torch.isfinite(states).all():
            raise ValueError('states must contain finite floating-point values')
        return B

    def _action_prior_unchecked(self, states: torch.Tensor, action: torch.Tensor,
                                beta: torch.Tensor) -> torch.Tensor:
        B = states.shape[0]
        if self.mode in {'noaction', 'ssmcontrol'}:
            action_features = states.new_zeros(B, 2 * self.embed_dim)
        else:
            action_features = self.W_a(action)
        d, v = action_features.chunk(2, dim=-1)
        normalized = F.layer_norm(states, (self.embed_dim,))
        delta = torch.tanh(v.unsqueeze(1) + d.unsqueeze(1) * normalized)
        return states + beta * delta

    def transition(self, states: torch.Tensor, action: torch.Tensor,
                   action_override=None, beta_override=None) -> torch.Tensor:
        """Validated one-step action-only transition."""
        B = self._validate_states(states)
        if not isinstance(action, torch.Tensor) or tuple(action.shape) != (B, self.action_dim):
            shape = tuple(action.shape) if isinstance(action, torch.Tensor) else type(action).__name__
            raise ValueError(f'action must have shape {(B, self.action_dim)}, got {shape}')
        if not action.is_floating_point() or not torch.isfinite(action).all():
            raise ValueError('action must contain finite floating-point values')
        action = action.to(device=states.device, dtype=states.dtype)
        if action_override is not None:
            action = self._coerce_action_override(action_override, action, 'action_override')
        beta = self._beta_sequence(B, 1, states, beta_override)[:, 0]
        return self._action_prior_unchecked(states, action, beta)

    def action_rollout(self, source_states: torch.Tensor, actions: torch.Tensor,
                       action_override=None, beta_override=None) -> torch.Tensor:
        """Advance source states with actions only; no future observation is consumed."""
        B = self._validate_states(source_states)
        if not isinstance(actions, torch.Tensor) or actions.dim() != 3:
            shape = tuple(actions.shape) if isinstance(actions, torch.Tensor) else type(actions).__name__
            raise ValueError(
                f'rollout actions must have shape (B,H,{self.action_dim}), got {shape}')
        H = actions.shape[1]
        actions = self._validate_actions(actions, B, H, name='rollout actions')
        actions = actions.to(device=source_states.device, dtype=source_states.dtype)
        if action_override is not None:
            actions = self._coerce_action_override(action_override, actions, 'action_override')
        beta_sequence = self._beta_sequence(B, H, source_states, beta_override)

        state = source_states
        trajectory = []
        for step in range(H):
            state = self._action_prior_unchecked(state, actions[:, step], beta_sequence[:, step])
            trajectory.append(state)
        if not trajectory:
            return source_states.new_empty(B, 0, self.K, self.embed_dim)
        return torch.stack(trajectory, dim=1)

    def _dynamic_gate(self, z_t: torch.Tensor, x_t: torch.Tensor,
                      prior: torch.Tensor) -> torch.Tensor:
        normalized_z = F.layer_norm(z_t, (self.embed_dim,))
        normalized_error = F.layer_norm(x_t.unsqueeze(1) - prior, (self.embed_dim,))
        z_score = torch.einsum('bd,d->b', normalized_z, self.w_z).unsqueeze(-1)
        error_score = torch.einsum('bkd,d->bk', normalized_error, self.w_e)
        logits = ((z_score + error_score) / math.sqrt(self.embed_dim)
                  + self.gate_bias.view(1, self.K))
        return torch.sigmoid(logits).unsqueeze(-1)

    def forward(self, z: torch.Tensor, actions: torch.Tensor,
                memory_update_mask: torch.Tensor = None, gate_override=None,
                action_override=None, beta_override=None, return_details: bool = False):
        """Return normalized memory and optionally exact states, gates, priors, and rates.

        ``actions[:,t]`` strictly maps ``z[:,t]`` to ``z[:,t+1]``.  A supplied update mask is
        shape/range checked but ignored, preventing visibility metadata from changing any V5 mode.
        ``details['betas']`` has shape ``(B,T,2,D)`` and records rates actually used, including an
        optional intervention.  The warm-start entry at index zero is recorded but not applied.
        """
        B, T, _ = self._validate_latents(z)
        actions = self._validate_actions(actions, B, T - 1)
        actions = actions.to(device=z.device, dtype=z.dtype)
        if action_override is not None:
            actions = self._coerce_action_override(action_override, actions, 'action_override')

        if memory_update_mask is not None:
            self._coerce_gates(
                memory_update_mask, B, T, device=z.device, dtype=z.dtype,
                name='memory_update_mask')
        override_gates = None
        if gate_override is not None:
            override_gates = self._coerce_gates(
                gate_override, B, T, device=z.device, dtype=z.dtype, name='gate_override')
        beta_sequence = self._beta_sequence(B, T, z, beta_override)

        x = self.W_x(z)
        initial = x[:, 0].unsqueeze(1).expand(-1, self.K, -1)
        states = [initial]
        priors = [initial]
        if override_gates is not None:
            gate_0 = override_gates[:, 0]
        elif self.mode == 'static':
            gate_0 = torch.sigmoid(self.gate_bias).view(1, self.K, 1).expand(B, -1, -1)
        elif self.mode == 'ssmcontrol':
            gate_0 = z.new_ones(B, self.K, 1)
        else:
            gate_0 = self._dynamic_gate(z[:, 0], x[:, 0], initial)
        gates = [gate_0]

        state = initial
        for t in range(1, T):
            beta = beta_sequence[:, t]
            prior = self._action_prior_unchecked(state, actions[:, t - 1], beta)
            if override_gates is not None:
                gate = override_gates[:, t]
            elif self.mode == 'static':
                gate = torch.sigmoid(self.gate_bias).view(1, self.K, 1).expand(B, -1, -1)
            elif self.mode == 'ssmcontrol':
                gate = z.new_ones(B, self.K, 1)
            else:
                gate = self._dynamic_gate(z[:, t], x[:, t], prior)
            state = prior + beta * gate * (x[:, t].unsqueeze(1) - prior)
            priors.append(prior)
            states.append(state)
            gates.append(gate)

        state_sequence = torch.stack(states, dim=1)
        route = self.route_weights().to(dtype=z.dtype)
        mixed = (state_sequence * route.view(1, 1, self.K, 1)).sum(dim=2)
        mixed = mixed * torch.rsqrt(
            mixed.square().mean(dim=-1, keepdim=True) + self.rms_eps)
        if not return_details:
            return mixed
        return mixed, {
            'x': x,
            'priors': torch.stack(priors, dim=1),
            'states': state_sequence,
            'gates': torch.stack(gates, dim=1),
            'route': route,
            'betas': beta_sequence,
        }

    def fuse(self, z: torch.Tensor, mixed: torch.Tensor) -> torch.Tensor:
        """Zero-initialized residual fusion ``z + W_o mixed``."""
        if not isinstance(z, torch.Tensor) or not isinstance(mixed, torch.Tensor):
            raise ValueError('z and mixed must be tensors with shape (B,T,D)')
        if z.dim() != 3 or mixed.dim() != 3 or z.shape != mixed.shape:
            z_shape = tuple(z.shape) if isinstance(z, torch.Tensor) else type(z).__name__
            mixed_shape = tuple(mixed.shape) if isinstance(mixed, torch.Tensor) else type(mixed).__name__
            raise ValueError(
                f'z and mixed must have identical (B,T,D) shapes, got {z_shape} and {mixed_shape}')
        if z.shape[-1] != self.embed_dim:
            raise ValueError(f'HACSSM-v5 expected final dim {self.embed_dim}, got {z.shape[-1]}')
        if not z.is_floating_point() or not mixed.is_floating_point():
            raise ValueError('z and mixed must be floating-point tensors')
        if not torch.isfinite(z).all() or not torch.isfinite(mixed).all():
            raise ValueError('z and mixed must contain only finite values')
        return z + self.W_o(mixed)

    def horizons(self) -> Dict[str, float]:
        """Summarize current per-channel rates/horizons for experiment logging."""
        with torch.no_grad():
            beta = self.betas
            tau = alpha_to_tau(beta)
            return {
                'tau_fast': float(tau[0].median()),
                'tau_slow': float(tau[1].median()),
                'alpha_fast': float(beta[0].median()),
                'alpha_slow': float(beta[1].median()),
                'tau_fast_min': float(tau[0].min()),
                'tau_fast_max': float(tau[0].max()),
                'tau_medium_min': float(tau[1].min()),
                'tau_medium_max': float(tau[1].max()),
                'n_banks': self.K,
            }


# Short name used by experiment configuration and checkpoint metadata.
HACSSMv5Memory = HierarchicalActionConditionedSSMMemory


class OCSMTMemory(nn.Module):
    """OC-SMT: Over-Complete fixed log-spaced EMA basis + L0 (hard-concrete) sparse READ gates.

    Learns an effective read-bank cardinality under a fixed over-complete ceiling M
    (docs/LEARNABLE_MEMORY.md §8). We keep the decays FIXED (learning them fails, §5.4), enlarge the
    basis, and learn a per-bank hard-concrete L0 gate, so 'how many banks are read' is a
    differentiable, penalized quantity rather than a hand-set K:

        gate logit   l_{t,m} = (W_g z_t)_m
        hard-concrete g_{t,m} ∈ {0} ∪ (0,1]   (exact zeros via the stretch [gamma,zeta], Louizos 2018)
        write gate   i_t = sigmoid(W_i z_t) ;  m^m_t = (1-a_m) m^m_{t-1} + a_m (i_t ⊙ z_t)   (a_m FIXED)
        read-out     o_t = W_o( Σ_m g_{t,m} m^m_t ) ;  z~_t = z_t + o_t        (additive, as SMT-v2)
        L0 penalty   λ0 · Σ_m P(g_{t,m} > 0)                                   (cost for being dense)

    The L0 penalty makes a uniform/dense gate costly, but does not guarantee a useful sparse set: the
    experiments find only a narrow band of static partial masks after task utility largely collapses.
    Decays never change. The current full-sequence code computes all M banks and treats gate
    cardinality as an analysis statistic; physical pruning or sparse evaluation is future work."""

    BETA, GAMMA, ZETA = 2.0 / 3.0, -0.1, 1.1            # hard-concrete temperature + stretch

    def __init__(self, embed_dim: int, M: int = 28, tau_min: float = 1.5,
                 tau_max: float = 256.0, stochastic_gates: bool = True):
        super().__init__()
        if M < 1:
            raise ValueError(f"M must be positive, got {M}")
        if not 0 < tau_min <= tau_max:
            raise ValueError(f"expected 0 < tau_min <= tau_max, got {tau_min}, {tau_max}")
        self.embed_dim = embed_dim
        self.M = M
        self.stochastic_gates = stochastic_gates
        taus = torch.logspace(math.log10(tau_min), math.log10(tau_max), M)
        self.register_buffer('taus', taus)
        self.register_buffer('alphas', 1.0 - torch.exp(-1.0 / taus))   # (M,) FIXED
        self.in_gate = nn.Linear(embed_dim, embed_dim)
        nn.init.constant_(self.in_gate.bias, 1.0)         # write gate starts ~open
        self.gate = nn.Linear(embed_dim, M)               # W_g: hard-concrete gate logits
        nn.init.constant_(self.gate.bias, 1.0)            # start banks OPEN so memory is used; an ANNEALED
        #   L0 (populate-then-sparsify, in train_memory) then prunes. NOTE (empirical): on clean short
        #   tasks useful performance remains near-dense; a narrow lambda band yields static partial
        #   masks only after long-gap usage largely collapses. See docs/LEARNABLE_MEMORY.md §9.
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        # Keep the initial summed residual approximately invariant when ablating M. M=28 exactly
        # preserves the checkpoints reported in §9; smaller banks receive a proportionally larger
        # projection because fewer gated states are summed.
        self.readout_init_std = 5e-4 * (28.0 / M)
        nn.init.normal_(self.out.weight, std=self.readout_init_std)
        self.last_l0 = None                               # stashed expected #open gates (for the loss)

    def _sample_gate(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training and self.stochastic_gates:
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
    def active_count(self, z: torch.Tensor, thresh: float = 0.0) -> torch.Tensor:
        """Mean number of read gates above ``thresh`` per step (not physical state pruning)."""
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
