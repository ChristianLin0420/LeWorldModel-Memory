"""
MemoryLeWorldModel: LeWorldModel augmented with a two-timescale EMA memory.

Design goal (the elegant part): change as little as possible. The encoder, the AdaLN
predictor and SIGReg are all reused verbatim. We only:

  1. compute two EMA memory banks over the encoder latents (lewm.models.memory), and
  2. additively inject them into the latents the predictor consumes (zero-init, so we
     start *exactly* at the memoryless baseline).

Crucially the predictor still attends over a window of only `history_len` latents -- we
train it with a *sliding short window* over a longer chunk, so any information that has
to travel further than `history_len` steps can only do so through the EMA memory. This
isolates the memory's contribution: it is the sole long-range channel.

Loss is unchanged in form:  L = L_pred + lambda * SIGReg(Z)   (2 terms, 1 lambda).
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from lewm.models.leworldmodel import LeWorldModel
from lewm.models.memory import (TwoTimescaleMemory, MemoryFusion, MultiTimescaleMemory,
                                GRUMemory, SSMMemory, RetrievalMemory)


class MemoryLeWorldModel(LeWorldModel):
    """LeWorldModel + two-timescale EMA memory injected into the predictor input.

    Extra args (everything else is inherited):
        memory_mode: 'none' | 'short' | 'long' | 'both' (the four ablations).
        tau_fast / tau_slow: initial effective horizons (steps) of the fast/slow banks.
        learnable_alpha: whether the EMA rates are learned (tau is always logged).
    """

    def __init__(
        self,
        *args,
        memory_mode: str = 'both',
        tau_fast: float = 2.0,
        tau_slow: float = 20.0,
        learnable_alpha: bool = True,
        memory_impl: str = 'ema',
        multi_taus=(2, 4, 8, 16, 32, 64),
        gru_hidden: int = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory_mode = memory_mode
        self.memory_impl = memory_impl
        # 'ema' is the default two-timescale design (unchanged param names -> old checkpoints load).
        # 'multi' (E3, log-spaced K-bank) and 'gru' (E2 learned-recurrent baseline) are additive.
        if memory_impl == 'ema':
            self.memory = TwoTimescaleMemory(
                embed_dim=self.embed_dim, tau_fast=tau_fast, tau_slow=tau_slow, learnable=learnable_alpha)
            self.fusion = MemoryFusion(embed_dim=self.embed_dim, mode=memory_mode)
        elif memory_impl == 'multi':
            self.mem_multi = MultiTimescaleMemory(embed_dim=self.embed_dim, taus=multi_taus)
        elif memory_impl == 'gru':
            self.mem_gru = GRUMemory(embed_dim=self.embed_dim, hidden=gru_hidden)
        elif memory_impl == 'ssm':
            self.mem_ssm = SSMMemory(embed_dim=self.embed_dim)
        elif memory_impl == 'retrieval':
            self.mem_ret = RetrievalMemory(embed_dim=self.embed_dim, num_heads=4)
        else:
            raise ValueError(f"unknown memory_impl '{memory_impl}'")

    def _inject(self, z: torch.Tensor) -> torch.Tensor:
        """Return the memory-augmented latents z~ the predictor consumes (branches by impl)."""
        if self.memory_impl == 'ema':
            m_fast, m_slow = self.memory(z)
            return self.fusion(z, m_fast, m_slow)
        if self.memory_impl == 'multi':
            return self.mem_multi.fuse(z, self.mem_multi.banks(z))
        if self.memory_impl == 'gru':
            return self.mem_gru.fuse(z, self.mem_gru(z))
        if self.memory_impl == 'ssm':
            return self.mem_ssm.fuse(z, self.mem_ssm(z))
        return self.mem_ret.fuse(z, self.mem_ret(z))  # retrieval

    def horizons(self):
        """Uniform horizon accessor across impls (for logging)."""
        if self.memory_impl == 'ema':
            return self.memory.horizons()
        if self.memory_impl == 'multi':
            return self.mem_multi.horizons()
        if self.memory_impl == 'gru':
            return self.mem_gru.horizons()
        if self.memory_impl == 'ssm':
            return self.mem_ssm.horizons()
        return self.mem_ret.horizons()

    # ---- core training loss (sliding short-window over a long chunk) ---------------
    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Two-term loss over a length-L chunk.

        Args:
            observations: (B, L, C, H, W) -- a contiguous chunk (L can be >> history_len).
            actions: (B, L-1, A) -- action a_t maps obs_t -> obs_{t+1}.
        Returns:
            dict with 'loss', 'pred_loss', 'sigreg_loss'.
        """
        B, L = observations.shape[0], observations.shape[1]
        h = self.history_len
        D, A = self.embed_dim, self.action_dim
        assert L >= h + 1, f"chunk length L={L} must be >= history_len+1={h + 1}"

        # Encode all frames (memoryless, per-frame).
        z = self.encode(observations)                      # (B, L, D)

        # Memory over the full causal history (ema / multi / gru), injected into the predictor input.
        z_tilde = self._inject(z)                          # (B, L, D)

        # Sliding windows of length h: window s = z~[s : s+h] predicts z_{s+h}.
        # Number of windows W = L - h (s = 0 .. L-h-1).
        W = L - h
        zt_win = z_tilde.unfold(1, h, 1)[:, :W]            # (B, W, D, h)
        zt_win = zt_win.permute(0, 1, 3, 2).reshape(B * W, h, D)
        act_win = actions.unfold(1, h, 1)[:, :W]           # (B, W, A, h)
        act_win = act_win.permute(0, 1, 3, 2).reshape(B * W, h, A)
        targets = z[:, h:L].reshape(B * W, D)              # z_{s+h}

        z_pred = self.predictor(zt_win, act_win)           # (B*W, h, D)
        z_pred_last = z_pred[:, -1, :]                     # predict next from last token

        pred_loss = F.mse_loss(z_pred_last, targets)
        sigreg_loss = self.sigreg(z.reshape(B * L, D))     # Gaussianize all latents
        total = pred_loss + self.sigreg_lambda * sigreg_loss
        return {'loss': total, 'pred_loss': pred_loss, 'sigreg_loss': sigreg_loss}

    def forward(self, observations, actions):
        return self.compute_loss(observations, actions)

    # ---- analysis utilities (used by probing / visualization) ----------------------
    @torch.no_grad()
    def encode_with_memory(self, observations: torch.Tensor):
        """Return (z, m_fast, m_slow, z_tilde). m_fast/m_slow are None for non-EMA impls."""
        z = self.encode(observations)
        if self.memory_impl == 'ema':
            m_fast, m_slow = self.memory(z)
            return z, m_fast, m_slow, self.fusion(z, m_fast, m_slow)
        return z, None, None, self._inject(z)

    @torch.no_grad()
    def memory_influence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Causal influence of each memory bank on the predicted next latent.

        We predict z_{L-1} (the last observed frame) from the window ending at L-2 and
        measure how much that prediction *moves* when a bank is ablated:

            infl_fast = || f(full) - f(ablate fast) ||_2 ,   similarly for slow.

        This is the empirical decision-influence of short- vs long-term memory.

        Args:
            observations: (B, L, C, H, W), L >= history_len + 1.
            actions: (B, L-1, A).
        Returns:
            dict: 'pred_full' (B, D), 'infl_fast' (B,), 'infl_slow' (B,).
        """
        h = self.history_len
        z = self.encode(observations)
        L = z.shape[1]
        assert L >= h + 1
        wsl = slice(L - 1 - h, L - 1)
        act = actions[:, wsl]

        if self.memory_impl == 'ema':
            m_fast, m_slow = self.memory(z)

            def pred(ablate_fast: bool, ablate_slow: bool) -> torch.Tensor:
                zt = self.fusion(z, m_fast, m_slow, ablate_fast=ablate_fast, ablate_slow=ablate_slow)
                return self.predictor(zt[:, wsl], act)[:, -1, :]

            full = pred(False, False)
            return {'pred_full': full,
                    'infl_fast': (full - pred(True, False)).norm(dim=-1),
                    'infl_slow': (full - pred(False, True)).norm(dim=-1)}
        # non-EMA: total memory influence = ablate ALL memory (fused window vs raw window)
        full = self.predictor(self._inject(z)[:, wsl], act)[:, -1, :]
        nomem = self.predictor(z[:, wsl], act)[:, -1, :]
        infl_all = (full - nomem).norm(dim=-1)
        return {'pred_full': full, 'infl_fast': infl_all, 'infl_slow': infl_all}

    @torch.no_grad()
    def rollout_latents(
        self,
        context_obs: torch.Tensor,
        future_actions: torch.Tensor,
        horizon: int,
        ablate_fast: bool = False,
        ablate_slow: bool = False,
    ) -> torch.Tensor:
        """Memory-aware autoregressive latent rollout (for imagination / planning).

        Seeds the EMA state from an observed context, then rolls forward, updating the
        memory with each *predicted* latent.

        Args:
            context_obs: (B, Lc, C, H, W) observed context (Lc >= history_len).
            future_actions: (B, horizon, A) actions to imagine.
            horizon: number of steps to roll out.
        Returns:
            z_future: (B, horizon, D) predicted latents.
        """
        h = self.history_len
        z = self.encode(context_obs)                       # (B, Lc, D)
        m_fast, m_slow = self.memory(z)                    # (B, Lc, D)
        mf, ms = m_fast[:, -1], m_slow[:, -1]              # (B, D) current memory state
        window = list(z[:, -h:].unbind(dim=1))             # last h latents

        preds = []
        for t in range(horizon):
            z_win = torch.stack(window[-h:], dim=1)        # (B, h, D)
            mf_b = mf.unsqueeze(1).expand(-1, h, -1)
            ms_b = ms.unsqueeze(1).expand(-1, h, -1)
            zt = self.fusion(z_win, mf_b, ms_b, ablate_fast=ablate_fast, ablate_slow=ablate_slow)
            a_t = future_actions[:, t:t + 1].expand(-1, h, -1)
            z_next = self.predictor(zt, a_t)[:, -1, :]     # (B, D)
            preds.append(z_next)
            mf, ms = self.memory.step(mf, ms, z_next)      # advance memory with prediction
            window.append(z_next)
        return torch.stack(preds, dim=1)
