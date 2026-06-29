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

import copy
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from lewm.models.leworldmodel import LeWorldModel
from lewm.models.memory import (TwoTimescaleMemory, MemoryFusion, MultiTimescaleMemory,
                                GRUMemory, SSMMemory, RetrievalMemory,
                                SelectiveMultiTimescaleMemory, SelectiveUpdateMemoryV3,
                                HierarchicalActionConditionedMemory,
                                HierarchicalActionConditionedSSMMemory,
                                HierarchicalCounterfactualRecoveryMemory,
                                SharedActionShrinkageMemory,
                                LearnedOrderedInnovationFilterMemory,
                                OrthogonalRecurrentBeliefMemory,
                                KickDriftInnovationObserverMemory, OCSMTMemory)


_SMTV3_MODES = {
    'smtv3': 'dynamic',
    'smtv3_static': 'static',
    'smtv3_oracle': 'oracle',
    'smtv3_old': 'old_update',
}

_HACSMV4_MODES = {
    'hacsmv4': 'dynamic',
    'hacsmv4_static': 'static',
    'hacsmv4_noaction': 'noaction',
    'hacsmv4_noaux': 'dynamic',
    'hacsmv4_single': 'single',
    'hacsmv4_oracle': 'oracle',
    'hacsmv4_two_noaux': 'dynamic',
}

# Each level is supervised only at horizons appropriate to its structural timescale.
_HACSMV4_AUX_HORIZONS = ((0, 1), (0, 2), (1, 4), (1, 8), (2, 16))

_HACSSMV5_MODES = {
    'hacssmv5': 'dynamic',
    'hacssmv5_static': 'static',
    'hacssmv5_noaction': 'noaction',
    'hacssmv5_fixedbeta': 'fixedbeta',
    'hacssmv5_fixedbeta_noaux': 'fixedbeta',
    'hacssmv5_noaux': 'dynamic',
    'hacssmv5_single': 'single',
    'hacssmv5_ssmcontrol': 'ssmcontrol',
}

# V5 deliberately drops V4's harmful slow level and uses only first-visible boundary targets.
_HACSSMV5_AUX_HORIZONS = ((0, 1), (0, 2), (1, 4), (1, 8))


# HACSSM-v6 deliberately returns to the strongest V4 inference architecture: two fixed scalar
# timescales.  The variants below differ only in the training-only consistency objective unless
# their names explicitly identify an inference mechanism control.
_HACSSMV6_MODES = {
    'hacssmv6': 'dynamic',
    'hacssmv6_noaux': 'dynamic',
    'hacssmv6_aux_noaction': 'dynamic',
    'hacssmv6_uniform': 'dynamic',
    'hacssmv6_sourcegrad': 'dynamic',
    'hacssmv6_fastonly': 'dynamic',
    'hacssmv6_mediumonly': 'dynamic',
    'hacssmv6_noaction': 'noaction',
    'hacssmv6_static': 'static',
    'hacssmv6_single': 'single',
}

_HACSSMV6_HIERARCHICAL_HORIZONS = ((0, 1), (0, 2), (1, 4), (1, 8))
_HACSSMV6_UNIFORM_HORIZONS = tuple(
    (level, horizon) for level in (0, 1) for horizon in (1, 2, 4, 8)
)

_HACSSMV7_MODES = {
    'hacssmv7': 'dynamic',
    'hacssmv7_noaux': 'noaux',
    'hacssmv7_sharedaction': 'sharedaction',
    'hacssmv7_noshrink': 'noshrink',
    'hacssmv7_actiononly': 'actiononly',
    'hacssmv7_uniform': 'uniform',
    'hacssmv7_norecovery': 'norecovery',
    'hacssmv7_noaction': 'noaction',
    'hacssmv7_single': 'single',
}
_HACSSMV7_HIERARCHICAL_HORIZONS = ((0, 1), (0, 2), (1, 4), (1, 8))
_HACSSMV7_UNIFORM_HORIZONS = tuple(
    (level, horizon) for level in (0, 1) for horizon in (1, 2, 4, 8)
)

_HACSSMV8_MODES = {
    'hacssmv8': 'learned',
    'hacssmv8_dynamic': 'rho1',
    'hacssmv8_static': 'rho0',
    'hacssmv8_levelaction': 'levelaction',
    'hacssmv8_redundant': 'redundant',
    'hacssmv8_noaction': 'noaction',
    'hacssmv8_single': 'single',
}

_LOIFV9_MODES = {
    'loifv9': 'learned',
    'loifv9_fixedalpha': 'fixedalpha',
    'loifv9_globalR': 'globalR',
    'loifv9_innovationonly': 'innovationonly',
    'loifv9_latentonly': 'latentonly',
    'loifv9_uniformfusion': 'uniformfusion',
    'loifv9_noaction': 'noaction',
    'loifv9_singlebank': 'singlebank',
}

_ORBITV10_MODES = {
    'orbitv10': 'orthogonal',
    'orbitv10_noaction': 'noaction',
    'orbitv10_additive': 'additive',
    'orbitv10_scaled': 'scaled',
    'orbitv10_static': 'static',
}

_KDIOV11_MODES = {
    'kdiov11': 'full',
    'kdiov11_unconstrained': 'unconstrained',
    'kdiov11_fixedscale': 'fixedscale',
    # These two names change only the training objective.  Their deployed inference path is
    # bit-identical to full KDIO and is selected by the V11 trainer, not the memory module.
    'kdiov11_h1': 'full',
    'kdiov11_noactionswap': 'full',
    'kdiov11_firstorder': 'firstorder',
    'kdiov11_nodrift': 'nodrift',
    'kdiov11_noaction': 'noaction',
    'kdiov11_noautonomy': 'noautonomy',
    'kdiov11_noreliability': 'noreliability',
    'kdiov11_static': 'static',
}


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
        encoder_type: str = 'vit',
        smt_router: str = 'softmax',
        oc_num: int = 28,
        oc_tau_min: float = 1.5,
        oc_tau_max: float = 256.0,
        oc_stochastic_gates: bool = True,
        l0_lambda: float = 0.0,
        hier_loss_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.memory_mode = memory_mode
        self.memory_impl = memory_impl
        self.encoder_type = encoder_type
        self.l0_lambda = l0_lambda
        if not math.isfinite(float(hier_loss_weight)) or float(hier_loss_weight) < 0.0:
            raise ValueError('hier_loss_weight must be non-negative and finite')
        self.hier_loss_weight = float(hier_loss_weight)
        if encoder_type == 'dino':                          # frozen pretrained DINOv2 backbone
            from lewm.models.encoder import FrozenDINOEncoder
            self.encoder = FrozenDINOEncoder(embed_dim=self.embed_dim)
        elif encoder_type == 'precomputed':                 # fixed external features loaded by dataset
            self.encoder = torch.nn.Identity()
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
        elif memory_impl == 'smt':                          # learnable selective multi-timescale
            self.mem_smt = SelectiveMultiTimescaleMemory(embed_dim=self.embed_dim, taus=multi_taus,
                                                         router_mode=smt_router)
        elif memory_impl in _SMTV3_MODES:                   # true selective-update SMT-v3 + controls
            self.mem_smtv3 = SelectiveUpdateMemoryV3(
                embed_dim=self.embed_dim, taus=multi_taus, mode=_SMTV3_MODES[memory_impl])
        elif memory_impl in _HACSMV4_MODES:                 # hierarchical action predict/correct memory
            self.mem_hacsmv4 = HierarchicalActionConditionedMemory(
                embed_dim=self.embed_dim, action_dim=self.action_dim,
                mode=_HACSMV4_MODES[memory_impl],
                taus=((2.0, 8.0) if memory_impl == 'hacsmv4_two_noaux' else None))
        elif memory_impl in _HACSSMV5_MODES:                # learned-rate action-conditioned SSM hierarchy
            self.mem_hacssmv5 = HierarchicalActionConditionedSSMMemory(
                embed_dim=self.embed_dim, action_dim=self.action_dim,
                mode=_HACSSMV5_MODES[memory_impl])
        elif memory_impl in _HACSSMV6_MODES:                # fixed-rate self-supervised hierarchy
            # A dedicated attribute/state-dict namespace keeps V6 checkpoints auditable even
            # though its inference primitive intentionally matches the V4 two-level control.
            self.mem_hacssmv6 = HierarchicalActionConditionedMemory(
                embed_dim=self.embed_dim, action_dim=self.action_dim,
                mode=_HACSSMV6_MODES[memory_impl], taus=(2.0, 8.0))
        elif memory_impl in _HACSSMV7_MODES:                # counterfactual-recovery hierarchy
            self.mem_hacssmv7 = HierarchicalCounterfactualRecoveryMemory(
                embed_dim=self.embed_dim, action_dim=self.action_dim,
                mode=_HACSSMV7_MODES[memory_impl])
            self.mem_hacssmv7_teacher = copy.deepcopy(self.mem_hacssmv7)
            self.mem_hacssmv7_teacher.requires_grad_(False)
            self.hier_teacher_momentum = 0.99
        elif memory_impl in _HACSSMV8_MODES:                # compact shared-action shrinkage
            self.mem_hacssmv8 = SharedActionShrinkageMemory(
                embed_dim=self.embed_dim, action_dim=self.action_dim,
                mode=_HACSSMV8_MODES[memory_impl])
        elif memory_impl in _LOIFV9_MODES:                  # learned ordered innovation filter
            self.mem_loifv9 = LearnedOrderedInnovationFilterMemory(
                embed_dim=self.embed_dim, action_dim=self.action_dim,
                mode=_LOIFV9_MODES[memory_impl])
        elif memory_impl in _ORBITV10_MODES:                # horizon-free orthogonal belief
            self.mem_orbitv10 = OrthogonalRecurrentBeliefMemory(
                embed_dim=self.embed_dim, action_dim=self.action_dim,
                mode=_ORBITV10_MODES[memory_impl])
        elif memory_impl in _KDIOV11_MODES:                 # kick--drift predictive observer
            self.mem_kdiov11 = KickDriftInnovationObserverMemory(
                embed_dim=self.embed_dim, action_dim=self.action_dim,
                mode=_KDIOV11_MODES[memory_impl])
        elif memory_impl == 'ocsmt':                        # over-complete basis + L0 sparse gates
            self.mem_ocsmt = OCSMTMemory(
                embed_dim=self.embed_dim, M=oc_num, tau_min=oc_tau_min,
                tau_max=oc_tau_max, stochastic_gates=oc_stochastic_gates)
        else:
            raise ValueError(f"unknown memory_impl '{memory_impl}'")

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == 'precomputed':
            if observations.dim() != 3 or observations.shape[-1] != self.embed_dim:
                raise ValueError(
                    f'precomputed encoder expects (B,L,{self.embed_dim}), got '
                    f'{tuple(observations.shape)}')
            return observations
        return super().encode(observations)

    def _inject(self, z: torch.Tensor, actions: torch.Tensor = None,
                memory_update_mask: torch.Tensor = None, gate_override=None,
                action_override=None, resistance_override=None,
                return_memory_details: bool = False):
        """Return the memory-augmented latents z~ the predictor consumes (branches by impl)."""
        if (return_memory_details and self.memory_impl not in _HACSMV4_MODES
                and self.memory_impl not in _HACSSMV5_MODES
                and self.memory_impl not in _HACSSMV6_MODES
                and self.memory_impl not in _HACSSMV7_MODES
                and self.memory_impl not in _HACSSMV8_MODES
                and self.memory_impl not in _LOIFV9_MODES
                and self.memory_impl not in _ORBITV10_MODES
                and self.memory_impl not in _KDIOV11_MODES):
            raise ValueError(
                'memory details are available only for HACSM-v4/HACSSM-v5/HACSSM-v6/'
                'v7/v8/LOIF-v9/ORBIT-v10/KDIO-v11')
        if self.memory_impl == 'ema':
            m_fast, m_slow = self.memory(z)
            return self.fusion(z, m_fast, m_slow)
        if self.memory_impl == 'multi':
            return self.mem_multi.fuse(z, self.mem_multi.banks(z))
        if self.memory_impl == 'gru':
            return self.mem_gru.fuse(z, self.mem_gru(z))
        if self.memory_impl == 'ssm':
            return self.mem_ssm.fuse(z, self.mem_ssm(z))
        if self.memory_impl == 'smt':
            return self.mem_smt.fuse(z, self.mem_smt(z))
        if self.memory_impl in _SMTV3_MODES:
            mixed = self.mem_smtv3(
                z, memory_update_mask=memory_update_mask, gate_override=gate_override)
            return self.mem_smtv3.fuse(z, mixed)
        if self.memory_impl in _HACSMV4_MODES:
            if actions is None:
                raise ValueError('HACSM-v4 requires actions with a_t mapping z_t to z_{t+1}')
            result = self.mem_hacsmv4(
                z, actions, memory_update_mask=memory_update_mask,
                gate_override=gate_override, action_override=action_override,
                return_details=return_memory_details)
            if return_memory_details:
                mixed, details = result
                return self.mem_hacsmv4.fuse(z, mixed), details
            return self.mem_hacsmv4.fuse(z, result)
        if self.memory_impl in _HACSSMV5_MODES:
            if actions is None:
                raise ValueError('HACSSM-v5 requires actions with a_t mapping z_t to z_{t+1}')
            result = self.mem_hacssmv5(
                z, actions, memory_update_mask=memory_update_mask,
                gate_override=gate_override, action_override=action_override,
                return_details=return_memory_details)
            if return_memory_details:
                mixed, details = result
                return self.mem_hacssmv5.fuse(z, mixed), details
            return self.mem_hacssmv5.fuse(z, result)
        if self.memory_impl in _HACSSMV6_MODES:
            if actions is None:
                raise ValueError('HACSSM-v6 requires actions with a_t mapping z_t to z_{t+1}')
            result = self.mem_hacssmv6(
                z, actions, memory_update_mask=memory_update_mask,
                gate_override=gate_override, action_override=action_override,
                return_details=return_memory_details)
            if return_memory_details:
                mixed, details = result
                return self.mem_hacssmv6.fuse(z, mixed), details
            return self.mem_hacssmv6.fuse(z, result)
        if self.memory_impl in _HACSSMV7_MODES:
            if actions is None:
                raise ValueError('HACSSM-v7 requires actions with a_t mapping z_t to z_{t+1}')
            result = self.mem_hacssmv7(
                z, actions, memory_update_mask=memory_update_mask,
                gate_override=gate_override, action_override=action_override,
                return_details=return_memory_details)
            if return_memory_details:
                mixed, details = result
                return self.mem_hacssmv7.fuse(z, mixed), details
            return self.mem_hacssmv7.fuse(z, result)
        if self.memory_impl in _HACSSMV8_MODES:
            if actions is None:
                raise ValueError('HACSSM-v8 requires actions with a_t mapping z_t to z_{t+1}')
            result = self.mem_hacssmv8(
                z, actions, memory_update_mask=memory_update_mask,
                gate_override=gate_override, action_override=action_override,
                return_details=return_memory_details)
            if return_memory_details:
                mixed, details = result
                return self.mem_hacssmv8.fuse(z, mixed), details
            return self.mem_hacssmv8.fuse(z, result)
        if self.memory_impl in _LOIFV9_MODES:
            if actions is None:
                raise ValueError('LOIF-v9 requires actions with a_t mapping z_t to z_{t+1}')
            if gate_override is not None:
                raise ValueError(
                    'LOIF-v9 has no gate override; use resistance_override for diagnostics')
            result = self.mem_loifv9(
                z, actions, memory_update_mask=memory_update_mask,
                action_override=action_override, resistance_override=resistance_override,
                return_details=return_memory_details)
            if return_memory_details:
                mixed, details = result
                return self.mem_loifv9.fuse(z, mixed), details
            return self.mem_loifv9.fuse(z, result)
        if self.memory_impl in _ORBITV10_MODES:
            if actions is None:
                raise ValueError('ORBIT-v10 requires actions with a_t mapping z_t to z_{t+1}')
            if resistance_override is not None:
                raise ValueError('ORBIT-v10 has no resistance override; use gate_override')
            result = self.mem_orbitv10(
                z, actions, memory_update_mask=memory_update_mask,
                gate_override=gate_override, action_override=action_override,
                return_details=return_memory_details)
            if return_memory_details:
                mixed, details = result
                return self.mem_orbitv10.fuse(z, mixed), details
            return self.mem_orbitv10.fuse(z, result)
        if self.memory_impl in _KDIOV11_MODES:
            if actions is None:
                raise ValueError('KDIO-v11 requires actions with a_t mapping z_t to z_{t+1}')
            if resistance_override is not None:
                raise ValueError('KDIO-v11 has no resistance override; use gate_override')
            result = self.mem_kdiov11(
                z, actions, memory_update_mask=memory_update_mask,
                gate_override=gate_override, action_override=action_override,
                return_details=return_memory_details)
            if return_memory_details:
                mixed, details = result
                return self.mem_kdiov11.fuse(z, mixed), details
            return self.mem_kdiov11.fuse(z, result)
        if self.memory_impl == 'ocsmt':
            return self.mem_ocsmt.fuse(z, self.mem_ocsmt(z))
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
        if self.memory_impl == 'smt':
            return self.mem_smt.horizons()
        if self.memory_impl in _SMTV3_MODES:
            return self.mem_smtv3.horizons()
        if self.memory_impl in _HACSMV4_MODES:
            return self.mem_hacsmv4.horizons()
        if self.memory_impl in _HACSSMV5_MODES:
            return self.mem_hacssmv5.horizons()
        if self.memory_impl in _HACSSMV6_MODES:
            return self.mem_hacssmv6.horizons()
        if self.memory_impl in _HACSSMV7_MODES:
            return self.mem_hacssmv7.horizons()
        if self.memory_impl in _HACSSMV8_MODES:
            return self.mem_hacssmv8.horizons()
        if self.memory_impl in _LOIFV9_MODES:
            return self.mem_loifv9.horizons()
        if self.memory_impl in _ORBITV10_MODES:
            return self.mem_orbitv10.horizons()
        if self.memory_impl in _KDIOV11_MODES:
            return self.mem_kdiov11.horizons()
        if self.memory_impl == 'ocsmt':
            return self.mem_ocsmt.horizons()
        return self.mem_ret.horizons()

    def _hierarchical_auxiliary_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
        target_valid_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Observation-free multi-horizon prediction from online posterior states.

        A source state at time ``t`` is rolled only with ``a_t ... a_{t+h-1}``; it never reuses
        an online state at the endpoint because that state has already consumed intervening
        observations.  Only endpoint validity determines whether a pair is supervised.
        """
        if self.memory_impl not in _HACSMV4_MODES:
            raise ValueError('hierarchical auxiliary loss requires HACSM-v4')
        if states.dim() != 4:
            raise ValueError(f'expected states (B,T,K,D), got {tuple(states.shape)}')
        B, T, K, D = states.shape
        if tuple(targets.shape) != (B, T, D):
            raise ValueError(f'target shape {tuple(targets.shape)} != {(B, T, D)}')
        if tuple(actions.shape) != (B, T - 1, self.action_dim):
            raise ValueError(
                f'action shape {tuple(actions.shape)} != {(B, T - 1, self.action_dim)}')
        if K != self.mem_hacsmv4.K:
            raise ValueError(f'state level count {K} != {self.mem_hacsmv4.K}')

        if target_valid_mask is None:
            mask = torch.ones(B, T, device=states.device, dtype=torch.bool)
            use_first_post = False
        else:
            if tuple(target_valid_mask.shape) != (B, T):
                raise ValueError(
                    f'target_valid_mask shape {tuple(target_valid_mask.shape)} != {(B, T)}')
            mask = target_valid_mask.to(device=states.device, dtype=torch.bool)
            use_first_post = True

        detached_targets = targets.detach()
        horizon_losses = {}
        configured_horizons = tuple(
            pair for pair in _HACSMV4_AUX_HORIZONS if pair[0] < K)
        level_terms = {level: [] for level in range(K)}
        for level, horizon in configured_horizons:
            n_sources = T - horizon
            if n_sources < 1:
                raise ValueError(
                    f'sequence length {T} is too short for HACSM-v4 horizon {horizon}')
            source = states[:, :n_sources].reshape(B * n_sources, K, D)
            action_windows = actions.unfold(1, horizon, 1)
            action_windows = action_windows.permute(0, 1, 3, 2).reshape(
                B * n_sources, horizon, self.action_dim)
            prediction = self.mem_hacsmv4.action_rollout(source, action_windows)[:, -1, level]
            prediction = prediction.reshape(B, n_sources, D)
            endpoint_targets = detached_targets[:, horizon:]
            per_pair = (prediction - endpoint_targets).square().mean(dim=-1)
            valid = mask[:, horizon:]
            if not bool(valid.any()):
                raise ValueError(f'no valid HACSM-v4 auxiliary targets at horizon {horizon}')
            all_valid = per_pair[valid].mean()
            loss_h = all_valid
            if use_first_post:
                first_post = valid & ~mask[:, horizon - 1:T - 1]
                if not bool(first_post.any()):
                    raise ValueError(
                        f'no masked-to-valid HACSM-v4 auxiliary target at horizon {horizon}')
                loss_h = 0.5 * all_valid + 0.5 * per_pair[first_post].mean()
            horizon_losses[f'hier_loss_h{horizon}'] = loss_h
            horizon_losses[f'hier_pairs_h{horizon}'] = valid.sum().to(dtype=states.dtype)
            level_terms[level].append(loss_h)

        fast = torch.stack(level_terms[0]).mean()
        medium = torch.stack(level_terms[1]).mean()
        level_losses = [fast, medium]
        result = {
            'hier_loss_fast': fast,
            'hier_loss_medium': medium,
            **horizon_losses,
        }
        if K == 3:
            slow = torch.stack(level_terms[2]).mean()
            level_losses.append(slow)
            result['hier_loss_slow'] = slow
        result['hier_loss'] = torch.stack(level_losses).mean()
        return result

    def _hierarchical_boundary_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
        target_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """V5 front-loaded action-rollout shaping at the first visible boundary only.

        Unlike V4's all-valid plus boundary objective, this loss contributes exactly the
        masked-to-visible endpoint for each configured horizon.  It is deliberately called
        boundary shaping rather than general self-supervision: all four targets are the canonical
        first-post frame, reached from different source times using actions only.
        """
        if self.memory_impl not in _HACSSMV5_MODES:
            raise ValueError('hierarchical boundary loss requires HACSSM-v5')
        if states.dim() != 4:
            raise ValueError(f'expected states (B,T,K,D), got {tuple(states.shape)}')
        B, T, K, D = states.shape
        if tuple(targets.shape) != (B, T, D):
            raise ValueError(f'target shape {tuple(targets.shape)} != {(B, T, D)}')
        if tuple(actions.shape) != (B, T - 1, self.action_dim):
            raise ValueError(
                f'action shape {tuple(actions.shape)} != {(B, T - 1, self.action_dim)}')
        if K != self.mem_hacssmv5.K:
            raise ValueError(f'state level count {K} != {self.mem_hacssmv5.K}')
        if target_valid_mask is None or tuple(target_valid_mask.shape) != (B, T):
            shape = None if target_valid_mask is None else tuple(target_valid_mask.shape)
            raise ValueError(f'HACSSM-v5 boundary shaping requires target mask {(B, T)}, got {shape}')

        mask = target_valid_mask.to(device=states.device, dtype=torch.bool)
        detached_targets = targets.detach()
        horizon_losses = {}
        level_terms = {0: [], 1: []}
        for level, horizon in _HACSSMV5_AUX_HORIZONS:
            n_sources = T - horizon
            if n_sources < 1:
                raise ValueError(
                    f'sequence length {T} is too short for HACSSM-v5 horizon {horizon}')
            source = states[:, :n_sources].reshape(B * n_sources, K, D)
            action_windows = actions.unfold(1, horizon, 1)
            action_windows = action_windows.permute(0, 1, 3, 2).reshape(
                B * n_sources, horizon, self.action_dim)
            prediction = self.mem_hacssmv5.action_rollout(
                source, action_windows)[:, -1, level].reshape(B, n_sources, D)
            per_pair = (prediction - detached_targets[:, horizon:]).square().mean(dim=-1)
            endpoint_valid = mask[:, horizon:]
            first_visible = endpoint_valid & ~mask[:, horizon - 1:T - 1]
            if not bool(first_visible.any()):
                raise ValueError(
                    f'no masked-to-visible HACSSM-v5 target at horizon {horizon}')
            loss_h = per_pair[first_visible].mean()
            horizon_losses[f'hier_loss_h{horizon}'] = loss_h
            horizon_losses[f'hier_pairs_h{horizon}'] = first_visible.sum().to(
                dtype=states.dtype)
            level_terms[level].append(loss_h)

        fast = torch.stack(level_terms[0]).mean()
        medium = torch.stack(level_terms[1]).mean()
        hierarchy = torch.stack((fast, medium)).mean()
        return {
            'hier_loss': hierarchy,
            'hier_loss_fast': fast,
            'hier_loss_medium': medium,
            **horizon_losses,
        }

    def _hierarchical_consistency_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """V6 dense, causal, same-level posterior consistency.

        Every valid endpoint contributes a training pair.  A detached source posterior is
        advanced using only the intervening actions and is matched to the detached posterior at
        the same hierarchy level.  Layer normalization makes the target scale-free, while
        Smooth-L1 limits the influence of outlying channels.  Hidden clean targets are never
        exposed: endpoint eligibility comes exclusively from ``target_valid_mask``.

        ``sourcegrad`` is the explicit ablation that allows gradients through source states;
        ``aux_noaction`` zeros only the actions supplied to this objective.  The inference
        recurrence is identical for those variants.
        """
        if self.memory_impl not in _HACSSMV6_MODES:
            raise ValueError('hierarchical consistency loss requires HACSSM-v6')
        if states.dim() != 4:
            raise ValueError(f'expected states (B,T,K,D), got {tuple(states.shape)}')
        B, T, K, D = states.shape
        if tuple(actions.shape) != (B, T - 1, self.action_dim):
            raise ValueError(
                f'action shape {tuple(actions.shape)} != {(B, T - 1, self.action_dim)}')
        if K != self.mem_hacssmv6.K:
            raise ValueError(f'state level count {K} != {self.mem_hacssmv6.K}')
        if target_valid_mask is None or tuple(target_valid_mask.shape) != (B, T):
            shape = None if target_valid_mask is None else tuple(target_valid_mask.shape)
            raise ValueError(
                f'HACSSM-v6 consistency requires target mask {(B, T)}, got {shape}')

        mask = target_valid_mask.to(device=states.device, dtype=torch.bool)
        if self.memory_impl == 'hacssmv6_uniform':
            configured_horizons = _HACSSMV6_UNIFORM_HORIZONS
        elif self.memory_impl == 'hacssmv6_fastonly':
            configured_horizons = tuple(
                pair for pair in _HACSSMV6_HIERARCHICAL_HORIZONS if pair[0] == 0)
        elif self.memory_impl == 'hacssmv6_mediumonly':
            configured_horizons = tuple(
                pair for pair in _HACSSMV6_HIERARCHICAL_HORIZONS if pair[0] == 1)
        else:
            configured_horizons = _HACSSMV6_HIERARCHICAL_HORIZONS

        level_names = {0: 'fast', 1: 'medium'}
        level_terms = {level: [] for level in range(K)}
        horizon_terms = {horizon: [] for _, horizon in configured_horizons}
        horizon_pairs = {horizon: [] for _, horizon in configured_horizons}
        result = {}
        for level, horizon in configured_horizons:
            n_sources = T - horizon
            if n_sources < 1:
                raise ValueError(
                    f'sequence length {T} is too short for HACSSM-v6 horizon {horizon}')
            source = states[:, :n_sources]
            if self.memory_impl != 'hacssmv6_sourcegrad':
                source = source.detach()
            source = source.reshape(B * n_sources, K, D)
            action_windows = actions.unfold(1, horizon, 1)
            action_windows = action_windows.permute(0, 1, 3, 2).reshape(
                B * n_sources, horizon, self.action_dim)
            if self.memory_impl == 'hacssmv6_aux_noaction':
                action_windows = torch.zeros_like(action_windows)
            prediction = self.mem_hacssmv6.action_rollout(
                source, action_windows)[:, -1, level].reshape(B, n_sources, D)
            endpoint = states[:, horizon:, level].detach()
            prediction = F.layer_norm(prediction, (D,))
            endpoint = F.layer_norm(endpoint, (D,))
            per_pair = F.smooth_l1_loss(prediction, endpoint, reduction='none').mean(dim=-1)
            valid = mask[:, horizon:]
            if not bool(valid.any()):
                raise ValueError(f'no visible HACSSM-v6 targets at horizon {horizon}')
            loss_h = per_pair[valid].mean()
            pair_count = valid.sum().to(dtype=states.dtype)
            level_name = level_names[level]
            result[f'hier_loss_{level_name}_h{horizon}'] = loss_h
            result[f'hier_pairs_{level_name}_h{horizon}'] = pair_count
            level_terms[level].append(loss_h)
            horizon_terms[horizon].append(loss_h)
            horizon_pairs[horizon].append(pair_count)

        active_level_losses = []
        zero = states.new_zeros(())
        for level, terms in level_terms.items():
            if terms:
                level_loss = torch.stack(terms).mean()
                result[f'hier_loss_{level_names[level]}'] = level_loss
                active_level_losses.append(level_loss)
            else:
                # Fixed result schema for the level-only ablations.  This zero is descriptive
                # and is intentionally excluded from the active-level hierarchy average.
                result[f'hier_loss_{level_names[level]}'] = zero
        for horizon, terms in horizon_terms.items():
            result[f'hier_loss_h{horizon}'] = torch.stack(terms).mean()
            result[f'hier_pairs_h{horizon}'] = torch.stack(horizon_pairs[horizon]).sum()
        result['hier_pairs'] = torch.stack(
            [pair_count for counts in horizon_pairs.values() for pair_count in counts]).sum()
        result['hier_loss'] = torch.stack(active_level_losses).mean()
        return result

    @torch.no_grad()
    def update_hierarchical_teacher(self) -> None:
        """EMA-update the V7 memory-only teacher after one optimizer step."""
        if self.memory_impl not in _HACSSMV7_MODES:
            return
        momentum = float(self.hier_teacher_momentum)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f'invalid hierarchy teacher momentum {momentum}')
        student_parameters = dict(self.mem_hacssmv7.named_parameters())
        teacher_parameters = dict(self.mem_hacssmv7_teacher.named_parameters())
        if student_parameters.keys() != teacher_parameters.keys():
            raise RuntimeError('V7 teacher/student parameter schemas differ')
        for name, teacher in teacher_parameters.items():
            teacher.mul_(momentum).add_(student_parameters[name], alpha=1.0 - momentum)
        student_buffers = dict(self.mem_hacssmv7.named_buffers())
        teacher_buffers = dict(self.mem_hacssmv7_teacher.named_buffers())
        if student_buffers.keys() != teacher_buffers.keys():
            raise RuntimeError('V7 teacher/student buffer schemas differ')
        for name, teacher in teacher_buffers.items():
            teacher.copy_(student_buffers[name])

    def _hierarchical_counterfactual_recovery_loss(
        self,
        states: torch.Tensor,
        z: torch.Tensor,
        actions: torch.Tensor,
        target_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """V7 visible-only counterfactual gap and recovery distillation.

        The EMA teacher consumes the original *occluded* trajectory.  For every contiguous
        window lying wholly inside an originally visible run, the student starts from a
        detached online state, replaces the next ``h`` visible latents by the canonical black
        latent already present in the input, and then consumes one restored visible latent.
        Bridge and recovery states are matched to stop-gradient teacher posteriors.  Fixed
        hidden clean frames are neither consumed by the teacher nor selected as targets.
        """
        if self.memory_impl not in _HACSSMV7_MODES:
            raise ValueError('counterfactual recovery loss requires HACSSM-v7')
        if states.dim() != 4:
            raise ValueError(f'expected states (B,T,K,D), got {tuple(states.shape)}')
        B, T, K, D = states.shape
        if tuple(z.shape) != (B, T, D):
            raise ValueError(f'z shape {tuple(z.shape)} != {(B, T, D)}')
        if tuple(actions.shape) != (B, T - 1, self.action_dim):
            raise ValueError(
                f'action shape {tuple(actions.shape)} != {(B, T - 1, self.action_dim)}')
        if K != self.mem_hacssmv7.K:
            raise ValueError(f'state level count {K} != {self.mem_hacssmv7.K}')
        if target_valid_mask is None or tuple(target_valid_mask.shape) != (B, T):
            shape = None if target_valid_mask is None else tuple(target_valid_mask.shape)
            raise ValueError(f'HACSSM-v7 requires target mask {(B, T)}, got {shape}')
        mask = target_valid_mask.to(device=states.device, dtype=torch.bool)
        hidden = ~mask
        hidden_counts = hidden.sum(dim=1)
        if bool((hidden_counts < 1).any()):
            raise ValueError('HACSSM-v7 requires an observed canonical black interval')

        # The black token comes only from the actually observed corrupted input.  All feature
        # transforms used in this auxiliary are detached so W_x remains primary-only.
        black = (z.detach() * hidden.unsqueeze(-1)).sum(dim=1)
        black = black / hidden_counts.to(dtype=z.dtype).unsqueeze(-1)
        with torch.no_grad():
            x_all = self.mem_hacssmv7.W_x(z.detach())
            x_black = self.mem_hacssmv7.W_x(black)
            _, teacher_details = self.mem_hacssmv7_teacher(
                z.detach(), actions.detach(), return_details=True)
            teacher_states = teacher_details['states'].detach()

        configured = (
            _HACSSMV7_UNIFORM_HORIZONS
            if self.memory_impl == 'hacssmv7_uniform'
            else _HACSSMV7_HIERARCHICAL_HORIZONS
        )
        level_names = {0: 'fast', 1: 'medium'}
        level_terms = {0: [], 1: []}
        horizon_terms: Dict[int, list[torch.Tensor]] = {}
        horizon_pairs: Dict[int, list[torch.Tensor]] = {}
        bridge_terms = []
        recovery_terms = []
        result: Dict[str, torch.Tensor] = {}
        total_pairs = states.new_zeros(())
        overlap_count = states.new_zeros(())

        for level, horizon in configured:
            # Each eligible window is [source, h synthetically hidden frames, restore].
            width = horizon + 2
            if T < width:
                raise ValueError(f'sequence length {T} is too short for V7 horizon {horizon}')
            eligible = mask.unfold(1, width, 1).all(dim=-1)
            batch_index, source_index = eligible.nonzero(as_tuple=True)
            if batch_index.numel() == 0:
                raise ValueError(f'no visible V7 counterfactual windows at horizon {horizon}')
            pair_count = eligible.sum().to(dtype=states.dtype)
            total_pairs = total_pairs + pair_count

            source = states[batch_index, source_index].detach()
            state = source
            for step in range(1, horizon + 1):
                action = actions[batch_index, source_index + step - 1]
                state, _, _ = self.mem_hacssmv7.correction_step(
                    state, black[batch_index], action,
                    x_t=x_black[batch_index])
            bridge = state[:, level]
            bridge_target = teacher_states[
                batch_index, source_index + horizon, level]
            bridge_error = F.smooth_l1_loss(
                F.layer_norm(bridge, (D,)),
                F.layer_norm(bridge_target, (D,)), reduction='none').mean(dim=-1)
            bridge_loss = bridge_error.mean()

            recovery_time = source_index + horizon + 1
            recovery_action = actions[batch_index, source_index + horizon]
            recovered, _, _ = self.mem_hacssmv7.correction_step(
                state, z[batch_index, recovery_time].detach(), recovery_action,
                x_t=x_all[batch_index, recovery_time])
            recovery_target = teacher_states[batch_index, recovery_time, level]
            recovery_error = F.smooth_l1_loss(
                F.layer_norm(recovered[:, level], (D,)),
                F.layer_norm(recovery_target, (D,)), reduction='none').mean(dim=-1)
            recovery_loss = recovery_error.mean()

            if self.memory_impl == 'hacssmv7_actiononly':
                action_windows = actions.unfold(1, horizon, 1)
                action_windows = action_windows.permute(0, 1, 3, 2)
                selected_actions = action_windows[batch_index, source_index]
                prediction = self.mem_hacssmv7.action_rollout(
                    source, selected_actions)[:, -1, level]
                loss_h = F.smooth_l1_loss(
                    F.layer_norm(prediction, (D,)),
                    F.layer_norm(bridge_target, (D,)), reduction='none').mean(dim=-1).mean()
                used_recovery = recovery_loss.detach() * 0.0
            elif self.memory_impl == 'hacssmv7_norecovery':
                loss_h = bridge_loss
                used_recovery = recovery_loss.detach() * 0.0
            else:
                loss_h = 0.5 * (bridge_loss + recovery_loss)
                used_recovery = recovery_loss

            level_terms[level].append(loss_h)
            horizon_terms.setdefault(horizon, []).append(loss_h)
            horizon_pairs.setdefault(horizon, []).append(pair_count)
            # Keep the aggregate diagnostic aligned with the loss actually optimized by each
            # control.  In ``actiononly`` there is no counterfactual correction bridge; its
            # action-rollout loss occupies the bridge slot so W&B cannot mislabel an unused
            # counterfactual quantity as active supervision.
            bridge_terms.append(
                loss_h if self.memory_impl == 'hacssmv7_actiononly' else bridge_loss)
            recovery_terms.append(used_recovery)
            result[f'hier_loss_{level_names[level]}_h{horizon}'] = loss_h
            result[f'hier_pairs_{level_names[level]}_h{horizon}'] = pair_count

            # Eligibility requires every synthetically hidden token to have been visible.
            for step in range(1, horizon + 1):
                overlap_count = overlap_count + hidden[
                    batch_index, source_index + step].sum().to(dtype=states.dtype)

        if float(overlap_count.detach()) != 0.0:
            raise RuntimeError('V7 counterfactual windows overlap original hidden targets')
        fast = torch.stack(level_terms[0]).mean()
        medium = torch.stack(level_terms[1]).mean()
        result.update({
            'hier_loss': torch.stack((fast, medium)).mean(),
            'hier_loss_fast': fast,
            'hier_loss_medium': medium,
            'hier_loss_bridge': torch.stack(bridge_terms).mean(),
            'hier_loss_recovery': torch.stack(recovery_terms).mean(),
            'hier_pairs': total_pairs,
            'hier_overlap': overlap_count,
        })
        for horizon, terms in horizon_terms.items():
            result[f'hier_loss_h{horizon}'] = torch.stack(terms).mean()
            result[f'hier_pairs_h{horizon}'] = torch.stack(horizon_pairs[horizon]).sum()
        return result

    # ---- core training loss (sliding short-window over a long chunk) ---------------
    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        target_observations: torch.Tensor = None,
        target_valid_mask: torch.Tensor = None,
        memory_update_mask: torch.Tensor = None,
        gate_override=None,
        first_post_loss_weight: float = 0.0,
        target_embeddings: torch.Tensor = None,
        diversity_embeddings: torch.Tensor = None,
        objective: str = 'lewm',
        detach_target_embeddings: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Prediction loss over a length-L chunk with an optional diversity objective.

        Args:
            observations: (B, L, C, H, W) -- a contiguous chunk (L can be >> history_len).
            actions: (B, L-1, A) -- action a_t maps obs_t -> obs_{t+1}.
            target_observations: optional synchronized clean frames. When provided, memory is
                driven by ``observations`` while prediction targets use these frames.
            target_valid_mask: optional (B,L) bool mask; false target frames do not contribute
                to prediction loss (used to keep hidden clean blackout frames evaluation-only).
            memory_update_mask: optional (B,L) visibility/update mask for an oracle memory. It is
                never inferred from ``target_valid_mask``; the oracle fails if it is omitted.
            gate_override: optional scalar/tensor SMT-v3 gate override for counterfactual analysis.
            first_post_loss_weight: convex weight on the first valid target immediately after a
                masked interval. Zero preserves the legacy all-valid prediction objective.
            target_embeddings: optional externally encoded synchronized targets. By default they
                are detached, preserving historical stop-gradient behavior for external targets.
            diversity_embeddings: online clean embeddings used by the V10-R/J variance and
                covariance terms.
            objective: ``lewm`` for prediction plus active SIGReg, or ``v10r``/``v10j`` for
                equal-weight prediction, variance, and off-diagonal covariance.
            detach_target_embeddings: whether external target embeddings are stop-gradient.
                V10-J explicitly sets this false so the clean target path remains end-to-end.
        Returns:
            dict with 'loss', 'pred_loss', 'sigreg_loss'.
        """
        B, L = observations.shape[0], observations.shape[1]
        h = self.history_len
        D, A = self.embed_dim, self.action_dim
        assert L >= h + 1, f"chunk length L={L} must be >= history_len+1={h + 1}"

        # Encode all frames (memoryless, per-frame).
        z = self.encode(observations)                      # (B, L, D)
        if target_observations is not None and target_embeddings is not None:
            raise ValueError('provide target observations or target embeddings, not both')
        if target_embeddings is not None:
            if target_embeddings.shape != (B, L, D):
                raise ValueError(
                    f'target_embeddings shape {tuple(target_embeddings.shape)} != {(B, L, D)}')
            z_target = (
                target_embeddings.detach()
                if detach_target_embeddings else target_embeddings)
        else:
            z_target = z if target_observations is None else self.encode(target_observations)

        # Memory over the full causal history (ema / multi / gru), injected into the predictor input.
        memory_details = None
        if (self.memory_impl in _HACSSMV8_MODES or self.memory_impl in _LOIFV9_MODES
                or self.memory_impl in _ORBITV10_MODES
                or self.memory_impl in _KDIOV11_MODES):
            # V8--V11 are action-conditioned inference memories with no teacher or internal-state
            # auxiliary in the shared LeWM loss.  Do not request details here: the generic tail
            # interprets non-None details as a request to add an auxiliary.
            z_tilde = self._inject(
                z, actions=actions, memory_update_mask=memory_update_mask,
                gate_override=gate_override)
        elif (self.memory_impl in _HACSMV4_MODES or self.memory_impl in _HACSSMV5_MODES
                or self.memory_impl in _HACSSMV6_MODES
                or self.memory_impl in _HACSSMV7_MODES):
            z_tilde, memory_details = self._inject(
                z, actions=actions, memory_update_mask=memory_update_mask,
                gate_override=gate_override, return_memory_details=True)
        else:
            z_tilde = self._inject(
                z, memory_update_mask=memory_update_mask,
                gate_override=gate_override)  # (B,L,D)

        # Sliding windows of length h: window s = z~[s : s+h] predicts z_{s+h}.
        # Number of windows W = L - h (s = 0 .. L-h-1).
        W = L - h
        zt_win = z_tilde.unfold(1, h, 1)[:, :W]            # (B, W, D, h)
        zt_win = zt_win.permute(0, 1, 3, 2).reshape(B * W, h, D)
        act_win = actions.unfold(1, h, 1)[:, :W]           # (B, W, A, h)
        act_win = act_win.permute(0, 1, 3, 2).reshape(B * W, h, A)
        targets = z_target[:, h:L].reshape(B * W, D)       # clean/shared z_{s+h}, when requested

        z_pred = self.predictor(zt_win, act_win)           # (B*W, h, D)
        z_pred_last = z_pred[:, -1, :]                     # predict next from last token

        if not 0.0 <= float(first_post_loss_weight) <= 1.0:
            raise ValueError('first_post_loss_weight must lie in [0,1]')
        per_window_error = (z_pred_last - targets).square().mean(dim=-1).reshape(B, W)
        pred_loss_first_post = None
        if target_valid_mask is None:
            if first_post_loss_weight:
                raise ValueError(
                    'first_post_loss_weight requires target_valid_mask to identify reappearance')
            pred_loss_all_valid = per_window_error.mean()
            pred_loss = pred_loss_all_valid
        else:
            if target_valid_mask.shape != (B, L):
                raise ValueError(
                    f'target_valid_mask shape {tuple(target_valid_mask.shape)} != {(B, L)}')
            mask = target_valid_mask.to(z_pred_last.device, dtype=torch.bool)
            valid = mask[:, h:L]
            if not valid.any():
                raise ValueError('target_valid_mask excludes every prediction target')
            pred_loss_all_valid = per_window_error[valid].mean()
            # A target at time t is first-post exactly when t is valid and t-1 was masked.
            first_post = mask[:, h:L] & ~mask[:, h - 1:L - 1]
            counts = first_post.sum(dim=1)
            if bool((counts != 1).any()):
                raise ValueError(
                    'target_valid_mask must contain exactly one masked-to-valid transition per '
                    f'episode for first-post accounting; got counts={counts.tolist()}')
            pred_loss_first_post = per_window_error[first_post].mean()
            pred_loss = ((1.0 - first_post_loss_weight) * pred_loss_all_valid +
                         first_post_loss_weight * pred_loss_first_post)
        if objective not in {'lewm', 'v10r', 'v10j'}:
            raise ValueError(f'unknown training objective {objective!r}')
        if objective in {'v10r', 'v10j'}:
            if objective == 'v10j' and detach_target_embeddings:
                raise ValueError('v10j requires active target-embedding gradients')
            if diversity_embeddings is None or diversity_embeddings.shape != (B, L, D):
                shape = None if diversity_embeddings is None else tuple(diversity_embeddings.shape)
                raise ValueError(
                    f'{objective} diversity_embeddings shape {shape} != {(B, L, D)}')
            if target_valid_mask is None:
                diversity_features = diversity_embeddings.reshape(B * L, D)
            else:
                diversity_features = diversity_embeddings[
                    target_valid_mask.to(diversity_embeddings.device, dtype=torch.bool)]
            if len(diversity_features) < 2:
                raise ValueError(
                    f'{objective} diversity requires at least two clean online embeddings')
            # Diversity statistics stay in FP32 under AMP; the cast remains differentiable.
            with torch.autocast(device_type=diversity_features.device.type, enabled=False):
                stats_features = diversity_features.float()
                centered = stats_features - stats_features.mean(dim=0, keepdim=True)
                variance = centered.square().sum(dim=0) / (len(centered) - 1)
                # The unit-variance target is fixed by affine-free LayerNorm. Averaging over D
                # makes this term independent of representation width.
                variance_loss = torch.relu(1.0 - torch.sqrt(
                    variance + torch.finfo(variance.dtype).eps)).mean()
                covariance = centered.T @ centered / (len(centered) - 1)
                off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
                covariance_loss = (
                    # Standard VICReg dimension normalization: a unit-variance rank-one
                    # representation costs D-1, while an identity covariance costs zero.
                    off_diagonal.square().sum() / D
                    if D > 1 else covariance.new_zeros(()))
            # SIGReg is retained only as a collapse diagnostic. It has no optimization
            # gradient in V10-R/J; all three optimized terms have exactly unit weight.
            with torch.no_grad():
                sigreg_loss = self.sigreg(diversity_features.detach())
            total = pred_loss + variance_loss + covariance_loss
            out = {
                'loss': total,
                'pred_loss': pred_loss,
                'variance_loss': variance_loss,
                'covariance_loss': covariance_loss,
                'sigreg_loss': sigreg_loss,
                'pred_loss_all_valid': pred_loss_all_valid,
            }
        else:
            if target_valid_mask is None:
                sigreg_features = z_target.reshape(B * L, D)
            else:
                sigreg_features = z_target[
                    target_valid_mask.to(z_target.device, dtype=torch.bool)]
            sigreg_loss = self.sigreg(sigreg_features)
            total = pred_loss + self.sigreg_lambda * sigreg_loss
            out = {'loss': total, 'pred_loss': pred_loss, 'sigreg_loss': sigreg_loss,
                   'pred_loss_all_valid': pred_loss_all_valid}
        if pred_loss_first_post is not None:
            out['pred_loss_first_post'] = pred_loss_first_post
        if self.memory_impl == 'ocsmt' and self.l0_lambda > 0 and self.mem_ocsmt.last_l0 is not None:
            l0 = self.mem_ocsmt.last_l0                     # expected #open gates (set in _inject)
            out['l0_loss'] = l0
            out['loss'] = total + self.l0_lambda * l0
        if memory_details is not None:
            if self.memory_impl in _HACSSMV7_MODES:
                auxiliary = self._hierarchical_counterfactual_recovery_loss(
                    memory_details['states'], z, actions, target_valid_mask)
            elif self.memory_impl in _HACSSMV6_MODES:
                auxiliary = self._hierarchical_consistency_loss(
                    memory_details['states'], actions, target_valid_mask)
            elif self.memory_impl in _HACSSMV5_MODES:
                auxiliary = self._hierarchical_boundary_loss(
                    memory_details['states'], actions, z_target, target_valid_mask)
            else:
                auxiliary = self._hierarchical_auxiliary_loss(
                    memory_details['states'], actions, z_target, target_valid_mask)
            out.update(auxiliary)
            effective_weight = (0.0 if self.memory_impl in {
                                    'hacsmv4_noaux', 'hacsmv4_two_noaux',
                                    'hacssmv5_noaux', 'hacssmv5_fixedbeta_noaux',
                                    'hacssmv5_ssmcontrol', 'hacssmv6_noaux',
                                    'hacssmv7_noaux'}
                                else self.hier_loss_weight)
            out['hier_loss_weight'] = auxiliary['hier_loss'].new_tensor(effective_weight)
            out['loss'] = out['loss'] + effective_weight * auxiliary['hier_loss']
        return out

    def forward(self, observations, actions):
        return self.compute_loss(observations, actions)

    # ---- analysis utilities (used by probing / visualization) ----------------------
    @torch.no_grad()
    def encode_with_memory(self, observations: torch.Tensor, actions: torch.Tensor = None,
                           memory_update_mask: torch.Tensor = None, gate_override=None,
                           action_override=None, resistance_override=None):
        """Return (z, m_fast, m_slow, z_tilde). m_fast/m_slow are None for non-EMA impls."""
        z = self.encode(observations)
        if self.memory_impl == 'ema':
            m_fast, m_slow = self.memory(z)
            return z, m_fast, m_slow, self.fusion(z, m_fast, m_slow)
        return z, None, None, self._inject(
            z, actions=actions, memory_update_mask=memory_update_mask,
            gate_override=gate_override, action_override=action_override,
            resistance_override=resistance_override)

    @torch.no_grad()
    def memory_influence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        memory_update_mask: torch.Tensor = None,
        gate_override=None,
        action_override=None,
        resistance_override=None,
    ) -> Dict[str, torch.Tensor]:
        """Causal influence of each memory bank on the predicted next latent.

        We predict z_{L-1} (the last observed frame) from the window ending at L-2 and
        measure how much that prediction *moves* when a bank is ablated:

            infl_fast = || f(full) - f(ablate fast) ||_2 ,   similarly for slow.

        EMA and hierarchical HACSM/HACSSM implementations expose distinct level ablations.
        Every implementation also exposes ``infl_all`` for the complete memory-path ablation.
        A single-state non-EMA memory reports only ``infl_all`` rather than duplicating one
        quantity under misleading fast/slow names.

        Args:
            observations: (B, L, C, H, W), L >= history_len + 1.
            actions: (B, L-1, A).
        Returns:
            dict: ``pred_full`` and ``infl_all``; additionally ``infl_fast`` and
            ``infl_slow`` when distinct corresponding recurrent states exist.
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
                    'infl_slow': (full - pred(False, True)).norm(dim=-1),
                    'infl_all': (full - pred(True, True)).norm(dim=-1)}

        hierarchical = (
            self.memory_impl in _HACSMV4_MODES
            or self.memory_impl in _HACSSMV5_MODES
            or self.memory_impl in _HACSSMV6_MODES
            or self.memory_impl in _HACSSMV7_MODES
            or self.memory_impl in _HACSSMV8_MODES
            or self.memory_impl in _LOIFV9_MODES
        )
        if hierarchical:
            z_full, details = self._inject(
                z, actions=actions, memory_update_mask=memory_update_mask,
                gate_override=gate_override, action_override=action_override,
                resistance_override=resistance_override,
                return_memory_details=True)
            if self.memory_impl in _HACSMV4_MODES:
                memory = self.mem_hacsmv4
            elif self.memory_impl in _HACSSMV5_MODES:
                memory = self.mem_hacssmv5
            elif self.memory_impl in _HACSSMV6_MODES:
                memory = self.mem_hacssmv6
            elif self.memory_impl in _HACSSMV7_MODES:
                memory = self.mem_hacssmv7
            elif self.memory_impl in _HACSSMV8_MODES:
                memory = self.mem_hacssmv8
            else:
                memory = self.mem_loifv9
            states = details['states']
            if self.memory_impl in _LOIFV9_MODES:
                dynamic_weights = details['read_weights'].to(
                    device=states.device, dtype=states.dtype)

                def ablated(level: int) -> torch.Tensor:
                    weights = dynamic_weights.clone()
                    weights[:, :, level] = 0.0
                    mixed = (states * weights.unsqueeze(-1)).sum(dim=2)
                    return memory.fuse(z, memory._rms_norm(mixed))
            else:
                route = details['route'].to(device=states.device, dtype=states.dtype)

                def ablated(level: int) -> torch.Tensor:
                    weights = route.clone()
                    weights[level] = 0.0
                    mixed = (states * weights.view(1, 1, memory.K, 1)).sum(dim=2)
                    mixed = mixed * torch.rsqrt(
                        mixed.square().mean(dim=-1, keepdim=True) + memory.rms_eps)
                    return memory.fuse(z, mixed)

            full = self.predictor(z_full[:, wsl], act)[:, -1, :]
            fast = self.predictor(ablated(0)[:, wsl], act)[:, -1, :]
            slow = self.predictor(ablated(memory.K - 1)[:, wsl], act)[:, -1, :]
            nomem = self.predictor(z[:, wsl], act)[:, -1, :]
            return {
                'pred_full': full,
                'infl_fast': (full - fast).norm(dim=-1),
                'infl_slow': (full - slow).norm(dim=-1),
                'infl_all': (full - nomem).norm(dim=-1),
            }

        # Other non-EMA memories have one undifferentiated or non-hierarchical memory path.
        full = self.predictor(self._inject(
            z, actions=actions, memory_update_mask=memory_update_mask,
            gate_override=gate_override, action_override=action_override,
            resistance_override=resistance_override)[:, wsl], act)[:, -1, :]
        nomem = self.predictor(z[:, wsl], act)[:, -1, :]
        infl_all = (full - nomem).norm(dim=-1)
        return {'pred_full': full, 'infl_all': infl_all}

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
