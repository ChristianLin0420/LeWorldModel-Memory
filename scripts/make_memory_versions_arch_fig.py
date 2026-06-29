#!/usr/bin/env python3
"""Generate the consolidated architecture/ablation map for the learnable-memory study."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'docs' / 'figures' / 'fig_memory_versions_arch.png'


def card(ax, x, y, w, h, title, body, color, *, title_size=12, body_size=8.5,
         edge='#263238', linewidth=1.5):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.04',
        facecolor=color, edgecolor=edge, linewidth=linewidth)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.28, title, ha='center', va='center',
            fontsize=title_size, fontweight='bold', color='#17202a')
    ax.text(x + w / 2, y + h / 2 - 0.12, body, ha='center', va='center',
            fontsize=body_size, color='#263238', linespacing=1.35)


def arrow(ax, start, end, label=''):
    patch = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14,
                            color='#546e7a', linewidth=1.7)
    ax.add_patch(patch)
    if label:
        ax.text((start[0] + end[0]) / 2, start[1] + 0.18, label,
                ha='center', va='bottom', fontsize=8.2, color='#546e7a')


def main() -> None:
    fig, ax = plt.subplots(figsize=(18, 25.4), dpi=180)
    fig.patch.set_facecolor('#fbfcfd')
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 25.4)
    ax.axis('off')

    ax.text(9, 25.03, 'Learnable-memory architecture map: V1–V8 and tested controls',
            ha='center', fontsize=18, fontweight='bold', color='#17202a')
    ax.text(9, 24.65,
            'Architecture-changing variants are shown explicitly; seeds, optimizer settings, '
            'mask shifts, and K/M/τ sweeps are experimental settings.',
            ha='center', fontsize=10.5, color='#607d8b')

    xs = tuple(0.10 + index * 2.23 for index in range(8))
    width, top_y, top_h = 2.00, 21.98, 2.05
    card(ax, xs[0], top_y, width, top_h, 'SMT-v1',
         r'value-gated EMA write' + '\n' + r'$i_t\odot z_t$; old state still decays' + '\n'
         + 'softmax horizon read', '#e3f2fd', body_size=6.1)
    card(ax, xs[1], top_y, width, top_h, 'SMT-v2',
         'same erasing write\nindependent sigmoid reads\n'
         'larger mass; gates become static', '#dcedc8', body_size=6.1)
    card(ax, xs[2], top_y, width, top_h, 'SMT-v3-W',
         'whole-update scalar gate\n$g_t=0$ exactly freezes all EMA banks\n'
         'global simplex; action-blind', '#fff3e0', body_size=6.1)
    card(ax, xs[3], top_y, width, top_h, 'HACSM-v4',
         'three belief levels $\\tau=\\{2,8,32\\}$\naction prior $p_t=T(m_{t-1},a_{t-1})$\n'
         'selective correction + fixed auxiliary', '#e1bee7', body_size=5.8)
    card(ax, xs[4], top_y, width, top_h, 'HACSSM-v5',
         'two fast/medium states\nhard-monotone channel gains\n'
         'action predict/correct\nboundary-only shaping', '#d1c4e9', body_size=5.8)
    card(ax, xs[5], top_y, width, top_h, 'HACSSM-v6',
         'fixed scalar $\\tau=\\{2,8\\}$ anchor\ndense visible-endpoint\n'
         'same-level action consistency', '#c5cae9', body_size=5.9)
    card(ax, xs[6], top_y, width, top_h, 'HACSSM-v7',
         'level-specific action heads\nstatic/dynamic gate shrinkage\n'
         'EMA counterfactual recovery', '#b39ddb', body_size=5.8)
    card(ax, xs[7], top_y, width, top_h, 'SAS-PC-v8',
         'one physical shared action head\nlearned gate shrinkage\n'
         'joint read; no internal auxiliary', '#9575cd', body_size=5.8)
    for index in range(7):
        arrow(ax, (xs[index] + width, 23.00), (xs[index + 1], 23.00))

    ax.text(0.35, 21.58,
            'V8 shared-action shrinkage controls (34,566 compact / 36,102 expanded parameters)',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v8_controls = (
        ('compact', 'physical shared head\nlearned shrinkage\njoint read', '#9575cd'),
        ('dynamic', '$\\rho_f=\\rho_m=1$\nretrained dynamic endpoint', '#f8bbd0'),
        ('static', '$\\rho_f=\\rho_m=0$\nretrained static endpoint', '#fff3e0'),
        ('level-action', 'separate fast/medium heads\naction-tying control', '#d1c4e9'),
        ('redundant', 'equal head halves; averaged\nstatistical equivalence', '#ffecb3'),
        ('no-action', 'shared action contribution 0\nstructural receipt', '#ffe0b2'),
        ('single', 'medium-only read\nboth states retained', '#b2ebf2'),
    )
    for index, (title, body, color) in enumerate(v8_controls):
        card(ax, 0.15 + index * 2.55, 19.90, 2.30, 1.34, title, body, color,
             title_size=8.5, body_size=6.2)

    ax.text(0.35, 19.38, 'Pre-V4 architecture controls used in the mechanism studies',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v3_controls = (
        ('scaled-softmax SMT', 'V1 softmax route × K/2\nmatched initial read mass\nisolates V1→V2 amplitude', '#e1f5fe'),
        ('V3 static', '$g_t=\\sigma(b)$\nremoves input conditioning\nmatched dynamic-gate parameters', '#ffe0b2'),
        ('V3 old-erasing', 'dynamic scalar gate on value only\nold state still decays\nisolates true-freeze semantics', '#ffecb3'),
        ('V3 hard visibility', 'known black/visible mask\nforces freeze/update timing\nnominal parameter control', '#f3e5f5'),
    )
    for index, (title, body, color) in enumerate(v3_controls):
        card(ax, 0.35 + index * 4.42, 17.60, 4.02, 1.42, title, body, color,
             title_size=10.2, body_size=7.6)

    ax.text(0.35, 17.22, 'V4 controls and the slow-removal bridge into V5/V6',
            fontsize=11.5, fontweight='bold', color='#37474f')
    controls = (
        ('full', 'dynamic correction\n+ action prior\n+ hierarchical auxiliary', '#d1c4e9'),
        ('static', '$g_t^k=\\sigma(b_k)$\nremoves input-conditioned\ncorrection timing', '#ffe0b2'),
        ('no-action', '$W_a$ instantiated\naction contribution forced 0\nisolates action transition', '#ffecb3'),
        ('no-aux', 'full online architecture\nauxiliary weight forced 0\nisolates self-supervision', '#c8e6c9'),
        ('single-level', 'dynamic/action states retained\nread fixed to middle $\\tau=8$\nisolates hierarchical read', '#b2ebf2'),
        ('two-level no-aux', '$\\tau=\\{2,8\\}$ only\nfixed scalar gains\nisolates slow removal', '#e0f2f1'),
    )
    for index, (title, body, color) in enumerate(controls):
        card(ax, 0.20 + index * 2.97, 15.42, 2.70, 1.50, title, body, color,
             title_size=9.3, body_size=6.8)

    ax.text(0.35, 15.03, 'V5 parameter-matched mechanism variants (34,820 memory parameters)',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v5_controls = (
        ('full', 'learned gains + action\ndynamic correction\nboundary shaping', '#d1c4e9'),
        ('no-aux', 'same online memory\nweight forced 0\nisolates objective', '#c8e6c9'),
        ('fixed-β no-aux', 'initial channel bands frozen\nno boundary shaping\nisolates gain learning', '#e3f2fd'),
        ('no-action', 'learned gains retained\naction contribution 0\naux schedule retained', '#ffecb3'),
        ('static', 'input-independent gates\naction + learned gains\naux schedule retained', '#ffe0b2'),
        ('single-medium', 'both states retained\nread fixed to medium\nisolates joint read', '#b2ebf2'),
        ('band-SSM', '$g=1$, action 0\nlearned two-state spectrum\nstate-matched control', '#eceff1'),
    )
    for index, (title, body, color) in enumerate(v5_controls):
        card(ax, 0.15 + index * 2.55, 13.18, 2.30, 1.48, title, body, color,
             title_size=8.4, body_size=6.2)

    ax.text(0.35, 12.78,
            'V6 fixed-rate inference anchor + complete training-objective/mechanism map '
            '(34,564 memory parameters)',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v6_controls = (
        ('full', 'detached source/target\nfast h=1,2; medium h=4,8', '#d1c4e9'),
        ('no-aux', 'same inference\nauxiliary weight = 0', '#c8e6c9'),
        ('aux-no-action', 'zero action only\nin auxiliary rollout', '#ffecb3'),
        ('uniform', 'both levels use\nh={1,2,4,8}', '#e1f5fe'),
        ('source-grad', 'source not detached\ntarget remains detached', '#f8bbd0'),
        ('fast-only', 'only fast auxiliary\nh={1,2}', '#dcedc8'),
        ('medium-only', 'only medium auxiliary\nh={4,8}', '#b2dfdb'),
        ('no-action', 'actions zeroed in\ninference + auxiliary', '#ffe0b2'),
        ('static', 'input-independent\ncorrection gates', '#fff3e0'),
        ('single', 'medium-only read\nboth states/aux retained', '#b2ebf2'),
    )
    for index, (title, body, color) in enumerate(v6_controls):
        row, col = divmod(index, 5)
        card(ax, 0.20 + col * 3.55, 11.00 - row * 1.72, 3.22, 1.42,
             title, body, color, title_size=9.0, body_size=6.7)

    ax.text(0.35, 8.88,
            'V7 counterfactual-recovery inference/objective map '
            '(36,102 trainable memory parameters)',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v7_controls = (
        ('full', 'level-specific action\nlearned gate shrinkage\nbridge + recovery', '#b39ddb'),
        ('no-aux', 'same online inference\nauxiliary weight = 0', '#c8e6c9'),
        ('shared-action', 'average both projected\naction heads per step', '#ffecb3'),
        ('no-shrink', '$\\rho_f=\\rho_m=1$\ndynamic correction only', '#f8bbd0'),
        ('action-only', 'V6-style action rollout\nno recovery correction', '#ffe0b2'),
        ('uniform', 'both levels use\nh={1,2,4,8}', '#e1f5fe'),
        ('no-recovery', 'counterfactual bridge only\nremove restored-frame loss', '#f3e5f5'),
        ('no-action', 'actions zeroed in\ninference + auxiliary', '#fff3e0'),
        ('single', 'medium-only read\nboth states/objectives retained', '#b2ebf2'),
    )
    for index, (title, body, color) in enumerate(v7_controls):
        row, col = divmod(index, 5)
        card(ax, 0.20 + col * 3.55, 7.10 - row * 1.72, 3.22, 1.42,
             title, body, color, title_size=9.0, body_size=6.7)

    ax.text(0.35, 4.98, 'External controls and historical memory families', fontsize=11.5,
            fontweight='bold', color='#37474f')
    baselines = (
        ('none', 'short predictor only\nno long-range channel'),
        ('fixed EMA', 'short / long / both\nhand-set recurrence'),
        ('fixed multi', '$K=6$ EMA banks\nseparate readouts'),
        ('GRU', 'learned recurrent state\nzero-init residual read'),
        ('diagonal SSM', 'learned channel decay\none $D$-state'),
        ('retrieval', 'causal soft attention\nover past latents'),
        ('OC-SMT', '$M=28$ EMA basis\nhard-concrete reads'),
    )
    for index, (title, body) in enumerate(baselines):
        card(ax, 0.15 + index * 2.55, 3.22, 2.30, 1.42, title, body, '#eceff1',
             title_size=9.0, body_size=6.8, edge='#607d8b', linewidth=1.2)

    card(ax, 0.35, 1.53, 17.30, 1.12, 'Shared leakage-safe V5/V6/V7/V8 comparison contract',
         'fixed DINOv2-PCA targets • $a_t:z_t\\rightarrow z_{t+1}$ • output norm = none '
         '(no cross-window statistics) • blackout targets excluded • '
         'first-post-balanced objective • final epoch, no best-checkpoint selection\n'
         'online W&B for every cell: 200 epoch logs + fixed evaluation-rollout '
         'trace / paired video / hashed artifact',
         '#e8eaf6', title_size=10.5, body_size=7.6, linewidth=1.4)

    ax.text(9, 1.05,
            'V5 complete: 300 runs.  V6 complete: 325 runs.  '
            'V7 complete: 325 runs.  V8 frozen: 325-run adaptive-development grid (§7.10).',
            ha='center', fontsize=8.8, color='#607d8b')
    ax.text(9, 0.66,
            'Historical V1–V4 cohorts retain their documented protocols; architecture cards do '
            'not imply that raw MSE values are pooled across incompatible target spaces.',
            ha='center', fontsize=8.8, color='#607d8b')
    ax.text(9, 0.27,
            'V8 removes the V7 teacher/objective: compact shared modes use 34,566 parameters; '
            'ordinary next-latent learning remains self-supervised.',
            ha='center', fontsize=8.8, fontweight='bold', color='#455a64')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    main()
