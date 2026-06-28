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
    fig, ax = plt.subplots(figsize=(17, 16), dpi=180)
    fig.patch.set_facecolor('#fbfcfd')
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 16)
    ax.axis('off')

    ax.text(8.5, 15.58, 'Learnable-memory architecture map: V1–V5 and tested controls',
            ha='center', fontsize=18, fontweight='bold', color='#17202a')
    ax.text(8.5, 15.20,
            'Architecture-changing variants are shown explicitly; seeds, optimizer settings, '
            'mask shifts, and K/M/τ sweeps are experimental settings.',
            ha='center', fontsize=10.5, color='#607d8b')

    xs = (0.25, 3.60, 6.95, 10.30, 13.65)
    width, top_y, top_h = 3.10, 12.48, 2.20
    card(ax, xs[0], top_y, width, top_h, 'SMT-v1',
         r'value-gated EMA write' + '\n' + r'$i_t\odot z_t$; old state still decays' + '\n'
         + 'softmax horizon read', '#e3f2fd', body_size=7.8)
    card(ax, xs[1], top_y, width, top_h, 'SMT-v2',
         'same erasing write\nindependent sigmoid reads\n'
         'larger mass; gates become static', '#dcedc8', body_size=7.8)
    card(ax, xs[2], top_y, width, top_h, 'SMT-v3-W',
         'whole-update scalar gate\n$g_t=0$ exactly freezes all EMA banks\n'
         'global simplex; action-blind', '#fff3e0', body_size=7.8)
    card(ax, xs[3], top_y, width, top_h, 'HACSM-v4',
         'three belief levels $\\tau=\\{2,8,32\\}$\naction prior $p_t=T(m_{t-1},a_{t-1})$\n'
         'selective correction + fixed auxiliary', '#e1bee7', body_size=7.8)
    card(ax, xs[4], top_y, width, top_h, 'HACSSM-v5',
         'two fast/medium states\nhard-monotone channel gains\n'
         'action predict/correct\nfront-loaded boundary shaping', '#d1c4e9', body_size=7.8)
    arrow(ax, (3.35, 13.58), (3.60, 13.58), 'mass')
    arrow(ax, (6.70, 13.58), (6.95, 13.58), 'freeze')
    arrow(ax, (10.05, 13.58), (10.30, 13.58), 'actions')
    arrow(ax, (13.40, 13.58), (13.65, 13.58), 'learn rates')

    ax.text(0.35, 12.10, 'Pre-V4 architecture controls used in the mechanism studies',
            fontsize=11.5, fontweight='bold', color='#37474f')
    v3_controls = (
        ('scaled-softmax SMT', 'V1 softmax route × K/2\nmatched initial read mass\nisolates V1→V2 amplitude', '#e1f5fe'),
        ('V3 static', '$g_t=\\sigma(b)$\nremoves input conditioning\nmatched dynamic-gate parameters', '#ffe0b2'),
        ('V3 old-erasing', 'dynamic scalar gate on value only\nold state still decays\nisolates true-freeze semantics', '#ffecb3'),
        ('V3 hard visibility', 'known black/visible mask\nforces freeze/update timing\nnominal parameter control', '#f3e5f5'),
    )
    for index, (title, body, color) in enumerate(v3_controls):
        card(ax, 0.35 + index * 4.20, 10.35, 3.65, 1.40, title, body, color,
             title_size=10.2, body_size=7.7)

    ax.text(0.35, 9.98, 'V4 controls and the slow-removal bridge into V5',
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
        card(ax, 0.25 + index * 2.80, 8.12, 2.55, 1.50, title, body, color,
             title_size=9.5, body_size=7.0)

    ax.text(0.35, 7.74, 'V5 parameter-matched mechanism variants (34,820 memory parameters)',
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
        card(ax, 0.20 + index * 2.40, 5.91, 2.18, 1.48, title, body, color,
             title_size=8.7, body_size=6.5)

    ax.text(0.35, 5.53, 'External controls and historical memory families', fontsize=11.5,
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
        card(ax, 0.20 + index * 2.40, 3.70, 2.18, 1.42, title, body, '#eceff1',
             title_size=9.2, body_size=7.0, edge='#607d8b', linewidth=1.2)

    card(ax, 0.35, 2.08, 16.25, 0.92, 'Shared leakage-safe V5 comparison contract',
         'fixed DINOv2-PCA targets • $a_t:z_t\\rightarrow z_{t+1}$ • output norm = none '
         '(no cross-window statistics) • blackout targets excluded • '
         'first-post-balanced objective • final epoch, no best-checkpoint selection',
         '#e8eaf6', title_size=10.5, body_size=8.3, linewidth=1.4)

    ax.text(8.5, 1.49,
            'Prospective V5 pilot: 5 environments × 12 designs × 3 seeds = 180 runs; '
            'the immutable screen is followed by the requested 120-run five-seed completion (§7.7).',
            ha='center', fontsize=8.8, color='#607d8b')
    ax.text(8.5, 1.10,
            'Historical V1–V3 cohorts retain their documented protocols; architecture cards do '
            'not imply that their raw MSE values are pooled with V5.',
            ha='center', fontsize=8.8, color='#607d8b')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    main()
