#!/usr/bin/env python3
"""Generate the HACSM-v4 architecture figure used by LEARNABLE_MEMORY.md."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'docs' / 'figures' / 'fig_hacsm_v4_arch.png'


def box(ax, xy, width, height, text, color, *, size=10, linewidth=1.6):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle='round,pad=0.025,rounding_size=0.035',
        facecolor=color, edgecolor='#263238', linewidth=linewidth, zorder=2)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha='center', va='center',
            fontsize=size, color='#17202a', zorder=3)
    return patch


def arrow(ax, start, end, *, color='#455a64', style='-|>', width=1.5,
          connection='arc3', dashed=False):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=13, linewidth=width,
        color=color, connectionstyle=connection, linestyle='--' if dashed else '-', zorder=1)
    ax.add_patch(patch)
    return patch


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('#fbfcfd')
    ax.set_facecolor('#fbfcfd')

    ax.text(8, 8.62, 'HACSM-v4: Hierarchical Action-Conditioned Selective Memory',
            ha='center', va='center', fontsize=18, fontweight='bold', color='#17202a')
    ax.text(8, 8.25,
            'strictly causal predict → correct belief update; fixed structural horizons; '
            'observation-free self-supervision',
            ha='center', va='center', fontsize=10.5, color='#546e7a')

    box(ax, (0.35, 5.65), 1.25, 0.72, r'observation $z_t$', '#e3f2fd', size=11)
    box(ax, (0.35, 3.15), 1.25, 0.72, r'action $a_{t-1}$', '#fff3e0', size=11)
    box(ax, (1.95, 5.65), 1.25, 0.72, r'$x_t=W_xz_t$', '#e8f5e9', size=11)
    box(ax, (1.95, 3.15), 1.45, 0.72, r'$(d,v)=W_aa_{t-1}$', '#fff8e1', size=10.5)
    arrow(ax, (1.6, 6.01), (1.95, 6.01))
    arrow(ax, (1.6, 3.51), (1.95, 3.51))

    colors = ('#dcedc8', '#b2dfdb', '#d1c4e9')
    levels = (('fast', 2, 6.6), ('medium', 8, 4.85), ('slow', 32, 3.10))
    for index, (name, tau, y) in enumerate(levels):
        box(ax, (3.95, y), 2.45, 0.95,
            rf'{name} prior  $p_t^{index}$' + '\n'
            + rf'$m_{{t-1}}^{index}+\beta_{index}\tanh(v+d\odot LN(m_{{t-1}}^{index}))$',
            colors[index], size=9.2)
        box(ax, (7.05, y), 2.15, 0.95,
            rf'correction gate  $g_t^{index}$' + '\n'
            + r'$\sigma((w_z^\top LN(z_t)+w_e^\top LN(x_t-p_t))$' + '\n'
            + r'$/\sqrt{D}+b_' + str(index) + r')$',
            '#ffecb3', size=8.0)
        box(ax, (9.85, y), 2.25, 0.95,
            rf'posterior  $m_t^{index}$  ($\tau={tau}$)' + '\n'
            + rf'$p_t^{index}+\beta_{index}g_t^{index}(x_t-p_t^{index})$',
            '#e1bee7' if index == 2 else '#c5cae9', size=9.2)
        arrow(ax, (3.4, 3.51), (3.95, y + 0.35), color='#ef6c00')
        arrow(ax, (6.4, y + 0.47), (7.05, y + 0.47))
        arrow(ax, (9.2, y + 0.47), (9.85, y + 0.47))
        arrow(ax, (3.2, 6.01), (7.05, y + 0.68), color='#1976d2')
        arrow(ax, (3.2, 6.01), (9.85, y + 0.22), color='#2e7d32', dashed=True)
        ax.text(4.05, y + 1.02, rf'$\beta_{index}=1-e^{{-1/{tau}}}$', fontsize=8.6,
                color='#455a64')

    box(ax, (12.7, 5.15), 1.45, 1.15,
        r'global read' + '\n' + r'$q_t=RMSNorm(\sum_k\pi_km_t^k)$', '#b3e5fc', size=10)
    box(ax, (14.55, 5.15), 1.15, 1.15,
        r'residual' + '\n' + r'$\tilde z_t=z_t+W_oq_t$', '#c8e6c9', size=10)
    for _, _, y in levels:
        arrow(ax, (12.1, y + 0.47), (12.7, 5.72), color='#5e35b1')
    arrow(ax, (14.15, 5.72), (14.55, 5.72))
    arrow(ax, (0.98, 6.37), (15.12, 6.30), color='#1976d2', dashed=True,
          connection='arc3,rad=-0.10')

    box(ax, (4.35, 0.65), 7.65, 1.35,
        'self-supervised action-only rollout (training only)\n'
        r'$r_{t,0}^k=m_t^k,\quad r_{t,j+1}^k=T_k(r_{t,j}^k,a_{t+j}),\quad '
        r'r_{t,h}^k\rightarrow stopgrad(z_{t+h}^{clean})$' + '\n'
        + r'fast $h=1,2$   •   medium $h=4,8$   •   slow $h=16$   •   hidden blackout targets excluded',
        '#f3e5f5', size=10.5, linewidth=1.9)
    # Every level receives its own horizon-matched auxiliary target.  Route the
    # three posterior states to distinct points on the training-only objective
    # so the diagram does not visually imply slow-only supervision.
    aux_targets = ((5.65, 'arc3,rad=-0.18'), (8.15, 'arc3,rad=-0.08'),
                   (10.90, 'arc3,rad=0.0'))
    for (_, _, y), (target_x, connection) in zip(levels, aux_targets):
        arrow(ax, (10.95, y), (target_x, 2.0), color='#7b1fa2', dashed=True,
              connection=connection)
    arrow(ax, (2.65, 3.15), (4.35, 1.32), color='#ef6c00', dashed=True)

    ax.text(0.4, 0.20,
            r'Initialization: $W_x=I$, $W_a=0$, $W_o=0$, $b_k=2$, $\pi_k=1/3$.  '
            r'Thus the predictor starts exactly memoryless while the auxiliary loss immediately '
            r'trains the belief dynamics.',
            fontsize=9.5, color='#455a64')
    ax.text(15.65, 0.58, 'aₜ maps zₜ → zₜ₊₁', ha='right', fontsize=9.5,
            fontweight='bold', color='#bf360c')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    main()
