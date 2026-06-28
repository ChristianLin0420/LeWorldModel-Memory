#!/usr/bin/env python3
"""Generate the frozen HACSSM-v7 architecture and complete variant map."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'docs' / 'figures' / 'fig_hacssm_v7_arch.png'


def box(ax, xy, width, height, text, color, *, size=8.6, linewidth=1.5,
        edge='#263238'):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle='round,pad=0.025,rounding_size=0.035',
        facecolor=color, edgecolor=edge, linewidth=linewidth, zorder=2)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha='center', va='center', fontsize=size, color='#17202a',
            zorder=3, linespacing=1.24)


def arrow(ax, start, end, *, color='#455a64', dashed=False, connection='arc3'):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=13, linewidth=1.4,
        color=color, linestyle='--' if dashed else '-',
        connectionstyle=connection, zorder=1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(18, 13.5), dpi=180)
    fig.patch.set_facecolor('#fbfcfd')
    ax.set_facecolor('#fbfcfd')
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 13.5)
    ax.axis('off')

    ax.text(9, 13.13,
            'HACSSM-v7 / HCRD: Hierarchical Counterfactual Recovery Distillation',
            ha='center', fontsize=18.2, fontweight='bold', color='#17202a')
    ax.text(9, 12.78,
            'Frozen before launch; completed 325-run study returned '
            'PILOT_NO_GO_FINAL_DESCRIPTIVE (§7.9)',
            ha='center', fontsize=10.1, color='#546e7a')

    # Online student inference.
    ax.text(0.35, 12.28, 'ONLINE STUDENT INFERENCE', fontsize=11.2,
            fontweight='bold', color='#37474f')
    box(ax, (0.35, 10.36), 1.22, 0.68, r'$z_t$', '#e3f2fd', size=11)
    box(ax, (0.35, 9.15), 1.22, 0.68, r'$a_{t-1}$', '#fff3e0', size=10.6)
    box(ax, (1.88, 10.36), 1.48, 0.68, r'$x_t=W_xz_t$', '#e8f5e9', size=9.5)
    arrow(ax, (1.57, 10.70), (1.88, 10.70), color='#1976d2')

    levels = (
        ('FAST', 'f', r'$\tau_f=2$', 10.76, '#dcedc8', '#c5cae9'),
        ('MEDIUM', 'm', r'$\tau_m=8$', 9.15, '#b2dfdb', '#d1c4e9'),
    )
    for name, symbol, tau, y, prior_color, posterior_color in levels:
        box(ax, (3.78, y), 2.43, 0.98,
            f'{name} action head  {tau}\n'
            + rf'$(d^{symbol},v^{symbol})=W_a^{symbol}a_{{t-1}}$' + '\n'
            + rf'$p_t^{symbol}=m_{{t-1}}^{symbol}+\beta_{symbol}\tanh($' + '\n'
            + rf'$v^{symbol}+d^{symbol}\odot LN(m_{{t-1}}^{symbol}))$',
            prior_color, size=6.8)
        box(ax, (6.63, y), 2.20, 0.98,
            rf'static  $s_{symbol}=\sigma(b_{symbol})$' + '\n'
            + rf'dynamic  $d_t^{symbol}=\sigma($' + '\n'
            + rf'$b_{symbol}+innovation_t^{symbol})$',
            '#ffecb3', size=7.8)
        box(ax, (9.24, y), 1.85, 0.98,
            rf'$\rho_{symbol}=\sigma(c_{symbol})$' + '\n'
            + rf'$g_t^{symbol}=(1-\rho_{symbol})s_{symbol}+\rho_{symbol}d_t^{symbol}$',
            '#f8bbd0', size=8.0)
        box(ax, (11.52, y), 2.44, 0.98,
            rf'posterior  $m_t^{symbol}$' + '\n'
            + rf'$p_t^{symbol}+\beta_{symbol}g_t^{symbol}$' + '\n'
            + rf'$(x_t-p_t^{symbol})$',
            posterior_color, size=8.0)
        arrow(ax, (1.57, 9.49), (3.78, y + 0.48), color='#ef6c00')
        arrow(ax, (3.36, 10.70), (6.63, y + 0.66), color='#1976d2')
        arrow(ax, (6.21, y + 0.49), (6.63, y + 0.49))
        arrow(ax, (8.83, y + 0.49), (9.24, y + 0.49))
        arrow(ax, (11.09, y + 0.49), (11.52, y + 0.49))

    box(ax, (14.35, 9.77), 1.50, 1.42,
        'joint read\n' + r'$q_t=RMSNorm($' + '\n'
        + r'$\pi_fm_t^f+\pi_mm_t^m)$', '#b3e5fc', size=8.3)
    box(ax, (16.23, 9.77), 1.42, 1.42,
        'residual\n' + r'$\tilde z_t=$' + '\n' + r'$z_t+W_oq_t$',
        '#c8e6c9', size=8.7)
    for _, _, _, y, _, _ in levels:
        arrow(ax, (13.96, y + 0.49), (14.35, 10.48), color='#5e35b1')
    arrow(ax, (15.85, 10.48), (16.23, 10.48))
    arrow(ax, (0.96, 11.04), (16.94, 11.38), color='#1976d2', dashed=True,
          connection='arc3,rad=-0.065')
    ax.text(9.02, 12.02,
            r'fixed scalar $\beta_k=1-e^{-1/\tau_k}$; $\rho_k=0.5$ at initialization; '
            'EMA teacher is absent from inference',
            ha='center', fontsize=8.9, fontweight='bold', color='#3949ab')

    # Training-only counterfactual recovery.
    ax.text(0.35, 8.63, 'TRAINING-ONLY SELF-SUPERVISION', fontsize=11.2,
            fontweight='bold', color='#37474f')
    box(ax, (0.38, 6.40), 4.30, 1.78,
        'EMA memory teacher  •  momentum 0.99\n'
        'consumes the original occluded trajectory only\n'
        r'$y^k_t=stopgrad(m^{k,teacher}_t)$' + '\n'
        'teacher parameters are not trainable\nand never enter inference',
        '#ede7f6', size=8.4, linewidth=1.8, edge='#512da8')
    box(ax, (5.05, 6.40), 7.90, 1.78,
        'counterfactual student window: [visible source, h synthetic black frames, visible restore]\n'
        r'$r_0=stopgrad(m_t)$; for $j=1\ldots h$: '
        r'$r_j=Correct(T(r_{j-1},a_{t+j-1}),z_{black})$' + '\n'
        r'$\mathcal{L}_{bridge}=SmoothL1(LN(r_h^k),LN(y^k_{t+h}))$' + '\n'
        r'$\mathcal{L}_{recovery}=SmoothL1(LN(Correct(T(r_h,a_{t+h}),z_{t+h+1})^k),'
        r'LN(y^k_{t+h+1}))$',
        '#f3e5f5', size=8.05, linewidth=1.9, edge='#6a1b9a')
    box(ax, (13.32, 6.40), 4.30, 1.78,
        'leakage and optimization boundary\n'
        'ALL eligible windows lie wholly inside\noriginally visible runs; overlap = 0\n'
        r'$W_x$ outputs detached; targets stop-gradient' + '\n'
        r'aux gradients: action + gate + $\rho$ only',
        '#e8eaf6', size=8.2, linewidth=1.8, edge='#3949ab')
    arrow(ax, (12.70, 9.15), (10.55, 8.18), color='#7b1fa2', dashed=True)
    arrow(ax, (2.55, 6.40), (5.05, 7.27), color='#7b1fa2')
    ax.text(9, 6.08,
            r'hierarchy: fast $h\in\{1,2\}$; medium $h\in\{4,8\}$  •  '
            r'$\lambda=.02$ through epoch 40, cosine to 0 at epoch 100, then 0',
            ha='center', fontsize=9.0, fontweight='bold', color='#455a64')

    # Every V7 variant.
    ax.text(0.35, 5.64, 'FROZEN V7 VARIANT MAP  •  all tensors instantiated; one exact change from full',
            fontsize=11.2, fontweight='bold', color='#37474f')
    variants = (
        ('hacssmv7 (full)', 'level-specific action heads\nshrinkage + bridge/recovery', '#d1c4e9'),
        ('hacssmv7_noaux', 'identical online inference\neffective auxiliary weight 0', '#c8e6c9'),
        ('hacssmv7_sharedaction', 'average projected action heads\nbefore both level transitions', '#ffecb3'),
        ('hacssmv7_noshrink', r'fix $\rho_f=\rho_m=1$' + '\ndynamic correction only', '#f8bbd0'),
        ('hacssmv7_actiononly', 'V6-style action-only bridge\nno correction/recovery loss', '#ffe0b2'),
        ('hacssmv7_uniform', 'both levels use every\nhorizon {1,2,4,8}', '#e1f5fe'),
        ('hacssmv7_norecovery', 'counterfactual bridge only\nremove restored-frame loss', '#f3e5f5'),
        ('hacssmv7_noaction', 'zero action features in\ninference and auxiliary', '#fff3e0'),
        ('hacssmv7_single', 'medium-only inference read\nboth states/objectives retained', '#b2ebf2'),
    )
    for index, (title, body, color) in enumerate(variants):
        row, col = divmod(index, 5)
        x = 0.35 + col * 3.52
        y = 3.75 - row * 1.66
        box(ax, (x, y), 3.20, 1.28, title + '\n' + body, color,
            size=7.55, linewidth=1.35)

    ax.text(9, 1.76,
            'Grid anchors: SSM • V4-two/no-aux • full V6 • static V6.  '
            'The V7 candidate and eight controls are parameter-matched.',
            ha='center', fontsize=8.8, color='#455a64')
    ax.text(9, 1.38,
            r'$D=128,A=6$: 36,102 trainable memory parameters; $2D$ recurrent floats.  '
            'The EMA teacher is checkpointed but frozen and excluded from the trainable count.',
            ha='center', fontsize=8.8, color='#455a64')
    ax.text(9, 0.94,
            'Frozen architecture; official 325-run grid completed with the locked negative '
            'result PILOT_NO_GO_FINAL_DESCRIPTIVE (§7.9).',
            ha='center', fontsize=9.1, fontweight='bold', color='#6a1b9a')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    main()
