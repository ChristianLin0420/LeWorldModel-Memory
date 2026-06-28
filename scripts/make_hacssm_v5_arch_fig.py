#!/usr/bin/env python3
"""Generate the prospective HACSSM-v5 architecture figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'docs' / 'figures' / 'fig_hacssm_v5_arch.png'


def box(ax, xy, width, height, text, color, *, size=9.5, linewidth=1.6):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle='round,pad=0.025,rounding_size=0.035',
        facecolor=color, edgecolor='#263238', linewidth=linewidth, zorder=2)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha='center', va='center',
            fontsize=size, color='#17202a', zorder=3, linespacing=1.25)


def arrow(ax, start, end, *, color='#455a64', dashed=False, connection='arc3'):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=13, linewidth=1.5,
        color=color, linestyle='--' if dashed else '-', connectionstyle=connection, zorder=1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=180)
    fig.patch.set_facecolor('#fbfcfd')
    ax.set_facecolor('#fbfcfd')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.5)
    ax.axis('off')

    ax.text(8, 9.08, 'HACSSM-v5: Hierarchical Action-Conditioned Selective SSM',
            ha='center', fontsize=18, fontweight='bold', color='#17202a')
    ax.text(8, 8.70,
            'V4 action predict/correct path + hard-monotone learned channel rates; '
            'slow level removed; front-loaded boundary shaping',
            ha='center', fontsize=10.3, color='#546e7a')

    box(ax, (0.35, 5.55), 1.25, 0.75, r'observation $z_t$', '#e3f2fd', size=10.5)
    box(ax, (0.35, 3.35), 1.25, 0.75, r'action $a_{t-1}$', '#fff3e0', size=10.5)
    box(ax, (1.95, 5.55), 1.35, 0.75, r'$x_t=W_xz_t$', '#e8f5e9', size=10.5)
    box(ax, (1.95, 3.35), 1.55, 0.75, r'$(d,v)=W_aa_{t-1}$', '#fff8e1', size=9.7)
    arrow(ax, (1.60, 5.93), (1.95, 5.93))
    arrow(ax, (1.60, 3.73), (1.95, 3.73))

    box(ax, (3.80, 7.05), 4.40, 1.05,
        r'hard-monotone learned transition/correction gains' + '\n'
        r'$\beta_m=\sigma(\theta_m),\quad '
        r'\beta_f=\beta_m+(1-\beta_m)\sigma(\theta_{gap})$' + '\n'
        r'init only: $\tau_f=1.5\ldots8,\;\tau_m=8\ldots64$ per channel',
        '#e8eaf6', size=9.4, linewidth=1.8)

    levels = (
        ('fast', 'f', 6.00, '#dcedc8', '#c5cae9'),
        ('medium', 'm', 4.35, '#b2dfdb', '#d1c4e9'),
    )
    for name, symbol, y, prior_color, posterior_color in levels:
        box(ax, (4.00, y), 2.35, 0.95,
            rf'{name} action prior  $p_t^{symbol}$' + '\n'
            + rf'$m_{{t-1}}^{symbol}+\beta_{symbol}\odot\tanh($' + '\n'
            + rf'$v+d\odot LN(m_{{t-1}}^{symbol}))$',
            prior_color, size=8.0)
        box(ax, (6.90, y), 2.25, 0.95,
            rf'selective correction  $g_t^{symbol}$' + '\n'
            + r'$\sigma((w_z^\top LN(z_t)+$' + '\n'
            + r'$w_e^\top LN(x_t-p_t))/\sqrt{D}+b)$',
            '#ffecb3', size=7.7)
        box(ax, (9.70, y), 2.35, 0.95,
            rf'posterior  $m_t^{symbol}$' + '\n'
            + rf'$p_t^{symbol}+\beta_{symbol}\odot g_t^{symbol}$' + '\n'
            + rf'$\odot(x_t-p_t^{symbol})$',
            posterior_color, size=8.1)
        arrow(ax, (3.50, 3.73), (4.00, y + 0.36), color='#ef6c00')
        arrow(ax, (6.35, y + 0.47), (6.90, y + 0.47))
        arrow(ax, (9.15, y + 0.47), (9.70, y + 0.47))
        arrow(ax, (3.30, 5.93), (6.90, y + 0.70), color='#1976d2')
        arrow(ax, (3.30, 5.93), (9.70, y + 0.22), color='#2e7d32', dashed=True)

    arrow(ax, (6.00, 7.05), (5.20, 6.95), color='#3949ab', dashed=True)
    arrow(ax, (6.00, 7.05), (5.20, 5.30), color='#3949ab', dashed=True,
          connection='arc3,rad=0.12')

    box(ax, (12.65, 5.05), 1.45, 1.20,
        r'global read' + '\n' + r'$q_t=RMSNorm($' + '\n'
        + r'$\pi_fm_t^f+\pi_mm_t^m)$', '#b3e5fc', size=8.5)
    box(ax, (14.50, 5.05), 1.18, 1.20,
        r'residual' + '\n' + r'$\tilde z_t=z_t+W_oq_t$', '#c8e6c9', size=9.6)
    for _, _, y, _, _ in levels:
        arrow(ax, (12.05, y + 0.47), (12.65, 5.65), color='#5e35b1')
    arrow(ax, (14.10, 5.65), (14.50, 5.65))
    arrow(ax, (0.98, 6.30), (15.09, 6.25), color='#1976d2', dashed=True,
          connection='arc3,rad=-0.10')

    box(ax, (4.25, 1.05), 7.80, 1.70,
        'training-only first-visible boundary shaping (not general hierarchical SSL)\n'
        r'$r_{t,0}^k=m_t^k,\quad r_{t,j+1}^k=T_k(r_{t,j}^k,a_{t+j}),\quad '
        r'r_{t,h}^k\rightarrow stopgrad(z_{16}^{clean})$' + '\n'
        + r'fast sources $t=15,14$ ($h=1,2$)  •  medium $t=12,8$ ($h=4,8$)' + '\n'
        + r'$\lambda=.05$ epochs 1–20  •  cosine to 0 at epoch 120  •  0 thereafter',
        '#f3e5f5', size=9.4, linewidth=1.9)
    arrow(ax, (10.85, 6.00), (5.75, 2.75), color='#7b1fa2', dashed=True,
          connection='arc3,rad=-0.12')
    arrow(ax, (10.85, 4.35), (9.85, 2.75), color='#7b1fa2', dashed=True)
    arrow(ax, (2.72, 3.35), (4.25, 1.75), color='#ef6c00', dashed=True)

    ax.text(8.00, 0.56,
            r'Initialization: $W_x=I$, $W_a=W_o=0$, $b_f=b_m=2$, $\pi=(1/2,1/2)$.  '
            r'Memory params: 34,820 at $D=128,A=6$; recurrent state: $2D$.',
            ha='center', fontsize=9.1, color='#455a64')
    ax.text(8.00, 0.20,
            r'actual observation rate = $\beta\odot g$; $\tau(\beta)$ is not an effective horizon',
            ha='center', fontsize=8.7, fontweight='bold', color='#bf360c')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    main()
