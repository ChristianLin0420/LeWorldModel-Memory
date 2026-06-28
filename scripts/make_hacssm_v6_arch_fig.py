#!/usr/bin/env python3
"""Generate the prospective HACSSM-v6 architecture and variant-contract figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'docs' / 'figures' / 'fig_hacssm_v6_arch.png'


def box(ax, xy, width, height, text, color, *, size=9.0, linewidth=1.5,
        edge='#263238'):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle='round,pad=0.025,rounding_size=0.035',
        facecolor=color, edgecolor=edge, linewidth=linewidth, zorder=2)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha='center', va='center',
            fontsize=size, color='#17202a', zorder=3, linespacing=1.25)


def arrow(ax, start, end, *, color='#455a64', dashed=False, connection='arc3'):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=13, linewidth=1.45,
        color=color, linestyle='--' if dashed else '-', connectionstyle=connection, zorder=1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(18, 12), dpi=180)
    fig.patch.set_facecolor('#fbfcfd')
    ax.set_facecolor('#fbfcfd')
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')

    ax.text(9, 11.62, 'HACSSM-v6: Dense Hierarchical Action Consistency',
            ha='center', fontsize=19, fontweight='bold', color='#17202a')
    ax.text(9, 11.27,
            'Exact fixed-rate V4-two inference anchor; causal self-distillation acts only during training',
            ha='center', fontsize=10.6, color='#546e7a')

    # Online inference path.
    ax.text(0.35, 10.73, 'ONLINE INFERENCE  •  identical for full / no-aux / auxiliary-only controls',
            fontsize=11.3, fontweight='bold', color='#37474f')
    box(ax, (0.35, 8.70), 1.30, 0.72, r'$z_t$', '#e3f2fd', size=11)
    box(ax, (0.35, 7.45), 1.30, 0.72, r'$a_{t-1}$', '#fff3e0', size=11)
    box(ax, (1.95, 8.70), 1.55, 0.72, r'$x_t=W_xz_t$', '#e8f5e9', size=10.3)
    box(ax, (1.95, 7.45), 1.85, 0.72, r'$(d,v)=W_aa_{t-1}$', '#fff8e1', size=9.5)
    arrow(ax, (1.65, 9.06), (1.95, 9.06))
    arrow(ax, (1.65, 7.81), (1.95, 7.81))

    levels = (
        ('FAST', 'f', r'$\tau_f=2$', 9.42, '#dcedc8', '#c5cae9'),
        ('MEDIUM', 'm', r'$\tau_m=8$', 7.92, '#b2dfdb', '#d1c4e9'),
    )
    for name, symbol, tau, y, prior_color, posterior_color in levels:
        box(ax, (4.25, y), 2.55, 0.95,
            f'{name} action prior  {tau}\n'
            + rf'$p_t^{symbol}=m_{{t-1}}^{symbol}+\beta_{symbol}\tanh($' + '\n'
            + rf'$v+d\odot LN(m_{{t-1}}^{symbol}))$', prior_color, size=7.9)
        box(ax, (7.30, y), 2.25, 0.95,
            rf'correction  $g_t^{symbol}$' + '\n'
            + r'$\sigma((w_z^\top LN(z_t)+$' + '\n'
            + r'$w_e^\top LN(x_t-p_t))/\sqrt{D}+b)$', '#ffecb3', size=7.5)
        box(ax, (10.05, y), 2.55, 0.95,
            rf'posterior  $m_t^{symbol}$' + '\n'
            + rf'$p_t^{symbol}+\beta_{symbol}g_t^{symbol}$' + '\n'
            + rf'$\odot(x_t-p_t^{symbol})$', posterior_color, size=8.0)
        arrow(ax, (3.80, 7.81), (4.25, y + 0.39), color='#ef6c00')
        arrow(ax, (6.80, y + 0.47), (7.30, y + 0.47))
        arrow(ax, (9.55, y + 0.47), (10.05, y + 0.47))
        arrow(ax, (3.50, 9.06), (7.30, y + 0.70), color='#1976d2')
        arrow(ax, (3.50, 9.06), (10.05, y + 0.22), color='#2e7d32', dashed=True)

    box(ax, (13.10, 8.58), 2.10, 1.28,
        'joint global read\n'
        + r'$q_t=RMSNorm($' + '\n' + r'$\pi_fm_t^f+\pi_mm_t^m)$',
        '#b3e5fc', size=8.8)
    box(ax, (15.65, 8.58), 1.90, 1.28,
        'residual injection\n' + r'$\tilde z_t=z_t+W_oq_t$', '#c8e6c9', size=9.1)
    for _, _, _, y, _, _ in levels:
        arrow(ax, (12.60, y + 0.47), (13.10, 9.22), color='#5e35b1')
    arrow(ax, (15.20, 9.22), (15.65, 9.22))
    arrow(ax, (1.00, 9.42), (16.60, 9.86), color='#1976d2', dashed=True,
          connection='arc3,rad=-0.08')
    ax.text(8.52, 10.51,
            r'fixed scalar gains $\beta_k=1-e^{-1/\tau_k}$; no learned V5 rate spectrum',
            ha='center', fontsize=9.1, fontweight='bold', color='#3949ab')

    # Training-only objective.
    ax.text(0.35, 6.94, 'TRAINING-ONLY SELF-SUPERVISION', fontsize=11.3,
            fontweight='bold', color='#37474f')
    box(ax, (0.40, 4.42), 11.60, 2.08,
        'dense same-level posterior-state action consistency\n'
        + r'$r_{t,0}^k=stopgrad(m_t^k),\qquad '
        + r'r_{t,j+1}^k=T_k(r_{t,j}^k,a_{t+j})$' + '\n'
        + r'$\mathcal{L}_{k,h}=SmoothL1(LayerNorm(r_{t,h}^k),'
        + r'LayerNorm(stopgrad(m_{t+h}^k)))$' + '\n'
        + r'FAST: $h\in\{1,2\}$   •   MEDIUM: $h\in\{4,8\}$   •   '
        + r'every endpoint with original $target\_valid\_mask=True$' + '\n'
        + 'no endpoint/intervening observation enters the action-only rollout; '
        + 'hidden clean blackout targets are never used',
        '#f3e5f5', size=9.1, linewidth=1.9, edge='#6a1b9a')
    arrow(ax, (11.20, 9.42), (9.45, 6.50), color='#7b1fa2', dashed=True)
    arrow(ax, (11.20, 7.92), (10.55, 6.50), color='#7b1fa2', dashed=True)
    arrow(ax, (2.85, 7.45), (2.85, 6.50), color='#ef6c00', dashed=True)
    box(ax, (12.45, 4.42), 5.10, 2.08,
        'bootstrap schedule\n'
        + r'$\lambda=.02$ for epochs 1–40' + '\n'
        + 'cosine decay to 0 at epoch 100\n'
        + r'$\lambda=0$ for epochs 101–200' + '\n'
        + 'default detach ⇒ auxiliary gradients\nonly update the action map $W_a$',
        '#ede7f6', size=9.1, linewidth=1.8, edge='#512da8')

    # Complete variant map.
    ax.text(0.35, 3.98, 'FROZEN V6 VARIANT MAP  •  one exact change from full unless stated',
            fontsize=11.3, fontweight='bold', color='#37474f')
    variants = (
        ('full', 'detached source + target\nfast 1/2; medium 4/8', '#d1c4e9'),
        ('noaux', 'identical inference\nauxiliary weight = 0', '#c8e6c9'),
        ('aux_noaction', 'zero actions only inside\nauxiliary rollouts', '#ffecb3'),
        ('uniform', 'both levels use every\nhorizon {1,2,4,8}', '#e1f5fe'),
        ('sourcegrad', 'source posterior is not\ndetached; target still is', '#f8bbd0'),
        ('fastonly', 'auxiliary uses only fast\nlevel at h={1,2}', '#dcedc8'),
        ('mediumonly', 'auxiliary uses only medium\nlevel at h={4,8}', '#b2dfdb'),
        ('noaction', 'zero action features in\ninference and auxiliary', '#ffe0b2'),
        ('static', 'input-independent correction\ngates; auxiliary unchanged', '#fff3e0'),
        ('single', 'medium-only inference read;\nboth states/aux retained', '#b2ebf2'),
    )
    for index, (title, body, color) in enumerate(variants):
        row, col = divmod(index, 5)
        x = 0.35 + col * 3.52
        y = 2.15 - row * 1.55
        box(ax, (x, y), 3.20, 1.22,
            (f'hacssmv6_{title}' if title != 'full' else 'hacssmv6 (full)')
            + '\n' + body,
            color, size=7.8, linewidth=1.4)

    ax.text(9, 0.19,
            r'$D=128,A=6$: 34,564 memory parameters; $2D$ recurrent state.  '
            'Auxiliary-only variants are exactly inference-identical at equal weights.',
            ha='center', fontsize=8.8, color='#455a64')

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    main()
