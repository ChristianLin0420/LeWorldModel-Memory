"""Render the OC-SMT architecture schematic for docs/LEARNABLE_MEMORY.md §9.
Contrast with SMT (Fig 1): an OVER-COMPLETE fixed basis (M=28) with per-bank L0 hard-concrete
gates that PRUNE to a learnable, variable-size active set (no constant K).
Color: blue = learned (W_i, W_g, W_o), gray = fixed decays, green = gate OPEN, faded = pruned."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

LEARN, FIXED, GATE, OPEN, SHUT = '#3182bd', '#9e9e9e', '#2ca25f', '#2ca25f', '#d9d9d9'
fig, ax = plt.subplots(figsize=(11.5, 6.0))
ax.set_xlim(0, 11.5); ax.set_ylim(0, 8.2); ax.axis('off')


def box(x, y, w, h, text, fc, fs=9, tc='white', r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                                fc=fc, ec='#333', lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc, zorder=5)


def arrow(x1, y1, x2, y2, c='#444', lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12,
                                 color=c, lw=lw, shrinkA=2, shrinkB=2))


# input
box(0.2, 3.6, 1.15, 0.9, r'$z_t$' + '\nlatent', '#37474f', fs=11)
arrow(1.35, 4.05, 2.0, 4.05)

# write gate (learned)
box(2.0, 6.0, 2.25, 0.8, r'write gate $i_t=\sigma(W_i z_t)$', LEARN, fs=8.5)
ax.text(2.0, 6.95, 'LEARNED', fontsize=7.5, color=LEARN, weight='bold')
arrow(1.0, 4.5, 1.0, 6.4); arrow(1.0, 6.4, 2.0, 6.4)
ax.add_patch(plt.Circle((4.85, 4.9), 0.2, fc=GATE, ec='#333', lw=1.1, zorder=4))
ax.text(4.85, 4.9, r'$\odot$', ha='center', va='center', fontsize=12, color='white', zorder=5)
arrow(4.25, 6.3, 4.85, 5.1); arrow(2.0, 4.2, 4.65, 4.9)
ax.text(5.05, 5.35, r'$i_t\!\odot z_t$', fontsize=8, color='#333')

# L0 gate logits (learned)
box(2.0, 1.25, 2.25, 0.8, r'L0 gate $W_g z_t$', LEARN, fs=8.5)
ax.text(2.0, 0.9, 'LEARNED', fontsize=7.5, color=LEARN, weight='bold')
arrow(1.0, 3.6, 1.0, 1.65); arrow(1.0, 1.65, 2.0, 1.65)

# over-complete fixed bank column (show 12 representative of M=28), with L0 gates (open/closed)
taus_shown = [1.5, 2.5, 4, 6, 10, 16, 26, 42, 68, 110, 175, 256]
# illustrative emergent active set: a few open (e.g. one short + a couple long), rest pruned
open_idx = {1, 6, 9, 11}
n = len(taus_shown); y0, bh, gap = 0.45, 0.50, 0.085
ax.text(6.5, 7.55, r'OVER-COMPLETE fixed basis  ($M{=}28$, $\tau\!=\!1.5\ldots256$)',
        ha='center', fontsize=9, color='#555', weight='bold')
gate_y = []
for k, t in enumerate(taus_shown):
    y = y0 + k * (bh + gap); gate_y.append(y + bh / 2)
    box(6.05, y, 1.5, bh, r'$\tau{=}%g$' % t, FIXED, fs=7.5)
    # L0 gate per bank
    is_open = k in open_idx
    ax.add_patch(plt.Circle((5.75, y + bh / 2), 0.12, fc=(OPEN if is_open else 'white'),
                            ec=(GATE if is_open else SHUT), lw=1.3, zorder=4))
    if not is_open:
        ax.plot([5.67, 5.83], [y + bh / 2, y + bh / 2], color=SHUT, lw=1.4, zorder=5)  # closed mark
arrow(5.0, 4.85, 5.55, gate_y[open_idx and max(open_idx)])  # gated input feeds banks (representative)
for gy in gate_y:
    ax.add_patch(FancyArrowPatch((5.1, 4.7), (5.62, gy), arrowstyle='-', color='#ddd', lw=0.5))
# L0 gate logits drive the per-bank gates
for gy in gate_y:
    ax.add_patch(FancyArrowPatch((4.25, 1.65), (5.63, gy), arrowstyle='-', color=LEARN, lw=0.4, alpha=0.4))
ax.text(3.0, 2.4, r'L0 penalty $\lambda_0\sum_m P(g_{t,m}{>}0)$' + '\n(annealed) prunes banks',
        fontsize=7.5, color=LEARN)

# weighted sum over OPEN banks -> W_o (learned) -> residual
ax.add_patch(plt.Circle((8.35, 4.0), 0.24, fc=GATE, ec='#333', lw=1.2, zorder=4))
ax.text(8.35, 4.0, r'$\Sigma$', ha='center', va='center', fontsize=12, color='white', zorder=5)
for k, gy in enumerate(gate_y):
    c = OPEN if k in open_idx else SHUT
    lw = 1.3 if k in open_idx else 0.5
    ax.add_patch(FancyArrowPatch((7.55, gy), (8.13, 4.0), arrowstyle='-|>', mutation_scale=8,
                                 color=c, lw=lw, alpha=1.0 if k in open_idx else 0.5))
ax.text(7.7, 5.7, r'only OPEN banks' + '\n' + r'(active set, size learned)', fontsize=7.5, color=GATE)
box(8.85, 3.6, 0.95, 0.8, r'$W_o$', LEARN, fs=10)
ax.text(8.85, 4.55, 'LEARNED', fontsize=7.5, color=LEARN, weight='bold')
arrow(8.59, 4.0, 8.85, 4.0)
ax.add_patch(plt.Circle((10.2, 4.55), 0.2, fc='#37474f', ec='#333', lw=1.2, zorder=4))
ax.text(10.2, 4.55, '+', ha='center', va='center', fontsize=13, color='white', zorder=5)
arrow(9.8, 4.0, 10.08, 4.4)
arrow(1.0, 4.5, 1.0, 7.9); ax.add_patch(FancyArrowPatch((1.0, 7.9), (10.2, 7.9), arrowstyle='-', color='#37474f', lw=1.1))
arrow(10.2, 7.9, 10.2, 4.77)
ax.text(5.5, 8.02, r'residual: $\tilde z_t = z_t + o_t$', fontsize=8.5, color='#37474f')
arrow(10.4, 4.55, 10.9, 4.55)
ax.text(10.65, 4.0, r'$\tilde z_t$', fontsize=9, color='#333', ha='center')

# legend
ax.add_patch(plt.Circle((0.35, 0.35), 0.1, fc=OPEN, ec=GATE)); ax.text(0.55, 0.35, 'gate OPEN (active)', fontsize=7.5, va='center')
ax.add_patch(plt.Circle((2.7, 0.35), 0.1, fc='white', ec=SHUT)); ax.text(2.9, 0.35, 'gate closed (pruned)', fontsize=7.5, va='center')
ax.add_patch(FancyBboxPatch((5.2, 0.25), 0.25, 0.2, boxstyle="round,pad=0.02", fc=FIXED, ec='#333')); ax.text(5.5, 0.35, 'fixed decay', fontsize=7.5, va='center')
ax.add_patch(FancyBboxPatch((6.9, 0.25), 0.25, 0.2, boxstyle="round,pad=0.02", fc=LEARN, ec='#333')); ax.text(7.2, 0.35, 'learned', fontsize=7.5, va='center')
ax.set_title('OC-SMT: over-complete fixed basis + L0 sparse gates → learnable variable-size active set (no constant K)',
             fontsize=10.5, weight='bold')
plt.tight_layout()
plt.savefig('docs/figures/fig_ocsmt_arch.png', dpi=150, bbox_inches='tight')
print('wrote docs/figures/fig_ocsmt_arch.png')
