"""Render the Selective Multi-Timescale (SMT) memory architecture schematic for
docs/LEARNABLE_MEMORY.md. Color code: blue = LEARNED modules, gray = FIXED (decays)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

LEARN = '#3182bd'      # learned modules
FIXED = '#9e9e9e'      # fixed decays
GATE = '#2ca25f'       # gating ops
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.set_xlim(0, 11); ax.set_ylim(0, 7.6); ax.axis('off')


def box(x, y, w, h, text, fc, ec='#333', fs=9, tc='white', r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc, zorder=5)


def arrow(x1, y1, x2, y2, c='#444', lw=1.4, style='-|>'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                 color=c, lw=lw, shrinkA=2, shrinkB=2))


# --- input ---
box(0.2, 3.3, 1.15, 0.9, r'$z_t$' + '\nlatent', '#37474f', fs=11)
arrow(1.35, 3.75, 2.0, 3.75)

# --- write gate (learned) + elementwise multiply ---
box(2.0, 5.5, 2.3, 0.85, r'write gate  $i_t=\sigma(W_i z_t)$', LEARN, fs=9)
ax.text(2.0, 6.55, 'LEARNED', fontsize=7.5, color=LEARN, weight='bold')
arrow(1.0, 4.2, 1.0, 5.95); arrow(1.0, 5.95, 2.0, 5.95)             # z_t up to write gate
ax.add_patch(plt.Circle((4.9, 4.6), 0.22, fc=GATE, ec='#333', lw=1.2, zorder=4))
ax.text(4.9, 4.6, r'$\odot$', ha='center', va='center', fontsize=13, color='white', zorder=5)
arrow(4.3, 5.9, 4.9, 4.82)                                          # i_t -> mult
arrow(2.0, 3.95, 4.68, 4.6)                                         # z_t -> mult (other input)
ax.text(3.0, 4.0, r'$z_t$', fontsize=8, color='#666')
ax.text(5.15, 5.05, r'$i_t\!\odot z_t$' + '\n(what to store)', fontsize=7.5, color='#333')

# --- K fixed EMA banks (log-spaced) ---
taus = [2, 4, 8, 16, 32, 64]
y0, bh, gap = 0.5, 0.62, 0.12
ax.text(6.55, 5.7, 'FIXED log-spaced EMA banks', ha='center', fontsize=8.5, color='#555', weight='bold')
bank_y = []
for k, t in enumerate(taus):
    y = y0 + k * (bh + gap)
    bank_y.append(y + bh / 2)
    box(5.7, y, 1.7, bh, r'$m^{%d}:\ \tau=%d$' % (k + 1, t), FIXED, fs=8, tc='white')
    # tiny decay glyph
    xs = np.linspace(0, 1, 20); ys = np.exp(-xs * (8.0 / t))
    ax.plot(7.0 + xs * 0.32, y + 0.12 + ys * (bh - 0.24), color='white', lw=0.8)
arrow(5.12, 4.55, 5.7, bank_y[-1])                                  # gated input -> banks (fan)
for by in bank_y[:-1]:
    ax.add_patch(FancyArrowPatch((5.4, 4.4), (5.7, by), arrowstyle='-', color='#bbb', lw=0.7))

# --- read router (learned) ---
box(2.0, 1.7, 2.3, 0.85, r'read router  $r_t=g(W_r z_t)$', LEARN, fs=9)
ax.text(2.0, 1.35, 'LEARNED', fontsize=7.5, color=LEARN, weight='bold')
arrow(1.0, 3.3, 1.0, 2.12); arrow(1.0, 2.12, 2.0, 2.12)            # z_t down to router
ax.text(1.05, 2.7, r'$z_t$', fontsize=8, color='#666')
ax.text(2.05, 0.95, 'softmax = mixture (v1)\nsigmoid = additive gates (v2)', fontsize=7, color='#888')

# --- weighted readout (gating) ---
ax.add_patch(plt.Circle((8.25, 3.2), 0.26, fc=GATE, ec='#333', lw=1.2, zorder=4))
ax.text(8.25, 3.2, r'$\Sigma$', ha='center', va='center', fontsize=13, color='white', zorder=5)
for by in bank_y:
    arrow(7.4, by, 8.02, 3.2, c='#9e9e9e', lw=0.9)                  # banks -> weighted sum
for by in [2.1, 3.2, 4.3]:
    ax.add_patch(FancyArrowPatch((4.3, 2.12), (8.0, 3.2), arrowstyle='-', color=LEARN, lw=0.6, alpha=0.5))
ax.text(6.0, 2.25, r'$\times\,r_{t,k}$', fontsize=8.5, color=LEARN)

# --- output projection (learned) + residual ---
box(8.75, 2.78, 1.0, 0.85, r'$W_o$', LEARN, fs=10)
ax.text(8.75, 3.7, 'LEARNED', fontsize=7.5, color=LEARN, weight='bold')
arrow(8.51, 3.2, 8.75, 3.2)
ax.add_patch(plt.Circle((10.15, 3.75), 0.22, fc='#37474f', ec='#333', lw=1.2, zorder=4))
ax.text(10.15, 3.75, '+', ha='center', va='center', fontsize=14, color='white', zorder=5)
arrow(9.75, 3.2, 10.05, 3.6)                                       # o_t -> add
arrow(1.0, 4.2, 1.0, 7.2); ax.add_patch(FancyArrowPatch((1.0, 7.2), (10.15, 7.2), arrowstyle='-', color='#37474f', lw=1.2))
arrow(10.15, 7.2, 10.15, 3.97)                                     # residual skip z_t -> add
ax.text(5.5, 7.32, r'residual: $\tilde z_t = z_t + o_t$', fontsize=8.5, color='#37474f')
arrow(10.37, 3.75, 10.85, 3.75)
ax.text(10.6, 3.2, r'$\tilde z_t$' + '\nto predictor', fontsize=8.5, color='#333', ha='center')

# legend
ax.add_patch(FancyBboxPatch((0.2, 0.15), 0.3, 0.3, boxstyle="round,pad=0.02", fc=LEARN, ec='#333'))
ax.text(0.6, 0.3, 'learned ($W_i,W_r,W_o$)', fontsize=8, va='center')
ax.add_patch(FancyBboxPatch((3.7, 0.15), 0.3, 0.3, boxstyle="round,pad=0.02", fc=FIXED, ec='#333'))
ax.text(4.1, 0.3, 'fixed (log-spaced decays $a_k$)', fontsize=8, va='center')
ax.set_title('Selective Multi-Timescale (SMT) memory: fixed timescale basis + learned input-conditioned gating',
             fontsize=10.5, weight='bold')
plt.tight_layout()
plt.savefig('docs/figures/fig_smt_arch.png', dpi=150, bbox_inches='tight')
print('wrote docs/figures/fig_smt_arch.png')
