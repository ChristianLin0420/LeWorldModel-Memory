"""Render the SMT-v3-W architecture schematic for docs/LEARNABLE_MEMORY.md.

Color code: blue = learned, gray = fixed, green = selective update, orange =
parameter-free normalization.  The diagram deliberately shows that actions bypass
the recurrent state and enter only the short-context predictor.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


LEARN = "#3182bd"
FIXED = "#8c8c8c"
UPDATE = "#238b45"
NORM = "#e67e22"
DARK = "#37474f"
LIGHT = "#f7f7f7"

fig, ax = plt.subplots(figsize=(12.4, 6.2))
ax.set_xlim(0, 12.4)
ax.set_ylim(0, 7.2)
ax.axis("off")


def box(x, y, w, h, text, fc, ec="#333", fs=9, tc="white", lw=1.3, r=0.08):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.03,rounding_size={r}",
            fc=fc,
            ec=ec,
            lw=lw,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=tc,
        zorder=5,
    )


def arrow(x1, y1, x2, y2, c="#444", lw=1.4, style="-|>", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=12,
            color=c,
            lw=lw,
            linestyle=ls,
            shrinkA=2,
            shrinkB=2,
        )
    )


# Observation input and learned scalar conditioner.
box(0.25, 3.25, 1.1, 0.82, r"$z_t$" + "\nlatent", DARK, fs=10.5)
box(
    1.85,
    5.35,
    2.6,
    0.92,
    r"$g_t=\sigma(w_g^\top[\mathrm{LN}(z_t)+e]+b_g)$"
    + "\nscalar update gate",
    LEARN,
    fs=8.8,
)
ax.text(1.88, 6.42, "LEARNED CONDITIONER", color=LEARN, fontsize=7.5, weight="bold")
arrow(0.8, 4.07, 0.8, 5.8)
arrow(0.8, 5.8, 1.85, 5.8)

# Whole-update operation.
box(
    4.95,
    4.95,
    2.65,
    1.45,
    r"$m_t^k=(1-a_k g_t)m_{t-1}^k$"
    + "\n"
    + r"$\qquad +\ a_k g_t z_t$"
    + "\n"
    + r"$g_t=0$: exact freeze",
    UPDATE,
    fs=9.2,
)
ax.text(5.0, 6.55, "TRUE SELECTIVE UPDATE", color=UPDATE, fontsize=7.5, weight="bold")
arrow(4.45, 5.8, 4.95, 5.8, c=LEARN)
arrow(1.35, 3.66, 5.15, 5.05, c=DARK)

# Fixed EMA states.
taus = [2, 4, 8, 16, 32, 64]
y0, bh, gap = 0.6, 0.55, 0.12
bank_y = []
ax.text(8.55, 4.7, "state bank with FIXED decays", ha="center", fontsize=8.5,
        color="#555", weight="bold")
for k, tau in enumerate(taus):
    y = y0 + k * (bh + gap)
    bank_y.append(y + bh / 2)
    box(7.75, y, 1.62, bh, rf"$m_t^{{{k + 1}}}:\ \tau={tau}$", FIXED, fs=7.8)
    xs = np.linspace(0, 1, 18)
    ys = np.exp(-xs * (8.0 / tau))
    ax.plot(9.05 + xs * 0.24, y + 0.1 + ys * (bh - 0.2), color="white", lw=0.7)
arrow(7.6, 5.62, 7.75, bank_y[-1], c=UPDATE)
for by in bank_y[:-1]:
    ax.add_patch(
        FancyArrowPatch(
            (7.56, 5.5), (7.75, by), arrowstyle="-", color="#aaaaaa", lw=0.7
        )
    )

# Global route and weighted read.
box(2.0, 1.25, 2.3, 0.8, r"$\pi=\mathrm{softmax}(r)$" + "\nglobal bank mixture", LEARN, fs=9)
ax.text(2.03, 0.95, "LEARNED, TIME-INDEPENDENT", color=LEARN, fontsize=7.2, weight="bold")
box(9.85, 2.72, 1.0, 0.85, r"$\sum_k\pi_k m_t^k$", UPDATE, fs=8.5)
for by in bank_y:
    arrow(9.37, by, 9.85, 3.15, c="#999", lw=0.8)
arrow(4.3, 1.65, 9.98, 2.72, c=LEARN, lw=1.1)

# RMS normalization, shared projection, and residual.
box(10.98, 2.72, 0.72, 0.85, "RMS\nnorm", NORM, fs=8.2)
arrow(10.85, 3.15, 10.98, 3.15)
box(10.98, 4.05, 0.72, 0.72, r"$W_o$", LEARN, fs=10)
arrow(11.34, 3.57, 11.34, 4.05)
ax.add_patch(plt.Circle((11.95, 4.42), 0.2, fc=DARK, ec="#333", lw=1.2, zorder=4))
ax.text(11.95, 4.42, "+", ha="center", va="center", fontsize=13, color="white", zorder=5)
arrow(11.7, 4.42, 11.75, 4.42)
arrow(0.8, 4.07, 0.8, 6.85)
ax.add_patch(
    FancyArrowPatch((0.8, 6.85), (11.95, 6.85), arrowstyle="-", color=DARK, lw=1.2)
)
arrow(11.95, 6.85, 11.95, 4.62)
arrow(12.15, 4.42, 12.38, 4.42)
ax.text(11.9, 3.72, r"$\tilde z_t$", fontsize=8.5, color=DARK)

# Explicitly show the action-blind recurrent path.
box(0.25, 0.25, 1.1, 0.65, r"$a_t$" + "\naction", "#6a51a3", fs=9)
box(10.55, 0.25, 1.55, 0.65, "short-context\npredictor", DARK, fs=8.5)
arrow(1.35, 0.58, 10.55, 0.58, c="#6a51a3", ls="--")
arrow(12.18, 4.35, 11.85, 0.9, c=DARK, ls="--")
ax.text(4.7, 0.7, "actions do not update recurrent state", color="#6a51a3", fontsize=8,
        ha="center", weight="bold")

# Compact control legend.
box(0.25, 2.05, 3.95, 0.7,
    "controls: static gate  |  old erasing update  |  hard visibility mask",
    LIGHT, ec="#777", fs=7.8, tc="#333", lw=1.0)

ax.set_title(
    "SMT-v3-W: scalar whole-update gating over a state bank with fixed timescales",
    fontsize=11,
    weight="bold",
)
plt.tight_layout()
plt.savefig("docs/figures/fig_smtv3_arch.png", dpi=160, bbox_inches="tight")
print("wrote docs/figures/fig_smtv3_arch.png")
