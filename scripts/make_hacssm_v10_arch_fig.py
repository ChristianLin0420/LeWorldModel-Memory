#!/usr/bin/env python3
"""Render final ORBIT-v10-J, its five-mode contract, and measured host audit."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "figures" / "fig_hacssm_v10_arch.png"


def box(ax, x, y, w, h, title, body, color, *, title_size=10, body_size=7.5,
        edge="#37474f", linewidth=1.5):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.035,rounding_size=0.06",
        facecolor=color, edgecolor=edge, linewidth=linewidth)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color="#17202a")
    ax.text(x + w / 2, y + h / 2 - 0.14, body, ha="center", va="center",
            fontsize=body_size, color="#263238", linespacing=1.35)


def arrow(ax, start, end, label="", *, color="#546e7a", bend=0.0):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14, color=color,
        linewidth=1.7, connectionstyle=f"arc3,rad={bend}")
    ax.add_patch(patch)
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.16,
                label, ha="center", va="bottom", fontsize=7.3, color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(18, 15.2), dpi=190)
    fig.patch.set_facecolor("#fbfcfd")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 15.2)
    ax.axis("off")

    ax.text(9, 14.86,
            "ORBIT-v10-J: orthogonal recurrent belief with a joint-gradient causal host",
            ha="center", fontsize=19, fontweight="bold", color="#17202a")
    ax.text(9, 14.48,
            "Five-task 100-epoch host audit observed • official five-mode grid not launched (0/225)",
            ha="center", fontsize=11, color="#607d8b")

    # Online causal path.
    box(ax, 0.35, 11.45, 2.05, 2.25, "observed RGB $o_t$",
        "raw pixels\npossibly corrupted context\nno DINO/precompute",
        "#e3f2fd", body_size=7.0)
    box(ax, 2.85, 11.45, 2.35, 2.25, "causal encoder $E_\\theta$",
        "ViT + affine-free per-frame LN\nno peer/future statistics\nbatch-size-one valid",
        "#dcedc8", body_size=7.0)
    box(ax, 5.65, 11.45, 2.05, 2.25, "observation $x_t$",
        "$z_t=E_\\theta(o_t)$\n$x_t=W_xz_t$\n$W_x=I$ initially",
        "#e1f5fe", body_size=7.2)
    box(ax, 8.15, 11.45, 2.55, 2.25, "orthogonal action prior",
        "$p_t=T_\\psi(a_{t-1})m_{t-1}$\n2 shuffled Givens layers\n"
        "$T^\\top T=I$; identity init",
        "#d1c4e9", body_size=7.0)
    box(ax, 11.15, 11.45, 2.50, 2.25, "innovation correction",
        "$g_t=(1-\\rho)s+\\rho q_t$\n$m_t=p_t+g_t(x_t-p_t)$\n"
        "$g_0=\\sigma(-2)=.119$",
        "#fff3e0", body_size=7.0)
    box(ax, 14.10, 11.45, 3.50, 2.25, "residual LeWM read",
        "$r_t=\\mathrm{RMSNorm}(m_t)$\n$\\tilde z_t=z_t+W_or_t$; $W_o=0$ init\n"
        "short causal predictor $\\rightarrow\\hat z_{t+1}$",
        "#b2ebf2", body_size=7.0)
    for x0, x1 in ((2.40, 2.85), (5.20, 5.65), (7.70, 8.15),
                   (10.70, 11.15), (13.65, 14.10)):
        arrow(ax, (x0, 12.58), (x1, 12.58))
    arrow(ax, (9.42, 13.70), (9.42, 14.05), "$a_{t-1}$", bend=0.0)
    arrow(ax, (9.00, 11.45), (6.90, 10.82), "$m_{t-1}$ streaming state", bend=-0.14)

    # Exact transport and objective panels.
    box(ax, 0.35, 8.15, 8.35, 2.55, "two-layer normalized complex transport",
        "$h^\\ell=W_a^\\ell a;\\quad (u,v)=(1+h_{2j},h_{2j+1});\\quad"
        "(c,s)=(u,v)/\\sqrt{u^2+v^2}$\n"
        "$R(c,s)[r,i]^\\top=[cr-si,\\;sr+ci]^\\top$\n"
        "layer 1 pairs adjacent coordinates; layer 2 perfect-shuffles, pairs across halves, "
        "then unshuffles\n"
        "Every block and their product preserve norm; the two overlapping pairings can form "
        "non-commuting action maps.",
        "#ede7f6", title_size=11, body_size=7.4)
    box(ax, 9.05, 8.15, 8.55, 2.55, "V10-J joint self-supervised objective",
        "clean RGB $o^{clean} \\rightarrow E_\\theta^{eval}$ (dropout off; gradients active) "
        "$\\rightarrow U$; no EMA/teacher/stop-gradient\n"
        "$C=(U-\\bar U)^\\top(U-\\bar U)/(N-1)$\n"
        "$\\mathcal{L}=\\mathcal{L}_{pred}+\\mathrm{mean}_d[1-\\sqrt{C_{dd}+\\epsilon}]_+"
        "+\\sum_{i\\ne j}C_{ij}^2/D$  (unit weights)\n"
        "SIGReg: detached diagnostic, optimization weight 0 • no horizon/visibility/memory loss\n"
        "Context and clean target are synchronized views of the same unlabeled trajectory.",
        "#e8f5e9", title_size=11, body_size=6.85)

    ax.text(0.40, 7.72, "Five same-schema ORBIT modes (official contrasts all PENDING)",
            fontsize=12.5, fontweight="bold", color="#37474f")
    modes = (
        ("orbitv10 — full", "normalized two-layer rotations\ndynamic V8-style shrinkage gate\n"
         "nominated isometric observer", "#7e57c2"),
        ("no-action", "$T(a)=I$ for every action\nall action tensors retained\n"
         "tests causal action transport", "#ffe0b2"),
        ("additive", "$p=m+\\tanh(v+d\\odot LN(m))$\nsame $W_a$ tensor/parameter count\n"
         "tests isometry vs V8-like transport", "#ffecb3"),
        ("scaled", "raw complex blocks $(u,v)$\nnormalization removed; identity init\n"
         "tests exact norm preservation", "#f8bbd0"),
        ("static", "$g_t=\\sigma(b)$\nnormalized orthogonal transport retained\n"
         "tests innovation conditioning", "#fff3e0"),
    )
    for index, (title, body, color) in enumerate(modes):
        box(ax, 0.35 + index * 3.52, 5.48, 3.20, 1.92, title, body, color,
            title_size=9.2, body_size=6.5)

    box(ax, 0.35, 2.75, 8.35, 2.20, "capacity and initialization contract",
        "$2D^2+2AD+2D+2=34{,}562$ trainable memory parameters at $D=128,A=6$\n"
        "$D=128$ recurrent floats (half V8); $O(D^2+AD)$ work per streaming step\n"
        "$W_x=I$; $W_a=W_o=w_z=w_e=0$; $\\rho=.5$; $b=-2$\n"
        "The gate bias is initialization only. There is no decay, pole, bank route, or fixed "
        "memory horizon.",
        "#eceff1", title_size=10.5, body_size=7.2)
    box(ax, 9.05, 2.75, 8.55, 2.20, "measured V10-J host receipts (excluded, 100 epochs)",
        "Walker: ceiling .8247; held-out 1.0387; late -7.53% • Pendulum: .9246 / 1.3097 / +.12%\n"
        "Cartpole: .1187 / .7704 / -3.63% • Fish: 1.0233 / 1.2181 / -3.02%\n"
        "Hopper: .7323 / 1.0969 / -5.46% • only Cartpole held-out <1\n"
        "variance .164–.401 • rank 36.86–78.91 • singleton/prefix <2e-6\n"
        "orthogonality <=2.98e-7 • streaming <=1.43e-6 • no matched baseline/control cell",
        "#f3e5f5", title_size=10.5, body_size=6.45)

    box(ax, 0.35, 0.55, 17.25, 1.62, "adaptive audit boundary and no-launch decision",
        "Normalization-only hosts failed: none -> point collapse; per-frame LN -> fixed vector; "
        "causal BN -> rank 1; group whitening -> train/eval mismatch. EMA+VICReg also remained "
        "weak/nonconverged.\n"
        "V10-J fixes representation collapse, not memory efficacy. Fish is not state-decodable "
        "and Walker/Hopper violate the 5% late-change ceiling, so the 225-cell grid was not launched.\n"
        "Official count 0/225 • five mechanism modes and all baselines PENDING • no executed return "
        "• no ICLR superiority claim.",
        "#ffebee", title_size=10.8, body_size=7.35, edge="#b71c1c", linewidth=1.8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
