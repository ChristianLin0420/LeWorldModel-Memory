#!/usr/bin/env python3
"""Generate the frozen prelaunch HACSSM-v8 / SAS-PC architecture map."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "figures" / "fig_hacssm_v8_arch.png"


def box(ax, xy, width, height, text, color, *, size=8.4, edge="#263238",
        linewidth=1.5):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.025,rounding_size=0.035",
        facecolor=color, edgecolor=edge, linewidth=linewidth, zorder=2)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center",
            va="center", fontsize=size, color="#17202a", linespacing=1.24,
            zorder=3)


def arrow(ax, start, end, *, color="#455a64", dashed=False, connection="arc3"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.4,
        color=color, linestyle="--" if dashed else "-",
        connectionstyle=connection, zorder=1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(18, 11.2), dpi=180)
    fig.patch.set_facecolor("#fbfcfd")
    ax.set_facecolor("#fbfcfd")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11.2)
    ax.axis("off")

    ax.text(9, 10.86, "HACSSM-v8 / SAS-PC: Shared-Action Shrinkage Predict/Correct",
            ha="center", fontsize=18.0, fontweight="bold", color="#17202a")
    ax.text(9, 10.51,
            "Frozen adaptive-development design; no V8 metric inspected before launch",
            ha="center", fontsize=10.0, color="#546e7a")

    ax.text(0.35, 10.02, "ONLINE INFERENCE", fontsize=11.2,
            fontweight="bold", color="#37474f")
    box(ax, (0.35, 8.39), 1.20, 0.70, r"$z_t$", "#e3f2fd", size=11)
    box(ax, (0.35, 7.25), 1.20, 0.70, r"$a_{t-1}$", "#fff3e0", size=10.6)
    box(ax, (1.83, 8.39), 1.52, 0.70, r"$x_t=W_xz_t$", "#e8f5e9", size=9.4)
    box(ax, (1.83, 7.25), 2.10, 0.70,
        "one physical action head\n" + r"$(d,v)=W_a a_{t-1}$",
        "#ffecb3", size=8.6, linewidth=1.8, edge="#ef6c00")
    arrow(ax, (1.55, 8.74), (1.83, 8.74), color="#1976d2")
    arrow(ax, (1.55, 7.60), (1.83, 7.60), color="#ef6c00")

    levels = (
        ("FAST", "f", r"$\tau_f=2$", 8.55, "#dcedc8", "#c5cae9"),
        ("MEDIUM", "m", r"$\tau_m=8$", 6.86, "#b2dfdb", "#d1c4e9"),
    )
    for name, symbol, tau, y, prior_color, posterior_color in levels:
        box(ax, (4.30, y), 2.50, 1.08,
            f"{name} prior  {tau}\n"
            + rf"$p_t^{symbol}=m_{{t-1}}^{symbol}+\beta_{symbol}\tanh($" + "\n"
            + rf"$v+d\odot LN(m_{{t-1}}^{symbol}))$",
            prior_color, size=7.6)
        box(ax, (7.20, y), 2.25, 1.08,
            rf"static $s_{symbol}=\sigma(b_{symbol})$" + "\n"
            + rf"dynamic $q_t^{symbol}=\sigma($" + "\n"
            + rf"$b_{symbol}+innovation_t^{symbol})$",
            "#fff0b3", size=7.8)
        box(ax, (9.85, y), 2.10, 1.08,
            rf"$\rho_{symbol}=\sigma(c_{symbol})$" + "\n"
            + rf"$g_t^{symbol}=(1-\rho_{symbol})s_{symbol}$" + "\n"
            + rf"$+\rho_{symbol}q_t^{symbol}$",
            "#f8bbd0", size=7.8)
        box(ax, (12.35, y), 2.38, 1.08,
            rf"posterior $m_t^{symbol}$" + "\n"
            + rf"$p_t^{symbol}+\beta_{symbol}g_t^{symbol}$" + "\n"
            + rf"$(x_t-p_t^{symbol})$",
            posterior_color, size=7.8)
        arrow(ax, (3.93, 7.60), (4.30, y + 0.35), color="#ef6c00")
        arrow(ax, (3.35, 8.74), (7.20, y + 0.72), color="#1976d2")
        arrow(ax, (6.80, y + 0.54), (7.20, y + 0.54))
        arrow(ax, (9.45, y + 0.54), (9.85, y + 0.54))
        arrow(ax, (11.95, y + 0.54), (12.35, y + 0.54))

    box(ax, (15.05, 7.39), 1.34, 1.52,
        "joint read\n" + r"$r_t=RMSNorm($" + "\n" + r"$\pi_fm_t^f+\pi_mm_t^m)$",
        "#b3e5fc", size=8.0)
    box(ax, (16.68, 7.39), 1.02, 1.52,
        "residual\n" + r"$\tilde z_t=$" + "\n" + r"$z_t+W_or_t$",
        "#c8e6c9", size=8.2)
    for _, _, _, y, _, _ in levels:
        arrow(ax, (14.73, y + 0.54), (15.05, 8.15), color="#5e35b1")
    arrow(ax, (16.39, 8.15), (16.68, 8.15))
    arrow(ax, (0.96, 9.09), (17.18, 9.25), color="#1976d2", dashed=True,
          connection="arc3,rad=-0.055")

    ax.text(9, 6.35, "TRAINING CONTRACT", fontsize=11.2, fontweight="bold",
            ha="center", color="#37474f")
    box(ax, (0.55, 4.69), 5.15, 1.25,
        "ordinary self-supervised next-latent prediction\n"
        + r"$\mathcal{L}=\mathcal{L}_{pred}+\lambda_{sig}\mathcal{L}_{SIGReg}$" + "\n"
        + "visible/all-target + first-post balance unchanged",
        "#e8f5e9", size=8.4, edge="#2e7d32", linewidth=1.7)
    box(ax, (6.42, 4.69), 5.15, 1.25,
        "no EMA teacher • no posterior-state matching\n"
        "no synthetic recovery • no internal auxiliary\n"
        "configured and effective hierarchy weight exactly zero",
        "#f3e5f5", size=8.4, edge="#6a1b9a", linewidth=1.7)
    box(ax, (12.30, 4.69), 5.15, 1.25,
        r"$W_x=I,\ W_o=0,\ W_a=0$" + "\n"
        + r"$b_f=b_m=2,\ c_f=c_m=0\Rightarrow\rho=.5$" + "\n"
        + "uniform route; train from scratch",
        "#e8eaf6", size=8.4, edge="#3949ab", linewidth=1.7)

    ax.text(0.35, 4.25, "FROZEN V8 VARIANT MAP", fontsize=11.2,
            fontweight="bold", color="#37474f")
    variants = (
        ("hacssmv8", "physical shared head\nlearned shrinkage + joint read", "#b39ddb"),
        ("hacssmv8_dynamic", r"$\rho_f=\rho_m=1$" + "\nretrained endpoint", "#f8bbd0"),
        ("hacssmv8_static", r"$\rho_f=\rho_m=0$" + "\nretrained endpoint", "#fff3e0"),
        ("hacssmv8_levelaction", "separate level heads\n36,102-param control", "#d1c4e9"),
        ("hacssmv8_redundant", "equal head halves; averaged\nstatistical equivalence control", "#ffecb3"),
        ("hacssmv8_noaction", "shared action contribution 0\nstructural receipt", "#ffe0b2"),
        ("hacssmv8_single", "medium-only read\nboth states retained", "#b2ebf2"),
    )
    for index, (title, body, color) in enumerate(variants):
        x = 0.28 + index * 2.53
        box(ax, (x, 2.54), 2.28, 1.25, title + "\n" + body, color,
            size=6.25, linewidth=1.3)

    ax.text(9, 2.05,
            "Compact shared modes: 34,566 trainable memory parameters; "
            "level-action/redundant: 36,102; recurrent state: 2D floats.",
            ha="center", fontsize=8.9, color="#455a64")
    ax.text(9, 1.71,
            "Official grid: 5 environments × (6 historical anchors + 7 V8 modes) × 5 seeds "
            "= 325 cells; seeds 0–2 pilot = 195, seeds 3–4 mandatory completion = 130.",
            ha="center", fontsize=8.45, color="#455a64")
    ax.text(9, 1.36,
            "Anchors: SSM, full/static V6, and V7 no-aux/shared-action/no-recovery; "
            "every cell logs 200 W&B epochs plus rollout NPZ/table/video/artifact.",
            ha="center", fontsize=8.25, color="#455a64")
    ax.text(9, 1.00,
            "Primary contrasts: retrained shrinkage endpoints • redundant↔level-action tying • "
            "compact↔redundant statistical equivalence • no-action • single-read.",
            ha="center", fontsize=8.35, color="#455a64")
    ax.text(9, 0.55,
            "Adaptive-development protocol only: the V1–V7 black-sentinel tasks selected V8; "
            "no untouched or ICLR confirmation claim is permitted.",
            ha="center", fontsize=9.0, fontweight="bold", color="#6a1b9a")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
