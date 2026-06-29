#!/usr/bin/env python3
"""Generate the completed HACSSM-v9 / LOIF architecture and result contract."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "figures" / "fig_hacssm_v9_proposal.png"


def box(ax, xy, width, height, text, color, *, size=8.0, edge="#263238",
        linewidth=1.45):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.025,rounding_size=0.035",
        facecolor=color, edgecolor=edge, linewidth=linewidth, zorder=2)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center",
            va="center", fontsize=size, color="#17202a", linespacing=1.22,
            zorder=3)


def arrow(ax, start, end, *, color="#455a64", dashed=False, connection="arc3"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.35,
        color=color, linestyle="--" if dashed else "-",
        connectionstyle=connection, zorder=1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(18, 11.4), dpi=180)
    fig.patch.set_facecolor("#fbfcfd")
    ax.set_facecolor("#fbfcfd")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11.4)
    ax.axis("off")

    ax.text(9, 11.02,
            "HACSSM-v9 / LOIF: Learned Ordered Innovation Filter",
            ha="center", fontsize=18.0, fontweight="bold", color="#17202a")
    ax.text(9, 10.65,
            "325 cells complete • immutable pilot NO_GO • final PILOT_NO_GO_FINAL_DESCRIPTIVE",
            ha="center", fontsize=10.0, color="#7b1fa2")

    ax.text(0.32, 10.12, "CAUSAL FILTER", fontsize=11.2,
            fontweight="bold", color="#37474f")
    box(ax, (0.35, 8.55), 1.16, 0.68, r"$z_t$", "#e3f2fd", size=11)
    box(ax, (0.35, 7.47), 1.16, 0.68, r"$a_{t-1}$", "#fff3e0", size=10.5)
    box(ax, (1.80, 8.55), 1.55, 0.68,
        r"$x_t=RMSNorm(W_xz_t)$", "#e8f5e9", size=7.7)
    box(ax, (1.80, 7.47), 2.05, 0.68,
        "one physical action map\n" + r"$(d,v)=W_a a_{t-1}$",
        "#ffecb3", size=8.3, edge="#ef6c00", linewidth=1.75)
    arrow(ax, (1.51, 8.89), (1.80, 8.89), color="#1976d2")
    arrow(ax, (1.51, 7.81), (1.80, 7.81), color="#ef6c00")

    box(ax, (4.18, 8.13), 2.45, 1.48,
        "learned ordered poles\n"
        + r"$0<\alpha_f<\alpha_s<1$" + "\n"
        + r"$q_k=1-\alpha_k^2$" + "\n"
        + "matched tau-init; poles then learned",
        "#dcedc8", size=8.2, edge="#558b2f", linewidth=1.7)

    levels = (
        ("FAST", "f", 8.48, "#c5e1a5", "#c5cae9"),
        ("SLOW", "s", 6.65, "#b2dfdb", "#d1c4e9"),
    )
    for name, symbol, y, prior_color, post_color in levels:
        box(ax, (6.98, y), 2.52, 1.18,
            f"{name} stable prior\n"
            + rf"$p_t^{symbol}=\alpha_{symbol}m_{{t-1}}^{symbol}$" + "\n"
            + rf"$+\sqrt{{q_{symbol}}}\tanh(v+d\odot LN(m_{{t-1}}^{symbol}))$" + "\n"
            + rf"$P_t^{{{symbol},-}}=\alpha_{symbol}^2P_{{t-1}}^{symbol}+q_{symbol}$",
            prior_color, size=7.25)
        box(ax, (12.15, y), 2.32, 1.18,
            f"{name} posterior\n"
            + rf"$K_t^{symbol}=P_t^{{{symbol},-}}/(P_t^{{{symbol},-}}+R_t)$" + "\n"
            + rf"$m_t^{symbol}=p_t^{symbol}+K_t^{symbol}(x_t-p_t^{symbol})$" + "\n"
            + rf"$P_t^{symbol}=P_t^{{{symbol},-}}R_t/(P_t^{{{symbol},-}}+R_t)$",
            post_color, size=6.85)
        arrow(ax, (6.63, 8.88), (6.98, y + 0.78), color="#558b2f")
        arrow(ax, (3.85, 7.81), (6.98, y + 0.28), color="#ef6c00",
              connection="arc3,rad=-0.08" if symbol == "f" else "arc3,rad=0.08")
        arrow(ax, (9.50, y + 0.59), (12.15, y + 0.59),
              connection="arc3,rad=-0.24" if symbol == "f" else "arc3,rad=0.24")
        arrow(ax, (3.35, 8.80), (12.15, y + 0.35), color="#1976d2",
              dashed=True,
              connection="arc3,rad=-0.10" if symbol == "f" else "arc3,rad=0.10")
        arrow(ax, (13.25, y), (8.20, y), color="#5e35b1", dashed=True,
              connection="arc3,rad=0.34" if symbol == "f" else "arc3,rad=-0.16")

    box(ax, (9.76, 6.02), 2.12, 0.82,
        "inverse-scale prior mixture\n"
        + r"$\omega_t^-=softmax(-\log P_t^-)$" + ";  "
        + r"$e_t=x_t-\sum_k\omega_t^{k,-}p_t^k$",
        "#e0f2f1", size=6.45, edge="#00897b", linewidth=1.45)
    box(ax, (9.78, 7.42), 2.08, 1.88,
        "shared causal evidence\n"
        + r"direct: $w_z^TLN(z_t)$" + "\n"
        + r"innovation: $w_e^Te_t$" + "\n"
        + r"$\ell_t=(w_z^TLN(z_t)+w_e^Te_t)/\sqrt{D}$" + "\n"
        + r"$R_t=softplus(b_R+\ell_t)+\epsilon$",
        "#f8bbd0", size=6.65, edge="#ad1457", linewidth=1.75)
    arrow(ax, (0.93, 9.23), (10.70, 9.30), color="#ad1457", dashed=True,
          connection="arc3,rad=-0.16")
    ax.text(7.12, 10.04, r"direct $LN(z_t)$ evidence", ha="center",
            fontsize=6.7, color="#ad1457")
    arrow(ax, (3.35, 8.62), (9.76, 6.42), color="#1976d2",
          connection="arc3,rad=0.25")
    arrow(ax, (9.14, 8.48), (10.12, 6.84),
          connection="arc3,rad=0.24")
    arrow(ax, (9.16, 6.65), (10.64, 6.84),
          connection="arc3,rad=-0.10")
    arrow(ax, (10.82, 6.84), (10.82, 7.42), color="#00897b")
    arrow(ax, (11.86, 8.48), (12.15, 9.06), color="#ad1457")
    arrow(ax, (11.86, 8.12), (12.15, 7.24), color="#ad1457")

    box(ax, (14.82, 7.55), 1.48, 1.72,
        "inverse-scale read\n"
        + r"$\pi_t=softmax(-\log P_t)$" + "\n"
        + r"$r_t=RMSNorm($" + "\n"
        + r"$\sum_k\pi_t^km_t^k)$",
        "#b3e5fc", size=7.7)
    box(ax, (16.64, 7.55), 1.02, 1.72,
        "residual\n" + r"$\tilde z_t=$" + "\n" + r"$z_t+W_or_t$",
        "#c8e6c9", size=8.0)
    for _, _, y, _, _ in levels:
        arrow(ax, (14.47, y + 0.59), (14.82, 8.41), color="#5e35b1")
    arrow(ax, (16.30, 8.41), (16.64, 8.41))
    arrow(ax, (0.93, 9.23), (17.15, 9.27), color="#1976d2", dashed=True,
          connection="arc3,rad=-0.09")
    ax.text(13.05, 6.24, r"dashed loops: $(m_t^k,P_t^k)\rightarrow$ next step",
            ha="center", fontsize=6.8, color="#5e35b1")

    ax.text(9, 5.70, "SELF-SUPERVISED CONTRACT", fontsize=11.2,
            fontweight="bold", ha="center", color="#37474f")
    box(ax, (0.52, 4.18), 5.30, 1.18,
        "effective trainable objective: unweighted visible next-latent MSE\n"
        "SIGReg logged but constant/no-gradient with fixed features\n"
        "all SSM/V7/V8 references retrained identically",
        "#e8f5e9", size=8.25, edge="#2e7d32", linewidth=1.7)
    box(ax, (6.35, 4.18), 5.30, 1.18,
        "no tau grid • matched tau-init; no fixed retention thereafter • no rho\n"
        "no route logits • no teacher/momentum • no auxiliary weight\n"
        "nominal direct old-state coefficient: alpha_k (1-K_t^k)",
        "#f3e5f5", size=8.15, edge="#6a1b9a", linewidth=1.7)
    box(ax, (12.18, 4.18), 5.30, 1.18,
        "constrained free mechanisms, not identifiability\nordered poles; q_k=1-alpha_k^2; one shared R_t\n"
        "operational scales—not calibrated variances",
        "#e8eaf6", size=8.1, edge="#3949ab", linewidth=1.7)

    ax.text(0.35, 3.84, "COMPLETED FALSIFICATION MAP", fontsize=11.2,
            fontweight="bold", color="#37474f")
    variants = (
        ("loifv9", "full filter: +2.551% vs SSM\nrank 7/13; locked NO_GO", "#9575cd"),
        ("fixedalpha", r"$\alpha=(e^{-1/2},e^{-1/8})$" + "\nfull delta = -0.671%", "#fff3e0"),
        ("globalR", r"$R_t=softplus(b_R)+\epsilon$" + "\nfull delta = +30.556%", "#f8bbd0"),
        ("innovationonly", "disconnect direct latent\nfull delta = +19.189%", "#e1f5fe"),
        ("latentonly", "disconnect innovation\nfull delta = +0.475%", "#fce4ec"),
        ("uniformfusion", "uniform prior + output\nfull delta = +0.340%", "#b3e5fc"),
        ("loifv9_noaction", "zero action innovation\nfull delta = +8.293%", "#ffe0b2"),
        ("singlebank", "fast state disconnected\nfull delta = -2.220%", "#b2ebf2"),
    )
    for index, (title, body, color) in enumerate(variants):
        x = 0.22 + index * 2.20
        box(ax, (x, 2.40), 1.98, 1.22, title + "\n" + body, color,
            size=5.35, linewidth=1.3)

    ax.text(9, 1.94,
            "325/325 cells • 65,000 W&B epochs • 325 rollout bundles • manifest e87b560a… • "
            "34,563 memory parameters • 258 streaming floats.",
            ha="center", fontsize=8.7, color="#455a64")
    ax.text(9, 1.53,
            "Full LOIF: +2.551% vs SSM, -4.149% vs compact V8, -4.994% vs the endpoint "
            "envelope; compact V8 leads the retrained grid at +6.377%.",
            ha="center", fontsize=8.45, color="#455a64")
    ax.text(9, 1.11,
            "Delta is paired full-LOIF reduction relative to each control: evidence/action paths "
            "are active, but fixed alpha and one slow bank beat the learned hierarchy.",
            ha="center", fontsize=8.45, color="#455a64")
    ax.text(9, 0.58,
            "V9 is negative adaptive-development evidence—not an overall-best/ICLR claim. "
            "Any confirmation requires unseen corruptions/seeds, state outcomes, and return.",
            ha="center", fontsize=8.9, fontweight="bold", color="#6a1b9a")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
