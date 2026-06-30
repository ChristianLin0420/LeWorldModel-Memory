#!/usr/bin/env python3
"""Render the prospective CF-EBO-v14 architecture and frozen screen contract."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures" / "fig_cf_ebo_v14_arch.png"

ONLINE = "#1565c0"
FIT = "#2e7d32"
CONTROL = "#ef6c00"
AUDIT = "#c62828"
INK = "#263238"


def box(ax, x, y, w, h, title, body, color, *, title_size=9.0, body_size=6.25,
        edge=INK, linewidth=1.3):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=color, edgecolor=edge, linewidth=linewidth)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - .30, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h / 2 - .10, body, ha="center", va="center",
            fontsize=body_size, color=INK, linespacing=1.30)


def arrow(ax, start, end, label="", color="#546e7a"):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        color=color, linewidth=1.55)
    ax.add_patch(patch)
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .14,
                label, ha="center", fontsize=6.8, color=color)


def tag(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=9.5, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=.24", facecolor="#ffffff",
                      edgecolor=color, linewidth=1.1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(20, 22), dpi=180)
    fig.patch.set_facecolor("#fbfcfd")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 22)
    ax.axis("off")

    ax.text(10, 21.56,
            "CF-EBO-v14: cross-fold-calibrated energy-bounded predictive observer",
            ha="center", fontsize=18, fontweight="bold", color=INK)
    ax.text(10, 21.18,
            "PROSPECTIVE / READY_NOT_RUN — frozen 40-cell screen; conditional 100-epoch manifest only (0/96)",
            ha="center", fontsize=10.3, fontweight="bold", color=AUDIT)

    tag(ax, .35, 20.72, "DETACHED TRAIN-ONLY COORDINATE FIT", FIT)
    box(ax, .35, 18.62, 2.70, 1.62, "paired active-encoder views",
        r"$y_t=E_\theta(o_t^{clean})$" "\n"
        r"$z_t=E_\theta(o_t^{observed})$" "\nexecuted IID actions; FP64 CPU fit",
        "#e3f2fd")
    box(ax, 3.38, 18.62, 3.42, 1.62, "pooled V13 normal coordinate",
        "even/odd all-lag action moments\npositive-part coefficient agreement\n"
        r"full-order Ho–Kalman $A,C$; normal stable blocks",
        "#dcedc8", body_size=6.05)
    box(ax, 7.13, 18.62, 3.52, 1.62, "infinite observable-energy map",
        r"$W=A^\top W A+C^\top C$" "\n"
        r"$h=W^{1/2}x:\quad F^\top F+H^\top H=P_{obs}$" "\n"
        "machine-observable support; inactive modes padded to zero\nfixed V13 source-order state schema",
        "#c8e6c9", body_size=6.15)
    box(ax, 10.98, 18.62, 4.05, 1.62, "rank-aware direct sum",
        r"$x_0=H^\dagger(y_0-\mu_y)$" "\n"
        r"$c_\perp=P_\perp(y_0-\mu_y)$" "\n"
        r"$P_\perp=0$ exactly when $H$ spans output space",
        "#b2dfdb", body_size=6.15)
    box(ax, 15.36, 18.62, 4.29, 1.62, "alternating LeWM fit",
        "fit before epoch 1 and after every epoch\nall operators detached; zero memory parameters\n"
        "encoder/predictor still receive end-to-end gradients",
        "#b2ebf2", body_size=6.1)
    for x0, x1 in ((3.05, 3.38), (6.80, 7.13), (10.65, 10.98), (15.03, 15.36)):
        arrow(ax, (x0, 19.43), (x1, 19.43), color=FIT)

    tag(ax, .35, 18.14, "CROSS-FOLD PREDICTIVE-RISK ACTION TRANSPORT", FIT)
    box(ax, .35, 15.86, 4.25, 1.76, "fixed-(F,H) fold action refits",
        r"$G_e=\arg\min_G\sum_j\|M_j^e-HF^jG\|_F^2$" "\n"
        r"$G_o$ analogously; $G_0$ uses pooled moments" "\n"
        "V13 coordinate is preserved; only transport is changed",
        "#e8f5e9", body_size=6.05)
    box(ax, 4.95, 15.86, 5.10, 1.76, "directional opposite-fold episode risk",
        r"$D=L_{no\ action}-L_{G_f}$ on the opposite fold" "\n"
        r"$\rho_f=1[m>0]\,[m^2-\mathrm{Var}(D)/N]_+/(m^2+\epsilon_{mach}s^2)$" "\n"
        "recursive open-loop clean prediction; no selected horizon",
        "#fff8e1", body_size=5.85)
    box(ax, 10.40, 15.86, 4.25, 1.76, "conservative symmetric support",
        r"$\alpha_B=\min(\rho_{e\to o},\rho_{o\to e})$" "\n"
        r"$G=\alpha_B G_0$" "\n"
        "unsupported action transport collapses continuously to zero",
        "#ffe0b2", body_size=6.15)
    box(ax, 15.00, 15.86, 4.65, 1.76, "claim boundary",
        "opposite-fold scoring calibrates risk shrinkage\n"
        "the pooled coordinate/final refit still use all train episodes\n"
        "therefore cross-fold calibrated, not a fully cross-fitted estimator",
        "#ffebee", edge=AUDIT, body_size=6.0)
    arrow(ax, (4.60, 16.74), (4.95, 16.74), color=FIT)
    arrow(ax, (10.05, 16.74), (10.40, 16.74), color=FIT)
    arrow(ax, (14.65, 16.74), (15.00, 16.74), color=FIT)

    tag(ax, .35, 15.38, "PAIRED ROBUST CORRECTION FIT", FIT)
    box(ax, .35, 12.90, 4.05, 1.96, "open-loop state error + innovation",
        "reconstruct clean predictive states from all futures\n"
        r"$e_t=x_t^{clean}-x_t^-$" "\n"
        r"$\nu_t=z_t^{observed}-(\mu_y+c_\perp+Hx_t^-)$",
        "#e8f5e9", body_size=6.15)
    box(ax, 4.75, 12.90, 4.30, 1.96, "OAS whitening",
        r"$\Sigma_\nu=\mathrm{OAS}(\nu),\quad J=\Sigma_\nu^{-1/2}$" "\n"
        r"$u=J\nu,\quad q=\|u\|^2$" "\n"
        "symmetric eigensolve; machine innovation rank d\nno corruption label enters the fit",
        "#d1c4e9", body_size=6.1)
    box(ax, 9.40, 12.90, 4.45, 1.96, "energy-coordinate correction map",
        r"$M_0=E[e u^\top]E[uu^\top]^\dagger$" "\n"
        r"$M_0=U\,\mathrm{diag}(s_i)V^\top$" "\n"
        r"$M=U\,\mathrm{diag}(\min(s_i,1))V^\top$" "\n"
        r"$\|M\|_2\leq1$ bounds observable future energy",
        "#fff3e0", body_size=6.0)
    box(ax, 14.20, 12.90, 5.45, 1.96, "cross-fold recursive correction risk",
        "fit a correction on one episode fold; recursively deploy on the other\n"
        r"$\alpha_K=\min(\rho^K_{e\to o},\rho^K_{o\to e})$" "\n"
        r"$\alpha_K=0$ falls back exactly to open loop; no DARE/Kalman gain" "\n"
        "same positive-part empirical-Bayes risk rule as action transport",
        "#ffccbc", body_size=5.9)
    for x0, x1 in ((4.40, 4.75), (9.05, 9.40), (13.85, 14.20)):
        arrow(ax, (x0, 13.88), (x1, 13.88), color=FIT)

    tag(ax, .35, 12.42, "ONLINE CAUSAL ENERGY-BOUNDED OBSERVER", ONLINE)
    box(ax, .35, 9.84, 4.20, 2.05, "action prior",
        r"$x_t^-=F x_{t-1}^+ + \alpha_B G_0(a_{t-1}-\mu_a)$" "\n"
        r"$h_t^-=\mu_y+c_\perp+Hx_t^-$" "\n"
        r"$F^\top F+H^\top H=P_{obs}$" "\n"
        "active-support energy is total future read energy\ninactive fixed-schema coordinates remain zero",
        "#e1f5fe", body_size=6.2)
    box(ax, 4.90, 9.84, 4.35, 2.05, "innovation evidence",
        r"$\nu_t=z_t-h_t^-,\quad u_t=J\nu_t$" "\n"
        r"$q_t=\|u_t\|^2,\quad d=\mathrm{rank}(\Sigma_\nu)$" "\n"
        r"$g_t=\min(1,d/\max(q_t,\epsilon_{tiny}))$" "\n"
        "large/off-fit innovations are radially redescended",
        "#d1c4e9", body_size=6.15)
    box(ax, 9.60, 9.84, 4.40, 2.05, "bounded correction",
        r"$\delta_t=\alpha_K M(g_tu_t)$" "\n"
        r"$x_t^+=x_t^-+\delta_t$" "\n"
        r"$\|\delta_t\|^2\leq\alpha_K^2d$ in full capped/radial mode" "\n"
        r"$\sum_{j\geq0}\|HF^j\delta_t\|^2=\|P_{obs}\delta_t\|^2$",
        "#fff3e0", body_size=6.15)
    box(ax, 14.35, 9.84, 5.30, 2.05, "direct LeWM predictor coordinate",
        r"$h_t=\mu_y+c_\perp+Hx_t^+$" "\n"
        r"$\hat y_{t+1}=P_\phi(h_t,a_t)$" "\n"
        r"$L=\|\hat y_{t+1}-E_\theta^{eval}(o_{t+1}^{clean})\|^2+L_{var}+L_{cov}$" "\n"
        "no bypass, suffix/rank loss, teacher, or memory loss",
        "#c8e6c9", body_size=5.95)
    for x0, x1 in ((4.55, 4.90), (9.25, 9.60), (14.00, 14.35)):
        arrow(ax, (x0, 10.87), (x1, 10.87), color=ONLINE)

    tag(ax, .35, 9.34, "SIX CANDIDATE MODES — ONE FIXED STATE/API SCHEMA", CONTROL)
    controls = (
        ("full", r"$\alpha_B,\alpha_K$ risk" "\ncap + radial gate", "#a5d6a7"),
        ("nocorrect", r"$\alpha_K=0$ exactly" "\nopen-loop control", "#ffe0b2"),
        ("noaction", r"$G_{eff}=0$ exactly" "\ncorrection retained", "#ffccbc"),
        ("norisk", r"$\alpha_B=\alpha_K=1$" "\ncap + radial retained", "#e1bee7"),
        ("noenergycap", r"$M=M_0$" "\nradial retained", "#f8bbd0"),
        ("noradial", r"$g_t=1$" "\nrisk + cap retained", "#b3e5fc"),
    )
    for index, (title, body, color) in enumerate(controls):
        box(ax, .35 + index * 3.23, 7.52, 2.88, 1.35, title, body, color,
            title_size=8.6, body_size=6.15)

    tag(ax, .35, 7.03, "FOUR FRESH BASELINES", CONTROL)
    baselines = (
        ("V13 nocorrect", "best V13 mode\nfixed normal open loop"),
        ("SSM", "fresh learned\ndiagonal recurrence"),
        ("compact V8", "fresh shared-action\ntwo-state filter"),
        ("KDIO-v11", "fresh raw-difference\nkick–drift observer"),
    )
    for index, (title, body) in enumerate(baselines):
        box(ax, .35 + index * 4.83, 5.47, 4.48, 1.10, title, body,
            "#eceff1", title_size=8.5, body_size=6.0)

    tag(ax, .35, 4.98, "FROZEN EVIDENCE CONTRACT", AUDIT)
    box(ax, .35, 2.53, 5.65, 1.95, "40-cell adaptive screen",
        "10 designs × 4 tasks × seed 14001 × exactly 30 epochs\n"
        "task-pinned GPUs Cart/Fish/Pend/Walker → 0/1/2/3\n"
        "online W&B + checkpoint + metrics + hashed rollout per cell\n"
        "clean/pushed HEAD, source/data/command hashes; no overwrite",
        "#e8eaf6", body_size=6.05)
    box(ax, 6.35, 2.53, 7.35, 1.95, "conjunctive continuation gates",
        "40/40 artifact integrity • full representation/causality • all-mode numerical exactness\n"
        "fixed source-order schema + machine-observable support/projector receipts; no selected rank threshold\n"
        "full beats V13 nocorrect, SSM, compact V8, KDIO, legal integrator, and direct controls\n"
        "positive action/correction risk support • Gaussian radial suppression and energy bound\n"
        "condition telemetry separates clean / val_train_view / held-out corruptions • registered convergence ceilings",
        "#fff8e1", edge=AUDIT, body_size=5.85)
    box(ax, 14.05, 2.53, 5.60, 1.95, "conditional continuation: manifest only",
        "8 retained designs × 4 tasks × seeds 14002–14004\n"
        "× 100 epochs = 96 prospective cells\n"
        "runner writes commands with status NOT_AUTHORIZED\n"
        "no automatic launch; independent analyzer/auditor required",
        "#ffebee", edge=AUDIT, body_size=6.0)
    arrow(ax, (6.00, 3.50), (6.35, 3.50), color=AUDIT)
    arrow(ax, (13.70, 3.50), (14.05, 3.50), color=AUDIT)

    box(ax, .35, .43, 19.30, 1.58, "prospective claim boundary",
        "CF-EBO is reward-free and state-label-free, but action-observed and paired-view supervised. "
        "Its novelty is a composition of opposite-fold empirical-Bayes risk calibration, observable-energy coordinates, "
        "OAS whitening, spectral correction capping, radial innovation suppression, and rank-aware direct-sum anchoring.\n"
        "It is not hyperparameter-free, fully cross-fitted, or yet an ICLR performance result. The architecture remains unvalidated until the frozen screen completes.",
        "#ffebee", edge=AUDIT, title_size=9.4, body_size=6.45)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
