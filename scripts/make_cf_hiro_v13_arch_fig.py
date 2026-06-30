#!/usr/bin/env python3
"""Render the prospective CF-HIRO-v13 architecture and frozen screen contract."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures" / "fig_cf_hiro_v13_arch.png"

ONLINE = "#1565c0"
FIT = "#2e7d32"
CONTROL = "#ef6c00"
AUDIT = "#c62828"
INK = "#263238"


def box(ax, x, y, w, h, title, body, color, *, title_size=9.2, body_size=6.5,
        edge=INK, linewidth=1.35):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=color, edgecolor=edge, linewidth=linewidth)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - .32, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h / 2 - .10, body, ha="center", va="center",
            fontsize=body_size, color=INK, linespacing=1.32)


def arrow(ax, start, end, label="", color="#546e7a"):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        color=color, linewidth=1.6)
    ax.add_patch(patch)
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + .16,
                label, ha="center", fontsize=7.0, color=color)


def tag(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=9.7, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=.24", facecolor="#ffffff",
                      edgecolor=color, linewidth=1.1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(20, 20.5), dpi=180)
    fig.patch.set_facecolor("#fbfcfd")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20.5)
    ax.axis("off")

    ax.text(10, 20.08,
            "CF-HIRO-v13: split-agreement normal predictive observer",
            ha="center", fontsize=18, fontweight="bold", color=INK)
    ax.text(10, 19.69,
            "PROSPECTIVE STATUS: READY_NOT_RUN — 0/36 screen cells, 0/72 conditional 100-epoch cells",
            ha="center", fontsize=10.3, fontweight="bold", color=AUDIT)

    tag(ax, .35, 19.22, "PAIRED TRAIN-ONLY SELF-SUPERVISION", ONLINE)
    box(ax, .35, 17.10, 2.35, 1.65, "paired RGB trajectory",
        r"$o_t^{clean},o_t^{obs}$" "\n" r"executed IID $a_t$" "\nno reward / state label",
        "#e3f2fd")
    box(ax, 3.05, 17.10, 2.40, 1.65, "shared causal encoder",
        r"$z_t=E_\theta(o_t)$" "\ndropout off for fits\nfit tensors detached FP64",
        "#dcedc8")
    box(ax, 5.80, 17.10, 3.30, 1.65, "two immutable episode folds",
        r"$M_j^{(f)}=\mathrm{Cov}_f(z_{t+j+1},a_t)\,\mathrm{Cov}_f(a_t)^\dagger$" "\n"
        r"all lags $j=0,\ldots,L-2$; source-time centered",
        "#e8f5e9", body_size=6.15)
    box(ax, 9.45, 17.10, 3.20, 1.65, "positive-part agreement",
        r"$\bar H_0=U\,\mathrm{diag}(s_i)V^\top$" "\n"
        r"$g_i=[m_i^2-d_i^2]_+/(m_i^2+\epsilon_{mach})$" "\n"
        r"$\tilde s_i=g_i s_i$; same filter on $H_1$",
        "#c8e6c9", body_size=5.9)
    box(ax, 13.00, 17.10, 3.20, 1.65, "normal modal realization",
        "balanced full-order Ho–Kalman state\nreal Schur; remove inter-block coupling\n"
        r"$1\times1$ contractions + $2\times2$ rotation-scale blocks",
        "#b2dfdb", body_size=5.9)
    box(ax, 16.55, 17.10, 3.10, 1.65, "all-lag action refit",
        r"$B=\arg\min_B\sum_j\|\tilde M_j-CA^jB\|_F^2$" "\n"
        r"fixed $A,C$; no selected horizon" "\nfull rectangular state schema retained",
        "#b2ebf2", body_size=5.9)
    for x0, x1 in ((2.70, 3.05), (5.45, 5.80), (9.10, 9.45),
                   (12.65, 13.00), (16.20, 16.55)):
        arrow(ax, (x0, 17.92), (x1, 17.92))

    tag(ax, .35, 16.55, "COVARIANCE FIT + OFFLINE RICCATI FIXED POINT", FIT)
    box(ax, .35, 14.25, 5.85, 1.80, "paired OAS process / measurement statistics",
        "reconstruct clean predictive states from all remaining clean outputs\n"
        r"$Q=\mathrm{OAS}(x_{t+1}^c-Ax_t^c-B(a_t-\mu_a))$" "\n"
        r"$R=\mathrm{OAS}(z_t^{obs}-z_t^{clean})$; pooled train fit, not cross-fitted",
        "#e8f5e9", body_size=6.45)
    box(ax, 6.60, 14.25, 5.75, 1.80, "DARE solved once per detached refit",
        r"$P_\infty^- = A P_\infty^+ A^\top+Q$" "\n"
        r"$K_\infty=P_\infty^-C^\top(CP_\infty^-C^\top+R)^{-1}$" "\n"
        "Joseph residual/PSD receipts; no online covariance recursion",
        "#fff8e1", body_size=6.35)
    box(ax, 12.75, 14.25, 6.90, 1.80, "what is and is not data-derived",
        r"data: all-lag moments, agreement weights, modal radii, $B,Q,R,K_\infty$" "\n"
        "structural choices: two folds, balanced Hankel split, full numerical order, OAS target, normal projection\n"
        "therefore not hyperparameter-free, not automatically sized, and not a fundamental-new-primitive claim",
        "#ffebee", edge=AUDIT, body_size=6.1)
    arrow(ax, (6.20, 15.15), (6.60, 15.15), color=FIT)
    arrow(ax, (12.35, 15.15), (12.75, 15.15), color=FIT)

    tag(ax, .35, 13.70, "ONLINE DIRECT-SUM OBSERVER — FIXED FITTED OPERATORS", ONLINE)
    box(ax, .35, 11.10, 4.45, 2.05, "canonical initialization",
        r"$P_C=CC^\dagger$" "\n" r"$x_0=C^\dagger(z_0-\mu_z)$" "\n"
        r"$c_\perp=(I-P_C)(z_0-\mu_z)$" "\n"
        r"$h_0=\mu_z+c_\perp+Cx_0=z_0$",
        "#e1f5fe", body_size=6.65)
    box(ax, 5.20, 11.10, 4.50, 2.05, "strict pre-observation prior",
        r"$x_t^-=Ax_{t-1}^+ + B(a_{t-1}-\mu_a)$" "\n"
        r"$h_t^-=\mu_z+c_\perp+Cx_t^-$" "\n"
        r"$c_\perp$ is copied exactly; normal blocks carry" "\n"
        "data-derived modal radii/phases",
        "#d1c4e9", body_size=6.55)
    box(ax, 10.10, 11.10, 4.45, 2.05, "fixed-gain correction",
        r"$\nu_t=z_t^{obs}-h_t^-$" "\n" r"$x_t^+=x_t^-+K_\infty\nu_t$" "\n"
        r"$h_t=\mu_z+c_\perp+Cx_t^+$" "\n"
        "streaming state: dynamic mean + complement; no $P_t$",
        "#fff3e0", body_size=6.55)
    box(ax, 14.95, 11.10, 4.70, 2.05, "direct one-token LeWM objective",
        r"$\hat z_{t+1}=P_\phi(h_t,a_t)$" "\n"
        r"$\mathcal{L}=\|\hat z_{t+1}-E_\theta^{eval}(o_{t+1}^{clean})\|^2$" "\n"
        r"$\qquad+\mathcal{L}_{var}+\mathcal{L}_{cov}$" "\n"
        "no bypass, teacher, suffix, rank, or memory loss",
        "#c8e6c9", body_size=6.35)
    for x0, x1 in ((4.80, 5.20), (9.70, 10.10), (14.55, 14.95)):
        arrow(ax, (x0, 12.12), (x1, 12.12), color=ONLINE)

    tag(ax, .35, 10.55, "SIX SAME-SCHEMA CANDIDATE MODES", CONTROL)
    controls = (
        ("full", "complement anchor\npositive-part $g$\n" r"normal $A$ + $K_\infty$", "#a5d6a7"),
        ("fullanchor", r"$c=z_0-\mu_z,\ x_0=0$" "\nisolates canonical\ndirect-sum initialization", "#b3e5fc"),
        ("triangular", "retain strict-upper\nreal-Schur coupling\nisolates normality", "#f8bbd0"),
        ("noshrink", r"$g_i=1$ exactly" "\nisolates split-fold\nagreement shrinkage", "#e1bee7"),
        ("noaction", r"$B_{eff}=0$ exactly" "\nfitted $B$ retained\naction-causal control", "#ffccbc"),
        ("nocorrect", r"$K_{eff}=0$ exactly" "\n" r"fitted $K_\infty$ retained" "\nobserver control", "#ffe0b2"),
    )
    for index, (title, body, color) in enumerate(controls):
        box(ax, .35 + index * 3.23, 8.55, 2.88, 1.55, title, body, color,
            title_size=8.8, body_size=6.2)

    tag(ax, .35, 8.03, "FROZEN FOUR-GPU SCREEN / CONDITIONAL CONTINUATION", AUDIT)
    box(ax, .35, 5.50, 6.15, 2.05, "30-epoch adaptive screen: 36 cells",
        "9 designs × 4 tasks × seed 13001; four persistent GPU workers\n"
        "six modes + fresh SSM + compact V8 + rawdiff KDIO-v11\n"
        "every cell: online W&B epoch rows + checkpoint + metrics\n"
        "+ rollout table/video/hashed NPZ; clean pushed HEAD required; current ledger 0/36",
        "#e8eaf6", body_size=6.35)
    box(ax, 6.90, 5.50, 7.00, 2.05, "conjunctive launch gates",
        "36/36 integrity • full rank 4/4 • all six numerical/streaming validity • late curves non-worsening\n"
        "≥5% vs noaction + SSM/V8/KDIO/integrator; ≥2% vs other controls; each ≥3/4 wins\n"
        r"held-fold action-predictive $R^2>0$ both directions on ≥3/4 tasks" "\n"
        "suffix advantage >0 + pair accuracy >.5 on same tasks; both direct-sum RMS >1e-8 on 4/4",
        "#fff8e1", edge=AUDIT, body_size=6.2)
    box(ax, 14.30, 5.50, 5.35, 2.05, "only if every gate passes: 72 cells",
        "100 epochs; three fresh seeds 13002–13004\n"
        "retained: full, noaction, noshrink,\nSSM, compact V8, KDIO-v11\n"
        "6 × 4 × 3 = 72; current ledger 0/72",
        "#ffebee", edge=AUDIT, body_size=6.35)
    arrow(ax, (6.50, 6.52), (6.90, 6.52), color=AUDIT)
    arrow(ax, (13.90, 6.52), (14.30, 6.52), color=AUDIT)

    box(ax, .35, 2.70, 19.30, 2.25,
        "implementation preflight only — not scientific evidence",
        "Four full-size D=128 / 6-layer-encoder / 4-layer-predictor, one-epoch, no-W&B smokes "
        "completed on four GPUs with two fits, checkpoints, metrics, rollouts, and streaming error 0.  "
        "Held-out values (Cart 1.230296, Fish 1.017661, Pendulum .957932, Walker .885543) are smoke telemetry, not results.\n"
        "Two earlier attempts trained/refit successfully but exposed diagnostics-only AMP dtype boundaries: "
        "FP32 actions × BF16 action map in suffix transition, then the same mismatch in action-effect RMS. "
        "Both paths now cast explicitly to operator dtype and have regression coverage. No V13 W&B run exists yet.",
        "#eceff1", edge="#607d8b", title_size=9.6, body_size=6.65)

    box(ax, .35, .48, 19.30, 1.70,
        "claim boundary before launch",
        "CF-HIRO is reward-free and state-label-free but action-observed, using paired clean/corrupted views. "
        "Its folds estimate agreement; pooled realization/covariance fits are not cross-fitted.\n"
        "Ho–Kalman/ERA, DMDc, predictive-state representations, OAS, DARE, and Kalman correction are established.\n"
        "READY_NOT_RUN means architecture/protocol readiness only—no superiority, hierarchy, calibration, or ICLR novelty evidence.",
        "#ffebee", edge=AUDIT, title_size=9.5, body_size=6.7)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
