#!/usr/bin/env python3
"""Render the completed SIRO-v12 architecture, controls, and negative evidence."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "figures" / "fig_siro_v12_arch.png"

INK = "#17202a"
TEXT = "#263238"
MUTED = "#607d8b"
EDGE = "#37474f"
ONLINE = "#1565c0"
FIT = "#2e7d32"
ABLATION = "#6a1b9a"
EVAL = "#c62828"


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body,
    color,
    *,
    title_size=9.4,
    body_size=6.6,
    edge=EDGE,
    linewidth=1.45,
    title_y=0.28,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.032,rounding_size=0.052",
        facecolor=color,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - title_y,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        x + w / 2,
        y + h / 2 - 0.13,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=TEXT,
        linespacing=1.30,
    )


def arrow(
    ax,
    start,
    end,
    label="",
    *,
    color=ONLINE,
    bend=0.0,
    linestyle="solid",
    linewidth=1.65,
    label_offset=0.13,
    label_size=6.7,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={bend}",
    )
    ax.add_patch(patch)
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + label_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=label_size,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "#fbfcfd",
                "edgecolor": "none",
                "alpha": 0.92,
            },
        )


def tag(ax, x, y, text, color):
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=7.7,
        fontweight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.27",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 1.15,
        },
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )
    fig, ax = plt.subplots(figsize=(20, 18.4), dpi=190)
    fig.patch.set_facecolor("#fbfcfd")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 18.4)
    ax.axis("off")

    ax.text(
        10,
        18.08,
        "SIRO-v12 closeout: anchor-centered self-supervised interventional residual observer",
        ha="center",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        10,
        17.70,
        "28/28 artifact bundles valid • 21 rank-valid / 7 rank-fail • SCREEN_NO_GO • contingent 100-epoch wave not launched",
        ha="center",
        fontsize=10.4,
        color=MUTED,
    )

    # Online differentiable path.
    tag(ax, 0.35, 17.25, "ONLINE CAUSAL / DIFFERENTIABLE PATH", ONLINE)
    box(
        ax,
        0.35,
        14.50,
        2.15,
        2.20,
        "paired RGB views",
        "$o_t^{obs}$: deployed causal view\n"
        "$o_t^{clean}$: synchronized train view\n"
        "same unlabeled trajectory",
        "#e3f2fd",
        body_size=6.8,
    )
    box(
        ax,
        2.85,
        14.50,
        2.25,
        2.20,
        "shared causal encoder $E_\\theta$",
        "affine-free per-frame LN\n"
        "no peer/future statistics\n"
        "clean target gradients active",
        "#dcedc8",
        title_size=9.0,
        body_size=6.75,
    )
    box(
        ax,
        5.50,
        14.50,
        2.30,
        2.20,
        "latent views / fixed anchor $c$",
        "$z_t^{obs},z_t^{clean}\\in\\mathbb{R}^D$\n"
        "$c=z_0$ is cached unchanged\n"
        "$r_0=u_0=0$; frame 0 is visible",
        "#e1f5fe",
        title_size=8.6,
        body_size=6.45,
    )
    box(
        ax,
        8.20,
        14.50,
        5.75,
        2.20,
        "two evolving states: residual $r$ and action history $u$",
        "$r_t^-=Ar_{t-1}+b,\\quad u_t^-=Au_{t-1}+Ba_{t-1}$\n"
        "$e_t^o=z_t^{obs}-(c+r_t^-+u_t^-)$\n"
        "$\\delta_t=\\mu_c+K(e_t^o-\\mu_o),\\quad r_t=r_t^-+\\delta_t$\n"
        "$K$ writes only $r$; observations never write $u$; `noaction': $u\\equiv0$",
        "#d1c4e9",
        title_size=9.2,
        body_size=6.6,
    )
    box(
        ax,
        14.35,
        14.50,
        2.00,
        2.20,
        "full read",
        "$h_t=c+r_t+u_t^-$\n"
        "unshrunk action mean\n"
        "$3D$ streaming floats",
        "#b2ebf2",
        body_size=6.7,
    )
    box(
        ax,
        16.75,
        14.50,
        2.90,
        2.20,
        "one-token LeWM predictor",
        "$P(h_t,a_t)\\rightarrow\\hat z_{t+1}$\n"
        "$\\mathcal{L}=\\mathcal{L}_{pred}+\\mathcal{L}_{var}+\\mathcal{L}_{cov}$\n"
        "prediction/VICReg only\n"
        "no memory-specific loss",
        "#c8e6c9",
        title_size=8.8,
        body_size=6.45,
    )
    for x0, x1 in ((2.50, 2.85), (5.10, 5.50), (7.80, 8.20),
                   (13.95, 14.35), (16.35, 16.75)):
        arrow(ax, (x0, 15.60), (x1, 15.60))
    arrow(ax, (11.05, 16.70), (11.05, 17.05), "$a_{t-1}$", label_offset=0.02)

    # Detached epoch-end fits.
    tag(ax, 0.35, 13.92, "DETACHED EPOCH-END FITS — TRAIN DATA ONLY", FIT)
    box(
        ax,
        0.35,
        11.08,
        5.75,
        2.35,
        "stable anchor-centered FWL identification",
        "$x_t=z_t^{clean}-z_0^{clean};\\quad x_{t+1}=A x_t+B a_t+b+\\epsilon_t$\n"
        "FWL residualizes actions against pre-action displacement $x_t$; exact sufficient statistics\n"
        "$A$ is projected to strict spectral stability; $A,B,b$ are fitted state, not optimizer parameters\n"
        "this fit assumes shared translation-equivariant displacement dynamics around each anchor",
        "#e8f5e9",
        title_size=10.0,
        body_size=6.65,
    )
    box(
        ax,
        6.48,
        11.08,
        5.15,
        2.35,
        "paired OAS–LMMSE innovation map",
        "$e_t^o=z_t^{obs}-h_t^-,\\quad e_t^c=z_t^{clean}-h_t^-$\n"
        "$K=\\Sigma_{co}\\Sigma_{oo}^{-1}$ with closed-form OAS covariance\n"
        "$\\delta_t=\\mu_c+K(e_t^o-\\mu_o)$; correction writes only to $r$; $c$ is immutable\n"
        "no corruption label, accept/reject threshold, learned calibration head, or fit loss",
        "#fff8e1",
        title_size=9.5,
        body_size=6.55,
    )
    box(
        ax,
        12.02,
        11.08,
        7.63,
        2.35,
        "shared-$A$ parity-split all-observed-horizon reachability / age spectrum  [diagnostic]",
        "even/odd episodes fit $B_0,B_1$, but reuse full-data $A$, action standardization, and covariance estimates\n"
        "$S_\\times=\\sum_j w_j\\,\\mathrm{sym}[A^jB_0Q_aB_1^\\top(A^\\top)^j],\\quad S=(S_\\times)_+=U\\,\\mathrm{diag}(\\kappa_i)U^\\top$\n"
        "$W_a=\\sum_jw_jA^jBQ_aB^\\top(A^\\top)^j,\\quad J_a=\\sum_j(j+1)w_jA^jBQ_aB^\\top(A^\\top)^j$\n"
        "$\\tau_i=(u_i^\\top J_a u_i)/(u_i^\\top W_a u_i+\\epsilon_{mach})$; $w_j$ is empirical episode survival\n"
        "$C_{\\epsilon c}=\\mathrm{Cov}(\\epsilon_t,c)$ is logged: residual–anchor dependence falsifies the translation assumption\n"
        "shared-$A$ parity agreement is a stability diagnostic—not fully cross-fitted, independent, or unbiased hierarchy evidence",
        "#e0f2f1",
        title_size=9.0,
        body_size=5.85,
        edge=FIT,
        linewidth=1.65,
    )
    arrow(
        ax,
        (3.25, 13.43),
        (9.25, 14.50),
        "$A,B,b$ refit",
        color=FIT,
        linestyle="dashed",
        bend=-0.06,
        label_offset=0.02,
    )
    arrow(
        ax,
        (9.05, 13.43),
        (12.85, 14.50),
        "$K,\\mu_o,\\mu_c$ refit",
        color=FIT,
        linestyle="dashed",
        bend=-0.05,
        label_offset=0.01,
    )
    arrow(
        ax,
        (5.00, 14.50),
        (3.20, 13.43),
        "clean latents + actions",
        color=FIT,
        linestyle="dashed",
        bend=0.10,
        label_offset=0.02,
    )
    arrow(
        ax,
        (5.85, 14.50),
        (8.65, 13.43),
        "paired residuals",
        color=FIT,
        linestyle="dashed",
        bend=-0.10,
        label_offset=0.02,
    )

    # Optional spectral shrink and evaluation-only probes.
    box(
        ax,
        0.35,
        8.62,
        9.35,
        1.88,
        "spectral-shrink ablation only — not the nominated full read",
        "$Q_\\epsilon=\\mathrm{OASCov}(\\epsilon_t),\\quad N=\\sum_jw_jA^jQ_\\epsilon(A^\\top)^j,\\quad R=S(S+N)^\\dagger$\n"
        "replace $h=c+r+u$ by $h=c+r+Ru$ and refit $K$ under that deployed prior\n"
        "$R$ is a signal-to-residual Wiener shrinker—not an orthogonal projector, causal probability, or calibrated quotient; it may bias a real conditional action mean.",
        "#f3e5f5",
        title_size=9.4,
        body_size=6.4,
        edge=ABLATION,
        linewidth=1.65,
    )
    box(
        ax,
        10.08,
        8.62,
        9.57,
        1.88,
        "evaluation-only probes — red path, zero training gradient",
        "held-out/clean/phase MSE + late convergence • true-vs-deranged and empirical $A^jB$ impulse agreement\n"
        "clean-train $\\rightarrow$ clean-val inverse-action and simulator-state ridge • singleton/prefix streaming\n"
        "fixed rollouts/videos, held-out state probes, residual–anchor covariance, timing/VRAM, replication/intervention receipts",
        "#ffebee",
        title_size=9.4,
        body_size=6.4,
        edge=EVAL,
        linewidth=1.7,
    )
    arrow(
        ax,
        (17.35, 14.50),
        (15.10, 10.50),
        "frozen checkpoint / rollouts",
        color=EVAL,
        linestyle="dashed",
        bend=0.18,
        label_offset=-0.02,
    )

    # Seven-mode screen.
    ax.text(
        0.38,
        8.18,
        "Completed seven-design / 28-cell four-GPU screen — held-out prior state NMSE (lower is better)",
        fontsize=11.4,
        fontweight="bold",
        color=EDGE,
    )
    variants = (
        (
            "full centered",
            "$c=z_0$; $h=c+r+u$\ncentered stable FWL + paired OAS $K$\n"
            "mean .935218 • rank-valid 3/4",
            "#a5d6a7",
            FIT,
        ),
        (
            "centered spectral shrink",
            "$h=c+r+Ru$; $R$ from shared-$A$ parity maps\n"
            "not a cross-fitted reachability estimate\n"
            "mean .958373 • rank-valid 3/4",
            "#e1bee7",
            ABLATION,
        ),
        (
            "centered identity $A$",
            "set $A=I$ in $r,u$\nretain centered $B,b,K$ schema\n"
            "mean 1.486104 • rank-valid 3/4",
            "#ffe0b2",
            EDGE,
        ),
        (
            "centered identity $K$",
            "$K=I$; raw observed innovation to $r$\nretain centered stable transition\n"
            "mean 1.351535 • rank-valid 3/4",
            "#fff3e0",
            EDGE,
        ),
        (
            "centered no action",
            "$u\\equiv0$ exactly\nretain $B$ tensors/statistics for schema\n"
            "mean 1.003325 • rank-valid 3/4",
            "#ffccbc",
            EDGE,
        ),
        (
            "absolute no-anchor",
            "$c=0,\\ r_0=z_0$; fit on absolute $z$\n$h=r+u$ under common $3D$ schema\n"
            "mean 1.495009 • rank-valid 2/4",
            "#b3e5fc",
            EDGE,
        ),
        (
            "retrained best V11",
            "frozen KDIO-v11 design/protocol\ncheckpoint retrained from scratch under\nthe same host/data budget\n"
            "mean .564069 • artifacts 4/4",
            "#d1c4e9",
            EDGE,
        ),
    )
    for index, (title, body, color, edge) in enumerate(variants):
        box(
            ax,
            0.35 + index * 2.76,
            5.72,
            2.56,
            2.05,
            title,
            body,
            color,
            title_size=8.75,
            body_size=5.25,
            edge=edge,
            linewidth=1.55,
        )

    # Contracts and red-line positioning.
    box(
        ax,
        0.35,
        3.05,
        9.35,
        2.15,
        "screen result + exact full-read collapse",
        "Full SIRO .935218 vs retrained KDIO-v11 .564069 vs its legal initial-frame integrator .469803.\n"
        "Because both streams use the same $A$ and full uses $R=I$, let $v=r+u$: $v^-=Av+Ba+b$ and $h^-=c+v^-$.\n"
        "The correction adds only to $r$, but the predictor sees only $v$; separate $r/u$ bookkeeping is therefore not a functionally\n"
        "identifiable hierarchy in full SIRO. All 28 bundles validate; 21 pass rank and 7 fail rank. SCREEN_NO_GO / NO 100e.",
        "#ffebee",
        title_size=9.7,
        body_size=6.25,
        edge=EVAL,
        linewidth=1.7,
    )
    box(
        ax,
        10.08,
        3.05,
        9.57,
        2.15,
        "V12b no-optimizer recursive replay — all five tested estimators",
        "Equal-task recursive clean-prior NMSE: old clean-history $K$/current $A$ .723578; identity $K$/current $A$ .727920;\n"
        "deployed history-LMMSE/current $A$ $2.008\\times10^{12}$; Riccati/current $A$ .724090;\n"
        "Riccati/normal stable $A$ .376639. Normal–Riccati repairs recursive prediction, but parity-held action partial-$R^2$\n"
        "is positive on 0/4 tasks. Stage A = false, decision = STOP; replay uses zero optimizer steps and is not a training result.",
        "#e8eaf6",
        title_size=9.7,
        body_size=6.15,
    )
    box(
        ax,
        0.35,
        0.55,
        19.30,
        1.95,
        "negative closeout and scope / novelty red lines",
        "Reward-free, action-observed self-supervision—not action-free learning. Randomization identifies the fitted linear action response only under the stated cohort assumptions.\n"
        "Anchor centering is a performance hypothesis imposing shared translation-equivariant displacement dynamics—not the novelty claim. The absolute no-anchor mode tests it directly.\n"
        "$u$ is the action-explainable response of the fitted model; $r$ is an unresolved residual/innovation stream, not a proven uncontrollable factor. "
        "The shared-$A$ parity spectrum is not a new Gramian, posterior probability, semantic hierarchy, or automatic memory sizing.\n"
        "The controlled composition remains an auditable negative result: it loses badly to V11 and the integrator, its full split collapses to $r+u$, and V12b finds no replicated action signal. No 100-epoch or confirmation launch.",
        "#ffebee",
        title_size=10.0,
        body_size=6.65,
        edge="#b71c1c",
        linewidth=1.8,
    )

    # Legend.
    legend_y = 0.22
    ax.text(0.45, legend_y, "solid blue: online gradient path", fontsize=6.8, color=ONLINE)
    ax.text(4.05, legend_y, "dashed green: detached epoch fit", fontsize=6.8, color=FIT)
    ax.text(8.05, legend_y, "purple: ablation only", fontsize=6.8, color=ABLATION)
    ax.text(11.05, legend_y, "red: evaluation only", fontsize=6.8, color=EVAL)
    ax.text(
        19.55,
        legend_y,
        "completed excluded adaptive screen • negative result",
        fontsize=6.8,
        ha="right",
        color=MUTED,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
