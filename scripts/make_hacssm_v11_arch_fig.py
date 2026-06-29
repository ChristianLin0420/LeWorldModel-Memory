#!/usr/bin/env python3
"""Render the evidence-complete KDIO-v11 architecture and experiment ledger."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "figures" / "fig_hacssm_v11_arch.png"

INK = "#17202a"
MUTED = "#607d8b"
EDGE = "#37474f"
TRAIN = "#1565c0"
OBJECTIVE = "#6a1b9a"
EVAL = "#c62828"
FAIL = "#b71c1c"


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
    title_size=9.2,
    body_size=6.4,
    edge=EDGE,
    linewidth=1.45,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.055",
        facecolor=color,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.27,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        x + w / 2,
        y + h / 2 - 0.12,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color="#263238",
        linespacing=1.30,
    )


def arrow(
    ax,
    start,
    end,
    label="",
    *,
    color=TRAIN,
    bend=0.0,
    linestyle="solid",
    label_offset=0.10,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        color=color,
        linewidth=1.55,
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
            fontsize=6.7,
            color=color,
        )


def tag(ax, x, y, text, color):
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.27",
            "facecolor": "white",
            "edgecolor": color,
            "linewidth": 1.15,
        },
    )


def ledger(ax, x, y, w, h, title, subtitle, rows, color, *, note=""):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.055",
        facecolor="#ffffff",
        edgecolor=color,
        linewidth=1.55,
    )
    ax.add_patch(patch)
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.035, y + h - 0.55),
            w - 0.07,
            0.515,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor=color,
            edgecolor=color,
            linewidth=0,
        )
    )
    ax.text(
        x + 0.16,
        y + h - 0.27,
        title,
        ha="left",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="white",
    )
    ax.text(
        x + 0.16,
        y + h - 0.78,
        subtitle,
        ha="left",
        va="center",
        fontsize=6.4,
        color=MUTED,
    )
    ax.text(
        x + 0.18,
        y + h - 1.08,
        "variant / objective",
        ha="left",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=EDGE,
    )
    ax.text(
        x + w - 0.18,
        y + h - 1.08,
        "held-out  /  clean",
        ha="right",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=EDGE,
    )
    row_y = y + h - 1.36
    step = 0.235 if len(rows) >= 8 else 0.34
    for index, (label, values, emphasis) in enumerate(rows):
        yy = row_y - index * step
        row_color = FAIL if emphasis == "fail" else ("#1b5e20" if emphasis == "best" else INK)
        weight = "bold" if emphasis in {"best", "fail"} else "normal"
        ax.text(
            x + 0.18,
            yy,
            label,
            ha="left",
            va="center",
            fontsize=6.15,
            fontfamily="DejaVu Sans Mono",
            fontweight=weight,
            color=row_color,
        )
        ax.text(
            x + w - 0.18,
            yy,
            values,
            ha="right",
            va="center",
            fontsize=6.15,
            fontfamily="DejaVu Sans Mono",
            fontweight=weight,
            color=row_color,
        )
    if note:
        ax.text(
            x + 0.18,
            y + 0.08,
            note,
            ha="left",
            va="bottom",
            fontsize=5.8,
            color=MUTED,
            linespacing=1.22,
        )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )
    fig, ax = plt.subplots(figsize=(20, 24.2), dpi=190)
    fig.patch.set_facecolor("#fbfcfd")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 24.2)
    ax.axis("off")

    ax.text(
        10,
        23.84,
        "KDIO-v11: completed evidence map and corrected V11b objective",
        ha="center",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        10,
        23.47,
        "Scaled-Stiefel kick–drift observer • live suffix prediction • detached relative-displacement log-ratio ranking",
        ha="center",
        fontsize=10.4,
        color=MUTED,
    )
    ax.text(
        10,
        23.08,
        "DEVELOPMENT SCREEN COMPLETE: NO_GO / NO_LAUNCH   •   REGISTERED OFFICIAL GRID UNLAUNCHED: 0 / 400",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=FAIL,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "#ffebee",
            "edgecolor": FAIL,
            "linewidth": 1.35,
        },
    )

    tag(ax, 0.35, 22.64, "CAUSAL END-TO-END HOST / DEPLOYED MEMORY", TRAIN)
    architecture = (
        (0.35, 1.45, "RGB $o_t$", "corrupted causal view\nclean synchronized\ntarget view", "#e3f2fd", 6.2),
        (2.10, 1.80, "encoder $E_\\theta$", "affine-free frame LN\nactive target gradients\nno peer/future statistics", "#dcedc8", 5.9),
        (4.20, 1.10, "$z_t$", "$D$-vector\ncausal token", "#e1f5fe", 6.5),
        (
            5.60,
            4.15,
            "scaled-Stiefel kick–drift prior",
            "$U=\\mathrm{qf}(M),\\ U^\\top U=I_A;\\quad \\gamma=e^{\\log\\gamma}>0$\n"
            "$f_t=\\tanh(w_q\\odot\\mathrm{RMSNorm}(q_{t-1})+b_f+\\gamma Ua_{t-1})$\n"
            "$v_t^-=v_{t-1}+f_t;\\quad q_t^-=q_{t-1}+v_t^-$",
            "#d1c4e9",
            5.45,
        ),
        (
            10.05,
            3.25,
            "OAS precision observer",
            "$e_t=z_t-q_t^-;\\quad q_t=q_t^-+g_t^q\\odot e_t$\n"
            "$v_t=v_t^-+g_t^v\\odot e_t$\n"
            "$0\\leq g_t^v\\leq g_t^q<1$",
            "#fff3e0",
            6.1,
        ),
        (
            13.60,
            2.10,
            "belief read",
            "$\\phi_t=\\mathrm{RMSNorm}(q_t+v_t)$\n"
            "$\\widetilde z_t=z_t+W_o\\phi_t$\n$W_o=0$ at init",
            "#b2ebf2",
            6.0,
        ),
        (
            16.00,
            3.65,
            "one-token LeWM predictor",
            "$P(\\widetilde z_t,a_t)\\rightarrow\\widehat z_{t+1}$\n"
            "end-to-end causal host\nno three-frame bypass",
            "#c8e6c9",
            6.4,
        ),
    )
    for x, w, title, body, color, body_size in architecture:
        box(
            ax,
            x,
            20.10,
            w,
            2.05,
            title,
            body,
            color,
            title_size=8.4,
            body_size=body_size,
        )
    for x0, x1 in ((1.80, 2.10), (3.90, 4.20), (5.30, 5.60), (9.75, 10.05), (13.30, 13.60), (15.70, 16.00)):
        arrow(ax, (x0, 21.12), (x1, 21.12))
    arrow(ax, (7.66, 22.15), (7.66, 22.52), "$a_{t-1}$", label_offset=0.00)
    arrow(
        ax,
        (11.7, 20.10),
        (7.0, 19.73),
        "persistent $(q_t,v_t)$",
        bend=-0.12,
        label_offset=-0.03,
    )

    box(
        ax,
        0.35,
        16.78,
        7.05,
        2.72,
        "closed-form epoch-end OAS calibration",
        "$u_t=\\sqrt{D}(z_t-q_t^-)/(\\|z_t\\|+\\|q_t^-\\|+\\varepsilon);\\quad y_t=Bu_t$ with fixed Helmert $B$\n"
        "$\\Sigma_{OAS}=(1-\\rho)S+\\rho\\,\\mathrm{tr}(S)I/(D-1)$; "
        "$\\mu=\\bar y,\\ C=\\mathrm{chol}(\\Sigma_{OAS})^{-1}$\n"
        "Fit $\\mu,C$ after each epoch on the clean deployed recurrence with reliability fixed open.\n"
        "$E_t=\\|C(y_t-\\mu)\\|^2/(D-1);\\ r_t=\\tau_t/(\\tau_t+E_t)$; "
        "$g_t^q=r_t\\alpha_t,\\ g_t^v=r_t\\beta_t$\n"
        "$\\tau,\\alpha,\\beta$ see only the prior. No calibration LR, loss weight, threshold, labels, or corruption classifier.",
        "#ede7f6",
        title_size=9.7,
        body_size=5.75,
    )
    box(
        ax,
        7.70,
        16.78,
        5.55,
        2.72,
        "live suffix prediction path",
        "For every observed posterior $s^{obs}_{i,t}=(q,v)$ and available $k$:\n"
        "$\\widehat z^{live}_{i,t,k}=\\mathrm{read}(T^k_{a_i[t:t+k]}(s^{obs}_{i,t};\\gamma))$\n"
        "$e^{live}_{i,t,k}=\\mathrm{mean}_D\\|\\widehat z^{live}_{i,t,k}-z^{clean}_{i,t+k}\\|^2$\n"
        "$\\mathcal{L}_{suffix}=\\mathrm{mean}_{k,i,t}e^{live}_{i,t,k}$\n"
        "This is the only suffix path with live source, clean target, and $\\gamma$ gradients.",
        "#e8f5e9",
        title_size=9.7,
        body_size=5.8,
        edge=OBJECTIVE,
        linewidth=1.65,
    )
    box(
        ax,
        13.55,
        16.78,
        6.10,
        2.72,
        "V11b detached relative-displacement rank path",
        "$\\Delta h^\\pm=\\mathrm{read}(T^k_{a^\\pm}(\\mathrm{sg}(s^{obs})))"
        "-\\mathrm{read}(\\mathrm{sg}(s^{obs}))$\n"
        "$\\Delta z=\\mathrm{sg}(z^{clean}_{t+k}-z^{clean}_t);\\quad "
        "e^\\pm=\\mathrm{mean}_D\\|\\Delta h^\\pm-\\Delta z\\|^2$\n"
        "$a^+$ is executed; $a^-$ is the complete cyclic-neighbor suffix; default rank uses $\\mathrm{sg}(\\gamma)$.\n"
        "$\\mathcal{L}_{rank}=\\mathrm{softplus}(\\log e^+-\\log e^-)$ "
        "$=\\mathrm{softplus}(\\log(e^+/e^-))$\n"
        "No margin/temperature. Endpoint, raw-difference, and live-$\\gamma$ are completed development modes.",
        "#f3e5f5",
        title_size=9.55,
        body_size=5.45,
        edge=OBJECTIVE,
        linewidth=1.65,
    )
    tag(ax, 7.90, 19.78, "SELF-SUPERVISED OBJECTIVE PATHS", OBJECTIVE)

    box(
        ax,
        0.35,
        15.05,
        19.30,
        1.25,
        "evaluation-only observability and action audit — zero training gradient",
        "Primary: frozen clean-train linear probe on strict pre-observation priors → corrupted held-out simulator observation; clean-posterior and direct-predictor ceilings.   "
        "Legal initial-frame action-only integrator; action ridge; streaming/QR/OAS/rollout receipts.   "
        "Raw physics state is archived, not scored. All screen means below are equal-task strict-prior NMSE (lower is better).",
        "#ffebee",
        title_size=9.3,
        body_size=6.2,
        edge=EVAL,
        linewidth=1.65,
    )

    ax.text(
        0.38,
        14.65,
        "Completed adaptive-development screens — every tested V11 variant",
        fontsize=11.6,
        fontweight="bold",
        color=EDGE,
    )
    ax.text(
        19.62,
        14.65,
        "excluded from official claims • one seed • four tasks • 30 epochs",
        ha="right",
        fontsize=7.2,
        color=MUTED,
    )
    ledger(
        ax,
        0.35,
        10.82,
        5.55,
        3.46,
        "superseded action-map screen · 12 cells",
        "fixed-unit/raw/no-action; inverse-head predecessor",
        (
            ("unit Stiefel", ".577906 / .534882", ""),
            ("raw unconstrained", ".556067 / .508061", "best"),
            ("no action", ".753364 / .553457", "fail"),
        ),
        "#455a64",
        note="unit initial integrator .473281; unit is +3.93% vs raw\nand +22.11% vs its integrator → fixed unit scale rejected",
    )
    ledger(
        ax,
        6.20,
        10.82,
        5.95,
        3.46,
        "V11a endpoint-ASR screen · 20 cells",
        "raw endpoint-energy difference; completed NO_GO",
        (
            ("full learned scale", ".581140 / .529385", ""),
            ("fixed scale", ".585329 / .536059", ""),
            ("free geometry", ".570260 / .529758", "best"),
            ("no optimized ASR", ".595159 / .562476", ""),
            ("no action", ".730247 / .556349", "fail"),
        ),
        "#6a1b9a",
        note="full initial integrator .485038; free geometry beats full;\nall 20 predictive curves worsen late → V11a NO_GO",
    )
    ledger(
        ax,
        12.45,
        10.82,
        7.20,
        3.46,
        "V11b corrected-objective screen · 32 cells",
        "live suffix + detached rank; completed NO_GO / NO_LAUNCH",
        (
            ("rawdiff displacement", ".577987 / .541326", "best"),
            ("default rel-displace", ".581525 / .546070", ""),
            ("relative endpoint", ".582805 / .554653", ""),
            ("live-gamma rel-disp", ".582887 / .543264", ""),
            ("fixed scale", ".582939 / .547465", ""),
            ("free geometry", ".584405 / .547287", ""),
            ("no optimized ASR", ".595045 / .557702", ""),
            ("no action", ".718123 / .556815", "fail"),
        ),
        "#283593",
        note="default initial integrator .476157; all 32 curves worsen late",
    )

    box(
        ax,
        0.35,
        9.66,
        19.30,
        0.72,
        "screen decision",
        "Raw-difference is the best V11b memory cell (.577987), but default is .581525 versus its legal integrator .476157 (+22.13%). "
        "Ranking geometry changes do not rescue the observer, and every V11b late-window change is negative:  NO_GO / NO_LAUNCH.",
        "#ffebee",
        title_size=8.2,
        body_size=6.15,
        edge=FAIL,
        linewidth=1.8,
    )

    ax.text(
        0.38,
        9.20,
        "Registered official grid — complete 16-design plan; every design remains UNLAUNCHED (0 / 25), total 0 / 400",
        fontsize=11.2,
        fontweight="bold",
        color=EDGE,
    )
    ax.text(
        19.62,
        9.20,
        "5 tasks × 5 seeds per design; never started after development NO_GO",
        ha="right",
        fontsize=7.0,
        color=FAIL,
    )
    designs = (
        ("SSM", "external baseline\none $D$-state", "#e3f2fd", TRAIN),
        ("compact V8", "external baseline\nadditive two-state", "#e3f2fd", TRAIN),
        ("ORBIT V10", "external baseline\northogonal prior", "#e3f2fd", TRAIN),
        ("full KDIO", "$\\gamma\\,\\mathrm{qf}(M)$ + OAS\nall-suffix V11b rank", "#d1c4e9", "#4527a0"),
        ("fixedscale", "$\\gamma=1$ exactly\nscale ablation", "#fff3e0", "#ef6c00"),
        ("free geometry", "$\\sqrt{A}M/\\|M\\|_F$\nQR bypass", "#fff3e0", "#ef6c00"),
        ("nocalibration", "$\\mu=0,C=I$\nfit disabled", "#fff3e0", "#ef6c00"),
        ("diagonal OAS", "diagonal fit only\nno covariance rotation", "#fff3e0", "#ef6c00"),
        ("h1", "rank + suffix at $k=1$\nfull inference", "#e8f5e9", "#2e7d32"),
        ("firstorder", "no velocity carry\nfirst-order update", "#fff3e0", "#ef6c00"),
        ("nodrift", "$q_t^-=q_{t-1}$\nno displacement", "#fff3e0", "#ef6c00"),
        ("noautonomy", "$w_q=b_f=0$\nremove autonomous force", "#fff3e0", "#ef6c00"),
        ("noaction", "$a_{t-1}=0$\naction path off", "#fff3e0", "#ef6c00"),
        ("noactionswap", "rank diagnosed; weight 0\nlive suffix retained", "#e8f5e9", "#2e7d32"),
        ("nosuffix", "suffix + rank off\nfull inference retained", "#e8f5e9", "#2e7d32"),
        ("noreliability", "$r_t=1$\nOAS stats retained", "#fff3e0", "#ef6c00"),
    )
    for index, (title, body, color, edge) in enumerate(designs):
        row, col = divmod(index, 4)
        x = 0.35 + col * 4.88
        y = 7.72 - row * 1.36
        box(
            ax,
            x,
            y,
            4.53,
            1.10,
            title,
            body + "\nUNLAUNCHED · 0/25",
            color,
            title_size=7.8,
            body_size=5.15,
            edge=edge,
            linewidth=1.45,
        )

    box(
        ax,
        0.35,
        1.34,
        19.30,
        1.30,
        "frozen protocol and status receipt",
        "Fresh iid bounded actions ($\\rho=0$), length 48, RGB64; same five adaptive DMC tasks; training never reads simulator task-observation or physics-state arrays. "
        "Four GPU workers were reserved for disjoint official cells with online W&B epoch logs and rollout artifacts.\n"
        "The reservation was intentionally not exercised: development screens were excluded adaptive evidence, the scientific gate failed, and the 16 × 5 × 5 official matrix remains exactly 0/400. "
        "There is no pending/final-in-progress V11 result.",
        "#eceff1",
        title_size=9.4,
        body_size=6.2,
    )
    ax.text(
        10,
        0.82,
        "Interpretation: V11 provides a clean intervention study of action scale, geometry, ranking energy, and action necessity; it does not establish a winning memory architecture.",
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="bold",
        color="#455a64",
    )
    ax.text(
        10,
        0.42,
        "Development counts: superseded 12 cells + V11a 20 cells + V11b 32 cells. Official counts: pilot 0/240; registered grid 0/400.",
        ha="center",
        va="center",
        fontsize=7.4,
        color=MUTED,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        metadata={"Software": "LeWorldModel-Memory deterministic figure script"},
    )
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
