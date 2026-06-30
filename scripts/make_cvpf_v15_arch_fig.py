#!/usr/bin/env python3
"""Render the CVPF-v15 architecture, controls, and current screen status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "figures" / "fig_cvpf_v15_arch.png"
ANALYSIS = ROOT / "outputs" / "hacssm_v15_screen_cvpf30" / "screen_analysis.json"


def box(
        ax, x: float, y: float, w: float, h: float, title: str, body: str,
        *, face: str, edge: str = "#263238", title_size: float = 10.0,
        body_size: float = 8.0) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.06",
        linewidth=1.25, edgecolor=edge, facecolor=face))
    ax.text(x + .07, y + h - .11, title, fontsize=title_size,
            fontweight="bold", va="top", color="#102027")
    ax.text(x + .07, y + h - .38, body, fontsize=body_size,
            va="top", color="#263238", linespacing=1.25)


def arrow(ax, start: tuple[float, float], end: tuple[float, float],
          *, color: str = "#455a64", style: str = "-|>", width: float = 1.5) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=12,
        linewidth=width, color=color, connectionstyle="arc3,rad=0"))


def _analysis() -> dict[str, Any] | None:
    if not ANALYSIS.is_file():
        return None
    value = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _result_text(value: dict[str, Any] | None) -> tuple[str, str]:
    if value is None:
        return (
            "PROSPECTIVE SCREEN — NOT YET RUN",
            "49 V15 tests pass; two excluded Pendulum smokes completed\n"
            "52 cells = 8 V15 designs (full + 7 interventions) + 5 fresh references × 4 tasks\n"
            "seed 15001 • 30 epochs • online W&B • held-out rollout artifacts\n"
            "The 156-command continuation is prewritten/tested but unauthorized.")
    status = str(value.get("status", "UNKNOWN"))
    means = value.get("design_means", {})
    rows = []
    if isinstance(means, dict):
        ranked = []
        for name, metrics in means.items():
            if isinstance(metrics, dict) and isinstance(
                    metrics.get("heldout_prior_state_nmse"), (int, float)):
                ranked.append((float(metrics["heldout_prior_state_nmse"]), name))
        for score, name in sorted(ranked)[:5]:
            rows.append(f"{name}: {score:.6f}")
    result = " • ".join(rows) if rows else "No complete finite design ledger."
    gate = "PASS" if value.get("scientific_gate_passed") is True else "FAIL"
    return (
        f"COMPLETED SCREEN — {status}",
        f"scientific continuation gate: {gate}\n{result}\n"
        f"completed {value.get('completed_cells', 0)}/{value.get('expected_cells', 52)} cells; "
        "continuation never auto-launched")


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 10), dpi=180)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    ax.text(.35, 9.72, "CVPF-v15: cross-view predictive filtration over all 47 future positions",
            fontsize=18, fontweight="bold", va="top", color="#102027")
    ax.text(.35, 9.34,
            "A finite-horizon, self-supervised 384-float stream state; predictive—not causal—action increments",
            fontsize=10.5, va="top", color="#455a64")

    box(ax, .35, 7.43, 2.38, 1.45, "Synchronized train views",
        "clean RGB → zᶜ\ncorrupted RGB → zᵒ\nexecuted IID actions a\nNo reward, task state, or val fit",
        face="#e3f2fd")
    box(ax, 3.08, 7.43, 2.52, 1.45, "All-suffix targets",
        "H = L−1 = 47 (structural)\nzero-padded future-output blocks\nOAS source covariance; no ridge\nQ×D cross-products, never Q×Q",
        face="#e8eaf6")
    box(ax, 5.95, 7.43, 2.57, 1.45, "Cross-fold mode evidence",
        "even-fit → odd risk score\nodd-fit → even risk score\npooled baseline/gauge: training heuristic\nper mode  gⱼ = ρⱼ wⱼ ∈ [0,1]",
        face="#f3e5f5", body_size=7.4)
    box(ax, 8.87, 7.43, 2.48, 1.45, "Projected suffix shifts",
        "install supported modes first\nTᵣ = argmin ‖DᵣT−SDᵣ‖²\nunit spectral projection\nclosure + support audited",
        face="#fff3e0")
    box(ax, 11.70, 7.43, 3.86, 1.45, "Alternating + differentiable training",
        "Global fit: dropout-off FP64; refresh each epoch\n"
        "Minibatch envelope: symmetric differentiable OAS solves\n"
        "L = next-clean + variance + covariance\n"
        "    + all-valid-suffix envelope",
        face="#e8f5e9", body_size=7.4)
    for left, right in ((2.73, 3.08), (5.60, 5.95), (8.52, 8.87), (11.35, 11.70)):
        arrow(ax, (left, 8.15), (right, 8.15))

    ax.text(.40, 7.08, "Interleaved deployed filtration", fontsize=12,
            fontweight="bold", color="#102027")
    box(ax, .40, 4.82, 3.38, 1.83, "1. Initial-anchor role  q⁰ ∈ ℝᴰ",
        "zᵒ₀ = zᶜ₀ is emitted exactly\nq⁰ predicts the complete clean suffix\n"
        "D₀ decodes future blocks; T₀ shifts them\nNo hidden-state reconstruction or H†",
        face="#bbdefb")
    box(ax, 4.18, 4.82, 3.38, 1.83, "2. Executed-action role  qᵃ ∈ ℝᴰ",
        "εᵃₜ = aₜ − E[a] under frozen IID policy\nqᵃ ← qᵃ + B εᵃₜ\n"
        "future-output impulse modes weighted by ρᵃⱼwᵃⱼ\nPredictive effect; no treatment-effect claim",
        face="#d1c4e9")
    box(ax, 7.96, 4.82, 3.38, 1.83, "3. Observation role  qᵒ ∈ ℝᴰ",
        "νₜ₊₁ = zᵒₜ₊₁ − priorₜ₊₁\nqᵒ ← qᵒ + K νₜ₊₁\n"
        "provisional fit → recursive replay → final refit\n"
        "clean-future PLS modes weighted by ρᵒⱼwᵒⱼ",
        face="#c8e6c9")
    box(ax, 11.74, 4.82, 3.82, 1.83, "Clean-coordinate read + shift",
        "prior/posterior = μ + first(D₀q⁰ + DₐGₐqᵃ\n"
        "                              + DₒGₒqᵒ)\n"
        "each role shifts independently with Tᵣ\n"
        "stream state = 3D = 384 floats; no online covariance\n"
        "posterior current coordinate enters the short LeWM predictor",
        face="#ffe0b2", body_size=7.4)
    for left, right in ((3.78, 4.18), (7.56, 7.96), (11.34, 11.74)):
        arrow(ax, (left, 5.73), (right, 5.73), color="#37474f", width=1.8)

    ax.text(.40, 4.43, "Registered falsification controls", fontsize=12,
            fontweight="bold", color="#102027")
    controls = (
        ("nocorrect", "Gₒ=0", "#c8e6c9"),
        ("noaction", "Gₐ=0", "#d1c4e9"),
        ("norisk", "w=1", "#f3e5f5"),
        ("norho", "ρ=1", "#e1bee7"),
        ("anchoronly", "Gₐ=Gₒ=0", "#bbdefb"),
        ("detachid", "stop ∂fit", "#ffecb3"),
        ("noenvelope", "Lfuture=0", "#ffe0b2"),
    )
    x = .40
    for title, body, color in controls:
        box(ax, x, 3.18, 1.82, .90, title, body, face=color,
            title_size=8.8, body_size=7.5)
        x += 2.08

    title, body = _result_text(_analysis())
    face = "#eceff1" if _analysis() is None else "#ffebee"
    box(ax, .40, 1.20, 15.16, 1.48, title, body, face=face,
        title_size=11, body_size=9)
    ax.text(.42, .76,
            "Claim boundary: fixed L=48/H=47 only • linear OAS/PLS approximation, not literal conditional expectation • "
            "opened-task adaptive evidence cannot support an ICLR confirmation claim",
            fontsize=8.7, color="#b71c1c", va="center")
    ax.text(.42, .38,
            "Prior-art boundary: PSRs, past/future CCA and subspace identification, innovations filtering, covariance "
            "shrinkage, cross-fitting, and differentiable matrix decompositions are established; novelty is only the tested composition.",
            fontsize=8.2, color="#455a64", va="center")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
