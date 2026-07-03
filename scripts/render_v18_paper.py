#!/usr/bin/env python3
"""Render the falsification manuscript from a complete, restart-bound V18 result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v18_release_common as common

DEFAULT_RESULTS = ROOT / "result"
DEFAULT_TEMPLATE = ROOT / "templates" / "ICLR.template.md"
DEFAULT_OUTPUT = ROOT / "generated" / "ICLR.md"
DEFAULT_RESTART_AUDIT = ROOT / "provenance" / "v18_restart_audit.v2.json"

LLM_USAGE_HEADING = "## Appendix H. LLM Usage Statement"
LLM_USAGE_STATEMENT = (
    "OpenAI Codex assisted with code review, experiment monitoring, artifact "
    "auditing, deterministic result-to-manuscript tooling, and manuscript "
    "drafting/editing. The authors verified the executed code, artifacts, "
    "statistics, citations, and final claims and retain responsibility for the work."
)

PRIMARY = "heldout_prior_state_nmse"
CLEAN = "clean_prior_state_nmse"
VARIANCE = "encoder_mean_channel_variance"
RANK = "encoder_covariance_effective_rank"
CONVERGENCE = "predictive_loss_convergence_relative_change"
INTEGRATOR = "initial_encoder_integrator_probe_nmse"

TASKS = (
    "acrobot.swingup",
    "manipulator.bring_ball",
    "quadruped.run",
    "stacker.stack_4",
    "swimmer.swimmer15",
)
TASK_LABELS = {
    "acrobot.swingup": "Acrobot",
    "manipulator.bring_ball": "Manipulator",
    "quadruped.run": "Quadruped",
    "stacker.stack_4": "Stacker",
    "swimmer.swimmer15": "Swimmer-15",
}
DESIGNS = (
    "vicreg_none",
    "vicreg_gru",
    "vicreg_ssm",
    "vicreg_hacssmv8",
    "vicreg_hacssmv8_noaction",
    "vicreg_hacssmv8_single",
    "vicreg_hacssmv8_static",
    "vicreg_hacssmv8_dynamic",
)
DESIGN_LABELS = {
    "vicreg_none": "No carrier",
    "vicreg_gru": "GRU",
    "vicreg_ssm": "Diag. SSM",
    "vicreg_hacssmv8": "SAS-PC",
    "vicreg_hacssmv8_noaction": "No action",
    "vicreg_hacssmv8_single": "Single read",
    "vicreg_hacssmv8_static": "Static",
    "vicreg_hacssmv8_dynamic": "Dynamic",
}

CONTRAST_KEYS = {
    "R": "vicreg_hacssmv8_vs_recurrent_envelope:heldout_prior_state_nmse",
    "N": "vicreg_hacssmv8_vs_vicreg_none:heldout_prior_state_nmse",
    "I": "vicreg_hacssmv8_vs_checkpoint_integrator:heldout_prior_state_nmse",
    "A": "vicreg_hacssmv8_vs_vicreg_hacssmv8_noaction:heldout_prior_state_nmse",
    "J": "vicreg_hacssmv8_vs_vicreg_hacssmv8_single:heldout_prior_state_nmse",
    "E": "vicreg_hacssmv8_vs_endpoint_envelope:heldout_prior_state_nmse",
    "D": "vicreg_hacssmv8_vs_recurrent_envelope:deep_prior_state_nmse",
    "C": "vicreg_hacssmv8_vs_recurrent_envelope:clean_prior_state_nmse",
}
GATE_KEYS = {
    "R": "v8_vs_per_cell_better_gru_ssm",
    "N": "v8_vs_none",
    "I": "v8_vs_checkpoint_integrator",
    "A": "action_causality",
    "J": "joint_state_use",
    "E": "learned_v8_vs_static_dynamic_envelope_noninferiority",
    "D": "deep_vs_per_cell_better_gru_ssm",
    "C": "clean_prior_guard_vs_per_cell_better_gru_ssm",
}


def pct(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):+.{digits}f}%"


def ci(contrast: dict) -> str:
    bootstrap = contrast["bootstrap"]
    return f"[{pct(bootstrap['ci95_low'])}, {pct(bootstrap['ci95_high'])}]"


def latex_pct(value: float, digits: int = 2) -> str:
    """Signed percent with an escaped percent sign for raw-LaTeX cells."""

    return f"{100.0 * float(value):+.{digits}f}\\%"


def latex_ci(record: dict, *, level: int = 95) -> str:
    bootstrap = record["bootstrap"]
    low = latex_pct(bootstrap[f"ci{level}_low"])
    high = latex_pct(bootstrap[f"ci{level}_high"])
    return f"[{low}, {high}]"


def latex_header(*labels: str) -> str:
    cells = " & ".join(
        rf"\textcolor{{NVIDIADark}}{{\textbf{{{label}}}}}" for label in labels
    )
    return cells + r" \\"


def latex_table(
    *,
    caption: str,
    label: str,
    column_spec: str,
    header: str,
    body: list[str],
    placement: str = "H",
    size: str = r"\small",
    extra_preamble: tuple[str, ...] = (),
) -> str:
    """Assemble one booktabs table with the shared manuscript styling."""

    lines = [
        rf"\begin{{table}}[{placement}]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        size,
        *extra_preamble,
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def gate(value: bool) -> str:
    return "PASS" if value else "FAIL"


def verdict(passed: bool) -> str:
    color = "NVIDIADark" if passed else "FailAmber"
    return rf"\textbf{{\textcolor{{{color}}}{{{gate(passed)}}}}}"


def scalar(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.3e}"
    return f"{value:.4f}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def change_text(contrast: dict, passed: bool, *, noninferiority: bool = False) -> str:
    effect = float(contrast["mean_paired_relative_reduction"])
    if noninferiority:
        outcome = "meets" if passed else "does not meet"
        return (
            f"{outcome} the registered noninferiority criterion "
            f"({pct(effect)}, 95% CI {ci(contrast)})"
        )
    direction = "reduces error" if effect >= 0.0 else "increases error"
    magnitude = f"{abs(100.0 * effect):.2f}%"
    outcome = "passes" if passed else "does not pass"
    return (
        f"{direction} by {magnitude} (95% CI {ci(contrast)}; "
        f"{contrast['paired_wins']}/25 cell and {contrast['task_mean_wins']}/5 "
        f"task wins) and {outcome} the registered gate"
    )


def load_inputs(root: Path) -> tuple[dict, dict, list[dict[str, str]], dict]:
    bundle = common.load_complete_bundle(root, require_failure=True)
    return bundle["report"], bundle["protocol"], bundle["cells"], bundle


def load_secondary_figure_manifest(path: Path, bundle: dict) -> dict:
    """Load the hash-bound descriptive slices used by Figure 3 and its prose."""

    manifest = common.read_json(path)
    expected = {
        "analysis_sha256": bundle["hashes"]["confirmation_analysis.json"],
        "protocol_sha256": bundle["hashes"]["confirmation_protocol.json"],
        "summary_sha256": bundle["hashes"]["confirmation_summary.json"],
    }
    mismatches = [
        f"{key}: {manifest.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    secondary = manifest.get("descriptive_secondary")
    if mismatches or not isinstance(secondary, dict):
        raise RuntimeError(
            "descriptive figure manifest is stale or malformed: "
            + "; ".join(mismatches or ["missing descriptive_secondary"])
        )
    if secondary.get("official_decision_changed") is not False \
            or secondary.get("decision_gates_defined") is not False \
            or secondary.get("claim_scope") != "descriptive decomposition only":
        raise RuntimeError("descriptive figure manifest changes the registered claim scope")
    return secondary


def task_design_table(rows: list[dict[str, str]]) -> str:
    """Appendix table for the four SAS-PC control arms."""

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        values[(row["task"], row["design"])].append(float(row[PRIMARY]))
    designs = DESIGNS[4:]
    body = []
    for task in TASKS:
        cells = []
        for design in designs:
            vector = np.asarray(values[(task, design)], dtype=np.float64)
            cells.append(f"{vector.mean():.3f} $\\pm$ {vector.std(ddof=1):.3f}")
        body.append(f"{TASK_LABELS[task]} & " + " & ".join(cells) + r" \\")
    return latex_table(
        caption=(
            "Held-out prior-state NMSE for SAS-PC component and endpoint "
            "controls by task (mean $\\pm$ SD over five seeds; lower is better)."
        ),
        label="tbl:component-controls",
        column_spec="@{}l" + "r" * len(designs) + "@{}",
        header=latex_header(
            "Task", *(DESIGN_LABELS[design] for design in designs)
        ),
        body=body,
    )


def task_recurrent_table(rows: list[dict[str, str]], report: dict) -> str:
    index = {
        (row["task"], int(row["seed"]), row["design"]): float(row[PRIMARY])
        for row in rows
    }
    body = []
    for task in TASKS:
        selected = defaultdict(int)
        for seed in range(18001, 18006):
            choices = {
                "GRU": index[(task, seed, "vicreg_gru")],
                "SSM": index[(task, seed, "vicreg_ssm")],
            }
            identity, _ = min(choices.items(), key=lambda pair: (pair[1], pair[0]))
            selected[identity] += 1
        body.append(
            f"{TASK_LABELS[task]} & {selected['GRU']}/5 & {selected['SSM']}/5 \\\\"
        )
    return latex_table(
        caption=(
            "Recurrent-envelope identities selected on the primary metric and "
            "reused for deep, clean, task, and corruption slices."
        ),
        label="tbl:recurrent-identities",
        column_spec="@{}lrr@{}",
        header=latex_header("Task", "GRU selected", "SSM selected"),
        body=body,
    )


def main_task_table(rows: list[dict[str, str]], report: dict) -> str:
    """Typeset the main carrier table with decimal-aligned uncertainties."""

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        task, design = row["task"], row["design"]
        grouped[(task, design)].append(float(row[PRIMARY]))
    labels = {
        "acrobot.swingup": "Acrobot",
        "manipulator.bring_ball": "Manipulator",
        "quadruped.run": "Quadruped",
        "stacker.stack_4": "Stacker",
        "swimmer.swimmer15": "Swimmer-15",
    }
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Held-out prior-state NMSE by task. Mean $\pm$ SD over five "
        r"seeds; lower is better. Green identifies the candidate header; "
        r"bold denotes the lowest mean across the four designs.}",
        r"\label{tbl:task-nmse}",
        r"\small",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\begin{tabular}{@{}l r@{\,\(\pm\)\,}l r@{\,\(\pm\)\,}l "
        r"r@{\,\(\pm\)\,}l r@{\,\(\pm\)\,}l@{}}",
        r"\toprule",
        r"& \multicolumn{8}{c}{\textcolor{NVIDIADark}{\textbf{Trained design}}} \\",
        r"\cmidrule(lr){2-9}",
        r"\textbf{Task} & \multicolumn{2}{c}{\textbf{No carrier}} & "
        r"\multicolumn{2}{c}{\textbf{GRU}} & "
        r"\multicolumn{2}{c}{\textbf{Diag. SSM}} & "
        r"\multicolumn{2}{c}{\colorbox{NVIDIAPale}{\strut\textcolor{NVIDIADark}{\textbf{SAS-PC}}}} \\",
        r"\midrule",
    ]
    for task in TASKS:
        vectors = [
            np.asarray(grouped[(task, design)], dtype=np.float64)
            for design in DESIGNS[:4]
        ]
        best = min(range(len(vectors)), key=lambda index: vectors[index].mean())
        cells: list[str] = []
        for index, vector in enumerate(vectors):
            mean = vector.mean()
            uncertainty = vector.std(ddof=1)
            value = f"{mean:.3f} & {uncertainty:.3f}"
            if index == best:
                value = (
                    rf"\bfseries {mean:.3f} & \bfseries {uncertainty:.3f}"
                )
            cells.append(value)
        lines.append(f"{labels[task]} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1pt}",
        r"\begin{minipage}{0.98\linewidth}\footnotesize\textit{Note.} NMSE is "
        r"standardized within task and is not pooled across tasks.\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def aggregate_evidence_table(report: dict) -> str:
    """All registered effect contrasts in one compact, sign-consistent table."""

    specifications = (
        ("GRU/SSM envelope", "R", ">=3%; CI>0; 18/25; 4/5"),
        ("no carrier", "N", ">=5%; 20/25; 4/5"),
        ("legal integrator", "I", ">=3%; 18/25; 4/5"),
        ("deep gap vs GRU/SSM", "D", "CI>0; 3/5"),
        ("no action transport", "A", ">=5%; CI>0; 18/25; 4/5"),
        ("single read", "J", ">=3%; CI>0; 18/25; 4/5"),
        ("endpoint envelope", "E", "mean, CI >=-1%"),
        ("clean vs GRU/SSM", "C", "effect >=-3%"),
    )
    lines = [
        "| comparison | frozen criterion | effect [95% CI] | cell/task wins | gate |",
        "|---|---|---:|---:|---|",
    ]
    for label, key, criterion in specifications:
        contrast = report["contrasts"][CONTRAST_KEYS[key]]
        passed = report["gates"][GATE_KEYS[key]]
        lines.append(
            f"| {label} | {criterion} | "
            f"{pct(contrast['mean_paired_relative_reduction'])} {ci(contrast)} | "
            f"{contrast['paired_wins']}/25; {contrast['task_mean_wins']}/5 | "
            f"**{gate(passed)}** |"
        )
    lines.extend([
        "",
        ": Registered effect contrasts and decision receipts. Effects are "
        "relative reductions; positive favors SAS-PC. Intervals are crossed "
        "task-by-seed bootstrap intervals. {#tbl:registered-effects}",
    ])
    return "\n".join(lines)


def validity_by_task_table(rows: list[dict[str, str]]) -> str:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Representation-rank and convergence receipts by task. Counts "
        r"aggregate eight designs $\times$ five seeds.}",
        r"\label{tbl:validity-by-task}",
        r"\small",
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"\textcolor{NVIDIADark}{\textbf{Task}} & "
        r"\textcolor{NVIDIADark}{\textbf{Rank $\geq 16$}} & "
        r"\textcolor{NVIDIADark}{\textbf{Change $\leq 5\%$}} & "
        r"\textcolor{NVIDIADark}{\textbf{Both guards}} \\",
        r"\midrule",
    ]
    overall: list[dict[str, str]] = []
    for task in TASKS:
        task_rows = grouped[task]
        overall.extend(task_rows)
        ranks = [float(row[RANK]) for row in task_rows]
        changes = [abs(float(row[CONVERGENCE])) for row in task_rows]
        lines.append(
            f"{TASK_LABELS[task]} & {sum(value >= 16 for value in ranks)}/40 & "
            f"{sum(value <= .05 for value in changes)}/40 & "
            f"{sum(rank >= 16 and change <= .05 for rank, change in zip(ranks, changes, strict=True))}/40 "
            r"\\"
        )
    ranks = [float(row[RANK]) for row in overall]
    changes = [abs(float(row[CONVERGENCE])) for row in overall]
    lines.append(
        r"\midrule\textbf{Overall} & "
        f"\\textbf{{{sum(value >= 16 for value in ranks)}/200}} & "
        f"\\textbf{{{sum(value <= .05 for value in changes)}/200}} & "
        f"\\textbf{{{sum(rank >= 16 and change <= .05 for rank, change in zip(ranks, changes, strict=True))}/200}} "
        r"\\"
    )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def secondary_component_table(secondary: dict) -> str:
    """Condition-specific component effects moved out of the main figure."""

    labels = {
        "freeze": "Freeze",
        "gaussian_noise": "Gaussian noise",
        "checkerboard": "Checkerboard",
        "long_freeze": "Long freeze",
    }
    policies = (
        ("no_carrier", "vs no carrier"),
        ("no_action_transport", "action transport"),
        ("single_read", "joint read"),
    )
    body = []
    for condition in ("freeze", "gaussian_noise", "checkerboard", "long_freeze"):
        records = secondary["condition_primary"][condition]
        cells = [
            latex_pct(records[policy]["mean_paired_relative_reduction"])
            for policy, _ in policies
        ]
        body.append(f"{labels[condition]} & " + " & ".join(cells) + r" \\")
    return latex_table(
        caption=(
            "Descriptive corruption-specific component effects. Positive "
            "values favor full SAS-PC; no multiplicity correction or decision "
            "gate is defined."
        ),
        label="tbl:secondary-components",
        column_spec="@{}lrrr@{}",
        header=latex_header("Corruption", *(label for _, label in policies)),
        body=body,
    )


def secondary_phase_table(secondary: dict) -> str:
    """Descriptive phase effects against the frozen recurrent identity."""

    labels = {
        "gap": "Whole gap",
        "deep": "Deep gap",
        "first_post": "First post-gap",
        "post": "Post-gap",
    }
    body = []
    for phase in ("gap", "deep", "first_post", "post"):
        record = secondary["phase_equal_condition_mean"][phase]
        body.append(
            f"{labels[phase]} & "
            f"{latex_pct(record['mean_paired_relative_reduction'])} "
            f"{latex_ci(record)} & {record['paired_wins']}/25; "
            f"{record['task_mean_wins']}/5 \\\\"
        )
    return latex_table(
        caption=(
            "Descriptive phase effects against the primary-selected GRU/SSM "
            "identity, averaged equally across corruptions. Deep gap is nested "
            "within whole gap; intervals are unadjusted and define no decision "
            "gate."
        ),
        label="tbl:secondary-phases",
        column_spec="@{}lrr@{}",
        header=latex_header("Phase", "Effect [95\\% CI]", "Cell/task wins"),
        body=body,
    )


def design_rank_table(rows: list[dict[str, str]]) -> str:
    values = {
        (row["task"], int(row["seed"]), row["design"]): float(row[PRIMARY])
        for row in rows
    }
    ranks: dict[str, list[int]] = defaultdict(list)
    for task in TASKS:
        for seed in range(18001, 18006):
            ordered = sorted(
                DESIGNS,
                key=lambda design: (values[(task, seed, design)], design),
            )
            for rank, design in enumerate(ordered, 1):
                ranks[design].append(rank)
    body = []
    ordered_designs = sorted(DESIGNS, key=lambda design: np.mean(ranks[design]))
    for design in ordered_designs:
        vector = np.asarray(ranks[design], dtype=np.int64)
        body.append(
            f"{DESIGN_LABELS[design]} & {vector.mean():.2f} & "
            f"{np.median(vector):.0f} & {np.sum(vector == 1)}/25 & "
            f"{np.sum(vector <= 3)}/25 \\\\"
        )
    return latex_table(
        caption=(
            "Descriptive within-task, within-seed rank summaries "
            "corresponding to Figure \\ref{fig:fig-v18-task-design}."
        ),
        label="tbl:design-ranks",
        column_spec="@{}lrrrr@{}",
        header=latex_header("Design", "Mean rank", "Median", "First", "Top-3"),
        body=body,
    )


def integrator_table(rows: list[dict[str, str]]) -> str:
    by_key = {
        (row["task"], int(row["seed"]), row["design"]): row
        for row in rows
    }
    body = []
    for task in TASKS:
        candidate, reference, effects = [], [], []
        for seed in range(18001, 18006):
            row = by_key[(task, seed, "vicreg_hacssmv8")]
            cand = float(row[PRIMARY])
            ref = float(row[INTEGRATOR])
            candidate.append(cand)
            reference.append(ref)
            effects.append((ref - cand) / max(abs(ref), 1e-12))
        task_effect = (np.mean(reference) - np.mean(candidate)) / max(abs(np.mean(reference)), 1e-12)
        body.append(
            f"{TASK_LABELS[task]} & {np.mean(candidate):.3f} & "
            f"{np.mean(reference):.3f} & {latex_pct(task_effect)} & "
            f"{sum(value > 0 for value in effects)}/5 \\\\"
        )
    return latex_table(
        caption=(
            "Legal initial-frame/action integrator guard by task. Positive "
            "paired reduction favors SAS-PC."
        ),
        label="tbl:integrator-guard",
        column_spec="@{}lrrrr@{}",
        header=latex_header(
            "Task", "SAS-PC prior NMSE", "Integrator NMSE",
            "Paired reduction", "Seed wins",
        ),
        body=body,
    )


def representation_table(rows: list[dict[str, str]]) -> str:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["design"]].append(row)
    health_body: list[str] = []
    convergence_body: list[str] = []
    for design in DESIGNS:
        arm = grouped[design]
        variances = [float(row[VARIANCE]) for row in arm]
        ranks = [float(row[RANK]) for row in arm]
        changes = [abs(float(row[CONVERGENCE])) for row in arm]
        health_body.append(
            f"{DESIGN_LABELS[design]} & {scalar(min(variances))} & "
            f"{min(ranks):.2f} & "
            f"{sum(value >= 1e-4 for value in variances)}/25 & "
            f"{sum(value >= 16 for value in ranks)}/25 \\\\"
        )
        convergence_body.append(
            f"{DESIGN_LABELS[design]} & {latex_pct(max(changes))} & "
            f"{sum(value <= .05 for value in changes)}/25 \\\\"
        )
    health = latex_table(
        caption="Representation-health receipts by design.",
        label="tbl:representation-by-design",
        column_spec="@{}lrrrr@{}",
        header=latex_header(
            "Design", "Min. variance", "Min. rank", "Variance pass",
            "Rank pass",
        ),
        body=health_body,
    )
    convergence = latex_table(
        caption="Late-window convergence receipts by design.",
        label="tbl:convergence-by-design",
        column_spec="@{}lrr@{}",
        header=latex_header(
            "Design", "Largest late change", "Convergence pass"
        ),
        body=convergence_body,
    )
    return health + "\n\n" + convergence


def artifact_table(
    report: dict,
    protocol: dict,
    canonical_commands_sha256: str,
) -> str:
    manifest = protocol["data"]["__manifest__"]
    rows = (
        ("clean worktree at freeze", str(protocol["git_worktree_clean"])),
        ("protocol SHA-256", report["input_protocol_sha256"]),
        ("canonical public command-list SHA-256", canonical_commands_sha256),
        ("cache-manifest SHA-256", manifest["path_sha256"]),
        ("cache-sidecar SHA-256", manifest["sidecar_sha256"]),
        ("artifact-manifest SHA-256", report["input_artifact_manifest_sha256"]),
        ("cell CSV SHA-256", report["cells_csv_sha256"]),
        ("contrast CSV SHA-256", report["contrasts_csv_sha256"]),
    )
    body = []
    for name, value in rows:
        text = str(value)
        display = text[:16] + r"\ldots" if len(text) > 20 else text
        body.append(rf"{name} & \texttt{{{display}}} \\")
    return latex_table(
        caption="Frozen execution and artifact identities.",
        label="tbl:artifact-identities",
        column_spec="@{}ll@{}",
        header=latex_header("Identity", "Frozen value or SHA-256 prefix"),
        body=body,
        size=r"\footnotesize",
    )


def gate_table(report: dict) -> str:
    """Main-text gate-receipt table with compact requirement/observed cells."""

    contrasts = {name: report["contrasts"][key] for name, key in CONTRAST_KEYS.items()}
    gates = report["gates"]
    representation = report["representation"]["observed"]
    convergence = report["convergence"]["observed"]

    def observed(name: str, *, with_ci: bool = False, with_wins: bool = True) -> str:
        contrast = contrasts[name]
        parts = [latex_pct(contrast["mean_paired_relative_reduction"])]
        if with_ci:
            parts.append(latex_ci(contrast))
        if with_wins:
            parts.append(f"{contrast['paired_wins']}/25")
            parts.append(f"{contrast['task_mean_wins']}/5")
        return "; ".join(parts)

    rows = [
        ("Integrity", "200/200 valid cells", "200/200", gates["integrity"]),
        ("Recurrent envelope",
         r"$\geq$3\%; CI$>$0; $\geq$18/25; $\geq$4/5",
         observed("R", with_ci=True), gates[GATE_KEYS["R"]]),
        ("No carrier", r"$\geq$5\%; $\geq$20/25; $\geq$4/5",
         observed("N"), gates[GATE_KEYS["N"]]),
        ("Integrator", r"$\geq$3\%; $\geq$18/25; $\geq$4/5",
         observed("I"), gates[GATE_KEYS["I"]]),
        ("Deep gap", r"CI$>$0; $\geq$3/5",
         f"{observed('D', with_ci=True, with_wins=False)}; "
         f"{contrasts['D']['task_mean_wins']}/5", gates[GATE_KEYS["D"]]),
        ("Action transport",
         r"$\geq$5\%; CI$>$0; $\geq$18/25; $\geq$4/5",
         observed("A", with_ci=True), gates[GATE_KEYS["A"]]),
        ("Joint read",
         r"$\geq$3\%; CI$>$0; $\geq$18/25; $\geq$4/5",
         observed("J", with_ci=True), gates[GATE_KEYS["J"]]),
        ("Shrinkage", r"mean and CI$\geq$-1\%",
         observed("E", with_ci=True, with_wins=False), gates[GATE_KEYS["E"]]),
        ("Clean guard", r"degradation$\leq$3\%",
         f"effect {observed('C', with_wins=False)}", gates[GATE_KEYS["C"]]),
        ("Representation",
         r"var$\geq10^{-4}$; rank$\geq$16; 200/200",
         f"var {scalar(representation['minimum_channel_variance'])} "
         f"({representation['variance_passing_cells']}/200); "
         f"rank {representation['minimum_effective_rank']:.2f} "
         f"({representation['rank_passing_cells']}/200)",
         gates["healthy_representation"]),
        ("Convergence", r"$|$late change$|\leq$5\%; 200/200",
         f"max {latex_pct(convergence['maximum_absolute_relative_change'])}; "
         f"{convergence['passing_cells']}/200", gates["convergence"]),
    ]
    body = [
        f"{name} & {requirement} & {receipt} & {verdict(passed)} \\\\"
        for name, requirement, receipt, passed in rows
    ]
    return latex_table(
        caption=(
            "Frozen confirmation gates and observed receipts. Decisions use "
            "unrounded values."
        ),
        label="tbl:gate-receipts",
        column_spec=(
            r"@{}l >{\raggedright\arraybackslash}p{0.285\linewidth} "
            r">{\raggedright\arraybackslash}p{0.405\linewidth} l@{}"
        ),
        header=latex_header(
            "Gate", "Registered requirement", "Observed", "Verdict"
        ),
        body=body,
        placement="!t",
        size=r"\footnotesize",
        extra_preamble=(r"\setlength{\tabcolsep}{3.5pt}",),
    )


def contrast_full_table(report: dict) -> str:
    """Appendix table with 95/90 intervals and pooled NMSE for every contrast."""

    names = (
        ("Recurrent envelope", "R"),
        ("No carrier", "N"),
        ("Legal integrator", "I"),
        ("Deep gap", "D"),
        ("Clean prior", "C"),
        ("Action transport", "A"),
        ("Joint read", "J"),
        ("Endpoint envelope", "E"),
    )
    body = []
    for label, key in names:
        contrast = report["contrasts"][CONTRAST_KEYS[key]]
        body.append(
            f"{label} & {latex_pct(contrast['mean_paired_relative_reduction'])} & "
            f"{latex_ci(contrast)} & {latex_ci(contrast, level=90)} & "
            f"{contrast['paired_wins']}/25 & {contrast['task_mean_wins']}/5 & "
            f"{contrast['candidate_mean']:.3f}/{contrast['reference_mean']:.3f} \\\\"
        )
    return latex_table(
        caption=(
            "Registered held-out primary contrasts with 95\\% and 90\\% "
            "crossed bootstrap intervals and pooled candidate/reference NMSE. "
            "Deep gap and clean prior use the deep and clean prior-state NMSE "
            "against the recurrent envelope; action transport and joint read "
            "compare against the no-action and single-read controls; the "
            "endpoint envelope is the shrinkage gate of "
            "Table \\ref{tbl:gate-receipts}."
        ),
        label="tbl:contrast-full",
        column_spec="@{}lrrrrrr@{}",
        header=latex_header(
            "Contrast", "Effect", "95\\% CI", "90\\% CI", "cells", "tasks",
            "cand./ref. NMSE",
        ),
        body=body,
        size=r"\footnotesize",
        extra_preamble=(r"\setlength{\tabcolsep}{3.4pt}",),
    )


def clean_task_table(rows: list[dict[str, str]]) -> str:
    """Appendix table of clean prior-state NMSE across the trained carriers."""

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["design"])].append(float(row[CLEAN]))
    body = []
    for task in TASKS:
        vectors = [
            np.asarray(grouped[(task, design)], dtype=np.float64)
            for design in DESIGNS[:4]
        ]
        best = min(range(len(vectors)), key=lambda index: vectors[index].mean())
        cells = []
        for index, vector in enumerate(vectors):
            cell = f"{vector.mean():.3f} $\\pm$ {vector.std(ddof=1):.3f}"
            if index == best:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        body.append(f"{TASK_LABELS[task]} & " + " & ".join(cells) + r" \\")
    return latex_table(
        caption=(
            "Clean prior-state NMSE by task (mean $\\pm$ SD over five seeds; "
            "lower is better). Bold denotes the lowest mean per task."
        ),
        label="tbl:clean-task-nmse",
        column_spec="@{}lrrrr@{}",
        header=latex_header(
            "Task", *(DESIGN_LABELS[design] for design in DESIGNS[:4])
        ),
        body=body,
    )


def replacements(
    report: dict,
    protocol: dict,
    rows: list[dict[str, str]],
    restart_audit: dict,
    canonical_commands_sha256: str,
    secondary: dict,
) -> dict[str, str]:
    contrasts = {name: report["contrasts"][key] for name, key in CONTRAST_KEYS.items()}
    gates = {name: report["gates"][key] for name, key in GATE_KEYS.items()}
    values: dict[str, str] = {"VALID": str(report["completed_valid_cells"])}
    for name in ("R", "N", "I", "A", "J", "E", "D"):
        contrast = contrasts[name]
        values.update({
            f"{name}_MEAN": pct(contrast["mean_paired_relative_reduction"]),
            f"{name}_CI": ci(contrast),
            f"{name}_WINS": str(contrast["paired_wins"]),
            f"{name}_TASKS": str(contrast["task_mean_wins"]),
            f"{name}_GATE": gate(gates[name]),
        })
    values.update({
        "R_TEXT": change_text(contrasts["R"], gates["R"]),
        "N_TEXT": change_text(contrasts["N"], gates["N"]),
        "I_TEXT": change_text(contrasts["I"], gates["I"]),
        "A_TEXT": change_text(contrasts["A"], gates["A"]),
        "J_TEXT": change_text(contrasts["J"], gates["J"]),
        "E_TEXT": change_text(contrasts["E"], gates["E"], noninferiority=True),
        "I_INTERPRETATION": (
            "This meets the registered superiority standard over the legal "
            "initial-state/action summary."
            if gates["I"] else
            "This does not meet the registered superiority standard over the legal "
            "initial-state/action summary; the failed clause may be magnitude, "
            "consistency, or both."
        ),
        "C_MEAN": pct(contrasts["C"]["mean_paired_relative_reduction"]),
        "C_CI": ci(contrasts["C"]),
        "C_WINS": str(contrasts["C"]["paired_wins"]),
        "C_TASKS": str(contrasts["C"]["task_mean_wins"]),
        "C_DEGRADE": pct(-contrasts["C"]["mean_paired_relative_reduction"]),
        "C_GATE": gate(gates["C"]),
        "GRU_COUNT": str(contrasts["R"]["selected_reference_counts"].get("vicreg_gru", 0)),
        "SSM_COUNT": str(contrasts["R"]["selected_reference_counts"].get("vicreg_ssm", 0)),
    })
    endpoint_counts = contrasts["E"]["selected_reference_counts"]
    values.update({
        "DYNAMIC_COUNT": str(endpoint_counts.get("vicreg_hacssmv8_dynamic", 0)),
        "STATIC_COUNT": str(endpoint_counts.get("vicreg_hacssmv8_static", 0)),
        "SAS_POOLED_PRIMARY": f"{contrasts['I']['candidate_mean']:.3f}",
        "INTEGRATOR_POOLED_PRIMARY": f"{contrasts['I']['reference_mean']:.3f}",
    })
    recurrent_slices = {
        condition: secondary["condition_primary"][condition]["primary_selected_gru_ssm"]
        for condition in ("freeze", "gaussian_noise", "checkerboard", "long_freeze")
    }
    values.update({
        "FREEZE_R": pct(recurrent_slices["freeze"]["mean_paired_relative_reduction"]),
        "GAUSSIAN_R": pct(
            recurrent_slices["gaussian_noise"]["mean_paired_relative_reduction"]
        ),
        "CHECKER_R": pct(
            recurrent_slices["checkerboard"]["mean_paired_relative_reduction"]
        ),
        "LONG_FREEZE_R": pct(
            recurrent_slices["long_freeze"]["mean_paired_relative_reduction"]
        ),
    })
    for condition, prefix in (
        ("freeze", "FREEZE"),
        ("gaussian_noise", "GAUSSIAN"),
        ("checkerboard", "CHECKER"),
        ("long_freeze", "LONG_FREEZE"),
    ):
        record = recurrent_slices[condition]
        values[f"{prefix}_WINS"] = str(record["paired_wins"])
        values[f"{prefix}_TASKS"] = str(record["task_mean_wins"])

    recurrent_cells = np.asarray(contrasts["R"]["cell_effects"], dtype=np.float64)
    task_prefixes = ("ACROBOT", "MANIPULATOR", "QUADRUPED", "STACKER", "SWIMMER")
    for index, (task, prefix) in enumerate(zip(TASKS, task_prefixes, strict=True)):
        task_cells = recurrent_cells[index]
        values[f"R_{prefix}"] = pct(contrasts["R"]["task_effects"][task])
        values[f"R_{prefix}_WINS"] = str(int(np.sum(task_cells > 0.0)))
        values[f"R_{prefix}_MIN"] = pct(float(task_cells.min()))
        values[f"R_{prefix}_MAX"] = pct(float(task_cells.max()))

    seed_metric_index = {
        (row["task"], int(row["seed"]), row["design"]): float(row[PRIMARY])
        for row in rows
    }
    fixed_seed_references = {
        "N": "vicreg_none",
        "A": "vicreg_hacssmv8_noaction",
        "J": "vicreg_hacssmv8_single",
    }
    for contrast_name in ("R", "N", "A", "J"):
        if contrast_name == "R":
            matrix = recurrent_cells
        else:
            reference_design = fixed_seed_references[contrast_name]
            matrix = np.asarray([
                [
                    (
                        seed_metric_index[(task, seed, reference_design)]
                        - seed_metric_index[(task, seed, "vicreg_hacssmv8")]
                    ) / max(
                        abs(seed_metric_index[(task, seed, reference_design)]),
                        1e-12,
                    )
                    for seed in range(18001, 18006)
                ]
                for task in TASKS
            ], dtype=np.float64)
        if matrix.shape != (len(TASKS), 5):
            raise RuntimeError(
                f"invalid cell-effect matrix for {contrast_name}: {matrix.shape}"
            )
        seed_effects = matrix.mean(axis=0)
        values[f"{contrast_name}_SEED_POS"] = str(int(np.sum(seed_effects > 0.0)))
        values[f"{contrast_name}_SEED_MIN"] = pct(float(seed_effects.min()))
        values[f"{contrast_name}_SEED_MAX"] = pct(float(seed_effects.max()))

    phase_records = secondary["phase_equal_condition_mean"]
    for phase, prefix in (
        ("gap", "GAP"),
        ("deep", "PHASE_DEEP"),
        ("first_post", "FIRST_POST"),
        ("post", "POST"),
    ):
        record = phase_records[phase]
        values.update({
            f"{prefix}_MEAN": pct(record["mean_paired_relative_reduction"]),
            f"{prefix}_CI": ci(record),
            f"{prefix}_WINS": str(record["paired_wins"]),
            f"{prefix}_TASKS": str(record["task_mean_wins"]),
        })

    condition_records = secondary["condition_primary"]
    action_condition_effects = [
        condition_records[condition]["no_action_transport"][
            "mean_paired_relative_reduction"
        ]
        for condition in ("freeze", "gaussian_noise", "checkerboard", "long_freeze")
    ]
    values.update({
        "ACTION_CONDITION_MIN": pct(min(action_condition_effects)),
        "ACTION_CONDITION_MAX": pct(max(action_condition_effects)),
        "JOINT_FREEZE": pct(condition_records["freeze"]["single_read"][
            "mean_paired_relative_reduction"]),
        "JOINT_GAUSSIAN": pct(condition_records["gaussian_noise"]["single_read"][
            "mean_paired_relative_reduction"]),
        "JOINT_CHECKER": pct(condition_records["checkerboard"]["single_read"][
            "mean_paired_relative_reduction"]),
        "JOINT_LONG_FREEZE": pct(condition_records["long_freeze"]["single_read"][
            "mean_paired_relative_reduction"]),
    })

    metric_index = {
        (row["task"], int(row["seed"]), row["design"]): float(row[PRIMARY])
        for row in rows
    }
    design_ranks: dict[str, list[int]] = defaultdict(list)
    for task in TASKS:
        for seed in range(18001, 18006):
            ordered = sorted(
                DESIGNS,
                key=lambda design: (metric_index[(task, seed, design)], design),
            )
            for rank, design in enumerate(ordered, 1):
                design_ranks[design].append(rank)
    values.update({
        "SINGLE_RANK": f"{np.mean(design_ranks['vicreg_hacssmv8_single']):.2f}",
        "SAS_RANK": f"{np.mean(design_ranks['vicreg_hacssmv8']):.2f}",
        "DYNAMIC_RANK": f"{np.mean(design_ranks['vicreg_hacssmv8_dynamic']):.2f}",
        "SINGLE_FIRST": str(sum(rank == 1 for rank in design_ranks[
            "vicreg_hacssmv8_single"])),
        "SINGLE_TOP3": str(sum(rank <= 3 for rank in design_ranks[
            "vicreg_hacssmv8_single"])),
        "SAS_FIRST": str(sum(rank == 1 for rank in design_ranks[
            "vicreg_hacssmv8"])),
        "SAS_TOP3": str(sum(rank <= 3 for rank in design_ranks[
            "vicreg_hacssmv8"])),
    })

    by_design: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_design[row["design"]].append(row)
    design_both = []
    for design in DESIGNS:
        design_both.append(sum(
            float(row[RANK]) >= 16
            and abs(float(row[CONVERGENCE])) <= .05
            for row in by_design[design]
        ))
    sas_both = design_both[DESIGNS.index("vicreg_hacssmv8")]
    values.update({
        "DESIGN_BOTH_MIN": str(min(design_both)),
        "DESIGN_BOTH_MAX": str(max(design_both)),
        "SAS_BOTH": str(sas_both),
    })

    integrator_task_effects = []
    by_cell = {
        (row["task"], int(row["seed"]), row["design"]): row for row in rows
    }
    for task in TASKS:
        candidate = []
        reference = []
        for seed in range(18001, 18006):
            row = by_cell[(task, seed, "vicreg_hacssmv8")]
            candidate.append(float(row[PRIMARY]))
            reference.append(float(row[INTEGRATOR]))
        candidate_mean = float(np.mean(candidate))
        reference_mean = float(np.mean(reference))
        integrator_task_effects.append(
            (reference_mean - candidate_mean) / max(abs(reference_mean), 1e-12)
        )
    values.update({
        "INTEGRATOR_CLOSEST_TASK": pct(max(integrator_task_effects)),
        "INTEGRATOR_WORST_TASK": pct(min(integrator_task_effects)),
    })
    representation = report["representation"]["observed"]
    convergence = report["convergence"]["observed"]
    values.update({
        "VAR_MIN": scalar(representation["minimum_channel_variance"]),
        "VAR_PASS": str(representation["variance_passing_cells"]),
        "RANK_MIN": f"{representation['minimum_effective_rank']:.2f}",
        "RANK_PASS": str(representation["rank_passing_cells"]),
        "REP_GATE": gate(report["gates"]["healthy_representation"]),
        "CONV_MAX": pct(convergence["maximum_absolute_relative_change"]),
        "CONV_PASS": str(convergence["passing_cells"]),
        "CONV_GATE": gate(report["gates"]["convergence"]),
    })
    validity = []
    if report["gates"]["healthy_representation"]:
        validity.append("All 200 cells pass the representation-health guard")
    else:
        validity.append(
            "Representation health fails: minimum encoder mean-channel variance is "
            f"{scalar(representation['minimum_channel_variance'])} with "
            f"{representation['variance_passing_cells']}/200 variance-passing cells, while "
            f"minimum rank is {representation['minimum_effective_rank']:.2f} with "
            f"{representation['rank_passing_cells']}/200 rank-passing cells"
        )
    if report["gates"]["convergence"]:
        validity.append("all 200 cells pass the convergence guard")
    else:
        validity.append(
            f"convergence fails: {convergence['passing_cells']}/200 cells pass and the maximum "
            f"late-window change is {pct(convergence['maximum_absolute_relative_change'])}"
        )
    values["VALIDITY_TEXT"] = "; ".join(validity) + "."
    values.update({
        "ARTIFACT_TABLE": artifact_table(report, protocol, canonical_commands_sha256),
        "GATE_TABLE": gate_table(report),
        "AGGREGATE_EVIDENCE_TABLE": aggregate_evidence_table(report),
        "MAIN_TASK_TABLE": main_task_table(rows, report),
        "VALIDITY_TASK_TABLE": validity_by_task_table(rows),
        "SECONDARY_COMPONENT_TABLE": secondary_component_table(secondary),
        "SECONDARY_PHASE_TABLE": secondary_phase_table(secondary),
        "TASK_DESIGN_TABLE": task_design_table(rows),
        "TASK_RECURRENT_TABLE": task_recurrent_table(rows, report),
        "DESIGN_RANK_TABLE": design_rank_table(rows),
        "REPRESENTATION_TABLE": representation_table(rows),
        "INTEGRATOR_TABLE": integrator_table(rows),
        "CONTRAST_FULL_TABLE": contrast_full_table(report),
        "CLEAN_TASK_TABLE": clean_task_table(rows),
    })
    restart_repro, restart_appendix = common.restart_text(restart_audit)
    values.update({
        "RESTART_REPRO_TEXT": restart_repro,
        "RESTART_APPENDIX_TEXT": restart_appendix,
    })
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--restart-audit", type=Path, default=DEFAULT_RESTART_AUDIT)
    parser.add_argument(
        "--log-root", type=Path,
        help="optional private log root; when supplied, verify every audited log byte",
    )
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument(
        "--figure-manifest", type=Path,
        help="figure provenance manifest; defaults to OUTPUT_DIR/figures/fig_v18_manifest.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    template = args.template.resolve()
    output = args.output.resolve()
    report, protocol, rows, bundle = load_inputs(root)
    figure_manifest_path = (
        args.figure_manifest.resolve()
        if args.figure_manifest
        else output.parent / "figures" / "fig_v18_manifest.json"
    )
    secondary = load_secondary_figure_manifest(figure_manifest_path, bundle)
    restart_path = args.restart_audit.resolve()
    restart_audit = common.validate_restart_audit(
        restart_path,
        bundle,
        log_root=args.log_root.resolve() if args.log_root else None,
    )
    private_repository_root = root.parent.parent.resolve()
    command_replacements = tuple(pair for pair in (
        (str(root), "$RESULTS"),
        (str(private_repository_root), "$REPO"),
        (str(Path.home().resolve()), "$HOME"),
        (str(protocol.get("wandb_entity", "")), "anonymous-review-entity"),
        (str(protocol.get("wandb_project", "")), "anonymous-review-project"),
        (str(protocol.get("git_commit", "")), "WITHHELD_GIT_COMMIT"),
        (str(protocol.get("git_upstream_commit", "")), "WITHHELD_UPSTREAM_COMMIT"),
    ) if pair[0])
    _, _, canonical_commands_sha256 = common.canonical_redacted_commands(
        protocol.get("commands"), command_replacements
    )
    rendered_values = replacements(
        report, protocol, rows, restart_audit, canonical_commands_sha256, secondary
    )
    text = common.render_template(
        template.read_text(encoding="utf-8"),
        rendered_values,
        label="manuscript",
    )
    common.atomic_write_text(output, text)
    manifest_path = (
        args.manifest_output.resolve()
        if args.manifest_output
        else output.with_suffix(".manifest.json")
    )
    manifest = {
        "schema_version": 2,
        "scientific_label": report["scientific_label"],
        "analysis_sha256": bundle["hashes"]["confirmation_analysis.json"],
        "cells_sha256": report["cells_csv_sha256"],
        "contrasts_sha256": report["contrasts_csv_sha256"],
        "protocol_sha256": bundle["hashes"]["confirmation_protocol.json"],
        "runs_sha256": bundle["hashes"]["confirmation_runs.json"],
        "attempts_sha256": bundle["hashes"]["confirmation_attempts.json"],
        "summary_sha256": bundle["hashes"]["confirmation_summary.json"],
        "restart_audit_sha256": common.sha256(restart_path),
        "template_sha256": common.sha256(template),
        "renderer_sha256": common.sha256(Path(__file__).resolve()),
        "common_validator_sha256": common.sha256(Path(common.__file__).resolve()),
        "figure_manifest_sha256": common.sha256(figure_manifest_path),
        "manuscript_sha256": common.sha256(output),
        "restart_interruptions": common.restart_interruption_count(restart_audit),
        "telemetry_disclosure_present": (
            "per-step gate vectors and route weights were not retained" in text
        ),
        "llm_usage_statement_present": (
            text.count(LLM_USAGE_HEADING) == 1
            and text.count(LLM_USAGE_STATEMENT) == 1
        ),
        "canonical_commands_sha256": canonical_commands_sha256,
    }
    if not manifest["telemetry_disclosure_present"]:
        raise RuntimeError("manuscript lacks the registered telemetry disclosure")
    if not manifest["llm_usage_statement_present"]:
        raise RuntimeError("manuscript lacks the required LLM Usage Statement")
    common.atomic_write_json(manifest_path, manifest)
    print(json.dumps({
        "output": str(output),
        "manifest": str(manifest_path.resolve()),
        "scientific_label": report["scientific_label"],
        "placeholders": len(rendered_values),
        "characters": len(text),
    }, indent=2))


if __name__ == "__main__":
    main()
