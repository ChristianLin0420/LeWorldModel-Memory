#!/usr/bin/env python3
"""Create provenance-bound method and result figures for the V18 paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1783036800")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "result"
DEFAULT_FIGURES = ROOT / "generated" / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v18_release_common as common

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": .015,
})

PRIMARY = "heldout_prior_state_nmse"
CONDITIONS = ("freeze", "gaussian_noise", "checkerboard", "long_freeze")
CONDITION_LABELS = ("Freeze", "Gaussian noise", "Checkerboard", "Long freeze")
PHASES = ("gap", "deep", "first_post", "post")
PHASE_LABELS = ("Whole gap", "Deep gap", "First post", "Post")
TASKS = (
    "acrobot.swingup",
    "manipulator.bring_ball",
    "quadruped.run",
    "stacker.stack_4",
    "swimmer.swimmer15",
)
TASK_LABELS = (
    "Acrobot\nSwingup",
    "Manipulator\nBring Ball",
    "Quadruped\nRun",
    "Stacker\nStack-4",
    "Swimmer-15",
)
TASK_SHORT_LABELS = ("Acrobot", "Manipulator", "Quadruped", "Stacker", "Swimmer")
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
DESIGN_LABELS = (
    "No\ncarrier",
    "GRU",
    "Diagonal\nSSM",
    "Compact\nV8",
    "V8\nno-action",
    "V8\nsingle-read",
    "V8\nstatic",
    "V8\ndynamic",
)

FOREST = (
    (
        "Selected GRU / SSM",
        "vicreg_hacssmv8_vs_recurrent_envelope:heldout_prior_state_nmse",
        "v8_vs_per_cell_better_gru_ssm",
        ">=3%; CI>0; 18/25; 4/5",
    ),
    (
        "No carrier",
        "vicreg_hacssmv8_vs_vicreg_none:heldout_prior_state_nmse",
        "v8_vs_none",
        ">=5%; 20/25; 4/5",
    ),
    (
        "Deep: selected recurrence",
        "vicreg_hacssmv8_vs_recurrent_envelope:deep_prior_state_nmse",
        "deep_vs_per_cell_better_gru_ssm",
        "CI>0; 3/5",
    ),
    (
        "Legal integrator",
        "vicreg_hacssmv8_vs_checkpoint_integrator:heldout_prior_state_nmse",
        "v8_vs_checkpoint_integrator",
        ">=3%; 18/25; 4/5",
    ),
    (
        "No action transport",
        "vicreg_hacssmv8_vs_vicreg_hacssmv8_noaction:heldout_prior_state_nmse",
        "action_causality",
        ">=5%; CI>0; 18/25; 4/5",
    ),
    (
        "Single read",
        "vicreg_hacssmv8_vs_vicreg_hacssmv8_single:heldout_prior_state_nmse",
        "joint_state_use",
        ">=3%; CI>0; 18/25; 4/5",
    ),
    (
        "Endpoint envelope",
        "vicreg_hacssmv8_vs_endpoint_envelope:heldout_prior_state_nmse",
        "learned_v8_vs_static_dynamic_envelope_noninferiority",
        ">=-1%; CI>=-1%",
    ),
    (
        "Clean: selected recurrence",
        "vicreg_hacssmv8_vs_recurrent_envelope:clean_prior_state_nmse",
        "clean_prior_guard_vs_per_cell_better_gru_ssm",
        "degradation <=3%",
    ),
)

SECONDARY_POLICIES = (
    ("Selected\nGRU/SSM", "primary_selected_gru_ssm"),
    ("No\ncarrier", "no_carrier"),
    ("No action\ntransport", "no_action_transport"),
    ("Single\nread", "single_read"),
)

NVIDIA = "#76B900"
NVIDIA_DARK = "#4B780A"
NVIDIA_DEEP = "#2F4F05"
GRAPHITE = "#252A2E"
SLATE = "#626E76"
GRID = "#D8DEE2"
AMBER = "#A45A1C"
TEAL = "#28756D"
LIGHT_GREEN = "#F1F7E8"
LIGHT_GRAY = "#F5F6F7"
LIGHT_AMBER = "#FBF2E8"
LIGHT_TEAL = "#EDF7F5"

# Semantic aliases used throughout the plotting helpers.
INK = GRAPHITE
MUTED = SLATE


def load_report(root: Path) -> tuple[dict, list[dict[str, str]], dict]:
    bundle = common.load_complete_bundle(root)
    report = bundle["report"]
    rows = bundle["cells"]
    for _, contrast_key, gate_key, _ in FOREST:
        if contrast_key not in report.get("contrasts", {}):
            raise RuntimeError(f"missing registered contrast {contrast_key}")
        if gate_key not in report.get("gate_receipts", {}):
            raise RuntimeError(f"missing registered gate {gate_key}")
    return report, rows, bundle


def _crossed_bootstrap(values: np.ndarray, draws: int = 100_000) -> dict[str, float | int]:
    if values.shape != (len(TASKS), 5) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid descriptive crossed matrix {values.shape}")
    rng = np.random.default_rng(18018)
    chunks = []
    remaining = int(draws)
    while remaining:
        count = min(10_000, remaining)
        task_ids = rng.integers(0, values.shape[0], size=(count, values.shape[0]))
        seed_ids = rng.integers(0, values.shape[1], size=(count, values.shape[1]))
        sampled = values[task_ids[:, :, None], seed_ids[:, None, :]]
        chunks.append(sampled.mean(axis=(1, 2)))
        remaining -= count
    estimates = np.concatenate(chunks)
    return {
        "ci95_low": float(np.quantile(estimates, .025, method="linear")),
        "ci95_high": float(np.quantile(estimates, .975, method="linear")),
        "draws": int(draws),
        "seed": 18018,
    }


def descriptive_secondary(root: Path, report: dict, bundle: dict) -> dict:
    """Derive frozen secondary slices without redefining confirmation gates."""

    root = root.resolve()
    cell_index = {
        (row["task"], int(row["seed"]), row["design"]): float(row[PRIMARY])
        for row in bundle["cells"]
    }
    deep_index = {
        (row["task"], int(row["seed"]), row["design"]):
        float(row["deep_prior_state_nmse"])
        for row in bundle["cells"]
    }
    metrics: dict[tuple[str, int, str], dict] = {}
    manifest_rows = []
    public_rows = []
    for run in bundle["runs"]:
        key = (str(run["task"]), int(run["seed"]), str(run["design"]))
        directory = (root / Path(str(run["directory"])).name).resolve()
        if directory.parent != root:
            raise RuntimeError(f"invalid local run directory: {directory}")
        path = directory / "metrics.json"
        expected = run["artifact_sha256"]["metrics.json"]
        if not path.is_file() or common.sha256(path) != expected:
            raise RuntimeError(f"descriptive metrics hash differs for {key}")
        value = common.read_json(path)
        if not isinstance(value, dict):
            raise RuntimeError(f"descriptive metrics are malformed for {key}")
        if (str(value.get("design")), int(value.get("seed", -1))) != (key[2], key[1]):
            raise RuntimeError(f"descriptive metrics identity differs for {key}")
        observed_primary = float(value[PRIMARY])
        if not math.isclose(observed_primary, cell_index[key], rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"descriptive primary differs from public cell row for {key}")
        metrics[key] = value
        manifest_rows.append({
            "task": key[0], "seed": key[1], "design": key[2],
            "metrics_json_sha256": expected,
        })
        condition_rows = {}
        for condition in CONDITIONS:
            slices = {
                "primary": float(value[f"{condition}_prior_state_nmse"]),
                **{
                    phase: float(value[f"{condition}_prior_state_nmse_{phase}"])
                    for phase in PHASES
                },
            }
            if not all(math.isfinite(item) for item in slices.values()):
                raise RuntimeError(f"nonfinite descriptive slice for {key}/{condition}")
            condition_rows[condition] = slices
        primary_mean = float(np.mean([
            condition_rows[condition]["primary"] for condition in CONDITIONS
        ]))
        deep_mean = float(np.mean([
            condition_rows[condition]["deep"] for condition in CONDITIONS
        ]))
        if not math.isclose(primary_mean, observed_primary, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"condition-primary mean differs for {key}")
        if not math.isclose(deep_mean, deep_index[key], rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"condition-deep mean differs for {key}")
        public_rows.append({
            "task": key[0], "seed": key[1], "design": key[2],
            "conditions": condition_rows,
        })
    if len(metrics) != 200:
        raise RuntimeError(f"descriptive analysis requires 200 metrics files, got {len(metrics)}")
    manifest_rows.sort(key=lambda row: (row["task"], row["seed"], row["design"]))
    public_rows.sort(key=lambda row: (row["task"], row["seed"], row["design"]))

    selected = {}
    for task in TASKS:
        for seed in range(18001, 18006):
            selected[(task, seed)] = min(
                ("vicreg_gru", "vicreg_ssm"),
                key=lambda design: (metrics[(task, seed, design)][PRIMARY], design),
            )

    reference_design = {
        "no_carrier": lambda task, seed: "vicreg_none",
        "primary_selected_gru_ssm": lambda task, seed: selected[(task, seed)],
        "no_action_transport": lambda task, seed: "vicreg_hacssmv8_noaction",
        "single_read": lambda task, seed: "vicreg_hacssmv8_single",
    }

    def contrast(value_fn, policy: str) -> dict:
        matrix = np.empty((len(TASKS), 5), dtype=np.float64)
        candidate_by_task = {task: [] for task in TASKS}
        reference_by_task = {task: [] for task in TASKS}
        for ti, task in enumerate(TASKS):
            for si, seed in enumerate(range(18001, 18006)):
                candidate = value_fn(metrics[(task, seed, "vicreg_hacssmv8")])
                reference = value_fn(metrics[(task, seed, reference_design[policy](task, seed))])
                if not math.isfinite(candidate) or not math.isfinite(reference):
                    raise RuntimeError(f"nonfinite descriptive value for {task}/{seed}/{policy}")
                matrix[ti, si] = (reference - candidate) / max(abs(reference), 1e-12)
                candidate_by_task[task].append(candidate)
                reference_by_task[task].append(reference)
        task_effects = {
            task: float(
                (np.mean(reference_by_task[task]) - np.mean(candidate_by_task[task]))
                / max(abs(np.mean(reference_by_task[task])), 1e-12)
            )
            for task in TASKS
        }
        return {
            "reference_policy": policy,
            "mean_paired_relative_reduction": float(matrix.mean()),
            "paired_wins": int((matrix > 0).sum()),
            "paired_ties": int((matrix == 0).sum()),
            "pairs": 25,
            "task_mean_wins": int(sum(value > 0 for value in task_effects.values())),
            "task_effects": task_effects,
            "cell_effects": matrix.tolist(),
            "bootstrap": _crossed_bootstrap(matrix),
        }

    anchor = contrast(lambda row: float(row[PRIMARY]), "primary_selected_gru_ssm")
    registered = report["contrasts"][
        "vicreg_hacssmv8_vs_recurrent_envelope:heldout_prior_state_nmse"
    ]
    if not np.array_equal(
        np.asarray(anchor["cell_effects"]), np.asarray(registered["cell_effects"])
    ):
        raise RuntimeError("descriptive recurrent selector differs from registered selector")

    condition_primary = {
        condition: {
            policy: contrast(
                lambda row, condition=condition: float(
                    row[f"{condition}_prior_state_nmse"]
                ),
                policy,
            )
            for _, policy in SECONDARY_POLICIES
        }
        for condition in CONDITIONS
    }
    phase_equal_condition_mean = {
        phase: contrast(
            lambda row, phase=phase: float(np.mean([
                row[f"{condition}_prior_state_nmse_{phase}"]
                for condition in CONDITIONS
            ])),
            "primary_selected_gru_ssm",
        )
        for phase in PHASES
    }
    registered_deep = report["contrasts"][
        "vicreg_hacssmv8_vs_recurrent_envelope:deep_prior_state_nmse"
    ]
    if not np.array_equal(
        np.asarray(phase_equal_condition_mean["deep"]["cell_effects"]),
        np.asarray(registered_deep["cell_effects"]),
    ):
        raise RuntimeError("descriptive deep slice differs from registered deep contrast")
    return {
        "schema_version": 1,
        "artifact_kind": "v18_descriptive_secondary_analysis",
        "scientific_label": report["scientific_label"],
        "official_decision_changed": False,
        "decision_gates_defined": False,
        "multiplicity_adjusted": False,
        "claim_scope": "descriptive decomposition only",
        "frozen_grid": {
            "tasks": list(TASKS),
            "seeds": list(range(18001, 18006)),
            "designs": list(DESIGNS),
            "conditions": list(CONDITIONS),
            "phases": list(PHASES),
        },
        "estimand": {
            "effect_definition": "(reference-candidate)/max(abs(reference),1e-12)",
            "condition_primary_definition": (
                "sample-weighted mean over deep union first_post within condition"
            ),
            "phase_aggregation": (
                "arithmetic mean of four condition-level phase NMSEs within each cell"
            ),
            "reference_selection": (
                "GRU/SSM identity selected once per task-seed on overall heldout prior NMSE"
            ),
            "phase_masks": {
                "gap": "b <= t < e",
                "deep": "b+H <= t < e",
                "first_post": "t == e",
                "post": "e < t <= e+H",
                "history": 3,
            },
        },
        "registration_status": (
            "condition and phase metrics were frozen secondary reports; these "
            "decompositions reuse the registered reference and bootstrap policies "
            "but define no confirmation gate"
        ),
        "bootstrap": {
            "method": "crossed_task_by_optimizer_seed_percentile",
            "generator": "numpy.random.Generator(PCG64)",
            "seed": 18018,
            "draws": 100_000,
            "percentiles": [0.025, 0.975],
            "quantile_method": "linear",
        },
        "input_file_sha256": bundle["hashes"],
        "input_artifact_manifest_sha256": report["input_artifact_manifest_sha256"],
        "metrics_json_manifest_sha256": common.json_sha256(manifest_rows),
        "condition_primary": condition_primary,
        "phase_equal_condition_mean": phase_equal_condition_mean,
        "public_prior_slices": public_rows,
    }


def _flag(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"frozen command lacks {name}") from exc


def _assert_architecture_contract(protocol: dict) -> None:
    commands = protocol.get("commands")
    if not isinstance(commands, list) or len(commands) != 200:
        raise RuntimeError("protocol lacks the frozen command expansion")
    expected = {
        "--history-len": "3",
        "--embed-dim": "128",
        "--img-size": "64",
        "--patch-size": "8",
        "--encoder-layers": "6",
        "--encoder-heads": "4",
        "--predictor-layers": "4",
        "--predictor-heads": "8",
        "--dropout": "0.1",
        "--sigreg-lambda": "0.0",
        "--epochs": "100",
        "--eval-target-key": "task_observation",
    }
    counts = {design: 0 for design in DESIGNS}
    for command in commands:
        argv = command.get("argv")
        if not isinstance(argv, list):
            raise RuntimeError("protocol command argv is malformed")
        observed = {name: _flag(argv, name) for name in expected}
        if observed != expected:
            raise RuntimeError(f"architecture figure contract differs: {observed}")
        design = _flag(argv, "--design")
        if design != command.get("design") or design not in counts:
            raise RuntimeError(f"architecture design identity differs: {design}")
        counts[design] += 1
    if set(counts.values()) != {25}:
        raise RuntimeError(f"architecture design grid differs: {counts}")


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], *,
           color: str = INK, linestyle: str = "-", width: float = 1.1,
           mutation: float = 8.0, zorder: int = 2, rad: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation,
        linewidth=width, color=color, linestyle=linestyle,
        transform=ax.transAxes, zorder=zorder,
        connectionstyle=f"arc3,rad={rad}",
    ))


def _node(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    *,
    edge: str = INK,
    face: str = "white",
    dashed: bool = False,
    title_size: float = 7.0,
    subtitle_size: float = 6.2,
    shadow: bool = False,
) -> None:
    """Draw a restrained academic diagram node with title/subtitle hierarchy."""

    if shadow:
        ax.add_patch(FancyBboxPatch(
            (x + .0055, y + .0075), width, height,
            boxstyle="round,pad=0.004,rounding_size=0.003",
            linewidth=.9, edgecolor=edge, facecolor="white",
            transform=ax.transAxes, zorder=2,
        ))
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.004,rounding_size=0.003",
        linewidth=1.0, edgecolor=edge, facecolor=face,
        linestyle="--" if dashed else "-", transform=ax.transAxes, zorder=3,
    )
    ax.add_patch(patch)
    center = x + width / 2
    if subtitle:
        sub_lines = subtitle.count("\n") + 1
        title_frac = .70 if sub_lines > 1 else .63
        sub_frac = .33 if sub_lines > 1 else .28
        ax.text(
            center, y + height * title_frac, title, ha="center", va="center",
            fontsize=title_size, weight="bold", color=INK,
            transform=ax.transAxes, zorder=4,
        )
        ax.text(
            center, y + height * sub_frac, subtitle, ha="center", va="center",
            fontsize=subtitle_size, color=MUTED, linespacing=1.25,
            transform=ax.transAxes, zorder=4,
        )
    else:
        ax.text(
            center, y + height / 2, title, ha="center", va="center",
            fontsize=title_size, weight="bold", color=INK,
            transform=ax.transAxes, zorder=4,
        )


def plot_architecture(protocol: dict, output: Path) -> None:
    """Draw the host/carrier dependency graph and the annotated SAS-PC step."""

    _assert_architecture_contract(protocol)
    fig, ax = plt.subplots(figsize=(5.5, 4.05))
    fig.subplots_adjust(left=.005, right=.995, bottom=.005, top=.995)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.plot([.015, .985], [.545, .545], color=GRID, linewidth=.8, zorder=0)
    ax.text(.015, .968, "(a) Pixel JEPA host with an episode-persistent carrier",
            fontsize=8.6, weight="bold", color=INK, va="bottom")
    ax.text(.015, .500, r"(b) One SAS-PC step: predict $\rightarrow$ correct"
            r" $\rightarrow$ read", fontsize=8.6, weight="bold", color=INK,
            va="bottom")

    # ---------------- Panel (a): host with persistent carrier ----------------
    # Corrupted lane (top) and clean lane (bottom) around two tied encoders.
    _node(ax, .015, .825, .12, .085, "Corrupted view",
          "$o^c_{1:L}$\n64×64 RGB", edge=AMBER, face=LIGHT_AMBER,
          title_size=6.9, subtitle_size=6.0)
    _node(ax, .165, .825, .12, .085, r"Encoder $E_\theta$",
          "6 layers · 4 heads\n8×8 patches", edge=GRAPHITE, face=LIGHT_GRAY,
          title_size=6.9, subtitle_size=6.0)
    _node(ax, .015, .585, .12, .08, "Clean view", "$o_{1:L}$",
          edge=TEAL, face=LIGHT_TEAL, title_size=6.9, subtitle_size=6.2)
    _node(ax, .165, .585, .12, .08, r"Encoder $E_\theta$", "shared weights",
          edge=GRAPHITE, face=LIGHT_GRAY, title_size=6.9, subtitle_size=6.0)
    ax.plot([.225, .225], [.665, .825], color=SLATE, linewidth=.8,
            linestyle=(0, (2, 2)), zorder=1)
    ax.text(.218, .745, "shared weights\nno dropout", fontsize=5.9, color=MUTED,
            ha="right", va="center", linespacing=1.25, style="italic",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                  "edgecolor": "none"})
    _arrow(ax, (.135, .8675), (.165, .8675), color=AMBER, width=.9, mutation=6)
    _arrow(ax, (.135, .625), (.165, .625), color=TEAL, width=.9, mutation=6)

    # Corrupted latent token strip with the finite H=3 window bracket.
    token_x = [.315 + .039 * index for index in range(6)]
    token_w, token_y0, token_h = .031, .827, .068
    for index, tx in enumerate(token_x):
        old = index < 3
        ax.add_patch(FancyBboxPatch(
            (tx, token_y0), token_w, token_h,
            boxstyle="round,pad=0.002,rounding_size=0.004",
            linewidth=.8, edgecolor=SLATE if old else AMBER,
            facecolor="white" if old else LIGHT_AMBER,
            linestyle=(0, (2, 1.6)) if old else "-",
            alpha=.5 if old else 1.0, zorder=3,
        ))
    for tx, tick in zip(token_x[3:], ("$z^c_{t-2}$", "$z^c_{t-1}$", "$z^c_t$")):
        ax.text(tx + token_w / 2, token_y0 + token_h / 2, tick, fontsize=6.1,
                ha="center", va="center", color=INK, zorder=4)
    ax.text(.333, token_y0 + token_h / 2, "…", fontsize=7.0, ha="center",
            va="center", color=SLATE, alpha=.75, zorder=4)
    ax.text(.315, .908, "$z^c$ stream", fontsize=6.2, color=MUTED, va="bottom")
    ax.text(.3695, .812, "older latents discarded", fontsize=5.8, color=SLATE,
            ha="center", va="top", style="italic")
    # Window bracket over the newest three tokens.
    bx0, bx1, by = token_x[3] - .004, token_x[5] + token_w + .004, .906
    ax.plot([bx0, bx0, bx1, bx1], [by, by + .012, by + .012, by],
            color=GRAPHITE, linewidth=.9, zorder=3)
    ax.text((bx0 + bx1) / 2, .922, "sliding window ($H{=}3$)", fontsize=6.2,
            color=INK, ha="center", va="bottom", weight="bold")
    _arrow(ax, (.285, .861), (.313, .861), color=AMBER, width=.9, mutation=6)

    # Predictor and loss.
    _node(ax, .60, .80, .19, .115, r"Causal predictor $P_\phi$",
          "action-conditioned\n4 layers · 8 heads", edge=GRAPHITE,
          face=LIGHT_GRAY, title_size=6.9, subtitle_size=6.0)
    _node(ax, .845, .68, .14, .115, "Prediction loss",
          "MSE/$D$ + var + cov\nclean targets only", edge=TEAL,
          face=LIGHT_TEAL, title_size=6.9, subtitle_size=5.9)
    _arrow(ax, (bx1 + .004, .861), (.60, .861), color=GRAPHITE, width=.9,
           mutation=6)
    _arrow(ax, (.79, .845), (.868, .795), color=GRAPHITE, width=.9, mutation=6)
    ax.text(.826, .833, r"$\hat z_{t+1}$", fontsize=6.2, color=MUTED,
            ha="center", va="bottom")

    # Episode-persistent carrier attached to the latent stream.
    _node(ax, .31, .625, .19, .115, "SAS-PC carrier",
          r"$\tau=(2,8)$ tracks", edge=NVIDIA_DARK, face=LIGHT_GREEN,
          title_size=6.9, subtitle_size=6.0)
    ax.add_patch(FancyArrowPatch(
        (.43, .740), (.345, .740), arrowstyle="-|>", mutation_scale=6,
        linewidth=1.1, color=NVIDIA_DARK, connectionstyle="arc3,rad=1.0",
        zorder=2,
    ))
    ax.text(.325, .792, r"episode-persistent state $m_t$ ($2{\times}D$ floats)",
            fontsize=6.0, color=NVIDIA_DARK, ha="center", va="bottom",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                  "edgecolor": "none"})
    _arrow(ax, (token_x[5] + token_w / 2, .825), (.468, .742),
           color=NVIDIA_DARK, width=.9, mutation=6)
    _arrow(ax, (.50, .722), (.612, .798), color=NVIDIA_DARK, width=1.0,
           mutation=6)
    ax.text(.548, .770, r"read $\tilde z_t$", fontsize=6.2, color=NVIDIA_DARK,
            ha="center", va="bottom", rotation=27)

    # Actions feed both the carrier and the predictor.
    _node(ax, .585, .655, .135, .058, r"Actions $a_{t-1}$", edge=AMBER,
          face=LIGHT_AMBER, title_size=6.6)
    _arrow(ax, (.585, .684), (.502, .684), color=AMBER, width=.9, mutation=6)
    _arrow(ax, (.6525, .713), (.6525, .798), color=AMBER, width=.9, mutation=6)

    # Clean targets close the loss path.
    _node(ax, .845, .555, .14, .08, "Clean targets", r"$z^\star_{t+1}$",
          edge=TEAL, face=LIGHT_TEAL, title_size=6.9, subtitle_size=6.2)
    _arrow(ax, (.285, .605), (.843, .592), color=TEAL, width=.9, mutation=6)
    _arrow(ax, (.915, .637), (.915, .678), color=TEAL, width=.9, mutation=6)

    # ---------------- Panel (b): one annotated SAS-PC step -------------------
    def _stage(x: float, width: float, label: str) -> None:
        ax.add_patch(FancyBboxPatch(
            (x, .400), width, .035,
            boxstyle="round,pad=0.004,rounding_size=0.003",
            linewidth=0, facecolor=NVIDIA_DARK, zorder=5,
        ))
        ax.text(x + width / 2, .4175, label, fontsize=6.4, weight="bold",
                color="white", ha="center", va="center", zorder=6)

    # Track states carried across the episode (two parallel tracks).
    _node(ax, .015, .330, .09, .075, r"$m^k_{t-1}$", edge=SLATE,
          face=LIGHT_GRAY, title_size=7.2, shadow=True)
    ax.text(.064, .316, "two tracks $k$:\nfast $\\tau{=}2$ · med $\\tau{=}8$",
            fontsize=5.9, color=MUTED, ha="center", va="top", linespacing=1.3)

    # Predict stage.
    _node(ax, .135, .270, .29, .165, "", edge=NVIDIA_DARK, face=LIGHT_GREEN,
          shadow=True)
    _stage(.135, .29, "PREDICT")
    ax.text(.28, .376, r"$(d,\,v) = W_a\, a_{t-1}$  (shared map)",
            fontsize=6.3, ha="center", va="center", color=INK, zorder=5)
    ax.text(.28, .341, r"$p^k_t = m^k_{t-1}\, +$", fontsize=6.3, ha="center",
            va="center", color=INK, zorder=5)
    ax.text(.28, .303, r"$\beta_k \tanh\!\left(v + d \odot \mathrm{LN}(m^k_{t-1})\right)$",
            fontsize=6.3, ha="center", va="center", color=INK, zorder=5)

    # Correct stage with the static-dynamic gate slider.
    _node(ax, .455, .270, .245, .165, "", edge=NVIDIA_DARK, face=LIGHT_GREEN,
          shadow=True)
    _stage(.455, .245, "CORRECT")
    ax.text(.5775, .379, r"$g^k_t = (1{-}\rho_k)\,\sigma(b_k) + \rho_k\, q^k_t$",
            fontsize=6.3, ha="center", va="center", color=INK, zorder=5)
    ax.text(.5775, .346, r"$m^k_t = p^k_t + \beta_k\, g^k_t\,(x_t - p^k_t)$",
            fontsize=6.3, ha="center", va="center", color=INK, zorder=5)
    ax.plot([.508, .648], [.303, .303], color=SLATE, linewidth=.8, zorder=5)
    ax.scatter([.578], [.303], s=8, color=NVIDIA_DARK, zorder=6)
    ax.text(.578, .310, r"$\rho_k$", fontsize=5.9, ha="center", va="bottom",
            color=NVIDIA_DARK, zorder=6)
    ax.text(.505, .296, "static", fontsize=5.6, ha="left", va="top",
            color=MUTED, zorder=6)
    ax.text(.650, .296, "dynamic", fontsize=5.6, ha="right", va="top",
            color=MUTED, zorder=6)

    # Read stage.
    _node(ax, .73, .270, .255, .165, "", edge=NVIDIA_DARK, face=LIGHT_GREEN)
    _stage(.73, .255, "READ")
    ax.text(.8575, .376, r"route $\pi = \mathrm{softmax}(\ell)$",
            fontsize=6.3, ha="center", va="center", color=INK, zorder=5)
    ax.text(.8575, .341, r"RMSNorm $\rightarrow\ W_o$", fontsize=6.3,
            ha="center", va="center", color=INK, zorder=5)
    ax.text(.8575, .303, r"$\tilde z_t = z^c_t + W_o\, r_t$", fontsize=6.3,
            ha="center", va="center", color=INK, zorder=5)

    _arrow(ax, (.105, .3675), (.133, .3675), color=SLATE, width=.9, mutation=6)
    _arrow(ax, (.425, .350), (.453, .350), color=NVIDIA_DARK, width=1.0,
           mutation=6)
    ax.text(.439, .356, "$p^k_t$", fontsize=6.0, ha="center", va="bottom",
            color=NVIDIA_DARK)
    _arrow(ax, (.70, .350), (.728, .350), color=NVIDIA_DARK, width=1.0,
           mutation=6)
    ax.text(.714, .356, "$m^k_t$", fontsize=6.0, ha="center", va="bottom",
            color=NVIDIA_DARK)

    # Inputs from below: previous action and current corrupted latent.
    _node(ax, .20, .155, .115, .058, r"$a_{t-1}$", edge=AMBER,
          face=LIGHT_AMBER, title_size=7.0)
    _arrow(ax, (.2575, .213), (.2575, .268), color=AMBER, width=.9, mutation=6)
    _node(ax, .50, .155, .115, .058, r"$z^c_t$", edge=AMBER,
          face=LIGHT_AMBER, title_size=7.0)
    _arrow(ax, (.5575, .213), (.5575, .268), color=AMBER, width=.9, mutation=6)
    ax.text(.572, .238, r"$x_t = W_x z^c_t$", fontsize=6.1, ha="left",
            va="center", color=AMBER)

    # Episode recurrence: the posterior is carried over the top to t+1.
    ax.plot([.578, .578, .06, .06], [.4425, .468, .468, .410], color=NVIDIA_DARK,
            linewidth=1.0, zorder=1)
    _arrow(ax, (.06, .418), (.06, .407), color=NVIDIA_DARK, width=1.0,
           mutation=6)
    ax.text(.32, .474, r"carry $m^k_t$ across the episode", fontsize=6.2,
            color=NVIDIA_DARK, ha="center", va="bottom")

    # Dashed, clearly separated evaluation-only tap.
    ax.add_patch(FancyBboxPatch(
        (.015, .005), .97, .115,
        boxstyle="round,pad=0.004,rounding_size=0.006",
        linewidth=.9, edgecolor=SLATE, facecolor="#FBFCFC",
        linestyle=(0, (3, 2)), zorder=1,
    ))
    ax.text(.095, .0625, "evaluation-only\ntap (no training\nsignal)",
            fontsize=5.9, color=MUTED, ha="center", va="center",
            style="italic", linespacing=1.3)
    _node(ax, .185, .028, .235, .068, "Routed prior",
          "action-transported, pre-obs.", edge=MUTED, dashed=True,
          title_size=6.5, subtitle_size=5.8)
    _node(ax, .475, .028, .20, .068, "Ridge probe", "fit after training",
          edge=MUTED, dashed=True, title_size=6.5, subtitle_size=5.8)
    _node(ax, .73, .028, .20, .068, "Prior-state NMSE", "held-out episodes",
          edge=MUTED, dashed=True, title_size=6.5, subtitle_size=5.8)
    _arrow(ax, (.35, .268), (.35, .098), color=MUTED, linestyle="--", width=.8,
           mutation=5)
    _arrow(ax, (.42, .062), (.473, .062), color=MUTED, linestyle="--",
           width=.8, mutation=5)
    _arrow(ax, (.675, .062), (.728, .062), color=MUTED, linestyle="--",
           width=.8, mutation=5)

    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white",
                metadata={"Software": "V18 release plotter"})
    fig.savefig(
        output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white",
        metadata={"Creator": "V18 release plotter", "CreationDate": None,
                  "ModDate": None},
    )
    plt.close(fig)


def plot_evidence(report: dict, output: Path) -> None:
    """Render the registered contrasts as an annotated forest plot."""

    raw_records = []
    for label, contrast_key, gate_key, _ in FOREST:
        contrast = report["contrasts"][contrast_key]
        gate = report["gate_receipts"][gate_key]
        requirement = gate["thresholds"].get("minimum_mean_paired_relative_reduction")
        raw_records.append({
            "label": label,
            "estimate": 100.0 * contrast["mean_paired_relative_reduction"],
            "low": 100.0 * contrast["bootstrap"]["ci95_low"],
            "high": 100.0 * contrast["bootstrap"]["ci95_high"],
            "wins": f"{contrast['paired_wins']}/25 · {contrast['task_mean_wins']}/5",
            "requirement": None if requirement is None else 100.0 * requirement,
            "passed": bool(gate["passed"]),
        })
    order = (0, 1, 3, 2, 7, 4, 5, 6)
    records = [raw_records[index] for index in order]
    y = np.asarray((9.0, 8.0, 7.0, 5.6, 4.6, 3.2, 2.2, 1.2))

    fig, ax = plt.subplots(figsize=(5.5, 2.65), constrained_layout=True)
    ax.axhspan(6.55, 9.48, color=LIGHT_GREEN, alpha=.58, zorder=-3)
    ax.axhspan(4.08, 6.05, color=LIGHT_GRAY, alpha=.78, zorder=-3)
    ax.axhspan(.72, 3.55, color=LIGHT_GREEN, alpha=.32, zorder=-3)
    for yi, row in zip(y, records, strict=True):
        if row["requirement"] is not None:
            ax.vlines(row["requirement"], yi - .30, yi + .30, color=GRAPHITE,
                      linewidth=.7, linestyle=(0, (1.6, 1.4)), zorder=2)
            ax.scatter(row["requirement"], yi + .38, marker="v", s=13,
                       facecolor="white", edgecolor=GRAPHITE, linewidth=.7,
                       zorder=3)
        ax.hlines(yi, row["low"], row["high"], color=SLATE, linewidth=1.2,
                  zorder=3)
        ax.vlines((row["low"], row["high"]), yi - .11, yi + .11,
                  color=SLATE, linewidth=.9, zorder=3)
        ax.scatter(row["estimate"], yi, s=58, color="white", edgecolor="none",
                   zorder=4)
        ax.scatter(row["estimate"], yi, s=30, color=NVIDIA,
                   edgecolor=NVIDIA_DEEP, linewidth=.55, zorder=5)
        ax.text(36, yi, f"{row['estimate']:+.1f}", va="center", ha="right",
                fontsize=7.0, color=INK)
        ax.text(60, yi, row["wins"], va="center", ha="right", fontsize=7.0,
                color=MUTED)
        verdict = "PASS" if row["passed"] else "FAIL"
        ax.text(70, yi, verdict, va="center", ha="center", fontsize=6.3,
                weight="bold", color=NVIDIA_DARK if row["passed"] else AMBER,
                bbox={"boxstyle": "round,pad=0.24",
                      "facecolor": LIGHT_GREEN if row["passed"] else LIGHT_AMBER,
                      "edgecolor": NVIDIA_DARK if row["passed"] else AMBER,
                      "linewidth": .6})
    ax.axvline(0, color=GRAPHITE, linewidth=.8, linestyle="--")
    for separator in (6.3, 3.9):
        ax.axhline(separator, color=GRID, linewidth=.65)
    ax.text(-69, 9.62, "CARRIER COMPARISONS", fontsize=6.6, color=NVIDIA_DARK,
            weight="bold")
    ax.text(-69, 6.18, "PERSISTENCE / CLEAN", fontsize=6.6, color=NVIDIA_DARK,
            weight="bold")
    ax.text(-69, 3.78, "COMPONENTS", fontsize=6.6, color=NVIDIA_DARK,
            weight="bold")
    header_y = 10.28
    ax.text(36, header_y, "effect", fontsize=6.8, color=MUTED, ha="right",
            weight="bold", va="center")
    ax.text(60, header_y, "cell · task wins", fontsize=6.8, color=MUTED,
            ha="right", weight="bold", va="center")
    ax.text(70, header_y, "verdict", fontsize=6.8, color=MUTED, ha="center",
            weight="bold", va="center")
    ax.plot((26.5, 76), (9.88, 9.88), color=GRID, linewidth=.7,
            clip_on=False)
    ax.text(-69, .62, r"$\triangledown$ registered mean requirement",
            fontsize=6.0, color=MUTED, va="center")
    ax.set_yticks(y, [row["label"] for row in records], fontsize=7.3)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.set_xlim(-70, 76)
    ax.set_ylim(.30, 10.62)
    ax.set_xticks((-60, -40, -20, 0, 20))
    ax.grid(axis="x", color=GRID, linewidth=.5)
    ax.set_xlabel(
        "paired relative reduction in prior-state NMSE (%)   ·   positive favors SAS-PC",
        fontsize=7.2,
    )
    ax.set_title("Registered effect estimates and crossed 95% intervals", loc="left",
                 fontsize=8.6, weight="bold", color=INK, pad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(output, dpi=300, facecolor="white",
                metadata={"Software": "V18 release plotter"})
    fig.savefig(
        output.with_suffix(".pdf"), facecolor="white",
        metadata={"Creator": "V18 release plotter", "CreationDate": None,
                  "ModDate": None},
    )
    plt.close(fig)


def _phase_timeline(ax) -> None:
    """Draw a miniature episode timeline that defines the four phase masks."""

    ax.set_xlim(0, 1)
    ax.set_ylim(-.14, 1.04)
    ax.axis("off")
    dark_amber = "#7A4113"
    ax.plot([.02, .93], [.38, .38], color=SLATE, linewidth=.8, zorder=1)
    ax.add_patch(FancyArrowPatch(
        (.90, .38), (.945, .38), arrowstyle="-|>", mutation_scale=5,
        linewidth=.8, color=SLATE, zorder=1,
    ))
    ax.text(.958, .38, "$t$", fontsize=5.9, color=MUTED, ha="left",
            va="center")
    b, bh, e, eh = .20, .36, .62, .80
    ax.add_patch(Rectangle(
        (b, .20), e - b, .36, facecolor=LIGHT_AMBER, edgecolor=AMBER,
        linewidth=.6, zorder=0,
    ))
    ax.add_patch(Rectangle(
        (bh, .20), e - bh, .36, facecolor=AMBER, alpha=.30, edgecolor="none",
        zorder=0,
    ))
    ax.text((b + bh) / 2, .64, "gap $[b,e)$", fontsize=5.8, color=dark_amber,
            ha="center", va="bottom")
    ax.text((bh + e) / 2 - .015, .38, "deep $[b{+}H,e)$", fontsize=5.6,
            color=dark_amber, ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.13", "facecolor": "white",
                  "alpha": .8, "edgecolor": "none"})
    ax.plot([e, e], [.04, .60], color=GRAPHITE, linewidth=.9, zorder=2)
    ax.text(e, -.03, "first post $t{=}e$", fontsize=5.7, color=INK,
            ha="center", va="top")
    ax.plot([e + .015, e + .015, eh, eh], [.60, .70, .70, .60], color=GRAPHITE,
            linewidth=.8, zorder=2)
    ax.text((e + eh) / 2 + .008, .76, "post $(e, e{+}H]$", fontsize=5.7,
            color=INK, ha="center", va="bottom")
    ax.text(b, .12, "$b$", fontsize=5.9, color=MUTED, ha="center", va="top")
    ax.text(bh, .12, "$b{+}H$", fontsize=5.9, color=MUTED, ha="center",
            va="top")


def plot_secondary(report: dict, secondary: dict, output: Path) -> None:
    """Show task, corruption, and phase heterogeneity without new gates."""

    registered = report["contrasts"][FOREST[0][1]]
    cell_effects = 100.0 * np.asarray(registered["cell_effects"], dtype=np.float64)
    if cell_effects.shape != (len(TASKS), 5):
        raise RuntimeError(f"invalid registered task/seed matrix {cell_effects.shape}")
    task_effects = np.asarray(
        [100.0 * registered["task_effects"][task] for task in TASKS],
        dtype=np.float64,
    )
    recurrent_conditions = [
        secondary["condition_primary"][condition]["primary_selected_gru_ssm"]
        for condition in CONDITIONS
    ]
    recurrent_phases = [
        secondary["phase_equal_condition_mean"][phase] for phase in PHASES
    ]

    fig = plt.figure(figsize=(5.5, 3.15), constrained_layout=True)
    grid = fig.add_gridspec(
        3, 2, height_ratios=(1.10, .40, 1.0), width_ratios=(1.08, 1.0),
        hspace=.06, wspace=.12,
    )
    task_ax = fig.add_subplot(grid[0, :])
    condition_ax = fig.add_subplot(grid[1:, 0])
    timeline_ax = fig.add_subplot(grid[1, 1])
    phase_ax = fig.add_subplot(grid[2, 1])

    # Exact task-by-seed observations plus the task-level ratio-of-means effect.
    task_y = np.arange(len(TASKS))[::-1]
    seed_jitter = np.linspace(-.13, .13, 5)
    for ti, yi in enumerate(task_y):
        values = cell_effects[ti]
        task_ax.hlines(
            yi, values.min(), values.max(), color=SLATE, linewidth=.7,
            alpha=.55, zorder=1,
        )
        task_ax.scatter(
            values, yi + seed_jitter, s=17, facecolor="white",
            edgecolor=SLATE, linewidth=.75, zorder=2,
        )
        task_ax.scatter(
            task_effects[ti], yi, s=35, marker="D", color=NVIDIA,
            edgecolor=NVIDIA_DEEP, linewidth=.5, zorder=3,
        )
        task_ax.text(
            31.8, yi, f"{task_effects[ti]:+.1f} ({np.sum(values > 0)}/5)",
            ha="right", va="center",
            fontsize=6.8, color=MUTED,
        )
    task_ax.axvline(0, color=GRAPHITE, linewidth=.75, linestyle="--")
    task_ax.set_yticks(task_y, TASK_SHORT_LABELS, fontsize=7.2)
    task_ax.set_xlim(-38, 33)
    task_ax.set_xticks((-30, -20, -10, 0, 10, 20))
    task_ax.set_ylim(-.55, len(TASKS) - .25)
    task_ax.grid(axis="x", color=GRID, linewidth=.5)
    task_ax.tick_params(axis="x", labelsize=6.8)
    task_ax.tick_params(axis="y", length=0)
    task_ax.set_title("(a) Task and seed heterogeneity", loc="left", fontsize=8.3,
                      weight="bold", color=INK, pad=4)
    seed_handle = task_ax.scatter(
        [], [], s=17, facecolor="white", edgecolor=SLATE, linewidth=.75,
    )
    mean_handle = task_ax.scatter(
        [], [], s=28, marker="D", color=NVIDIA, edgecolor=NVIDIA_DEEP, linewidth=.5,
    )
    legend = task_ax.legend(
        (seed_handle, mean_handle), ("seed cell", "task effect"),
        loc="lower left", bbox_to_anchor=(.005, .02), ncol=2, frameon=True,
        handletextpad=.25, columnspacing=.75, borderaxespad=0, fontsize=6.3,
        labelcolor=MUTED, facecolor="white", edgecolor=GRID, framealpha=1.0,
        borderpad=.45,
    )
    legend.set_zorder(6)
    task_ax.text(31.8, 4.42, "task effect (wins)", ha="right", fontsize=6.3,
                 weight="bold", color=MUTED)
    task_ax.spines[["top", "right", "left"]].set_visible(False)

    # Corruption slices use the same descriptive crossed-resampling recipe.
    condition_y = np.arange(len(CONDITIONS))[::-1]
    for yi, record in zip(condition_y, recurrent_conditions, strict=True):
        estimate = 100.0 * record["mean_paired_relative_reduction"]
        low = 100.0 * record["bootstrap"]["ci95_low"]
        high = 100.0 * record["bootstrap"]["ci95_high"]
        condition_ax.hlines(yi, low, high, color=SLATE, linewidth=1.15)
        condition_ax.vlines((low, high), yi - .09, yi + .09,
                            color=SLATE, linewidth=.8)
        condition_ax.scatter(
            estimate, yi, s=27, color=NVIDIA, edgecolor=NVIDIA_DEEP,
            linewidth=.5, zorder=3,
        )
        condition_ax.text(32.0, yi, f"{estimate:+.1f}", ha="right", va="center",
                          fontsize=6.9, color=INK)
    condition_ax.axvline(0, color=GRAPHITE, linewidth=.75, linestyle="--")
    condition_ax.set_yticks(condition_y, CONDITION_LABELS, fontsize=7.1)
    condition_ax.set_xlim(-55, 33)
    condition_ax.set_xticks((-50, -25, 0, 25))
    condition_ax.set_ylim(-.55, len(CONDITIONS) - .25)
    condition_ax.grid(axis="x", color=GRID, linewidth=.5)
    condition_ax.tick_params(axis="x", labelsize=6.8)
    condition_ax.tick_params(axis="y", length=0)
    condition_ax.set_title("(b) Corruption-specific effects", loc="left",
                           fontsize=8.1, weight="bold", color=INK, pad=4)
    condition_ax.text(32.0, 3.42, "effect", ha="right", fontsize=6.3,
                      weight="bold", color=MUTED)
    condition_ax.spines[["top", "right", "left"]].set_visible(False)

    # Miniature timeline glyph defining the phase masks, then the phase slices.
    _phase_timeline(timeline_ax)
    timeline_ax.set_title("(c) Phase-specific effects", loc="left",
                          fontsize=8.1, weight="bold", color=INK, pad=3)

    # Equal-condition phase slices distinguish transport from recovery behavior.
    phase_y = np.arange(len(PHASES))[::-1]
    for yi, record in zip(phase_y, recurrent_phases, strict=True):
        estimate = 100.0 * record["mean_paired_relative_reduction"]
        low = 100.0 * record["bootstrap"]["ci95_low"]
        high = 100.0 * record["bootstrap"]["ci95_high"]
        phase_ax.hlines(yi, low, high, color=SLATE, linewidth=1.15)
        phase_ax.vlines((low, high), yi - .09, yi + .09,
                        color=SLATE, linewidth=.8)
        phase_ax.scatter(
            estimate, yi, s=27, color=NVIDIA, edgecolor=NVIDIA_DEEP,
            linewidth=.5, zorder=3,
        )
        phase_ax.text(19.0, yi, f"{estimate:+.1f}", ha="right", va="center",
                      fontsize=6.9, color=INK)
    phase_ax.axvline(0, color=GRAPHITE, linewidth=.75, linestyle="--")
    phase_ax.set_yticks(phase_y, PHASE_LABELS, fontsize=7.1)
    phase_ax.set_xlim(-20, 20)
    phase_ax.set_xticks((-20, -10, 0, 10, 20))
    phase_ax.set_ylim(-.55, len(PHASES) - .25)
    phase_ax.grid(axis="x", color=GRID, linewidth=.5)
    phase_ax.tick_params(axis="x", labelsize=6.8)
    phase_ax.tick_params(axis="y", length=0)
    phase_ax.text(19.0, 3.42, "effect", ha="right", fontsize=6.3,
                  weight="bold", color=MUTED)
    phase_ax.spines[["top", "right", "left"]].set_visible(False)

    fig.supxlabel(
        "relative reduction (%)  ·  positive favors SAS-PC",
        fontsize=7.0,
    )
    fig.savefig(
        output, dpi=300, facecolor="white",
        metadata={"Software": "V18 release plotter"},
    )
    fig.savefig(
        output.with_suffix(".pdf"), facecolor="white",
        metadata={"Creator": "V18 release plotter", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def _guard_pass_counts(rows: list[dict[str, str]]) -> np.ndarray:
    """Count seeds passing both registered validity guards per task/design.

    Mirrors the frozen guards (encoder covariance effective rank >= 16 and
    |predictive-loss convergence relative change| <= 0.05); descriptive only.
    """

    if len(rows) != 200:
        raise RuntimeError(f"guard counting requires 200 cell rows, got {len(rows)}")
    task_index = {task: ti for ti, task in enumerate(TASKS)}
    design_index = {design: di for di, design in enumerate(DESIGNS)}
    counts = np.zeros((len(TASKS), len(DESIGNS)), dtype=np.int64)
    for row in rows:
        rank = float(row["encoder_covariance_effective_rank"])
        change = float(row["predictive_loss_convergence_relative_change"])
        if not (math.isfinite(rank) and math.isfinite(change)):
            raise RuntimeError(f"nonfinite guard metrics for {row['task']}")
        if rank >= 16.0 and abs(change) <= 0.05:
            counts[task_index[row["task"]], design_index[row["design"]]] += 1
    if counts.max() > 5:
        raise RuntimeError("guard pass counts exceed the 5-seed grid")
    return counts


def plot_task_design(rows: list[dict[str, str]], output: Path) -> None:
    """Plot within-block rank distributions and joint validity-guard passes."""

    values: dict[tuple[str, int, str], float] = {
        (row["task"], int(row["seed"]), row["design"]): float(row[PRIMARY])
        for row in rows
    }
    ranks: dict[str, list[int]] = {design: [] for design in DESIGNS}
    for task in TASKS:
        for seed in range(18001, 18006):
            ordered = sorted(
                DESIGNS, key=lambda design: (values[(task, seed, design)], design)
            )
            for rank, design in enumerate(ordered, 1):
                ranks[design].append(rank)
    ordered_designs = sorted(DESIGNS, key=lambda design: np.mean(ranks[design]))
    labels = {
        "vicreg_none": "No carrier",
        "vicreg_gru": "GRU",
        "vicreg_ssm": "Diag. SSM",
        "vicreg_hacssmv8": "SAS-PC",
        "vicreg_hacssmv8_noaction": "No action",
        "vicreg_hacssmv8_single": "Single read",
        "vicreg_hacssmv8_static": "Static",
        "vicreg_hacssmv8_dynamic": "Dynamic",
    }
    column_labels = {
        "vicreg_none": "No\ncarrier",
        "vicreg_gru": "GRU",
        "vicreg_ssm": "Diagonal\nSSM",
        "vicreg_hacssmv8": "SAS-PC",
        "vicreg_hacssmv8_noaction": "No\naction",
        "vicreg_hacssmv8_single": "Single\nread",
        "vicreg_hacssmv8_static": "Static\ngate",
        "vicreg_hacssmv8_dynamic": "Dynamic\ngate",
    }
    guard_counts = _guard_pass_counts(rows)

    fig = plt.figure(figsize=(5.5, 3.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.32, 1.0), hspace=.10)
    ax = fig.add_subplot(grid[0])
    heat_ax = fig.add_subplot(grid[1])

    y = np.arange(len(ordered_designs))[::-1]
    jitter = np.linspace(-.16, .16, 25)
    for yi, design in zip(y, ordered_designs, strict=True):
        vector = np.asarray(ranks[design], dtype=np.float64)
        is_candidate = design == "vicreg_hacssmv8"
        point_color = "#A8CD72" if is_candidate else "#C3C9CD"
        summary_color = NVIDIA_DARK if is_candidate else GRAPHITE
        ax.scatter(vector, yi + jitter, s=10, color=point_color, alpha=.72,
                   edgecolor="none", zorder=2)
        q1, median, q3 = np.percentile(vector, (25, 50, 75))
        ax.hlines(yi, q1, q3, color=summary_color, linewidth=1.5, zorder=3)
        ax.vlines(median, yi - .12, yi + .12, color=summary_color,
                  linewidth=1.0, zorder=3)
        ax.scatter(vector.mean(), yi, s=31, marker="D", color=summary_color,
                   edgecolor="white", linewidth=.45, zorder=4)
        ax.text(8.75, yi, f"{vector.mean():.2f}", ha="right", va="center",
                fontsize=6.9, color=summary_color)
    ax.set_yticks(y, [labels[design] for design in ordered_designs], fontsize=7.2)
    ax.set_xticks(range(1, 9))
    ax.set_xlim(.65, 8.9)
    ax.set_ylim(-.55, len(ordered_designs) - .15)
    ax.grid(axis="x", color=GRID, linewidth=.5)
    ax.tick_params(axis="x", labelsize=6.9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("rank within each task × seed block  (1 = lowest NMSE)",
                  fontsize=7.1, labelpad=1.5)
    ax.set_title("(a) Within-block rank distributions across 25 task–seed blocks",
                 loc="left", fontsize=8.4, weight="bold", color=INK, pad=4)
    ax.text(8.75, len(ordered_designs) - .02, "mean", ha="right", fontsize=6.5,
            weight="bold", color=MUTED)
    ax.spines[["top", "right", "left"]].set_visible(False)

    # Joint validity-guard passes, ordered exactly as panel (a) columns.
    design_order = [DESIGNS.index(design) for design in ordered_designs]
    heat = guard_counts[:, design_order]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "nvidia_guard", [LIGHT_GREEN, NVIDIA], N=256,
    )
    heat_ax.pcolormesh(
        np.arange(len(ordered_designs) + 1), np.arange(len(TASKS) + 1),
        heat[::-1], cmap=cmap, vmin=0, vmax=5, edgecolors="white",
        linewidth=1.4,
    )
    for ti in range(len(TASKS)):
        for di in range(len(ordered_designs)):
            value = int(heat[ti, di])
            heat_ax.text(
                di + .5, len(TASKS) - ti - .5, str(value), ha="center",
                va="center", fontsize=6.9,
                color="white" if value >= 4 else GRAPHITE,
                weight="bold" if value >= 4 else "normal",
            )
    heat_ax.set_xticks(
        np.arange(len(ordered_designs)) + .5,
        [column_labels[design] for design in ordered_designs], fontsize=6.4,
        linespacing=1.1,
    )
    heat_ax.set_yticks(
        np.arange(len(TASKS)) + .5, TASK_SHORT_LABELS[::-1], fontsize=6.9,
    )
    heat_ax.tick_params(length=0)
    heat_ax.set_xlim(0, len(ordered_designs))
    heat_ax.set_ylim(0, len(TASKS))
    heat_ax.set_title(
        "(b) Joint validity-guard passes per task × design (of 5 seeds)",
        loc="left", fontsize=8.4, weight="bold", color=INK, pad=4,
    )
    for spine in heat_ax.spines.values():
        spine.set_visible(False)

    fig.savefig(
        output, dpi=300, facecolor="white",
        metadata={"Software": "V18 release plotter"},
    )
    fig.savefig(
        output.with_suffix(".pdf"), facecolor="white",
        metadata={"Creator": "V18 release plotter", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    root = args.root.resolve()
    report, rows, bundle = load_report(root)
    protocol = bundle["protocol"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".v18-figures-", dir=output.parent))
    secondary = descriptive_secondary(root, report, bundle)
    names = (
        "fig_v18_architecture",
        "fig_v18_evidence",
        "fig_v18_secondary",
        "fig_v18_task_design",
    )
    try:
        plot_architecture(protocol, staging / f"{names[0]}.png")
        plot_evidence(report, staging / f"{names[1]}.png")
        plot_secondary(report, secondary, staging / f"{names[2]}.png")
        plot_task_design(rows, staging / f"{names[3]}.png")
        files = [staging / f"{name}.{suffix}" for name in names for suffix in ("png", "pdf")]
        if any(not path.is_file() or path.stat().st_size == 0 for path in files):
            raise RuntimeError("V18 plot generation produced a missing or empty figure")
        provenance = {
            "schema_version": 3,
            "artifact_kind": "v18_provenance_bound_paper_figures",
            "scientific_label": report["scientific_label"],
            "analysis_sha256": bundle["hashes"]["confirmation_analysis.json"],
            "cells_sha256": report["cells_csv_sha256"],
            "contrasts_sha256": report["contrasts_csv_sha256"],
            "protocol_sha256": bundle["hashes"]["confirmation_protocol.json"],
            "summary_sha256": bundle["hashes"]["confirmation_summary.json"],
            "plotter_sha256": common.sha256(Path(__file__).resolve()),
            "common_validator_sha256": common.sha256(Path(common.__file__).resolve()),
            "tool_versions": {
                "matplotlib": matplotlib.__version__,
                "numpy": np.__version__,
            },
            "descriptive_secondary": secondary,
            "figures": {path.name: common.sha256(path) for path in files},
        }
        manifest = staging / "fig_v18_manifest.json"
        common.atomic_write_json(manifest, provenance)
        for path in (*files, manifest):
            os.replace(path, output / path.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps({
        "scientific_label": report["scientific_label"],
        "architecture": str(output / "fig_v18_architecture.png"),
        "evidence": str(output / "fig_v18_evidence.png"),
        "secondary": str(output / "fig_v18_secondary.png"),
        "task_design": str(output / "fig_v18_task_design.png"),
        "manifest": str(output / "fig_v18_manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
