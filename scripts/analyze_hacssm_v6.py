#!/usr/bin/env python3
"""Fail-closed analysis for the prospectively frozen HACSSM-v6 study.

V6 keeps the strongest fixed two-rate action hierarchy and tests whether dense,
visible-endpoint action-consistency distillation improves it.  Raw latent MSE is
never pooled across environments; cross-environment summaries use paired relative
reductions on matched environment/seed cells.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.analyze_hacssm_v5 as shared


OCC_TO_CLEAN = shared.OCC_TO_CLEAN
DESIGNS = (
    "ssm",
    "hacsmv4_two_noaux",
    "hacssmv5_noaux",
    "hacssmv6_noaux",
    "hacssmv6_aux_noaction",
    "hacssmv6_uniform",
    "hacssmv6_sourcegrad",
    "hacssmv6_fastonly",
    "hacssmv6_mediumonly",
    "hacssmv6_noaction",
    "hacssmv6_static",
    "hacssmv6_single",
    "hacssmv6",
)
PILOT_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
V6_DESIGNS = frozenset(d for d in DESIGNS if d.startswith("hacssmv6"))
HIER_DESIGNS = frozenset(d for d in DESIGNS if d.startswith(("hacsmv4", "hacssmv5", "hacssmv6")))
PRIMARY = "clean_mse_first_post"
EPOCHS = 200
WANDB_ENTITY = "crlc112358"
WANDB_PROJECT = "lewm-memory-popgym"
WANDB_MODE = "online"
WANDB_STUDY = "hacssm-v6"
CANDIDATE = "hacssmv6"


def design_aux_contract(design: str) -> tuple[float, str, bool]:
    """Return base weight, schedule, and whether gradients are prospectively active."""
    if design in V6_DESIGNS:
        return 0.02, "v6_bootstrap", design != "hacssmv6_noaux"
    if design == "hacsmv4_two_noaux":
        return 0.1, "fixed", False
    if design == "hacssmv5_noaux":
        return 0.05, "v5_frontload", False
    if design == "ssm":
        return 0.0, "fixed", False
    raise ValueError(f"unknown V6 design {design!r}")


def scheduled_weight(base: float, schedule: str, epoch: int) -> float:
    if epoch < 1:
        raise ValueError(f"epoch must be positive, got {epoch}")
    if schedule == "fixed":
        return float(base)
    if schedule == "v5_frontload":
        if epoch <= 20:
            return float(base)
        if epoch <= 120:
            return float(base) * 0.5 * (1.0 + math.cos(math.pi * (epoch - 20) / 100.0))
        return 0.0
    if schedule == "v6_bootstrap":
        if epoch <= 40:
            return float(base)
        if epoch <= 100:
            return float(base) * 0.5 * (1.0 + math.cos(math.pi * (epoch - 40) / 60.0))
        return 0.0
    raise ValueError(f"unknown hierarchy schedule {schedule!r}")


def configure_shared() -> None:
    shared.DESIGNS = DESIGNS
    shared.PILOT_SEEDS = PILOT_SEEDS
    shared.FINAL_SEEDS = FINAL_SEEDS
    shared.V5_DESIGNS = V6_DESIGNS
    shared.HIER_DESIGNS = HIER_DESIGNS
    shared.NO_AUX_DESIGNS = frozenset({"hacsmv4_two_noaux", "hacssmv5_noaux", "hacssmv6_noaux"})
    shared.PRIMARY = PRIMARY
    shared.EPOCHS = EPOCHS
    shared.WANDB_ENTITY = WANDB_ENTITY
    shared.WANDB_PROJECT = WANDB_PROJECT
    shared.WANDB_MODE = WANDB_MODE
    shared.WANDB_STUDY = WANDB_STUDY
    shared.design_aux_contract = design_aux_contract
    shared.scheduled_weight = scheduled_weight


def environment_wins(rows: Sequence[Mapping[str, Any]], reference: str) -> int:
    candidate = shared.environment_means(rows, CANDIDATE)
    baseline = shared.environment_means(rows, reference)
    return sum(candidate[env] < baseline[env] for env in OCC_TO_CLEAN)


def contrast_map(contrasts: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        design: shared.overall_contrast(contrasts, design)
        for design in DESIGNS if design != CANDIDATE
    }


def pilot_decision(rows, convergence, contrasts) -> dict[str, Any]:
    compared = contrast_map(contrasts)
    env_wins = {d: environment_wins(rows, d) for d in compared}
    absolute = np.abs(np.asarray(
        [float(row["relative_improvement"]) for row in convergence], dtype=np.float64))

    criteria = {
        "vs_ssm_reduction_ge_3pct":
            float(compared["ssm"]["mean_paired_relative_reduction"]) >= 0.03,
        "vs_ssm_wins_ge_9_of_15": int(compared["ssm"]["paired_wins"]) >= 9,
        "vs_ssm_env_wins_ge_3_of_5": env_wins["ssm"] >= 3,
        "vs_v4_two_reduction_ge_1pct":
            float(compared["hacsmv4_two_noaux"]["mean_paired_relative_reduction"]) >= 0.01,
        "vs_v4_two_wins_ge_9_of_15":
            int(compared["hacsmv4_two_noaux"]["paired_wins"]) >= 9,
        "vs_v4_two_env_wins_ge_3_of_5": env_wins["hacsmv4_two_noaux"] >= 3,
        "vs_noaux_reduction_ge_1pct":
            float(compared["hacssmv6_noaux"]["mean_paired_relative_reduction"]) >= 0.01,
        "vs_noaux_wins_ge_9_of_15": int(compared["hacssmv6_noaux"]["paired_wins"]) >= 9,
        "vs_noaux_env_wins_ge_3_of_5": env_wins["hacssmv6_noaux"] >= 3,
    }
    for reference in (
        "hacssmv6_aux_noaction", "hacssmv6_uniform", "hacssmv6_sourcegrad",
        "hacssmv6_fastonly", "hacssmv6_mediumonly", "hacssmv6_noaction",
        "hacssmv6_static", "hacssmv6_single",
    ):
        label = reference.removeprefix("hacssmv6_")
        criteria[f"vs_{label}_positive"] = (
            float(compared[reference]["mean_paired_relative_reduction"]) > 0.0)
        criteria[f"vs_{label}_wins_ge_8_of_15"] = int(compared[reference]["paired_wins"]) >= 8
    criteria.update({
        "convergence_absolute_median_lt_1pct": float(np.median(absolute)) < 0.01,
        "convergence_absolute_p95_lt_3pct": float(np.quantile(absolute, 0.95)) < 0.03,
        "convergence_absolute_max_lt_5pct": float(absolute.max()) < 0.05,
    })
    passed = all(criteria.values())
    return {
        "schema_version": 1,
        "phase": "pilot",
        "decision": "PILOT_PASS" if passed else "NO_GO",
        "pilot_screen_passed": passed,
        "criteria": criteria,
        "observed": {
            "overall_contrasts": compared,
            "environment_mean_wins": env_wins,
            "convergence_absolute_median": float(np.median(absolute)),
            "convergence_absolute_p95": float(np.quantile(absolute, 0.95)),
            "convergence_absolute_max": float(absolute.max()),
        },
        "note": (
            "Immutable prospective screen. All five seeds run regardless; a failed screen "
            "cannot be rescued by the descriptive completion."
        ),
    }


def final_summary(rows, convergence, contrasts, *, pilot_screen_passed: bool) -> dict[str, Any]:
    compared = contrast_map(contrasts)
    env_wins = {d: environment_wins(rows, d) for d in compared}
    candidate_means = shared.environment_means(rows, CANDIDATE)
    design_means = {d: shared.environment_means(rows, d) for d in DESIGNS}
    hold = shared.environment_means(rows, CANDIDATE, "last_visible_mse_first_post")
    envelope = sum(
        candidate_means[env] <= min(design_means[d][env] for d in DESIGNS)
        for env in OCC_TO_CLEAN)
    hold_wins = sum(candidate_means[env] < hold[env] for env in OCC_TO_CLEAN)
    absolute = np.abs(np.asarray(
        [float(row["relative_improvement"]) for row in convergence], dtype=np.float64))

    criteria = {
        "vs_ssm_reduction_ge_5pct":
            float(compared["ssm"]["mean_paired_relative_reduction"]) >= 0.05,
        "vs_ssm_wins_ge_18_of_25": int(compared["ssm"]["paired_wins"]) >= 18,
        "vs_ssm_env_wins_ge_4_of_5": env_wins["ssm"] >= 4,
        "vs_v4_two_reduction_ge_1pct":
            float(compared["hacsmv4_two_noaux"]["mean_paired_relative_reduction"]) >= 0.01,
        "vs_v4_two_wins_ge_15_of_25":
            int(compared["hacsmv4_two_noaux"]["paired_wins"]) >= 15,
        "vs_v4_two_env_wins_ge_3_of_5": env_wins["hacsmv4_two_noaux"] >= 3,
        "vs_noaux_reduction_ge_1pct":
            float(compared["hacssmv6_noaux"]["mean_paired_relative_reduction"]) >= 0.01,
        "vs_noaux_wins_ge_15_of_25": int(compared["hacssmv6_noaux"]["paired_wins"]) >= 15,
        "vs_noaux_env_wins_ge_3_of_5": env_wins["hacssmv6_noaux"] >= 3,
        "locked_grid_envelope_wins_ge_4_of_5": envelope >= 4,
        "beats_hold_ge_4_of_5": hold_wins >= 4,
        "convergence_absolute_median_lt_1pct": float(np.median(absolute)) < 0.01,
        "convergence_absolute_p95_lt_3pct": float(np.quantile(absolute, 0.95)) < 0.03,
        "convergence_absolute_max_lt_5pct": float(absolute.max()) < 0.05,
    }
    for reference in (
        "hacssmv6_aux_noaction", "hacssmv6_uniform", "hacssmv6_sourcegrad",
        "hacssmv6_fastonly", "hacssmv6_mediumonly", "hacssmv6_static",
    ):
        label = reference.removeprefix("hacssmv6_")
        criteria[f"vs_{label}_positive"] = (
            float(compared[reference]["mean_paired_relative_reduction"]) > 0.0)
        criteria[f"vs_{label}_wins_ge_13_of_25"] = int(compared[reference]["paired_wins"]) >= 13
        criteria[f"vs_{label}_env_wins_ge_3_of_5"] = env_wins[reference] >= 3
    for reference in ("hacssmv6_noaction", "hacssmv6_single"):
        label = reference.removeprefix("hacssmv6_")
        criteria[f"vs_{label}_reduction_ge_3pct"] = (
            float(compared[reference]["mean_paired_relative_reduction"]) >= 0.03)
        criteria[f"vs_{label}_wins_ge_17_of_25"] = int(compared[reference]["paired_wins"]) >= 17
        criteria[f"vs_{label}_env_wins_ge_3_of_5"] = env_wins[reference] >= 3

    good_enough = all(criteria.values())
    primary_positive = all(
        float(compared[d]["mean_paired_relative_reduction"]) > 0.0
        for d in ("ssm", "hacsmv4_two_noaux", "hacssmv6_noaux"))
    if not pilot_screen_passed:
        decision = "PILOT_NO_GO_FINAL_DESCRIPTIVE"
    elif good_enough:
        decision = "OVERALL_BEST_IN_LOCKED_GRID"
    elif primary_positive:
        decision = "PROMISING_NOT_OVERALL_BEST"
    else:
        decision = "NO_GO"
    return {
        "schema_version": 1,
        "phase": "final",
        "decision": decision,
        "pilot_screen_passed": pilot_screen_passed,
        "good_enough_for_v6_stop": bool(pilot_screen_passed and good_enough),
        "trigger_v7": not bool(pilot_screen_passed and good_enough),
        "criteria": criteria,
        "completed_runs": len(rows),
        "observed": {
            "overall_contrasts": compared,
            "environment_mean_wins": env_wins,
            "locked_grid_envelope_env_wins": envelope,
            "hold_environment_wins": hold_wins,
            "convergence_absolute_median": float(np.median(absolute)),
            "convergence_absolute_p95": float(np.quantile(absolute, 0.95)),
            "convergence_absolute_max": float(absolute.max()),
        },
        "limitations": [
            "The fixed seed-7777 trajectories and exact black corruption are adaptive development data.",
            "The objective sees only originally visible endpoints, but the same corruption family defines them.",
            "No simulator-state metric or executed-control return is measured.",
        ],
        "note": (
            "OVERALL_BEST_IN_LOCKED_GRID is a deterministic development-grid label, not an "
            "untouched-test or ICLR claim. Any other label prospectively triggers V7."
        ),
    }


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/hacssm_v6_shared"))
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    configure_shared()
    args = parse_args(argv)
    seeds = PILOT_SEEDS if args.phase == "pilot" else FINAL_SEEDS
    expected = len(OCC_TO_CLEAN) * len(DESIGNS) * len(seeds)
    rows, convergence = shared.load_cells(args.root, seeds)
    if len(rows) != expected:
        raise ValueError(f"{args.phase} grid has {len(rows)} rows, expected {expected}")
    grouped = shared.grouped_rows(rows)
    contrasts = shared.contrast_rows(rows, candidate=CANDIDATE)
    prefix = "pilot_" if args.phase == "pilot" else ""
    if args.phase == "pilot":
        decision = pilot_decision(rows, convergence, contrasts)
    else:
        pilot_path = args.root / "pilot_decision.json"
        pilot = shared.read_json(pilot_path)
        if not isinstance(pilot, dict) or type(pilot.get("pilot_screen_passed")) is not bool:
            raise ValueError(f"invalid immutable pilot decision: {pilot_path}")
        decision = final_summary(
            rows, convergence, contrasts,
            pilot_screen_passed=pilot["pilot_screen_passed"])
    shared.atomic_csv(args.root / f"{prefix}per_run.csv", rows)
    shared.atomic_csv(args.root / f"{prefix}grouped.csv", grouped)
    shared.atomic_csv(args.root / f"{prefix}paired_contrasts.csv", contrasts)
    shared.atomic_csv(args.root / f"{prefix}convergence.csv", convergence)
    shared.atomic_json(
        args.root / ("pilot_decision.json" if args.phase == "pilot" else "decision.json"),
        decision)
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
