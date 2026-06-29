#!/usr/bin/env python3
"""Synthetic decision and fail-closed tests for the HACSSM-v9 analyzer."""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.analyze_hacssm_v9 as analysis


def _candidate_receipts() -> dict[str, float | bool]:
    values: dict[str, float | bool] = {
        "alpha_fast": 0.50,
        "alpha_slow": 0.75,
        "q_fast": 0.75,
        "q_slow": 0.4375,
        "loif_pole_separation": 0.25,
        "loif_fast_boundary_margin": 0.50,
        "loif_slow_boundary_margin": 0.25,
        "loif_pole_boundary_margin": 0.25,
        "loif_saturation_tolerance": 0.01,
        "loif_log_scale_extreme_threshold": 20.0,
        "loif_gain_saturated_fraction": 0.0,
        "loif_log_R_extreme_fraction": 0.0,
        "loif_log_P_extreme_fraction": 0.0,
        "loif_nonfinite_diagnostic_count": 0.0,
        "loif_streaming_batch_size": 1.0,
        "loif_streaming_tolerance": 1e-5,
        "loif_streaming_mixed_max_abs": 0.0,
        "loif_streaming_state_max_abs": 0.0,
        "loif_streaming_log_P_max_abs": 0.0,
        "loif_pole_collapsed": False,
        "loif_boundary_saturated": False,
        "loif_streaming_equivalent": True,
    }
    for phase in analysis.DIAGNOSTIC_PHASES:
        values[f"loif_innovation_or_log_R_constant_{phase}"] = False
        values[f"loif_innovation_log_R_corr_{phase}"] = 0.5
        for stat in analysis.DIAGNOSTIC_STATS:
            if stat == "log_R" or stat.startswith("log_P"):
                value = 0.0
            elif stat.startswith(("omega_", "pi_")):
                value = 0.5
            elif stat.startswith("K_"):
                value = 0.4
            elif stat.startswith("direct_"):
                value = 0.3
            else:
                value = 1.0
            values[f"loif_{stat}_{phase}"] = value
    return values


def synthetic_rows(seeds) -> list[dict]:
    """A comfortably passing grid with constant paired effects in every cell."""
    rows = []
    for env in analysis.OCC_TO_CLEAN:
        for design in analysis.DESIGNS:
            for seed in seeds:
                candidate = design == analysis.CANDIDATE
                mse = 0.80 if candidate else 1.00
                row = {
                    "run": f"{env}:{design}:{seed}",
                    "env": env,
                    "design": design,
                    "seed": seed,
                    analysis.PRIMARY: mse,
                    "clean_mse_deep_blackout": mse,
                    "clean_mse_all": mse,
                    "last_visible_mse_first_post": 1.10,
                }
                if candidate:
                    for phase in analysis.INTERVENTION_PHASES:
                        row[f"clean_mse_{phase}"] = mse
                        for kind in analysis.INTERVENTION_KINDS:
                            row[f"clean_mse_{phase}_resistance_{kind}"] = 1.00
                    row.update(_candidate_receipts())
                rows.append(row)
    return rows


def stable_convergence(rows, value: float = 0.001) -> list[dict]:
    return [
        {
            "run": row["run"],
            "env": row["env"],
            "design": row["design"],
            "seed": row["seed"],
            "relative_improvement": value,
        }
        for row in rows
    ]


def _candidate_rows(rows):
    return [row for row in rows if row["design"] == analysis.CANDIDATE]


def test_positive_pilot_and_final_paths_and_pilot_lock() -> None:
    pilot_rows = synthetic_rows(analysis.PILOT_SEEDS)
    pilot = analysis.pilot_decision(pilot_rows, stable_convergence(pilot_rows))
    assert pilot["decision"] == "PILOT_OVERALL_BEST_PASS"
    assert pilot["pilot_screen_passed"] is True
    assert all(pilot["criteria"].values())
    assert pilot["scope"] == "adaptive_development_only"

    final_rows = synthetic_rows(analysis.FINAL_SEEDS)
    final = analysis.final_summary(
        final_rows, stable_convergence(final_rows), pilot_screen_passed=True)
    assert final["decision"] == "OVERALL_BEST_ADAPTIVE_DEV"
    assert final["final_gates_passed"] is True
    assert final["best_in_locked_grid"] is True
    assert final["completed_runs"] == 325
    assert all(final["criteria"].values())

    # Completion can describe a strong result but can never reopen a failed pilot.
    locked = analysis.final_summary(
        final_rows, stable_convergence(final_rows), pilot_screen_passed=False)
    assert locked["final_gates_passed"] is True
    assert locked["decision"] == "PILOT_NO_GO_FINAL_DESCRIPTIVE"
    assert locked["best_in_locked_grid"] is False
    assert locked["good_enough_for_overall_best_claim"] is False


def test_threshold_edges_are_inclusive_or_strict_as_frozen() -> None:
    inclusive = {
        "mean_paired_relative_reduction": 0.005,
        "paired_wins": 9,
        "environment_mean_wins": 3,
    }
    assert analysis._pair_gate(inclusive, 0.005, 9, 3) == (True, True, True)
    below = dict(inclusive)
    below["mean_paired_relative_reduction"] = math.nextafter(0.005, 0.0)
    assert analysis._pair_gate(below, 0.005, 9, 3)[0] is False

    zero = dict(inclusive)
    zero["mean_paired_relative_reduction"] = 0.0
    assert analysis._pair_gate(
        zero, 0.0, 9, 3, strict_reduction=True)[0] is False
    zero["mean_paired_relative_reduction"] = math.nextafter(0.0, 1.0)
    assert analysis._pair_gate(
        zero, 0.0, 9, 3, strict_reduction=True)[0] is True

    # The phase and convergence inequalities are also strict at their named bounds.
    phase_observed = {
        "phase_vs_v7_sharedaction": {
            metric: {
                "mean_paired_relative_reduction": -0.01,
                "environment_mean_reductions": {
                    env: -0.01 for env in analysis.OCC_TO_CLEAN
                },
            }
            for metric in analysis.PHASE_METRICS
        }
    }
    phase = analysis._phase_criteria(phase_observed)
    assert not any(phase.values())

    pilot_rows = synthetic_rows(analysis.PILOT_SEEDS)
    exact_convergence = stable_convergence(pilot_rows, value=0.01)
    decision = analysis.pilot_decision(pilot_rows, exact_convergence)
    assert decision["criteria"]["convergence_absolute_median_lt_1pct"] is False
    assert decision["decision"] == "NO_GO"


def test_endpoint_envelope_is_cellwise_and_environment_rule_is_separate() -> None:
    rows = synthetic_rows(analysis.FINAL_SEEDS)
    for row in rows:
        if row["design"] == analysis.DYNAMIC_ENDPOINT:
            row[analysis.PRIMARY] = 0.90 if row["seed"] % 2 == 0 else 1.10
        elif row["design"] == analysis.STATIC_ENDPOINT:
            row[analysis.PRIMARY] = 1.10 if row["seed"] % 2 == 0 else 0.90

    matrix = analysis.endpoint_matrix(rows)
    expected_cell = (0.90 - 0.80) / 0.90
    assert np.allclose(matrix, expected_cell, rtol=0.0, atol=1e-14)

    summary = analysis.endpoint_summary(rows)
    # Across five seeds dynamic averages .98 while static averages 1.02. The environment
    # contrast deliberately selects the better complete endpoint mean, not cellwise minima.
    expected_environment = (0.98 - 0.80) / 0.98
    assert all(math.isclose(value, expected_environment)
               for value in summary["environment_mean_reductions"].values())
    assert summary["environment_mean_wins"] == 5


def test_bootstrap_is_deterministic_hash_bound_and_finite() -> None:
    assert analysis.BOOTSTRAP_DRAWS == 100_000
    assert analysis.BOOTSTRAP_SEED == 8_008
    matrix = np.full((5, 5), 0.2, dtype=np.float64)
    first = analysis.crossed_bootstrap(matrix, "synthetic")
    second = analysis.crossed_bootstrap(matrix, "synthetic")
    assert first == second
    assert first["contract_sha256"] == analysis.BOOTSTRAP_CONTRACT_SHA256
    assert len(first["contract_sha256"]) == 64
    assert math.isclose(first["point_mean_paired_relative_reduction"], 0.2)
    assert all(math.isclose(value, 0.2) for value in first["ci90"])
    json.dumps(first, allow_nan=False)


def test_intervention_and_phase_receipts_have_the_declared_direction() -> None:
    rows = synthetic_rows(analysis.PILOT_SEEDS)
    for kind in analysis.INTERVENTION_KINDS:
        matrix = analysis.intervention_matrix(rows, kind)
        assert np.allclose(matrix, 0.2)
        summary = analysis.intervention_summary(rows, kind)
        assert math.isclose(summary["mean_paired_relative_reduction"], 0.2)
        assert summary["paired_wins"] == 15
        assert summary["environment_mean_wins"] == 5

    observed = analysis._observed(rows, stable_convergence(rows))
    phase = analysis._phase_criteria(observed)
    assert all(phase.values())
    phase_rows = analysis.phase_contrast_rows(rows)
    intervention_rows = analysis.intervention_contrast_rows(rows)
    assert len(phase_rows) == len(analysis.PHASE_METRICS) * 6
    assert len(intervention_rows) == (
        len(analysis.INTERVENTION_PHASES) * len(analysis.INTERVENTION_KINDS) * 6)


def test_each_finite_scientific_stop_forces_no_go_without_crashing() -> None:
    cases = (
        (
            "pole collapse",
            {"alpha_slow": 0.5001, "loif_pole_collapsed": True},
            "no_pole_collapse_cells",
        ),
        (
            "pole boundary saturation",
            {"alpha_fast": 0.005, "loif_boundary_saturated": True},
            "no_pole_boundary_saturation_cells",
        ),
        (
            "gain saturation",
            {"loif_gain_saturated_fraction": 1e-6},
            "no_gain_saturation_samples",
        ),
        (
            "extreme log R",
            {"loif_log_R_extreme_fraction": 1e-6},
            "no_extreme_log_R_samples",
        ),
        (
            "extreme log P",
            {"loif_log_P_extreme_fraction": 1e-6},
            "no_extreme_log_P_samples",
        ),
        (
            "streaming mismatch",
            {
                "loif_streaming_equivalent": False,
                "loif_streaming_mixed_max_abs": 1e-4,
                "loif_streaming_state_max_abs": 1e-4,
                "loif_streaming_log_P_max_abs": 1e-4,
            },
            "batch_size_one_streaming_equivalent_all_cells",
        ),
    )
    for label, mutation, criterion in cases:
        rows = synthetic_rows(analysis.PILOT_SEEDS)
        for row in _candidate_rows(rows):
            row.update(mutation)
        decision = analysis.pilot_decision(rows, stable_convergence(rows))
        assert decision["decision"] == "NO_GO", label
        assert decision["pilot_screen_passed"] is False, label
        assert decision["criteria"][criterion] is False, label
        # The stop receipt itself remains finite and serializable; it is a negative result,
        # not an analyzer exception or NaN propagation path.
        json.dumps(decision, allow_nan=False)


def test_final_main_recomputes_and_exactly_locks_the_pilot() -> None:
    rows = synthetic_rows(analysis.FINAL_SEEDS)
    convergence = stable_convergence(rows)
    pilot_rows = [row for row in rows if row["seed"] in analysis.PILOT_SEEDS]
    pilot_convergence = [
        row for row in convergence if row["seed"] in analysis.PILOT_SEEDS
    ]
    tampered = analysis.pilot_decision(pilot_rows, pilot_convergence)
    tampered = copy.deepcopy(tampered)
    tampered["note"] = "tampered after pilot"

    originals = {
        "strict_validate_cells": analysis.strict_validate_cells,
        "load_cells": analysis.load_cells,
        "contrast_rows": analysis.contrast_rows,
        "phase_contrast_rows": analysis.phase_contrast_rows,
        "intervention_contrast_rows": analysis.intervention_contrast_rows,
        "grouped_rows": analysis.shared.grouped_rows,
        "read_json": analysis.shared.read_json,
    }
    analysis.strict_validate_cells = lambda *_args, **_kwargs: None
    analysis.load_cells = lambda *_args, **_kwargs: (rows, convergence)
    analysis.contrast_rows = lambda *_args, **_kwargs: []
    analysis.phase_contrast_rows = lambda *_args, **_kwargs: []
    analysis.intervention_contrast_rows = lambda *_args, **_kwargs: []
    analysis.shared.grouped_rows = lambda *_args, **_kwargs: []
    analysis.shared.read_json = lambda *_args, **_kwargs: tampered
    try:
        with tempfile.TemporaryDirectory() as directory:
            try:
                analysis.main(["--root", directory, "--phase", "final"])
            except ValueError as exc:
                assert "invalid immutable pilot decision" in str(exc)
            else:
                raise AssertionError("final analysis accepted a modified pilot receipt")
    finally:
        analysis.strict_validate_cells = originals["strict_validate_cells"]
        analysis.load_cells = originals["load_cells"]
        analysis.contrast_rows = originals["contrast_rows"]
        analysis.phase_contrast_rows = originals["phase_contrast_rows"]
        analysis.intervention_contrast_rows = originals["intervention_contrast_rows"]
        analysis.shared.grouped_rows = originals["grouped_rows"]
        analysis.shared.read_json = originals["read_json"]


if __name__ == "__main__":
    tests = (
        test_positive_pilot_and_final_paths_and_pilot_lock,
        test_threshold_edges_are_inclusive_or_strict_as_frozen,
        test_endpoint_envelope_is_cellwise_and_environment_rule_is_separate,
        test_bootstrap_is_deterministic_hash_bound_and_finite,
        test_intervention_and_phase_receipts_have_the_declared_direction,
        test_each_finite_scientific_stop_forces_no_go_without_crashing,
        test_final_main_recomputes_and_exactly_locks_the_pilot,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
    print(f"All {len(tests)} HACSSM-v9 analyzer tests passed.")
