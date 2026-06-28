#!/usr/bin/env python3
"""Focused synthetic tests for the fail-closed HACSSM/HCRD-v7 analyzer."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.analyze_hacssm_v7 as analysis


def synthetic_rows(seeds, *, candidate: float = 0.75):
    values = {
        "ssm": 1.00,
        "hacsmv4_two_noaux": 0.95,
        "hacssmv6": 0.90,
        "hacssmv6_static": 0.88,
        "hacssmv7_noaux": 0.92,
        "hacssmv7_sharedaction": 0.86,
        "hacssmv7_noshrink": 0.85,
        "hacssmv7_actiononly": 0.87,
        "hacssmv7_uniform": 0.84,
        "hacssmv7_norecovery": 0.83,
        "hacssmv7_noaction": 1.08,
        "hacssmv7_single": 1.07,
        "hacssmv7": candidate,
    }
    return [
        {
            "env": env,
            "design": design,
            "seed": seed,
            analysis.PRIMARY: mse,
            "last_visible_mse_first_post": 1.10,
        }
        for env in analysis.OCC_TO_CLEAN
        for design, mse in values.items()
        for seed in seeds
    ]


def stable_convergence(rows, value: float = 0.001):
    return [
        {
            "run": f"{row['env']}:{row['design']}:{row['seed']}",
            "seed": row['seed'],
            "relative_improvement": value,
        }
        for row in rows
    ]


def pilot_boundary_contrasts():
    tiny = math.nextafter(0.0, 1.0)
    reductions = {
        "ssm": 0.05,
        "hacsmv4_two_noaux": 0.01,
        "hacssmv6": 0.005,
        "hacssmv6_static": tiny,
        "hacssmv7_noaux": 0.01,
        "hacssmv7_sharedaction": tiny,
        "hacssmv7_noshrink": tiny,
        "hacssmv7_actiononly": tiny,
        "hacssmv7_uniform": tiny,
        "hacssmv7_norecovery": tiny,
        "hacssmv7_noaction": tiny,
        "hacssmv7_single": tiny,
    }
    wins = {
        "ssm": 10,
        "hacsmv4_two_noaux": 9,
        "hacssmv6": 9,
        "hacssmv6_static": 8,
        "hacssmv7_noaux": 9,
        "hacssmv7_sharedaction": 8,
        "hacssmv7_noshrink": 8,
        "hacssmv7_actiononly": 8,
        "hacssmv7_uniform": 8,
        "hacssmv7_norecovery": 8,
        "hacssmv7_noaction": 8,
        "hacssmv7_single": 8,
    }
    return {
        design: {
            "reference": design,
            "mean_paired_relative_reduction": reductions[design],
            "paired_wins": wins[design],
        }
        for design in reductions
    }


def final_boundary_contrasts():
    tiny = math.nextafter(0.0, 1.0)
    reductions = {
        "ssm": 0.06,
        "hacsmv4_two_noaux": 0.015,
        "hacssmv6": 0.01,
        "hacssmv6_static": tiny,
        "hacssmv7_noaux": 0.01,
        "hacssmv7_sharedaction": tiny,
        "hacssmv7_noshrink": tiny,
        "hacssmv7_actiononly": tiny,
        "hacssmv7_uniform": tiny,
        "hacssmv7_norecovery": tiny,
        "hacssmv7_noaction": 0.03,
        "hacssmv7_single": 0.03,
    }
    wins = {
        "ssm": 20,
        "hacsmv4_two_noaux": 17,
        "hacssmv6": 15,
        "hacssmv6_static": 13,
        "hacssmv7_noaux": 15,
        "hacssmv7_sharedaction": 13,
        "hacssmv7_noshrink": 13,
        "hacssmv7_actiononly": 13,
        "hacssmv7_uniform": 13,
        "hacssmv7_norecovery": 13,
        "hacssmv7_noaction": 17,
        "hacssmv7_single": 17,
    }
    return {
        design: {
            "reference": design,
            "mean_paired_relative_reduction": reductions[design],
            "paired_wins": wins[design],
        }
        for design in reductions
    }


def with_fake_contrasts(compared, callback):
    original = analysis.contrast_map
    analysis.contrast_map = lambda _contrasts: compared
    try:
        return callback()
    finally:
        analysis.contrast_map = original


def test_observed_synthetic_pass_and_failure_labels() -> None:
    analysis.configure_shared()
    pilot_rows = synthetic_rows(analysis.PILOT_SEEDS)
    pilot = analysis.pilot_decision(
        pilot_rows, stable_convergence(pilot_rows),
        analysis.shared.contrast_rows(pilot_rows, candidate=analysis.CANDIDATE))
    assert pilot["decision"] == "PILOT_PASS"
    assert pilot["pilot_screen_passed"] is True

    final_rows = synthetic_rows(analysis.FINAL_SEEDS)
    contrasts = analysis.shared.contrast_rows(final_rows, candidate=analysis.CANDIDATE)
    passed = analysis.final_summary(
        final_rows, stable_convergence(final_rows), contrasts,
        pilot_screen_passed=True)
    assert passed["decision"] == "OVERALL_BEST_IN_LOCKED_GRID"
    assert passed["good_enough_for_overall_best_claim"] is True

    preserved = analysis.final_summary(
        final_rows, stable_convergence(final_rows), contrasts,
        pilot_screen_passed=False)
    assert preserved["decision"] == "PILOT_NO_GO_FINAL_DESCRIPTIVE"
    assert preserved["pilot_screen_passed"] is False
    assert preserved["good_enough_for_overall_best_claim"] is False

    losing_rows = synthetic_rows(analysis.FINAL_SEEDS, candidate=0.93)
    losing = analysis.final_summary(
        losing_rows, stable_convergence(losing_rows),
        analysis.shared.contrast_rows(losing_rows, candidate=analysis.CANDIDATE),
        pilot_screen_passed=True)
    assert losing["decision"] != "OVERALL_BEST_IN_LOCKED_GRID"
    assert losing["good_enough_for_overall_best_claim"] is False


def test_pilot_thresholds_are_inclusive_except_positive_and_convergence() -> None:
    rows = synthetic_rows(analysis.PILOT_SEEDS)
    compared = pilot_boundary_contrasts()
    exact = with_fake_contrasts(
        compared,
        lambda: analysis.pilot_decision(rows, stable_convergence(rows), []))
    assert exact["decision"] == "PILOT_PASS"
    assert all(exact["criteria"].values())

    below = copy.deepcopy(compared)
    below["ssm"]["mean_paired_relative_reduction"] = math.nextafter(0.05, 0.0)
    failed_reduction = with_fake_contrasts(
        below,
        lambda: analysis.pilot_decision(rows, stable_convergence(rows), []))
    assert failed_reduction["decision"] == "NO_GO"
    assert not failed_reduction["criteria"]["vs_ssm_reduction_ge_5pct"]

    too_few_wins = copy.deepcopy(compared)
    too_few_wins["hacssmv6"]["paired_wins"] = 8
    failed_wins = with_fake_contrasts(
        too_few_wins,
        lambda: analysis.pilot_decision(rows, stable_convergence(rows), []))
    assert not failed_wins["criteria"]["vs_v6_wins_ge_9_of_15"]

    zero_positive = copy.deepcopy(compared)
    zero_positive["hacssmv6_static"]["mean_paired_relative_reduction"] = 0.0
    failed_positive = with_fake_contrasts(
        zero_positive,
        lambda: analysis.pilot_decision(rows, stable_convergence(rows), []))
    assert not failed_positive["criteria"]["vs_v6_static_positive"]

    failed_convergence = with_fake_contrasts(
        compared,
        lambda: analysis.pilot_decision(rows, stable_convergence(rows, 0.01), []))
    assert not failed_convergence["criteria"]["convergence_absolute_median_lt_1pct"]


def test_final_thresholds_are_exact_and_fail_one_step_below() -> None:
    rows = synthetic_rows(analysis.FINAL_SEEDS)
    compared = final_boundary_contrasts()
    exact = with_fake_contrasts(
        compared,
        lambda: analysis.final_summary(
            rows, stable_convergence(rows), [], pilot_screen_passed=True))
    assert exact["decision"] == "OVERALL_BEST_IN_LOCKED_GRID"
    assert exact["good_enough_for_overall_best_claim"] is True
    assert all(exact["criteria"].values())

    below = copy.deepcopy(compared)
    below["hacsmv4_two_noaux"]["mean_paired_relative_reduction"] = math.nextafter(
        0.015, 0.0)
    failed_reduction = with_fake_contrasts(
        below,
        lambda: analysis.final_summary(
            rows, stable_convergence(rows), [], pilot_screen_passed=True))
    assert failed_reduction["good_enough_for_overall_best_claim"] is False
    assert not failed_reduction["criteria"]["vs_v4_two_reduction_ge_1_5pct"]

    too_few_wins = copy.deepcopy(compared)
    too_few_wins["ssm"]["paired_wins"] = 19
    failed_wins = with_fake_contrasts(
        too_few_wins,
        lambda: analysis.final_summary(
            rows, stable_convergence(rows), [], pilot_screen_passed=True))
    assert not failed_wins["criteria"]["vs_ssm_wins_ge_20_of_25"]

    zero_positive = copy.deepcopy(compared)
    zero_positive["hacssmv7_norecovery"]["mean_paired_relative_reduction"] = 0.0
    failed_positive = with_fake_contrasts(
        zero_positive,
        lambda: analysis.final_summary(
            rows, stable_convergence(rows), [], pilot_screen_passed=True))
    assert not failed_positive["criteria"]["vs_norecovery_positive"]


def test_final_main_refuses_missing_immutable_pilot_record() -> None:
    rows = synthetic_rows(analysis.FINAL_SEEDS)
    converged = stable_convergence(rows)
    original_load = analysis.shared.load_cells
    original_strict = analysis.strict_validate_cells
    analysis.shared.load_cells = lambda _root, _seeds: (rows, converged)
    analysis.strict_validate_cells = lambda _root, _seeds: None
    try:
        with tempfile.TemporaryDirectory() as temporary:
            try:
                analysis.main(["--root", temporary, "--phase", "final"])
            except FileNotFoundError as exc:
                assert "pilot_decision.json" in str(exc)
            else:
                raise AssertionError("final analysis accepted a missing pilot decision")
            fake = Path(temporary) / 'pilot_decision.json'
            fake.write_text(json.dumps({'pilot_screen_passed': True}))
            try:
                analysis.main(["--root", temporary, "--phase", "final"])
            except ValueError as exc:
                assert 'invalid immutable pilot decision' in str(exc)
            else:
                raise AssertionError('final analysis accepted an unrecomputed pilot decision')
    finally:
        analysis.shared.load_cells = original_load
        analysis.strict_validate_cells = original_strict


def test_standalone_analyzer_enforces_v7_overlap_contract() -> None:
    analysis.configure_shared()
    assert analysis.shared.validate_history is analysis.validate_history
    split = {
        'loss': 1.0,
        'pred_loss': 0.9,
        'sigreg_loss': 0.1,
        'hier_loss': 0.2,
        'hier_loss_fast': 0.2,
        'hier_loss_medium': 0.2,
        'hier_loss_weight': 0.02,
        'hier_loss_bridge': 0.2,
        'hier_loss_recovery': 0.2,
        'hier_overlap': 0.0,
    }
    history = [
        {'epoch': epoch, 'train': copy.deepcopy(split), 'val': copy.deepcopy(split)}
        for epoch in range(1, analysis.EPOCHS + 1)
    ]
    for epoch, record in enumerate(history, 1):
        wanted = analysis.scheduled_weight(0.02, 'v6_bootstrap', epoch)
        record['train']['hier_loss_weight'] = wanted
        record['val']['hier_loss_weight'] = wanted
    analysis.validate_history(history, 'hacssmv7', Path('synthetic-v7'))

    history[41]['val']['hier_overlap'] = 1.0
    try:
        analysis.validate_history(history, 'hacssmv7', Path('synthetic-v7'))
    except ValueError as exc:
        assert 'overlap original hidden targets' in str(exc)
    else:
        raise AssertionError('standalone analyzer accepted nonzero hidden-target overlap')


if __name__ == "__main__":
    tests = (
        test_observed_synthetic_pass_and_failure_labels,
        test_pilot_thresholds_are_inclusive_except_positive_and_convergence,
        test_final_thresholds_are_exact_and_fail_one_step_below,
        test_final_main_refuses_missing_immutable_pilot_record,
        test_standalone_analyzer_enforces_v7_overlap_contract,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
    print(f"All {len(tests)} HACSSM-v7 analyzer tests passed.")
