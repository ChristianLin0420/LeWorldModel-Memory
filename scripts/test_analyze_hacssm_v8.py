#!/usr/bin/env python3
"""Synthetic boundary tests for the fail-closed HACSSM-v8 analyzer."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import tempfile


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.analyze_hacssm_v8 as analysis


def synthetic_values(*, candidate: float = 0.70) -> dict[str, float]:
    return {
        "ssm": 1.00,
        "hacssmv6": 0.82,
        "hacssmv6_static": 0.80,
        "hacssmv7_noaux": 0.72,
        "hacssmv7_sharedaction": 0.72,
        "hacssmv7_norecovery": 0.73,
        "hacssmv8_dynamic": 0.72,
        "hacssmv8_static": 0.73,
        "hacssmv8_levelaction": 0.72,
        "hacssmv8_redundant": candidate,
        "hacssmv8_noaction": 0.90,
        "hacssmv8_single": 0.86,
        "hacssmv8": candidate,
    }


def synthetic_rows(seeds, values: dict[str, float] | None = None):
    values = synthetic_values() if values is None else values
    assert set(values) == set(analysis.DESIGNS)
    return [
        {
            "run": f"{env}:{design}:{seed}",
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
            "run": row["run"],
            "env": row["env"],
            "design": row["design"],
            "seed": row["seed"],
            "relative_improvement": value,
        }
        for row in rows
    ]


def contrasts(rows):
    analysis.configure_shared()
    return analysis.shared.contrast_rows(rows, candidate=analysis.CANDIDATE)


def replace_design(rows, design: str, value: float):
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["design"] == design:
            row[analysis.PRIMARY] = value
    return changed


def valid_v8_history():
    split = {
        "loss": 1.0,
        "pred_loss": 0.9,
        "sigreg_loss": 0.1,
        "pred_loss_all_valid": 0.9,
        "pred_loss_first_post": 1.1,
    }
    return [
        {"epoch": epoch, "train": copy.deepcopy(split), "val": copy.deepcopy(split)}
        for epoch in range(1, analysis.EPOCHS + 1)
    ]


def valid_anchor_history(design: str):
    base, schedule, active = analysis.design_aux_contract(design)
    records = []
    for epoch in range(1, analysis.EPOCHS + 1):
        split = {
            "loss": 1.0,
            "pred_loss": 0.9,
            "sigreg_loss": 0.1,
            "hier_loss": 0.2,
            "hier_loss_fast": 0.2,
            "hier_loss_medium": 0.2,
            "hier_loss_weight": (
                analysis.scheduled_weight(base, schedule, epoch) if active else 0.0
            ),
        }
        if design in analysis.V7_DESIGNS:
            split.update({
                "hier_loss_bridge": 0.2,
                "hier_loss_recovery": 0.2,
                "hier_overlap": 0.0,
            })
        records.append({
            "epoch": epoch,
            "train": copy.deepcopy(split),
            "val": copy.deepcopy(split),
        })
    return records


def test_grid_contract_and_no_auxiliary_history() -> None:
    assert len(analysis.DESIGNS) == 13
    assert len(analysis.PILOT_SEEDS) * len(analysis.DESIGNS) * 5 == 195
    assert len(analysis.FINAL_SEEDS) * len(analysis.DESIGNS) * 5 == 325
    for design in analysis.V8_DESIGNS:
        assert analysis.design_aux_contract(design) == (0.0, "fixed", False)
        analysis.validate_history(valid_v8_history(), design, Path("synthetic-v8"))

    poisoned = valid_v8_history()
    poisoned[41]["val"]["hier_loss"] = 0.0
    try:
        analysis.validate_history(poisoned, "hacssmv8", Path("synthetic-v8"))
    except ValueError as exc:
        assert "must have no hierarchical auxiliary fields" in str(exc)
    else:
        raise AssertionError("V8 history accepted a hier_* field")

    for design in (*analysis.V6_DESIGNS, *analysis.V7_DESIGNS):
        analysis.validate_history(valid_anchor_history(design), design, Path("anchor"))
    overlap = valid_anchor_history("hacssmv7_sharedaction")
    overlap[10]["train"]["hier_overlap"] = 1.0
    try:
        analysis.validate_history(overlap, "hacssmv7_sharedaction", Path("anchor"))
    except ValueError as exc:
        assert "overlap original hidden targets" in str(exc)
    else:
        raise AssertionError("V7 anchor accepted hidden-target overlap")

    analysis.configure_shared()
    expected = analysis.shared.expected_args_subset(
        next(iter(analysis.OCC_TO_CLEAN)), "hacssmv8", 0
    )
    assert expected["hier_loss_weight"] == 0.0
    assert expected["hier_loss_schedule"] == "fixed"


def test_bootstrap_is_deterministic_and_hash_bound() -> None:
    assert analysis.BOOTSTRAP_DRAWS == 100_000
    assert analysis.BOOTSTRAP_SEED == 8_008
    assert analysis.BOOTSTRAP_CONTRACT_SHA256 == (
        "b387010d207f96e9e6777c272ec51629764bfc190cbfd3f323fe6196c38f969e"
    )
    values = synthetic_values(candidate=0.98)
    values[analysis.V7_LEADER] = 1.0
    values[analysis.REDUNDANT] = 0.98
    rows = synthetic_rows(analysis.FINAL_SEEDS, values)
    first = analysis.crossed_bootstrap_summary(
        rows, analysis.CANDIDATE, analysis.V7_LEADER
    )
    second = analysis.crossed_bootstrap_summary(
        rows, analysis.CANDIDATE, analysis.V7_LEADER
    )
    assert first == second
    assert first["contract_sha256"] == analysis.BOOTSTRAP_CONTRACT_SHA256
    assert math.isclose(first["point_mean_paired_relative_reduction"], 0.02)
    assert all(math.isclose(value, 0.02) for value in first["ci90"])
    assert all(math.isclose(value, 0.02) for value in first["ci95"])


def test_observed_synthetic_strict_pass_and_pilot_lock() -> None:
    pilot_rows = synthetic_rows(analysis.PILOT_SEEDS)
    pilot = analysis.pilot_decision(
        pilot_rows, stable_convergence(pilot_rows), contrasts(pilot_rows)
    )
    assert pilot["decision"] == "PILOT_OVERALL_BEST_PASS"
    assert pilot["pilot_screen_passed"] is True
    assert all(pilot["criteria"].values())
    assert pilot["adaptive_development_only"] is True

    final_rows = synthetic_rows(analysis.FINAL_SEEDS)
    final = analysis.final_summary(
        final_rows,
        stable_convergence(final_rows),
        contrasts(final_rows),
        pilot_screen_passed=True,
        pilot_noninferiority_screen_passed=True,
    )
    assert final["decision"] == "OVERALL_BEST_ADAPTIVE_DEV"
    assert final["good_enough_for_overall_best_claim"] is True
    assert final["best_in_focused_locked_grid"] is True
    assert final["compact_noninferior_to_v7_leader"] is False
    assert final["scope"] == "adaptive_development_only"
    assert all(final["criteria"].values())
    assert analysis.REDUNDANT not in final["observed"]["performance_envelope_designs"]

    locked = analysis.final_summary(
        final_rows,
        stable_convergence(final_rows),
        contrasts(final_rows),
        pilot_screen_passed=False,
        pilot_noninferiority_screen_passed=False,
    )
    assert locked["decision"] == "PILOT_NO_GO_FINAL_DESCRIPTIVE"
    assert locked["good_enough_for_overall_best_claim"] is False


def test_candidate_thresholds_are_inclusive_and_positive_is_strict() -> None:
    rows = synthetic_rows(analysis.PILOT_SEEDS)
    compared = analysis.contrast_map(contrasts(rows))
    exact = copy.deepcopy(compared)
    exact["ssm"]["mean_paired_relative_reduction"] = 0.06
    exact["ssm"]["paired_wins"] = 10
    original = analysis.contrast_map
    analysis.contrast_map = lambda _unused: exact
    try:
        decision = analysis.pilot_decision(rows, stable_convergence(rows), [])
        assert decision["criteria"]["vs_ssm_reduction_ge_6pct"]
        assert decision["criteria"]["vs_ssm_wins_ge_10_of_15"]
    finally:
        analysis.contrast_map = original

    below = copy.deepcopy(compared)
    below["ssm"]["mean_paired_relative_reduction"] = math.nextafter(0.06, 0.0)
    original = analysis.contrast_map
    analysis.contrast_map = lambda _unused: below
    try:
        decision = analysis.pilot_decision(rows, stable_convergence(rows), [])
        assert not decision["criteria"]["vs_ssm_reduction_ge_6pct"]
    finally:
        analysis.contrast_map = original

    zero = copy.deepcopy(compared)
    zero["hacssmv6_static"]["mean_paired_relative_reduction"] = 0.0
    original = analysis.contrast_map
    analysis.contrast_map = lambda _unused: zero
    try:
        decision = analysis.pilot_decision(rows, stable_convergence(rows), [])
        assert not decision["criteria"]["vs_v6_static_positive"]
    finally:
        analysis.contrast_map = original

    final_rows = synthetic_rows(analysis.FINAL_SEEDS)
    final_compared = analysis.contrast_map(contrasts(final_rows))
    exact_final = copy.deepcopy(final_compared)
    exact_final["ssm"]["mean_paired_relative_reduction"] = 0.07
    exact_final["ssm"]["paired_wins"] = 20
    original = analysis.contrast_map
    analysis.contrast_map = lambda _unused: exact_final
    try:
        decision = analysis.final_summary(
            final_rows,
            stable_convergence(final_rows),
            [],
            pilot_screen_passed=True,
            pilot_noninferiority_screen_passed=True,
        )
        assert decision["criteria"]["vs_ssm_reduction_ge_7pct"]
        assert decision["criteria"]["vs_ssm_wins_ge_20_of_25"]
    finally:
        analysis.contrast_map = original


def test_action_endpoint_and_equivalence_gates_fail_independently() -> None:
    base = synthetic_rows(analysis.PILOT_SEEDS)

    # Redundant and level-specific heads become indistinguishable: the declared
    # action-sharing effect threshold, rather than the compact bundle, must fail.
    no_action_effect = replace_design(base, analysis.ACTION_REFERENCE, 0.70)
    failed_action = analysis.pilot_decision(
        no_action_effect,
        stable_convergence(no_action_effect),
        contrasts(no_action_effect),
    )
    assert not failed_action["criteria"][
        "redundant_vs_levelaction_reduction_ge_0_5pct"
    ]

    weak_endpoints = replace_design(base, analysis.DYNAMIC_ENDPOINT, 0.701)
    weak_endpoints = replace_design(weak_endpoints, analysis.STATIC_ENDPOINT, 0.701)
    failed_endpoint = analysis.pilot_decision(
        weak_endpoints,
        stable_convergence(weak_endpoints),
        contrasts(weak_endpoints),
    )
    assert not failed_endpoint["criteria"][
        "learned_vs_endpoint_envelope_reduction_ge_0_5pct"
    ]

    non_equivalent = replace_design(base, analysis.REDUNDANT, 0.71)
    failed_equivalence = analysis.pilot_decision(
        non_equivalent,
        stable_convergence(non_equivalent),
        contrasts(non_equivalent),
    )
    assert not failed_equivalence["criteria"][
        "compact_redundant_abs_mean_le_0_25pct"
    ]


def test_compact_noninferiority_label_and_exact_margin_failure() -> None:
    # Compact is 0.4% worse than the V7 leader, inside both the point and
    # crossed-bootstrap margins, while all shrinkage/action gates still pass.
    values = synthetic_values(candidate=1.0)
    values.update({
        "ssm": 1.10,
        "hacssmv6": 1.05,
        "hacssmv6_static": 1.04,
        "hacssmv7_noaux": 1.02,
        "hacssmv7_sharedaction": 0.996,
        "hacssmv7_norecovery": 1.03,
        "hacssmv8_dynamic": 1.02,
        "hacssmv8_static": 1.03,
        "hacssmv8_levelaction": 1.02,
        "hacssmv8_redundant": 1.0,
        "hacssmv8_noaction": 1.15,
        "hacssmv8_single": 1.12,
        "hacssmv8": 1.0,
    })
    pilot_rows = synthetic_rows(analysis.PILOT_SEEDS, values)
    pilot = analysis.pilot_decision(
        pilot_rows, stable_convergence(pilot_rows), contrasts(pilot_rows)
    )
    assert pilot["pilot_screen_passed"] is False
    assert pilot["pilot_noninferiority_screen_passed"] is True
    assert pilot["decision"] == "NO_GO"

    final_rows = synthetic_rows(analysis.FINAL_SEEDS, values)
    locked = analysis.final_summary(
        final_rows,
        stable_convergence(final_rows),
        contrasts(final_rows),
        pilot_screen_passed=False,
        pilot_noninferiority_screen_passed=True,
    )
    assert locked["decision"] == "PILOT_NO_GO_FINAL_DESCRIPTIVE"

    # Noninferiority is a final fallback only when the immutable strict pilot
    # passed but the five-seed superiority bar subsequently missed.
    final = analysis.final_summary(
        final_rows,
        stable_convergence(final_rows),
        contrasts(final_rows),
        pilot_screen_passed=True,
        pilot_noninferiority_screen_passed=True,
    )
    assert final["decision"] == "COMPACT_NONINFERIOR_ADAPTIVE_DEV"
    assert final["good_enough_for_overall_best_claim"] is False
    assert final["good_enough_for_compact_noninferiority_claim"] is True
    assert final["best_in_focused_locked_grid"] is False
    assert final["compact_noninferior_to_v7_leader"] is True

    outside = dict(values)
    outside[analysis.V7_LEADER] = 0.98
    outside_rows = synthetic_rows(analysis.FINAL_SEEDS, outside)
    failed = analysis.final_summary(
        outside_rows,
        stable_convergence(outside_rows),
        contrasts(outside_rows),
        pilot_screen_passed=True,
        pilot_noninferiority_screen_passed=True,
    )
    assert not failed["noninferiority_criteria"][
        "vs_v7_leader_point_gt_minus_0_5pct"
    ]
    assert failed["decision"] != "COMPACT_NONINFERIOR_ADAPTIVE_DEV"


def test_final_refuses_nonrecomputed_pilot() -> None:
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
                raise AssertionError("final analyzer accepted a missing pilot record")

            pilot_path = Path(temporary) / "pilot_decision.json"
            pilot_path.write_text(json.dumps({"pilot_screen_passed": True}))
            try:
                analysis.main(["--root", temporary, "--phase", "final"])
            except ValueError as exc:
                assert "invalid immutable pilot decision" in str(exc)
            else:
                raise AssertionError("final analyzer accepted an unrecomputed pilot record")
    finally:
        analysis.shared.load_cells = original_load
        analysis.strict_validate_cells = original_strict


if __name__ == "__main__":
    tests = (
        test_grid_contract_and_no_auxiliary_history,
        test_bootstrap_is_deterministic_and_hash_bound,
        test_observed_synthetic_strict_pass_and_pilot_lock,
        test_candidate_thresholds_are_inclusive_and_positive_is_strict,
        test_action_endpoint_and_equivalence_gates_fail_independently,
        test_compact_noninferiority_label_and_exact_margin_failure,
        test_final_refuses_nonrecomputed_pilot,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
    print(f"All {len(tests)} HACSSM-v8 analyzer tests passed.")
