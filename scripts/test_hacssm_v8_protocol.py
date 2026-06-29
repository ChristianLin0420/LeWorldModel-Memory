#!/usr/bin/env python3
"""Dependency-light protocol tests for the locked HACSSM-v8 study runner."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.run_hacssm_v8 as runner


def job(design: str = "hacssmv8"):
    occ, clean = runner.ENVIRONMENTS[0]
    return runner.shared.Job("pilot", 0, occ, clean, design)


def clean_history():
    values = {
        "loss": 1.0,
        "pred_loss": 0.8,
        "sigreg_loss": 2.0,
        "pred_loss_all_valid": 0.8,
        "pred_loss_first_post": 0.9,
    }
    return [
        {"epoch": epoch, "train": dict(values), "val": dict(values)}
        for epoch in range(1, 201)
    ]


def final(decision: str, *, pilot: bool, best: bool, noninferior: bool):
    return {
        "decision": decision,
        "completed_runs": 325,
        "pilot_screen_passed": pilot,
        "best_in_focused_locked_grid": best,
        "compact_noninferior_to_v7_leader": noninferior,
        "scope": "adaptive_development_only",
    }


def test_grid_is_exact_and_unique() -> None:
    assert len(runner.ENVIRONMENTS) == 5
    assert len(runner.DESIGNS) == 13
    assert len(runner.V8_DESIGNS) == 7
    assert len(runner.PILOT_JOBS) == 195
    assert len(runner.COMPLETION_JOBS) == 130
    assert len(runner.ALL_JOBS) == 325
    assert len({item.run_name for item in runner.ALL_JOBS}) == 325
    assert runner.DESIGNS == (
        "ssm", "hacssmv6", "hacssmv6_static", "hacssmv7_noaux",
        "hacssmv7_sharedaction", "hacssmv7_norecovery", "hacssmv8",
        "hacssmv8_dynamic", "hacssmv8_static", "hacssmv8_levelaction",
        "hacssmv8_redundant", "hacssmv8_noaction", "hacssmv8_single",
    )


def test_auxiliary_contracts_are_exact() -> None:
    for design in runner.V8_DESIGNS:
        assert runner.design_aux_contract(design) == (0.0, "fixed", False)
        metadata = runner.objective_metadata(design)
        assert metadata["memory_internal_auxiliary"] == "none"
        assert metadata["memory_teacher_present"] is False
    assert runner.design_aux_contract("ssm") == (0.0, "fixed", False)
    assert runner.design_aux_contract("hacssmv6") == (0.02, "v6_bootstrap", True)
    assert runner.design_aux_contract("hacssmv6_static") == (0.02, "v6_bootstrap", True)
    assert runner.design_aux_contract("hacssmv7_noaux") == (0.02, "v6_bootstrap", False)
    assert runner.design_aux_contract("hacssmv7_sharedaction") == (
        0.02, "v6_bootstrap", True)


def test_parameter_contract_distinguishes_compact_and_expanded() -> None:
    contract = runner.memory_contract()
    counts = contract["v8_counts_by_design"]
    assert all(counts[design] == 34_566 for design in runner.COMPACT_V8_DESIGNS)
    assert all(counts[design] == 36_102 for design in runner.EXPANDED_V8_DESIGNS)
    assert contract["memory_parameters"]["ssm"] == 33_024
    assert contract["memory_parameters"]["hacssmv6_all_modes"] == 34_564
    assert contract["memory_parameters"]["hacssmv7_student"] == 36_102
    assert contract["v8_checkpoint_contains_teacher"] is False
    assert "intentionally not parameter matched" in contract["parameter_matching_scope"]


def test_expected_args_and_command_force_zero_v8_auxiliary() -> None:
    sample = job("hacssmv8")
    args = runner.expected_args(sample)
    assert args["hier_loss_weight"] == 0.0
    assert args["hier_loss_schedule"] == "fixed"
    assert args["wandb_study"] == "hacssm-v8"
    command = runner.train_command("python", sample)
    weight_index = command.index("--hier-loss-weight")
    schedule_index = command.index("--hier-loss-schedule")
    assert float(command[weight_index + 1]) == 0.0
    assert command[schedule_index + 1] == "fixed"
    assert command[command.index("--memory-mode") + 1] == "hacssmv8"


def test_v8_history_rejects_every_hierarchy_field() -> None:
    history = clean_history()
    runner.validate_history(history, job())
    history[19]["val"]["hier_loss"] = 0.0
    try:
        runner.validate_history(history, job())
    except runner.RunnerError as exc:
        assert "forbidden hierarchy fields" in str(exc)
    else:
        raise AssertionError("V8 history accepted a hier_* field")


def test_v8_model_state_namespace_shape_teacher_and_redundancy() -> None:
    compact_state = {
        "mem_hacssmv8.W_a.weight": torch.zeros(256, 6),
        "mem_hacssmv8.shrink_logits": torch.zeros(2),
    }
    runner.validate_model_state(compact_state, job("hacssmv8"))

    redundant_state = {
        "mem_hacssmv8.W_a.weight": torch.zeros(512, 6),
        "mem_hacssmv8.shrink_logits": torch.zeros(2),
    }
    runner.validate_model_state(redundant_state, job("hacssmv8_redundant"))
    redundant_state["mem_hacssmv8.W_a.weight"][300, 0] = 1.0
    try:
        runner.validate_model_state(redundant_state, job("hacssmv8_redundant"))
    except runner.RunnerError as exc:
        assert "redundant action heads diverged" in str(exc)
    else:
        raise AssertionError("diverged redundant heads were accepted")

    with_teacher = dict(compact_state)
    with_teacher["mem_hacssmv8_teacher.W_a.weight"] = torch.zeros(256, 6)
    try:
        runner.validate_model_state(with_teacher, job("hacssmv8"))
    except runner.RunnerError as exc:
        assert "teacher/V7 tensors" in str(exc)
    else:
        raise AssertionError("V8 teacher tensor was accepted")


def test_v7_student_namespace_mapping_is_exact() -> None:
    v7 = {
        "predictor.weight": torch.arange(3.0),
        "mem_hacssmv7.W_a.weight": torch.arange(12.0).reshape(4, 3),
        "mem_hacssmv7.shrink_logits": torch.ones(2),
        "mem_hacssmv7_teacher.W_a.weight": torch.full((4, 3), 9.0),
    }
    mapped = runner._canonical_v7_student_state(v7)
    assert set(mapped) == {
        "predictor.weight", "mem_hacssmv8.W_a.weight", "mem_hacssmv8.shrink_logits",
    }
    assert torch.equal(mapped["mem_hacssmv8.W_a.weight"], v7["mem_hacssmv7.W_a.weight"])


def test_sealed_v7_reference_is_bound() -> None:
    contract = runner.v7_reference_contract()
    assert contract["manifest_sha256"] == runner.V7_REFERENCE_SHA256
    assert contract["completed_runs"] == 325
    assert set(contract["identity_designs"]) == runner.ANCHOR_DESIGNS


def test_protocol_freezes_scope_grid_thresholds_and_wandb() -> None:
    originals = (
        runner.shared.feature_snapshot,
        runner.shared.eval_rollout_snapshot,
        runner.shared.source_snapshot,
    )
    runner.shared.feature_snapshot = lambda: {"features": {"sha256": "f" * 64}}
    runner.shared.eval_rollout_snapshot = lambda: {"rollout": {"sha256": "e" * 64}}
    runner.shared.source_snapshot = lambda: {"source": {"sha256": "s" * 64}}
    try:
        protocol = runner.build_protocol("a" * 40, True, {"mode": "online"})
    finally:
        (
            runner.shared.feature_snapshot,
            runner.shared.eval_rollout_snapshot,
            runner.shared.source_snapshot,
        ) = originals
    assert protocol["adaptive_development_only"] is True
    assert protocol["stages"]["pilot"]["runs"] == 195
    assert protocol["stages"]["completion"]["runs"] == 130
    assert protocol["stages"]["completion"]["runs_regardless_of_pilot_screen"] is True
    assert protocol["analysis_gate"]["pilot_pass_result"] == "PILOT_OVERALL_BEST_PASS"
    assert protocol["analysis_gate"]["scope"] == "adaptive_development_only"
    assert protocol["performance_envelope"] == {
        "designs": list(runner.PERFORMANCE_ENVELOPE_DESIGNS),
        "excluded": ["hacssmv8", "hacssmv8_redundant"],
        "rationale": (
            "the candidate cannot be its own reference; redundant is a statistical "
            "equivalence/optimization receipt, not a distinct deployable architecture"
        ),
    }
    assert protocol["endpoint_envelope"] == {
        "pairwise_reduction_and_wins": (
            "select min(dynamic, static) independently within every environment-seed cell"
        ),
        "environment_wins": (
            "compare the candidate environment mean with the minimum of the dynamic and "
            "static endpoint environment means"
        ),
    }
    assert protocol["wandb_requirements"]["complete_epoch_history_per_cell"] == 200
    assert protocol["wandb_requirements"]["evaluation_rollout_npz_table_video_per_cell"] is True
    assert protocol["pilot_success_criteria"] == {
        "vs_ssm": ">=6% reduction, >=10/15 wins, >=4/5 environment wins",
        "vs_v7_sharedaction": ">=0.5% reduction, >=9/15 wins, >=3/5 environment wins",
        "vs_v6_static": "positive reduction, >=8/15 wins, >=3/5 environment wins",
        "vs_full_v6": ">=1% reduction, >=9/15 wins, >=3/5 environment wins",
        "vs_v7_norecovery": "positive reduction, >=8/15 wins, >=3/5 environment wins",
        "redundant_vs_levelaction": ">=0.5% reduction, >=9/15 wins, >=3/5 environment wins",
        "vs_each_shrinkage_endpoint": (
            "positive reduction, >=8/15 wins, >=3/5 environment wins"
        ),
        "vs_better_endpoint_envelope": (
            ">=0.5% reduction, >=9/15 wins, >=3/5 environment wins"
        ),
        "compact_redundant_equivalence": (
            "absolute mean <=0.25%, every environment within +/-1%, crossed-bootstrap "
            "90% interval inside +/-1%"
        ),
        "structural_controls": ">=3% reduction, >=11/15 wins, >=3/5 environment wins",
        "convergence": "absolute median <1%, p95 <3%, max <5%",
    }
    assert protocol["final_success_criteria"] == {
        "requires_pilot_pass": True,
        "vs_ssm": ">=7% reduction, >=20/25 wins, >=4/5 environment wins",
        "vs_v7_sharedaction": ">=1% reduction, >=15/25 wins, >=3/5 environment wins",
        "vs_v6_static": ">=1% reduction, >=15/25 wins, >=3/5 environment wins",
        "vs_full_v6": ">=1% reduction, >=15/25 wins, >=3/5 environment wins",
        "vs_v7_norecovery": "positive reduction, >=13/25 wins, >=3/5 environment wins",
        "redundant_vs_levelaction": ">=0.5% reduction, >=15/25 wins, >=3/5 environment wins",
        "vs_better_endpoint_envelope": (
            ">=1% reduction, >=15/25 wins, >=3/5 environment wins"
        ),
        "vs_each_shrinkage_endpoint": (
            "positive reduction, >=13/25 wins, >=3/5 environment wins"
        ),
        "compact_redundant_equivalence": (
            "absolute mean <=0.25%, every environment within +/-1%, crossed-bootstrap "
            "90% interval inside +/-1%"
        ),
        "structural_controls": ">=3% reduction, >=17/25 wins, >=3/5 environment wins",
        "envelope_and_hold": (
            ">=3/5 wins over the frozen performance-envelope design list (redundant "
            "excluded) and >=4/5 last-visible-hold wins"
        ),
        "convergence": "absolute median <1%, p95 <3%, max <5%",
    }
    noninferiority = protocol["compact_noninferiority_criteria"]
    assert noninferiority["shared_requirements"] == {
        "vs_v7_sharedaction": (
            "point estimate >-0.5%, >=4/5 environment effects >-1%, deterministic "
            "crossed-bootstrap 95% lower bound >-1%"
        ),
        "vs_ssm": ">=6% reduction",
        "compact_redundant_equivalence": (
            "absolute mean <=0.25%, every environment within +/-1%, crossed-bootstrap "
            "90% interval inside +/-1%"
        ),
        "convergence": "absolute median <1%, p95 <3%, max <5%",
    }
    assert noninferiority["pilot"] == {
        "redundant_vs_levelaction": (
            ">=0.5% reduction, >=9/15 wins, >=3/5 environment wins"
        ),
        "vs_better_endpoint_envelope": (
            ">=0.5% reduction, >=9/15 wins, >=3/5 environment wins"
        ),
        "role": "diagnostic receipt only; cannot reopen a failed strict pilot",
    }
    assert noninferiority["final"] == {
        "requires_strict_pilot_pass": True,
        "redundant_vs_levelaction": (
            ">=0.5% reduction, >=15/25 wins, >=3/5 environment wins"
        ),
        "vs_better_endpoint_envelope": (
            ">=1% reduction, >=15/25 wins, >=3/5 environment wins"
        ),
    }
    assert noninferiority["bootstrap_contract"] == runner.BOOTSTRAP_CONTRACT
    assert noninferiority["bootstrap_contract_sha256"] == (
        "b387010d207f96e9e6777c272ec51629764bfc190cbfd3f323fe6196c38f969e"
    )
    assert runner.BOOTSTRAP_CONTRACT_SHA256 == noninferiority["bootstrap_contract_sha256"]
    for design in runner.V8_DESIGNS:
        entry = protocol["design_protocol"][design]
        assert entry["hier_loss_weight"] == 0.0
        assert entry["hier_loss_schedule"] == "fixed"
        assert entry["auxiliary_gradients_active"] is False


def test_pilot_labels_are_fail_closed() -> None:
    original = runner.PILOT_DECISION_PATH
    with tempfile.TemporaryDirectory() as directory:
        runner.PILOT_DECISION_PATH = Path(directory) / "pilot_decision.json"
        try:
            runner.PILOT_DECISION_PATH.write_text(json.dumps({
                "decision": "PILOT_OVERALL_BEST_PASS",
                "pilot_screen_passed": True,
                "adaptive_development_only": True,
            }))
            passed, _ = runner.read_pilot_decision()
            assert passed is True
            runner.PILOT_DECISION_PATH.write_text(json.dumps({
                "decision": "NO_GO",
                "pilot_screen_passed": False,
                "adaptive_development_only": True,
            }))
            passed, _ = runner.read_pilot_decision()
            assert passed is False
        finally:
            runner.PILOT_DECISION_PATH = original


def test_final_labels_and_fields_are_consistent() -> None:
    assert runner.validate_final_decision(final(
        "OVERALL_BEST_ADAPTIVE_DEV", pilot=True, best=True, noninferior=False), True
    ) == (True, False)
    assert runner.validate_final_decision(final(
        "COMPACT_NONINFERIOR_ADAPTIVE_DEV", pilot=True, best=False, noninferior=True), True
    ) == (False, True)
    assert runner.validate_final_decision(final(
        "NO_GO", pilot=True, best=False, noninferior=False), True
    ) == (False, False)
    assert runner.validate_final_decision(final(
        "PILOT_NO_GO_FINAL_DESCRIPTIVE", pilot=False, best=False, noninferior=False), False
    ) == (False, False)
    invalid_failed_pilot = final(
        "PILOT_NO_GO_FINAL_DESCRIPTIVE", pilot=False, best=False, noninferior=True)
    try:
        runner.validate_final_decision(invalid_failed_pilot, False)
    except runner.RunnerError:
        pass
    else:
        raise AssertionError("failed-pilot decision carried a noninferiority claim")
    invalid = final("OVERALL_BEST_ADAPTIVE_DEV", pilot=True, best=False, noninferior=True)
    try:
        runner.validate_final_decision(invalid, True)
    except runner.RunnerError:
        pass
    else:
        raise AssertionError("inconsistent overall-best decision was accepted")


if __name__ == "__main__":
    tests = (
        test_grid_is_exact_and_unique,
        test_auxiliary_contracts_are_exact,
        test_parameter_contract_distinguishes_compact_and_expanded,
        test_expected_args_and_command_force_zero_v8_auxiliary,
        test_v8_history_rejects_every_hierarchy_field,
        test_v8_model_state_namespace_shape_teacher_and_redundancy,
        test_v7_student_namespace_mapping_is_exact,
        test_sealed_v7_reference_is_bound,
        test_protocol_freezes_scope_grid_thresholds_and_wandb,
        test_pilot_labels_are_fail_closed,
        test_final_labels_and_fields_are_consistent,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
    print(f"All {len(tests)} HACSSM-v8 protocol tests passed.")
