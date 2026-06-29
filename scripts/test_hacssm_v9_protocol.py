#!/usr/bin/env python3
"""Dependency-light protocol tests for the locked HACSSM-v9 LOIF study."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.run_hacssm_v9 as runner
import scripts.hacssm_v9_diagnostics as diagnostics


V9_DESIGNS = {
    "loifv9",
    "loifv9_fixedalpha",
    "loifv9_globalR",
    "loifv9_innovationonly",
    "loifv9_latentonly",
    "loifv9_uniformfusion",
    "loifv9_noaction",
    "loifv9_singlebank",
}
REFERENCE_DESIGNS = {
    "ssm", "hacssmv7_sharedaction", "hacssmv8", "hacssmv8_dynamic",
    "hacssmv8_static",
}


def job(design: str = "loifv9"):
    occ, clean = runner.ENVIRONMENTS[0]
    return runner.shared.Job("pilot", 0, occ, clean, design)


def clean_history(*, hierarchy_diagnostics: bool = False) -> list[dict]:
    values = {
        "loss": 1.0,
        "pred_loss": 0.8,
        "sigreg_loss": 2.0,
        "pred_loss_all_valid": 0.8,
        "pred_loss_first_post": 0.9,
    }
    if hierarchy_diagnostics:
        values.update({
            "hier_loss": 1.0,
            "hier_loss_fast": 1.0,
            "hier_loss_medium": 1.0,
            "hier_loss_bridge": 1.0,
            "hier_loss_recovery": 1.0,
            "hier_overlap": 0.0,
            "hier_loss_weight": 0.0,
        })
    return [
        {"epoch": epoch, "train": dict(values), "val": dict(values)}
        for epoch in range(1, 201)
    ]


def _build_protocol() -> dict:
    originals = (
        runner.shared.feature_snapshot,
        runner.shared.eval_rollout_snapshot,
        runner.shared.source_snapshot,
    )
    runner.shared.feature_snapshot = lambda: {"features": {"sha256": "f" * 64}}
    runner.shared.eval_rollout_snapshot = lambda: {"rollout": {"sha256": "e" * 64}}
    runner.shared.source_snapshot = lambda: {"source": {"sha256": "s" * 64}}
    try:
        return runner.build_protocol("a" * 40, True, {"mode": "online"})
    finally:
        (
            runner.shared.feature_snapshot,
            runner.shared.eval_rollout_snapshot,
            runner.shared.source_snapshot,
        ) = originals


def test_grid_is_exact_unique_and_stage_locked() -> None:
    assert len(runner.ENVIRONMENTS) == 5
    assert set(runner.DESIGNS) == V9_DESIGNS | REFERENCE_DESIGNS
    assert len(runner.DESIGNS) == 13
    assert set(runner.V9_DESIGNS) == V9_DESIGNS
    assert runner.PILOT_SEEDS == (0, 1, 2)
    assert runner.COMPLETION_SEEDS == (3, 4)
    assert len(runner.PILOT_JOBS) == 195
    assert len(runner.COMPLETION_JOBS) == 130
    assert len(runner.ALL_JOBS) == 325
    assert len({item.run_name for item in runner.ALL_JOBS}) == 325
    assert all(item.stage == "pilot" for item in runner.PILOT_JOBS)
    assert all(item.stage == "completion" for item in runner.COMPLETION_JOBS)

    protocol = _build_protocol()
    # Protocol publication is a JSON round trip; tuple/list drift would make the freshly
    # published lock differ from the still-live in-memory object before the first cell.
    assert runner.shared.stable_equal(
        protocol, json.loads(json.dumps(protocol, allow_nan=False))
    )
    assert protocol["adaptive_development_only"] is True
    assert protocol["stages"]["pilot"] == {
        "designs": list(runner.DESIGNS), "seeds": [0, 1, 2], "runs": 195,
    }
    assert protocol["stages"]["completion"]["runs"] == 130
    assert protocol["stages"]["completion"]["completed_total_runs"] == 325
    assert protocol["stages"]["completion"]["runs_regardless_of_pilot_screen"] is True


def test_every_cell_uses_the_same_primary_objective() -> None:
    assert runner.COMMON["first_post_loss_weight"] == 0.0
    assert runner.COMMON["fixed_alpha"] is False
    for design in runner.DESIGNS:
        assert runner.design_aux_contract(design) == (0.0, "fixed", False)
        command = runner.train_command("python", job(design))
        assert "--fixed-alpha" not in command
        assert float(command[command.index("--first-post-loss-weight") + 1]) == 0.0
        assert float(command[command.index("--hier-loss-weight") + 1]) == 0.0
        assert command[command.index("--hier-loss-schedule") + 1] == "fixed"
        assert command[command.index("--memory-mode") + 1] == design

    protocol = _build_protocol()
    for design, entry in protocol["design_protocol"].items():
        assert design in runner.DESIGNS
        assert entry["first_post_loss_weight"] == 0.0
        assert entry["hier_loss_weight"] == 0.0
        assert entry["hier_loss_schedule"] == "fixed"
        assert entry["auxiliary_gradients_active"] is False
    architecture = protocol["v9_architecture_contract"]
    assert architecture["training_signal"] == "ordinary visible-target next-latent MSE only"
    assert architecture["internal_auxiliary"] == "none"
    assert architecture["teacher"] == "none"
    assert architecture["hidden_clean_blackout_targets_used"] is False
    assert architecture["teacher_or_hidden_clean_training_target"] is False
    assert architecture["fixed_memory_timescale"] is False
    assert architecture["memory_specific_objective_weight"] is False


def test_references_are_retrained_and_no_checkpoint_is_reused() -> None:
    protocol = _build_protocol()
    # All references occupy ordinary cells in the V9 root. Historical checkpoints are
    # provenance only and cannot enter a command or initialize a model.
    expected_reference_jobs = 5 * len(REFERENCE_DESIGNS) * 5
    assert sum(item.design in REFERENCE_DESIGNS for item in runner.ALL_JOBS) == (
        expected_reference_jobs)
    for design in REFERENCE_DESIGNS:
        command = runner.train_command("python", job(design))
        assert "--resume" not in command
        assert "--encoder-checkpoint" not in command
        assert all("hacssm_v7_shared" not in token and "hacssm_v8_shared" not in token
                   for token in command)
        output = Path(command[command.index("--output-dir") + 1])
        assert output.name == "hacssm_v9_shared"

    fresh = protocol["fresh_training_contract"]
    assert fresh == {
        "all_cells_trained_from_scratch": True,
        "checkpoint_reuse_allowed": False,
        "optimizer_state_reuse_allowed": False,
        "history_reuse_allowed": False,
        "sealed_v7_v8_models_are_not_inputs": True,
        "objective_mismatched_checkpoint_reuse": False,
        "reused_inputs_only": ["fixed feature caches", "fixed rollout pixel caches"],
        "command_has_checkpoint_or_resume_input": False,
    }


def test_history_contract_forbids_v9_aux_but_allows_zero_weight_v7_receipts() -> None:
    for design in V9_DESIGNS | {"ssm", "hacssmv8", "hacssmv8_dynamic", "hacssmv8_static"}:
        runner.validate_history(clean_history(), job(design))
        contaminated = clean_history()
        contaminated[19]["val"]["hier_loss"] = 0.0
        try:
            runner.validate_history(contaminated, job(design))
        except runner.RunnerError as exc:
            assert "hier" in str(exc).lower()
        else:
            raise AssertionError(f"{design} accepted an internal hierarchy field")

    # V7 retains its teacher as an architecture diagnostic, but the weight and effective
    # objective contribution are exactly zero in this same-objective rerun.
    v7_history = clean_history(hierarchy_diagnostics=True)
    runner.validate_history(v7_history, job("hacssmv7_sharedaction"))
    v7_history[100]["train"]["hier_loss_weight"] = 1e-3
    try:
        runner.validate_history(v7_history, job("hacssmv7_sharedaction"))
    except runner.RunnerError as exc:
        assert "auxiliary" in str(exc).lower() or "weight" in str(exc).lower()
    else:
        raise AssertionError("V7 reference accepted a nonzero auxiliary weight")


def test_wandb_rollout_and_logging_contract_is_online_and_per_cell() -> None:
    protocol = _build_protocol()
    assert runner.WANDB_MODE == "online"
    assert runner.WANDB_STUDY == "hacssm-v9"
    assert protocol["wandb_requirements"] == {
        "all_cells_online": True,
        "complete_epoch_history_per_cell": 200,
        "evaluation_rollout_npz_table_video_per_cell": True,
    }
    for design in runner.DESIGNS:
        command = runner.train_command("python", job(design))
        assert "--wandb" in command
        assert command[command.index("--wandb-mode") + 1] == "online"
        assert command[command.index("--wandb-study") + 1] == "hacssm-v9"
        assert "--eval-rollout-cache" in command
        assert int(command[command.index("--epochs") + 1]) == 200
    # Hash the indirect package imports as well as the obvious trainer/model files so the
    # source artifact is a complete record of the code imported by an official cell.
    source_paths = {path.as_posix() for path in runner.SOURCE_FILES}
    assert {
        "lewm/__init__.py",
        "lewm/envs/__init__.py",
        "lewm/envs/memory_envs.py",
        "lewm/envs/two_room.py",
        "lewm/eval/__init__.py",
        "lewm/eval/memory_probe.py",
        "lewm/eval/probing.py",
        "lewm/models/__init__.py",
    } <= source_paths
    snapshot = runner.source_snapshot()
    assert set(snapshot) == source_paths
    assert snapshot["lewm/__init__.py"]["bytes"] == 0
    assert snapshot["lewm/__init__.py"]["sha256"] == (
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )


def test_diagnostics_contract_required_fields_and_hash_are_frozen() -> None:
    protocol = _build_protocol()
    contract = protocol["diagnostics_contract"]
    assert contract == runner.diagnostic_contract()
    assert contract["schema_version"] == diagnostics.DIAGNOSTICS_SCHEMA_VERSION == 1
    assert contract["donor_seed"] == diagnostics.DONOR_SEED == 9009
    assert contract["donor_contract"] == diagnostics.DONOR_CONTRACT
    assert contract["donor_contract_sha256"] == diagnostics.DONOR_CONTRACT_SHA256
    assert len(contract["donor_contract_sha256"]) == 64
    assert contract["required_candidate_global_fields"] == [
        "alpha_fast", "alpha_slow", "q_fast", "q_slow",
    ]
    phase_fields = contract["required_candidate_phase_fields"]
    assert len(phase_fields) == (
        len(runner.V9_DIAGNOSTIC_PHASES) * len(runner.V9_DIAGNOSTIC_STATS))
    assert set(phase_fields) == {
        f"loif_{stat}_{phase}"
        for phase in runner.V9_DIAGNOSTIC_PHASES
        for stat in runner.V9_DIAGNOSTIC_STATS
    }
    intervention_fields = contract["required_candidate_intervention_fields"]
    assert set(intervention_fields) == {
        f"clean_mse_{phase}_resistance_{kind}"
        for phase in runner.V9_INTERVENTION_PHASES
        for kind in ("permuted", "mean")
    }
    assert contract["training_only_donors"] is True
    assert contract["future_or_validation_donors"] is False
    assert protocol["bootstrap_contract"] == runner.BOOTSTRAP_CONTRACT
    assert protocol["bootstrap_contract_sha256"] == runner.BOOTSTRAP_CONTRACT_SHA256
    assert len(protocol["bootstrap_contract_sha256"]) == 64


def test_thresholds_are_frozen_exactly_as_documented() -> None:
    protocol = _build_protocol()
    pilot = protocol["pilot_success_criteria"]
    final = protocol["final_success_criteria"]
    assert pilot["vs_ssm"] == ">=6% reduction, >=10/15 cells, >=4/5 environments"
    assert final["vs_ssm"] == ">=7% reduction, >=20/25 cells, >=4/5 environments"
    assert pilot["vs_each_headline_reference"] == (
        ">=0.5% reduction, >=9/15, >=3/5"
    )
    assert final["vs_each_headline_reference"] == (
        ">=1% reduction, >=15/25, >=3/5"
    )
    assert pilot["vs_each_adaptive_evidence_control"] == (
        ">=0.25% reduction, >=9/15, >=3/5"
    )
    assert final["vs_each_adaptive_evidence_control"] == (
        ">=0.5% reduction, >=14/25, >=3/5"
    )
    assert pilot["vs_better_v8_endpoint_envelope"] == (
        ">0% reduction, >=9/15, >=3/5"
    )
    assert final["vs_better_v8_endpoint_envelope"] == (
        ">=0.5% reduction, >=14/25, >=3/5"
    )
    assert pilot["vs_uniform_fusion"] == (
        ">0% reduction, >=8/15, >=3/5"
    )
    assert final["vs_uniform_fusion"] == (
        ">=0.5% reduction, >=14/25, >=3/5"
    )
    assert pilot["vs_each_structural_control"] == (
        ">=3% reduction, >=11/15, >=3/5"
    )
    assert final["vs_each_structural_control"] == (
        ">=3% reduction, >=17/25, >=3/5"
    )
    assert final["bootstrap"] == (
        "crossed environment x seed 90% lower bound >0 for both headline references, "
        "four adaptive-evidence controls, and endpoint envelope"
    )
    assert final["full_grid_environment_envelope"] == ">=3/5"
    assert final["last_visible_hold"] == ">=4/5"
    assert final["convergence"] == "absolute median <1%, p95 <3%, maximum <5%"
    assert protocol["phase_success_criteria"] == {
        "reference": "hacssmv7_sharedaction",
        "metrics": ["clean_mse_deep_blackout", "clean_mse_all"],
        "each_metric": ">-1% paired reduction and >=3/5 environment effects >-1%",
    }
    assert protocol["intervention_success_criteria"] == {
        "candidate": "loifv9",
        "metric": "clean_mse_first_post",
        "pilot_each_intervention": ">=0.25% reduction, >=9/15, >=3/5",
        "final_each_intervention": ">=0.5% reduction, >=14/25, >=3/5",
        "interventions": ["resistance_permuted", "resistance_mean"],
    }
    diagnostics = protocol["diagnostics_contract"]
    assert diagnostics["donor_seed"] == 9009
    assert diagnostics["training_only_donors"] is True
    assert diagnostics["future_or_validation_donors"] is False


def test_pilot_lock_is_fail_closed_and_cannot_be_rescued_by_completion() -> None:
    protocol = _build_protocol()
    gate = protocol["analysis_gate"]
    assert gate["fail_closed_result"] == "NO_GO"
    assert gate["scope"] == "adaptive_development_only"
    assert protocol["final_success_criteria"]["requires_pilot_pass"] is True
    assert protocol["stages"]["completion"]["runs_regardless_of_pilot_screen"] is True

    original = runner.PILOT_DECISION_PATH
    with tempfile.TemporaryDirectory() as directory:
        runner.PILOT_DECISION_PATH = Path(directory) / "pilot_decision.json"
        try:
            runner.PILOT_DECISION_PATH.write_text(json.dumps({
                "decision": gate["pilot_pass_result"],
                "pilot_screen_passed": True,
                "scope": "adaptive_development_only",
            }))
            passed, _ = runner.read_pilot_decision()
            assert passed is True

            runner.PILOT_DECISION_PATH.write_text(json.dumps({
                "decision": "NO_GO",
                "pilot_screen_passed": False,
                "scope": "adaptive_development_only",
            }))
            passed, _ = runner.read_pilot_decision()
            assert passed is False

            # A contradictory claim after a locked miss must fail closed.
            runner.PILOT_DECISION_PATH.write_text(json.dumps({
                "decision": gate["pilot_pass_result"],
                "pilot_screen_passed": False,
                "scope": "adaptive_development_only",
            }))
            try:
                runner.read_pilot_decision()
            except runner.RunnerError:
                pass
            else:
                raise AssertionError("contradictory pilot label was accepted")
        finally:
            runner.PILOT_DECISION_PATH = original


if __name__ == "__main__":
    tests = (
        test_grid_is_exact_unique_and_stage_locked,
        test_every_cell_uses_the_same_primary_objective,
        test_references_are_retrained_and_no_checkpoint_is_reused,
        test_history_contract_forbids_v9_aux_but_allows_zero_weight_v7_receipts,
        test_wandb_rollout_and_logging_contract_is_online_and_per_cell,
        test_diagnostics_contract_required_fields_and_hash_are_frozen,
        test_thresholds_are_frozen_exactly_as_documented,
        test_pilot_lock_is_fail_closed_and_cannot_be_rescued_by_completion,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
    print(f"All {len(tests)} HACSSM-v9 protocol tests passed.")
