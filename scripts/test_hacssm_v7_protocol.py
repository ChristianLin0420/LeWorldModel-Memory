#!/usr/bin/env python3
"""Dependency-light tests for the locked HACSSM/HCRD-v7 study contract."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.analyze_hacssm_v7 as analysis
import scripts.run_hacssm_v7 as runner


def synthetic_rows(seeds, *, candidate=0.75):
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


def convergence(rows):
    return [
        {
            "run": f"{row['env']}:{row['design']}:{row['seed']}",
            "relative_improvement": 0.001,
        }
        for row in rows
    ]


def history(design, *, overlap=0.0):
    base, schedule, active = runner.design_aux_contract(design)
    records = []
    for epoch in range(1, 201):
        values = {"loss": 1.0, "pred_loss": 0.9, "sigreg_loss": 0.1}
        if design in runner.HIER_DESIGNS:
            values.update({
                "hier_loss": 0.2,
                "hier_loss_fast": 0.2,
                "hier_loss_medium": 0.2,
                "hier_loss_weight": (
                    runner.scheduled_weight(base, schedule, epoch) if active else 0.0),
            })
        if design in runner.V7_DESIGNS:
            values.update({
                "hier_loss_bridge": 0.2,
                "hier_loss_recovery": 0.2,
                "hier_overlap": overlap,
            })
        records.append({"epoch": epoch, "train": dict(values), "val": dict(values)})
    return records


def test_schedule_grid_and_parameter_contract() -> None:
    assert len(runner.DESIGNS) == 13
    assert len(runner.PILOT_JOBS) == 195
    assert len(runner.COMPLETION_JOBS) == 130
    assert len(runner.ALL_JOBS) == 325
    assert runner.scheduled_weight(.02, "v6_bootstrap", 40) == .02
    assert abs(runner.scheduled_weight(.02, "v6_bootstrap", 70) - .01) < 1e-12
    assert runner.scheduled_weight(.02, "v6_bootstrap", 100) == 0.0
    assert runner.scheduled_weight(.02, "v6_bootstrap", 200) == 0.0
    assert runner.design_aux_contract("hacssmv7_noaux") == (
        .02, "v6_bootstrap", False)
    assert runner.design_aux_contract("hacssmv7") == (.02, "v6_bootstrap", True)
    assert runner.design_aux_contract("hacssmv6_static") == (
        .02, "v6_bootstrap", True)
    contract = runner.memory_contract()
    assert contract["memory_parameters"]["hacssmv7_all_modes"] == 36_102
    assert contract["trainable_memory_parameters_include_teacher"] is False
    assert contract["checkpoint_contains_frozen_ema_teacher"] is True
    assert contract["streaming_recurrent_floats"]["hacssmv7_all_modes"] == 256


def test_objective_metadata_covers_v6_anchors_and_v7_variants() -> None:
    v6 = runner.objective_metadata("hacssmv6_static")
    assert v6["hier_objective_schema_version"] == 1
    assert v6["hier_target_kind"] == "same_level_posterior_stop_gradient"
    full = runner.objective_metadata("hacssmv7")
    assert full["hier_objective_schema_version"] == 2
    assert full["hier_teacher_momentum"] == 0.99
    assert full["hier_level_specific_action"] is True
    assert full["hier_shrinkage_kind"] == "learned_static_dynamic_convex"
    assert full["hier_aux_kind"] == "counterfactual_bridge_and_recovery"
    assert full["hier_hidden_clean_targets_used"] is False
    assert runner.objective_metadata("hacssmv7_sharedaction")[
        "hier_level_specific_action"] is False
    assert runner.objective_metadata("hacssmv7_noshrink")[
        "hier_shrinkage_kind"] == "dynamic_only"
    assert runner.objective_metadata("hacssmv7_actiononly")[
        "hier_aux_kind"] == "action_only"
    assert runner.objective_metadata("hacssmv7_norecovery")[
        "hier_aux_kind"] == "bridge_only"


def test_history_fails_closed_on_hidden_overlap() -> None:
    job = next(job for job in runner.PILOT_JOBS if job.design == "hacssmv7")
    runner.validate_history(history(job.design), job)
    try:
        runner.validate_history(history(job.design, overlap=1.0), job)
    except runner.RunnerError as exc:
        assert "overlaps original hidden targets" in str(exc)
    else:
        raise AssertionError("nonzero original-hidden overlap was accepted")


def test_online_wandb_rollout_command_is_mandatory() -> None:
    runner.configure_shared()
    job = next(job for job in runner.PILOT_JOBS if job.design == "hacssmv7")
    expected = runner.expected_args(job)
    assert expected["wandb"] is True
    assert expected["wandb_mode"] == "online"
    assert expected["wandb_study"] == "hacssm-v7"
    assert expected["eval_rollout_episode"] == 0
    command = runner.train_command(sys.executable, job)
    for flag in (
        "--wandb", "--wandb-project", "--wandb-entity", "--wandb-mode",
        "--wandb-study", "--eval-rollout-cache", "--eval-rollout-episode",
    ):
        assert flag in command
    assert "--no-wandb" not in command
    assert command[command.index("--hier-loss-schedule") + 1] == "v6_bootstrap"


def test_protocol_is_stable_and_declares_visible_only_supervision() -> None:
    runner.configure_shared()
    preflight = {
        "authenticated": True,
        "base_url": runner.shared.WANDB_BASE_URL,
        "entity": runner.WANDB_ENTITY,
        "mode": runner.WANDB_MODE,
        "project": runner.WANDB_PROJECT,
        "sdk_version": "test",
        "study": runner.WANDB_STUDY,
    }
    protocol = runner.build_protocol("0" * 40, True, preflight)
    ssl = protocol["self_supervision_contract"]
    assert ssl["hidden_clean_blackout_targets_used"] is False
    assert ssl["required_original_hidden_overlap"] == 0
    assert ssl["hierarchy"] == {"fast": [1, 2], "medium": [4, 8]}
    assert set(ssl["variants"]) == runner.V7_DESIGNS
    assert protocol["wandb_requirements"]["complete_epoch_history_per_cell"] == 200
    assert protocol == json.loads(json.dumps(protocol))


def test_pilot_and_final_require_improvement_over_v6_and_controls() -> None:
    analysis.configure_shared()
    pilot_rows = synthetic_rows(analysis.PILOT_SEEDS)
    pilot = analysis.pilot_decision(
        pilot_rows, convergence(pilot_rows),
        analysis.shared.contrast_rows(pilot_rows, candidate=analysis.CANDIDATE))
    assert pilot["decision"] == "PILOT_PASS"
    assert all(pilot["criteria"].values())

    final_rows = synthetic_rows(analysis.FINAL_SEEDS)
    final = analysis.final_summary(
        final_rows, convergence(final_rows),
        analysis.shared.contrast_rows(final_rows, candidate=analysis.CANDIDATE),
        pilot_screen_passed=True)
    assert final["decision"] == "OVERALL_BEST_IN_LOCKED_GRID"
    assert final["good_enough_for_overall_best_claim"] is True
    assert all(final["criteria"].values())

    no_ssl_gain = synthetic_rows(analysis.FINAL_SEEDS, candidate=0.93)
    failed = analysis.final_summary(
        no_ssl_gain, convergence(no_ssl_gain),
        analysis.shared.contrast_rows(no_ssl_gain, candidate=analysis.CANDIDATE),
        pilot_screen_passed=True)
    assert failed["decision"] != "OVERALL_BEST_IN_LOCKED_GRID"
    assert failed["good_enough_for_overall_best_claim"] is False
    assert not failed["criteria"]["vs_noaux_reduction_ge_1pct"]


if __name__ == "__main__":
    tests = (
        test_schedule_grid_and_parameter_contract,
        test_objective_metadata_covers_v6_anchors_and_v7_variants,
        test_history_fails_closed_on_hidden_overlap,
        test_online_wandb_rollout_command_is_mandatory,
        test_protocol_is_stable_and_declares_visible_only_supervision,
        test_pilot_and_final_require_improvement_over_v6_and_controls,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
    print(f"All {len(tests)} HACSSM-v7 protocol tests passed.")
