#!/usr/bin/env python3
"""Dependency-light tests for the locked HACSSM-v6 runner/analyzer contract."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.analyze_hacssm_v6 as analysis
import scripts.run_hacssm_v6 as runner


def synthetic_rows(seeds, *, candidate=0.80):
    values = {
        "ssm": 1.00,
        "hacsmv4_two_noaux": 0.95,
        "hacssmv5_noaux": 1.02,
        "hacssmv6_noaux": 0.92,
        "hacssmv6_aux_noaction": 0.93,
        "hacssmv6_uniform": 0.91,
        "hacssmv6_sourcegrad": 0.90,
        "hacssmv6_fastonly": 0.91,
        "hacssmv6_mediumonly": 0.92,
        "hacssmv6_noaction": 1.08,
        "hacssmv6_static": 0.90,
        "hacssmv6_single": 1.07,
        "hacssmv6": candidate,
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
        {"run": f"{row['env']}:{row['design']}:{row['seed']}",
         "relative_improvement": 0.001}
        for row in rows
    ]


def test_schedule_and_grid_contract() -> None:
    assert len(runner.DESIGNS) == 13
    assert len(runner.PILOT_JOBS) == 195
    assert len(runner.COMPLETION_JOBS) == 130
    assert len(runner.ALL_JOBS) == 325
    assert runner.scheduled_weight(.02, "v6_bootstrap", 40) == .02
    assert abs(runner.scheduled_weight(.02, "v6_bootstrap", 70) - .01) < 1e-12
    assert runner.scheduled_weight(.02, "v6_bootstrap", 100) == 0.0
    assert runner.scheduled_weight(.02, "v6_bootstrap", 200) == 0.0
    assert runner.design_aux_contract("hacssmv6_noaux") == (.02, "v6_bootstrap", False)
    assert runner.design_aux_contract("hacssmv6") == (.02, "v6_bootstrap", True)
    contract = runner.memory_contract()
    assert contract["memory_parameters"]["hacssmv6_all_modes"] == 34_564
    assert contract["streaming_recurrent_floats"]["hacssmv6_all_modes"] == 256


def test_online_wandb_rollout_command_is_mandatory() -> None:
    runner.configure_shared()
    job = runner.PILOT_JOBS[0]
    expected = runner.expected_args(job)
    assert expected["wandb"] is True
    assert expected["wandb_mode"] == "online"
    assert expected["wandb_study"] == "hacssm-v6"
    assert expected["eval_rollout_episode"] == 0
    command = runner.train_command(sys.executable, job)
    for flag in (
        "--wandb", "--wandb-project", "--wandb-entity", "--wandb-mode",
        "--wandb-study", "--eval-rollout-cache", "--eval-rollout-episode",
    ):
        assert flag in command
    assert "--no-wandb" not in command


def test_protocol_is_json_stable_and_declares_no_hidden_targets() -> None:
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
    assert protocol["self_supervision_contract"]["hidden_clean_blackout_targets_used"] is False
    assert protocol["self_supervision_contract"]["hierarchy"] == {
        "fast": [1, 2], "medium": [4, 8]}
    assert protocol == json.loads(json.dumps(protocol))


def test_pilot_and_final_success_require_self_supervised_gain() -> None:
    analysis.configure_shared()
    pilot_rows = synthetic_rows(analysis.PILOT_SEEDS)
    pilot_contrasts = analysis.shared.contrast_rows(
        pilot_rows, candidate=analysis.CANDIDATE)
    pilot = analysis.pilot_decision(
        pilot_rows, convergence(pilot_rows), pilot_contrasts)
    assert pilot["decision"] == "PILOT_PASS"
    assert all(pilot["criteria"].values())

    final_rows = synthetic_rows(analysis.FINAL_SEEDS)
    final_contrasts = analysis.shared.contrast_rows(
        final_rows, candidate=analysis.CANDIDATE)
    final = analysis.final_summary(
        final_rows, convergence(final_rows), final_contrasts,
        pilot_screen_passed=True)
    assert final["decision"] == "OVERALL_BEST_IN_LOCKED_GRID"
    assert final["good_enough_for_v6_stop"] is True
    assert final["trigger_v7"] is False
    assert all(final["criteria"].values())

    no_ssl_gain = synthetic_rows(analysis.FINAL_SEEDS, candidate=0.93)
    failed = analysis.final_summary(
        no_ssl_gain, convergence(no_ssl_gain),
        analysis.shared.contrast_rows(no_ssl_gain, candidate=analysis.CANDIDATE),
        pilot_screen_passed=True)
    assert failed["decision"] != "OVERALL_BEST_IN_LOCKED_GRID"
    assert failed["trigger_v7"] is True
    assert not failed["criteria"]["vs_noaux_reduction_ge_1pct"]


if __name__ == "__main__":
    tests = (
        test_schedule_and_grid_contract,
        test_online_wandb_rollout_command_is_mandatory,
        test_protocol_is_json_stable_and_declares_no_hidden_targets,
        test_pilot_and_final_success_require_self_supervised_gain,
    )
    for test in tests:
        test()
        print(f"{test.__name__}: OK")
    print(f"All {len(tests)} HACSSM-v6 protocol tests passed.")
