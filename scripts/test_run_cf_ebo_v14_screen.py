#!/usr/bin/env python3
"""Frozen launcher-contract tests for the CF-EBO-v14 screen."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_cf_ebo_v14_screen as run
from scripts.train_cf_ebo_v14 import DESIGNS


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_frozen_grid_and_task_pinning_contract() -> None:
    assert run.TASKS == (
        "cartpole.swingup", "fish.swim", "pendulum.swingup", "walker.walk")
    assert run.SEED == 14001
    assert len(DESIGNS) == 10
    assert len(run.TASKS) * len(DESIGNS) == 40
    assert run.DEFAULT_STUDY == "hacssm-v14-screen-cfebo30"
    assert run.DEFAULT_OUTPUT_ROOT == Path("outputs/hacssm_v14_screen_cfebo30")
    assert run.DEFAULT_LOG_ROOT == Path("logs/hacssm_v14_screen_cfebo30")
    assert run.BLAS_THREADS == 4


def test_commands_keep_exact_host_budget_and_online_wandb() -> None:
    root = Path("/tmp/frozen-v14")
    for task in run.TASKS:
        for design in DESIGNS:
            command = run.train_command(
                "python", root, run.DEFAULT_STUDY, 30, task, design)
            expected = {
                "--memory-mode": design, "--seed": "14001", "--epochs": "30",
                "--batch-size": "64", "--lr": "0.0003",
                "--weight-decay": "0.00001", "--num-workers": "2",
                "--img-size": "64", "--patch-size": "8", "--embed-dim": "128",
                "--encoder-layers": "6", "--encoder-heads": "4",
                "--predictor-layers": "4", "--predictor-heads": "8",
                "--history-len": "3", "--dropout": "0.1",
                "--sigreg-lambda": "0.1", "--sigreg-projections": "512",
                "--probe-ridge": "0.001", "--corruption-seed": "11012",
                "--wandb-entity": run.WANDB_ENTITY,
                "--wandb-project": run.WANDB_PROJECT,
                "--wandb-mode": "online", "--wandb-study": run.DEFAULT_STUDY,
            }
            for flag, value in expected.items():
                assert _flag(command, flag) == value
            assert "--wandb" in command
            assert Path(command[1]).name == "train_cf_ebo_v14.py"


def test_run_names_match_candidate_and_kdio_delegation() -> None:
    root = Path("/tmp/v14")
    ordinary = run.run_directory(root, "fish.swim", "cfebov14")
    assert ordinary.name == "lewm-dmc:fish.swim-cfebov14-s14001"
    v13 = run.run_directory(root, "fish.swim", "cfhirov13_nocorrect")
    assert v13.name == "lewm-dmc:fish.swim-cfhirov13_nocorrect-s14001"
    kdio = run.run_directory(root, "fish.swim", "kdiov11")
    assert kdio.name.endswith("-rank-rawdiff_displacement_detached")


def test_continuation_is_complete_but_never_authorized_or_launched() -> None:
    manifest = run.continuation_manifest("python")
    assert manifest["status"] == "CONDITIONAL_NOT_AUTHORIZED"
    assert manifest["launch_performed"] is False
    assert manifest["automatic_launch_supported"] is False
    assert manifest["designs"] == list(run.CONTINUATION_DESIGNS)
    assert manifest["seeds"] == [14002, 14003, 14004]
    assert manifest["epochs"] == 100
    assert manifest["runs"] == 8 * 4 * 3 == 96
    commands = manifest["commands"]
    assert isinstance(commands, list) and len(commands) == 96
    assert len({tuple(command) for command in commands}) == 96
    for command in commands:
        assert _flag(command, "--memory-mode") in run.CONTINUATION_DESIGNS
        assert int(_flag(command, "--seed")) in run.CONTINUATION_SEEDS
        assert _flag(command, "--epochs") == "100"
        assert _flag(command, "--wandb-study") == run.CONTINUATION_STUDY
        assert "--wandb" in command


def test_source_manifest_covers_v14_and_inherited_training_paths() -> None:
    required = {
        Path("lewm/models/cf_ebo.py"),
        Path("lewm/models/cf_hiro.py"),
        Path("lewm/models/memory_model.py"),
        Path("scripts/train_cf_ebo_v14.py"),
        Path("scripts/run_cf_ebo_v14_screen.py"),
        Path("scripts/analyze_cf_ebo_v14_screen.py"),
        Path("scripts/audit_cf_ebo_v14_screen.py"),
        Path("scripts/train_cf_hiro_v13.py"),
        Path("scripts/train_siro_v12.py"),
        Path("scripts/train_hacssm_v11.py"),
        Path("scripts/hacssm_v11_data.py"),
    }
    assert required <= set(run.SOURCE_PATHS)
    assert all((ROOT / path).is_file() for path in run.SOURCE_PATHS)


def test_git_receipt_requires_clean_exactly_pushed_head() -> None:
    original = run.subprocess.run

    def install(values):
        def fake(arguments, **kwargs):
            del kwargs
            key = tuple(arguments[1:])
            return SimpleNamespace(stdout=values[key] + "\n")
        run.subprocess.run = fake

    common = {
        ("branch", "--show-current"): "learnable-memory",
        ("rev-parse", "HEAD"): "a" * 40,
        ("rev-parse", "@{upstream}"): "a" * 40,
    }
    try:
        install({
            **common,
            ("status", "--porcelain", "--untracked-files=all"): " M dirty.py",
        })
        try:
            run._git_receipt()
        except RuntimeError as error:
            assert "dirty worktree" in str(error)
        else:
            raise AssertionError("dirty worktree was accepted")

        install({
            **common,
            ("status", "--porcelain", "--untracked-files=all"): "",
            ("rev-parse", "@{upstream}"): "b" * 40,
        })
        try:
            run._git_receipt()
        except RuntimeError as error:
            assert "before push" in str(error)
        else:
            raise AssertionError("unpushed HEAD was accepted")

        install({
            **common,
            ("status", "--porcelain", "--untracked-files=all"): "",
        })
        receipt = run._git_receipt()
        assert receipt == {
            "git_branch": "learnable-memory",
            "git_commit": "a" * 40,
            "git_upstream_commit": "a" * 40,
            "git_worktree_clean": True,
            "git_head_pushed": True,
        }
    finally:
        run.subprocess.run = original


def main() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} V14 screen-runner tests passed.")


if __name__ == "__main__":
    main()
