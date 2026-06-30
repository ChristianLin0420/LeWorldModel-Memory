#!/usr/bin/env python3
"""Frozen launcher-contract tests for the CVPF-v15 screen."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_cvpf_v15_screen as run


def flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_frozen_grid_and_task_pinning() -> None:
    assert run.TASKS == (
        "cartpole.swingup", "fish.swim", "pendulum.swingup", "walker.walk")
    assert run.SEED == 15001 and run.EPOCHS == 30
    assert len(run.CVPF_DESIGNS) == 8
    assert len(run.BASELINES) == 5
    assert len(run.DESIGNS) == 13
    assert len(run.TASKS) * len(run.DESIGNS) == 52
    assert run.DEFAULT_STUDY == "hacssm-v15-screen-cvpf30"
    assert run.DEFAULT_OUTPUT_ROOT == Path("outputs/hacssm_v15_screen_cvpf30")
    assert run.DEFAULT_LOG_ROOT == Path("logs/hacssm_v15_screen_cvpf30")
    assert run.BLAS_THREADS == 4


def test_commands_freeze_budget_and_online_wandb() -> None:
    root = Path("/tmp/frozen-v15")
    for task in run.TASKS:
        for design in run.DESIGNS:
            command = run.train_command(
                "python", root, run.DEFAULT_STUDY, 30, task, design)
            expected = {
                "--memory-mode": design, "--seed": "15001", "--epochs": "30",
                "--batch-size": "64", "--lr": "0.0003",
                "--weight-decay": "0.00001", "--num-workers": "2",
                "--embed-dim": "128", "--encoder-layers": "6",
                "--predictor-layers": "4", "--history-len": "3",
                "--probe-ridge": "0.001", "--corruption-seed": "11012",
                "--wandb-mode": "online", "--wandb-study": run.DEFAULT_STUDY,
            }
            for name, value in expected.items():
                assert flag(command, name) == value
            assert "--wandb" in command
            assert Path(command[1]).name == "train_cvpf_v15.py"


def test_names_include_kdio_ranking_delegation() -> None:
    root = Path("/tmp/v15")
    assert run.run_directory(root, "fish.swim", "cvpfv15").name == (
        "lewm-dmc:fish.swim-cvpfv15-s15001")
    assert run.run_directory(root, "fish.swim", "kdiov11").name.endswith(
        "-rank-rawdiff_displacement_detached")


def test_continuation_is_complete_and_never_pre_authorized() -> None:
    manifest = run.continuation_manifest("python")
    assert manifest["status"] == "CONDITIONAL_NOT_AUTHORIZED"
    assert manifest["launch_performed"] is False
    assert manifest["automatic_launch_supported"] is False
    assert manifest["designs"] == list(run.DESIGNS)
    assert manifest["seeds"] == [15002, 15003, 15004]
    assert manifest["epochs"] == 100
    assert manifest["runs"] == 13 * 4 * 3 == 156
    commands = manifest["commands"]
    assert isinstance(commands, list) and len(commands) == 156
    assert len({tuple(command) for command in commands}) == 156
    assert manifest["commands_sha256"] == run.json_sha256(commands)


def test_source_manifest_covers_v15_and_inherited_paths() -> None:
    required = {
        Path("lewm/models/cvpf.py"), Path("scripts/train_cvpf_v15.py"),
        Path("scripts/run_cvpf_v15_screen.py"),
        Path("scripts/analyze_cvpf_v15_screen.py"),
        Path("scripts/audit_cvpf_v15_screen.py"),
        Path("lewm/models/cf_ebo.py"), Path("scripts/train_cf_ebo_v14.py"),
        Path("lewm/models/cf_hiro.py"), Path("scripts/train_cf_hiro_v13.py"),
        Path("lewm/models/siro.py"), Path("scripts/train_siro_v12.py"),
        Path("scripts/train_hacssm_v11.py"), Path("scripts/train_hacssm_v10.py"),
        Path("scripts/hacssm_v11_data.py"),
    }
    assert required <= set(run.SOURCE_PATHS)


def test_git_receipt_requires_clean_pushed_head() -> None:
    original = run.subprocess.run

    def install(values):
        def fake(arguments, **kwargs):
            del kwargs
            return SimpleNamespace(stdout=values[tuple(arguments[1:])] + "\n")
        run.subprocess.run = fake

    common = {
        ("branch", "--show-current"): "learnable-memory",
        ("rev-parse", "HEAD"): "a" * 40,
        ("rev-parse", "@{upstream}"): "a" * 40,
    }
    try:
        install({**common, ("status", "--porcelain", "--untracked-files=all"): " M x"})
        try:
            run.git_receipt()
        except RuntimeError as error:
            assert "dirty worktree" in str(error)
        else:
            raise AssertionError("dirty worktree accepted")
        install({
            **common, ("status", "--porcelain", "--untracked-files=all"): "",
            ("rev-parse", "@{upstream}"): "b" * 40})
        try:
            run.git_receipt()
        except RuntimeError as error:
            assert "before push" in str(error)
        else:
            raise AssertionError("unpushed head accepted")
        install({**common, ("status", "--porcelain", "--untracked-files=all"): ""})
        assert run.git_receipt()["git_head_pushed"] is True
    finally:
        run.subprocess.run = original


def main() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} V15 screen-runner tests passed.")


if __name__ == "__main__":
    main()
