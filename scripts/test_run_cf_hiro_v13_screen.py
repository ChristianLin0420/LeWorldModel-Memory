#!/usr/bin/env python3
"""Frozen launcher-contract tests for the CF-HIRO-v13 screen."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_cf_hiro_v13_screen as run
from scripts.train_cf_hiro_v13 import DESIGNS


def _flag(command, name):
    index = command.index(name)
    return command[index + 1]


def test_frozen_grid_and_task_pinning_contract() -> None:
    assert run.TASKS == (
        "cartpole.swingup", "fish.swim", "pendulum.swingup", "walker.walk")
    assert run.SEED == 13001
    assert len(DESIGNS) == 9
    assert len(run.TASKS) * len(DESIGNS) == 36
    assert run.DEFAULT_STUDY == "hacssm-v13-screen-cfhiro30"
    assert run.BLAS_THREADS == 4


def test_commands_keep_exact_v12_host_and_online_wandb() -> None:
    root = Path("/tmp/frozen-v13")
    for task in run.TASKS:
        for design in DESIGNS:
            command = run.train_command("python", root, run.DEFAULT_STUDY, 30, task, design)
            expected = {
                "--memory-mode": design, "--seed": "13001", "--epochs": "30",
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


def test_run_names_match_candidate_and_kdio_delegates() -> None:
    root = Path("/tmp/v13")
    ordinary = run.run_directory(root, "fish.swim", "cfhirov13")
    assert ordinary.name == "lewm-dmc:fish.swim-cfhirov13-s13001"
    kdio = run.run_directory(root, "fish.swim", "kdiov11")
    assert kdio.name.endswith("-rank-rawdiff_displacement_detached")


def main() -> None:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} V13 screen-runner tests passed.")


if __name__ == "__main__":
    main()
