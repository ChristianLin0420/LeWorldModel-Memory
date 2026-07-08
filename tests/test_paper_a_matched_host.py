from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import aggregate_paper_a_matched_host as aggregate
from scripts import launch_paper_a_matched_host as launcher
from scripts import train_paper_a_matched_host as trainer
from scripts import launch_paper_a_tworoom_use as use_launcher
from scripts.paper_a_matched_host_spec import (
    AGES,
    DEFAULT_SPEC,
    validate_spec,
)


def test_protocol_is_valid_before_lock() -> None:
    value = yaml.safe_load(DEFAULT_SPEC.read_text())
    validate_spec(value, verify_inputs=False)
    assert value["sequence"]["cue_intervals"] == {
        "age-4": [12, 15], "age-8": [8, 11], "age-15": [1, 4]}
    for start, stop in value["sequence"]["cue_intervals"].values():
        assert stop <= 16


def test_aligned_latent_changes_only_registered_cue() -> None:
    base = {
        "z_base": np.zeros((2, 20, 192), dtype=np.float32),
        "episode_index": np.asarray([4, 7], dtype=np.int64),
        "local_start": np.asarray([2, 3], dtype=np.int64),
    }
    cue = {
        "z_cue": np.ones((2, 3, 192), dtype=np.float32),
        "episode_index": base["episode_index"],
        "local_start": base["local_start"],
        "cue_on": np.full(2, 8, dtype=np.int64),
        "cue_off": np.full(2, 11, dtype=np.int64),
    }
    value = trainer._aligned_latent(base, cue)
    assert np.all(value[:, 8:11] == 1)
    assert np.all(value[:, :8] == 0)
    assert np.all(value[:, 11:] == 0)


def test_launcher_expands_exact_75_gpu0_cells() -> None:
    commands = launcher._carrier_commands(DEFAULT_SPEC, DEFAULT_SPEC)
    assert len(commands) == 75
    assert len({(host, arm, seed) for host, arm, seed, _, _ in commands}) == 75
    assert all(command[command.index("--device") + 1] == "cuda:0"
               for _, _, _, command, _ in commands)


def test_use_launcher_expands_exact_25_gpu0_cells() -> None:
    spec = DEFAULT_SPEC
    sha = DEFAULT_SPEC
    cells = [
        (arm, seed, use_launcher._command(
            "scripts/evaluate_paper_a_tworoom_use.py", spec, sha,
            "--arm", arm, "--seed", str(seed), "--device", "cuda:0"))
        for arm in use_launcher.ARMS for seed in use_launcher.SEEDS
    ]
    assert len(cells) == 25
    assert len({(arm, seed) for arm, seed, _ in cells}) == 25
    assert all(command[command.index("--device") + 1] == "cuda:0"
               for _, _, command in cells)


def test_hierarchical_bootstrap_is_deterministic_and_host_independent() -> None:
    correct = np.zeros((3, 5, 5, 2, 3, 480), dtype=np.float32)
    # Make every host/arm/target/age deterministic but different.
    for host in range(3):
        for arm in range(5):
            correct[host, arm, :, :, :, :120 + 10 * arm] = 1
    joint = np.tile(np.arange(16, dtype=np.int64).repeat(30), (3, 1))
    point_a, sample_a = aggregate._bootstrap(
        correct, joint, draws=40, seed=19)
    point_b, sample_b = aggregate._bootstrap(
        correct, joint, draws=40, seed=19)
    assert np.array_equal(point_a, point_b)
    assert np.array_equal(sample_a, sample_b)
    assert point_a.shape == (3, 5, 2, 3)
    assert sample_a.shape == (40, 3, 5, 2, 3)


def test_tmpfs_dataset_is_pinned_without_following_arbitrary_absolute_path() -> None:
    value = yaml.safe_load(DEFAULT_SPEC.read_text())
    record = value["inputs"]["tworoom"]["dataset"]
    assert record["external_tmpfs"] is True
    assert record["path"] == "/dev/shm/paper_a_matched_host_v1/tworoom.h5"
