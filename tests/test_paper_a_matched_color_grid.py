from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import aggregate_paper_a_matched_color as aggregate
from scripts import launch_paper_a_matched_color as launcher
from scripts.train_paper_a_matched_color import _nuisance_breakdown


def test_wave1b_launcher_has_exact_75_gpu0_cells() -> None:
    path = Path("configs/paper_a_matched_color_v1.yaml")
    cells = launcher._carrier_commands(path, path)
    assert len(cells) == 75
    assert len({(host, arm, seed)
                for host, arm, seed, _, _ in cells}) == 75
    assert all(command[command.index("--device") + 1] == "cuda:0"
               for _, _, _, command, _ in cells)


def test_nuisance_breakdown_reports_worst_location() -> None:
    location = np.repeat(np.arange(4), 4)
    correct = np.asarray(
        [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    value = _nuisance_breakdown(correct, location)
    assert value["per_location_accuracy"] == [1.0, 0.5, 0.25, 0.0]
    assert value["worst_location_accuracy"] == 0.0


def test_wave1b_bootstrap_is_deterministic_and_preserves_nuisance() -> None:
    correct = np.zeros((3, 5, 5, 3, 480), dtype=np.float32)
    joint = np.tile(np.repeat(np.arange(16), 30), (3, 1))
    location = joint % 4
    for host in range(3):
        for arm in range(5):
            for nuisance in range(4):
                mask = location[host] == nuisance
                count = min(mask.sum(), 20 + 4 * arm + nuisance)
                rows = np.flatnonzero(mask)[:count]
                correct[host, arm, :, :, rows] = 1.0
    first = aggregate._bootstrap(
        correct, joint, location, draws=32, seed=7)
    second = aggregate._bootstrap(
        correct, joint, location, draws=32, seed=7)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
    point, samples, location_point, location_samples = first
    assert point.shape == (3, 5, 3)
    assert samples.shape == (32, 3, 5, 3)
    assert location_point.shape == (3, 5, 3, 4)
    assert location_samples.shape == (32, 3, 5, 3, 4)

