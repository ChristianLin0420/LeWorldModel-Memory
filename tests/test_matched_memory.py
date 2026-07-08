from __future__ import annotations

import numpy as np
import pytest

from lewm.official_tasks.matched_memory import (
    MatchedCueError,
    balanced_joint_labels,
    cue_bounds,
    render_joint_cue,
    validate_joint_counterfactuals,
)


def test_balanced_joint_labels_are_exact_and_reproducible() -> None:
    first = balanced_joint_labels(480, 17)
    second = balanced_joint_labels(480, 17)
    assert np.array_equal(first.combination, second.combination)
    assert np.array_equal(np.bincount(first.combination), np.full(16, 30))
    assert np.array_equal(np.bincount(first.color), np.full(4, 120))
    assert np.array_equal(np.bincount(first.location), np.full(4, 120))


def test_renderer_is_cue_only_and_exhaustively_unique() -> None:
    rng = np.random.default_rng(8)
    base = rng.integers(0, 256, size=(20, 64, 64, 3), dtype=np.uint8)
    value = render_joint_cue(base, 2, 3, 8, 3)
    assert np.array_equal(value[:8], base[:8])
    assert np.array_equal(value[11:], base[11:])
    assert not value.flags.writeable
    assert validate_joint_counterfactuals(base, 8, 3) == {
        "counterfactuals": 16,
        "unique_cue_signatures": 16,
        "offcue_mismatches": 0,
    }


def test_bounds_scale_with_resolution() -> None:
    small = cue_bounds(64, 64, 0)
    large = cue_bounds(224, 224, 0)
    assert small[2] - small[0] == 13
    assert large[2] - large[0] == 45


def test_invalid_joint_design_is_rejected() -> None:
    with pytest.raises(MatchedCueError):
        balanced_joint_labels(17, 0)
    base = np.zeros((20, 64, 64, 3), dtype=np.uint8)
    with pytest.raises(MatchedCueError):
        render_joint_cue(base, 4, 0, 1, 3)
