from __future__ import annotations

import numpy as np

from lewm.official_tasks.matched_token import (
    balanced_token_labels, draw_token, render_token_cue,
    validate_token_counterfactuals,
)


def test_token_labels_are_exact_balanced_and_deterministic() -> None:
    first = balanced_token_labels(480, 31)
    second = balanced_token_labels(480, 31)
    assert np.array_equal(first.combination, second.combination)
    assert np.array_equal(
        np.bincount(first.combination, minlength=16), np.full(16, 30))
    assert np.array_equal(
        first.combination, first.token * 4 + first.location)


def test_all_composite_tokens_are_unique_at_both_host_resolutions() -> None:
    for size in (64, 224):
        frame = np.full((size, size, 3), 127, dtype=np.uint8)
        signatures = {
            draw_token(frame, token, location).tobytes()
            for token in range(4) for location in range(4)}
        assert len(signatures) == 16


def test_token_cue_changes_only_registered_frames() -> None:
    frames = np.random.default_rng(2).integers(
        0, 256, size=(20, 64, 64, 3), dtype=np.uint8)
    rendered = render_token_cue(frames, 2, 3, 8, 3)
    assert np.array_equal(rendered[:8], frames[:8])
    assert np.array_equal(rendered[11:], frames[11:])
    assert not np.array_equal(rendered[8:11], frames[8:11])
    validate_token_counterfactuals(frames, 8, 3)

