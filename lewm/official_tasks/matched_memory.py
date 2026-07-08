"""Shared color-by-location cue used by the matched-host memory audit.

Every episode contains one colored square at one of four locations during a
declared cue interval.  Color and location are independently balanced.  The
same rendered episode therefore supports two four-way targets without a
renderer, salience, class-count, or carrier-training difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CUE_COLORS = np.asarray([
    (230, 57, 70),
    (40, 160, 84),
    (45, 108, 223),
    (239, 174, 45),
], dtype=np.uint8)

# Normalized centers, ordered top-left, top-right, bottom-left, bottom-right.
CUE_LOCATIONS = ((0.18, 0.18), (0.82, 0.18),
                 (0.18, 0.82), (0.82, 0.82))
NUM_COLORS = 4
NUM_LOCATIONS = 4
NUM_COMBINATIONS = NUM_COLORS * NUM_LOCATIONS


class MatchedCueError(ValueError):
    """The paired cue or label contract is invalid."""


@dataclass(frozen=True)
class MatchedLabels:
    color: np.ndarray
    location: np.ndarray
    combination: np.ndarray

    def __post_init__(self) -> None:
        arrays = (self.color, self.location, self.combination)
        if any(value.dtype != np.int64 or value.ndim != 1 for value in arrays):
            raise MatchedCueError("matched labels must be one-dimensional int64")
        if len({len(value) for value in arrays}) != 1:
            raise MatchedCueError("matched label arrays differ in length")
        if not np.array_equal(
                self.combination, self.color * NUM_LOCATIONS + self.location):
            raise MatchedCueError("combination does not encode color x location")
        if np.any((self.color < 0) | (self.color >= NUM_COLORS)) \
                or np.any((self.location < 0)
                          | (self.location >= NUM_LOCATIONS)):
            raise MatchedCueError("matched label is outside its vocabulary")


def balanced_joint_labels(count: int, seed: int) -> MatchedLabels:
    """Return an exactly balanced, deterministically shuffled 4x4 design."""

    if isinstance(count, bool) or not isinstance(count, int) \
            or count <= 0 or count % NUM_COMBINATIONS:
        raise MatchedCueError(
            f"count must be a positive multiple of {NUM_COMBINATIONS}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise MatchedCueError("seed must be a non-negative integer")
    combination = np.tile(
        np.arange(NUM_COMBINATIONS, dtype=np.int64),
        count // NUM_COMBINATIONS)
    np.random.default_rng(seed).shuffle(combination)
    color = combination // NUM_LOCATIONS
    location = combination % NUM_LOCATIONS
    return MatchedLabels(color=color, location=location,
                         combination=combination)


def cue_bounds(height: int, width: int, location: int
               ) -> tuple[int, int, int, int]:
    if min(height, width) < 16:
        raise MatchedCueError("cue frames must be at least 16x16")
    if isinstance(location, bool) or not isinstance(location, int) \
            or not 0 <= location < NUM_LOCATIONS:
        raise MatchedCueError("location label is invalid")
    side = max(4, int(round(min(height, width) * 0.20)))
    cx = int(round(CUE_LOCATIONS[location][0] * (width - 1)))
    cy = int(round(CUE_LOCATIONS[location][1] * (height - 1)))
    x0 = min(max(0, cx - side // 2), width - side)
    y0 = min(max(0, cy - side // 2), height - side)
    return x0, y0, x0 + side, y0 + side


def render_joint_cue(base_frames: np.ndarray, color: int, location: int,
                     cue_start: int, cue_length: int) -> np.ndarray:
    """Render one cue while preserving every off-cue byte exactly."""

    frames = np.asarray(base_frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 \
            or frames.shape[-1] != 3:
        raise MatchedCueError("base_frames must be uint8 THWC RGB")
    if isinstance(color, bool) or not isinstance(color, int) \
            or not 0 <= color < NUM_COLORS:
        raise MatchedCueError("color label is invalid")
    if isinstance(cue_start, bool) or not isinstance(cue_start, int) \
            or isinstance(cue_length, bool) or not isinstance(cue_length, int) \
            or cue_length <= 0 \
            or not 0 <= cue_start < cue_start + cue_length <= len(frames):
        raise MatchedCueError("cue interval is invalid")
    x0, y0, x1, y1 = cue_bounds(
        int(frames.shape[1]), int(frames.shape[2]), location)
    result = frames.copy()
    result[cue_start:cue_start + cue_length, y0:y1, x0:x1] = CUE_COLORS[color]
    if not np.array_equal(result[:cue_start], frames[:cue_start]) \
            or not np.array_equal(result[cue_start + cue_length:],
                                  frames[cue_start + cue_length:]):
        raise MatchedCueError("renderer changed an off-cue pixel")
    if np.array_equal(result[cue_start:cue_start + cue_length],
                      frames[cue_start:cue_start + cue_length]):
        raise MatchedCueError("renderer did not change the cue interval")
    result.setflags(write=False)
    return result


def validate_joint_counterfactuals(base_frames: np.ndarray, cue_start: int,
                                   cue_length: int) -> dict[str, int]:
    """Exhaustively certify all 16 cue combinations on one base sequence."""

    signatures: set[bytes] = set()
    reference_outside: bytes | None = None
    cue_end = cue_start + cue_length
    for color in range(NUM_COLORS):
        for location in range(NUM_LOCATIONS):
            value = render_joint_cue(
                base_frames, color, location, cue_start, cue_length)
            outside = value[:cue_start].tobytes() + value[cue_end:].tobytes()
            if reference_outside is None:
                reference_outside = outside
            elif outside != reference_outside:
                raise MatchedCueError("counterfactual identity leaks outside cue")
            signatures.add(value[cue_start:cue_end].tobytes())
    if len(signatures) != NUM_COMBINATIONS:
        raise MatchedCueError("two joint cue combinations render identically")
    return {"counterfactuals": NUM_COMBINATIONS,
            "unique_cue_signatures": len(signatures),
            "offcue_mismatches": 0}


__all__ = [
    "CUE_COLORS", "CUE_LOCATIONS", "MatchedCueError", "MatchedLabels",
    "NUM_COLORS", "NUM_COMBINATIONS", "NUM_LOCATIONS",
    "balanced_joint_labels", "cue_bounds", "render_joint_cue",
    "validate_joint_counterfactuals",
]
