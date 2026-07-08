"""High-contrast composite token renderer for adaptive matched-host recall."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TOKEN_COLORS = np.asarray([
    [225, 35, 45],
    [25, 155, 65],
    [35, 90, 225],
    [235, 155, 20],
], dtype=np.uint8)
TOKEN_NAMES = ("vertical-bar", "horizontal-bar", "x", "plus")
LOCATION_CENTERS = ((0.20, 0.20), (0.80, 0.20),
                    (0.20, 0.80), (0.80, 0.80))
SIDE_FRACTION = 0.28


@dataclass(frozen=True)
class TokenLabels:
    token: np.ndarray
    location: np.ndarray
    combination: np.ndarray


def balanced_token_labels(count: int, seed: int) -> TokenLabels:
    if count < 16 or count % 16:
        raise ValueError("token deck size must be a positive multiple of 16")
    combination = np.tile(np.arange(16, dtype=np.int64), count // 16)
    np.random.default_rng(seed).shuffle(combination)
    token = combination // 4
    location = combination % 4
    for value in (combination, token, location):
        value.setflags(write=False)
    return TokenLabels(token=token, location=location,
                       combination=combination)


def _bounds(height: int, width: int, location: int
            ) -> tuple[int, int, int, int]:
    if not 0 <= location < 4:
        raise ValueError("location must be in [0,3]")
    side = max(14, int(round(min(height, width) * SIDE_FRACTION)))
    side = min(side, height, width)
    cx = int(round(LOCATION_CENTERS[location][0] * (width - 1)))
    cy = int(round(LOCATION_CENTERS[location][1] * (height - 1)))
    left = int(np.clip(cx - side // 2, 0, width - side))
    top = int(np.clip(cy - side // 2, 0, height - side))
    return top, top + side, left, left + side


def draw_token(frame: np.ndarray, token: int, location: int) -> np.ndarray:
    """Draw one black-bordered, white-field, colored geometric token."""

    value = np.asarray(frame)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("token renderer requires uint8 HWC RGB")
    if not 0 <= token < 4:
        raise ValueError("token must be in [0,3]")
    output = value.copy()
    top, bottom, left, right = _bounds(*value.shape[:2], location)
    side = bottom - top
    border = max(2, int(round(side * 0.10)))
    output[top:bottom, left:right] = np.asarray([0, 0, 0], dtype=np.uint8)
    output[top + border:bottom - border,
           left + border:right - border] = np.asarray(
               [250, 250, 250], dtype=np.uint8)
    inner_top, inner_bottom = top + border, bottom - border
    inner_left, inner_right = left + border, right - border
    inner_h, inner_w = inner_bottom - inner_top, inner_right - inner_left
    thickness = max(2, int(round(min(inner_h, inner_w) * 0.22)))
    cy = (inner_top + inner_bottom - 1) // 2
    cx = (inner_left + inner_right - 1) // 2
    color = TOKEN_COLORS[token]
    if token == 0:  # vertical bar
        output[inner_top:inner_bottom,
               cx - thickness // 2:cx + (thickness + 1) // 2] = color
    elif token == 1:  # horizontal bar
        output[cy - thickness // 2:cy + (thickness + 1) // 2,
               inner_left:inner_right] = color
    elif token == 2:  # X, rasterized with deterministic thick diagonals
        for row in range(inner_h):
            first = int(round(row * (inner_w - 1) / max(inner_h - 1, 1)))
            second = inner_w - 1 - first
            for center in (first, second):
                start = max(0, center - thickness // 2)
                stop = min(inner_w, center + (thickness + 1) // 2)
                output[inner_top + row, inner_left + start:inner_left + stop] = color
    else:  # plus
        output[inner_top:inner_bottom,
               cx - thickness // 2:cx + (thickness + 1) // 2] = color
        output[cy - thickness // 2:cy + (thickness + 1) // 2,
               inner_left:inner_right] = color
    return output


def render_token_cue(frames: np.ndarray, token: int, location: int,
                     cue_on: int, cue_length: int) -> np.ndarray:
    values = np.asarray(frames)
    if values.dtype != np.uint8 or values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("frames must be uint8 LxHxWx3")
    if cue_length < 1 or cue_on < 0 or cue_on + cue_length > len(values):
        raise ValueError("cue interval leaves the sequence")
    output = values.copy()
    for step in range(cue_on, cue_on + cue_length):
        output[step] = draw_token(output[step], token, location)
    return output


def validate_token_counterfactuals(frames: np.ndarray, cue_on: int,
                                   cue_length: int) -> None:
    values = np.asarray(frames)
    rendered = [
        render_token_cue(values, token, location, cue_on, cue_length)
        for token in range(4) for location in range(4)
    ]
    offcue = np.ones(len(values), dtype=np.bool_)
    offcue[cue_on:cue_on + cue_length] = False
    if any(not np.array_equal(item[offcue], values[offcue])
           for item in rendered):
        raise RuntimeError("token renderer changed an off-cue frame")
    signatures = {
        np.ascontiguousarray(item[cue_on:cue_on + cue_length]).tobytes()
        for item in rendered}
    if len(signatures) != 16:
        raise RuntimeError("token renderer does not create 16 unique cues")


__all__ = [
    "LOCATION_CENTERS", "SIDE_FRACTION", "TOKEN_COLORS", "TOKEN_NAMES",
    "TokenLabels", "balanced_token_labels", "draw_token",
    "render_token_cue", "validate_token_counterfactuals",
]
