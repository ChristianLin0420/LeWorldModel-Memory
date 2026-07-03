"""Pure-numpy sprite compositing and exogenous motion for the V19 P1a tasks.

All exogenous elements are drawn as opaque (alpha=1) overlays on the rendered
DMC frames with integer pixel math.  This is a deliberate constraint: because
the overlay is a pure function of (frame, overlay state), two episode branches
that share every nuisance draw and differ only in xi render *byte-identically*
outside the cue window — the identical-rendering certificate can demand a max
absolute pixel difference of exactly zero instead of a tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Four maximally color-separated cue colors: linear pixel probes must separate
# them from pooled RGB statistics alone (non-vacuousness certificate).
CUE_COLORS = ((230, 60, 60), (60, 140, 230), (240, 200, 50), (60, 200, 120))
GRAY = (128, 128, 128)


def draw_rect(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int,
              color: tuple[int, int, int]) -> None:
    """Fill the half-open box [x0,x1) x [y0,y1) in place, clipped to the frame."""
    height, width = frame.shape[:2]
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), width), min(int(y1), height)
    if x0 < x1 and y0 < y1:
        frame[y0:y1, x0:x1] = np.asarray(color, dtype=np.uint8)


def _disc_distance2(frame: np.ndarray, cx: int, cy: int, r: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Squared distance grid over the clipped bounding box around (cx, cy)."""
    height, width = frame.shape[:2]
    x0, x1 = max(cx - r, 0), min(cx + r + 1, width)
    y0, y1 = max(cy - r, 0), min(cy + r + 1, height)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return (xs - cx) ** 2 + (ys - cy) ** 2, frame[y0:y1, x0:x1]


def draw_disc(frame: np.ndarray, cx: int, cy: int, r: int,
              color: tuple[int, int, int]) -> None:
    """Fill the disc of radius ``r`` centered at (cx, cy) in place."""
    dist2, region = _disc_distance2(frame, int(cx), int(cy), int(r))
    region[dist2 <= r * r] = np.asarray(color, dtype=np.uint8)


def draw_ring(frame: np.ndarray, cx: int, cy: int, r: int,
              color: tuple[int, int, int], thickness: int = 1) -> None:
    """Draw the annulus with outer radius ``r`` and given thickness in place."""
    dist2, region = _disc_distance2(frame, int(cx), int(cy), int(r))
    inner = max(r - thickness, 0)
    region[(dist2 <= r * r) & (dist2 > inner * inner)] = np.asarray(color, dtype=np.uint8)


@dataclass(frozen=True)
class OUProcess2D:
    """Reflected position/velocity process driving the exogenous sprites.

        x_{t+1} = x_t + v_t
        v_{t+1} = (1 - theta) * v_t + sigma * eps_t,   eps_t ~ N(0, I)

    with positions folded back into the bounds (velocity sign flipped on each
    reflection).  The process is exogenous by construction: its randomness
    comes from a dedicated rng that never touches actions or xi draws, so
    trajectories are shared verbatim across paired xi branches.
    """

    theta: float
    sigma: float
    x_bounds: tuple[float, float]
    y_bounds: tuple[float, float]

    @property
    def stationary_std(self) -> float:
        """Stationary velocity std of the unreflected recursion."""
        return self.sigma / np.sqrt(1.0 - (1.0 - self.theta) ** 2)

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.array([self.x_bounds[0], self.y_bounds[0]], dtype=np.float64)
        high = np.array([self.x_bounds[1], self.y_bounds[1]], dtype=np.float64)
        return low, high

    def reflect(self, pos: np.ndarray, vel: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """Fold positions (..., 2) into bounds via the exact triangle-wave map.

        Equivalent to iterating single reflections until inside, so it is
        correct even for excursions longer than the box (vectorized, no loop).
        Velocity components flip sign once per odd fold count.
        """
        low, high = self._bounds()
        width = high - low
        offset = pos - low
        folds = np.floor(offset / width).astype(np.int64)
        remainder = offset % (2.0 * width)
        folded = low + np.minimum(remainder, 2.0 * width - remainder)
        return folded, np.where(folds % 2 == 1, -vel, vel)

    def rollout(self, num_episodes: int, length: int, rng: np.random.Generator
                ) -> tuple[np.ndarray, np.ndarray]:
        """Sample (pos, vel) traces, each float32 (E, length, 2).

        Initial positions are uniform in the bounds and initial velocities are
        drawn from the stationary velocity law so t=0 is not distinguishably
        "cold" — the exogenous process has no privileged origin the integrator
        probe could exploit.
        """
        low, high = self._bounds()
        pos = np.empty((num_episodes, length, 2), dtype=np.float32)
        vel = np.empty((num_episodes, length, 2), dtype=np.float32)
        p = low + (high - low) * rng.random((num_episodes, 2))
        v = self.stationary_std * rng.standard_normal((num_episodes, 2))
        for t in range(length):
            pos[:, t] = p
            vel[:, t] = v
            p, v = self.reflect(p + v, (1.0 - self.theta) * v
                                + self.sigma * rng.standard_normal((num_episodes, 2)))
        return pos, vel

    def conditional_mean(self, pos: np.ndarray, vel: np.ndarray,
                         horizon: np.ndarray) -> np.ndarray:
        """Closed-form E[x_{t+h} | x_t, v_t] of the unreflected process.

        Summing the geometric velocity decay gives
        ``x + v * (1 - (1-theta)^h) / theta``; the mean is then folded into the
        bounds.  Folding the unbounded mean is an approximation (the exact
        reflected-OU mean has no closed form) but is the registered predictor
        for the T4 posterior-mean certificate.
        """
        horizon = np.asarray(horizon, dtype=np.float64)[..., None]
        decay = (1.0 - (1.0 - self.theta) ** horizon) / self.theta
        mean = np.asarray(pos, dtype=np.float64) + np.asarray(vel, dtype=np.float64) * decay
        folded, _ = self.reflect(mean, np.zeros_like(mean))
        return folded
