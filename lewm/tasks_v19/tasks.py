"""The six V19 P1a tasks: exogenous overlays on the reacher base scene.

Every task follows the same leakage-proofing recipe (V19_PROPOSAL.md 4.4):

- Randomness is split into three independent streams derived from one seed:
  the *base* stream (scene + actions), the *nuisance* stream (cue timing,
  swap patterns, OU trajectories), and the *xi* stream.  ``paired_branches``
  replays the base and nuisance streams verbatim and shifts only the xi draw,
  so identical rendering outside the cue window is checkable to the byte.
- Cue onset/duration are drawn from the nuisance stream, never from xi.
- Overlays are visual-only (no contact path into the physics), live in the
  top rows and corners, and post-cue rendering is xi-independent.

Amendment 2 (cue salience, V19_PROPOSAL.md section 9): P1b showed the trained
VICReg encoders carry zero cue information while raw pixels decode it at
0.95+, so the exogenous elements are enlarged and the cue windows augmented
with a frame-border tint (xi-colored for T1/T3; fixed-color for T2) so the
exogenous factor carries non-negligible pixel variance.  Every leakage proof
is preserved: borders exist only during the cue window, so post-cue frames
remain byte-identical across xi.  Amendment-1 values are noted next to each
changed parameter; ``describe()`` reports ``amendment: 2``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from lewm.tasks_v19.base import ACTION_DIM, IMG_SIZE, EpisodeBatch, V19Task
from lewm.tasks_v19.dmc_base import DOMAIN, TASK, collect_base
from lewm.tasks_v19.overlays import (CUE_COLORS, GRAY, OUProcess2D,
                                     draw_border, draw_disc, draw_rect,
                                     draw_ring)

TASKS = ("t1", "t2", "t3", "t4", "t1dev", "t2dev")

# Domain-separation salt for seed derivation: keeps a task's random streams
# disjoint from every other task's at the same user seed.
_SEED_SALT = 0x5619
_TASK_IDS = {name: index for index, name in enumerate(TASKS, start=1)}

_STREAM_SPEC = {
    "iid": "a_t = bounds(tanh(g_t)), g_t ~ N(0,1) per dim (V18 convention)",
    "script": "a_t = bounds(tanh(alpha*sin(omega*t+phi))), alpha~U(0.5,1.5), "
              "omega~U(0.05,0.3), phi~U(0,2pi) per episode/dim",
}


class _OverlayTask(V19Task):
    """Common machinery: seed splitting, base collection, paired branches."""

    def _rngs(self, seed: int
              ) -> tuple[int, np.random.Generator, np.random.Generator]:
        root = np.random.SeedSequence((_SEED_SALT, _TASK_IDS[self.name], seed))
        base_ss, nuisance_ss, xi_ss = root.spawn(3)
        return (int(base_ss.generate_state(1)[0]),
                np.random.default_rng(nuisance_ss),
                np.random.default_rng(xi_ss))

    def sample_script(self, num_episodes: int, seed: int, xi_shift: int = 0
                      ) -> dict[str, np.ndarray]:
        """Draw the exogenous script (events, trajectories, xi) — no rendering.

        Exposed separately so independence properties (cue timing vs xi, swap
        pattern vs xi) can be tested over many draws without paying for MuJoCo.
        ``xi_shift`` rotates the categorical xi draw without consuming any
        additional randomness, which is how paired branches differ only in xi.
        """
        _, nuisance_rng, xi_rng = self._rngs(seed)
        return self._sample_script(num_episodes, nuisance_rng, xi_rng, xi_shift)

    def generate(self, stream: str, num_episodes: int, seed: int) -> EpisodeBatch:
        return self._generate(stream, num_episodes, seed, xi_shift=0)

    def paired_branches(self, num_episodes: int, seed: int
                        ) -> tuple[EpisodeBatch, EpisodeBatch]:
        return (self._generate("iid", num_episodes, seed, xi_shift=0),
                self._generate("iid", num_episodes, seed, xi_shift=1))

    def _generate(self, stream: str, num_episodes: int, seed: int,
                  xi_shift: int) -> EpisodeBatch:
        base_seed, nuisance_rng, xi_rng = self._rngs(seed)
        script = self._sample_script(num_episodes, nuisance_rng, xi_rng, xi_shift)
        frames, actions, endo_state = collect_base(
            num_episodes, self.length, base_seed, stream)
        exo_state = self._render(frames, script)
        return EpisodeBatch(
            frames=frames, actions=actions, xi=script["xi"],
            xi_kind=self.xi_kind, n_classes=self.n_classes,
            endo_state=endo_state, exo_state=exo_state,
            events={key: script[key] for key in self.event_keys},
            stream=stream, task=self.name, seed=seed)

    def describe(self) -> dict:
        return {
            **dataclasses.asdict(self),
            "amendment": 2,
            "xi_kind": self.xi_kind,
            "n_classes": self.n_classes,
            "length": self.length,
            "img_size": IMG_SIZE,
            "action_dim": ACTION_DIM,
            "base_env": f"{DOMAIN}.{TASK}",
            "cue_palette": CUE_COLORS[:max(self.n_classes, 1)],
            "streams": _STREAM_SPEC,
        }

    # Subclasses define: event_keys, _sample_script, _render.
    event_keys: tuple[str, ...]

    def _sample_script(self, num_episodes: int, nuisance_rng: np.random.Generator,
                       xi_rng: np.random.Generator, xi_shift: int
                       ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _render(self, frames: np.ndarray, script: dict[str, np.ndarray]
                ) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class TransientCueTask(_OverlayTask):
    """T1/T1dev: candidate markers drawn always; marker xi flashes once.

    The gray marker rings are rendered on *every* frame identically so that
    the only xi-dependent pixels are the colored fill inside marker xi and the
    CUE_COLORS[xi] frame border during [cue_on, cue_off) — post-cue frames are
    xi-independent by construction (the border exists only during the cue).
    """

    name: str
    n_classes: int
    markers: tuple[tuple[int, int], ...]
    onset_range: tuple[int, int]        # inclusive bounds for cue_on
    duration_range: tuple[int, int]     # inclusive bounds for cue length
    cue_shape: str                      # 'disc' | 'square'
    xi_kind: str = "cat"
    marker_radius: int = 6              # amendment 1: 4
    # Cue disc radius / square half-size, scaled with the marker so the fill
    # still exactly tiles the ring interior (cue_half = marker_radius - 1).
    cue_half: int = 5                   # amendment 1: 3
    # Amendment 2: xi-colored frame border drawn during the cue window ONLY.
    cue_border_px: int = 3

    event_keys = ("cue_on", "cue_off")

    def _sample_script(self, num_episodes, nuisance_rng, xi_rng, xi_shift):
        onset = nuisance_rng.integers(self.onset_range[0], self.onset_range[1] + 1,
                                      size=num_episodes)
        duration = nuisance_rng.integers(self.duration_range[0],
                                         self.duration_range[1] + 1,
                                         size=num_episodes)
        xi = (xi_rng.integers(0, self.n_classes, size=num_episodes)
              + xi_shift) % self.n_classes
        return {"xi": xi.astype(np.int64), "cue_on": onset.astype(np.int64),
                "cue_off": (onset + duration).astype(np.int64)}

    def _render(self, frames, script):
        num_episodes, length = frames.shape[:2]
        exo_state = np.zeros((num_episodes, length, 1 + self.n_classes),
                             dtype=np.float32)
        for episode in range(num_episodes):
            xi = int(script["xi"][episode])
            cue_on, cue_off = (int(script["cue_on"][episode]),
                               int(script["cue_off"][episode]))
            for t in range(length):
                frame = frames[episode, t]
                for cx, cy in self.markers:
                    draw_ring(frame, cx, cy, self.marker_radius, GRAY)
                if cue_on <= t < cue_off:
                    draw_border(frame, self.cue_border_px, CUE_COLORS[xi])
                    cx, cy = self.markers[xi]
                    if self.cue_shape == "disc":
                        draw_disc(frame, cx, cy, self.cue_half, CUE_COLORS[xi])
                    else:
                        draw_rect(frame, cx - self.cue_half, cy - self.cue_half,
                                  cx + self.cue_half, cy + self.cue_half,
                                  CUE_COLORS[xi])
                    exo_state[episode, t, 0] = 1.0
                    exo_state[episode, t, 1 + xi] = 1.0
        return exo_state


@dataclass(frozen=True)
class ShellGameTask(_OverlayTask):
    """T2/T2dev: shell game with identical cups and visible smooth swaps.

    xi is the *final* slot of the cued cup.  Because the cups are pixel
    identical, every frame outside the cue phase is xi-independent; xi is
    recoverable only by remembering the cued slot and integrating the visible
    swap motions.  With uniformly random swap pairs the induced permutation is
    exactly uniform on A3 after two swaps, so even the true initial ball slot
    (part of the initial exogenous state fed to the integrator probe) predicts
    the final slot at chance.

    Amendment 2 adds a FIXED-color frame border during the cue phase only —
    identical across xi0, so it flags "the cue is happening" without carrying
    any slot information — and enlarges cups and ball.  Overlap check for the
    enlarged cups (documented, t2 geometry slot_x=(12,32,52), width 12): cups
    at rest span [6,18)/[26,38)/[46,58) with 8 px gaps; the two cups being
    exchanged stay >= 13 px apart at the sampled frames of the long (0,2)
    swap; exchanging cups on the *adjacent* pairs necessarily cross mid-swap
    (linear exchange along a line — already true in amendment 1), and a (0,2)
    mover transiently overlaps the stationary middle cup by <= 6 px.  Both
    overlaps are pure functions of the nuisance swap pattern with a fixed
    entity drawing order, hence xi-independent; no slot adjustment can remove
    them in a 64 px frame (it would need half-spacing >= 3x cup width).
    """

    name: str
    slot_x: tuple[int, int, int]
    swap_times: tuple[int, ...]
    cup_size: tuple[int, int]           # (width, height); amendment 1: t2 (8,10), t2dev (7,9)
    n_classes: int = 3
    xi_kind: str = "cat"
    cue_window: tuple[int, int] = (4, 8)   # [on, off)
    swap_frames: int = 4                    # frames per swap (linear motion)
    lift: int = 3                           # cue-phase cup lift in px
    strip_height: int = 12
    cup_y: int = 3
    ball_y: int = 15
    ball_radius: int = 4                    # amendment 1: 2
    # Amendment 2: fixed-color border during the cue phase ONLY (never
    # xi-colored — the ball/lift carry the slot; the border only adds salience).
    cue_border_px: int = 3
    cue_border_color: tuple[int, int, int] = (200, 200, 200)
    table_color: tuple[int, int, int] = (70, 55, 40)
    cup_color: tuple[int, int, int] = (150, 150, 150)
    ball_color: tuple[int, int, int] = CUE_COLORS[0]

    PAIRS = ((0, 1), (0, 2), (1, 2))
    event_keys = ("cue_on", "cue_off", "shuffle_off", "ball_slot0", "swap_pairs")

    def _sample_script(self, num_episodes, nuisance_rng, xi_rng, xi_shift):
        num_swaps = len(self.swap_times)
        pair_index = nuisance_rng.integers(0, len(self.PAIRS),
                                           size=(num_episodes, num_swaps))
        ball_slot0 = (xi_rng.integers(0, self.n_classes, size=num_episodes)
                      + xi_shift) % self.n_classes
        entity_x = np.empty((num_episodes, self.length, 3), dtype=np.float32)
        xi = np.empty(num_episodes, dtype=np.int64)
        slots = np.asarray(self.slot_x, dtype=np.float32)
        for episode in range(num_episodes):
            entity_x[episode] = self._entity_positions(pair_index[episode])
            # The ball stays under the same cup *entity*; entities start at
            # their own slot index, so the cued entity id equals ball_slot0.
            final_x = entity_x[episode, -1, ball_slot0[episode]]
            xi[episode] = int(np.argmin(np.abs(slots - final_x)))
        return {
            "xi": xi,
            "ball_slot0": ball_slot0.astype(np.int64),
            "swap_pairs": pair_index.astype(np.int64),
            "entity_x": entity_x,
            "cue_on": np.full(num_episodes, self.cue_window[0], dtype=np.int64),
            "cue_off": np.full(num_episodes, self.cue_window[1], dtype=np.int64),
            "shuffle_off": np.full(num_episodes,
                                   self.swap_times[-1] + self.swap_frames,
                                   dtype=np.int64),
        }

    def _entity_positions(self, pair_index: np.ndarray) -> np.ndarray:
        """Per-frame x position of each cup entity, (L, 3) float32.

        Swap m moves the two cups occupying slots (i, j) along a linear path
        over ``swap_frames`` frames starting at swap_times[m]; the motion is
        deliberately visible so a sighted tracker can follow the cued cup.
        """
        positions = np.empty((self.length, 3), dtype=np.float32)
        current = np.asarray(self.slot_x, dtype=np.float32).copy()  # per entity
        slot_entity = np.arange(3)                                   # slot -> entity
        cursor = 0
        for start, index in zip(self.swap_times, pair_index, strict=True):
            i, j = self.PAIRS[int(index)]
            positions[cursor:start] = current
            entity_i, entity_j = slot_entity[i], slot_entity[j]
            for u in range(self.swap_frames):
                progress = u / (self.swap_frames - 1)
                positions[start + u] = current
                positions[start + u, entity_i] = (
                    self.slot_x[i] + progress * (self.slot_x[j] - self.slot_x[i]))
                positions[start + u, entity_j] = (
                    self.slot_x[j] + progress * (self.slot_x[i] - self.slot_x[j]))
            current = current.copy()
            current[entity_i], current[entity_j] = self.slot_x[j], self.slot_x[i]
            slot_entity[i], slot_entity[j] = entity_j, entity_i
            cursor = start + self.swap_frames
        positions[cursor:] = current
        return positions

    def _render(self, frames, script):
        num_episodes, length = frames.shape[:2]
        width, height = self.cup_size
        exo_state = np.zeros((num_episodes, length, 5), dtype=np.float32)
        for episode in range(num_episodes):
            ball_slot0 = int(script["ball_slot0"][episode])
            entity_x = script["entity_x"][episode]
            for t in range(length):
                frame = frames[episode, t]
                in_cue = self.cue_window[0] <= t < self.cue_window[1]
                draw_rect(frame, 0, 0, IMG_SIZE, self.strip_height, self.table_color)
                if in_cue:
                    # Fixed color: identical across xi0 (salience only); drawn
                    # before ball/cups so it never occludes the actual cue.
                    draw_border(frame, self.cue_border_px, self.cue_border_color)
                    draw_disc(frame, self.slot_x[ball_slot0], self.ball_y,
                              self.ball_radius, self.ball_color)
                for entity in range(3):
                    cx = int(round(float(entity_x[t, entity])))
                    y0 = self.cup_y - (self.lift if in_cue and entity == ball_slot0
                                       else 0)
                    draw_rect(frame, cx - width // 2, y0,
                              cx - width // 2 + width, y0 + height, self.cup_color)
                exo_state[episode, t, 0] = entity_x[t, ball_slot0]
                exo_state[episode, t, 1:4] = entity_x[t]
                exo_state[episode, t, 4] = float(in_cue)
        return exo_state


@dataclass(frozen=True)
class DrifterTask(_OverlayTask):
    """T3: autonomously drifting sprite whose color flashes xi once.

    The drifter trajectory comes from the nuisance rng and is shared verbatim
    across paired xi branches; xi only recolors the sprite (and, amendment 2,
    tints the frame border) during the flash window, so post-flash frames are
    byte-identical across xi by construction (the sprite keeps moving, but
    identically in both branches).
    """

    name: str = "t3"
    n_classes: int = 4
    xi_kind: str = "cat"
    onset_range: tuple[int, int] = (6, 14)
    duration_range: tuple[int, int] = (4, 6)
    sprite_size: int = 12               # amendment 1: 6
    # Amendment 2: border in CUE_COLORS[xi] during the flash window ONLY.
    cue_border_px: int = 3
    ou: OUProcess2D = OUProcess2D(theta=0.12, sigma=0.9,
                                  x_bounds=(34.0, 58.0), y_bounds=(4.0, 22.0))

    event_keys = ("cue_on", "cue_off")

    def _sample_script(self, num_episodes, nuisance_rng, xi_rng, xi_shift):
        pos, vel = self.ou.rollout(num_episodes, self.length, nuisance_rng)
        onset = nuisance_rng.integers(self.onset_range[0], self.onset_range[1] + 1,
                                      size=num_episodes)
        duration = nuisance_rng.integers(self.duration_range[0],
                                         self.duration_range[1] + 1,
                                         size=num_episodes)
        xi = (xi_rng.integers(0, self.n_classes, size=num_episodes)
              + xi_shift) % self.n_classes
        return {"xi": xi.astype(np.int64), "cue_on": onset.astype(np.int64),
                "cue_off": (onset + duration).astype(np.int64),
                "pos": pos, "vel": vel}

    def _render(self, frames, script):
        num_episodes, length = frames.shape[:2]
        half = self.sprite_size // 2
        exo_state = np.zeros((num_episodes, length, 5 + self.n_classes),
                             dtype=np.float32)
        for episode in range(num_episodes):
            xi = int(script["xi"][episode])
            cue_on, cue_off = (int(script["cue_on"][episode]),
                               int(script["cue_off"][episode]))
            for t in range(length):
                px, py = script["pos"][episode, t]
                flashing = cue_on <= t < cue_off
                color = CUE_COLORS[xi] if flashing else GRAY
                if flashing:
                    draw_border(frames[episode, t], self.cue_border_px,
                                CUE_COLORS[xi])
                x0 = int(round(float(px))) - half
                y0 = int(round(float(py))) - half
                draw_rect(frames[episode, t], x0, y0, x0 + self.sprite_size,
                          y0 + self.sprite_size, color)
                exo_state[episode, t, 0:2] = script["pos"][episode, t]
                exo_state[episode, t, 2:4] = script["vel"][episode, t]
                if flashing:
                    exo_state[episode, t, 4] = 1.0
                    exo_state[episode, t, 5 + xi] = 1.0
        return exo_state


@dataclass(frozen=True)
class FreezeTrackTask(_OverlayTask):
    """T4: OU-driven target; observations freeze while the truth advances.

    xi is the (continuous) target position at gap end, normalized to [-1, 1].
    Stochastic motion is the whole point: a deterministic target would be
    computable from (frame 0, t), whereas here the best gap-bridging predictor
    is the OU posterior mean given the last pre-freeze state — a belief, not
    an integral of actions.
    """

    name: str = "t4"
    n_classes: int = 0
    xi_kind: str = "cont"
    gap_on_range: tuple[int, int] = (24, 30)     # inclusive bounds for b
    gap_len_range: tuple[int, int] = (16, 20)    # inclusive bounds for e - b
    # Exogenous respawn (P1a amendment 1): at t_r ~ U[respawn_range] the target
    # teleports to a fresh uniform position with a fresh stationary velocity.
    # Without it the OU process does not mix by t = e, so the *initial* target
    # state retains xi information and the integrator probe reads ~0.15 R2
    # (measured) instead of chance.  The respawn — visible, exogenous, and
    # strictly before the gap (max onset 20 < min gap 24) — severs the s0
    # channel by construction while leaving pre-gap predictability intact.
    respawn_range: tuple[int, int] = (12, 20)    # inclusive bounds for t_r
    target_radius: int = 6                       # amendment 1: 3
    target_color: tuple[int, int, int] = CUE_COLORS[0]
    # Amendment 2: 1 px white halo ring drawn whenever the target is drawn.
    halo_color: tuple[int, int, int] = (255, 255, 255)
    ou: OUProcess2D = OUProcess2D(theta=0.15, sigma=0.55,
                                  x_bounds=(6.0, 58.0), y_bounds=(6.0, 58.0))

    event_keys = ("gap_on", "gap_off", "respawn")

    def _normalize(self, pos: np.ndarray) -> np.ndarray:
        low = np.array([self.ou.x_bounds[0], self.ou.y_bounds[0]], dtype=np.float64)
        high = np.array([self.ou.x_bounds[1], self.ou.y_bounds[1]], dtype=np.float64)
        return (2.0 * (np.asarray(pos, dtype=np.float64) - low) / (high - low)
                - 1.0).astype(np.float32)

    def _sample_script(self, num_episodes, nuisance_rng, xi_rng, xi_shift):
        # xi is a deterministic readout of the nuisance OU trajectory, so the
        # xi rng and xi_shift are intentionally unused (see paired_branches).
        del xi_rng, xi_shift
        pos, vel = self.ou.rollout(num_episodes, self.length, nuisance_rng)
        # Respawn: splice an independent OU trajectory in from t_r onward.
        # Drawing the replacement rollout from the same nuisance rng keeps the
        # bank byte-deterministic; per-episode alignment uses a gather because
        # t_r varies across episodes.
        respawn = nuisance_rng.integers(self.respawn_range[0],
                                        self.respawn_range[1] + 1,
                                        size=num_episodes)
        pos2, vel2 = self.ou.rollout(num_episodes, self.length, nuisance_rng)
        index = np.arange(self.length)[None, :]                    # (1, L)
        take = index >= respawn[:, None]                           # (E, L)
        shifted = np.clip(index - respawn[:, None], 0, self.length - 1)
        gather = np.take_along_axis
        for full, fresh in ((pos, pos2), (vel, vel2)):
            for axis in range(2):
                spliced = gather(fresh[:, :, axis], shifted, axis=1)
                full[:, :, axis] = np.where(take, spliced, full[:, :, axis])
        gap_on = nuisance_rng.integers(self.gap_on_range[0],
                                       self.gap_on_range[1] + 1, size=num_episodes)
        gap_off = gap_on + nuisance_rng.integers(self.gap_len_range[0],
                                                 self.gap_len_range[1] + 1,
                                                 size=num_episodes)
        xi = self._normalize(pos[np.arange(num_episodes), gap_off])
        return {"xi": xi, "gap_on": gap_on.astype(np.int64),
                "gap_off": gap_off.astype(np.int64),
                "respawn": respawn.astype(np.int64), "pos": pos, "vel": vel}

    def _render(self, frames, script):
        num_episodes, length = frames.shape[:2]
        exo_state = np.zeros((num_episodes, length, 4), dtype=np.float32)
        for episode in range(num_episodes):
            for t in range(length):
                px, py = script["pos"][episode, t]
                cx, cy = int(round(float(px))), int(round(float(py)))
                draw_disc(frames[episode, t], cx, cy, self.target_radius,
                          self.target_color)
                draw_ring(frames[episode, t], cx, cy, self.target_radius + 1,
                          self.halo_color, thickness=1)
            exo_state[episode, :, 0:2] = script["pos"][episode]
            exo_state[episode, :, 2:4] = script["vel"][episode]
            # Freeze corruption: observations repeat frame b-1 while the
            # ground-truth exo_state keeps advancing through the gap.
            gap_on = int(script["gap_on"][episode])
            gap_off = int(script["gap_off"][episode])
            frames[episode, gap_on:gap_off] = frames[episode, gap_on - 1]
        return exo_state

    def paired_branches(self, num_episodes: int, seed: int):
        raise NotImplementedError(
            "t4 has no paired identical-rendering check: xi is a continuous "
            "readout of the shared nuisance OU trajectory, so there is no "
            "xi branch disjoint from the nuisance draws, and post-gap frames "
            "legitimately re-show the target (xi) by design.")

    def posterior_mean_prediction(self, batch: EpisodeBatch) -> np.ndarray:
        """Closed-form OU conditional mean of p_e given (p_{b-1}, v_{b-1}).

        This is the registered no-learning reference the T4 certificate scores
        against: R2(posterior mean) - R2(integrator) is the task's certified
        memory demand.  Returns normalized coordinates, float32 (E, 2).
        """
        index = np.arange(batch.num_episodes)
        gap_on, gap_off = batch.events["gap_on"], batch.events["gap_off"]
        pos = batch.exo_state[index, gap_on - 1, 0:2]
        vel = batch.exo_state[index, gap_on - 1, 2:4]
        mean = self.ou.conditional_mean(pos, vel, gap_off - (gap_on - 1))
        return self._normalize(mean)


def _build_registry() -> dict[str, V19Task]:
    return {
        "t1": TransientCueTask(
            name="t1", n_classes=4,
            markers=((9, 9), (54, 9), (9, 54), (54, 54)),
            onset_range=(6, 14), duration_range=(4, 6), cue_shape="disc"),
        "t1dev": TransientCueTask(
            name="t1dev", n_classes=3,
            markers=((9, 9), (54, 9), (32, 54)),
            onset_range=(8, 16), duration_range=(5, 5), cue_shape="square"),
        # Amendment-2 cup sizes: t2 (8,10)->(12,14), t2dev (7,9)->(11,13).
        # Slot spacing re-checked for the wider cups (see ShellGameTask
        # docstring): resting spans stay disjoint and in-frame at the original
        # slot_x, and the (0,2)-swap movers still clear each other at every
        # sampled frame, so no slot adjustment is needed.
        "t2": ShellGameTask(
            name="t2", slot_x=(12, 32, 52), swap_times=(12, 20, 28, 36),
            cup_size=(12, 14)),
        "t2dev": ShellGameTask(
            name="t2dev", slot_x=(14, 32, 50), swap_times=(10, 16, 22, 28, 34, 40),
            cup_size=(11, 13)),
        "t3": DrifterTask(),
        "t4": FreezeTrackTask(),
    }


def make_task(name: str) -> V19Task:
    """Instantiate a registered P1a task by name."""
    registry = _build_registry()
    if name not in registry:
        raise KeyError(f"unknown V19 task {name!r}; expected one of {TASKS}")
    return registry[name]
