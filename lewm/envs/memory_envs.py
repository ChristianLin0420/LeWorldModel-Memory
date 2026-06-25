"""
Memory-stressing environments for studying short- vs long-term memory in JEPA world models.

The base TwoRoom env is fully observable (Markovian), so memory provably cannot help
there -- it is our *control*. To make memory matter, every env here contains a
**cue-determined event**: something that appears *later* in the episode is decided by a
cue shown *earlier* and is *not* recoverable from the current frame or the current
action. A memoryless world model cannot predict that event; a model with the right
memory horizon can.

Each episode also contains a **controllable agent dot** doing a random walk, with the
applied velocity recorded as the action. This gives genuine action-conditioned dynamics
(the predictor needs the action for the agent dot) that are *orthogonal* to the memory
channel (the cue event). So the model must use actions for the controllable part and
memory for the cue-determined part -- a clean separation.

Four environments (one per GPU), spanning the short<->long memory axis:

  tmaze      (LONG  term)  : a corner cue picks an arm; a goal appears in that arm only
                            after a long delay -> needs the slow bank.
  occlusion  (SHORT term)  : a target moves in one of two lanes, is hidden by a bar for a
                            few steps, then reappears -> needs the fast bank (short gap).
  recall     (MIXED)       : a short colour sequence is shown, then replayed after a delay
                            -> ordered working+episodic memory.
  distractor (LONG + interference): like tmaze but random distractor flashes occur during
                            the delay; only the first cue matters -> robust long-term memory.

All envs render RGB images in [0,1] (uint8 internally) and use a 2-D continuous action
(agent-dot velocity), matching the base model (action_dim=2).
"""

from typing import Callable, Dict, Tuple

import numpy as np

# ---- colours (RGB, 0..255) --------------------------------------------------------
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (230, 30, 30)        # agent dot
GREEN = (30, 200, 30)      # goal / event
BLUE = (40, 80, 230)       # cue A / sequence colour 0
YELLOW = (235, 200, 20)    # cue B / sequence colour 1
MAGENTA = (220, 40, 220)   # sequence colour 2
CYAN = (30, 200, 220)      # distractor flashes
SEQ_COLORS = [BLUE, YELLOW, MAGENTA]


class _Renderer:
    """Fast vectorized renderer: filled disks / rectangles via boolean masks."""

    def __init__(self, img_size: int):
        self.S = img_size
        ys, xs = np.mgrid[0:img_size, 0:img_size]
        self.xs = xs.astype(np.float32)
        self.ys = ys.astype(np.float32)

    def blank(self) -> np.ndarray:
        img = np.empty((self.S, self.S, 3), dtype=np.uint8)
        img[:] = WHITE
        return img

    def disk(self, img: np.ndarray, cx: float, cy: float, r: float, color) -> None:
        rpx = max(1.0, r * self.S)
        cxp, cyp = cx * self.S, cy * self.S
        mask = (self.xs - cxp) ** 2 + (self.ys - cyp) ** 2 <= rpx ** 2
        img[mask] = color

    def rect(self, img: np.ndarray, x0: float, y0: float, x1: float, y1: float, color) -> None:
        S = self.S
        xa, xb = int(x0 * S), int(x1 * S)
        ya, yb = int(y0 * S), int(y1 * S)
        xa, xb = max(0, min(xa, xb)), min(S, max(xa, xb))
        ya, yb = max(0, min(ya, yb)), min(S, max(ya, yb))
        img[ya:yb, xa:xb] = color


# ---- shared agent-dot random walk -------------------------------------------------
def _random_walk(rng: np.random.Generator, length: int, step: float = 0.06):
    """Return (positions (length,2) in [0.1,0.9], actions (length-1,2) in [-1,1])."""
    pos = np.empty((length, 2), dtype=np.float32)
    act = np.empty((length - 1, 2), dtype=np.float32)
    pos[0] = rng.uniform(0.2, 0.8, size=2)
    for t in range(length - 1):
        v = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        act[t] = v
        pos[t + 1] = np.clip(pos[t] + v * step, 0.08, 0.92)
    return pos, act


EpisodeFn = Callable[[np.random.Generator], Tuple[np.ndarray, np.ndarray, Dict]]


# ---- environment episode generators ----------------------------------------------
def make_tmaze(img_size: int = 64, length: int = 32, cue_len: int = 3,
               reveal: int = 24, agent_radius: float = 0.05,
               goal_radius: float = 0.06) -> EpisodeFn:
    """Long-term recall: corner cue (top/bottom) -> goal appears in that arm at `reveal`."""
    R = _Renderer(img_size)

    def gen(rng: np.random.Generator):
        cue = int(rng.integers(0, 2))                       # 0 = top arm, 1 = bottom arm
        cue_color = BLUE if cue == 0 else YELLOW
        cue_cy = 0.12 if cue == 0 else 0.88
        pos, act = _random_walk(rng, length)
        obs = np.empty((length, img_size, img_size, 3), dtype=np.uint8)
        for t in range(length):
            img = R.blank()
            R.rect(img, 0.46, 0.0, 0.54, 1.0, GRAY)         # central corridor wall hint
            if t < cue_len:                                  # show cue early, then hide
                R.rect(img, 0.02, cue_cy - 0.08, 0.18, cue_cy + 0.08, cue_color)
            if t >= reveal:                                  # cue-determined goal appears
                R.disk(img, 0.85, cue_cy, goal_radius, GREEN)
            R.disk(img, pos[t, 0], pos[t, 1], agent_radius, RED)
            obs[t] = img
        info = {'cue': cue, 'cue_end': cue_len, 'reveal': reveal, 'n_cue_classes': 2}
        return obs, act, info

    return gen


def make_occlusion(img_size: int = 64, length: int = 32, occ_start: int = 12,
                   occ_end: int = 17, agent_radius: float = 0.05,
                   target_radius: float = 0.05) -> EpisodeFn:
    """Short-term / object permanence: target crosses in one of two lanes, hidden by a bar
    for a short window, then reappears at the extrapolated position."""
    R = _Renderer(img_size)

    def gen(rng: np.random.Generator):
        lane = int(rng.integers(0, 2))                      # 0 = top lane, 1 = bottom lane
        ty = 0.3 if lane == 0 else 0.7
        vx = float(rng.uniform(0.025, 0.04))                # rightward speed
        x0 = float(rng.uniform(0.05, 0.15))
        pos, act = _random_walk(rng, length)
        obs = np.empty((length, img_size, img_size, 3), dtype=np.uint8)
        for t in range(length):
            img = R.blank()
            R.rect(img, 0.44, 0.0, 0.56, 1.0, GRAY)         # occluder bar
            tx = np.clip(x0 + vx * t, 0.0, 1.0)
            occluded = occ_start <= t < occ_end
            if not occluded:
                R.disk(img, float(tx), ty, target_radius, GREEN)
            R.disk(img, pos[t, 0], pos[t, 1], agent_radius, RED)
            obs[t] = img
        info = {'cue': lane, 'cue_end': occ_start, 'reveal': occ_end, 'n_cue_classes': 2}
        return obs, act, info

    return gen


def make_recall(img_size: int = 64, length: int = 32, seq_len: int = 3,
                show_start: int = 2, replay_start: int = 20,
                agent_radius: float = 0.05) -> EpisodeFn:
    """Sequential recall: a colour sequence is shown, hidden, then replayed after a delay."""
    R = _Renderer(img_size)

    def gen(rng: np.random.Generator):
        seq = rng.integers(0, len(SEQ_COLORS), size=seq_len)
        pos, act = _random_walk(rng, length)
        obs = np.empty((length, img_size, img_size, 3), dtype=np.uint8)
        for t in range(length):
            img = R.blank()
            # show phase
            if show_start <= t < show_start + seq_len:
                R.rect(img, 0.35, 0.35, 0.65, 0.65, SEQ_COLORS[seq[t - show_start]])
            # replay phase (cue-determined) -- requires remembering the sequence
            if replay_start <= t < replay_start + seq_len:
                R.rect(img, 0.35, 0.35, 0.65, 0.65, SEQ_COLORS[seq[t - replay_start]])
            R.disk(img, pos[t, 0], pos[t, 1], agent_radius, RED)
            obs[t] = img
        # probe target: the FIRST colour of the sequence
        info = {'cue': int(seq[0]), 'cue_end': show_start + seq_len,
                'reveal': replay_start, 'n_cue_classes': len(SEQ_COLORS),
                'sequence': seq.tolist()}
        return obs, act, info

    return gen


def make_distractor(img_size: int = 64, length: int = 32, cue_len: int = 3,
                    reveal: int = 26, n_distract: int = 5, agent_radius: float = 0.05,
                    goal_radius: float = 0.06) -> EpisodeFn:
    """Long-term recall under interference: like tmaze, but random distractor flashes occur
    during the delay; only the first cue decides the goal arm."""
    R = _Renderer(img_size)

    def gen(rng: np.random.Generator):
        cue = int(rng.integers(0, 2))
        cue_color = BLUE if cue == 0 else YELLOW
        cue_cy = 0.12 if cue == 0 else 0.88
        # distractor flash times in the delay window
        flash_times = set(rng.choice(np.arange(cue_len + 1, reveal - 1),
                                     size=min(n_distract, reveal - cue_len - 2),
                                     replace=False).tolist())
        pos, act = _random_walk(rng, length)
        obs = np.empty((length, img_size, img_size, 3), dtype=np.uint8)
        for t in range(length):
            img = R.blank()
            R.rect(img, 0.46, 0.0, 0.54, 1.0, GRAY)
            if t < cue_len:
                R.rect(img, 0.02, cue_cy - 0.08, 0.18, cue_cy + 0.08, cue_color)
            if t in flash_times:                            # interference: random arm flash
                d_cy = 0.12 if rng.random() < 0.5 else 0.88
                R.rect(img, 0.02, d_cy - 0.06, 0.12, d_cy + 0.06, CYAN)
            if t >= reveal:
                R.disk(img, 0.85, cue_cy, goal_radius, GREEN)
            R.disk(img, pos[t, 0], pos[t, 1], agent_radius, RED)
            obs[t] = img
        info = {'cue': cue, 'cue_end': cue_len, 'reveal': reveal, 'n_cue_classes': 2}
        return obs, act, info

    return gen


def make_tworoom(img_size: int = 64, length: int = 32, agent_radius: float = 0.05,
                 goal_radius: float = 0.06) -> EpisodeFn:
    """Markovian control: agent + goal both always visible (no memory needed)."""
    R = _Renderer(img_size)

    def gen(rng: np.random.Generator):
        goal = rng.uniform(0.6, 0.92, size=2).astype(np.float32)
        pos, act = _random_walk(rng, length)
        obs = np.empty((length, img_size, img_size, 3), dtype=np.uint8)
        for t in range(length):
            img = R.blank()
            R.rect(img, 0.46, 0.0, 0.54, 1.0, GRAY)
            ds = int((0.5 - 0.075) * img_size), int((0.5 + 0.075) * img_size)
            img[ds[0]:ds[1], int(0.46 * img_size):int(0.54 * img_size)] = WHITE  # door
            R.disk(img, float(goal[0]), float(goal[1]), goal_radius, GREEN)
            R.disk(img, pos[t, 0], pos[t, 1], agent_radius, RED)
            obs[t] = img
        info = {'cue': 0, 'cue_end': 0, 'reveal': 0, 'n_cue_classes': 1}
        return obs, act, info

    return gen


ENV_REGISTRY: Dict[str, Callable[..., EpisodeFn]] = {
    'tmaze': make_tmaze,
    'occlusion': make_occlusion,
    'recall': make_recall,
    'distractor': make_distractor,
    'tworoom': make_tworoom,
}

# Default short/long character of each env (for documentation / tags).
ENV_MEMORY_KIND = {
    'tmaze': 'long',
    'occlusion': 'short',
    'recall': 'mixed',
    'distractor': 'long-interference',
    'tworoom': 'markovian-control',
}


def make_episode_fn(env_name: str, img_size: int = 64, length: int = 32, **kwargs) -> EpisodeFn:
    """Factory: return an episode generator gen(rng) -> (obs, actions, info)."""
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"unknown env '{env_name}', choices: {list(ENV_REGISTRY)}")
    return ENV_REGISTRY[env_name](img_size=img_size, length=length, **kwargs)
