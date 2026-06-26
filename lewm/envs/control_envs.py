"""Interactive memory-control env for E7 (downstream closed-loop planning).

TMazeControlEnv: the agent must navigate to the arm indicated by a cue shown briefly at
the start and then hidden — a genuine memory-dependent *control* task (unlike the offline
prediction envs, here actions move the agent and success = reaching the cued arm). Visuals
match the offline `tmaze` env so a model trained there transfers.
"""
import numpy as np
from lewm.envs.memory_envs import _Renderer, BLUE, YELLOW, GREEN, GRAY, RED, WHITE


class TMazeControlEnv:
    def __init__(self, img_size: int = 64, cue_len: int = 3, max_steps: int = 40,
                 agent_radius: float = 0.05, goal_radius: float = 0.06, step: float = 0.06):
        self.S = img_size; self.R = _Renderer(img_size)
        self.cue_len = cue_len; self.max_steps = max_steps
        self.ar, self.gr, self.step_sz = agent_radius, goal_radius, step
        self.arm_y = {0: 0.15, 1: 0.85}            # 0 = top arm, 1 = bottom arm
        self.goal_x = 0.85

    def reset(self, rng):
        self.cue = int(rng.integers(0, 2))
        self.pos = np.array([0.12, 0.5], dtype=np.float32)
        self.t = 0
        return self._render()

    def _render(self, show_goal: bool = False, pos=None, goal_arm=None):
        pos = self.pos if pos is None else pos
        img = self.R.blank()
        self.R.rect(img, 0.46, 0.0, 0.54, 1.0, GRAY)
        self.R.rect(img, 0.46, 0.44, 0.54, 0.56, WHITE)      # door
        if self.t < self.cue_len:                            # cue (blue=top / yellow=bottom)
            cc = BLUE if self.cue == 0 else YELLOW
            self.R.rect(img, 0.02, self.arm_y[self.cue] - 0.08, 0.18, self.arm_y[self.cue] + 0.08, cc)
        if show_goal:
            self.R.disk(img, self.goal_x, self.arm_y[goal_arm], self.gr, GREEN)
        self.R.disk(img, pos[0], pos[1], self.ar, RED)
        return img

    def goal_template(self, arm: int):
        """A reveal-like frame with the goal at `arm` and the agent at a fixed junction
        position (identical across arms, so z(top)-z(bottom) isolates the goal direction)."""
        save_t, save_pos = self.t, self.pos
        self.t = self.cue_len + 1                              # cue off
        img = self._render(show_goal=True, pos=np.array([0.8, 0.5], dtype=np.float32), goal_arm=arm)
        self.t, self.pos = save_t, save_pos
        return img

    def act(self, action):
        action = np.clip(action, -1, 1) * self.step_sz
        self.pos = np.clip(self.pos + action.astype(np.float32), 0.05, 0.95)
        self.t += 1
        dist = np.linalg.norm(self.pos - np.array([self.goal_x, self.arm_y[self.cue]]))
        wrong = np.linalg.norm(self.pos - np.array([self.goal_x, self.arm_y[1 - self.cue]]))
        success = dist < self.gr * 2.2
        entered_wrong = wrong < self.gr * 2.2
        done = success or entered_wrong or self.t >= self.max_steps
        return self._render(), success, entered_wrong, done
