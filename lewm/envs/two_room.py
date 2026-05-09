"""
Simple 2D navigation environment for testing LeWorldModel.
TwoRoom-like environment: agent must navigate through a door to reach a goal.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple


class TwoRoomEnv(gym.Env):
    """
    Simple 2D navigation environment with two rooms separated by a wall.
    The agent (red dot) must navigate through a door to reach a goal (green dot).

    Observation: RGB image (H, W, 3) in [0, 1]
    Action: Continuous 2D velocity (dx, dy) in [-1, 1]
    """

    metadata = {'render_modes': ['rgb_array', 'human']}

    def __init__(
        self,
        img_size: int = 64,
        max_steps: int = 100,
        door_width: float = 0.15,
        agent_radius: float = 0.03,
        goal_radius: float = 0.05,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.max_steps = max_steps
        self.door_width = door_width
        self.agent_radius = agent_radius
        self.goal_radius = goal_radius
        self.render_mode = render_mode

        # Wall position (vertical wall at x=0.5)
        self.wall_x = 0.5
        self.door_y = 0.5  # Door center
        self.door_half = door_width / 2

        # Action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(img_size, img_size, 3), dtype=np.float32
        )

        # State
        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0

    def _get_obs(self) -> np.ndarray:
        """Render current state as RGB image."""
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.float32)

        # Draw wall
        wall_px = int(self.wall_x * self.img_size)
        door_start = int((self.door_y - self.door_half) * self.img_size)
        door_end = int((self.door_y + self.door_half) * self.img_size)

        # Wall (gray)
        img[:, wall_px-1:wall_px+1, :] = 0.5
        # Remove wall at door
        img[door_start:door_end, wall_px-1:wall_px+1, :] = 1.0

        # Draw agent (red)
        ax, ay = self.agent_pos
        apx = int(ax * self.img_size)
        apy = int(ay * self.img_size)
        r = max(1, int(self.agent_radius * self.img_size))
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if dx*dx + dy*dy <= r*r:
                    px, py = apx + dx, apy + dy
                    if 0 <= px < self.img_size and 0 <= py < self.img_size:
                        img[py, px] = [1.0, 0.0, 0.0]

        # Draw goal (green)
        gx, gy = self.goal_pos
        gpx = int(gx * self.img_size)
        gpy = int(gy * self.img_size)
        r = max(1, int(self.goal_radius * self.img_size))
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if dx*dx + dy*dy <= r*r:
                    px, py = gpx + dx, gpy + dy
                    if 0 <= px < self.img_size and 0 <= py < self.img_size:
                        img[py, px] = [0.0, 1.0, 0.0]

        return img

    def _check_wall_collision(self, pos: np.ndarray) -> np.ndarray:
        """Check and resolve wall collisions."""
        x, y = pos

        # Check vertical wall
        if abs(x - self.wall_x) < 0.02:
            # Check if at door
            if abs(y - self.door_y) > self.door_half:
                # Blocked by wall
                if x < self.wall_x:
                    x = self.wall_x - 0.02
                else:
                    x = self.wall_x + 0.02

        # Clamp to bounds
        x = np.clip(x, 0.0, 1.0)
        y = np.clip(y, 0.0, 1.0)

        return np.array([x, y], dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.steps = 0

        # Random agent position in left room
        self.agent_pos = np.array([
            np.random.uniform(0.05, 0.4),
            np.random.uniform(0.1, 0.9),
        ], dtype=np.float32)

        # Random goal position in right room
        self.goal_pos = np.array([
            np.random.uniform(0.6, 0.95),
            np.random.uniform(0.1, 0.9),
        ], dtype=np.float32)

        obs = self._get_obs()
        info = {'goal_obs': self._get_goal_obs()}
        return obs, info

    def _get_goal_obs(self) -> np.ndarray:
        """Get observation with only the goal visible (for goal conditioning)."""
        img = np.ones((self.img_size, self.img_size, 3), dtype=np.float32)

        # Draw wall
        wall_px = int(self.wall_x * self.img_size)
        door_start = int((self.door_y - self.door_half) * self.img_size)
        door_end = int((self.door_y + self.door_half) * self.img_size)
        img[:, wall_px-1:wall_px+1, :] = 0.5
        img[door_start:door_end, wall_px-1:wall_px+1, :] = 1.0

        # Draw goal only
        gx, gy = self.goal_pos
        gpx = int(gx * self.img_size)
        gpy = int(gy * self.img_size)
        r = max(1, int(self.goal_radius * self.img_size))
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if dx*dx + dy*dy <= r*r:
                    px, py = gpx + dx, gpy + dy
                    if 0 <= px < self.img_size and 0 <= py < self.img_size:
                        img[py, px] = [0.0, 1.0, 0.0]

        return img

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1

        # Move agent
        action = np.clip(action, -1.0, 1.0) * 0.05  # Scale step size
        new_pos = self.agent_pos + action
        self.agent_pos = self._check_wall_collision(new_pos)

        # Check goal
        dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        success = dist < self.goal_radius * 2

        # Reward: negative distance
        reward = -dist

        # Termination
        terminated = success
        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = {'success': success, 'distance': dist}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._get_obs()
        return None


def collect_random_trajectories(
    env: TwoRoomEnv,
    num_episodes: int = 1000,
    max_steps: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect random trajectories from the environment.

    Returns:
        observations: (N, H, W, C) array
        actions: (N, A) array
    """
    all_obs = []
    all_act = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            action = env.action_space.sample()
            all_obs.append(obs)
            all_act.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    return np.array(all_obs, dtype=np.float32), np.array(all_act, dtype=np.float32)
