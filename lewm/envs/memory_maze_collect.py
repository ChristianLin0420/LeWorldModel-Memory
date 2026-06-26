"""Collect trajectories from Memory-Maze (3D first-person POMDP; MuJoCo/EGL headless).

Memory-Maze requires the agent to remember target colours/locations across a 3D maze — a
recognized 3D memory benchmark. We roll out a random policy and store 64x64 RGB frames +
discrete actions, in the same .npz format as the POPGym adapter so the rest of the pipeline
(PopgymDataset, train_popgym) is reused unchanged.
"""
import os
import numpy as np


def collect_mmaze(task: str, num_episodes: int, length: int, img_size: int = 64, seed: int = 0):
    """task e.g. '9x9' -> memory_maze.tasks.memory_maze_9x9(). Returns
    (obs uint8 (E,L,H,W,3), actions int (E,L-1), n_actions)."""
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from memory_maze import tasks
    env = getattr(tasks, f'memory_maze_{task}')()
    n_actions = int(env.action_spec().num_values)
    rng = np.random.default_rng(seed)
    obs_all, act_all = [], []
    for _ in range(num_episodes):
        ts = env.reset(); frames = [ts.observation['image']]; acts = []
        for _t in range(length - 1):
            a = int(rng.integers(0, n_actions))
            ts = env.step(a)
            if ts.last():
                ts = env.reset()
            frames.append(ts.observation['image']); acts.append(a)
        o = np.stack(frames)
        if o.shape[1] != img_size:
            import cv2
            o = np.stack([cv2.resize(fr, (img_size, img_size)) for fr in o])
        obs_all.append(o.astype(np.uint8)); act_all.append(np.array(acts, dtype=np.int32))
    return np.stack(obs_all), np.stack(act_all), n_actions
