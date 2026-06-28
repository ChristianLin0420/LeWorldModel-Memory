"""Collect pixel trajectories from OGBench (the goal-conditioned manipulation benchmark used
by DINO-WM / LeWorldModel) — here the **Cube** robot-arm manipulation task. Same .npz format
and continuous->discrete action-prototype trick as the dm_control adapter, so train_popgym is
reused unchanged.

'.occ' suffix blanks a mid-episode window (the manipulation continues under known actions, so
predicting the post-occlusion frames needs memory) — identical protocol to §5.15's dm_control.
"""
import os
import warnings
import numpy as np

K_PROTOTYPES = 6


def _parse(spec: str):
    """'cube-single' -> ('cube-single-v0', False); 'cube-single.occ' -> (..., True)."""
    occ = spec.endswith('.occ')
    base = spec[:-4] if occ else spec
    return f'{base}-v0', occ


def collect_ogbench(spec: str, num_episodes: int, length: int, img_size: int = 64,
                    seed: int = 0, prototype_seed: int = 0):
    """spec e.g. 'cube-single' or 'cube-single.occ'. Returns
    (obs uint8 (E,L,H,W,3), actions int (E,L-1) in [0,K), n_actions=K,
    action_prototypes). ``prototype_seed`` is independent of the rollout/env seed."""
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import gymnasium as gym
    import cv2
    import ogbench  # noqa: F401  (registers the envs)

    envid, occ = _parse(spec)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        env = gym.make(envid, render_mode='rgb_array')
    lo = np.asarray(env.action_space.low, dtype=np.float32)
    hi = np.asarray(env.action_space.high, dtype=np.float32)
    lo = np.nan_to_num(lo, neginf=-1.0); hi = np.nan_to_num(hi, posinf=1.0)

    proto_rng = np.random.default_rng(prototype_seed)
    rollout_rng = np.random.default_rng(seed)
    protos = lo + (hi - lo) * proto_rng.random((K_PROTOTYPES, *lo.shape)).astype(np.float32)

    occ_start = length // 3
    occ_end = min(length, occ_start + max(4, length // 5))

    def frame():
        f = env.render()
        if f.shape[0] != img_size:
            f = cv2.resize(f, (img_size, img_size))
        return f.astype(np.uint8)

    obs_all, act_all = [], []
    for ep in range(num_episodes):
        env.reset(seed=seed + ep)
        frames = [frame()]; acts = []
        for _t in range(length - 1):
            k = int(rollout_rng.integers(0, K_PROTOTYPES))
            _, _, term, trunc, _ = env.step(protos[k])
            if term or trunc:
                env.reset(seed=seed + ep)
            frames.append(frame()); acts.append(k)
        o = np.stack(frames).astype(np.uint8)
        if occ:
            o[occ_start:occ_end] = 0
        obs_all.append(o); act_all.append(np.array(acts, dtype=np.int32))
    env.close()
    return np.stack(obs_all), np.stack(act_all), K_PROTOTYPES, protos
