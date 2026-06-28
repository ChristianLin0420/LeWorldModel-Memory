"""Collect pixel trajectories from the DeepMind Control Suite (dm_control) — real MuJoCo
robots (reacher arm, ball-in-cup, finger spinner, cheetah, walker). This gives a *genuine
robotic simulator* (not the lightweight pixel proxies of §5.9), reusing the same .npz format
as the POPGym/Memory-Maze adapters so train_popgym is reused unchanged.

Continuous action spaces are handled with the same trick as Memory-Maze's discrete actions:
we fix K random "action prototypes" (continuous vectors clipped to the action bounds) and the
stored action is the *index* of the chosen prototype, so action_dim = n_actions = K and the
one-hot dataset path is unchanged.

A partially-observable memory variant ('.occ' suffix) blanks a contiguous window of frames in
the middle of each episode: the robot keeps moving under known actions, so predicting the
post-occlusion frames requires *remembering* the pre-occlusion state across the blackout — a
memory test on real robot dynamics (the real-sim analogue of our Occlusion env).
"""
import os
import numpy as np

K_PROTOTYPES = 6  # number of discrete continuous-action prototypes


def _parse(spec: str):
    """'reacher.hard' -> ('reacher','hard',False); 'ball_in_cup.catch.occ' -> (...,True)."""
    parts = spec.split('.')
    occ = parts[-1] == 'occ'
    if occ:
        parts = parts[:-1]
    domain, task = parts[0], parts[1]
    return domain, task, occ


def collect_dmc(spec: str, num_episodes: int, length: int, img_size: int = 64,
                seed: int = 0, prototype_seed: int = 0):
    """spec e.g. 'reacher.hard' or 'reacher.hard.occ'. Returns
    (obs uint8 (E,L,H,W,3), actions int (E,L-1) in [0,K), n_actions=K,
    action_prototypes). ``prototype_seed`` is deliberately independent of the
    rollout/environment seed so action index k has identical semantics in train and val."""
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from dm_control import suite

    domain, task, occ = _parse(spec)
    env = suite.load(domain, task, task_kwargs={'random': seed})
    aspec = env.action_spec()
    lo = np.broadcast_to(aspec.minimum, aspec.shape).astype(np.float32)
    hi = np.broadcast_to(aspec.maximum, aspec.shape).astype(np.float32)
    lo = np.nan_to_num(lo, neginf=-1.0); hi = np.nan_to_num(hi, posinf=1.0)

    proto_rng = np.random.default_rng(prototype_seed)
    rollout_rng = np.random.default_rng(seed)
    # K fixed continuous action prototypes shared across train/val rollout seeds.
    protos = lo + (hi - lo) * proto_rng.random((K_PROTOTYPES, *aspec.shape)).astype(np.float32)

    # occlusion window (middle third-ish of the episode)
    occ_start = length // 3
    occ_len = max(4, length // 5)
    occ_end = min(length, occ_start + occ_len)

    obs_all, act_all = [], []
    for _ in range(num_episodes):
        ts = env.reset()
        frames = [env.physics.render(img_size, img_size, camera_id=0)]
        acts = []
        for _t in range(length - 1):
            k = int(rollout_rng.integers(0, K_PROTOTYPES))
            ts = env.step(protos[k])
            if ts.last():
                ts = env.reset()
            frames.append(env.physics.render(img_size, img_size, camera_id=0))
            acts.append(k)
        o = np.stack(frames).astype(np.uint8)              # (L,H,W,3)
        if occ:
            o[occ_start:occ_end] = 0                        # blank the window (memory must bridge)
        obs_all.append(o)
        act_all.append(np.array(acts, dtype=np.int32))
    return np.stack(obs_all), np.stack(act_all), K_PROTOTYPES, protos
