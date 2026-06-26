"""Adapter: collect pixel trajectories from POPGym Arcade (JAX/gymnax POMDPs) for
offline world-model training in PyTorch.

POPGym Arcade envs are partially-observable Atari-style tasks designed to require memory.
We roll out a random policy (vectorized with jax.vmap + jit) to produce fixed-length
chunks of (observation, action) pairs, downsample to img_size, and store as .npz. Our
MemoryLeWorldModel then trains to predict next latents; discrete actions are one-hot
encoded so action_dim = n_actions.

We only use JAX (CPU here) to *generate* data; all model training stays in PyTorch.
"""

from pathlib import Path
import numpy as np


def collect_popgym(env_id: str, num_episodes: int, length: int, img_size: int = 64,
                   partial_obs: bool = True, num_envs: int = 64, seed: int = 0):
    """Roll out a random policy; return (obs uint8 (E,L,h,w,3), actions int (E,L-1), n_actions)."""
    import jax, jax.numpy as jnp
    import popgym_arcade as pa

    env, params = pa.make(env_id, partial_obs=partial_obs)
    try:
        n_actions = int(env.num_actions)
    except NotImplementedError:
        n_actions = int(env.action_space(params).n)
    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))

    key = jax.random.PRNGKey(seed)
    nbatch = (num_episodes + num_envs - 1) // num_envs
    obs_chunks, act_chunks = [], []
    for _ in range(nbatch):
        key, kr = jax.random.split(key)
        obs, state = reset(jax.random.split(kr, num_envs), params)   # (E,H,W,3)
        ep_obs, ep_act = [obs], []
        for _t in range(length - 1):
            key, ka, ks = jax.random.split(key, 3)
            acts = jax.random.randint(ka, (num_envs,), 0, n_actions)
            obs, state, r, done, info = step(jax.random.split(ks, num_envs), state, acts, params)
            ep_obs.append(obs); ep_act.append(acts)
        o = jnp.stack(ep_obs, axis=1)                                # (E,L,H,W,3)
        a = jnp.stack(ep_act, axis=1)                               # (E,L-1)
        # downsample H,W to img_size by uniform striding (128 -> 64 etc.)
        H = o.shape[2]
        if H != img_size:
            assert H % img_size == 0, f"obs size {H} not a multiple of img_size {img_size}"
            s = H // img_size
            o = o[:, :, ::s, ::s, :]
        obs_chunks.append(np.asarray(o, dtype=np.uint8))
        act_chunks.append(np.asarray(a, dtype=np.int32))
    obs = np.concatenate(obs_chunks)[:num_episodes]
    act = np.concatenate(act_chunks)[:num_episodes]
    return obs, act, n_actions


def get_or_collect(env_id: str, num_episodes: int, length: int, img_size: int = 64,
                   seed: int = 0, data_dir: str = 'outputs/popgym_data'):
    """Cache trajectories to .npz so collection happens once per (env, size, seed)."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(data_dir) / f"{env_id.replace(':', '_')}_n{num_episodes}_L{length}_s{img_size}_seed{seed}.npz"
    if fp.exists():
        d = np.load(fp)
        return d['obs'], d['actions'], int(d['n_actions'])
    if env_id.startswith('mmaze:'):                          # Memory-Maze (3D) via MuJoCo
        from lewm.envs.memory_maze_collect import collect_mmaze
        obs, act, n_actions = collect_mmaze(env_id.split(':', 1)[1], num_episodes, length, img_size=img_size, seed=seed)
    elif env_id.startswith('dmc:'):                          # DeepMind Control (real MuJoCo robots)
        from lewm.envs.dmc_collect import collect_dmc
        obs, act, n_actions = collect_dmc(env_id.split(':', 1)[1], num_episodes, length, img_size=img_size, seed=seed)
    else:
        obs, act, n_actions = collect_popgym(env_id, num_episodes, length, img_size=img_size, seed=seed)
    np.savez_compressed(fp, obs=obs, actions=act, n_actions=n_actions)
    return obs, act, n_actions
