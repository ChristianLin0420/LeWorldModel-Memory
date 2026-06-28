"""Adapter: collect pixel trajectories from POPGym Arcade (JAX/gymnax POMDPs) for
offline world-model training in PyTorch.

POPGym Arcade envs are partially-observable Atari-style tasks designed to require memory.
We roll out a random policy (vectorized with jax.vmap + jit) to produce fixed-length
chunks of (observation, action) pairs, downsample to img_size, and store as .npz. Our
MemoryLeWorldModel then trains to predict next latents; discrete actions are one-hot
encoded so action_dim = n_actions.

We only use JAX (CPU here) to *generate* data; all model training stays in PyTorch.
"""

import hashlib
from pathlib import Path
import numpy as np


_PROTOTYPE_FINGERPRINTS = {}


def _validate_action_prototypes(env_id, prototype_seed, n_actions, prototypes):
    prototypes = np.asarray(prototypes)
    if (prototypes.ndim < 2 or prototypes.shape[0] != n_actions or
            not np.issubdtype(prototypes.dtype, np.floating) or
            not np.isfinite(prototypes).all()):
        raise ValueError(
            f'{env_id}: invalid action prototypes shape/dtype/values: '
            f'{prototypes.shape}, {prototypes.dtype}')
    array = np.ascontiguousarray(prototypes)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(repr(array.shape).encode('ascii'))
    digest.update(array.tobytes(order='C'))
    fingerprint = digest.hexdigest()
    base_env = env_id[:-4] if env_id.endswith('.occ') else env_id
    key = (base_env, int(prototype_seed))
    previous = _PROTOTYPE_FINGERPRINTS.setdefault(key, fingerprint)
    if previous != fingerprint:
        raise ValueError(
            f'{env_id}: action prototypes differ from another cache for '
            f'{base_env} at prototype_seed={prototype_seed}')
    return array


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
                   seed: int = 0, data_dir: str = 'outputs/popgym_data',
                   prototype_seed: int = 0):
    """Cache trajectories to .npz so collection happens once per (env, size, seed).

    DMC/OGBench caches use schema v3.  Continuous-action prototypes are independent
    of the rollout seed, and every ``.occ`` cache is derived by masking its clean cache
    rather than by independently resetting the simulator.  The latter is necessary for
    exact clean/occluded pairing (OGBench rendering is not reset-bit-reproducible).
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    uses_prototypes = env_id.startswith(('dmc:', 'ogbench:'))
    schema_version = 3
    version = f"_v{schema_version}_proto{prototype_seed}" if uses_prototypes else ""
    fp = Path(data_dir) / f"{env_id.replace(':', '_')}{version}_n{num_episodes}_L{length}_s{img_size}_seed{seed}.npz"
    if fp.exists():
        with np.load(fp, allow_pickle=False) as d:
            n_actions = int(d['n_actions'])
            if uses_prototypes:
                required = {'action_prototypes', 'prototype_seed', 'schema_version', 'cache_role'}
                missing = required - set(d.files)
                if missing:
                    raise ValueError(f"prototype cache {fp} missing metadata: {sorted(missing)}")
                if int(d['schema_version']) != schema_version or int(d['prototype_seed']) != prototype_seed:
                    raise ValueError(f"prototype cache metadata mismatch: {fp}")
                role = str(d['cache_role'])
                if env_id.endswith('.occ'):
                    expected_clean = env_id[:-4]
                    if role != 'paired_occluded' or 'clean_env_id' not in d.files:
                        raise ValueError(f"occluded cache lacks paired-clean provenance: {fp}")
                    if str(d['clean_env_id']) != expected_clean:
                        raise ValueError(f"occluded cache clean-env mismatch: {fp}")
                elif role != 'clean_or_full':
                    raise ValueError(f"unexpected cache role {role!r}: {fp}")
                _validate_action_prototypes(
                    env_id, prototype_seed, n_actions, d['action_prototypes'])
            obs = np.array(d['obs'], copy=True)
            actions = np.array(d['actions'], copy=True)
        return obs, actions, n_actions

    # A paired occlusion view must come from the exact clean pixels/actions. Recursive
    # collection creates or loads the clean cache first, then masks a copy.
    if uses_prototypes and env_id.endswith('.occ'):
        clean_env_id = env_id[:-4]
        clean_obs, act, n_actions = get_or_collect(
            clean_env_id, num_episodes, length, img_size=img_size, seed=seed,
            data_dir=data_dir, prototype_seed=prototype_seed)
        clean_fp = Path(data_dir) / (
            f"{clean_env_id.replace(':', '_')}{version}_n{num_episodes}_L{length}_s{img_size}_seed{seed}.npz")
        with np.load(clean_fp) as clean_cache:
            protos = np.array(clean_cache['action_prototypes'], copy=True)
        _validate_action_prototypes(env_id, prototype_seed, n_actions, protos)
        obs = np.array(clean_obs, copy=True)
        occ_start = length // 3
        occ_end = min(length, occ_start + max(4, length // 5))
        obs[:, occ_start:occ_end] = 0
        np.savez_compressed(
            fp, obs=obs, actions=act, n_actions=n_actions,
            action_prototypes=protos, prototype_seed=prototype_seed,
            schema_version=schema_version, cache_role='paired_occluded',
            clean_env_id=clean_env_id)
        return obs, act, n_actions

    protos = None
    if env_id.startswith('mmaze:'):                          # Memory-Maze (3D) via MuJoCo
        from lewm.envs.memory_maze_collect import collect_mmaze
        obs, act, n_actions = collect_mmaze(env_id.split(':', 1)[1], num_episodes, length, img_size=img_size, seed=seed)
    elif env_id.startswith('dmc:'):                          # DeepMind Control (real MuJoCo robots)
        from lewm.envs.dmc_collect import collect_dmc
        obs, act, n_actions, protos = collect_dmc(
            env_id.split(':', 1)[1], num_episodes, length, img_size=img_size,
            seed=seed, prototype_seed=prototype_seed)
    elif env_id.startswith('ogbench:'):                      # OGBench manipulation (Cube robot arm)
        from lewm.envs.ogbench_collect import collect_ogbench
        obs, act, n_actions, protos = collect_ogbench(
            env_id.split(':', 1)[1], num_episodes, length, img_size=img_size,
            seed=seed, prototype_seed=prototype_seed)
    else:
        obs, act, n_actions = collect_popgym(env_id, num_episodes, length, img_size=img_size, seed=seed)
    if uses_prototypes:
        _validate_action_prototypes(env_id, prototype_seed, n_actions, protos)
        np.savez_compressed(
            fp, obs=obs, actions=act, n_actions=n_actions,
            action_prototypes=protos, prototype_seed=prototype_seed,
            schema_version=schema_version, cache_role='clean_or_full')
    else:
        np.savez_compressed(fp, obs=obs, actions=act, n_actions=n_actions)
    return obs, act, n_actions
