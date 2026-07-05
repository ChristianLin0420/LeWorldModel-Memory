"""Endogenous base scene for the V19 P1a tasks: dm_control reacher pixels.

The reacher provides the action-controlled (endogenous) part of every task;
all exogenous content is composited on top by ``lewm.tasks_v19.tasks``.  The
native reacher target is made invisible before rendering: its position is
resampled from the environment seed each episode and would otherwise be an
uncontrolled salient pixel factor sitting outside the leakage ledger.  With it
hidden, the rendered base frame is a pure function of (qpos, qvel), which is
exactly what the endogenous state trace records.
"""

from __future__ import annotations

import os

import numpy as np

from lewm.tasks_v19.base import ACTION_DIM, IMG_SIZE, STREAMS

DOMAIN = "reacher"
TASK = "easy"
CAMERA_ID = 0

# Open-loop script stream: a_t = tanh(alpha * sin(omega * t + phi)) per action
# dim, with per-episode parameters drawn from the rollout rng.  Smooth and
# xi-independent by construction (V19_PROPOSAL.md section 4.4 stream rule).
SCRIPT_ALPHA_RANGE = (0.5, 1.5)
SCRIPT_OMEGA_RANGE = (0.05, 0.3)


def collect_base(num_episodes: int, length: int, seed: int, stream: str,
                 action_override: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the reacher under an xi-independent action stream and render.

    Args:
        num_episodes: number of episodes E.
        length: frames per episode L (actions have length L-1).
        seed: seeds both the dm_control task randomness and the action rng;
            byte-identical output for identical arguments is what lets paired
            xi branches share the base scene exactly.
        stream: 'iid' (per-step bounded tanh-Gaussian, the V18 convention) or
            'script' (per-episode smooth open-loop sinusoid).
        action_override: optional float (E, L-1, A) actions executed verbatim
            (clipped to the action spec) instead of sampling from ``stream``.
            Episode initial states are unchanged because dm_control draws its
            reset randomness from ``task_kwargs['random']``, which is
            independent of the action rng — this is the entry point for
            action-swap counterfactual branches that must share the base
            scene and every exogenous draw with the factual branch
            (scripts/counterfactual_v19.py).  ``None`` (the default) leaves
            the historical sampling path byte-identical.

    Returns:
        frames uint8 (E, L, 64, 64, 3), actions float32 (E, L-1, 2),
        endo_state float32 (E, L, S) with S = dim(qpos) + dim(qvel).
    """
    if stream not in STREAMS:
        raise ValueError(f"stream must be one of {STREAMS}, got {stream!r}")
    if action_override is not None:
        action_override = np.asarray(action_override, dtype=np.float32)
        expected = (num_episodes, length - 1, ACTION_DIM)
        if action_override.shape != expected:
            raise ValueError(f"action_override must have shape {expected}, "
                             f"got {action_override.shape}")
        if not np.isfinite(action_override).all():
            raise ValueError("action_override must be finite")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from dm_control import suite

    # V21 §12/F2b: second-scene override. Unset (the default) leaves every
    # frozen bank byte-identical; set LEWM_DMC_DOMAIN/LEWM_DMC_TASK only for
    # the registered off-reacher instrument runs.
    domain = os.environ.get("LEWM_DMC_DOMAIN", DOMAIN)
    dmc_task = os.environ.get("LEWM_DMC_TASK", TASK)
    env = suite.load(domain, dmc_task, task_kwargs={"random": seed % 2**32})
    if domain == "reacher":
        # Hide the native target (model-level change: persists across
        # resets); other domains have no target geom.
        env.physics.named.model.geom_rgba["target", 3] = 0.0

    spec = env.action_spec()
    if spec.shape != (ACTION_DIM,):
        raise RuntimeError(f"{domain}.{dmc_task} action dim {spec.shape} != ({ACTION_DIM},)")
    low = np.asarray(spec.minimum, dtype=np.float32)
    high = np.asarray(spec.maximum, dtype=np.float32)
    center, half_range = (low + high) * 0.5, (high - low) * 0.5

    rng = np.random.default_rng(seed)
    state_dim = env.physics.data.qpos.size + env.physics.data.qvel.size
    frames = np.empty((num_episodes, length, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    actions = np.empty((num_episodes, length - 1, ACTION_DIM), dtype=np.float32)
    endo_state = np.empty((num_episodes, length, state_dim), dtype=np.float32)

    def read_state() -> np.ndarray:
        return np.concatenate([env.physics.data.qpos, env.physics.data.qvel]
                              ).astype(np.float32)

    for episode in range(num_episodes):
        if stream == "script" and action_override is None:
            alpha = rng.uniform(*SCRIPT_ALPHA_RANGE, size=ACTION_DIM)
            omega = rng.uniform(*SCRIPT_OMEGA_RANGE, size=ACTION_DIM)
            phi = rng.uniform(0.0, 2.0 * np.pi, size=ACTION_DIM)
        timestep = env.reset()
        frames[episode, 0] = env.physics.render(IMG_SIZE, IMG_SIZE, camera_id=CAMERA_ID)
        endo_state[episode, 0] = read_state()
        for t in range(length - 1):
            if action_override is not None:
                action = np.clip(action_override[episode, t], low, high)
            elif stream == "iid":
                squashed = np.tanh(rng.standard_normal(ACTION_DIM))
                action = np.clip(center + half_range * squashed.astype(np.float32),
                                 low, high)
            else:
                squashed = np.tanh(alpha * np.sin(omega * t + phi))
                action = np.clip(center + half_range * squashed.astype(np.float32),
                                 low, high)
            timestep = env.step(action)
            if timestep.last():
                # L=64 is far below the reacher time limit; a mid-episode reset
                # would silently desynchronize frames/actions/state, so fail.
                raise RuntimeError(f"unexpected episode termination at step {t}")
            actions[episode, t] = action
            frames[episode, t + 1] = env.physics.render(IMG_SIZE, IMG_SIZE,
                                                        camera_id=CAMERA_ID)
            endo_state[episode, t + 1] = read_state()
    return frames, actions, endo_state
