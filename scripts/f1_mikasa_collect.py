#!/usr/bin/env python3
"""V21 F1 stage 1 — collect RememberColor9 banks (runs in .venv-mikasa).

Phase structure of the env (mikasa_robo_suite.memory_envs.remember_color):
cue cube visible at center for t in [0, 5); all cubes hidden [5, 10); all
nine cubes visible at shuffled positions from t = 10 with the cued color
unmarked — the decision phase carries no xi information by construction.
xi = true_color_indices (9 classes, chance 1/9).

Actions are xi-independent uniform random draws from a bank-seeded numpy
RNG.  Frames: base_camera rgb 128x128 -> bilinear 64x64 uint8 (program
resolution).  Writes one NPZ per (bank_seed, split) under
outputs/v21_f1/banks/: frames (E,60,64,64,3) u8, actions (E,59,8) f32,
xi (E,) i64, cue_on/cue_off/decision_on (E,) i64.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

L = 60
CUE_ON, CUE_OFF, DECISION_ON = 0, 4, 10   # inclusive cue window [0, 4]
E_TRAIN, E_EVAL = 512, 256                # p1b defaults
BANK_SEEDS = (0, 1, 2)
SEED_SALT = 271_500


def collect(env_id: str, episodes: int, seed: int, num_envs: int,
            out: Path) -> None:
    import gymnasium as gym
    import mikasa_robo_suite  # noqa: F401  (registers envs)

    env = gym.make(env_id, num_envs=num_envs, obs_mode="rgb",
                   render_mode="rgb_array")
    action_low = -1.0
    action_high = 1.0
    action_dim = env.action_space.shape[-1]
    rng = np.random.default_rng(seed)

    frames = np.empty((episodes, L, 64, 64, 3), dtype=np.uint8)
    actions = np.empty((episodes, L - 1, action_dim), dtype=np.float32)
    xi = np.empty(episodes, dtype=np.int64)

    done = 0
    batch = 0
    while done < episodes:
        take = min(num_envs, episodes - done)
        obs, _ = env.reset(seed=(seed * 100_003 + batch) % (2**31))
        xi_batch = env.unwrapped.true_color_indices.cpu().numpy()
        xi[done:done + take] = xi_batch[:take]

        for t in range(L):
            rgb = obs["sensor_data"]["base_camera"]["rgb"]
            small = torch.nn.functional.interpolate(
                rgb.float().permute(0, 3, 1, 2), size=(64, 64),
                mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
            frames[done:done + take, t] = small[:take].cpu().numpy()
            if t == L - 1:
                break
            act = rng.uniform(action_low, action_high,
                              (num_envs, action_dim)).astype(np.float32)
            actions[done:done + take, t] = act[:take]
            obs, _, _, _, _ = env.step(torch.from_numpy(act).to(rgb.device))
        done += take
        batch += 1
        print(f"[f1-collect] {out.name}: {done}/{episodes}", flush=True)

    env.close()
    np.savez_compressed(
        out, frames=frames, actions=actions, xi=xi,
        cue_on=np.full(episodes, CUE_ON, dtype=np.int64),
        cue_off=np.full(episodes, CUE_OFF, dtype=np.int64),
        decision_on=np.full(episodes, DECISION_ON, dtype=np.int64))
    print(f"[f1-collect] wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="RememberColor9-v0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--out", default="outputs/v21_f1/banks")
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    for bank_seed in BANK_SEEDS:
        for split, episodes in (("train", E_TRAIN), ("eval", E_EVAL)):
            out = root / f"rc9_s{bank_seed}_{split}.npz"
            if out.exists():
                print(f"[f1-collect] {out} exists, skip", flush=True)
                continue
            collect(args.env_id, episodes,
                    SEED_SALT + 17 * bank_seed + (0 if split == "train"
                                                  else 1), args.num_envs, out)


if __name__ == "__main__":
    main()
