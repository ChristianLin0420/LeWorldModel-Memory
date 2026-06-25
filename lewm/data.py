"""Datasets and eval-batch helpers for the memory-stressing environments."""

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from lewm.envs.memory_envs import make_episode_fn


def _episode_to_tensors(obs: np.ndarray, act: np.ndarray):
    """(L,H,W,3) uint8, (L-1,2) -> obs (L,3,H,W) float in [0,1], act (L-1,2) float."""
    obs_t = torch.from_numpy(obs.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
    act_t = torch.from_numpy(np.ascontiguousarray(act, dtype=np.float32))
    return obs_t, act_t


class MemoryEpisodeDataset(Dataset):
    """On-the-fly dataset: each index deterministically generates one episode (chunk).

    No frames are stored; episode `idx` is reproducible from (seed, idx), giving a fixed
    finite dataset with zero memory footprint and trivial multi-worker sharding.
    """

    def __init__(self, env_name: str, num_episodes: int, img_size: int = 64,
                 length: int = 32, seed: int = 0, **env_kwargs):
        self.gen = make_episode_fn(env_name, img_size=img_size, length=length, **env_kwargs)
        self.num_episodes = num_episodes
        self.seed = seed
        self.env_name = env_name
        self.length = length

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        obs, act, _info = self.gen(rng)
        return _episode_to_tensors(obs, act)


def generate_eval_batch(env_name: str, num_episodes: int, img_size: int = 64,
                        length: int = 32, seed: int = 10_000, **env_kwargs
                        ) -> Dict[str, object]:
    """Generate a batch with probe metadata (for probing / visualization).

    Returns dict:
        obs: (B, L, 3, H, W) float tensor
        actions: (B, L-1, 2) float tensor
        cue: (B,) int64 tensor (the variable to remember)
        cue_end: (B,) int  | reveal: (B,) int
        n_cue_classes: int
    """
    gen = make_episode_fn(env_name, img_size=img_size, length=length, **env_kwargs)
    obs_list, act_list, cues, cue_ends, reveals = [], [], [], [], []
    n_classes = 1
    for i in range(num_episodes):
        rng = np.random.default_rng(seed * 7_654_321 + i)
        obs, act, info = gen(rng)
        o, a = _episode_to_tensors(obs, act)
        obs_list.append(o)
        act_list.append(a)
        cues.append(int(info['cue']))
        cue_ends.append(int(info['cue_end']))
        reveals.append(int(info['reveal']))
        n_classes = int(info['n_cue_classes'])
    return {
        'obs': torch.stack(obs_list),
        'actions': torch.stack(act_list),
        'cue': torch.tensor(cues, dtype=torch.long),
        'cue_end': torch.tensor(cue_ends, dtype=torch.long),
        'reveal': torch.tensor(reveals, dtype=torch.long),
        'n_cue_classes': n_classes,
        'env_name': env_name,
    }
