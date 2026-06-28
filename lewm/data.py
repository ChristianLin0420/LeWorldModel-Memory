"""Datasets and eval-batch helpers for the memory-stressing environments."""

import hashlib
import json
from pathlib import Path
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


class PopgymDataset(Dataset):
    """Offline POPGym Arcade trajectories: serves (obs (L,3,H,W) float, action one-hot
    (L-1, n_actions) float). Trajectories are collected/cached once via get_or_collect.

    When ``target_env_id`` is provided, a synchronized clean trajectory is returned as a
    third tensor.  This is used by the occlusion integrity experiment: the model sees the
    ``.occ`` frames but predicts the corresponding unoccluded frame in a shared encoder.
    The action arrays must match exactly, so a cache or collection mismatch fails closed.
    """

    def __init__(self, env_id: str, num_episodes: int, length: int = 32, img_size: int = 64,
                 seed: int = 0, data_dir: str = 'outputs/popgym_data',
                 prototype_seed: int = 0, target_env_id: str = None,
                 mask_occluded_target_loss: bool = False):
        from lewm.envs.popgym_arcade import get_or_collect
        self.obs, self.act, self.n_actions = get_or_collect(
            env_id, num_episodes, length, img_size=img_size, seed=seed,
            data_dir=data_dir, prototype_seed=prototype_seed)
        self.target_obs = None
        self.target_valid_mask = None
        if mask_occluded_target_loss and target_env_id is None:
            raise ValueError('mask_occluded_target_loss requires target_env_id')
        if target_env_id is not None:
            self.target_obs, target_act, target_n_actions = get_or_collect(
                target_env_id, num_episodes, length, img_size=img_size, seed=seed,
                data_dir=data_dir, prototype_seed=prototype_seed)
            if target_n_actions != self.n_actions:
                raise ValueError(
                    f"action-count mismatch for {env_id} -> {target_env_id}: "
                    f"{self.n_actions} != {target_n_actions}")
            if not np.array_equal(self.act, target_act):
                raise ValueError(
                    f"action trajectories are not synchronized for {env_id} -> {target_env_id} "
                    f"(seed={seed}, prototype_seed={prototype_seed})")
            if self.target_obs.shape != self.obs.shape:
                raise ValueError(
                    f"observation-shape mismatch for {env_id} -> {target_env_id}: "
                    f"{self.obs.shape} != {self.target_obs.shape}")
            if env_id.endswith('.occ'):
                occ_start = length // 3
                occ_end = min(length, occ_start + max(4, length // 5))
                if not np.array_equal(self.obs[:, :occ_start], self.target_obs[:, :occ_start]):
                    raise ValueError(f'pre-occlusion pixels differ for {env_id} -> {target_env_id}')
                if not np.array_equal(self.obs[:, occ_end:], self.target_obs[:, occ_end:]):
                    raise ValueError(f'post-occlusion pixels differ for {env_id} -> {target_env_id}')
                if np.any(self.obs[:, occ_start:occ_end]):
                    raise ValueError(f'occluded input is nonblack for {env_id}')
                if not np.any(self.target_obs[:, occ_start:occ_end]):
                    raise ValueError(f'clean target interval is entirely black for {target_env_id}')
            if mask_occluded_target_loss:
                if not env_id.endswith('.occ'):
                    raise ValueError('mask_occluded_target_loss requires a .occ input environment')
                self.target_valid_mask = np.ones(length, dtype=np.bool_)
                self.target_valid_mask[occ_start:occ_end] = False

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int):
        o = torch.from_numpy(self.obs[idx].astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
        a = self.act[idx].astype(np.int64)
        a1h = np.zeros((a.shape[0], self.n_actions), dtype=np.float32)
        a1h[np.arange(a.shape[0]), a] = 1.0
        a1h = torch.from_numpy(a1h)
        if self.target_obs is None:
            return o, a1h
        target = torch.from_numpy(self.target_obs[idx].astype(np.float32) / 255.0)
        target = target.permute(0, 3, 1, 2).contiguous()
        if self.target_valid_mask is not None:
            return o, a1h, target, torch.from_numpy(self.target_valid_mask.copy())
        return o, a1h, target


class PrecomputedFeatureDataset(Dataset):
    """Validated paired feature trajectories for a fixed external encoder.

    The NPZ is produced by ``scripts/precompute_dino_clean_features.py``. Inputs are
    the occluded DINO-PCA features, targets are synchronized clean features, and the
    blackout target mask keeps hidden clean frames evaluation-only during training.
    """

    def __init__(self, path: str, manifest_path: str = None):
        artifact_path = Path(path).resolve()
        path = str(artifact_path)
        with np.load(path, allow_pickle=False) as data:
            required = {
                'schema_version', 'split', 'clean_env', 'occ_env', 'features_input',
                'features_target', 'actions', 'target_valid_mask', 'n_actions',
                'constant_target', 'feature_dim', 'manifest_sha256',
            }
            missing = required - set(data.files)
            if missing:
                raise ValueError(f'{path}: missing feature-cache fields {sorted(missing)}')
            content_fields = required - {'manifest_sha256'}
            content_hashes = {}
            for name in content_fields:
                array = np.ascontiguousarray(np.asarray(data[name]))
                digest = hashlib.sha256()
                digest.update(str(array.dtype).encode('ascii'))
                digest.update(repr(array.shape).encode('ascii'))
                digest.update(array.tobytes(order='C'))
                content_hashes[name] = digest.hexdigest()
            if int(data['schema_version']) != 1:
                raise ValueError(f'{path}: unsupported feature-cache schema')
            self.split = str(data['split'])
            self.clean_env = str(data['clean_env'])
            self.occ_env = str(data['occ_env'])
            self.features_input = np.array(data['features_input'], dtype=np.float32, copy=True)
            self.features_target = np.array(data['features_target'], dtype=np.float32, copy=True)
            self.act = np.array(data['actions'], dtype=np.int64, copy=True)
            self.target_valid_mask = np.array(data['target_valid_mask'], dtype=np.bool_, copy=True)
            self.n_actions = int(data['n_actions'])
            self.constant_target = np.array(data['constant_target'], dtype=np.float32, copy=True)
            self.feature_dim = int(data['feature_dim'])
            self.manifest_sha256 = str(data['manifest_sha256'])
        if manifest_path is not None:
            manifest_path = Path(manifest_path).resolve()
            if not manifest_path.is_file():
                raise FileNotFoundError(f'feature manifest not found: {manifest_path}')
            manifest_bytes = manifest_path.read_bytes()
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            if self.manifest_sha256 != manifest_sha256:
                raise ValueError(f'{path}: embedded manifest hash does not match {manifest_path}')
            manifest = json.loads(manifest_bytes)
            artifact_files = manifest.get('artifact_files', {})
            if artifact_files.get(self.split) != artifact_path.name:
                raise ValueError(f'{path}: artifact name does not match manifest split {self.split!r}')
            expected_hashes = manifest.get('output_content_hashes', {}).get(self.split)
            if not isinstance(expected_hashes, dict) or set(expected_hashes) != content_fields:
                raise ValueError(f'{manifest_path}: incomplete {self.split} content hashes')
            for name, actual in content_hashes.items():
                if expected_hashes[name] != actual:
                    raise ValueError(f'{path}: content hash mismatch for {name}')
        if self.split not in {'train', 'val'}:
            raise ValueError(f'{path}: invalid split {self.split!r}')
        if self.features_input.shape != self.features_target.shape or self.features_input.ndim != 3:
            raise ValueError(f'{path}: paired feature shapes do not match')
        episodes, length, dim = self.features_input.shape
        if dim != self.feature_dim or self.constant_target.shape != (dim,):
            raise ValueError(f'{path}: feature dimension metadata mismatch')
        if self.act.shape != (episodes, length - 1):
            raise ValueError(f'{path}: action shape mismatch')
        if self.target_valid_mask.shape != (length,):
            raise ValueError(f'{path}: target_valid_mask shape mismatch')
        if self.n_actions < 1 or self.act.min() < 0 or self.act.max() >= self.n_actions:
            raise ValueError(f'{path}: invalid action indices/count')
        if not (np.isfinite(self.features_input).all() and
                np.isfinite(self.features_target).all() and
                np.isfinite(self.constant_target).all()):
            raise ValueError(f'{path}: non-finite features')
        occ_start = length // 3
        occ_end = min(length, occ_start + max(4, length // 5))
        expected_mask = np.ones(length, dtype=np.bool_)
        expected_mask[occ_start:occ_end] = False
        if not np.array_equal(self.target_valid_mask, expected_mask):
            raise ValueError(f'{path}: target-valid mask does not match the occlusion protocol')
        if not np.array_equal(
                self.features_input[:, expected_mask], self.features_target[:, expected_mask]):
            raise ValueError(f'{path}: input/target features differ outside the blackout')

    def __len__(self):
        return len(self.features_input)

    def __getitem__(self, idx):
        actions = self.act[idx]
        one_hot = np.zeros((actions.shape[0], self.n_actions), dtype=np.float32)
        one_hot[np.arange(actions.shape[0]), actions] = 1.0
        return (
            torch.from_numpy(self.features_input[idx]),
            torch.from_numpy(one_hot),
            torch.from_numpy(self.features_target[idx]),
            torch.from_numpy(self.target_valid_mask.copy()),
        )


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
