#!/usr/bin/env python3
"""Dependency-light tests for deterministic W&B evaluation-rollout artifacts."""

from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.train_popgym as training


class _Dataset:
    def __init__(self):
        self.n_actions = 3
        self.target_valid_mask = np.asarray(
            [True, True, False, False, True, True, True, True], dtype=np.bool_)
        self.act = np.asarray([[0, 1, 2, 0, 1, 2, 0]], dtype=np.int64)
        base = np.arange(8 * 4, dtype=np.float32).reshape(8, 4) / 10.0
        self.features_input = base[None].copy()
        self.features_input[:, ~self.target_valid_mask] = -1.0
        self.features_target = base[None].copy()

    def __len__(self):
        return 1

    def __getitem__(self, index):
        actions = np.zeros((7, self.n_actions), dtype=np.float32)
        actions[np.arange(7), self.act[index]] = 1.0
        return (
            torch.from_numpy(self.features_input[index]),
            torch.from_numpy(actions),
            torch.from_numpy(self.features_target[index]),
            torch.from_numpy(self.target_valid_mask.copy()),
        )


class _Predictor:
    def __call__(self, latent_windows, _action_windows):
        return latent_windows


class _Model:
    predictor = _Predictor()

    def eval(self):
        return self

    def encode(self, values):
        return values

    def _inject(self, values, **_kwargs):
        return values + 0.25


def test_rollout_trace_and_archive() -> None:
    dataset = _Dataset()
    arrays = training.evaluation_rollout_trace(
        _Model(), dataset, 0, torch.device('cpu'), False, history_len=2)
    assert int(arrays['schema_version']) == 1
    assert arrays['target_times'].tolist() == [2, 3, 4, 5, 6, 7]
    assert arrays['actions_to_target'].tolist() == [1, 2, 0, 1, 2, 0]
    assert arrays['prediction'].shape == (6, 4)
    assert arrays['target'].shape == (6, 4)
    assert np.isfinite(arrays['mse']).all()
    assert not np.array_equal(arrays['mse'], arrays['mse_no_memory'])

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'eval_rollout.npz'
        np.savez_compressed(path, **arrays)
        assert len(training.sha256_file(path)) == 64
        with np.load(path, allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            assert saved['prediction'].shape == (6, 4)


def test_paired_rollout_video_validation() -> None:
    dataset = _Dataset()
    obs = np.arange(1 * 8 * 4 * 5 * 3, dtype=np.uint8).reshape(1, 8, 4, 5, 3)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'clean_val.npz'
        np.savez(
            path,
            obs=obs,
            actions=dataset.act.astype(np.int32),
            n_actions=np.asarray(dataset.n_actions),
            prototype_seed=np.asarray(0),
            schema_version=np.asarray(3),
            cache_role=np.asarray('clean_or_full'),
        )
        video = training.evaluation_rollout_video(path, dataset, 0)
        assert video.shape == (8, 3, 4, 14)
        # Left panel is black during the hidden interval; right panel stays clean.
        assert video[0, :, :, :5].any()
        assert not video[2, :, :, :5].any()
        assert video[2, :, :, 9:].any()
        assert (video[:, :, :, 5:9] == 255).all()


if __name__ == '__main__':
    tests = (test_rollout_trace_and_archive, test_paired_rollout_video_validation)
    for test in tests:
        test()
        print(f'{test.__name__}: OK')
    print(f'All {len(tests)} W&B evaluation-rollout tests passed.')
