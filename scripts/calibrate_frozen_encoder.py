#!/usr/bin/env python3
"""Calibrate deterministic projector normalization for a frozen shared encoder.

The repository's ViT projector uses BatchNorm with ``track_running_stats=False``;
therefore even ``eval()`` latents depend on the other frames in the current batch.
For common-target comparisons we instead estimate the projector input's per-channel
mean/variance once from CLEAN TRAIN frames and later install those values as frozen
running statistics. Validation frames are never used here.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.data import PopgymDataset
from lewm.models.memory_model import MemoryLeWorldModel


def build_source(args, action_dim):
    mode = args['memory_mode']
    impl = mode if mode in ('multi', 'gru', 'ssm', 'retrieval', 'smt') else 'ema'
    ema_mode = 'both' if impl != 'ema' else mode
    return MemoryLeWorldModel(
        img_size=args['img_size'], patch_size=args['patch_size'],
        embed_dim=args['embed_dim'], action_dim=action_dim,
        encoder_layers=args['encoder_layers'], encoder_heads=args['encoder_heads'],
        predictor_layers=args['predictor_layers'], predictor_heads=args['predictor_heads'],
        history_len=args['history_len'], dropout=args['dropout'],
        sigreg_lambda=args['sigreg_lambda'], sigreg_projections=args['sigreg_projections'],
        memory_mode=ema_mode, memory_impl=impl, tau_fast=args['tau_fast'],
        tau_slow=args['tau_slow'], learnable_alpha=not args.get('fixed_alpha', True),
        smt_router=args.get('smt_router', 'softmax'))


def validate_existing(path, source, clean_env, n, length, prototype_seed,
                      min_mean_channel_var, min_effective_rank):
    if not path.is_file():
        return False
    stats = torch.load(path, map_location='cpu', weights_only=False)
    expected = {
        'schema_version': 2,
        'source_checkpoint': str(source.resolve()),
        'source_checkpoint_size': source.stat().st_size,
        'source_checkpoint_mtime_ns': source.stat().st_mtime_ns,
        'clean_env': clean_env,
        'train_episodes': n,
        'length': length,
        'prototype_seed': prototype_seed,
        'frame_count': n * length,
        'min_mean_channel_var': min_mean_channel_var,
        'min_effective_rank': min_effective_rank,
    }
    for key, value in expected.items():
        if stats.get(key) != value:
            raise ValueError(f'{path}: existing {key}={stats.get(key)!r}, expected {value!r}')
    mean = torch.as_tensor(stats.get('pre_bn_mean'))
    var = torch.as_tensor(stats.get('pre_bn_var'))
    if mean.ndim != 1 or var.shape != mean.shape or not torch.isfinite(mean).all():
        raise ValueError(f'{path}: invalid existing mean/variance')
    if not torch.isfinite(var).all() or (var < 0).any():
        raise ValueError(f'{path}: invalid existing variance')
    if not stats.get('quality_pass'):
        raise ValueError(
            f"{path}: target encoder failed quality gate: {stats.get('quality_failure')}")
    print(f'validated existing encoder statistics: {path}')
    return True


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--clean-env', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--data-dir', default='outputs/popgym_data')
    ap.add_argument('--num-episodes', type=int, default=600)
    ap.add_argument('--length', type=int, default=32)
    ap.add_argument('--img-size', type=int, default=64)
    ap.add_argument('--prototype-seed', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=8, help='episodes per calibration batch')
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--min-mean-channel-var', type=float, default=1e-5)
    ap.add_argument('--min-effective-rank', type=float, default=2.0)
    args = ap.parse_args()

    source = args.checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if validate_existing(
            args.output, source, args.clean_env, args.num_episodes,
            args.length, args.prototype_seed, args.min_mean_channel_var,
            args.min_effective_rank):
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dataset = PopgymDataset(
        args.clean_env, args.num_episodes, args.length, args.img_size, seed=0,
        data_dir=args.data_dir, prototype_seed=args.prototype_seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0)

    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    cfg = checkpoint['args']
    if cfg.get('env_id') != args.clean_env:
        raise ValueError(
            f"source checkpoint env {cfg.get('env_id')!r} != clean env {args.clean_env!r}")
    model = build_source(cfg, dataset.n_actions)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    bn = model.encoder.projector[1]
    if not isinstance(bn, nn.BatchNorm1d) or bn.track_running_stats:
        raise TypeError('expected source projector BatchNorm1d(track_running_stats=False)')
    embed_dim = cfg['embed_dim']
    eps = float(bn.eps)
    affine_weight = bn.weight.detach().cpu().double().clone()
    affine_bias = bn.bias.detach().cpu().double().clone()
    model.encoder.projector[1] = nn.Identity()

    total = torch.zeros(embed_dim, dtype=torch.float64)
    total_sq = torch.zeros(embed_dim, dtype=torch.float64)
    gram = torch.zeros(embed_dim, embed_dim, dtype=torch.float64)
    temporal_sq = 0.0
    temporal_count = 0
    count = 0
    for batch in loader:
        obs = batch[0].to(device, non_blocking=True)
        features_bt = model.encode(obs).detach().cpu().double()
        features = features_bt.reshape(-1, embed_dim)
        total += features.sum(0)
        total_sq += features.square().sum(0)
        gram += features.T @ features
        temporal_sq += float((features_bt[:, 1:] - features_bt[:, :-1]).square().sum())
        temporal_count += features_bt.shape[0] * (features_bt.shape[1] - 1) * embed_dim
        count += features.shape[0]
    expected_count = args.num_episodes * args.length
    if count != expected_count:
        raise ValueError(f'calibrated {count} frames, expected {expected_count}')
    mean = total / count
    var = (total_sq / count - mean.square()).clamp_min(0)
    if not torch.isfinite(mean).all() or not torch.isfinite(var).all():
        raise ValueError('non-finite calibration statistics')

    covariance = gram / count - torch.outer(mean, mean)
    covariance = (covariance + covariance.T) * 0.5
    eigvals = torch.linalg.eigvalsh(covariance).clamp_min(0)
    eigsum = float(eigvals.sum())
    effective_rank = float(eigsum ** 2 / max(float(eigvals.square().sum()), 1e-30))
    mean_channel_var = float(var.mean())
    fixed_output_var = affine_weight.square() * var / (var + eps)
    fixed_mean_channel_var = float(fixed_output_var.mean())
    temporal_step_mse = temporal_sq / max(temporal_count, 1)
    quality_failures = []
    if mean_channel_var < args.min_mean_channel_var:
        quality_failures.append(
            f'mean_channel_var={mean_channel_var:.6g} < {args.min_mean_channel_var:.6g}')
    if effective_rank < args.min_effective_rank:
        quality_failures.append(
            f'effective_rank={effective_rank:.6g} < {args.min_effective_rank:.6g}')

    record = {
        'schema_version': 2,
        'protocol': 'pre-BN channel moments on clean training frames only; population variance',
        'source_checkpoint': str(source),
        'source_checkpoint_size': source.stat().st_size,
        'source_checkpoint_mtime_ns': source.stat().st_mtime_ns,
        'clean_env': args.clean_env,
        'data_dir': str(Path(args.data_dir).resolve()),
        'train_seed': 0,
        'train_episodes': args.num_episodes,
        'length': args.length,
        'img_size': args.img_size,
        'prototype_seed': args.prototype_seed,
        'dataset_schema_version': 3,
        'frame_count': count,
        'embed_dim': embed_dim,
        'bn_eps': eps,
        'pre_bn_mean': mean,
        'pre_bn_var': var,
        'source_bn_affine_weight': affine_weight,
        'source_bn_affine_bias': affine_bias,
        'mean_channel_var': mean_channel_var,
        'fixed_mean_channel_var': fixed_mean_channel_var,
        'covariance_effective_rank': effective_rank,
        'covariance_eigenvalues': eigvals,
        'temporal_step_mse': temporal_step_mse,
        'min_mean_channel_var': args.min_mean_channel_var,
        'min_effective_rank': args.min_effective_rank,
        'quality_pass': not quality_failures,
        'quality_failure': '; '.join(quality_failures),
    }
    temporary = args.output.with_name(f'.{args.output.name}.tmp-{os.getpid()}')
    try:
        torch.save(record, temporary)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f'wrote {args.output}: frames={count}, '
        f'mean_var={mean_channel_var:.6g}, effective_rank={effective_rank:.3f}, '
        f'fixed_var={fixed_mean_channel_var:.6g}, device={device}')
    if quality_failures:
        raise SystemExit(f'target encoder quality gate failed: {record["quality_failure"]}')


if __name__ == '__main__':
    main()
