#!/usr/bin/env python3
"""Evaluate shifted/longer blackout masks without retraining shared-feature models."""

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.data import PrecomputedFeatureDataset  # noqa: E402
from lewm.models.memory_model import MemoryLeWorldModel  # noqa: E402


ENVS = (
    'dmc:reacher.hard.occ', 'dmc:ball_in_cup.catch.occ',
    'dmc:finger.spin.occ', 'dmc:cheetah.run.occ', 'ogbench:cube-single.occ',
)
DEFAULT_DESIGNS = ('none', 'multi', 'gru', 'ssm', 'smt')
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
CONDITIONS = {
    'original_10_16': (10, 16),
    'early_6_12': (6, 12),
    'late_14_20': (14, 20),
    'longer_10_19': (10, 19),
}


def build_model(cfg, action_dim):
    mode = cfg['memory_mode']
    impl = mode if mode in (
        'multi', 'gru', 'ssm', 'retrieval', 'smt',
        'smtv3', 'smtv3_static', 'smtv3_old', 'smtv3_oracle') else 'ema'
    ema_mode = 'both' if impl != 'ema' else mode
    return MemoryLeWorldModel(
        img_size=cfg['img_size'], patch_size=cfg['patch_size'], embed_dim=cfg['embed_dim'],
        action_dim=action_dim, encoder_layers=cfg['encoder_layers'],
        encoder_heads=cfg['encoder_heads'], predictor_layers=cfg['predictor_layers'],
        predictor_heads=cfg['predictor_heads'], history_len=cfg['history_len'],
        dropout=cfg['dropout'], sigreg_lambda=cfg['sigreg_lambda'],
        sigreg_projections=cfg['sigreg_projections'], memory_mode=ema_mode,
        memory_impl=impl, tau_fast=cfg['tau_fast'], tau_slow=cfg['tau_slow'],
        learnable_alpha=not cfg.get('fixed_alpha', True),
        smt_router=cfg.get('smt_router', 'softmax'), encoder_type='precomputed')


@torch.no_grad()
def evaluate(model, loader, device, use_amp, start, end, constant_target):
    model.eval(); h = model.history_len
    sums = defaultdict(float); episodes = 0
    for obs_original, act, target, _mask in loader:
        act = act.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # All original blackout entries are the same exact DINO-PCA black-frame feature.
        black = obs_original[:, 10].to(device, non_blocking=True)
        if not torch.equal(black, obs_original[:, 15].to(device, non_blocking=True)):
            raise ValueError('feature cache does not contain a constant original black token')
        if not torch.equal(black, black[:1].expand_as(black)):
            raise ValueError('black feature differs across episodes')
        obs = target.clone(); obs[:, start:end] = black[:, None]
        ctx = torch.autocast('cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
        with ctx:
            update_mask = torch.ones((obs.shape[0], obs.shape[1]), dtype=torch.bool,
                                     device=device)
            update_mask[:, start:end] = False
            z_full = model._inject(obs, memory_update_mask=update_mask)
            B, L, D = obs.shape; W = L - h
            full_win = z_full.unfold(1, h, 1)[:, :W].permute(0, 1, 3, 2).reshape(B * W, h, D)
            raw_win = obs.unfold(1, h, 1)[:, :W].permute(0, 1, 3, 2).reshape(B * W, h, D)
            act_win = act.unfold(1, h, 1)[:, :W].permute(0, 1, 3, 2).reshape(B * W, h, -1)
            pred = model.predictor(full_win, act_win)[:, -1].reshape(B, W, D)
            pred_ab = model.predictor(raw_win, act_win)[:, -1].reshape(B, W, D)
            error = (pred - target[:, h:]).float().square().mean(-1)
            error_ab = (pred_ab - target[:, h:]).float().square().mean(-1)
        target_times = torch.arange(h, L)
        first = int((target_times == end).nonzero(as_tuple=False)[0])
        recovery = (target_times > end) & (target_times < end + h)
        deep = (target_times >= start + h) & (target_times < end)
        if not recovery.any() or not deep.any():
            raise ValueError(f'invalid mask phase for [{start},{end})')
        constant = constant_target.to(device).view(1, 1, D)
        constant_error = (constant - target[:, h:]).float().square().mean(-1)
        last_visible = obs[:, start - 1:start]
        last_visible_first = (last_visible[:, 0] - target[:, end]).float().square().mean(-1)
        sums['first_post_mse'] += float(error[:, first].sum())
        sums['first_post_mse_ablated'] += float(error_ab[:, first].sum())
        sums['recovery_mse'] += float(error[:, recovery].sum()) / int(recovery.sum())
        sums['recovery_mse_ablated'] += float(error_ab[:, recovery].sum()) / int(recovery.sum())
        sums['deep_blackout_mse'] += float(error[:, deep].sum()) / int(deep.sum())
        sums['deep_blackout_mse_ablated'] += float(error_ab[:, deep].sum()) / int(deep.sum())
        sums['constant_first_post_mse'] += float(constant_error[:, first].sum())
        sums['last_visible_first_post_mse'] += float(last_visible_first.sum())
        episodes += B
    if episodes == 0:
        raise ValueError('empty validation feature dataset')
    return {key: value / episodes for key, value in sums.items()}


def mean(xs): return sum(xs) / len(xs)


def popstd(xs):
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('outputs/shared_clean_occlusion'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--designs', nargs='+', default=list(DEFAULT_DESIGNS))
    ap.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS))
    args = ap.parse_args(); root = args.root.resolve()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    expected = set(itertools.product(ENVS, args.designs, args.seeds))
    rows = []; seen = set(); dataset_cache = {}
    for model_path in sorted(root.glob('*/model.pt')):
        ck = torch.load(model_path, map_location='cpu', weights_only=False); cfg = ck['args']
        key = (cfg.get('env_id'), cfg.get('memory_mode'), cfg.get('seed'))
        if key not in expected or key in seen:
            raise SystemExit(f'unexpected/duplicate checkpoint {model_path}: {key}')
        seen.add(key); env, design, seed = key
        if cfg.get('encoder_type') != 'precomputed' or not cfg.get('mask_occluded_target_loss'):
            raise SystemExit(f'{model_path}: not a fixed-feature masked-target run')
        val_path = Path(cfg['val_feature_cache']).resolve()
        if env not in dataset_cache:
            dataset = PrecomputedFeatureDataset(
                str(val_path), str(Path(cfg['feature_manifest']).resolve()))
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                                persistent_workers=args.num_workers > 0)
            dataset_cache[env] = (dataset, loader)
        dataset, loader = dataset_cache[env]
        model = build_model(cfg, dataset.n_actions)
        model.load_state_dict(ck['model_state_dict'], strict=True); model.to(device).eval()
        constant = torch.from_numpy(dataset.constant_target)
        for condition, (start, end) in CONDITIONS.items():
            result = evaluate(model, loader, device, use_amp, start, end, constant)
            row = {'run': model_path.parent.name, 'env': env, 'design': design, 'seed': seed,
                   'condition': condition, 'mask_start': start, 'mask_end': end,
                   'mask_length': end - start, **result}
            row['first_post_r2'] = 1.0 - row['first_post_mse'] / row['constant_first_post_mse']
            rows.append(row)
        original = rows[-len(CONDITIONS)]
        reference = ck['final_metrics']['clean_mse_first_post']
        if abs(original['first_post_mse'] - reference) > 2e-4:
            raise SystemExit(
                f'{model_path}: original-mask parity failed '
                f"{original['first_post_mse']} vs {reference}")
        del model, ck
        if device.type == 'cuda': torch.cuda.empty_cache()
    if seen != expected:
        raise SystemExit(f'incomplete factorial {len(seen)}/{len(expected)}')
    write_csv(root / 'mask_generalization_per_run.csv', rows)
    groups = defaultdict(list)
    for row in rows: groups[(row['env'], row['design'], row['condition'])].append(row)
    grouped = []
    metric_keys = [
        k for k in rows[0]
        if k.endswith('_mse') or k.endswith('_mse_ablated') or k == 'first_post_r2'
    ]
    for (env, design, condition), group in sorted(groups.items()):
        out = {'env': env, 'design': design, 'condition': condition, 'n_seeds': len(group)}
        for metric in metric_keys:
            values = [r[metric] for r in group]
            out[f'{metric}_mean'] = mean(values); out[f'{metric}_std'] = popstd(values)
        grouped.append(out)
    write_csv(root / 'mask_generalization_grouped.csv', grouped)
    print(f'validated {len(seen)} checkpoints; wrote {len(rows)} mask evaluations')


if __name__ == '__main__':
    main()
