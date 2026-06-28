#!/usr/bin/env python3
"""Fail-closed staged analysis for the leakage-safe HACSM-v4 cohort.

Raw PCA MSE is never averaged across environments.  Cross-environment summaries use paired
relative changes within environment/optimizer seed; tables retain environment-specific scales.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lewm.data import PrecomputedFeatureDataset
from lewm.models.memory_model import MemoryLeWorldModel


OCC_TO_CLEAN = {
    'dmc:reacher.hard.occ': 'dmc:reacher.hard',
    'dmc:ball_in_cup.catch.occ': 'dmc:ball_in_cup.catch',
    'dmc:finger.spin.occ': 'dmc:finger.spin',
    'dmc:cheetah.run.occ': 'dmc:cheetah.run',
    'ogbench:cube-single.occ': 'ogbench:cube-single',
}
PILOT_DESIGNS = (
    'none', 'multi', 'ssm', 'smtv3', 'hacsmv4_static', 'hacsmv4_noaction',
    'hacsmv4_noaux', 'hacsmv4_single', 'hacsmv4',
)
EXPANSION_DESIGNS = ('gru', 'smtv1', 'smtv2')
FINAL_DESIGNS = PILOT_DESIGNS + EXPANSION_DESIGNS
HAC_DESIGNS = tuple(design for design in PILOT_DESIGNS if design.startswith('hacsmv4'))
PILOT_SEEDS = (0, 1, 2)
FINAL_SEEDS = (0, 1, 2, 3, 4)
PRIMARY = 'clean_mse_first_post'
EPOCHS = 200
WINDOW = 10

MASKS = {
    'original': (10, 16),
    'early': (6, 12),
    'late': (14, 20),
    'longer': (10, 19),
}


def finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} is not finite: {value!r}')
    return result


def safe_env(value: str) -> str:
    return ''.join(char if char.isalnum() else '_' for char in value).strip('_')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f'refusing to write empty CSV {path}')
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f'inconsistent columns for {path}')
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_cells(root: Path, designs: Sequence[str], seeds: Sequence[int]):
    expected = {
        (env, design, seed): root / f'lewm-{env}-{design}-s{seed}'
        for env in OCC_TO_CLEAN for design in designs for seed in seeds
    }
    rows = []
    checkpoints = {}
    convergence = []
    for key, run_dir in sorted(expected.items()):
        env, design, seed = key
        model_path, metrics_path = run_dir / 'model.pt', run_dir / 'metrics.json'
        if not model_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f'missing complete checkpoint pair: {run_dir}')
        metrics = json.loads(metrics_path.read_text(), parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f'non-RFC JSON constant {token}')))
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if metrics != checkpoint.get('final_metrics'):
            raise ValueError(f'{run_dir}: metrics.json/checkpoint mismatch')
        cfg = checkpoint.get('args')
        history = checkpoint.get('history')
        if not isinstance(cfg, dict) or not isinstance(history, list) or len(history) != EPOCHS:
            raise ValueError(f'{run_dir}: malformed args/history')
        exact = {
            'env_id': env, 'target_env_id': OCC_TO_CLEAN[env], 'memory_mode': design,
            'seed': seed, 'num_episodes': 600, 'val_episodes': 150, 'length': 32,
            'epochs': EPOCHS, 'batch_size': 64, 'lr': 3e-4, 'weight_decay': 1e-5,
            'embed_dim': 128, 'history_len': 3, 'predictor_norm': 'none',
            'first_post_loss_weight': 0.5, 'hier_loss_weight': 0.1,
            'encoder_type': 'precomputed', 'mask_occluded_target_loss': True,
        }
        for name, wanted in exact.items():
            if cfg.get(name) != wanted:
                raise ValueError(f'{run_dir}: {name}={cfg.get(name)!r}, expected {wanted!r}')
        for epoch, record in enumerate(history, 1):
            if record.get('epoch') != epoch:
                raise ValueError(f'{run_dir}: non-contiguous history at epoch {epoch}')
            for split in ('train', 'val'):
                values = record.get(split, {})
                for metric in ('loss', 'pred_loss', 'sigreg_loss'):
                    finite(values.get(metric), f'{run_dir} {epoch} {split}.{metric}')
            if design.startswith('hacsmv4'):
                for split in ('train', 'val'):
                    for metric in ('hier_loss', 'hier_loss_fast', 'hier_loss_medium',
                                   'hier_loss_slow', 'hier_loss_h1', 'hier_loss_h2',
                                   'hier_loss_h4', 'hier_loss_h8', 'hier_loss_h16'):
                        finite(record[split].get(metric), f'{run_dir} {epoch} {split}.{metric}')
        primary = finite(metrics.get(PRIMARY), f'{run_dir} {PRIMARY}')
        metric_fields = (
            'val_pred_loss', 'clean_mse_pre', 'clean_mse_blackout_transition',
            'clean_mse_deep_blackout', 'clean_mse_first_post', 'clean_mse_recovery',
            'clean_mse_late_post', 'clean_mse_all', 'clean_mse_first_post_ablated',
            'clean_input_mse_first_post', 'last_visible_mse_first_post',
            'constant_mse_first_post', 'persistence_mse_first_post',
        )
        row = {'run': run_dir.name, 'env': env, 'design': design, 'seed': seed}
        for name in metric_fields:
            row[name] = finite(metrics.get(name), f'{run_dir} {name}')
        row['trainable_parameters'] = int(metrics['trainable_parameters'])
        for name in ('val_hier_loss', 'val_hier_loss_fast', 'val_hier_loss_medium',
                     'val_hier_loss_slow', 'val_hier_loss_h1', 'val_hier_loss_h2',
                     'val_hier_loss_h4', 'val_hier_loss_h8', 'val_hier_loss_h16'):
            row[name] = (finite(metrics[name], f'{run_dir} {name}')
                         if name in metrics else '')
        rows.append(row)
        previous = mean(finite(item['val']['pred_loss'], 'previous pred')
                        for item in history[-2 * WINDOW:-WINDOW])
        recent = mean(finite(item['val']['pred_loss'], 'recent pred')
                      for item in history[-WINDOW:])
        convergence.append({
            'run': run_dir.name, 'env': env, 'design': design, 'seed': seed,
            'previous_window_mean': previous, 'recent_window_mean': recent,
            'relative_improvement': (previous - recent) / previous,
        })
        checkpoints[key] = {
            'path': model_path, 'cfg': cfg, 'metrics': metrics,
            'sha256': sha256_file(model_path),
        }
    return rows, checkpoints, convergence


def grouped_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    numeric = [name for name in rows[0] if name not in {'run', 'env', 'design', 'seed'}]
    result = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row['env'], row['design'])].append(row)
    for (env, design), values in sorted(groups.items()):
        out = {'env': env, 'design': design, 'n_seeds': len(values)}
        for name in numeric:
            observed = [float(row[name]) for row in values if row[name] != '']
            out[f'{name}_mean'] = mean(observed) if observed else ''
            out[f'{name}_std'] = pstdev(observed) if observed else ''
        result.append(out)
    return result


def contrast_rows(rows: Sequence[Mapping[str, Any]], candidate: str = 'hacsmv4'):
    lookup = {(row['env'], row['design'], int(row['seed'])): row for row in rows}
    designs = sorted({str(row['design']) for row in rows if row['design'] != candidate})
    result = []
    for reference in designs:
        for env in list(OCC_TO_CLEAN) + ['__overall__']:
            envs = list(OCC_TO_CLEAN) if env == '__overall__' else [env]
            pairs = []
            for current_env in envs:
                common_seeds = sorted({int(row['seed']) for row in rows
                                       if row['env'] == current_env and row['design'] == candidate}
                                      & {int(row['seed']) for row in rows
                                         if row['env'] == current_env and row['design'] == reference})
                for seed in common_seeds:
                    cand = float(lookup[(current_env, candidate, seed)][PRIMARY])
                    ref = float(lookup[(current_env, reference, seed)][PRIMARY])
                    pairs.append((cand, ref))
            if not pairs:
                continue
            relative = [(ref - cand) / ref for cand, ref in pairs]
            result.append({
                'candidate': candidate, 'reference': reference, 'env': env,
                'n_pairs': len(pairs),
                'candidate_mean_mse': (mean(x[0] for x in pairs)
                                       if env != '__overall__' else ''),
                'reference_mean_mse': (mean(x[1] for x in pairs)
                                       if env != '__overall__' else ''),
                'mean_paired_relative_reduction': mean(relative),
                'paired_wins': sum(cand < ref for cand, ref in pairs),
                'paired_ties': sum(cand == ref for cand, ref in pairs),
            })
    return result


def _environment_means(rows: Sequence[Mapping[str, Any]], design: str, metric: str = PRIMARY):
    return {
        env: mean(float(row[metric]) for row in rows
                  if row['env'] == env and row['design'] == design)
        for env in OCC_TO_CLEAN
    }


def pilot_decision(rows: Sequence[Mapping[str, Any]], convergence: Sequence[Mapping[str, Any]]):
    contrasts = {(row['reference'], row['env']): row for row in contrast_rows(rows)}
    ssm = contrasts[('ssm', '__overall__')]
    noaction = contrasts[('hacsmv4_noaction', '__overall__')]
    noaux = contrasts[('hacsmv4_noaux', '__overall__')]
    v4_env = _environment_means(rows, 'hacsmv4')
    ssm_env = _environment_means(rows, 'ssm')
    hold_env = _environment_means(rows, 'hacsmv4', 'last_visible_mse_first_post')
    clean_v4 = [float(row['clean_input_mse_first_post']) for row in rows
                if row['design'] == 'hacsmv4']
    clean_ssm_lookup = {(row['env'], row['seed']): float(row['clean_input_mse_first_post'])
                        for row in rows if row['design'] == 'ssm'}
    clean_pairs = [(value, clean_ssm_lookup[(row['env'], row['seed'])])
                   for row, value in zip(
                       [row for row in rows if row['design'] == 'hacsmv4'], clean_v4)]
    convergence_values = np.asarray([float(row['relative_improvement']) for row in convergence])
    criteria = {
        'v4_vs_ssm_mean_paired_reduction_ge_1pct':
            float(ssm['mean_paired_relative_reduction']) >= 0.01,
        'v4_vs_ssm_wins_ge_9_of_15': int(ssm['paired_wins']) >= 9,
        'v4_vs_ssm_env_mean_wins_ge_3_of_5':
            sum(v4_env[env] < ssm_env[env] for env in OCC_TO_CLEAN) >= 3,
        'v4_vs_noaction_positive_and_wins_ge_8_of_15':
            float(noaction['mean_paired_relative_reduction']) > 0 and int(noaction['paired_wins']) >= 8,
        'v4_vs_noaux_positive_and_wins_ge_8_of_15':
            float(noaux['mean_paired_relative_reduction']) > 0 and int(noaux['paired_wins']) >= 8,
        'v4_beats_hold_in_ge_3_of_5_env_means':
            sum(v4_env[env] < hold_env[env] for env in OCC_TO_CLEAN) >= 3,
        'clean_input_worsening_vs_ssm_le_10pct':
            mean((v4 - base) / base for v4, base in clean_pairs) <= 0.10,
    }
    observed = {
        'v4_vs_ssm_mean_paired_relative_reduction': ssm['mean_paired_relative_reduction'],
        'v4_vs_ssm_paired_wins': ssm['paired_wins'],
        'v4_vs_ssm_env_mean_wins': sum(v4_env[e] < ssm_env[e] for e in OCC_TO_CLEAN),
        'v4_vs_noaction_mean_paired_relative_reduction': noaction['mean_paired_relative_reduction'],
        'v4_vs_noaction_paired_wins': noaction['paired_wins'],
        'v4_vs_noaux_mean_paired_relative_reduction': noaux['mean_paired_relative_reduction'],
        'v4_vs_noaux_paired_wins': noaux['paired_wins'],
        'v4_hold_env_wins': sum(v4_env[e] < hold_env[e] for e in OCC_TO_CLEAN),
        'clean_input_relative_worsening_vs_ssm': mean(
            (v4 - base) / base for v4, base in clean_pairs),
        'descriptive_convergence_median_last10_vs_previous10':
            float(np.median(convergence_values)),
        'descriptive_convergence_p95_last10_vs_previous10':
            float(np.quantile(convergence_values, 0.95)),
        'descriptive_convergence_max_last10_vs_previous10': float(convergence_values.max()),
    }
    expand = all(criteria.values())
    return {
        'schema_version': 1, 'phase': 'pilot', 'decision': 'EXPAND' if expand else 'NO_GO',
        'expand': expand, 'criteria': criteria, 'observed': observed,
        'note': ('Prospective permissive screen. A pass authorizes the five-seed/full-baseline '
                 'expansion; it is not an overall-best or publication claim.'),
    }


def build_model(cfg: Mapping[str, Any], action_dim: int) -> MemoryLeWorldModel:
    logical = str(cfg['memory_mode'])
    if logical in ('smtv1', 'smtv2'):
        impl = 'smt'
        router = 'softmax' if logical == 'smtv1' else 'sigmoid'
    else:
        impl = logical if logical in (
            'multi', 'gru', 'ssm', 'retrieval', 'smt', 'smtv3', 'smtv3_static',
            'smtv3_old', 'smtv3_oracle', 'hacsmv4', 'hacsmv4_static',
            'hacsmv4_noaction', 'hacsmv4_noaux', 'hacsmv4_single', 'hacsmv4_oracle') else 'ema'
        router = cfg.get('smt_router', 'softmax')
    return MemoryLeWorldModel(
        img_size=cfg['img_size'], patch_size=cfg['patch_size'], embed_dim=cfg['embed_dim'],
        action_dim=action_dim, encoder_layers=cfg['encoder_layers'],
        encoder_heads=cfg['encoder_heads'], predictor_layers=cfg['predictor_layers'],
        predictor_heads=cfg['predictor_heads'], predictor_norm=cfg['predictor_norm'],
        history_len=cfg['history_len'], dropout=cfg['dropout'],
        sigreg_lambda=cfg['sigreg_lambda'], sigreg_projections=cfg['sigreg_projections'],
        memory_mode='both' if impl != 'ema' else logical, memory_impl=impl,
        tau_fast=cfg['tau_fast'], tau_slow=cfg['tau_slow'],
        learnable_alpha=not cfg.get('fixed_alpha', True), smt_router=router,
        hier_loss_weight=cfg.get('hier_loss_weight', 0.1), encoder_type='precomputed')


def _predict_first_post(model: MemoryLeWorldModel, observations: torch.Tensor,
                        actions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                        *, gate_override=None, action_override=None,
                        route_override: torch.Tensor | None = None) -> torch.Tensor:
    hidden = torch.nonzero(~mask[0].bool(), as_tuple=False).flatten()
    if hidden.numel() < 1 or not torch.equal(hidden, torch.arange(hidden[0], hidden[-1] + 1,
                                                                  device=hidden.device)):
        raise ValueError('expected one contiguous blackout')
    end = int(hidden[-1]) + 1
    h = model.history_len
    if route_override is None:
        injected = model._inject(
            observations, actions=actions, memory_update_mask=mask,
            gate_override=gate_override, action_override=action_override)
    else:
        if model.memory_impl not in HAC_DESIGNS:
            raise ValueError('route override requires HACSM-v4')
        _unused, details = model._inject(
            observations, actions=actions, memory_update_mask=mask,
            gate_override=gate_override, action_override=action_override,
            return_memory_details=True)
        route = route_override.to(device=observations.device, dtype=observations.dtype)
        route = route / route.sum()
        mixed = (details['states'] * route.view(1, 1, -1, 1)).sum(2)
        mixed = mixed * torch.rsqrt(
            mixed.square().mean(-1, keepdim=True) + model.mem_hacsmv4.rms_eps)
        injected = model.mem_hacsmv4.fuse(observations, mixed)
    prediction = model.predictor(
        injected[:, end - h:end], actions[:, end - h:end])[:, -1]
    return (prediction - targets[:, end]).float().square().mean(-1)


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if scores.shape != labels.shape or positive == 0 or negative == 0:
        raise ValueError('invalid AUROC inputs')
    order = np.argsort(scores, kind='mergesort')
    ranked = scores[order]
    ordered_labels = labels[order]
    rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end] == ranked[start]:
            end += 1
        rank_sum += (((start + 1) + end) / 2.0) * ordered_labels[start:end].sum()
        start = end
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))


@torch.no_grad()
def final_interventions(root: Path, checkpoints: Mapping, device: torch.device):
    """Replay masks and HAC-specific causal interventions after the full cohort is locked."""
    feature_root = Path(checkpoints[next(iter(checkpoints))]['cfg']['feature_manifest']).parent
    mask_rows, gate_rows, action_rows, level_rows = [], [], [], []
    use_amp = device.type == 'cuda'
    for env, clean in OCC_TO_CLEAN.items():
        safe = safe_env(clean)
        manifest = feature_root / f'{safe}_manifest.json'
        train_set = PrecomputedFeatureDataset(feature_root / f'{safe}_train.npz', manifest)
        val_set = PrecomputedFeatureDataset(feature_root / f'{safe}_val.npz', manifest)
        train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
        val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
        # Materialize once; 150x32x128 is small and exact interventions need episode alignment.
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        tr_obs, tr_actions, _tr_targets, tr_mask = (tensor.to(device) for tensor in train_batch)
        val_obs, val_actions, val_targets, _val_mask = (tensor.to(device) for tensor in val_batch)
        # The trained missingness sentinel is the projected all-black image, not the dataset's
        # clean-train constant-prediction baseline.  Recover it from the canonical cached gap and
        # require exact identity across episodes/times before using it for shifted masks.
        canonical_black = val_obs[:, 10:16]
        black_token = canonical_black[0, 0].view(1, 1, -1)
        if not torch.equal(canonical_black, black_token.expand_as(canonical_black)):
            raise ValueError(f'{env}: canonical validation blackout is not one exact token')
        for design in FINAL_DESIGNS:
            for seed in FINAL_SEEDS:
                info = checkpoints[(env, design, seed)]
                checkpoint = torch.load(info['path'], map_location='cpu', weights_only=False)
                model = build_model(info['cfg'], val_set.n_actions)
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                model.to(device).eval()
                context = torch.autocast('cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
                with context:
                    for condition, (start, end) in MASKS.items():
                        observations = val_targets.clone()
                        observations[:, start:end] = black_token
                        mask = torch.ones(observations.shape[:2], device=device, dtype=torch.bool)
                        mask[:, start:end] = False
                        error = _predict_first_post(
                            model, observations, val_actions, val_targets, mask)
                        if condition == 'original':
                            reference = float(info['metrics'][PRIMARY])
                            if abs(float(error.mean()) - reference) > 2e-4:
                                raise ValueError(
                                    f"{env}/{design}/s{seed}: causal replay {float(error.mean())} "
                                    f"does not reproduce checkpoint {reference}")
                        mask_rows.append({
                            'env': env, 'design': design, 'seed': seed,
                            'condition': condition, 'mask_start': start, 'mask_end': end,
                            'n_episodes': error.numel(), 'first_post_mse': float(error.mean()),
                        })
                if design not in HAC_DESIGNS and design not in ('ssm',):
                    del model, checkpoint
                    continue

                observations = val_targets.clone()
                observations[:, 10:16] = black_token
                mask = torch.ones(observations.shape[:2], device=device, dtype=torch.bool)
                mask[:, 10:16] = False
                original = _predict_first_post(
                    model, observations, val_actions, val_targets, mask)
                if design in HAC_DESIGNS:
                    # Calibrate a per-level causal-prefix gate mean on disjoint training episodes.
                    _, train_details = model._inject(
                        tr_obs, actions=tr_actions, memory_update_mask=tr_mask,
                        return_memory_details=True)
                    calibration = train_details['gates'][:, 1:16].mean((0, 1), keepdim=True)
                    _, val_details = model._inject(
                        observations, actions=val_actions, memory_update_mask=mask,
                        return_memory_details=True)
                    gates = val_details['gates'][:, 1:16, :, 0].float().cpu().numpy()
                    labels = mask[:, 1:16].cpu().numpy()
                    gate_override = val_details['gates'].clone()
                    gate_override[:, 1:16] = calibration
                    mean_error = _predict_first_post(
                        model, observations, val_actions, val_targets, mask,
                        gate_override=gate_override)
                    for level, tau in enumerate((2, 8, 32)):
                        level_gates = gates[:, :, level]
                        expanded_labels = np.broadcast_to(labels, level_gates.shape)
                        gate_rows.append({
                            'env': env, 'design': design, 'seed': seed, 'level': level,
                            'tau': tau, 'causal_gate_mean': float(level_gates.mean()),
                            'visible_mean': float(level_gates[expanded_labels].mean()),
                            'black_mean': float(level_gates[~expanded_labels].mean()),
                            'visible_minus_black': float(
                                level_gates[expanded_labels].mean()
                                - level_gates[~expanded_labels].mean()),
                            'visible_auroc': binary_auroc(level_gates, expanded_labels),
                            'temporal_std': float(level_gates.mean(0).std()),
                            'input_std': float(level_gates.std(0).mean()),
                            'original_first_post_mse': float(original.mean()),
                            'mean_gate_first_post_mse': float(mean_error.mean()),
                            'mean_gate_relative_change': float(
                                (mean_error.mean() - original.mean()) / original.mean()),
                        })
                    if design == 'hacsmv4':
                        for level, label in enumerate(('fast_only', 'medium_only', 'slow_only')):
                            route = torch.zeros(3, device=device)
                            route[level] = 1
                            error = _predict_first_post(
                                model, observations, val_actions, val_targets, mask,
                                route_override=route)
                            level_rows.append({
                                'env': env, 'design': design, 'seed': seed, 'condition': label,
                                'first_post_mse': float(error.mean()),
                                'relative_change': float(
                                    (error.mean() - original.mean()) / original.mean()),
                            })
                        for level, label in enumerate(('drop_fast', 'drop_medium', 'drop_slow')):
                            route = model.mem_hacsmv4.route_weights().detach().clone()
                            route[level] = 0
                            error = _predict_first_post(
                                model, observations, val_actions, val_targets, mask,
                                route_override=route)
                            level_rows.append({
                                'env': env, 'design': design, 'seed': seed, 'condition': label,
                                'first_post_mse': float(error.mean()),
                                'relative_change': float(
                                    (error.mean() - original.mean()) / original.mean()),
                            })

                overrides = {'zero_early_blackout': val_actions.clone(),
                             'zero_full_gap': val_actions.clone(),
                             'cross_episode_early_blackout': val_actions.clone()}
                overrides['zero_early_blackout'][:, 10:13] = 0
                overrides['zero_full_gap'][:, 9:15] = 0
                permutation = torch.arange(val_actions.shape[0] - 1, -1, -1, device=device)
                overrides['cross_episode_early_blackout'][:, 10:13] = (
                    val_actions[permutation, 10:13])
                for label, override in overrides.items():
                    error = _predict_first_post(
                        model, observations, val_actions, val_targets, mask,
                        action_override=override)
                    action_rows.append({
                        'env': env, 'design': design, 'seed': seed, 'condition': label,
                        'original_first_post_mse': float(original.mean()),
                        'intervention_first_post_mse': float(error.mean()),
                        'relative_change': float((error.mean() - original.mean()) / original.mean()),
                        'changed_action_fraction': float(
                            (override != val_actions).any(dim=-1).float().mean()),
                    })
                del model, checkpoint
    return mask_rows, gate_rows, action_rows, level_rows


def final_decision(rows: Sequence[Mapping[str, Any]], mask_rows, gate_rows, action_rows,
                   convergence):
    contrasts = {(row['reference'], row['env']): row for row in contrast_rows(rows)}
    v4 = _environment_means(rows, 'hacsmv4')
    learned = [design for design in FINAL_DESIGNS if design != 'none']
    env_means = {design: _environment_means(rows, design) for design in learned}
    env_wins = {
        reference: sum(v4[env] < env_means[reference][env] for env in OCC_TO_CLEAN)
        for reference in ('ssm', 'multi', 'smtv3')
    }
    envelope_wins = sum(
        v4[env] <= min(env_means[design][env] for design in learned if design != 'hacsmv4')
        for env in OCC_TO_CLEAN)
    ssm = contrasts[('ssm', '__overall__')]
    hold = _environment_means(rows, 'hacsmv4', 'last_visible_mse_first_post')
    mask_lookup = defaultdict(dict)
    for row in mask_rows:
        mask_lookup[(row['env'], row['design'], int(row['seed']))][row['condition']] = float(
            row['first_post_mse'])
    shifted = {}
    for condition in MASKS:
        effects = []
        for env in OCC_TO_CLEAN:
            for seed in FINAL_SEEDS:
                candidate = mask_lookup[(env, 'hacsmv4', seed)][condition]
                baseline = mask_lookup[(env, 'ssm', seed)][condition]
                effects.append((baseline - candidate) / baseline)
        shifted[condition] = mean(effects)
    clean_lookup = {(row['env'], row['design'], row['seed']): row for row in rows}
    clean_worsening = mean(
        (float(clean_lookup[(env, 'hacsmv4', seed)]['clean_input_mse_first_post'])
         - float(clean_lookup[(env, 'ssm', seed)]['clean_input_mse_first_post']))
        / float(clean_lookup[(env, 'ssm', seed)]['clean_input_mse_first_post'])
        for env in OCC_TO_CLEAN for seed in FINAL_SEEDS)
    architecture = {}
    for reference in ('hacsmv4_static', 'hacsmv4_noaction', 'hacsmv4_noaux'):
        overall = contrasts[(reference, '__overall__')]
        candidate_env = _environment_means(rows, reference)
        architecture[reference] = {
            'mean_paired_reduction': float(overall['mean_paired_relative_reduction']),
            'paired_wins': int(overall['paired_wins']),
            'env_mean_wins': sum(v4[e] < candidate_env[e] for e in OCC_TO_CLEAN),
        }
    action_summary = {}
    for design in ('hacsmv4', 'hacsmv4_noaction', 'ssm'):
        for condition in ('zero_early_blackout', 'cross_episode_early_blackout', 'zero_full_gap'):
            values = [float(row['relative_change']) for row in action_rows
                      if row['design'] == design and row['condition'] == condition]
            action_summary[f'{design}:{condition}'] = mean(values)
    gate_change = mean(float(row['mean_gate_relative_change']) for row in gate_rows
                       if row['design'] == 'hacsmv4')
    conv = np.asarray([float(row['relative_improvement']) for row in convergence])
    conv_abs = np.abs(conv)
    convergence_pass = (float(np.median(conv_abs)) < 0.01
                        and float(np.quantile(conv_abs, 0.95)) < 0.03
                        and float(conv_abs.max()) < 0.05)
    criteria = {
        'v4_beats_ssm_multi_v3_in_ge_4_of_5_each': all(value >= 4 for value in env_wins.values()),
        'v4_is_learned_baseline_envelope_winner_in_ge_4_of_5': envelope_wins >= 4,
        'v4_vs_ssm_mean_reduction_ge_5pct_and_wins_ge_18_of_25':
            float(ssm['mean_paired_relative_reduction']) >= 0.05 and int(ssm['paired_wins']) >= 18,
        'v4_beats_hold_in_ge_4_of_5': sum(v4[e] < hold[e] for e in OCC_TO_CLEAN) >= 4,
        'v4_vs_ssm_positive_on_every_mask': all(value > 0 for value in shifted.values()),
        'clean_input_worsening_vs_ssm_le_5pct': clean_worsening <= 0.05,
        'full_beats_static_noaction_noaux_mechanism_bar': all(
            value['env_mean_wins'] >= 3 and value['mean_paired_reduction'] >= 0.03
            and value['paired_wins'] >= 17 for value in architecture.values()),
        'in_distribution_cross_episode_memory_actions_worsen_v4_ge_3pct':
            action_summary['hacsmv4:cross_episode_early_blackout'] >= 0.03,
        'noaction_and_ssm_action_override_invariant': max(
            abs(value) for key, value in action_summary.items()
            if key.startswith(('hacsmv4_noaction:', 'ssm:'))) <= 1e-6,
        'mean_gate_replacement_worsens_v4_ge_3pct': gate_change >= 0.03,
        'convergence_gate_passes': convergence_pass,
    }
    overall_best = all(criteria.values())
    if overall_best:
        decision = 'OVERALL_BEST'
    elif float(ssm['mean_paired_relative_reduction']) > 0:
        decision = 'PROMISING_NOT_OVERALL_BEST'
    else:
        decision = 'NO_GO'
    return {
        'schema_version': 1, 'phase': 'final', 'decision': decision,
        'criteria': criteria,
        'observed': {
            'env_mean_wins_vs_key_baselines': env_wins,
            'learned_baseline_envelope_env_wins': envelope_wins,
            'v4_vs_ssm_mean_paired_relative_reduction': ssm['mean_paired_relative_reduction'],
            'v4_vs_ssm_paired_wins': ssm['paired_wins'],
            'v4_hold_env_wins': sum(v4[e] < hold[e] for e in OCC_TO_CLEAN),
            'shifted_mask_v4_vs_ssm_relative_reductions': shifted,
            'clean_input_relative_worsening_vs_ssm': clean_worsening,
            'architecture_ablation_contrasts': architecture,
            'action_interventions': action_summary,
            'mean_gate_relative_change': gate_change,
            'convergence_signed_median': float(np.median(conv)),
            'convergence_absolute_median': float(np.median(conv_abs)),
            'convergence_absolute_p95': float(np.quantile(conv_abs, 0.95)),
            'convergence_absolute_max': float(conv_abs.max()),
        },
        'limitations': [
            'The fixed-DINO validation trajectories use rollout seed 7777 and are not untouched.',
            'Every mask uses the same synthetic black-token corruption family.',
            'The outcome is latent prediction, not simulator-state estimation or executed return.',
        ],
    }


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/hacsm_v4_shared'))
    parser.add_argument('--phase', choices=('pilot', 'final'), required=True)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    designs = PILOT_DESIGNS if args.phase == 'pilot' else FINAL_DESIGNS
    seeds = PILOT_SEEDS if args.phase == 'pilot' else FINAL_SEEDS
    rows, checkpoints, convergence = load_cells(args.root, designs, seeds)
    prefix = 'pilot_' if args.phase == 'pilot' else ''
    grouped = grouped_rows(rows)
    contrasts = contrast_rows(rows)
    write_csv(args.root / f'{prefix}per_run.csv', rows)
    write_csv(args.root / f'{prefix}grouped.csv', grouped)
    write_csv(args.root / f'{prefix}paired_contrasts.csv', contrasts)
    write_csv(args.root / f'{prefix}convergence.csv', convergence)
    if args.phase == 'pilot':
        decision = pilot_decision(rows, convergence)
        atomic_json(args.root / 'pilot_decision.json', decision)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        mask_rows, gate_rows, action_rows, level_rows = final_interventions(
            args.root, checkpoints, device)
        write_csv(args.root / 'mask_generalization_per_run.csv', mask_rows)
        write_csv(args.root / 'v4_gate_per_run.csv', gate_rows)
        write_csv(args.root / 'v4_action_interventions.csv', action_rows)
        write_csv(args.root / 'v4_level_ablations.csv', level_rows)
        decision = final_decision(rows, mask_rows, gate_rows, action_rows, convergence)
        atomic_json(args.root / 'decision.json', decision)
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
