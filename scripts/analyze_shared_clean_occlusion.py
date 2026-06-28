#!/usr/bin/env python3
"""Validate and aggregate the shared-encoder clean-target occlusion factorial."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


OCC_TO_CLEAN = {
    'dmc:reacher.hard.occ': 'dmc:reacher.hard',
    'dmc:ball_in_cup.catch.occ': 'dmc:ball_in_cup.catch',
    'dmc:finger.spin.occ': 'dmc:finger.spin',
    'dmc:cheetah.run.occ': 'dmc:cheetah.run',
    'ogbench:cube-single.occ': 'ogbench:cube-single',
}
REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE_KEYS = [
    'clean_mse_pre', 'clean_mse_blackout_transition', 'clean_mse_deep_blackout',
    'clean_mse_first_post', 'clean_mse_recovery', 'clean_mse_late_post',
    'clean_mse_all',
]
BASELINE_KEYS = [
    *(f'constant_mse_{key.removeprefix("clean_mse_")}' for key in PHASE_KEYS),
    *(f'persistence_mse_{key.removeprefix("clean_mse_")}' for key in PHASE_KEYS),
    *(f'last_visible_mse_{key.removeprefix("clean_mse_")}' for key in PHASE_KEYS),
]
CLEAN_INPUT_KEYS = [f'clean_input_mse_{key.removeprefix("clean_mse_")}' for key in PHASE_KEYS]
R2_KEYS = [f'r2_{key.removeprefix("clean_mse_")}' for key in PHASE_KEYS]


def mean(xs):
    return sum(xs) / len(xs)


def popstd(xs):
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def samplestd(xs):
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def resolve_repo_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_env_name(env):
    return ''.join(c if c.isalnum() else '_' for c in env).strip('_')


def state_fingerprint(state, prefix):
    digest = hashlib.sha256()
    keys = sorted(k for k in state if k.startswith(prefix))
    if not keys:
        raise ValueError(f'no state keys with prefix {prefix!r}')
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes(order='C'))
    return digest.hexdigest()


def exact_bootstrap_ci(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    means = np.fromiter(
        (values[list(idx)].mean() for idx in itertools.product(range(n), repeat=n)),
        dtype=np.float64, count=n ** n)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_csv(path, rows, fieldnames):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


def semantically_equal(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            semantically_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(map(lambda pair: semantically_equal(*pair), zip(left, right)))
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return left == right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('outputs/shared_clean_occlusion'))
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--designs', nargs='+', default=['none', 'multi', 'gru', 'ssm', 'smt'])
    ap.add_argument('--num-episodes', type=int, default=600)
    ap.add_argument('--val-episodes', type=int, default=150)
    ap.add_argument('--length', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--feature-dim', type=int, default=128)
    args = ap.parse_args()
    root = args.root.resolve()

    expected = set(itertools.product(OCC_TO_CLEAN, args.designs, args.seeds))
    rows = []
    seen = set()
    encoder_fingerprints = defaultdict(set)
    validated_manifests = {}
    for metrics_path in sorted(root.glob('*/metrics.json')):
        run_dir = metrics_path.parent
        model_path = run_dir / 'model.pt'
        if not model_path.is_file():
            raise SystemExit(f'missing checkpoint beside {metrics_path}')
        metrics = json.loads(metrics_path.read_text())
        ck = torch.load(model_path, map_location='cpu', weights_only=False)
        if not semantically_equal(metrics, ck.get('final_metrics')):
            raise SystemExit(f'{run_dir.name}: metrics.json differs from checkpoint final_metrics')
        cfg = ck.get('args', {})
        key = (cfg.get('env_id'), cfg.get('memory_mode'), cfg.get('seed'))
        if key not in expected:
            raise SystemExit(f'unexpected run/config under {root}: {run_dir.name} -> {key}')
        if key in seen:
            raise SystemExit(f'duplicate factorial cell: {key}')
        seen.add(key)
        env, design, seed = key
        checks = {
            'target_env_id': OCC_TO_CLEAN[env], 'prototype_seed': 0,
            'freeze_encoder': False, 'num_episodes': args.num_episodes,
            'val_episodes': args.val_episodes, 'length': args.length,
            'epochs': args.epochs, 'mask_occluded_target_loss': True,
            'encoder_type': 'precomputed', 'embed_dim': args.feature_dim,
            'img_size': 64, 'patch_size': 8,
            'encoder_layers': 6, 'encoder_heads': 4,
            'predictor_layers': 4, 'predictor_heads': 8,
            'history_len': 3, 'dropout': 0.1,
            'sigreg_lambda': 0.1, 'sigreg_projections': 512,
            'tau_fast': 3.0, 'tau_slow': 25.0, 'fixed_alpha': True,
            'smt_router': 'sigmoid', 'lr': 3e-4, 'weight_decay': 1e-5,
            'batch_size': 64, 'no_amp': False,
        }
        for name, want in checks.items():
            if cfg.get(name) != want:
                raise SystemExit(f'{run_dir.name}: {name}={cfg.get(name)!r}, expected {want!r}')
        clean_env = OCC_TO_CLEAN[env]
        safe = safe_env_name(clean_env)
        feature_root = root / f'dino_features_d{args.feature_dim}'
        expected_manifest = (feature_root / f'{safe}_manifest.json').resolve()
        expected_train = (feature_root / f'{safe}_train.npz').resolve()
        expected_val = (feature_root / f'{safe}_val.npz').resolve()
        manifest_path = resolve_repo_path(cfg.get('feature_manifest', ''))
        train_feature_path = resolve_repo_path(cfg.get('train_feature_cache', ''))
        val_feature_path = resolve_repo_path(cfg.get('val_feature_cache', ''))
        if (manifest_path != expected_manifest or train_feature_path != expected_train or
                val_feature_path != expected_val):
            raise SystemExit(f'{run_dir.name}: unexpected DINO feature artifact path')
        if not (manifest_path.is_file() and train_feature_path.is_file() and val_feature_path.is_file()):
            raise SystemExit(f'{run_dir.name}: missing DINO feature artifact')
        manifest_hash = file_sha256(manifest_path)
        if cfg.get('feature_manifest_sha256') != manifest_hash:
            raise SystemExit(f'{run_dir.name}: feature manifest hash mismatch')
        if env not in validated_manifests:
            manifest = json.loads(manifest_path.read_text())
            config = manifest.get('config', {})
            required_config = {
                'clean_env': clean_env, 'occ_env': env,
                'train_episodes': args.num_episodes, 'val_episodes': args.val_episodes,
                'length': args.length, 'feature_dim': args.feature_dim,
                'pixel_schema_version': 3, 'feature_schema_version': 1,
                'prototype_seed': 0, 'train_rollout_seed': 0,
                'val_rollout_seed': 7777,
            }
            for name, want in required_config.items():
                if config.get(name) != want:
                    raise SystemExit(f'{manifest_path}: {name}={config.get(name)!r}, expected {want!r}')
            raw_quality = manifest.get('raw_quality_clean_train_visible', {})
            pca = manifest.get('pca', {})
            projected_val = manifest.get('projected_quality_clean_val_valid', {})
            baselines = manifest.get('clean_val_baselines', {})
            if raw_quality.get('mean_channel_variance', 0) < 1e-4:
                raise SystemExit(f'{manifest_path}: raw DINO variance quality gate failed')
            if raw_quality.get('covariance_effective_rank', 0) < 2:
                raise SystemExit(f'{manifest_path}: raw DINO rank quality gate failed')
            if pca.get('retained_explained_variance_ratio', 0) < .95:
                raise SystemExit(f'{manifest_path}: PCA retention quality gate failed')
            if projected_val.get('covariance_effective_rank', 0) < 1:
                raise SystemExit(f'{manifest_path}: projected validation features collapsed')
            for name in ('constant_train_mean_mse', 'immediate_persistence_mse',
                         'last_visible_hold_mse'):
                value = baselines.get(name)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise SystemExit(f'{manifest_path}: invalid baseline {name}={value!r}')
            if manifest.get('artifact_files', {}).get('train') != expected_train.name:
                raise SystemExit(f'{manifest_path}: wrong train artifact name')
            if manifest.get('artifact_files', {}).get('val') != expected_val.name:
                raise SystemExit(f'{manifest_path}: wrong val artifact name')
            validated_manifests[env] = manifest_hash
        elif validated_manifests[env] != manifest_hash:
            raise SystemExit(f'{env}: manifest changed during analysis')
        if (metrics.get('target_env') != clean_env or
                not metrics.get('external_features_fixed') or metrics.get('encoder_frozen') or
                not metrics.get('masked_clean_blackout_loss')):
            raise SystemExit(f'{run_dir.name}: metrics do not attest paired/frozen evaluation')
        if metrics.get('dataset_schema_version') != 3 or metrics.get('feature_schema_version') != 1:
            raise SystemExit(f'{run_dir.name}: dataset/feature schema mismatch')
        if metrics.get('feature_manifest_sha256') != manifest_hash:
            raise SystemExit(f'{run_dir.name}: metrics feature manifest hash mismatch')
        if not isinstance(metrics.get('trainable_parameters'), int) or metrics['trainable_parameters'] <= 0:
            raise SystemExit(f'{run_dir.name}: invalid trainable parameter count')
        if any(k.startswith('encoder.') for k in ck['model_state_dict']):
            raise SystemExit(f'{run_dir.name}: precomputed-feature checkpoint unexpectedly stores an encoder')
        encoder_hash = manifest_hash
        encoder_fingerprints[env].add(encoder_hash)
        numeric = ['val_pred_loss', *PHASE_KEYS,
                   *(f'{k}_ablated' for k in PHASE_KEYS), *BASELINE_KEYS,
                   *CLEAN_INPUT_KEYS]
        for name in numeric:
            value = metrics.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise SystemExit(f'{run_dir.name}: invalid {name}={value!r}')
        target_times = metrics.get('prediction_target_times')
        per_t = metrics.get('clean_mse_by_target_t')
        per_t_ab = metrics.get('clean_mse_by_target_t_ablated')
        per_t_constant = metrics.get('constant_mse_by_target_t')
        per_t_persistence = metrics.get('persistence_mse_by_target_t')
        per_t_last_visible = metrics.get('last_visible_mse_by_target_t')
        per_t_clean_input = metrics.get('clean_input_mse_by_target_t')
        expected_windows = args.length - int(cfg.get('history_len', -1))
        if not (isinstance(target_times, list) and isinstance(per_t, list) and
                isinstance(per_t_ab, list) and isinstance(per_t_constant, list) and
                isinstance(per_t_persistence, list) and isinstance(per_t_last_visible, list) and
                isinstance(per_t_clean_input, list) and
                len(target_times) == len(per_t) == len(per_t_ab) ==
                len(per_t_constant) == len(per_t_persistence) ==
                len(per_t_last_visible) == len(per_t_clean_input) == expected_windows):
            raise SystemExit(f'{run_dir.name}: invalid per-timestep error arrays')
        if any(not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in per_t + per_t_ab + per_t_constant + per_t_persistence +
               per_t_last_visible + per_t_clean_input):
            raise SystemExit(f'{run_dir.name}: non-finite per-timestep errors')
        row = {'run': run_dir.name, 'env': env, 'target_env': OCC_TO_CLEAN[env],
               'design': design, 'seed': seed,
               'feature_manifest': cfg['feature_manifest'],
               'train_feature_cache': cfg['train_feature_cache'],
               'val_feature_cache': cfg['val_feature_cache'],
               'encoder_fingerprint': encoder_hash,
               'feature_manifest_sha256': manifest_hash,
               'trainable_parameters': metrics.get('trainable_parameters'),
               'prediction_target_times_json': json.dumps(target_times),
               'clean_mse_by_target_t_json': json.dumps(per_t),
               'clean_mse_by_target_t_ablated_json': json.dumps(per_t_ab),
               'constant_mse_by_target_t_json': json.dumps(per_t_constant),
               'persistence_mse_by_target_t_json': json.dumps(per_t_persistence),
               'last_visible_mse_by_target_t_json': json.dumps(per_t_last_visible),
               'clean_input_mse_by_target_t_json': json.dumps(per_t_clean_input)}
        row.update({k: metrics[k] for k in numeric})
        for phase_key, r2_key in zip(PHASE_KEYS, R2_KEYS):
            suffix = phase_key.removeprefix('clean_mse_')
            constant = metrics[f'constant_mse_{suffix}']
            if constant <= 0:
                raise SystemExit(f'{run_dir.name}: non-positive constant baseline for {suffix}')
            row[r2_key] = 1.0 - metrics[phase_key] / constant
        rows.append(row)

    missing = expected - seen
    extra = seen - expected
    if missing or extra:
        sample = sorted(missing)[:8]
        raise SystemExit(
            f'incomplete factorial: found {len(seen)}/{len(expected)}; '
            f'missing sample={sample}; extra={sorted(extra)[:8]}')
    for env, hashes in encoder_fingerprints.items():
        if len(hashes) != 1:
            raise SystemExit(f'{env}: encoder differs across designs/seeds: {len(hashes)} fingerprints')
    for env in OCC_TO_CLEAN:
        env_rows = [r for r in rows if r['env'] == env]
        for key in BASELINE_KEYS:
            values = [r[key] for r in env_rows]
            if max(values) - min(values) > 1e-6:
                raise SystemExit(f'{env}: shared {key} differs across runs by {max(values)-min(values)}')

    rows.sort(key=lambda r: (r['env'], r['design'], r['seed']))
    per_run_fields = list(rows[0])
    write_csv(root / 'per_run.csv', rows, per_run_fields)

    grouped = []
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row['env'], row['design'])].append(row)
    metric_keys = ['val_pred_loss', *PHASE_KEYS, *(f'{k}_ablated' for k in PHASE_KEYS),
                   *BASELINE_KEYS, *CLEAN_INPUT_KEYS, *R2_KEYS]
    for (env, design), group in sorted(by_group.items()):
        out = {'env': env, 'design': design, 'n_seeds': len(group)}
        for key in metric_keys:
            values = [r[key] for r in group]
            out[f'{key}_mean'] = mean(values)
            out[f'{key}_std'] = popstd(values)
        grouped.append(out)
    grouped_fields = list(grouped[0])
    write_csv(root / 'grouped.csv', grouped, grouped_fields)

    lookup = {(r['env'], r['design'], r['seed']): r for r in rows}
    deltas = []
    for env in OCC_TO_CLEAN:
        for design in args.designs:
            if design == 'none':
                continue
            for seed in args.seeds:
                base = lookup[(env, 'none', seed)]
                row = lookup[(env, design, seed)]
                out = {'env': env, 'design': design, 'reference': 'none', 'seed': seed}
                for key in PHASE_KEYS:
                    if base[key] <= 0:
                        raise SystemExit(f'{env} seed {seed}: non-positive reference {key}')
                    out[f'delta_{key}'] = row[key] - base[key]
                    out[f'rel_{key}'] = row[key] / base[key] - 1.0
                    out[f'ablation_delta_{key}'] = row[f'{key}_ablated'] - row[key]
                deltas.append(out)
    delta_fields = list(deltas[0])
    write_csv(root / 'paired_deltas.csv', deltas, delta_fields)

    paired_grouped = []
    delta_metric_keys = [
        *(f'delta_{k}' for k in PHASE_KEYS),
        *(f'rel_{k}' for k in PHASE_KEYS),
        *(f'ablation_delta_{k}' for k in PHASE_KEYS),
    ]
    by_pair = defaultdict(list)
    for row in deltas:
        by_pair[(row['env'], row['design'])].append(row)
    for (env, design), group in sorted(by_pair.items()):
        out = {'env': env, 'design': design, 'reference': 'none', 'n_pairs': len(group)}
        for key in delta_metric_keys:
            values = [r[key] for r in group]
            lo, hi = exact_bootstrap_ci(values)
            out[f'{key}_mean'] = mean(values)
            out[f'{key}_sample_std'] = samplestd(values)
            out[f'{key}_bootstrap_lo'] = lo
            out[f'{key}_bootstrap_hi'] = hi
        paired_grouped.append(out)
    write_csv(root / 'paired_grouped.csv', paired_grouped, list(paired_grouped[0]))

    print(f'validated {len(rows)} runs; wrote per_run.csv, grouped.csv, paired_deltas.csv, paired_grouped.csv')
    print(f"{'env':<31} {'design':<6} {'deep_blk':>10} {'first_post':>11} {'recovery':>10}")
    for row in grouped:
        print(f"{row['env']:<31} {row['design']:<6} "
              f"{row['clean_mse_deep_blackout_mean']:>10.4f} "
              f"{row['clean_mse_first_post_mean']:>11.4f} "
              f"{row['clean_mse_recovery_mean']:>10.4f}")


if __name__ == '__main__':
    main()
