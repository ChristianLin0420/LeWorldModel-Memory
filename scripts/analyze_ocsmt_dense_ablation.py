#!/usr/bin/env python3
"""Strict aggregation for the 3x2x2 dense OC-SMT Occlusion factorial."""

import csv
import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch


SEEDS = (0, 1, 2)
MS = (6, 12, 28)
TAU_MAXES = (64.0, 256.0)
GATE_MODES = ('stochastic', 'deterministic')
METRICS = ('usage_matched', 'val_mse', 'mean_active_count',
           'mean_gate_mass', 'expected_open_count')


def read_csv(path):
    if not path.is_file():
        raise SystemExit(f'missing prerequisite: {path}')
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise SystemExit(f'non-finite {label}: {value}')
    return value


def mean(xs):
    return sum(xs) / len(xs)


def popstd(xs):
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))


def write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'outputs/ocsmt_dense_ablation').resolve()
    master_rows = read_csv(root / 'master_metrics.csv')
    gate_rows = read_csv(root / 'ocsmt_gate_metrics.csv')
    master = {r['run']: r for r in master_rows}
    gates = {r['run']: r for r in gate_rows}
    if len(master) != len(master_rows) or len(gates) != len(gate_rows):
        raise SystemExit('duplicate run key in prerequisite CSV')

    expected = set(itertools.product(SEEDS, MS, TAU_MAXES, GATE_MODES))
    seen = set(); rows = []
    checkpoints = sorted(root.glob('*/model.pt'))
    for checkpoint_path in checkpoints:
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        cfg = ck['args']; run = checkpoint_path.parent.name
        key = (int(cfg['seed']), int(cfg['oc_num']), float(cfg['oc_tau_max']), cfg['oc_gate_mode'])
        if key not in expected:
            raise SystemExit(f'unexpected dense-factorial config {run}: {key}')
        if key in seen:
            raise SystemExit(f'duplicate dense-factorial config: {key}')
        seen.add(key)
        checks = {
            'env': 'occlusion', 'memory_mode': 'ocsmt', 'l0_lambda': 0.0,
            'oc_tau_min': 1.5, 'gate_lr_mult': 8.0, 'epochs': 30,
            'num_episodes': 4000, 'length': 32,
        }
        for name, want in checks.items():
            if cfg.get(name) != want:
                raise SystemExit(f'{run}: {name}={cfg.get(name)!r}, expected {want!r}')
        if run not in master or run not in gates:
            raise SystemExit(f'{run}: missing canonical usage or gate row')
        mr, gr = master[run], gates[run]
        if mr.get('design') != 'ocsmt' or gr.get('design') != 'ocsmt':
            raise SystemExit(f'{run}: wrong design in prerequisite CSV')
        row = {'run': run, 'seed': key[0], 'M': key[1], 'tau_min': 1.5,
               'tau_max': key[2], 'gate_mode': key[3]}
        row.update({
            'usage_matched': finite(mr['usage_matched'], f'{run} usage'),
            'val_mse': finite(mr['val_mse'], f'{run} val_mse'),
            'mean_active_count': finite(gr['mean_active_count'], f'{run} active'),
            'mean_gate_mass': finite(gr['mean_gate_mass'], f'{run} mass'),
            'expected_open_count': finite(gr['expected_open_count'], f'{run} expected'),
        })
        rows.append(row)
    if seen != expected or len(checkpoints) != len(expected):
        raise SystemExit(
            f'incomplete dense factorial: checkpoints={len(checkpoints)}, '
            f'cells={len(seen)}/{len(expected)}, missing={sorted(expected-seen)[:8]}')
    if set(master) != {r['run'] for r in rows} or set(gates) != {r['run'] for r in rows}:
        raise SystemExit('prerequisite CSV contains stale or extra runs')
    rows.sort(key=lambda r: (r['M'], r['tau_max'], r['gate_mode'], r['seed']))
    write(root / 'dense_factorial_per_run.csv', rows)

    grouped = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row['M'], row['tau_max'], row['gate_mode'])].append(row)
    for (m, tau, mode), group in sorted(groups.items()):
        out = {'M': m, 'tau_min': 1.5, 'tau_max': tau, 'gate_mode': mode,
               'n_seeds': len(group), 'seed_list': '0;1;2'}
        for metric in METRICS:
            values = [r[metric] for r in group]
            out[f'{metric}_mean'] = mean(values)
            out[f'{metric}_std'] = popstd(values)
        grouped.append(out)
    write(root / 'dense_factorial_grouped.csv', grouped)

    lookup = {(r['seed'], r['M'], r['tau_max'], r['gate_mode']): r for r in rows}
    contrasts = []
    for seed in SEEDS:
        for tau in TAU_MAXES:
            for mode in GATE_MODES:
                for high, low in ((12, 6), (28, 12), (28, 6)):
                    contrasts.append((f'M{high}-M{low}', seed,
                                      lookup[(seed, high, tau, mode)],
                                      lookup[(seed, low, tau, mode)]))
        for m in MS:
            for mode in GATE_MODES:
                contrasts.append(('tau256-tau64', seed,
                                  lookup[(seed, m, 256.0, mode)],
                                  lookup[(seed, m, 64.0, mode)]))
            for tau in TAU_MAXES:
                contrasts.append(('deterministic-stochastic', seed,
                                  lookup[(seed, m, tau, 'deterministic')],
                                  lookup[(seed, m, tau, 'stochastic')]))
    paired = []
    for name, seed, high, low in contrasts:
        out = {'contrast': name, 'seed': seed, 'high_M': high['M'], 'low_M': low['M'],
               'high_tau_max': high['tau_max'], 'low_tau_max': low['tau_max'],
               'high_gate_mode': high['gate_mode'], 'low_gate_mode': low['gate_mode']}
        for metric in METRICS:
            out[f'delta_{metric}'] = high[metric] - low[metric]
        paired.append(out)
    write(root / 'dense_factorial_paired_contrasts.csv', paired)
    print(f'validated {len(rows)} runs; wrote dense factorial per-run/grouped/contrast CSVs')


if __name__ == '__main__':
    main()
