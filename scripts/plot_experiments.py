"""Make the experiment figures (1)-(4) from outputs/mem/master_metrics.csv."""

import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent / 'outputs' / 'mem'
COL = {'none': '#999999', 'short': '#d62728', 'long': '#1f77b4', 'both': '#2ca02c'}
DESIGNS = ['none', 'short', 'long', 'both']
MEM_ENVS = ['tmaze', 'occlusion', 'recall', 'distractor']


def fnum(x):
    try:
        v = float(x)
        return v
    except (ValueError, TypeError):
        return np.nan


def load():
    rows = []
    with open(ROOT / 'master_metrics.csv') as f:
        for r in rows_reader(f):
            rows.append(r)
    return rows


def rows_reader(f):
    for r in csv.DictReader(f):
        for k in ['tau_fast', 'tau_slow', 'val_mse', 'acc_z', 'acc_fast', 'acc_slow',
                  'usage_matched', 'chance', 'gap', 'reveal', 'length']:
            r[k] = fnum(r.get(k))
        r['seed'] = int(fnum(r.get('seed')))
        r['learnable'] = int(fnum(r.get('learnable')))
        yield r


def fig_seeds(rows):
    # main matrix: default suffix, fixed alpha, tau=(3,25); aggregate over seeds
    sel = [r for r in rows if r['suffix'] == '' and r['learnable'] == 0
           and round(r['tau_slow']) == 25 and round(r['tau_fast']) == 3]
    envs = [e for e in MEM_ENVS + ['tworoom'] if any(r['env'] == e for r in sel)]
    agg = defaultdict(lambda: defaultdict(list))   # env -> design -> [vals]
    aggm = defaultdict(lambda: defaultdict(list))
    seeds = sorted({r['seed'] for r in sel})
    for r in sel:
        agg[r['env']][r['design']].append(r['usage_matched'])
        aggm[r['env']][r['design']].append(r['val_mse'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4.6))
    x = np.arange(len(envs)); w = 0.2
    for i, d in enumerate(DESIGNS):
        mu = [np.nanmean(agg[e][d]) if agg[e][d] else np.nan for e in envs]
        sd = [np.nanstd(agg[e][d]) if len(agg[e][d]) > 1 else 0 for e in envs]
        ax1.bar(x + (i - 1.5) * w, mu, w, yerr=sd, capsize=3, label=d, color=COL[d])
        mum = [np.nanmean(aggm[e][d]) if aggm[e][d] else np.nan for e in envs]
        sdm = [np.nanstd(aggm[e][d]) if len(aggm[e][d]) > 1 else 0 for e in envs]
        ax2.bar(x + (i - 1.5) * w, mum, w, yerr=sdm, capsize=3, label=d, color=COL[d])
    for e_i, e in enumerate(envs):
        ch = next((r['chance'] for r in sel if r['env'] == e and not np.isnan(r['chance'])), np.nan)
        if not np.isnan(ch):
            ax1.hlines(ch, e_i - 2 * w, e_i + 2 * w, color='k', ls=':', lw=1)
    for ax, t, yl in [(ax1, 'Decision uses memory (matched probe)', 'cue acc from prediction'),
                      (ax2, 'World-model prediction error', 'val next-latent MSE')]:
        ax.set_xticks(x); ax.set_xticklabels(envs); ax.set_title(t); ax.set_ylabel(yl)
        ax.legend(title='design', ncol=4, fontsize=8)
    ax1.set_ylim(0, 1.02)
    fig.suptitle(f"(1) Across seeds {seeds} (mean +/- std). tworoom = Markovian control "
                 f"(memory gives no advantage)", y=1.02)
    fig.tight_layout(); fig.savefig(ROOT / 'exp1_seeds.png', dpi=110, bbox_inches='tight')
    print("saved exp1_seeds.png  seeds:", seeds)


def fig_tau_slow(rows):
    sel = sorted([r for r in rows if r['suffix'].startswith('tslow')], key=lambda r: r['tau_slow'])
    if not sel:
        print("no tau_slow sweep runs"); return
    gap = sel[0]['gap']
    ts = [r['tau_slow'] for r in sel]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ts, [r['usage_matched'] for r in sel], 'o-', color='#2ca02c', label='usage (cue from prediction)')
    ax.plot(ts, [r['acc_slow'] for r in sel], 's--', color='#1f77b4', label='availability (cue in m_slow)')
    ax.axhline(sel[0]['chance'], ls=':', c='gray', label='chance')
    ax.axvline(gap, ls='--', c='k', alpha=.6); ax.text(gap + .5, 0.05, f'gap Delta={gap:.0f}', rotation=90, fontsize=9)
    ax.set_xlabel('slow-bank horizon tau_slow (steps)'); ax.set_ylabel('accuracy'); ax.set_ylim(0, 1.02)
    ax.set_title('(2) tmaze: usage tracks the gap -- decision recovers the cue once tau_slow >= Delta')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT / 'exp2_tau_slow_sweep.png', dpi=110, bbox_inches='tight')
    print("saved exp2_tau_slow_sweep.png")


def fig_learnable(rows):
    sel = [r for r in rows if r['suffix'] == 'learnable']
    if not sel:
        print("no learnable runs"); return
    envs = [e for e in MEM_ENVS if any(r['env'] == e for r in sel)]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(envs)); w = 0.35
    tf = [next(r['tau_fast'] for r in sel if r['env'] == e) for e in envs]
    tsl = [next(r['tau_slow'] for r in sel if r['env'] == e) for e in envs]
    gaps = [next(r['gap'] for r in sel if r['env'] == e) for e in envs]
    ax.bar(x - w / 2, tf, w, label='learned tau_fast', color='#d62728')
    ax.bar(x + w / 2, tsl, w, label='learned tau_slow', color='#1f77b4')
    ax.plot(x, gaps, 'k*', ms=14, label='task gap Delta')
    for xi, g in zip(x, gaps):
        ax.text(xi, g + 0.5, f'{g:.0f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(envs); ax.set_ylabel('horizon tau (steps)')
    ax.set_title('(3) Discovered memory horizons (alpha learned): tau adapts toward the task gap')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT / 'exp3_learnable_tau.png', dpi=110, bbox_inches='tight')
    print("saved exp3_learnable_tau.png")


def fig_gap(rows):
    sel = [r for r in rows if r['suffix'].startswith('gap')]
    if not sel:
        print("no gap sweep runs"); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))
    for d, c in [('none', '#999999'), ('both', '#2ca02c')]:
        pts = sorted([r for r in sel if r['design'] == d], key=lambda r: r['gap'])
        if not pts: continue
        g = [r['gap'] for r in pts]
        ax1.plot(g, [r['usage_matched'] for r in pts], 'o-', color=c, label=d)
        ax2.plot(g, [r['val_mse'] for r in pts], 'o-', color=c, label=d)
    ch = next((r['chance'] for r in sel if not np.isnan(r['chance'])), 0.5)
    ax1.axhline(ch, ls=':', c='gray', label='chance')
    tau_slow = next((r['tau_slow'] for r in sel if r['design'] == 'both'), 25)
    h = 3
    for ax in (ax1, ax2):
        ax.axvline(h, ls='--', c='#999999', alpha=.7); ax.text(h + .3, ax.get_ylim()[0], f'window h={h}', rotation=90, fontsize=8)
        ax.axvline(tau_slow, ls='--', c='#1f77b4', alpha=.7); ax.text(tau_slow + .3, ax.get_ylim()[0], f'tau_slow={tau_slow:.0f}', rotation=90, fontsize=8, color='#1f77b4')
        ax.set_xlabel('cue->decision gap Delta (steps)'); ax.legend(fontsize=8)
    ax1.set_ylabel('usage (cue from prediction)'); ax1.set_ylim(0, 1.02)
    ax1.set_title('(4) Memory vs finite-window baseline as the gap grows')
    ax2.set_ylabel('val next-latent MSE')
    fig.suptitle('(4) tmaze gap sweep: baseline (none, window h=3) cliffs once Delta>h; '
                 'memory (both) holds until Delta approaches tau_slow', y=1.02)
    fig.tight_layout(); fig.savefig(ROOT / 'exp4_gap_sweep.png', dpi=110, bbox_inches='tight')
    print("saved exp4_gap_sweep.png")


if __name__ == '__main__':
    rows = load()
    fig_seeds(rows)
    fig_tau_slow(rows)
    fig_learnable(rows)
    fig_gap(rows)
