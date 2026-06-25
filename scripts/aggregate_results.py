"""Aggregate per-run metrics.json into an env x design summary table + figure."""

import json
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DESIGNS = ['none', 'short', 'long', 'both']
KEY = 'cue_acc_from_prediction'   # headline "decision uses memory" metric
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'mem')


def load():
    runs = {}
    for fp in glob.glob(os.path.join(ROOT, '*', 'metrics.json')):
        try:
            with open(fp) as f:
                m = json.load(f)
            runs[(m['env'], m['design'])] = m
        except Exception as e:
            print('skip', fp, e)
    return runs


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "


def main():
    runs = load()
    if not runs:
        print("no metrics found in", ROOT)
        return
    envs = sorted({e for (e, _) in runs})

    print("\n" + "=" * 78)
    print(f"DECISION USES MEMORY  ({KEY}; chance shown in parens)")
    print("=" * 78)
    print(f"{'env':<12} " + " ".join(f"{d:>8}" for d in DESIGNS) + "   chance")
    for e in envs:
        row = []
        chance = None
        for d in DESIGNS:
            m = runs.get((e, d))
            row.append(fmt(m.get(KEY)) if m else "  -  ")
            if m and 'chance' in m:
                chance = m['chance']
        print(f"{e:<12} " + " ".join(f"{c:>8}" for c in row) + f"   {fmt(chance) if chance else '-'}")

    # availability table: cue retained in slow vs fast vs z at the decision step (use 'both')
    print("\n" + "=" * 78)
    print("AVAILABILITY at decision step (design=both): acc of decoding cue from stream")
    print("=" * 78)
    print(f"{'env':<12} {'z':>8} {'m_fast':>8} {'m_slow':>8}   long_adv")
    for e in envs:
        m = runs.get((e, 'both'))
        if not m:
            continue
        print(f"{e:<12} {fmt(m.get('acc_z_delay')):>8} {fmt(m.get('acc_m_fast_delay')):>8} "
              f"{fmt(m.get('acc_m_slow_delay')):>8}   {fmt(m.get('long_mem_advantage'))}")

    # grouped bar chart of the headline metric
    fig, ax = plt.subplots(figsize=(1.6 * len(envs) + 3, 4.2))
    x = np.arange(len(envs))
    w = 0.2
    colors = {'none': '#999999', 'short': '#d62728', 'long': '#1f77b4', 'both': '#2ca02c'}
    for i, d in enumerate(DESIGNS):
        vals = [runs.get((e, d), {}).get(KEY, np.nan) for e in envs]
        ax.bar(x + (i - 1.5) * w, vals, w, label=d, color=colors[d])
    chances = [runs.get((e, 'both'), {}).get('chance', np.nan) for e in envs]
    for xi, c in zip(x, chances):
        ax.hlines(c, xi - 2 * w, xi + 2 * w, color='k', ls=':', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(envs)
    ax.set_ylabel(KEY); ax.set_ylim(0, 1.02)
    ax.set_title('Decision uses memory: cue decodable from the predicted reveal-latent\n'
                 '(dotted = chance; higher = the model recruited memory for the decision)')
    ax.legend(title='design', ncol=4, fontsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, 'summary_decision.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f"\nsaved {out}")

    # csv
    csv = os.path.join(ROOT, 'summary.csv')
    keys = ['env', 'design', KEY, 'cue_acc_from_true_latent', 'chance',
            'acc_z_delay', 'acc_m_fast_delay', 'acc_m_slow_delay',
            'long_mem_advantage', 'short_mem_advantage', 'infl_fast', 'infl_slow',
            'tau_fast', 'tau_slow', 'val/pred_loss']
    with open(csv, 'w') as f:
        f.write(','.join(keys) + '\n')
        for (e, d), m in sorted(runs.items()):
            f.write(','.join(str(m.get(k, '')) for k in keys) + '\n')
    print(f"saved {csv}")


if __name__ == '__main__':
    main()
