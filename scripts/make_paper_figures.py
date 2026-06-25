"""Publication-quality figures for ICLR.md, from the lewm-memory-4ens runs (3 seeds).

Fig A (fig_dissociation): the headline short-vs-long dissociation -- cue-decoding
   availability over time for z / m_fast / m_slow, T-Maze (long) vs Occlusion (short),
   mean +/- std over seeds.
Fig B (fig_usage_bar): decision-usage (cue decodable from the model's prediction) across
   the 4 envs x 4 designs, mean +/- std with chance lines.
"""
import sys, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from lewm.models.memory_model import MemoryLeWorldModel
from lewm.data import generate_eval_batch
from lewm.eval.memory_probe import extract_timewise, probe_cue_over_time

ROOT = Path(__file__).parent.parent / 'outputs' / '4ens'
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [0, 1, 2]
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
                     'legend.fontsize': 8.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'figure.dpi': 120, 'savefig.dpi': 150, 'axes.grid': True,
                     'grid.alpha': 0.25, 'axes.spines.top': False, 'axes.spines.right': False})
CST = {'z': '#777777', 'm_fast': '#d62728', 'm_slow': '#1f77b4'}
LBL = {'z': r'$z$  (memoryless encoder)', 'm_fast': r'$m^{\mathrm{fast}}$ ($\tau{=}3$)',
       'm_slow': r'$m^{\mathrm{slow}}$ ($\tau{=}25$)'}


def load(env, design, seed):
    ck = torch.load(ROOT / f'lewm-{env}-{design}-s{seed}' / 'model.pt', map_location=dev, weights_only=False)
    a = ck['args']
    m = MemoryLeWorldModel(img_size=a['img_size'], patch_size=a['patch_size'], embed_dim=a['embed_dim'],
        action_dim=2, encoder_layers=a['encoder_layers'], encoder_heads=a['encoder_heads'],
        predictor_layers=a['predictor_layers'], predictor_heads=a['predictor_heads'], history_len=a['history_len'],
        dropout=a['dropout'], sigreg_lambda=a['sigreg_lambda'], sigreg_projections=a['sigreg_projections'],
        memory_mode=a['memory_mode'], tau_fast=a['tau_fast'], tau_slow=a['tau_slow'],
        learnable_alpha=not a.get('fixed_alpha', True)).to(dev)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m


@torch.no_grad()
def dissociation():
    panels = [('tmaze', 'T-Maze (long-term, gap $\\Delta{\\approx}21$)'),
              ('occlusion', 'Occlusion (short-term, gap $\\Delta{\\approx}5$)')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, (env, title) in zip(axes, panels):
        b = generate_eval_batch(env, 320, length=32, seed=2024)
        cue = b['cue'].numpy(); nclass = b['n_cue_classes']
        cue_end = int(b['cue_end'].float().mean()); reveal = int(b['reveal'].float().mean())
        curves = {s: [] for s in ['z', 'm_fast', 'm_slow']}
        for seed in SEEDS:
            feats = extract_timewise(load(env, 'both', seed), b['obs'], dev)
            for s in curves:
                curves[s].append(probe_cue_over_time(feats[s], cue, nclass, seed=seed))
        T = len(curves['z'][0]); t = np.arange(T)
        for s in ['z', 'm_fast', 'm_slow']:
            arr = np.stack(curves[s]); mu = arr.mean(0); sd = arr.std(0)
            ax.plot(t, mu, color=CST[s], lw=2.0, label=LBL[s])
            ax.fill_between(t, mu - sd, mu + sd, color=CST[s], alpha=0.15)
        ax.axhline(1.0 / nclass, ls=':', c='k', lw=1, alpha=.6)
        ax.axvline(cue_end, ls='--', c='k', lw=1, alpha=.5)
        ax.axvline(reveal, ls='--', c='green', lw=1, alpha=.6)
        ax.text(cue_end + .3, .03, 'cue off', rotation=90, fontsize=8, alpha=.7)
        ax.text(reveal + .3, .03, 'decision', rotation=90, fontsize=8, color='green', alpha=.8)
        ax.set_title(title); ax.set_xlabel('time step'); ax.set_ylim(0, 1.03)
    axes[0].set_ylabel('cue-decoding accuracy')
    axes[0].legend(loc='lower left', framealpha=.9)
    fig.tight_layout()
    fig.savefig(ROOT / 'fig_dissociation.png', bbox_inches='tight')
    print('saved fig_dissociation.png')


def usage_bar():
    rows = list(csv.DictReader(open(ROOT / 'master_metrics.csv')))
    for r in rows:
        for k in ['usage_matched', 'chance']:
            try: r[k] = float(r[k])
            except: r[k] = np.nan
    envs = ['tmaze', 'distractor', 'recall', 'occlusion']
    des = ['none', 'short', 'long', 'both']
    col = {'none': '#999999', 'short': '#d62728', 'long': '#1f77b4', 'both': '#2ca02c'}
    lbl = {'none': 'none (vanilla LeWM)', 'short': 'short', 'long': 'long', 'both': 'both'}
    def ms(e, d):
        v = [r['usage_matched'] for r in rows if r['env'] == e and r['design'] == d and not np.isnan(r['usage_matched'])]
        return (np.mean(v), np.std(v)) if v else (np.nan, 0)
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(envs)); w = 0.2
    for i, d in enumerate(des):
        mu = [ms(e, d)[0] for e in envs]; sd = [ms(e, d)[1] for e in envs]
        ax.bar(x + (i - 1.5) * w, mu, w, yerr=sd, capsize=3, label=lbl[d], color=col[d])
    for xi, e in zip(x, envs):
        ch = next((r['chance'] for r in rows if r['env'] == e and not np.isnan(r['chance'])), np.nan)
        ax.hlines(ch, xi - 2.2 * w, xi + 2.2 * w, color='k', ls=':', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(['T-Maze\n(Δ21)', 'Distractor\n(Δ23)', 'Recall\n(Δ15)', 'Occlusion\n(Δ5)'])
    ax.set_ylabel('cue acc. from prediction'); ax.set_ylim(0, 1.0)
    ax.set_title('Does the decision use memory? (dotted = chance; 3 seeds, mean±std)')
    ax.legend(ncol=4, loc='upper center', framealpha=.9)
    fig.tight_layout(); fig.savefig(ROOT / 'fig_usage_bar.png', bbox_inches='tight')
    print('saved fig_usage_bar.png')


if __name__ == '__main__':
    dissociation()
    usage_bar()
