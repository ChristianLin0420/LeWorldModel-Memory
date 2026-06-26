"""Aggregate the reviewer-response experiments into tables + figures for ICLR.md.
Reads outputs/4ens/master_metrics.csv (E5 seeds, E3 multi, E2 gru) and
outputs/rev/master_metrics.csv (E3 single-tau, E4 grid, E2 long-context)."""
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F4 = os.path.join(ROOT, 'outputs', '4ens', 'master_metrics.csv')
FR = os.path.join(ROOT, 'outputs', 'rev', 'master_metrics.csv')
FIG = os.path.join(ROOT, 'docs', 'figures')


def load(fp):
    if not os.path.exists(fp):
        return []
    rows = list(csv.DictReader(open(fp)))
    for r in rows:
        for k in ['tau_fast', 'tau_slow', 'val_mse', 'acc_slow', 'usage_matched', 'gap', 'chance', 'length']:
            try: r[k] = float(r[k])
            except: r[k] = np.nan
        r['seed'] = int(float(r.get('seed', 0)))
    return rows


def ag(rows, pred, key):
    v = [r[key] for r in rows if pred(r) and not np.isnan(r[key])]
    return (np.mean(v), np.std(v), len(v)) if v else (np.nan, np.nan, 0)


def main():
    r4, rr = load(F4), load(FR)

    # ---- E5 + E3(multi) + E2(gru): headline matrix, all designs, all seeds ----
    if r4:
        envs = ['tmaze', 'distractor', 'recall', 'occlusion']
        designs = ['none', 'short', 'long', 'both', 'multi', 'gru']
        print("\n==== E5/E3/E2: usage_matched by design (4-env matrix, all seeds) ====")
        print(f"{'env':<11}" + "".join(f"{d:>13}" for d in designs))
        for e in envs:
            cells = [ag(r4, lambda r, e=e, d=d: r['env'] == e and r['design'] == d and r['suffix'] == '', 'usage_matched') for d in designs]
            print(f"{e:<11}" + "".join(f"{c[0]:>8.3f}({c[2]})" for c in cells))
        ns = ag(r4, lambda r: r['env'] == 'tmaze' and r['design'] == 'both' and r['suffix'] == '', 'usage_matched')[2]
        print(f"(seeds for tmaze/both: {ns})")

    # ---- E3 single-tau sweep (rev) ----
    st = [r for r in rr if r['suffix'].startswith('single')]
    if st:
        print("\n==== E3 single-timescale sweep (tmaze, design long, gap≈21) ====")
        taus = sorted({r['tau_slow'] for r in st})
        print(f"{'tau':>6}{'usage':>9}{'avail':>9}{'n':>4}")
        xs, us, av = [], [], []
        for t in taus:
            u = ag(st, lambda r, t=t: r['tau_slow'] == t, 'usage_matched')
            a = ag(st, lambda r, t=t: r['tau_slow'] == t, 'acc_slow')
            print(f"{t:>6.0f}{u[0]:>9.3f}{a[0]:>9.3f}{u[2]:>4}")
            xs.append(t); us.append(u[0]); av.append(a[0])
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(xs, us, 'o-', color='#2ca02c', label='usage'); ax.plot(xs, av, 's--', color='#1f77b4', label='availability')
        ax.axvline(21, ls='--', c='k', alpha=.5); ax.set_xscale('log', base=2)
        ax.set_xlabel('single-bank tau'); ax.set_ylabel('accuracy'); ax.set_ylim(0, 1.02)
        ax.set_title('E3: single-timescale EMA sweep (tmaze, gap≈21)'); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIG, 'exp_E3_singletau.png'), dpi=120, bbox_inches='tight')
        print("saved exp_E3_singletau.png")

    # ---- E4 horizon-law grid (rev) ----
    gr = [r for r in rr if r['suffix'].startswith('grid_')]
    if gr:
        print("\n==== E4 horizon-law grid: usage(gap, tau) ====")
        gaps = sorted({int(r['gap']) for r in gr}); taus = sorted({r['tau_slow'] for r in gr})
        print(f"{'gap\\tau':>8}" + "".join(f"{t:>8.0f}" for t in taus))
        M = np.zeros((len(gaps), len(taus)))
        for i, g in enumerate(gaps):
            row = []
            for j, t in enumerate(taus):
                u = ag(gr, lambda r, g=g, t=t: int(r['gap']) == g and r['tau_slow'] == t, 'usage_matched')
                M[i, j] = u[0]; row.append(f"{u[0]:>8.2f}")
            print(f"{g:>8}" + "".join(row))
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(M, origin='lower', aspect='auto', cmap='viridis', vmin=0.25, vmax=1.0)
        ax.set_xticks(range(len(taus))); ax.set_xticklabels([f'{t:.0f}' for t in taus])
        ax.set_yticks(range(len(gaps))); ax.set_yticklabels([f'{g}' for g in gaps])
        ax.set_xlabel('slow-bank tau'); ax.set_ylabel('cue→decision gap Δ')
        ax.set_title('E4: usage(Δ, τ) — usable iff τ ≳ Δ'); fig.colorbar(im, label='usage')
        fig.tight_layout(); fig.savefig(os.path.join(FIG, 'exp_E4_horizon_law.png'), dpi=120, bbox_inches='tight')
        print("saved exp_E4_horizon_law.png")

    # ---- E2 long-context baseline (rev) vs memory ----
    lc = [r for r in rr if r['suffix'].startswith('h')]
    if lc and r4:
        print("\n==== E2 long-context (tmaze, design none, window h) vs EMA memory ====")
        hs = sorted({int(r['suffix'][1:]) for r in lc})
        for h in hs:
            u = ag(lc, lambda r, h=h: r['suffix'] == f'h{h}', 'usage_matched')
            print(f"  none h={h:<3} usage={u[0]:.3f} (n={u[2]})")
        u_both = ag(r4, lambda r: r['env'] == 'tmaze' and r['design'] == 'both' and r['suffix'] == '', 'usage_matched')
        print(f"  EMA both (h=3) usage={u_both[0]:.3f}")


if __name__ == '__main__':
    main()
