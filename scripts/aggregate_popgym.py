"""Aggregate POPGym Arcade runs: vanilla vs memory, per env x design (mean+/-std over seeds)."""
import json, glob, os, re
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'popgym')
DES = ['none', 'short', 'long', 'both']
COL = {'none': '#999999', 'short': '#d62728', 'long': '#1f77b4', 'both': '#2ca02c'}


def main():
    rows = []
    for f in glob.glob(os.path.join(ROOT, '*', 'metrics.json')):
        m = json.load(open(f))
        sd = re.search(r'-s(\d+)$', os.path.basename(os.path.dirname(f)))
        m['seed'] = int(sd.group(1)) if sd else 0
        rows.append(m)
    if not rows:
        print('no popgym runs found'); return
    envs = sorted({r['env'] for r in rows})

    def ag(e, d, k):
        v = [r[k] for r in rows if r['env'] == e and r['design'] == d and k in r]
        return (np.mean(v), np.std(v), len(v)) if v else (np.nan, np.nan, 0)

    print("\n" + "=" * 88)
    print("POPGym Arcade: vanilla (none) vs memory  [mean+/-std over seeds]")
    print("=" * 88)
    for metric, lo_better in [('val_pred_loss', True), ('infl_slow', False), ('infl_fast', False)]:
        print(f"\n{metric}  ({'lower=better' if lo_better else 'memory influence on prediction'})")
        print(f"{'env':<18}" + "".join(f"{d:>14}" for d in DES))
        for e in envs:
            print(f"{e:<18}" + "".join(f"{ag(e,d,metric)[0]:>8.3f}±{ag(e,d,metric)[1]:>4.2f}" for d in DES))
    print(f"\nseeds/cell: {ag(envs[0],'both','val_pred_loss')[2]}")

    # figure: val_pred_loss + infl_slow per env x design
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(envs)); w = 0.2
    for i, d in enumerate(DES):
        a1.bar(x + (i - 1.5) * w, [ag(e, d, 'val_pred_loss')[0] for e in envs], w,
               yerr=[ag(e, d, 'val_pred_loss')[1] for e in envs], capsize=3, label=d, color=COL[d])
        a2.bar(x + (i - 1.5) * w, [ag(e, d, 'infl_slow')[0] for e in envs], w,
               yerr=[ag(e, d, 'infl_slow')[1] for e in envs], capsize=3, label=d, color=COL[d])
    for ax, t, yl in [(a1, 'World-model prediction error (lower=better)', 'val next-latent MSE'),
                      (a2, 'Long-memory influence on prediction', 'infl_slow')]:
        ax.set_xticks(x); ax.set_xticklabels(envs, fontsize=8); ax.set_title(t); ax.set_ylabel(yl)
        ax.legend(title='design', ncol=4, fontsize=8)
    fig.suptitle('POPGym Arcade (standard memory POMDPs): vanilla LeWM vs two-timescale memory', y=1.02)
    fig.tight_layout(); out = os.path.join(ROOT, 'summary_popgym.png')
    fig.savefig(out, dpi=110, bbox_inches='tight'); print(f"\nsaved {out}")
    with open(os.path.join(ROOT, 'summary.csv'), 'w') as f:
        f.write('env,design,val_pred_loss,infl_fast,infl_slow,tau_fast,tau_slow\n')
        for e in envs:
            for d in DES:
                f.write(f"{e},{d},{ag(e,d,'val_pred_loss')[0]:.4f},{ag(e,d,'infl_fast')[0]:.4f},"
                        f"{ag(e,d,'infl_slow')[0]:.4f},{ag(e,d,'tau_fast')[0]:.2f},{ag(e,d,'tau_slow')[0]:.2f}\n")
    print(f"saved {os.path.join(ROOT,'summary.csv')}")


if __name__ == '__main__':
    main()
