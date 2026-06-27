"""Router visualization (docs/LEARNABLE_MEMORY.md §5). SMT's timescales are FIXED and known, so
the read router is interpretable as a preference over KNOWN horizons. We report two honest views:
 (a) per-horizon mean read weight vs the uniform baseline 1/K -- does the router prefer
     task-appropriate horizons?
 (b) temporal std of the router per horizon -- does it switch horizons over the sequence?
Uses trained smt(sigmoid) checkpoints (outputs/smt_v2)."""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from lewm.models.memory_model import MemoryLeWorldModel
from lewm.data import generate_eval_batch

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = Path('outputs/smt_v2')
ENVS = ['tmaze', 'distractor', 'occlusion']
COL = {'tmaze': '#3182bd', 'distractor': '#e6550d', 'occlusion': '#31a354'}


def build(ck):
    a = ck['args']
    return MemoryLeWorldModel(
        img_size=a['img_size'], patch_size=a['patch_size'], embed_dim=a['embed_dim'], action_dim=2,
        encoder_layers=a['encoder_layers'], encoder_heads=a['encoder_heads'],
        predictor_layers=a['predictor_layers'], predictor_heads=a['predictor_heads'],
        history_len=a['history_len'], dropout=a['dropout'], sigreg_lambda=a['sigreg_lambda'],
        sigreg_projections=a['sigreg_projections'], memory_mode='both', memory_impl='smt',
        multi_taus=tuple(a.get('multi_taus', (2, 4, 8, 16, 32, 64))),
        smt_router=a.get('smt_router', 'sigmoid'))


@torch.no_grad()
def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    taus = None
    perstep_var = {}
    for env in ENVS:
        ck = torch.load(ROOT / f'lewm-{env}-smt-s0' / 'model.pt', map_location=DEV, weights_only=False)
        m = build(ck).to(DEV); m.load_state_dict(ck['model_state_dict']); m.eval()
        taus = m.mem_smt.taus
        b = generate_eval_batch(env, 256, img_size=ck['args']['img_size'], length=ck['args']['length'], seed=4242)
        w = m.mem_smt.route_weights(m.encode(b['obs'].to(DEV)))          # (B,L,K)
        wn = (w / (w.sum(-1, keepdim=True) + 1e-8)).cpu().numpy()        # per-step distribution
        perstep_var[env] = wn.mean(0).std(0)                            # (K,) temporal std of mean
        pref = wn.mean((0, 1))                                          # (K,) mean preference
        x = np.arange(len(taus))
        off = (ENVS.index(env) - 1) * 0.26
        ax1.bar(x + off, pref, 0.26, label=env, color=COL[env])
    K = len(taus)
    ax1.axhline(1.0 / K, color='k', ls='--', lw=1, label=f'uniform (1/{K})')
    ax1.set_xticks(range(K)); ax1.set_xticklabels([f'{t}' for t in taus])
    ax1.set_xlabel('memory horizon $\\tau$'); ax1.set_ylabel('mean read weight')
    ax1.set_title('(a) per-horizon read preference (vs uniform)', fontsize=10)
    ax1.legend(fontsize=7.5); ax1.spines[['top', 'right']].set_visible(False)
    for env in ENVS:
        ax2.plot(range(K), perstep_var[env], '-o', ms=4, color=COL[env], label=env)
    ax2.set_xticks(range(K)); ax2.set_xticklabels([f'{t}' for t in taus])
    ax2.set_xlabel('memory horizon $\\tau$'); ax2.set_ylabel('temporal std of read weight')
    ax2.set_title('(b) how much routing varies over time (per horizon)', fontsize=10)
    ax2.legend(fontsize=7.5); ax2.spines[['top', 'right']].set_visible(False)
    fig.suptitle('SMT learned read router: weakly task-appropriate, near-static over time', fontsize=11, weight='bold')
    plt.tight_layout()
    out = Path('docs/figures/fig_smt_router.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); print('wrote', out)
    # print the numbers for the writeup
    print('uniform =', round(1.0 / K, 3))
    for env in ENVS:
        print(env, 'temporal-std max =', round(float(perstep_var[env].max()), 4))


if __name__ == '__main__':
    main()
