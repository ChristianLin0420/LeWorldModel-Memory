"""Deep analysis of trained LeWM-Memory models: where (in time) memory reduces
prediction error, and a corrected 'decision uses memory' probe."""

import os, sys, glob
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from lewm.models.memory_model import MemoryLeWorldModel
from lewm.data import generate_eval_batch

ROOT = Path(__file__).parent.parent / 'outputs' / 'mem'
DESIGNS = ['none', 'short', 'long', 'both']
COL = {'none': '#999999', 'short': '#d62728', 'long': '#1f77b4', 'both': '#2ca02c'}
ENVS = ['tmaze', 'occlusion', 'recall', 'distractor', 'tworoom']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(run_dir):
    ck = torch.load(run_dir / 'model.pt', map_location=dev, weights_only=False)
    a = ck['args']
    m = MemoryLeWorldModel(
        img_size=a['img_size'], patch_size=a['patch_size'], embed_dim=a['embed_dim'], action_dim=2,
        encoder_layers=a['encoder_layers'], encoder_heads=a['encoder_heads'],
        predictor_layers=a['predictor_layers'], predictor_heads=a['predictor_heads'],
        history_len=a['history_len'], dropout=a['dropout'], sigreg_lambda=a['sigreg_lambda'],
        sigreg_projections=a['sigreg_projections'], memory_mode=a['memory_mode'],
        tau_fast=a['tau_fast'], tau_slow=a['tau_slow'], learnable_alpha=not a['fixed_alpha']).to(dev)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m, a


@torch.no_grad()
def per_timestep_mse(model, obs, act):
    """Mean (over episodes & dims) next-latent prediction MSE at each predicted frame t."""
    h = model.history_len
    z, mf, ms, zt = model.encode_with_memory(obs.to(dev))
    B, L, D = z.shape
    W = L - h
    zt_win = zt.unfold(1, h, 1)[:, :W].permute(0, 1, 3, 2).reshape(B * W, h, D)
    act_win = act.to(dev).unfold(1, h, 1)[:, :W].permute(0, 1, 3, 2).reshape(B * W, h, model.action_dim)
    tgt = z[:, h:L].reshape(B * W, D)
    pred = model.predictor(zt_win, act_win)[:, -1, :]
    mse = ((pred - tgt) ** 2).mean(-1).reshape(B, W).mean(0).cpu().numpy()  # (W,) for t=h..L-1
    return mse, np.arange(h, L)


@torch.no_grad()
def cue_from_prediction(model, batch, seed=0):
    """Two protocols: probe trained on TRUE reveal latent (cross-dist) vs on PRED latent (matched)."""
    n_classes = batch['n_cue_classes']
    if n_classes < 2:
        return {}
    h = model.history_len
    reveal = int(batch['reveal'].float().mean())
    obs, act, cue = batch['obs'], batch['actions'], batch['cue'].numpy()
    z, mf, ms, zt = model.encode_with_memory(obs.to(dev))
    L = z.shape[1]
    t = min(reveal, L - 1)
    z_true = z[:, t].cpu().numpy()
    win = zt[:, t - h:t]; a_w = act[:, t - h:t].to(dev)
    z_pred = model.predictor(win, a_w)[:, -1, :].cpu().numpy()
    B = len(cue); rng = np.random.default_rng(seed); p = rng.permutation(B); n = int(.7 * B)
    tr, te = p[:n], p[n:]

    def probe(Xtr, ytr, Xte, yte):
        sc = StandardScaler().fit(Xtr)
        return float(LogisticRegression(max_iter=400).fit(sc.transform(Xtr), ytr).score(sc.transform(Xte), yte))
    return {
        'pred_acc_crossdist': probe(z_true[tr], cue[tr], z_pred[te], cue[te]),   # train true -> test pred
        'pred_acc_matched': probe(z_pred[tr], cue[tr], z_pred[te], cue[te]),     # train pred -> test pred
        'true_acc': probe(z_true[tr], cue[tr], z_true[te], cue[te]),
        'chance': 1.0 / n_classes, 'reveal': t,
    }


def main():
    batches = {e: generate_eval_batch(e, 400, length=32, seed=123) for e in ENVS}

    # ---- per-timestep MSE figure (the money plot) ----
    fig, axes = plt.subplots(1, len(ENVS), figsize=(4.2 * len(ENVS), 3.8), sharey=False)
    print("\n==== CORRECTED 'decision uses memory' (probe matched to prediction distribution) ====")
    print(f"{'env':<11}{'design':<7}{'pred(crossdist)':>16}{'pred(matched)':>15}{'true':>8}{'chance':>8}")
    usage = {}
    for ci, env in enumerate(ENVS):
        ax = axes[ci]
        b = batches[env]
        reveal = int(b['reveal'].float().mean()); cue_end = int(b['cue_end'].float().mean())
        for d in DESIGNS:
            rd = ROOT / f"lewm-{env}-{d}-s0"
            if not (rd / 'model.pt').exists():
                continue
            m, a = load_model(rd)
            mse, ts = per_timestep_mse(m, b['obs'], b['actions'])
            ax.plot(ts, mse, color=COL[d], lw=1.8, marker='.', ms=4, label=d)
            u = cue_from_prediction(m, b)
            if u:
                usage[(env, d)] = u
                print(f"{env:<11}{d:<7}{u['pred_acc_crossdist']:>16.3f}{u['pred_acc_matched']:>15.3f}"
                      f"{u['true_acc']:>8.3f}{u['chance']:>8.3f}")
        if env != 'tworoom':
            ax.axvline(reveal, ls='--', c='green', alpha=.6)
            ax.text(reveal + .2, ax.get_ylim()[1] * 0.9, 'reveal', color='green', fontsize=8, rotation=90)
            ax.axvline(cue_end, ls='--', c='k', alpha=.4)
        ax.set_title(f"{env}"); ax.set_xlabel('predicted frame t'); ax.set_ylim(bottom=0)
        if ci == 0: ax.set_ylabel('next-latent prediction MSE')
        ax.legend(fontsize=8)
    fig.suptitle('Where memory helps: per-frame next-latent prediction error by design '
                 '(memory cuts error exactly at the cue-determined frames; tworoom flat)', y=1.02)
    fig.tight_layout()
    out = ROOT / 'analysis_mse_by_time.png'
    fig.savefig(out, dpi=110, bbox_inches='tight'); print(f"\nsaved {out}")

    # ---- corrected decision figure (matched probe) ----
    envs2 = [e for e in ENVS if any((e, d) in usage for d in DESIGNS)]
    fig2, ax2 = plt.subplots(figsize=(1.7 * len(envs2) + 3, 4.2))
    x = np.arange(len(envs2)); w = 0.2
    for i, d in enumerate(DESIGNS):
        vals = [usage.get((e, d), {}).get('pred_acc_matched', np.nan) for e in envs2]
        ax2.bar(x + (i - 1.5) * w, vals, w, label=d, color=COL[d])
    for xi, e in zip(x, envs2):
        ch = usage.get((e, 'both'), {}).get('chance', np.nan)
        ax2.hlines(ch, xi - 2 * w, xi + 2 * w, color='k', ls=':', lw=1)
    ax2.set_xticks(x); ax2.set_xticklabels(envs2); ax2.set_ylim(0, 1.02)
    ax2.set_ylabel('cue acc from predicted reveal-latent')
    ax2.set_title("Decision uses memory (corrected, matched probe): is the cue decodable from the\n"
                  "model's PREDICTION? (dotted=chance; long/both recover it, none/short stay ~chance)")
    ax2.legend(title='design', ncol=4, fontsize=8); fig2.tight_layout()
    fig2.savefig(ROOT / 'summary_decision_corrected.png', dpi=110, bbox_inches='tight')
    print('saved', ROOT / 'summary_decision_corrected.png')

    # ---- summary: prediction-error reduction vs baseline ----
    print("\n==== val-region prediction MSE (mean over episode) and reduction vs 'none' ====")
    print(f"{'env':<11}" + "".join(f"{d:>9}" for d in DESIGNS) + f"{'both_red%':>10}")
    for env in ENVS:
        row = {}
        for d in DESIGNS:
            rd = ROOT / f"lewm-{env}-{d}-s0"
            if (rd / 'model.pt').exists():
                m, _ = load_model(rd)
                mse, _ = per_timestep_mse(m, batches[env]['obs'], batches[env]['actions'])
                row[d] = float(mse.mean())
        if 'none' in row and 'both' in row:
            red = 100 * (row['none'] - row['both']) / row['none']
        else:
            red = float('nan')
        print(f"{env:<11}" + "".join(f"{row.get(d, float('nan')):>9.3f}" for d in DESIGNS) + f"{red:>9.1f}%")


if __name__ == '__main__':
    main()
