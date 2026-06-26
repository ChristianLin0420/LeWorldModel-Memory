"""E1 — Counterfactual memory swap: is the EMA memory CAUSALLY controlling the prediction,
or merely carrying decodable info?

For each episode i we predict the reveal-latent using i's current frames+actions but a
*different* episode j's memory banks (cue[j] != cue[i]). A cue probe trained on REAL reveal
latents is applied to the prediction:
  follow_self    = P(decode(pred | own memory)     == cue[i])   (control: should be high)
  follow_memory  = P(decode(pred | swapped memory) == cue[j])   (causal: high => memory drives it)
  follow_current = P(decode(pred | swapped memory) == cue[i])   (should collapse to ~chance)

Runs on already-trained `both` models (no retraining).
"""
import sys, argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from lewm.models.memory_model import MemoryLeWorldModel
from lewm.data import generate_eval_batch

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load(run_dir):
    ck = torch.load(run_dir / 'model.pt', map_location=dev, weights_only=False)
    a = ck['args']
    m = MemoryLeWorldModel(
        img_size=a['img_size'], patch_size=a['patch_size'], embed_dim=a['embed_dim'], action_dim=2,
        encoder_layers=a['encoder_layers'], encoder_heads=a['encoder_heads'],
        predictor_layers=a['predictor_layers'], predictor_heads=a['predictor_heads'],
        history_len=a['history_len'], dropout=a['dropout'], sigreg_lambda=a['sigreg_lambda'],
        sigreg_projections=a['sigreg_projections'], memory_mode=a['memory_mode'],
        tau_fast=a['tau_fast'], tau_slow=a['tau_slow'], learnable_alpha=not a.get('fixed_alpha', True)).to(dev)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m


@torch.no_grad()
def swap_eval(model, env, n=400, data_seed=999, split_seed=0):
    b = generate_eval_batch(env, n, length=32, seed=data_seed)
    obs, act = b['obs'].to(dev), b['actions'].to(dev)
    cue = b['cue'].numpy(); ncls = b['n_cue_classes']
    h = model.history_len
    reveal = int(b['reveal'].float().mean()); t = min(reveal, obs.shape[1] - 1)
    z, mf, ms, _ = model.encode_with_memory(obs)
    win = slice(t - h, t)

    def predict(z_win, mf_win, ms_win, a_win):
        zt = model.fusion(z_win, mf_win, ms_win)
        return model.predictor(zt, a_win)[:, -1, :].float().cpu().numpy()

    own = predict(z[:, win], mf[:, win], ms[:, win], act[:, win])
    # partner with a DIFFERENT cue for each episode
    rng = np.random.default_rng(split_seed)
    B = len(cue); partner = np.arange(B)
    for i in range(B):
        cand = np.where(cue != cue[i])[0]
        partner[i] = cand[rng.integers(len(cand))] if len(cand) else i
    pt = torch.as_tensor(partner, device=dev)
    swapped = predict(z[:, win], mf[pt][:, win], ms[pt][:, win], act[:, win])

    # MATCHED probe: train on the model's OWN predicted reveal-latents (not real latents),
    # so it reads the prediction distribution; then test whether swapped predictions follow
    # the partner's cue.
    perm = rng.permutation(B); ntr = int(0.7 * B); tr, te = perm[:ntr], perm[ntr:]
    sc = StandardScaler().fit(own[tr])
    clf = LogisticRegression(max_iter=400).fit(sc.transform(own[tr]), cue[tr])

    def acc(pred, target):
        return float((clf.predict(sc.transform(pred[te])) == target[te]).mean())

    return {'follow_self': acc(own, cue), 'follow_memory': acc(swapped, cue[partner]),
            'follow_current': acc(swapped, cue), 'chance': 1.0 / ncls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs/4ens')
    ap.add_argument('--envs', nargs='+', default=['tmaze', 'distractor', 'recall', 'occlusion'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    args = ap.parse_args()
    root = Path(args.root)

    rows = {}
    print(f"\n{'env':<12}{'follow_self':>13}{'follow_memory':>15}{'follow_current':>16}{'chance':>8}")
    for env in args.envs:
        accs = {k: [] for k in ['follow_self', 'follow_memory', 'follow_current', 'chance']}
        for s in args.seeds:
            rd = root / f'lewm-{env}-both-s{s}'
            if not (rd / 'model.pt').exists():
                continue
            r = swap_eval(load(rd), env)
            for k in accs:
                accs[k].append(r[k])
        if not accs['follow_self']:
            continue
        rows[env] = {k: (np.mean(v), np.std(v)) for k, v in accs.items()}
        m = rows[env]
        print(f"{env:<12}{m['follow_self'][0]:>8.3f}±{m['follow_self'][1]:>4.2f}"
              f"{m['follow_memory'][0]:>10.3f}±{m['follow_memory'][1]:>4.2f}"
              f"{m['follow_current'][0]:>11.3f}±{m['follow_current'][1]:>4.2f}{m['chance'][0]:>8.2f}")

    # figure
    envs = list(rows)
    fig, ax = plt.subplots(figsize=(1.6 * len(envs) + 3, 4.2))
    x = np.arange(len(envs)); w = 0.25
    series = [('follow_self', '#2ca02c', 'own memory → cue (control)'),
              ('follow_memory', '#1f77b4', 'swapped memory → swapped cue (causal)'),
              ('follow_current', '#d62728', 'swapped memory → current cue')]
    for i, (k, c, lab) in enumerate(series):
        ax.bar(x + (i - 1) * w, [rows[e][k][0] for e in envs], w,
               yerr=[rows[e][k][1] for e in envs], capsize=3, color=c, label=lab)
    for xi, e in zip(x, envs):
        ax.hlines(rows[e]['chance'][0], xi - 1.5 * w, xi + 1.5 * w, color='k', ls=':', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(envs); ax.set_ylim(0, 1.02)
    ax.set_ylabel('cue decoded from prediction')
    ax.set_title('E1: counterfactual memory swap — prediction follows the INJECTED memory, not the current frame')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout(); out = root / 'fig_causal_swap.png'
    fig.savefig(out, dpi=120, bbox_inches='tight'); print(f"\nsaved {out}")


if __name__ == '__main__':
    main()
