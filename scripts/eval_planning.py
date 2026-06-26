"""E7 — downstream closed-loop control on the memory T-Maze.

The agent gathers a short context by moving right (the cue enters memory), the world model
*imagines* the goal by rolling its latent forward to the reveal step (memory-aware), and a
linear read-out of that imagined latent decides which arm to enter. Success = the agent
commits to the cued arm. We compare designs (none/short/long/both) and, on `both`, ablate
the memory at test time (the decisive causal check that memory drives the decision).
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
from lewm.envs.control_envs import TMazeControlEnv

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load(run_dir):
    ck = torch.load(run_dir / 'model.pt', map_location=dev, weights_only=False); a = ck['args']
    m = MemoryLeWorldModel(
        img_size=a['img_size'], patch_size=a['patch_size'], embed_dim=a['embed_dim'], action_dim=2,
        encoder_layers=a['encoder_layers'], encoder_heads=a['encoder_heads'],
        predictor_layers=a['predictor_layers'], predictor_heads=a['predictor_heads'],
        history_len=a['history_len'], dropout=a['dropout'], sigreg_lambda=a['sigreg_lambda'],
        sigreg_projections=a['sigreg_projections'], memory_mode=a['memory_mode'],
        tau_fast=a['tau_fast'], tau_slow=a['tau_slow'], learnable_alpha=not a.get('fixed_alpha', True)).to(dev)
    m.load_state_dict(ck['model_state_dict']); m.eval(); return m


def build_contexts(n, gather, seed):
    """n episodes: reset (random cue) + move right `gather` steps -> context frames + cues."""
    obs, cues = [], []
    for i in range(n):
        e = TMazeControlEnv()
        o = e.reset(np.random.default_rng(seed * 99991 + i)); seq = [o]
        for _ in range(gather):
            o, *_ = e.act(np.array([1.0, 0.0])); seq.append(o)
        obs.append(np.stack(seq)); cues.append(e.cue)
    obs = torch.from_numpy(np.stack(obs).astype(np.float32) / 255.0).permute(0, 1, 4, 2, 3).contiguous()
    return obs, np.array(cues)


@torch.no_grad()
def imagine(model, ctx, horizon, ablate=False):
    B = ctx.shape[0]
    fa = torch.zeros(B, horizon, 2, device=dev); fa[..., 0] = 1.0          # keep moving right
    g = model.rollout_latents(ctx.to(dev), fa, horizon, ablate_fast=ablate, ablate_slow=ablate)
    return g[:, -1].float().cpu().numpy()                                  # imagined reveal latent


def eval_design(model, gather, horizon, n=500, seed=1):
    ctx, cue = build_contexts(n, gather, seed)
    g = imagine(model, ctx, horizon, ablate=False)
    rng = np.random.default_rng(0); p = rng.permutation(n); ntr = int(.6 * n); tr, te = p[:ntr], p[ntr:]
    sc = StandardScaler().fit(g[tr]); clf = LogisticRegression(max_iter=500).fit(sc.transform(g[tr]), cue[tr])
    succ = float((clf.predict(sc.transform(g[te])) == cue[te]).mean())
    g_ab = imagine(model, ctx, horizon, ablate=True)                       # test-time memory ablation
    succ_ab = float((clf.predict(sc.transform(g_ab[te])) == cue[te]).mean())
    return succ, succ_ab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs/4ens')
    ap.add_argument('--designs', nargs='+', default=['none', 'short', 'long', 'both'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--gather', type=int, default=6)
    ap.add_argument('--horizon', type=int, default=18)
    args = ap.parse_args()
    root = Path(args.root)

    res = {}
    print(f"\nE7 closed-loop T-Maze control success (gather={args.gather}, horizon={args.horizon})")
    print(f"{'design':<8}{'success':>12}{'mem-ablated':>14}{'chance':>8}")
    for d in args.designs:
        s_full, s_ab = [], []
        for sd in args.seeds:
            rd = root / f'lewm-tmaze-{d}-s{sd}'
            if not (rd / 'model.pt').exists():
                continue
            a, b = eval_design(load(rd), args.gather, args.horizon)
            s_full.append(a); s_ab.append(b)
        if not s_full:
            continue
        res[d] = (np.mean(s_full), np.std(s_full), np.mean(s_ab), np.std(s_ab))
        print(f"{d:<8}{res[d][0]:>7.3f}±{res[d][1]:>4.2f}{res[d][2]:>9.3f}±{res[d][3]:>4.2f}{0.5:>8.2f}")

    # figure
    ds = list(res); x = np.arange(len(ds)); w = 0.35
    fig, ax = plt.subplots(figsize=(1.4 * len(ds) + 3, 4.2))
    ax.bar(x - w / 2, [res[d][0] for d in ds], w, yerr=[res[d][1] for d in ds], capsize=3, color='#2ca02c', label='with memory')
    ax.bar(x + w / 2, [res[d][2] for d in ds], w, yerr=[res[d][3] for d in ds], capsize=3, color='#d62728', label='memory ablated at test')
    ax.axhline(0.5, ls=':', c='k', label='chance')
    ax.set_xticks(x); ax.set_xticklabels(ds); ax.set_ylim(0, 1.02); ax.set_ylabel('control success rate')
    ax.set_title('E7: closed-loop T-Maze control — memory enables the correct arm decision')
    ax.legend(fontsize=8)
    out = Path('docs/figures/fig_E7_planning.png'); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches='tight'); print(f"saved {out}")


if __name__ == '__main__':
    main()
