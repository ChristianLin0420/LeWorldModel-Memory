"""Scan every trained run in outputs/mem, recompute metrics from saved weights, and
write a tidy master CSV (outputs/mem/master_metrics.csv) used by plot_experiments.py.

Recomputing from model.pt (not metrics.json) makes this robust to code/metric changes
across runs and gives a single consistent definition of every metric (incl. the matched
usage probe and learned tau)."""

import sys, csv
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from lewm.models.memory_model import MemoryLeWorldModel
from lewm.data import generate_eval_batch
from lewm.eval.memory_probe import extract_timewise, _fit_probe

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (Path(__file__).parent.parent / 'outputs' / 'mem')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_EVAL = 256

# default cue->decision gap Delta per env (when --reveal not overridden)
DEFAULT_GAP = {'tmaze': 24 - 3, 'distractor': 26 - 3, 'occlusion': 17 - 12,
               'recall': 20 - 5, 'tworoom': 0}


def build_model(a):
    mode = a['memory_mode']
    impl = mode if mode in ('multi', 'gru', 'ssm', 'retrieval', 'smt', 'ocsmt') else 'ema'
    ema_mode = 'both' if impl != 'ema' else mode
    m = MemoryLeWorldModel(
        img_size=a['img_size'], patch_size=a['patch_size'], embed_dim=a['embed_dim'], action_dim=2,
        encoder_layers=a['encoder_layers'], encoder_heads=a['encoder_heads'],
        predictor_layers=a['predictor_layers'], predictor_heads=a['predictor_heads'],
        history_len=a['history_len'], dropout=a['dropout'], sigreg_lambda=a['sigreg_lambda'],
        sigreg_projections=a['sigreg_projections'], memory_mode=ema_mode,
        tau_fast=a['tau_fast'], tau_slow=a['tau_slow'], learnable_alpha=not a.get('fixed_alpha', True),
        memory_impl=impl, multi_taus=tuple(a.get('multi_taus', (2, 4, 8, 16, 32, 64))),
        encoder_type=a.get('encoder', 'vit'), smt_router=a.get('smt_router', 'softmax'),
        oc_num=a.get('oc_num', 28), l0_lambda=a.get('l0_lambda', 0.0))
    return m


@torch.no_grad()
def analyze(run_dir):
    ck = torch.load(run_dir / 'model.pt', map_location=dev, weights_only=False)
    a = ck['args']
    m = build_model(a).to(dev); m.load_state_dict(ck['model_state_dict']); m.eval()
    env, design, seed = a['env'], a['memory_mode'], a['seed']
    length = a['length']; h = a['history_len']
    env_kwargs = {}
    if a.get('reveal') is not None: env_kwargs['reveal'] = a['reveal']
    if a.get('cue_len') is not None: env_kwargs['cue_len'] = a['cue_len']
    if a.get('n_distract') is not None: env_kwargs['n_distract'] = a['n_distract']
    if a.get('seq_len') is not None: env_kwargs['seq_len'] = a['seq_len']
    b = generate_eval_batch(env, N_EVAL, img_size=a['img_size'], length=length, seed=4242, **env_kwargs)
    obs, act, cue = b['obs'], b['actions'], b['cue'].numpy()
    n_classes = b['n_cue_classes']
    reveal = int(b['reveal'].float().mean()); cue_end = int(b['cue_end'].float().mean())
    gap = (a['reveal'] - (a.get('cue_len') or 3)) if a.get('reveal') is not None else DEFAULT_GAP.get(env, reveal - cue_end)

    feats = extract_timewise(m, obs, dev)
    z = torch.from_numpy(feats['z']).to(dev)
    # per-frame prediction MSE (mean over time & episodes)
    L, D = z.shape[1], z.shape[2]; W = L - h
    _, mf, ms, zt = m.encode_with_memory(obs.to(dev))
    zt_win = zt.unfold(1, h, 1)[:, :W].permute(0, 1, 3, 2).reshape(-1, h, D)
    act_win = act.to(dev).unfold(1, h, 1)[:, :W].permute(0, 1, 3, 2).reshape(-1, h, 2)
    tgt = z[:, h:L].reshape(-1, D)
    val_mse = float(((m.predictor(zt_win, act_win)[:, -1, :] - tgt) ** 2).mean())

    # availability at decision (one step before reveal) + matched usage probe at reveal
    row = dict(run=run_dir.name, env=env, design=design, seed=seed,
               suffix=a.get('run_suffix', ''), exp=a.get('extra_tag', ''),
               tau_fast=m.horizons()['tau_fast'], tau_slow=m.horizons()['tau_slow'],
               learnable=int(not a.get('fixed_alpha', True)), reveal=reveal, length=length,
               gap=gap, n_classes=n_classes, chance=1.0 / max(n_classes, 1), val_mse=val_mse,
               acc_z=np.nan, acc_fast=np.nan, acc_slow=np.nan, usage_matched=np.nan)
    if n_classes > 1:
        rng = np.random.default_rng(0); p = rng.permutation(len(cue)); ntr = int(.7 * len(cue))
        tr, te = p[:ntr], p[ntr:]
        dt = int(np.clip(reveal - 1, max(cue_end, 0), L - 1))
        for key, nm in [('z', 'acc_z'), ('m_fast', 'acc_fast'), ('m_slow', 'acc_slow')]:
            if key in feats:                                # m_fast/m_slow absent for non-EMA impls
                row[nm] = _fit_probe(feats[key][tr, dt], cue[tr], feats[key][te, dt], cue[te], n_classes)
        t = min(reveal, L - 1)
        win = zt[:, t - h:t]; a_w = act[:, t - h:t].to(dev)
        z_pred = m.predictor(win, a_w)[:, -1, :].cpu().numpy()
        row['usage_matched'] = _fit_probe(z_pred[tr], cue[tr], z_pred[te], cue[te], n_classes)
    return row


def main():
    runs = sorted([d for d in ROOT.glob('*') if (d / 'model.pt').exists()])
    print(f"analyzing {len(runs)} runs...")
    rows = []
    for d in runs:
        try:
            rows.append(analyze(d)); print(f"  {d.name}: usage={rows[-1]['usage_matched']:.3f} "
                                            f"val_mse={rows[-1]['val_mse']:.3f}")
        except Exception as e:
            print(f"  SKIP {d.name}: {e}")
    if not rows:
        return
    keys = list(rows[0].keys())
    out = ROOT / 'master_metrics.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {out} ({len(rows)} runs)")


if __name__ == '__main__':
    main()
