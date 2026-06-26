"""Memory-attribution timeline: for one episode, show (A) the frames, (B) how much each
step's next-latent prediction relies on the fast vs slow memory bank, and (C) which past
frames the decision step actually reads (gradient attribution + the exponential kernels).

Answers: "how does the current step decide the next prediction, and from which frames?"
Runs on a trained EMA `both` model (no retraining)."""
import sys, argparse
from pathlib import Path
import numpy as np
import torch
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
        sigreg_projections=a['sigreg_projections'], memory_mode='both',
        tau_fast=a['tau_fast'], tau_slow=a['tau_slow'], learnable_alpha=not a.get('fixed_alpha', True)).to(dev)
    m.load_state_dict(ck['model_state_dict']); m.eval()
    return m


@torch.no_grad()
def influence_timeline(model, obs, act):
    """I_fast(t), I_slow(t): movement of the next-latent prediction when each bank is ablated."""
    h = model.history_len
    z, mf, ms, _ = model.encode_with_memory(obs.to(dev))
    L = z.shape[1]; a = act.to(dev)
    If, Is = np.full(L, np.nan), np.full(L, np.nan)
    for t in range(h, L):
        w = slice(t - h, t)

        def pr(af, as_):
            zt = model.fusion(z[:, w], mf[:, w], ms[:, w], ablate_fast=af, ablate_slow=as_)
            return model.predictor(zt, a[:, w])[:, -1, :]
        full = pr(False, False)
        If[t] = (full - pr(True, False)).norm(dim=-1).mean().item()
        Is[t] = (full - pr(False, True)).norm(dim=-1).mean().item()
    return If, Is


def frame_attribution(model, obs1, act1, t_star):
    """||d yhat_{t*} / d z_s|| for every source frame s (grad through window + EMA banks)."""
    h = model.history_len
    z = model.encode(obs1.to(dev)).detach().requires_grad_(True)   # (1,L,D)
    mf, ms = model.memory(z)
    zt = model.fusion(z, mf, ms)
    w = slice(t_star - h, t_star)
    yhat = model.predictor(zt[:, w], act1.to(dev)[:, w])[:, -1, :]
    (yhat ** 2).sum().backward()
    return z.grad[0].norm(dim=-1).cpu().numpy()                    # (L,)


def make_figure(env, run_dir, out_png, n_inf=128):
    model = load(run_dir)
    b = generate_eval_batch(env, n_inf, length=32, seed=7)
    obs, act, cue = b['obs'], b['actions'], b['cue']
    h = model.history_len
    cue_end = int(b['cue_end'].float().mean()); reveal = int(b['reveal'].float().mean())
    L = obs.shape[1]
    If, Is = influence_timeline(model, obs, act)
    # one representative episode (first with cue 0) for filmstrip + attribution
    idx = int((cue == cue[0]).nonzero()[0])
    obs1, act1 = obs[idx:idx + 1], act[idx:idx + 1]
    attr = frame_attribution(model, obs1, act1, reveal)
    tau = model.memory.horizons()
    kf = model.memory.kernel(L, 'fast').cpu().numpy()
    ks = model.memory.kernel(L, 'slow').cpu().numpy()

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.1, 1.3], hspace=0.45)

    # (A) filmstrip
    nshow = 16; ts = np.linspace(0, L - 1, nshow).astype(int)
    axf = fig.add_subplot(gs[0])
    strip = np.concatenate([obs1[0, t].permute(1, 2, 0).numpy() for t in ts], axis=1)
    axf.imshow(strip); axf.set_yticks([]); axf.set_xticks([])
    for j, t in enumerate(ts):
        axf.text(j * 64 + 32, -4, str(t), ha='center', va='bottom', fontsize=7)
    axf.set_title(f'(A) [{env}] episode frames (cue at t<{cue_end}, decision at t={reveal})', fontsize=11)

    # (B) per-step fast vs slow influence on the next prediction
    axi = fig.add_subplot(gs[1])
    tgrid = np.arange(L)
    axi.plot(tgrid, If, 'o-', color='#d62728', label=f'fast bank ($\\tau$={tau["tau_fast"]:.0f})')
    axi.plot(tgrid, Is, 's-', color='#1f77b4', label=f'slow bank ($\\tau$={tau["tau_slow"]:.0f})')
    axi.axvline(cue_end, ls='--', c='k', alpha=.5); axi.axvline(reveal, ls='--', c='green', alpha=.6)
    axi.set_xlim(-0.5, L - 0.5); axi.set_xlabel('decision step t'); axi.set_ylabel('influence on next pred')
    axi.set_title('(B) how much each step relies on short- vs long-term memory', fontsize=11)
    axi.legend(fontsize=8)

    # (C) which frames the decision at t=reveal reads
    axa = fig.add_subplot(gs[2])
    s = np.arange(L)
    a_n = attr / (attr.max() + 1e-9)
    axa.bar(s, a_n, color='#888888', width=0.8, label='grad attribution $\\|\\partial \\hat y_{rev}/\\partial z_s\\|$')
    # exponential kernels anchored at the decision (reach back from reveal)
    lag = reveal - s
    kf_r = np.where(lag >= 0, model.memory.alpha_fast.item() * (1 - model.memory.alpha_fast.item()) ** np.clip(lag, 0, None), 0)
    ks_r = np.where(lag >= 0, model.memory.alpha_slow.item() * (1 - model.memory.alpha_slow.item()) ** np.clip(lag, 0, None), 0)
    axa.plot(s, kf_r / (kf_r.max() + 1e-9), color='#d62728', lw=2, label='fast kernel $K_f(t{-}s)$')
    axa.plot(s, ks_r / (ks_r.max() + 1e-9), color='#1f77b4', lw=2, label='slow kernel $K_s(t{-}s)$')
    axa.axvspan(-0.5, cue_end - 0.5, color='gold', alpha=0.25, label='cue frames')
    axa.axvline(reveal, ls='--', c='green', alpha=.6)
    axa.set_xlim(-0.5, L - 0.5); axa.set_xlabel('source frame s'); axa.set_ylabel('normalized attribution')
    axa.set_title(f'(C) which frames the decision at t={reveal} reads (slow kernel reaches the cue; fast does not)', fontsize=11)
    axa.legend(fontsize=8, ncol=2)

    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f"saved {out_png}  (cue_end={cue_end} reveal={reveal} tau_f={tau['tau_fast']:.0f} tau_s={tau['tau_slow']:.0f})")
    print(f"  attribution on cue frames (0..{cue_end-1}): {attr[:cue_end].mean():.3e} | "
          f"mid-delay frames: {attr[cue_end:reveal-h].mean():.3e} | window frames: {attr[reveal-h:reveal].mean():.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs/4ens')
    ap.add_argument('--envs', nargs='+', default=['tmaze', 'occlusion'])
    args = ap.parse_args()
    outdir = Path('docs/figures'); outdir.mkdir(parents=True, exist_ok=True)
    for env in args.envs:
        rd = Path(args.root) / f'lewm-{env}-both-s0'
        if (rd / 'model.pt').exists():
            make_figure(env, rd, outdir / f'fig_attribution_{env}.png')
        else:
            print(f"skip {env}: no model at {rd}")


if __name__ == '__main__':
    main()
