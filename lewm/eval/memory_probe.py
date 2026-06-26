"""
Probing + visualization: *how do short- and long-term memory affect the decision?*

Two complementary views, matching the two halves of the claim:

  (A) AVAILABILITY -- "where does the cue live, and for how long?"
      For each time step t we train a linear probe to read the episode's cue from each
      feature stream (the memoryless encoder latent z, the fast bank m^f, the slow bank
      m^s). Plotting probe accuracy vs t shows that z forgets the cue the instant it
      leaves the frame, m^f keeps it for ~tau_f steps, and m^s keeps it for ~tau_s steps.
      This is the empirical signature of the exponential memory kernel.

  (B) USAGE -- "does the model's *decision* use it?"
      The decision proxy is the predictor's imagined latent at the reveal step, z_hat_reveal.
      We train a probe on the *true* reveal latent and apply it to the model's *prediction*.
      If the world model recruited long-term memory, its prediction already encodes the
      cue-determined event -> the cue is decodable from the prediction. A memoryless (or
      short-only) model cannot know the cue at reveal -> the cue is not decodable. We also
      report the causal influence ||f(full) - f(ablate bank)|| of each bank.

Together: (A) shows the information is *there* in the slow bank (a property of the math);
(B) shows that injecting it is what lets the *decision* depend on long-term memory.
"""

from typing import Dict, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def extract_timewise(model, obs: torch.Tensor, device, batch_size: int = 64
                     ) -> Dict[str, np.ndarray]:
    """Return per-time feature streams as (B, L, D) numpy arrays."""
    model.eval()
    out = {'z': [], 'z_tilde': []}
    for i in range(0, obs.shape[0], batch_size):
        o = obs[i:i + batch_size].to(device)
        z, m_fast, m_slow, z_tilde = model.encode_with_memory(o)
        out['z'].append(z.float().cpu())
        out['z_tilde'].append(z_tilde.float().cpu())
        if m_fast is not None:                              # EMA only; non-EMA impls have no banks
            out.setdefault('m_fast', []).append(m_fast.float().cpu())
            out.setdefault('m_slow', []).append(m_slow.float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def _fit_probe(Xtr, ytr, Xte, yte, n_classes):
    """Standardize + multinomial logistic regression; return test accuracy."""
    if len(np.unique(ytr)) < 2:
        return 1.0 / max(n_classes, 1)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=300, C=1.0)
    try:
        clf.fit(sc.transform(Xtr), ytr)
        return float(clf.score(sc.transform(Xte), yte))
    except Exception:
        return 1.0 / max(n_classes, 1)


def probe_cue_over_time(feats: np.ndarray, cue: np.ndarray, n_classes: int,
                        train_ratio: float = 0.7, seed: int = 0) -> np.ndarray:
    """Probe accuracy of decoding `cue` from `feats[:, t, :]` for every t -> (L,)."""
    B, L, _ = feats.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(B)
    ntr = int(B * train_ratio)
    tr, te = perm[:ntr], perm[ntr:]
    acc = np.empty(L)
    for t in range(L):
        acc[t] = _fit_probe(feats[tr, t], cue[tr], feats[te, t], cue[te], n_classes)
    return acc


@torch.no_grad()
def _predict_latent_at(model, obs, actions, t: int, device) -> np.ndarray:
    """Model's imagined latent at time t (predicted from the fused window ending t-1)."""
    h = model.history_len
    z, m_fast, m_slow, z_tilde = model.encode_with_memory(obs.to(device))
    win = z_tilde[:, t - h:t]
    act = actions[:, t - h:t].to(device)
    return model.predictor(win, act)[:, -1, :].float().cpu().numpy()


def decision_uses_memory(model, eval_batch, feats, device, train_ratio: float = 0.7,
                         seed: int = 0) -> Dict[str, float]:
    """USAGE metric: is the cue decodable from the model's *predicted* reveal latent?"""
    n_classes = eval_batch['n_cue_classes']
    h = model.history_len
    reveal = int(eval_batch['reveal'].float().mean())
    L = feats['z'].shape[1]
    if n_classes < 2 or reveal < h or reveal >= L:
        return {}
    cue = eval_batch['cue'].numpy()
    obs, actions = eval_batch['obs'], eval_batch['actions']

    z_real = feats['z'][:, reveal, :]                       # true reveal latent
    z_pred = _predict_latent_at(model, obs, actions, reveal, device)  # imagined reveal latent

    B = z_real.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(B)
    ntr = int(B * train_ratio)
    tr, te = perm[:ntr], perm[ntr:]
    # probe trained on TRUE reveal latents, evaluated on the model's PREDICTION (cross-dist)
    acc_from_pred = _fit_probe(z_real[tr], cue[tr], z_pred[te], cue[te], n_classes)
    acc_from_true = _fit_probe(z_real[tr], cue[tr], z_real[te], cue[te], n_classes)
    # MATCHED probe: trained and tested on the model's PREDICTIONS (the correct "does the
    # decision encode the cue?" measure -- avoids the encoder->predictor distribution shift)
    acc_matched = _fit_probe(z_pred[tr], cue[tr], z_pred[te], cue[te], n_classes)
    return {
        'cue_acc_from_prediction': acc_from_pred,
        'cue_acc_from_prediction_matched': acc_matched,
        'cue_acc_from_true_latent': acc_from_true,
        'chance': 1.0 / n_classes,
    }


def plot_probe_curves(streams: Dict[str, np.ndarray], cue_end: int, reveal: int,
                      n_classes: int, env: str, tau: Dict[str, float]):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = {'z': '#888888', 'm_fast': '#d62728', 'm_slow': '#1f77b4', 'z_tilde': '#2ca02c'}
    labels = {'z': 'z (memoryless encoder)', 'm_fast': f"m_fast (tau={tau['tau_fast']:.1f})",
              'm_slow': f"m_slow (tau={tau['tau_slow']:.1f})", 'z_tilde': 'z~ (predictor input)'}
    for name, acc in streams.items():
        ax.plot(acc, marker='o', ms=3, lw=1.8, color=colors.get(name), label=labels.get(name, name))
    ax.axhline(1.0 / n_classes, ls=':', c='gray', lw=1, label='chance')
    ax.axvline(cue_end, ls='--', c='k', alpha=.6)
    ax.axvline(reveal, ls='--', c='green', alpha=.6)
    ymax = 1.02
    ax.text(cue_end + 0.2, 0.05, 'cue off', rotation=90, fontsize=8, alpha=.7)
    ax.text(reveal + 0.2, 0.05, 'reveal/decision', rotation=90, fontsize=8, color='green', alpha=.8)
    ax.set_xlabel('time step'); ax.set_ylabel('cue decode accuracy')
    ax.set_ylim(0, ymax)
    ax.set_title(f'[{env}] AVAILABILITY: cue decodability of memory streams over time')
    ax.legend(loc='center right', fontsize=8)
    fig.tight_layout()
    return fig


def plot_memory_kernels(model, length: int = 45):
    kf = model.memory.kernel(length, 'fast').detach().cpu().numpy()
    ks = model.memory.kernel(length, 'slow').detach().cpu().numpy()
    tau = model.memory.horizons()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(kf, color='#d62728', lw=2, label=f"fast  K(k)=a(1-a)^k, tau={tau['tau_fast']:.1f}")
    ax.plot(ks, color='#1f77b4', lw=2, label=f"slow  K(k)=a(1-a)^k, tau={tau['tau_slow']:.1f}")
    ax.axhline(0, color='k', lw=.5)
    ax.set_xlabel('lag k (steps into the past)'); ax.set_ylabel('memory weight K(k)')
    ax.set_title('Two-timescale exponential memory kernels')
    ax.legend(fontsize=8); fig.tight_layout()
    return fig


def run_memory_eval(model, eval_batch, device, max_probe_episodes: int = 400,
                    seed: int = 0) -> Tuple[Dict[str, float], Dict[str, object]]:
    """Full memory eval -> (scalar metrics, {fig_name: matplotlib figure})."""
    env = eval_batch['env_name']
    n_classes = eval_batch['n_cue_classes']
    n = min(max_probe_episodes, eval_batch['obs'].shape[0])
    sub = {k: (v[:n] if torch.is_tensor(v) else v) for k, v in eval_batch.items()}
    obs = sub['obs']
    cue = sub['cue'].numpy()
    cue_end = int(sub['cue_end'].float().mean())
    reveal = int(sub['reveal'].float().mean())

    feats = extract_timewise(model, obs, device)
    is_ema = getattr(model, 'memory_impl', 'ema') == 'ema'
    tau = model.horizons() if hasattr(model, 'horizons') else model.memory.horizons()
    metrics: Dict[str, float] = dict(tau)
    figs: Dict[str, object] = {}
    if is_ema:
        figs['memory_kernels'] = plot_memory_kernels(model)

    if n_classes > 1:
        stream_names = [s for s in ['z', 'm_fast', 'm_slow'] if s in feats]
        streams = {name: probe_cue_over_time(feats[name], cue, n_classes, seed=seed)
                   for name in stream_names}
        L = streams['z'].shape[0]
        # Decision-relevant time = the last DELAY step (just before the event appears).
        dt = int(np.clip(reveal - 1, max(cue_end, 0), L - 1))
        metrics['decision_time'] = float(dt)
        for name, acc in streams.items():
            metrics[f'acc_{name}_delay'] = float(acc[dt])
        if 'm_slow' in streams:
            figs['probe_cue_over_time'] = plot_probe_curves(streams, cue_end, reveal, n_classes, env, tau)
            metrics['long_mem_advantage'] = float(streams['m_slow'][dt] - streams['z'][dt])
            metrics['short_mem_advantage'] = float(streams['m_fast'][dt] - streams['z'][dt])
        metrics.update(decision_uses_memory(model, sub, feats, device, seed=seed))

    # causal influence of each bank on the prediction (USAGE)
    infl = model.memory_influence(obs.to(device), sub['actions'].to(device))
    metrics['infl_fast'] = float(infl['infl_fast'].mean())
    metrics['infl_slow'] = float(infl['infl_slow'].mean())
    return metrics, figs
