"""Visualize what a (memoryless vs memory) world model 'sees' the robot doing through an
occlusion, and log it to wandb so you can watch the rollout.

JEPA models predict in *latent* space and have no pixel decoder, so we train a small probe
decoder (latent -> 64x64 RGB) on each model's own frozen encoder, then decode the model's
one-step next-latent predictions on the occluded input. We render a 4-panel video per episode:

    [ TRUE ROBOT (un-occluded) | MODEL INPUT (occluded) | MEMORYLESS pred | MEMORY pred ]

During the blackout the memoryless model has nothing to go on; the memory (K-bank) model
carries the pre-occlusion state and, with the known actions, keeps tracking the robot.
The decoder is for *visualization only* — it is not part of the world model.
"""
import argparse, os, sys
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from lewm.models.memory_model import MemoryLeWorldModel
from lewm.envs.popgym_arcade import get_or_collect

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(a, n_actions):
    mode = a['memory_mode']
    impl = mode if mode in ('multi', 'gru', 'ssm', 'retrieval') else 'ema'
    ema_mode = 'both' if impl != 'ema' else mode
    return MemoryLeWorldModel(
        img_size=a['img_size'], patch_size=a['patch_size'], embed_dim=a['embed_dim'],
        action_dim=n_actions, encoder_layers=a['encoder_layers'], encoder_heads=a['encoder_heads'],
        predictor_layers=a['predictor_layers'], predictor_heads=a['predictor_heads'],
        history_len=a['history_len'], dropout=a['dropout'], sigreg_lambda=a['sigreg_lambda'],
        sigreg_projections=a['sigreg_projections'], memory_mode=ema_mode, memory_impl=impl,
        tau_fast=a['tau_fast'], tau_slow=a['tau_slow'], learnable_alpha=not a.get('fixed_alpha', True))


class Decoder(nn.Module):
    """latent (D,) -> 64x64x3, visualization-only probe."""
    def __init__(self, D):
        super().__init__()
        self.fc = nn.Linear(D, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),  # 8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),    # 16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),     # 32
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())                        # 64

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 256, 4, 4))


def obs_to_tensor(obs_np):
    return torch.from_numpy(obs_np.astype(np.float32) / 255.0).permute(0, 1, 4, 2, 3).contiguous()


@torch.no_grad()
def encode_all(model, obs_t, bs=64):
    zs = []
    for i in range(0, obs_t.shape[0], bs):
        z, _, _, _ = model.encode_with_memory(obs_t[i:i + bs].to(DEV))
        zs.append(z.cpu())
    return torch.cat(zs)                                    # (N, L, D)


def train_decoder(model, obs_t, D, epochs, lr=1e-3, bs=256):
    """Fit Dec(encoder latent) -> frame on this model's frozen encoder."""
    z = encode_all(model, obs_t).reshape(-1, D)             # (N*L, D)
    y = obs_t.reshape(-1, *obs_t.shape[2:])                 # (N*L, 3, H, W)
    dec = Decoder(D).to(DEV)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    idx = np.arange(len(z))
    for ep in range(epochs):
        np.random.default_rng(ep).shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            zb = z[b].to(DEV); yb = y[b].to(DEV)
            pred = dec(zb)
            loss = ((pred - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        print(f"    dec epoch {ep+1}/{epochs} mse={tot/len(idx):.4f}", flush=True)
    return dec.eval()


@torch.no_grad()
def one_step_preds(model, obs_t, act_t):
    """Teacher-forced next-latent prediction at each t in [h, L). Returns (N, L-h, D)."""
    h = model.history_len
    _, _, _, zt = model.encode_with_memory(obs_t.to(DEV))   # fused window latents (N,L,D)
    L = zt.shape[1]
    preds = []
    for t in range(h, L):
        p = model.predictor(zt[:, t - h:t], act_t[:, t - h:t].to(DEV))[:, -1, :]
        preds.append(p)
    return torch.stack(preds, dim=1)                        # (N, L-h, D)


@torch.no_grad()
def decode(dec, lat):                                       # (N,T,D) -> (N,T,3,H,W) uint8
    N, T, D = lat.shape
    img = dec(lat.reshape(-1, D).to(DEV)).clamp(0, 1).cpu()
    return (img.reshape(N, T, 3, 64, 64).permute(0, 1, 3, 4, 2).numpy() * 255).astype(np.uint8)


def label(img, text, color=(255, 255, 255)):
    cv2.putText(img, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', required=True, help='e.g. reacher.hard')
    ap.add_argument('--robotic-dir', default='outputs/robotic')
    ap.add_argument('--out-dir', default='outputs/robotic_viz')
    ap.add_argument('--wandb-project', default='lewm-memory-robotic-viz')
    ap.add_argument('--n-eps', type=int, default=6)
    ap.add_argument('--dec-epochs', type=int, default=8)
    ap.add_argument('--dec-train-eps', type=int, default=200)
    ap.add_argument('--viz-seed', type=int, default=31337)
    ap.add_argument('--no-wandb', action='store_true')
    args = ap.parse_args()

    robot = args.robot
    env_occ, env_full = f'dmc:{robot}.occ', f'dmc:{robot}'
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # viz episodes: occluded (model input) + full (true robot), same seed -> aligned dynamics
    occ_obs, occ_act, n_act = get_or_collect(env_occ, args.n_eps, 32, seed=args.viz_seed)
    full_obs, _, _ = get_or_collect(env_full, args.n_eps, 32, seed=args.viz_seed)
    # decoder training frames (clean, from the cached training distribution)
    dec_obs, _, _ = get_or_collect(env_full, args.dec_train_eps, 32, seed=0)
    occ_t = obs_to_tensor(occ_obs); full_t = obs_to_tensor(full_obs); dec_t = obs_to_tensor(dec_obs)
    act_t = torch.zeros(args.n_eps, 31, n_act)
    act_t.scatter_(2, torch.from_numpy(occ_act.astype(np.int64)).unsqueeze(-1), 1.0)

    L = 32; occ_s = L // 3; occ_e = min(L, occ_s + max(4, L // 5))  # blackout window (match dmc_collect)
    decoded = {}; err_curve = {}
    for design in ['none', 'multi']:
        ckp = Path(args.robotic_dir) / f'lewm-{env_occ}-{design}-s0' / 'model.pt'
        ck = torch.load(ckp, map_location=DEV, weights_only=False)
        m = build_model(ck['args'], n_act).to(DEV); m.load_state_dict(ck['model_state_dict']); m.eval()
        D = ck['args']['embed_dim']; h = ck['args']['history_len']
        print(f"  [{design}] training viz decoder...", flush=True)
        dec = train_decoder(m, dec_t, D, args.dec_epochs)
        preds = one_step_preds(m, occ_t, act_t)            # (N, L-h, D) pred on OCCLUDED input
        decoded[design] = decode(dec, preds)               # (N, L-h, H, W, 3)
        # prediction-vs-TRUE-robot error over time (latent space; decode-independent)
        true_lat = encode_all(m, full_t)                   # (N, L, D) latent of the un-occluded robot
        err = (preds.cpu() - true_lat[:, h:]).norm(dim=-1).mean(0).numpy()   # (L-h,)
        base = err[:max(1, occ_s - h)].mean()              # pre-occlusion baseline
        err_curve[design] = err / (base + 1e-8)            # normalized so both start ~1.0
    h = ck['args']['history_len']

    runs = None
    if not args.no_wandb:
        import wandb
        runs = wandb.init(project=args.wandb_project, name=f'rollout-{robot}',
                          tags=['exp:robotic_viz', f'robot:{robot}'], reinit=True)

    up = 3                                                   # upscale for visibility
    montages = []
    for e in range(args.n_eps):
        frames = []
        for j, t in enumerate(range(h, L)):                # predicted timesteps
            cols = [full_obs[e, t], occ_obs[e, t], decoded['none'][e, j], decoded['multi'][e, j]]
            names = ['TRUE robot', 'model input', 'memoryless', 'MEMORY (K-bank)']
            tiles = []
            occluded_now = occ_s <= t < occ_e
            for img, nm in zip(cols, names):
                im = cv2.resize(img.astype(np.uint8), (64 * up, 64 * up), interpolation=cv2.INTER_NEAREST).copy()
                col = (60, 60, 255) if (occluded_now and nm != 'TRUE robot') else (255, 255, 255)
                label(im, nm, col)
                if occluded_now and nm != 'TRUE robot':
                    cv2.rectangle(im, (0, 0), (64 * up - 1, 64 * up - 1), (60, 60, 255), 2)
                tiles.append(im)
                tiles.append(np.full((64 * up, 2, 3), 30, np.uint8))   # separator
            frames.append(np.concatenate(tiles[:-1], axis=1))
        vid = np.stack(frames)                              # (T, H, W, 3)
        mid_occ_j = (occ_s + occ_e) // 2 - h               # a frame DURING the blackout
        montages.append(vid[np.clip(mid_occ_j, 0, len(vid) - 1)])
        if runs is not None:
            import wandb
            runs.log({f'rollout/ep{e}': wandb.Video(vid.transpose(0, 3, 1, 2), fps=4, format='mp4')})
        cv2.imwrite(str(out / f'{robot}_ep{e}.png'), cv2.cvtColor(vid[np.clip(mid_occ_j, 0, len(vid) - 1)], cv2.COLOR_RGB2BGR))
    # one combined montage png for quick inspection
    grid = np.concatenate(montages, axis=0)
    cv2.imwrite(str(out / f'{robot}_montage.png'), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    # prediction-vs-true-robot error over time (the decode-independent, quantitative view)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ts = np.arange(h, L)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.axvspan(occ_s, occ_e - 1, color='0.85', label='occluded (blackout)')
    ax.plot(ts, err_curve['none'], '-o', ms=3, color='#9ecae1', label='memoryless (none)')
    ax.plot(ts, err_curve['multi'], '-o', ms=3, color='#3182bd', label='MEMORY (K-bank)')
    ax.set_xlabel('time step'); ax.set_ylabel('pred-vs-true error\n(normalized to pre-occlusion)')
    ax.set_title(f'{robot}: predicting the robot through the blackout')
    ax.legend(fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout(); plt.savefig(str(out / f'{robot}_errcurve.png'), dpi=150); plt.close(fig)

    if runs is not None:
        import wandb
        # true robot episode videos (so you can see the actual robots + the blackout)
        for e in range(min(args.n_eps, 3)):
            tv = np.stack([cv2.resize(full_obs[e, t], (192, 192), interpolation=cv2.INTER_NEAREST)
                           for t in range(L)])
            runs.log({f'true_robot/ep{e}': wandb.Video(tv.transpose(0, 3, 1, 2), fps=4, format='mp4')})
        runs.log({'montage': wandb.Image(str(out / f'{robot}_montage.png'),
                                         caption='cols: TRUE | input | memoryless | MEMORY (red=occluded)'),
                  'pred_error_over_time': wandb.Image(str(out / f'{robot}_errcurve.png'))})
        runs.finish()
    print(f"done {robot}: wrote {out}/{robot}_*.png", flush=True)


if __name__ == '__main__':
    main()
