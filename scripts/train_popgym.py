"""Train MemoryLeWorldModel on a POPGym Arcade POMDP (standard memory benchmark).

These envs have no clean exposed cue label, so we evaluate memory by what it does to the
world model itself: (a) next-latent validation MSE (vanilla `none` vs memory), and
(b) the memory-ablation influence ||f(full) - f(ablate bank)|| on the prediction.
wandb project: lewm-memory-popgym; run name lewm-<EnvId>-<design>-s<seed>.
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from lewm.models.memory_model import MemoryLeWorldModel
from lewm.data import PopgymDataset
from lewm.eval.memory_probe import plot_memory_kernels


def run_epoch(model, loader, opt, device, train, use_amp):
    model.train(train)
    tot = {'loss': 0.0, 'pred_loss': 0.0, 'sigreg_loss': 0.0}; n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for obs, act in loader:
            obs = obs.to(device, non_blocking=True); act = act.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    losses = model.compute_loss(obs, act)
            else:
                losses = model.compute_loss(obs, act)
            if train:
                opt.zero_grad(set_to_none=True); losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            for k in tot: tot[k] += float(losses[k].detach())
            n += 1
    return {k: v / max(n, 1) for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env-id', required=True)
    p.add_argument('--memory-mode', default='both', choices=['none', 'short', 'long', 'both'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output-dir', default='outputs/popgym')
    p.add_argument('--num-episodes', type=int, default=4000)
    p.add_argument('--val-episodes', type=int, default=512)
    p.add_argument('--length', type=int, default=32)
    p.add_argument('--img-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--patch-size', type=int, default=8)
    p.add_argument('--embed-dim', type=int, default=128)
    p.add_argument('--encoder-layers', type=int, default=6)
    p.add_argument('--encoder-heads', type=int, default=4)
    p.add_argument('--predictor-layers', type=int, default=4)
    p.add_argument('--predictor-heads', type=int, default=8)
    p.add_argument('--history-len', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--sigreg-lambda', type=float, default=0.1)
    p.add_argument('--sigreg-projections', type=int, default=512)
    p.add_argument('--tau-fast', type=float, default=3.0)
    p.add_argument('--tau-slow', type=float, default=25.0)
    p.add_argument('--fixed-alpha', action='store_true')
    p.add_argument('--wandb', dest='wandb', action='store_true', default=True)
    p.add_argument('--no-wandb', dest='wandb', action='store_false')
    p.add_argument('--wandb-project', default='lewm-memory-popgym')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'

    run_name = f"lewm-{args.env_id}-{args.memory_mode}-s{args.seed}"
    out_dir = Path(args.output_dir) / run_name; out_dir.mkdir(parents=True, exist_ok=True)

    # data: FIXED data seeds (decoupled from model seed) so it is collected once per env and
    # shared across model-init seeds. Pre-collect before launching parallel runs (see run_popgym.sh)
    # so the training process never imports JAX (avoids the fork/threading hazard with DataLoader).
    train_ds = PopgymDataset(args.env_id, args.num_episodes, args.length, args.img_size, seed=0)
    val_ds = PopgymDataset(args.env_id, args.val_episodes, args.length, args.img_size, seed=7777)
    n_actions = train_ds.n_actions

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=run_name, group=args.env_id,
                        job_type=args.memory_mode,
                        tags=[f"env:{args.env_id}", f"design:{args.memory_mode}", "popgym-arcade", "lewm-memory"],
                        config=vars(args) | {'n_actions': n_actions, 'benchmark': 'popgym-arcade'})

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=pin, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=pin, persistent_workers=args.num_workers > 0)

    model = MemoryLeWorldModel(
        img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim, action_dim=n_actions,
        encoder_layers=args.encoder_layers, encoder_heads=args.encoder_heads,
        predictor_layers=args.predictor_layers, predictor_heads=args.predictor_heads,
        history_len=args.history_len, dropout=args.dropout, sigreg_lambda=args.sigreg_lambda,
        sigreg_projections=args.sigreg_projections, memory_mode=args.memory_mode,
        tau_fast=args.tau_fast, tau_slow=args.tau_slow, learnable_alpha=not args.fixed_alpha).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"=== {run_name} | n_actions={n_actions} | params={model.num_parameters():,} | amp={use_amp} ===", flush=True)

    # a fixed val batch for the influence metric
    vb_obs = torch.stack([val_ds[i][0] for i in range(min(256, len(val_ds)))])
    vb_act = torch.stack([val_ds[i][1] for i in range(min(256, len(val_ds)))])

    best = {}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, opt, device, True, use_amp)
        va = run_epoch(model, val_loader, opt, device, False, use_amp)
        tau = model.memory.horizons()
        print(f"e{epoch:3d}/{args.epochs} ({time.time()-t0:.1f}s) train {tr['loss']:.4f}(pred {tr['pred_loss']:.4f}) "
              f"| val pred {va['pred_loss']:.4f} | tau_f={tau['tau_fast']:.1f} tau_s={tau['tau_slow']:.1f}", flush=True)
        if wb is not None:
            wb.log({'train/loss': tr['loss'], 'train/pred_loss': tr['pred_loss'],
                    'val/loss': va['loss'], 'val/pred_loss': va['pred_loss'],
                    'mem/tau_fast': tau['tau_fast'], 'mem/tau_slow': tau['tau_slow']}, step=epoch)

    # final eval: influence of each memory bank on the prediction
    infl = model.memory_influence(vb_obs.to(device), vb_act.to(device))
    tau = model.memory.horizons()
    best = {'env': args.env_id, 'design': args.memory_mode, 'n_actions': n_actions,
            'val_pred_loss': va['pred_loss'], 'infl_fast': float(infl['infl_fast'].mean()),
            'infl_slow': float(infl['infl_slow'].mean()), **tau}
    fig = plot_memory_kernels(model); fp = out_dir / 'memory_kernels.png'
    fig.savefig(fp, dpi=100, bbox_inches='tight')
    if wb is not None:
        import wandb, matplotlib.pyplot as plt
        wb.log({'eval/memory_kernels': wandb.Image(str(fp)), **{f'eval/{k}': v for k, v in best.items() if isinstance(v, (int, float))}})
        plt.close(fig)
    torch.save({'model_state_dict': model.state_dict(), 'args': vars(args), 'final_metrics': best}, out_dir / 'model.pt')
    json.dump(best, open(out_dir / 'metrics.json', 'w'), indent=2)
    print(f"=== done {run_name}: val_pred={best['val_pred_loss']:.4f} infl_fast={best['infl_fast']:.3f} "
          f"infl_slow={best['infl_slow']:.3f} ===", flush=True)
    if wb is not None:
        wb.summary.update(best); wb.finish()


if __name__ == '__main__':
    main()
