"""
Train MemoryLeWorldModel on a memory-stressing environment, with wandb logging.

One run = one (env, design) cell of the ablation matrix. The design is the memory mode:
  none  : baseline memoryless JEPA (control)
  short : fast EMA bank only
  long  : slow EMA bank only
  both  : fast + slow

Run names and tags are chosen so the env and the design are trivially separable in wandb:
  name  = lewm-<env>-<design>-s<seed>
  group = <env>          (groups all designs of one env together)
  job_type = <design>
  tags  = ["env:<env>", "design:<design>", "kind:<memory-kind>", "lewm-memory"]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from lewm.models.memory_model import MemoryLeWorldModel
from lewm.data import MemoryEpisodeDataset, generate_eval_batch
from lewm.envs.memory_envs import ENV_MEMORY_KIND, ENV_REGISTRY
from lewm.eval.memory_probe import run_memory_eval


def build_model(args, device) -> MemoryLeWorldModel:
    # non-EMA modes select a memory implementation; none/short/long/both use the EMA impl.
    impl = args.memory_mode if args.memory_mode in ('multi', 'gru', 'ssm', 'retrieval', 'smt') else 'ema'
    ema_mode = 'both' if impl != 'ema' else args.memory_mode
    model = MemoryLeWorldModel(
        img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
        action_dim=2, encoder_layers=args.encoder_layers, encoder_heads=args.encoder_heads,
        predictor_layers=args.predictor_layers, predictor_heads=args.predictor_heads,
        history_len=args.history_len, dropout=args.dropout,
        sigreg_lambda=args.sigreg_lambda, sigreg_projections=args.sigreg_projections,
        memory_mode=ema_mode, tau_fast=args.tau_fast, tau_slow=args.tau_slow,
        learnable_alpha=not args.fixed_alpha,
        memory_impl=impl, multi_taus=tuple(args.multi_taus), encoder_type=args.encoder,
        smt_router=getattr(args, 'smt_router', 'softmax'),
    ).to(device)
    return model


def run_epoch(model, loader, optimizer, device, train: bool, use_amp: bool) -> Dict[str, float]:
    model.train(train)
    tot = {'loss': 0.0, 'pred_loss': 0.0, 'sigreg_loss': 0.0}
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for obs, act in loader:
            obs = obs.to(device, non_blocking=True)
            act = act.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    losses = model.compute_loss(obs, act)
            else:
                losses = model.compute_loss(obs, act)
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            for k in tot:
                tot[k] += float(losses[k].detach())
            n += 1
    return {k: v / max(n, 1) for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser()
    # experiment identity
    p.add_argument('--env', required=True, choices=list(ENV_REGISTRY))
    p.add_argument('--memory-mode', default='both',
                   choices=['none', 'short', 'long', 'both', 'multi', 'gru', 'ssm', 'retrieval', 'smt'])
    p.add_argument('--multi-taus', type=float, nargs='+', default=[2, 4, 8, 16, 32, 64])
    p.add_argument('--smt-router', default='softmax', choices=['softmax', 'sigmoid'],
                   help="SMT read-out: softmax mixture (v1) or independent additive sigmoid gates (v2)")
    p.add_argument('--freeze-encoder', action='store_true', help='freeze the encoder (train only memory+predictor)')
    p.add_argument('--init-from', default=None, help='load encoder weights from this checkpoint (for frozen-backbone)')
    p.add_argument('--encoder', default='vit', choices=['vit', 'dino'], help="encoder backbone ('dino' = frozen DINOv2)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output-dir', default='outputs/mem')
    p.add_argument('--run-suffix', default='', help='appended to run name + output dir (for sweeps)')
    p.add_argument('--extra-tag', default='', help='comma-separated extra wandb tags (e.g. exp:tau_slow_sweep)')
    # env overrides (passed through to the env generator when set; e.g. tmaze gap sweep)
    p.add_argument('--reveal', type=int, default=None)
    p.add_argument('--cue-len', type=int, default=None)
    # data / episodes
    p.add_argument('--num-episodes', type=int, default=6000)
    p.add_argument('--val-episodes', type=int, default=512)
    p.add_argument('--length', type=int, default=32)
    p.add_argument('--img-size', type=int, default=64)
    # training
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--num-workers', type=int, default=6)
    p.add_argument('--no-amp', action='store_true')
    # model
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
    # memory
    p.add_argument('--tau-fast', type=float, default=2.0)
    p.add_argument('--tau-slow', type=float, default=20.0)
    p.add_argument('--fixed-alpha', action='store_true', help='freeze EMA rates (known horizons)')
    # eval / logging
    p.add_argument('--eval-interval', type=int, default=10)
    p.add_argument('--probe-episodes', type=int, default=400)
    p.add_argument('--wandb', dest='wandb', action='store_true', default=True)
    p.add_argument('--no-wandb', dest='wandb', action='store_false')
    p.add_argument('--wandb-project', default='lewm-memory')
    p.add_argument('--wandb-entity', default=None)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'

    kind = ENV_MEMORY_KIND[args.env]
    run_name = f"lewm-{args.env}-{args.memory_mode}-s{args.seed}"
    if args.run_suffix:
        run_name += f"-{args.run_suffix}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # env kwargs passed through only when explicitly set (keeps defaults otherwise)
    env_kwargs = {}
    if args.reveal is not None:
        env_kwargs['reveal'] = args.reveal
    if args.cue_len is not None:
        env_kwargs['cue_len'] = args.cue_len

    # ---- wandb ----
    wb = None
    if args.wandb:
        import wandb
        tags = [f"env:{args.env}", f"design:{args.memory_mode}", f"kind:{kind}", "lewm-memory"]
        if args.extra_tag:
            tags += [t.strip() for t in args.extra_tag.split(',') if t.strip()]
        wb = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, name=run_name,
            group=args.env, job_type=args.memory_mode, tags=tags,
            config=vars(args) | {'memory_kind': kind},
        )

    # ---- data ----
    train_ds = MemoryEpisodeDataset(args.env, args.num_episodes, img_size=args.img_size,
                                    length=args.length, seed=args.seed, **env_kwargs)
    val_ds = MemoryEpisodeDataset(args.env, args.val_episodes, img_size=args.img_size,
                                  length=args.length, seed=args.seed + 99991, **env_kwargs)
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin,
                            persistent_workers=args.num_workers > 0)
    eval_batch = generate_eval_batch(args.env, args.probe_episodes, img_size=args.img_size,
                                     length=args.length, seed=args.seed + 7, **env_kwargs)

    # ---- model / optim ----
    model = build_model(args, device)
    if args.init_from:                                  # frozen-backbone: load a pretrained encoder
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        enc_sd = {k[len('encoder.'):]: v for k, v in ck['model_state_dict'].items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(enc_sd); print(f"  loaded encoder from {args.init_from}")
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        print("  encoder FROZEN (training memory+predictor only)")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)

    print(f"=== {run_name} | kind={kind} | params={model.num_parameters():,} | "
          f"amp={use_amp} | device={device} ===", flush=True)

    def do_eval(epoch):
        metrics, figs = run_memory_eval(model, eval_batch, device,
                                        max_probe_episodes=args.probe_episodes, seed=args.seed)
        # save figures locally + log to wandb
        for name, fig in figs.items():
            fp = out_dir / f"{name}_e{epoch}.png"
            fig.savefig(fp, dpi=100, bbox_inches='tight')
            if wb is not None:
                import wandb
                wb.log({f"eval/{name}": wandb.Image(str(fp))}, step=epoch)
            import matplotlib.pyplot as plt
            plt.close(fig)
        if wb is not None:
            wb.log({f"eval/{k}": v for k, v in metrics.items()}, step=epoch)
        msg = " ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                       if k in ('long_mem_advantage', 'cue_acc_from_prediction',
                                'acc_m_slow_delay', 'acc_z_delay', 'infl_slow'))
        print(f"  [eval e{epoch}] {msg}", flush=True)
        return metrics

    best = {}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, device, True, use_amp)
        va = run_epoch(model, val_loader, optimizer, device, False, use_amp)
        tau = model.horizons()
        dt = time.time() - t0
        print(f"e{epoch:3d}/{args.epochs} ({dt:.1f}s) "
              f"train {tr['loss']:.4f} (pred {tr['pred_loss']:.4f}) | "
              f"val {va['loss']:.4f} (pred {va['pred_loss']:.4f}) | "
              f"tau_f={tau['tau_fast']:.1f} tau_s={tau['tau_slow']:.1f}", flush=True)
        if wb is not None:
            wb.log({
                'train/loss': tr['loss'], 'train/pred_loss': tr['pred_loss'],
                'train/sigreg_loss': tr['sigreg_loss'],
                'val/loss': va['loss'], 'val/pred_loss': va['pred_loss'],
                'val/sigreg_loss': va['sigreg_loss'],
                'mem/tau_fast': tau['tau_fast'], 'mem/tau_slow': tau['tau_slow'],
                'mem/alpha_fast': tau['alpha_fast'], 'mem/alpha_slow': tau['alpha_slow'],
                'time/epoch_s': dt,
            }, step=epoch)
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            best = do_eval(epoch)

    # ---- save ----
    ckpt = {'model_state_dict': model.state_dict(), 'args': vars(args), 'final_metrics': best}
    torch.save(ckpt, out_dir / 'model.pt')
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'run': run_name, 'env': args.env, 'design': args.memory_mode,
                   'kind': kind, **best}, f, indent=2)
    print(f"=== done {run_name} -> {out_dir} ===", flush=True)
    if wb is not None:
        wb.summary.update(best)
        wb.finish()


if __name__ == '__main__':
    main()
