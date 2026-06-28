"""Train MemoryLeWorldModel on a POPGym Arcade POMDP (standard memory benchmark).

These envs have no clean exposed cue label, so we evaluate memory by what it does to the
world model itself: (a) next-latent validation MSE (vanilla `none` vs memory), and
(b) the memory-ablation influence ||f(full) - f(ablate bank)|| on the prediction.
wandb project: lewm-memory-popgym; run name lewm-<EnvId>-<design>-s<seed>.
"""
import os, sys, json, time, argparse, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from lewm.models.memory_model import MemoryLeWorldModel
from lewm.data import PopgymDataset, PrecomputedFeatureDataset
from lewm.eval.memory_probe import plot_memory_kernels


def run_epoch(model, loader, opt, device, train, use_amp, first_post_loss_weight=0.0):
    model.train(train)
    if not any(p.requires_grad for p in model.encoder.parameters()):
        model.encoder.eval()  # a frozen shared target must also have deterministic dropout/norm behavior
    metric_names = (
        'loss', 'pred_loss', 'sigreg_loss', 'pred_loss_all_valid',
        'pred_loss_first_post', 'hier_loss', 'hier_loss_fast',
        'hier_loss_medium', 'hier_loss_slow', 'hier_loss_h1', 'hier_loss_h2',
        'hier_loss_h4', 'hier_loss_h8', 'hier_loss_h16',
    )
    tot = {name: 0.0 for name in metric_names}; counts = {name: 0 for name in metric_names}
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            obs, act = batch[:2]
            target_obs = batch[2] if len(batch) == 3 else None
            target_mask = batch[3] if len(batch) == 4 else None
            if len(batch) == 4:
                target_obs = batch[2]
            obs = obs.to(device, non_blocking=True); act = act.to(device, non_blocking=True)
            if target_obs is not None:
                target_obs = target_obs.to(device, non_blocking=True)
            if target_mask is not None:
                target_mask = target_mask.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    losses = model.compute_loss(
                        obs, act, target_obs, target_mask,
                        memory_update_mask=target_mask,
                        first_post_loss_weight=first_post_loss_weight)
            else:
                losses = model.compute_loss(
                    obs, act, target_obs, target_mask,
                    memory_update_mask=target_mask,
                    first_post_loss_weight=first_post_loss_weight)
            if train:
                opt.zero_grad(set_to_none=True); losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            batch_n = obs.shape[0]
            for name in metric_names:
                if name in losses:
                    tot[name] += float(losses[name].detach()) * batch_n
                    counts[name] += batch_n
            n += batch_n
    return {name: tot[name] / max(counts[name], 1)
            for name in metric_names if counts[name]}


@torch.no_grad()
def phase_mse(model, loader, device, use_amp, length, history_len, constant_latent):
    """Common-target MSE by occlusion phase, with a paired all-memory ablation.

    This metric is defined only for loaders that return synchronized target observations.
    Prediction time ``t`` is the target frame index for each sliding window.
    """
    model.eval()
    if constant_latent is None:
        raise ValueError('phase_mse requires a clean-train constant-target baseline')
    sums = None; sums_ablate = None; sums_constant = None; sums_persistence = None
    sums_last_visible = None; sums_clean_input = None; count = 0
    for batch in loader:
        if len(batch) not in (3, 4):
            raise ValueError('phase_mse requires paired target observations')
        obs, act, target_obs = (x.to(device, non_blocking=True) for x in batch[:3])
        target_mask = (batch[3].to(device, non_blocking=True)
                       if len(batch) == 4 else None)
        amp_ctx = torch.autocast('cuda', dtype=torch.bfloat16) if use_amp else torch.no_grad()
        with amp_ctx:
            z = model.encode(obs)
            z_target = model.encode(target_obs)
            z_full = model._inject(z, actions=act, memory_update_mask=target_mask)
            clean_update_mask = (torch.ones_like(target_mask)
                                 if target_mask is not None else None)
            z_clean_full = model._inject(
                z_target, actions=act, memory_update_mask=clean_update_mask)
            B, L, D = z.shape; h = history_len; W = L - h
            full_win = z_full.unfold(1, h, 1)[:, :W].permute(0, 1, 3, 2).reshape(B * W, h, D)
            raw_win = z.unfold(1, h, 1)[:, :W].permute(0, 1, 3, 2).reshape(B * W, h, D)
            clean_win = z_clean_full.unfold(1, h, 1)[:, :W].permute(0, 1, 3, 2).reshape(B * W, h, D)
            act_win = act.unfold(1, h, 1)[:, :W].permute(0, 1, 3, 2).reshape(B * W, h, -1)
            target = z_target[:, h:L]
            pred = model.predictor(full_win, act_win)[:, -1].reshape(B, W, D)
            pred_ablate = model.predictor(raw_win, act_win)[:, -1].reshape(B, W, D)
            pred_clean_input = model.predictor(clean_win, act_win)[:, -1].reshape(B, W, D)
            pred_constant = constant_latent.to(device=device, dtype=z_target.dtype).view(1, 1, D)
            pred_persistence = z[:, h - 1:L - 1]
            target_times_t = torch.arange(h, L, device=device)
            visible = (torch.arange(L, device=device) < L // 3) | (
                torch.arange(L, device=device) >= min(L, L // 3 + max(4, L // 5)))
            last_visible_indices = []
            for target_time in target_times_t.tolist():
                candidates = torch.nonzero(visible[:target_time], as_tuple=False).flatten()
                last_visible_indices.append(int(candidates[-1]))
            pred_last_visible = z[:, torch.tensor(last_visible_indices, device=device)]
            err = (pred - target).float().square().mean(-1).sum(0).cpu().numpy()
            err_ablate = (pred_ablate - target).float().square().mean(-1).sum(0).cpu().numpy()
            err_constant = (pred_constant - target).float().square().mean(-1).sum(0).cpu().numpy()
            err_persistence = (pred_persistence - target).float().square().mean(-1).sum(0).cpu().numpy()
            err_last_visible = (pred_last_visible - target).float().square().mean(-1).sum(0).cpu().numpy()
            err_clean_input = (pred_clean_input - target).float().square().mean(-1).sum(0).cpu().numpy()
        sums = err if sums is None else sums + err
        sums_ablate = err_ablate if sums_ablate is None else sums_ablate + err_ablate
        sums_constant = err_constant if sums_constant is None else sums_constant + err_constant
        sums_persistence = err_persistence if sums_persistence is None else sums_persistence + err_persistence
        sums_last_visible = err_last_visible if sums_last_visible is None else sums_last_visible + err_last_visible
        sums_clean_input = err_clean_input if sums_clean_input is None else sums_clean_input + err_clean_input
        count += B
    if count == 0:
        raise ValueError('empty validation loader')
    per_t = sums / count; per_t_ablate = sums_ablate / count
    per_t_constant = sums_constant / count; per_t_persistence = sums_persistence / count
    per_t_last_visible = sums_last_visible / count; per_t_clean_input = sums_clean_input / count
    target_t = np.arange(history_len, length)
    occ_start = length // 3
    occ_end = min(length, occ_start + max(4, length // 5))
    deep_start = min(occ_end, occ_start + history_len)
    late_start = min(length, occ_end + history_len)
    masks = {
        'pre': target_t < occ_start,
        'blackout_transition': (target_t >= occ_start) & (target_t < deep_start),
        'deep_blackout': (target_t >= deep_start) & (target_t < occ_end),
        'first_post': target_t == occ_end,
        'recovery': (target_t > occ_end) & (target_t < late_start),
        'late_post': target_t >= late_start,
    }
    out = {}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f'no prediction windows in phase {name}')
        out[f'clean_mse_{name}'] = float(per_t[mask].mean())
        out[f'clean_mse_{name}_ablated'] = float(per_t_ablate[mask].mean())
        out[f'constant_mse_{name}'] = float(per_t_constant[mask].mean())
        out[f'persistence_mse_{name}'] = float(per_t_persistence[mask].mean())
        out[f'last_visible_mse_{name}'] = float(per_t_last_visible[mask].mean())
        out[f'clean_input_mse_{name}'] = float(per_t_clean_input[mask].mean())
    out['clean_mse_all'] = float(per_t.mean())
    out['clean_mse_all_ablated'] = float(per_t_ablate.mean())
    out['constant_mse_all'] = float(per_t_constant.mean())
    out['persistence_mse_all'] = float(per_t_persistence.mean())
    out['last_visible_mse_all'] = float(per_t_last_visible.mean())
    out['clean_input_mse_all'] = float(per_t_clean_input.mean())
    out['prediction_target_times'] = target_t.tolist()
    out['clean_mse_by_target_t'] = per_t.tolist()
    out['clean_mse_by_target_t_ablated'] = per_t_ablate.tolist()
    out['constant_mse_by_target_t'] = per_t_constant.tolist()
    out['persistence_mse_by_target_t'] = per_t_persistence.tolist()
    out['last_visible_mse_by_target_t'] = per_t_last_visible.tolist()
    out['clean_input_mse_by_target_t'] = per_t_clean_input.tolist()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env-id', required=True)
    p.add_argument('--memory-mode', default='both',
                   choices=['none', 'short', 'long', 'both', 'multi', 'gru', 'ssm', 'retrieval',
                            'smt', 'smtv1', 'smtv2',
                            'smtv3', 'smtv3_static', 'smtv3_old', 'smtv3_oracle',
                            'hacsmv4', 'hacsmv4_static', 'hacsmv4_noaction',
                            'hacsmv4_noaux', 'hacsmv4_single', 'hacsmv4_oracle'])
    p.add_argument('--smt-router', default='softmax',
                   choices=['softmax', 'scaled_softmax', 'sigmoid'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output-dir', default='outputs/popgym')
    p.add_argument('--num-episodes', type=int, default=4000)
    p.add_argument('--val-episodes', type=int, default=512)
    p.add_argument('--data-dir', default='outputs/popgym_data')
    p.add_argument('--prototype-seed', type=int, default=0,
                   help='fixed continuous-action prototype seed for DMC/OGBench; shared by train and val')
    p.add_argument('--target-env-id', default=None,
                   help='synchronized clean target env (for example dmc:reacher.hard for a .occ input)')
    p.add_argument('--mask-occluded-target-loss', action='store_true',
                   help='exclude hidden clean blackout frames from train/val prediction loss')
    p.add_argument('--first-post-loss-weight', type=float, default=0.0,
                   help='weight on the first visible target after the masked interval; '
                        '0 preserves the legacy all-valid loss, 0.5 gives equal all/first-post weight')
    p.add_argument('--encoder-checkpoint', default=None,
                   help='checkpoint whose encoder weights initialize this run')
    p.add_argument('--encoder-stats', default=None,
                   help='clean-train fixed projector statistics for a frozen shared encoder')
    p.add_argument('--freeze-encoder', action='store_true',
                   help='freeze the initialized encoder to define a shared target space')
    p.add_argument('--encoder-type', default='vit', choices=['vit', 'precomputed'])
    p.add_argument('--train-feature-cache', default=None)
    p.add_argument('--val-feature-cache', default=None)
    p.add_argument('--feature-manifest', default=None)
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
    p.add_argument('--predictor-norm', default='batch', choices=['batch', 'layer', 'none'],
                   help='predictor output normalization; layer is independent across '
                        'flattened sliding windows, batch preserves legacy behavior')
    p.add_argument('--history-len', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--sigreg-lambda', type=float, default=0.1)
    p.add_argument('--sigreg-projections', type=int, default=512)
    p.add_argument('--hier-loss-weight', type=float, default=0.1,
                   help='HACSM-v4 action-rollout auxiliary weight; noaux forces effective zero')
    p.add_argument('--tau-fast', type=float, default=3.0)
    p.add_argument('--tau-slow', type=float, default=25.0)
    p.add_argument('--fixed-alpha', action='store_true')
    p.add_argument('--wandb', dest='wandb', action='store_true', default=True)
    p.add_argument('--no-wandb', dest='wandb', action='store_false')
    p.add_argument('--wandb-project', default='lewm-memory-popgym')
    p.add_argument('--extra-tag', default='', help='comma-separated extra wandb tags')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    if not 0.0 <= args.first_post_loss_weight <= 1.0:
        raise ValueError('--first-post-loss-weight must be in [0,1]')
    if not np.isfinite(args.hier_loss_weight) or args.hier_loss_weight < 0:
        raise ValueError('--hier-loss-weight must be non-negative and finite')

    feature_mode = any((args.train_feature_cache, args.val_feature_cache, args.feature_manifest))
    if feature_mode and not all((args.train_feature_cache, args.val_feature_cache, args.feature_manifest)):
        raise ValueError('feature mode requires train cache, val cache, and manifest together')
    fixed_target = feature_mode or (args.freeze_encoder and args.encoder_stats)
    if feature_mode and not args.mask_occluded_target_loss:
        raise ValueError(
            'precomputed paired features always carry the blackout target mask; '
            'pass --mask-occluded-target-loss explicitly')
    if args.target_env_id and not fixed_target:
        raise ValueError(
            '--target-env-id requires precomputed features or a calibrated frozen encoder')
    if args.mask_occluded_target_loss and not (args.target_env_id and fixed_target):
        raise ValueError(
            '--mask-occluded-target-loss requires paired targets plus a fixed feature encoder')
    if feature_mode and args.encoder_type != 'precomputed':
        raise ValueError('feature caches require --encoder-type precomputed')
    if args.encoder_type == 'precomputed' and not feature_mode:
        raise ValueError('--encoder-type precomputed requires feature caches')

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'

    run_name = f"lewm-{args.env_id}-{args.memory_mode}-s{args.seed}"
    out_dir = Path(args.output_dir) / run_name; out_dir.mkdir(parents=True, exist_ok=True)

    # data: FIXED data seeds (decoupled from model seed) so it is collected once per env and
    # shared across model-init seeds. Pre-collect before launching parallel runs (see run_popgym.sh)
    # so the training process never imports JAX (avoids the fork/threading hazard with DataLoader).
    if feature_mode:
        manifest_path = Path(args.feature_manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(f'feature manifest not found: {manifest_path}')
        args.feature_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        train_ds = PrecomputedFeatureDataset(args.train_feature_cache, manifest_path)
        val_ds = PrecomputedFeatureDataset(args.val_feature_cache, manifest_path)
        if train_ds.manifest_sha256 != args.feature_manifest_sha256:
            raise ValueError('train feature cache does not match manifest hash')
        if val_ds.manifest_sha256 != args.feature_manifest_sha256:
            raise ValueError('validation feature cache does not match manifest hash')
        if train_ds.split != 'train' or val_ds.split != 'val':
            raise ValueError('feature-cache split mismatch')
        if not (train_ds.occ_env == val_ds.occ_env == args.env_id and
                train_ds.clean_env == val_ds.clean_env == args.target_env_id):
            raise ValueError('feature-cache environment metadata mismatch')
        if len(train_ds) != args.num_episodes or len(val_ds) != args.val_episodes:
            raise ValueError('feature-cache episode count mismatch')
        if train_ds.features_input.shape[1] != args.length or val_ds.features_input.shape[1] != args.length:
            raise ValueError('feature-cache sequence length mismatch')
        if train_ds.feature_dim != args.embed_dim or val_ds.feature_dim != args.embed_dim:
            raise ValueError('feature-cache dimension does not match --embed-dim')
        if train_ds.n_actions != val_ds.n_actions:
            raise ValueError('feature-cache action-count mismatch')
    else:
        train_ds = PopgymDataset(
            args.env_id, args.num_episodes, args.length, args.img_size, seed=0,
            data_dir=args.data_dir, prototype_seed=args.prototype_seed,
            target_env_id=args.target_env_id,
            mask_occluded_target_loss=args.mask_occluded_target_loss)
        val_ds = PopgymDataset(
            args.env_id, args.val_episodes, args.length, args.img_size, seed=7777,
            data_dir=args.data_dir, prototype_seed=args.prototype_seed,
            target_env_id=args.target_env_id,
            mask_occluded_target_loss=args.mask_occluded_target_loss)
    n_actions = train_ds.n_actions

    wb = None
    if args.wandb:
        import wandb
        tags = [f"env:{args.env_id}", f"design:{args.memory_mode}", "popgym-arcade", "lewm-memory"]
        if args.extra_tag:
            tags += [t.strip() for t in args.extra_tag.split(',') if t.strip()]
        wb = wandb.init(project=args.wandb_project, name=run_name, group=args.env_id,
                        job_type=args.memory_mode, tags=tags,
                        config=vars(args) | {'n_actions': n_actions, 'benchmark': 'popgym-arcade'})

    pin = device.type == 'cuda'
    train_generator = torch.Generator().manual_seed(10_000 + args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=train_generator, num_workers=args.num_workers,
                              pin_memory=pin, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=pin, persistent_workers=args.num_workers > 0)

    if args.memory_mode in ('smtv1', 'smtv2'):
        impl = 'smt'
        args.smt_router = 'softmax' if args.memory_mode == 'smtv1' else 'sigmoid'
    else:
        impl = args.memory_mode if args.memory_mode in (
            'multi', 'gru', 'ssm', 'retrieval', 'smt',
            'smtv3', 'smtv3_static', 'smtv3_old', 'smtv3_oracle',
            'hacsmv4', 'hacsmv4_static', 'hacsmv4_noaction',
            'hacsmv4_noaux', 'hacsmv4_single', 'hacsmv4_oracle') else 'ema'
    ema_mode = 'both' if impl != 'ema' else args.memory_mode
    model = MemoryLeWorldModel(
        img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim, action_dim=n_actions,
        encoder_layers=args.encoder_layers, encoder_heads=args.encoder_heads,
        predictor_layers=args.predictor_layers, predictor_heads=args.predictor_heads,
        predictor_norm=args.predictor_norm,
        history_len=args.history_len, dropout=args.dropout, sigreg_lambda=args.sigreg_lambda,
        sigreg_projections=args.sigreg_projections, memory_mode=ema_mode, memory_impl=impl,
        tau_fast=args.tau_fast, tau_slow=args.tau_slow, learnable_alpha=not args.fixed_alpha,
        smt_router=getattr(args, 'smt_router', 'softmax'),
        hier_loss_weight=args.hier_loss_weight,
        encoder_type=args.encoder_type).to(device)
    constant_latent = (torch.from_numpy(train_ds.constant_target).to(device)
                       if feature_mode else None)
    if args.encoder_checkpoint:
        source = Path(args.encoder_checkpoint)
        if not source.is_file():
            raise FileNotFoundError(f'encoder checkpoint not found: {source}')
        source_ck = torch.load(source, map_location='cpu', weights_only=False)
        encoder_state = {
            k.removeprefix('encoder.'): v for k, v in source_ck['model_state_dict'].items()
            if k.startswith('encoder.')
        }
        if not encoder_state:
            raise ValueError(f'checkpoint has no encoder state: {source}')
        model.encoder.load_state_dict(encoder_state, strict=True)
    if args.encoder_stats:
        if not args.encoder_checkpoint:
            raise ValueError('--encoder-stats requires --encoder-checkpoint')
        stats_path = Path(args.encoder_stats)
        if not stats_path.is_file():
            raise FileNotFoundError(f'encoder statistics not found: {stats_path}')
        stats = torch.load(stats_path, map_location='cpu', weights_only=False)
        args.encoder_stats_sha256 = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        args.encoder_source_size = Path(args.encoder_checkpoint).stat().st_size
        args.encoder_source_mtime_ns = Path(args.encoder_checkpoint).stat().st_mtime_ns
        if int(stats.get('schema_version', -1)) != 2:
            raise ValueError(f'unsupported encoder-statistics schema: {stats_path}')
        if not stats.get('quality_pass'):
            raise ValueError(f'encoder statistics failed representation quality gate: {stats_path}')
        if Path(stats.get('source_checkpoint', '')).resolve() != Path(args.encoder_checkpoint).resolve():
            raise ValueError(f'encoder-statistics source mismatch: {stats_path}')
        if args.target_env_id is not None and stats.get('clean_env') != args.target_env_id:
            raise ValueError(
                f"encoder-statistics clean env {stats.get('clean_env')!r} != target {args.target_env_id!r}")
        mean = torch.as_tensor(stats.get('pre_bn_mean'))
        var = torch.as_tensor(stats.get('pre_bn_var'))
        if mean.shape != (args.embed_dim,) or var.shape != (args.embed_dim,):
            raise ValueError(f'encoder-statistics shape mismatch: {stats_path}')
        if not torch.isfinite(mean).all() or not torch.isfinite(var).all() or (var < 0).any():
            raise ValueError(f'encoder statistics are non-finite or have negative variance: {stats_path}')
        old_bn = model.encoder.projector[1]
        if not isinstance(old_bn, torch.nn.BatchNorm1d):
            raise TypeError(f'expected BatchNorm1d encoder projector, got {type(old_bn).__name__}')
        fixed_bn = torch.nn.BatchNorm1d(
            args.embed_dim, eps=old_bn.eps, momentum=0.0, affine=True,
            track_running_stats=True, device=device)
        with torch.no_grad():
            fixed_bn.weight.copy_(old_bn.weight)
            fixed_bn.bias.copy_(old_bn.bias)
            fixed_bn.running_mean.copy_(mean.to(device=device, dtype=fixed_bn.running_mean.dtype))
            fixed_bn.running_var.copy_(var.to(device=device, dtype=fixed_bn.running_var.dtype))
            fixed_bn.num_batches_tracked.fill_(int(stats.get('frame_count', 0)))
        fixed_bn.requires_grad_(False)
        fixed_bn.eval()
        model.encoder.projector[1] = fixed_bn
        constant_latent = fixed_bn.bias.detach().clone()
    if args.freeze_encoder:
        if not args.encoder_checkpoint:
            raise ValueError('--freeze-encoder requires --encoder-checkpoint for reproducible shared features')
        if not args.encoder_stats:
            raise ValueError(
                '--freeze-encoder requires --encoder-stats; track_running_stats=False BN is batch-dependent')
        model.encoder.requires_grad_(False)
    elif args.encoder_stats:
        raise ValueError('--encoder-stats is only supported with --freeze-encoder')
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError('model has no trainable parameters')
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    print(f"=== {run_name} | n_actions={n_actions} | params={model.num_parameters():,} | amp={use_amp} ===", flush=True)

    # a fixed val batch for the influence metric
    vb_obs = torch.stack([val_ds[i][0] for i in range(min(256, len(val_ds)))])
    vb_act = torch.stack([val_ds[i][1] for i in range(min(256, len(val_ds)))])
    vb_update_mask = None
    if len(val_ds[0]) == 4:
        vb_update_mask = torch.stack(
            [val_ds[i][3] for i in range(min(256, len(val_ds)))])

    best = {}
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, opt, device, True, use_amp,
                       args.first_post_loss_weight)
        va = run_epoch(model, val_loader, opt, device, False, use_amp,
                       args.first_post_loss_weight)
        tau = model.horizons()
        aux_text = (f" hier={va['hier_loss']:.4f}" if 'hier_loss' in va else '')
        print(f"e{epoch:3d}/{args.epochs} ({time.time()-t0:.1f}s) train {tr['loss']:.4f}(pred {tr['pred_loss']:.4f}) "
              f"| val pred {va['pred_loss']:.4f}{aux_text} | "
              f"tau_f={tau['tau_fast']:.1f} tau_s={tau['tau_slow']:.1f}", flush=True)
        if wb is not None:
            wb.log({**{f'train/{key}': value for key, value in tr.items()},
                    **{f'val/{key}': value for key, value in va.items()},
                    'mem/tau_fast': tau['tau_fast'], 'mem/tau_slow': tau['tau_slow']}, step=epoch)
        history.append({'epoch': epoch, 'train': tr, 'val': va})

    # final eval: influence of each memory bank on the prediction
    infl = model.memory_influence(
        vb_obs.to(device), vb_act.to(device),
        memory_update_mask=(vb_update_mask.to(device) if vb_update_mask is not None else None))
    tau = model.horizons()
    tau_json = {k: (float(v) if np.isfinite(v) else None) for k, v in tau.items()}
    best = {'env': args.env_id, 'design': args.memory_mode, 'n_actions': n_actions,
            'val_pred_loss': va['pred_loss'], 'infl_fast': float(infl['infl_fast'].mean()),
            'infl_slow': float(infl['infl_slow'].mean()),
            'prototype_seed': args.prototype_seed,
            'dataset_schema_version': 3 if args.env_id.startswith(('dmc:', 'ogbench:')) else 1,
            'feature_schema_version': 1 if feature_mode else None,
            'feature_manifest': args.feature_manifest,
            'feature_manifest_sha256': getattr(args, 'feature_manifest_sha256', None),
            'target_env': args.target_env_id,
            'masked_clean_blackout_loss': bool(args.mask_occluded_target_loss),
            'first_post_loss_weight': float(args.first_post_loss_weight),
            'hier_loss_weight': float(args.hier_loss_weight),
            'hier_loss_weight_effective': (
                0.0 if args.memory_mode == 'hacsmv4_noaux' else
                float(args.hier_loss_weight) if args.memory_mode.startswith('hacsmv4') else 0.0),
            'val_pred_loss_target_kind': 'observed_pre_post_only' if args.mask_occluded_target_loss else 'all',
            'deep_blackout_target_kind': 'evaluation_only_hidden_clean',
            'primary_common_target_metric': 'clean_mse_first_post',
            'encoder_frozen': bool(args.freeze_encoder),
            'encoder_type': args.encoder_type,
            'predictor_norm': args.predictor_norm,
            'external_features_fixed': bool(feature_mode),
            'encoder_checkpoint': args.encoder_checkpoint,
            'encoder_stats': args.encoder_stats,
            'encoder_stats_sha256': getattr(args, 'encoder_stats_sha256', None),
            'trainable_parameters': model.num_parameters(),
            **tau_json}
    for key, value in va.items():
        if key.startswith('hier_'):
            best[f'val_{key}'] = float(value)
    if args.target_env_id is not None:
        best.update(phase_mse(
            model, val_loader, device, use_amp, args.length, args.history_len,
            constant_latent))
    if wb is not None:
        import wandb, matplotlib.pyplot as plt
        log = {f'eval/{k}': v for k, v in best.items() if isinstance(v, (int, float))}
        if model.memory_impl == 'ema':                      # kernels only defined for EMA banks
            fig = plot_memory_kernels(model); fp = out_dir / 'memory_kernels.png'
            fig.savefig(fp, dpi=100, bbox_inches='tight')
            log['eval/memory_kernels'] = wandb.Image(str(fp)); plt.close(fig)
        wb.log(log)
    torch.save({'model_state_dict': model.state_dict(), 'args': vars(args),
                'final_metrics': best, 'history': history}, out_dir / 'model.pt')
    json.dump(best, open(out_dir / 'metrics.json', 'w'), indent=2)
    print(f"=== done {run_name}: val_pred={best['val_pred_loss']:.4f} infl_fast={best['infl_fast']:.3f} "
          f"infl_slow={best['infl_slow']:.3f} ===", flush=True)
    if wb is not None:
        wb.summary.update(best); wb.finish()


if __name__ == '__main__':
    main()
