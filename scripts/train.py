"""
Training script for LeWorldModel.
Supports mixed precision, gradient accumulation, cosine LR with warmup.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lewm.models.leworldmodel import LeWorldModel


class OfflineTrajectoryDataset(Dataset):
    """
    Offline trajectory dataset for LeWM training.
    Loads trajectories of (observations, actions) and samples sub-trajectories.

    Supports:
      - .npz files with 'observations' and 'actions' keys
      - .npy files containing a dict with 'observations' and 'actions'
      - Trajectory-segmented data (list of trajectories)
    """

    def __init__(
        self,
        data_path: str,
        history_len: int = 3,
        frame_skip: int = 5,
        img_size: int = 224,
    ):
        self.history_len = history_len
        self.frame_skip = frame_skip
        self.img_size = img_size

        # Load data
        print(f"Loading data from {data_path}...")
        suffix = Path(data_path).suffix

        if suffix == '.npz':
            data = np.load(data_path, allow_pickle=True)
            self.observations = data['observations']
            self.actions = data['actions']
        elif suffix == '.npy':
            data = np.load(data_path, allow_pickle=True)
            if hasattr(data, 'item'):
                data = data.item()
            self.observations = data['observations']
            self.actions = data['actions']
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        # Ensure observations are float32 in [0, 1]
        if self.observations.dtype == np.uint8:
            self.observations = self.observations.astype(np.float32) / 255.0
        elif self.observations.dtype != np.float32:
            self.observations = self.observations.astype(np.float32)

        # Ensure channel-first format (N, C, H, W)
        if self.observations.ndim == 4 and self.observations.shape[-1] in [1, 3]:
            self.observations = np.transpose(self.observations, (0, 3, 1, 2))

        # Ensure actions are float32
        self.actions = self.actions.astype(np.float32)

        # Build trajectory boundaries
        self.num_samples = max(1, len(self.observations) - history_len * frame_skip - 1)

        print(f"  Observations: {self.observations.shape}, dtype={self.observations.dtype}")
        print(f"  Actions: {self.actions.shape}, dtype={self.actions.dtype}")
        print(f"  Samples: {self.num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        indices = [idx + i * self.frame_skip for i in range(self.history_len + 1)]
        obs = self.observations[indices]  # (N+1, C, H, W)
        act_idx = [idx + i * self.frame_skip for i in range(self.history_len)]
        act = self.actions[act_idx]  # (N, A)

        obs = torch.from_numpy(obs.copy()).float()
        act = torch.from_numpy(act.copy()).float()

        return obs, act


class SyntheticPushDataset(Dataset):
    """
    Synthetic dataset for testing LeWM training.
    Generates random observations with temporal correlation.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        history_len: int = 3,
        img_size: int = 64,
        action_dim: int = 2,
    ):
        self.num_samples = num_samples
        self.history_len = history_len
        self.img_size = img_size
        self.action_dim = action_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        torch.manual_seed(idx)
        obs = torch.randn(self.history_len + 1, 3, self.img_size, self.img_size) * 0.1
        for i in range(1, self.history_len + 1):
            obs[i] = 0.8 * obs[i - 1] + 0.2 * obs[i]
        actions = torch.randn(self.history_len, self.action_dim) * 0.5
        return obs, actions


def train_epoch(
    model: LeWorldModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    """Train for one epoch with optional mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_sigreg_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, (obs, actions) in enumerate(dataloader):
        obs = obs.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        if scaler is not None:
            with autocast('cuda'):
                losses = model.compute_loss(obs, actions)
                loss = losses['loss'] / grad_accum_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            losses = model.compute_loss(obs, actions)
            loss = losses['loss'] / grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += losses['loss'].item()
        total_pred_loss += losses['pred_loss'].item()
        total_sigreg_loss += losses['sigreg_loss'].item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'pred_loss': total_pred_loss / num_batches,
        'sigreg_loss': total_sigreg_loss / num_batches,
    }


@torch.no_grad()
def validate(
    model: LeWorldModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_sigreg_loss = 0.0
    num_batches = 0

    for obs, actions in dataloader:
        obs = obs.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)

        losses = model.compute_loss(obs, actions)
        total_loss += losses['loss'].item()
        total_pred_loss += losses['pred_loss'].item()
        total_sigreg_loss += losses['sigreg_loss'].item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'pred_loss': total_pred_loss / num_batches,
        'sigreg_loss': total_sigreg_loss / num_batches,
    }


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min: float = 1e-6,
):
    """Cosine annealing with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(eta_min, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description='Train LeWorldModel')

    # Data
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--use-synthetic', action='store_true')
    parser.add_argument('--num-synthetic', type=int, default=10000)
    parser.add_argument('--output-dir', type=str, default='outputs/lewm')

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--min-lr', type=float, default=1e-6)

    # Model
    parser.add_argument('--embed-dim', type=int, default=192)
    parser.add_argument('--action-dim', type=int, default=2)
    parser.add_argument('--history-len', type=int, default=3)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--encoder-layers', type=int, default=12)
    parser.add_argument('--encoder-heads', type=int, default=3)
    parser.add_argument('--predictor-layers', type=int, default=6)
    parser.add_argument('--predictor-heads', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)

    # SIGReg
    parser.add_argument('--sigreg-lambda', type=float, default=0.1)
    parser.add_argument('--sigreg-projections', type=int, default=1024)

    # Misc
    parser.add_argument('--frame-skip', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print("LeWorldModel Training")
    print("=" * 60)
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    print(f"  Output: {output_dir}")
    print(f"  AMP: {args.use_amp}")
    print(f"  Batch size: {args.batch_size} (x{args.grad_accum_steps} accum = {args.batch_size * args.grad_accum_steps} effective)")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr} (warmup {args.warmup_steps} steps)")
    print(f"  Model: embed={args.embed_dim}, action={args.action_dim}, history={args.history_len}")
    print(f"  SIGReg: lambda={args.sigreg_lambda}, projections={args.sigreg_projections}")

    # Create model
    model = LeWorldModel(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        action_dim=args.action_dim,
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
        predictor_layers=args.predictor_layers,
        predictor_heads=args.predictor_heads,
        history_len=args.history_len,
        dropout=args.dropout,
        sigreg_lambda=args.sigreg_lambda,
        sigreg_projections=args.sigreg_projections,
    ).to(device)

    print(f"  Parameters: {model.num_parameters():,}")

    # Create datasets
    if args.use_synthetic or args.data_path is None:
        print("  Using synthetic dataset")
        train_dataset = SyntheticPushDataset(
            num_samples=args.num_synthetic,
            history_len=args.history_len,
            img_size=args.img_size,
            action_dim=args.action_dim,
        )
        val_dataset = SyntheticPushDataset(
            num_samples=max(100, args.num_synthetic // 5),
            history_len=args.history_len,
            img_size=args.img_size,
            action_dim=args.action_dim,
        )
    else:
        print(f"  Loading data from {args.data_path}")
        train_dataset = OfflineTrajectoryDataset(
            data_path=args.data_path,
            history_len=args.history_len,
            frame_skip=args.frame_skip,
            img_size=args.img_size,
        )
        val_dataset = train_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # LR scheduler with warmup
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
        eta_min=args.min_lr,
    )

    # Mixed precision
    scaler = GradScaler('cuda') if args.use_amp and torch.cuda.is_available() else None

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"  Resumed from epoch {start_epoch - 1}, best val loss: {best_val_loss:.4f}")

    # TensorBoard
    writer = SummaryWriter(str(output_dir / 'logs'))

    # Training loop
    print("=" * 60)
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, device, scaler, args.grad_accum_steps
        )
        val_metrics = validate(model, val_loader, device)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s, lr={current_lr:.2e}) | "
            f"Train: {train_metrics['loss']:.4f} "
            f"(pred={train_metrics['pred_loss']:.4f}, sigreg={train_metrics['sigreg_loss']:.4f}) | "
            f"Val: {val_metrics['loss']:.4f} "
            f"(pred={val_metrics['pred_loss']:.4f}, sigreg={val_metrics['sigreg_loss']:.4f})"
        )

        # Log to TensorBoard
        writer.add_scalars('loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss'],
        }, epoch)
        writer.add_scalars('pred_loss', {
            'train': train_metrics['pred_loss'],
            'val': val_metrics['pred_loss'],
        }, epoch)
        writer.add_scalars('sigreg_loss', {
            'train': train_metrics['sigreg_loss'],
            'val': val_metrics['sigreg_loss'],
        }, epoch)
        writer.add_scalar('lr', current_lr, epoch)

        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']

        if epoch % args.save_interval == 0 or is_best:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'train_loss': train_metrics['loss'],
                'args': vars(args),
            }
            if scaler is not None:
                ckpt['scaler_state_dict'] = scaler.state_dict()

            if is_best:
                path = output_dir / 'best_model.pt'
            else:
                path = output_dir / f'checkpoint_epoch_{epoch}.pt'

            torch.save(ckpt, str(path))
            if is_best:
                print(f"  New best! Saved to {path}")

    writer.close()
    print("=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
