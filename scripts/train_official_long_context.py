#!/usr/bin/env python3
"""Train an official-LeWM long-context predictor on cached official latents.

The cache contract is ``<cache-root>/<task>/{train,val}.npz`` with arrays
``z`` (E,L,192), ``actions`` (E,L-1,10), ``xi``, ``endo_state``,
``exo_state``, ``event_*``, and scalar JSON string ``meta_json``.  Pixels and
the official encoder are deliberately absent from optimization: the cached
``z`` values are the frozen official encoder/projector outputs.

For H in {3,16,32,56}, the trainable model is initialized from the released
H=3 official checkpoint.  Transformer, action-encoder, and prediction-
projection weights are copied exactly; the three learned temporal position
vectors are linearly interpolated (or periodically repeated) to H.  The
training objective matches this repository's LeWM sliding-window convention:
the final causal output of every H-token window predicts the next latent.

Besides next-latent MSE, evaluation fits a semantic readout on strictly legal
features (only z/actions before the target are consumed).  Transient-marker
and drifting-color tasks use a standardized logistic readout from the final
contextual prediction at decision time.  Occluded tracking uses
standardized-X/standardized-y RidgeCV on the prediction at ``gap_off``.
Target-window coverage and partial/full-context counts are saved explicitly.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models import official_lewm


SCHEMA_VERSION = 1
HISTORY_CHOICES = (3, 16, 32, 56)
TASK_FAMILIES = ("transient-marker", "drifting-color", "occluded-tracking")
RIDGECV_ALPHAS = np.logspace(-3, 3, 7)


# ---------------------------------------------------------------------------
# Reproducibility and small utilities
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()
                   if parameter.requires_grad or not trainable_only))


def _json_scalar(value: np.ndarray, name: str) -> dict[str, Any]:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar JSON string, got {array.shape}")
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    if not isinstance(item, str):
        raise ValueError(f"{name} must contain a JSON string, got {type(item).__name__}")
    parsed = json.loads(item)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} JSON must decode to an object")
    return parsed


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_as_jsonable(payload), indent=2,
                                    sort_keys=True) + "\n")
    temporary.replace(path)


# ---------------------------------------------------------------------------
# Fixed latent-cache contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentCache:
    path: Path
    z: np.ndarray
    actions: np.ndarray
    xi: np.ndarray
    events: dict[str, np.ndarray]
    metadata: dict[str, Any]
    xi_kind: str
    n_classes: int
    task_family: str

    @property
    def episodes(self) -> int:
        return int(self.z.shape[0])

    @property
    def length(self) -> int:
        return int(self.z.shape[1])


def canonical_task_family(task: str) -> str:
    normalized = task.strip().lower().replace("_", "-")
    if normalized.startswith("t1") or normalized in {
            "transient", "transient-cue", "transient-marker"}:
        return "transient-marker"
    if normalized.startswith("t3") or normalized in {
            "drifter", "drifting-color", "drifting-colour"}:
        return "drifting-color"
    if normalized == "t4" or normalized in {
            "freeze-track", "occluded-track", "occluded-tracking"}:
        return "occluded-tracking"
    raise ValueError(
        f"cannot map task {task!r} to one of {TASK_FAMILIES}; use --task-family")


def load_cache(path: str | Path, task_family: str = "auto") -> LatentCache:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"z", "actions", "xi", "endo_state", "exo_state", "meta_json"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{path}: missing required arrays {missing}")
        z = np.asarray(archive["z"], dtype=np.float32)
        actions = np.asarray(archive["actions"], dtype=np.float32)
        xi = np.asarray(archive["xi"])
        endo_state = np.asarray(archive["endo_state"])
        exo_state = np.asarray(archive["exo_state"])
        metadata = _json_scalar(archive["meta_json"], "meta_json")
        events = {
            key.removeprefix("event_"): np.asarray(archive[key])
            for key in archive.files if key.startswith("event_")
        }

    if z.ndim != 3 or z.shape[-1] != official_lewm.OFFICIAL_EMBED_DIM:
        raise ValueError(
            f"{path}: z must have shape (E,L,{official_lewm.OFFICIAL_EMBED_DIM}), "
            f"got {z.shape}")
    episodes, length, _ = z.shape
    expected_actions = (episodes, length - 1, official_lewm.OFFICIAL_ACTION_DIM)
    if actions.shape != expected_actions:
        raise ValueError(f"{path}: actions must have shape {expected_actions}, "
                         f"got {actions.shape}")
    for name, array in (("z", z), ("actions", actions),
                        ("endo_state", endo_state), ("exo_state", exo_state)):
        if name in {"endo_state", "exo_state"} and (
                array.ndim < 3 or array.shape[:2] != (episodes, length)):
            raise ValueError(f"{path}: {name} must start with (E,L), got {array.shape}")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"{path}: {name} must contain finite numeric values")
    if xi.shape[0] != episodes:
        raise ValueError(f"{path}: xi leading dimension must be {episodes}, got {xi.shape}")
    for name, value in events.items():
        if value.shape != (episodes,) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError(
                f"{path}: event_{name} must be integer (E,), got {value.dtype} {value.shape}")

    xi_kind = str(metadata.get("xi_kind", "")).lower()
    if not xi_kind:
        xi_kind = "cat" if xi.ndim == 1 and np.issubdtype(
            xi.dtype, np.integer) else "cont"
    if xi_kind not in {"cat", "cont"}:
        raise ValueError(f"{path}: meta_json xi_kind must be 'cat' or 'cont'")
    if xi_kind == "cat":
        if xi.ndim != 1 or not np.issubdtype(xi.dtype, np.integer):
            raise ValueError(f"{path}: categorical xi must be integer (E,), got {xi.shape}")
        xi = xi.astype(np.int64, copy=False)
        n_classes = int(metadata.get("n_classes", int(xi.max()) + 1))
        if n_classes < 2 or xi.min() < 0 or xi.max() >= n_classes:
            raise ValueError(f"{path}: categorical xi is outside n_classes={n_classes}")
    else:
        if xi.ndim != 2 or xi.shape[1] < 1 or not np.isfinite(xi).all():
            raise ValueError(f"{path}: continuous xi must be finite (E,K), got {xi.shape}")
        xi = xi.astype(np.float32, copy=False)
        n_classes = 0

    if task_family == "auto":
        task_name = str(metadata.get("task", path.parent.name))
        family = canonical_task_family(task_name)
    else:
        family = task_family
    expected_kind = "cont" if family == "occluded-tracking" else "cat"
    if xi_kind != expected_kind:
        raise ValueError(f"{path}: {family} requires xi_kind={expected_kind}, "
                         f"got {xi_kind}")
    required_event = "gap_off" if family == "occluded-tracking" else "cue_off"
    if required_event not in events:
        raise ValueError(f"{path}: {family} requires event_{required_event}")
    return LatentCache(path, z, actions, xi, events, metadata, xi_kind,
                       n_classes, family)


def validate_cache_pair(train: LatentCache, validation: LatentCache,
                        history: int) -> None:
    if train.task_family != validation.task_family:
        raise ValueError("train and validation caches map to different task families")
    if train.xi_kind != validation.xi_kind or train.n_classes != validation.n_classes:
        raise ValueError("train and validation xi schemas differ")
    for name, cache in (("train", train), ("validation", validation)):
        if cache.length <= history:
            raise ValueError(f"{name} sequence length {cache.length} must exceed H={history}")


class SlidingWindowDataset(Dataset):
    """All exact-H causal windows; only the final next latent is supervised."""

    def __init__(self, cache: LatentCache, history: int) -> None:
        self.z = cache.z
        self.actions = cache.actions
        self.history = history
        self.windows_per_episode = cache.length - history

    def __len__(self) -> int:
        return int(self.z.shape[0] * self.windows_per_episode)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        episode, start = divmod(int(index), self.windows_per_episode)
        stop = start + self.history
        return (
            torch.from_numpy(self.z[episode, start:stop]),
            torch.from_numpy(self.actions[episode, start:stop]),
            torch.from_numpy(self.z[episode, stop]),
        )


# ---------------------------------------------------------------------------
# Official initialization and trainable model
# ---------------------------------------------------------------------------


def resize_position_embedding(source: torch.Tensor, history: int,
                              method: str) -> torch.Tensor:
    expected = (1, official_lewm.OFFICIAL_HISTORY,
                official_lewm.OFFICIAL_EMBED_DIM)
    if tuple(source.shape) != expected:
        raise ValueError(f"official position embedding must have shape {expected}, "
                         f"got {tuple(source.shape)}")
    if history not in HISTORY_CHOICES:
        raise ValueError(f"history must be one of {HISTORY_CHOICES}")
    if history == official_lewm.OFFICIAL_HISTORY:
        return source.detach().clone()
    if method == "interpolate":
        return F.interpolate(
            source.detach().transpose(1, 2).float(), size=history,
            mode="linear", align_corners=True).transpose(1, 2).to(source.dtype)
    if method == "repeat":
        repeats = math.ceil(history / source.shape[1])
        return source.detach().repeat(1, repeats, 1)[:, :history].clone()
    raise ValueError(f"unknown position initialization {method!r}")


class LongContextPredictor(nn.Module):
    def __init__(self, history: int) -> None:
        super().__init__()
        self.history = history
        self.predictor = official_lewm.Predictor(num_frames=history)
        self.action_encoder = official_lewm.Embedder()
        self.pred_proj = official_lewm.MLP()

    def forward(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[-1] != official_lewm.OFFICIAL_EMBED_DIM:
            raise ValueError(f"latent must be (B,T,192), got {tuple(latent.shape)}")
        if actions.shape != (*latent.shape[:2], official_lewm.OFFICIAL_ACTION_DIM):
            raise ValueError(f"actions must be (B,T,10), got {tuple(actions.shape)}")
        if latent.shape[1] < 1 or latent.shape[1] > self.history:
            raise ValueError(f"context length must be in [1,{self.history}]")
        condition = self.action_encoder(actions)
        hidden = self.predictor(latent, condition)
        batch, steps, dimension = hidden.shape
        projected = self.pred_proj(hidden.reshape(batch * steps, dimension))
        return projected.reshape(batch, steps, dimension)


def initialize_from_official(
        checkpoint: str | Path, history: int, position_init: str,
        ) -> tuple[LongContextPredictor, dict[str, Any]]:
    """Load released weights through the repository's strict official loader."""
    checkpoint = Path(checkpoint)
    source = official_lewm.load_official_reacher_checkpoint(checkpoint, "cpu")
    source_position = source.predictor.pos_embedding.detach().clone()
    model = LongContextPredictor(history)

    predictor_state = source.predictor.state_dict()
    predictor_state["pos_embedding"] = resize_position_embedding(
        source_position, history, position_init)
    model.predictor.load_state_dict(predictor_state, strict=True)
    model.action_encoder.load_state_dict(source.action_encoder.state_dict(), strict=True)
    model.pred_proj.load_state_dict(source.pred_proj.state_dict(), strict=True)
    model.requires_grad_(True)

    source_component_parameters = (
        parameter_count(source.predictor)
        + parameter_count(source.action_encoder)
        + parameter_count(source.pred_proj)
    )
    report = {
        "loader": "lewm.models.official_lewm.load_official_reacher_checkpoint",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_history": official_lewm.OFFICIAL_HISTORY,
        "target_history": history,
        "position_initialization": position_init,
        "copied_modules": ["predictor.transformer", "action_encoder", "pred_proj"],
        "source_component_parameters": source_component_parameters,
        "target_parameters": parameter_count(model),
        "added_position_parameters": (
            history - official_lewm.OFFICIAL_HISTORY
        ) * official_lewm.OFFICIAL_EMBED_DIM,
        "encoder": "frozen upstream; not instantiated after cached-latent initialization",
    }
    del source
    gc.collect()
    return model, report


# ---------------------------------------------------------------------------
# Training and prediction MSE
# ---------------------------------------------------------------------------


def make_loader(dataset: Dataset, args: argparse.Namespace,
                *, train: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed + 10_003) if train else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=args.device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )


def autocast_context(args: argparse.Namespace):
    if not args.amp:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type=args.device.type, dtype=dtype)


def make_grad_scaler(args: argparse.Namespace):
    enabled = bool(args.amp and args.device.type == "cuda"
                   and args.amp_dtype == "float16")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch 2.0 compatibility
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_epoch(model: LongContextPredictor, loader: DataLoader,
                optimizer: torch.optim.Optimizer, scaler: Any,
                args: argparse.Namespace) -> float:
    model.train()
    total_squared = 0.0
    total_values = 0
    for latent, actions, target in loader:
        latent = latent.to(args.device, non_blocking=True)
        actions = actions.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args):
            prediction = model(latent, actions)[:, -1]
            loss = F.mse_loss(prediction.float(), target.float())
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        values = target.numel()
        total_squared += float(loss.detach()) * values
        total_values += values
    return total_squared / max(total_values, 1)


@torch.no_grad()
def prediction_mse(model: LongContextPredictor, loader: DataLoader,
                   args: argparse.Namespace) -> tuple[float, int]:
    model.eval()
    total_squared = 0.0
    total_values = 0
    windows = 0
    for latent, actions, target in loader:
        latent = latent.to(args.device, non_blocking=True)
        actions = actions.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        with autocast_context(args):
            prediction = model(latent, actions)[:, -1]
        squared = (prediction.float() - target.float()).square()
        total_squared += float(squared.sum())
        total_values += squared.numel()
        windows += target.shape[0]
    return total_squared / max(total_values, 1), windows


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone()
            for name, value in module.state_dict().items()}


# ---------------------------------------------------------------------------
# Legal decision-target features and probes
# ---------------------------------------------------------------------------


def target_times(cache: LatentCache) -> np.ndarray:
    if cache.task_family == "occluded-tracking":
        target = np.asarray(cache.events["gap_off"], dtype=np.int64)
    elif "decision_time" in cache.events:
        target = np.asarray(cache.events["decision_time"], dtype=np.int64)
    else:
        value = cache.metadata.get("decision_time", cache.length - 1)
        target = np.full(cache.episodes, int(value), dtype=np.int64)
    if (target < 1).any() or (target >= cache.length).any():
        raise ValueError(
            f"{cache.path}: semantic target times must lie in [1,{cache.length - 1}]")
    return target


@torch.no_grad()
def legal_semantic_features(
        model: LongContextPredictor, cache: LatentCache,
        args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Features at target q consume only z[start:q] and actions[start:q]."""
    model.eval()
    targets = target_times(cache)
    starts = np.maximum(0, targets - args.history_len)
    context_lengths = targets - starts
    features: list[np.ndarray | None] = [None] * cache.episodes

    # Variable target times (occluded tracking) imply variable legal lengths.
    # Grouping retains vectorized GPU execution without padding future tokens.
    for context_length in sorted(np.unique(context_lengths).tolist()):
        episode_indices = np.flatnonzero(context_lengths == context_length)
        for offset in range(0, len(episode_indices), args.batch_size):
            selected = episode_indices[offset:offset + args.batch_size]
            latent_np = np.stack([
                cache.z[index, starts[index]:targets[index]] for index in selected
            ])
            action_np = np.stack([
                cache.actions[index, starts[index]:targets[index]]
                for index in selected
            ])
            latent = torch.from_numpy(latent_np).to(args.device, non_blocking=True)
            actions = torch.from_numpy(action_np).to(args.device, non_blocking=True)
            with autocast_context(args):
                prediction = model(latent, actions)
            prediction_np = prediction.float().cpu().numpy()
            for row, episode in enumerate(selected):
                # The repository's sliding objective supervises the final
                # transformer position only.  Probe that trained coordinate,
                # not unsupervised earlier positions from the same forward.
                features[episode] = prediction_np[row, -1]

    if any(feature is None for feature in features):
        raise RuntimeError("internal error: one or more semantic features were not built")
    matrix = np.stack(features).astype(np.float64)
    coverage: dict[str, Any] = {
        "requested_target_windows": cache.episodes,
        "valid_target_windows": int(len(matrix)),
        "full_H_context_windows": int((context_lengths == args.history_len).sum()),
        "partial_context_windows": int((context_lengths < args.history_len).sum()),
        "target_time_min": int(targets.min()),
        "target_time_max": int(targets.max()),
        "context_length_min": int(context_lengths.min()),
        "context_length_mean": float(context_lengths.mean()),
        "context_length_max": int(context_lengths.max()),
        "future_target_observation_consumed": False,
        "legal_input_contract": "z[start:target], actions[start:target] only",
    }
    if cache.task_family != "occluded-tracking":
        cue_on = cache.events.get("cue_on", cache.events["cue_off"] - 1)
        cue_off = cache.events["cue_off"]
        cue_frames_reachable = np.maximum(
            0, np.minimum(targets, cue_off) - np.maximum(starts, cue_on))
        cue_fully_reachable = ((starts <= cue_on) & (targets >= cue_off))
        coverage.update({
            "cue_any_frame_reachable_from_context": int(
                (cue_frames_reachable > 0).sum()),
            "cue_full_window_reachable_from_context": int(
                cue_fully_reachable.sum()),
            "cue_frames_reachable_min": int(cue_frames_reachable.min()),
            "cue_frames_reachable_mean": float(cue_frames_reachable.mean()),
            "cue_frames_reachable_max": int(cue_frames_reachable.max()),
        })
    return matrix, cache.xi, coverage


def semantic_readout(
        model: LongContextPredictor, train: LatentCache,
        validation: LatentCache, args: argparse.Namespace) -> dict[str, Any]:
    train_x, train_y, train_counts = legal_semantic_features(
        model, train, args)
    val_x, val_y, val_counts = legal_semantic_features(
        model, validation, args)
    common = {
        "task_family": train.task_family,
        "feature_dimension": int(train_x.shape[1]),
        "train_target_windows": train_counts,
        "validation_target_windows": val_counts,
    }
    if train.xi_kind == "cat":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed),
        )
        probe.fit(train_x, train_y)
        accuracy = float(probe.score(val_x, val_y))
        return {
            **common,
            "metric": "accuracy",
            "value": accuracy,
            "chance": 1.0 / train.n_classes,
            "readout": "StandardScaler + LogisticRegression",
            "parameters": {"C": 1.0, "max_iter": 2000,
                           "random_state": args.seed},
            "feature": "final contextual prediction at decision time",
        }

    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    x_scaler = StandardScaler().fit(train_x)
    y_scaler = StandardScaler().fit(train_y)
    probe = RidgeCV(alphas=RIDGECV_ALPHAS)
    probe.fit(x_scaler.transform(train_x), y_scaler.transform(train_y))
    prediction = y_scaler.inverse_transform(
        probe.predict(x_scaler.transform(val_x)))
    per_dimension = [float(r2_score(val_y[:, index], prediction[:, index]))
                     for index in range(val_y.shape[1])]
    alpha = np.asarray(probe.alpha_)
    return {
        **common,
        "metric": "r2",
        "value": float(r2_score(val_y, prediction)),
        "per_dimension_r2": per_dimension,
        "readout": "StandardScaler(X) + StandardScaler(y) + RidgeCV",
        "parameters": {
            "alphas": RIDGECV_ALPHAS.tolist(),
            "selected_alpha": alpha.tolist() if alpha.ndim else float(alpha),
            "target_standardized_during_fit": True,
            "prediction_inverse_transformed_before_r2": True,
        },
        "feature": "final legal prediction at gap_off",
    }


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--val-cache", required=True, type=Path)
    parser.add_argument("--official-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--history-len", type=int, choices=HISTORY_CHOICES,
                        required=True)
    parser.add_argument("--position-init", choices=("interpolate", "repeat"),
                        default="interpolate")
    parser.add_argument("--task-family", choices=("auto", *TASK_FAMILIES),
                        default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"),
                        default="bfloat16")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        parser.error("optimizer hyperparameters are outside their valid range")
    args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA device requested but torch.cuda.is_available() is false")
    if args.amp and args.device.type != "cuda":
        args.amp = False
    return args


def prepare_output(directory: Path, force: bool) -> None:
    products = ("checkpoint.pt", "history.csv", "metrics.json")
    existing = [directory / name for name in products if (directory / name).exists()]
    if existing and not force:
        raise FileExistsError(
            f"output products already exist: {existing}; pass --force to replace")
    directory.mkdir(parents=True, exist_ok=True)


def write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty history")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare_output(args.output_dir, args.force)
    seed_everything(args.seed)
    train_cache = load_cache(args.train_cache, args.task_family)
    val_cache = load_cache(args.val_cache, args.task_family)
    validate_cache_pair(train_cache, val_cache, args.history_len)

    model, initialization = initialize_from_official(
        args.official_checkpoint, args.history_len, args.position_init)
    model.to(args.device)
    train_dataset = SlidingWindowDataset(train_cache, args.history_len)
    val_dataset = SlidingWindowDataset(val_cache, args.history_len)
    train_loader = make_loader(train_dataset, args, train=True)
    val_loader = make_loader(val_dataset, args, train=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = make_grad_scaler(args)

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_mse = float("inf")
    best_epoch = -1
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_mse = train_epoch(model, train_loader, optimizer, scaler, args)
        val_mse, val_windows = prediction_mse(model, val_loader, args)
        if val_mse < best_mse:
            best_mse = val_mse
            best_epoch = epoch
            best_state = cpu_state_dict(model)
        row = {
            "epoch": epoch,
            "train_prediction_mse": train_mse,
            "val_prediction_mse": val_mse,
            "val_target_windows": val_windows,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(row)
        write_history(args.output_dir / "history.csv", history)
        print(
            f"[official-long-context] epoch {epoch:03d}/{args.epochs:03d} "
            f"train_mse={train_mse:.6g} val_mse={val_mse:.6g} "
            f"windows={val_windows} seconds={row['epoch_seconds']:.1f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("training completed without a checkpoint state")
    model.load_state_dict(best_state, strict=True)
    model.to(args.device)
    final_train_mse, train_windows = prediction_mse(model, train_loader, args)
    final_val_mse, val_windows = prediction_mse(model, val_loader, args)
    semantic = semantic_readout(model, train_cache, val_cache, args)

    cache_summary = {
        "train": {
            "path": str(train_cache.path.resolve()),
            "sha256": sha256_file(train_cache.path),
            "episodes": train_cache.episodes,
            "length": train_cache.length,
            "sliding_target_windows": train_windows,
            "metadata": train_cache.metadata,
        },
        "validation": {
            "path": str(val_cache.path.resolve()),
            "sha256": sha256_file(val_cache.path),
            "episodes": val_cache.episodes,
            "length": val_cache.length,
            "sliding_target_windows": val_windows,
            "metadata": val_cache.metadata,
        },
    }
    config = {
        "history_len": args.history_len,
        "position_init": args.position_init,
        "task_family": train_cache.task_family,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "device": str(args.device),
        "amp": args.amp,
        "amp_dtype": args.amp_dtype,
        "objective": "final-token next-latent MSE over all exact-H windows",
        "encoder_frozen": True,
        "encoder_instantiated_during_training": False,
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "study": "official-lewm-long-context",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "initialization": initialization,
        "parameters": {
            "total": parameter_count(model),
            "trainable": parameter_count(model, trainable_only=True),
        },
        "caches": cache_summary,
        "best_epoch": best_epoch,
        "best_validation_prediction_mse_during_training": best_mse,
        "best_checkpoint_prediction_mse": {
            "train": final_train_mse,
            "validation": final_val_mse,
        },
        "semantic_target_readout": semantic,
        "elapsed_seconds": time.time() - started,
    }

    checkpoint_payload = {
        "schema_version": SCHEMA_VERSION,
        "study": "official-lewm-long-context",
        "model_state_dict": cpu_state_dict(model),
        "config": config,
        "initialization": initialization,
        "best_epoch": best_epoch,
        "best_validation_prediction_mse": best_mse,
        "cache_sha256": {
            "train": cache_summary["train"]["sha256"],
            "validation": cache_summary["validation"]["sha256"],
        },
    }
    checkpoint_tmp = args.output_dir / "checkpoint.pt.tmp"
    torch.save(checkpoint_payload, checkpoint_tmp)
    checkpoint_tmp.replace(args.output_dir / "checkpoint.pt")
    _atomic_json(args.output_dir / "metrics.json", metrics)
    print(
        f"[official-long-context] best epoch={best_epoch} "
        f"val_mse={final_val_mse:.6g} semantic_{semantic['metric']}="
        f"{semantic['value']:.6g} -> {args.output_dir}", flush=True)
    return metrics


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
